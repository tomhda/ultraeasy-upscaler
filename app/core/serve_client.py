"""UEU常駐ヘルパーのバイナリプロトコルクライアント。"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .jobs import Cancelled

MAGIC_READY = b"UEUH"
MAGIC_FRAME = b"UEUF"
MAGIC_DATA = b"UEUD"
MAGIC_ERROR = b"UEUE"

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


class ServeClientError(RuntimeError):
    """ヘルパーの起動・接続・プロトコル自体が利用できない。"""


class ServeClient:
    """起動引数をそのまま受け取り、UEUプロトコルでRGB画像を処理する。"""

    def __init__(
        self,
        command: Sequence[str | os.PathLike[str]],
        workdir: str | os.PathLike[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not command:
            raise ValueError("helper command must not be empty")
        self.command = [os.fspath(arg) for arg in command]
        self.workdir = os.fspath(workdir) if workdir is not None else None
        self.env = env
        self.log = log
        self.proc: subprocess.Popen | None = None
        self.scale = 0
        self.tile_w = 0
        self.tile_h = 0
        self.stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._io_lock = threading.Lock()

    def connect(
        self,
        timeout: float | None = None,
        *,
        progress: Callable[[float], None] | None = None,
        cancel=None,
    ) -> None:
        """プロセスを起動し、UEUHを受け取るまで待つ。"""
        if self.proc is not None:
            raise ServeClientError("helper is already connected")

        try:
            self.proc = subprocess.Popen(
                self.command,
                cwd=self.workdir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=self.env,
                creationflags=_CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise ServeClientError(f"AI helperを起動できません: {self.command[0]}") from exc

        self._start_stderr_drain()
        finished = threading.Event()
        result: dict[str, object] = {}

        def _read_ready() -> None:
            try:
                result["header"] = self._read_exact(16)
            except BaseException as exc:  # メイン待機側へ引き渡す
                result["error"] = exc
            finally:
                finished.set()

        reader = threading.Thread(target=_read_ready, name="ueu-ready", daemon=True)
        reader.start()
        started = time.monotonic()
        try:
            while not finished.wait(0.1):
                elapsed = time.monotonic() - started
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                if timeout is not None and elapsed >= timeout:
                    raise ServeClientError(f"AI helperの準備が{timeout:g}秒以内に完了しませんでした")
                if progress is not None:
                    progress(elapsed)

            if "error" in result:
                exc = result["error"]
                if isinstance(exc, BaseException):
                    raise ServeClientError(self._failure_message(str(exc))) from exc
                raise ServeClientError(self._failure_message("ready header read failed"))

            header = result.get("header")
            if not isinstance(header, bytes) or header[:4] != MAGIC_READY:
                magic = header[:4] if isinstance(header, bytes) else header
                raise ServeClientError(self._failure_message(f"unexpected ready magic: {magic!r}"))
            self.scale, self.tile_w, self.tile_h = struct.unpack("<iii", header[4:16])
            if self.scale <= 0 or self.tile_w <= 0 or self.tile_h <= 0:
                raise ServeClientError(
                    self._failure_message(
                        f"invalid ready values: scale={self.scale}, tile={self.tile_w}x{self.tile_h}"
                    )
                )
        except BaseException:
            self.close(force=True)
            raise

    def send(self, image: np.ndarray) -> None:
        """RGB24フレームを送信する。"""
        proc = self._require_connected()
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 uint8 RGB, got {image.dtype} {image.shape}")
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image size: {width}x{height}")
        payload = np.ascontiguousarray(image).tobytes()
        try:
            assert proc.stdin is not None
            proc.stdin.write(MAGIC_FRAME + struct.pack("<ii", width, height) + payload)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ServeClientError(self._failure_message("AI helperへの送信に失敗しました")) from exc

    def receive(self, cancel=None) -> np.ndarray:
        """次の応答を読む。UEUEはそのフレームだけのValueErrorにする。"""
        if cancel is not None:
            finished = threading.Event()
            result: dict[str, object] = {}
            def _read() -> None:
                try:
                    result["value"] = self._receive_once()
                except BaseException as exc:
                    result["error"] = exc
                finally:
                    finished.set()
            reader = threading.Thread(target=_read, name="ueu-frame-read", daemon=True)
            reader.start()
            while not finished.wait(0.05):
                if cancel.is_set():
                    self.close(force=True)
                    raise Cancelled()
            if cancel.is_set():
                self.close(force=True)
                raise Cancelled()
            error = result.get("error")
            if isinstance(error, BaseException):
                raise error
            return result["value"]  # type: ignore[return-value]
        return self._receive_once()

    def _receive_once(self) -> np.ndarray:
        """キャンセル監視スレッドからも呼べる同期受信本体。"""
        self._require_connected()
        try:
            magic = self._read_exact(4)
            if magic == MAGIC_ERROR:
                (length,) = struct.unpack("<i", self._read_exact(4))
                if length < 0 or length > 16 * 1024 * 1024:
                    raise ServeClientError(f"invalid UEUE length: {length}")
                message = self._read_exact(length).decode("utf-8", errors="replace")
                raise ValueError(message)
            if magic != MAGIC_DATA:
                raise ServeClientError(f"unexpected response magic: {magic!r}")
            out_w, out_h = struct.unpack("<ii", self._read_exact(8))
            if out_w <= 0 or out_h <= 0 or out_w > 65536 or out_h > 65536:
                raise ServeClientError(f"invalid response size: {out_w}x{out_h}")
            data = self._read_exact(out_w * out_h * 3)
            return np.frombuffer(data, dtype=np.uint8).reshape(out_h, out_w, 3).copy()
        except ValueError:
            raise
        except ServeClientError:
            raise
        except (EOFError, OSError, struct.error) as exc:
            raise ServeClientError(self._failure_message(str(exc))) from exc

    def upscale(self, image: np.ndarray, cancel=None) -> np.ndarray:
        """1フレームを同期送受信する。"""
        with self._io_lock:
            self.send(image)
            return self.receive(cancel=cancel)

    def close(self, *, force: bool = False) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()  # EOFでサーバーを正常終了させる
            if force and proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=10 if not force else 3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        except Exception:
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        finally:
            self.proc = None
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=2)
                self._stderr_thread = None

    def _require_connected(self) -> subprocess.Popen:
        if self.proc is None:
            raise ServeClientError("connect() first")
        return self.proc

    def _read_exact(self, size: int) -> bytes:
        proc = self._require_connected()
        assert proc.stdout is not None
        buf = bytearray()
        while len(buf) < size:
            chunk = proc.stdout.read(size - len(buf))
            if not chunk:
                raise EOFError(
                    f"AI helperがstdoutを閉じました (need={size}, got={len(buf)}, exit={proc.poll()})"
                )
            buf += chunk
        return bytes(buf)

    def _start_stderr_drain(self) -> None:
        proc = self._require_connected()
        if proc.stderr is None:
            return

        def _drain() -> None:
            try:
                for raw in iter(proc.stderr.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                    if not line:
                        continue
                    self.stderr_lines.append(line)
                    if self.log is not None:
                        self.log(line)
                    else:
                        print(line, file=sys.stderr, flush=True)
            except Exception:
                pass

        self._stderr_thread = threading.Thread(target=_drain, name="ueu-stderr", daemon=True)
        self._stderr_thread.start()

    def _failure_message(self, message: str) -> str:
        tail = "\n".join(self.stderr_lines[-12:])
        return f"{message}\n{tail}" if tail else message

    def __enter__(self) -> "ServeClient":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

