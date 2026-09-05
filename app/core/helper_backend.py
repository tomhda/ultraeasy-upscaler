"""DirectML GPU / Ryzen AI NPUのUEU常駐ヘルパーバックエンド。

GUIから選ばれる新NPU経路は、旧 ``npu_worker`` ではなく
``tools/npu-serve/npu_serve.py`` を常駐させて使う。
"""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from . import binaries, jobs, media
from .jobs import ProgressCb
from .serve_client import ServeClient, ServeClientError
from .settings import (
    DEFAULT_HELPER_MODEL,
    DEFAULT_MODEL,
    DEFAULT_MODELS_DIR,
    DEFAULT_NPU_CACHE_DIR,
    DEFAULT_VENDOR_MODELS_DIR,
    HELPER_MODEL_FILES,
    HELPER_MODEL_AMD_RRDB,
    HELPER_MODEL_SWINIR,
    canonical_helper_model,
    ModelFamily,
    UpscaleBackend,
    UpscaleSettings,
)

WINML_HELPER_ENV = "UEU_WINML_HELPER"
MODELS_DIR_ENV = "UEU_MODELS_DIR"
NPU_PYTHON_ENV = "UEU_NPU_PYTHON"
NPU_CACHE_ENV = "UEU_NPU_CACHE"
OVERLAP = 16
HELPER_BACKENDS = frozenset({UpscaleBackend.WINML_GPU, UpscaleBackend.NPU_NATIVE})


class HelperBackendUnavailable(RuntimeError):
    """モデル解決またはヘルパー起動に失敗し、Vulkan退避が必要。"""


@dataclass
class HelperSession:
    client: ServeClient
    backend: UpscaleBackend
    model_path: Path
    cache_hit: bool = True

    @property
    def scale(self) -> int:
        return self.client.scale

    def upscale(self, image: np.ndarray, cancel=None) -> np.ndarray:
        return self.client.upscale(image, cancel=cancel)

    def close(self, *, force: bool = False) -> None:
        self.client.close(force=force)


def _noop(_fraction: float, _message: str) -> None:
    pass


def is_helper_backend(backend: UpscaleBackend) -> bool:
    return backend in HELPER_BACKENDS


def models_dir() -> Path:
    return Path(os.environ.get(MODELS_DIR_ENV, str(DEFAULT_MODELS_DIR))).expanduser()


def npu_cache_dir() -> Path:
    return Path(os.environ.get(NPU_CACHE_ENV, str(DEFAULT_NPU_CACHE_DIR))).expanduser()


def _model_key(settings: UpscaleSettings) -> str:
    """設定からhelperの具体的なモデルキーを取り出す。

    現行統合版は ``model_family`` に抽象値を保存していたため、
    ``model=DEFAULT_MODEL`` の旧形式だけは系統値を優先して読み戻す。
    新形式では ``model`` が具体的なキーなので、表示名では解決しない。
    """
    model_key = canonical_helper_model(settings.model)
    if settings.model in (None, DEFAULT_MODEL):
        legacy_key = canonical_helper_model(settings.model_family)
        if legacy_key is not None:
            model_key = legacy_key
    if model_key is None:
        model_key = DEFAULT_HELPER_MODEL
    return model_key


def _discarded_pixels(width: int, height: int, tile: int, overlap: int = OVERLAP) -> int:
    core = tile - 2 * overlap
    padded_w = math.ceil(width / core) * core
    padded_h = math.ceil(height / core) * core
    return padded_w * padded_h - width * height


def choose_gpu_tile(width: int, height: int) -> int:
    """256/512タイルのコア合成後に捨てる画素数が少ない方を選ぶ。"""
    scores = {tile: _discarded_pixels(width, height, tile) for tile in (256, 512)}
    return min(scores, key=lambda tile: (scores[tile], tile))


def effective_backend(backend: UpscaleBackend, width: int, height: int) -> UpscaleBackend:
    if backend == UpscaleBackend.NPU_NATIVE and min(width, height) < 480:
        return UpscaleBackend.WINML_GPU
    return backend


def _resolve_model(backend: UpscaleBackend, model: str | ModelFamily, tile: int) -> Path:
    model_key = canonical_helper_model(model)
    try:
        filename = HELPER_MODEL_FILES[backend][model_key][tile]
    except KeyError as exc:
        raise HelperBackendUnavailable(
            f"対応モデルがありません: backend={backend.value}, model={model_key}, tile={tile}"
        ) from exc

    root = models_dir()
    root_dirs = (root, root / "span")
    search_dirs = (
        (*root_dirs, DEFAULT_VENDOR_MODELS_DIR)
        if MODELS_DIR_ENV in os.environ
        else (DEFAULT_VENDOR_MODELS_DIR, *root_dirs)
    )
    candidates = tuple(search_dir / filename for search_dir in search_dirs)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise HelperBackendUnavailable(
        f"AIモデルが見つかりません: {filename}\n"
        f"探索先: {', '.join(str(search_dir) for search_dir in search_dirs)}"
    )


def _winml_helper() -> Path:
    override = os.environ.get(WINML_HELPER_ENV)
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise HelperBackendUnavailable(f"{WINML_HELPER_ENV} のファイルが見つかりません: {candidate}")

    helper_root = binaries.repo_root() / "tools" / "winml-sr" / "bin"
    patterns = (
        "Release/net*/win-x64/winml-sr.exe",
        "Release/net*/win-arm64/winml-sr.exe",
        "Debug/net*/win-x64/winml-sr.exe",
        "Debug/net*/win-arm64/winml-sr.exe",
    )
    for pattern in patterns:
        matches = sorted(helper_root.glob(pattern), reverse=True)
        if matches:
            return matches[0].resolve()

    found = shutil.which("winml-sr.exe") or shutil.which("winml-sr")
    if found:
        return Path(found).resolve()
    raise HelperBackendUnavailable(
        "winml-sr.exeが見つかりません。tools/winml-srをビルドするか、"
        f"{WINML_HELPER_ENV}を指定してください。"
    )


def _npu_python() -> Path:
    default = Path.home() / "miniforge3" / "envs" / "ryzen-ai-1.8.0" / "python.exe"
    candidate = Path(os.environ.get(NPU_PYTHON_ENV, str(default))).expanduser()
    if not candidate.is_file():
        raise HelperBackendUnavailable(f"NPU用Pythonが見つかりません: {candidate}")
    return candidate.resolve()


def _npu_script() -> Path:
    script = binaries.repo_root() / "tools" / "npu-serve" / "npu_serve.py"
    if not script.is_file():
        raise HelperBackendUnavailable(f"NPUワーカーが見つかりません: {script}")
    return script.resolve()


def _cache_hit(model_path: Path) -> bool:
    cache = npu_cache_dir() / f"modelcachekey_{model_path.stem}"
    return (cache / "context.json").is_file() and any(cache.rglob("*.rai"))


def _helper_env() -> dict[str, str]:
    """常駐ヘルパーへ渡す環境を作る。

    .NET SDK/runtimeをユーザー領域へ入れたWindowsでは、親プロセスが古い
    PATHを保持したまま起動されることがあるため、標準のユーザー配置先を
    見つけた場合だけDOTNET_ROOTを補う。
    """
    env = os.environ.copy()
    dotnet_root = Path.home() / ".dotnet"
    if "DOTNET_ROOT" not in env and (dotnet_root / "dotnet.exe").is_file():
        env["DOTNET_ROOT"] = str(dotnet_root)
        env.setdefault("DOTNET_ROOT_X64", str(dotnet_root))
        env["PATH"] = str(dotnet_root) + os.pathsep + env.get("PATH", "")
    env.setdefault("PYTHONUTF8", "1")
    return env


def _session_spec(
    settings: UpscaleSettings, width: int, height: int
) -> tuple[UpscaleBackend, int, Path]:
    if int(settings.scale) != 4:
        raise HelperBackendUnavailable("新しいGPU/NPUバックエンドは4xモデル専用です。倍率を4xにしてください。")
    requested = settings.backend
    backend = effective_backend(requested, width, height)
    model_key = _model_key(settings)
    if model_key in (HELPER_MODEL_AMD_RRDB, HELPER_MODEL_SWINIR):
        tile = 256
    else:
        tile = 512 if backend == UpscaleBackend.NPU_NATIVE else choose_gpu_tile(width, height)
    model_path = _resolve_model(backend, model_key, tile)
    return backend, tile, model_path


def open_session(
    settings: UpscaleSettings,
    width: int,
    height: int,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> HelperSession:
    """入力寸法に合うモデルとヘルパーを解決してUEUHまで接続する。"""
    progress = progress or _noop
    try:
        backend, _tile, model_path = _session_spec(settings, width, height)
        if settings.backend == UpscaleBackend.NPU_NATIVE and backend == UpscaleBackend.WINML_GPU:
            progress(0.0, "短辺480px未満のためGPUへ自動切替…")
        else:
            progress(0.0, "AI準備中…")

        cache_hit = True
        if backend == UpscaleBackend.WINML_GPU:
            helper = _winml_helper()
            command = [
                str(helper), "serve", "--model", str(model_path),
                "--ep-name", "DmlExecutionProvider",
                "--overlap", str(OVERLAP), "--warmup", "1",
            ]
            workdir = helper.parent
            timeout = 120.0
        else:
            python = _npu_python()
            script = _npu_script()
            cache = npu_cache_dir()
            cache_hit = _cache_hit(model_path)
            command = [
                str(python), str(script), "--model", str(model_path),
                "--cache-dir", str(cache),
                "--overlap", str(OVERLAP), "--warmup", "1",
            ]
            workdir = binaries.repo_root()
            # 初回VAIMLコンパイルはモデル次第で長い（av3dp512: 約15分、SwinIR-M: 約51分）。
            # キャッシュ有無でタイムアウトを分け、初回コンパイルを打ち切らない。
            timeout = 15 * 60.0 if cache_hit else 120 * 60.0
            if not cache_hit:
                progress(0.0, "初回のみNPU最適化中（数分〜1時間・次回から数秒）")

        env = _helper_env()
        client = ServeClient(command, workdir, env=env)

        last_second = -1

        def _waiting(elapsed: float) -> None:
            nonlocal last_second
            second = int(elapsed)
            if second == last_second:
                return
            last_second = second
            if backend == UpscaleBackend.NPU_NATIVE and not cache_hit:
                progress(0.0, "初回のみNPU最適化中（数分〜1時間・次回から数秒）")
            else:
                progress(0.0, "AI準備中…")

        client.connect(timeout=timeout, progress=_waiting, cancel=cancel)
        return HelperSession(client, backend, model_path, cache_hit)
    except jobs.Cancelled:
        raise
    except HelperBackendUnavailable:
        raise
    except (ServeClientError, OSError, ValueError) as exc:
        raise HelperBackendUnavailable(str(exc) or exc.__class__.__name__) from exc


def _load_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def upscale_image(
    in_path: str,
    out_path: str,
    settings: UpscaleSettings,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> None:
    progress = progress or _noop
    image = _load_rgb(in_path)
    height, width = image.shape[:2]
    session = open_session(settings, width, height, progress, cancel)
    try:
        if cancel is not None and cancel.is_set():
            raise jobs.Cancelled()
        progress(0.1, "アップスケール中…")
        output = session.upscale(image, cancel=cancel) if cancel is not None else session.upscale(image)
        if cancel is not None and cancel.is_set():
            raise jobs.Cancelled()
        _save_rgb(output, Path(out_path))
    finally:
        session.close(force=cancel is not None and cancel.is_set())
    progress(1.0, "完了")


def upscale_folder(
    in_dir: str,
    out_dir: str,
    settings: UpscaleSettings,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> None:
    progress = progress or _noop
    source = Path(in_dir)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    images = [
        path for path in sorted(source.iterdir())
        if path.is_file() and path.suffix.lower() in media.IMAGE_EXTS
    ]
    total = len(images)
    if not images:
        progress(1.0, "0/0 枚")
        return

    sessions: dict[tuple[UpscaleBackend, Path], HelperSession] = {}
    fmt = (settings.image_format or "png").lower().replace("jpeg", "jpg")
    try:
        for index, path in enumerate(images, start=1):
            if cancel is not None and cancel.is_set():
                raise jobs.Cancelled()
            image = _load_rgb(str(path))
            height, width = image.shape[:2]
            backend, _tile, model_path = _session_spec(settings, width, height)
            key = (backend, model_path)
            session = sessions.get(key)
            if session is None:
                session = open_session(
                    settings, width, height,
                    progress=lambda _f, message, i=index: progress(
                        (i - 1) / total, f"{i}/{total} 枚 {message}"
                    ),
                    cancel=cancel,
                )
                sessions[key] = session
            output = session.upscale(image, cancel=cancel) if cancel is not None else session.upscale(image)
            _save_rgb(output, target / f"{path.stem}.{fmt}")
            progress(index / total, f"{index}/{total} 枚")
    finally:
        force = cancel is not None and cancel.is_set()
        for session in sessions.values():
            session.close(force=force)
