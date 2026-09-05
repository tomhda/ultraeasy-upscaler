"""UEUバイナリプロトコルのモックpipe単体テスト。"""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from app.core import serve_client


class _FakeProc:
    def __init__(self, stdout_bytes: bytes, stderr_bytes: bytes = b"") -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO(stderr_bytes)
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        del timeout
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def _ready(scale: int = 4, tile: int = 256) -> bytes:
    return b"UEUH" + struct.pack("<iii", scale, tile, tile)


def _data(image: np.ndarray) -> bytes:
    height, width = image.shape[:2]
    return b"UEUD" + struct.pack("<ii", width, height) + image.tobytes()


def test_ueuh_ueuf_ueud_roundtrip(monkeypatch, tmp_path) -> None:
    output = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    fake = _FakeProc(_ready() + _data(output), b"helper log\n")
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return fake

    monkeypatch.setattr(serve_client.subprocess, "Popen", fake_popen)
    client = serve_client.ServeClient(["fake-helper", "serve"], tmp_path)
    client.connect(timeout=1)

    source = np.arange(2 * 1 * 3, dtype=np.uint8).reshape(1, 2, 3)
    actual = client.upscale(source)

    assert client.scale == 4
    assert (client.tile_w, client.tile_h) == (256, 256)
    assert np.array_equal(actual, output)
    assert fake.stdin.getvalue() == b"UEUF" + struct.pack("<ii", 2, 1) + source.tobytes()
    assert calls[0][0] == ["fake-helper", "serve"]
    assert calls[0][1]["cwd"] == str(tmp_path)
    client.close()


def test_ueue_becomes_value_error_and_next_frame_can_continue(monkeypatch) -> None:
    message = "bad frame".encode("utf-8")
    output = np.full((4, 4, 3), 17, dtype=np.uint8)
    stream = _ready(tile=512) + b"UEUE" + struct.pack("<i", len(message)) + message + _data(output)
    fake = _FakeProc(stream)
    monkeypatch.setattr(serve_client.subprocess, "Popen", lambda *_a, **_kw: fake)
    client = serve_client.ServeClient(["fake-helper"])
    client.connect(timeout=1)
    source = np.zeros((1, 1, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="bad frame"):
        client.upscale(source)
    actual = client.upscale(source)

    assert np.array_equal(actual, output)
    expected_frame = b"UEUF" + struct.pack("<ii", 1, 1) + source.tobytes()
    assert fake.stdin.getvalue() == expected_frame * 2
    client.close()


def test_rejects_non_rgb_uint8_before_writing(monkeypatch) -> None:
    fake = _FakeProc(_ready())
    monkeypatch.setattr(serve_client.subprocess, "Popen", lambda *_a, **_kw: fake)
    client = serve_client.ServeClient(["fake-helper"])
    client.connect(timeout=1)

    with pytest.raises(ValueError, match="HxWx3 uint8"):
        client.send(np.zeros((4, 4), dtype=np.uint8))
    assert fake.stdin.getvalue() == b""
    client.close()



def test_receive_cancel_terminates_blocked_helper():
    class BlockingStdout:
        def read(self, _size):
            threading.Event().wait()
            return b""

    class BlockingProc(_FakeProc):
        def __init__(self):
            super().__init__(b"")
            self.stdout = BlockingStdout()
        def terminate(self):
            self.returncode = -15
        def wait(self, timeout=None):
            del timeout
            return self.returncode

    import threading
    fake = BlockingProc()
    client = serve_client.ServeClient(["fake-helper"])
    client.proc = fake
    cancel = threading.Event()
    thread = threading.Thread(target=lambda: (threading.Event().wait(0.1), cancel.set()))
    thread.start()
    with pytest.raises(serve_client.Cancelled):
        client.receive(cancel=cancel)
    thread.join()
    assert fake.returncode == -15
    assert client.proc is None
