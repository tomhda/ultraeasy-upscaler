"""NPU worker image IO helpers."""
from __future__ import annotations

import sys
import types
from pathlib import Path


def test_read_image_uses_unicode_safe_decode(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "日本語 入力.png"
    src.write_bytes(b"image-bytes")
    calls: dict[str, object] = {}

    def fromfile(path, dtype):
        calls["fromfile"] = (path, dtype)
        return b"image-bytes"

    def imdecode(data, flags):
        calls["imdecode"] = (data, flags)
        return object()

    fake_np = types.SimpleNamespace(uint8=object(), fromfile=fromfile)

    fake_cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        imdecode=imdecode,
    )

    monkeypatch.setitem(sys.modules, "numpy", fake_np)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    from app.core import npu_worker

    assert npu_worker._read_image(src) is not None
    assert calls["fromfile"][0] == str(src)
    assert calls["imdecode"][1] == fake_cv2.IMREAD_COLOR


def test_write_image_uses_unicode_safe_encode(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "日本語 出力.png"
    calls: dict[str, object] = {}

    class Encoded:
        def tofile(self, path: str) -> None:
            calls["tofile"] = path
            Path(path).write_bytes(b"encoded")

    def imencode(ext, img):
        calls["imencode"] = (ext, img)
        return True, Encoded()

    fake_cv2 = types.SimpleNamespace(imencode=imencode)

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    from app.core import npu_worker

    image = object()
    npu_worker._write_image(out, image)

    assert out.read_bytes() == b"encoded"
    assert calls["imencode"] == (".png", image)
    assert calls["tofile"] == str(out)
