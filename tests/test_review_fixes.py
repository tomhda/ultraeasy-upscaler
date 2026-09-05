"""レビュー指摘（EXIF向き / 透過 / ICC / フォルダ衝突 / 一時名 / HDR拒否）の回帰テスト。"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageCms

from app.core import engine, helper_backend, media, video
from app.core.settings import UpscaleBackend, UpscaleSettings


def _exif_orientation(value: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = value
    return exif.tobytes()


def test_load_image_applies_exif_orientation(tmp_path):
    src = tmp_path / "exif6.jpg"
    Image.new("RGB", (64, 96), (10, 20, 30)).save(src, exif=_exif_orientation(6), quality=95)

    loaded = helper_backend._load_image(str(src))

    # Orientation=6（90度CW）は幅高が入れ替わる
    assert loaded.rgb.shape == (64, 96, 3)
    assert loaded.alpha is None


def test_save_rgb_keeps_alpha_and_flattens_for_jpeg(tmp_path):
    src = tmp_path / "alpha.png"
    arr = np.zeros((32, 32, 4), np.uint8)
    arr[:, :, :3] = (255, 0, 0)
    arr[:, :, 3] = 255
    arr[:, :16] = (0, 0, 255, 0)  # 左半分は透明
    Image.fromarray(arr, "RGBA").save(src)

    loaded = helper_backend._load_image(str(src))
    assert loaded.alpha is not None
    upscaled = np.full((64, 64, 3), 128, np.uint8)

    out_png = tmp_path / "out.png"
    helper_backend._save_rgb(upscaled, out_png, alpha=loaded.alpha)
    with Image.open(out_png) as image:
        assert image.mode == "RGBA"
        alpha = np.asarray(image)[..., 3]
    assert alpha[:, :24].max() == 0
    assert alpha[:, 40:].min() == 255

    out_jpg = tmp_path / "out.jpg"
    helper_backend._save_rgb(upscaled, out_jpg, alpha=loaded.alpha)
    with Image.open(out_jpg) as image:
        assert image.mode == "RGB"
        assert image.getpixel((4, 4)) == (255, 255, 255)  # 透明部は白背景


def test_icc_profile_is_carried_to_output(tmp_path):
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    src = tmp_path / "icc.png"
    Image.new("RGB", (16, 16), (10, 200, 30)).save(src, icc_profile=icc)

    loaded = helper_backend._load_image(str(src))
    assert loaded.icc_profile

    out = tmp_path / "out.png"
    helper_backend._save_rgb(np.zeros((32, 32, 3), np.uint8), out, icc_profile=loaded.icc_profile)
    with Image.open(out) as image:
        assert image.info.get("icc_profile") == loaded.icc_profile


def test_folder_outputs_resolve_batch_and_existing_collisions(tmp_path):
    target = tmp_path / "out"
    target.mkdir()
    (target / "a.png").write_bytes(b"KEEP")
    images = [tmp_path / "a.jpg", tmp_path / "a.png"]

    kept = helper_backend._folder_outputs(images, target, "png", overwrite=False)
    assert kept[images[0]] == target / "a(1).png"
    assert kept[images[1]] == target / "a(2).png"

    replaced = helper_backend._folder_outputs(images, target, "png", overwrite=True)
    assert replaced[images[0]] == target / "a.png"
    assert replaced[images[1]] == target / "a(1).png"


def test_part_path_is_unique_and_leaves_existing_part_alone(tmp_path):
    out = tmp_path / "x_x4.png"
    stale = tmp_path / "x_x4.part.png"
    stale.write_bytes(b"OLD")

    first = engine._part_path(out)
    second = engine._part_path(out)
    try:
        assert first != second
        assert first.parent == out.parent and first.suffix == ".png"
        assert first.name.startswith("x_x4.part-")
        assert stale.read_bytes() == b"OLD"
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def _hdr_probe(_path: str) -> dict:
    return {
        "streams": [{
            "codec_type": "video", "width": 852, "height": 480,
            "avg_frame_rate": "24/1", "nb_frames": "15",
            "color_transfer": "smpte2084", "color_primaries": "bt2020",
        }],
        "format": {"duration": "0.625"},
    }


def test_probe_reads_color_transfer(monkeypatch, tmp_path):
    src = tmp_path / "hdr.mp4"
    src.write_bytes(b"video")
    monkeypatch.setattr(media, "_run_ffprobe", _hdr_probe)
    info = media.probe(str(src))
    assert info.color_transfer == "smpte2084"
    assert info.color_primaries == "bt2020"


def test_video_paths_reject_hdr_before_processing(monkeypatch, tmp_path):
    src = tmp_path / "hdr.mp4"
    src.write_bytes(b"video")
    monkeypatch.setattr(media, "_run_ffprobe", _hdr_probe)
    settings = UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model="animevideov3", scale=4)

    with pytest.raises(ValueError, match="HDR"):
        video.upscale_video_piped(str(src), str(tmp_path / "out.mp4"), settings)
    with pytest.raises(ValueError, match="HDR"):
        video.extract_frames(str(src), str(tmp_path / "frames"))
