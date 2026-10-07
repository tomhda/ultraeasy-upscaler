"""CLI (app/cli.py) のテスト。実際の推論・補間・変換は回さない。

engine.process_job / upscaler.upscale_image / npu_prepare.convert などは
差し替え、CLI は app.cli.main(argv) を直接呼ぶ。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from app import cli
from app.core import helper_backend


def _make_png(path: Path, size: tuple[int, int] = (64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 120, 200)).save(path, "PNG")
    return path


@pytest.fixture(autouse=True)
def _english_and_gpu(monkeypatch):
    """CLI テストは既定で英語・GPU 解決にする（conftest の ja を上書き）。"""
    monkeypatch.setenv("UEU_LANG", "en")
    monkeypatch.setattr(
        helper_backend, "_winml_helper", lambda: Path("C:/fake/winml-sr.exe"))
    from app.core import binaries
    monkeypatch.setattr(binaries, "ffmpeg_exe", lambda: "C:/fake/ffmpeg.exe")
    monkeypatch.setattr(binaries, "ffprobe_exe", lambda: "C:/fake/ffprobe.exe")
    monkeypatch.setattr(
        binaries, "available_interpolation_models",
        lambda: ["rife-v4.6", binaries.FILM_MODEL])


def _run(argv: list[str], capsys):
    code = cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def _json_out(out: str) -> dict:
    assert out.strip().count("\n") == 0, f"stdout is not one JSON: {out!r}"
    return json.loads(out)


def _fail_copy_upscale(monkeypatch, fail_on: str = ""):
    """upscaler.upscale_image の差し替え（入力をそのまま写す）。"""
    from app.core import upscaler

    def fake(in_path, out_path, settings, progress=None, cancel=None):
        if Path(in_path).name == fail_on:
            raise RuntimeError("boom")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(Path(in_path).read_bytes())
        if progress:
            progress(1.0, "fake")

    monkeypatch.setattr(upscaler, "upscale_image", fake)


# ------------------------------------------------------------------ status

def test_status_json_shape(monkeypatch, capsys):
    from app.core import addon_kits, binaries

    monkeypatch.setattr(
        cli, "_npu_state", lambda: (False, "NPU kit is not installed."))
    monkeypatch.setattr(
        binaries, "realesrgan_exe", lambda: "C:/fake/realesrgan.exe")
    for name in ("film_kit_installed", "npu_kit_installed",
                 "adcsr_gpu_installed", "adcsr_npu_installed"):
        monkeypatch.setattr(addon_kits, name, lambda: False)

    code, out, _err = _run(["status", "--json"], capsys)
    assert code == 0
    payload = _json_out(out)
    assert payload["ok"] is True
    assert payload["version"] == "0.11.0"
    assert isinstance(payload["app_root"], str)
    assert payload["ffmpeg"] == {"found": True, "path": "C:/fake/ffmpeg.exe"}
    assert payload["backends"]["gpu"] == {"available": True}
    assert payload["backends"]["npu"]["available"] is False
    assert isinstance(payload["backends"]["npu"]["reason"], str)
    assert payload["backends"]["vulkan"] == {"available": True}
    assert payload["kits"] == {
        "film": False, "npu": False, "adcsr_gpu": False, "adcsr_npu": False}


def test_status_ffmpeg_missing(monkeypatch, capsys):
    from app.core import binaries

    monkeypatch.setattr(
        binaries, "ffmpeg_exe",
        lambda: (_ for _ in ()).throw(RuntimeError("no ffmpeg")))
    code, out, _err = _run(["status", "--json"], capsys)
    assert code == 0
    payload = _json_out(out)
    assert payload["ffmpeg"] == {"found": False, "path": None}


# ------------------------------------------------------------------ models

def test_models_json_shape(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_model_available", lambda _b, _k: True)
    code, out, _err = _run(["models", "--backend", "gpu", "--json"], capsys)
    assert code == 0
    payload = _json_out(out)
    assert payload["backend"] == "gpu"
    assert len(payload["upscale"]) == 5
    for entry in payload["upscale"]:
        assert isinstance(entry["key"], str)
        assert isinstance(entry["name"], str) and entry["name"]
        assert isinstance(entry["image"], bool)
        assert isinstance(entry["video"], bool)
        assert entry["scales"] == [4]
        assert isinstance(entry["available"], bool)
        assert isinstance(entry["note"], str) and entry["note"]
        assert "converted" not in entry
    by_key = {e["key"]: e for e in payload["upscale"]}
    assert by_key["animevideov3"]["video"] is True
    assert by_key["AdcSR"]["video"] is False
    assert [e["key"] for e in payload["interpolation"]] == [
        "rife-v4.6", "film-style"]
    for entry in payload["interpolation"]:
        assert isinstance(entry["available"], bool)
        assert isinstance(entry["note"], str) and entry["note"]


def test_models_npu_has_converted(monkeypatch, capsys):
    from app.core import helper_backend as hb

    monkeypatch.setattr(hb, "npu_compiled", lambda _k: False)
    code, out, _err = _run(["models", "--backend", "npu", "--json"], capsys)
    assert code == 0
    payload = _json_out(out)
    assert payload["backend"] == "npu"
    for entry in payload["upscale"]:
        assert entry["converted"] is False


def test_models_keys_work_in_run(monkeypatch, capsys, tmp_path):
    """models の key は run --dry-run で通る（画像・補間とも）。"""
    from app.core import binaries

    monkeypatch.setattr(cli, "_model_available", lambda _b, _k: True)
    _code, out, _err = _run(["models", "--backend", "gpu", "--json"], capsys)
    payload = json.loads(out)
    src = _make_png(tmp_path / "a.png")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video")
    for entry in payload["upscale"]:
        code, _o, _e = _run(
            ["run", str(src), "--model", entry["key"], "--dry-run"], capsys)
        assert code == 0, entry["key"]
    for entry in payload["interpolation"]:
        code, _o, _e = _run(
            ["run", str(clip), "--interpolation", entry["key"],
             "--dry-run", "--json"], capsys)
        assert code == 0, entry["key"]


# -------------------------------------------------------------------- info

def test_info_image_json(tmp_path, capsys):
    src = _make_png(tmp_path / "a.png", (100, 80))
    code, out, _err = _run(["info", str(src), "--json"], capsys)
    assert code == 0
    payload = _json_out(out)
    assert payload == {
        "ok": True, "path": str(src.resolve()), "kind": "image",
        "width": 100, "height": 80, "fps": None, "frames": None,
        "duration": None, "audio": False,
    }


def test_info_missing_is_usage_error(capsys):
    code, out, err = _run(["info", "no-such-file.png", "--json"], capsys)
    assert code == 2
    payload = _json_out(out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "input_not_found"
    assert payload["error"]["fix"]


# --------------------------------------------------------------------- run

def test_run_image_success_json(monkeypatch, tmp_path, capsys):
    _fail_copy_upscale(monkeypatch)
    src = _make_png(tmp_path / "a.png", (64, 48))
    code, out, err = _run(
        ["run", str(src), "--model", "animevideov3", "--json"], capsys)
    assert code == 0
    payload = _json_out(out)
    assert payload["ok"] is True
    assert payload["dry_run"] is False
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["status"] == "done"
    assert result["kind"] == "image"
    assert result["input"] == str(src.resolve())
    assert result["output"].endswith("a_x4.png")
    assert Path(result["output"]).is_file()
    assert Path(result["output"]).stat().st_size > 0
    assert result["width"] == 64 and result["height"] == 48
    assert isinstance(result["seconds"], (int, float))
    assert result["settings"] == {
        "backend": "gpu", "model": "animevideov3", "scale": 4,
        "interpolation": None, "factor": None, "fps": None,
    }
    assert "[1/1]" in err  # progress goes to stderr, not stdout


def test_run_continues_after_failure(monkeypatch, tmp_path, capsys):
    _fail_copy_upscale(monkeypatch, fail_on="b.png")
    inputs = [_make_png(tmp_path / name) for name in ("a.png", "b.png", "c.png")]
    code, out, _err = _run(
        ["run", *(str(p) for p in inputs), "--json"], capsys)
    assert code == 1
    payload = _json_out(out)
    assert payload["ok"] is False
    assert [r["status"] for r in payload["results"]] == [
        "done", "failed", "done"]
    failed = payload["results"][1]
    assert failed["error"]["code"]
    assert failed["error"]["message"]
    assert failed["error"]["fix"]


def test_run_dry_run_matches_real_with_numbering(
        monkeypatch, tmp_path, capsys):
    """dry-run は作らず、計画の出力先が本番と一致する（番号付け含む）。"""
    _fail_copy_upscale(monkeypatch)
    src = _make_png(tmp_path / "a.png")
    before = sorted(p for p in tmp_path.rglob("*"))

    code, out, _err = _run(["run", str(src), "--dry-run", "--json"], capsys)
    assert code == 0
    planned = _json_out(out)["results"][0]["output"]
    assert planned.endswith(str(Path("upscaled") / "a_x4.png"))
    assert sorted(p for p in tmp_path.rglob("*")) == before

    code, out, _err = _run(["run", str(src), "--json"], capsys)
    assert code == 0
    assert _json_out(out)["results"][0]["output"] == planned

    # 同名あり: 2 回目は (1) 付きで一致する。
    code, out, _err = _run(["run", str(src), "--dry-run", "--json"], capsys)
    assert code == 0
    planned2 = _json_out(out)["results"][0]["output"]
    assert planned2.endswith(str(Path("upscaled") / "a_x4(1).png"))
    code, out, _err = _run(["run", str(src), "--json"], capsys)
    assert code == 0
    assert _json_out(out)["results"][0]["output"] == planned2


def test_run_image_ignores_interpolation_with_note(
        monkeypatch, tmp_path, capsys):
    _fail_copy_upscale(monkeypatch)
    src = _make_png(tmp_path / "a.png")
    code, out, _err = _run(
        ["run", str(src), "--interpolation", "rife-v4.6", "--json"], capsys)
    assert code == 0
    result = _json_out(out)["results"][0]
    assert result["status"] == "done"
    assert result["note"] == "Interpolation is ignored for images."
    assert result["settings"]["interpolation"] is None


@pytest.mark.parametrize("argv_extra,model_args,video,code_name", [
    (["no-such.png"], [], False, "input_not_found"),
    (["{img}"], ["--model", "no-such-model"], False, "model_unknown"),
    (["{vid}"], ["--model", "AdcSR"], True, "model_not_for_video"),
    (["{vid}"], ["--interpolation", "film-style", "--fps", "60"],
     True, "fps_not_multiple"),
    (["{vid}"], ["--factor", "4", "--fps", "60"],
     True, "factor_fps_conflict"),
])
def test_run_validation_processes_nothing(
        monkeypatch, tmp_path, capsys, argv_extra, model_args, video,
        code_name):
    """検証の誤りでは何も処理しない（終了コード 2）。"""
    from app.core import engine

    def _boom(*_a, **_k):
        raise AssertionError("must not process")

    monkeypatch.setattr(engine, "process_job", _boom)
    img = _make_png(tmp_path / "a.png")
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"fake-video")
    argv = ["run"] + [
        a.format(img=img, vid=vid) for a in argv_extra] + model_args
    code, out, _err = _run(argv + ["--json"], capsys)
    assert code == 2
    payload = _json_out(out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == code_name
    assert payload["error"]["fix"]


def test_run_npu_not_converted_processes_nothing(
        monkeypatch, tmp_path, capsys):
    from app.core import engine, media, npu_prepare

    monkeypatch.setattr(engine, "process_job",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            AssertionError("must not process")))
    monkeypatch.setattr(cli, "_npu_state", lambda: (True, None))
    monkeypatch.setattr(npu_prepare, "is_converted", lambda _k: False)
    monkeypatch.setattr(
        media, "probe",
        lambda _p: media.MediaInfo(kind="image", width=512, height=512))
    src = _make_png(tmp_path / "a.png")
    code, out, _err = _run(
        ["run", str(src), "--backend", "npu", "--json"], capsys)
    assert code == 2
    payload = _json_out(out)
    assert payload["error"]["code"] == "npu_not_converted"
    assert payload["error"]["fix"]


def test_run_scale2_needs_vulkan(tmp_path, capsys):
    src = _make_png(tmp_path / "a.png")
    code, out, _err = _run(
        ["run", str(src), "--scale", "2", "--backend", "gpu", "--json"],
        capsys)
    assert code == 2
    assert _json_out(out)["error"]["code"] == "scale_not_supported"


def test_run_overwrite(monkeypatch, tmp_path, capsys):
    """--overwrite なしは番号付け、ありは上書き。"""
    from app.core import upscaler

    def fake(_in, out_path, _settings, progress=None, cancel=None):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"new")

    monkeypatch.setattr(upscaler, "upscale_image", fake)
    src = _make_png(tmp_path / "a.png")
    first = tmp_path / "upscaled" / "a_x4.png"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"old")

    code, out, _err = _run(["run", str(src), "--json"], capsys)
    assert code == 0
    numbered = _json_out(out)["results"][0]["output"]
    assert numbered.endswith("a_x4(1).png")
    assert first.read_bytes() == b"old"

    code, out, _err = _run(
        ["run", str(src), "--overwrite", "--json"], capsys)
    assert code == 0
    assert _json_out(out)["results"][0]["output"] == str(first.resolve())
    assert first.read_bytes() == b"new"


def test_run_video_success_json(monkeypatch, tmp_path, capsys):
    from app.core import media, video

    monkeypatch.setattr(
        media, "probe",
        lambda _p: media.MediaInfo(
            kind="video", width=16, height=16, fps=30.0, has_audio=True))

    def fake_pipe(_src, out_path, _settings, progress=None, cancel=None):
        Path(out_path).write_bytes(b"video")
        if progress:
            progress(1.0, "fake")

    monkeypatch.setattr(video, "upscale_video_piped", fake_pipe)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video")
    code, out, _err = _run(["run", str(clip), "--json"], capsys)
    assert code == 0
    result = _json_out(out)["results"][0]
    assert result["status"] == "done"
    assert result["kind"] == "video"
    assert result["output"].endswith(".mp4")
    assert result["width"] == 16 and result["height"] == 16
    assert result["fps"] == 30.0


# -------------------------------------------------------------- quick-check

def test_quick_check_image(monkeypatch, tmp_path, capsys):
    _fail_copy_upscale(monkeypatch)
    src = _make_png(tmp_path / "a.png", (100, 80))
    out = tmp_path / "check.png"
    code, out_s, _err = _run(
        ["quick-check", str(src), "--out", str(out), "--json"], capsys)
    assert code == 0
    payload = _json_out(out_s)
    assert payload["ok"] is True
    assert payload["output"] == str(out.resolve())
    assert payload["source_frame"] == str(src.resolve())
    assert (payload["width"], payload["height"]) == (100, 80)
    assert isinstance(payload["seconds"], (int, float))
    assert payload["settings"]["model"] == "animevideov3"


def test_quick_check_video_rect(monkeypatch, tmp_path, capsys):
    from app.core import trial as trial_core
    from app.core import upscaler

    def fake_extract(_video, _sec, out_png):
        _make_png(Path(out_png), (100, 80))

    def fake_upscale(in_path, out_path, _settings, progress=None, cancel=None):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(Path(in_path).read_bytes())

    monkeypatch.setattr(trial_core, "extract_frame",
                        lambda v, s, o: fake_extract(v, s, o))
    monkeypatch.setattr(upscaler, "upscale_image", fake_upscale)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video")
    out = tmp_path / "check.png"
    code, out_s, _err = _run(
        ["quick-check", str(clip), "--out", str(out),
         "--rect", "10,10,50,40", "--json"], capsys)
    assert code == 0
    payload = _json_out(out_s)
    assert payload["source_frame"].endswith("source_frame.png")
    assert (payload["width"], payload["height"]) == (50, 40)


def test_quick_check_existing_out_needs_overwrite(tmp_path, capsys):
    src = _make_png(tmp_path / "a.png")
    out = _make_png(tmp_path / "check.png")
    code, out_s, _err = _run(
        ["quick-check", str(src), "--out", str(out), "--json"], capsys)
    assert code == 2
    assert _json_out(out_s)["error"]["code"] == "output_exists"


# -------------------------------------------------------------- npu-convert

def test_npu_convert_already_converted(monkeypatch, capsys):
    from app.core import npu_prepare

    seen = []
    monkeypatch.setattr(cli, "_npu_state", lambda: (True, None))
    monkeypatch.setattr(npu_prepare, "is_converted", lambda _k: True)
    monkeypatch.setattr(
        npu_prepare, "convert", lambda *a, **k: seen.append((a, k)))
    code, out, _err = _run(["npu-convert", "animevideov3", "--json"], capsys)
    assert code == 0
    assert _json_out(out) == {
        "ok": True, "model": "animevideov3", "converted": True,
        "already_converted": True,
    }
    assert seen == []


def test_npu_convert_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "_npu_state", lambda: (False, "NPU kit is not installed."))
    code, out, _err = _run(["npu-convert", "animevideov3", "--json"], capsys)
    assert code == 1
    payload = _json_out(out)
    assert payload["error"]["code"] == "npu_unavailable"
    assert payload["error"]["fix"]


# ------------------------------------------------------------------ wording

def test_default_is_english(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("UEU_LANG", raising=False)
    code, _out, err = _run(["info", "no-such-file.png"], capsys)
    assert code == 2
    assert "error: Input not found:" in err
    assert err.count("fix: ") == 1


def test_lang_ja_core_wording(monkeypatch, capsys):
    """--lang ja で core の文言が日本語になる（models の note で確認）。"""
    monkeypatch.setattr(cli, "_model_available", lambda _b, _k: True)
    code, out, _err = _run(
        ["models", "--backend", "gpu", "--lang", "ja", "--json"], capsys)
    assert code == 0
    notes = [e["note"] for e in json.loads(out)["upscale"]]
    assert "アニメ向け・速い" in notes
    code, out, _err = _run(["models", "--backend", "gpu", "--json"], capsys)
    assert code == 0
    notes = [e["note"] for e in json.loads(out)["upscale"]]
    assert "For anime · Fast" in notes


def test_help_has_args_defaults_examples(capsys):
    for argv in (["status", "--help"], ["models", "--help"],
                 ["info", "--help"], ["run", "--help"],
                 ["quick-check", "--help"], ["npu-convert", "--help"]):
        code, out, _err = _run(argv, capsys)
        assert code == 0, argv
        assert "Example" in out, argv


def test_version(capsys):
    code, out, _err = _run(["--version"], capsys)
    assert code == 0
    assert "0.11.0" in out


# ----------------------------------------------------------------------- env

def test_cli_does_not_import_qt():
    """app.cli を import しても Qt (PySide6) が読み込まれないこと。"""
    script = (
        "import sys; "
        "assert 'PySide6' not in sys.modules; "
        "import app.cli; "
        "assert 'PySide6' not in sys.modules, "
        "[m for m in sys.modules if 'PySide' in m]; "
        "print('no-qt-ok')"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert "no-qt-ok" in done.stdout


def test_frozen_cli_resolves_next_to_cli_exe(monkeypatch, tmp_path):
    """CLI exe 版でも同梱の models/ や vendor/ を exe の隣から探す。"""
    from app.core import binaries, settings

    exe = tmp_path / "ultraeasy-upscaler-cli.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert settings._app_root() == tmp_path.resolve()
    assert binaries.repo_root() == tmp_path.resolve()
