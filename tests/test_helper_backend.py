import numpy as np
import pytest
from pathlib import Path

from app.core import helper_backend
from app.core.settings import (
    HELPER_MODEL_AMD_RRDB,
    HELPER_MODEL_ANIME,
    HELPER_MODEL_SPAN,
    HELPER_MODEL_SWINIR,
    HELPER_MODEL_ADCSR,
    UpscaleBackend,
    UpscaleSettings,
    canonical_helper_model,
    helper_model_family,
    vulkan_fallback_settings,
)


def test_gpu_tile_uses_less_discarded_pixel_plan() -> None:
    assert helper_backend.choose_gpu_tile(854, 480) == 512
    assert helper_backend.choose_gpu_tile(640, 360) == 256


def test_npu_short_edge_falls_back_to_gpu() -> None:
    assert (
        helper_backend.effective_backend(UpscaleBackend.NPU_NATIVE, 854, 479)
        == UpscaleBackend.WINML_GPU
    )
    assert (
        helper_backend.effective_backend(UpscaleBackend.NPU_NATIVE, 854, 480)
        == UpscaleBackend.NPU_NATIVE
    )


def test_realesrgan_winml_resolves_256_model_from_env_first(
    monkeypatch, tmp_path
) -> None:
    filename = "realesrgan_nchw_256x256_fp32.onnx"
    env_models = tmp_path / "env-models"
    env_models.mkdir()
    expected = env_models / filename
    expected.touch()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(env_models))

    settings = UpscaleSettings(
        backend=UpscaleBackend.WINML_GPU,
        model=HELPER_MODEL_AMD_RRDB,
    )
    backend, tile, model_path = helper_backend._session_spec(settings, 854, 480)

    assert backend == UpscaleBackend.WINML_GPU
    assert tile == 256
    assert model_path == expected.resolve()


def test_realesrgan_npu_resolves_256_model_from_vendor(
    monkeypatch, tmp_path
) -> None:
    filename = "realesrgan_nchw_256x256_bf16cast.onnx"
    env_models = tmp_path / "env-models"
    vendor_models = tmp_path / "vendor-models"
    env_models.mkdir()
    vendor_models.mkdir()
    expected = vendor_models / filename
    expected.touch()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(env_models))
    monkeypatch.setattr(helper_backend, "DEFAULT_VENDOR_MODELS_DIR", vendor_models)

    settings = UpscaleSettings(
        backend=UpscaleBackend.NPU_NATIVE,
        model=HELPER_MODEL_AMD_RRDB,
    )
    backend, tile, model_path = helper_backend._session_spec(settings, 854, 480)

    assert backend == UpscaleBackend.NPU_NATIVE
    assert tile == 256
    assert model_path == expected.resolve()


def test_winml_helper_uses_renamed_tool_layout(monkeypatch, tmp_path) -> None:
    helper = (
        tmp_path / "tools" / "winml-sr" / "bin" / "Release"
        / "net8.0-windows10.0.22621.0" / "win-x64" / "winml-sr.exe"
    )
    helper.parent.mkdir(parents=True)
    helper.touch()
    monkeypatch.setattr(helper_backend.binaries, "repo_root", lambda: tmp_path)
    monkeypatch.delenv(helper_backend.WINML_HELPER_ENV, raising=False)

    assert helper_backend._winml_helper() == helper.resolve()


def test_adcsr_uses_fixed_128_gpu_tile(monkeypatch) -> None:
    model = helper_backend.binaries.repo_root() / "tmp" / "adcsr" / "onnx" / "adcsr_nchw_128x128_fp32.onnx"
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(model.parent))
    settings = UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model=HELPER_MODEL_ADCSR)
    backend, tile, path = helper_backend._session_spec(settings, 1280, 534)
    assert backend == UpscaleBackend.WINML_GPU
    assert tile == 128
    assert path == model.resolve()

def test_adcsr_npu_request_switches_to_gpu_with_message(monkeypatch) -> None:
    model = helper_backend.binaries.repo_root() / "tmp" / "adcsr" / "onnx" / "adcsr_nchw_128x128_fp32.onnx"
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(model.parent))
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_ADCSR)
    messages = []
    monkeypatch.setattr(helper_backend, "_winml_helper", lambda: model.parent / "missing.exe")
    try:
        helper_backend.open_session(settings, 1280, 534, progress=lambda _f, msg: messages.append(msg))
    except helper_backend.HelperBackendUnavailable:
        pass
    assert "AdcSRはNPU非対応のためGPUで実行…" in messages

def test_adcsr_alias_family_and_vulkan_fallback() -> None:
    from app.core.settings import ModelFamily
    assert canonical_helper_model("adcsr") == "AdcSR"
    assert helper_model_family("adcsr") == ModelFamily.PHOTO
    settings = UpscaleSettings(model="adcsr")
    fallback = vulkan_fallback_settings(settings)
    assert fallback.model == "realesrgan-x4plus"
    assert fallback.backend == UpscaleBackend.VULKAN

def test_overlap_for_model_gives_adcsr_32_and_default_16() -> None:
    assert helper_backend.overlap_for_model(HELPER_MODEL_ADCSR) == 32
    assert helper_backend.overlap_for_model("adcsr") == 32
    assert helper_backend.overlap_for_model(HELPER_MODEL_ANIME) == 16
    assert helper_backend.overlap_for_model(HELPER_MODEL_SPAN) == 16
    assert helper_backend.overlap_for_model(HELPER_MODEL_AMD_RRDB) == 16
    assert helper_backend.overlap_for_model(HELPER_MODEL_SWINIR) == 16
    assert helper_backend.overlap_for_model("unknown-model") == 16
    assert helper_backend.overlap_for_model(None) == 16


class _CapturingServeClient:
    last_command: list = []
    last_connect_kwargs: dict = {}

    def __init__(self, command, workdir, env=None, log=None) -> None:
        type(self).last_command = list(command)

    def connect(self, **kwargs) -> None:
        type(self).last_connect_kwargs = dict(kwargs)
        return None


def _serve_overlap(monkeypatch, settings, width=1280, height=534) -> str:
    from pathlib import Path

    model_dir = helper_backend.binaries.repo_root() / "tmp" / "adcsr" / "onnx"
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(model_dir))
    monkeypatch.setattr(helper_backend, "_winml_helper", lambda: Path("winml-sr.exe"))
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)
    session = helper_backend.open_session(settings, width, height)
    assert session.backend == UpscaleBackend.WINML_GPU
    command = _CapturingServeClient.last_command
    return command[command.index("--overlap") + 1]


def test_adcsr_serve_uses_overlap_32(monkeypatch) -> None:
    settings = UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model=HELPER_MODEL_ADCSR)
    assert _serve_overlap(monkeypatch, settings) == "32"


def test_non_adcsr_serve_uses_overlap_16(monkeypatch, tmp_path) -> None:
    filename = "realesrgan_nchw_256x256_fp32.onnx"
    (tmp_path / filename).touch()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(tmp_path))
    from pathlib import Path

    monkeypatch.setattr(helper_backend, "_winml_helper", lambda: Path("winml-sr.exe"))
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)
    settings = UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model=HELPER_MODEL_AMD_RRDB)
    session = helper_backend.open_session(settings, 854, 480)
    assert session.backend == UpscaleBackend.WINML_GPU
    command = _CapturingServeClient.last_command
    assert command[command.index("--overlap") + 1] == "16"


def test_npu_serve_goes_through_overlap_for_model(monkeypatch, tmp_path) -> None:
    from pathlib import Path

    filename = "realesrgan_nchw_256x256_bf16cast.onnx"
    (tmp_path / filename).touch()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(helper_backend, "_npu_python", lambda: tmp_path / "python.exe")
    monkeypatch.setattr(helper_backend, "_npu_script", lambda: tmp_path / "npu_serve.py")
    monkeypatch.setattr(helper_backend, "_cache_hit", lambda _path: True)
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_AMD_RRDB)
    session = helper_backend.open_session(settings, 854, 480)
    assert session.backend == UpscaleBackend.NPU_NATIVE
    command = _CapturingServeClient.last_command
    assert command[command.index("--overlap") + 1] == str(
        helper_backend.overlap_for_model(HELPER_MODEL_AMD_RRDB)
    )


def _serve_command_for_settings(monkeypatch, settings, width=1280, height=534):
    from pathlib import Path

    model_dir = helper_backend.binaries.repo_root() / "tmp" / "adcsr" / "onnx"
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(model_dir))
    monkeypatch.setattr(helper_backend, "_winml_helper", lambda: Path("winml-sr.exe"))
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)
    session = helper_backend.open_session(settings, width, height)
    return session, list(_CapturingServeClient.last_command)


def test_adcsr_serve_includes_seam_template(monkeypatch) -> None:
    from pathlib import Path

    settings = UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model=HELPER_MODEL_ADCSR)
    _session, command = _serve_command_for_settings(monkeypatch, settings)
    assert "--seam-template" in command
    template = Path(command[command.index("--seam-template") + 1])
    assert template.is_file()
    assert template.name == "adcsr_ov32_p256.json"


def test_adcsr_npu_request_gets_seam_template_via_gpu(monkeypatch) -> None:
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_ADCSR)
    session, command = _serve_command_for_settings(monkeypatch, settings)
    assert session.backend == UpscaleBackend.WINML_GPU
    assert "--seam-template" in command


def test_non_adcsr_serve_has_no_seam_template(monkeypatch, tmp_path) -> None:
    from pathlib import Path

    filename = "realesrgan_nchw_256x256_fp32.onnx"
    (tmp_path / filename).touch()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(helper_backend, "_winml_helper", lambda: Path("winml-sr.exe"))
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)
    settings = UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model=HELPER_MODEL_AMD_RRDB)
    helper_backend.open_session(settings, 854, 480)
    assert "--seam-template" not in _CapturingServeClient.last_command


def test_seam_template_path_resolution() -> None:
    resolved = helper_backend.seam_template_path(HELPER_MODEL_ADCSR, 32)
    assert resolved is not None and resolved.is_file()
    assert resolved.name == "adcsr_ov32_p256.json"
    assert helper_backend.seam_template_path("adcsr", 32) == resolved
    ov16 = helper_backend.seam_template_path(HELPER_MODEL_ADCSR, 16)
    assert ov16 is not None and ov16.is_file()
    assert ov16.name == "adcsr_ov16_p384.json"
    assert helper_backend.seam_template_path(HELPER_MODEL_ANIME, 16) is None
    assert helper_backend.seam_template_path(HELPER_MODEL_ADCSR, 99) is None


def _stage_two_stage_files(monkeypatch, tmp_path):
    """front/back/manifest＋キャッシュをそろえ、解決先を向ける。"""
    import hashlib
    import json

    front = tmp_path / "adcsr_front_nchw_128x128_bf16cast.onnx"
    back = tmp_path / "adcsr_back_nchw_128x128_bf16cast.onnx"
    front.write_bytes(b"front-model")
    back.write_bytes(b"back-model")
    for name in ("cache_f", "cache_b"):
        cache = tmp_path / name
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "context.json").write_text("{}", encoding="utf-8")
        (cache / "x.rai").write_bytes(b"r")
    manifest = {
        "model_family": "AdcSR",
        "front": {"file": front.name,
                  "sha256": hashlib.sha256(b"front-model").hexdigest(),
                  "cache_key": "cache_f"},
        "back": {"file": back.name,
                 "sha256": hashlib.sha256(b"back-model").hexdigest(),
                 "cache_key": "cache_b"},
        "boundary": ["main", "mean", "std"],
    }
    (tmp_path / "adcsr_npu_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "adcsr_nchw_128x128_fp32.onnx").write_bytes(b"gpu-model")
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(helper_backend, "npu_cache_dir", lambda: tmp_path)
    monkeypatch.delenv("UEU_ADCSR_NPU2", raising=False)
    return front, back


def test_npu_two_stage_settings_tables() -> None:
    from app.core.settings import (
        ADCSR_NPU2_ENV,
        ADCSR_NPU_MANIFEST,
        HELPER_MODEL_FILES,
        HELPER_MODEL_NPU_BACK,
    )

    assert HELPER_MODEL_FILES[UpscaleBackend.NPU_NATIVE][HELPER_MODEL_ADCSR] == {
        128: "adcsr_front_nchw_128x128_bf16cast.onnx"
    }
    assert HELPER_MODEL_NPU_BACK[HELPER_MODEL_ADCSR] == "adcsr_back_nchw_128x128_bf16cast.onnx"
    assert ADCSR_NPU_MANIFEST == "adcsr_npu_manifest.json"
    assert ADCSR_NPU2_ENV == "UEU_ADCSR_NPU2"


def test_adcsr_npu_two_stage_spec_stays_npu(monkeypatch, tmp_path) -> None:
    front, _back = _stage_two_stage_files(monkeypatch, tmp_path)
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_ADCSR)
    backend, tile, path = helper_backend._session_spec(settings, 1280, 534)
    assert backend == UpscaleBackend.NPU_NATIVE
    assert tile == 128
    assert path == front.resolve()


def test_adcsr_npu_two_stage_disabled_or_missing_falls_back(monkeypatch, tmp_path) -> None:
    _stage_two_stage_files(monkeypatch, tmp_path)
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_ADCSR)
    monkeypatch.setenv("UEU_ADCSR_NPU2", "0")
    backend, _tile, _path = helper_backend._session_spec(settings, 1280, 534)
    assert backend == UpscaleBackend.WINML_GPU
    monkeypatch.delenv("UEU_ADCSR_NPU2")
    (tmp_path / "adcsr_npu_manifest.json").unlink()
    backend, _tile, _path = helper_backend._session_spec(settings, 1280, 534)
    assert backend == UpscaleBackend.WINML_GPU


def _open_npu_session(monkeypatch, settings, width=1280, height=534):
    monkeypatch.setattr(helper_backend, "_npu_python", lambda: Path("python.exe"))
    monkeypatch.setattr(helper_backend, "_npu_script", lambda: Path("npu_serve.py"))
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)
    return helper_backend.open_session(settings, width, height)


def test_adcsr_npu_two_stage_command_hit(monkeypatch, tmp_path) -> None:
    _stage_two_stage_files(monkeypatch, tmp_path)
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_ADCSR)
    session = _open_npu_session(monkeypatch, settings)
    assert session.backend == UpscaleBackend.NPU_NATIVE
    command = _CapturingServeClient.last_command
    assert "--model-back" in command
    assert "--manifest" in command
    assert "--seam-template" in command
    assert "--require-cache" in command
    assert "--allow-compile" not in command


def test_adcsr_npu_two_stage_command_miss_allows_compile(monkeypatch, tmp_path) -> None:
    _stage_two_stage_files(monkeypatch, tmp_path)
    (tmp_path / "cache_f" / "x.rai").unlink()
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_ADCSR)
    _open_npu_session(monkeypatch, settings)
    command = _CapturingServeClient.last_command
    assert "--allow-compile" in command
    assert "--require-cache" not in command


def test_non_adcsr_npu_has_no_two_stage_flags(monkeypatch, tmp_path) -> None:
    filename = "realesrgan_nchw_256x256_bf16cast.onnx"
    (tmp_path / filename).touch()
    monkeypatch.setenv(helper_backend.MODELS_DIR_ENV, str(tmp_path))
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_AMD_RRDB)
    _open_npu_session(monkeypatch, settings)
    assert "--model-back" not in _CapturingServeClient.last_command
    assert "--manifest" not in _CapturingServeClient.last_command


def test_two_stage_fatal_retries_on_gpu(monkeypatch) -> None:
    from app.core.serve_client import HelperOutputInvalid

    image = np.zeros((8, 8, 3), dtype=np.uint8)

    class _FailClient:
        def upscale(self, img, cancel=None):
            raise HelperOutputInvalid("TWO_STAGE_FATAL stage=front tensor=main x")

    class _GpuClient:
        def upscale(self, img, cancel=None):
            return img + 1

        def close(self, force=False):
            return None

    opened = {}

    def fake_open(settings, width, height, progress=None, cancel=None):
        opened["backend"] = settings.backend
        return helper_backend.HelperSession(
            client=_GpuClient(), backend=settings.backend, model_path=Path("g.onnx"))

    monkeypatch.setattr(helper_backend, "open_session", fake_open)
    session = helper_backend.HelperSession(
        client=_FailClient(), backend=UpscaleBackend.NPU_NATIVE, model_path=Path("f.onnx"))
    settings = UpscaleSettings(backend=UpscaleBackend.NPU_NATIVE, model=HELPER_MODEL_ADCSR)
    out = helper_backend._upscale_with_adcsr_gpu_retry(
        session, settings, 8, 8, image, lambda _f, _m: None, None)
    assert np.all(out == 1)
    assert opened["backend"] == UpscaleBackend.WINML_GPU


def test_two_stage_fatal_reraises_for_non_npu(monkeypatch) -> None:
    from app.core.serve_client import HelperOutputInvalid

    class _FailClient:
        def upscale(self, img, cancel=None):
            raise HelperOutputInvalid("TWO_STAGE_FATAL x")

    session = helper_backend.HelperSession(
        client=_FailClient(), backend=UpscaleBackend.WINML_GPU, model_path=Path("g.onnx"))
    settings = UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model=HELPER_MODEL_ADCSR)
    with pytest.raises(HelperOutputInvalid):
        helper_backend._upscale_with_adcsr_gpu_retry(
            session, settings, 8, 8, np.zeros((8, 8, 3), dtype=np.uint8),
            lambda _f, _m: None, None)
def test_swinir_cuda_serve_uses_separate_python_worker(monkeypatch, tmp_path) -> None:
    python = tmp_path / "python.exe"
    model = tmp_path / "swinir.pth"
    python.touch()
    model.touch()
    monkeypatch.setenv(helper_backend.SWINIR_PYTHON_ENV, str(python))
    monkeypatch.setenv(helper_backend.SWINIR_MODEL_ENV, str(model))
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)

    settings = UpscaleSettings(
        backend=UpscaleBackend.SWINIR_CUDA,
        model=HELPER_MODEL_SWINIR,
    )
    session = helper_backend.open_session(settings, 640, 480)
    command = _CapturingServeClient.last_command

    assert session.backend == UpscaleBackend.SWINIR_CUDA
    assert session.model_path == model.resolve()
    assert command[0] == str(python.resolve())
    assert command[1].endswith("tools\\swinir\\worker.py")
    assert command[2] == "serve"
    assert command[command.index("--precision") + 1] == "bf16"
    assert command[command.index("--tile") + 1] == "256"
    assert command[command.index("--tile-overlap") + 1] == "32"
    assert _CapturingServeClient.last_connect_kwargs["timeout"] == 30 * 60.0


def test_swinir_cuda_startup_timeout_can_be_extended(monkeypatch, tmp_path) -> None:
    python = tmp_path / "python.exe"
    model = tmp_path / "swinir.pth"
    python.touch()
    model.touch()
    monkeypatch.setenv(helper_backend.SWINIR_PYTHON_ENV, str(python))
    monkeypatch.setenv(helper_backend.SWINIR_MODEL_ENV, str(model))
    monkeypatch.setenv(helper_backend.SWINIR_STARTUP_TIMEOUT_ENV, "7200")
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)

    helper_backend.open_session(
        UpscaleSettings(
            backend=UpscaleBackend.SWINIR_CUDA,
            model="swinir-m",
        ),
        640,
        480,
    )

    assert _CapturingServeClient.last_connect_kwargs["timeout"] == 7200.0


def test_swinir_cuda_rejects_other_models(monkeypatch, tmp_path) -> None:
    model = tmp_path / "swinir.pth"
    model.touch()
    monkeypatch.setenv(helper_backend.SWINIR_MODEL_ENV, str(model))
    settings = UpscaleSettings(
        backend=UpscaleBackend.SWINIR_CUDA,
        model=HELPER_MODEL_ANIME,
    )

    try:
        helper_backend._session_spec(settings, 640, 480)
    except helper_backend.HelperBackendUnavailable as exc:
        assert "SwinIR-Mモデル専用" in str(exc)
    else:
        raise AssertionError("SwinIR CUDA accepted a non-SwinIR model")


def test_swinir_cuda_respects_memory_tile_and_gpu(monkeypatch, tmp_path) -> None:
    python = tmp_path / "python.exe"
    model = tmp_path / "swinir.pth"
    python.touch()
    model.touch()
    monkeypatch.setenv(helper_backend.SWINIR_PYTHON_ENV, str(python))
    monkeypatch.setenv(helper_backend.SWINIR_MODEL_ENV, str(model))
    monkeypatch.setattr(helper_backend, "ServeClient", _CapturingServeClient)
    settings = UpscaleSettings(
        backend=UpscaleBackend.SWINIR_CUDA,
        model=HELPER_MODEL_SWINIR,
        tile_size=128,
        gpu_id=1,
    )

    helper_backend.open_session(settings, 640, 480)
    command = _CapturingServeClient.last_command
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--tile") + 1] == "128"
    assert command[command.index("--tile-overlap") + 1] == "32"
