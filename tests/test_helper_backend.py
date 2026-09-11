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

    def __init__(self, command, workdir, env=None) -> None:
        type(self).last_command = list(command)

    def connect(self, **kwargs) -> None:
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
