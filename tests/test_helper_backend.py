from app.core import helper_backend
from app.core.settings import ModelFamily, UpscaleBackend, UpscaleSettings


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
        model_family=ModelFamily.REALESRGAN,
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
        model_family=ModelFamily.REALESRGAN,
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
