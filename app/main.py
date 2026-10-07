"""エントリポイント: GUI を起動する。"""
from __future__ import annotations

import sys


def portable_self_test() -> int:
    """ポータブル版が同梱バイナリを解決できるかを終了コードで返す。"""
    from app.core import binaries

    binaries.ffmpeg_exe()
    binaries.ffprobe_exe()
    binaries.realesrgan_exe()
    binaries.rife_exe()
    if not binaries.available_models():
        return 2
    if "rife-v4.6" not in binaries.available_interpolation_models():
        return 3
    # GPU ヘルパーを同梱した版では、そのモデルも exe の隣から見つかること
    from app.core import helper_backend
    from app.core.settings import DEFAULT_HELPER_MODEL, UpscaleBackend

    if (binaries.repo_root() / "vendor" / "winml-sr" / "winml-sr.exe").is_file():
        try:
            helper_backend._winml_helper()
            helper_backend._resolve_model(UpscaleBackend.WINML_GPU, DEFAULT_HELPER_MODEL, 256)
        except helper_backend.HelperBackendUnavailable:
            return 4
    # NPU キットを展開してある場合は、そのモデルも exe の隣から見つかること
    # （Ryzen AI Software の有無は PC 次第なので、ここでは見ない）
    if (binaries.repo_root() / "tools" / "npu-serve" / "npu_serve.py").is_file():
        from app.core import npu_prepare

        if not npu_prepare.npu_models():
            return 5
    # FILM キットを置いてあるときだけ、ヘルパーとモデルが解決できること。
    # 置いていなければ何もしない（起動はしない）。
    film_helper = binaries.repo_root() / "vendor" / "winml-film" / "winml-film.exe"
    film_model = binaries.repo_root() / "models" / "film" / "film_style_fp32.onnx"
    if film_helper.is_file() or film_model.is_file():
        try:
            binaries.film_helper_exe()
            binaries.film_model_path()
        except binaries.BinaryError:
            return 6
    return 0


def main() -> int:
    if "--portable-self-test" in sys.argv:
        return portable_self_test()
    from app.i18n import init_language

    init_language()
    from app.gui.main_window import run  # GUI エージェントが提供する run(argv)->int
    return run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
