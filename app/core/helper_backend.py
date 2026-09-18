"""DirectML GPU / Ryzen AI NPUのUEU常駐ヘルパーバックエンド。

GUIから選ばれる新NPU経路は、旧 ``npu_worker`` ではなく
``tools/npu-serve/npu_serve.py`` を常駐させて使う。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import os
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from . import binaries, jobs, media
from .jobs import ProgressCb
from .serve_client import HelperOutputInvalid, ServeClient, ServeClientError
from .settings import (
    DEFAULT_HELPER_MODEL,
    DEFAULT_MODEL,
    DEFAULT_MODELS_DIR,
    DEFAULT_NPU_CACHE_DIR,
    DEFAULT_VENDOR_MODELS_DIR,
    HELPER_MODEL_FILES,
    HELPER_MODEL_AMD_RRDB,
    HELPER_MODEL_ADCSR,
    HELPER_MODEL_SWINIR,
    HELPER_DEFAULT_OVERLAP,
    HELPER_MODEL_OVERLAP,
    ADCSR_NPU2_ENV,
    ADCSR_NPU_MANIFEST,
    HELPER_MODEL_NPU_BACK,
    HELPER_MODEL_NPU_TAIL,
    NPU_TAILCUT_ENV,
    HELPER_SEAM_TEMPLATES,
    canonical_helper_model,
    ModelFamily,
    UpscaleBackend,
    UpscaleSettings,
)

WINML_HELPER_ENV = "UEU_WINML_HELPER"
MODELS_DIR_ENV = "UEU_MODELS_DIR"
NPU_PYTHON_ENV = "UEU_NPU_PYTHON"
NPU_CACHE_ENV = "UEU_NPU_CACHE"
SWINIR_PYTHON_ENV = "UEU_SWINIR_PYTHON"
SWINIR_MODEL_ENV = "UEU_SWINIR_MODEL"
SWINIR_STARTUP_TIMEOUT_ENV = "UEU_SWINIR_STARTUP_TIMEOUT"
SWINIR_DEFAULT_STARTUP_TIMEOUT = 30 * 60.0
SWINIR_MODEL_NAME = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
OVERLAP = 16  # 全モデル共通の既定値（settings.HELPER_DEFAULT_OVERLAP と同じ）。
HELPER_BACKENDS = frozenset({
    UpscaleBackend.WINML_GPU,
    UpscaleBackend.NPU_NATIVE,
    UpscaleBackend.SWINIR_CUDA,
})


# NPU 2段モード (AdcSR 前半/後半) の既定値。
NPU2_WORKER_TIMEOUT = 60.0
NPU2_STARTUP_TIMEOUT_HIT = 15 * 60.0
NPU2_COMPILE_TIMEOUT = 4 * 3600.0


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


def overlap_for_model(model: str | ModelFamily | None) -> int:
    """serve の --overlap（片側マージン、入力 px）をモデル別に返す。

    AdcSR だけ 32（128 タイル → コア 64）。それ以外は既定 16。
    """
    return HELPER_MODEL_OVERLAP.get(canonical_helper_model(model), HELPER_DEFAULT_OVERLAP)


def seam_template_path(
    model: str | ModelFamily | None,
    overlap: int,
    helper_dir: Path | str | None = None,
) -> Path | None:
    """AdcSR の格子補正テンプレートの絶対パスを返す。無ければ None。

    AdcSR 以外・overlap に対応するファイル名が無い・ファイルが無い場合は
    補正なし（None）。探索順は exe と同じ出力先（helper_dir/seam_templates/）、
    次に配布版の vendor/winml-sr/seam_templates/、最後に開発用の
    tools/winml-sr/seam_templates/。
    """
    if canonical_helper_model(model) != HELPER_MODEL_ADCSR:
        return None
    filename = HELPER_SEAM_TEMPLATES.get(overlap)
    if filename is None:
        return None
    candidates: list[Path] = []
    if helper_dir is not None:
        candidates.append(Path(helper_dir) / "seam_templates" / filename)
    candidates.append(binaries.repo_root() / "vendor" / "winml-sr" / "seam_templates" / filename)
    candidates.append(binaries.repo_root() / "tools" / "winml-sr" / "seam_templates" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


# _discarded_pixels / choose_gpu_tile は 256/512 タイル用なので既定 16 のまま。
# AdcSR（128 タイル）は overlap_for_model で 32 を使い、コアは 64 になる。
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

    # 配布版（setup.ps1 が vendor/winml-sr/ へ展開したビルド済みヘルパー）を
    # 開発ビルドより先に探す。UEU_WINML_HELPER の明示指定はこの前段で優先される。
    distributed = binaries.repo_root() / "vendor" / "winml-sr" / "winml-sr.exe"
    if distributed.is_file():
        return distributed.resolve()

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


def _swinir_python() -> Path:
    default = binaries.repo_root() / "tmp" / "swinir-venv" / "Scripts" / "python.exe"
    candidate = Path(os.environ.get(SWINIR_PYTHON_ENV, str(default))).expanduser()
    if not candidate.is_file():
        raise HelperBackendUnavailable(
            "SwinIR用Pythonが見つかりません。"
            "scripts\\setup_swinir.ps1を実行するか、"
            f"{SWINIR_PYTHON_ENV}を指定してください: {candidate}"
        )
    return candidate.resolve()


def _swinir_script() -> Path:
    script = binaries.repo_root() / "tools" / "swinir" / "worker.py"
    if not script.is_file():
        raise HelperBackendUnavailable(f"SwinIRワーカーが見つかりません: {script}")
    return script.resolve()


def _swinir_model() -> Path:
    default = binaries.repo_root() / "tmp" / "swinir-models" / SWINIR_MODEL_NAME
    candidate = Path(os.environ.get(SWINIR_MODEL_ENV, str(default))).expanduser()
    if not candidate.is_file():
        raise HelperBackendUnavailable(
            "SwinIR-Mモデルが見つかりません。"
            "scripts\\setup_swinir.ps1を実行するか、"
            f"{SWINIR_MODEL_ENV}を指定してください: {candidate}"
        )
    return candidate.resolve()


def _swinir_startup_timeout() -> float:
    """CUDA初期化が遅い環境向けに、起動待ちを秒単位で上書き可能にする。"""
    raw = os.environ.get(SWINIR_STARTUP_TIMEOUT_ENV)
    if raw is None:
        return SWINIR_DEFAULT_STARTUP_TIMEOUT
    try:
        seconds = float(raw)
    except ValueError:
        return SWINIR_DEFAULT_STARTUP_TIMEOUT
    if not math.isfinite(seconds):
        return SWINIR_DEFAULT_STARTUP_TIMEOUT
    return max(30.0, min(24 * 60 * 60.0, seconds))


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


def adcsr_npu2_enabled() -> bool:
    """UEU_ADCSR_NPU2=0 で無効。それ以外は既定有効。"""
    return os.environ.get(ADCSR_NPU2_ENV, "1") != "0"


def npu_tailcut_enabled() -> bool:
    """UEU_NPU_TAILCUT=0 で無効。それ以外は既定有効。"""
    return os.environ.get(NPU_TAILCUT_ENV, "1") != "0"


def npu_tail_files(model: str | ModelFamily | None, tile: int) -> tuple[Path, Path] | None:
    """tail-cut の (body, manifest)。両方が models に無ければ None。

    v0.9.0 の配布物しか無い環境では None になり、呼び出し側は従来の
    全体モデルへフォールバックする。
    """
    entry = HELPER_MODEL_NPU_TAIL.get(canonical_helper_model(model), {}).get(tile)
    if entry is None:
        return None
    body_name, manifest_name = entry
    body = _search_model_file(body_name)
    manifest = _search_model_file(manifest_name)
    if body is None or manifest is None:
        return None
    return body, manifest


def _search_model_file(filename: str) -> Path | None:
    """models 探索順で filename を探す。無ければ None。"""
    root = models_dir()
    root_dirs = (root, root / "span")
    search_dirs = (
        (*root_dirs, DEFAULT_VENDOR_MODELS_DIR)
        if MODELS_DIR_ENV in os.environ
        else (DEFAULT_VENDOR_MODELS_DIR, *root_dirs)
    )
    for search_dir in search_dirs:
        candidate = search_dir / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def adcsr_two_stage_files() -> tuple[Path, Path, Path] | None:
    """2 段モードの (front, back, manifest)。揃わなければ None。"""
    try:
        front_name = HELPER_MODEL_FILES[UpscaleBackend.NPU_NATIVE][HELPER_MODEL_ADCSR][128]
        back_name = HELPER_MODEL_NPU_BACK[HELPER_MODEL_ADCSR]
    except KeyError:
        return None
    front = _search_model_file(front_name)
    back = _search_model_file(back_name)
    manifest = _search_model_file(ADCSR_NPU_MANIFEST)
    if front is None or back is None or manifest is None:
        return None
    return front, back, manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _two_stage_cache_hit(manifest_path: Path, front_path: Path, back_path: Path) -> bool:
    """F/G 両方のキャッシュ照合。context.json＋.rai＋マニフェスト SHA-256 一致。"""
    try:
        manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return False
    for side, model_path in (("front", front_path), ("back", back_path)):
        try:
            entry = manifest[side]
            cache = npu_cache_dir() / entry["cache_key"]
        except (KeyError, TypeError):
            return False
        if not (cache / "context.json").is_file():
            return False
        if not any(cache.rglob("*.rai")):
            return False
        try:
            digest = _sha256_file(model_path)
        except OSError:
            return False
        if digest != entry.get("sha256"):
            return False
    return True


def _session_spec(
    settings: UpscaleSettings, width: int, height: int
) -> tuple[UpscaleBackend, int, Path]:
    if int(settings.scale) != 4:
        raise HelperBackendUnavailable("新しいGPU/NPUバックエンドは4xモデル専用です。倍率を4xにしてください。")
    requested = settings.backend
    backend = effective_backend(requested, width, height)
    model_key = _model_key(settings)
    if backend == UpscaleBackend.SWINIR_CUDA:
        if model_key != HELPER_MODEL_SWINIR:
            raise HelperBackendUnavailable("CUDA版SwinIRはSwinIR-Mモデル専用です。")
        return backend, 256, _swinir_model()
    if model_key == HELPER_MODEL_ADCSR and backend == UpscaleBackend.NPU_NATIVE:
        two_stage_pre = adcsr_two_stage_files() if adcsr_npu2_enabled() else None
        if two_stage_pre is None:
            backend = UpscaleBackend.WINML_GPU
    if model_key == HELPER_MODEL_ADCSR:
        tile = 128
    elif model_key in (HELPER_MODEL_AMD_RRDB, HELPER_MODEL_SWINIR):
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
        backend, tile, model_path = _session_spec(settings, width, height)
        model_key = _model_key(settings)
        overlap = overlap_for_model(model_key)
        if backend == UpscaleBackend.SWINIR_CUDA:
            progress(0.0, "SwinIR-MをCUDAへ読み込み中…")
        elif settings.backend == UpscaleBackend.NPU_NATIVE and backend == UpscaleBackend.WINML_GPU:
            if model_key == HELPER_MODEL_ADCSR:
                progress(0.0, "AdcSRはNPU非対応のためGPUで実行…")
            else:
                progress(0.0, "短辺480px未満のためGPUへ自動切替…")
        else:
            progress(0.0, "AI準備中…")

        cache_hit = True
        if backend == UpscaleBackend.SWINIR_CUDA:
            python = _swinir_python()
            script = _swinir_script()
            swinir_tile = int(settings.tile_size) if int(settings.tile_size) > 0 else 256
            if swinir_tile % 8:
                raise HelperBackendUnavailable("SwinIRのタイルサイズは8の倍数にしてください。")
            swinir_overlap = min(32, swinir_tile // 2)
            swinir_device = (
                f"cuda:{int(settings.gpu_id)}"
                if int(settings.gpu_id) >= 0
                else "cuda"
            )
            command = [
                str(python), str(script), "serve",
                "--model", str(model_path),
                "--model-kind", "real_sr_m",
                "--device", swinir_device,
                "--precision", "bf16",
                "--tile", str(swinir_tile),
                "--tile-overlap", str(swinir_overlap),
            ]
            workdir = script.parent
            timeout = _swinir_startup_timeout()
        elif backend == UpscaleBackend.WINML_GPU:
            helper = _winml_helper()
            command = [
                str(helper), "serve", "--model", str(model_path),
                "--ep-name", "DmlExecutionProvider",
                "--overlap", str(overlap), "--warmup", "1",
            ]
            seam_template = seam_template_path(model_key, overlap, helper.parent)
            if seam_template is not None:
                command += ["--seam-template", str(seam_template)]
            elif model_key == HELPER_MODEL_ADCSR:
                print("警告: AdcSR の格子補正テンプレートが見つからないため補正なしで実行",
                      file=sys.stderr, flush=True)
            workdir = helper.parent
            timeout = 120.0
        else:
            python = _npu_python()
            script = _npu_script()
            cache = npu_cache_dir()
            two_stage: tuple[Path, Path, Path] | None = None
            back_path: Path | None = None
            manifest_path: Path | None = None
            if model_key == HELPER_MODEL_ADCSR and adcsr_npu2_enabled():
                two_stage = adcsr_two_stage_files()
            if two_stage is not None:
                front_path, back_path, manifest_path = two_stage
                model_path = front_path
                cache_hit = _two_stage_cache_hit(manifest_path, front_path, back_path)
            elif model_key == HELPER_MODEL_ADCSR and backend == UpscaleBackend.NPU_NATIVE:
                raise HelperBackendUnavailable(
                    "AdcSR の NPU 2段モード用ファイル (front/back/manifest) が見つかりません")
            else:
                cache_hit = _cache_hit(model_path)
            tail_manifest: Path | None = None
            if (
                two_stage is None
                and backend == UpscaleBackend.NPU_NATIVE
                and model_key != HELPER_MODEL_ADCSR
                and npu_tailcut_enabled()
            ):
                tail_found = npu_tail_files(model_key, tile)
                if tail_found is not None:
                    # body とマニフェストが両方あれば tail-cut。cache_hit は
                    # body の stem で判定する。無ければ従来の全体モデルのまま。
                    model_path, tail_manifest = tail_found
                    cache_hit = _cache_hit(model_path)
            command = [
                str(python), str(script), "--model", str(model_path),
                "--cache-dir", str(cache),
                "--overlap", str(overlap), "--warmup", "1",
            ]
            if tail_manifest is not None:
                command += ["--tail", str(tail_manifest)]
            seam_template = seam_template_path(model_key, overlap)
            if seam_template is not None:
                command += ["--seam-template", str(seam_template)]
            workdir = binaries.repo_root()
            if two_stage is not None:
                assert back_path is not None and manifest_path is not None
                command += [
                    "--model-back", str(back_path),
                    "--manifest", str(manifest_path),
                    "--worker-timeout", str(NPU2_WORKER_TIMEOUT),
                ]
                command += ["--allow-compile"] if not cache_hit else ["--require-cache"]
            # 初回VAIMLコンパイルはモデル次第で長い（av3dp512: 約15分、SwinIR-M: 約51分）。
            # キャッシュ有無でタイムアウトを分け、初回コンパイルを打ち切らない。
            if two_stage is not None:
                timeout = NPU2_STARTUP_TIMEOUT_HIT if cache_hit else NPU2_COMPILE_TIMEOUT
            else:
                timeout = 15 * 60.0 if cache_hit else 120 * 60.0
            if not cache_hit:
                if two_stage is not None:
                    progress(0.0, "NPU 前半を最適化中 1/2（初回のみ。次回はキャッシュを利用）")
                else:
                    progress(0.0, "初回のみNPU最適化中（数分〜1時間・次回はキャッシュを利用）")

        env = _helper_env()
        stage_state: dict[str, object] = {"tag": None}
        boot_start = time.monotonic()

        def _format_stage(tag: str, elapsed: float) -> str:
            mm_ss = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
            if tag == "front-compile":
                return f"NPU 前半を最適化中 1/2（経過 {mm_ss}。この検証機では約93分）"
            if tag == "back-compile":
                return f"NPU 後半を最適化中 2/2（経過 {mm_ss}。この検証機では約30分）"
            if tag == "selftest":
                return f"NPU 動作検査中（経過 {mm_ss}）"
            if tag == "ready":
                return "NPU 準備完了"
            return "AI準備中…"

        def _log_line(line: str) -> None:
            for tag in ("front-compile", "back-compile", "selftest", "ready"):
                if f"[stage] {tag}" in line:
                    stage_state["tag"] = tag
                    progress(0.0, _format_stage(tag, time.monotonic() - boot_start))
                    break

        client = ServeClient(command, workdir, env=env, log=_log_line)

        last_second = -1

        def _waiting(elapsed: float) -> None:
            nonlocal last_second
            second = int(elapsed)
            if second == last_second:
                return
            last_second = second
            if backend == UpscaleBackend.SWINIR_CUDA:
                progress(0.0, "SwinIR-MをCUDAへ読み込み中…")
            elif backend == UpscaleBackend.NPU_NATIVE and not cache_hit:
                tag = stage_state["tag"]
                if two_stage is not None:
                    if isinstance(tag, str):
                        progress(0.0, _format_stage(tag, elapsed))
                    else:
                        progress(0.0, _format_stage("front-compile", elapsed))
                else:
                    progress(0.0, "初回のみNPU最適化中（数分〜1時間・次回はキャッシュを利用）")
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


@dataclass
class _LoadedImage:
    rgb: np.ndarray
    alpha: Image.Image | None = None
    icc_profile: bytes | None = None


def _load_image(path: str) -> _LoadedImage:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        icc_profile = image.info.get("icc_profile")
        has_alpha = "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info)
        alpha = image.convert("RGBA").getchannel("A") if has_alpha else None
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return _LoadedImage(rgb, alpha, icc_profile)


def _load_rgb(path: str) -> np.ndarray:
    return _load_image(path).rgb


def _save_rgb(image: np.ndarray, path: Path, *, alpha: Image.Image | None = None,
              icc_profile: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = Image.fromarray(image, mode="RGB")
    if alpha is not None:
        alpha = alpha.resize(output.size, Image.Resampling.LANCZOS)
        output = Image.merge("RGBA", (*output.split(), alpha))
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            background = Image.new("RGB", output.size, "white")
            background.paste(output, mask=alpha)
            output = background
    output.save(path, **({"icc_profile": icc_profile} if icc_profile else {}))


def _folder_outputs(images: list[Path], target: Path, fmt: str, overwrite: bool) -> dict[Path, Path]:
    outputs, reserved = {}, set()
    for source in images:
        candidate = target / f"{source.stem}.{fmt}"
        index = 1
        while candidate in reserved or (not overwrite and candidate.exists()):
            candidate = target / f"{source.stem}({index}).{fmt}"
            index += 1
        outputs[source] = candidate
        reserved.add(candidate)
    return outputs


def _upscale_with_adcsr_gpu_retry(
    session: HelperSession,
    settings: UpscaleSettings,
    width: int,
    height: int,
    image: "np.ndarray",
    progress: ProgressCb,
    cancel=None,
) -> "np.ndarray":
    """NPU 2段の TWO_STAGE_FATAL時は同じ AdcSR の DirectML で画像単位に再処理する。

    フォルダ処理で完了済みの画像は再処理しない (呼び出し側が画像単位で呼ぶ)。
    """
    try:
        if cancel is not None:
            return session.upscale(image, cancel=cancel)
        return session.upscale(image)
    except HelperOutputInvalid:
        if session.backend != UpscaleBackend.NPU_NATIVE:
            raise
        if _model_key(settings) != HELPER_MODEL_ADCSR:
            raise
        progress(0.1, "NPU 2段で失敗したためGPUで再処理…")
        gpu_session = open_session(
            replace(settings, backend=UpscaleBackend.WINML_GPU),
            width, height, progress, cancel,
        )
        try:
            if cancel is not None:
                return gpu_session.upscale(image, cancel=cancel)
            return gpu_session.upscale(image)
        finally:
            gpu_session.close(force=cancel is not None and cancel.is_set())


def upscale_image(
    in_path: str,
    out_path: str,
    settings: UpscaleSettings,
    progress: Optional[ProgressCb] = None,
    cancel=None,
) -> None:
    progress = progress or _noop
    loaded = _load_image(in_path)
    image = loaded.rgb
    height, width = image.shape[:2]
    session = open_session(settings, width, height, progress, cancel)
    try:
        if cancel is not None and cancel.is_set():
            raise jobs.Cancelled()
        progress(0.1, "アップスケール中…")
        output = _upscale_with_adcsr_gpu_retry(session, settings, width, height, image, progress, cancel)
        if cancel is not None and cancel.is_set():
            raise jobs.Cancelled()
        _save_rgb(output, Path(out_path), alpha=loaded.alpha, icc_profile=loaded.icc_profile)
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
    outputs = _folder_outputs(images, target, fmt, settings.overwrite)
    gpu_retry: HelperSession | None = None

    def _upscale_one(active: HelperSession, image: np.ndarray, index: int) -> np.ndarray:
        nonlocal gpu_retry
        try:
            if cancel is not None:
                return active.upscale(image, cancel=cancel)
            return active.upscale(image)
        except HelperOutputInvalid:
            if active.backend != UpscaleBackend.NPU_NATIVE:
                raise
            if _model_key(settings) != HELPER_MODEL_ADCSR:
                raise
            progress((index - 1) / total, f"{index}/{total} 枚 NPU 2段で失敗したためGPUで再処理…")
            height0, width0 = image.shape[:2]
            if gpu_retry is None:
                gpu_retry = open_session(
                    replace(settings, backend=UpscaleBackend.WINML_GPU),
                    width0, height0, progress, cancel,
                )
            if cancel is not None:
                return gpu_retry.upscale(image, cancel=cancel)
            return gpu_retry.upscale(image)
    try:
        for index, path in enumerate(images, start=1):
            if cancel is not None and cancel.is_set():
                raise jobs.Cancelled()
            loaded = _load_image(str(path))
            image = loaded.rgb
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
            output = _upscale_one(session, image, index)
            _save_rgb(output, outputs[path], alpha=loaded.alpha, icc_profile=loaded.icc_profile)
            progress(index / total, f"{index}/{total} 枚")
    finally:
        force = cancel is not None and cancel.is_set()
        for session in sessions.values():
            session.close(force=force)
        if gpu_retry is not None:
            gpu_retry.close(force=force)
