"""realesrgan-ncnn-vulkan ラッパ: 画像 / フレームフォルダのアップスケール。

実装担当: アップスケールコア エージェント。共通規約:
  - binaries.realesrgan_exe() を subprocess 実行。
  - models は -m <binaries.models_dir()> で明示する（cwd 依存を避ける）。
  - progress(fraction 0..1, message) を呼ぶ。
  - cancel は threading.Event 互換。True なら subprocess を terminate/kill し
    jobs.Cancelled を送出する。
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import binaries, jobs, media
from .jobs import ProgressCb
from .settings import UpscaleBackend, UpscaleSettings, vulkan_fallback_settings

# stderr に出る進捗行（例: "25.00%"）を拾う。
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Windows でコンソールウィンドウを出さないためのフラグ。
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def _noop(_f: float, _m: str) -> None:
    pass


def _build_cmd(in_path: str, out_path: str, settings: UpscaleSettings) -> list[str]:
    """realesrgan-ncnn-vulkan のコマンドライン引数列を組み立てる。

    モデル/スケールの選択規則（exe 実機検証済み）:
      exe は `-n <model>` と `-s <scale>` から param ファイルを
      `<model>-x<scale>.param` → `<model>.param` の順で探す。よって呼び出し側は
      `-n` にスケール接尾辞を付けず、素のモデル名と `-s <scale>` を渡せばよい。
        - realesr-animevideov3: x2/x3/x4 の param のみ → -s 2/3/4 が有効。
        - realesrgan-x4plus(-anime): 単一 param → -s は出力倍率として機能（2/3/4 可）。
      対応する param が一切無い (model, scale) は binaries.model_supports_scale が
      False を返すので、防御的に ValueError を送出する。
    """
    model = settings.model
    scale = int(settings.scale)

    # 防御的チェック: GUI 側は model_supports_scale でガードしているが念のため。
    if not binaries.model_supports_scale(model, scale):
        raise ValueError(
            f"モデル '{model}' は倍率 x{scale} に対応していません"
            f"（{binaries.models_dir()} に該当 param がありません）。"
        )

    cmd: list[str] = [
        binaries.realesrgan_exe(),
        "-i", in_path,
        "-o", out_path,
        "-n", model,
        "-s", str(scale),
        "-m", str(binaries.models_dir()),
    ]

    # タイルサイズ: 0(自動)のときは渡さない（exe 既定の自動推定に任せる）。
    if settings.tile_size and settings.tile_size > 0:
        cmd += ["-t", str(settings.tile_size)]

    # GPU 指定: -1(既定)のときは渡さない。
    if settings.gpu_id is not None and settings.gpu_id >= 0:
        cmd += ["-g", str(settings.gpu_id)]

    # スレッド設定 load:proc:save。
    if settings.threads:
        cmd += ["-j", str(settings.threads)]

    # TTA（高品質・低速）。
    if settings.tta_mode:
        cmd += ["-x"]

    return cmd


def _spawn(cmd: list[str]) -> subprocess.Popen:
    """Popen を生成（stderr をテキストで取得、Windows ではコンソール非表示）。"""
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        # % 行を取りこぼさないよう行バッファ・テキストモード。不正バイトは置換。
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=_CREATE_NO_WINDOW,
    )


def _terminate(proc: subprocess.Popen) -> None:
    """プロセスを terminate→（生きていれば）kill する。"""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


def upscale_image(in_path: str, out_path: str, settings: UpscaleSettings,
                  progress: Optional[ProgressCb] = None, cancel=None) -> None:
    """単一画像をアップスケールして out_path に書く。

    コマンド例:
      realesrgan-ncnn-vulkan -i <in> -o <out> -n <model> -s <scale>
        [-t <tile>] [-g <gpu>] [-j <threads>] [-x] -f <fmt> -m <models_dir>
    進捗は stderr の "xx.xx%" を解析する。out_path の親フォルダは事前に作成すること。
    """
    progress = progress or _noop

    # 旧API互換。新GUIのNPU選択は下のNPU_NATIVEへ寄せる。
    if settings.backend == UpscaleBackend.NPU:
        from . import npu_backend
        npu_backend.upscale_image(in_path, out_path, settings, progress, cancel)
        return

    if settings.backend in {UpscaleBackend.WINML_GPU, UpscaleBackend.NPU_NATIVE}:
        from . import helper_backend

        try:
            helper_backend.upscale_image(in_path, out_path, settings, progress, cancel)
            return
        except helper_backend.HelperBackendUnavailable as exc:
            # DirectML/NPUのモデル不在・helper起動失敗時は、既存のVulkan資産へ退避する。
            settings = vulkan_fallback_settings(settings)
            progress(0.0, f"Vulkanへ切替（モデル: {settings.model} で代替） ({exc})")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_cmd(in_path, out_path, settings)
    # 出力拡張子に合わせて -f を渡す（拡張子が無ければ設定値）。
    fmt = (out.suffix.lstrip(".") or settings.image_format).lower()
    if fmt == "jpeg":
        fmt = "jpg"
    cmd += ["-f", fmt]

    proc = _spawn(cmd)
    stderr_lines: list[str] = []
    last_fraction = 0.0
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            if cancel is not None and cancel.is_set():
                _terminate(proc)
                raise jobs.Cancelled()
            line = line.strip()
            if not line:
                continue
            stderr_lines.append(line)
            m = _PCT_RE.search(line)
            if m:
                frac = max(0.0, min(1.0, float(m.group(1)) / 100.0))
                if frac >= last_fraction:
                    last_fraction = frac
                    progress(frac, f"アップスケール中… {int(frac * 100)}%")
        ret = proc.wait()
    finally:
        if proc.poll() is None:
            _terminate(proc)

    # キャンセル中に EOF した場合の保険。
    if cancel is not None and cancel.is_set():
        raise jobs.Cancelled()

    if ret != 0:
        tail = "\n".join(stderr_lines[-8:])
        raise RuntimeError(
            f"realesrgan-ncnn-vulkan が失敗しました (exit={ret})\ncmd: {' '.join(cmd)}\n{tail}"
        )

    if not out.exists():
        tail = "\n".join(stderr_lines[-8:])
        raise RuntimeError(f"出力ファイルが生成されませんでした: {out}\n{tail}")

    progress(1.0, "完了")


def _count_images(d: Path) -> int:
    """ディレクトリ直下の画像ファイル数（media.IMAGE_EXTS 基準）。"""
    if not d.exists():
        return 0
    return sum(1 for p in d.iterdir()
               if p.is_file() and p.suffix.lower() in media.IMAGE_EXTS)


def upscale_folder(in_dir: str, out_dir: str, settings: UpscaleSettings,
                   progress: Optional[ProgressCb] = None, cancel=None) -> None:
    """in_dir 内の全画像をアップスケールして out_dir へ（realesrgan のフォルダ入出力）。

    realesrgan-ncnn-vulkan -i <in_dir> -o <out_dir> -n <model> -s <scale> ...
    進捗は out_dir の生成ファイル数 / in_dir の画像数 で算出するのが堅実
    （別スレッドでポーリング）。out_dir は事前に作成しておくこと。
    """
    progress = progress or _noop

    # 旧API互換。新GUIのNPU選択は下のNPU_NATIVEへ寄せる。
    if settings.backend == UpscaleBackend.NPU:
        from . import npu_backend
        npu_backend.upscale_folder(in_dir, out_dir, settings, progress, cancel)
        return

    if settings.backend in {UpscaleBackend.WINML_GPU, UpscaleBackend.NPU_NATIVE}:
        from . import helper_backend

        try:
            helper_backend.upscale_folder(in_dir, out_dir, settings, progress, cancel)
            return
        except helper_backend.HelperBackendUnavailable as exc:
            settings = vulkan_fallback_settings(settings)
            progress(0.0, f"Vulkanへ切替（モデル: {settings.model} で代替） ({exc})")

    in_p = Path(in_dir)
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    total = _count_images(in_p)
    if total == 0:
        # 画像が無ければ何もせず完了扱い。
        progress(1.0, "0/0 枚")
        return

    cmd = _build_cmd(in_dir, out_dir, settings)
    # フォルダ出力でも形式を固定（設定の image_format）。
    fmt = (settings.image_format or "png").lower()
    if fmt == "jpeg":
        fmt = "jpg"
    cmd += ["-f", fmt]

    proc = _spawn(cmd)

    stop_poll = threading.Event()

    def _poll() -> None:
        # out_dir の生成枚数を監視して進捗を報告する。
        # フォルダモードの stderr % 行はファイル間で交錯し信頼できないため。
        while not stop_poll.is_set():
            done = min(_count_images(out_p), total)
            frac = done / total if total else 1.0
            progress(min(frac, 0.999), f"{done}/{total} 枚")
            if stop_poll.wait(0.3):
                break

    poller = threading.Thread(target=_poll, daemon=True)
    poller.start()

    stderr_lines: list[str] = []
    try:
        assert proc.stderr is not None
        # stderr は読み捨て（キャンセル監視のため逐次読む）。
        for line in proc.stderr:
            if cancel is not None and cancel.is_set():
                _terminate(proc)
                raise jobs.Cancelled()
            line = line.strip()
            if line:
                stderr_lines.append(line)
        # stderr が EOF でもプロセス終了待ち。途中で polling 継続。
        while True:
            if cancel is not None and cancel.is_set():
                _terminate(proc)
                raise jobs.Cancelled()
            try:
                ret = proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        stop_poll.set()
        poller.join(timeout=2)
        if proc.poll() is None:
            _terminate(proc)

    if cancel is not None and cancel.is_set():
        raise jobs.Cancelled()

    if ret != 0:
        tail = "\n".join(stderr_lines[-8:])
        raise RuntimeError(
            f"realesrgan-ncnn-vulkan が失敗しました (exit={ret})\ncmd: {' '.join(cmd)}\n{tail}"
        )

    done = _count_images(out_p)
    progress(1.0, f"{done}/{total} 枚")
