"""英語の訳文。

キーは日本語の原文そのまま。`t()` と `N_()` に渡している原文を
もれなく載せる（tests/test_i18n.py が網羅を確かめる）。
"""
from __future__ import annotations

EN: dict[str, str | None] = {
    # --- 共通: 言語とウィンドウ
    # 閉じる前の案内など、ウィンドウ題名に足す文言。{text}=状況行の文言
    'ultraeasy-upscaler — {text}': 'ultraeasy-upscaler — {text}',
    # 詳細設定の項目名
    '表示言語': 'Display language',
    # 表示言語の選択肢（OSに合わせる）
    'Windows の設定に合わせる': 'Same as Windows',
    # 表示言語の選択肢
    '日本語': '日本語',
    # 表示言語の選択肢（原文のまま）
    'English': 'English',
    # 表示言語を変えたとき項目の下に出る1行
    '次回の起動から切り替わります。': 'The change takes effect the next time the app starts.',
    # --- メイン画面: ヘッダ
    # 歯車ボタンのツールチップ／詳細設定の見出し（2箇所で同じ文言）
    '詳細設定': 'More settings',
    # ヘッダの一括設定の見出し
    '画像の拡大モデル': 'Image model',
    # ヘッダの一括設定の見出し
    '動画の拡大モデル': 'Video model',
    # --- メイン画面: 左列（メディア一覧・ドロップ）
    # 左列の見出し
    'メディア': 'Media',
    # 左列のリンクボタン
    'すべてやり直す': 'Redo all',
    # 左列のリンクボタン
    'すべて削除': 'Remove all',
    # 全面ドロップ枠の大見出し
    '画像・動画をドロップ': 'Drop images or videos here',
    # 全面ドロップ枠の案内
    'クリックでファイル選択 / 複数選択OK': 'Or click to choose files. You can add several at once.',
    # 一覧下の小さい追加枠
    'ここにドロップ、またはクリックで追加': 'Drop here, or click to add',
    # 複数追加時の状況行。{added}=追加数、{skipped}=スキップ数、{names}=先頭3件の名、{more}=「…」か空
    '{added} 件追加 / {skipped} 件は未対応のためスキップ: {names}{more}': 'Added {added}. Skipped {skipped} unsupported: {names}{more}',
    # --- メイン画面: 中央（クイック確認・比較）
    # クイック確認ボタン（画像・動画で同じ）
    'クイック確認': 'Quick check',
    # 範囲ボタン（通常）
    '範囲を選択してクイック確認': 'Quick check on a selected area',
    # 範囲ボタン（範囲あり）
    '範囲を解除': 'Clear area',
    # 範囲選択中の画像上の案内
    '確認したい部分をドラッグで囲んでください': 'Drag over the part you want to check',
    # クイック確認の「?」の説明文（3 行。改行も原文どおり）
    '仕上がり確認のため、1 枚だけ拡大処理を行い、元の画像と比較できます。\n動画は、画像下のスライダーで拡大するフレームを選択できます。\n「範囲を選択してクイック確認」を使うと、その 1 枚のさらに一部分だけに処理を限定できます。': 'Enlarges just one image so you can compare it with the original before the full run.\nFor a video, choose the frame to enlarge with the slider under the picture.\n"Quick check on a selected area" limits the processing to part of that one image.',
    # 確認の実行中のクイック確認ボタン／NPU変換の実行中のボタン（2箇所で同じ文言）
    '中止': 'Cancel',
    # 比較ビューの表示ボタンのツールチップ（画像に重ねる小ボタン）
    '全体表示': 'Fit',
    # 比較ビューの表示ボタンのツールチップ（画像に重ねる小ボタン）
    '等倍': '100%',
    # 画の札（未確認・比較の左）
    '元の画像': 'Original',
    # ファイル未選択の中央文言
    '左の一覧からファイルを選ぶと、ここに表示します。': 'Select a file in the list on the left to show it here.',
    # 読み込み失敗の中央文言
    'このファイルは表示できません。処理はできます。': "This file can't be shown here. It can still be processed.",
    # フォルダ選択の中央文言
    'フォルダの中身はここには表示されません。': "The contents of a folder aren't shown here.",
    # 元画像の読み込み中の中央文言／旧NPUフォルダ処理の画像読み込みの進捗
    '読み込み中…': 'Loading…',
    # 確認の実行中の状況行
    '確認中…': 'Checking…',
    # 確認の進捗の状況行。{pct}=0〜100の整数
    '確認中… {pct}%': 'Checking… {pct}%',
    # 確認の失敗の状況行。{message}=整形済みエラー文
    '試せませんでした: {message}': "Couldn't try it: {message}",
    # 試し・NPU変換の中止の状況行（2箇所で同じ文言）
    '中止しました': 'Canceled',
    # 変換中の確認の状況行
    'NPU の変換中はクイック確認できません': "Quick check isn't available while a model is being converted for the NPU",
    # 本処理中の確認の状況行
    '処理中はクイック確認できません': "Quick check isn't available while processing",
    # フォルダ選択の確認の状況行
    'フォルダはクイック確認できません。中の画像を 1 枚追加すると確認できます。': "Folders can't be quick-checked. Add one of the images inside to check it.",
    # モデル未選択の確認の状況行
    '拡大モデルを選ぶとクイック確認できます': 'Choose an upscaling model to use quick check',
    # エラー文が空のときの代替表示
    '不明なエラー': 'Unknown error',
    # --- メイン画面: 右列（設定と開始）
    # 右列の見出し
    '設定': 'Settings',
    # ファイル未選択のときの案内1行
    'ファイルを選ぶと、そのファイルだけ設定を変えられます': 'Select a file to change the settings for that file only.',
    # 行の種別表示
    '画像': 'Image',
    # 行の種別表示
    '動画': 'Video',
    # 右列の項目名
    '拡大モデル': 'Upscaling model',
    # 右列の項目名
    '倍率': 'Scale',
    # 個別設定の案内1行
    'このファイルだけ別の設定です': 'This file uses its own settings.',
    # 個別設定のボタン（動画）
    'ほかの動画にも使う': 'Use for other videos too',
    # 個別設定のボタン（画像・フォルダ）
    'ほかの画像にも使う': 'Use for other images too',
    # 個別設定のボタン
    '一括設定に戻す': 'Use the settings at the top',
    # モデルコンボの選択肢
    'なし（拡大しない）': 'None (no upscaling)',
    # フレーム補間コンボの選択肢
    'なし（補間しない）': 'None (no interpolation)',
    # フレーム補間コンボの選択肢（rife-v4.6用）
    'RIFE v4.6': 'RIFE v4.6',
    # フレーム補間コンボの選択肢（FILM用。表示名は訳さない）
    'FILM (Style)': 'FILM (Style)',
    # FILM未導入の項目。{name}=補間モデルの表示名（例: FILM (Style)）
    '{name}（未導入）': '{name} (not installed)',
    # FILM未導入の項目のツールチップ
    '詳細設定の「追加キット」からダウンロードできます': 'You can download it from "Add-on kits" in the settings',
    # フレーム補間モデルの説明行（右列・ヘッダーのツールチップ）
    '高速': 'Fast',
    # フレーム補間モデルの説明行（右列・ヘッダーのツールチップ）
    '低速・高品質': 'Slow · High quality',
    # モデル・補間コンボの無効な選択肢（2箇所で同じ文言）
    'モデル未検出': 'No model found',
    # 右列の項目名（動画側のみ表示）
    'フレーム補間モデル': 'Frame interpolation model',
    # 出力先コンボの選択肢
    '元の場所': 'Same folder as the source',
    # 出力先コンボの選択肢
    'フォルダ選択…': 'Choose a folder…',
    # 右列の項目名／行のツールチップ先頭（「出力先: {path}」は別項目）
    '出力先': 'Output',
    # 出力先に選んだフォルダの表示。{name}=フォルダ名（例: out）
    '📁 {name}': '📁 {name}',
    # 一時停止ボタン（通常）
    '一時停止': 'Pause',
    # 一時停止ボタンのツールチップ
    '現在のジョブ完了後に停止します': 'Stops after the current file is finished',
    # 開始ボタン（待機中）
    '開始': 'Start',
    # 開始ボタンの実行中表示／行の状態表示（2箇所で同じ文言）
    '処理中…': 'Processing…',
    # 一時停止ボタンの停止要求中の表示
    '停止中…': 'Pausing…',
    # 開始時（空）の状況行
    '処理するファイルがありません。': 'There are no files to process.',
    # 開始時（個別未選択）の状況行。{name}=ファイル名（例: photo.png）
    '{name}の拡大モデルを選んでください。': 'Choose an upscaling model for {name}.',
    # 開始時（動画・未選択）の状況行
    '動画の拡大モデルかフレーム補間モデルを選んでください。': 'Choose an upscaling model or a frame interpolation model for videos.',
    # 開始時（画像・未選択）の状況行
    '画像の拡大モデルを選んでください。': 'Choose an upscaling model for images.',
    # 変換中の開始時の状況行
    'NPU の変換中は処理を始められません': "You can't start while a model is being converted for the NPU",
    # 未変換モデルの開始・試し時の状況行。{label}=モデル表示名（例: Anime Video v3）
    '{label}は NPU 用の変換がまだです。詳細設定の「NPU の準備」で変換してください。': "{label} hasn't been converted for the NPU yet. Convert it under NPU setup in More settings.",
    # 一時停止を押したときの状況行
    '現在のジョブ完了後に停止します。今すぐ中止するには行の × を押してください。': 'It will stop after the current file is finished. To cancel right away, select × on the row.',
    # 一時停止完了の状況行
    '一時停止しました。「開始」で再開できます。': 'Paused. Select Start to continue.',
    # 閉じる操作の待機中の状況行
    '終了処理中… 現在の処理を停止しています': 'Closing… stopping the current work',
    # モデル「なし」選択の説明行
    '拡大はしません。フレーム補間だけ実行できます。': 'No upscaling. Only frame interpolation can run.',
    # 出力先ダイアログの題名
    '出力先フォルダを選択': 'Choose the output folder',
    # 実行中行の削除時の行表示
    'キャンセル中…': 'Canceling…',
    # 中止済み行の表示／行の状態表示（2箇所で同じ文言）
    'キャンセルされました': 'Canceled',
    # 失敗行の表示。{message}=エラー文
    'エラー: {message}': 'Error: {message}',
    # --- モデル: 表示名
    # モデル表示名（Vulkan用）
    'Real-ESRGAN': 'Real-ESRGAN',
    # モデル表示名（Vulkan用）
    'Real-ESRGAN Anime': 'Real-ESRGAN Anime',
    # モデル表示名（Vulkan・GPU・NPU用で同じ文言）
    'Anime Video v3': 'Anime Video v3',
    # モデル表示名（Vulkan用）
    'General Video v3（ノイズ除去強）': 'General Video v3 (strong denoise)',
    # モデル表示名（Vulkan用）
    'General Video v3（ノイズ除去弱）': 'General Video v3 (weak denoise)',
    # モデル表示名（GPU・NPU用）
    '4xNomosUni SPAN': '4xNomosUni SPAN',
    # モデル表示名（GPU・NPU用）
    'Real-ESRGAN（AMD縮小版）': 'Real-ESRGAN (AMD compact)',
    # モデル表示名（GPU・NPU・CUDA用）
    'SwinIR-M': 'SwinIR-M',
    # モデル表示名（GPU・NPU用）
    'AdcSR': 'AdcSR',
    # CUDA用モデルコンボの選択肢
    'SwinIR-M（real-world x4）': 'SwinIR-M (real-world x4)',
    # --- モデル: 説明（1 行）・特性の印
    # 説明行（Anime Video v3。GPU/NPU/Vulkan とも同じ）
    'アニメ向け・速い': 'For anime · Fast',
    # 説明行（4xNomosUni SPAN）
    '実写向け・速い': 'For live action · Fast',
    # 説明行（Real-ESRGAN AMD縮小版）
    '実写向け・くっきり・やや遅い': 'For live action · Sharp · Somewhat slow',
    # 説明行（SwinIR-M）
    '実写の静止画向け・高精細・遅い': 'For live-action stills · Fine detail · Slow',
    # 説明行（AdcSR）
    '実写の静止画向け・最高画質・とても遅い': 'For live-action stills · Highest quality · Very slow',
    # 説明行（Real-ESRGAN Vulkan）
    '実写向け・高画質・遅い': 'For live action · High quality · Slow',
    # 説明行（Real-ESRGAN Anime Vulkan）
    'アニメ向け・高画質・遅い': 'For anime · High quality · Slow',
    # 説明行（General Video v3 ノイズ除去強）
    '実写・アニメ兼用・ノイズ除去強め': 'For live action and anime · Strong denoising',
    # 説明行（General Video v3 ノイズ除去弱）
    '実写向け・ノイズ除去弱め': 'For live action · Light denoising',
    # 一覧の印の前半。{speed}=◎○△✕極遅の印
    '速度{speed}': 'Speed: {speed}',
    # 一覧の印の後半・ツールチップ。{quality}=◎○◎◎の印
    '画質{quality}': 'Quality: {quality}',
    # ツールチップの適性。{anime}{live}=◎○△の印
    'アニメ{anime}・実写{live}': 'Anime: {anime}, live action: {live}',
    # ツールチップの推奨タグ。{star}=アニメ実写静止画など
    '★{star}に推奨': '★ Recommended for {star}',
    # 開いた一覧の印。{speed}{quality}=上と同様
    '速度{speed} 画質{quality}': 'Speed: {speed}, quality: {quality}',
    # 開いた一覧の推奨タグ（前あき）。{star}=上と同様
    ' ★{star}': ' ★ {star}',
    # 推奨タグの値
    'アニメ': 'anime',
    # 推奨タグの値
    '実写': 'live action',
    # 推奨タグの値
    '静止画': 'stills',
    # 推奨タグの値（SwinIR CUDA）
    '実写・再開可': 'live action, resumable',
    # 速さの印の値（SwinIR CUDA）
    '極遅': 'Extremely low',
    # 速度・画質・適性の印（英語では言葉にする。日本語はそのまま）
    '◎◎': 'Best',
    '◎': 'High',
    '○': 'Medium',
    '△': 'Low',
    '✕': 'Very low',
    # NPU未変換の一覧の印／閉じた表示の除去対象（2箇所で同じ文言）
    '（未変換）': ' (not converted)',
    # 個別設定の行の印。{suffix}として差し込まれる
    '（個別）': ' (custom)',
    # 待機行の表示。{model}=モデル表示名、{scale}=4、{suffix}=「（個別）」か空
    '{model}・{scale}x{suffix}': '{model}, {scale}x{suffix}',
    # 待機行の表示（補間あり）。差し込みは上と同様
    '{model}・{scale}x・補間あり{suffix}': '{model}, {scale}x, interpolation{suffix}',
    # 待機行の表示（補間のみ）。{suffix}=上と同様
    '補間のみ{suffix}': 'Interpolation only{suffix}',
    # 待機行の表示（未選択）。{suffix}=上と同様
    'モデル未選択{suffix}': 'No model chosen{suffix}',
    # 完了行の表示。{model}=モデル表示名（例: Anime Video v3）
    '完了・{model}': 'Done: {model}',
    # 完了行の表示（補間のみ）
    '完了・補間のみ': 'Done: interpolation only',
    # 完了行・完了進捗・行の状態表示（複数箇所で同じ文言）
    '完了': 'Done',
    # --- 詳細設定: 項目と選択肢
    # 詳細設定の項目名
    'AI実行先': 'Run on',
    # AI実行先の選択肢
    '自動（GPU優先）': 'Auto (GPU first)',
    # AI実行先の選択肢
    'GPU（DirectML）': 'GPU (DirectML)',
    # AI実行先の選択肢
    'NPU（GPU温存）': 'NPU (keeps the GPU free)',
    # AI実行先の選択肢
    'SwinIR-M（CUDA・超低速）': 'SwinIR-M (CUDA, extremely slow)',
    # AI実行先の選択肢
    'Vulkan': 'Vulkan',
    # 詳細設定の項目名
    '画像の保存形式': 'Image format',
    # 詳細設定の項目名
    '動画の保存形式': 'Video format',
    # 詳細設定の項目名
    '動画の画質 (CRF/QP)': 'Video quality (CRF/QP)',
    # 動画の画質の選択肢
    '高画質（容量大）': 'High quality (large file)',
    # 動画の画質の選択肢
    '標準': 'Standard',
    # 動画の画質の選択肢
    '軽量（容量小）': 'Light (small file)',
    # 詳細設定の項目名
    '分割処理 (タイル)': 'Tiling',
    # タイル・使うGPUの選択肢（2箇所で同じ文言）
    '自動': 'Auto',
    # タイルの選択肢
    'メモリ節約': 'Save memory',
    # タイルの選択肢
    '強めに節約': 'Save more memory',
    # 詳細設定の項目名
    '使うGPU': 'GPU to use',
    # 詳細設定の項目名
    '出力フォルダ名': 'Output folder name',
    # 詳細設定の項目名
    '補間後のfps': 'Interpolated frame rate',
    # 補間後fpsの選択肢
    '元動画の2倍': 'Twice the source',
    # 補間後fpsの選択肢
    '元動画の4倍': '4 times the source',
    # 補間後fpsの選択肢
    '元動画の8倍': '8 times the source',
    # FILM選択中に選べないfps項目のツールチップ
    'FILM (Style) は元動画の倍数だけに対応します': 'FILM (Style) supports only multiples of the source frame rate',
    # 詳細設定の項目名
    '処理の順番': 'Order of processing',
    # 処理順の選択肢
    'アプコン → 補間（速い）': 'Upscale → interpolate (fast)',
    # 処理順の選択肢
    '補間 → アプコン（省メモリ）': 'Interpolate → upscale (less memory)',
    # 詳細設定のチェックボックス
    '動画の保存を速くする': 'Save videos faster',
    # 詳細設定のチェックボックス
    '動画の音声を残す': 'Keep the audio of videos',
    # 詳細設定のチェックボックス
    '高品質モード (TTA)': 'High-quality mode (TTA)',
    # 詳細設定のチェックボックス
    '出力フォルダを作る': 'Create an output folder',
    # 詳細設定のチェックボックス
    '補間の倍率を細かく選ぶ': 'Choose the interpolation rate in detail',
    # --- 詳細設定: ヘルプ
    # 出力先のヘルプ
    '処理したファイルの保存先です。「元の場所」は元のファイルと同じ場所、「フォルダ選択…」は指定したフォルダに保存します。': 'Where processed files are saved. "Same folder as the source" saves next to the original file, and "Choose a folder…" saves to the folder you pick.',
    # AI実行先のヘルプ
    'AIの実行先です。自動はDirectML GPUを優先し、起動できない場合はVulkanへ切り替えます。NPUはGPU負荷を抑えます。SwinIR CUDAは実写向けですが動画処理は非常に低速です。': "Where the AI runs. Auto uses the GPU with DirectML and switches to Vulkan if that can't start. NPU keeps the load off the GPU. SwinIR CUDA suits live action but is extremely slow for videos.",
    # 画像保存形式のヘルプ
    '画像を書き出す形式です。pngは劣化なし、jpgは容量小、webpは容量を抑えやすい形式です。': 'The format for saved images. png is lossless, jpg is small, and webp keeps files small with good quality.',
    # 動画保存形式のヘルプ
    '動画ファイルの保存形式です。mp4は再生互換性が高く、mkv/movは用途に合わせて選びます。': 'The format for saved videos. mp4 plays almost everywhere. Choose mkv or mov when you need them.',
    # 動画画質のヘルプ
    'CRF/QPは動画の圧縮品質です。数字が小さいほど高画質で容量は大きくなります。': 'CRF/QP sets how much the video is compressed. A smaller number means higher quality and a larger file.',
    # タイルのヘルプ
    'タイルは画像を分割して処理する単位です。通常は自動でOK。メモリ不足で失敗するときだけ節約側にします。': 'The picture is split into tiles for processing. Auto is usually fine. Choose a memory-saving option only if processing fails for lack of memory.',
    # 使うGPUのヘルプ
    '通常は自動でOK。GPUが複数あるPCで、使うGPUを固定したい時だけ番号を選びます。': 'Auto is usually fine. Choose a number only on a PC with several GPUs, when you want to fix which one is used.',
    # 出力フォルダ名のヘルプ
    '出力をまとめるフォルダ名です。上段の出力先が「元の場所」なら、元画像の横にこの名前のフォルダを作ります。': 'The name of the folder that collects the output. If Output is set to the same folder as the source, a folder with this name is created next to the source file.',
    # 高速保存のヘルプ
    '動画の書き出しにGPUを使います。対応していれば速くなります。失敗時は通常エンコードに戻します。': 'Uses the GPU to encode videos. It is faster when the GPU supports it. If it fails, normal encoding is used instead.',
    # 音声保持のヘルプ
    '元動画の音声を、新しく作る動画にも入れます。': 'Copies the audio of the source video into the new video.',
    # TTAのヘルプ
    'TTAは同じ画像を反転などで複数回処理して仕上げる高品質モードです。少し良くなる場合がありますが、かなり遅くなります。': 'TTA processes the same picture several times, flipped and so on, and combines the results. It can look slightly better but is much slower.',
    # 出力フォルダ作成のヘルプ
    'チェックすると、出力を指定名のフォルダにまとめます。外すと入力ファイルと同じ場所へ直接出力します。': 'When on, the output goes into a folder with the given name. When off, it is saved directly next to the source file.',
    # 補間倍率チェックボックスのヘルプ
    '入れると「補間後のfps」を選べるようになります。切っている間は、元動画の 2 倍になります。': 'Turn this on to choose "Interpolated frame rate". While it is off, the frame rate is twice the source.',
    # 補間後fpsのヘルプ
    'フレーム補間後の滑らかさです。通常は元動画の2倍を選びます。指定fpsが元動画以下なら処理できません。': "How smooth the video is after frame interpolation. Twice the source is the usual choice. A frame rate that isn't higher than the source can't be processed.",
    # 処理順のヘルプ
    'アップスケールとフレーム補間を両方行うときの順番です。通常は「アプコン→補間」が速くおすすめ。高解像度でメモリ不足になるときだけ「補間→アプコン」にします。FILM (Style) は、メモリを抑えるため常に「補間 → アプコン」の順で処理します。': 'The order used when both upscaling and frame interpolation run. Upscale → interpolate is faster and is the usual choice. Choose Interpolate → upscale only when a high-resolution result runs out of memory. FILM (Style) always uses Interpolate → upscale, to keep memory use down.',
    # --- 追加キット
    # 詳細設定内の見出し
    '追加キット': 'Add-on kits',
    # 追加キットの説明文
    'ダウンロードした zip を、ultraeasy-upscaler.exe のあるフォルダに展開してください。次回の起動から使えます。': 'Extract the downloaded zip into the folder that contains ultraeasy-upscaler.exe. It becomes available the next time the app starts.',
    # 追加キットの行の名前
    'NPU キット': 'NPU kit',
    # 追加キットの行の名前
    'AdcSR（GPU 用）': 'AdcSR (for GPU)',
    # 追加キットの行の名前
    'AdcSR（NPU 用）': 'AdcSR (for NPU)',
    # 追加キットの行の状態（未導入）
    '未導入': 'Not installed',
    # 追加キットの行の状態（導入済み）
    '導入済み': 'Installed',
    # 追加キットの行のボタン
    'ダウンロード': 'Download',
    # --- NPU の準備
    # 詳細設定内の見出し／未変換案内の文中の名称（2箇所で同じ文言）
    'NPU の準備': 'NPU setup',
    # NPU欄の説明文
    'NPU で使うモデルは、最初に一度だけ変換が必要です。使うモデルだけ変換してください。変換中も PC は使えますが、SwinIR-M と AdcSR はメモリを多く使います。': 'A model must be converted once before it can run on the NPU. Convert only the models you will use. You can keep using the PC during a conversion, but SwinIR-M and AdcSR use a lot of memory.',
    # モデル行の状態（変換済み）
    '変換済み': 'Converted',
    # モデル行の状態（未変換）
    '未変換': 'Not converted',
    # モデル行の状態（直前の変換が失敗）
    '失敗': 'Failed',
    # モデル行のボタン（未変換・失敗時）
    'NPU 用に変換': 'Convert for NPU',
    # 変換完了の状況行
    '変換が終わりました。': 'The conversion has finished.',
    # 変換失敗の状況行。{message}=整形済みエラー文
    '変換できませんでした: {message}': "Couldn't convert it: {message}",
    # 本処理・試し実行中の変換ボタンの状況行
    '処理中は変換できません': "You can't convert while files are being processed",
    # 変換中の状態表示。{body}=経過時間（例: 12:34、1時間以上は1:02:03）
    '変換中（経過 {body}）': 'Converting ({body} elapsed)',
    # 未変換行の目安。{minutes}=分数（例: 15）
    '初回変換の目安: 約{minutes}分': 'Estimated time: about {minutes} min',
    # 未変換行の目安（60分以上）。{hours}=時間（例: 2）
    '初回変換の目安: 約{hours}時間': 'Estimated time: about {hours} h',
    # 未変換行の目安（端数あり）。{hours}=時間、{rest}=分（例: 1と30）
    '初回変換の目安: 約{hours}時間{rest}分': 'Estimated time: about {hours} h {rest} min',
    # 変換の内部エラー文。{model}=モデルキー
    'NPU で変換できないモデルです: {model}': "This model can't be converted for the NPU: {model}",
    # 変換の内部エラー文。{model}=モデルキー
    'NPU 用のファイルが揃っていません: {model}': 'Files for the NPU are missing: {model}',
    # --- ファイル選択ダイアログ
    # ファイル選択ダイアログの題名
    'ファイルを選択（複数可）': 'Choose files',
    # ファイル選択ダイアログのフィルタ名（拡張子は変えない）
    '対応ファイル (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.flv *.mpg *.mpeg *.ts *.m2ts);;すべて (*.*)': 'Supported files (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.mp4 *.mkv *.mov *.avi *.webm *.m4v *.wmv *.flv *.mpg *.mpeg *.ts *.m2ts);;All files (*.*)',
    # フォルダ選択ダイアログの題名
    'フォルダを選択': 'Choose a folder',
    # --- キュー行
    # 行の種別表示
    'フォルダ': 'Folder',
    # 行の状態表示／ジョブの既定文言（2箇所で同じ文言）
    '待機中': 'Waiting',
    # 行の状態表示
    '解析中…': 'Analyzing…',
    # 行の状態表示
    'エラー': 'Error',
    # 行の状態表示
    'キャンセル': 'Canceled',
    # 行のやり直しボタンのツールチップ
    'やり直す': 'Redo',
    # 行の削除ボタンのツールチップ
    '一覧から削除 / 処理中ならキャンセル': 'Remove from the list, or cancel if it is being processed',
    # 行の削除ボタンのツールチップ（待機中）
    '一覧から削除': 'Remove from the list',
    # 行の削除ボタンのツールチップ（処理中）
    '処理を中止': 'Cancel processing',
    # 完了した行の保存先ボタンのツールチップ
    '保存先を開く': 'Open the output folder',
    # 完了行のツールチップ。{path}=出力パス
    '出力先: {path}': 'Output: {path}',
    # 行のツールチップ（設定の内訳）。{backend}=実行先ID、{upscale}=モデル名（4x）か「なし」、{interpolation}=補間モデルか「なし」
    'AI実行先: {backend}\nアップスケール: {upscale}\nフレーム補間: {interpolation}': 'Run on: {backend}\nUpscaling: {upscale}\nFrame interpolation: {interpolation}',
    # 行のツールチップ内の「なし」
    'なし': 'None',
    # --- 進捗: 画像・フォルダ
    # 画像・フォルダ処理の進捗（複数箇所で同じ文言）
    'アップスケール中…': 'Upscaling…',
    # Vulkan画像処理の進捗。{pct}=0〜100の整数
    'アップスケール中… {pct}%': 'Upscaling… {pct}%',
    # フォルダ処理の開始時
    'フォルダを一括アップスケール中…': 'Upscaling the folder…',
    # 空フォルダ・空キューの完了表示（複数箇所で同じ文言）
    '0/0 枚': '0/0 images',
    # フォルダ処理の進捗。{done}=済み枚数、{total}=全枚数
    '{done}/{total} 枚': '{done}/{total} images',
    # NPUフォルダ処理の進捗。差し込みは上と同様
    '{index}/{total} 枚': '{index}/{total} images',
    # 旧NPUフォルダ処理の進捗。{message}=その画像の進捗文
    '{index}/{total} 枚: {message}': '{index}/{total} images: {message}',
    # フォルダ処理の進捗。{i}=何枚目、{message}=その画像の進捗文
    '{i}/{total} 枚 {message}': '{i}/{total} images {message}',
    # NPUフォルダ処理のGPU再処理の進捗
    '{index}/{total} 枚 NPU 2段で失敗したためGPUで再処理…': '{index}/{total} images: failed on the NPU, processing again on the GPU…',
    # --- 進捗: 動画
    # 動画処理の開始時
    '動画を解析中…': 'Analyzing the video…',
    # 動画処理の抽出開始時
    'フレームを抽出中…': 'Extracting frames…',
    # 抽出の下請け進捗の代替文言
    'フレーム抽出中…': 'Extracting frames…',
    # 抽出の完了時
    'フレーム抽出完了': 'Frames extracted',
    # 補間の進捗（複数箇所で同じ文言）
    'RIFEでフレーム補間中…': 'Interpolating frames with RIFE…',
    # FILM補間の進捗（複数箇所で同じ文言）
    'FILM (Style) でフレーム補間中…': 'Interpolating frames with FILM (Style)…',
    # 補間の完了時
    'フレーム補間完了': 'Frame interpolation finished',
    # 動画フレーム拡大の開始時
    'フレームをアップスケール中…': 'Upscaling frames…',
    # 新経路の拡大進捗。{processed}=済み、{total}=全数
    '動画をアップスケール中… {processed}/{total}フレーム': 'Upscaling the video… {processed}/{total} frames',
    # 新経路の拡大進捗（総数不明時）。{processed}=済み数
    '動画をアップスケール中… {processed}フレーム': 'Upscaling the video… {processed} frames',
    # 再結合前の自動縮小。{width}{height}=縮小後の寸法（例: 3840と2160）
    '出力を{width}x{height}へ縮小（H.264上限のため）': 'Reducing the output to {width}x{height} (the H.264 limit)',
    # SwinIR動画の拡大進捗。{done}=済みチャンク数、{total}=全チャンク数
    'SwinIR動画 {done}/{total}フレーム': 'SwinIR video: {done}/{total} frames',
    # 再結合の開始時
    '動画を再結合中…': 'Putting the video back together…',
    # 再結合の下請け進捗の代替文言
    '再結合中…': 'Putting the video back together…',
    # 再結合の完了時
    '再結合完了': 'Video finished',
    # ヘルパー失敗時のVulkan退避。{model}=代替モデル名、{error}=失敗文
    'Vulkanへ切替（モデル: {model} で代替） ({error})': 'Switched to Vulkan (using {model} instead) ({error})',
    # --- 進捗: ヘルパー起動・NPU最適化
    # ヘルパー起動の進捗（複数箇所で同じ文言）
    'AI準備中…': 'Preparing the AI…',
    # NPU起動の完了時
    'NPU 準備完了': 'NPU ready',
    # AdcSR前半の初回最適化
    'NPU 前半を最適化中 1/2（初回のみ。次回はキャッシュを利用）': 'Converting the first half for the NPU, 1/2 (first time only)',
    # AdcSR前半の最適化の経過。{elapsed}=MM:SS
    'NPU 前半を最適化中 1/2（経過 {elapsed}。この検証機では約93分）': 'Converting the first half for the NPU, 1/2 ({elapsed} elapsed; about 93 min on the test PC)',
    # AdcSR後半の最適化の経過。{elapsed}=MM:SS
    'NPU 後半を最適化中 2/2（経過 {elapsed}。この検証機では約30分）': 'Converting the second half for the NPU, 2/2 ({elapsed} elapsed; about 30 min on the test PC)',
    # NPU起動の動作検査の経過。{elapsed}=MM:SS
    'NPU 動作検査中（経過 {elapsed}）': 'Checking that the NPU works ({elapsed} elapsed)',
    # NPU初回最適化の進捗
    '初回のみNPU最適化中（数分〜1時間・次回はキャッシュを利用）': 'Converting the model for the NPU (first time only; minutes to an hour)',
    # SwinIR起動の進捗
    'SwinIR-MをCUDAへ読み込み中…': 'Loading SwinIR-M onto the NVIDIA GPU…',
    # AdcSR選択時の自動切替の進捗
    'AdcSRはNPU非対応のためGPUで実行…': "AdcSR can't run on the NPU here, so it runs on the GPU…",
    # 小さい入力の自動切替の進捗
    '短辺480px未満のためGPUへ自動切替…': 'The short side is under 480 px, so it runs on the GPU…',
    # AdcSR画像のGPU再処理の進捗
    'NPU 2段で失敗したためGPUで再処理…': 'Failed on the NPU, processing again on the GPU…',
    # 旧NPU経路の前処理の進捗
    'NPU前処理中…': 'Preparing for the NPU…',
    # 旧NPU経路のタイル進捗。{idx}=何個目、{total}=全タイル数
    '{idx}/{total} タイル': '{idx}/{total} tiles',
    # 旧NPU経路の後処理の進捗
    'NPU後処理中…': 'Finishing after the NPU…',
    # 旧NPU経路の読み込みの進捗
    '画像を読み込み中…': 'Loading the image…',
    # 旧NPU経路の書き出しの進捗
    '画像を書き出し中…': 'Saving the image…',
    # 旧NPUフォルダ処理の画像書き出しの進捗
    '書き出し中…': 'Saving…',
    # SwinIR動画の再開確認の進捗
    'SwinIR動画の再開データを確認中…': 'Checking the data for resuming the SwinIR video…',
    # SwinIR動画の総数確認の進捗
    'SwinIR動画のフレーム数を確認中…': 'Counting the frames of the SwinIR video…',
    # SwinIR動画の結合待ちの進捗
    'SwinIR動画のチャンクを結合中…': 'Joining the parts of the SwinIR video…',
    # --- エラー: 設定・形式・試し
    # 画像ジョブの設定不足
    '画像にはアップスケーラーモデルを選択してください。': 'Choose an upscaling model for images.',
    # フォルダジョブの設定不足
    '画像フォルダにはアップスケーラーモデルを選択してください。': 'Choose an upscaling model for image folders.',
    # 動画にAdcSRを選んだとき
    'AdcSRは静止画専用です。動画には他のモデルを選んでください': 'AdcSR is for still images only. Choose another model for videos',
    # SwinIR動画に補間を併せたとき
    'SwinIR CUDA動画ではフレーム補間を併用できません': "Frame interpolation can't be combined with SwinIR CUDA for videos",
    # 動画ジョブの両方未選択
    'アップスケーラーモデルまたはフレーム補間モデルを選択してください。': 'Choose an upscaling model or a frame interpolation model.',
    # 内部エラー文。{kind}=種別ID
    '未知のジョブ種別: {kind}': 'Unknown kind of item: {kind}',
    # 対応外ファイルの追加時。{name}=ファイル名
    '未対応の形式です: {name}': 'Unsupported format: {name}',
    # 試しの範囲切り出しの失敗時
    '範囲が画像の外です': 'The area is outside the picture',
    # 試しのコマ抽出の失敗時。{error}=失敗内容
    'コマを取り出せませんでした: {error}': "Couldn't get the frame: {error}",
    # 補間設定の内部エラー文
    'フレーム補間モデルが選択されていません。': 'No frame interpolation model is chosen.',
    # 補間設定の内部エラー文
    '元動画のfpsを取得できません。': "Couldn't read the frame rate of the source video.",
    # 補間設定の内部エラー文
    'フレーム補間には2枚以上のフレームが必要です。': 'Frame interpolation needs at least two frames.',
    # 補間設定の内部エラー文。{fps}=元動画のfps（例: 29.970）
    '補間後のfpsは元動画より大きい値にしてください（元: {fps:.3f}fps）。': 'The frame rate after interpolation must be higher than the source ({fps:.3f} fps).',
    # FILM補間のfps制約。{source}=元動画のfps、{requested}=指定fps（例: 24.000）
    'FILM (Style) は元動画の 2 倍・4 倍・8 倍の fps だけに対応します（元: {source}fps / 指定: {requested}fps）。': 'FILM (Style) supports only 2, 4, or 8 times the source frame rate (source: {source} fps, requested: {requested} fps).',
    # FILM補間実行の失敗時。{detail}=実行ログ末尾
    'FILM (Style) の処理に失敗しました:\n{detail}': 'FILM (Style) failed:\n{detail}',
    # FILM補間結果の検証時。{expected}=予定枚数、{actual}=実際
    'FILM (Style) が作ったフレームの数が合いません（予定 {expected} / 実際 {actual}）。': 'FILM (Style) produced a different number of frames than expected (expected {expected}, got {actual}).',
    # 補間実行の失敗時。{tail}=実行ログ末尾
    'RIFEが失敗しました:\n{tail}': 'RIFE failed:\n{tail}',
    # 補間結果の検証時。{planned}=予定枚数、{actual}=実際
    'RIFEの生成枚数が一致しません（予定 {planned} / 実際 {actual}）。': 'RIFE made a different number of frames than expected (expected {planned}, got {actual}).',
    # 寸法設定の形式エラー。{value}=入力値
    '動画の最大寸法は WIDTHxHEIGHT で指定してください: {value!r}': 'Give the maximum video size as WIDTHxHEIGHT: {value!r}',
    # 寸法設定の形式エラー。{value}=入力値
    '動画の最大寸法は (幅, 高さ) で指定してください: {value!r}': 'Give the maximum video size as (width, height): {value!r}',
    # 寸法設定の範囲エラー。{width}{height}=入力寸法
    '動画の最大寸法は2以上で指定してください: {width}x{height}': 'The maximum video size must be 2 or more: {width}x{height}',
    # 寸法設定の内部エラー文
    '動画の寸法が不正です: {width}x{height}': 'Invalid video size: {width}x{height}',
    # HDR動画の入力時（複数箇所で同じ文言）
    'HDR動画（PQ/HLG）は未対応です。SDRに変換してから処理してください': "HDR video (PQ/HLG) isn't supported. Convert it to SDR first",
    # SwinIR設定の内部エラー文
    'SwinIR CUDAチャンク経路にはSWINIR_CUDA backendが必要です': 'SwinIR CUDA video processing needs Run on set to SwinIR-M (CUDA)',
    # SwinIR設定の内部エラー文
    'SwinIR CUDA経路ではSwinIR-Mモデルを選択してください': 'Choose the SwinIR-M model when running on SwinIR CUDA',
    # SwinIR設定の内部エラー文
    'SwinIRチャンク処理には共有decoderが必要です': 'SwinIR video processing needs a shared decoder',
    # 倍率設定の内部エラー文
    '新しいGPU/NPUバックエンドは4xモデル専用です。倍率を4xにしてください。': 'GPU and NPU processing supports 4x only. Set the scale to 4x.',
    # モデル設定の内部エラー文
    'CUDA版SwinIRはSwinIR-Mモデル専用です。': 'SwinIR CUDA supports the SwinIR-M model only.',
    # タイル設定の内部エラー文
    'SwinIRのタイルサイズは8の倍数にしてください。': 'The SwinIR tile size must be a multiple of 8.',
    # 旧NPU経路の倍率エラー文
    'NPU backend は現在 x4 のみ対応です。倍率を 4x にしてください。': 'NPU processing supports 4x only. Set the scale to 4x.',
    # Vulkanモデル設定の内部エラー文。{model}=モデル名、{scale}=倍率、{models_dir}=モデル場所
    "モデル '{model}' は倍率 x{scale} に対応していません（{models_dir} に該当 param がありません）。": "The model '{model}' doesn't support {scale}x (no matching param file in {models_dir}).",
    # --- エラー: モデル解決・ヘルパー起動
    # モデル解決の内部エラー文。{backend}=実行先ID、{model}=モデルキー、{tile}=タイル
    '対応モデルがありません: backend={backend}, model={model}, tile={tile}': 'No matching model: backend={backend}, model={model}, tile={tile}',
    # モデル解決の内部エラー文。{filename}=ファイル名、{dirs}=探索先一覧
    'AIモデルが見つかりません: {filename}\n探索先: {dirs}': 'Model file not found: {filename}\nLooked in: {dirs}',
    # ヘルパー指定の内部エラー文。{env}=環境変数名（例: UEU_WINML_HELPER）、{path}=指定場所
    '{env} のファイルが見つかりません: {path}': 'The file set in {env} was not found: {path}',
    # GPUヘルパー不在時。{env}=環境変数名
    'winml-sr.exeが見つかりません。tools/winml-srをビルドするか、{env}を指定してください。': 'winml-sr.exe was not found. Build tools/winml-sr, or set {env}.',
    # NPU実行環境不在時。{path}=探した場所
    'NPU用Pythonが見つかりません: {path}': 'Python for the NPU was not found: {path}',
    # NPUキット不在時。{path}=探した場所
    'NPUワーカーが見つかりません: {path}': 'The NPU worker was not found: {path}',
    # SwinIR実行環境不在時。{env}=環境変数名、{path}=探した場所
    'SwinIR用Pythonが見つかりません。scripts\\setup_swinir.ps1を実行するか、{env}を指定してください: {path}': 'Python for SwinIR was not found. Run scripts\\setup_swinir.ps1, or set {env}: {path}',
    # SwinIR作業ファイル不在時。{path}=探した場所
    'SwinIRワーカーが見つかりません: {path}': 'The SwinIR worker was not found: {path}',
    # SwinIR重み不在時。{env}=環境変数名、{path}=探した場所
    'SwinIR-Mモデルが見つかりません。scripts\\setup_swinir.ps1を実行するか、{env}を指定してください: {path}': 'The SwinIR-M model was not found. Run scripts\\setup_swinir.ps1, or set {env}: {path}',
    # SwinIR実行環境不在時（動画経路）。{path}=探した場所
    'SwinIR CUDA用Pythonが見つかりません: {path}': 'Python for SwinIR CUDA was not found: {path}',
    # SwinIR作業ファイル不在時（動画経路）。{path}=探した場所
    'SwinIR CUDAワーカーが見つかりません: {path}': 'The SwinIR CUDA worker was not found: {path}',
    # SwinIR重み不在時（動画経路）。{path}=探した場所
    'SwinIR CUDAの重みが見つかりません: {path}': 'The SwinIR CUDA model file was not found: {path}',
    # AdcSR用ファイル不足時
    'AdcSR の NPU 2段モード用ファイル (front/back/manifest) が見つかりません': 'The files for running AdcSR on the NPU (front/back/manifest) were not found',
    # 旧NPU経路の未対応モデル時。{key}=モデル名
    "モデル '{key}' はNPUバックエンドに対応していません。GPU (Vulkan) を選ぶか、NPU対応モデルに切り替えてください。": "The model '{key}' can't run on the NPU. Choose GPU (Vulkan), or switch to a model that supports the NPU.",
    # 外部バイナリ不在時。{name}=バイナリ名（例: ffmpeg）
    '{name} が見つかりません（PATH を確認してください）。': '{name} was not found (check PATH).',
    # Vulkan実行ファイル不在時。{exe}=探した場所
    'realesrgan-ncnn-vulkan.exe が見つかりません: {exe}\nvendor/realesrgan/ に展開してください。': 'realesrgan-ncnn-vulkan.exe was not found: {exe}\nExtract it into vendor/realesrgan/.',
    # RIFE実行ファイル不在時。{base}=探した場所
    'rife-ncnn-vulkan.exe が見つかりません: {base}\nモデル取得スクリプトを実行してください。': 'rife-ncnn-vulkan.exe was not found: {base}\nRun the script that downloads the models.',
    # FILM実行ファイル不在時（キット未導入）
    'FILM (Style) の実行ファイルが見つかりません。FILM キットを追加してください。': 'The FILM (Style) program was not found. Add the FILM kit.',
    # FILM実行ファイル不在時。{path}=探した場所
    'FILM (Style) の実行ファイルが見つかりません: {path}': 'The FILM (Style) program was not found: {path}',
    # FILMモデル不在時。{path}=探した場所
    'FILM (Style) のモデルが見つかりません: {path}': 'The FILM (Style) model was not found: {path}',
    # RIFEモデル不在時。{path}=探した場所
    'フレーム補間モデルが見つかりません: {path}': 'The frame interpolation model was not found: {path}',
    # 旧NPU経路のモデル不在時。{name}=モデル名、{model}=探した場所
    'NPU ONNXモデルが見つかりません ({name}): {model}': 'The ONNX model for the NPU was not found ({name}): {model}',
    # 旧NPU経路のconda不在時
    'conda が見つかりません。Miniforge / Ryzen AI 環境を確認してください。': 'conda was not found. Check the Miniforge / Ryzen AI environment.',
    # ヘルパー起動失敗時。{command}=実行ファイル
    'AI helperを起動できません: {command}': "Couldn't start the AI helper: {command}",
    # ヘルパー起動待ち超過時。{timeout}=秒数
    'AI helperの準備が{timeout:g}秒以内に完了しませんでした': "The AI helper wasn't ready within {timeout:g} seconds",
    # ヘルパー通信失敗時
    'AI helperへの送信に失敗しました': "Couldn't send data to the AI helper",
    # ヘルパー送信待ち超過時。{timeout}=秒数
    'AI helperへの送信が{timeout:g}秒以内に完了しませんでした': "Sending data to the AI helper didn't finish within {timeout:g} seconds",
    # ヘルパー通信失敗時（詳細あり）。{error}=失敗内容
    'AI helperへの送信に失敗しました: {error}': "Couldn't send data to the AI helper: {error}",
    # ヘルパー異常終了時。{need}=要求バイト数、{got}=受信済み、{exit}=終了コード
    'AI helperがstdoutを閉じました (need={need}, got={got}, exit={exit})': 'The AI helper stopped unexpectedly (need={need}, got={got}, exit={exit})',
    # 旧NPU経路の実行環境エラー。{available}=利用可能な実行先一覧
    'VitisAIExecutionProvider が利用できません。{available}': 'VitisAIExecutionProvider is not available. {available}',
    # 旧NPU経路のNPU不在時
    '対応するRyzen AI NPUが見つかりません。': 'No supported Ryzen AI NPU was found.',
    # --- エラー: 実行時の失敗
    # Vulkan実行の失敗時。{ret}=終了コード、{cmd}=実行行、{tail}=実行ログ末尾
    'realesrgan-ncnn-vulkan が失敗しました (exit={ret})\ncmd: {cmd}\n{tail}': 'realesrgan-ncnn-vulkan failed (exit={ret})\ncmd: {cmd}\n{tail}',
    # Vulkan画像出力の検証時。{out}=出力パス、{tail}=実行ログ末尾
    '出力ファイルが生成されませんでした: {out}\n{tail}': 'No output file was created: {out}\n{tail}',
    # 動画処理の失敗時。{ret}=終了コード、{tail}=実行ログ末尾
    'ffmpeg が失敗しました (exit {ret}):\n{tail}': 'ffmpeg failed (exit {ret}):\n{tail}',
    # 新経路の動画出力の検証時。{out}=出力パス
    '動画出力が生成されませんでした: {out}': 'No output video was created: {out}',
    # 新経路の入力解析時。{path}=入力パス
    '動画の解像度を取得できません: {path}': "Couldn't read the size of the video: {path}",
    # 新経路の失敗時（詳細なし）。{label}=動画エンコード／動画デコード
    '{label}に失敗しました': '{label} failed',
    # 新経路の失敗時。{label}=上と同様、{tail}=実行ログ末尾
    '{label}に失敗しました:\n{tail}': '{label} failed:\n{tail}',
    # 新経路の失敗時。{returncode}=終了コード
    '{label}に失敗しました（ffmpeg exit {returncode}、stderrなし）': '{label} failed (ffmpeg exit {returncode}, no error output)',
    # 新経路のパイプ切断時
    '{label}に失敗しました（ffmpegとのパイプが閉じられました。ffmpegのstderrは取得できませんでした）': '{label} failed (the connection to ffmpeg was closed, and no error output could be read)',
    # 新経路の失敗文の工程名
    '動画エンコード': 'Video encoding',
    # 新経路の失敗文の工程名
    '動画デコード': 'Video decoding',
    # 旧NPU経路の失敗時。{ret}=終了コード、{tail}=実行ログ末尾
    'NPU backend が失敗しました (exit={ret})\n{tail}': 'NPU processing failed (exit={ret})\n{tail}',
    # 旧NPU経路の出力検証時。{out}=出力パス
    'NPU出力ファイルが生成されませんでした: {out}': 'No output file was created by the NPU: {out}',
    # 旧NPU経路の画像処理時。{path}=画像パス
    '画像をエンコードできませんでした: {path}': "Couldn't encode the image: {path}",
    # 旧NPU経路の画像処理時。{path}=画像パス
    '画像を書き出せませんでした: {path}': "Couldn't save the image: {path}",
    # 旧NPU経路の画像処理時。{path}=画像パス
    '画像を読めませんでした: {path}': "Couldn't read the image: {path}",
    # SwinIR動画の復号失敗時。{tail}=実行ログ末尾
    'SwinIR動画のデコードに失敗しました:\n{tail}': 'Decoding the SwinIR video failed:\n{tail}',
    # SwinIR動画の検証時。{detail}=総数+超過数（例: 100+2）
    'SwinIR動画のフレーム数が事前確認後に変化しました: {detail}': 'The number of frames in the SwinIR video changed after it was counted: {detail}',
    # SwinIR動画の総数確認時。{tail}=実行ログ末尾
    'SwinIR動画のフレーム数確認に失敗しました:\n{tail}': 'Counting the frames of the SwinIR video failed:\n{tail}',
    # SwinIR動画の総数確認時
    'SwinIR動画のCFR変換後フレーム数を取得できません': "Couldn't get the number of frames of the SwinIR video after converting to a constant frame rate",
    # SwinIR動画の復号時
    'SwinIR動画decoderのフレーム数が不足しています': 'The SwinIR video decoder returned too few frames',
    # SwinIR動画の検証時。{decoded}=復号数、{count}=予定数
    'SwinIRチャンクのフレーム数が不足しています: {decoded}/{count}': 'A part of the SwinIR video has too few frames: {decoded}/{count}',
    # SwinIR動画の結合時。{tail}=実行ログ末尾
    'SwinIRチャンクのffmpeg処理に失敗しました:\n{tail}': 'ffmpeg failed on a part of the SwinIR video:\n{tail}',
    # SwinIR動画の結合時
    'SwinIRチャンクが生成されませんでした': 'No part of the SwinIR video was created',
    # SwinIR動画の結合時。{tail}=実行ログ末尾
    'SwinIRチャンクの結合に失敗しました:\n{tail}': 'Joining the parts of the SwinIR video failed:\n{tail}',
    # SwinIR動画の出力検証時。{out}=出力パス
    'SwinIR動画出力が生成されませんでした: {out}': 'No SwinIR output video was created: {out}',
    # SwinIR動画の入力解析時
    'SwinIR動画の解像度を取得できません': "Couldn't read the size of the SwinIR video",
}

