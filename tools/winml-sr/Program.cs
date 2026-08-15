// Windows ML で自前ONNX（Real-ESRGAN系・固定256x256タイル）を実行するPoC。
//
//   winml-sr list [--download]
//   winml-sr run --model <onnx> --input <img> --output <img>
//                [--ep-policy npu|gpu|cpu|default|power|perf|efficiency]
//                [--ep-name VitisAIExecutionProvider [--device-type NPU]]
//                [--overlap 16] [--compile] [--download] [--warmup 2]
//   winml-sr psnr --a <img> --b <img>
//
// タイル分割/結合は ultraeasy-upscaler app/core/npu_runner.py と同一ロジック
// （reflectパディング→オーバーラップ付き切り出し→コア領域のみ合成）。
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Windows.AI.MachineLearning;

internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }

        try
        {
            if (args.Length == 0 || Has(args, "--help") || Has(args, "-h"))
            {
                PrintUsage();
                return 0;
            }

            switch (args[0].ToLowerInvariant())
            {
                case "list":
                    return await ListAsync(args, Has(args, "--download"));
                case "run":
                    return await RunAsync(args);
                case "serve":
                    return await ServeAsync(args);
                case "psnr":
                    return Psnr(Required(args, "--a"), Required(args, "--b"));
                default:
                    Console.Error.WriteLine($"Unknown command: {args[0]}");
                    PrintUsage();
                    return 2;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"{ex.GetType().Name}: {ex.Message}");
            if (ex.InnerException is not null)
            {
                Console.Error.WriteLine($"  inner: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}");
            }
            return 1;
        }
    }

    // ---------------------------------------------------------------- commands

    private static async Task<int> ListAsync(string[] args, bool allowDownload)
    {
        AddPackageDependencies(args);
        using OrtEnv env = CreateEnv();
        RegisterEpLibraries(env, args);
        await InitializeProvidersAsync(allowDownload);
        PrintEpDevices(env);
        return 0;
    }

    private static async Task<int> RunAsync(string[] args)
    {
        string modelPath = Path.GetFullPath(Required(args, "--model"));
        string inputPath = Path.GetFullPath(Required(args, "--input"));
        string outputPath = Path.GetFullPath(Required(args, "--output"));
        string? epPolicy = Value(args, "--ep-policy");
        string? epName = Value(args, "--ep-name");
        string? deviceType = Value(args, "--device-type");
        Dictionary<string, string> epOptions = ParseEpOptions(args);
        int overlap = int.Parse(Value(args, "--overlap") ?? "16");
        int warmup = int.Parse(Value(args, "--warmup") ?? "2");
        bool compile = Has(args, "--compile");
        bool download = Has(args, "--download");

        if (epPolicy is null && epName is null)
        {
            epPolicy = "npu";
        }

        var swTotal = Stopwatch.StartNew();
        AddPackageDependencies(args);
        using OrtEnv env = CreateEnv();
        RegisterEpLibraries(env, args);
        await InitializeProvidersAsync(download);
        PrintEpDevices(env);

        SessionOptions sessionOptions = BuildSessionOptions(env, epPolicy, epName, deviceType, epOptions);

        // EPコンパイル（EPContextモデル生成）: 初回のみ。以後はコンパイル済みを再利用。
        string actualModelPath = modelPath;
        if (compile)
        {
            string suffix = epPolicy ?? $"{epName}_{deviceType ?? "any"}";
            string compiledPath = Path.Combine(
                Path.GetDirectoryName(modelPath)!,
                $"{Path.GetFileNameWithoutExtension(modelPath)}_ctx_{Sanitize(suffix)}.onnx");
            if (!File.Exists(compiledPath))
            {
                Console.WriteLine($"[compile] {compiledPath}");
                var swCompile = Stopwatch.StartNew();
                using var compileOptions = new OrtModelCompilationOptions(sessionOptions);
                compileOptions.SetInputModelPath(modelPath);
                compileOptions.SetOutputModelPath(compiledPath);
                try
                {
                    compileOptions.CompileModel();
                    Console.WriteLine($"[compile] done in {swCompile.Elapsed.TotalSeconds:F1}s");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[compile] failed ({ex.Message}) -> 元モデルをそのまま使用");
                }
            }
            else
            {
                Console.WriteLine($"[compile] found precompiled: {compiledPath}");
            }
            if (File.Exists(compiledPath))
            {
                actualModelPath = compiledPath;
            }
        }

        Console.WriteLine($"[session] creating from {Path.GetFileName(actualModelPath)} ...");
        var swSession = Stopwatch.StartNew();
        using var session = new InferenceSession(actualModelPath, sessionOptions);
        Console.WriteLine($"[session] created in {swSession.Elapsed.TotalSeconds:F1}s");

        // 入出力メタデータ確認（固定 1x3xHxW / float32 前提）
        var (inputName, outputName, tileW, tileH, scale, outTileW, outTileH) = InspectModel(session);
        Console.WriteLine($"[model] input {tileW}x{tileH}, scale x{scale}, output {outTileW}x{outTileH}");

        using var tileRunner = new TileRunner(session, inputName, outputName, tileW, tileH, outTileW, outTileH);

        // 画像読み込み → RGB CHW float
        var (img, w, h) = LoadImageChw(inputPath);
        Console.WriteLine($"[image] {Path.GetFileName(inputPath)} {w}x{h}");

        // タイル分割 → 推論（ウォームアップ含む）→ 結合 → 保存
        var timings = new TileTimings();
        float[] merged = UpscaleChw(tileRunner, img, w, h, tileW, tileH, scale, overlap,
            warmup, timings, Console.WriteLine);
        SaveImageChw(merged, w * scale, h * scale, outputPath);

        // 統計
        var sorted = timings.TotalMs.OrderBy(v => v).ToList();
        var pureRunSorted = timings.PureRunMs.OrderBy(v => v).ToList();
        double median = sorted[sorted.Count / 2];
        Console.WriteLine();
        Console.WriteLine($"[result] {outputPath}");
        Console.WriteLine($"[timing] tiles={timings.TotalMs.Count}  median={median:F1} ms/tile  mean={timings.TotalMs.Average():F1} ms  " +
                          $"min={sorted.First():F1}  max={sorted.Last():F1}");
        Console.WriteLine($"[timing] pure-run median={pureRunSorted[pureRunSorted.Count / 2]:F1} ms  " +
                          $"mean={timings.PureRunMs.Average():F1} ms  total={timings.PureRunMs.Sum() / 1000.0:F2}s");
        Console.WriteLine($"[timing] tile-preprocess total={timings.PreprocessMs.Sum():F1} ms  " +
                          $"merge-copy total={timings.MergeMs.Sum():F1} ms  wall total={swTotal.Elapsed.TotalSeconds:F2}s");
        return 0;
    }

    private static int Psnr(string pathA, string pathB)
    {
        var (a, wa, ha) = LoadImageChw(pathA);
        var (b, wb, hb) = LoadImageChw(pathB);
        if (wa != wb || ha != hb)
        {
            Console.Error.WriteLine($"サイズ不一致: {wa}x{ha} vs {wb}x{hb}");
            return 1;
        }
        double mse = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (a[i] - b[i]) * 255.0;
            mse += d * d;
        }
        mse /= a.Length;
        double psnr = mse == 0 ? 99.0 : 10.0 * Math.Log10(255.0 * 255.0 / mse);
        Console.WriteLine($"PSNR: {psnr:F2} dB");
        return 0;
    }

    // ------------------------------------------------------------ Windows ML

    private static OrtEnv CreateEnv()
    {
        EnvironmentCreationOptions envOptions = new()
        {
            logId = "winml-sr",
            logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
        };
        return OrtEnv.CreateInstanceWithOptions(ref envOptions);
    }

    // --pkg-dep <PackageFamilyName> : MSIX動的依存でFrameworkパッケージをプロセスの
    // package graph に追加する（カタログの EnsureReady が内部でやっている環境構築の再現）。
    // OrtEnv 作成・EPライブラリロードより前に呼ぶこと。
    [DllImport("kernelbase.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern int TryCreatePackageDependency(
        IntPtr user, string packageFamilyName, ulong minVersion, int architectures,
        int lifetimeKind, string? lifetimeArtifact, int options, out IntPtr packageDependencyId);

    [DllImport("kernelbase.dll", CharSet = CharSet.Unicode, ExactSpelling = true)]
    private static extern int AddPackageDependency(
        IntPtr packageDependencyId, int rank, int options,
        out IntPtr packageDependencyContext, out IntPtr packageFullName);

    private static void AddPackageDependencies(string[] args)
    {
        for (int i = 1; i < args.Length - 1; i++)
        {
            if (!args[i].Equals("--pkg-dep", StringComparison.OrdinalIgnoreCase)) continue;
            string family = args[i + 1];
            int hr = TryCreatePackageDependency(IntPtr.Zero, family, 0UL,
                0x01 | 0x04 /* Neutral|X64 */, 0 /* Process lifetime */, null, 0, out IntPtr depId);
            if (hr < 0)
            {
                Console.WriteLine($"[pkg-dep] TryCreatePackageDependency({family}) failed: 0x{hr:X8}");
                continue;
            }
            string idText = Marshal.PtrToStringUni(depId) ?? "?";
            int hr2 = AddPackageDependency(depId, 0, 0, out _, out IntPtr fullNamePtr);
            if (hr2 < 0)
            {
                Console.WriteLine($"[pkg-dep] AddPackageDependency({family}) failed: 0x{hr2:X8}");
                continue;
            }
            Console.WriteLine($"[pkg-dep] added: {Marshal.PtrToStringUni(fullNamePtr)} (id={idText[..Math.Min(24, idText.Length)]}...)");
        }
    }

    /// <summary>--ep-lib NAME=PATH で自前EPライブラリを登録（Bring your own EP）。
    /// カタログ配信が止まったEPをローカルDLLから復活させる用。依存DLL解決のためPATHへ追加する。</summary>
    private static void RegisterEpLibraries(OrtEnv env, string[] args)
    {
        for (int i = 1; i < args.Length - 1; i++)
        {
            if (!args[i].Equals("--ep-lib", StringComparison.OrdinalIgnoreCase)) continue;
            string[] kv = args[i + 1].Split('=', 2);
            if (kv.Length != 2)
            {
                throw new ArgumentException($"--ep-lib は NAME=PATH 形式で指定してください: {args[i + 1]}");
            }
            string name = kv[0];
            string path = Path.GetFullPath(kv[1]);
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"EPライブラリが見つかりません: {path}");
            }
            string dir = Path.GetDirectoryName(path)!;
            string current = Environment.GetEnvironmentVariable("PATH") ?? "";
            if (!current.Split(';').Contains(dir, StringComparer.OrdinalIgnoreCase))
            {
                Environment.SetEnvironmentVariable("PATH", dir + ";" + current);
            }
            env.RegisterExecutionProviderLibrary(name, path);
            Console.WriteLine($"[ep] registered library: {name} <- {path}");
        }
    }

    private static async Task InitializeProvidersAsync(bool allowDownload)
    {
        var catalog = ExecutionProviderCatalog.GetDefault();
        var providers = catalog.FindAllProviders();
        if (providers is null || providers.Length == 0)
        {
            Console.WriteLine("[ep] catalog is empty");
            return;
        }

        foreach (var provider in providers)
        {
            try
            {
                var state = provider.ReadyState;
                Console.WriteLine($"[ep] {provider.Name}: {state} (lib='{provider.LibraryPath}')");
                if (allowDownload || state != ExecutionProviderReadyState.NotPresent)
                {
                    var op = provider.EnsureReadyAsync();
                    op.Progress = (_, p) => Console.Write($"\r[ep] {provider.Name}: downloading {p:F0}%   ");
                    var result = await op;
                    Console.WriteLine();
                    Console.WriteLine($"[ep] {provider.Name}: EnsureReady status={result.Status} " +
                                      $"error={(result.ExtendedError is null ? "none" : result.ExtendedError.Message)} " +
                                      $"diag='{result.DiagnosticText}' -> {provider.ReadyState}");
                }
                bool registered = provider.TryRegister();
                Console.WriteLine($"[ep] {provider.Name}: TryRegister={registered}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ep] {provider.Name}: 初期化失敗 ({ex.GetType().Name}: {ex.Message})");
            }
        }
    }

    private static void PrintEpDevices(OrtEnv env)
    {
        IReadOnlyList<OrtEpDevice> devices = env.GetEpDevices();
        Console.WriteLine("[ep] discovered devices:");
        foreach (var d in devices)
        {
            Console.WriteLine($"  {d.EpName,-34} {d.EpVendor,-14} {d.HardwareDevice.Type}");
        }
    }

    private static Dictionary<string, string> ParseEpOptions(string[] args)
    {
        var options = new Dictionary<string, string>();
        for (int i = 1; i < args.Length - 1; i++)
        {
            if (args[i].Equals("--ep-option", StringComparison.OrdinalIgnoreCase))
            {
                string[] kv = args[i + 1].Split('=', 2);
                if (kv.Length == 2) options[kv[0]] = kv[1];
            }
        }
        return options;
    }

    private static SessionOptions BuildSessionOptions(OrtEnv env, string? epPolicy, string? epName, string? deviceType,
        Dictionary<string, string>? epOptions = null)
    {
        var sessionOptions = new SessionOptions();

        if (epName is not null)
        {
            var devices = env.GetEpDevices()
                .Where(d => d.EpName.Equals(epName, StringComparison.OrdinalIgnoreCase))
                .Where(d => deviceType is null ||
                            d.HardwareDevice.Type.ToString().Equals(deviceType, StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (devices.Count == 0)
            {
                throw new InvalidOperationException($"EP '{epName}' (device={deviceType ?? "any"}) が見つかりません。'winml-sr list' で確認してください。");
            }
            Console.WriteLine($"[session] explicit EP: {epName} ({string.Join(",", devices.Select(d => d.HardwareDevice.Type))})");
            sessionOptions.AppendExecutionProvider(env, devices, new Dictionary<string, string>());
            return sessionOptions;
        }

        ExecutionProviderDevicePolicy policy = epPolicy!.ToLowerInvariant() switch
        {
            "npu" => ExecutionProviderDevicePolicy.PREFER_NPU,
            "gpu" => ExecutionProviderDevicePolicy.PREFER_GPU,
            "cpu" => ExecutionProviderDevicePolicy.PREFER_CPU,
            "power" => ExecutionProviderDevicePolicy.MIN_OVERALL_POWER,
            "perf" => ExecutionProviderDevicePolicy.MAX_PERFORMANCE,
            "efficiency" => ExecutionProviderDevicePolicy.MAX_EFFICIENCY,
            "default" => ExecutionProviderDevicePolicy.DEFAULT,
            _ => throw new ArgumentException($"unknown --ep-policy: {epPolicy}"),
        };
        Console.WriteLine($"[session] EP policy: {policy}");
        sessionOptions.SetEpSelectionPolicy(policy);
        return sessionOptions;
    }

    /// <summary>
    /// 固定shapeの入出力バッファを一度だけピン留めし、OrtValue経由で再利用する。
    /// session.Runの出力引数に同じOrtValueを渡すため、結果のManaged配列コピーは発生しない。
    /// </summary>
    private sealed class TileRunner : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions;
        private readonly string[] _inputNames;
        private readonly string[] _outputNames;
        private readonly OrtValue _inputValue;
        private readonly OrtValue _outputValue;
        private readonly OrtValue[] _inputValues;
        private readonly OrtValue[] _outputValues;

        public float[] InputBuffer { get; }
        public float[] OutputBuffer { get; }
        public int TileW { get; }
        public int TileH { get; }
        public int OutTileW { get; }
        public int OutTileH { get; }

        public TileRunner(InferenceSession session, string inputName, string outputName,
            int tileW, int tileH, int outTileW, int outTileH)
        {
            _session = session;
            TileW = tileW;
            TileH = tileH;
            InputBuffer = new float[3 * tileW * tileH];
            OutputBuffer = new float[3 * outTileW * outTileH];
            OutTileW = outTileW;
            OutTileH = outTileH;

            _inputValue = OrtValue.CreateTensorValueFromMemory<float>(
                InputBuffer, new long[] { 1, 3, tileH, tileW });
            _outputValue = OrtValue.CreateTensorValueFromMemory<float>(
                OutputBuffer, new long[] { 1, 3, outTileH, outTileW });
            _inputNames = new[] { inputName };
            _outputNames = new[] { outputName };
            _inputValues = new[] { _inputValue };
            _outputValues = new[] { _outputValue };
            _runOptions = new RunOptions();
        }

        public double RunPrepared()
        {
            var swRun = Stopwatch.StartNew();
            _session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);
            return swRun.Elapsed.TotalMilliseconds;
        }

        public void Dispose()
        {
            _runOptions.Dispose();
            _outputValue.Dispose();
            _inputValue.Dispose();
        }
    }

    // ------------------------------------------------------- shared run/serve

    /// <summary>入出力メタデータ検証（固定 1x3xHxW / float32 前提）。runと同一の検証。</summary>
    private static (string inputName, string outputName, int tileW, int tileH, int scale, int outTileW, int outTileH) InspectModel(
        InferenceSession session)
    {
        var inputMeta = session.InputMetadata.First();
        var outputMeta = session.OutputMetadata.First();
        int[] inDims = inputMeta.Value.Dimensions;
        int[] outDims = outputMeta.Value.Dimensions;
        if (inDims.Length != 4 || inDims[0] != 1 || inDims[1] != 3 || inDims[2] <= 0 || inDims[3] <= 0)
        {
            throw new InvalidOperationException($"想定外の入力shape: [{string.Join(",", inDims)}] (NCHW 1x3xHxW 固定モデルが必要)");
        }
        if (inputMeta.Value.ElementDataType != TensorElementType.Float)
        {
            throw new InvalidOperationException($"想定外の入力型: {inputMeta.Value.ElementDataType} (float32が必要)");
        }
        int tileH = inDims[2], tileW = inDims[3];
        if (outDims.Length != 4 || outDims[0] != 1 || outDims[1] != 3 ||
            outDims[2] <= 0 || outDims[3] <= 0 || outDims[2] % tileH != 0 || outDims[3] % tileW != 0)
        {
            throw new InvalidOperationException($"想定外の出力shape: [{string.Join(",", outDims)}] (入力の整数倍の1x3xH'xW'が必要)");
        }
        int scale = outDims[2] / tileH;
        if (outDims[3] / tileW != scale)
        {
            throw new InvalidOperationException($"縦横で異なるscaleは未対応: input={tileW}x{tileH}, output={outDims[3]}x{outDims[2]}");
        }
        if (outputMeta.Value.ElementDataType != TensorElementType.Float)
        {
            throw new InvalidOperationException($"想定外の出力型: {outputMeta.Value.ElementDataType} (float32が必要)");
        }
        return (inputMeta.Key, outputMeta.Key, tileW, tileH, scale, outDims[3], outDims[2]);
    }

    private sealed class TileTimings
    {
        public readonly List<double> TotalMs = new();
        public readonly List<double> PreprocessMs = new();
        public readonly List<double> PureRunMs = new();
        public readonly List<double> MergeMs = new();

        public static double Median(IEnumerable<double> values)
        {
            var sorted = values.OrderBy(v => v).ToList();
            return sorted[sorted.Count / 2];
        }
    }

    /// <summary>
    /// タイル分割→（任意でウォームアップ）→タイル毎推論→結合。run/serve共用。
    /// タイルごとに入力充填、session.Run純粋時間、マージ書き写しを分離して計測する。
    /// log が null ならログを出さない。
    /// </summary>
    private static float[] UpscaleChw(
        TileRunner runner, float[] img, int w, int h, int tileW, int tileH,
        int scale, int overlap, int warmup, TileTimings timings, Action<string>? log)
    {
        // タイル分割（reflect padding + overlap）。タイル配列は作らず、runner.InputBufferへ直接充填する。
        var plan = TilePlan.Create(w, h, tileW, tileH, overlap);
        log?.Invoke($"[tiles] {plan.TileCount} tiles ({plan.NTilesX}x{plan.NTilesY}), overlap={overlap}");

        // ウォームアップ
        for (int i = 0; i < warmup; i++)
        {
            var swWarm = Stopwatch.StartNew();
            plan.FillTile(img, 0, runner.InputBuffer);
            runner.RunPrepared();
            log?.Invoke($"[warmup {i + 1}/{warmup}] {swWarm.Elapsed.TotalMilliseconds:F0} ms");
        }

        var merged = new float[3 * w * scale * h * scale];
        int outW = w * scale;
        int outH = h * scale;
        int outOverlap = overlap * scale;
        for (int tileIndex = 0; tileIndex < plan.TileCount; tileIndex++)
        {
            var swTile = Stopwatch.StartNew();

            var swPre = Stopwatch.StartNew();
            plan.FillTile(img, tileIndex, runner.InputBuffer);
            double preprocessMs = swPre.Elapsed.TotalMilliseconds;

            double pureRunMs = runner.RunPrepared();

            var swMerge = Stopwatch.StartNew();
            MergeTile(runner.OutputBuffer, merged, outW, outH, runner.OutTileW, runner.OutTileH,
                outOverlap, plan.NTilesX, tileIndex);
            double mergeMs = swMerge.Elapsed.TotalMilliseconds;

            timings.PreprocessMs.Add(preprocessMs);
            timings.PureRunMs.Add(pureRunMs);
            timings.MergeMs.Add(mergeMs);
            timings.TotalMs.Add(swTile.Elapsed.TotalMilliseconds);
        }

        return merged;
    }

    // ------------------------------------------------------------------- serve
    //
    // stdin/stdout バイナリプロトコル（int32は全てリトルエンディアン、RGB24行連続・padding無し）:
    //   準備完了(stdout): 'UEUH' + int32 scale + int32 tileW + int32 tileH
    //   フレーム要求(stdin): 'UEUF' + int32 w + int32 h + w*h*3 bytes RGB
    //   フレーム応答(stdout): 'UEUD' + int32 outW + int32 outH + outW*outH*3 bytes RGB
    //   エラー応答(stdout): 'UEUE' + int32 len + UTF-8メッセージ（処理継続）
    //   stdin EOF → 正常終了
    // serve中の stdout はプロトコル専用。ログは全て stderr。

    private const int MaxFrameDim = 16384;

    private static async Task<int> ServeAsync(string[] args)
    {
        // 既存ヘルパー(InitializeProvidersAsync等)は Console.WriteLine でログを出すため、
        // serve では Console.Out を stderr に付け替えて stdout をバイナリ専用にする。
        Console.SetOut(Console.Error);

        string modelPath = Path.GetFullPath(Required(args, "--model"));
        string? epPolicy = Value(args, "--ep-policy");
        string? epName = Value(args, "--ep-name");
        string? deviceType = Value(args, "--device-type");
        Dictionary<string, string> epOptions = ParseEpOptions(args);
        int overlap = int.Parse(Value(args, "--overlap") ?? "16");
        int warmup = int.Parse(Value(args, "--warmup") ?? "2");
        bool download = Has(args, "--download");

        if (epPolicy is null && epName is null)
        {
            epPolicy = "npu";
        }

        AddPackageDependencies(args);
        using OrtEnv env = CreateEnv();
        RegisterEpLibraries(env, args);
        await InitializeProvidersAsync(download);
        PrintEpDevices(env);

        SessionOptions sessionOptions = BuildSessionOptions(env, epPolicy, epName, deviceType, epOptions);

        Console.Error.WriteLine($"[session] creating from {Path.GetFileName(modelPath)} ...");
        var swSession = Stopwatch.StartNew();
        using var session = new InferenceSession(modelPath, sessionOptions);
        Console.Error.WriteLine($"[session] created in {swSession.Elapsed.TotalSeconds:F1}s");

        var (inputName, outputName, tileW, tileH, scale, outTileW, outTileH) = InspectModel(session);
        Console.Error.WriteLine($"[model] input {tileW}x{tileH}, scale x{scale}, output {outTileW}x{outTileH}");

        using var tileRunner = new TileRunner(session, inputName, outputName, tileW, tileH, outTileW, outTileH);

        // ウォームアップ（乱数ダミータイル）
        var rnd = new Random(12345);
        var dummy = new float[3 * tileW * tileH];
        for (int i = 0; i < dummy.Length; i++) dummy[i] = (float)rnd.NextDouble();
        for (int i = 0; i < warmup; i++)
        {
            var swWarm = Stopwatch.StartNew();
            dummy.AsSpan().CopyTo(tileRunner.InputBuffer);
            tileRunner.RunPrepared();
            Console.Error.WriteLine($"[warmup {i + 1}/{warmup}] {swWarm.Elapsed.TotalMilliseconds:F0} ms");
        }

        using Stream stdin = Console.OpenStandardInput();
        using Stream stdout = Console.OpenStandardOutput();

        // 準備完了通知
        var ready = new byte[16];
        "UEUH"u8.CopyTo(ready.AsSpan());
        BinaryPrimitives.WriteInt32LittleEndian(ready.AsSpan(4), scale);
        BinaryPrimitives.WriteInt32LittleEndian(ready.AsSpan(8), tileW);
        BinaryPrimitives.WriteInt32LittleEndian(ready.AsSpan(12), tileH);
        stdout.Write(ready, 0, ready.Length);
        stdout.Flush();
        Console.Error.WriteLine($"[serve] ready (scale x{scale}, tile {tileW}x{tileH}, overlap={overlap})");

        // 各段の仕掛かりを1フレームに制限する3段パイプライン。
        // inference段だけがTileRunnerを触るため、固定OrtValueの再利用も安全に維持できる。
        var received = new BlockingCollection<ServeFrame>(boundedCapacity: 1);
        var inferred = new BlockingCollection<ServeFrame>(boundedCapacity: 1);
        using var pipelineCts = new CancellationTokenSource();
        var failure = new PipelineFailure();
        void Fail(Exception ex)
        {
            failure.Set(ex);
            pipelineCts.Cancel();
        }

        var stats = new ServeStats();
        Task receiveTask = Task.Run(() => ReceiveLoop(stdin, received, pipelineCts.Token, Fail, stats));
        Task inferenceTask = Task.Run(() => InferenceLoop(received, inferred, pipelineCts.Token, Fail,
            tileRunner, scale, overlap, stats));
        Task sendTask = Task.Run(() => SendLoop(inferred, stdout, pipelineCts.Token, Fail, scale, stats));

        try
        {
            await Task.WhenAll(receiveTask, inferenceTask, sendTask);
        }
        catch (Exception ex)
        {
            Fail(ex);
        }

        pipelineCts.Cancel();
        stats.Print();
        if (failure.Exception is not null)
        {
            throw failure.Exception;
        }
        return 0;
    }

    private sealed class ServeFrame
    {
        public int Number { get; init; }
        public int Width { get; init; }
        public int Height { get; init; }
        public long StartedTimestamp { get; init; }
        public float[]? ImageChw { get; set; }
        public float[]? MergedChw { get; set; }
        public TileTimings? Timings { get; set; }
        public Exception? Error { get; set; }
        public double InputConvertMs { get; set; }
        public double OutputConvertMs { get; set; }
    }

    private sealed class PipelineFailure
    {
        public Exception? Exception;

        public void Set(Exception ex)
        {
            Interlocked.CompareExchange(ref Exception, ex, null);
        }
    }

    private sealed class ServeStats
    {
        private readonly object _gate = new();
        private readonly List<double> _pureRunMs = new();
        private readonly List<double> _preprocessMs = new();
        private readonly List<double> _mergeMs = new();
        private readonly List<double> _inputConvertMs = new();
        private readonly List<double> _outputConvertMs = new();
        private int _successFrames;
        private int _errorFrames;
        private int _tileCount;

        public void AddInput(double ms)
        {
            lock (_gate) _inputConvertMs.Add(ms);
        }

        public void AddTiming(TileTimings timings)
        {
            lock (_gate)
            {
                _successFrames++;
                _tileCount += timings.PureRunMs.Count;
                _pureRunMs.AddRange(timings.PureRunMs);
                _preprocessMs.AddRange(timings.PreprocessMs);
                _mergeMs.AddRange(timings.MergeMs);
            }
        }

        public void AddOutput(double ms)
        {
            lock (_gate) _outputConvertMs.Add(ms);
        }

        public void AddError()
        {
            lock (_gate) _errorFrames++;
        }

        public void Print()
        {
            lock (_gate)
            {
                if (_pureRunMs.Count == 0)
                {
                    Console.Error.WriteLine($"[timing] frames={_successFrames + _errorFrames} success={_successFrames} error={_errorFrames} pure-run median=n/a");
                    return;
                }

                Console.Error.WriteLine(
                    $"[timing] frames={_successFrames + _errorFrames} success={_successFrames} error={_errorFrames} " +
                    $"tiles={_tileCount} pure-run median={TileTimings.Median(_pureRunMs):F1} ms " +
                    $"mean={_pureRunMs.Average():F1} ms total={_pureRunMs.Sum() / 1000.0:F2}s");
                Console.Error.WriteLine(
                    $"[timing] tile-preprocess total={_preprocessMs.Sum():F1} ms  " +
                    $"merge-copy total={_mergeMs.Sum():F1} ms  " +
                    $"input-convert total={_inputConvertMs.Sum():F1} ms  " +
                    $"output-convert total={_outputConvertMs.Sum():F1} ms");
            }
        }
    }

    private static void ReceiveLoop(Stream stdin, BlockingCollection<ServeFrame> output,
        CancellationToken cancellationToken, Action<Exception> fail, ServeStats stats)
    {
        var magic = new byte[4];
        var header = new byte[8];
        int frameNo = 0;
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                long started = Stopwatch.GetTimestamp();
                int first = stdin.Read(magic, 0, 1);
                if (first == 0)
                {
                    Console.Error.WriteLine($"[serve] stdin EOF, exiting after {frameNo} frames");
                    return;
                }
                stdin.ReadExactly(magic, 1, 3);
                if (!magic.AsSpan().SequenceEqual("UEUF"u8))
                {
                    throw new InvalidDataException($"bad magic: {BitConverter.ToString(magic)} (protocol desync)");
                }

                stdin.ReadExactly(header);
                int w = BinaryPrimitives.ReadInt32LittleEndian(header);
                int h = BinaryPrimitives.ReadInt32LittleEndian(header.AsSpan(4));
                if (w <= 0 || h <= 0 || w > MaxFrameDim || h > MaxFrameDim)
                {
                    throw new InvalidDataException($"invalid frame size {w}x{h}");
                }

                var rgb = new byte[checked(w * h * 3)];
                stdin.ReadExactly(rgb);
                frameNo++;

                var frame = new ServeFrame
                {
                    Number = frameNo,
                    Width = w,
                    Height = h,
                    StartedTimestamp = started,
                };
                var swInput = Stopwatch.StartNew();
                try
                {
                    frame.ImageChw = RgbToChw(rgb, w, h);
                    frame.InputConvertMs = swInput.Elapsed.TotalMilliseconds;
                    stats.AddInput(frame.InputConvertMs);
                }
                catch (Exception ex)
                {
                    frame.Error = ex;
                }
                output.Add(frame, cancellationToken);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            fail(ex);
        }
        finally
        {
            output.CompleteAdding();
        }
    }

    private static void InferenceLoop(BlockingCollection<ServeFrame> input, BlockingCollection<ServeFrame> output,
        CancellationToken cancellationToken, Action<Exception> fail, TileRunner runner,
        int scale, int overlap, ServeStats stats)
    {
        try
        {
            foreach (ServeFrame frame in input.GetConsumingEnumerable(cancellationToken))
            {
                if (frame.Error is null)
                {
                    try
                    {
                        var timings = new TileTimings();
                        frame.MergedChw = UpscaleChw(runner, frame.ImageChw!, frame.Width, frame.Height,
                            runner.TileW, runner.TileH, scale, overlap, warmup: 0, timings, log: null);
                        frame.Timings = timings;
                        stats.AddTiming(timings);
                    }
                    catch (Exception ex)
                    {
                        frame.Error = ex;
                    }
                }
                output.Add(frame, cancellationToken);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            fail(ex);
        }
        finally
        {
            output.CompleteAdding();
        }
    }

    private static void SendLoop(BlockingCollection<ServeFrame> input, Stream stdout,
        CancellationToken cancellationToken, Action<Exception> fail, int scale, ServeStats stats)
    {
        try
        {
            foreach (ServeFrame frame in input.GetConsumingEnumerable(cancellationToken))
            {
                if (frame.Error is not null)
                {
                    WriteError(stdout, frame.Error);
                    stats.AddError();
                    LogFrameError(frame);
                    continue;
                }

                byte[] outRgb;
                try
                {
                    var swOutput = Stopwatch.StartNew();
                    outRgb = ChwToRgb(frame.MergedChw!, frame.Width * scale, frame.Height * scale);
                    frame.OutputConvertMs = swOutput.Elapsed.TotalMilliseconds;
                    stats.AddOutput(frame.OutputConvertMs);
                }
                catch (Exception ex)
                {
                    frame.Error = ex;
                    WriteError(stdout, ex);
                    stats.AddError();
                    LogFrameError(frame);
                    continue;
                }

                try
                {
                    var respHeader = new byte[12];
                    "UEUD"u8.CopyTo(respHeader.AsSpan());
                    BinaryPrimitives.WriteInt32LittleEndian(respHeader.AsSpan(4), frame.Width * scale);
                    BinaryPrimitives.WriteInt32LittleEndian(respHeader.AsSpan(8), frame.Height * scale);
                    stdout.Write(respHeader, 0, respHeader.Length);
                    stdout.Write(outRgb, 0, outRgb.Length);
                    stdout.Flush();
                }
                catch (Exception ex)
                {
                    fail(ex);
                    return;
                }

                double wallMs = (Stopwatch.GetTimestamp() - frame.StartedTimestamp) * 1000.0 / Stopwatch.Frequency;
                TileTimings timings = frame.Timings!;
                Console.Error.WriteLine(
                    $"frame {frame.Number}: {wallMs:F0} ms (pure-run median {TileTimings.Median(timings.PureRunMs):F0} ms, " +
                    $"tile-pre {timings.PreprocessMs.Sum():F0} ms, merge {timings.MergeMs.Sum():F0} ms, " +
                    $"in/out {frame.InputConvertMs + frame.OutputConvertMs:F0} ms, {timings.PureRunMs.Count} tiles)");
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            fail(ex);
        }
    }

    private static void WriteError(Stream stdout, Exception ex)
    {
        string message = $"{ex.GetType().Name}: {ex.Message}";
        byte[] msgBytes = System.Text.Encoding.UTF8.GetBytes(message);
        var errHeader = new byte[8];
        "UEUE"u8.CopyTo(errHeader.AsSpan());
        BinaryPrimitives.WriteInt32LittleEndian(errHeader.AsSpan(4), msgBytes.Length);
        stdout.Write(errHeader, 0, errHeader.Length);
        stdout.Write(msgBytes, 0, msgBytes.Length);
        stdout.Flush();
    }

    private static void LogFrameError(ServeFrame frame)
    {
        double wallMs = (Stopwatch.GetTimestamp() - frame.StartedTimestamp) * 1000.0 / Stopwatch.Frequency;
        Console.Error.WriteLine($"frame {frame.Number}: ERROR {wallMs:F0} ms ({frame.Error!.GetType().Name}: {frame.Error.Message})");
    }

    /// <summary>RGB24（行連続・padding無し）→ float CHW（/255, R/G/Bプレーン）。LoadImageChwのピクセル変換と同一。</summary>
    private static float[] RgbToChw(byte[] rgb, int w, int h)
    {
        int plane = w * h;
        var chw = new float[3 * plane];
        for (int i = 0, p = 0; i < plane; i++, p += 3)
        {
            chw[i] = rgb[p] / 255f;              // R
            chw[plane + i] = rgb[p + 1] / 255f;  // G
            chw[2 * plane + i] = rgb[p + 2] / 255f;  // B
        }
        return chw;
    }

    /// <summary>float CHW → RGB24（行連続・padding無し）。SaveImageChwのピクセル変換と同一（ClampByte使用）。</summary>
    private static byte[] ChwToRgb(float[] chw, int w, int h)
    {
        int plane = w * h;
        var rgb = new byte[plane * 3];
        for (int i = 0, p = 0; i < plane; i++, p += 3)
        {
            rgb[p] = ClampByte(chw[i]);              // R
            rgb[p + 1] = ClampByte(chw[plane + i]);  // G
            rgb[p + 2] = ClampByte(chw[2 * plane + i]);  // B
        }
        return rgb;
    }

    // ------------------------------------------------- tiling (npu_runner.py準拠)

    /// <summary>np.pad mode="reflect" と同じ折り返しインデックス。</summary>
    private static int Reflect(int i, int n)
    {
        if (n == 1) return 0;
        int period = 2 * n - 2;
        i = ((i % period) + period) % period;
        return i < n ? i : period - i;
    }

    private sealed class TilePlan
    {
        private readonly int _w;
        private readonly int _h;
        private readonly int _tileW;
        private readonly int _tileH;
        private readonly int[] _lutX;
        private readonly int[] _lutY;

        public int CoreW { get; }
        public int CoreH { get; }
        public int NTilesX { get; }
        public int NTilesY { get; }
        public int TileCount => NTilesX * NTilesY;

        private TilePlan(int w, int h, int tileW, int tileH, int overlap,
            int nTilesX, int nTilesY, int[] lutX, int[] lutY)
        {
            _w = w;
            _h = h;
            _tileW = tileW;
            _tileH = tileH;
            CoreW = tileW - 2 * overlap;
            CoreH = tileH - 2 * overlap;
            NTilesX = nTilesX;
            NTilesY = nTilesY;
            _lutX = lutX;
            _lutY = lutY;
        }

        public static TilePlan Create(int w, int h, int tileW, int tileH, int overlap)
        {
            int coreW = tileW - 2 * overlap;
            int coreH = tileH - 2 * overlap;
            if (coreW <= 0 || coreH <= 0) throw new ArgumentException("overlapが大きすぎます");

            int nTilesX = (w + coreW - 1) / coreW;
            int nTilesY = (h + coreH - 1) / coreH;
            int paddedW = nTilesX * coreW;
            int paddedH = nTilesY * coreH;

            // 2段のreflectパディング（右下へgrid合わせ→全周overlap）を合成したLUT
            int bigW = paddedW + 2 * overlap;
            int bigH = paddedH + 2 * overlap;
            int[] lutX = new int[bigW];
            int[] lutY = new int[bigH];
            for (int x = 0; x < bigW; x++) lutX[x] = Reflect(Reflect(x - overlap, paddedW), w);
            for (int y = 0; y < bigH; y++) lutY[y] = Reflect(Reflect(y - overlap, paddedH), h);

            return new TilePlan(w, h, tileW, tileH, overlap, nTilesX, nTilesY, lutX, lutY);
        }

        public void FillTile(float[] imgChw, int tileIndex, float[] tileBuffer)
        {
            if (imgChw.Length != 3 * _w * _h || tileBuffer.Length != 3 * _tileW * _tileH)
            {
                throw new ArgumentException("tile buffer shape mismatch");
            }
            if ((uint)tileIndex >= (uint)TileCount)
            {
                throw new ArgumentOutOfRangeException(nameof(tileIndex));
            }

            int iy = tileIndex / NTilesX;
            int ix = tileIndex % NTilesX;
            int y0 = iy * CoreH;
            int x0 = ix * CoreW;
            int planeSrc = _w * _h;
            int planeTile = _tileW * _tileH;
            for (int c = 0; c < 3; c++)
            {
                int srcBase = c * planeSrc;
                int dstBase = c * planeTile;
                for (int ty = 0; ty < _tileH; ty++)
                {
                    int sy = _lutY[y0 + ty] * _w + srcBase;
                    int dy = dstBase + ty * _tileW;
                    for (int tx = 0; tx < _tileW; tx++)
                    {
                        tileBuffer[dy + tx] = imgChw[sy + _lutX[x0 + tx]];
                    }
                }
            }
        }
    }

    private static void MergeTile(
        float[] tile, float[] outImg, int outW, int outH,
        int tileW, int tileH, int overlap, int nTilesX, int tileIndex)
    {
        int coreW = tileW - 2 * overlap;
        int coreH = tileH - 2 * overlap;
        int planeOut = outW * outH;
        int planeTile = tileW * tileH;
        int iy = tileIndex / nTilesX;
        int ix = tileIndex % nTilesX;
        int y0 = iy * coreH;
        int x0 = ix * coreW;
        int copyH = Math.Min(coreH, outH - y0);
        int copyW = Math.Min(coreW, outW - x0);
        if (copyH <= 0 || copyW <= 0) return;
        for (int c = 0; c < 3; c++)
        {
            int dstBase = c * planeOut;
            int srcBase = c * planeTile;
            for (int y = 0; y < copyH; y++)
            {
                int src = srcBase + (overlap + y) * tileW + overlap;
                int dst = dstBase + (y0 + y) * outW + x0;
                Array.Copy(tile, src, outImg, dst, copyW);
            }
        }
    }

    // ---------------------------------------------------------------- image IO

    private static (float[] chw, int w, int h) LoadImageChw(string path)
    {
        using var bmp = new Bitmap(path);
        int w = bmp.Width, h = bmp.Height;
        var rect = new Rectangle(0, 0, w, h);
        BitmapData data = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        try
        {
            int stride = Math.Abs(data.Stride);
            var raw = new byte[stride * h];
            Marshal.Copy(data.Scan0, raw, 0, raw.Length);

            var chw = new float[3 * w * h];
            int plane = w * h;
            for (int y = 0; y < h; y++)
            {
                int row = y * stride;
                int rowOut = y * w;
                for (int x = 0; x < w; x++)
                {
                    int p = row + x * 3;                       // BGR
                    chw[0 * plane + rowOut + x] = raw[p + 2] / 255f;  // R
                    chw[1 * plane + rowOut + x] = raw[p + 1] / 255f;  // G
                    chw[2 * plane + rowOut + x] = raw[p + 0] / 255f;  // B
                }
            }
            return (chw, w, h);
        }
        finally
        {
            bmp.UnlockBits(data);
        }
    }

    private static void SaveImageChw(float[] chw, int w, int h, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var bmp = new Bitmap(w, h, PixelFormat.Format24bppRgb);
        var rect = new Rectangle(0, 0, w, h);
        BitmapData data = bmp.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
        try
        {
            int stride = Math.Abs(data.Stride);
            var raw = new byte[stride * h];
            int plane = w * h;
            for (int y = 0; y < h; y++)
            {
                int row = y * stride;
                int rowIn = y * w;
                for (int x = 0; x < w; x++)
                {
                    int p = row + x * 3;
                    raw[p + 2] = ClampByte(chw[0 * plane + rowIn + x]);
                    raw[p + 1] = ClampByte(chw[1 * plane + rowIn + x]);
                    raw[p + 0] = ClampByte(chw[2 * plane + rowIn + x]);
                }
            }
            Marshal.Copy(raw, 0, data.Scan0, raw.Length);
        }
        finally
        {
            bmp.UnlockBits(data);
        }
        bmp.Save(path, ImageFormat.Png);
    }

    private static byte ClampByte(float v)
    {
        float scaled = v * 255f;
        return scaled <= 0f ? (byte)0 : scaled >= 255f ? (byte)255 : (byte)MathF.Round(scaled);
    }

    // --------------------------------------------------------------- CLI utils

    private static string Sanitize(string s)
    {
        foreach (char c in Path.GetInvalidFileNameChars()) s = s.Replace(c, '-');
        return s;
    }

    private static string Required(string[] args, string key)
        => Value(args, key) ?? throw new ArgumentException($"Missing required argument: {key}");

    private static string? Value(string[] args, string key)
    {
        for (int i = 1; i < args.Length - 1; i++)
        {
            if (args[i].Equals(key, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        }
        return null;
    }

    private static bool Has(string[] args, string key)
        => args.Any(a => a.Equals(key, StringComparison.OrdinalIgnoreCase));

    private static void PrintUsage()
    {
        Console.WriteLine("winml-sr list [--download]");
        Console.WriteLine("winml-sr run --model <onnx> --input <img> --output <img>");
        Console.WriteLine("             [--ep-policy npu|gpu|cpu|default|power|perf|efficiency]");
        Console.WriteLine("             [--ep-name <EpName> [--device-type NPU|GPU|CPU]]");
        Console.WriteLine("             [--overlap 16] [--compile] [--download] [--warmup 2]");
        Console.WriteLine("winml-sr serve --model <onnx> (--ep-name <EP> [--device-type T] | --ep-policy <p>)");
        Console.WriteLine("               [--overlap 16] [--warmup 2]   # stdin/stdoutバイナリプロトコル常駐モード");
        Console.WriteLine("winml-sr psnr --a <img> --b <img>");
    }
}
