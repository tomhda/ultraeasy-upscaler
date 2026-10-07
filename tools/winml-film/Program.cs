// FILM ONNX を Windows ML / DirectML で実行し、連番 PNG を補間する。
// 入力: x0, x1 (1x3xHxW, RGB [0,1]), time (1x1, 0.5)
// 出力: image (1x3xHxW, RGB [0,1])
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.Windows.AI.MachineLearning;

internal static class Program
{
    private static async Task<int> Main(string[] args)
    {
        try
        {
            if (args.Length == 0 || args.Contains("--help"))
            {
                Console.WriteLine("winml-film --model FILE --input-dir DIR --output-dir DIR --factor 2|4|8 [--ep-name DmlExecutionProvider] [--device-index N]");
                return 0;
            }
            string model = Path.GetFullPath(Required(args, "--model"));
            string inputDir = Path.GetFullPath(Required(args, "--input-dir"));
            string outputDir = Path.GetFullPath(Required(args, "--output-dir"));
            int factor = int.Parse(Required(args, "--factor"));
            if (factor is not (2 or 4 or 8)) throw new ArgumentException("--factor は 2, 4, 8 のいずれかです。");
            string epName = Value(args, "--ep-name") ?? "DmlExecutionProvider";
            string? deviceIndex = Value(args, "--device-index");
            if (!Directory.Exists(inputDir)) throw new DirectoryNotFoundException(inputDir);
            if (!File.Exists(model)) throw new FileNotFoundException("FILM ONNXがありません", model);
            Directory.CreateDirectory(outputDir);
            var frames = Directory.EnumerateFiles(inputDir, "frame_*.png")
                .OrderBy(x => x, StringComparer.Ordinal).ToArray();
            if (frames.Length < 2) throw new InvalidDataException("入力フレームは2枚以上必要です。");

            var first = LoadImage(frames[0]);
            int width = first.Width, height = first.Height;
            int paddedWidth = ((width + 63) / 64) * 64;
            int paddedHeight = ((height + 63) / 64) * 64;
            Console.Error.WriteLine($"[image] {width}x{height}, padded={paddedWidth}x{paddedHeight}, frames={frames.Length}");

            // Windows ML のカタログを初期化してから明示的に GPU の DML EP を選ぶ。
            using OrtEnv env = CreateEnv();
            var provider = ExecutionProviderCatalog.GetDefault().FindAllProviders()
                .FirstOrDefault(p => p.Name.Equals(epName, StringComparison.OrdinalIgnoreCase));
            if (provider is not null)
            {
                await provider.EnsureReadyAsync();
            }
            var devices = env.GetEpDevices()
                .Where(d => d.EpName.Equals(epName, StringComparison.OrdinalIgnoreCase) &&
                            d.HardwareDevice.Type.ToString().Equals("GPU", StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (devices.Count == 0) throw new InvalidOperationException($"GPU の {epName} が見つかりません。");
            OrtEpDevice device;
            if (deviceIndex is not null)
            {
                int index = int.Parse(deviceIndex);
                if (index < 0 || index >= devices.Count) throw new ArgumentOutOfRangeException(nameof(deviceIndex));
                device = devices[index];
            }
            else
            {
                device = devices.FirstOrDefault(d => d.HardwareDevice.VendorId == 0x10DE) ?? devices[0];
            }
            Console.Error.WriteLine($"[ep] {epName}: {device.HardwareDevice.Vendor}, {device.HardwareDevice.Type}");
            using var options = new SessionOptions();
            options.AppendExecutionProvider(env, new List<OrtEpDevice> { device }, new Dictionary<string, string>());
            using var session = new InferenceSession(model, options);
            CheckModel(session);
            using var runner = new FilmRunner(session, paddedWidth, paddedHeight);

            var left = Pad(first.Pixels, width, height, paddedWidth, paddedHeight);
            int nextOutput = 1;
            for (int i = 0; i < frames.Length - 1; i++)
            {
                var rightImage = LoadImage(frames[i + 1]);
                if (rightImage.Width != width || rightImage.Height != height)
                    throw new InvalidDataException($"フレームの解像度が一致しません: {frames[i + 1]}");
                var right = Pad(rightImage.Pixels, width, height, paddedWidth, paddedHeight);
                CopyFrame(frames[i], OutputPath(outputDir, nextOutput++));
                foreach (var middle in Between(runner, left, right, factor))
                    SaveImage(middle, paddedWidth, paddedHeight, width, height, OutputPath(outputDir, nextOutput++));
                left = right;
                Console.WriteLine($"frame={nextOutput - 1}");
            }
            CopyFrame(frames[^1], OutputPath(outputDir, nextOutput++));
            for (int i = 1; i < factor; i++)
                CopyFrame(frames[^1], OutputPath(outputDir, nextOutput++));
            if (nextOutput - 1 != frames.Length * factor)
                throw new InvalidDataException("出力フレーム枚数が一致しません。");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"{ex.GetType().Name}: {ex.Message}");
            if (ex.InnerException is not null) Console.Error.WriteLine($"inner: {ex.InnerException.Message}");
            return 1;
        }
    }

    private static IEnumerable<float[]> Between(FilmRunner runner, float[] left, float[] right, int factor)
    {
        float[] middle = runner.Run(left, right);
        if (factor == 2)
        {
            yield return middle;
            yield break;
        }
        foreach (var frame in Between(runner, left, middle, factor / 2)) yield return frame;
        yield return middle;
        foreach (var frame in Between(runner, middle, right, factor / 2)) yield return frame;
    }

    private sealed class FilmRunner : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly RunOptions _options = new();
        private readonly int _width;
        private readonly int _height;

        public FilmRunner(InferenceSession session, int width, int height)
        {
            _session = session;
            _width = width;
            _height = height;
        }

        public float[] Run(float[] left, float[] right)
        {
            var output = new float[3 * _width * _height];
            using var x0 = OrtValue.CreateTensorValueFromMemory(left, new long[] { 1, 3, _height, _width });
            using var x1 = OrtValue.CreateTensorValueFromMemory(right, new long[] { 1, 3, _height, _width });
            using var time = OrtValue.CreateTensorValueFromMemory(new float[] { 0.5f }, new long[] { 1, 1 });
            using var result = OrtValue.CreateTensorValueFromMemory(output, new long[] { 1, 3, _height, _width });
            _session.Run(_options,
                new[] { "x0", "x1", "time" }, new[] { x0, x1, time },
                new[] { "image" }, new[] { result });
            return output;
        }

        public void Dispose() => _options.Dispose();
    }

    private static void CheckModel(InferenceSession session)
    {
        foreach (string name in new[] { "x0", "x1", "time" })
            if (!session.InputMetadata.ContainsKey(name))
                throw new InvalidDataException($"FILMモデルの入力 {name} がありません。");
        if (!session.OutputMetadata.ContainsKey("image"))
            throw new InvalidDataException("FILMモデルの出力 image がありません。");
    }

    private static float[] Pad(float[] src, int width, int height, int paddedWidth, int paddedHeight)
    {
        if (width == paddedWidth && height == paddedHeight) return src;
        var dst = new float[3 * paddedWidth * paddedHeight];
        int srcPlane = width * height, dstPlane = paddedWidth * paddedHeight;
        int left = (paddedWidth - width) / 2, top = (paddedHeight - height) / 2;
        for (int channel = 0; channel < 3; channel++)
            for (int y = 0; y < height; y++)
                Array.Copy(src, channel * srcPlane + y * width,
                    dst, channel * dstPlane + (y + top) * paddedWidth + left, width);
        return dst;
    }

    private static (float[] Pixels, int Width, int Height) LoadImage(string path)
    {
        using var bitmap = new Bitmap(path);
        int width = bitmap.Width, height = bitmap.Height;
        var rect = new Rectangle(0, 0, width, height);
        var data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
        try
        {
            int stride = Math.Abs(data.Stride);
            var bytes = new byte[stride * height];
            Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);
            var result = new float[3 * width * height];
            int plane = width * height;
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int p = y * stride + x * 3, at = y * width + x;
                    result[at] = bytes[p + 2] / 255f;
                    result[plane + at] = bytes[p + 1] / 255f;
                    result[2 * plane + at] = bytes[p] / 255f;
                }
            return (result, width, height);
        }
        finally { bitmap.UnlockBits(data); }
    }

    private static void SaveImage(float[] pixels, int paddedWidth, int paddedHeight,
        int width, int height, string path)
    {
        using var bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb);
        var data = bitmap.LockBits(new Rectangle(0, 0, width, height),
            ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
        try
        {
            int stride = Math.Abs(data.Stride), plane = paddedWidth * paddedHeight;
            int left = (paddedWidth - width) / 2, top = (paddedHeight - height) / 2;
            var bytes = new byte[stride * height];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int at = (y + top) * paddedWidth + x + left, p = y * stride + x * 3;
                    bytes[p + 2] = Clamp(pixels[at]);
                    bytes[p + 1] = Clamp(pixels[plane + at]);
                    bytes[p] = Clamp(pixels[2 * plane + at]);
                }
            Marshal.Copy(bytes, 0, data.Scan0, bytes.Length);
        }
        finally { bitmap.UnlockBits(data); }
        bitmap.Save(path, ImageFormat.Png);
    }

    private static byte Clamp(float value)
    {
        float scaled = value * 255f;
        return scaled <= 0 ? (byte)0 : scaled >= 255 ? (byte)255 : (byte)MathF.Round(scaled);
    }

    private static string OutputPath(string dir, int index) =>
        Path.Combine(dir, $"frame_{index:D8}.png");
    private static void CopyFrame(string source, string destination) => File.Copy(source, destination, true);
    private static string Required(string[] args, string key) =>
        Value(args, key) ?? throw new ArgumentException($"Missing {key}");
    private static string? Value(string[] args, string key)
    {
        for (int i = 0; i < args.Length - 1; i++)
            if (args[i].Equals(key, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
        return null;
    }
    private static OrtEnv CreateEnv()
    {
        EnvironmentCreationOptions envOptions = new()
        {
            logId = "winml-film",
            logLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
        };
        return OrtEnv.CreateInstanceWithOptions(ref envOptions);
    }
}
