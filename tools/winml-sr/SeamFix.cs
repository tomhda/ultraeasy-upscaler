// AdcSR の格子補正（固定テンプレート減算・平坦領域限定）。
//
// Python 参照実装 tmp/adcsr-seam/flat/methods.py::apply_template と
// tmp/adcsr-seam/flat/flatlib.py::flat_weights/local_std/gauss_blur と
// 同じ数式・同じ既定値にする。float 合成バッファ（MergeTile 完了後、
// uint8 化前）に適用すること。uint8 化後では 0.5 階調未満の補正が
// 丸めで消えるため効かない。
//
// 処理: (1)輝度 Y(Rec.601, 0-255 スケール)→33x33 局所標準偏差
// （box 平均の差分による厳密値、reflect 端処理）→hard マスク(<= tau)→
// σ=8 のガウス近似（box 半径 8 を x/y に 3 パス）でフェザー、
// 1e-3 未満は 0 切り捨て、(2)各画素に -(gx(dx)+gy(dy))*mask を
// RGB 全 ch に加算（dx,dy は最寄り継ぎ目からの距離を周期 P で折り畳んだ
// もの。継ぎ目は出力座標 k*P、k>=1 のみ。位相 ph=P）、
// (3)補正量を +-clip にクリップしてから加算（0-255 階調単位）。
// merged は CHW float 0-1 のため、加算時は /255 する。
internal static class SeamFix
{
    public sealed class SeamTemplate
    {
        public int Pitch { get; init; }
        public double[] Gx { get; init; } = Array.Empty<double>();
        public double[] Gy { get; init; } = Array.Empty<double>();
        public double Tau { get; init; } = 4.0;
        public double Clip { get; init; } = 2.0;
        public double FeatherSigma { get; init; } = 8.0;
        public int StdWindow { get; init; } = 33;
    }

    public static SeamTemplate Load(string path)
    {
        string json = File.ReadAllText(path);
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        var root = doc.RootElement;
        int pitch = root.GetProperty("pitch").GetInt32();
        double[] gx = root.GetProperty("gx").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        double[] gy = root.GetProperty("gy").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        if (gx.Length != pitch || gy.Length != pitch)
        {
            throw new InvalidDataException(
                $"seam template の長さが pitch と不一致: gx={gx.Length} gy={gy.Length} pitch={pitch} ({path})");
        }
        double tau = root.TryGetProperty("tau", out var tauEl) ? tauEl.GetDouble() : 4.0;
        double clip = root.TryGetProperty("clip", out var clipEl) ? clipEl.GetDouble() : 2.0;
        double feather = root.TryGetProperty("feather_sigma", out var featherEl) ? featherEl.GetDouble() : 8.0;
        int window = root.TryGetProperty("std_window", out var windowEl) ? windowEl.GetInt32() : 33;
        return new SeamTemplate
        {
            Pitch = pitch,
            Gx = gx,
            Gy = gy,
            Tau = tau,
            Clip = clip,
            FeatherSigma = feather,
            StdWindow = window,
        };
    }

    /// <summary>merged（CHW float 0-1）を in-place で補正し、処理時間を ms で返す。</summary>
    public static double ApplyInPlace(float[] merged, int outW, int outH, SeamTemplate template)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        int plane = outW * outH;
        int P = template.Pitch;
        int half = P / 2;

        // 輝度 Y(Rec.601)。Python 側は 0-255 float のため *255 して単位を合わせる。
        var y = new double[plane];
        for (int i = 0; i < plane; i++)
        {
            y[i] = (0.299 * merged[i] + 0.587 * merged[plane + i] + 0.114 * merged[2 * plane + i]) * 255.0;
        }

        // 33x33 局所標準偏差 → hard マスク → σ=8 フェザー（flat_weights と同一）。
        int r = template.StdWindow / 2;
        double[] std = LocalStd(y, outW, outH, r);
        var mask = new double[plane];
        for (int i = 0; i < plane; i++)
        {
            mask[i] = std[i] <= template.Tau ? 1.0 : 0.0;
        }
        mask = GaussBlur(mask, outW, outH, template.FeatherSigma);
        for (int i = 0; i < plane; i++)
        {
            if (mask[i] < 1e-3) mask[i] = 0.0;
        }

        // 距離 d 基準のテンプレート値を周期インデックス k 基準に並べ替え。
        // Python: tmpx[k] = gx[d + P//2]（d=k (k<P-P//2) または d=k-P）。
        var tmpx = new double[P];
        var tmpy = new double[P];
        for (int k = 0; k < P; k++)
        {
            int idx = k + half < P ? k + half : k + half - P;
            tmpx[k] = template.Gx[idx];
            tmpy[k] = template.Gy[idx];
        }

        // 位相 ph=P。kx=(x-ph) mod P（正規化）。ph=P のため x%P と等価だが、
        // Python と同じ式で書く。
        int ph = P;
        double clip = template.Clip;
        for (int row = 0; row < outH; row++)
        {
            int ky = (row - ph) % P;
            if (ky < 0) ky += P;
            double gy = tmpy[ky];
            int baseRow = row * outW;
            for (int col = 0; col < outW; col++)
            {
                double m = mask[baseRow + col];
                if (m == 0.0) continue;
                int kx = (col - ph) % P;
                if (kx < 0) kx += P;
                double corr = -(tmpx[kx] + gy) * m;
                if (corr > clip) corr = clip;
                else if (corr < -clip) corr = -clip;
                if (corr == 0.0) continue;
                float add = (float)(corr / 255.0);
                int pi = baseRow + col;
                merged[pi] += add;
                merged[plane + pi] += add;
                merged[2 * plane + pi] += add;
            }
        }

        return sw.Elapsed.TotalMilliseconds;
    }

    private static int Reflect(int i, int n)
    {
        if (n == 1) return 0;
        int period = 2 * n - 2;
        i = ((i % period) + period) % period;
        return i < n ? i : period - i;
    }

    /// <summary>一様窓（半径 r、reflect 端処理）の x 方向 1 パス。flatlib.box_blur_1pass と同一。</summary>
    private static double[] BoxBlurX(double[] src, int w, int h, int r)
    {
        if (r <= 0) return (double[])src.Clone();
        int win = 2 * r + 1;
        int pw = w + 2 * r;
        var lut = new int[pw];
        for (int j = 0; j < pw; j++) lut[j] = Reflect(j - r, w);
        var dst = new double[src.Length];
        var cs = new double[pw + 1];
        for (int row = 0; row < h; row++)
        {
            int rowBase = row * w;
            cs[0] = 0.0;
            for (int j = 0; j < pw; j++) cs[j + 1] = cs[j] + src[rowBase + lut[j]];
            for (int x = 0; x < w; x++) dst[rowBase + x] = (cs[x + win] - cs[x]) / win;
        }
        return dst;
    }

    /// <summary>一様窓（半径 r、reflect 端処理）の y 方向 1 パス。</summary>
    private static double[] BoxBlurY(double[] src, int w, int h, int r)
    {
        if (r <= 0) return (double[])src.Clone();
        int win = 2 * r + 1;
        int ph = h + 2 * r;
        var lut = new int[ph];
        for (int j = 0; j < ph; j++) lut[j] = Reflect(j - r, h);
        var dst = new double[src.Length];
        var cs = new double[ph + 1];
        for (int x = 0; x < w; x++)
        {
            cs[0] = 0.0;
            for (int j = 0; j < ph; j++) cs[j + 1] = cs[j] + src[lut[j] * w + x];
            for (int y = 0; y < h; y++) dst[y * w + x] = (cs[y + win] - cs[y]) / win;
        }
        return dst;
    }

    /// <summary>窓幅 (2r+1) の局所標準偏差。flatlib.local_std と同一（x→y の順）。</summary>
    private static double[] LocalStd(double[] y, int w, int h, int r)
    {
        double[] m = BoxBlurY(BoxBlurX(y, w, h, r), w, h, r);
        var y2 = new double[y.Length];
        for (int i = 0; i < y.Length; i++) y2[i] = y[i] * y[i];
        double[] m2 = BoxBlurY(BoxBlurX(y2, w, h, r), w, h, r);
        var std = new double[y.Length];
        for (int i = 0; i < std.Length; i++)
        {
            std[i] = Math.Sqrt(Math.Max(m2[i] - m[i] * m[i], 0.0));
        }
        return std;
    }

    /// <summary>ガウス近似ぼかし（box 半径 sigma を x/y に 3 パス）。flatlib.gauss_blur と同一。</summary>
    private static double[] GaussBlur(double[] src, int w, int h, double sigma)
    {
        int r = (int)Math.Round(sigma);
        double[] cur = src;
        for (int p = 0; p < 3; p++)
        {
            cur = BoxBlurY(BoxBlurX(cur, w, h, r), w, h, r);
        }
        return cur;
    }
}
