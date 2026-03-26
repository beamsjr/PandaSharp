using PandaSharp.Column;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PandaSharp.Vision;

/// <summary>
/// Image dataset analysis utilities for computing normalization statistics and validating images.
/// </summary>
public static class ImageStats
{
    /// <summary>
    /// Compute per-channel mean and standard deviation across a sample of images.
    /// Useful for deriving normalization parameters for a dataset.
    /// </summary>
    /// <param name="df">DataFrame containing image paths.</param>
    /// <param name="pathColumn">Name of the column containing file paths.</param>
    /// <param name="sampleSize">Maximum number of images to sample.</param>
    /// <param name="seed">Optional random seed for reproducible sampling.</param>
    /// <returns>A tuple of (Mean, Std) arrays, each with 3 elements for R, G, B channels.</returns>
    public static (float[] Mean, float[] Std) ComputeNormalization(
        DataFrame df, string pathColumn, int sampleSize = 1000, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(pathColumn);
        var col = df.GetStringColumn(pathColumn);
        var indices = Enumerable.Range(0, col.Length).ToArray();
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();

        // Fisher-Yates shuffle for sampling
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
        int n = Math.Min(sampleSize, indices.Length);

        double totalR = 0, totalG = 0, totalB = 0;
        double totalR2 = 0, totalG2 = 0, totalB2 = 0;
        long totalPix = 0;
        var lockObj = new object();

        Parallel.For(0, n,
            () => (sr: 0.0, sg: 0.0, sb: 0.0, sr2: 0.0, sg2: 0.0, sb2: 0.0, cnt: 0L),
            (i, _, local) =>
            {
                try
                {
                    var path = col[indices[i]];
                    if (path is null) return local;
                    using var img = Image.Load<Rgb24>(path);
                    img.ProcessPixelRows(acc =>
                    {
                        for (int y = 0; y < img.Height; y++)
                        {
                            var row = acc.GetRowSpan(y);
                            for (int x = 0; x < img.Width; x++)
                            {
                                float r = row[x].R / 255f;
                                float g = row[x].G / 255f;
                                float b = row[x].B / 255f;
                                local.sr += r;
                                local.sg += g;
                                local.sb += b;
                                local.sr2 += r * r;
                                local.sg2 += g * g;
                                local.sb2 += b * b;
                                local.cnt++;
                            }
                        }
                    });
                }
                catch (Exception)
                {
                    // Skip unreadable images
                }
                return local;
            },
            local =>
            {
                lock (lockObj)
                {
                    totalR += local.sr;
                    totalG += local.sg;
                    totalB += local.sb;
                    totalR2 += local.sr2;
                    totalG2 += local.sg2;
                    totalB2 += local.sb2;
                    totalPix += local.cnt;
                }
            });

        if (totalPix == 0)
            return (new float[] { 0, 0, 0 }, new float[] { 1, 1, 1 });

        float meanR = (float)(totalR / totalPix);
        float meanG = (float)(totalG / totalPix);
        float meanB = (float)(totalB / totalPix);
        float stdR = (float)Math.Sqrt(totalR2 / totalPix - meanR * (double)meanR);
        float stdG = (float)Math.Sqrt(totalG2 / totalPix - meanG * (double)meanG);
        float stdB = (float)Math.Sqrt(totalB2 / totalPix - meanB * (double)meanB);

        return (
            new[] { meanR, meanG, meanB },
            new[] { Math.Max(stdR, 1e-6f), Math.Max(stdG, 1e-6f), Math.Max(stdB, 1e-6f) }
        );
    }

    /// <summary>
    /// Validate images in a DataFrame, detecting corrupt, missing, or unreadable files.
    /// </summary>
    /// <param name="df">DataFrame containing image paths.</param>
    /// <param name="pathColumn">Name of the column containing file paths.</param>
    /// <returns>A DataFrame with columns: path, status, width, height.</returns>
    public static DataFrame ValidateImages(DataFrame df, string pathColumn)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(pathColumn);
        var col = df.GetStringColumn(pathColumn);
        var paths = new string?[col.Length];
        var statuses = new string?[col.Length];
        var widths = new int[col.Length];
        var heights = new int[col.Length];

        Parallel.For(0, col.Length, i =>
        {
            paths[i] = col[i];
            try
            {
                if (col[i] is null)
                {
                    statuses[i] = "null_path";
                    return;
                }
                if (!File.Exists(col[i]))
                {
                    statuses[i] = "missing";
                    return;
                }
                using var img = Image.Load<Rgb24>(col[i]!);
                widths[i] = img.Width;
                heights[i] = img.Height;
                statuses[i] = "ok";
            }
            catch (Exception ex)
            {
                statuses[i] = $"error: {ex.Message}";
            }
        });

        return new DataFrame(
            new StringColumn(pathColumn, paths),
            new StringColumn("status", statuses),
            new Column<int>("width", widths),
            new Column<int>("height", heights));
    }

    /// <summary>
    /// Produce a summary DataFrame with total, valid, and invalid image counts.
    /// </summary>
    /// <param name="df">DataFrame containing image paths.</param>
    /// <param name="pathColumn">Name of the column containing file paths.</param>
    /// <returns>A DataFrame with columns: metric, value.</returns>
    public static DataFrame DatasetSummary(DataFrame df, string pathColumn)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(pathColumn);
        var validation = ValidateImages(df, pathColumn);
        int okCount = 0;
        int errCount = 0;
        var statusCol = validation.GetStringColumn("status");
        for (int i = 0; i < statusCol.Length; i++)
        {
            if (statusCol[i] == "ok")
                okCount++;
            else
                errCount++;
        }

        return new DataFrame(
            new StringColumn("metric", new string?[] { "total_images", "valid", "invalid" }),
            new Column<int>("value", new[] { df.RowCount, okCount, errCount }));
    }
}
