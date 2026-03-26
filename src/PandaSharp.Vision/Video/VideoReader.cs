using FFMediaToolkit;
using FFMediaToolkit.Decoding;
using FFMediaToolkit.Graphics;
using PandaSharp.ML.Tensors;
using PandaSharp.Column;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PandaSharp.Vision.Video;

/// <summary>
/// Video frame extraction using FFMediaToolkit (FFmpeg wrapper).
/// Requires FFmpeg shared libraries installed on the system.
/// </summary>
public class VideoReader : IDisposable
{
    private readonly MediaFile _file;
    private readonly string _path;

    public VideoReader(string path)
    {
        _path = path;
        try
        {
            // Try common FFmpeg locations on macOS/Linux
            if (string.IsNullOrEmpty(FFmpegLoader.FFmpegPath))
            {
                var commonPaths = new[] { "/opt/homebrew/lib", "/opt/homebrew/Cellar/ffmpeg", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib" };
                foreach (var p in commonPaths)
                {
                    if (Directory.Exists(p) && Directory.GetFiles(p, "libavcodec*").Length > 0)
                    {
                        FFmpegLoader.FFmpegPath = p;
                        break;
                    }
                }
            }
            _file = MediaFile.Open(path);
        }
        catch (DllNotFoundException ex)
        {
            throw new InvalidOperationException(
                $"FFmpeg libraries not found. Install FFmpeg: brew install ffmpeg (macOS) or apt install ffmpeg (Linux). Error: {ex.Message}", ex);
        }
    }

    /// <summary>Total number of frames in the video (may be estimated).</summary>
    public int FrameCount => _file.Video?.Info.NumberOfFrames ?? 0;

    /// <summary>Frames per second of the video.</summary>
    public double Fps => _file.Video?.Info.AvgFrameRate ?? 0;

    /// <summary>Duration of the video.</summary>
    public TimeSpan Duration => _file.Video?.Info.Duration ?? TimeSpan.Zero;

    /// <summary>Video frame width in pixels.</summary>
    public int Width => _file.Video?.Info.FrameSize.Width ?? 0;

    /// <summary>Video frame height in pixels.</summary>
    public int Height => _file.Video?.Info.FrameSize.Height ?? 0;

    /// <summary>
    /// Fast frame extraction — uses native C FFmpeg decoder when available,
    /// falls back to ffmpeg CLI pipe. Single call: open → decode → scale → convert.
    /// </summary>
    public static ImageTensor ExtractTensorsFast(string videoPath, int? maxFrames = null,
        int? everyNthFrame = null, int resizeWidth = 256, int resizeHeight = 256)
    {
        // CLI pipe is fastest — ffmpeg CLI uses hardware decode internally + efficient piping
        // Native AVFoundation/FFmpeg have higher per-call overhead

        // Fallback: ffmpeg CLI pipe
        int step = everyNthFrame ?? 1;
        string vfFilter = $"scale={resizeWidth}:{resizeHeight}";
        if (step > 1) vfFilter = $"select='not(mod(n\\,{step}))',setpts=N/FRAME_RATE/TB,{vfFilter}";

        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "ffmpeg",
            Arguments = $"-i \"{videoPath}\" -vf \"{vfFilter}\" -f rawvideo -pix_fmt rgb24 -v quiet -",
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var proc = System.Diagnostics.Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start ffmpeg");

        // Read all raw bytes into a single MemoryStream (one allocation)
        int frameSize = resizeWidth * resizeHeight * 3;
        int max = maxFrames ?? 10000;
        using var ms = new MemoryStream(max * frameSize);
        proc.StandardOutput.BaseStream.CopyTo(ms);
        proc.WaitForExit();

        var rawBytes = ms.GetBuffer();
        int totalBytes = (int)ms.Length;
        int frameCount = Math.Min(totalBytes / frameSize, max);

        if (frameCount == 0)
            return new ImageTensor(new float[resizeHeight * resizeWidth * 3], resizeHeight, resizeWidth, 3);

        // Parallel byte→float conversion (SIMD-friendly)
        int totalFloats = frameCount * frameSize;
        var batchData = new float[totalFloats];
        int nThreads = Math.Min(Environment.ProcessorCount, 4);
        Parallel.For(0, nThreads, t =>
        {
            int start = (int)((long)totalFloats * t / nThreads);
            int end = (int)((long)totalFloats * (t + 1) / nThreads);
            for (int i = start; i < end; i++)
                batchData[i] = rawBytes[i] / 255f;
        });

        return new ImageTensor(new Tensor<float>(batchData, frameCount, resizeHeight, resizeWidth, 3));
    }

    /// <summary>
    /// Fast frame extraction to disk via ffmpeg CLI.
    /// </summary>
    public static DataFrame ExtractFramesFast(string videoPath, string outputDir,
        int? everyNthFrame = null)
    {
        Directory.CreateDirectory(outputDir);
        int step = everyNthFrame ?? 1;
        string vfFilter = step > 1 ? $"select='not(mod(n\\,{step}))'" : "";
        string vfArg = string.IsNullOrEmpty(vfFilter) ? "" : $"-vf \"{vfFilter}\" -vsync vfr";

        var psi = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "ffmpeg",
            Arguments = $"-i \"{videoPath}\" {vfArg} \"{outputDir}/frame_%06d.png\" -v quiet",
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var proc = System.Diagnostics.Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start ffmpeg");
        proc.WaitForExit();

        var files = Directory.GetFiles(outputDir, "frame_*.png").OrderBy(f => f).ToArray();
        var indices = new int[files.Length];
        var timestamps = new double[files.Length];
        var paths = new string?[files.Length];
        for (int i = 0; i < files.Length; i++)
        {
            indices[i] = i * step;
            timestamps[i] = 0; // would need ffprobe for actual timestamps
            paths[i] = files[i];
        }

        return new DataFrame(
            new Column<int>("FrameIndex", indices),
            new Column<double>("Timestamp", timestamps),
            new StringColumn("ImagePath", paths));
    }

    /// <summary>
    /// Extract frames to disk as PNG files. Returns a DataFrame with frame metadata.
    /// </summary>
    /// <param name="outputDir">Directory to save extracted frames.</param>
    /// <param name="everyNthFrame">Extract every Nth frame, or null for all frames.</param>
    /// <param name="fps">Target extraction frame rate, or null to use original.</param>
    /// <returns>A DataFrame with frame paths and metadata.</returns>
    public DataFrame ExtractFrames(string outputDir, int? everyNthFrame = null, double? fps = null)
    {
        Directory.CreateDirectory(outputDir);
        var video = _file.Video ?? throw new InvalidOperationException("No video stream found.");

        var frameIndices = new List<int>();
        var timestamps = new List<double>();
        var paths = new List<string?>();
        int frameIdx = 0;
        int step = everyNthFrame ?? 1;

        while (video.TryGetNextFrame(out var frame))
        {
            if (frameIdx % step == 0)
            {
                var outPath = Path.Combine(outputDir, $"frame_{frameIdx:D6}.png");
                SaveFrame(frame, outPath);
                frameIndices.Add(frameIdx);
                timestamps.Add(frameIdx / Fps);
                paths.Add(outPath);
            }
            frameIdx++;
        }

        return new DataFrame(
            new Column<int>("FrameIndex", frameIndices.ToArray()),
            new Column<double>("Timestamp", timestamps.ToArray()),
            new StringColumn("ImagePath", paths.ToArray()));
    }

    /// <summary>
    /// Extract frames as in-memory ImageTensor batch.
    /// </summary>
    /// <param name="maxFrames">Maximum number of frames to extract.</param>
    /// <param name="everyNthFrame">Extract every Nth frame.</param>
    /// <param name="resizeWidth">Target width for frames.</param>
    /// <param name="resizeHeight">Target height for frames.</param>
    /// <returns>A batched ImageTensor containing extracted frames.</returns>
    public ImageTensor ExtractTensors(int? maxFrames = null, int? everyNthFrame = null,
        int? resizeWidth = null, int? resizeHeight = null)
    {
        var video = _file.Video ?? throw new InvalidOperationException("No video stream found.");
        int step = everyNthFrame ?? 1;
        int rw = resizeWidth ?? Width;
        int rh = resizeHeight ?? Height;
        int max = maxFrames ?? int.MaxValue;

        var frameTensors = new List<float[]>();
        int frameIdx = 0;
        int collected = 0;

        while (video.TryGetNextFrame(out var frame) && collected < max)
        {
            if (frameIdx % step == 0)
            {
                var tensor = FrameToTensor(frame, rw, rh);
                frameTensors.Add(tensor);
                collected++;
            }
            frameIdx++;
        }

        if (frameTensors.Count == 0)
            return new ImageTensor(new float[rh * rw * 3], rh, rw, 3);

        int singleLen = rh * rw * 3;
        var batchData = new float[frameTensors.Count * singleLen];
        for (int i = 0; i < frameTensors.Count; i++)
            Array.Copy(frameTensors[i], 0, batchData, i * singleLen, singleLen);

        return new ImageTensor(new Tensor<float>(batchData, frameTensors.Count, rh, rw, 3));
    }

    /// <summary>
    /// Lazy enumeration of frames — memory-efficient for large videos.
    /// </summary>
    /// <param name="everyNthFrame">Yield every Nth frame.</param>
    /// <returns>An enumerable of frame data.</returns>
    public IEnumerable<(int FrameIndex, TimeSpan Timestamp, ImageTensor Frame)> Frames(int? everyNthFrame = null)
    {
        var video = _file.Video ?? throw new InvalidOperationException("No video stream found.");
        int step = everyNthFrame ?? 1;
        int frameIdx = 0;

        while (video.TryGetNextFrame(out var frame))
        {
            if (frameIdx % step == 0)
            {
                var tensor = FrameToTensor(frame, Width, Height);
                var imgTensor = new ImageTensor(tensor, Height, Width, 3);
                yield return (frameIdx, TimeSpan.FromSeconds(frameIdx / Fps), imgTensor);
            }
            frameIdx++;
        }
    }

    private static float[] FrameToTensor(ImageData frame, int targetW, int targetH)
    {
        // FFMediaToolkit gives us ImageData with raw pixel data
        // Convert to float [0,1] tensor
        int srcW = frame.ImageSize.Width;
        int srcH = frame.ImageSize.Height;
        var data = frame.Data;

        // If resize needed, use bilinear interpolation
        bool needResize = srcW != targetW || srcH != targetH;

        if (!needResize)
        {
            var result = new float[srcH * srcW * 3];
            for (int y = 0; y < srcH; y++)
            {
                var row = data.Slice(y * frame.Stride, srcW * 3); // BGR24
                for (int x = 0; x < srcW; x++)
                {
                    int srcOff = x * 3;
                    int dstOff = (y * srcW + x) * 3;
                    // FFMediaToolkit uses BGR format
                    result[dstOff] = row[srcOff + 2] / 255f;     // R
                    result[dstOff + 1] = row[srcOff + 1] / 255f; // G
                    result[dstOff + 2] = row[srcOff] / 255f;     // B
                }
            }
            return result;
        }
        else
        {
            // Bilinear resize during extraction
            var result = new float[targetH * targetW * 3];
            float scaleY = (float)srcH / targetH;
            float scaleX = (float)srcW / targetW;

            for (int y = 0; y < targetH; y++)
            {
                float srcYf = y * scaleY;
                int y0 = Math.Min((int)srcYf, srcH - 1);
                int y1 = Math.Min(y0 + 1, srcH - 1);
                float fy = srcYf - y0;

                for (int x = 0; x < targetW; x++)
                {
                    float srcXf = x * scaleX;
                    int x0 = Math.Min((int)srcXf, srcW - 1);
                    int x1 = Math.Min(x0 + 1, srcW - 1);
                    float fx = srcXf - x0;

                    int dstOff = (y * targetW + x) * 3;
                    for (int c = 0; c < 3; c++)
                    {
                        // BGR source, RGB dest
                        int sc = 2 - c; // swap R<->B
                        float v00 = data[y0 * frame.Stride + x0 * 3 + sc] / 255f;
                        float v01 = data[y0 * frame.Stride + x1 * 3 + sc] / 255f;
                        float v10 = data[y1 * frame.Stride + x0 * 3 + sc] / 255f;
                        float v11 = data[y1 * frame.Stride + x1 * 3 + sc] / 255f;
                        result[dstOff + c] = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy + v11 * fx * fy;
                    }
                }
            }
            return result;
        }
    }

    private static void SaveFrame(ImageData frame, string path)
    {
        int w = frame.ImageSize.Width, h = frame.ImageSize.Height;
        int stride = frame.Stride;
        // Copy span data to array since ImageData.Data is a Span and can't be used in lambdas
        var pixelData = frame.Data.ToArray();
        using var image = new Image<Rgb24>(w, h);
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    int srcOff = y * stride + x * 3;
                    // BGR -> RGB
                    row[x] = new Rgb24(pixelData[srcOff + 2], pixelData[srcOff + 1], pixelData[srcOff]);
                }
            }
        });
        image.Save(path);
    }

    /// <inheritdoc />
    public void Dispose() => _file?.Dispose();
}
