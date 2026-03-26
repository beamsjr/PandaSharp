using System.Runtime.InteropServices;

namespace PandaSharp.Vision.Native;

/// <summary>
/// P/Invoke bindings to native video decoders.
/// Prefers Apple-native AVFoundation (VideoToolbox hardware decode) on macOS.
/// Falls back to FFmpeg-based libpandavision (threaded software decode).
/// </summary>
internal static class NativeVideoOps
{
    private const string FFmpegLib = "libpandavision";
    private const string AppleLib = "libpandavision_apple";

    private static readonly bool _ffmpegAvailable;
    private static readonly bool _appleAvailable;

    static NativeVideoOps()
    {
        _ffmpegAvailable = TryLoadLib(FFmpegLib, "libpandavision.dylib", "libpandavision.so");
        _appleAvailable = TryLoadLib(AppleLib, "libpandavision_apple.dylib");
    }

    private static bool TryLoadLib(string libName, params string[] fileNames)
    {
        try
        {
            if (NativeLibrary.TryLoad(libName, typeof(NativeVideoOps).Assembly, null, out _))
                return true;
            var asmDir = Path.GetDirectoryName(typeof(NativeVideoOps).Assembly.Location);
            if (asmDir is null) return false;
            foreach (var name in fileNames)
            {
                var path = Path.Combine(asmDir, "Native", name);
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out _))
                    return true;
            }
        }
        catch { }
        return false;
    }

    public static bool IsAvailable => _appleAvailable || _ffmpegAvailable;
    public static bool IsAppleAvailable => _appleAvailable;

    // Apple AVFoundation (VideoToolbox hardware decode) — persistent handle API
    [DllImport(AppleLib)] private static extern IntPtr apple_video_open(string path);
    [DllImport(AppleLib)] private static extern int apple_handle_extract(
        IntPtr handle, IntPtr output, int maxFrames, int everyNth, int targetW, int targetH);
    [DllImport(AppleLib)] private static extern void apple_video_close(IntPtr handle);
    // One-shot convenience
    [DllImport(AppleLib)] private static extern int apple_extract_frames(
        string path, IntPtr output, int maxFrames, int everyNth, int targetW, int targetH);

    // FFmpeg software decode
    [DllImport(FFmpegLib)] private static extern IntPtr video_open(string path);
    [DllImport(FFmpegLib)] private static extern int video_width(IntPtr handle);
    [DllImport(FFmpegLib)] private static extern int video_height(IntPtr handle);
    [DllImport(FFmpegLib)] private static extern int video_frame_count(IntPtr handle);
    [DllImport(FFmpegLib)] private static extern double video_fps(IntPtr handle);
    [DllImport(FFmpegLib)] private static extern double video_duration(IntPtr handle);
    [DllImport(FFmpegLib)] private static extern int video_extract_frames(
        IntPtr handle, IntPtr output, int maxFrames, int everyNth, int targetW, int targetH);
    [DllImport(FFmpegLib)] private static extern void video_close(IntPtr handle);

    /// <summary>
    /// Extract frames using the fastest available decoder.
    /// Apple-native (VideoToolbox hardware) > FFmpeg threaded software > CLI pipe.
    /// </summary>
    public static (ImageTensor Frames, int Width, int Height, int FrameCount, double Fps, double Duration)
        ExtractFrames(string path, int maxFrames, int everyNth, int targetW, int targetH)
    {
        int framePixels = targetW * targetH * 3;
        var buffer = new float[maxFrames * framePixels];
        int extracted;

        // Prefer Apple-native (VideoToolbox hardware decode)
        if (_appleAvailable)
        {
            var handle = apple_video_open(path);
            if (handle != IntPtr.Zero)
            {
                try
                {
                    unsafe
                    {
                        fixed (float* pBuf = buffer)
                            extracted = apple_handle_extract(handle, (IntPtr)pBuf,
                                maxFrames, everyNth, targetW, targetH);
                    }
                    if (extracted > 0)
                    {
                        var trimmed = extracted == maxFrames ? buffer : buffer[..(extracted * framePixels)];
                        var tensor = new ImageTensor(
                            new ML.Tensors.Tensor<float>(trimmed, extracted, targetH, targetW, 3));
                        return (tensor, targetW, targetH, extracted, 0, 0);
                    }
                }
                finally { apple_video_close(handle); }
            }
        }

        // Fallback: FFmpeg threaded software decode
        if (_ffmpegAvailable)
        {
            var handle = video_open(path);
            if (handle == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to open video: {path}");

            try
            {
                int w = video_width(handle);
                int h = video_height(handle);
                int fc = video_frame_count(handle);
                double fps = video_fps(handle);
                double dur = video_duration(handle);

                unsafe
                {
                    fixed (float* pBuf = buffer)
                        extracted = video_extract_frames(handle, (IntPtr)pBuf,
                            maxFrames, everyNth, targetW, targetH);
                }

                if (extracted == 0)
                {
                    var empty = new ImageTensor(new float[targetH * targetW * 3], targetH, targetW, 3);
                    return (empty, w, h, fc, fps, dur);
                }

                var trimmed = extracted == maxFrames ? buffer : buffer[..(extracted * framePixels)];
                var tensor = new ImageTensor(
                    new ML.Tensors.Tensor<float>(trimmed, extracted, targetH, targetW, 3));
                return (tensor, w, h, fc, fps, dur);
            }
            finally
            {
                video_close(handle);
            }
        }

        throw new InvalidOperationException("No native video decoder available");
    }
}
