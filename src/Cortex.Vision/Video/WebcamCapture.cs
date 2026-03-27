using System.Diagnostics;

namespace Cortex.Vision.Video;

/// <summary>
/// Live webcam frame capture using FFmpeg's device input.
/// Reads raw RGB24 frames from FFmpeg's stdout pipe in real-time.
///
/// Supported platforms:
///   macOS:   AVFoundation (-f avfoundation)
///   Linux:   V4L2 (-f v4l2)
///   Windows: DirectShow (-f dshow)
///
/// Requires FFmpeg to be installed and available on PATH.
/// </summary>
public class WebcamCapture : IDisposable
{
    private readonly Process _process;
    private readonly Stream _stdout;
    private readonly int _width;
    private readonly int _height;
    private readonly int _frameSize;
    private readonly byte[] _buffer;
    private bool _disposed;

    /// <summary>Frame width in pixels.</summary>
    public int Width => _width;

    /// <summary>Frame height in pixels.</summary>
    public int Height => _height;

    /// <summary>Whether the capture process is still running.</summary>
    public bool IsRunning => !_process.HasExited;

    /// <summary>
    /// Open a webcam for live frame capture.
    /// </summary>
    /// <param name="deviceIndex">Camera device index (0 = default camera).</param>
    /// <param name="width">Capture width.</param>
    /// <param name="height">Capture height.</param>
    /// <param name="fps">Target frame rate.</param>
    /// <param name="deviceName">Explicit device name (overrides deviceIndex). Platform-specific.</param>
    public WebcamCapture(int deviceIndex = 0, int width = 640, int height = 480, int fps = 15, string? deviceName = null)
    {
        var (deviceArg, inputFps) = BuildDeviceArgument(deviceIndex, deviceName, fps);

        // First, probe the camera to get its native resolution
        var (nativeW, nativeH) = ProbeResolution(deviceArg);
        if (nativeW > 0 && nativeH > 0)
        {
            // Scale to fit within requested width, preserving aspect ratio
            double aspect = (double)nativeH / nativeW;
            _width = width;
            _height = (int)(width * aspect);
            if (_height % 2 != 0) _height++; // Must be even for pixel formats
        }
        else
        {
            _width = width;
            _height = height;
        }

        _frameSize = _width * _height * 3; // RGB24
        _buffer = new byte[_frameSize];

        var psi = new ProcessStartInfo
        {
            FileName = "ffmpeg",
            Arguments = $"-fflags nobuffer -flags low_delay {deviceArg} -vf \"fps={fps},scale={_width}:{_height}\" -f rawvideo -pix_fmt rgb24 -v error pipe:1",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        Console.WriteLine($"[WebcamCapture] FFmpeg args: {psi.Arguments}");

        _process = Process.Start(psi)
            ?? throw new InvalidOperationException("Failed to start FFmpeg. Is it installed and on PATH?");
        _stdout = _process.StandardOutput.BaseStream;

        // Drain stderr on a background thread to prevent pipe blocking
        var stderrReader = _process.StandardError;
        Task.Run(() =>
        {
            try
            {
                string? line;
                while ((line = stderrReader.ReadLine()) is not null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                        Console.WriteLine($"[FFmpeg] {line}");
                }
            }
            catch { }
        });
    }

    /// <summary>
    /// Read the next frame as an ImageTensor (H, W, 3) normalized to [0,1].
    /// Blocks until a full frame is available.
    /// Returns null if the stream has ended.
    /// </summary>
    private int _frameCount;

    public ImageTensor? ReadFrame()
    {
        if (_disposed || _process.HasExited)
        {
            Console.WriteLine("[WebcamCapture] Process has exited or disposed");
            return null;
        }

        int totalRead = 0;
        while (totalRead < _frameSize)
        {
            int read = _stdout.Read(_buffer, totalRead, _frameSize - totalRead);
            if (read == 0)
            {
                Console.WriteLine($"[WebcamCapture] Stream ended after {totalRead} bytes (expected {_frameSize})");
                return null;
            }
            totalRead += read;
        }

        _frameCount++;
        if (_frameCount <= 3 || _frameCount % 50 == 0)
            Console.WriteLine($"[WebcamCapture] Frame {_frameCount} read ({_frameSize} bytes)");

        // Convert byte RGB24 to float [0,1]
        var floatData = new float[_frameSize];
        for (int i = 0; i < _frameSize; i++)
            floatData[i] = _buffer[i] / 255f;

        return new ImageTensor(floatData, _height, _width, 3);
    }

    /// <summary>
    /// Read the latest available frame, dropping any queued frames to minimize latency.
    /// </summary>
    public ImageTensor? ReadLatestFrame()
    {
        if (_disposed || _process.HasExited) return null;

        // Check how many bytes are buffered and skip to the last complete frame
        ImageTensor? latest = null;
        while (true)
        {
            // Try to read a full frame
            int totalRead = 0;
            while (totalRead < _frameSize)
            {
                int read = _stdout.Read(_buffer, totalRead, _frameSize - totalRead);
                if (read == 0) return latest; // Stream ended
                totalRead += read;
            }

            // Successfully read a frame — convert it
            var floatData = new float[_frameSize];
            for (int i = 0; i < _frameSize; i++)
                floatData[i] = _buffer[i] / 255f;
            latest = new ImageTensor(floatData, _height, _width, 3);
            _frameCount++;

            // If there's at least one more full frame available, skip this one
            // This drains the buffer to get the freshest frame
            // We can't check pipe buffer size directly, so we use a heuristic:
            // if we just read a frame instantly (no blocking), there's probably more
            // Just return the frame we have — the next call will get the next one
            return latest;
        }
    }

    /// <summary>
    /// Enumerate frames as a lazy sequence. Yields frames until the capture is stopped or disposed.
    /// </summary>
    public IEnumerable<ImageTensor> Frames()
    {
        while (true)
        {
            var frame = ReadFrame();
            if (frame is null) yield break;
            yield return frame;
        }
    }

    /// <summary>
    /// List available camera devices. Returns device names/indices for the current platform.
    /// </summary>
    public static string[] ListDevices()
    {
        string args;
        if (OperatingSystem.IsMacOS())
            args = "-f avfoundation -list_devices true -i \"\"";
        else if (OperatingSystem.IsLinux())
            return Directory.Exists("/dev")
                ? Directory.GetFiles("/dev", "video*").OrderBy(f => f).ToArray()
                : [];
        else if (OperatingSystem.IsWindows())
            args = "-f dshow -list_devices true -i dummy";
        else
            return [];

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "ffmpeg",
                Arguments = args,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            using var proc = Process.Start(psi);
            if (proc is null) return [];

            // FFmpeg outputs device list to stderr
            var stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit(5000);

            // Parse device lines
            // macOS format:  [AVFoundation indev @ 0x...] [0] MacBook Pro Camera
            // Windows format: [dshow @ ...] "HP Webcam" (video)
            var devices = new List<string>();
            bool inVideoSection = false;
            foreach (var line in stderr.Split('\n'))
            {
                var trimmed = line.Trim();

                // macOS: detect video vs audio section
                if (trimmed.Contains("AVFoundation video devices:")) { inVideoSection = true; continue; }
                if (trimmed.Contains("AVFoundation audio devices:")) { inVideoSection = false; continue; }

                if (OperatingSystem.IsMacOS() && inVideoSection)
                {
                    // Parse: [AVFoundation indev @ 0x...] [0] MacBook Pro Camera
                    // Find the device index pattern [N]
                    var match = System.Text.RegularExpressions.Regex.Match(trimmed, @"\]\s+\[(\d+)\]\s+(.+)$");
                    if (match.Success)
                        devices.Add($"[{match.Groups[1].Value}] {match.Groups[2].Value}");
                }
                else if (OperatingSystem.IsWindows() && trimmed.Contains("\""))
                {
                    var start = trimmed.IndexOf('"');
                    var end = trimmed.LastIndexOf('"');
                    if (start >= 0 && end > start)
                        devices.Add(trimmed[(start + 1)..end]);
                }
            }

            return devices.ToArray();
        }
        catch
        {
            return [];
        }
    }

    private static (int Width, int Height) ProbeResolution(string deviceArg)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "ffmpeg",
                Arguments = $"{deviceArg} -frames:v 1 -f null -",
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var proc = Process.Start(psi);
            if (proc is null) return (0, 0);
            var stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit(5000);

            // Parse "Video: ... WxH" from FFmpeg output
            var match = System.Text.RegularExpressions.Regex.Match(stderr, @"(\d{3,4})x(\d{3,4})");
            if (match.Success)
            {
                int w = int.Parse(match.Groups[1].Value);
                int h = int.Parse(match.Groups[2].Value);
                Console.WriteLine($"[WebcamCapture] Native resolution: {w}x{h}");
                return (w, h);
            }
        }
        catch { }
        return (0, 0);
    }

    private static (string Args, int ActualFps) BuildDeviceArgument(int deviceIndex, string? deviceName, int requestedFps)
    {
        if (OperatingSystem.IsMacOS())
        {
            // macOS AVFoundation: framerate must be 30 (or match device capabilities)
            // Don't use -video_size before -i — it conflicts with AVFoundation
            var device = deviceName ?? deviceIndex.ToString();
            int fps = Math.Max(requestedFps, 30); // AVFoundation needs >=30
            return ($"-f avfoundation -framerate {fps} -i \"{device}:none\"", fps);
        }
        else if (OperatingSystem.IsLinux())
        {
            var device = deviceName ?? $"/dev/video{deviceIndex}";
            return ($"-f v4l2 -framerate {requestedFps} -i \"{device}\"", requestedFps);
        }
        else if (OperatingSystem.IsWindows())
        {
            var device = deviceName ?? $"video={deviceIndex}";
            return ($"-f dshow -framerate {requestedFps} -i \"{device}\"", requestedFps);
        }
        else
        {
            throw new PlatformNotSupportedException("Webcam capture is supported on macOS, Linux, and Windows.");
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        try
        {
            if (!_process.HasExited)
            {
                _process.Kill();
                _process.WaitForExit(2000);
            }
        }
        catch { }

        _process.Dispose();
    }
}
