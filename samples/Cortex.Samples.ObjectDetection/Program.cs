using System.Diagnostics;
using System.Net;
using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;
using Cortex.Vision;
using Cortex.Vision.Models;
using Cortex.Vision.Video;

/*
 * Cortex Object Detection — Live Webcam Demo
 * Uses HttpListener (raw HTTP) instead of Kestrel to get proper MJPEG streaming.
 */

// Find YOLO model
// Try largest model first, fall back to smaller
var modelPath = new[] {
    "models/yolov8x.onnx", "models/yolov8m.onnx", "models/yolov8n.onnx",
    "yolov8x.onnx", "yolov8m.onnx", "yolov8n.onnx",
    Path.Combine(AppContext.BaseDirectory, "models/yolov8x.onnx"),
}.FirstOrDefault(File.Exists);
ObjectDetector? detector = null;

if (modelPath is not null)
{
    Console.WriteLine($"Loading YOLO model from: {modelPath}");
    detector = new ObjectDetector(modelPath);
    Console.WriteLine($"Model loaded. Input: {detector.InputWidth}x{detector.InputHeight}");
}
else
{
    Console.WriteLine("WARNING: yolov8n.onnx not found.");
}

var cameras = WebcamCapture.ListDevices();
Console.WriteLine($"Cameras: {cameras.Length}");
foreach (var cam in cameras) Console.WriteLine($"  {cam}");

// Start raw HTTP server
var listener = new HttpListener();
listener.Prefixes.Add("http://localhost:5050/");
listener.Start();

Console.WriteLine("\n  Object Detection Demo");
Console.WriteLine("  Open http://localhost:5050");
Console.WriteLine("  Press Ctrl+C to stop\n");

var wwwroot = Path.Combine(AppContext.BaseDirectory, "wwwroot");
if (!Directory.Exists(wwwroot))
    wwwroot = Path.Combine(Directory.GetCurrentDirectory(), "wwwroot");

while (true)
{
    var ctx = listener.GetContext();
    var path = ctx.Request.Url?.AbsolutePath ?? "/";

    if (path == "/api/stream")
    {
        // Handle MJPEG stream on a thread so we can keep accepting other requests
        ThreadPool.QueueUserWorkItem(_ => HandleStream(ctx, detector));
    }
    else if (path == "/api/status")
    {
        var json = Encoding.UTF8.GetBytes(
            $"{{\"modelLoaded\":{(detector is not null ? "true" : "false")},\"cameras\":{cameras.Length}}}");
        ctx.Response.ContentType = "application/json";
        ctx.Response.ContentLength64 = json.Length;
        ctx.Response.OutputStream.Write(json);
        ctx.Response.Close();
    }
    else
    {
        // Serve static files
        var filePath = path == "/" ? "index.html" : path.TrimStart('/');
        var fullPath = Path.Combine(wwwroot, filePath);
        if (File.Exists(fullPath))
        {
            var bytes = File.ReadAllBytes(fullPath);
            ctx.Response.ContentType = filePath.EndsWith(".html") ? "text/html" :
                filePath.EndsWith(".js") ? "application/javascript" :
                filePath.EndsWith(".css") ? "text/css" : "application/octet-stream";
            ctx.Response.ContentLength64 = bytes.Length;
            ctx.Response.OutputStream.Write(bytes);
        }
        else
        {
            ctx.Response.StatusCode = 404;
        }
        ctx.Response.Close();
    }
}

static void HandleStream(HttpListenerContext ctx, ObjectDetector? detector)
{
    if (detector is null)
    {
        ctx.Response.StatusCode = 503;
        ctx.Response.Close();
        return;
    }

    ctx.Response.ContentType = "multipart/x-mixed-replace; boundary=frame";
    ctx.Response.SendChunked = false;
    ctx.Response.ContentLength64 = long.MaxValue;
    var output = ctx.Response.OutputStream;

    using var webcam = new WebcamCapture(deviceIndex: 0, width: 640, height: 480, fps: 15);
    int W = webcam.Width, H = webcam.Height;
    Console.WriteLine($"Capture resolution: {W}x{H}");

    // Warmup
    var warmup = webcam.ReadFrame();
    if (warmup is not null)
    {
        var sw = Stopwatch.StartNew();
        detector.Detect(warmup, 0.4f, originalWidth: W, originalHeight: H);
        Console.WriteLine($"ONNX warmup: {sw.ElapsedMilliseconds}ms");
    }

    Console.WriteLine("MJPEG stream started");
    int n = 0;

    try
    {
        foreach (var frame in webcam.Frames())
        {
            var sw = Stopwatch.StartNew();
            var dets = detector.Detect(frame, 0.4f, originalWidth: W, originalHeight: H);

            var jpeg = RenderFrame(frame, dets, W, H, n);

            // Save first 3 frames to disk for debugging
            if (n < 3)
            {
                var debugPath = $"debug_frame_{n}.jpg";
                File.WriteAllBytes(debugPath, jpeg);
                Console.WriteLine($"Saved {debugPath} ({jpeg.Length} bytes)");
            }

            var header = Encoding.ASCII.GetBytes(
                $"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {jpeg.Length}\r\n\r\n");

            output.Write(header);
            output.Write(jpeg);
            output.Write(Encoding.ASCII.GetBytes("\r\n"));
            output.Flush();

            n++;
            if (n % 30 == 0)
                Console.WriteLine($"Frame {n}: {dets.Count} objects, {sw.ElapsedMilliseconds}ms, {jpeg.Length / 1024}KB");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Stream ended: {ex.GetType().Name}");
    }

    Console.WriteLine($"MJPEG stream ended after {n} frames");
}

static byte[] RenderFrame(ImageTensor frame, List<ObjectDetector.Detection> detections, int w, int h, int frameNum)
{
    var image = new Image<Rgb24>(w, h);
    var span = frame.Span;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
        {
            int i = (y * w + x) * 3;
            image[x, y] = new Rgb24((byte)(span[i] * 255), (byte)(span[i + 1] * 255), (byte)(span[i + 2] * 255));
        }

    var font = SystemFonts.CreateFont("Arial", 14, FontStyle.Bold);
    var bigFont = SystemFonts.CreateFont("Arial", 28, FontStyle.Bold);

    // Stamp frame number on image so we can tell if frames are updating
    image.Mutate(ctx =>
    {
        var frameLabel = $"Frame #{frameNum}";
        ctx.Fill(SixLabors.ImageSharp.Color.Black, new RectangleF(w - 180, 10, 170, 36));
        ctx.DrawText(frameLabel, bigFont, SixLabors.ImageSharp.Color.Yellow, new PointF(w - 175, 12));
    });

    foreach (var det in detections)
    {
        var color = SixLabors.ImageSharp.Color.LimeGreen;
        var label = $"{det.ClassName} {det.Confidence:P0}";
        image.Mutate(ctx =>
        {
            ctx.Draw(color, 2, new RectangleF(det.X1, det.Y1, det.Width, det.Height));
            var ts = TextMeasurer.MeasureSize(label, new TextOptions(font));
            ctx.Fill(color, new RectangleF(det.X1, det.Y1 - ts.Height - 4, ts.Width + 8, ts.Height + 4));
            ctx.DrawText(label, font, SixLabors.ImageSharp.Color.White, new PointF(det.X1 + 4, det.Y1 - ts.Height - 2));
        });
    }

    using var ms = new MemoryStream();
    image.SaveAsJpeg(ms);
    return ms.ToArray();
}
