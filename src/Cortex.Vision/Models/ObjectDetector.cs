using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Cortex.Vision.Transforms;

namespace Cortex.Vision.Models;

/// <summary>
/// YOLO-based object detection using ONNX models.
/// Detects objects in images and returns bounding boxes with class labels and confidence scores.
/// Supports YOLOv8 ONNX models (input: [1,3,H,W], output: [1,84,N] for 80 COCO classes).
/// </summary>
public class ObjectDetector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int[] _inputShape;
    private readonly string[] _classNames;
    private readonly int _inputWidth;
    private readonly int _inputHeight;

    /// <summary>A single detected object with bounding box, class, and confidence.</summary>
    public record Detection(
        float X1, float Y1, float X2, float Y2,
        string ClassName, int ClassId, float Confidence)
    {
        /// <summary>Bounding box width.</summary>
        public float Width => X2 - X1;
        /// <summary>Bounding box height.</summary>
        public float Height => Y2 - Y1;
    }

    /// <summary>
    /// Create an object detector from a YOLO ONNX model.
    /// </summary>
    /// <param name="modelPath">Path to the .onnx model file (e.g. yolov8n.onnx).</param>
    /// <param name="classNames">Class label names (80 COCO classes by default).</param>
    /// <summary>
    /// Create an object detector from a YOLO ONNX model.
    /// </summary>
    /// <param name="modelPath">Path to the .onnx model file (e.g. yolov8n.onnx).</param>
    /// <param name="classNames">Class label names (80 COCO classes by default).</param>
    /// <param name="useGpu">Try to use CoreML (macOS) or CUDA (Linux/Windows) acceleration.</param>
    public ObjectDetector(string modelPath, string[]? classNames = null, bool useGpu = true)
    {
        var opts = new SessionOptions();
        if (useGpu)
        {
            try
            {
                // macOS: CoreML execution provider (uses Apple Neural Engine / GPU)
                if (OperatingSystem.IsMacOS())
                    opts.AppendExecutionProvider_CoreML();
                Console.WriteLine("[ObjectDetector] GPU acceleration enabled");
            }
            catch
            {
                Console.WriteLine("[ObjectDetector] GPU acceleration not available, using CPU");
            }
        }

        _session = new InferenceSession(modelPath, opts);
        var inputMeta = _session.InputMetadata.First();
        _inputName = inputMeta.Key;
        _inputShape = inputMeta.Value.Dimensions;

        // YOLOv8 input: [1, 3, H, W]
        _inputHeight = _inputShape.Length >= 3 ? Math.Abs(_inputShape[2]) : 640;
        _inputWidth = _inputShape.Length >= 4 ? Math.Abs(_inputShape[3]) : 640;
        if (_inputHeight <= 0 || _inputHeight > 4096) _inputHeight = 640;
        if (_inputWidth <= 0 || _inputWidth > 4096) _inputWidth = 640;

        _classNames = classNames ?? CocoClassNames;
    }

    /// <summary>Model input width.</summary>
    public int InputWidth => _inputWidth;

    /// <summary>Model input height.</summary>
    public int InputHeight => _inputHeight;

    /// <summary>
    /// Detect objects in an image.
    /// </summary>
    /// <param name="image">Image tensor (H, W, C) normalized to [0,1].</param>
    /// <param name="confidenceThreshold">Minimum confidence to keep a detection.</param>
    /// <param name="iouThreshold">IoU threshold for Non-Maximum Suppression.</param>
    /// <param name="originalWidth">Original image width (for scaling boxes back). If 0, uses input width.</param>
    /// <param name="originalHeight">Original image height (for scaling boxes back). If 0, uses input height.</param>
    public List<Detection> Detect(ImageTensor image, float confidenceThreshold = 0.5f,
        float iouThreshold = 0.45f, int originalWidth = 0, int originalHeight = 0)
    {
        // Preprocess: resize to model input size
        var resized = new Resize(_inputWidth, _inputHeight).Transform(image);
        var src = resized.Span;
        int h = _inputHeight, w = _inputWidth, c = 3;

        // Convert to NCHW float tensor
        var inputData = new float[1 * c * h * w];
        for (int ch = 0; ch < c; ch++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    inputData[ch * h * w + y * w + x] = src[(y * w + x) * c + ch];

        var inputTensor = new DenseTensor<float>(inputData, [1, c, h, w]);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) };

        using var results = _session.Run(inputs);
        var output = results.First();

        if (output.Value is not DenseTensor<float> outputTensor)
            return [];

        return ParseYoloOutput(outputTensor, confidenceThreshold, iouThreshold,
            originalWidth > 0 ? originalWidth : _inputWidth,
            originalHeight > 0 ? originalHeight : _inputHeight);
    }

    /// <summary>
    /// Detect objects from raw JPEG/PNG bytes (e.g. from a webcam frame).
    /// </summary>
    public List<Detection> DetectFromBytes(byte[] imageBytes, float confidenceThreshold = 0.5f,
        float iouThreshold = 0.45f)
    {
        using var ms = new MemoryStream(imageBytes);
        var image = ImageIO.Load(ms);
        int origW = image.Width;
        int origH = image.Height;
        return Detect(image, confidenceThreshold, iouThreshold, origW, origH);
    }

    /// <summary>
    /// Detect objects from a file path.
    /// </summary>
    public List<Detection> DetectFromFile(string imagePath, float confidenceThreshold = 0.5f,
        float iouThreshold = 0.45f)
    {
        var image = ImageIO.Load(imagePath);
        int origW = image.Width;
        int origH = image.Height;
        return Detect(image, confidenceThreshold, iouThreshold, origW, origH);
    }

    // ═══════════════════════════════════════════════════════════
    // YOLOv8 output parsing
    // ═══════════════════════════════════════════════════════════

    private List<Detection> ParseYoloOutput(DenseTensor<float> output,
        float confThresh, float iouThresh, int origW, int origH)
    {
        // YOLOv8 output shape: [1, 84, N] where 84 = 4 bbox + 80 classes, N = num candidates
        // Need to transpose to [N, 84]
        var dims = output.Dimensions.ToArray();
        int numClasses;
        int numDetections;
        bool transposed;

        if (dims.Length == 3 && dims[1] < dims[2])
        {
            // Shape [1, 84, 8400] — YOLOv8 format, needs transpose
            numClasses = dims[1] - 4;
            numDetections = dims[2];
            transposed = true;
        }
        else if (dims.Length == 3)
        {
            // Shape [1, 8400, 84] — already transposed
            numClasses = dims[2] - 4;
            numDetections = dims[1];
            transposed = false;
        }
        else if (dims.Length == 2)
        {
            numClasses = dims[1] - 4;
            numDetections = dims[0];
            transposed = false;
        }
        else
        {
            return [];
        }

        float scaleX = (float)origW / _inputWidth;
        float scaleY = (float)origH / _inputHeight;

        var candidates = new List<Detection>();

        for (int i = 0; i < numDetections; i++)
        {
            // Extract bbox (cx, cy, w, h) and class scores
            float cx, cy, bw, bh;
            if (transposed)
            {
                cx = output[0, 0, i];
                cy = output[0, 1, i];
                bw = output[0, 2, i];
                bh = output[0, 3, i];
            }
            else
            {
                cx = output.Length > numDetections ? output[0, i, 0] : output[i, 0];
                cy = output.Length > numDetections ? output[0, i, 1] : output[i, 1];
                bw = output.Length > numDetections ? output[0, i, 2] : output[i, 2];
                bh = output.Length > numDetections ? output[0, i, 3] : output[i, 3];
            }

            // Find best class
            float bestScore = 0;
            int bestClass = 0;
            for (int cls = 0; cls < numClasses && cls < _classNames.Length; cls++)
            {
                float score = transposed ? output[0, 4 + cls, i] : (output.Length > numDetections ? output[0, i, 4 + cls] : output[i, 4 + cls]);
                if (score > bestScore) { bestScore = score; bestClass = cls; }
            }

            if (bestScore < confThresh) continue;

            // Convert from center format to corner format, scale to original image
            float x1 = (cx - bw / 2) * scaleX;
            float y1 = (cy - bh / 2) * scaleY;
            float x2 = (cx + bw / 2) * scaleX;
            float y2 = (cy + bh / 2) * scaleY;

            // Clamp to image bounds
            x1 = Math.Max(0, x1);
            y1 = Math.Max(0, y1);
            x2 = Math.Min(origW, x2);
            y2 = Math.Min(origH, y2);

            candidates.Add(new Detection(x1, y1, x2, y2,
                bestClass < _classNames.Length ? _classNames[bestClass] : $"class_{bestClass}",
                bestClass, bestScore));
        }

        // Non-Maximum Suppression
        return Nms(candidates, iouThresh);
    }

    private static List<Detection> Nms(List<Detection> detections, float iouThreshold)
    {
        var sorted = detections.OrderByDescending(d => d.Confidence).ToList();
        var result = new List<Detection>();

        while (sorted.Count > 0)
        {
            var best = sorted[0];
            result.Add(best);
            sorted.RemoveAt(0);

            sorted.RemoveAll(d => d.ClassId == best.ClassId && Iou(best, d) > iouThreshold);
        }

        return result;
    }

    private static float Iou(Detection a, Detection b)
    {
        float x1 = Math.Max(a.X1, b.X1), y1 = Math.Max(a.Y1, b.Y1);
        float x2 = Math.Min(a.X2, b.X2), y2 = Math.Min(a.Y2, b.Y2);
        float inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        float areaA = a.Width * a.Height, areaB = b.Width * b.Height;
        return inter / (areaA + areaB - inter + 1e-6f);
    }

    public void Dispose() => _session.Dispose();

    // ═══════════════════════════════════════════════════════════
    // COCO 80 class names
    // ═══════════════════════════════════════════════════════════
    public static readonly string[] CocoClassNames =
    [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ];
}
