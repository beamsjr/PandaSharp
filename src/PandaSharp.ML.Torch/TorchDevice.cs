using TorchSharp;
using static TorchSharp.torch;

namespace PandaSharp.ML.Torch;

/// <summary>
/// Provides automatic device detection and selection for TorchSharp operations.
/// Prefers CUDA > MPS > CPU based on availability.
/// </summary>
public static class TorchDevice
{
    private static readonly Lazy<DeviceDetection> _detection = new(Detect);

    /// <summary>
    /// Auto-detected best available device: CUDA if available, then MPS, then CPU.
    /// </summary>
    public static torch.Device Best => _detection.Value.BestDevice;

    /// <summary>Whether CUDA (NVIDIA GPU) is available.</summary>
    public static bool HasCuda => _detection.Value.CudaAvailable;

    /// <summary>Whether MPS (Apple Silicon GPU) is available.</summary>
    public static bool HasMps => _detection.Value.MpsAvailable;

    /// <summary>
    /// Returns a human-readable summary of device availability.
    /// </summary>
    public static string DeviceInfo()
    {
        var det = _detection.Value;
        return $"Best: {det.BestDevice} | CUDA: {det.CudaAvailable} | MPS: {det.MpsAvailable}";
    }

    /// <summary>
    /// Resolve a device string ("auto", "cuda", "mps", "cpu") to a torch.Device.
    /// </summary>
    public static torch.Device Resolve(string? device)
    {
        if (string.IsNullOrEmpty(device) || device == "auto")
            return Best;

        return device.ToLowerInvariant() switch
        {
            "cuda" => torch.CUDA,
            "mps" => new torch.Device("mps"),
            "cpu" => torch.CPU,
            _ => new torch.Device(device)
        };
    }

    private static DeviceDetection Detect()
    {
        bool cuda = false;
        bool mps = false;

        try { cuda = torch.cuda.is_available(); }
        catch { /* CUDA backend not loaded */ }

        try
        {
            // TorchSharp doesn't have a built-in mps_is_available in all versions.
            // Attempt to create a tiny tensor on MPS to probe availability.
            using var probe = torch.zeros(1, device: new torch.Device("mps"));
            mps = true;
        }
        catch { /* MPS not available */ }

        torch.Device best = cuda ? torch.CUDA
            : mps ? new torch.Device("mps")
            : torch.CPU;

        return new DeviceDetection(best, cuda, mps);
    }

    private sealed record DeviceDetection(torch.Device BestDevice, bool CudaAvailable, bool MpsAvailable);
}
