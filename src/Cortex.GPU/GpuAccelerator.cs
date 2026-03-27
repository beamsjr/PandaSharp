using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace Cortex.GPU;

/// <summary>
/// Singleton GPU accelerator context. Automatically selects the best available device
/// (CUDA > OpenCL > CPU fallback). Lazily initialized on first use.
/// </summary>
public sealed class GpuAccelerator : IDisposable
{
    private static GpuAccelerator? _instance;
    private static readonly object _lock = new();

    private readonly Context _context;
    private readonly Accelerator _accelerator;
    private bool _disposed;

    /// <summary>The ILGPU accelerator instance.</summary>
    public Accelerator Device => _accelerator;

    /// <summary>The type of device in use (CPU, Cuda, OpenCL).</summary>
    public AcceleratorType DeviceType => _accelerator.AcceleratorType;

    /// <summary>Whether a real GPU (non-CPU) is available.</summary>
    public bool HasGpu => _accelerator.AcceleratorType != AcceleratorType.CPU;

    /// <summary>Device name string.</summary>
    public string DeviceName => _accelerator.Name;

    private GpuAccelerator(Context context, Accelerator accelerator)
    {
        _context = context;
        _accelerator = accelerator;
    }

    /// <summary>
    /// Gets or creates the shared GPU accelerator instance.
    /// Tries CUDA first, then OpenCL, then falls back to CPU.
    /// </summary>
    public static GpuAccelerator Instance
    {
        get
        {
            if (_instance is not null && !_instance._disposed)
                return _instance;

            lock (_lock)
            {
                if (_instance is not null && !_instance._disposed)
                    return _instance;

                _instance = Create();
                return _instance;
            }
        }
    }

    /// <summary>
    /// Creates a new accelerator targeting the best available device.
    /// </summary>
    private static GpuAccelerator Create()
    {
        var context = Context.Create(builder =>
        {
            builder.Default()
                   .EnableAlgorithms();
            // Try to enable CUDA
            try { builder.Cuda(); } catch { /* CUDA not available */ }
            // Try to enable OpenCL
            try { builder.OpenCL(); } catch { /* OpenCL not available */ }
        });

        Accelerator accelerator;

        // Prefer CUDA
        var cudaDevices = context.GetCudaDevices();
        if (cudaDevices.Count > 0)
        {
            accelerator = cudaDevices[0].CreateAccelerator(context);
            return new GpuAccelerator(context, accelerator);
        }

        // Then OpenCL
        var clDevices = context.GetCLDevices();
        if (clDevices.Count > 0)
        {
            // Prefer GPU-type OpenCL devices
            Device? gpuDevice = null;
            foreach (var dev in clDevices)
            {
                if (dev.Name.Contains("GPU", StringComparison.OrdinalIgnoreCase) ||
                    dev.Name.Contains("Graphics", StringComparison.OrdinalIgnoreCase))
                {
                    gpuDevice = dev;
                    break;
                }
            }
            gpuDevice ??= clDevices[0];
            accelerator = gpuDevice.CreateAccelerator(context);
            return new GpuAccelerator(context, accelerator);
        }

        // Fallback to CPU
        accelerator = context.CreateCPUAccelerator(0);
        return new GpuAccelerator(context, accelerator);
    }

    /// <summary>
    /// Force a specific accelerator type. Useful for testing or benchmarking.
    /// Disposes any existing instance.
    /// </summary>
    public static GpuAccelerator ForceDevice(AcceleratorType type)
    {
        lock (_lock)
        {
            _instance?.Dispose();

            var context = Context.Create(builder =>
            {
                builder.Default().EnableAlgorithms();
                if (type == AcceleratorType.Cuda) builder.Cuda();
                if (type == AcceleratorType.OpenCL) builder.OpenCL();
            });

            Accelerator acc = type switch
            {
                AcceleratorType.Cuda => context.GetCudaDevices()[0].CreateAccelerator(context),
                AcceleratorType.OpenCL => context.GetCLDevices()[0].CreateAccelerator(context),
                _ => context.CreateCPUAccelerator(0),
            };

            _instance = new GpuAccelerator(context, acc);
            return _instance;
        }
    }

    /// <summary>Dispose the accelerator and context.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _accelerator.Dispose();
        _context.Dispose();
    }
}
