using PandaSharp.SafeTensors;
using TorchSharp;
using static TorchSharp.torch;

namespace PandaSharp.ML.Torch;

/// <summary>
/// Loads SafeTensors weight files into TorchSharp modules.
/// Reads tensor data via <see cref="SafeTensorReader"/> and copies values
/// into matching module parameters by name.
/// </summary>
public static class SafeTensorsLoader
{
    /// <summary>
    /// Load weights from a SafeTensors file into a TorchSharp module.
    /// Matches tensor names from the file to module parameter names and copies data.
    /// Parameters not found in the file are left unchanged (partial loading is supported).
    /// </summary>
    /// <param name="path">Path to the .safetensors file.</param>
    /// <param name="module">The TorchSharp module whose parameters will be populated.</param>
    /// <param name="strict">
    /// If true, throws when the file contains tensor names not found in the module.
    /// If false (default), unmatched tensors are silently skipped.
    /// </param>
    /// <returns>List of parameter names that were successfully loaded.</returns>
    /// <exception cref="FileNotFoundException">The safetensors file does not exist.</exception>
    /// <exception cref="InvalidOperationException">In strict mode, when tensor names don't match module parameters.</exception>
    public static IReadOnlyList<string> LoadSafeTensors(string path, torch.nn.Module module, bool strict = false)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"SafeTensors file not found: {path}", path);

        using var reader = SafeTensorReader.Open(path);
        var tensorNames = reader.GetTensorNames();

        // Build lookup of module parameters by name
        var moduleParams = new Dictionary<string, torch.Tensor>();
        foreach (var (name, param) in module.named_parameters())
            moduleParams[name] = param;
        // Also include buffers (e.g., batch norm running_mean/running_var)
        foreach (var (name, buffer) in module.named_buffers())
            moduleParams[name] = buffer;

        var loaded = new List<string>();
        var unmatched = new List<string>();

        foreach (var name in tensorNames)
        {
            // Try exact match, then common name transformations
            string paramName = name;
            if (!moduleParams.ContainsKey(paramName))
            {
                // Try replacing '/' with '.' (common in HuggingFace naming)
                paramName = name.Replace('/', '.');
            }

            if (!moduleParams.TryGetValue(paramName, out var targetParam))
            {
                unmatched.Add(name);
                continue;
            }

            // Read tensor data as float and copy into the module parameter
            var safeTensor = reader.GetTensor<float>(name);
            var data = safeTensor.ToArray();
            var shape = safeTensor.Shape.Select(s => (long)s).ToArray();

            using var sourceTensor = torch.tensor(data, shape);
            using var converted = sourceTensor.to(targetParam.dtype).to(targetParam.device);

            // Verify shapes match
            if (!ShapesMatch(converted.shape, targetParam.shape))
            {
                if (strict)
                    throw new InvalidOperationException(
                        $"Shape mismatch for parameter '{paramName}': " +
                        $"file has [{string.Join(", ", converted.shape)}], " +
                        $"module expects [{string.Join(", ", targetParam.shape)}].");
                unmatched.Add(name);
                continue;
            }

            // Copy data into the existing parameter tensor
            using (torch.no_grad())
            {
                targetParam.copy_(converted);
            }

            loaded.Add(paramName);
        }

        if (strict && unmatched.Count > 0)
        {
            throw new InvalidOperationException(
                $"Strict loading failed. {unmatched.Count} tensor(s) in file have no matching " +
                $"module parameter: {string.Join(", ", unmatched.Take(10))}" +
                (unmatched.Count > 10 ? $" (and {unmatched.Count - 10} more)" : ""));
        }

        return loaded;
    }

    private static bool ShapesMatch(long[] a, long[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }
}
