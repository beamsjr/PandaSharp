using System.Text.Json;

namespace PandaSharp.ML.Models;

/// <summary>
/// Serializes and deserializes ML models to/from JSON files.
/// Uses reflection to capture model type, parameters, and fitted state.
/// </summary>
public static class ModelSerializer
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    /// <summary>
    /// Save a model's type and public properties to a JSON file.
    /// </summary>
    /// <param name="model">The model to serialize.</param>
    /// <param name="path">File path to write to.</param>
    public static void Save(IModel model, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(path);
        var type = model.GetType();
        var properties = new Dictionary<string, object?>();

        foreach (var prop in type.GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance))
        {
            if (!prop.CanRead) continue;
            // Skip complex non-serializable properties
            if (prop.PropertyType.Namespace?.StartsWith("PandaSharp.ML.Tensors") == true) continue;

            try
            {
                properties[prop.Name] = prop.GetValue(model);
            }
            catch
            {
                // Skip properties that throw on access (e.g., unfitted state guards)
            }
        }

        var envelope = new Dictionary<string, object?>
        {
            ["version"] = 1,
            ["typeName"] = type.FullName,
            ["assemblyName"] = type.Assembly.GetName().Name,
            ["isFitted"] = model.IsFitted,
            ["properties"] = properties
        };

        var json = JsonSerializer.Serialize(envelope, JsonOpts);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// Load a model from a JSON file previously saved with <see cref="Save"/>.
    /// The model's assembly must be loaded in the current AppDomain.
    /// </summary>
    /// <param name="path">File path to read from.</param>
    /// <returns>The deserialized model instance.</returns>
    public static IModel Load(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        var json = File.ReadAllText(path);
        var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var typeName = root.GetProperty("typeName").GetString()
            ?? throw new InvalidOperationException("Missing typeName in model file.");
        var assemblyName = root.GetProperty("assemblyName").GetString();

        // Resolve the type from loaded assemblies
        Type? modelType = null;
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assemblyName is not null && assembly.GetName().Name != assemblyName) continue;
            modelType = assembly.GetType(typeName);
            if (modelType is not null) break;
        }

        modelType ??= Type.GetType(typeName);
        if (modelType is null)
            throw new InvalidOperationException($"Cannot resolve model type '{typeName}'. Ensure the assembly is loaded.");

        var instance = Activator.CreateInstance(modelType)
            ?? throw new InvalidOperationException($"Cannot create instance of '{typeName}'. Ensure it has a parameterless constructor.");

        // Restore public writable properties
        if (root.TryGetProperty("properties", out var propsElement))
        {
            foreach (var jsonProp in propsElement.EnumerateObject())
            {
                var propInfo = modelType.GetProperty(jsonProp.Name,
                    System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
                if (propInfo is null || !propInfo.CanWrite) continue;

                try
                {
                    var value = JsonSerializer.Deserialize(jsonProp.Value.GetRawText(), propInfo.PropertyType, JsonOpts);
                    propInfo.SetValue(instance, value);
                }
                catch
                {
                    // Skip properties that fail to deserialize
                }
            }
        }

        return (IModel)instance;
    }
}
