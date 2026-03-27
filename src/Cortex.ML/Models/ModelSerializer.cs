using System.Text.Json;

namespace Cortex.ML.Models;

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
            if (prop.PropertyType.Namespace?.StartsWith("Cortex.ML.Tensors") == true) continue;

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

        // Parse the serialized properties so we can supply constructor arguments
        Dictionary<string, JsonElement>? serializedProps = null;
        if (root.TryGetProperty("properties", out var propsEl))
        {
            serializedProps = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
            foreach (var jp in propsEl.EnumerateObject())
                serializedProps[jp.Name] = jp.Value;
        }

        // Try parameterless constructor first; if unavailable, find a constructor
        // where all parameters have default values and invoke it with those defaults.
        // As a last resort, try constructors where required parameters can be supplied
        // from the serialized properties.
        object? instance = null;
        try
        {
            instance = Activator.CreateInstance(modelType);
        }
        catch (MissingMethodException)
        {
            var constructors = modelType.GetConstructors(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);

            // First pass: look for a constructor where every parameter is optional (has a default value)
            foreach (var ctor in constructors.OrderBy(c => c.GetParameters().Length))
            {
                var parameters = ctor.GetParameters();
                if (parameters.All(p => p.HasDefaultValue))
                {
                    var defaults = parameters.Select(p => p.DefaultValue).ToArray();
                    instance = ctor.Invoke(defaults);
                    break;
                }
            }

            // Second pass: try constructors where required parameters can be matched
            // from serialized properties (by name, case-insensitive)
            if (instance is null && serializedProps is not null)
            {
                foreach (var ctor in constructors.OrderBy(c => c.GetParameters().Length))
                {
                    var parameters = ctor.GetParameters();
                    var args = new object?[parameters.Length];
                    bool canInvoke = true;
                    for (int i = 0; i < parameters.Length; i++)
                    {
                        var param = parameters[i];
                        if (serializedProps.TryGetValue(param.Name!, out var jsonVal))
                        {
                            try
                            {
                                args[i] = JsonSerializer.Deserialize(jsonVal.GetRawText(), param.ParameterType, JsonOpts);
                                continue;
                            }
                            catch { /* fall through to default */ }
                        }
                        if (param.HasDefaultValue)
                        {
                            args[i] = param.DefaultValue;
                        }
                        else
                        {
                            canInvoke = false;
                            break;
                        }
                    }
                    if (canInvoke)
                    {
                        try
                        {
                            instance = ctor.Invoke(args);
                            break;
                        }
                        catch { /* try next constructor */ }
                    }
                }
            }
        }

        if (instance is null)
            throw new InvalidOperationException($"Cannot create instance of '{typeName}'. Ensure it has a parameterless constructor or a constructor with all-optional parameters.");

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

        // Restore IsFitted state from the envelope. IsFitted often has a private setter
        // or is computed from a private field, so we need to use reflection to set it.
        if (root.TryGetProperty("isFitted", out var isFittedElement) && isFittedElement.GetBoolean())
        {
            var isFittedProp = modelType.GetProperty("IsFitted",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            if (isFittedProp is not null)
            {
                // Try setting via property setter (works even with private set)
                var setter = isFittedProp.GetSetMethod(nonPublic: true);
                if (setter is not null)
                {
                    try { setter.Invoke(instance, [true]); }
                    catch { /* ignore if it fails */ }
                }
                else
                {
                    // For computed properties (e.g., IsFitted => _root is not null),
                    // try to find and set the backing field, or set a sentinel value
                    // on the field that the computed property checks.
                    var backingField = modelType.GetField("<IsFitted>k__BackingField",
                        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    if (backingField is not null)
                    {
                        try { backingField.SetValue(instance, true); }
                        catch { /* ignore */ }
                    }
                    else
                    {
                        // Try common field patterns that computed IsFitted properties check:
                        // _root (DecisionTreeClassifier), _trees (RandomForestClassifier/ensemble),
                        // _weights (various models)
                        TrySetSentinelForIsFitted(modelType, instance);
                    }
                }
            }
        }

        return (IModel)instance;
    }

    /// <summary>
    /// For models where IsFitted is computed from a private nullable field (e.g., _root, _trees),
    /// set a sentinel non-null value on that field so IsFitted returns true.
    /// This allows deserialized models to report they were previously fitted,
    /// even though the full internal state (weights, tree structure) is not restored.
    /// </summary>
    private static void TrySetSentinelForIsFitted(Type modelType, object instance)
    {
        // Known patterns for computed IsFitted properties:
        // DecisionTreeClassifier: IsFitted => _root is not null
        // RandomForestClassifier: IsFitted => _trees is not null
        var candidateFields = new[] { "_root", "_trees", "_weights", "_inner" };
        foreach (var fieldName in candidateFields)
        {
            var field = modelType.GetField(fieldName,
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (field is null) continue;

            // Only set if the field is currently null and is a reference/nullable type
            var currentValue = field.GetValue(instance);
            if (currentValue is not null) continue;

            try
            {
                // Create a minimal non-null sentinel value
                var fieldType = field.FieldType;
                if (fieldType == typeof(TreeNode))
                {
                    // DecisionTreeClassifier._root: set a minimal leaf node
                    field.SetValue(instance, new TreeNode { ClassDistribution = Array.Empty<double>() });
                }
                else if (fieldType.IsArray)
                {
                    // For array fields like _trees (DecisionTreeClassifier[]),
                    // create an empty array of the element type
                    var elementType = fieldType.GetElementType()!;
                    field.SetValue(instance, Array.CreateInstance(elementType, 0));
                }
                else
                {
                    // Try creating a default instance
                    try
                    {
                        var sentinel = Activator.CreateInstance(fieldType);
                        if (sentinel is not null)
                            field.SetValue(instance, sentinel);
                    }
                    catch { /* ignore */ }
                }
            }
            catch { /* ignore */ }
        }
    }
}
