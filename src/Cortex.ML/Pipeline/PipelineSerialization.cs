using System.Text.Json;
using Cortex.ML.Transformers;

namespace Cortex.ML.Pipeline;

/// <summary>
/// Serialize/deserialize feature pipelines for production deployment.
/// Includes learned parameters for all fitted transformers.
/// </summary>
public static class PipelineSerialization
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };

    public static byte[] Serialize(this FeaturePipeline pipeline)
    {
        var steps = new List<Dictionary<string, object?>>();
        foreach (var step in pipeline.Steps)
        {
            var entry = new Dictionary<string, object?>
            {
                ["type"] = step.GetType().Name,
                ["name"] = step.Name,
                ["params"] = ExtractParams(step)
            };
            steps.Add(entry);
        }

        var envelope = new Dictionary<string, object?>
        {
            ["version"] = 2,
            ["steps"] = steps
        };

        return JsonSerializer.SerializeToUtf8Bytes(envelope, JsonOpts);
    }

    public static string SerializeToJson(this FeaturePipeline pipeline)
        => System.Text.Encoding.UTF8.GetString(pipeline.Serialize());

    /// <summary>
    /// Extract learned parameters from a fitted transformer.
    /// </summary>
    private static object? ExtractParams(ITransformer step)
    {
        // Use reflection to extract private fitted fields
        var type = step.GetType();
        var result = new Dictionary<string, object?>();

        // StandardScaler: _params Dictionary<string, (double Mean, double Std)>
        if (step is StandardScaler)
        {
            var field = type.GetField("_params", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (field?.GetValue(step) is Dictionary<string, (double Mean, double Std)> p)
            {
                var dict = new Dictionary<string, object>();
                foreach (var (k, v) in p) dict[k] = new { v.Mean, v.Std };
                return dict;
            }
        }

        // MinMaxScaler: _params Dictionary<string, (double Min, double Max)>
        if (step is MinMaxScaler)
        {
            var field = type.GetField("_params", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (field?.GetValue(step) is Dictionary<string, (double Min, double Max)> p)
            {
                var dict = new Dictionary<string, object>();
                foreach (var (k, v) in p) dict[k] = new { v.Min, v.Max };
                return dict;
            }
        }

        // RobustScaler: _params Dictionary<string, (double Median, double IQR)>
        if (step is RobustScaler)
        {
            var field = type.GetField("_params", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (field?.GetValue(step) is Dictionary<string, (double Median, double IQR)> p)
            {
                var dict = new Dictionary<string, object>();
                foreach (var (k, v) in p) dict[k] = new { v.Median, v.IQR };
                return dict;
            }
        }

        // LabelEncoder: _mappings Dictionary<string, Dictionary<string, int>>
        if (step is LabelEncoder le)
        {
            var field = type.GetField("_mappings", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return field?.GetValue(step);
        }

        // OneHotEncoder: _vocabularies Dictionary<string, string[]>
        if (step is OneHotEncoder)
        {
            var field = type.GetField("_vocabularies", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return field?.GetValue(step);
        }

        // Imputer: _fillValues Dictionary<string, double>
        if (step is Imputer)
        {
            var field = type.GetField("_fillValues", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return field?.GetValue(step);
        }

        // TextVectorizer: _vocabulary, _idfWeights
        if (step is TextVectorizer)
        {
            var vocabField = type.GetField("_vocabulary", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var idfField = type.GetField("_idfWeights", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return new { vocabulary = vocabField?.GetValue(step), idfWeights = idfField?.GetValue(step) };
        }

        // Discretizer: _binEdges Dictionary<string, double[]>
        if (step is Discretizer)
        {
            var field = type.GetField("_binEdges", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return field?.GetValue(step);
        }

        // TargetEncoder: _encodings Dictionary<string, Dictionary<string, double>> + _globalMean
        if (step is TargetEncoder)
        {
            var encodingsField = type.GetField("_encodings", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var globalMeanField = type.GetField("_globalMean", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return new { encodings = encodingsField?.GetValue(step), globalMean = globalMeanField?.GetValue(step) };
        }

        // PolynomialFeatures: stateless transformer — only stores input column names
        // (no learned parameters). _inputColumns is set during Fit but only determines
        // which columns to expand, not learned statistics.
        if (step is PolynomialFeatures)
        {
            var field = type.GetField("_inputColumns", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            return new { inputColumns = field?.GetValue(step), note = "stateless: no learned parameters" };
        }

        return null; // unfitted or no params to serialize
    }

    /// <summary>
    /// Deserialize a pipeline from JSON bytes.
    /// Not yet implemented — pipeline deserialization requires a type registry and
    /// constructor-matching strategy for each transformer type. Contributions welcome.
    /// </summary>
    /// <param name="data">JSON bytes produced by <see cref="Serialize"/>.</param>
    /// <returns>A reconstructed FeaturePipeline.</returns>
    /// <exception cref="NotImplementedException">Always thrown. Deserialization is not yet implemented.</exception>
    public static FeaturePipeline Deserialize(byte[] data)
    {
        throw new NotImplementedException(
            "Pipeline deserialization is not yet implemented. " +
            "Serialized pipelines can be inspected as JSON but cannot be reconstructed automatically. " +
            "To restore a pipeline, reconstruct the steps manually and re-fit on training data.");
    }
}
