using PandaSharp;
using PandaSharp.ML.Transformers;

namespace PandaSharp.ML.Pipeline;

/// <summary>
/// Composable feature engineering pipeline. Fit/Transform applied in sequence.
/// </summary>
public class FeaturePipeline : ITransformer
{
    private readonly ITransformer[] _steps;

    public string Name => "Pipeline";

    public FeaturePipeline(params ITransformer[] steps) => _steps = steps;

    /// <summary>
    /// Fit all steps sequentially. Note: calling Fit() again re-fits all steps
    /// with new data, replacing any previously learned parameters.
    /// </summary>
    public ITransformer Fit(DataFrame df)
    {
        var current = df;
        foreach (var step in _steps)
        {
            step.Fit(current);
            current = step.Transform(current);
        }
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        var current = df;
        foreach (var step in _steps)
            current = step.Transform(current);
        return current;
    }

    public DataFrame FitTransform(DataFrame df)
    {
        var current = df;
        foreach (var step in _steps)
            current = step.FitTransform(current);
        return current;
    }

    public IReadOnlyList<ITransformer> Steps => _steps;
}
