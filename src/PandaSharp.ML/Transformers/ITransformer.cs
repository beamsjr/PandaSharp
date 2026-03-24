using PandaSharp;

namespace PandaSharp.ML.Transformers;

/// <summary>
/// Fit/Transform pattern for feature engineering.
/// All transformers learn parameters from training data (Fit) and apply them consistently (Transform).
/// </summary>
public interface ITransformer
{
    string Name { get; }
    ITransformer Fit(DataFrame df);
    DataFrame Transform(DataFrame df);
    DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
}
