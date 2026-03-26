using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>
/// Unified interface for all ML models (classification + regression).
/// Models follow a fit/predict pattern analogous to the ITransformer fit/transform pattern.
/// </summary>
public interface IModel
{
    /// <summary>Human-readable model name.</summary>
    string Name { get; }

    /// <summary>Whether the model has been fitted to training data.</summary>
    bool IsFitted { get; }

    /// <summary>Fit the model on feature matrix X and target vector y.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <param name="y">Target vector of shape (n_samples) or (n_samples, 1).</param>
    /// <returns>The fitted model (fluent API).</returns>
    IModel Fit(Tensor<double> X, Tensor<double> y);

    /// <summary>Predict target values for the given feature matrix.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>Predictions of shape (n_samples).</returns>
    Tensor<double> Predict(Tensor<double> X);

    /// <summary>Evaluate the model on the given data, returning a quality score (higher is better).</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <param name="y">True target values of shape (n_samples).</param>
    /// <returns>A score where higher values indicate better performance (e.g., R² for regression, accuracy for classification).</returns>
    double Score(Tensor<double> X, Tensor<double> y);
}

/// <summary>
/// Classification model that can predict class probabilities in addition to class labels.
/// </summary>
public interface IClassifier : IModel
{
    /// <summary>Predict class probabilities for each sample.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>Probability matrix of shape (n_samples, n_classes).</returns>
    Tensor<double> PredictProba(Tensor<double> X);

    /// <summary>Number of distinct classes learned during fitting.</summary>
    int NumClasses { get; }
}
