using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Models;
using PandaSharp.ML.Tensors;
using StringColumn = PandaSharp.Column.StringColumn;

namespace PandaSharp.ML.ModelSelection;

/// <summary>
/// Result of a single hyperparameter combination evaluated during grid search.
/// </summary>
/// <param name="Params">The parameter combination tested.</param>
/// <param name="MeanScore">Mean cross-validation score across all folds.</param>
/// <param name="StdScore">Standard deviation of cross-validation scores.</param>
/// <param name="FoldScores">Individual fold scores.</param>
public record HyperparameterResult(
    Dictionary<string, object> Params,
    double MeanScore,
    double StdScore,
    double[] FoldScores);

/// <summary>
/// Exhaustive grid search over specified parameter values with cross-validation.
/// Evaluates all combinations and selects the best based on mean CV score.
/// </summary>
public class GridSearchCV
{
    private readonly IModel _model;
    private readonly Dictionary<string, object[]> _paramGrid;
    private readonly int _nFolds;
    private readonly int? _seed;

    private HyperparameterResult? _bestResult;
    private List<HyperparameterResult>? _allResults;
    private DataFrame? _cachedResultsDf;

    /// <summary>Create a new grid search.</summary>
    /// <param name="model">Base model to evaluate.</param>
    /// <param name="paramGrid">Parameter grid: keys are parameter names, values are arrays of values to try.</param>
    /// <param name="nFolds">Number of cross-validation folds (default 5).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public GridSearchCV(IModel model, Dictionary<string, object[]> paramGrid, int nFolds = 5, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(paramGrid);
        _model = model;
        _paramGrid = paramGrid;
        _nFolds = nFolds;
        _seed = seed;
    }

    /// <summary>Best parameter combination found after fitting.</summary>
    public Dictionary<string, object> BestParams =>
        _bestResult?.Params ?? throw new InvalidOperationException("Grid search has not been fitted yet.");

    /// <summary>Best mean cross-validation score found after fitting.</summary>
    public double BestScore =>
        _bestResult?.MeanScore ?? throw new InvalidOperationException("Grid search has not been fitted yet.");

    /// <summary>All results as a DataFrame with columns for each parameter, mean_score, and std_score.</summary>
    public DataFrame AllResults
    {
        get
        {
            if (_allResults is null)
                throw new InvalidOperationException("Grid search has not been fitted yet.");

            if (_cachedResultsDf is not null)
                return _cachedResultsDf;

            var count = _allResults.Count;
            var columns = new List<IColumn>();

            // Parameter columns (as string since params can be any type)
            foreach (var paramName in _paramGrid.Keys)
            {
                var values = new string?[count];
                for (int i = 0; i < count; i++)
                    values[i] = _allResults[i].Params[paramName]?.ToString() ?? "";
                columns.Add(new StringColumn(paramName, values));
            }

            // Score columns — build arrays directly without LINQ
            var meanScores = new double[count];
            var stdScores = new double[count];
            for (int i = 0; i < count; i++)
            {
                meanScores[i] = _allResults[i].MeanScore;
                stdScores[i] = _allResults[i].StdScore;
            }
            columns.Add(new Column<double>("mean_score", meanScores));
            columns.Add(new Column<double>("std_score", stdScores));

            _cachedResultsDf = new DataFrame(columns);
            return _cachedResultsDf;
        }
    }

    /// <summary>
    /// Run the grid search: evaluate all parameter combinations using cross-validation.
    /// </summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <param name="y">Target vector of shape (n_samples).</param>
    /// <returns>This instance for fluent API usage.</returns>
    public GridSearchCV Fit(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        var combinations = GenerateCombinations(_paramGrid);
        _allResults = new List<HyperparameterResult>(combinations.Count);
        _bestResult = null;
        _cachedResultsDf = null;

        foreach (var combo in combinations)
        {
            // Apply parameters to the model via reflection
            ApplyParameters(_model, combo);

            var foldScores = CrossValidation.CrossValScore(_model, X, y, _nFolds, _seed);
            double mean = foldScores.Average();
            double std = StandardDeviation(foldScores, mean);

            var result = new HyperparameterResult(
                new Dictionary<string, object>(combo),
                mean,
                std,
                foldScores);

            _allResults.Add(result);

            if (_bestResult is null || mean > _bestResult.MeanScore)
                _bestResult = result;
        }

        // Apply best params to the model
        if (_bestResult is not null)
            ApplyParameters(_model, _bestResult.Params);

        return this;
    }

    /// <summary>Generate all combinations of parameter values from the grid.</summary>
    private static List<Dictionary<string, object>> GenerateCombinations(Dictionary<string, object[]> grid)
    {
        var keys = grid.Keys.ToList();
        var results = new List<Dictionary<string, object>>();
        GenerateCombinationsRecursive(grid, keys, 0, new Dictionary<string, object>(), results);
        return results;
    }

    private static void GenerateCombinationsRecursive(
        Dictionary<string, object[]> grid,
        List<string> keys,
        int depth,
        Dictionary<string, object> current,
        List<Dictionary<string, object>> results)
    {
        if (depth == keys.Count)
        {
            results.Add(new Dictionary<string, object>(current));
            return;
        }

        string key = keys[depth];
        foreach (var value in grid[key])
        {
            current[key] = value;
            GenerateCombinationsRecursive(grid, keys, depth + 1, current, results);
        }
        current.Remove(key);
    }

    /// <summary>Apply parameter values to a model via public property setters.</summary>
    private static void ApplyParameters(IModel model, Dictionary<string, object> parameters)
    {
        var type = model.GetType();
        foreach (var (name, value) in parameters)
        {
            var prop = type.GetProperty(name, System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            if (prop is null)
                throw new ArgumentException($"Parameter '{name}' does not match any public property on {type.Name}.");
            if (!prop.CanWrite)
                throw new ArgumentException($"Parameter '{name}' on {type.Name} is read-only and cannot be set.");
            var converted = Convert.ChangeType(value, prop.PropertyType);
            prop.SetValue(model, converted);
        }
    }

    private static double StandardDeviation(double[] values, double mean)
    {
        if (values.Length == 0) return 0;
        double sumSq = 0;
        foreach (var v in values)
        {
            double diff = v - mean;
            sumSq += diff * diff;
        }
        return Math.Sqrt(sumSq / values.Length);
    }
}
