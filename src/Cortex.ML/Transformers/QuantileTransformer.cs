using Cortex;
using Cortex.Column;

namespace Cortex.ML.Transformers;

/// <summary>Target output distribution for quantile mapping.</summary>
public enum QuantileOutputDistribution
{
    /// <summary>Map to uniform [0, 1].</summary>
    Uniform,
    /// <summary>Map to standard normal via inverse CDF (probit).</summary>
    Normal
}

/// <summary>
/// Transforms features to follow a uniform or normal distribution using quantile mapping.
/// Fit sorts each column and stores quantile edges. Transform uses binary search + interpolation.
/// </summary>
public class QuantileTransformer : ITransformer
{
    private readonly string[] _columns;
    private readonly int _nQuantiles;
    private readonly QuantileOutputDistribution _outputDistribution;

    /// <summary>Sorted quantile edge values per column, learned during Fit.</summary>
    private Dictionary<string, double[]>? _quantileEdges;

    public string Name => "QuantileTransformer";

    /// <summary>Create a quantile transformer.</summary>
    /// <param name="nQuantiles">Number of quantile bins (default 1000).</param>
    /// <param name="outputDistribution">Target distribution (Uniform or Normal).</param>
    /// <param name="columns">Columns to transform. Empty = all numeric.</param>
    public QuantileTransformer(
        int nQuantiles = 1000,
        QuantileOutputDistribution outputDistribution = QuantileOutputDistribution.Uniform,
        params string[] columns)
    {
        if (nQuantiles < 2)
            throw new ArgumentOutOfRangeException(nameof(nQuantiles), "Must have at least 2 quantiles.");
        _nQuantiles = nQuantiles;
        _outputDistribution = outputDistribution;
        _columns = columns;
    }

    public ITransformer Fit(DataFrame df)
    {
        _quantileEdges = new Dictionary<string, double[]>();
        var cols = _columns.Length > 0
            ? _columns
            : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();

        foreach (var name in cols)
        {
            var col = df[name];
            // Collect non-null values
            var values = new List<double>(col.Length);
            for (int i = 0; i < col.Length; i++)
            {
                if (!col.IsNull(i))
                    values.Add(TypeHelpers.GetDouble(col, i));
            }

            if (values.Count == 0)
            {
                _quantileEdges[name] = Array.Empty<double>();
                continue;
            }

            values.Sort();

            // Compute quantile edges: nQuantiles+1 evenly spaced quantile points
            int nEdges = Math.Min(_nQuantiles + 1, values.Count);
            var edges = new double[nEdges];
            if (nEdges == 1)
            {
                edges[0] = values[0];
            }
            else
            {
                for (int q = 0; q < nEdges; q++)
                {
                    double frac = (double)q / (nEdges - 1);
                    double idx = frac * (values.Count - 1);
                    int lo = (int)idx;
                    int hi = Math.Min(lo + 1, values.Count - 1);
                    double t = idx - lo;
                    edges[q] = values[lo] * (1 - t) + values[hi] * t;
                }
            }

            _quantileEdges[name] = edges;
        }

        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_quantileEdges is null)
            throw new InvalidOperationException("Call Fit() first.");

        var result = df;
        foreach (var (name, edges) in _quantileEdges)
        {
            if (!df.ColumnNames.Contains(name)) continue;
            if (edges.Length == 0) continue;

            var col = df[name];
            var values = new double[df.RowCount];

            for (int i = 0; i < df.RowCount; i++)
            {
                if (col.IsNull(i))
                {
                    values[i] = double.NaN;
                    continue;
                }

                double v = TypeHelpers.GetDouble(col, i);
                double quantile = InterpolateQuantile(edges, v);

                values[i] = _outputDistribution == QuantileOutputDistribution.Normal
                    ? Probit(quantile)
                    : quantile;
            }

            result = result.Assign(name, new Column<double>(name, values));
        }

        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);

    /// <summary>
    /// Binary search on sorted edges to find the quantile position of value v.
    /// Returns a value in [0, 1].
    /// </summary>
    private static double InterpolateQuantile(double[] edges, double v)
    {
        if (edges.Length == 1) return 0.5;

        // Clamp to edge range
        if (v <= edges[0]) return 0.0;
        if (v >= edges[^1]) return 1.0;

        // Binary search for the interval containing v
        int lo = 0, hi = edges.Length - 1;
        while (lo < hi - 1)
        {
            int mid = (lo + hi) >> 1;
            if (edges[mid] <= v)
                lo = mid;
            else
                hi = mid;
        }

        // Linear interpolation within the interval
        double range = edges[hi] - edges[lo];
        double t = range > 0 ? (v - edges[lo]) / range : 0.5;
        double qLo = (double)lo / (edges.Length - 1);
        double qHi = (double)hi / (edges.Length - 1);

        return qLo + t * (qHi - qLo);
    }

    /// <summary>
    /// Inverse CDF of the standard normal distribution (probit function).
    /// Uses rational approximation (Abramowitz and Stegun / Peter Acklam).
    /// Clamps input to (epsilon, 1-epsilon) to avoid infinity.
    /// </summary>
    internal static double Probit(double p)
    {
        const double epsilon = 1e-8;
        p = Math.Clamp(p, epsilon, 1.0 - epsilon);

        // Rational approximation coefficients (Acklam)
        const double a1 = -3.969683028665376e+01;
        const double a2 = 2.209460984245205e+02;
        const double a3 = -2.759285104469687e+02;
        const double a4 = 1.383577518672690e+02;
        const double a5 = -3.066479806614716e+01;
        const double a6 = 2.506628277459239e+00;

        const double b1 = -5.447609879822406e+01;
        const double b2 = 1.615858368580409e+02;
        const double b3 = -1.556989798598866e+02;
        const double b4 = 6.680131188771972e+01;
        const double b5 = -1.328068155288572e+01;

        const double c1 = -7.784894002430293e-03;
        const double c2 = -3.223964580411365e-01;
        const double c3 = -2.400758277161838e+00;
        const double c4 = -2.549732539343734e+00;
        const double c5 = 4.374664141464968e+00;
        const double c6 = 2.938163982698783e+00;

        const double d1 = 7.784695709041462e-03;
        const double d2 = 3.224671290700398e-01;
        const double d3 = 2.445134137142996e+00;
        const double d4 = 3.754408661907416e+00;

        const double pLow = 0.02425;
        const double pHigh = 1.0 - pLow;

        double q, r;

        if (p < pLow)
        {
            q = Math.Sqrt(-2.0 * Math.Log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
        else if (p <= pHigh)
        {
            q = p - 0.5;
            r = q * q;
            return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        }
        else
        {
            q = Math.Sqrt(-2.0 * Math.Log(1.0 - p));
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                    ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
    }

    private static bool IsNumeric(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
