using Cortex;
using Cortex.Column;

namespace Cortex.ML.Transformers;

/// <summary>Power transform method.</summary>
public enum PowerMethod
{
    /// <summary>Box-Cox transform (requires strictly positive data).</summary>
    BoxCox,
    /// <summary>Yeo-Johnson transform (handles positive and negative data).</summary>
    YeoJohnson
}

/// <summary>
/// Applies a power transform to make data more Gaussian-like.
/// Estimates optimal lambda per column via Brent's method (maximising log-likelihood).
/// Optionally standardises the result to zero mean and unit variance.
/// </summary>
public class PowerTransformer : ITransformer
{
    private readonly string[] _columns;
    private readonly PowerMethod _method;
    private readonly bool _standardize;

    /// <summary>Fitted lambda and optional standardisation params per column.</summary>
    private Dictionary<string, (double Lambda, double Mean, double Std)>? _params;

    public string Name => "PowerTransformer";

    /// <summary>Create a power transformer.</summary>
    /// <param name="method">BoxCox (positive only) or YeoJohnson (any sign).</param>
    /// <param name="standardize">Whether to standardise output to zero mean, unit variance.</param>
    /// <param name="columns">Columns to transform. Empty = all numeric.</param>
    public PowerTransformer(
        PowerMethod method = PowerMethod.YeoJohnson,
        bool standardize = true,
        params string[] columns)
    {
        _method = method;
        _standardize = standardize;
        _columns = columns;
    }

    public ITransformer Fit(DataFrame df)
    {
        _params = new Dictionary<string, (double, double, double)>();
        var cols = _columns.Length > 0
            ? _columns
            : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();

        foreach (var name in cols)
        {
            var col = df[name];
            var values = CollectValues(col);

            if (values.Length == 0)
            {
                _params[name] = (1.0, 0.0, 1.0);
                continue;
            }

            if (_method == PowerMethod.BoxCox)
            {
                for (int i = 0; i < values.Length; i++)
                {
                    if (values[i] <= 0)
                        throw new ArgumentException(
                            $"Box-Cox requires strictly positive data, but column '{name}' contains non-positive values.");
                }
            }

            // Find optimal lambda via Brent's method on negative log-likelihood
            double lambda = BrentMinimize(
                lam => -LogLikelihood(values, lam, _method),
                -5.0, 5.0, 1e-8, 200);

            // Compute mean/std of transformed data for standardisation
            double mean = 0, m2 = 0;
            if (_standardize)
            {
                var transformed = ApplyTransform(values, lambda, _method);
                for (int i = 0; i < transformed.Length; i++)
                    mean += transformed[i];
                mean /= transformed.Length;

                for (int i = 0; i < transformed.Length; i++)
                {
                    double d = transformed[i] - mean;
                    m2 += d * d;
                }
                double std = transformed.Length > 1 ? Math.Sqrt(m2 / (transformed.Length - 1)) : 1.0;
                _params[name] = (lambda, mean, std == 0 ? 1.0 : std);
            }
            else
            {
                _params[name] = (lambda, 0.0, 1.0);
            }
        }

        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_params is null)
            throw new InvalidOperationException("Call Fit() first.");

        var result = df;
        foreach (var (name, (lambda, mean, std)) in _params)
        {
            if (!df.ColumnNames.Contains(name)) continue;

            var col = df[name];
            var output = new double[df.RowCount];

            for (int i = 0; i < df.RowCount; i++)
            {
                if (col.IsNull(i))
                {
                    output[i] = double.NaN;
                    continue;
                }

                double v = TypeHelpers.GetDouble(col, i);
                double t = _method == PowerMethod.BoxCox
                    ? BoxCoxTransform(v, lambda)
                    : YeoJohnsonTransform(v, lambda);

                if (_standardize)
                    t = (t - mean) / std;

                output[i] = t;
            }

            result = result.Assign(name, new Column<double>(name, output));
        }

        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);

    // ---- Transform functions ----

    internal static double BoxCoxTransform(double x, double lambda)
    {
        if (Math.Abs(lambda) < 1e-10)
            return Math.Log(x);
        return (Math.Pow(x, lambda) - 1.0) / lambda;
    }

    internal static double YeoJohnsonTransform(double x, double lambda)
    {
        if (x >= 0)
        {
            if (Math.Abs(lambda) < 1e-10)
                return Math.Log(x + 1.0);
            return (Math.Pow(x + 1.0, lambda) - 1.0) / lambda;
        }
        else
        {
            if (Math.Abs(lambda - 2.0) < 1e-10)
                return -Math.Log(-x + 1.0);
            return -(Math.Pow(-x + 1.0, 2.0 - lambda) - 1.0) / (2.0 - lambda);
        }
    }

    // ---- Log-likelihood for lambda optimisation ----

    private static double LogLikelihood(double[] values, double lambda, PowerMethod method)
    {
        int n = values.Length;
        double sumLogLik = 0;
        double sumTransformed = 0;
        double sumTransformedSq = 0;

        var transformed = ApplyTransform(values, lambda, method);

        for (int i = 0; i < n; i++)
            sumTransformed += transformed[i];
        double mean = sumTransformed / n;

        for (int i = 0; i < n; i++)
        {
            double d = transformed[i] - mean;
            sumTransformedSq += d * d;
        }

        double variance = sumTransformedSq / n;
        if (variance <= 0) return double.NegativeInfinity;

        // Log-likelihood = -n/2 * ln(variance) + (lambda - 1) * sum(sign * ln(|x| + offset))
        sumLogLik = -0.5 * n * Math.Log(variance);

        for (int i = 0; i < n; i++)
        {
            if (method == PowerMethod.BoxCox)
                sumLogLik += (lambda - 1.0) * Math.Log(values[i]);
            else
                sumLogLik += (lambda - 1.0) * Math.Sign(values[i]) * Math.Log(Math.Abs(values[i]) + 1.0);
        }

        return sumLogLik;
    }

    private static double[] ApplyTransform(double[] values, double lambda, PowerMethod method)
    {
        var result = new double[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = method == PowerMethod.BoxCox
                ? BoxCoxTransform(values[i], lambda)
                : YeoJohnsonTransform(values[i], lambda);
        }
        return result;
    }

    // ---- Brent's method for scalar minimisation ----

    /// <summary>
    /// Brent's method to find the minimum of f on [a, b].
    /// Returns the x that minimises f.
    /// </summary>
    internal static double BrentMinimize(Func<double, double> f, double a, double b, double tol, int maxIter)
    {
        const double goldenRatio = 0.3819660112501051;

        double x = a + goldenRatio * (b - a);
        double w = x, v = x;
        double fx = f(x), fw = fx, fv = fx;
        double d = 0, e = 0;

        for (int iter = 0; iter < maxIter; iter++)
        {
            double midpoint = 0.5 * (a + b);
            double tol1 = tol * Math.Abs(x) + 1e-10;
            double tol2 = 2.0 * tol1;

            if (Math.Abs(x - midpoint) <= tol2 - 0.5 * (b - a))
                return x;

            bool useParabolic = false;
            double u = 0;

            if (Math.Abs(e) > tol1)
            {
                // Try parabolic interpolation
                double r = (x - w) * (fx - fv);
                double q = (x - v) * (fx - fw);
                double p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if (q > 0) p = -p; else q = -q;

                if (Math.Abs(p) < Math.Abs(0.5 * q * e) && p > q * (a - x) && p < q * (b - x))
                {
                    u = x + p / q;
                    if (u - a < tol2 || b - u < tol2)
                        u = x < midpoint ? x + tol1 : x - tol1;
                    useParabolic = true;
                    e = d;
                }
            }

            if (!useParabolic)
            {
                e = (x < midpoint ? b : a) - x;
                d = goldenRatio * e;
                u = x + d;
            }
            else
            {
                d = (u - x);
            }

            if (Math.Abs(u - x) < tol1)
                u = x + Math.Sign(u - x) * tol1;

            double fu = f(u);

            if (fu <= fx)
            {
                if (u < x) b = x; else a = x;
                v = w; fv = fw;
                w = x; fw = fx;
                x = u; fx = fu;
            }
            else
            {
                if (u < x) a = u; else b = u;
                if (fu <= fw || Math.Abs(w - x) < 1e-15)
                {
                    v = w; fv = fw;
                    w = u; fw = fu;
                }
                else if (fu <= fv || Math.Abs(v - x) < 1e-15 || Math.Abs(v - w) < 1e-15)
                {
                    v = u; fv = fu;
                }
            }
        }

        return x;
    }

    // ---- Helpers ----

    private static double[] CollectValues(IColumn col)
    {
        var values = new List<double>(col.Length);
        for (int i = 0; i < col.Length; i++)
        {
            if (!col.IsNull(i))
                values.Add(TypeHelpers.GetDouble(col, i));
        }
        return values.ToArray();
    }

    private static bool IsNumeric(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
