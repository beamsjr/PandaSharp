using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Transformers;

namespace Cortex.ML.Tests;

public class TransformerAdvancedTests
{
    // ----------------------------------------------------------------
    // QuantileTransformer
    // ----------------------------------------------------------------

    [Fact]
    public void QuantileTransformer_Uniform_OutputIsApproximatelyUniform()
    {
        // Generate a right-skewed distribution (exponential-like)
        var rng = new Random(42);
        var values = new double[500];
        for (int i = 0; i < values.Length; i++)
            values[i] = -Math.Log(1.0 - rng.NextDouble()); // Exponential(1)

        var df = new DataFrame(new Column<double>("X", values));
        var qt = new QuantileTransformer(nQuantiles: 100, outputDistribution: QuantileOutputDistribution.Uniform, columns: "X");
        var result = qt.FitTransform(df);

        var col = result.GetColumn<double>("X");

        // All values should be in [0, 1]
        for (int i = 0; i < col.Length; i++)
        {
            col[i]!.Value.Should().BeGreaterThanOrEqualTo(0.0);
            col[i]!.Value.Should().BeLessThanOrEqualTo(1.0);
        }

        // Check approximate uniformity: mean should be ~0.5
        double mean = 0;
        for (int i = 0; i < col.Length; i++)
            mean += col[i]!.Value;
        mean /= col.Length;

        mean.Should().BeApproximately(0.5, 0.1, "uniform distribution has mean 0.5");
    }

    [Fact]
    public void QuantileTransformer_Normal_OutputHasApproximatelyZeroMean()
    {
        var rng = new Random(123);
        var values = new double[500];
        for (int i = 0; i < values.Length; i++)
            values[i] = -Math.Log(1.0 - rng.NextDouble());

        var df = new DataFrame(new Column<double>("X", values));
        var qt = new QuantileTransformer(nQuantiles: 100, outputDistribution: QuantileOutputDistribution.Normal, columns: "X");
        var result = qt.FitTransform(df);

        var col = result.GetColumn<double>("X");

        double mean = 0;
        for (int i = 0; i < col.Length; i++)
            mean += col[i]!.Value;
        mean /= col.Length;

        mean.Should().BeApproximately(0.0, 0.2, "normal distribution has mean ~0");
    }

    [Fact]
    public void QuantileTransformer_Transform_UseFittedParams()
    {
        var train = new DataFrame(new Column<double>("X", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        var test = new DataFrame(new Column<double>("X", [5.0]));

        var qt = new QuantileTransformer(nQuantiles: 10, outputDistribution: QuantileOutputDistribution.Uniform, columns: "X");
        qt.Fit(train);
        var result = qt.Transform(test);

        var col = result.GetColumn<double>("X");
        // Median of [1..10] → should map to ~0.5
        col[0]!.Value.Should().BeApproximately(0.44, 0.15);
    }

    [Fact]
    public void QuantileTransformer_ThrowsBeforeFit()
    {
        var qt = new QuantileTransformer();
        var df = new DataFrame(new Column<double>("X", [1, 2, 3]));

        var act = () => qt.Transform(df);
        act.Should().Throw<InvalidOperationException>();
    }

    // ----------------------------------------------------------------
    // PowerTransformer
    // ----------------------------------------------------------------

    [Fact]
    public void PowerTransformer_BoxCox_ReducesSkewnessOfLogNormalData()
    {
        // Generate log-normal data (highly right-skewed)
        var rng = new Random(42);
        var values = new double[300];
        for (int i = 0; i < values.Length; i++)
        {
            // Box-Muller for normal, then exp for log-normal
            double u1 = rng.NextDouble();
            double u2 = rng.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            values[i] = Math.Exp(z * 0.5 + 1.0); // log-normal with some spread
        }

        // Compute skewness of original data
        double originalSkewness = ComputeSkewness(values);
        originalSkewness.Should().BeGreaterThan(0.5, "log-normal data should be right-skewed");

        var df = new DataFrame(new Column<double>("X", values));
        var pt = new PowerTransformer(PowerMethod.BoxCox, standardize: true, columns: "X");
        var result = pt.FitTransform(df);

        var col = result.GetColumn<double>("X");
        var transformed = new double[col.Length];
        for (int i = 0; i < col.Length; i++)
            transformed[i] = col[i]!.Value;

        double transformedSkewness = ComputeSkewness(transformed);
        Math.Abs(transformedSkewness).Should().BeLessThan(Math.Abs(originalSkewness),
            "Box-Cox should reduce skewness");
    }

    [Fact]
    public void PowerTransformer_YeoJohnson_HandlesNegativeValues()
    {
        var values = new double[] { -5, -3, -1, 0, 1, 3, 5, 10, 20 };
        var df = new DataFrame(new Column<double>("X", values));

        var pt = new PowerTransformer(PowerMethod.YeoJohnson, standardize: true, columns: "X");
        var result = pt.FitTransform(df);

        var col = result.GetColumn<double>("X");
        col.Length.Should().Be(values.Length);

        // Check output is roughly standardised
        double mean = 0;
        for (int i = 0; i < col.Length; i++)
            mean += col[i]!.Value;
        mean /= col.Length;

        mean.Should().BeApproximately(0.0, 0.15, "standardised output should have ~0 mean");
    }

    [Fact]
    public void PowerTransformer_BoxCox_ThrowsOnNonPositiveData()
    {
        var df = new DataFrame(new Column<double>("X", [-1, 0, 1, 2]));
        var pt = new PowerTransformer(PowerMethod.BoxCox, columns: "X");

        var act = () => pt.Fit(df);
        act.Should().Throw<ArgumentException>().WithMessage("*positive*");
    }

    [Fact]
    public void PowerTransformer_ThrowsBeforeFit()
    {
        var pt = new PowerTransformer();
        var df = new DataFrame(new Column<double>("X", [1, 2, 3]));

        var act = () => pt.Transform(df);
        act.Should().Throw<InvalidOperationException>();
    }

    // ---- Helpers ----

    private static double ComputeSkewness(double[] data)
    {
        int n = data.Length;
        double mean = 0;
        for (int i = 0; i < n; i++) mean += data[i];
        mean /= n;

        double m2 = 0, m3 = 0;
        for (int i = 0; i < n; i++)
        {
            double d = data[i] - mean;
            m2 += d * d;
            m3 += d * d * d;
        }
        m2 /= n;
        m3 /= n;

        double std = Math.Sqrt(m2);
        if (std < 1e-10) return 0;
        return m3 / (std * std * std);
    }
}
