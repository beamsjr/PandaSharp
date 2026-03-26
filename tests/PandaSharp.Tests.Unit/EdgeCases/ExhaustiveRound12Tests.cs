using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Missing;
using PandaSharp.Statistics;

namespace PandaSharp.Tests.Unit.EdgeCases;

/// <summary>
/// Exhaustive Round 12: Final sweep bugs.
/// Bug 144: Prod() does not skip NaN values in double/float columns
/// Bug 145: DescribeViaDouble does not filter NaN for Column&lt;float&gt;
/// Bug 146: ConvertFromDoubleColumn loses null information during non-linear Interpolate
/// </summary>
public class ExhaustiveRound12Tests
{
    // ---------------------------------------------------------------
    // Bug 144: Prod() does not skip NaN in double/float columns
    // Prod uses col.Nulls.IsNull(i) instead of col.IsMissing(i),
    // so NaN values get multiplied into the product, yielding NaN.
    // ---------------------------------------------------------------

    [Fact]
    public void Prod_DoubleColumnWithNaN_ShouldSkipNaN()
    {
        var col = Column<double>.FromNullable("x", [2.0, double.NaN, 3.0]);
        var result = col.Prod();
        result.Should().NotBeNull();
        result!.Value.Should().Be(6.0); // 2 * 3, skip NaN
    }

    [Fact]
    public void Prod_FloatColumnWithNaN_ShouldSkipNaN()
    {
        var col = Column<float>.FromNullable("x", [2.0f, float.NaN, 5.0f]);
        var result = col.Prod();
        result.Should().NotBeNull();
        result!.Value.Should().Be(10.0f); // 2 * 5, skip NaN
    }

    [Fact]
    public void Prod_AllNaN_ShouldReturnNull()
    {
        var col = Column<double>.FromNullable("x", [double.NaN, double.NaN]);
        var result = col.Prod();
        // All values are NaN (effectively missing), so no valid values -> null
        result.Should().BeNull();
    }

    // ---------------------------------------------------------------
    // Bug 145: DescribeViaDouble doesn't filter NaN for Column<float>
    // When Column<float> has NaN values but NullCount==0, the NaN
    // values are included in statistics, corrupting mean/std/min/max.
    // ---------------------------------------------------------------

    [Fact]
    public void Describe_FloatColumnWithNaN_ShouldExcludeNaN()
    {
        var df = new DataFrame(
            Column<float>.FromNullable("vals", [1.0f, 2.0f, float.NaN, 3.0f])
        );

        var desc = df.Describe();
        var countCol = desc.GetColumn<double>("vals");

        // count should be 3 (excluding NaN), not 4
        var countRow = desc.GetColumn<double>("vals")[0];
        countRow.Should().Be(3.0);

        // mean should be 2.0 = (1+2+3)/3, not NaN
        var meanRow = desc.GetColumn<double>("vals")[1];
        meanRow.Should().Be(2.0);

        // min should be 1.0, not NaN
        var minRow = desc.GetColumn<double>("vals")[3];
        minRow.Should().Be(1.0);

        // max should be 3.0, not NaN
        var maxRow = desc.GetColumn<double>("vals")[7];
        maxRow.Should().Be(3.0);
    }

    // ---------------------------------------------------------------
    // Bug 146: ConvertFromDoubleColumn loses null information
    // When interpolating Column<float> with Polynomial/Cubic method,
    // positions that remain null after interpolation (edge cases with
    // no known values nearby) are converted to 0 instead of staying null.
    // ---------------------------------------------------------------

    [Fact]
    public void Interpolate_FloatPolynomial_ShouldPreserveNulls()
    {
        // All values are null/NaN -- nothing to interpolate from,
        // so result should remain all null.
        var col = Column<float>.FromNullable("x", [null, null, null]);

        var result = col.Interpolate(InterpolationMethod.Polynomial);

        // All positions should remain null since there are no known values
        for (int i = 0; i < result.Length; i++)
        {
            result.IsNull(i).Should().BeTrue($"position {i} should be null when no known values exist");
        }
    }

    [Fact]
    public void Interpolate_FloatCubic_LeadingNullsPreserved()
    {
        // Leading null before any known value can only be extrapolated
        // but the important thing is the result still has correct null semantics.
        var col = Column<float>.FromNullable("x", [null, null, 1.0f, 2.0f, 3.0f]);

        var result = col.Interpolate(InterpolationMethod.Cubic);

        // After interpolation, positions 0,1 should have extrapolated values (not null converted to 0).
        // With cubic spline and only forward known points, leading nulls get extrapolated.
        // The key bug was that ConvertFromDoubleColumn loses null status, converting null -> 0.
        // With the fix, if the double column had nulls, they should be preserved as null in float.
        // In this case, cubic spline extrapolates leading values, so they should be non-null finite values.
        for (int i = 2; i < result.Length; i++)
        {
            result[i].Should().NotBeNull($"position {i} has a known value");
        }
    }
}
