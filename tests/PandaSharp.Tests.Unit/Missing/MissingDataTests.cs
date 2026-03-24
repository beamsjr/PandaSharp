using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Missing;

namespace PandaSharp.Tests.Unit.Missing;

public class MissingDataTests
{
    [Fact]
    public void IsNa_ReturnsTrueForNulls()
    {
        var col = Column<int>.FromNullable("x", [1, null, 3]);
        col.IsNa().Should().Equal([false, true, false]);
    }

    [Fact]
    public void NotNa_ReturnsTrueForNonNulls()
    {
        var col = Column<int>.FromNullable("x", [1, null, 3]);
        col.NotNa().Should().Equal([true, false, true]);
    }

    [Fact]
    public void FillNa_Scalar_ReplacesNulls()
    {
        var col = Column<int>.FromNullable("x", [1, null, 3, null]);
        var filled = col.FillNa(0);

        filled[0].Should().Be(1);
        filled[1].Should().Be(0);
        filled[2].Should().Be(3);
        filled[3].Should().Be(0);
        filled.NullCount.Should().Be(0);
    }

    [Fact]
    public void FillNa_Forward_FillsWithLastKnownValue()
    {
        var col = Column<int>.FromNullable("x", [1, null, null, 4, null]);
        var filled = col.FillNa(FillStrategy.Forward);

        filled[0].Should().Be(1);
        filled[1].Should().Be(1);
        filled[2].Should().Be(1);
        filled[3].Should().Be(4);
        filled[4].Should().Be(4);
    }

    [Fact]
    public void FillNa_Forward_LeadingNullsRemainNull()
    {
        var col = Column<int>.FromNullable("x", [null, null, 3]);
        var filled = col.FillNa(FillStrategy.Forward);

        filled[0].Should().BeNull();
        filled[1].Should().BeNull();
        filled[2].Should().Be(3);
    }

    [Fact]
    public void FillNa_Backward_FillsWithNextKnownValue()
    {
        var col = Column<int>.FromNullable("x", [null, 2, null, null, 5]);
        var filled = col.FillNa(FillStrategy.Backward);

        filled[0].Should().Be(2);
        filled[1].Should().Be(2);
        filled[2].Should().Be(5);
        filled[3].Should().Be(5);
        filled[4].Should().Be(5);
    }

    [Fact]
    public void FillNa_String_Scalar()
    {
        var col = new StringColumn("s", ["a", null, "c"]);
        var filled = col.FillNa("?");

        filled[0].Should().Be("a");
        filled[1].Should().Be("?");
        filled[2].Should().Be("c");
    }

    [Fact]
    public void FillNa_String_Forward()
    {
        var col = new StringColumn("s", ["a", null, null, "d"]);
        var filled = col.FillNa(FillStrategy.Forward);

        filled[1].Should().Be("a");
        filled[2].Should().Be("a");
    }

    [Fact]
    public void DropNa_RemovesRowsWithNulls()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3]),
            new StringColumn("B", ["x", "y", null])
        );

        var result = df.DropNa();
        result.RowCount.Should().Be(1);
        result.GetColumn<int>("A")[0].Should().Be(1);
    }

    [Fact]
    public void DropNa_WithThreshold_KeepsRowsWithEnoughNonNull()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3]),
            new StringColumn("B", ["x", "y", null])
        );

        var result = df.DropNa(threshold: 1); // need at least 1 non-null
        result.RowCount.Should().Be(3); // all rows have at least 1 non-null
    }

    [Fact]
    public void DropNa_Axis1_DropsColumnsWithNulls()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            Column<int>.FromNullable("B", [1, null, 3])
        );

        var result = df.DropNa(axis: 1);
        result.ColumnCount.Should().Be(1);
        result.ColumnNames.Should().Contain("A");
    }

    [Fact]
    public void Interpolate_Linear_FillsGaps()
    {
        var col = Column<double>.FromNullable("x", [1.0, null, null, 4.0]);
        var interp = col.Interpolate();

        interp[0].Should().Be(1.0);
        interp[1].Should().BeApproximately(2.0, 0.001);
        interp[2].Should().BeApproximately(3.0, 0.001);
        interp[3].Should().Be(4.0);
    }

    [Fact]
    public void Interpolate_LeadingNull_UsesNextValue()
    {
        var col = Column<double>.FromNullable("x", [null, 2.0, 4.0]);
        var interp = col.Interpolate();

        interp[0].Should().Be(2.0);
    }

    [Fact]
    public void Interpolate_TrailingNull_UsesLastValue()
    {
        var col = Column<double>.FromNullable("x", [1.0, 2.0, null]);
        var interp = col.Interpolate();

        interp[2].Should().Be(2.0);
    }

    // ===== Polynomial interpolation =====

    [Fact]
    public void Interpolate_Polynomial_BasicGap()
    {
        // Known: 0→1, 1→4, 3→16 (y = x^2 + ... ish)
        var col = Column<double>.FromNullable("x", [1.0, 4.0, null, 16.0]);
        var interp = col.Interpolate(InterpolationMethod.Polynomial);

        interp.NullCount.Should().Be(0);
        interp[2].Should().NotBe(0); // should interpolate, not leave as 0
        // Polynomial through (0,1), (1,4), (3,16) should give ~9 at x=2
        interp[2]!.Value.Should().BeApproximately(9.0, 1.0);
    }

    [Fact]
    public void Interpolate_Polynomial_NoNulls_ReturnsOriginal()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0]);
        var interp = col.Interpolate(InterpolationMethod.Polynomial);

        interp[0].Should().Be(1.0);
        interp[1].Should().Be(2.0);
        interp[2].Should().Be(3.0);
    }

    [Fact]
    public void Interpolate_Polynomial_LinearData_MatchesLinear()
    {
        // For perfectly linear data, polynomial should match linear
        var col = Column<double>.FromNullable("x", [0.0, null, null, 6.0]);
        var linear = col.Interpolate(InterpolationMethod.Linear);
        var poly = col.Interpolate(InterpolationMethod.Polynomial);

        poly[1]!.Value.Should().BeApproximately(linear[1]!.Value, 0.01);
        poly[2]!.Value.Should().BeApproximately(linear[2]!.Value, 0.01);
    }

    // ===== Cubic spline interpolation =====

    [Fact]
    public void Interpolate_Cubic_BasicGap()
    {
        var col = Column<double>.FromNullable("x", [0.0, null, null, 3.0, null, 5.0]);
        var interp = col.Interpolate(InterpolationMethod.Cubic);

        interp.NullCount.Should().Be(0);
        // Cubic spline through (0,0), (3,3), (5,5) should give ~1 and ~2 at positions 1,2
        interp[1]!.Value.Should().BeApproximately(1.0, 0.5);
        interp[2]!.Value.Should().BeApproximately(2.0, 0.5);
    }

    [Fact]
    public void Interpolate_Cubic_SmoothCurve()
    {
        // Known: sin-like data with gaps
        var col = Column<double>.FromNullable("x", [0.0, null, 1.0, null, 0.0]);
        var interp = col.Interpolate(InterpolationMethod.Cubic);

        interp.NullCount.Should().Be(0);
        // Spline should produce smooth values between known points
        interp[1]!.Value.Should().BeInRange(-0.5, 1.5);
        interp[3]!.Value.Should().BeInRange(-0.5, 1.5);
    }

    [Fact]
    public void Interpolate_Spline_SameAsCubic()
    {
        var col = Column<double>.FromNullable("x", [1.0, null, 3.0, null, 5.0]);
        var cubic = col.Interpolate(InterpolationMethod.Cubic);
        var spline = col.Interpolate(InterpolationMethod.Spline);

        cubic[1].Should().Be(spline[1]);
        cubic[3].Should().Be(spline[3]);
    }

    [Fact]
    public void Interpolate_Cubic_IntColumn()
    {
        var col = Column<int>.FromNullable("x", [0, null, null, 6]);
        var interp = col.Interpolate(InterpolationMethod.Cubic);

        interp.NullCount.Should().Be(0);
        interp[1]!.Value.Should().BeInRange(1, 3);
        interp[2]!.Value.Should().BeInRange(3, 5);
    }
}
