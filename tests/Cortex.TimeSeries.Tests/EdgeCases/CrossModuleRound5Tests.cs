using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.TimeSeries.Features;
using Xunit;

namespace Cortex.TimeSeries.Tests.EdgeCases;

/// <summary>
/// Round 5 bug hunting: cross-module data flow issues in TimeSeries features.
/// </summary>
public class CrossModuleRound5Tests
{
    /// <summary>
    /// Bug: RollingFeatures uses a running sum over GetDoubleArray() output.
    /// GetDoubleArray returns NaN for null values. Once NaN enters the running sum,
    /// it poisons ALL subsequent windows — even those that contain no null values.
    ///
    /// For example, with values [1, null, 3, 4, 5] and window=3:
    /// - Window [1,null,3]: sum = NaN (expected: null or handle gracefully)
    /// - Window [null,3,4]: sum = NaN (correct, null is in window)
    /// - Window [3,4,5]: sum = NaN (BUG! No nulls in this window, should be 12)
    ///
    /// The running sum approach computes: sum += values[i]; if (i >= w) sum -= values[i-w];
    /// But NaN + anything = NaN, and NaN - NaN = NaN, so once NaN enters, sum stays NaN forever.
    /// </summary>
    [Fact]
    public void RollingFeatures_NullValueDoesNotPoisonSubsequentWindows()
    {
        // Arrange: column with a null at index 1
        var values = new double?[] { 1.0, null, 3.0, 4.0, 5.0, 6.0 };
        var df = new DataFrame(
            Column<double>.FromNullable("value", values)
        );

        var rolling = new RollingFeatures("value", 3);
        rolling.Fit(df);

        // Act
        var result = rolling.Transform(df);

        // Assert: the window [3,4,5] (indices 2,3,4) should NOT be NaN
        // It contains no null values, so the rolling mean should be (3+4+5)/3 = 4.0
        var meanCol = result.GetColumn<double>("value_rolling_3_mean");

        // Index 4: window is [3, 4, 5] — no nulls, should be 4.0
        meanCol.IsNull(4).Should().BeFalse("window [3,4,5] has no nulls");
        meanCol[4].Should().BeApproximately(4.0, 0.001,
            "rolling mean of [3,4,5] should be 4.0, not NaN from poison");

        // Index 5: window is [4, 5, 6] — no nulls, should be 5.0
        meanCol.IsNull(5).Should().BeFalse("window [4,5,6] has no nulls");
        meanCol[5].Should().BeApproximately(5.0, 0.001,
            "rolling mean of [4,5,6] should be 5.0, not NaN from poison");
    }

    /// <summary>
    /// Bug: RollingFeatures min/max uses comparisons that fail with NaN.
    /// NaN < anything is always false, so if a NaN is in the window:
    /// - min stays at double.MaxValue
    /// - max stays at double.MinValue
    /// After NaN leaves the window, min/max should recover to correct values.
    /// </summary>
    [Fact]
    public void RollingFeatures_MinMax_RecoverAfterNullLeavesWindow()
    {
        var values = new double?[] { 1.0, null, 3.0, 4.0, 5.0 };
        var df = new DataFrame(
            Column<double>.FromNullable("value", values)
        );

        var rolling = new RollingFeatures("value", 3);
        rolling.Fit(df);
        var result = rolling.Transform(df);

        // Window at index 4 is [3, 4, 5] — no nulls
        var minCol = result.GetColumn<double>("value_rolling_3_min");
        var maxCol = result.GetColumn<double>("value_rolling_3_max");

        minCol.IsNull(4).Should().BeFalse();
        minCol[4].Should().Be(3.0, "min of [3,4,5] should be 3.0");

        maxCol.IsNull(4).Should().BeFalse();
        maxCol[4].Should().Be(5.0, "max of [3,4,5] should be 5.0");
    }

    /// <summary>
    /// Bug: FourierFeatures.Transform with a DateTime index column on an empty DataFrame
    /// crashes with IndexOutOfRangeException because it accesses span[0] to get the origin
    /// date before checking if the DataFrame has any rows.
    /// </summary>
    [Fact]
    public void FourierFeatures_EmptyDataFrame_WithDateTimeIndex_DoesNotCrash()
    {
        // Create a DataFrame with DateTime column but 0 rows (via filter)
        var df = new DataFrame(
            new Column<DateTime>("date", new DateTime[] { DateTime.Now }),
            new Column<double>("value", new double[] { 1.0 })
        );
        var emptyDf = df.WhereDouble("value", v => v > 100); // filters all out
        emptyDf.RowCount.Should().Be(0);

        var fourier = new FourierFeatures(new[] { 7.0 }, harmonics: 2, indexColumn: "date");
        fourier.Fit(emptyDf);

        // This should not crash
        var act = () => fourier.Transform(emptyDf);
        act.Should().NotThrow("FourierFeatures should handle empty DataFrames gracefully");
    }

    /// <summary>
    /// RollingFeatures: windows that contain a null should produce null for all stats,
    /// and windows after the null leaves should produce correct values.
    /// </summary>
    [Fact]
    public void RollingFeatures_WindowContainingNull_ProducesNull()
    {
        var values = new double?[] { 1.0, null, 3.0, 4.0, 5.0 };
        var df = new DataFrame(
            Column<double>.FromNullable("value", values)
        );

        var rolling = new RollingFeatures("value", 3);
        rolling.Fit(df);
        var result = rolling.Transform(df);

        var meanCol = result.GetColumn<double>("value_rolling_3_mean");

        // Index 2: window is [1, null, 3] — has null, should be null
        meanCol.IsNull(2).Should().BeTrue("window [1,null,3] contains a null");

        // Index 3: window is [null, 3, 4] — has null, should be null
        meanCol.IsNull(3).Should().BeTrue("window [null,3,4] contains a null");
    }
}
