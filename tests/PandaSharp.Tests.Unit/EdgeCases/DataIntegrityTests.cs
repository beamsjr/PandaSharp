using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Missing;
using PandaSharp.Reshape;
using PandaSharp.Statistics;

namespace PandaSharp.Tests.Unit.EdgeCases;

public class DataIntegrityTests
{
    // ═══════════════════════════════════════════════════════════════
    // Bug 1: Describe mean is wrong when NaN values are present.
    // count includes NaN in the divisor but NaN is skipped in the sum,
    // producing an artificially low mean.
    // ═══════════════════════════════════════════════════════════════
    [Fact]
    public void Describe_WithNaN_MeanShouldExcludeNaN()
    {
        // 3 real values: 1, 2, 3 → mean should be 2.0
        var col = new Column<double>("x", [1.0, 2.0, double.NaN, 3.0]);
        var df = new DataFrame(col);
        var desc = df.Describe();

        // "mean" is at row index 1
        var meanCol = desc.GetColumn<double>("x");
        double mean = meanCol[1]!.Value;
        // Correct mean of {1, 2, 3} = 2.0
        mean.Should().BeApproximately(2.0, 0.001,
            "mean should exclude NaN values from both sum and count");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 2: Describe std is NaN when any value is NaN.
    // The std loop iterates over all data[] including NaN entries,
    // and (NaN - mean)^2 = NaN propagates to sumSq.
    // ═══════════════════════════════════════════════════════════════
    [Fact]
    public void Describe_WithNaN_StdShouldNotBeNaN()
    {
        var col = new Column<double>("x", [1.0, 2.0, double.NaN, 3.0]);
        var df = new DataFrame(col);
        var desc = df.Describe();

        var stdCol = desc.GetColumn<double>("x");
        double std = stdCol[2]!.Value; // row 2 = "std"
        double.IsNaN(std).Should().BeFalse(
            "std should be computed from non-NaN values only");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 3: FastCorr includes NaN in mean calculation.
    // When all columns are Column<double>, the fast path sums all
    // values including NaN, making mean=NaN and corrupting the
    // entire correlation matrix.
    // ═══════════════════════════════════════════════════════════════
    [Fact]
    public void Corr_WithNaN_ShouldNotCorruptMatrix()
    {
        // a has a NaN; ignoring NaN, a=[1,2,4] and b=[2,4,8] are perfectly correlated
        var df = new DataFrame(
            new Column<double>("a", [1.0, 2.0, double.NaN, 4.0]),
            new Column<double>("b", [2.0, 4.0, 6.0, 8.0])
        );
        var corr = df.Corr();
        // a's self-correlation should still be 1.0 (not NaN)
        var aCol = corr.GetColumn<double>("a");
        double selfCorrA = aCol[0]!.Value;
        double.IsNaN(selfCorrA).Should().BeFalse(
            "self-correlation of column 'a' should not be NaN even though a contains NaN values");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 4: DropColumn silently succeeds for non-existent column.
    // Unlike RemoveColumn in pandas which raises, DropColumn just
    // filters by name and returns the same DataFrame with no error.
    // ═══════════════════════════════════════════════════════════════
    [Fact]
    public void DropColumn_NonExistent_ShouldThrow()
    {
        var df = new DataFrame(new Column<int>("a", [1, 2, 3]));
        var act = () => df.DropColumn("nonexistent");
        act.Should().Throw<KeyNotFoundException>(
            "dropping a non-existent column should throw");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 5: CrossTab string fast path treats null as empty string.
    // Null values are coerced to "" in the string fast path,
    // making null and "" indistinguishable in the cross-tabulation.
    // ═══════════════════════════════════════════════════════════════
    [Fact]
    public void CrossTab_StringWithNulls_ShouldDistinguishNullFromEmpty()
    {
        var df = new DataFrame(
            new StringColumn("row", [null, "", "a"]),
            new StringColumn("col", ["x", "x", "x"])
        );
        var ct = df.CrossTab("row", "col");
        // Should have 3 distinct row values: null (or "null"/""), "", and "a"
        // The bug: null gets coerced to "" so only 2 rows appear
        ct.RowCount.Should().Be(3,
            "null and empty string should be distinct categories");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 6: Describe QuickSelect is corrupted by NaN.
    // NaN values in the compacted data array break QuickSelect
    // because NaN < pivot is always false, causing NaN to be
    // incorrectly placed and corrupting quantile results.
    // ═══════════════════════════════════════════════════════════════
    [Fact]
    public void Describe_WithNaN_QuantilesShouldBeCorrect()
    {
        // Known values: 1, 2, 3, 4 + a NaN
        var col = new Column<double>("x", [1.0, double.NaN, 2.0, 3.0, 4.0]);
        var df = new DataFrame(col);
        var desc = df.Describe();

        var xCol = desc.GetColumn<double>("x");
        double min = xCol[3]!.Value;  // row 3 = "min"
        double max = xCol[7]!.Value;  // row 7 = "max"

        min.Should().BeApproximately(1.0, 0.001, "min should exclude NaN");
        max.Should().BeApproximately(4.0, 0.001, "max should exclude NaN");

        double median = xCol[5]!.Value; // row 5 = "50%"
        double.IsNaN(median).Should().BeFalse("median should not be NaN");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 7: AddColumn on empty DataFrame skips length check.
    // When RowCount is 0, the condition `column.Length != RowCount && RowCount > 0`
    // is false, so any-length column can be added. Then adding a second
    // column of different length also succeeds (first col sets RowCount).
    // This creates an inconsistent DataFrame.
    // ═══════════════════════════════════════════════════════════════
    [Fact]
    public void AddColumn_ToEmptyDataFrame_ShouldSetRowCount()
    {
        var df = new DataFrame(); // 0 rows, 0 columns
        var df2 = df.AddColumn(new Column<int>("a", [1, 2, 3]));
        df2.RowCount.Should().Be(3);

        // Now adding a column of different length should fail
        var act = () => df2.AddColumn(new Column<int>("b", [1, 2]));
        act.Should().Throw<ArgumentException>(
            "adding a column with mismatched length should throw");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 8: Melt doesn't validate idVars.
    // When idVars contains a column not in the DataFrame, it throws
    // a raw KeyNotFoundException from the indexer instead of a
    // descriptive error.
    // ═══════════════════════════════════════════════════════════════
    [Fact]
    public void Melt_WithInvalidIdVars_ShouldThrow()
    {
        var df = new DataFrame(
            new Column<int>("a", [1, 2]),
            new Column<int>("b", [3, 4])
        );
        // This should throw (it does throw KeyNotFoundException already,
        // but let's verify the behavior is not silently broken)
        var act = () => df.Melt(["nonexistent"], ["a"]);
        act.Should().Throw<Exception>();
    }
}
