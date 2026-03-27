using Cortex.Column;
using Cortex.GroupBy;
using Cortex.Joins;
using Cortex.Missing;
using Cortex.Window;

namespace Cortex.Tests.Unit.EdgeCases;

/// <summary>
/// Round 8: Systematic NaN audit. Every method touching double/float data must handle NaN.
/// NaN in a Column&lt;double&gt; buffer with a valid null bitmask is the key scenario.
/// </summary>
public class NaNConsistencyTests
{
    /// <summary>
    /// Helper: create a Column&lt;double&gt; where NaN is stored in the buffer
    /// but the null bitmask says ALL values are valid (NullCount == 0).
    /// This is the tricky case that many code paths miss.
    /// </summary>
    private static Column<double> MakeNaNColumn(string name, params double[] values)
    {
        // Column<double>(name, values) sets NullBitmask.AllValid, so NaN is in buffer but IsNull is false
        return new Column<double>(name, values);
    }

    // ============================================================
    // Area 1: Window functions NaN audit
    // ============================================================

    [Fact]
    public void Rolling_Mean_WithNaN_SkipsNaN()
    {
        // NaN in buffer, null bitmask all-valid. Use minPeriods=1 so fewer valid values still produce output.
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0, 4.0);
        var rolling = col.Rolling(3, minPeriods: 1);
        var result = rolling.Mean();

        // Window [1, NaN, 3]: NaN should be skipped, mean of [1, 3] = 2.0
        Assert.True(result[2].HasValue, "Should have a value when NaN is skipped");
        Assert.False(double.IsNaN(result[2]!.Value), "Mean should not be NaN when valid values exist");
        Assert.Equal(2.0, result[2]!.Value, 5);
    }

    [Fact]
    public void Rolling_Sum_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0);
        var rolling = col.Rolling(3, minPeriods: 1);
        var result = rolling.Sum();

        // Window [1, NaN, 3]: should sum to 4.0, not NaN
        Assert.True(result[2].HasValue);
        Assert.False(double.IsNaN(result[2]!.Value), "Sum should skip NaN");
        Assert.Equal(4.0, result[2]!.Value, 5);
    }

    [Fact]
    public void Rolling_Min_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 5.0, double.NaN, 3.0);
        var rolling = col.Rolling(3, minPeriods: 1);
        var result = rolling.Min();

        // NaN < 3.0 is false, NaN < 5.0 is false — but NaN should be SKIPPED, not compared
        Assert.True(result[2].HasValue);
        Assert.Equal(3.0, result[2]!.Value, 5);
    }

    [Fact]
    public void Rolling_Max_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 5.0);
        var rolling = col.Rolling(3, minPeriods: 1);
        var result = rolling.Max();

        Assert.True(result[2].HasValue);
        Assert.Equal(5.0, result[2]!.Value, 5);
    }

    [Fact]
    public void Rolling_Std_WithNaN_SkipsNaN()
    {
        // Std needs at least 2 values, so use minPeriods=2
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0, 5.0);
        var rolling = col.Rolling(3, minPeriods: 2);
        var result = rolling.Std();

        // Window [1, NaN, 3]: should compute std of [1, 3]
        Assert.True(result[2].HasValue);
        Assert.False(double.IsNaN(result[2]!.Value), "Std should skip NaN values");
    }

    [Fact]
    public void Rolling_Var_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0, 5.0);
        var rolling = col.Rolling(3, minPeriods: 2);
        var result = rolling.Var();

        Assert.True(result[2].HasValue);
        Assert.False(double.IsNaN(result[2]!.Value), "Var should skip NaN values");
    }

    [Fact]
    public void RollingMeanFast_WithNaN_HandlesCorrectly()
    {
        // Column with NaN but NullCount == 0 — should NOT take the fast O(n) path
        // because that path doesn't handle NaN. Instead it should fall back to Apply.
        var col = MakeNaNColumn("val", 1.0, 2.0, double.NaN, 4.0, 5.0);
        var rolling = col.Rolling(3, minPeriods: 1);
        var result = rolling.Mean();

        // Window [2, NaN, 4] at index 3: NaN should be skipped, mean of [2, 4] = 3.0
        Assert.True(result[3].HasValue);
        Assert.False(double.IsNaN(result[3]!.Value), "RollingMeanFast should handle NaN");
        Assert.Equal(3.0, result[3]!.Value, 5);
    }

    // ============================================================
    // Area 1b: Expanding window NaN audit
    // ============================================================

    [Fact]
    public void Expanding_Sum_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0);
        var expanding = col.Expanding();
        var result = expanding.Sum();

        // After index 2: sum of [1, 3] = 4, not NaN
        Assert.True(result[2].HasValue);
        Assert.False(double.IsNaN(result[2]!.Value));
        Assert.Equal(4.0, result[2]!.Value, 5);
    }

    [Fact]
    public void Expanding_Mean_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0);
        var expanding = col.Expanding();
        var result = expanding.Mean();

        // Mean of [1, 3] = 2.0
        Assert.True(result[2].HasValue);
        Assert.False(double.IsNaN(result[2]!.Value));
        Assert.Equal(2.0, result[2]!.Value, 5);
    }

    [Fact]
    public void Expanding_Min_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 5.0, double.NaN, 3.0);
        var expanding = col.Expanding();
        var result = expanding.Min();

        Assert.True(result[2].HasValue);
        Assert.Equal(3.0, result[2]!.Value, 5);
    }

    [Fact]
    public void Expanding_Max_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 5.0);
        var expanding = col.Expanding();
        var result = expanding.Max();

        Assert.True(result[2].HasValue);
        Assert.Equal(5.0, result[2]!.Value, 5);
    }

    [Fact]
    public void Expanding_Std_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0);
        var expanding = col.Expanding();
        var result = expanding.Std();

        // Std of [1, 3] with ddof=1 = sqrt(2)
        Assert.True(result[2].HasValue);
        Assert.False(double.IsNaN(result[2]!.Value));
    }

    // ============================================================
    // Area 1c: EWM NaN audit
    // ============================================================

    [Fact]
    public void Ewm_Mean_WithNaN_SkipsNaN()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0);
        var ewm = col.Ewm(span: 3);
        var result = ewm.Mean();

        // NaN row should be null or at least not corrupt subsequent values
        // Index 0: 1.0 (first value)
        Assert.Equal(1.0, result[0]!.Value, 5);
        // Index 1 (NaN): should be null/skipped
        // Index 2: 3.0 should be incorporated without NaN corrupting
        Assert.True(result[2].HasValue);
        Assert.False(double.IsNaN(result[2]!.Value), "EWM should skip NaN, not propagate it");
    }

    // ============================================================
    // Area 2: GroupBy aggregation NaN audit
    // ============================================================

    [Fact]
    public void GroupBy_Count_DoesNotCountNaN()
    {
        var df = new DataFrame(
            new Column<double>("val", [1.0, double.NaN, 3.0, double.NaN]),
            new StringColumn("grp", ["A", "A", "A", "A"])
        );
        var result = df.GroupBy("grp").Count();
        var countCol = (Column<int>)result["val"];

        // NaN should NOT be counted as non-null. Group A has 2 valid values (1.0, 3.0).
        Assert.Equal(2, countCol[0]);
    }

    [Fact]
    public void GroupBy_First_SkipsNaN()
    {
        var df = new DataFrame(
            new Column<double>("val", [double.NaN, 2.0, 3.0]),
            new StringColumn("grp", ["A", "A", "A"])
        );
        var result = df.GroupBy("grp").First();
        var valCol = (Column<double>)result["val"];

        // First non-NaN value in group A is 2.0
        Assert.Equal(2.0, valCol.Buffer.Span[0], 5);
    }

    [Fact]
    public void GroupBy_Last_SkipsNaN()
    {
        var df = new DataFrame(
            new Column<double>("val", [1.0, 2.0, double.NaN]),
            new StringColumn("grp", ["A", "A", "A"])
        );
        var result = df.GroupBy("grp").Last();
        var valCol = (Column<double>)result["val"];

        // Last non-NaN value in group A is 2.0
        Assert.Equal(2.0, valCol.Buffer.Span[0], 5);
    }

    [Fact]
    public void GroupBy_Median_SkipsNaN()
    {
        var df = new DataFrame(
            new Column<double>("val", [1.0, double.NaN, 5.0, 3.0]),
            new StringColumn("grp", ["A", "A", "A", "A"])
        );
        var result = df.GroupBy("grp").Median();
        var valCol = (Column<double>)result["val"];

        // Median of [1.0, 3.0, 5.0] = 3.0 (NaN skipped)
        Assert.Equal(3.0, Convert.ToDouble(valCol.GetObject(0)), 5);
    }

    [Fact]
    public void GroupBy_Var_TypedFastPath_SkipsNaN()
    {
        var df = new DataFrame(
            new Column<double>("val", [1.0, double.NaN, 3.0]),
            new StringColumn("grp", ["A", "A", "A"])
        );
        var result = df.GroupBy("grp").Var();
        var valCol = (Column<double>)result["val"];

        // Var of [1.0, 3.0] with ddof=1 = 2.0
        var v = valCol.Buffer.Span[0];
        Assert.False(double.IsNaN(v), "Var should skip NaN");
        Assert.Equal(2.0, v, 5);
    }

    [Fact]
    public void GroupBy_ComputeNumericAggregate_SkipsNaN_ForFloatColumn()
    {
        var df = new DataFrame(
            new Column<float>("val", [1.0f, float.NaN, 3.0f]),
            new StringColumn("grp", ["A", "A", "A"])
        );
        var result = df.GroupBy("grp").Sum();
        var valCol = (Column<double>)result["val"];

        // NaN should be skipped, sum = 4.0
        Assert.False(double.IsNaN(valCol.Buffer.Span[0]), "Float NaN should be skipped in aggregation");
        Assert.Equal(4.0, valCol.Buffer.Span[0], 5);
    }

    // ============================================================
    // Area 3: Join with NaN keys
    // ============================================================

    [Fact]
    public void Join_NaN_Double_Keys_ShouldNotMatch()
    {
        var left = new DataFrame(
            new Column<double>("key", [1.0, double.NaN]),
            new StringColumn("left_val", ["a", "b"])
        );
        var right = new DataFrame(
            new Column<double>("key", [1.0, double.NaN]),
            new StringColumn("right_val", ["x", "y"])
        );

        var result = left.Join(right, "key", how: JoinType.Inner);

        // NaN keys should NOT match (IEEE 754: NaN != NaN)
        // Only key=1.0 should produce a match
        Assert.Equal(1, result.RowCount);
    }

    // ============================================================
    // Area 4: DropDuplicates with NaN
    // ============================================================

    [Fact]
    public void DropDuplicates_NaN_TreatedAsSameValue()
    {
        // In pandas, NaN == NaN for DropDuplicates purposes
        var df = new DataFrame(
            new Column<double>("val", [1.0, double.NaN, double.NaN, 2.0])
        );
        var result = df.DropDuplicates("val");

        // Should have 3 unique rows: 1.0, NaN, 2.0 (the two NaNs are considered duplicates)
        Assert.Equal(3, result.RowCount);
    }

    // ============================================================
    // Area 5: FillNa / Interpolate with NaN
    // ============================================================

    [Fact]
    public void FillNa_FillsNaN_InDoubleColumn()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0);
        var filled = col.FillNa(0.0);

        // NaN position should be filled with 0.0
        Assert.Equal(0.0, filled[1]!.Value, 5);
    }

    [Fact]
    public void FillNa_ForwardFill_FillsNaN()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0);
        var filled = col.FillNa(FillStrategy.Forward);

        // NaN should be forward-filled with 1.0
        Assert.Equal(1.0, filled[1]!.Value, 5);
    }

    [Fact]
    public void FillNa_BackwardFill_FillsNaN()
    {
        var col = MakeNaNColumn("val", double.NaN, 2.0, 3.0);
        var filled = col.FillNa(FillStrategy.Backward);

        // NaN should be backward-filled with 2.0
        Assert.Equal(2.0, filled[0]!.Value, 5);
    }

    [Fact]
    public void IsNa_DetectsNaN_InDoubleColumn()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0);
        var isNa = col.IsNa();

        Assert.False(isNa[0]);
        Assert.True(isNa[1], "NaN should be detected as NA");
        Assert.False(isNa[2]);
    }

    [Fact]
    public void Interpolate_TreatsNaN_AsMissing()
    {
        var col = MakeNaNColumn("val", 1.0, double.NaN, 3.0);
        var interpolated = col.Interpolate();

        // NaN should be interpolated: (1.0 + 3.0) / 2 = 2.0
        // The interpolated column may use null bitmask or NaN-free buffer
        var rawVal = interpolated.Buffer.Span[1];
        var nullableVal = interpolated[1];
        // Either it's null (and bitmask-based) or the value should be 2.0
        if (nullableVal.HasValue)
            Assert.Equal(2.0, nullableVal.Value, 5);
        else
            Assert.Equal(2.0, rawVal, 5);
    }

    [Fact]
    public void DropNa_DropsNaN_Rows()
    {
        var df = new DataFrame(
            new Column<double>("val", [1.0, double.NaN, 3.0])
        );
        var result = df.DropNa();

        // NaN row should be dropped
        Assert.Equal(2, result.RowCount);
    }

    // ============================================================
    // Area 6: GroupBy Min/Max NaN edge case
    // ============================================================

    [Fact]
    public void GroupBy_Min_TypedFastPath_SkipsNaN()
    {
        var df = new DataFrame(
            new Column<double>("val", [double.NaN, 2.0, double.NaN]),
            new StringColumn("grp", ["A", "A", "A"])
        );
        var result = df.GroupBy("grp").Min();
        var valCol = (Column<double>)result["val"];

        // Min should skip NaN and return 2.0
        Assert.Equal(2.0, valCol.Buffer.Span[0], 5);
    }

    [Fact]
    public void GroupBy_Max_TypedFastPath_SkipsNaN()
    {
        var df = new DataFrame(
            new Column<double>("val", [double.NaN, 2.0, double.NaN]),
            new StringColumn("grp", ["A", "A", "A"])
        );
        var result = df.GroupBy("grp").Max();
        var valCol = (Column<double>)result["val"];

        // Max should skip NaN and return 2.0
        Assert.Equal(2.0, valCol.Buffer.Span[0], 5);
    }
}
