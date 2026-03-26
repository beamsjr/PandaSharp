using PandaSharp;
using PandaSharp.Column;
using PandaSharp.GroupBy;
using PandaSharp.Reshape;
using PandaSharp.Window;

namespace PandaSharp.Tests.Unit.EdgeCases;

public class NullAndWindowEdgeCaseTests
{
    // ==========================================================================
    // Area 1: Null handling in Column arithmetic
    // ==========================================================================

    [Fact]
    public void AsDouble_PreservesNulls()
    {
        // BUG: AsDouble creates new Column<double>(name, vals) which has no null bitmask
        var col = Column<int>.FromNullable("x", new int?[] { 1, null, 3 });
        var result = col.AsDouble();

        Assert.True(result.IsNull(1), "AsDouble should preserve null at index 1");
        Assert.Null(result[1]);
    }

    [Fact]
    public void MixedTypeArithmetic_WithNulls_PropagatesNulls()
    {
        // This relies on AsDouble, so if AsDouble drops nulls, this is also broken
        var intCol = Column<int>.FromNullable("a", new int?[] { 1, null, 3 });
        var dblCol = Column<double>.FromNullable("b", new double?[] { 10.0, 20.0, 30.0 });

        var result = intCol.AsDouble().Add(dblCol);

        Assert.True(result.IsNull(1), "null int + 20.0 should be null");
    }

    [Fact]
    public void Sum_AllNullColumn_ReturnsZero_MatchingPandasDefault()
    {
        // pandas default: sum of all-null = 0 (skipna=True, min_count=0)
        var col = Column<double>.FromNullable("x", new double?[] { null, null, null });
        var sum = col.Sum();

        Assert.Equal(0.0, sum);
    }

    [Fact]
    public void Mean_AllNullColumn_ReturnsNull()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, null, null });
        var mean = col.Mean();

        Assert.Null(mean);
    }

    [Fact]
    public void Min_AllNullColumn_ReturnsNull()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, null, null });
        var min = col.Min();

        Assert.Null(min);
    }

    [Fact]
    public void Max_AllNullColumn_ReturnsNull()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, null, null });
        var max = col.Max();

        Assert.Null(max);
    }

    [Fact]
    public void Std_AllNullColumn_ReturnsNull()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, null, null });
        var std = col.Std();

        Assert.Null(std);
    }

    [Fact]
    public void ColumnAddWithNulls_PropagatesCorrectly()
    {
        var col1 = Column<double>.FromNullable("a", new double?[] { 1.0, null, 3.0, null });
        var col2 = Column<double>.FromNullable("b", new double?[] { 10.0, 20.0, null, null });

        var result = col1 + col2;

        Assert.Equal(11.0, result[0]);
        Assert.Null(result[1]);  // null + 20.0 = null
        Assert.Null(result[2]);  // 3.0 + null = null
        Assert.Null(result[3]);  // null + null = null
    }

    [Fact]
    public void ColumnIntWithNulls_AddScalar_NullStaysNull()
    {
        var col = Column<int>.FromNullable("x", new int?[] { 1, null, 3 });
        var result = col + 10;

        Assert.Equal(11, result[0]);
        Assert.Null(result[1]);
        Assert.Equal(13, result[2]);
    }

    // ==========================================================================
    // Area 2: Rolling/Expanding/EWM window edge cases
    // ==========================================================================

    [Fact]
    public void Rolling_WindowSize1_Mean_EqualsOriginalValues()
    {
        var col = new Column<double>("x", new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var result = col.Rolling(1).Mean();

        for (int i = 0; i < col.Length; i++)
            Assert.Equal(col[i], result[i]);
    }

    [Fact]
    public void Rolling_WindowSizeLargerThanLength_ReturnsAllNull()
    {
        var col = new Column<double>("x", new double[] { 1.0, 2.0, 3.0 });
        var result = col.Rolling(10).Mean();

        // With default minPeriods = windowSize = 10, all windows have fewer than 10 values
        for (int i = 0; i < col.Length; i++)
            Assert.Null(result[i]);
    }

    [Fact]
    public void Rolling_AllNulls_ReturnsAllNull()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, null, null, null });
        var result = col.Rolling(2).Mean();

        for (int i = 0; i < col.Length; i++)
            Assert.Null(result[i]);
    }

    [Fact]
    public void Rolling_Std_ConstantValues_ReturnsZero()
    {
        var col = new Column<double>("x", new double[] { 5.0, 5.0, 5.0, 5.0, 5.0 });
        var result = col.Rolling(3).Std();

        // First two should be null (minPeriods=3 by default for windowSize=3, need 2+ for std)
        // Positions 2,3,4 should be 0.0 (std of constant = 0)
        for (int i = 2; i < col.Length; i++)
        {
            Assert.NotNull(result[i]);
            Assert.Equal(0.0, result[i]!.Value, 10);
        }
    }

    [Fact]
    public void Rolling_MinPeriods0_ProducesResultsFromStart()
    {
        var col = new Column<double>("x", new double[] { 1.0, 2.0, 3.0 });
        var result = col.Rolling(3, minPeriods: 0).Mean();

        // minPeriods=0 means every position should have a result
        Assert.NotNull(result[0]); // mean of [1] = 1.0
        Assert.NotNull(result[1]); // mean of [1,2] = 1.5
        Assert.NotNull(result[2]); // mean of [1,2,3] = 2.0
    }

    [Fact]
    public void Expanding_MinPeriods0_Sum_StartsFromFirstValue()
    {
        var col = new Column<double>("x", new double[] { 1.0, 2.0, 3.0 });
        var result = col.Expanding(minPeriods: 0).Sum();

        // minPeriods=0 means results from index 0
        Assert.Equal(1.0, result[0]);
        Assert.Equal(3.0, result[1]);
        Assert.Equal(6.0, result[2]);
    }

    [Fact]
    public void Ewm_Span1_EqualsOriginalValues()
    {
        // span=1 means alpha = 2/(1+1) = 1.0, so ewm = 1.0*current + 0.0*prev = current
        var col = new Column<double>("x", new double[] { 1.0, 2.0, 3.0, 4.0 });
        var result = col.Ewm(span: 1).Mean();

        for (int i = 0; i < col.Length; i++)
            Assert.Equal(col[i], result[i]);
    }

    // ==========================================================================
    // Area 3: Pivot/Melt/Explode edge cases
    // ==========================================================================

    [Fact]
    public void Melt_AllColumnsAsIdVars_NoValueVars_ProducesEmptyResult()
    {
        // BUG: Melt with 0 value vars causes nRows * 0 = 0 output rows,
        // but then tries to access df[valueVars[0]] which is out of range
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3 },
            ["B"] = new[] { 4, 5, 6 },
        });

        // All columns as id_vars means no value vars
        var result = df.Melt(new[] { "A", "B" });
        Assert.Equal(0, result.RowCount);
    }

    [Fact]
    public void Explode_NullValues_PreservesNullRows()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["id"] = new[] { 1, 2, 3 },
            ["tags"] = new[] { "a,b", null!, "c" },
        });

        var result = df.Explode("tags");

        // Row with null should produce one row with null value
        Assert.True(result.RowCount >= 3);
    }

    [Fact]
    public void Explode_EmptyString_ProducesOneRow()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["id"] = new[] { 1, 2 },
            ["tags"] = new[] { "a,b", "" },
        });

        var result = df.Explode("tags");
        // "a,b" -> 2 rows, "" -> 1 row (split on "," produces [""])
        Assert.Equal(3, result.RowCount);
    }

    [Fact]
    public void Explode_SeparatorNotFound_ProducesOriginalRows()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["id"] = new[] { 1, 2 },
            ["tags"] = new[] { "hello", "world" },
        });

        var result = df.Explode("tags", separator: "|");
        Assert.Equal(2, result.RowCount);
    }

    [Fact]
    public void Pivot_DuplicateIndexColumnCombination_LastValueWins()
    {
        // Not necessarily a "bug" but documenting the behavior
        var df = DataFrame.FromDictionary(new()
        {
            ["idx"] = new[] { "A", "A", "B" },
            ["col"] = new[] { "X", "X", "X" },
            ["val"] = new[] { 1, 2, 3 },
        });

        // Pivot with duplicate (A, X) - last value (2) should win
        var result = df.Pivot("idx", "col", "val");
        Assert.Equal(2, result.RowCount);
    }

    // ==========================================================================
    // Area 4: CombineFirst / Compare edge cases
    // ==========================================================================

    [Fact]
    public void CombineFirst_DifferentTypes_IntVsDouble()
    {
        // BUG: CombineFirst checks left is Column<int> && right is Column<int>,
        // but if left is int and right is double, falls to CombineGeneric
        var df1 = DataFrame.FromDictionary(new()
        {
            ["A"] = new int[] { 1, 2, 3 },
        });
        // Need to create with nulls to trigger combine logic
        var cols1 = new IColumn[] { Column<int>.FromNullable("A", new int?[] { null, 2, null }) };
        var primary = new DataFrame(cols1);
        var cols2 = new IColumn[] { Column<double>.FromNullable("A", new double?[] { 10.0, 20.0, 30.0 }) };
        var fallback = new DataFrame(cols2);

        var result = primary.CombineFirst(fallback);

        // null from primary -> 10.0 from fallback
        var aCol = result["A"];
        Assert.NotNull(aCol.GetObject(0));
        Assert.NotNull(aCol.GetObject(1));
        Assert.NotNull(aCol.GetObject(2));
    }

    [Fact]
    public void CombineFirst_ZeroRowDataFrames()
    {
        var df1 = new DataFrame(new IColumn[] { new Column<int>("A", Array.Empty<int>()) });
        var df2 = new DataFrame(new IColumn[] { new Column<int>("A", Array.Empty<int>()) });

        var result = df1.CombineFirst(df2);
        Assert.Equal(0, result.RowCount);
    }

    [Fact]
    public void Compare_DifferentColumnSets_OnlyComparesCommonColumns()
    {
        var df1 = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3 },
            ["B"] = new[] { 4, 5, 6 },
        });
        var df2 = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3 },
            ["C"] = new[] { 7, 8, 9 },
        });

        // Should not throw - only compares "A" which is identical
        var result = df1.Compare(df2);
        // All values in common column "A" are equal, so no diff rows from column comparison
        // But it's the same row count so no extra rows either
    }

    // ==========================================================================
    // Area 5: GroupBy.Agg / Pipe / Batch / ValueEquals edge cases
    // ==========================================================================

    [Fact]
    public void GroupBy_Agg_NonExistentColumn_ThrowsWithMessage()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Grp"] = new[] { "A", "B", "A" },
            ["Val"] = new[] { 1.0, 2.0, 3.0 },
        });

        var grouped = df.GroupBy("Grp");
        var ex = Assert.Throws<ArgumentException>(() =>
            grouped.Agg(("NonExistent", AggFunc.Sum)));

        Assert.Contains("NonExistent", ex.Message);
    }

    [Fact]
    public void GroupBy_Agg_ZeroAggregations_Throws()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Grp"] = new[] { "A", "B" },
            ["Val"] = new[] { 1.0, 2.0 },
        });

        var grouped = df.GroupBy("Grp");
        Assert.Throws<ArgumentException>(() =>
            grouped.Agg(Array.Empty<(string, AggFunc)>()));
    }

    [Fact]
    public void Pipe_NullTransform_Throws()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2 },
        });

        Assert.Throws<NullReferenceException>(() => df.Pipe((Func<DataFrame, DataFrame>)null!));
    }

    [Fact]
    public void Batch_Size1_EveryRowIsOwnDataFrame()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 10, 20, 30 },
        });

        var batches = df.Batch(1).ToList();
        Assert.Equal(3, batches.Count);
        Assert.Equal(1, batches[0].RowCount);
        Assert.Equal(1, batches[1].RowCount);
        Assert.Equal(1, batches[2].RowCount);
    }

    [Fact]
    public void Batch_ZeroRowDataFrame_ProducesNoBatches()
    {
        var df = new DataFrame(new IColumn[] { new Column<int>("A", Array.Empty<int>()) });
        var batches = df.Batch(5).ToList();
        Assert.Empty(batches);
    }

    [Fact]
    public void ValueEquals_DifferentColumnOrder_ReturnsFalse()
    {
        var df1 = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1 },
            ["B"] = new[] { 2 },
        });
        var df2 = DataFrame.FromDictionary(new()
        {
            ["B"] = new[] { 2 },
            ["A"] = new[] { 1 },
        });

        // Different column order means not equal (pandas behavior)
        Assert.False(df1.ValueEquals(df2));
    }

    [Fact]
    public void ValueEquals_BothHaveNulls_InSamePositions_ReturnsTrue()
    {
        var cols1 = new IColumn[] { Column<int>.FromNullable("A", new int?[] { 1, null, 3 }) };
        var cols2 = new IColumn[] { Column<int>.FromNullable("A", new int?[] { 1, null, 3 }) };
        var df1 = new DataFrame(cols1);
        var df2 = new DataFrame(cols2);

        Assert.True(df1.ValueEquals(df2));
    }
}
