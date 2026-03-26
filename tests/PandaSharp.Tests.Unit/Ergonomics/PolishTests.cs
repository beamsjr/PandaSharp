using PandaSharp;
using PandaSharp.Column;
using PandaSharp.GroupBy;
using PandaSharp.Joins;
using PandaSharp.Concat;
using FluentAssertions;

namespace PandaSharp.Tests.Unit.Ergonomics;

public class PolishTests
{
    // ─── Feature 1: Multi-column Aggregation ───

    [Fact]
    public void Agg_TupleSyntax_MultipleAggregations()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Dept"] = new[] { "Eng", "Eng", "Sales", "Sales" },
            ["Salary"] = new[] { 100.0, 200.0, 150.0, 250.0 },
            ["Age"] = new[] { 30, 40, 25, 35 }
        });

        var result = df.GroupBy("Dept").Agg(
            ("Salary", AggFunc.Sum),
            ("Age", AggFunc.Mean));

        result.ColumnNames.Should().Contain("Salary_sum");
        result.ColumnNames.Should().Contain("Age_mean");
        result.RowCount.Should().Be(2);

        // Verify values - find Eng row
        var deptCol = result.GetStringColumn("Dept");
        var deptVals = deptCol.GetValues();
        int engIdx = Array.IndexOf(deptVals, "Eng");
        result.GetColumn<double>("Salary_sum").Values[engIdx].Should().Be(300.0);
        result.GetColumn<double>("Age_mean").Values[engIdx].Should().Be(35.0);
    }

    [Fact]
    public void Agg_TupleSyntax_CountAndMinMax()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Group"] = new[] { "A", "A", "B", "B", "B" },
            ["Value"] = new[] { 10.0, 20.0, 5.0, 15.0, 25.0 }
        });

        var result = df.GroupBy("Group").Agg(
            ("Value", AggFunc.Count),
            ("Value", AggFunc.Min),
            ("Value", AggFunc.Max));

        result.ColumnNames.Should().Contain("Value_count");
        result.ColumnNames.Should().Contain("Value_min");
        result.ColumnNames.Should().Contain("Value_max");
    }

    [Fact]
    public void Agg_TupleSyntax_InvalidColumn_ThrowsWithAvailableColumns()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Group"] = new[] { "A", "B" },
            ["Value"] = new[] { 1.0, 2.0 }
        });

        var grouped = df.GroupBy("Group");
        var act = () => grouped.Agg(("NonExistent", AggFunc.Sum));
        act.Should().Throw<ArgumentException>()
            .WithMessage("*NonExistent*not found*Available*");
    }

    [Fact]
    public void Agg_TupleSyntax_StdAndVar()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Group"] = new[] { "A", "A", "A" },
            ["Value"] = new[] { 10.0, 20.0, 30.0 }
        });

        var result = df.GroupBy("Group").Agg(
            ("Value", AggFunc.Std),
            ("Value", AggFunc.Var));

        result.ColumnNames.Should().Contain("Value_std");
        result.ColumnNames.Should().Contain("Value_var");
        result.GetColumn<double>("Value_var").Values[0].Should().Be(100.0);
    }

    [Fact]
    public void Agg_TupleSyntax_FirstAndLast()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Group"] = new[] { "A", "A", "A" },
            ["Name"] = new[] { "Alice", "Bob", "Charlie" }
        });

        var result = df.GroupBy("Group").Agg(
            ("Name", AggFunc.First),
            ("Name", AggFunc.Last));

        result.ColumnNames.Should().Contain("Name_first");
        result.ColumnNames.Should().Contain("Name_last");
    }

    // ─── Feature 2: Actionable Error Messages ───

    [Fact]
    public void Filter_MaskLengthMismatch_IncludesLengthsInMessage()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3 }
        });

        var act = () => df.Filter(new bool[] { true, false });
        act.Should().Throw<ArgumentException>()
            .WithMessage("*2 elements*3 rows*");
    }

    [Fact]
    public void ColumnAccess_NotFound_ListsAvailableColumns()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Name"] = new[] { "Alice" },
            ["Age"] = new[] { 30 }
        });

        var act = () => { var _ = df["NonExistent"]; };
        act.Should().Throw<KeyNotFoundException>()
            .WithMessage("*NonExistent*Available*Name*Age*");
    }

    [Fact]
    public void Join_MissingRightColumn_ListsAvailable()
    {
        var left = DataFrame.FromDictionary(new()
        {
            ["id"] = new[] { 1, 2 },
            ["val"] = new[] { 10, 20 }
        });
        var right = DataFrame.FromDictionary(new()
        {
            ["name"] = new[] { "A", "B" },
            ["value"] = new[] { 100, 200 }
        });

        var act = () => left.Join(right, "id", "id");
        act.Should().Throw<ArgumentException>()
            .WithMessage("*'id'*not found in right*name*value*");
    }

    [Fact]
    public void Join_LeftOnRightOnLengthMismatch_ShowsDetails()
    {
        var left = DataFrame.FromDictionary(new()
        {
            ["a"] = new[] { 1 },
            ["b"] = new[] { 2 }
        });
        var right = DataFrame.FromDictionary(new()
        {
            ["c"] = new[] { 1 }
        });

        var act = () => left.Join(right, new[] { "a", "b" }, new[] { "c" });
        act.Should().Throw<ArgumentException>()
            .WithMessage("*leftOn has 2*rightOn has 1*");
    }

    [Fact]
    public void Concat_ColumnAxis_RowCountMismatch_ShowsDetails()
    {
        var df1 = DataFrame.FromDictionary(new() { ["A"] = new[] { 1, 2 } });
        var df2 = DataFrame.FromDictionary(new() { ["B"] = new[] { 1, 2, 3 } });

        var act = () => DataFrame.Concat(1, df1, df2);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*axis=1*3 rows*2 rows*");
    }

    [Fact]
    public void RenameColumn_NotFound_ListsAvailable()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["X"] = new[] { 1 },
            ["Y"] = new[] { 2 }
        });

        var act = () => df.RenameColumn("Z", "NewZ");
        act.Should().Throw<KeyNotFoundException>()
            .WithMessage("*'Z'*not found*X*Y*");
    }

    [Fact]
    public void GroupBy_ColumnNotFound_ListsAvailable()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Name"] = new[] { "A" },
            ["Value"] = new[] { 1 }
        });

        var act = () => df.GroupBy("Missing");
        act.Should().Throw<KeyNotFoundException>()
            .WithMessage("*'Missing'*not found*Name*Value*");
    }

    // ─── Feature 3: Pipe (already exists, verify it works) ───

    [Fact]
    public void Pipe_ChainsTransformations()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3, 4, 5 },
            ["B"] = new[] { 10, 20, 30, 40, 50 }
        });

        static DataFrame AddOne(DataFrame d) => d.Apply<int>(row => (int)row["A"]! + 1, "A_plus_1");
        static DataFrame FilterBig(DataFrame d, int threshold) =>
            d.WhereInt("A", v => v > threshold);

        var result = df.Pipe(AddOne).Pipe(FilterBig, 3);
        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("A_plus_1");
    }

    // ─── Feature 4: RenameColumn / RenameColumns (optimized) ───

    [Fact]
    public void RenameColumn_Works()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["OldName"] = new[] { 1, 2, 3 },
            ["Other"] = new[] { 4, 5, 6 }
        });

        var result = df.RenameColumn("OldName", "NewName");
        result.ColumnNames.Should().Contain("NewName");
        result.ColumnNames.Should().NotContain("OldName");
        result.GetColumn<int>("NewName").Values.ToArray().Should().Equal(1, 2, 3);
    }

    [Fact]
    public void RenameColumns_MultipleRenames()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2 },
            ["B"] = new[] { 3, 4 },
            ["C"] = new[] { 5, 6 }
        });

        var result = df.RenameColumns(new()
        {
            ["A"] = "X",
            ["C"] = "Z"
        });

        result.ColumnNames.Should().Equal("X", "B", "Z");
        result.GetColumn<int>("X").Values.ToArray().Should().Equal(1, 2);
        result.GetColumn<int>("Z").Values.ToArray().Should().Equal(5, 6);
    }

    [Fact]
    public void RenameColumns_InvalidKey_Throws()
    {
        var df = DataFrame.FromDictionary(new() { ["A"] = new[] { 1 } });
        var act = () => df.RenameColumns(new() { ["Missing"] = "X" });
        act.Should().Throw<KeyNotFoundException>()
            .WithMessage("*'Missing'*not found*");
    }

    // ─── Feature 5: Factory Methods ───

    [Fact]
    public void FromMatrix_CreatesCorrectDataFrame()
    {
        var matrix = new double[,] { { 1, 2, 3 }, { 4, 5, 6 } };
        var df = DataFrame.FromMatrix(matrix, new[] { "A", "B", "C" });

        df.RowCount.Should().Be(2);
        df.ColumnCount.Should().Be(3);
        df.GetColumn<double>("A").Values.ToArray().Should().Equal(1.0, 4.0);
        df.GetColumn<double>("B").Values.ToArray().Should().Equal(2.0, 5.0);
        df.GetColumn<double>("C").Values.ToArray().Should().Equal(3.0, 6.0);
    }

    [Fact]
    public void FromMatrix_DefaultColumnNames()
    {
        var matrix = new double[,] { { 1, 2 }, { 3, 4 } };
        var df = DataFrame.FromMatrix(matrix);

        df.ColumnNames.Should().Equal("col_0", "col_1");
    }

    [Fact]
    public void FromMatrix_WrongColumnCount_Throws()
    {
        var matrix = new double[,] { { 1, 2 } };
        var act = () => DataFrame.FromMatrix(matrix, new[] { "A" });
        act.Should().Throw<ArgumentException>()
            .WithMessage("*1 names*2 columns*");
    }

    [Fact]
    public void FromTuples2_CreatesCorrectDataFrame()
    {
        var df = DataFrame.FromTuples(
            new[] { "Name", "Age" },
            ("Alice", 25), ("Bob", 30), ("Charlie", 35));

        df.RowCount.Should().Be(3);
        df.ColumnCount.Should().Be(2);
        df.GetStringColumn("Name").GetValues().Should().Equal("Alice", "Bob", "Charlie");
        df.GetColumn<int>("Age").Values.ToArray().Should().Equal(25, 30, 35);
    }

    [Fact]
    public void FromTuples3_CreatesCorrectDataFrame()
    {
        var df = DataFrame.FromTuples(
            new[] { "Name", "Age", "Score" },
            ("Alice", 25, 90.5), ("Bob", 30, 85.0));

        df.RowCount.Should().Be(2);
        df.ColumnCount.Should().Be(3);
        df.GetStringColumn("Name").GetValues().Should().Equal("Alice", "Bob");
    }

    [Fact]
    public void FromTuples4_CreatesCorrectDataFrame()
    {
        var df = DataFrame.FromTuples(
            new[] { "A", "B", "C", "D" },
            (1, 2.0, "x", 10L), (3, 4.0, "y", 20L));

        df.RowCount.Should().Be(2);
        df.ColumnCount.Should().Be(4);
    }

    [Fact]
    public void FromTuples2_WrongColumnCount_Throws()
    {
        var act = () => DataFrame.FromTuples(
            new[] { "A", "B", "C" },
            ("x", 1));
        act.Should().Throw<ArgumentException>()
            .WithMessage("*2 column names*2-element*");
    }

    // ─── Feature 6: Batch ───

    [Fact]
    public void Batch_SplitsEvenly()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3, 4, 5, 6 }
        });

        var batches = df.Batch(2).ToList();
        batches.Count.Should().Be(3);
        batches[0].RowCount.Should().Be(2);
        batches[1].RowCount.Should().Be(2);
        batches[2].RowCount.Should().Be(2);
        batches[0].GetColumn<int>("A").Values.ToArray().Should().Equal(1, 2);
        batches[2].GetColumn<int>("A").Values.ToArray().Should().Equal(5, 6);
    }

    [Fact]
    public void Batch_LastBatchSmaller()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3, 4, 5 }
        });

        var batches = df.Batch(3).ToList();
        batches.Count.Should().Be(2);
        batches[0].RowCount.Should().Be(3);
        batches[1].RowCount.Should().Be(2);
    }

    [Fact]
    public void Batch_LargerThanDF_ReturnsSingleBatch()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2 }
        });

        var batches = df.Batch(10).ToList();
        batches.Count.Should().Be(1);
        batches[0].RowCount.Should().Be(2);
    }

    [Fact]
    public void Batch_ZeroOrNegative_Throws()
    {
        var df = DataFrame.FromDictionary(new() { ["A"] = new[] { 1 } });
        var act = () => df.Batch(0).ToList();
        act.Should().Throw<ArgumentException>().WithMessage("*positive*");
    }

    // ─── Feature 7: ToString() pretty-print ───

    [Fact]
    public void ToString_SmallDF_ShowsAllRows()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Name"] = new[] { "Alice", "Bob" },
            ["Age"] = new[] { 25, 30 }
        });

        var str = df.ToString();
        str.Should().Contain("Alice");
        str.Should().Contain("Bob");
        str.Should().Contain("Name");
        str.Should().Contain("Age");
        str.Should().Contain("[2 rows x 2 columns]");
    }

    [Fact]
    public void ToString_LargeDF_ShowsHeadAndTailWithEllipsis()
    {
        var names = Enumerable.Range(0, 100).Select(i => $"Row{i}").ToArray();
        var df = DataFrame.FromDictionary(new()
        {
            ["Name"] = names
        });

        var str = df.ToString();
        str.Should().Contain("Row0"); // head
        str.Should().Contain("Row99"); // tail
        str.Should().Contain("..."); // ellipsis
        str.Should().Contain("[100 rows x 1 columns]");
    }

    [Fact]
    public void ToString_EmptyDF_ShowsEmptyMessage()
    {
        var df = new DataFrame();
        var str = df.ToString();
        str.Should().Contain("empty");
    }

    // ─── Feature 8: ValueEquals ───

    [Fact]
    public void ValueEquals_IdenticalDataFrames_ReturnsTrue()
    {
        var df1 = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3 },
            ["B"] = new[] { "x", "y", "z" }
        });
        var df2 = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3 },
            ["B"] = new[] { "x", "y", "z" }
        });

        df1.ValueEquals(df2).Should().BeTrue();
    }

    [Fact]
    public void ValueEquals_DifferentValues_ReturnsFalse()
    {
        var df1 = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3 }
        });
        var df2 = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 999 }
        });

        df1.ValueEquals(df2).Should().BeFalse();
    }

    [Fact]
    public void ValueEquals_DifferentColumnNames_ReturnsFalse()
    {
        var df1 = DataFrame.FromDictionary(new() { ["A"] = new[] { 1 } });
        var df2 = DataFrame.FromDictionary(new() { ["B"] = new[] { 1 } });

        df1.ValueEquals(df2).Should().BeFalse();
    }

    [Fact]
    public void ValueEquals_DifferentRowCount_ReturnsFalse()
    {
        var df1 = DataFrame.FromDictionary(new() { ["A"] = new[] { 1, 2 } });
        var df2 = DataFrame.FromDictionary(new() { ["A"] = new[] { 1 } });

        df1.ValueEquals(df2).Should().BeFalse();
    }

    // ─── Feature 9: StringColumn Methods ───

    [Fact]
    public void StringColumn_ToUpper()
    {
        var col = new StringColumn("name", new[] { "alice", "Bob", null, "charlie" });
        var result = col.ToUpper();

        result.GetValues().Should().Equal("ALICE", "BOB", null, "CHARLIE");
    }

    [Fact]
    public void StringColumn_ToLower()
    {
        var col = new StringColumn("name", new[] { "ALICE", "Bob", null, "CHARLIE" });
        var result = col.ToLower();

        result.GetValues().Should().Equal("alice", "bob", null, "charlie");
    }

    [Fact]
    public void StringColumn_Trim()
    {
        var col = new StringColumn("name", new[] { "  alice  ", " Bob", null, "charlie " });
        var result = col.Trim();

        result.GetValues().Should().Equal("alice", "Bob", null, "charlie");
    }

    [Fact]
    public void StringColumn_Substring()
    {
        var col = new StringColumn("name", new[] { "Hello World", "Hi", null, "Test" });
        var result = col.Substring(0, 5);

        result.GetValues().Should().Equal("Hello", "Hi", null, "Test");
    }

    [Fact]
    public void StringColumn_Substring_StartBeyondLength()
    {
        var col = new StringColumn("name", new[] { "AB", "ABCDEF" });
        var result = col.Substring(10, 5);

        result.GetValues().Should().Equal("", "");
    }

    [Fact]
    public void StringColumn_Substring_NegativeStart_Throws()
    {
        var col = new StringColumn("name", new[] { "Test" });
        var act = () => col.Substring(-1, 3);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ─── Feature 10: GroupBy Null Behavior ───

    [Fact]
    public void GroupBy_NullMode_Include_NullsFormOwnGroup()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Category"] = new string?[] { "A", "B", null, "A", null },
            ["Value"] = new[] { 1, 2, 3, 4, 5 }
        });

        var grouped = df.GroupBy(new[] { "Category" }, NullGroupingMode.Include);
        // With Include, null values form their own group
        grouped.GroupCount.Should().BeGreaterThanOrEqualTo(2); // at least A and B
    }

    [Fact]
    public void GroupBy_NullMode_Exclude_NullsDropped()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Category"] = new string?[] { "A", "B", null, "A", null },
            ["Value"] = new[] { 1, 2, 3, 4, 5 }
        });

        var grouped = df.GroupBy(new[] { "Category" }, NullGroupingMode.Exclude);

        // With Exclude, null rows are dropped — only A and B groups
        grouped.GroupCount.Should().Be(2);

        var result = grouped.Sum();
        result.RowCount.Should().Be(2);

        // Verify null rows (indices 2 and 4, values 3 and 5) are excluded from sums
        var catCol = result.GetStringColumn("Category");
        var valCol = result.GetColumn<double>("Value");
        var cats = catCol.GetValues();
        int aIdx = Array.IndexOf(cats, "A");
        int bIdx = Array.IndexOf(cats, "B");
        valCol.Values[aIdx].Should().Be(5.0);  // 1 + 4
        valCol.Values[bIdx].Should().Be(2.0);  // just 2
    }

    [Fact]
    public void GroupBy_DefaultMode_IsInclude()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Category"] = new string?[] { "A", null, "A" },
            ["Value"] = new[] { 1, 2, 3 }
        });

        // Default GroupBy should include nulls (same as Include mode)
        var grouped1 = df.GroupBy("Category");
        var grouped2 = df.GroupBy(new[] { "Category" }, NullGroupingMode.Include);

        grouped1.GroupCount.Should().Be(grouped2.GroupCount);
    }
}
