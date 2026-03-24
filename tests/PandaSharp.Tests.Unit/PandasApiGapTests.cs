using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.IO;

namespace PandaSharp.Tests.Unit;

/// <summary>
/// Tests for pandas-equivalent API methods that fill coverage gaps.
/// </summary>
public class PandasApiGapTests
{
    // ===== Prod =====

    [Fact]
    public void Prod_IntColumn()
    {
        var col = new Column<int>("x", [2, 3, 4]);
        col.Prod().Should().Be(24); // 2*3*4
    }

    [Fact]
    public void Prod_DoubleColumn()
    {
        var col = new Column<double>("x", [1.5, 2.0, 3.0]);
        col.Prod().Should().Be(9.0); // 1.5*2*3
    }

    [Fact]
    public void Prod_WithNulls_SkipsNulls()
    {
        var col = Column<int>.FromNullable("x", [2, null, 5]);
        col.Prod().Should().Be(10); // 2*5
    }

    [Fact]
    public void Prod_EmptyColumn_ReturnsNull()
    {
        var col = new Column<int>("x", Array.Empty<int>());
        col.Prod().Should().BeNull();
    }

    // ===== CastColumn =====

    [Fact]
    public void CastColumn_IntToDouble()
    {
        var df = new DataFrame(new Column<int>("x", [1, 2, 3]));
        var result = df.CastColumn("x", typeof(double));

        result["x"].DataType.Should().Be(typeof(double));
        result.GetColumn<double>("x")[1].Should().Be(2.0);
    }

    [Fact]
    public void CastColumn_DoubleToInt()
    {
        var df = new DataFrame(new Column<double>("x", [1.9, 2.1, 3.5]));
        var result = df.CastColumn("x", typeof(int));

        result["x"].DataType.Should().Be(typeof(int));
        result.GetColumn<int>("x")[0].Should().Be(2); // rounded
    }

    [Fact]
    public void CastColumn_IntToString()
    {
        var df = new DataFrame(new Column<int>("x", [1, 2, 3]));
        var result = df.CastColumn("x", typeof(string));

        result["x"].DataType.Should().Be(typeof(string));
        result.GetStringColumn("x")[0].Should().Be("1");
    }

    [Fact]
    public void CastColumn_IntToLong()
    {
        var df = new DataFrame(new Column<int>("x", [1, 2, 3]));
        var result = df.CastColumn("x", typeof(long));

        result["x"].DataType.Should().Be(typeof(long));
        result.GetColumn<long>("x")[2].Should().Be(3L);
    }

    [Fact]
    public void CastColumn_PreservesOtherColumns()
    {
        var df = new DataFrame(
            new Column<int>("a", [1, 2]),
            new StringColumn("b", ["x", "y"])
        );
        var result = df.CastColumn("a", typeof(double));

        result.ColumnCount.Should().Be(2);
        result["b"].DataType.Should().Be(typeof(string));
    }

    // ===== Column<T>.Cast =====

    [Fact]
    public void ColumnCast_IntToDouble()
    {
        var col = new Column<int>("x", [1, 2, 3]);
        var result = col.Cast<int, double>();

        result.Name.Should().Be("x");
        result[1].Should().Be(2.0);
    }

    [Fact]
    public void ColumnCast_WithNulls()
    {
        var col = Column<int>.FromNullable("x", [1, null, 3]);
        var result = col.Cast<int, double>();

        result[0].Should().Be(1.0);
        result.IsNull(1).Should().BeTrue();
        result[2].Should().Be(3.0);
    }

    // ===== String padding =====

    [Fact]
    public void ZFill_PadsWithZeros()
    {
        var col = new StringColumn("x", ["42", "7", "123"]);
        var result = col.Str.ZFill(5);

        result[0].Should().Be("00042");
        result[1].Should().Be("00007");
        result[2].Should().Be("00123");
    }

    [Fact]
    public void Center_CentersString()
    {
        var col = new StringColumn("x", ["hi", "a"]);
        var result = col.Str.Center(6);

        result[0].Should().Be("  hi  ");
        result[1].Should().Be("  a   ");
    }

    [Fact]
    public void Center_WithFillChar()
    {
        var col = new StringColumn("x", ["hi"]);
        var result = col.Str.Center(6, '*');

        result[0].Should().Be("**hi**");
    }

    [Fact]
    public void LJust_LeftJustifies()
    {
        var col = new StringColumn("x", ["hi"]);
        var result = col.Str.LJust(5);
        result[0].Should().Be("hi   ");
    }

    [Fact]
    public void RJust_RightJustifies()
    {
        var col = new StringColumn("x", ["hi"]);
        var result = col.Str.RJust(5);
        result[0].Should().Be("   hi");
    }

    [Fact]
    public void ZFill_HandlesNulls()
    {
        var col = new StringColumn("x", ["42", null, "7"]);
        var result = col.Str.ZFill(4);

        result[0].Should().Be("0042");
        result[1].Should().BeNull();
        result[2].Should().Be("0007");
    }

    // ===== JSON Lines =====

    [Fact]
    public void JsonLines_ReadsMultipleObjects()
    {
        var jsonl = "{\"name\":\"Alice\",\"age\":25}\n{\"name\":\"Bob\",\"age\":30}\n{\"name\":\"Charlie\",\"age\":35}";
        var df = JsonReader.ReadLinesString(jsonl);

        df.RowCount.Should().Be(3);
        df.ColumnNames.Should().Contain("name");
        df.ColumnNames.Should().Contain("age");
        df.GetStringColumn("name")[0].Should().Be("Alice");
    }

    [Fact]
    public void JsonLines_HandlesBlankLines()
    {
        var jsonl = "{\"x\":1}\n\n{\"x\":2}\n";
        var df = JsonReader.ReadLinesString(jsonl);

        df.RowCount.Should().Be(2);
    }

    [Fact]
    public void JsonLines_EmptyInput()
    {
        var df = JsonReader.ReadLinesString("");
        df.RowCount.Should().Be(0);
    }

    [Fact]
    public void JsonLines_FileRoundTrip()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, "{\"id\":1,\"val\":\"a\"}\n{\"id\":2,\"val\":\"b\"}\n");

        var df = JsonReader.ReadLines(path);
        df.RowCount.Should().Be(2);
        df.GetStringColumn("val")[1].Should().Be("b");

        File.Delete(path);
    }
}
