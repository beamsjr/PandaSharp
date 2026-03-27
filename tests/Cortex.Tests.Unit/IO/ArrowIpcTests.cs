using FluentAssertions;
using Cortex.Column;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class ArrowIpcTests
{
    [Fact]
    public void ArrowIpc_RoundTrip_IntAndString()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
            new Column<int>("Age", [25, 30, 35])
        );

        using var ms = new MemoryStream();
        ArrowIpcWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = ArrowIpcReader.Read(ms);

        df2.RowCount.Should().Be(3);
        df2.ColumnNames.Should().Equal(["Name", "Age"]);
        df2.GetStringColumn("Name")[0].Should().Be("Alice");
        df2.GetColumn<int>("Age")[2].Should().Be(35);
    }

    [Fact]
    public void ArrowIpc_RoundTrip_Double()
    {
        var df = new DataFrame(
            new Column<double>("Value", [1.5, 2.7, 3.14])
        );

        using var ms = new MemoryStream();
        ArrowIpcWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = ArrowIpcReader.Read(ms);

        df2.GetColumn<double>("Value")[0].Should().Be(1.5);
        df2.GetColumn<double>("Value")[2].Should().Be(3.14);
    }

    [Fact]
    public void ArrowIpc_RoundTrip_WithNulls()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3]),
            new StringColumn("B", ["x", null, "z"])
        );

        using var ms = new MemoryStream();
        ArrowIpcWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = ArrowIpcReader.Read(ms);

        df2["A"].IsNull(1).Should().BeTrue();
        df2["B"].IsNull(1).Should().BeTrue();
        df2.GetColumn<int>("A")[0].Should().Be(1);
        df2.GetStringColumn("B")[2].Should().Be("z");
    }

    [Fact]
    public void ArrowIpc_RoundTrip_Bool()
    {
        var df = new DataFrame(
            new Column<bool>("Flag", [true, false, true])
        );

        using var ms = new MemoryStream();
        ArrowIpcWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = ArrowIpcReader.Read(ms);

        df2.GetColumn<bool>("Flag")[0].Should().Be(true);
        df2.GetColumn<bool>("Flag")[1].Should().Be(false);
    }

    [Fact]
    public void ArrowIpc_RoundTrip_LargeDataFrame()
    {
        int n = 10_000;
        var ints = new int[n];
        var strings = new string?[n];
        for (int i = 0; i < n; i++)
        {
            ints[i] = i;
            strings[i] = $"row_{i}";
        }

        var df = new DataFrame(
            new Column<int>("Id", ints),
            new StringColumn("Label", strings)
        );

        using var ms = new MemoryStream();
        ArrowIpcWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = ArrowIpcReader.Read(ms);

        df2.RowCount.Should().Be(n);
        df2.GetColumn<int>("Id")[9999].Should().Be(9999);
    }
}
