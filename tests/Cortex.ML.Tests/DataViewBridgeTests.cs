using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using Cortex;
using Cortex.Column;
using Cortex.ML.MLNet;

namespace Cortex.ML.Tests;

public class DataViewBridgeTests
{
    private static readonly MLContext MlContext = new(seed: 42);

    [Fact]
    public void ToDataView_PreservesRowCount()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0]),
            new Column<int>("B", [10, 20, 30])
        );

        var dv = df.ToDataView(MlContext);

        dv.GetRowCount().Should().Be(3);
    }

    [Fact]
    public void ToDataView_PreservesColumnNames()
    {
        var df = new DataFrame(
            new Column<double>("Price", [1.5, 2.5]),
            new StringColumn("Name", ["A", "B"])
        );

        var dv = df.ToDataView(MlContext);

        dv.Schema.Select(c => c.Name).Should().Contain("Price");
        dv.Schema.Select(c => c.Name).Should().Contain("Name");
    }

    [Fact]
    public void ToDataView_NumericColumns_MapToCorrectTypes()
    {
        var df = new DataFrame(
            new Column<double>("X", [1.0, 2.0]),
            new Column<int>("Y", [10, 20])
        );

        var dv = df.ToDataView(MlContext);

        dv.Schema["X"].Type.Should().Be(NumberDataViewType.Double);
        dv.Schema["Y"].Type.Should().Be(NumberDataViewType.Int32);
    }

    [Fact]
    public void ToDataView_StringColumns_MapToText()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"])
        );

        var dv = df.ToDataView(MlContext);

        dv.Schema["Name"].Type.Should().BeOfType<TextDataViewType>();
    }

    [Fact]
    public void ToDataView_BoolColumns_MapToBoolean()
    {
        var df = new DataFrame(
            new Column<bool>("Flag", [true, false, true])
        );

        var dv = df.ToDataView(MlContext);

        dv.Schema["Flag"].Type.Should().Be(BooleanDataViewType.Instance);
    }

    [Fact]
    public void ToDataView_ReadBack_NumericValues()
    {
        var df = new DataFrame(
            new Column<double>("X", [1.5, 2.5, 3.5])
        );

        var dv = df.ToDataView(MlContext);
        using var cursor = dv.GetRowCursor(dv.Schema);
        var getter = cursor.GetGetter<double>(dv.Schema["X"]);

        var values = new List<double>();
        while (cursor.MoveNext())
        {
            double val = 0;
            getter(ref val);
            values.Add(val);
        }

        values.Should().HaveCount(3);
        values[0].Should().BeApproximately(1.5, 0.001);
        values[1].Should().BeApproximately(2.5, 0.001);
        values[2].Should().BeApproximately(3.5, 0.001);
    }

    [Fact]
    public void ToDataView_ReadBack_StringValues()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"])
        );

        var dv = df.ToDataView(MlContext);
        using var cursor = dv.GetRowCursor(dv.Schema);
        var getter = cursor.GetGetter<ReadOnlyMemory<char>>(dv.Schema["Name"]);

        var values = new List<string>();
        while (cursor.MoveNext())
        {
            var val = default(ReadOnlyMemory<char>);
            getter(ref val);
            values.Add(val.ToString());
        }

        values.Should().Equal(["Alice", "Bob"]);
    }

    [Fact]
    public void ToDataView_ReadBack_BoolValues()
    {
        var df = new DataFrame(
            new Column<bool>("Flag", [true, false])
        );

        var dv = df.ToDataView(MlContext);
        using var cursor = dv.GetRowCursor(dv.Schema);
        var getter = cursor.GetGetter<bool>(dv.Schema["Flag"]);

        var values = new List<bool>();
        while (cursor.MoveNext())
        {
            bool val = false;
            getter(ref val);
            values.Add(val);
        }

        values.Should().Equal([true, false]);
    }

    [Fact]
    public void RoundTrip_ToDataView_ToDataFrame()
    {
        var original = new DataFrame(
            new Column<double>("Price", [1.5, 2.5, 3.5]),
            new Column<int>("Qty", [10, 20, 30])
        );

        var dv = original.ToDataView(MlContext);
        var roundTripped = dv.ToDataFrame();

        roundTripped.RowCount.Should().Be(3);
        roundTripped.ColumnNames.Should().Contain("Price");
        roundTripped.ColumnNames.Should().Contain("Qty");

        // Values come back as double (via Single round-trip)
        var priceCol = (Column<double>)roundTripped["Price"];
        ((double?)priceCol[0]).Should().BeApproximately(1.5, 0.01);
        ((double?)priceCol[1]).Should().BeApproximately(2.5, 0.01);
    }

    [Fact]
    public void ToDataView_MixedColumns()
    {
        var df = new DataFrame(
            new Column<double>("Score", [0.9, 0.8]),
            new StringColumn("Label", ["cat", "dog"]),
            new Column<bool>("Active", [true, false])
        );

        var dv = df.ToDataView(MlContext);

        dv.Schema.Count.Should().Be(3);
        dv.GetRowCount().Should().Be(2);
    }

    [Fact]
    public void ToDataView_EmptyDataFrame()
    {
        var df = new DataFrame(
            new Column<double>("X", Array.Empty<double>())
        );

        var dv = df.ToDataView(MlContext);

        dv.GetRowCount().Should().Be(0);
        dv.Schema["X"].Type.Should().Be(NumberDataViewType.Double);
    }

    [Fact]
    public void ToDataView_IntColumn_ReadAsInt32()
    {
        var df = new DataFrame(
            new Column<int>("Count", [100, 200, 300])
        );

        var dv = df.ToDataView(MlContext);
        dv.Schema["Count"].Type.Should().Be(NumberDataViewType.Int32);

        using var cursor = dv.GetRowCursor(dv.Schema);
        var getter = cursor.GetGetter<int>(dv.Schema["Count"]);

        var values = new List<int>();
        while (cursor.MoveNext())
        {
            int val = 0;
            getter(ref val);
            values.Add(val);
        }

        values.Should().Equal([100, 200, 300]);
    }

    [Fact]
    public void ToDataView_DoubleColumn_ReadAsDouble()
    {
        var df = new DataFrame(
            new Column<double>("Value", [1.5, 2.5, 3.5])
        );

        var dv = df.ToDataView(MlContext);
        dv.Schema["Value"].Type.Should().Be(NumberDataViewType.Double);

        using var cursor = dv.GetRowCursor(dv.Schema);
        var getter = cursor.GetGetter<double>(dv.Schema["Value"]);

        var values = new List<double>();
        while (cursor.MoveNext())
        {
            double val = 0;
            getter(ref val);
            values.Add(val);
        }

        values.Should().Equal([1.5, 2.5, 3.5]);
    }

    [Fact]
    public void ToDataView_LongColumn_ReadAsInt64()
    {
        var df = new DataFrame(
            new Column<long>("BigId", [1_000_000_000L, 2_000_000_000L])
        );

        var dv = df.ToDataView(MlContext);
        dv.Schema["BigId"].Type.Should().Be(NumberDataViewType.Int64);

        using var cursor = dv.GetRowCursor(dv.Schema);
        var getter = cursor.GetGetter<long>(dv.Schema["BigId"]);

        var values = new List<long>();
        while (cursor.MoveNext())
        {
            long val = 0;
            getter(ref val);
            values.Add(val);
        }

        values.Should().Equal([1_000_000_000L, 2_000_000_000L]);
    }

    [Fact]
    public void RoundTrip_PreservesIntTypes()
    {
        var df = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new Column<double>("Score", [9.5, 8.0, 7.5]),
            new StringColumn("Name", ["A", "B", "C"])
        );

        var dv = df.ToDataView(MlContext);
        var roundTripped = dv.ToDataFrame();

        roundTripped.RowCount.Should().Be(3);
        roundTripped["Id"].DataType.Should().Be(typeof(int));
        roundTripped["Score"].DataType.Should().Be(typeof(double));
        roundTripped["Name"].DataType.Should().Be(typeof(string));
        roundTripped.GetColumn<int>("Id")[2].Should().Be(3);
        roundTripped.GetColumn<double>("Score")[0].Should().Be(9.5);
    }
}
