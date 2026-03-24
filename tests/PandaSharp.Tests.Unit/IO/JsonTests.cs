using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.IO;

namespace PandaSharp.Tests.Unit.IO;

public class JsonTests
{
    [Fact]
    public void ReadJson_RecordOriented()
    {
        var json = """
        [
            {"Name": "Alice", "Age": 25, "Active": true},
            {"Name": "Bob", "Age": 30, "Active": false}
        ]
        """;

        var df = JsonReader.ReadString(json);

        df.RowCount.Should().Be(2);
        df.GetStringColumn("Name")[0].Should().Be("Alice");
        df.GetColumn<int>("Age")[1].Should().Be(30);
        df.GetColumn<bool>("Active")[0].Should().Be(true);
    }

    [Fact]
    public void ReadJson_ColumnOriented()
    {
        var json = """
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30]
        }
        """;

        var df = JsonReader.ReadString(json);

        df.RowCount.Should().Be(2);
        df.GetStringColumn("Name")[1].Should().Be("Bob");
        df.GetColumn<int>("Age")[0].Should().Be(25);
    }

    [Fact]
    public void ReadJson_WithNulls()
    {
        var json = """
        [
            {"A": 1, "B": "x"},
            {"A": null, "B": "y"}
        ]
        """;

        var df = JsonReader.ReadString(json);
        df["A"].IsNull(1).Should().BeTrue();
    }

    [Fact]
    public void WriteJson_Records_RoundTrips()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        var json = JsonWriter.WriteString(df, JsonOrient.Records);
        var df2 = JsonReader.ReadString(json);

        df2.RowCount.Should().Be(2);
        df2.GetStringColumn("Name")[0].Should().Be("Alice");
        df2.GetColumn<int>("Age")[1].Should().Be(30);
    }

    [Fact]
    public void WriteJson_Columns()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice"]),
            new Column<int>("Age", [25])
        );

        var json = JsonWriter.WriteString(df, JsonOrient.Columns);

        json.Should().Contain("\"Name\"");
        json.Should().Contain("\"Alice\"");

        var df2 = JsonReader.ReadString(json);
        df2.RowCount.Should().Be(1);
    }

    [Fact]
    public void WriteJson_HandlesNulls()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3])
        );

        var json = JsonWriter.WriteString(df);
        json.Should().Contain("null");
    }
}
