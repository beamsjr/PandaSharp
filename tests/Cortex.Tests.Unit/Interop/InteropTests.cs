using System.Data;
using FluentAssertions;
using Cortex.Column;
using Cortex.Interop;
using Cortex.Schema;
using DataFrameSchema = Cortex.Schema.Schema;

namespace Cortex.Tests.Unit.Interop;

public class InteropTests
{
    // -- LINQ --

    public class Person
    {
        public string Name { get; set; } = "";
        public int Age { get; set; }
        public double Salary { get; set; }
    }

    [Fact]
    public void FromEnumerable_CreatesDataFrame()
    {
        var people = new[]
        {
            new Person { Name = "Alice", Age = 25, Salary = 50_000 },
            new Person { Name = "Bob", Age = 30, Salary = 62_000 }
        };

        var df = LinqExtensions.FromEnumerable(people);

        df.RowCount.Should().Be(2);
        df.ColumnNames.Should().Contain("Name");
        df.ColumnNames.Should().Contain("Age");
        df.GetStringColumn("Name")[0].Should().Be("Alice");
        df.GetColumn<int>("Age")[1].Should().Be(30);
    }

    [Fact]
    public void AsEnumerable_ProjectsToObjects()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30]),
            new Column<double>("Salary", [50_000, 62_000])
        );

        var people = df.AsEnumerable<Person>().ToList();

        people.Should().HaveCount(2);
        people[0].Name.Should().Be("Alice");
        people[0].Age.Should().Be(25);
        people[1].Salary.Should().Be(62_000);
    }

    [Fact]
    public void FromEnumerable_AnonymousTypes()
    {
        var items = new[]
        {
            new { X = 1, Y = "a" },
            new { X = 2, Y = "b" }
        };

        var df = LinqExtensions.FromEnumerable(items);

        df.RowCount.Should().Be(2);
        df.GetColumn<int>("X")[0].Should().Be(1);
        df.GetStringColumn("Y")[1].Should().Be("b");
    }

    // -- DataTable --

    [Fact]
    public void FromDataTable_CreatesDataFrame()
    {
        var table = new DataTable();
        table.Columns.Add("Name", typeof(string));
        table.Columns.Add("Age", typeof(int));
        table.Rows.Add("Alice", 25);
        table.Rows.Add("Bob", 30);

        var df = DataTableBridge.FromDataTable(table);

        df.RowCount.Should().Be(2);
        df.GetStringColumn("Name")[0].Should().Be("Alice");
        df.GetColumn<int>("Age")[1].Should().Be(30);
    }

    [Fact]
    public void ToDataTable_RoundTrips()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        var table = df.ToDataTable();

        table.Rows.Count.Should().Be(2);
        table.Columns["Name"]!.DataType.Should().Be(typeof(string));
        table.Rows[0]["Name"].Should().Be("Alice");
        table.Rows[1]["Age"].Should().Be(30);
    }

    [Fact]
    public void ToDataTable_HandlesNulls()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3])
        );

        var table = df.ToDataTable();
        table.Rows[1]["A"].Should().Be(DBNull.Value);
    }

    // -- Schema --

    [Fact]
    public void Schema_Valid_Passes()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice"]),
            new Column<int>("Age", [25])
        );

        var schema = new DataFrameSchema(
            new ColumnSchema("Name", typeof(string)),
            new ColumnSchema("Age", typeof(int))
        );

        df.MatchesSchema(schema).Should().BeTrue();
    }

    [Fact]
    public void Schema_MissingColumn_Fails()
    {
        var df = new DataFrame(new StringColumn("Name", ["Alice"]));

        var schema = new DataFrameSchema(
            new ColumnSchema("Name", typeof(string)),
            new ColumnSchema("Age", typeof(int))
        );

        df.MatchesSchema(schema).Should().BeFalse();
    }

    [Fact]
    public void Schema_WrongType_Fails()
    {
        var df = new DataFrame(
            new StringColumn("Age", ["25"])
        );

        var schema = new DataFrameSchema(new ColumnSchema("Age", typeof(int)));

        df.MatchesSchema(schema).Should().BeFalse();
    }

    [Fact]
    public void Schema_NonNullable_WithNulls_Fails()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("Age", [25, null])
        );

        var schema = new DataFrameSchema(new ColumnSchema("Age", typeof(int), Nullable: false));

        df.MatchesSchema(schema).Should().BeFalse();
    }

    [Fact]
    public void Schema_Validate_ThrowsWithDetails()
    {
        var df = new DataFrame(new StringColumn("Name", ["Alice"]));
        var schema = new DataFrameSchema(new ColumnSchema("Missing", typeof(int)));

        var act = () => df.ValidateSchema(schema);
        act.Should().Throw<SchemaValidationException>()
            .Which.Errors.Should().Contain(e => e.Contains("Missing"));
    }
}
