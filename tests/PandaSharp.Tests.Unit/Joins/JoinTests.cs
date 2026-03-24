using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Joins;

namespace PandaSharp.Tests.Unit.Joins;

public class JoinTests
{
    private static DataFrame Employees() => new(
        new Column<int>("EmpId", [1, 2, 3, 4]),
        new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana"]),
        new Column<int>("DeptId", [10, 20, 10, 30])
    );

    private static DataFrame Departments() => new(
        new Column<int>("DeptId", [10, 20, 40]),
        new StringColumn("DeptName", ["Sales", "Engineering", "Marketing"])
    );

    [Fact]
    public void InnerJoin_ReturnsMatchingRows()
    {
        var result = Employees().Join(Departments(), "DeptId", how: JoinType.Inner);

        result.RowCount.Should().Be(3); // Alice(10), Bob(20), Charlie(10)
        result.ColumnNames.Should().Contain("DeptName");
        result.ColumnNames.Should().Contain("Name");
    }

    [Fact]
    public void LeftJoin_KeepsAllLeftRows()
    {
        var result = Employees().Join(Departments(), "DeptId", how: JoinType.Left);

        result.RowCount.Should().Be(4); // all employees, Diana has null DeptName
        // Diana's DeptId=30, not in Departments
        var lastDeptName = result.GetStringColumn("DeptName")[3];
        lastDeptName.Should().BeNull();
    }

    [Fact]
    public void RightJoin_KeepsAllRightRows()
    {
        var result = Employees().Join(Departments(), "DeptId", how: JoinType.Right);

        // 3 matches + 1 unmatched right (Marketing, DeptId=40)
        result.RowCount.Should().Be(4);
    }

    [Fact]
    public void OuterJoin_KeepsAllRows()
    {
        var result = Employees().Join(Departments(), "DeptId", how: JoinType.Outer);

        // 3 matches + 1 unmatched left (Diana) + 1 unmatched right (Marketing)
        result.RowCount.Should().Be(5);
    }

    [Fact]
    public void AntiJoin_ReturnsUnmatchedLeft()
    {
        var result = Employees().Join(Departments(), "DeptId", how: JoinType.Anti);

        result.RowCount.Should().Be(1); // Diana (DeptId=30)
        result.GetStringColumn("Name")[0].Should().Be("Diana");
    }

    [Fact]
    public void CrossJoin_ReturnsCartesianProduct()
    {
        var left = new DataFrame(new StringColumn("A", ["x", "y"]));
        var right = new DataFrame(new StringColumn("B", ["1", "2", "3"]));

        var result = left.Join(right, Array.Empty<string>(), Array.Empty<string>(), how: JoinType.Cross);

        result.RowCount.Should().Be(6); // 2 * 3
    }

    [Fact]
    public void Join_DifferentKeyNames()
    {
        var left = new DataFrame(
            new Column<int>("Id", [1, 2]),
            new StringColumn("Name", ["Alice", "Bob"])
        );
        var right = new DataFrame(
            new Column<int>("EmpId", [1, 2]),
            new StringColumn("Role", ["Dev", "PM"])
        );

        var result = left.Join(right, leftOn: "Id", rightOn: "EmpId");

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("Role");
    }

    [Fact]
    public void Join_MultiKey()
    {
        var left = new DataFrame(
            new StringColumn("Dept", ["Sales", "Eng"]),
            new StringColumn("Level", ["Jr", "Sr"]),
            new Column<int>("Count", [5, 3])
        );
        var right = new DataFrame(
            new StringColumn("Dept", ["Sales", "Sales"]),
            new StringColumn("Level", ["Jr", "Sr"]),
            new Column<double>("Budget", [100.0, 200.0])
        );

        var result = left.Join(right, ["Dept", "Level"], ["Dept", "Level"]);

        result.RowCount.Should().Be(1); // only Sales+Jr matches
        result.GetColumn<double>("Budget")[0].Should().Be(100.0);
    }

    [Fact]
    public void Join_OverlappingColumnNames_GetSuffix()
    {
        var left = new DataFrame(
            new Column<int>("Id", [1]),
            new StringColumn("Value", ["left"])
        );
        var right = new DataFrame(
            new Column<int>("Id", [1]),
            new StringColumn("Value", ["right"])
        );

        var result = left.Join(right, "Id");

        result.ColumnNames.Should().Contain("Value_x");
        result.ColumnNames.Should().Contain("Value_y");
    }

    [Fact]
    public void Join_DuplicateKeysInRight_ProduceMultipleRows()
    {
        var left = new DataFrame(
            new Column<int>("Id", [1]),
            new StringColumn("Name", ["Alice"])
        );
        var right = new DataFrame(
            new Column<int>("Id", [1, 1]),
            new StringColumn("Tag", ["A", "B"])
        );

        var result = left.Join(right, "Id");

        result.RowCount.Should().Be(2); // 1 left * 2 right matches
    }
}
