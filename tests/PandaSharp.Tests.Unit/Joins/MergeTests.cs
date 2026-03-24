using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Joins;

namespace PandaSharp.Tests.Unit.Joins;

public class MergeTests
{
    [Fact]
    public void Merge_IsAliasForJoin()
    {
        var left = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie"])
        );
        var right = new DataFrame(
            new Column<int>("Id", [1, 2]),
            new StringColumn("Role", ["Dev", "PM"])
        );

        var result = left.Merge(right, "Id");

        result.RowCount.Should().Be(2); // inner join default
        result.ColumnNames.Should().Contain("Role");
    }

    [Fact]
    public void Merge_WithDifferentKeys()
    {
        var left = new DataFrame(
            new Column<int>("LeftId", [1, 2]),
            new StringColumn("Name", ["Alice", "Bob"])
        );
        var right = new DataFrame(
            new Column<int>("RightId", [1, 2]),
            new StringColumn("Role", ["Dev", "PM"])
        );

        var result = left.Merge(right, leftOn: "LeftId", rightOn: "RightId");
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void Merge_OuterJoin()
    {
        var left = new DataFrame(
            new Column<int>("Id", [1, 2]),
            new StringColumn("Name", ["Alice", "Bob"])
        );
        var right = new DataFrame(
            new Column<int>("Id", [2, 3]),
            new StringColumn("Role", ["PM", "QA"])
        );

        var result = left.Merge(right, "Id", how: JoinType.Outer);
        result.RowCount.Should().Be(3); // 1, 2, 3
    }
}
