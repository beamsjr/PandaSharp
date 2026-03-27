using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using Cortex.Joins;
using Cortex.Lazy;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit.Lazy;

public class LazyJoinTests
{
    [Fact]
    public void Lazy_Join_Inner()
    {
        var left = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie"])
        );
        var right = new DataFrame(
            new Column<int>("Id", [1, 2]),
            new StringColumn("Role", ["Dev", "PM"])
        );

        var result = left.Lazy()
            .Join(right.Lazy(), "Id")
            .Collect();

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("Role");
    }

    [Fact]
    public void Lazy_Join_Left()
    {
        var left = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie"])
        );
        var right = new DataFrame(
            new Column<int>("Id", [1, 2]),
            new StringColumn("Role", ["Dev", "PM"])
        );

        var result = left.Lazy()
            .Join(right.Lazy(), "Id", JoinType.Left)
            .Collect();

        result.RowCount.Should().Be(3);
        result.GetStringColumn("Role")[2].Should().BeNull(); // Charlie has no match
    }

    [Fact]
    public void Lazy_FilterThenJoin()
    {
        var left = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new Column<int>("Value", [10, 20, 30])
        );
        var right = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new StringColumn("Label", ["A", "B", "C"])
        );

        var result = left.Lazy()
            .Filter(Col("Value") > Lit(15))
            .Join(right.Lazy(), "Id")
            .Collect();

        result.RowCount.Should().Be(2); // Id 2 and 3
    }

    [Fact]
    public void Lazy_Join_Explain()
    {
        var left = new DataFrame(new Column<int>("Id", [1]));
        var right = new DataFrame(new Column<int>("Id", [1]));

        var plan = left.Lazy().Join(right.Lazy(), "Id").Explain();

        plan.Should().Contain("Join");
        plan.Should().Contain("Id");
    }
}
