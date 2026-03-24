using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Joins;

namespace PandaSharp.Tests.Unit.Joins;

public class AsOfJoinTests
{
    [Fact]
    public void AsOfJoin_Backward_FindsNearestPrior()
    {
        var trades = new DataFrame(
            new Column<int>("Time", [1, 5, 10]),
            new StringColumn("Symbol", ["AAPL", "AAPL", "AAPL"]),
            new Column<double>("Price", [100.0, 102.0, 101.0])
        );

        var quotes = new DataFrame(
            new Column<int>("Time", [2, 4, 7, 11]),
            new Column<double>("Bid", [99.5, 101.0, 101.5, 100.5])
        );

        var result = trades.JoinAsOf(quotes, on: "Time", direction: AsOfDirection.Backward);

        result.RowCount.Should().Be(3);
        result.ColumnNames.Should().Contain("Bid");

        // Time=1: nearest backward quote is none → null? Actually Time=2 is > 1, no backward match
        // Time=5: nearest backward quote <= 5 is Time=4 → Bid=101.0
        // Time=10: nearest backward quote <= 10 is Time=7 → Bid=101.5
        result.GetColumn<double>("Bid")[1].Should().Be(101.0);  // Time=5 matched Time=4
        result.GetColumn<double>("Bid")[2].Should().Be(101.5);  // Time=10 matched Time=7
    }

    [Fact]
    public void AsOfJoin_Forward_FindsNearestFuture()
    {
        var left = new DataFrame(
            new Column<int>("Time", [3, 6]),
            new Column<double>("Value", [10.0, 20.0])
        );

        var right = new DataFrame(
            new Column<int>("Time", [1, 5, 8]),
            new Column<double>("Ref", [100.0, 200.0, 300.0])
        );

        var result = left.JoinAsOf(right, on: "Time", direction: AsOfDirection.Forward);

        // Time=3: nearest forward >= 3 is Time=5 → Ref=200
        // Time=6: nearest forward >= 6 is Time=8 → Ref=300
        result.GetColumn<double>("Ref")[0].Should().Be(200.0);
        result.GetColumn<double>("Ref")[1].Should().Be(300.0);
    }

    [Fact]
    public void AsOfJoin_WithByColumn()
    {
        var left = new DataFrame(
            new StringColumn("Sym", ["A", "B", "A"]),
            new Column<int>("Time", [5, 5, 10]),
            new Column<double>("Price", [1.0, 2.0, 3.0])
        );

        var right = new DataFrame(
            new StringColumn("Sym", ["A", "B", "A"]),
            new Column<int>("Time", [3, 4, 8]),
            new Column<double>("Bid", [10.0, 20.0, 30.0])
        );

        var result = left.JoinAsOf(right, on: "Time", by: "Sym");

        // Row 0: Sym=A, Time=5 → backward match Sym=A, Time=3 → Bid=10
        // Row 1: Sym=B, Time=5 → backward match Sym=B, Time=4 → Bid=20
        // Row 2: Sym=A, Time=10 → backward match Sym=A, Time=8 → Bid=30
        result.GetColumn<double>("Bid")[0].Should().Be(10.0);
        result.GetColumn<double>("Bid")[1].Should().Be(20.0);
        result.GetColumn<double>("Bid")[2].Should().Be(30.0);
    }
}
