using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.TimeSeries;

namespace PandaSharp.Tests.Unit.TimeSeries;

public class ResampleTests
{
    [Fact]
    public void Resample_ByHour_GroupsCorrectly()
    {
        var df = new DataFrame(
            new Column<DateTime>("Time", [
                new DateTime(2024, 1, 1, 10, 0, 0),
                new DateTime(2024, 1, 1, 10, 15, 0),
                new DateTime(2024, 1, 1, 10, 45, 0),
                new DateTime(2024, 1, 1, 11, 0, 0),
                new DateTime(2024, 1, 1, 11, 30, 0),
            ]),
            new Column<double>("Value", [1.0, 2.0, 3.0, 4.0, 5.0])
        );

        var result = df.Resample("Time", TimeSpan.FromHours(1)).Mean();

        result.RowCount.Should().Be(2); // 10:00 bucket and 11:00 bucket
    }

    [Fact]
    public void Resample_Sum_AggregatesWithinBuckets()
    {
        var df = new DataFrame(
            new Column<DateTime>("Time", [
                new DateTime(2024, 1, 1, 0, 0, 0),
                new DateTime(2024, 1, 1, 0, 30, 0),
                new DateTime(2024, 1, 1, 1, 0, 0),
            ]),
            new Column<double>("Value", [10.0, 20.0, 30.0])
        );

        var result = df.Resample("Time", TimeSpan.FromHours(1)).Sum();

        // First hour: 10+20=30, second hour: 30
        for (int i = 0; i < result.RowCount; i++)
        {
            var time = result.GetColumn<DateTime>("Time")[i];
            var val = result.GetColumn<double>("Value")[i];
            if (time == new DateTime(2024, 1, 1, 0, 0, 0))
                val.Should().Be(30.0);
            else
                val.Should().Be(30.0);
        }
    }

    [Fact]
    public void Resample_StringFrequency_1h()
    {
        var df = new DataFrame(
            new Column<DateTime>("Time", [
                new DateTime(2024, 1, 1, 10, 0, 0),
                new DateTime(2024, 1, 1, 11, 0, 0),
            ]),
            new Column<double>("Value", [1.0, 2.0])
        );

        var result = df.Resample("Time", "1h").Sum();
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void Resample_ByDay()
    {
        var df = new DataFrame(
            new Column<DateTime>("Date", [
                new DateTime(2024, 1, 1),
                new DateTime(2024, 1, 1),
                new DateTime(2024, 1, 2),
            ]),
            new Column<double>("Value", [10.0, 20.0, 30.0])
        );

        var result = df.Resample("Date", "1d").Sum();
        result.RowCount.Should().Be(2);
    }
}
