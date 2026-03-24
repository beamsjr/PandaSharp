using FluentAssertions;
using PandaSharp.Accessors;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Accessors;

public class DateTimeAccessorTests
{
    private static Column<DateTime> Col(params DateTime[] values) => new("dt", values);

    private static readonly DateTime D1 = new(2024, 1, 15, 10, 30, 45);
    private static readonly DateTime D2 = new(2024, 6, 30, 23, 59, 59);
    private static readonly DateTime D3 = new(2024, 12, 31, 0, 0, 0);

    [Fact]
    public void Year_ExtractsYear()
    {
        var result = Col(D1, D2).Dt().Year();
        result[0].Should().Be(2024);
        result[1].Should().Be(2024);
    }

    [Fact]
    public void Month_ExtractsMonth()
    {
        var result = Col(D1, D2).Dt().Month();
        result[0].Should().Be(1);
        result[1].Should().Be(6);
    }

    [Fact]
    public void Day_ExtractsDay()
    {
        var result = Col(D1, D2).Dt().Day();
        result[0].Should().Be(15);
        result[1].Should().Be(30);
    }

    [Fact]
    public void Hour_ExtractsHour()
    {
        var result = Col(D1).Dt().Hour();
        result[0].Should().Be(10);
    }

    [Fact]
    public void Minute_ExtractsMinute()
    {
        var result = Col(D1).Dt().Minute();
        result[0].Should().Be(30);
    }

    [Fact]
    public void Second_ExtractsSecond()
    {
        var result = Col(D1).Dt().Second();
        result[0].Should().Be(45);
    }

    [Fact]
    public void DayOfWeek_ReturnsCorrectDay()
    {
        // 2024-01-15 is a Monday = 1
        var result = Col(D1).Dt().DayOfWeek();
        result[0].Should().Be(1);
    }

    [Fact]
    public void DayOfYear_ReturnsCorrectDayOfYear()
    {
        var result = Col(D1).Dt().DayOfYear();
        result[0].Should().Be(15);
    }

    [Fact]
    public void Quarter_ReturnsCorrectQuarter()
    {
        var result = Col(D1, D2, D3).Dt().Quarter();
        result[0].Should().Be(1); // Jan
        result[1].Should().Be(2); // Jun
        result[2].Should().Be(4); // Dec
    }

    [Fact]
    public void IsMonthStart_IdentifiesFirstOfMonth()
    {
        var result = Col(new DateTime(2024, 3, 1), new DateTime(2024, 3, 15)).Dt().IsMonthStart();
        result[0].Should().Be(true);
        result[1].Should().Be(false);
    }

    [Fact]
    public void IsMonthEnd_IdentifiesLastOfMonth()
    {
        var result = Col(D2, D1).Dt().IsMonthEnd(); // June 30 is end of month
        result[0].Should().Be(true);
        result[1].Should().Be(false);
    }

    [Fact]
    public void IsYearStart_IdentifiesJan1()
    {
        var result = Col(new DateTime(2024, 1, 1), D1).Dt().IsYearStart();
        result[0].Should().Be(true);
        result[1].Should().Be(false);
    }

    [Fact]
    public void IsYearEnd_IdentifiesDec31()
    {
        var result = Col(D3, D1).Dt().IsYearEnd();
        result[0].Should().Be(true);
        result[1].Should().Be(false);
    }

    [Fact]
    public void Floor_FloorsToHour()
    {
        var result = Col(D1).Dt().Floor(TimeSpan.FromHours(1));
        result[0].Should().Be(new DateTime(2024, 1, 15, 10, 0, 0));
    }

    [Fact]
    public void Ceil_CeilsToHour()
    {
        var result = Col(D1).Dt().Ceil(TimeSpan.FromHours(1));
        result[0].Should().Be(new DateTime(2024, 1, 15, 11, 0, 0));
    }

    [Fact]
    public void Round_RoundsToHour()
    {
        // D1 is 10:30:45 — rounds to 11:00
        var result = Col(D1).Dt().Round(TimeSpan.FromHours(1));
        result[0].Should().Be(new DateTime(2024, 1, 15, 11, 0, 0));
    }

    [Fact]
    public void AddDays_AddsCorrectly()
    {
        var result = Col(D1).Dt().AddDays(5);
        result[0].Should().Be(new DateTime(2024, 1, 20, 10, 30, 45));
    }

    [Fact]
    public void AddMonths_AddsCorrectly()
    {
        var result = Col(D1).Dt().AddMonths(2);
        result[0].Should().Be(new DateTime(2024, 3, 15, 10, 30, 45));
    }

    [Fact]
    public void Date_ExtractsDateOnly()
    {
        var result = Col(D1).Dt().Date();
        result[0].Should().Be(new DateTime(2024, 1, 15));
    }

    [Fact]
    public void NullPropagation_Works()
    {
        var col = Column<DateTime>.FromNullable("dt", [D1, null]);
        var result = col.Dt().Year();
        result[0].Should().Be(2024);
        result[1].Should().BeNull();
    }

    [Fact]
    public void DateRange_GeneratesSequence()
    {
        var col = DateTimeAccessor.DateRange(
            new DateTime(2024, 1, 1), new DateTime(2024, 1, 5), TimeSpan.FromDays(1));

        col.Length.Should().Be(5);
        col[0].Should().Be(new DateTime(2024, 1, 1));
        col[4].Should().Be(new DateTime(2024, 1, 5));
    }

    [Fact]
    public void DateRange_WithPeriods()
    {
        var col = DateTimeAccessor.DateRange(
            new DateTime(2024, 1, 1), periods: 3, TimeSpan.FromDays(7));

        col.Length.Should().Be(3);
        col[0].Should().Be(new DateTime(2024, 1, 1));
        col[1].Should().Be(new DateTime(2024, 1, 8));
        col[2].Should().Be(new DateTime(2024, 1, 15));
    }
}
