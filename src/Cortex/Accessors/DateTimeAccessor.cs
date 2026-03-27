using Cortex.Column;

namespace Cortex.Accessors;

/// <summary>
/// Vectorized datetime operations accessible via Column&lt;DateTime&gt;.Dt() extension.
/// All operations are null-propagating.
/// </summary>
public class DateTimeAccessor
{
    private readonly Column<DateTime> _column;

    internal DateTimeAccessor(Column<DateTime> column) => _column = column;

    private string Name => _column.Name;
    private int Length => _column.Length;

    // -- Component extraction --

    public Column<int> Year() => MapInt(dt => dt.Year);
    public Column<int> Month() => MapInt(dt => dt.Month);
    public Column<int> Day() => MapInt(dt => dt.Day);
    public Column<int> Hour() => MapInt(dt => dt.Hour);
    public Column<int> Minute() => MapInt(dt => dt.Minute);
    public Column<int> Second() => MapInt(dt => dt.Second);
    public Column<int> Millisecond() => MapInt(dt => dt.Millisecond);
    public Column<int> Microsecond() => MapInt(dt => dt.Microsecond);
    public Column<int> DayOfWeek() => MapInt(dt => (int)dt.DayOfWeek);
    public Column<int> DayOfYear() => MapInt(dt => dt.DayOfYear);
    public Column<int> Quarter() => MapInt(dt => (dt.Month - 1) / 3 + 1);

    // -- Boolean properties --

    public Column<bool> IsMonthStart() => MapBool(dt => dt.Day == 1);
    public Column<bool> IsMonthEnd() => MapBool(dt => dt.Day == DateTime.DaysInMonth(dt.Year, dt.Month));
    public Column<bool> IsQuarterStart() => MapBool(dt => dt.Day == 1 && (dt.Month - 1) % 3 == 0);
    public Column<bool> IsQuarterEnd() => MapBool(dt =>
    {
        int endMonth = ((dt.Month - 1) / 3 + 1) * 3;
        return dt.Month == endMonth && dt.Day == DateTime.DaysInMonth(dt.Year, endMonth);
    });
    public Column<bool> IsYearStart() => MapBool(dt => dt.Month == 1 && dt.Day == 1);
    public Column<bool> IsYearEnd() => MapBool(dt => dt.Month == 12 && dt.Day == 31);
    public Column<bool> IsLeapYear() => MapBool(dt => DateTime.IsLeapYear(dt.Year));

    // -- Date only --

    public Column<DateTime> Date() => MapDateTime(dt => dt.Date);

    // -- Floor / Ceil / Round --

    public Column<DateTime> Floor(TimeSpan freq) => MapDateTime(dt =>
    {
        long ticks = dt.Ticks / freq.Ticks * freq.Ticks;
        return new DateTime(ticks, dt.Kind);
    });

    public Column<DateTime> Ceil(TimeSpan freq) => MapDateTime(dt =>
    {
        long ticks = (dt.Ticks + freq.Ticks - 1) / freq.Ticks * freq.Ticks;
        return new DateTime(ticks, dt.Kind);
    });

    public Column<DateTime> Round(TimeSpan freq) => MapDateTime(dt =>
    {
        long ticks = (dt.Ticks + freq.Ticks / 2) / freq.Ticks * freq.Ticks;
        return new DateTime(ticks, dt.Kind);
    });

    // -- Arithmetic --

    public Column<DateTime> AddDays(int days) => MapDateTime(dt => dt.AddDays(days));
    public Column<DateTime> AddMonths(int months) => MapDateTime(dt => dt.AddMonths(months));
    public Column<DateTime> AddYears(int years) => MapDateTime(dt => dt.AddYears(years));
    public Column<DateTime> AddHours(double hours) => MapDateTime(dt => dt.AddHours(hours));

    // -- Static factories --

    /// <summary>
    /// Generate a date range as a Column&lt;DateTime&gt;.
    /// </summary>
    public static Column<DateTime> DateRange(DateTime start, DateTime end, TimeSpan step, string name = "date")
    {
        var dates = new List<DateTime>();
        for (var dt = start; dt <= end; dt += step)
            dates.Add(dt);
        return new Column<DateTime>(name, dates.ToArray());
    }

    public static Column<DateTime> DateRange(DateTime start, int periods, TimeSpan step, string name = "date")
    {
        var dates = new DateTime[periods];
        for (int i = 0; i < periods; i++)
            dates[i] = start + TimeSpan.FromTicks(step.Ticks * i);
        return new Column<DateTime>(name, dates);
    }

    // -- Helpers --

    private Column<int> MapInt(Func<DateTime, int> func)
    {
        var result = new int?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _column[i] is { } dt ? func(dt) : null;
        return Column<int>.FromNullable(Name, result);
    }

    private Column<bool> MapBool(Func<DateTime, bool> func)
    {
        var result = new bool?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _column[i] is { } dt ? func(dt) : null;
        return Column<bool>.FromNullable(Name, result);
    }

    private Column<DateTime> MapDateTime(Func<DateTime, DateTime> func)
    {
        var result = new DateTime?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _column[i] is { } dt ? func(dt) : null;
        return Column<DateTime>.FromNullable(Name, result);
    }
}
