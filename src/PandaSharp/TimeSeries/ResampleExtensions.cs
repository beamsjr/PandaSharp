using PandaSharp.Column;
using PandaSharp.GroupBy;

namespace PandaSharp.TimeSeries;

public static class ResampleExtensions
{
    /// <summary>
    /// Resample a DataFrame by a datetime column to a given frequency.
    /// Groups rows into time buckets and returns a GroupedDataFrame for aggregation.
    /// Usage: df.Resample("Date", TimeSpan.FromHours(1)).Mean()
    /// </summary>
    public static GroupedDataFrame Resample(this DataFrame df, string dateColumn, TimeSpan frequency)
    {
        var dtCol = df.GetColumn<DateTime>(dateColumn);

        // Floor each datetime to the frequency bucket
        var buckets = new DateTime?[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
        {
            if (dtCol[i] is { } dt)
            {
                long ticks = dt.Ticks / frequency.Ticks * frequency.Ticks;
                buckets[i] = new DateTime(ticks, dt.Kind);
            }
        }

        // Replace the date column with bucketed version and group by it
        var bucketCol = Column<DateTime>.FromNullable(dateColumn, buckets);
        var columns = new List<IColumn>();
        foreach (var name in df.ColumnNames)
        {
            columns.Add(name == dateColumn ? bucketCol : df[name]);
        }

        var bucketed = new DataFrame(columns);
        return bucketed.GroupBy(dateColumn);
    }

    /// <summary>
    /// Resample with a string frequency spec: "1h", "30m", "1d", "1w".
    /// </summary>
    public static GroupedDataFrame Resample(this DataFrame df, string dateColumn, string frequency)
    {
        var ts = ParseFrequency(frequency);
        return df.Resample(dateColumn, ts);
    }

    private static TimeSpan ParseFrequency(string freq)
    {
        freq = freq.Trim().ToLowerInvariant();

        if (freq.EndsWith("ms"))
        {
            if (double.TryParse(freq[..^2], out double ms))
                return TimeSpan.FromMilliseconds(ms);
        }

        char unit = freq[^1];
        if (!double.TryParse(freq[..^1], out double value))
            throw new ArgumentException($"Invalid frequency: '{freq}'");

        return unit switch
        {
            's' => TimeSpan.FromSeconds(value),
            'm' => TimeSpan.FromMinutes(value),
            'h' => TimeSpan.FromHours(value),
            'd' => TimeSpan.FromDays(value),
            'w' => TimeSpan.FromDays(value * 7),
            _ => throw new ArgumentException($"Unknown frequency unit: '{unit}'. Use s, m, h, d, or w.")
        };
    }
}
