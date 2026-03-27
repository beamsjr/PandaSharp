using Cortex;
using Cortex.Column;
using Cortex.ML.Transformers;

namespace Cortex.TimeSeries.Features;

/// <summary>
/// Transformer that extracts calendar features from a DateTime column:
/// day_of_week, month, quarter, is_weekend, day_of_year.
/// </summary>
public class DateTimeFeatures : ITransformer
{
    private readonly string _dateColumn;

    /// <inheritdoc />
    public string Name => "DateTimeFeatures";

    /// <summary>
    /// Create a DateTime feature extractor.
    /// </summary>
    /// <param name="dateColumn">Name of the DateTime column to extract features from.</param>
    public DateTimeFeatures(string dateColumn)
    {
        if (string.IsNullOrEmpty(dateColumn))
            throw new ArgumentException("Date column name must not be empty.", nameof(dateColumn));
        _dateColumn = dateColumn;
    }

    /// <inheritdoc />
    public ITransformer Fit(DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(df);
        if (!df.ColumnNames.Contains(_dateColumn))
            throw new KeyNotFoundException($"Column '{_dateColumn}' not found.");
        return this;
    }

    /// <inheritdoc />
    public DataFrame Transform(DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(df);
        var col = df.GetColumn<DateTime>(_dateColumn);
        int n = df.RowCount;
        var span = col.Values;

        var dayOfWeek = new int[n];
        var month = new int[n];
        var quarter = new int[n];
        var isWeekend = new int[n];
        var dayOfYear = new int[n];

        for (int i = 0; i < n; i++)
        {
            DateTime dt = span[i];
            dayOfWeek[i] = (int)dt.DayOfWeek;
            month[i] = dt.Month;
            quarter[i] = (dt.Month - 1) / 3 + 1;
            isWeekend[i] = dt.DayOfWeek is DayOfWeek.Saturday or DayOfWeek.Sunday ? 1 : 0;
            dayOfYear[i] = dt.DayOfYear;
        }

        var result = df;
        result = result.Assign($"{_dateColumn}_day_of_week", new Column<int>($"{_dateColumn}_day_of_week", dayOfWeek));
        result = result.Assign($"{_dateColumn}_month", new Column<int>($"{_dateColumn}_month", month));
        result = result.Assign($"{_dateColumn}_quarter", new Column<int>($"{_dateColumn}_quarter", quarter));
        result = result.Assign($"{_dateColumn}_is_weekend", new Column<int>($"{_dateColumn}_is_weekend", isWeekend));
        result = result.Assign($"{_dateColumn}_day_of_year", new Column<int>($"{_dateColumn}_day_of_year", dayOfYear));

        return result;
    }

    /// <inheritdoc />
    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
}
