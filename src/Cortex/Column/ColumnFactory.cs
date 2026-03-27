namespace Cortex.Column;

/// <summary>
/// Factory methods for creating columns from various input types.
/// </summary>
public static class ColumnFactory
{
    public static Column<int> Create(string name, int[] values) => new(name, values);
    public static Column<int> Create(string name, int?[] values) => Column<int>.FromNullable(name, values);

    public static Column<long> Create(string name, long[] values) => new(name, values);
    public static Column<long> Create(string name, long?[] values) => Column<long>.FromNullable(name, values);

    public static Column<double> Create(string name, double[] values) => new(name, values);
    public static Column<double> Create(string name, double?[] values) => Column<double>.FromNullable(name, values);

    public static Column<float> Create(string name, float[] values) => new(name, values);
    public static Column<float> Create(string name, float?[] values) => Column<float>.FromNullable(name, values);

    public static Column<bool> Create(string name, bool[] values) => new(name, values);
    public static Column<bool> Create(string name, bool?[] values) => Column<bool>.FromNullable(name, values);

    public static Column<DateTime> Create(string name, DateTime[] values) => new(name, values);
    public static Column<DateTime> Create(string name, DateTime?[] values) => Column<DateTime>.FromNullable(name, values);

    public static StringColumn Create(string name, string?[] values) => new(name, values);
}
