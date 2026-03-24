namespace PandaSharp.Column;

/// <summary>Shared type checking utilities.</summary>
public static class TypeHelpers
{
    public static bool IsNumeric(Type type) =>
        type == typeof(int) || type == typeof(long) ||
        type == typeof(double) || type == typeof(float) ||
        type == typeof(decimal) || type == typeof(short) ||
        type == typeof(byte);

    /// <summary>
    /// Read a numeric column value as double without boxing.
    /// Falls back to GetObject + Convert.ToDouble for unknown types.
    /// </summary>
    public static double GetDouble(IColumn col, int index)
    {
        if (col.IsNull(index)) return double.NaN;

        if (col is Column<double> dCol) return dCol.Values[index];
        if (col is Column<int> iCol) return iCol.Values[index];
        if (col is Column<float> fCol) return fCol.Values[index];
        if (col is Column<long> lCol) return lCol.Values[index];
        if (col is Column<short> sCol) return sCol.Values[index];
        if (col is Column<byte> bCol) return bCol.Values[index];
        if (col is Column<decimal> decCol) return (double)decCol.Values[index];

        return Convert.ToDouble(col.GetObject(index));
    }

    /// <summary>
    /// Extract all values from a numeric column as double[] without boxing.
    /// </summary>
    public static double[] GetDoubleArray(IColumn col)
    {
        var result = new double[col.Length];

        if (col is Column<double> dCol)
        {
            var span = dCol.Values;
            for (int i = 0; i < result.Length; i++)
                result[i] = dCol.IsNull(i) ? double.NaN : span[i];
            return result;
        }
        if (col is Column<int> iCol)
        {
            var span = iCol.Values;
            for (int i = 0; i < result.Length; i++)
                result[i] = iCol.IsNull(i) ? double.NaN : span[i];
            return result;
        }
        if (col is Column<float> fCol)
        {
            var span = fCol.Values;
            for (int i = 0; i < result.Length; i++)
                result[i] = fCol.IsNull(i) ? double.NaN : span[i];
            return result;
        }
        if (col is Column<long> lCol)
        {
            var span = lCol.Values;
            for (int i = 0; i < result.Length; i++)
                result[i] = lCol.IsNull(i) ? double.NaN : span[i];
            return result;
        }

        // Fallback for rare types
        for (int i = 0; i < result.Length; i++)
            result[i] = col.IsNull(i) ? double.NaN : Convert.ToDouble(col.GetObject(i));
        return result;
    }
}
