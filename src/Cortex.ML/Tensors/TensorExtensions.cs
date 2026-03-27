using System.Numerics;
using Cortex;
using Cortex.Column;

namespace Cortex.ML.Tensors;

public static class TensorExtensions
{
    /// <summary>Convert a typed column to a 1D tensor.</summary>
    public static Tensor<T> AsTensor<T>(this Column<T> column) where T : struct, INumber<T>
        => new(column);

    /// <summary>Convert selected numeric columns to a 2D tensor (rows × cols).</summary>
    public static Tensor<double> ToTensor(this DataFrame df, params string[] columns)
    {
        var cols = columns.Length > 0 ? columns : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();
        int rows = df.RowCount;
        int ncols = cols.Length;
        var data = new double[rows * ncols];

        for (int c = 0; c < ncols; c++)
        {
            var col = df[cols[c]];
            for (int r = 0; r < rows; r++)
                data[r * ncols + c] = TypeHelpers.GetDouble(col, r);
        }

        return new Tensor<double>(data, rows, ncols);
    }

    /// <summary>Convert selected columns to a typed 2D tensor.</summary>
    public static Tensor<T> ToTensor<T>(this DataFrame df, params string[] columns) where T : struct, INumber<T>
    {
        var cols = columns.Length > 0 ? columns : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();
        int rows = df.RowCount;
        int ncols = cols.Length;
        var data = new T[rows * ncols];

        for (int c = 0; c < ncols; c++)
        {
            var col = df[cols[c]];
            for (int r = 0; r < rows; r++)
                data[r * ncols + c] = col.IsNull(r) ? T.Zero : T.CreateChecked(TypeHelpers.GetDouble(col, r));
        }

        return new Tensor<T>(data, rows, ncols);
    }

    /// <summary>Convert a 1D tensor back to a Column.</summary>
    public static Column<T> ToColumn<T>(this Tensor<T> tensor, string name) where T : struct, INumber<T>
    {
        if (tensor.Rank != 1)
            throw new ArgumentException("ToColumn requires a 1D tensor.");
        return new Column<T>(name, tensor.ToArray());
    }

    /// <summary>Convert a 2D tensor back to a DataFrame with given column names.</summary>
    public static DataFrame ToDataFrame(this Tensor<double> tensor, string[]? columnNames = null)
    {
        if (tensor.Rank != 2)
            throw new ArgumentException("ToDataFrame requires a 2D tensor.");
        int rows = tensor.Shape[0], cols = tensor.Shape[1];
        columnNames ??= Enumerable.Range(0, cols).Select(i => $"col_{i}").ToArray();

        var columns = new IColumn[cols];
        for (int c = 0; c < cols; c++)
        {
            var data = new double[rows];
            for (int r = 0; r < rows; r++)
                data[r] = tensor[r, c];
            columns[c] = new Column<double>(columnNames[c], data);
        }

        return new DataFrame(columns);
    }

    private static bool IsNumeric(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
