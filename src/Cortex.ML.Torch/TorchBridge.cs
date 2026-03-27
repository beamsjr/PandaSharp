using Cortex;
using Cortex.Column;
using Cortex.ML.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace Cortex.ML.Torch;

/// <summary>
/// Bridge between Cortex DataFrames/Tensors and TorchSharp tensors.
/// Enables Cortex data pipeline → TorchSharp model training.
/// </summary>
public static class TorchBridge
{
    /// <summary>
    /// Convert selected DataFrame columns to a TorchSharp float tensor.
    /// Numeric column values are read via typed fast paths (Column&lt;double&gt;, Column&lt;float&gt;,
    /// Column&lt;int&gt;) to avoid boxing overhead from GetObject().
    /// </summary>
    public static Tensor ToTorchTensor(this DataFrame df, params string[] columns)
    {
        var cols = columns.Length > 0 ? columns
            : df.ColumnNames.Where(n => TypeHelpers.IsNumeric(df[n].DataType)).ToArray();

        int rows = df.RowCount;
        int ncols = cols.Length;
        var data = new float[rows * ncols];

        for (int c = 0; c < ncols; c++)
        {
            var col = df[cols[c]];
            for (int r = 0; r < rows; r++)
                data[r * ncols + c] = col.IsNull(r) ? 0f : (float)TypeHelpers.GetDouble(col, r);
        }

        return torch.tensor(data, [rows, ncols]);
    }

    /// <summary>
    /// Convert a single DataFrame column to a 1D TorchSharp float tensor.
    /// Uses typed fast paths to avoid boxing. Null values are replaced with 0.
    /// </summary>
    public static Tensor ToTorchTensor1D(this DataFrame df, string column)
    {
        var col = df[column];
        var data = new float[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
            data[i] = col.IsNull(i) ? 0f : (float)TypeHelpers.GetDouble(col, i);
        return torch.tensor(data);
    }

    /// <summary>
    /// Convert a Cortex Tensor to a TorchSharp tensor.
    /// </summary>
    public static Tensor ToTorchTensor(this Tensor<float> tensor)
    {
        var longShape = tensor.Shape.Select(s => (long)s).ToArray();
        return torch.tensor(tensor.ToArray(), longShape);
    }

    /// <summary>
    /// Convert a Cortex double Tensor to a TorchSharp float tensor.
    /// Note: This involves precision loss from double (64-bit) to float (32-bit).
    /// Values outside the float range will be clamped to float.MaxValue/MinValue,
    /// and precision beyond ~7 significant digits will be lost.
    /// </summary>
    public static Tensor ToTorchTensor(this Tensor<double> tensor)
    {
        var floatData = new float[tensor.Length];
        var span = tensor.Span;
        for (int i = 0; i < tensor.Length; i++)
            floatData[i] = (float)span[i];
        var longShape = tensor.Shape.Select(s => (long)s).ToArray();
        return torch.tensor(floatData, longShape);
    }

    /// <summary>
    /// Convert a TorchSharp tensor back to a Cortex DataFrame.
    /// Requires 2D tensor.
    /// </summary>
    public static DataFrame ToDataFrame(this Tensor tensor, string[]? columnNames = null)
    {
        if (tensor.dim() != 2)
            throw new ArgumentException("ToDataFrame requires a 2D tensor.");

        int rows = (int)tensor.shape[0];
        int cols = (int)tensor.shape[1];
        columnNames ??= Enumerable.Range(0, cols).Select(i => $"col_{i}").ToArray();

        var cpuTensor = tensor.cpu().to(ScalarType.Float64);
        var columns = new List<IColumn>();

        for (int c = 0; c < cols; c++)
        {
            var colTensor = cpuTensor[.., c].data<double>().ToArray();
            columns.Add(new Column<double>(columnNames[c], colTensor));
        }

        return new DataFrame(columns);
    }

    /// <summary>
    /// Batched DataLoader that yields TorchSharp tensors.
    /// </summary>
    public static IEnumerable<(Tensor Features, Tensor Labels)> ToTorchDataLoader(
        this DataFrame df, string[] features, string label,
        int batchSize = 32, bool shuffle = true, int? seed = null)
    {
        var loader = new DataFrameDataLoader(df, features, label, batchSize, shuffle, seed);
        foreach (var (featureTensor, labelTensor) in loader)
        {
            yield return (
                featureTensor.ToTorchTensor(),
                labelTensor.ToTorchTensor()
            );
        }
    }

}
