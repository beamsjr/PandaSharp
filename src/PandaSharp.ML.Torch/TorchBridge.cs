using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Tensors;
using TorchSharp;
using static TorchSharp.torch;

namespace PandaSharp.ML.Torch;

/// <summary>
/// Bridge between PandaSharp DataFrames/Tensors and TorchSharp tensors.
/// Enables PandaSharp data pipeline → TorchSharp model training.
/// </summary>
public static class TorchBridge
{
    /// <summary>
    /// Convert selected DataFrame columns to a TorchSharp float tensor.
    /// </summary>
    public static Tensor ToTorchTensor(this DataFrame df, params string[] columns)
    {
        var cols = columns.Length > 0 ? columns
            : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();

        int rows = df.RowCount;
        int ncols = cols.Length;
        var data = new float[rows * ncols];

        for (int c = 0; c < ncols; c++)
        {
            var col = df[cols[c]];
            for (int r = 0; r < rows; r++)
                data[r * ncols + c] = col.IsNull(r) ? 0f : Convert.ToSingle(col.GetObject(r));
        }

        return torch.tensor(data, [rows, ncols]);
    }

    /// <summary>
    /// Convert a single DataFrame column to a 1D TorchSharp float tensor.
    /// </summary>
    public static Tensor ToTorchTensor1D(this DataFrame df, string column)
    {
        var col = df[column];
        var data = new float[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
            data[i] = col.IsNull(i) ? 0f : Convert.ToSingle(col.GetObject(i));
        return torch.tensor(data);
    }

    /// <summary>
    /// Convert a PandaSharp Tensor to a TorchSharp tensor.
    /// </summary>
    public static Tensor ToTorchTensor(this Tensor<float> tensor)
    {
        var longShape = tensor.Shape.Select(s => (long)s).ToArray();
        return torch.tensor(tensor.ToArray(), longShape);
    }

    /// <summary>
    /// Convert a PandaSharp double Tensor to a TorchSharp float tensor.
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
    /// Convert a TorchSharp tensor back to a PandaSharp DataFrame.
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

    private static bool IsNumeric(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
