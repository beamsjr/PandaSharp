using Apache.Arrow;
using Apache.Arrow.Ipc;
using Apache.Arrow.Types;
using PandaSharp.Column;

namespace PandaSharp.IO;

public static class ArrowIpcReader
{
    public static DataFrame Read(string path)
    {
        using var stream = File.OpenRead(path);
        return Read(stream);
    }

    public static DataFrame Read(Stream stream)
    {
        using var reader = new ArrowFileReader(stream, leaveOpen: true);
        var batch = reader.ReadNextRecordBatch();
        if (batch is null) return new DataFrame();

        return FromRecordBatch(batch);
    }

    public static DataFrame FromRecordBatch(RecordBatch batch)
    {
        var columns = new List<IColumn>();

        for (int c = 0; c < batch.ColumnCount; c++)
        {
            var field = batch.Schema.GetFieldByIndex(c);
            var array = batch.Column(c);
            columns.Add(FromArrowArray(field.Name, array));
        }

        return new DataFrame(columns);
    }

    private static IColumn FromArrowArray(string name, IArrowArray array)
    {
        return array switch
        {
            Int32Array a => FromInt32(name, a),
            Int64Array a => FromInt64(name, a),
            DoubleArray a => FromDouble(name, a),
            FloatArray a => FromFloat(name, a),
            BooleanArray a => FromBool(name, a),
            StringArray a => FromString(name, a),
            _ => FromStringFallback(name, array)
        };
    }

    private static Column<int> FromInt32(string name, Int32Array array)
    {
        var values = new int?[array.Length];
        for (int i = 0; i < array.Length; i++)
            values[i] = array.IsNull(i) ? null : array.GetValue(i);
        return Column<int>.FromNullable(name, values);
    }

    private static Column<long> FromInt64(string name, Int64Array array)
    {
        var values = new long?[array.Length];
        for (int i = 0; i < array.Length; i++)
            values[i] = array.IsNull(i) ? null : array.GetValue(i);
        return Column<long>.FromNullable(name, values);
    }

    private static Column<double> FromDouble(string name, DoubleArray array)
    {
        var values = new double?[array.Length];
        for (int i = 0; i < array.Length; i++)
            values[i] = array.IsNull(i) ? null : array.GetValue(i);
        return Column<double>.FromNullable(name, values);
    }

    private static Column<float> FromFloat(string name, FloatArray array)
    {
        var values = new float?[array.Length];
        for (int i = 0; i < array.Length; i++)
            values[i] = array.IsNull(i) ? null : array.GetValue(i);
        return Column<float>.FromNullable(name, values);
    }

    private static Column<bool> FromBool(string name, BooleanArray array)
    {
        var values = new bool?[array.Length];
        for (int i = 0; i < array.Length; i++)
            values[i] = array.IsNull(i) ? null : array.GetValue(i);
        return Column<bool>.FromNullable(name, values);
    }

    private static StringColumn FromString(string name, StringArray array)
    {
        var values = new string?[array.Length];
        for (int i = 0; i < array.Length; i++)
            values[i] = array.IsNull(i) ? null : array.GetString(i);
        return new StringColumn(name, values);
    }

    private static StringColumn FromStringFallback(string name, IArrowArray array)
    {
        var values = new string?[array.Length];
        for (int i = 0; i < array.Length; i++)
            values[i] = array.IsNull(i) ? null : array.GetType().GetMethod("GetString")?.Invoke(array, [i]) as string;
        return new StringColumn(name, values);
    }
}
