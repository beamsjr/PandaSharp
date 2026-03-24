using Apache.Arrow;
using Apache.Arrow.Ipc;
using Apache.Arrow.Types;
using PandaSharp.Column;
using ArrowSchema = Apache.Arrow.Schema;

namespace PandaSharp.IO;

public static class ArrowIpcWriter
{
    public static void Write(DataFrame df, string path)
    {
        using var stream = File.Create(path);
        Write(df, stream);
    }

    public static void Write(DataFrame df, Stream stream)
    {
        var schema = BuildArrowSchema(df);
        var batch = BuildRecordBatch(df, schema);

        using var writer = new ArrowFileWriter(stream, schema, leaveOpen: true);
        writer.WriteRecordBatch(batch);
        writer.WriteEnd();
    }

    private static ArrowSchema BuildArrowSchema(DataFrame df)
    {
        var fields = new List<Field>();
        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            var arrowType = ToArrowType(col.DataType);
            fields.Add(new Field(name, arrowType, col.NullCount > 0));
        }
        return new ArrowSchema(fields, null);
    }

    private static RecordBatch BuildRecordBatch(DataFrame df, ArrowSchema schema)
    {
        var arrays = new List<IArrowArray>();

        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            arrays.Add(BuildArrowArray(col));
        }

        return new RecordBatch(schema, arrays, df.RowCount);
    }

    private static IArrowArray BuildArrowArray(IColumn col)
    {
        if (col is StringColumn sc)
        {
            var builder = new StringArray.Builder();
            for (int i = 0; i < sc.Length; i++)
            {
                if (sc.IsNull(i)) builder.AppendNull();
                else builder.Append(sc[i]!);
            }
            return builder.Build();
        }

        if (col.DataType == typeof(int)) return BuildPrimitive<int>(col);
        if (col.DataType == typeof(long)) return BuildPrimitive<long>(col);
        if (col.DataType == typeof(double)) return BuildPrimitive<double>(col);
        if (col.DataType == typeof(float)) return BuildPrimitive<float>(col);
        if (col.DataType == typeof(bool)) return BuildBoolArray(col);

        // Fallback: convert to string
        var sb = new StringArray.Builder();
        for (int i = 0; i < col.Length; i++)
        {
            if (col.IsNull(i)) sb.AppendNull();
            else sb.Append(col.GetObject(i)?.ToString() ?? "");
        }
        return sb.Build();
    }

    private static IArrowArray BuildPrimitive<T>(IColumn col) where T : struct
    {
        var typed = (Column<T>)col;
        var span = typed.Values;
        var builder = new ArrowBuffer.Builder<T>();
        var validityBuilder = new ArrowBuffer.BitmapBuilder();

        for (int i = 0; i < col.Length; i++)
        {
            builder.Append(span[i]);
            validityBuilder.Append(!col.IsNull(i));
        }

        var data = new ArrayData(
            ToArrowType(typeof(T)),
            col.Length,
            col.NullCount,
            0,
            [validityBuilder.Build(), builder.Build()]);

        if (typeof(T) == typeof(int)) return new Int32Array(data);
        if (typeof(T) == typeof(long)) return new Int64Array(data);
        if (typeof(T) == typeof(double)) return new DoubleArray(data);
        if (typeof(T) == typeof(float)) return new FloatArray(data);

        throw new NotSupportedException($"Unsupported Arrow type: {typeof(T)}");
    }

    private static IArrowArray BuildBoolArray(IColumn col)
    {
        var builder = new BooleanArray.Builder();
        for (int i = 0; i < col.Length; i++)
        {
            if (col.IsNull(i)) builder.AppendNull();
            else builder.Append((bool)col.GetObject(i)!);
        }
        return builder.Build();
    }

    private static IArrowType ToArrowType(Type type)
    {
        if (type == typeof(int)) return Int32Type.Default;
        if (type == typeof(long)) return Int64Type.Default;
        if (type == typeof(double)) return DoubleType.Default;
        if (type == typeof(float)) return FloatType.Default;
        if (type == typeof(bool)) return BooleanType.Default;
        if (type == typeof(string)) return StringType.Default;
        if (type == typeof(DateTime)) return StringType.Default; // serialize as string for now
        return StringType.Default;
    }
}
