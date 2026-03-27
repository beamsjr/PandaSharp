using Apache.Arrow;
using Apache.Arrow.Ipc;
using Cortex.Column;
using Cortex.IO;

namespace Cortex.Storage;

/// <summary>
/// A DataFrame spilled to disk as an Arrow IPC file.
/// Columns are lazily loaded on demand, enabling out-of-core processing
/// for datasets larger than available RAM.
///
/// Usage:
///   var spilled = df.Spill("/tmp/data.arrow"); // writes to disk, frees RAM
///   var col = spilled["Value"]; // loads only this column from disk
///   var materialized = spilled.Materialize(); // loads everything back to RAM
///   spilled.Dispose(); // deletes the temp file
/// </summary>
public class SpilledDataFrame : IDisposable
{
    private readonly string _path;
    private readonly bool _ownsFile;
    private readonly string[] _columnNames;
    private readonly Dictionary<string, int> _columnIndex;
    private readonly Dictionary<string, IColumn> _cache = new();
    private bool _disposed;

    /// <summary>Number of rows in the spilled DataFrame.</summary>
    public int RowCount { get; }
    /// <summary>Number of columns.</summary>
    public int ColumnCount => _columnNames.Length;
    /// <summary>Column names.</summary>
    public IReadOnlyList<string> ColumnNames => _columnNames;
    /// <summary>Path to the Arrow IPC file on disk.</summary>
    public string FilePath => _path;
    /// <summary>Number of columns currently loaded in memory.</summary>
    public int CachedColumnCount => _cache.Count;

    private SpilledDataFrame(string path, bool ownsFile, string[] columnNames, int rowCount)
    {
        _path = path;
        _ownsFile = ownsFile;
        _columnNames = columnNames;
        _columnIndex = new Dictionary<string, int>();
        for (int i = 0; i < columnNames.Length; i++)
            _columnIndex[columnNames[i]] = i;
        RowCount = rowCount;
    }

    /// <summary>
    /// Spill a DataFrame to disk. The source DataFrame can then be GC'd to free RAM.
    /// </summary>
    public static SpilledDataFrame Spill(DataFrame df, string path)
    {
        ArrowIpcWriter.Write(df, path);
        return new SpilledDataFrame(path, ownsFile: true,
            df.ColumnNames.ToArray(), df.RowCount);
    }

    /// <summary>
    /// Open an existing Arrow IPC file as a SpilledDataFrame without loading data.
    /// </summary>
    public static SpilledDataFrame Open(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new ArrowFileReader(stream);
        var schema = reader.Schema;

        // Read row count from first batch without materializing columns
        var batch = reader.ReadNextRecordBatch();
        int rowCount = batch?.Length ?? 0;

        var columnNames = schema.FieldsList.Select(f => f.Name).ToArray();
        return new SpilledDataFrame(path, ownsFile: false, columnNames, rowCount);
    }

    /// <summary>Access a column by name. Loads it from disk on first access, then caches.</summary>
    public IColumn this[string columnName]
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_cache.TryGetValue(columnName, out var cached))
                return cached;

            if (!_columnIndex.TryGetValue(columnName, out int colIdx))
                throw new KeyNotFoundException($"Column '{columnName}' not found.");

            var column = LoadColumn(colIdx);
            _cache[columnName] = column;
            return column;
        }
    }

    /// <summary>Check if a column is currently loaded in memory.</summary>
    public bool IsLoaded(string columnName) => _cache.ContainsKey(columnName);

    /// <summary>Evict a cached column from memory (it will be reloaded from disk on next access).</summary>
    public void Evict(string columnName) => _cache.Remove(columnName);

    /// <summary>Evict all cached columns.</summary>
    public void EvictAll() => _cache.Clear();

    /// <summary>
    /// Load all columns and return a fully materialized in-memory DataFrame.
    /// </summary>
    public DataFrame Materialize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return ArrowIpcReader.Read(_path);
    }

    /// <summary>
    /// Materialize only the specified columns into a DataFrame.
    /// </summary>
    public DataFrame Materialize(params string[] columns)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var cols = columns.Select(name => this[name].Clone()).ToList();
        return new DataFrame(cols);
    }

    /// <summary>
    /// Select specific columns and return a new SpilledDataFrame backed by a new file.
    /// Only the selected columns are written, reducing disk footprint.
    /// </summary>
    public SpilledDataFrame Select(string outputPath, params string[] columns)
    {
        var df = Materialize(columns);
        return Spill(df, outputPath);
    }

    /// <summary>
    /// Filter rows and return a new SpilledDataFrame backed by a new file.
    /// </summary>
    public SpilledDataFrame Filter(string outputPath, Func<DataFrame, bool[]> predicate)
    {
        var df = Materialize();
        var mask = predicate(df);
        var filtered = df.Filter(mask);
        return Spill(filtered, outputPath);
    }

    /// <summary>Head: materialize first N rows.</summary>
    public DataFrame Head(int count = 5)
    {
        var df = Materialize();
        return df.Head(count);
    }

    /// <summary>Tail: materialize last N rows.</summary>
    public DataFrame Tail(int count = 5)
    {
        var df = Materialize();
        return df.Tail(count);
    }

    private IColumn LoadColumn(int columnIndex)
    {
        using var stream = File.OpenRead(_path);
        using var reader = new ArrowFileReader(stream);
        var batch = reader.ReadNextRecordBatch();
        if (batch is null)
            throw new InvalidDataException("Arrow IPC file contains no record batches.");

        var field = batch.Schema.GetFieldByIndex(columnIndex);
        var array = batch.Column(columnIndex);
        return ArrowIpcReader.FromRecordBatch(
            new RecordBatch(
                new Apache.Arrow.Schema(new[] { field }, null),
                new[] { array },
                batch.Length)
        )[field.Name];
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _cache.Clear();
        if (_ownsFile && File.Exists(_path))
            File.Delete(_path);
    }
}

/// <summary>Extension methods for spilling DataFrames to disk.</summary>
public static class SpillExtensions
{
    /// <summary>
    /// Spill this DataFrame to an Arrow IPC file on disk.
    /// Returns a SpilledDataFrame that lazily loads columns on demand.
    /// The file is auto-deleted when the SpilledDataFrame is disposed.
    /// </summary>
    public static SpilledDataFrame Spill(this DataFrame df, string? path = null)
    {
        path ??= Path.Combine(Path.GetTempPath(), $"pandasharp_spill_{Guid.NewGuid():N}.arrow");
        return SpilledDataFrame.Spill(df, path);
    }
}
