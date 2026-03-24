namespace PandaSharp.Interop;

public static class FromRecordsExtensions
{
    /// <summary>
    /// Create a DataFrame from anonymous objects or POCOs.
    /// Usage: DataFrame.FromRecords(new[] { new { Name = "Alice", Age = 25 } })
    /// </summary>
    public static DataFrame FromRecords<T>(IEnumerable<T> records) =>
        LinqExtensions.FromEnumerable(records);
}
