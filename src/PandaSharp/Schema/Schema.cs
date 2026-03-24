namespace PandaSharp.Schema;

public record ColumnSchema(string Name, Type DataType, bool Nullable = true);

public class Schema
{
    private readonly List<ColumnSchema> _columns;

    public Schema(params ColumnSchema[] columns)
    {
        _columns = new List<ColumnSchema>(columns);
    }

    public IReadOnlyList<ColumnSchema> Columns => _columns;

    /// <summary>
    /// Validate that a DataFrame matches this schema. Throws on mismatch.
    /// </summary>
    public void Validate(DataFrame df)
    {
        var errors = new List<string>();

        foreach (var expected in _columns)
        {
            if (!df.ColumnNames.Contains(expected.Name))
            {
                errors.Add($"Missing column: '{expected.Name}'");
                continue;
            }

            var actual = df[expected.Name];
            if (actual.DataType != expected.DataType)
                errors.Add($"Column '{expected.Name}': expected type {expected.DataType.Name}, got {actual.DataType.Name}");

            if (!expected.Nullable && actual.NullCount > 0)
                errors.Add($"Column '{expected.Name}': contains {actual.NullCount} nulls but schema requires non-nullable");
        }

        if (errors.Count > 0)
            throw new SchemaValidationException(errors);
    }

    /// <summary>
    /// Check if a DataFrame matches this schema without throwing.
    /// </summary>
    public bool IsValid(DataFrame df)
    {
        try { Validate(df); return true; }
        catch (SchemaValidationException) { return false; }
    }
}

public class SchemaValidationException : Exception
{
    public IReadOnlyList<string> Errors { get; }

    public SchemaValidationException(List<string> errors)
        : base($"Schema validation failed:\n{string.Join("\n", errors.Select(e => $"  - {e}"))}")
    {
        Errors = errors;
    }
}

public static class SchemaExtensions
{
    public static void ValidateSchema(this DataFrame df, Schema schema) => schema.Validate(df);
    public static bool MatchesSchema(this DataFrame df, Schema schema) => schema.IsValid(df);
}
