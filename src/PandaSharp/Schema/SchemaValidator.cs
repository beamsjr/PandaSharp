using System.Text.RegularExpressions;
using PandaSharp.Column;

namespace PandaSharp.Schema;

/// <summary>
/// Fluent DataFrame schema validator — like Python's pandera.
/// Validates column types, nullability, value ranges, uniqueness, patterns, and custom rules.
///
/// Usage:
///   var schema = DataFrameSchema.Define()
///       .Column("id", type: typeof(int), unique: true, noNulls: true)
///       .Column("age", type: typeof(int), min: 0, max: 150)
///       .Column("email", type: typeof(string), pattern: @".+@.+\..+")
///       .Column("status", allowedValues: ["active", "inactive"])
///       .HasColumns("id", "age", "email", "status")
///       .MinRows(1);
///
///   var result = schema.Validate(df);
///   if (!result.IsValid) Console.WriteLine(result);
/// </summary>
public class DataFrameSchema
{
    private readonly List<ColumnRule> _columnRules = new();
    private readonly List<Func<DataFrame, ValidationError?>> _globalRules = new();

    public static DataFrameSchema Define() => new();

    /// <summary>Add a column validation rule.</summary>
    public DataFrameSchema Column(string name,
        Type? type = null,
        bool noNulls = false,
        bool unique = false,
        double? min = null,
        double? max = null,
        string? pattern = null,
        string[]? allowedValues = null,
        Func<IColumn, ValidationError?>? check = null)
    {
        _columnRules.Add(new ColumnRule
        {
            Name = name,
            ExpectedType = type,
            NoNulls = noNulls,
            Unique = unique,
            Min = min,
            Max = max,
            Pattern = pattern is not null ? new Regex(pattern) : null,
            AllowedValues = allowedValues is not null ? new HashSet<string>(allowedValues) : null,
            CustomCheck = check
        });
        return this;
    }

    /// <summary>Require the DataFrame to have exactly these columns (in any order).</summary>
    public DataFrameSchema HasColumns(params string[] requiredColumns)
    {
        var required = new HashSet<string>(requiredColumns);
        _globalRules.Add(df =>
        {
            var missing = required.Except(df.ColumnNames).ToList();
            if (missing.Count > 0)
                return new ValidationError("Schema", $"Missing required columns: {string.Join(", ", missing)}");
            return null;
        });
        return this;
    }

    /// <summary>Require no extra columns beyond what's defined.</summary>
    public DataFrameSchema NoExtraColumns()
    {
        _globalRules.Add(df =>
        {
            var defined = new HashSet<string>(_columnRules.Select(r => r.Name));
            var extra = df.ColumnNames.Where(n => !defined.Contains(n)).ToList();
            if (extra.Count > 0)
                return new ValidationError("Schema", $"Unexpected columns: {string.Join(", ", extra)}");
            return null;
        });
        return this;
    }

    /// <summary>Require a minimum number of rows.</summary>
    public DataFrameSchema MinRows(int min)
    {
        _globalRules.Add(df =>
            df.RowCount < min
                ? new ValidationError("Schema", $"Expected at least {min} rows, got {df.RowCount}")
                : null);
        return this;
    }

    /// <summary>Require a maximum number of rows.</summary>
    public DataFrameSchema MaxRows(int max)
    {
        _globalRules.Add(df =>
            df.RowCount > max
                ? new ValidationError("Schema", $"Expected at most {max} rows, got {df.RowCount}")
                : null);
        return this;
    }

    /// <summary>Require no duplicate rows.</summary>
    public DataFrameSchema NoDuplicateRows()
    {
        _globalRules.Add(df =>
        {
            var seen = new HashSet<long>();
            var cols = df.ColumnNames.Select(n => df[n]).ToArray();
            for (int r = 0; r < df.RowCount; r++)
            {
                long hash = 17;
                for (int c = 0; c < cols.Length; c++)
                    hash = hash * 31 + (cols[c].GetObject(r)?.GetHashCode() ?? 0);
                if (!seen.Add(hash))
                    return new ValidationError("Schema", $"Duplicate row detected at index {r}");
            }
            return null;
        });
        return this;
    }

    /// <summary>Add a custom global validation rule.</summary>
    public DataFrameSchema Check(string name, Func<DataFrame, bool> predicate, string? message = null)
    {
        _globalRules.Add(df =>
            !predicate(df)
                ? new ValidationError(name, message ?? $"Custom check '{name}' failed")
                : null);
        return this;
    }

    /// <summary>Validate a DataFrame against this schema.</summary>
    public ValidationResult Validate(DataFrame df)
    {
        var errors = new List<ValidationError>();

        // Global rules
        foreach (var rule in _globalRules)
        {
            var error = rule(df);
            if (error is not null) errors.Add(error);
        }

        // Column rules
        foreach (var rule in _columnRules)
        {
            if (!df.ColumnNames.Contains(rule.Name))
            {
                errors.Add(new ValidationError(rule.Name, $"Column '{rule.Name}' not found"));
                continue;
            }

            var col = df[rule.Name];
            errors.AddRange(ValidateColumn(col, rule));
        }

        return new ValidationResult(errors);
    }

    private static List<ValidationError> ValidateColumn(IColumn col, ColumnRule rule)
    {
        var errors = new List<ValidationError>();

        // Type check
        if (rule.ExpectedType is not null && col.DataType != rule.ExpectedType)
            errors.Add(new ValidationError(rule.Name,
                $"Expected type {rule.ExpectedType.Name}, got {col.DataType.Name}"));

        // Null check
        if (rule.NoNulls && col.NullCount > 0)
            errors.Add(new ValidationError(rule.Name,
                $"Contains {col.NullCount} null values (noNulls=true)"));

        // Uniqueness
        if (rule.Unique)
        {
            var seen = new HashSet<object?>();
            for (int i = 0; i < col.Length; i++)
            {
                if (col.IsNull(i)) continue;
                if (!seen.Add(col.GetObject(i)))
                {
                    errors.Add(new ValidationError(rule.Name,
                        $"Contains duplicate values (unique=true)"));
                    break;
                }
            }
        }

        // Min/Max range
        if (rule.Min.HasValue || rule.Max.HasValue)
        {
            for (int i = 0; i < col.Length; i++)
            {
                if (col.IsNull(i)) continue;
                double val;
                try { val = Convert.ToDouble(col.GetObject(i)); }
                catch { continue; }

                if (rule.Min.HasValue && val < rule.Min.Value)
                {
                    errors.Add(new ValidationError(rule.Name,
                        $"Value {val} at index {i} is below minimum {rule.Min.Value}"));
                    break;
                }
                if (rule.Max.HasValue && val > rule.Max.Value)
                {
                    errors.Add(new ValidationError(rule.Name,
                        $"Value {val} at index {i} is above maximum {rule.Max.Value}"));
                    break;
                }
            }
        }

        // Regex pattern
        if (rule.Pattern is not null)
        {
            for (int i = 0; i < col.Length; i++)
            {
                if (col.IsNull(i)) continue;
                var str = col.GetObject(i)?.ToString() ?? "";
                if (!rule.Pattern.IsMatch(str))
                {
                    errors.Add(new ValidationError(rule.Name,
                        $"Value '{str}' at index {i} doesn't match pattern '{rule.Pattern}'"));
                    break;
                }
            }
        }

        // Allowed values
        if (rule.AllowedValues is not null)
        {
            for (int i = 0; i < col.Length; i++)
            {
                if (col.IsNull(i)) continue;
                var str = col.GetObject(i)?.ToString() ?? "";
                if (!rule.AllowedValues.Contains(str))
                {
                    errors.Add(new ValidationError(rule.Name,
                        $"Value '{str}' at index {i} is not in allowed values: [{string.Join(", ", rule.AllowedValues)}]"));
                    break;
                }
            }
        }

        // Custom check
        if (rule.CustomCheck is not null)
        {
            var error = rule.CustomCheck(col);
            if (error is not null) errors.Add(error);
        }

        return errors;
    }

    private class ColumnRule
    {
        public string Name { get; init; } = "";
        public Type? ExpectedType { get; init; }
        public bool NoNulls { get; init; }
        public bool Unique { get; init; }
        public double? Min { get; init; }
        public double? Max { get; init; }
        public Regex? Pattern { get; init; }
        public HashSet<string>? AllowedValues { get; init; }
        public Func<IColumn, ValidationError?>? CustomCheck { get; init; }
    }
}

/// <summary>A single validation error with column name and description.</summary>
public record ValidationError(string Column, string Message);

/// <summary>Result of DataFrame validation.</summary>
public class ValidationResult
{
    public IReadOnlyList<ValidationError> Errors { get; }
    public bool IsValid => Errors.Count == 0;
    public int ErrorCount => Errors.Count;

    public ValidationResult(List<ValidationError> errors) => Errors = errors.AsReadOnly();

    public override string ToString()
    {
        if (IsValid) return "Validation passed.";
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Validation failed ({ErrorCount} errors):");
        foreach (var err in Errors)
            sb.AppendLine($"  [{err.Column}] {err.Message}");
        return sb.ToString();
    }
}

/// <summary>Extension method for DataFrame validation.</summary>
public static class SchemaValidationExtensions
{
    /// <summary>Validate this DataFrame against a schema. Throws if invalid.</summary>
    public static DataFrame ValidateSchema(this DataFrame df, DataFrameSchema schema)
    {
        var result = schema.Validate(df);
        if (!result.IsValid)
            throw new InvalidDataException(result.ToString());
        return df;
    }
}
