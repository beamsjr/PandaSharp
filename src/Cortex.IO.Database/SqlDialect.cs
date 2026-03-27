namespace Cortex.IO.Database;

/// <summary>
/// SQL dialect abstraction for database-specific syntax.
/// </summary>
public abstract class SqlDialect
{
    public abstract string QuoteIdentifier(string name);
    public abstract string LimitClause(int count);
    public abstract string OffsetClause(int offset, int limit);
    public virtual string ParameterPrefix => "@";
    public virtual string CastSyntax(string expr, string targetType) => $"CAST({expr} AS {targetType})";
}

public class PostgresDialect : SqlDialect
{
    public override string QuoteIdentifier(string name) => $"\"{name}\"";
    public override string LimitClause(int count) => $"LIMIT {count}";
    public override string OffsetClause(int offset, int limit) => $"LIMIT {limit} OFFSET {offset}";
    public override string ParameterPrefix => "@";
}

public class SqlServerDialect : SqlDialect
{
    public override string QuoteIdentifier(string name) => $"[{name}]";
    public override string LimitClause(int count) => $"TOP {count}";
    public override string OffsetClause(int offset, int limit) =>
        $"OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY";
}

public class SqliteDialect : SqlDialect
{
    public override string QuoteIdentifier(string name) => $"\"{name}\"";
    public override string LimitClause(int count) => $"LIMIT {count}";
    public override string OffsetClause(int offset, int limit) => $"LIMIT {limit} OFFSET {offset}";
}

public class MySqlDialect : SqlDialect
{
    public override string QuoteIdentifier(string name) => $"`{name}`";
    public override string LimitClause(int count) => $"LIMIT {count}";
    public override string OffsetClause(int offset, int limit) => $"LIMIT {limit} OFFSET {offset}";
}
