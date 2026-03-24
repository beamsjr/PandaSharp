using System.Data;
using System.Data.Common;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Expressions;

namespace PandaSharp.IO.Database;

/// <summary>
/// Scan a database table into a PandaSharp DataFrame with expression push-down.
/// Translates filter/select/sort/limit expressions to SQL before execution.
/// </summary>
public class DatabaseScanner
{
    private readonly DbConnection _connection;
    private readonly string _tableName;
    private readonly SqlDialect _dialect;
#pragma warning disable CS0414 // reserved for future Dispose pattern
    private readonly bool _ownsConnection;
#pragma warning restore CS0414

    public DatabaseScanner(DbConnection connection, string tableName, SqlDialect? dialect = null)
    {
        _connection = connection;
        _tableName = tableName;
        _dialect = dialect ?? DetectDialect(connection);
        _ownsConnection = false;
    }

    public DatabaseScanner(string connectionString, string tableName,
        Func<string, DbConnection> connectionFactory, SqlDialect? dialect = null)
    {
        _connection = connectionFactory(connectionString);
        _tableName = tableName;
        _dialect = dialect ?? DetectDialect(_connection);
        _ownsConnection = true;
    }

    /// <summary>
    /// Execute a push-down query and return results as a DataFrame.
    /// </summary>
    public DataFrame Scan(
        string[]? selectColumns = null,
        Expr? filter = null,
        (string Column, bool Ascending)[]? orderBy = null,
        int? limit = null,
        bool distinct = false,
        CancellationToken cancellationToken = default)
    {
        var generator = new SqlGenerator(_dialect);
        var (sql, parameters) = generator.GenerateFromExpressions(
            _tableName, selectColumns, filter, null, orderBy, limit, null, distinct);

        return ExecuteQuery(sql, parameters, cancellationToken);
    }

    /// <summary>
    /// Execute a raw SQL query and return results as a DataFrame.
    /// </summary>
    public DataFrame ScanSql(string sql, CancellationToken cancellationToken = default)
    {
        return ExecuteQuery(sql, [], cancellationToken);
    }

    /// <summary>
    /// Execute a push-down query with GroupBy + aggregation.
    /// </summary>
    public DataFrame ScanGroupBy(
        string[] groupByColumns,
        (string Column, string AggFunc, string Alias)[] aggregations,
        Expr? filter = null,
        (string Column, bool Ascending)[]? orderBy = null,
        int? limit = null,
        CancellationToken cancellationToken = default)
    {
        var generator = new SqlGenerator(_dialect);
        var sb = new System.Text.StringBuilder();

        sb.Append("SELECT ");
        var selectParts = new List<string>();
        foreach (var col in groupByColumns)
            selectParts.Add(_dialect.QuoteIdentifier(col));
        foreach (var (col, func, alias) in aggregations)
            selectParts.Add($"{func}({_dialect.QuoteIdentifier(col)}) AS {_dialect.QuoteIdentifier(alias)}");
        sb.Append(string.Join(", ", selectParts));

        sb.Append($" FROM {_dialect.QuoteIdentifier(_tableName)}");

        var parameters = new List<(string Name, object? Value)>();
        if (filter is not null)
        {
            sb.Append(" WHERE ");
            sb.Append(generator.ExprToSql(filter));
        }

        sb.Append(" GROUP BY ");
        sb.Append(string.Join(", ", groupByColumns.Select(c => _dialect.QuoteIdentifier(c))));

        if (orderBy is not null && orderBy.Length > 0)
        {
            sb.Append(" ORDER BY ");
            sb.Append(string.Join(", ", orderBy.Select(o =>
                $"{_dialect.QuoteIdentifier(o.Column)} {(o.Ascending ? "ASC" : "DESC")}")));
        }

        if (limit.HasValue)
            sb.Append($" {_dialect.LimitClause(limit.Value)}");

        return ExecuteQuery(sb.ToString(), parameters, cancellationToken);
    }

    /// <summary>
    /// Create a LazyFrame backed by this database table.
    /// Operations (Filter, Select, Sort, Head, GroupBy) are pushed down to SQL
    /// when Collect() is called, minimizing data transfer.
    /// </summary>
    public PandaSharp.Lazy.LazyFrame Lazy()
    {
        var plan = new PandaSharp.Lazy.ExternalScanPlan(
            fullPlan => ExecutePushDown(fullPlan),
            $"db://{_tableName}");
        return new PandaSharp.Lazy.LazyFrame(plan);
    }

    /// <summary>
    /// Compile the full plan to SQL and execute against the database.
    /// Falls back to client-side execution for unsupported plan nodes.
    /// </summary>
    private DataFrame ExecutePushDown(PandaSharp.Lazy.LogicalPlan fullPlan)
    {
        var generator = new SqlGenerator(_dialect);
        try
        {
            var (sql, parameters) = generator.Generate(fullPlan, _tableName);
            return ExecuteQuery(sql, parameters, default);
        }
        catch (NotSupportedException)
        {
            // Plan contains nodes that can't be pushed to SQL (e.g., WithColumn with complex expressions).
            // Fall back: scan the full table, then execute the plan client-side.
            var fullTable = Scan();
            var rewrittenPlan = RewriteWithLocalScan(fullPlan, fullTable);
            return PandaSharp.Lazy.Executor.ExecuteLocal(rewrittenPlan);
        }
    }

    /// <summary>
    /// Replace the ExternalScanPlan leaf with a ScanPlan holding the materialized data.
    /// </summary>
    private static PandaSharp.Lazy.LogicalPlan RewriteWithLocalScan(
        PandaSharp.Lazy.LogicalPlan plan, DataFrame data)
    {
        return plan switch
        {
            PandaSharp.Lazy.ExternalScanPlan => new PandaSharp.Lazy.ScanPlan(data),
            PandaSharp.Lazy.SelectPlan s => s with { Input = RewriteWithLocalScan(s.Input, data) },
            PandaSharp.Lazy.FilterPlan f => f with { Input = RewriteWithLocalScan(f.Input, data) },
            PandaSharp.Lazy.SortPlan s => s with { Input = RewriteWithLocalScan(s.Input, data) },
            PandaSharp.Lazy.HeadPlan h => h with { Input = RewriteWithLocalScan(h.Input, data) },
            PandaSharp.Lazy.GroupByAggPlan g => g with { Input = RewriteWithLocalScan(g.Input, data) },
            PandaSharp.Lazy.WithColumnPlan w => w with { Input = RewriteWithLocalScan(w.Input, data) },
            _ => plan
        };
    }

    /// <summary>
    /// Create a LazyFrame that joins this table with another table in the same database.
    /// The join is pushed down to SQL when Collect() is called.
    /// </summary>
    public PandaSharp.Lazy.LazyFrame LazyJoin(string rightTableName, string onColumn,
        PandaSharp.Joins.JoinType how = PandaSharp.Joins.JoinType.Inner)
    {
        var leftPlan = new PandaSharp.Lazy.ExternalScanPlan(
            fullPlan => ExecutePushDown(fullPlan),
            $"db://{_tableName}");
        var rightPlan = new PandaSharp.Lazy.ExternalScanPlan(
            fullPlan => ExecutePushDown(fullPlan),
            $"db://{rightTableName}");
        var joinPlan = new PandaSharp.Lazy.JoinPlan(leftPlan, rightPlan, onColumn, how);

        // Wrap in an ExternalScanPlan that can push the whole thing down
        var executorPlan = new PandaSharp.Lazy.ExternalScanPlan(
            fullPlan => ExecutePushDown(fullPlan),
            $"db://{_tableName}+{rightTableName}");

        // Return a LazyFrame that, when Collect() is called, uses the join plan
        return new PandaSharp.Lazy.LazyFrame(joinPlan);
    }

    /// <summary>Get table schema (column names + types) without loading data.</summary>
    public DataFrame Schema()
    {
        var sql = $"SELECT * FROM {_dialect.QuoteIdentifier(_tableName)} WHERE 1=0";
        return ExecuteQuery(sql, [], default);
    }

    /// <summary>Get row count without loading data.</summary>
    public long Count(Expr? filter = null)
    {
        var generator = new SqlGenerator(_dialect);
        var sb = new System.Text.StringBuilder();
        sb.Append($"SELECT COUNT(*) FROM {_dialect.QuoteIdentifier(_tableName)}");

        if (filter is not null)
        {
            sb.Append(" WHERE ");
            sb.Append(generator.ExprToSql(filter));
        }

        EnsureOpen();
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = sb.ToString();

        // Add parameters from expression compilation
        foreach (var (name, value) in generator.Parameters)
        {
            var param = cmd.CreateParameter();
            param.ParameterName = name;
            param.Value = value ?? DBNull.Value;
            cmd.Parameters.Add(param);
        }

        return Convert.ToInt64(cmd.ExecuteScalar());
    }

    private DataFrame ExecuteQuery(string sql, List<(string Name, object? Value)> parameters,
        CancellationToken cancellationToken)
    {
        EnsureOpen();

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = sql;

        foreach (var (name, value) in parameters)
        {
            var param = cmd.CreateParameter();
            param.ParameterName = name;
            param.Value = value ?? DBNull.Value;
            cmd.Parameters.Add(param);
        }

        using var reader = cmd.ExecuteReader();
        return ReadToDataFrame(reader);
    }

    /// <summary>
    /// Stream results into a PandaSharp DataFrame using columnar assembly.
    /// </summary>
    private static DataFrame ReadToDataFrame(IDataReader reader)
    {
        int fieldCount = reader.FieldCount;
        var columnNames = new string[fieldCount];
        var columnTypes = new Type[fieldCount];
        var builders = new List<object?>[fieldCount];

        // Handle duplicate column names (e.g., from JOINs) by adding suffixes
        var nameCount = new Dictionary<string, int>();
        for (int c = 0; c < fieldCount; c++)
        {
            var name = reader.GetName(c);
            if (nameCount.TryGetValue(name, out int count))
            {
                nameCount[name] = count + 1;
                name = $"{name}_{count}";
            }
            else
            {
                nameCount[name] = 1;
            }
            columnNames[c] = name;
            columnTypes[c] = reader.GetFieldType(c);
            builders[c] = new List<object?>();
        }

        while (reader.Read())
        {
            for (int c = 0; c < fieldCount; c++)
                builders[c].Add(reader.IsDBNull(c) ? null : reader.GetValue(c));
        }

        var columns = new IColumn[fieldCount];
        for (int c = 0; c < fieldCount; c++)
            columns[c] = BuildColumn(columnNames[c], columnTypes[c], builders[c]);

        return new DataFrame(columns);
    }

    private static IColumn BuildColumn(string name, Type type, List<object?> values)
    {
        if (type == typeof(int)) return BuildTyped<int>(name, values);
        if (type == typeof(long)) return BuildTyped<long>(name, values);
        if (type == typeof(double)) return BuildTyped<double>(name, values);
        if (type == typeof(float)) return BuildTyped<float>(name, values);
        if (type == typeof(bool)) return BuildTyped<bool>(name, values);
        if (type == typeof(decimal))
        {
            var arr = new double?[values.Count];
            for (int i = 0; i < values.Count; i++)
                arr[i] = values[i] is null ? null : Convert.ToDouble(values[i]);
            return Column<double>.FromNullable(name, arr);
        }
        if (type == typeof(DateTime)) return BuildTyped<DateTime>(name, values);
        return new StringColumn(name, values.Select(v => v?.ToString()).ToArray());
    }

    private static Column<T> BuildTyped<T>(string name, List<object?> values) where T : struct
    {
        var arr = new T?[values.Count];
        for (int i = 0; i < values.Count; i++)
            arr[i] = values[i] is null ? null : (T)Convert.ChangeType(values[i]!, typeof(T));
        return Column<T>.FromNullable(name, arr);
    }

    private void EnsureOpen()
    {
        if (_connection.State != ConnectionState.Open)
            _connection.Open();
    }

    private static SqlDialect DetectDialect(DbConnection conn)
    {
        var typeName = conn.GetType().Name.ToLowerInvariant();
        if (typeName.Contains("npgsql") || typeName.Contains("postgres")) return new PostgresDialect();
        if (typeName.Contains("sqlserver") || typeName.Contains("sqlconnection")) return new SqlServerDialect();
        if (typeName.Contains("sqlite")) return new SqliteDialect();
        if (typeName.Contains("mysql")) return new MySqlDialect();
        return new PostgresDialect(); // default
    }
}
