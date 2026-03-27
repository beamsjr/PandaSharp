using System.Data.Common;
using Cortex;
using Cortex.Expressions;

namespace Cortex.IO.Database;

/// <summary>
/// Extension methods for scanning databases into DataFrames.
/// </summary>
public static class DataFrameDatabaseExtensions
{
    /// <summary>
    /// Scan a database table using an existing connection.
    /// </summary>
    public static DatabaseScanner ScanDatabase(DbConnection connection, string tableName, SqlDialect? dialect = null)
        => new(connection, tableName, dialect);

    /// <summary>
    /// Execute a raw SQL query against a connection and return a DataFrame.
    /// </summary>
    public static DataFrame ReadSql(DbConnection connection, string sql)
    {
        var scanner = new DatabaseScanner(connection, "__raw__");
        return scanner.ScanSql(sql);
    }

    /// <summary>
    /// Write a DataFrame to a database table via batched parameterized INSERT.
    /// </summary>
    /// <param name="batchSize">Number of rows per INSERT statement (default 1000).</param>
    public static void ToSql(this DataFrame df, DbConnection connection, string tableName,
        SqlDialect? dialect = null, bool createTable = false, int batchSize = 1000)
    {
        dialect ??= new PostgresDialect();
        if (batchSize < 1) throw new ArgumentOutOfRangeException(nameof(batchSize));

        if (connection.State != System.Data.ConnectionState.Open)
            connection.Open();

        if (createTable)
            CreateTableFromSchema(df, connection, tableName, dialect);

        InsertRowsBatched(df, connection, tableName, dialect, batchSize);
    }

    private static void CreateTableFromSchema(DataFrame df, DbConnection conn, string tableName, SqlDialect dialect)
    {
        var cols = new List<string>();
        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            var sqlType = col.DataType switch
            {
                Type t when t == typeof(int) => "INTEGER",
                Type t when t == typeof(long) => "BIGINT",
                Type t when t == typeof(double) => "DOUBLE PRECISION",
                Type t when t == typeof(float) => "REAL",
                Type t when t == typeof(bool) => "BOOLEAN",
                Type t when t == typeof(DateTime) => "TIMESTAMP",
                _ => "TEXT"
            };
            cols.Add($"{dialect.QuoteIdentifier(name)} {sqlType}");
        }

        var sql = $"CREATE TABLE IF NOT EXISTS {dialect.QuoteIdentifier(tableName)} ({string.Join(", ", cols)})";
        using var cmd = conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Insert rows in batches using multi-row INSERT statements.
    /// INSERT INTO table (cols) VALUES (@p0_0, @p0_1), (@p1_0, @p1_1), ...
    /// Much faster than one INSERT per row due to reduced round-trips.
    /// </summary>
    private static void InsertRowsBatched(DataFrame df, DbConnection conn, string tableName,
        SqlDialect dialect, int batchSize)
    {
        if (df.RowCount == 0) return;

        var colNames = string.Join(", ", df.ColumnNames.Select(n => dialect.QuoteIdentifier(n)));
        var columns = df.ColumnNames.Select(n => df[n]).ToArray();
        int colCount = df.ColumnCount;

        using var transaction = conn.BeginTransaction();

        for (int batchStart = 0; batchStart < df.RowCount; batchStart += batchSize)
        {
            int batchEnd = Math.Min(batchStart + batchSize, df.RowCount);
            int batchRows = batchEnd - batchStart;

            // Build multi-row VALUES clause: VALUES (@p0_0, @p0_1), (@p1_0, @p1_1), ...
            var sb = new System.Text.StringBuilder();
            sb.Append($"INSERT INTO {dialect.QuoteIdentifier(tableName)} ({colNames}) VALUES ");

            for (int r = 0; r < batchRows; r++)
            {
                if (r > 0) sb.Append(", ");
                sb.Append('(');
                for (int c = 0; c < colCount; c++)
                {
                    if (c > 0) sb.Append(", ");
                    sb.Append($"@p{r}_{c}");
                }
                sb.Append(')');
            }

            using var cmd = conn.CreateCommand();
            cmd.CommandText = sb.ToString();
            cmd.Transaction = transaction;

            // Add all parameters for this batch
            for (int r = 0; r < batchRows; r++)
            {
                int sourceRow = batchStart + r;
                for (int c = 0; c < colCount; c++)
                {
                    var param = cmd.CreateParameter();
                    param.ParameterName = $"@p{r}_{c}";
                    param.Value = columns[c].GetObject(sourceRow) ?? DBNull.Value;
                    cmd.Parameters.Add(param);
                }
            }

            cmd.ExecuteNonQuery();
        }

        transaction.Commit();
    }
}
