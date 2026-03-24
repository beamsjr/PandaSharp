using System.Collections.Concurrent;
using System.Data.Common;

namespace PandaSharp.IO.Database;

/// <summary>
/// Named connection registry for database access.
/// Register connections once, then access tables by name throughout your application.
///
/// Usage:
///   DatabasePool.Register("mydb", new SqliteConnection("Data Source=mydb.db"));
///   var df = DatabasePool.Table("mydb", "users").Scan();
///   var lazy = DatabasePool.Table("mydb", "orders").Lazy().Filter(...).Collect();
///   DatabasePool.Dispose(); // cleanup all connections
/// </summary>
public static class DatabasePool
{
    private static readonly ConcurrentDictionary<string, DbConnection> _connections = new();
    private static readonly ConcurrentDictionary<string, SqlDialect> _dialects = new();

    /// <summary>
    /// Register a named database connection.
    /// The connection will be opened automatically when first used.
    /// </summary>
    public static void Register(string name, DbConnection connection, SqlDialect? dialect = null)
    {
        _connections[name] = connection;
        if (dialect is not null)
            _dialects[name] = dialect;
    }

    /// <summary>
    /// Get a DatabaseScanner for a table in a registered database.
    /// </summary>
    public static DatabaseScanner Table(string dbName, string tableName)
    {
        if (!_connections.TryGetValue(dbName, out var conn))
            throw new KeyNotFoundException($"Database '{dbName}' is not registered. Call DatabasePool.Register first.");

        _dialects.TryGetValue(dbName, out var dialect);
        return new DatabaseScanner(conn, tableName, dialect);
    }

    /// <summary>Check if a database name is registered.</summary>
    public static bool IsRegistered(string name) => _connections.ContainsKey(name);

    /// <summary>Get all registered database names.</summary>
    public static IEnumerable<string> RegisteredNames => _connections.Keys;

    /// <summary>Remove a named connection (does not close it).</summary>
    public static bool Unregister(string name)
    {
        _dialects.TryRemove(name, out _);
        return _connections.TryRemove(name, out _);
    }

    /// <summary>Dispose and remove all registered connections.</summary>
    public static void DisposeAll()
    {
        foreach (var (name, conn) in _connections)
        {
            try { conn.Dispose(); } catch { }
        }
        _connections.Clear();
        _dialects.Clear();
    }

    /// <summary>Execute a raw SQL query against a registered database.</summary>
    public static PandaSharp.DataFrame Query(string dbName, string sql)
    {
        if (!_connections.TryGetValue(dbName, out var conn))
            throw new KeyNotFoundException($"Database '{dbName}' is not registered.");
        return DataFrameDatabaseExtensions.ReadSql(conn, sql);
    }
}
