using System.Text;

namespace PandaSharp.IO;

/// <summary>
/// Fluent builder for database connection strings.
/// Supports PostgreSQL, SQL Server, SQLite, and MySQL dialects.
/// </summary>
public class DatabaseConnectionBuilder
{
    private readonly string _dialect;
    private string? _host;
    private int? _port;
    private string? _database;
    private string? _username;
    private string? _password;
    private readonly Dictionary<string, string> _options = new();

    /// <summary>The database dialect (e.g., "Npgsql", "SqlClient", "Sqlite", "MySql").</summary>
    public string Dialect => _dialect;

    private DatabaseConnectionBuilder(string dialect)
    {
        _dialect = dialect;
    }

    /// <summary>Create a builder for PostgreSQL (Npgsql).</summary>
    public static DatabaseConnectionBuilder ForPostgres() => new("Npgsql");

    /// <summary>Create a builder for SQL Server.</summary>
    public static DatabaseConnectionBuilder ForSqlServer() => new("SqlClient");

    /// <summary>Create a builder for SQLite.</summary>
    public static DatabaseConnectionBuilder ForSqlite() => new("Sqlite");

    /// <summary>Create a builder for MySQL.</summary>
    public static DatabaseConnectionBuilder ForMySql() => new("MySql");

    /// <summary>Set the host/server address.</summary>
    public DatabaseConnectionBuilder Host(string host)
    {
        ArgumentNullException.ThrowIfNull(host);
        _host = host;
        return this;
    }

    /// <summary>Set the port number.</summary>
    public DatabaseConnectionBuilder Port(int port)
    {
        _port = port;
        return this;
    }

    /// <summary>Set the database name.</summary>
    public DatabaseConnectionBuilder Database(string database)
    {
        ArgumentNullException.ThrowIfNull(database);
        _database = database;
        return this;
    }

    /// <summary>Set the username.</summary>
    public DatabaseConnectionBuilder Username(string username)
    {
        ArgumentNullException.ThrowIfNull(username);
        _username = username;
        return this;
    }

    /// <summary>Set the password.</summary>
    public DatabaseConnectionBuilder Password(string password)
    {
        ArgumentNullException.ThrowIfNull(password);
        _password = password;
        return this;
    }

    /// <summary>Set a custom connection string option.</summary>
    public DatabaseConnectionBuilder Option(string key, string value)
    {
        _options[key] = value;
        return this;
    }

    /// <summary>
    /// Build the connection string for the configured dialect.
    /// </summary>
    public string Build()
    {
        return _dialect switch
        {
            "Npgsql" => BuildPostgres(),
            "SqlClient" => BuildSqlServer(),
            "Sqlite" => BuildSqlite(),
            "MySql" => BuildMySql(),
            _ => throw new NotSupportedException($"Unsupported dialect: {_dialect}")
        };
    }

    private string BuildPostgres()
    {
        var sb = new StringBuilder(128);
        if (_host is not null) sb.Append($"Host={_host};");
        if (_port is not null) sb.Append($"Port={_port};");
        if (_database is not null) sb.Append($"Database={_database};");
        if (_username is not null) sb.Append($"Username={_username};");
        if (_password is not null) sb.Append($"Password={_password};");
        AppendOptions(sb);
        return TrimTrailingSemicolon(sb);
    }

    private string BuildSqlServer()
    {
        var sb = new StringBuilder(128);
        if (_host is not null)
        {
            sb.Append($"Server={_host}");
            if (_port is not null) sb.Append($",{_port}");
            sb.Append(';');
        }
        if (_database is not null) sb.Append($"Database={_database};");
        if (_username is not null) sb.Append($"User Id={_username};");
        if (_password is not null) sb.Append($"Password={_password};");
        AppendOptions(sb);
        return TrimTrailingSemicolon(sb);
    }

    private string BuildSqlite()
    {
        var sb = new StringBuilder(64);
        if (_database is not null)
            sb.Append($"Data Source={_database};");
        else
            sb.Append("Data Source=:memory:;");
        AppendOptions(sb);
        return TrimTrailingSemicolon(sb);
    }

    private string BuildMySql()
    {
        var sb = new StringBuilder(128);
        if (_host is not null) sb.Append($"Server={_host};");
        if (_port is not null) sb.Append($"Port={_port};");
        if (_database is not null) sb.Append($"Database={_database};");
        if (_username is not null) sb.Append($"Uid={_username};");
        if (_password is not null) sb.Append($"Pwd={_password};");
        AppendOptions(sb);
        return TrimTrailingSemicolon(sb);
    }

    private void AppendOptions(StringBuilder sb)
    {
        foreach (var (key, value) in _options)
            sb.Append($"{key}={value};");
    }

    private static string TrimTrailingSemicolon(StringBuilder sb)
    {
        if (sb.Length > 0 && sb[sb.Length - 1] == ';')
            sb.Length--;
        return sb.ToString();
    }
}
