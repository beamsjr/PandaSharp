# Cortex.IO.Database

ADO.NET database I/O for Cortex with support for PostgreSQL, SQL Server, SQLite, and MySQL.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package. Database-specific ADO.NET providers (Npgsql, Microsoft.Data.SqlClient, etc.) are required for each database.

## Features

- **Read SQL queries** directly into DataFrames with automatic type mapping
- **Write DataFrames** to database tables with bulk insert
- **Expression push-down** — translates Cortex filters to SQL WHERE clauses
- **Lazy database scans** — combine filters, sorts, and limits into a single SQL query
- **Multi-provider** — works with any ADO.NET-compatible database
- **Connection pooling** and parameterized queries for production use

## Installation

```bash
dotnet add package Cortex.IO.Database
```

## Quick Start

```csharp
using Cortex;
using Cortex.IO.Database;

var connStr = "Host=localhost;Database=mydb;Username=user;Password=pass";

var df = DataFrame.ReadSql(connStr, "SELECT * FROM orders WHERE total > 100");
df.GroupBy("status").Agg(x => x.Count()).Print();

df.WriteSql(connStr, "orders_summary", ifExists: "replace");
```

## Lazy Database Scans

```csharp
var scanner = new DatabaseScanner(conn, "orders", new PostgresDialect());
var result = scanner.Lazy()
    .Filter(Col("amount") > Lit(1000))
    .Head(100)
    .Collect();  // single optimized SQL query
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.Cloud** | Cloud storage I/O (S3, Azure, GCS) |
| **Cortex.Flight** | Arrow Flight for distributed data transport |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
