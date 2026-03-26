# PandaSharp.IO.Database

ADO.NET database I/O for PandaSharp with support for PostgreSQL, SQL Server, SQLite, and MySQL.

## Features

- **Read SQL queries** directly into DataFrames with automatic type mapping
- **Write DataFrames** to database tables with bulk insert
- **Expression push-down** — translates PandaSharp filters to SQL WHERE clauses
- **Multi-provider** — works with any ADO.NET-compatible database
- **Connection pooling** and parameterized queries for production use

## Installation

```bash
dotnet add package PandaSharp.IO.Database
```

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.IO.Database;

var connStr = "Host=localhost;Database=mydb;Username=user;Password=pass";

var df = DataFrame.ReadSql(connStr, "SELECT * FROM orders WHERE total > 100");
df.GroupBy("status").Agg(x => x.Count()).Print();

df.WriteSql(connStr, "orders_summary", ifExists: "replace");
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
