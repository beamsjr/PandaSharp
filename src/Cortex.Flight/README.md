# Cortex.Flight

Apache Arrow Flight RPC client for distributed DataFrame transport and Flight SQL.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **Arrow Flight RPC** — high-performance gRPC-based data transfer
- **Flight SQL** — query remote databases using Arrow Flight SQL protocol
- **Zero-copy streaming** — transfer DataFrames between services without serialization overhead
- **Parallel data retrieval** with endpoint-aware ticket routing

## Installation

```bash
dotnet add package Cortex.Flight
```

## Quick Start

```csharp
using Cortex;
using Cortex.Flight;

var client = new FlightClient("grpc://flight-server:8815");

// List available datasets
var flights = await client.ListFlightsAsync();

// Retrieve a dataset as a DataFrame
var df = await client.GetDataFrameAsync("sales_2025");
df.Head(10).Print();
```

## Flight SQL

```csharp
var client = new FlightSqlClient("grpc://flight-sql-server:8815");
var df = await client.QueryAsync("SELECT * FROM orders WHERE total > 1000");
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.IO.Database** | ADO.NET database I/O (alternative for direct DB access) |
| **Cortex.Cloud** | Cloud storage I/O |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
