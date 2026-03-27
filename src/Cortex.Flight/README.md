# Cortex.Flight

Apache Arrow Flight RPC client for distributed DataFrame transport and Flight SQL.

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

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
