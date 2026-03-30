# Cortex.Cloud

Cloud storage I/O for Cortex with S3, Azure Blob, and GCS support plus built-in resilience.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package. Cloud provider SDKs are included.

## Features

- **Read/write DataFrames** directly from S3, Azure Blob Storage, and Google Cloud Storage
- **Multiple formats** — CSV, Parquet, and JSON on cloud storage
- **Built-in resilience** — retries, exponential backoff, circuit breaker, and timeout handling
- **Streaming reads** for large files without full memory buffering
- **Credential auto-discovery** — environment variables, profiles, and managed identity

## Installation

```bash
dotnet add package Cortex.Cloud
```

## Quick Start

```csharp
using Cortex;
using Cortex.Cloud;

// Read from S3
var df = DataFrame.ReadParquet("s3://my-bucket/data/sales.parquet");

// Write to Azure Blob
df.Filter(df["year"] == 2025)
  .WriteParquet("az://container/filtered/sales_2025.parquet");
```

## Google Cloud Storage

```csharp
var df = DataFrame.ReadCsv("gs://my-bucket/data.csv");
df.Save("gs://my-bucket/output.parquet");
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.IO.Database** | SQL database I/O |
| **Cortex.Flight** | Arrow Flight for distributed data transport |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
