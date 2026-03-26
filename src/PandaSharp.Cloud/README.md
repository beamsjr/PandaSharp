# PandaSharp.Cloud

Cloud storage I/O for PandaSharp with S3, Azure Blob, and GCS support plus built-in resilience.

## Features

- **Read/write DataFrames** directly from S3, Azure Blob Storage, and Google Cloud Storage
- **Multiple formats** — CSV, Parquet, and JSON on cloud storage
- **Built-in resilience** — retries, exponential backoff, and timeout handling
- **Streaming reads** for large files without full memory buffering
- **Credential auto-discovery** — environment variables, profiles, and managed identity

## Installation

```bash
dotnet add package PandaSharp.Cloud
```

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.Cloud;

// Read from S3
var df = DataFrame.ReadParquet("s3://my-bucket/data/sales.parquet");

// Write to Azure Blob
df.Filter(df["year"] == 2025)
  .WriteParquet("az://container/filtered/sales_2025.parquet");
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
