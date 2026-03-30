# Cortex.Geo

Geospatial operations for Cortex with R-tree indexing and coordinate transforms.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **Spatial operations** — distance calculations, bounding boxes, point-in-polygon
- **R-tree spatial indexing** for fast nearest-neighbor and range queries
- **Coordinate transforms** — WGS84, UTM, and custom CRS via ProjNET
- **Spatial joins** — join DataFrames on geographic proximity
- **GeoJSON, WKT, and GeoParquet** import/export

## Installation

```bash
dotnet add package Cortex.Geo
```

## Quick Start

```csharp
using Cortex;
using Cortex.Geo;

var stores = DataFrame.ReadCsv("stores.csv");    // has lat, lon columns
var customers = DataFrame.ReadCsv("customers.csv");

var nearest = customers.Geo.NearestJoin(stores, leftLat: "lat", leftLon: "lon",
    rightLat: "lat", rightLon: "lon", maxDistanceKm: 10);

nearest.Head(5).Print();
```

## Distance Calculations

```csharp
var distance = GeoDistance.Haversine(lat1: 40.7128, lon1: -74.0060,
                                     lat2: 51.5074, lon2: -0.1278);
// distance in kilometers
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.Viz** | Map visualizations |
| **Cortex.IO.Database** | PostGIS spatial queries |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
