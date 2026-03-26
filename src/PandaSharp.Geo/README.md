# PandaSharp.Geo

Geospatial operations for PandaSharp with R-tree indexing and coordinate transforms.

## Features

- **Spatial operations** — distance calculations, bounding boxes, point-in-polygon
- **R-tree spatial indexing** for fast nearest-neighbor and range queries
- **Coordinate transforms** — WGS84, UTM, and custom CRS via ProjNET
- **Spatial joins** — join DataFrames on geographic proximity
- **GeoJSON and WKT** import/export

## Installation

```bash
dotnet add package PandaSharp.Geo
```

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.Geo;

var stores = DataFrame.ReadCsv("stores.csv");    // has lat, lon columns
var customers = DataFrame.ReadCsv("customers.csv");

var nearest = customers.Geo.NearestJoin(stores, leftLat: "lat", leftLon: "lon",
    rightLat: "lat", rightLon: "lon", maxDistanceKm: 10);

nearest.Head(5).Print();
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
