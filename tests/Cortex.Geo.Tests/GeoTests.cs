using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Geo;
using Xunit;

namespace Cortex.Geo.Tests;

public class GeoTests
{
    // Known distances for verification:
    // NYC (40.7128, -74.0060) to LA (34.0522, -118.2437) ≈ 3944 km
    // NYC to London (51.5074, -0.1278) ≈ 5570 km
    // NYC to nearby JFK (40.6413, -73.7781) ≈ 18.9 km

    private static readonly GeoPoint NYC = new(40.7128, -74.0060);
    private static readonly GeoPoint LA = new(34.0522, -118.2437);
    private static readonly GeoPoint London = new(51.5074, -0.1278);
    private static readonly GeoPoint JFK = new(40.6413, -73.7781);
    private static readonly GeoPoint Tokyo = new(35.6762, 139.6503);

    // ===== GeoOps =====

    [Fact]
    public void HaversineKm_NYC_to_LA()
    {
        var dist = GeoOps.HaversineKm(NYC, LA);
        dist.Should().BeApproximately(3944, 10); // ±10 km tolerance
    }

    [Fact]
    public void HaversineKm_NYC_to_London()
    {
        var dist = GeoOps.HaversineKm(NYC, London);
        dist.Should().BeApproximately(5570, 10);
    }

    [Fact]
    public void HaversineKm_SamePoint_IsZero()
    {
        GeoOps.HaversineKm(NYC, NYC).Should().Be(0);
    }

    [Fact]
    public void HaversineKm_ShortDistance()
    {
        var dist = GeoOps.HaversineKm(NYC, JFK);
        dist.Should().BeApproximately(19, 2);
    }

    [Fact]
    public void Bearing_NYC_to_London()
    {
        var bearing = GeoOps.Bearing(NYC, London);
        bearing.Should().BeInRange(40, 60); // roughly northeast
    }

    [Fact]
    public void Destination_NYC_East_100km()
    {
        var dest = GeoOps.Destination(NYC, 90, 100); // east 100km
        dest.Latitude.Should().BeApproximately(NYC.Latitude, 0.5);
        dest.Longitude.Should().BeGreaterThan(NYC.Longitude);
    }

    // ===== BoundingBox =====

    [Fact]
    public void BoundingBox_Contains_PointInside()
    {
        var bbox = new BoundingBox(40.0, -75.0, 41.0, -73.0);
        bbox.Contains(NYC).Should().BeTrue();
    }

    [Fact]
    public void BoundingBox_Contains_PointOutside()
    {
        var bbox = new BoundingBox(40.0, -75.0, 41.0, -73.0);
        bbox.Contains(LA).Should().BeFalse();
    }

    [Fact]
    public void BoundingBox_Intersects()
    {
        var a = new BoundingBox(40.0, -75.0, 41.0, -73.0);
        var b = new BoundingBox(40.5, -74.5, 41.5, -73.5);
        a.Intersects(b).Should().BeTrue();
    }

    [Fact]
    public void BoundingBox_NotIntersects()
    {
        var a = new BoundingBox(40.0, -75.0, 41.0, -73.0);
        var b = new BoundingBox(50.0, 0.0, 52.0, 2.0);
        a.Intersects(b).Should().BeFalse();
    }

    // ===== GeoColumn =====

    [Fact]
    public void GeoColumn_DistanceTo_Point()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London]);
        var distances = geo.DistanceTo(JFK);

        distances.Length.Should().Be(3);
        distances[0]!.Value.Should().BeApproximately(19, 2);   // NYC to JFK
        distances[1]!.Value.Should().BeGreaterThan(3000);       // LA to JFK
    }

    [Fact]
    public void GeoColumn_DistanceTo_Column()
    {
        var a = new GeoColumn("from", [NYC, London]);
        var b = new GeoColumn("to", [LA, Tokyo]);
        var distances = a.DistanceTo(b);

        distances[0]!.Value.Should().BeApproximately(3944, 10);  // NYC-LA
        distances[1]!.Value.Should().BeApproximately(9560, 50);  // London-Tokyo
    }

    [Fact]
    public void GeoColumn_Within_BoundingBox()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London, JFK]);
        var nycArea = new BoundingBox(40.0, -75.0, 41.0, -73.0);
        var mask = geo.Within(nycArea);

        mask[0].Should().BeTrue();  // NYC
        mask[1].Should().BeFalse(); // LA
        mask[2].Should().BeFalse(); // London
        mask[3].Should().BeTrue();  // JFK (near NYC)
    }

    [Fact]
    public void GeoColumn_WithinDistance()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London, JFK]);
        var mask = geo.WithinDistance(NYC, 50); // 50 km radius

        mask[0].Should().BeTrue();  // NYC (0 km)
        mask[1].Should().BeFalse(); // LA (3944 km)
        mask[2].Should().BeFalse(); // London (5570 km)
        mask[3].Should().BeTrue();  // JFK (~19 km)
    }

    [Fact]
    public void GeoColumn_Bounds()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London]);
        var bbox = geo.Bounds();

        bbox.MinLat.Should().BeApproximately(34.05, 0.01); // LA
        bbox.MaxLat.Should().BeApproximately(51.51, 0.01); // London
    }

    [Fact]
    public void GeoColumn_Centroid()
    {
        var geo = new GeoColumn("loc", [new GeoPoint(0, 0), new GeoPoint(10, 10)]);
        var c = geo.Centroid();
        c.Latitude.Should().BeApproximately(5, 0.001);
        c.Longitude.Should().BeApproximately(5, 0.001);
    }

    [Fact]
    public void GeoColumn_Filter()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London]);
        var filtered = geo.Filter([true, false, true]);
        filtered.Length.Should().Be(2);
        filtered[0].Should().Be(NYC);
        filtered[1].Should().Be(London);
    }

    // ===== DataFrame Extensions =====

    [Fact]
    public void WithDistance_AddsDistanceColumn()
    {
        var df = new DataFrame(
            new StringColumn("city", ["NYC", "LA", "London"]),
            new Column<double>("lat", [40.7128, 34.0522, 51.5074]),
            new Column<double>("lon", [-74.0060, -118.2437, -0.1278])
        );

        var result = df.WithDistance("lat", "lon", JFK);
        result.ColumnNames.Should().Contain("distance_km");
        result.GetColumn<double>("distance_km")[0]!.Value.Should().BeLessThan(25);
    }

    [Fact]
    public void FilterByDistance_FiltersCorrectly()
    {
        var df = new DataFrame(
            new StringColumn("city", ["NYC", "LA", "London", "JFK"]),
            new Column<double>("lat", [40.7128, 34.0522, 51.5074, 40.6413]),
            new Column<double>("lon", [-74.0060, -118.2437, -0.1278, -73.7781])
        );

        var nearby = df.FilterByDistance("lat", "lon", NYC, 50);
        nearby.RowCount.Should().Be(2); // NYC + JFK
    }

    [Fact]
    public void FilterByBounds_FiltersCorrectly()
    {
        var df = new DataFrame(
            new StringColumn("city", ["NYC", "LA", "London"]),
            new Column<double>("lat", [40.7128, 34.0522, 51.5074]),
            new Column<double>("lon", [-74.0060, -118.2437, -0.1278])
        );

        var eastCoast = new BoundingBox(35, -80, 45, -70);
        var result = df.FilterByBounds("lat", "lon", eastCoast);
        result.RowCount.Should().Be(1);
        result.GetStringColumn("city")[0].Should().Be("NYC");
    }

    // ===== Spatial Join =====

    [Fact]
    public void NearestJoin_FindsClosestPoint()
    {
        var cities = new DataFrame(
            new StringColumn("city", ["NYC", "LA"]),
            new Column<double>("lat", [40.7128, 34.0522]),
            new Column<double>("lon", [-74.0060, -118.2437])
        );

        var airports = new DataFrame(
            new StringColumn("airport", ["JFK", "LAX", "LHR"]),
            new Column<double>("lat", [40.6413, 33.9425, 51.4700]),
            new Column<double>("lon", [-73.7781, -118.4081, -0.4543])
        );

        var result = cities.SpatialJoinNearest("lat", "lon", airports, "lat", "lon");

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("airport");
        result.ColumnNames.Should().Contain("distance_km");

        // NYC's nearest airport should be JFK
        result.GetStringColumn("airport")[0].Should().Be("JFK");
        // LA's nearest should be LAX
        result.GetStringColumn("airport")[1].Should().Be("LAX");
    }

    [Fact]
    public void WithinJoin_FindsAllWithinRadius()
    {
        var centers = new DataFrame(
            new StringColumn("name", ["NYC"]),
            new Column<double>("lat", [40.7128]),
            new Column<double>("lon", [-74.0060])
        );

        var points = new DataFrame(
            new StringColumn("poi", ["Central Park", "JFK", "LAX"]),
            new Column<double>("lat", [40.7829, 40.6413, 33.9425]),
            new Column<double>("lon", [-73.9654, -73.7781, -118.4081])
        );

        var result = centers.SpatialJoinWithin("lat", "lon", points, "lat", "lon", 30);

        // Central Park (~8km) and JFK (~19km) within 30km, LAX is not
        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("poi");
    }

    [Fact]
    public void NearestJoin_WithMaxDistance_ExcludesFarPoints()
    {
        var left = new DataFrame(
            new StringColumn("city", ["NYC", "Tokyo"]),
            new Column<double>("lat", [40.7128, 35.6762]),
            new Column<double>("lon", [-74.0060, 139.6503])
        );

        var right = new DataFrame(
            new StringColumn("airport", ["JFK"]),
            new Column<double>("lat", [40.6413]),
            new Column<double>("lon", [-73.7781])
        );

        var result = left.SpatialJoinNearest("lat", "lon", right, "lat", "lon", maxDistanceKm: 100);
        result.RowCount.Should().Be(2);
        // NYC finds JFK (19km, within 100km)
        result.GetStringColumn("airport")[0].Should().Be("JFK");
        // Tokyo is too far — should have NaN distance
        double.IsNaN(result.GetColumn<double>("distance_km")[1]!.Value).Should().BeTrue();
    }

    // ===== R-tree =====

    [Fact]
    public void RTree_Query_FindsPointsInBBox()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London, JFK, Tokyo]);
        var tree = RTree.Build(geo);

        var nycArea = new BoundingBox(40.0, -75.0, 41.0, -73.0);
        var results = tree.Query(nycArea);

        results.Should().Contain(0); // NYC
        results.Should().Contain(3); // JFK
        results.Should().NotContain(1); // LA
    }

    [Fact]
    public void RTree_QueryRadius_FindsNearbyPoints()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London, JFK, Tokyo]);
        var tree = RTree.Build(geo);

        var nearby = tree.QueryRadius(NYC, 50);
        nearby.Should().Contain(r => r.Index == 0); // NYC itself
        nearby.Should().Contain(r => r.Index == 3); // JFK (~19km)
        nearby.Should().NotContain(r => r.Index == 1); // LA
    }

    [Fact]
    public void RTree_Nearest_FindsClosest()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London, JFK, Tokyo]);
        var tree = RTree.Build(geo);

        var (idx, dist) = tree.Nearest(new GeoPoint(40.75, -73.95)); // near central Manhattan
        idx.Should().Be(0); // NYC is closest
        dist.Should().BeLessThan(10);
    }

    [Fact]
    public void RTree_KNearest_ReturnsKClosest()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London, JFK, Tokyo]);
        var tree = RTree.Build(geo);

        var results = tree.KNearest(NYC, 2);
        results.Should().HaveCount(2);
        results[0].Index.Should().Be(0); // NYC itself (0km)
        results[1].Index.Should().Be(3); // JFK (~19km)
    }

    [Fact]
    public void RTree_LargeDataset_MatchesBruteForce()
    {
        // Generate 1000 random points
        var rng = new Random(42);
        var lats = new double[1000];
        var lons = new double[1000];
        for (int i = 0; i < 1000; i++)
        {
            lats[i] = 30 + rng.NextDouble() * 20;
            lons[i] = -120 + rng.NextDouble() * 50;
        }
        var geo = new GeoColumn("pts", lats, lons);
        var tree = RTree.Build(geo);

        // Test nearest: compare R-tree result with brute-force
        var target = new GeoPoint(40.0, -90.0);
        var (treeIdx, treeDist) = tree.Nearest(target);

        double bruteBestDist = double.MaxValue;
        int bruteBestIdx = -1;
        for (int i = 0; i < 1000; i++)
        {
            double d = GeoOps.HaversineKm(geo[i], target);
            if (d < bruteBestDist) { bruteBestDist = d; bruteBestIdx = i; }
        }

        treeIdx.Should().Be(bruteBestIdx);
        treeDist.Should().BeApproximately(bruteBestDist, 0.001);
    }

    [Fact]
    public void RTree_Empty()
    {
        var geo = new GeoColumn("empty", Array.Empty<double>(), Array.Empty<double>());
        var tree = RTree.Build(geo);

        tree.Query(new BoundingBox(0, 0, 90, 180)).Should().BeEmpty();
    }

    // ===== Polygon =====

    [Fact]
    public void Polygon_Contains_PointInside()
    {
        // Simple square polygon around NYC area
        var poly = new GeoPolygon(
            (40.0, -75.0), (41.0, -75.0), (41.0, -73.0), (40.0, -73.0)
        );

        poly.Contains(NYC).Should().BeTrue();
        poly.Contains(JFK).Should().BeTrue();
    }

    [Fact]
    public void Polygon_Contains_PointOutside()
    {
        var poly = new GeoPolygon(
            (40.0, -75.0), (41.0, -75.0), (41.0, -73.0), (40.0, -73.0)
        );

        poly.Contains(LA).Should().BeFalse();
        poly.Contains(London).Should().BeFalse();
    }

    [Fact]
    public void Polygon_Contains_TrianglePolygon()
    {
        // Triangle: (0,0), (10,0), (5,10)
        var poly = new GeoPolygon(
            (0, 0), (10, 0), (5, 10)
        );

        poly.Contains(new GeoPoint(3, 3)).Should().BeTrue();   // inside
        poly.Contains(new GeoPoint(0, 10)).Should().BeFalse(); // outside
        poly.Contains(new GeoPoint(8, 1)).Should().BeTrue();   // inside near edge
    }

    [Fact]
    public void Polygon_GeoColumn_Within()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London, JFK]);
        var poly = new GeoPolygon(
            (40.0, -75.0), (41.0, -75.0), (41.0, -73.0), (40.0, -73.0)
        );

        var mask = geo.Within(poly);
        mask[0].Should().BeTrue();  // NYC
        mask[1].Should().BeFalse(); // LA
        mask[2].Should().BeFalse(); // London
        mask[3].Should().BeTrue();  // JFK
    }

    [Fact]
    public void Polygon_AreaKm2()
    {
        // Rough 1-degree square near equator ≈ ~111km × 111km ≈ 12321 km²
        var poly = new GeoPolygon(
            (0, 0), (1, 0), (1, 1), (0, 1)
        );

        var area = poly.AreaKm2();
        area.Should().BeInRange(10000, 15000);
    }

    [Fact]
    public void Polygon_PerimeterKm()
    {
        // 1-degree square near equator ≈ 4 × 111km ≈ 444 km
        var poly = new GeoPolygon(
            (0, 0), (1, 0), (1, 1), (0, 1)
        );

        poly.PerimeterKm().Should().BeInRange(400, 500);
    }

    [Fact]
    public void Polygon_Centroid()
    {
        var poly = new GeoPolygon(
            (0, 0), (10, 0), (10, 10), (0, 10)
        );
        var c = poly.Centroid();
        c.Latitude.Should().BeApproximately(5, 0.001);
        c.Longitude.Should().BeApproximately(5, 0.001);
    }

    [Fact]
    public void FilterByPolygon_FiltersCorrectly()
    {
        var df = new DataFrame(
            new StringColumn("city", ["NYC", "LA", "London", "JFK"]),
            new Column<double>("lat", [40.7128, 34.0522, 51.5074, 40.6413]),
            new Column<double>("lon", [-74.0060, -118.2437, -0.1278, -73.7781])
        );

        var nycRegion = new GeoPolygon(
            (40.0, -75.0), (41.0, -75.0), (41.0, -73.0), (40.0, -73.0)
        );

        var result = df.FilterByPolygon("lat", "lon", nycRegion);
        result.RowCount.Should().Be(2); // NYC + JFK
    }

    // ===== LineString =====

    [Fact]
    public void LineString_LengthKm()
    {
        var line = new GeoLineString([NYC, JFK]);
        line.LengthKm().Should().BeApproximately(19, 2);
    }

    [Fact]
    public void LineString_Interpolate_Midpoint()
    {
        var line = new GeoLineString([new GeoPoint(0, 0), new GeoPoint(10, 0)]);
        var mid = line.Interpolate(0.5);
        mid.Latitude.Should().BeApproximately(5, 0.1);
        mid.Longitude.Should().BeApproximately(0, 0.1);
    }

    [Fact]
    public void LineString_Bounds()
    {
        var line = new GeoLineString([NYC, LA]);
        var bbox = line.Bounds();
        bbox.MinLat.Should().BeApproximately(34.05, 0.01);
        bbox.MaxLat.Should().BeApproximately(40.71, 0.01);
    }

    // ===== WKB encoding =====

    [Fact]
    public void Wkb_EncodeDecodePoint_RoundTrip()
    {
        var point = NYC;
        var bytes = Wkb.EncodePoint(point);

        bytes.Length.Should().Be(21);

        var decoded = Wkb.DecodePoint(bytes);
        decoded.Latitude.Should().BeApproximately(point.Latitude, 0.000001);
        decoded.Longitude.Should().BeApproximately(point.Longitude, 0.000001);
    }

    [Fact]
    public void Wkb_EncodeDecodePoints_RoundTrip()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London]);
        var bytes = Wkb.EncodePoints(geo);

        bytes.Length.Should().Be(63); // 3 × 21

        var decoded = Wkb.DecodePoints("loc", bytes);
        decoded.Length.Should().Be(3);
        decoded[0].Latitude.Should().BeApproximately(NYC.Latitude, 0.000001);
        decoded[1].Latitude.Should().BeApproximately(LA.Latitude, 0.000001);
        decoded[2].Longitude.Should().BeApproximately(London.Longitude, 0.000001);
    }

    [Fact]
    public void Wkb_DecodePointArray_RoundTrip()
    {
        var points = new[] { NYC, Tokyo };
        var wkbArray = points.Select(p => Wkb.EncodePoint(p)).ToArray();

        var decoded = Wkb.DecodePointArray("loc", wkbArray);
        decoded.Length.Should().Be(2);
        decoded[0].Should().Be(NYC);
        decoded[1].Latitude.Should().BeApproximately(Tokyo.Latitude, 0.000001);
    }

    // ===== GeoParquet =====

    [Fact]
    public void GeoParquet_WriteRead_RoundTrip()
    {
        var df = new DataFrame(
            new StringColumn("city", ["NYC", "LA", "London"]),
            new Column<double>("population", [8_000_000, 4_000_000, 9_000_000])
        );
        var geo = new GeoColumn("geometry", [NYC, LA, London]);

        var path = Path.Combine(Path.GetTempPath(), $"geoparquet_test_{Guid.NewGuid():N}.parquet");

        try
        {
            GeoParquet.Write(df, geo, path);
            File.Exists(path).Should().BeTrue();

            var (loadedDf, loadedGeo) = GeoParquet.Read(path);

            loadedDf.RowCount.Should().Be(3);
            loadedDf.ColumnNames.Should().Contain("city");
            loadedDf.ColumnNames.Should().Contain("population");
            loadedDf.ColumnNames.Should().NotContain("geometry"); // geometry is in GeoColumn

            loadedGeo.Length.Should().Be(3);
            loadedGeo[0].Latitude.Should().BeApproximately(NYC.Latitude, 0.000001);
            loadedGeo[1].Longitude.Should().BeApproximately(LA.Longitude, 0.000001);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(path + ".geo.json")) File.Delete(path + ".geo.json");
        }
    }

    [Fact]
    public void GeoParquet_ReadAsDataFrame()
    {
        var df = new DataFrame(
            new StringColumn("name", ["A", "B"]),
            new Column<int>("id", [1, 2])
        );
        var geo = new GeoColumn("geometry", [NYC, LA]);

        var path = Path.Combine(Path.GetTempPath(), $"geoparquet_df_{Guid.NewGuid():N}.parquet");

        try
        {
            GeoParquet.Write(df, geo, path);

            var result = GeoParquet.ReadAsDataFrame(path);
            result.ColumnNames.Should().Contain("latitude");
            result.ColumnNames.Should().Contain("longitude");
            result.ColumnNames.Should().Contain("name");
            result.GetColumn<double>("latitude")[0].Should().BeApproximately(NYC.Latitude, 0.000001);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(path + ".geo.json")) File.Delete(path + ".geo.json");
        }
    }

    [Fact]
    public void GeoParquet_WritesGeoMetadata()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var geo = new GeoColumn("geometry", [NYC]);

        var path = Path.Combine(Path.GetTempPath(), $"geoparquet_meta_{Guid.NewGuid():N}.parquet");

        try
        {
            GeoParquet.Write(df, geo, path);

            var metadataPath = path + ".geo.json";
            File.Exists(metadataPath).Should().BeTrue();

            var json = File.ReadAllText(metadataPath);
            json.Should().Contain("WKB");
            json.Should().Contain("Point");
            json.Should().Contain("bbox");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
            if (File.Exists(path + ".geo.json")) File.Delete(path + ".geo.json");
        }
    }

    // ===== Dissolve =====

    [Fact]
    public void Dissolve_GroupsByKeyAndComputesCentroids()
    {
        var df = new DataFrame(
            new StringColumn("region", ["East", "East", "West", "West"]),
            new Column<double>("value", [100, 200, 300, 400]),
            new Column<double>("lat", [40.0, 41.0, 34.0, 35.0]),
            new Column<double>("lon", [-74.0, -73.0, -118.0, -117.0])
        );

        var result = df.Dissolve("lat", "lon", "region", Cortex.GroupBy.AggFunc.Sum);

        result.RowCount.Should().Be(2); // East, West
        result.ColumnNames.Should().Contain("centroid_lat");
        result.ColumnNames.Should().Contain("centroid_lon");
    }

    [Fact]
    public void Dissolve_SumAggregation()
    {
        var df = new DataFrame(
            new StringColumn("cat", ["A", "A", "B"]),
            new Column<double>("val", [10, 20, 30]),
            new Column<double>("lat", [0, 1, 2]),
            new Column<double>("lon", [0, 1, 2])
        );

        var geo = df.ToGeoColumn("lat", "lon");
        var (data, dissolved) = DissolveOps.Dissolve(df, geo, "cat", Cortex.GroupBy.AggFunc.Sum);

        data.RowCount.Should().Be(2);
        dissolved.Length.Should().Be(2);

        // Centroid of A's points: (0+1)/2, (0+1)/2 = (0.5, 0.5)
        // Find which row is "A"
        for (int i = 0; i < data.RowCount; i++)
        {
            if (data.GetStringColumn("cat")[i] == "A")
            {
                dissolved[i].Latitude.Should().BeApproximately(0.5, 0.01);
                dissolved[i].Longitude.Should().BeApproximately(0.5, 0.01);
            }
        }
    }

    // ===== Coordinate Reprojection =====

    [Fact]
    public void Reproject_Wgs84ToWebMercator()
    {
        var geo = new GeoColumn("loc", [NYC]);
        var projected = geo.Reproject(Crs.Wgs84, Crs.WebMercator);

        // Web Mercator coordinates for NYC should be large numbers (meters)
        projected[0].Longitude.Should().NotBe(NYC.Longitude);
        Math.Abs(projected[0].Longitude).Should().BeGreaterThan(1_000_000);
        Math.Abs(projected[0].Latitude).Should().BeGreaterThan(1_000_000);
    }

    [Fact]
    public void Reproject_WebMercatorBackToWgs84()
    {
        var geo = new GeoColumn("loc", [NYC, LA]);
        var projected = geo.Reproject(Crs.Wgs84, Crs.WebMercator);
        var backToWgs84 = projected.Reproject(Crs.WebMercator, Crs.Wgs84);

        // Should round-trip back to original within floating point tolerance
        backToWgs84[0].Latitude.Should().BeApproximately(NYC.Latitude, 0.0001);
        backToWgs84[0].Longitude.Should().BeApproximately(NYC.Longitude, 0.0001);
        backToWgs84[1].Latitude.Should().BeApproximately(LA.Latitude, 0.0001);
    }

    [Fact]
    public void Reproject_Wgs84ToUtm()
    {
        // NYC is in UTM zone 18N
        var geo = new GeoColumn("loc", [NYC]);
        var utm = geo.Reproject(Crs.Wgs84, Crs.Utm(18));

        // UTM easting should be around 500000 ± 500km
        utm[0].Longitude.Should().BeInRange(100_000, 900_000);
        // UTM northing for 40°N should be around 4.5M meters
        utm[0].Latitude.Should().BeInRange(4_000_000, 5_000_000);
    }

    [Fact]
    public void Reproject_EpsgCodes()
    {
        var geo = new GeoColumn("loc", [London]);
        var projected = geo.Reproject(4326, 3857); // WGS84 → Web Mercator

        Math.Abs(projected[0].Longitude).Should().BeGreaterThan(1000);
    }

    [Fact]
    public void Reproject_PreservesLength()
    {
        var geo = new GeoColumn("loc", [NYC, LA, London]);
        var projected = geo.Reproject(Crs.Wgs84, Crs.WebMercator);
        projected.Length.Should().Be(3);
    }

    [Fact]
    public void Dissolve_CountAggregation()
    {
        var df = new DataFrame(
            new StringColumn("region", ["East", "East", "East", "West"]),
            new Column<double>("val", [1, 2, 3, 4]),
            new Column<double>("lat", [40, 41, 42, 34]),
            new Column<double>("lon", [-74, -73, -72, -118])
        );

        var geo = df.ToGeoColumn("lat", "lon");
        var (data, _) = DissolveOps.Dissolve(df, geo, "region", Cortex.GroupBy.AggFunc.Count);

        data.RowCount.Should().Be(2);
    }
}
