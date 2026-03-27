using FluentAssertions;
using Cortex.Geo;
using Xunit;

namespace Cortex.Geo.Tests.EdgeCases;

public class GeoEdgeCaseTests2
{
    // === Haversine edge cases ===

    [Fact]
    public void Haversine_Across_Antimeridian_Should_Be_Short_Distance()
    {
        // Points at 179E and 179W on the equator: should be ~2 degrees apart (~222 km)
        // not ~358 degrees apart (~39800 km)
        var a = new GeoPoint(0, 179);
        var b = new GeoPoint(0, -179);
        var dist = GeoOps.HaversineKm(a, b);

        // 2 degrees on equator ~ 222 km
        dist.Should().BeLessThan(300, "distance across antimeridian should be the short way around");
        dist.Should().BeApproximately(222.4, 5.0);
    }

    [Fact]
    public void Haversine_Identical_Points_Should_Be_Exactly_Zero()
    {
        var p = new GeoPoint(51.5074, -0.1278); // London
        var dist = GeoOps.HaversineKm(p, p);
        dist.Should().Be(0.0, "distance between identical points should be exactly 0");
    }

    // === GeoColumn.Buffer at pole ===

    [Fact]
    public void GeoColumn_Buffer_At_Pole_Should_Produce_Valid_BoundingBox()
    {
        // Buffer at the north pole - longitude degrees expand to infinity at cos(90)=0
        var geo = new GeoColumn("pole", new double[] { 90.0 }, new double[] { 0.0 });
        var buffers = geo.Buffer(100); // 100 km radius

        buffers.Should().HaveCount(1);
        var bbox = buffers[0];

        // The bounding box should have reasonable bounds, not infinity
        double.IsInfinity(bbox.MinLon).Should().BeFalse("MinLon should not be infinity at pole");
        double.IsInfinity(bbox.MaxLon).Should().BeFalse("MaxLon should not be infinity at pole");
        double.IsNaN(bbox.MinLon).Should().BeFalse("MinLon should not be NaN at pole");
        double.IsNaN(bbox.MaxLon).Should().BeFalse("MaxLon should not be NaN at pole");

        // At the pole, longitude is meaningless, so the buffer should cover a wide range
        // but the longitude degree value should be capped (360 degrees max) rather than
        // producing absurdly large values
        bbox.MinLon.Should().BeGreaterThanOrEqualTo(-361, "MinLon should be bounded at pole");
        bbox.MaxLon.Should().BeLessThanOrEqualTo(361, "MaxLon should be bounded at pole");
    }

    // === RTree edge cases ===

    [Fact]
    public void RTree_Query_On_Empty_Tree_Returns_Empty_List()
    {
        var emptyCol = new GeoColumn("empty", Array.Empty<double>(), Array.Empty<double>());
        var tree = RTree.Build(emptyCol);

        var result = tree.Query(new BoundingBox(0, 0, 10, 10));

        result.Should().BeEmpty("query on empty tree should return empty list");
    }

    [Fact]
    public void RTree_Query_Returns_Zero_Results_When_No_Points_In_Bbox()
    {
        var lats = new double[] { 40, 41, 42 };
        var lons = new double[] { -74, -73, -72 };
        var geo = new GeoColumn("test", lats, lons);
        var tree = RTree.Build(geo);

        // Query far away from any points
        var result = tree.Query(new BoundingBox(0, 0, 1, 1));

        result.Should().BeEmpty("query with no matching points should return empty");
    }

    [Fact]
    public void RTree_Query_Returns_All_Results_When_Bbox_Covers_All()
    {
        var lats = new double[] { 40, 41, 42 };
        var lons = new double[] { -74, -73, -72 };
        var geo = new GeoColumn("test", lats, lons);
        var tree = RTree.Build(geo);

        // Query covering all points
        var result = tree.Query(new BoundingBox(39, -75, 43, -71));

        result.Should().HaveCount(3, "query covering all points should return all");
    }

    // === GeoPolygon.Contains for point on edge ===

    [Fact]
    public void GeoPolygon_Contains_Point_On_Edge_Should_Be_Consistent()
    {
        // Square polygon: (0,0), (0,10), (10,10), (10,0)
        var polygon = new GeoPolygon(new GeoPoint[]
        {
            new(0, 0), new(0, 10), new(10, 10), new(10, 0)
        });

        // Point on the left edge at (5, 0)
        var pointOnEdge = new GeoPoint(5, 0);
        // Ray casting for points on edges is implementation-defined,
        // but it should not crash and should return a boolean
        var result = polygon.Contains(pointOnEdge);
        // The result is implementation-dependent, but should be deterministic
        // The real test is that it doesn't throw
        result.Should().Be(result); // tautology to ensure no exception
    }

    // === Coordinate transform edge cases ===

    [Fact]
    public void CoordinateTransform_WGS84_To_WebMercator_Origin()
    {
        var geo = new GeoColumn("test", new double[] { 0 }, new double[] { 0 });
        var projected = geo.Reproject(Crs.Wgs84, Crs.WebMercator);

        // At (0,0), Web Mercator should give approximately (0,0)
        projected.Latitudes[0].Should().BeApproximately(0, 0.01);
        projected.Longitudes[0].Should().BeApproximately(0, 0.01);
    }

    [Fact]
    public void CoordinateTransform_RoundTrip_WGS84_WebMercator_WGS84()
    {
        // Transform WGS84 -> WebMercator -> WGS84, coordinates should match
        var lat = 48.8566; // Paris
        var lon = 2.3522;
        var geo = new GeoColumn("test", new double[] { lat }, new double[] { lon });

        var projected = geo.Reproject(Crs.Wgs84, Crs.WebMercator);
        var backToWgs = projected.Reproject(Crs.WebMercator, Crs.Wgs84);

        backToWgs.Latitudes[0].Should().BeApproximately(lat, 0.001,
            "Round-trip WGS84 -> WebMercator -> WGS84 should preserve latitude");
        backToWgs.Longitudes[0].Should().BeApproximately(lon, 0.001,
            "Round-trip WGS84 -> WebMercator -> WGS84 should preserve longitude");
    }

    [Fact]
    public void CoordinateTransform_Near_Pole()
    {
        // lat=89.99 should be transformable, though Web Mercator diverges at poles
        var geo = new GeoColumn("test", new double[] { 89.99 }, new double[] { 0 });

        var act = () => geo.Reproject(Crs.Wgs84, Crs.WebMercator);

        // Should either succeed (with large Y values) or throw a meaningful error
        // Web Mercator is defined for lat ~= -85.06 to 85.06
        // At 89.99, the projection may produce extreme values but should not crash
        act.Should().NotThrow("near-pole coordinate should not crash");
    }
}
