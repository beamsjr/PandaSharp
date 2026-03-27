using FluentAssertions;
using Cortex.Geo;
using Xunit;

namespace Cortex.Geo.Tests.EdgeCases;

public class GeoEdgeCaseTests
{
    // === GeoPoint edge cases ===

    [Fact]
    public void GeoPoint_LatitudeAbove90_ShouldThrow()
    {
        // Latitude must be between -90 and 90
        var act = () => new GeoPoint(91, 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GeoPoint_LatitudeBelow_Minus90_ShouldThrow()
    {
        var act = () => new GeoPoint(-91, 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GeoPoint_LongitudeAbove180_ShouldThrow()
    {
        var act = () => new GeoPoint(0, 181);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GeoPoint_LongitudeBelow_Minus180_ShouldThrow()
    {
        var act = () => new GeoPoint(0, -181);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GeoPoint_ExactBoundary_ShouldBeValid()
    {
        // Boundary values should be accepted
        var p1 = new GeoPoint(90, 180);
        p1.Latitude.Should().Be(90);
        p1.Longitude.Should().Be(180);

        var p2 = new GeoPoint(-90, -180);
        p2.Latitude.Should().Be(-90);
        p2.Longitude.Should().Be(-180);
    }

    // === GeoPolygon edge cases ===

    [Fact]
    public void GeoPolygon_LessThan3Vertices_ShouldThrow()
    {
        var act = () => new GeoPolygon(new GeoPoint[]
        {
            new(0, 0),
            new(1, 1)
        });
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void GeoPolygon_DuplicateVertices_ShouldHandleGracefully()
    {
        // All duplicate vertices form a degenerate polygon
        // Should either throw or return area = 0
        var polygon = new GeoPolygon(new GeoPoint[]
        {
            new(0, 0),
            new(0, 0),
            new(0, 0)
        });

        // Degenerate polygon: area should be 0
        polygon.AreaKm2().Should().Be(0);
        // Contains should return false for any point (zero-area polygon)
        polygon.Contains(new GeoPoint(0, 0)).Should().BeFalse();
    }

    [Fact]
    public void GeoPolygon_ContainsPoint_InsideTriangle()
    {
        var polygon = new GeoPolygon(new GeoPoint[]
        {
            new(0, 0),
            new(0, 10),
            new(10, 0)
        });

        polygon.Contains(new GeoPoint(1, 1)).Should().BeTrue();
        polygon.Contains(new GeoPoint(20, 20)).Should().BeFalse();
    }

    // === BoundingBox edge cases ===

    [Fact]
    public void BoundingBox_InvertedBox_MinGreaterThanMax_ContainsShouldReturnFalse()
    {
        // Inverted box where min > max (not antimeridian wrap)
        var box = new BoundingBox(MinLat: 10, MinLon: 10, MaxLat: 0, MaxLon: 0);
        // A point that would be inside a normal box should NOT be inside an inverted one
        box.Contains(new GeoPoint(5, 5)).Should().BeFalse();
    }

    [Fact]
    public void BoundingBox_ZeroAreaBox_PointOnBoundary()
    {
        // Zero-area box (a point)
        var box = new BoundingBox(MinLat: 5, MinLon: 5, MaxLat: 5, MaxLon: 5);
        box.Contains(new GeoPoint(5, 5)).Should().BeTrue();
        box.Contains(new GeoPoint(5.0001, 5)).Should().BeFalse();
    }

    [Fact]
    public void BoundingBox_PointOnExactBoundary_ShouldBeContained()
    {
        var box = new BoundingBox(MinLat: 0, MinLon: 0, MaxLat: 10, MaxLon: 10);
        // Points on all four edges
        box.Contains(new GeoPoint(0, 5)).Should().BeTrue();   // bottom edge
        box.Contains(new GeoPoint(10, 5)).Should().BeTrue();  // top edge
        box.Contains(new GeoPoint(5, 0)).Should().BeTrue();   // left edge
        box.Contains(new GeoPoint(5, 10)).Should().BeTrue();  // right edge
        // Corners
        box.Contains(new GeoPoint(0, 0)).Should().BeTrue();
        box.Contains(new GeoPoint(10, 10)).Should().BeTrue();
    }

    // === Haversine distance edge cases ===

    [Fact]
    public void Haversine_SamePoint_ShouldBeZero()
    {
        var p = new GeoPoint(40.7128, -74.0060); // NYC
        GeoOps.HaversineKm(p, p).Should().Be(0);
    }

    [Fact]
    public void Haversine_AntipodalPoints_ShouldBeHalfCircumference()
    {
        // North pole to south pole
        var north = new GeoPoint(90, 0);
        var south = new GeoPoint(-90, 0);
        var dist = GeoOps.HaversineKm(north, south);
        // Half circumference of Earth ~= PI * 6371 ~= 20015 km
        dist.Should().BeApproximately(Math.PI * 6371.0, 1.0);
    }

    [Fact]
    public void Haversine_PointsOnEquator_ShouldBeAccurate()
    {
        // Two points on the equator, 90 degrees apart
        var a = new GeoPoint(0, 0);
        var b = new GeoPoint(0, 90);
        var dist = GeoOps.HaversineKm(a, b);
        // Quarter circumference ~= PI/2 * 6371 ~= 10007.5 km
        dist.Should().BeApproximately(Math.PI / 2 * 6371.0, 1.0);
    }

    [Fact]
    public void Haversine_SamePoint_AtOrigin_ShouldBeZero()
    {
        var origin = new GeoPoint(0, 0);
        GeoOps.HaversineKm(origin, origin).Should().Be(0);
    }

    [Fact]
    public void Haversine_NearlyAntipodalPoints_ShouldNotReturnNaN()
    {
        // Nearly antipodal points can cause floating-point issues with asin(sqrt(h)) when h > 1
        var a = new GeoPoint(89.999, 0);
        var b = new GeoPoint(-89.999, 180);
        var dist = GeoOps.HaversineKm(a, b);
        double.IsNaN(dist).Should().BeFalse();
        double.IsInfinity(dist).Should().BeFalse();
        dist.Should().BeGreaterThan(0);
    }

    // === CoordinateTransform edge cases ===

    [Fact]
    public void CoordinateTransform_Origin_ShouldWork()
    {
        // Transform (0,0) from WGS84 to Web Mercator
        var geo = new GeoColumn("test", new double[] { 0 }, new double[] { 0 });
        var projected = geo.Reproject(Crs.Wgs84, Crs.WebMercator);
        // At (0,0) Web Mercator should give (0,0)
        projected.Latitudes[0].Should().BeApproximately(0, 1.0);
        projected.Longitudes[0].Should().BeApproximately(0, 1.0);
    }

    [Fact]
    public void CoordinateTransform_ExtremeLatitude_ShouldThrow()
    {
        // Latitude 91 should be rejected when reprojecting from geographic CRS
        var geo = new GeoColumn("test", new double[] { 91 }, new double[] { 0 });
        var act = () => geo.Reproject(Crs.Wgs84, Crs.WebMercator);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void CoordinateTransform_ExtremeLongitude_ShouldThrow()
    {
        // Longitude 181 should be rejected
        var geo = new GeoColumn("test", new double[] { 0 }, new double[] { 181 });
        var act = () => geo.Reproject(Crs.Wgs84, Crs.WebMercator);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // === Bearing edge cases ===

    [Fact]
    public void Bearing_SamePoint_ShouldBeZeroOrDefined()
    {
        var p = new GeoPoint(0, 0);
        var bearing = GeoOps.Bearing(p, p);
        // Bearing of same point is mathematically undefined, but should not be NaN
        double.IsNaN(bearing).Should().BeFalse();
    }

    [Fact]
    public void Bearing_DueNorth_ShouldBeZero()
    {
        var a = new GeoPoint(0, 0);
        var b = new GeoPoint(10, 0);
        GeoOps.Bearing(a, b).Should().BeApproximately(0, 0.1);
    }

    [Fact]
    public void Bearing_DueEast_ShouldBe90()
    {
        var a = new GeoPoint(0, 0);
        var b = new GeoPoint(0, 10);
        GeoOps.Bearing(a, b).Should().BeApproximately(90, 0.1);
    }

    // === KmToDegrees edge cases ===

    [Fact]
    public void KmToDegrees_AtPole_ShouldNotDivideByZero()
    {
        // At exactly 90 degrees latitude, cos(90) = 0, which would cause division by zero
        // for the longitude component
        var (latDeg, lonDeg) = GeoOps.KmToDegrees(100, atLatitude: 90);
        double.IsInfinity(lonDeg).Should().BeFalse("longitude degrees at pole should not be infinity");
        double.IsNaN(lonDeg).Should().BeFalse("longitude degrees at pole should not be NaN");
    }
}
