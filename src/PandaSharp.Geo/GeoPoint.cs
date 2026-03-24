namespace PandaSharp.Geo;

/// <summary>
/// A 2D geographic point with latitude and longitude.
/// </summary>
public readonly record struct GeoPoint(double Latitude, double Longitude)
{
    public override string ToString() => $"({Latitude:F6}, {Longitude:F6})";
}

/// <summary>
/// An axis-aligned bounding box defined by min/max lat/lon.
/// </summary>
public readonly record struct BoundingBox(double MinLat, double MinLon, double MaxLat, double MaxLon)
{
    public bool Contains(GeoPoint point) =>
        point.Latitude >= MinLat && point.Latitude <= MaxLat &&
        point.Longitude >= MinLon && point.Longitude <= MaxLon;

    public bool Intersects(BoundingBox other) =>
        MinLat <= other.MaxLat && MaxLat >= other.MinLat &&
        MinLon <= other.MaxLon && MaxLon >= other.MinLon;

    /// <summary>Create a bounding box by buffering a point by a radius in degrees.</summary>
    public static BoundingBox FromPoint(GeoPoint center, double radiusDegrees) =>
        new(center.Latitude - radiusDegrees, center.Longitude - radiusDegrees,
            center.Latitude + radiusDegrees, center.Longitude + radiusDegrees);
}
