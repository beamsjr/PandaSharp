namespace Cortex.Geo;

/// <summary>
/// A 2D geographic point with latitude and longitude.
/// </summary>
public readonly record struct GeoPoint
{
    public double Latitude { get; }
    public double Longitude { get; }

    public GeoPoint(double Latitude, double Longitude)
    {
        if (Latitude < -90 || Latitude > 90)
            throw new ArgumentOutOfRangeException(nameof(Latitude), Latitude, "Latitude must be between -90 and 90.");
        if (Longitude < -180 || Longitude > 180)
            throw new ArgumentOutOfRangeException(nameof(Longitude), Longitude, "Longitude must be between -180 and 180.");
        this.Latitude = Latitude;
        this.Longitude = Longitude;
    }

    // Private constructor that skips validation, used for projected coordinates
    private GeoPoint(double latitude, double longitude, bool skipValidation)
    {
        Latitude = latitude;
        Longitude = longitude;
    }

    /// <summary>
    /// Creates a GeoPoint from projected coordinates without geographic bounds validation.
    /// Use this for coordinates in projected CRS (e.g., Web Mercator meters, UTM).
    /// </summary>
    internal static GeoPoint FromProjected(double latitude, double longitude)
        => new(latitude, longitude, skipValidation: true);

    public override string ToString() => $"({Latitude:F6}, {Longitude:F6})";
}

/// <summary>
/// An axis-aligned bounding box defined by min/max lat/lon.
/// </summary>
public readonly record struct BoundingBox(double MinLat, double MinLon, double MaxLat, double MaxLon)
{
    public bool Contains(GeoPoint point)
    {
        bool latIn = point.Latitude >= MinLat && point.Latitude <= MaxLat;
        // Handle normalized bounding boxes where MinLon > MaxLon (antimeridian wrap)
        bool lonIn = MinLon <= MaxLon
            ? point.Longitude >= MinLon && point.Longitude <= MaxLon
            : point.Longitude >= MinLon || point.Longitude <= MaxLon;
        return latIn && lonIn;
    }

    public bool Intersects(BoundingBox other)
    {
        bool latOverlap = MinLat <= other.MaxLat && MaxLat >= other.MinLat;
        // Handle normalized bounding boxes where MinLon > MaxLon (antimeridian wrap)
        bool lonOverlap;
        if (MinLon <= MaxLon && other.MinLon <= other.MaxLon)
            lonOverlap = MinLon <= other.MaxLon && MaxLon >= other.MinLon;
        else
            // At least one box wraps the antimeridian — conservatively assume overlap
            lonOverlap = true;
        return latOverlap && lonOverlap;
    }

    /// <summary>Create a bounding box by buffering a point by a radius in degrees.</summary>
    public static BoundingBox FromPoint(GeoPoint center, double radiusDegrees) =>
        new(center.Latitude - radiusDegrees, center.Longitude - radiusDegrees,
            center.Latitude + radiusDegrees, center.Longitude + radiusDegrees);

    /// <summary>Create a bounding box with separate latitude and longitude degree radii.</summary>
    public static BoundingBox FromPoint(GeoPoint center, double latDegrees, double lonDegrees) =>
        new(center.Latitude - latDegrees, center.Longitude - lonDegrees,
            center.Latitude + latDegrees, center.Longitude + lonDegrees);
}
