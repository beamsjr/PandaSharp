namespace PandaSharp.Geo;

/// <summary>
/// Core geographic operations: distance, bearing, buffering.
/// </summary>
public static class GeoOps
{
    private const double EarthRadiusKm = 6371.0;
    private const double EarthRadiusMiles = 3958.8;
    private const double DegToRad = Math.PI / 180.0;

    /// <summary>
    /// Haversine distance between two points in kilometers.
    /// Accurate for any distance on Earth's surface.
    /// </summary>
    public static double HaversineKm(GeoPoint a, GeoPoint b)
    {
        double dLat = (b.Latitude - a.Latitude) * DegToRad;
        double dLon = (b.Longitude - a.Longitude) * DegToRad;
        double lat1 = a.Latitude * DegToRad;
        double lat2 = b.Latitude * DegToRad;

        double h = Math.Sin(dLat / 2) * Math.Sin(dLat / 2) +
                   Math.Cos(lat1) * Math.Cos(lat2) *
                   Math.Sin(dLon / 2) * Math.Sin(dLon / 2);

        return 2 * EarthRadiusKm * Math.Asin(Math.Sqrt(h));
    }

    /// <summary>Haversine distance in miles.</summary>
    public static double HaversineMiles(GeoPoint a, GeoPoint b) =>
        HaversineKm(a, b) * EarthRadiusMiles / EarthRadiusKm;

    /// <summary>Euclidean distance in degrees (fast approximation for short distances).</summary>
    public static double EuclideanDegrees(GeoPoint a, GeoPoint b)
    {
        double dLat = a.Latitude - b.Latitude;
        double dLon = a.Longitude - b.Longitude;
        return Math.Sqrt(dLat * dLat + dLon * dLon);
    }

    /// <summary>
    /// Convert a distance in kilometers to approximate degrees at a given latitude.
    /// Returns both latitude and longitude degree values since they differ at non-equatorial latitudes.
    /// Useful for creating bounding boxes from kilometer distances.
    /// </summary>
    public static (double LatDeg, double LonDeg) KmToDegrees(double km, double atLatitude = 0)
    {
        // 1 degree of latitude ≈ 111.32 km
        double latDeg = km / 111.32;
        // Longitude degrees shrink with cos(latitude)
        double cosLat = Math.Cos(atLatitude * DegToRad);
        double lonDeg;
        if (cosLat < 1e-10)
        {
            // At or very near the pole, longitude is meaningless — cap to full range
            lonDeg = 360.0;
        }
        else
        {
            lonDeg = km / (111.32 * cosLat);
        }
        return (latDeg, lonDeg);
    }

    /// <summary>
    /// Initial bearing from point a to point b in degrees (0-360).
    /// </summary>
    public static double Bearing(GeoPoint a, GeoPoint b)
    {
        double lat1 = a.Latitude * DegToRad;
        double lat2 = b.Latitude * DegToRad;
        double dLon = (b.Longitude - a.Longitude) * DegToRad;

        double y = Math.Sin(dLon) * Math.Cos(lat2);
        double x = Math.Cos(lat1) * Math.Sin(lat2) -
                   Math.Sin(lat1) * Math.Cos(lat2) * Math.Cos(dLon);

        double bearing = Math.Atan2(y, x) * (180.0 / Math.PI);
        return (bearing + 360) % 360;
    }

    /// <summary>
    /// Compute the destination point given start, bearing, and distance in km.
    /// </summary>
    public static GeoPoint Destination(GeoPoint start, double bearingDegrees, double distanceKm)
    {
        double d = distanceKm / EarthRadiusKm;
        double brng = bearingDegrees * DegToRad;
        double lat1 = start.Latitude * DegToRad;
        double lon1 = start.Longitude * DegToRad;

        double lat2 = Math.Asin(Math.Sin(lat1) * Math.Cos(d) +
                                Math.Cos(lat1) * Math.Sin(d) * Math.Cos(brng));
        double lon2 = lon1 + Math.Atan2(
            Math.Sin(brng) * Math.Sin(d) * Math.Cos(lat1),
            Math.Cos(d) - Math.Sin(lat1) * Math.Sin(lat2));

        return new GeoPoint(lat2 / DegToRad, lon2 / DegToRad);
    }
}
