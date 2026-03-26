namespace PandaSharp.Geo;

/// <summary>
/// A closed polygon defined by an ordered list of vertices (lat/lon).
/// The last vertex is implicitly connected back to the first.
/// Uses the ray-casting algorithm for point-in-polygon tests.
/// </summary>
public class GeoPolygon
{
    private readonly GeoPoint[] _vertices;
    private readonly BoundingBox _bounds;

    /// <summary>Vertices of the polygon (in order).</summary>
    public ReadOnlySpan<GeoPoint> Vertices => _vertices;

    /// <summary>Number of vertices.</summary>
    public int VertexCount => _vertices.Length;

    /// <summary>Bounding box of the polygon.</summary>
    public BoundingBox Bounds => _bounds;

    public GeoPolygon(GeoPoint[] vertices)
    {
        if (vertices.Length < 3)
            throw new ArgumentException("A polygon must have at least 3 vertices.");

        // Check for antimeridian crossing: polygon spanning >180 degrees longitude
        // is not supported by the simple ray-casting algorithm
        double minLon = double.MaxValue, maxLon = double.MinValue;
        foreach (var v in vertices)
        {
            if (v.Longitude < minLon) minLon = v.Longitude;
            if (v.Longitude > maxLon) maxLon = v.Longitude;
        }
        if (maxLon - minLon > 180)
            throw new ArgumentException("Polygon spans more than 180° longitude. Antimeridian-crossing polygons are not supported.");

        _vertices = vertices;
        _bounds = ComputeBounds(vertices);
    }

    public GeoPolygon(params (double Lat, double Lon)[] vertices)
        : this(vertices.Select(v => new GeoPoint(v.Lat, v.Lon)).ToArray()) { }

    /// <summary>
    /// Test if a point is inside the polygon using the ray-casting algorithm.
    /// Casts a horizontal ray from the point to the right and counts edge crossings.
    /// Odd crossings = inside, even = outside.
    /// </summary>
    public bool Contains(GeoPoint point)
    {
        // Fast reject: bounding box check
        if (!_bounds.Contains(point))
            return false;

        return RayCast(point);
    }

    /// <summary>
    /// Test which points in a GeoColumn are inside this polygon.
    /// Uses bounding box pre-filter + ray-casting.
    /// </summary>
    public bool[] Contains(GeoColumn points)
    {
        var result = new bool[points.Length];
        for (int i = 0; i < points.Length; i++)
            result[i] = Contains(points[i]);
        return result;
    }

    /// <summary>
    /// Approximate area of the polygon in square kilometers using the Shoelace formula
    /// on projected coordinates. Accuracy decreases for large polygons or near poles.
    /// </summary>
    public double AreaKm2()
    {
        // Shoelace formula in degrees, then convert
        double area = 0;
        int n = _vertices.Length;
        for (int i = 0; i < n; i++)
        {
            int j = (i + 1) % n;
            area += _vertices[i].Longitude * _vertices[j].Latitude;
            area -= _vertices[j].Longitude * _vertices[i].Latitude;
        }
        area = Math.Abs(area) / 2.0;

        // Convert from degree² to km²
        double midLat = (_bounds.MinLat + _bounds.MaxLat) / 2;
        double latKm = 111.32; // km per degree latitude
        double lonKm = 111.32 * Math.Cos(midLat * Math.PI / 180); // km per degree longitude
        return area * latKm * lonKm;
    }

    /// <summary>Compute the centroid of the polygon.</summary>
    public GeoPoint Centroid()
    {
        double sumLat = 0, sumLon = 0;
        foreach (var v in _vertices)
        {
            sumLat += v.Latitude;
            sumLon += v.Longitude;
        }
        return new GeoPoint(sumLat / _vertices.Length, sumLon / _vertices.Length);
    }

    /// <summary>Approximate perimeter in km.</summary>
    public double PerimeterKm()
    {
        double total = 0;
        for (int i = 0; i < _vertices.Length; i++)
        {
            int j = (i + 1) % _vertices.Length;
            total += GeoOps.HaversineKm(_vertices[i], _vertices[j]);
        }
        return total;
    }

    private bool RayCast(GeoPoint point)
    {
        bool inside = false;
        int n = _vertices.Length;

        for (int i = 0, j = n - 1; i < n; j = i++)
        {
            double yi = _vertices[i].Latitude;
            double xi = _vertices[i].Longitude;
            double yj = _vertices[j].Latitude;
            double xj = _vertices[j].Longitude;

            // Does the edge cross the horizontal ray from point going right?
            if ((yi > point.Latitude) != (yj > point.Latitude))
            {
                double intersectX = xj + (point.Latitude - yj) / (yi - yj) * (xi - xj);
                if (point.Longitude < intersectX)
                    inside = !inside;
            }
        }

        return inside;
    }

    private static BoundingBox ComputeBounds(GeoPoint[] vertices)
    {
        double minLat = double.MaxValue, maxLat = double.MinValue;
        double minLon = double.MaxValue, maxLon = double.MinValue;
        foreach (var v in vertices)
        {
            if (v.Latitude < minLat) minLat = v.Latitude;
            if (v.Latitude > maxLat) maxLat = v.Latitude;
            if (v.Longitude < minLon) minLon = v.Longitude;
            if (v.Longitude > maxLon) maxLon = v.Longitude;
        }
        return new BoundingBox(minLat, minLon, maxLat, maxLon);
    }
}

/// <summary>
/// A LineString: an ordered sequence of connected points.
/// </summary>
public class GeoLineString
{
    private readonly GeoPoint[] _points;

    public ReadOnlySpan<GeoPoint> Points => _points;
    public int PointCount => _points.Length;

    public GeoLineString(GeoPoint[] points)
    {
        if (points.Length < 2)
            throw new ArgumentException("A line string must have at least 2 points.");
        _points = points;
    }

    /// <summary>Total length in km.</summary>
    public double LengthKm()
    {
        double total = 0;
        for (int i = 0; i < _points.Length - 1; i++)
            total += GeoOps.HaversineKm(_points[i], _points[i + 1]);
        return total;
    }

    /// <summary>Bounding box of the line string.</summary>
    public BoundingBox Bounds()
    {
        double minLat = double.MaxValue, maxLat = double.MinValue;
        double minLon = double.MaxValue, maxLon = double.MinValue;
        foreach (var p in _points)
        {
            if (p.Latitude < minLat) minLat = p.Latitude;
            if (p.Latitude > maxLat) maxLat = p.Latitude;
            if (p.Longitude < minLon) minLon = p.Longitude;
            if (p.Longitude > maxLon) maxLon = p.Longitude;
        }
        return new BoundingBox(minLat, minLon, maxLat, maxLon);
    }

    /// <summary>Get a point at a given fraction along the line (0.0 = start, 1.0 = end).</summary>
    public GeoPoint Interpolate(double fraction)
    {
        if (fraction <= 0) return _points[0];
        if (fraction >= 1) return _points[^1];

        double totalLen = LengthKm();
        double targetDist = fraction * totalLen;
        double accumulated = 0;

        for (int i = 0; i < _points.Length - 1; i++)
        {
            double segLen = GeoOps.HaversineKm(_points[i], _points[i + 1]);
            if (accumulated + segLen >= targetDist)
            {
                double segFraction = (targetDist - accumulated) / segLen;
                double lat = _points[i].Latitude + segFraction * (_points[i + 1].Latitude - _points[i].Latitude);
                double lon = _points[i].Longitude + segFraction * (_points[i + 1].Longitude - _points[i].Longitude);
                return new GeoPoint(lat, lon);
            }
            accumulated += segLen;
        }

        return _points[^1];
    }
}
