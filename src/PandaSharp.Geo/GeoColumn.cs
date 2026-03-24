using PandaSharp.Column;

namespace PandaSharp.Geo;

/// <summary>
/// A columnar geometry column storing 2D points as parallel lat/lon arrays.
/// Backed by PandaSharp Column&lt;double&gt; for zero-copy interop.
/// </summary>
public class GeoColumn
{
    private readonly double[] _latitudes;
    private readonly double[] _longitudes;

    public string Name { get; }
    public int Length => _latitudes.Length;

    public GeoColumn(string name, double[] latitudes, double[] longitudes)
    {
        if (latitudes.Length != longitudes.Length)
            throw new ArgumentException("Latitude and longitude arrays must have the same length.");
        Name = name;
        _latitudes = latitudes;
        _longitudes = longitudes;
    }

    public GeoColumn(string name, GeoPoint[] points)
    {
        Name = name;
        _latitudes = new double[points.Length];
        _longitudes = new double[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            _latitudes[i] = points[i].Latitude;
            _longitudes[i] = points[i].Longitude;
        }
    }

    public GeoPoint this[int index] => new(_latitudes[index], _longitudes[index]);

    public ReadOnlySpan<double> Latitudes => _latitudes;
    public ReadOnlySpan<double> Longitudes => _longitudes;

    /// <summary>
    /// Compute Haversine distance from every point to a target point. Returns km.
    /// </summary>
    public Column<double> DistanceTo(GeoPoint target)
    {
        var result = new double[Length];
        for (int i = 0; i < Length; i++)
            result[i] = GeoOps.HaversineKm(this[i], target);
        return new Column<double>($"{Name}_dist_km", result);
    }

    /// <summary>
    /// Compute pairwise distance between this column and another (same length). Returns km.
    /// </summary>
    public Column<double> DistanceTo(GeoColumn other)
    {
        if (Length != other.Length)
            throw new ArgumentException("GeoColumns must have the same length.");
        var result = new double[Length];
        for (int i = 0; i < Length; i++)
            result[i] = GeoOps.HaversineKm(this[i], other[i]);
        return new Column<double>($"dist_km", result);
    }

    /// <summary>
    /// Check which points fall within a bounding box. Returns a boolean mask.
    /// </summary>
    public bool[] Within(BoundingBox bbox)
    {
        var result = new bool[Length];
        for (int i = 0; i < Length; i++)
            result[i] = bbox.Contains(this[i]);
        return result;
    }

    /// <summary>
    /// Check which points are within a given distance (km) of a target point.
    /// Uses bounding box pre-filter + exact Haversine check.
    /// </summary>
    public bool[] WithinDistance(GeoPoint target, double radiusKm)
    {
        var result = new bool[Length];
        double approxDeg = GeoOps.KmToDegrees(radiusKm, target.Latitude);
        var bbox = BoundingBox.FromPoint(target, approxDeg);

        for (int i = 0; i < Length; i++)
        {
            // Fast reject: bounding box check
            if (!bbox.Contains(this[i]))
            {
                result[i] = false;
                continue;
            }
            // Exact check
            result[i] = GeoOps.HaversineKm(this[i], target) <= radiusKm;
        }
        return result;
    }

    /// <summary>
    /// Buffer each point by a radius in km, returning bounding boxes.
    /// </summary>
    public BoundingBox[] Buffer(double radiusKm)
    {
        var result = new BoundingBox[Length];
        for (int i = 0; i < Length; i++)
        {
            double deg = GeoOps.KmToDegrees(radiusKm, _latitudes[i]);
            result[i] = BoundingBox.FromPoint(this[i], deg);
        }
        return result;
    }

    /// <summary>
    /// Compute the bounding box of all points.
    /// </summary>
    public BoundingBox Bounds()
    {
        if (Length == 0)
            return new BoundingBox(0, 0, 0, 0);
        double minLat = _latitudes[0], maxLat = _latitudes[0];
        double minLon = _longitudes[0], maxLon = _longitudes[0];
        for (int i = 1; i < Length; i++)
        {
            if (_latitudes[i] < minLat) minLat = _latitudes[i];
            if (_latitudes[i] > maxLat) maxLat = _latitudes[i];
            if (_longitudes[i] < minLon) minLon = _longitudes[i];
            if (_longitudes[i] > maxLon) maxLon = _longitudes[i];
        }
        return new BoundingBox(minLat, minLon, maxLat, maxLon);
    }

    /// <summary>Return the centroid (mean lat/lon) of all points.</summary>
    public GeoPoint Centroid()
    {
        double sumLat = 0, sumLon = 0;
        for (int i = 0; i < Length; i++)
        {
            sumLat += _latitudes[i];
            sumLon += _longitudes[i];
        }
        return new GeoPoint(sumLat / Length, sumLon / Length);
    }

    /// <summary>Bearing from each point to a target, in degrees (0-360).</summary>
    public Column<double> BearingTo(GeoPoint target)
    {
        var result = new double[Length];
        for (int i = 0; i < Length; i++)
            result[i] = GeoOps.Bearing(this[i], target);
        return new Column<double>($"{Name}_bearing", result);
    }

    /// <summary>
    /// Check which points fall within a polygon. Returns a boolean mask.
    /// Uses bounding box pre-filter + ray-casting.
    /// </summary>
    public bool[] Within(GeoPolygon polygon) => polygon.Contains(this);

    /// <summary>Filter to a subset of points by index mask.</summary>
    public GeoColumn Filter(bool[] mask)
    {
        if (mask.Length != Length)
            throw new ArgumentException("Mask length must match column length.");
        int count = 0;
        for (int i = 0; i < mask.Length; i++) if (mask[i]) count++;
        var lats = new double[count];
        var lons = new double[count];
        int j = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i])
            {
                lats[j] = _latitudes[i];
                lons[j] = _longitudes[i];
                j++;
            }
        }
        return new GeoColumn(Name, lats, lons);
    }

    /// <summary>Take rows at given indices.</summary>
    public GeoColumn TakeRows(int[] indices)
    {
        var lats = new double[indices.Length];
        var lons = new double[indices.Length];
        for (int i = 0; i < indices.Length; i++)
        {
            lats[i] = _latitudes[indices[i]];
            lons[i] = _longitudes[indices[i]];
        }
        return new GeoColumn(Name, lats, lons);
    }
}
