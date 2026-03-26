using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.Geo;

/// <summary>
/// Extension methods for creating GeoColumns from DataFrames.
/// </summary>
public static class GeoExtensions
{
    /// <summary>
    /// Create a GeoColumn from latitude and longitude columns in a DataFrame.
    /// </summary>
    public static GeoColumn ToGeoColumn(this DataFrame df, string latColumn, string lonColumn, string name = "geometry")
    {
        var latCol = df.GetColumn<double>(latColumn);
        var lonCol = df.GetColumn<double>(lonColumn);
        return new GeoColumn(name, latCol, lonCol);
    }

    /// <summary>
    /// Add a distance column to the DataFrame (Haversine km from each row's point to a target).
    /// </summary>
    public static DataFrame WithDistance(this DataFrame df, string latColumn, string lonColumn,
        GeoPoint target, string outputColumn = "distance_km")
    {
        var geo = df.ToGeoColumn(latColumn, lonColumn);
        var distCol = geo.DistanceTo(target);
        return df.AddColumn(distCol.Clone(outputColumn));
    }

    /// <summary>
    /// Filter DataFrame to rows within a given distance of a target point.
    /// </summary>
    public static DataFrame FilterByDistance(this DataFrame df, string latColumn, string lonColumn,
        GeoPoint target, double maxDistanceKm)
    {
        var geo = df.ToGeoColumn(latColumn, lonColumn);
        var mask = geo.WithinDistance(target, maxDistanceKm);
        return df.Filter(mask);
    }

    /// <summary>
    /// Filter DataFrame to rows within a bounding box.
    /// </summary>
    public static DataFrame FilterByBounds(this DataFrame df, string latColumn, string lonColumn,
        BoundingBox bounds)
    {
        var geo = df.ToGeoColumn(latColumn, lonColumn);
        var mask = geo.Within(bounds);
        return df.Filter(mask);
    }

    /// <summary>
    /// Nearest-neighbor spatial join: for each row in this DataFrame, find the closest row in other.
    /// </summary>
    public static DataFrame SpatialJoinNearest(this DataFrame df,
        string latColumn, string lonColumn,
        DataFrame other, string otherLatColumn, string otherLonColumn,
        double? maxDistanceKm = null)
    {
        var leftGeo = df.ToGeoColumn(latColumn, lonColumn);
        var rightGeo = other.ToGeoColumn(otherLatColumn, otherLonColumn);
        return SpatialJoin.NearestJoin(df, leftGeo, other, rightGeo, maxDistanceKm);
    }

    /// <summary>
    /// Filter DataFrame to rows where the point is inside a polygon.
    /// </summary>
    public static DataFrame FilterByPolygon(this DataFrame df, string latColumn, string lonColumn,
        GeoPolygon polygon)
    {
        var geo = df.ToGeoColumn(latColumn, lonColumn);
        var mask = geo.Within(polygon);
        return df.Filter(mask);
    }

    /// <summary>
    /// Within-distance spatial join: for each row, find all rows in other within radiusKm.
    /// </summary>
    public static DataFrame SpatialJoinWithin(this DataFrame df,
        string latColumn, string lonColumn,
        DataFrame other, string otherLatColumn, string otherLonColumn,
        double radiusKm)
    {
        var leftGeo = df.ToGeoColumn(latColumn, lonColumn);
        var rightGeo = other.ToGeoColumn(otherLatColumn, otherLonColumn);
        return SpatialJoin.WithinJoin(df, leftGeo, other, rightGeo, radiusKm);
    }
}
