using PandaSharp;
using PandaSharp.Column;
using PandaSharp.GroupBy;

namespace PandaSharp.Geo;

/// <summary>
/// Dissolve: spatial GroupBy that aggregates data and merges geometry.
/// Like GeoPandas dissolve() — groups by a column, applies an aggregation function
/// to numeric columns, and computes the centroid of each group's points.
///
/// Usage:
///   var result = Dissolve.Aggregate(df, geoColumn, "region", AggFunc.Sum);
///   // Returns (DataFrame with aggregated values, GeoColumn with group centroids)
/// </summary>
public static class DissolveOps
{
    /// <summary>
    /// Dissolve: group by key columns, aggregate values, and merge geometry into centroids.
    /// </summary>
    public static (DataFrame Data, GeoColumn Geometry) Dissolve(
        DataFrame df, GeoColumn geo,
        string keyColumn, AggFunc aggFunc = AggFunc.Sum)
    {
        return Dissolve(df, geo, [keyColumn], aggFunc);
    }

    /// <summary>
    /// Dissolve with multiple key columns.
    /// </summary>
    public static (DataFrame Data, GeoColumn Geometry) Dissolve(
        DataFrame df, GeoColumn geo,
        string[] keyColumns, AggFunc aggFunc = AggFunc.Sum)
    {
        // Group rows by key
        var groups = new Dictionary<string, List<int>>();
        for (int r = 0; r < df.RowCount; r++)
        {
            var keyParts = new string[keyColumns.Length];
            for (int k = 0; k < keyColumns.Length; k++)
                keyParts[k] = df[keyColumns[k]].GetObject(r)?.ToString() ?? "";
            var key = string.Join("|", keyParts);

            if (!groups.TryGetValue(key, out var list))
            {
                list = new List<int>();
                groups[key] = list;
            }
            list.Add(r);
        }

        // Aggregate the DataFrame using PandaSharp's GroupBy
        var grouped = GroupByExtensions.GroupBy(df, keyColumns);
        DataFrame aggDf = aggFunc switch
        {
            AggFunc.Sum => grouped.Sum(),
            AggFunc.Mean => grouped.Mean(),
            AggFunc.Min => grouped.Min(),
            AggFunc.Max => grouped.Max(),
            AggFunc.Count => grouped.Count(),
            AggFunc.First => grouped.First(),
            AggFunc.Last => grouped.Last(),
            AggFunc.Median => grouped.Median(),
            _ => grouped.Sum()
        };

        // Compute centroids per group (in the same order as the grouped DataFrame)
        var groupOrder = groups.Keys.ToList();
        var lats = new double[groupOrder.Count];
        var lons = new double[groupOrder.Count];

        for (int g = 0; g < groupOrder.Count; g++)
        {
            var indices = groups[groupOrder[g]];
            double sumLat = 0, sumLon = 0;
            foreach (var idx in indices)
            {
                sumLat += geo[idx].Latitude;
                sumLon += geo[idx].Longitude;
            }
            lats[g] = sumLat / indices.Count;
            lons[g] = sumLon / indices.Count;
        }

        var dissolved = new GeoColumn("geometry", lats, lons);
        return (aggDf, dissolved);
    }
}

/// <summary>Extension methods for dissolve on DataFrames with geometry.</summary>
public static class DissolveExtensions
{
    /// <summary>
    /// Dissolve: group by key, aggregate values, merge geometry.
    /// Returns a DataFrame with the centroid lat/lon added as columns.
    /// </summary>
    public static DataFrame Dissolve(this DataFrame df,
        string latColumn, string lonColumn,
        string keyColumn, AggFunc aggFunc = AggFunc.Sum,
        string centroidLatCol = "centroid_lat", string centroidLonCol = "centroid_lon")
    {
        var geo = df.ToGeoColumn(latColumn, lonColumn);
        var (data, dissolved) = DissolveOps.Dissolve(df, geo, keyColumn, aggFunc);

        // Add centroid columns to the aggregated result
        return data
            .AddColumn(new Column<double>(centroidLatCol, dissolved.Latitudes.ToArray()))
            .AddColumn(new Column<double>(centroidLonCol, dissolved.Longitudes.ToArray()));
    }
}
