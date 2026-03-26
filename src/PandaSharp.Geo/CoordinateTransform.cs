using ProjNet;
using ProjNet.CoordinateSystems;
using ProjNet.CoordinateSystems.Transformations;

namespace PandaSharp.Geo;

/// <summary>
/// Coordinate reference system (CRS) transformation using ProjNet.
/// Transforms GeoColumn coordinates between different projections
/// (e.g., WGS84 ↔ UTM, WGS84 ↔ Web Mercator).
///
/// Usage:
///   var utm = geo.Reproject(Crs.Wgs84, Crs.Utm(18));
///   var mercator = geo.Reproject(Crs.Wgs84, Crs.WebMercator);
/// </summary>
public static class CoordinateTransform
{
    private static readonly CoordinateTransformationFactory _factory = new();

    /// <summary>
    /// Reproject a GeoColumn from one CRS to another.
    /// </summary>
    public static GeoColumn Reproject(this GeoColumn geo, CoordinateSystem source, CoordinateSystem target)
    {
        // Validate bounds only when source is a geographic CRS (lat/lon in degrees)
        bool isGeographic = source is GeographicCoordinateSystem;
        var transform = _factory.CreateFromCoordinateSystems(source, target);
        var mt = transform.MathTransform;

        var lats = new double[geo.Length];
        var lons = new double[geo.Length];

        for (int i = 0; i < geo.Length; i++)
        {
            var point = geo[i];
            if (isGeographic)
            {
                if (point.Latitude < -90 || point.Latitude > 90)
                    throw new ArgumentOutOfRangeException($"Latitude at index {i} is {point.Latitude}, must be between -90 and 90.");
                if (point.Longitude < -180 || point.Longitude > 180)
                    throw new ArgumentOutOfRangeException($"Longitude at index {i} is {point.Longitude}, must be between -180 and 180.");
            }
            // ProjNet uses (x=lon, y=lat) order for geographic CRS
            var projected = mt.Transform(new[] { point.Longitude, point.Latitude });
            lons[i] = projected[0];
            lats[i] = projected[1];
        }

        return new GeoColumn(geo.Name, lats, lons);
    }

    /// <summary>Reproject using EPSG codes.</summary>
    public static GeoColumn Reproject(this GeoColumn geo, int fromEpsg, int toEpsg)
    {
        return geo.Reproject(Crs.FromEpsg(fromEpsg), Crs.FromEpsg(toEpsg));
    }
}

/// <summary>Common coordinate reference systems.</summary>
public static class Crs
{
    private static readonly CoordinateSystemFactory _factory = new();

    /// <summary>WGS 84 — standard GPS coordinates (EPSG:4326).</summary>
    public static CoordinateSystem Wgs84 => GeographicCoordinateSystem.WGS84;

    /// <summary>Web Mercator (EPSG:3857).</summary>
    public static CoordinateSystem WebMercator => (CoordinateSystem)_factory.CreateFromWkt(
        "PROJCS[\"WGS 84 / Pseudo-Mercator\"," +
        "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]]," +
        "PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]," +
        "PROJECTION[\"Mercator_2SP\"]," +
        "PARAMETER[\"standard_parallel_1\",0]," +
        "PARAMETER[\"central_meridian\",0]," +
        "PARAMETER[\"false_easting\",0]," +
        "PARAMETER[\"false_northing\",0]," +
        "UNIT[\"metre\",1]]");

    /// <summary>Create a UTM zone CRS.</summary>
    public static CoordinateSystem Utm(int zone, bool north = true)
    {
        double cm = (zone - 1) * 6 - 180 + 3;
        double fn = north ? 0 : 10000000;
        string h = north ? "N" : "S";
        return (CoordinateSystem)_factory.CreateFromWkt(
            $"PROJCS[\"WGS 84 / UTM zone {zone}{h}\"," +
            "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]]," +
            "PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]," +
            "PROJECTION[\"Transverse_Mercator\"]," +
            "PARAMETER[\"latitude_of_origin\",0]," +
            $"PARAMETER[\"central_meridian\",{cm}]," +
            "PARAMETER[\"scale_factor\",0.9996]," +
            "PARAMETER[\"false_easting\",500000]," +
            $"PARAMETER[\"false_northing\",{fn}]," +
            "UNIT[\"metre\",1]]");
    }

    /// <summary>Create CRS from EPSG code.</summary>
    public static CoordinateSystem FromEpsg(int epsg) => epsg switch
    {
        4326 => Wgs84,
        3857 => WebMercator,
        >= 32601 and <= 32660 => Utm(epsg - 32600, north: true),
        >= 32701 and <= 32760 => Utm(epsg - 32700, north: false),
        _ => throw new ArgumentException($"EPSG:{epsg} not in built-in catalog.")
    };

    /// <summary>Create CRS from WKT.</summary>
    public static CoordinateSystem FromWkt(string wkt) => (CoordinateSystem)_factory.CreateFromWkt(wkt);
}
