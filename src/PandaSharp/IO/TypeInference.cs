using System.Globalization;

namespace PandaSharp.IO;

internal static class TypeInference
{
    public static Type InferType(IReadOnlyList<string?> samples, string? dateFormat = null)
    {
        bool allNull = true;
        bool canBeInt = true;
        bool canBeLong = true;
        bool canBeDouble = true;
        bool canBeBool = true;
        bool canBeDateTime = true;

        foreach (var s in samples)
        {
            if (string.IsNullOrEmpty(s)) continue;
            allNull = false;

            if (canBeInt && !int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out _))
                canBeInt = false;
            if (!canBeInt && canBeLong && !long.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out _))
                canBeLong = false;
            if (!canBeLong && canBeDouble && !double.TryParse(s, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out _))
                canBeDouble = false;
            if (canBeBool && !IsBoolString(s))
                canBeBool = false;
            if (canBeDateTime && !TryParseDateTime(s, dateFormat))
                canBeDateTime = false;
        }

        if (allNull) return typeof(string);
        if (canBeInt) return typeof(int);
        if (canBeLong) return typeof(long);
        if (canBeDouble) return typeof(double);
        if (canBeBool) return typeof(bool);
        if (canBeDateTime) return typeof(DateTime);
        return typeof(string);
    }

    private static bool IsBoolString(string s) =>
        s.Equals("true", StringComparison.OrdinalIgnoreCase) ||
        s.Equals("false", StringComparison.OrdinalIgnoreCase);

    private static bool TryParseDateTime(string s, string? format)
    {
        if (format is not null)
            return DateTime.TryParseExact(s, format, CultureInfo.InvariantCulture, DateTimeStyles.None, out _);
        return DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.None, out _);
    }
}
