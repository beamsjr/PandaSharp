using System.Numerics;
using PandaSharp.Column;

namespace PandaSharp.Window;

public static class WindowExtensions
{
    public static RollingWindow<T> Rolling<T>(this Column<T> column, int windowSize, int minPeriods = -1, bool center = false)
        where T : struct, INumber<T>
        => new(column, windowSize, minPeriods, center);

    public static ExpandingWindow<T> Expanding<T>(this Column<T> column, int minPeriods = 1)
        where T : struct, INumber<T>
        => new(column, minPeriods);

    public static EwmWindow<T> Ewm<T>(this Column<T> column, double? span = null, double? alpha = null)
        where T : struct, INumber<T>
        => new(column, span, alpha);
}
