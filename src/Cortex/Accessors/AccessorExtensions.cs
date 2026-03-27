using Cortex.Column;

namespace Cortex.Accessors;

public static class AccessorExtensions
{
    /// <summary>
    /// Access vectorized datetime operations on a DateTime column.
    /// </summary>
    public static DateTimeAccessor Dt(this Column<DateTime> column) => new(column);
}
