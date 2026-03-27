namespace Cortex.GroupBy;

/// <summary>
/// Controls how null values are handled during GroupBy operations.
/// </summary>
public enum NullGroupingMode
{
    /// <summary>Null values form their own group (default).</summary>
    Include,
    /// <summary>Rows with null key values are excluded from all groups.</summary>
    Exclude
}
