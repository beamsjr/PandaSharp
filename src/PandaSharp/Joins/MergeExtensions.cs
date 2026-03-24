namespace PandaSharp.Joins;

/// <summary>
/// Merge is a pandas-style alias for Join with slightly different defaults.
/// </summary>
public static class MergeExtensions
{
    /// <summary>
    /// Merge two DataFrames (pandas-style alias for Join).
    /// Default is inner join.
    /// </summary>
    public static DataFrame Merge(this DataFrame left, DataFrame right, string on,
        JoinType how = JoinType.Inner, string leftSuffix = "_x", string rightSuffix = "_y")
    {
        return left.Join(right, on, how, leftSuffix, rightSuffix);
    }

    public static DataFrame Merge(this DataFrame left, DataFrame right,
        string leftOn, string rightOn,
        JoinType how = JoinType.Inner, string leftSuffix = "_x", string rightSuffix = "_y")
    {
        return left.Join(right, leftOn, rightOn, how, leftSuffix, rightSuffix);
    }
}
