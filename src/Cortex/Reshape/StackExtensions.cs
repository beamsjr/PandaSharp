using Cortex.Column;

namespace Cortex.Reshape;

public static class StackExtensions
{
    /// <summary>
    /// Stack: convert wide format to long format by "stacking" value columns into rows.
    /// Similar to Melt but with cleaner semantics.
    /// The id column identifies each row, remaining columns become variable/value pairs.
    /// </summary>
    public static DataFrame Stack(this DataFrame df, string idColumn,
        string varName = "variable", string valName = "value")
    {
        var valueCols = df.ColumnNames.Where(c => c != idColumn).ToArray();
        return df.Melt(idVars: [idColumn], valueVars: valueCols, varName: varName, valueName: valName);
    }

    /// <summary>
    /// Unstack: convert long format to wide format.
    /// The indexColumn identifies rows, the columnColumn values become new column headers,
    /// and the valueColumn values fill the cells.
    /// </summary>
    public static DataFrame Unstack(this DataFrame df, string indexColumn, string columnColumn, string valueColumn)
    {
        return df.Pivot(index: indexColumn, columns: columnColumn, values: valueColumn);
    }
}
