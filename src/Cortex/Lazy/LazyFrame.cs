using Cortex.Expressions;
using Cortex.GroupBy;

namespace Cortex.Lazy;

/// <summary>
/// A deferred DataFrame that records operations as a query plan.
/// Call .Collect() to materialize.
/// </summary>
public class LazyFrame
{
    private LogicalPlan _plan;

    internal LazyFrame(LogicalPlan plan) => _plan = plan;

    public LazyFrame Select(params string[] columns) =>
        new(new SelectPlan(_plan, columns));

    public LazyFrame Filter(Expr predicate) =>
        new(new FilterPlan(_plan, predicate));

    public LazyFrame Sort(string column, bool ascending = true) =>
        new(new SortPlan(_plan, column, ascending));

    public LazyFrame WithColumn(Expr expression, string name) =>
        new(new WithColumnPlan(_plan, expression, name));

    public LazyFrame Head(int count = 5) =>
        new(new HeadPlan(_plan, count));

    /// <summary>
    /// Add multiple computed columns in a single lazy step.
    /// </summary>
    public LazyFrame WithColumns(params (Expr Expression, string Name)[] columns)
    {
        var result = this;
        foreach (var (expr, name) in columns)
            result = result.WithColumn(expr, name);
        return result;
    }

    /// <summary>
    /// Lazy GroupBy returning a LazyGroupedFrame for deferred aggregation.
    /// </summary>
    public LazyGroupedFrame GroupBy(params string[] keyColumns) =>
        new(_plan, keyColumns);

    /// <summary>
    /// Lazy join with another LazyFrame.
    /// </summary>
    public LazyFrame Join(LazyFrame right, string on, Joins.JoinType how = Joins.JoinType.Inner) =>
        new(new JoinPlan(_plan, right._plan, on, how));

    /// <summary>
    /// Get the logical plan (for inspection/debugging).
    /// </summary>
    public LogicalPlan Plan => _plan;

    /// <summary>
    /// Optimize and materialize the query plan into a concrete DataFrame.
    /// </summary>
    public DataFrame Collect()
    {
        var optimized = Optimizer.Optimize(_plan);
        return Executor.Execute(optimized);
    }

    /// <summary>
    /// Describe the plan as a string for debugging.
    /// </summary>
    public string Explain()
    {
        var optimized = Optimizer.Optimize(_plan);
        return FormatPlan(optimized, 0);
    }

    private static string FormatPlan(LogicalPlan plan, int indent)
    {
        var prefix = new string(' ', indent * 2);
        return plan switch
        {
            ScanPlan s => $"{prefix}Scan [{s.Source.RowCount} rows x {s.Source.ColumnCount} cols]",
            SelectPlan s => $"{prefix}Select [{string.Join(", ", s.Columns)}]\n{FormatPlan(s.Input, indent + 1)}",
            FilterPlan f => $"{prefix}Filter [{f.Predicate.Name}]\n{FormatPlan(f.Input, indent + 1)}",
            SortPlan s => $"{prefix}Sort [{s.Column} {(s.Ascending ? "ASC" : "DESC")}]\n{FormatPlan(s.Input, indent + 1)}",
            WithColumnPlan w => $"{prefix}WithColumn [{w.Name} = {w.Expression.Name}]\n{FormatPlan(w.Input, indent + 1)}",
            HeadPlan h => $"{prefix}Head [{h.Count}]\n{FormatPlan(h.Input, indent + 1)}",
            GroupByAggPlan g => $"{prefix}GroupBy [{string.Join(", ", g.KeyColumns)}].{g.AggFunc}\n{FormatPlan(g.Input, indent + 1)}",
            JoinPlan j => $"{prefix}Join [{j.On}, {j.How}]\n{FormatPlan(j.Left, indent + 1)}\n{FormatPlan(j.Right, indent + 1)}",
            ExternalScanPlan ext => $"{prefix}ExternalScan [{ext.SourceDescription}]",
            _ => $"{prefix}Unknown"
        };
    }
}

/// <summary>
/// Deferred GroupBy result — records the aggregation to apply when Collect() is called.
/// </summary>
public class LazyGroupedFrame
{
    private readonly LogicalPlan _input;
    private readonly string[] _keyColumns;

    internal LazyGroupedFrame(LogicalPlan input, string[] keyColumns)
    {
        _input = input;
        _keyColumns = keyColumns;
    }

    public LazyFrame Sum() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.Sum));
    public LazyFrame Mean() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.Mean));
    public LazyFrame Min() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.Min));
    public LazyFrame Max() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.Max));
    public LazyFrame Count() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.Count));
    public LazyFrame Median() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.Median));
    public LazyFrame Std() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.Std));
    public LazyFrame Var() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.Var));
    public LazyFrame First() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.First));
    public LazyFrame Last() => new(new GroupByAggPlan(_input, _keyColumns, AggFunc.Last));
}

public static class LazyFrameExtensions
{
    /// <summary>
    /// Convert a DataFrame to a LazyFrame for deferred execution.
    /// </summary>
    public static LazyFrame Lazy(this DataFrame df) => new(new ScanPlan(df));
}
