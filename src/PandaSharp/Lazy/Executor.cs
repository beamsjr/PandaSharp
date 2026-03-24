using PandaSharp.Expressions;
using PandaSharp.GroupBy;
using PandaSharp.Joins;

namespace PandaSharp.Lazy;

/// <summary>
/// Materializes a LogicalPlan into a concrete DataFrame.
/// </summary>
internal static class Executor
{
    public static DataFrame Execute(LogicalPlan plan)
    {
        // Check if the plan tree is rooted on an external scan — if so, let it
        // push down the entire plan to the external source (e.g., database).
        if (FindExternalScan(plan) is { } externalScan)
            return externalScan.Executor(plan);

        return ExecuteLocal(plan);
    }

    private static ExternalScanPlan? FindExternalScan(LogicalPlan plan)
    {
        return plan switch
        {
            ExternalScanPlan ext => ext,
            ScanPlan => null,
            SelectPlan s => FindExternalScan(s.Input),
            FilterPlan f => FindExternalScan(f.Input),
            SortPlan s => FindExternalScan(s.Input),
            WithColumnPlan w => FindExternalScan(w.Input),
            HeadPlan h => FindExternalScan(h.Input),
            GroupByAggPlan g => FindExternalScan(g.Input),
            JoinPlan j => FindExternalScan(j.Left) ?? FindExternalScan(j.Right),
            _ => null
        };
    }

    internal static DataFrame ExecuteLocal(LogicalPlan plan)
    {
        return plan switch
        {
            ScanPlan scan => scan.Source,

            ExternalScanPlan ext => ext.Executor(plan),

            SelectPlan select => ExecuteLocal(select.Input).Select(select.Columns),

            FilterPlan filter => ExprExtensions.Filter(ExecuteLocal(filter.Input), filter.Predicate),

            SortPlan sort => ExecuteLocal(sort.Input).Sort(sort.Column, sort.Ascending),

            WithColumnPlan withCol => ExprExtensions.WithColumn(ExecuteLocal(withCol.Input), withCol.Expression, withCol.Name),

            HeadPlan head => ExecuteLocal(head.Input).Head(head.Count),

            GroupByAggPlan groupBy => ExecuteGroupByAgg(ExecuteLocal(groupBy.Input), groupBy.KeyColumns, groupBy.AggFunc),

            JoinPlan join => ExecuteLocal(join.Left).Join(ExecuteLocal(join.Right), join.On, join.How),

            _ => throw new NotSupportedException($"Unknown plan node: {plan.GetType().Name}")
        };
    }

    private static DataFrame ExecuteGroupByAgg(DataFrame df, string[] keyColumns, AggFunc func)
    {
        var grouped = df.GroupBy(keyColumns);
        return func switch
        {
            AggFunc.Sum => grouped.Sum(),
            AggFunc.Mean => grouped.Mean(),
            AggFunc.Min => grouped.Min(),
            AggFunc.Max => grouped.Max(),
            AggFunc.Count => grouped.Count(),
            AggFunc.Median => grouped.Median(),
            AggFunc.Std => grouped.Std(),
            AggFunc.Var => grouped.Var(),
            AggFunc.First => grouped.First(),
            AggFunc.Last => grouped.Last(),
            _ => throw new NotSupportedException($"Unknown aggregate: {func}")
        };
    }
}
