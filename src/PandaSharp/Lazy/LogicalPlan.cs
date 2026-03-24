using PandaSharp.Expressions;

namespace PandaSharp.Lazy;

public abstract record LogicalPlan;

public record ScanPlan(DataFrame Source) : LogicalPlan;

public record SelectPlan(LogicalPlan Input, string[] Columns) : LogicalPlan;

public record FilterPlan(LogicalPlan Input, Expr Predicate) : LogicalPlan;

public record SortPlan(LogicalPlan Input, string Column, bool Ascending) : LogicalPlan;

public record WithColumnPlan(LogicalPlan Input, Expr Expression, string Name) : LogicalPlan;

public record HeadPlan(LogicalPlan Input, int Count) : LogicalPlan;

public record GroupByAggPlan(LogicalPlan Input, string[] KeyColumns, GroupBy.AggFunc AggFunc) : LogicalPlan;

public record JoinPlan(LogicalPlan Left, LogicalPlan Right, string On, Joins.JoinType How) : LogicalPlan;

/// <summary>
/// A scan backed by an external data source (e.g., database).
/// The executor function receives the full plan tree above this node and returns a materialized DataFrame.
/// This enables push-down: the external source can inspect the plan and execute operations natively.
/// </summary>
public record ExternalScanPlan(Func<LogicalPlan, DataFrame> Executor, string SourceDescription) : LogicalPlan;
