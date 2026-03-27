namespace Cortex.Lazy;

/// <summary>
/// Simple query optimizer applying predicate pushdown and projection pushdown.
/// </summary>
internal static class Optimizer
{
    public static LogicalPlan Optimize(LogicalPlan plan)
    {
        plan = PushDownPredicates(plan);
        plan = PushDownProjections(plan);
        return plan;
    }

    /// <summary>
    /// Push Filter nodes closer to the Scan to reduce rows early.
    /// </summary>
    private static LogicalPlan PushDownPredicates(LogicalPlan plan)
    {
        // If we have Select(Filter(...)), swap to Filter(Select(...)) only if filter
        // columns are in the select list — for now, push Filter past Sort and WithColumn.
        return plan switch
        {
            // Filter over Sort → Sort over Filter (filter first = fewer rows to sort)
            FilterPlan { Input: SortPlan sort } filter =>
                new SortPlan(
                    PushDownPredicates(new FilterPlan(sort.Input, filter.Predicate)),
                    sort.Column, sort.Ascending),

            // Filter over WithColumn → push filter below if it doesn't reference the new column
            // (conservative: don't push if we can't determine column refs)

            // Recurse into children
            SelectPlan s => s with { Input = PushDownPredicates(s.Input) },
            FilterPlan f => f with { Input = PushDownPredicates(f.Input) },
            SortPlan s => s with { Input = PushDownPredicates(s.Input) },
            WithColumnPlan w => w with { Input = PushDownPredicates(w.Input) },
            HeadPlan h => h with { Input = PushDownPredicates(h.Input) },
            GroupByAggPlan g => g with { Input = PushDownPredicates(g.Input) },
            JoinPlan j => j with { Left = PushDownPredicates(j.Left), Right = PushDownPredicates(j.Right) },

            _ => plan
        };
    }

    /// <summary>
    /// Push Select (projection) closer to Scan to reduce columns early.
    /// When a SelectPlan sits above Sort/Filter/Head, push it down past them
    /// while ensuring that columns needed by those nodes are included.
    /// </summary>
    private static LogicalPlan PushDownProjections(LogicalPlan plan)
    {
        return plan switch
        {
            // Select over Sort → push wider Select below Sort, keep narrow Select on top
            SelectPlan { Input: SortPlan sort } select when !select.Columns.Contains(sort.Column) =>
                new SelectPlan(
                    new SortPlan(
                        PushDownProjections(new SelectPlan(sort.Input, MergeColumns(select.Columns, sort.Column))),
                        sort.Column, sort.Ascending),
                    select.Columns),

            // Select over Sort → sort column already in projection, just push through
            SelectPlan { Input: SortPlan sort } select =>
                new SortPlan(
                    PushDownProjections(new SelectPlan(sort.Input, select.Columns)),
                    sort.Column, sort.Ascending),

            // Select over Filter → push wider Select below Filter, keep narrow Select on top
            SelectPlan { Input: FilterPlan filter } select =>
                WrapIfNeeded(select.Columns,
                    new FilterPlan(
                        PushDownProjections(new SelectPlan(filter.Input,
                            MergeColumns(select.Columns, ExtractColumnRefs(filter.Predicate)))),
                        filter.Predicate)),

            // Select over Head → push Select below Head
            SelectPlan { Input: HeadPlan head } select =>
                new HeadPlan(
                    PushDownProjections(new SelectPlan(head.Input, select.Columns)),
                    head.Count),

            // Select over Select → merge projections
            SelectPlan { Input: SelectPlan inner } select =>
                PushDownProjections(new SelectPlan(inner.Input,
                    select.Columns.Intersect(inner.Columns).ToArray()
                    is { Length: > 0 } merged ? merged : select.Columns)),

            // Recurse into children for other node types
            SelectPlan s => s with { Input = PushDownProjections(s.Input) },
            FilterPlan f => f with { Input = PushDownProjections(f.Input) },
            SortPlan s => s with { Input = PushDownProjections(s.Input) },
            WithColumnPlan w => w with { Input = PushDownProjections(w.Input) },
            HeadPlan h => h with { Input = PushDownProjections(h.Input) },
            GroupByAggPlan g => g with { Input = PushDownProjections(g.Input) },
            JoinPlan j => j with { Left = PushDownProjections(j.Left), Right = PushDownProjections(j.Right) },
            _ => plan
        };
    }

    /// <summary>Wrap a plan in a SelectPlan if the inner plan has more columns than needed.</summary>
    private static LogicalPlan WrapIfNeeded(string[] desiredColumns, LogicalPlan inner)
    {
        // If the inner plan's Select already matches, no wrapping needed
        if (inner is SelectPlan s && s.Columns.SequenceEqual(desiredColumns))
            return inner;
        return new SelectPlan(inner, desiredColumns);
    }

    /// <summary>Merge extra columns into a projection list.</summary>
    private static string[] MergeColumns(string[] projection, string extraColumn)
    {
        if (projection.Contains(extraColumn)) return projection;
        return [.. projection, extraColumn];
    }

    private static string[] MergeColumns(string[] projection, HashSet<string> extraColumns)
    {
        var merged = new HashSet<string>(projection);
        merged.UnionWith(extraColumns);
        return merged.ToArray();
    }

    /// <summary>
    /// Extract column name references from an expression.
    /// Used to determine which columns a filter predicate needs.
    /// </summary>
    private static HashSet<string> ExtractColumnRefs(Expressions.Expr expr)
    {
        var refs = new HashSet<string>();
        CollectColumnRefs(expr, refs);
        return refs;
    }

    private static void CollectColumnRefs(Expressions.Expr expr, HashSet<string> refs)
    {
        switch (expr)
        {
            case Expressions.ColExpr col:
                refs.Add(col.Name);
                break;
            case Expressions.BinaryExpr bin:
                CollectColumnRefs(bin.Left, refs);
                CollectColumnRefs(bin.Right, refs);
                break;
            case Expressions.ComparisonExpr comp:
                CollectColumnRefs(comp.Left, refs);
                CollectColumnRefs(comp.Right, refs);
                break;
            case Expressions.LogicalExpr logic:
                CollectColumnRefs(logic.Left, refs);
                CollectColumnRefs(logic.Right, refs);
                break;
            case Expressions.NotExpr not:
                CollectColumnRefs(not.Inner, refs);
                break;
            case Expressions.AliasExpr alias:
                CollectColumnRefs(alias.Inner, refs);
                break;
        }
    }
}
