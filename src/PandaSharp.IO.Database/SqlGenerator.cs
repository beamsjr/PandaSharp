using System.Text;
using PandaSharp.Expressions;
using PandaSharp.Lazy;

namespace PandaSharp.IO.Database;

/// <summary>
/// Compiles a LazyFrame logical plan into parameterized SQL.
/// Pushes filters, projections, sorts, limits, and aggregations to the database.
/// </summary>
public class SqlGenerator
{
    private readonly SqlDialect _dialect;
    private readonly List<(string Name, object? Value)> _parameters = new();
    private int _paramIndex;

    public SqlGenerator(SqlDialect dialect) => _dialect = dialect;

    /// <summary>Parameters accumulated during expression compilation.</summary>
    public IReadOnlyList<(string Name, object? Value)> Parameters => _parameters;

    /// <summary>
    /// Generate SQL from a logical plan rooted at a database scan.
    /// Returns the SQL string and parameter list.
    /// </summary>
    public (string Sql, List<(string Name, object? Value)> Parameters) Generate(LogicalPlan plan, string tableName)
    {
        _parameters.Clear();
        _paramIndex = 0;

        var query = AnalyzePlan(plan, tableName);
        return (query.ToSql(_dialect), _parameters.ToList());
    }

    /// <summary>
    /// Generate SQL from an expression-based filter/select chain.
    /// </summary>
    public (string Sql, List<(string Name, object? Value)> Parameters) GenerateFromExpressions(
        string tableName,
        string[]? selectColumns = null,
        Expr? filter = null,
        string[]? groupByColumns = null,
        (string Column, bool Ascending)[]? orderBy = null,
        int? limit = null,
        int? offset = null,
        bool distinct = false)
    {
        _parameters.Clear();
        _paramIndex = 0;

        var sb = new StringBuilder();
        sb.Append("SELECT ");

        if (distinct) sb.Append("DISTINCT ");

        // SELECT columns
        if (selectColumns is not null && selectColumns.Length > 0)
            sb.Append(string.Join(", ", selectColumns.Select(c => _dialect.QuoteIdentifier(c))));
        else
            sb.Append('*');

        sb.Append($" FROM {_dialect.QuoteIdentifier(tableName)}");

        // WHERE
        if (filter is not null)
        {
            sb.Append(" WHERE ");
            sb.Append(ExprToSql(filter));
        }

        // GROUP BY
        if (groupByColumns is not null && groupByColumns.Length > 0)
        {
            sb.Append(" GROUP BY ");
            sb.Append(string.Join(", ", groupByColumns.Select(c => _dialect.QuoteIdentifier(c))));
        }

        // ORDER BY
        if (orderBy is not null && orderBy.Length > 0)
        {
            sb.Append(" ORDER BY ");
            sb.Append(string.Join(", ", orderBy.Select(o =>
                $"{_dialect.QuoteIdentifier(o.Column)} {(o.Ascending ? "ASC" : "DESC")}")));
        }

        // LIMIT / OFFSET
        if (limit.HasValue && offset.HasValue)
            sb.Append($" {_dialect.OffsetClause(offset.Value, limit.Value)}");
        else if (limit.HasValue)
            sb.Append($" {_dialect.LimitClause(limit.Value)}");

        return (sb.ToString(), _parameters.ToList());
    }

    /// <summary>Compile a PandaSharp expression to a SQL fragment.</summary>
    public string ExprToSql(Expr expr)
    {
        return expr switch
        {
            ColExpr col => _dialect.QuoteIdentifier(col.Name),

            LitExpr<int> lit => AddParameter(lit.Name, int.Parse(lit.Name)),
            LitExpr<double> lit => AddParameter(lit.Name, double.Parse(lit.Name)),
            StringLitExpr lit => AddParameter(lit.Name, lit.Name),

            BinaryExpr bin => $"({ExprToSql(bin.Left)} {BinaryOpToSql(bin.Op)} {ExprToSql(bin.Right)})",
            ComparisonExpr comp => $"({ExprToSql(comp.Left)} {ComparisonOpToSql(comp.Op)} {ExprToSql(comp.Right)})",
            LogicalExpr logic => $"({ExprToSql(logic.Left)} {LogicalOpToSql(logic.Op)} {ExprToSql(logic.Right)})",
            NotExpr not => $"(NOT {ExprToSql(not.Inner)})",
            AliasExpr alias => $"{ExprToSql(alias.Inner)} AS {_dialect.QuoteIdentifier(alias.Name)}",

            AggExpr agg => AggExprToSql(agg),

            _ => throw new NotSupportedException($"Expression type {expr.GetType().Name} cannot be compiled to SQL.")
        };
    }

    private string AddParameter(string hint, object? value)
    {
        var name = $"{_dialect.ParameterPrefix}p{_paramIndex++}";
        _parameters.Add((name, value));
        return name;
    }

    private static string BinaryOpToSql(BinaryOp op) => op switch
    {
        BinaryOp.Add => "+",
        BinaryOp.Subtract => "-",
        BinaryOp.Multiply => "*",
        BinaryOp.Divide => "/",
        _ => throw new NotSupportedException($"Binary op {op}")
    };

    private static string ComparisonOpToSql(ComparisonOp op) => op switch
    {
        ComparisonOp.Gt => ">",
        ComparisonOp.Lt => "<",
        ComparisonOp.Gte => ">=",
        ComparisonOp.Lte => "<=",
        ComparisonOp.Eq => "=",
        ComparisonOp.Neq => "<>",
        _ => throw new NotSupportedException($"Comparison op {op}")
    };

    private static string LogicalOpToSql(LogicalOp op) => op switch
    {
        LogicalOp.And => "AND",
        LogicalOp.Or => "OR",
        _ => throw new NotSupportedException($"Logical op {op}")
    };

    private string AggExprToSql(AggExpr agg)
    {
        var colSql = ExprToSql(agg.Source);
        var op = agg.Op;
        return op switch
        {
            AggExprOp.Sum => $"SUM({colSql})",
            AggExprOp.Mean => $"AVG({colSql})",
            AggExprOp.Min => $"MIN({colSql})",
            AggExprOp.Max => $"MAX({colSql})",
            AggExprOp.Count => $"COUNT({colSql})",
            AggExprOp.Std => $"STDDEV({colSql})",
            AggExprOp.Median => $"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {colSql})",
            _ => throw new NotSupportedException($"Aggregate {op}")
        };
    }

    // -- Plan walking --
    // Walks the LogicalPlan tree bottom-up, accumulating SQL clauses.

    private QueryPlan AnalyzePlan(LogicalPlan plan, string tableName)
    {
        var qp = new QueryPlan { Table = tableName };
        WalkPlan(plan, qp);
        return qp;
    }

    private void WalkPlan(LogicalPlan plan, QueryPlan qp)
    {
        switch (plan)
        {
            case ScanPlan:
                // Base case — table name already set
                break;

            case FilterPlan filter:
                WalkPlan(filter.Input, qp);
                var whereSql = ExprToSql(filter.Predicate);
                qp.WhereFragments.Add(whereSql);
                break;

            case SelectPlan select:
                WalkPlan(select.Input, qp);
                qp.SelectColumns = select.Columns.ToList();
                break;

            case SortPlan sort:
                WalkPlan(sort.Input, qp);
                qp.OrderBy.Add((sort.Column, sort.Ascending));
                break;

            case HeadPlan head:
                WalkPlan(head.Input, qp);
                // Only take the smallest limit if multiple Heads are chained
                qp.Limit = qp.Limit.HasValue ? Math.Min(qp.Limit.Value, head.Count) : head.Count;
                break;

            case GroupByAggPlan groupBy:
                WalkPlan(groupBy.Input, qp);
                qp.GroupByColumns = groupBy.KeyColumns.ToList();
                qp.AggFunc = groupBy.AggFunc;
                break;

            case WithColumnPlan withCol:
                WalkPlan(withCol.Input, qp);
                var exprSql = ExprToSql(withCol.Expression);
                qp.ComputedColumns.Add((exprSql, withCol.Name));
                break;

            case JoinPlan join:
                WalkPlan(join.Left, qp);
                var joinKeyword = join.How switch
                {
                    PandaSharp.Joins.JoinType.Inner => "INNER JOIN",
                    PandaSharp.Joins.JoinType.Left => "LEFT JOIN",
                    PandaSharp.Joins.JoinType.Right => "RIGHT JOIN",
                    PandaSharp.Joins.JoinType.Outer => "FULL OUTER JOIN",
                    PandaSharp.Joins.JoinType.Cross => "CROSS JOIN",
                    _ => throw new NotSupportedException($"Join type {join.How} cannot be pushed to SQL.")
                };
                // Extract right table name from the plan leaf
                var rightTable = ExtractTableName(join.Right);
                if (rightTable is null)
                    throw new NotSupportedException("Join push-down requires the right side to reference a known table.");
                qp.Joins.Add(new JoinInfo(joinKeyword, join.On, rightTable, qp.Table));
                break;

            case ExternalScanPlan:
                // Base case for database-backed scans — table name already set
                break;

            default:
                throw new NotSupportedException(
                    $"Plan node {plan.GetType().Name} cannot be compiled to SQL. " +
                    "Collect this node client-side or restructure the query.");
        }
    }

    private class QueryPlan
    {
        public string Table { get; set; } = "";
        public List<string>? SelectColumns { get; set; }
        public List<string> WhereFragments { get; } = new();
        public List<(string Column, bool Ascending)> OrderBy { get; } = new();
        public int? Limit { get; set; }
        public List<string>? GroupByColumns { get; set; }
        public PandaSharp.GroupBy.AggFunc? AggFunc { get; set; }
        public List<(string ExprSql, string Alias)> ComputedColumns { get; } = new();
        public List<JoinInfo> Joins { get; } = new();

        public string ToSql(SqlDialect dialect)
        {
            var sb = new StringBuilder();

            // SELECT
            sb.Append("SELECT ");

            if (GroupByColumns is not null && AggFunc.HasValue)
            {
                // GroupBy + Agg: SELECT key_cols, AGG(non_key_cols)
                var parts = new List<string>();
                foreach (var col in GroupByColumns)
                    parts.Add(dialect.QuoteIdentifier(col));

                // When we have a GroupBy, emit the aggregate function on all
                // selected non-key columns. If SelectColumns is specified, use those;
                // otherwise, use * with the aggregate.
                var aggSql = AggFunc.Value switch
                {
                    PandaSharp.GroupBy.AggFunc.Sum => "SUM",
                    PandaSharp.GroupBy.AggFunc.Mean => "AVG",
                    PandaSharp.GroupBy.AggFunc.Min => "MIN",
                    PandaSharp.GroupBy.AggFunc.Max => "MAX",
                    PandaSharp.GroupBy.AggFunc.Count => "COUNT",
                    PandaSharp.GroupBy.AggFunc.Std => "STDDEV",
                    PandaSharp.GroupBy.AggFunc.Var => "VARIANCE",
                    PandaSharp.GroupBy.AggFunc.Median => "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY",
                    PandaSharp.GroupBy.AggFunc.First => "MIN", // approximate First as MIN
                    PandaSharp.GroupBy.AggFunc.Last => "MAX",  // approximate Last as MAX
                    _ => throw new NotSupportedException($"Aggregate {AggFunc}")
                };

                if (SelectColumns is not null)
                {
                    foreach (var col in SelectColumns)
                    {
                        if (!GroupByColumns.Contains(col))
                        {
                            if (AggFunc.Value == PandaSharp.GroupBy.AggFunc.Median)
                                parts.Add($"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {dialect.QuoteIdentifier(col)}) AS {dialect.QuoteIdentifier(col)}");
                            else
                                parts.Add($"{aggSql}({dialect.QuoteIdentifier(col)}) AS {dialect.QuoteIdentifier(col)}");
                        }
                    }
                }
                else
                {
                    // Emit AGG(*) for count, otherwise we can't know columns
                    if (AggFunc.Value == PandaSharp.GroupBy.AggFunc.Count)
                        parts.Add("COUNT(*) AS \"count\"");
                }

                sb.Append(string.Join(", ", parts));
            }
            else if (SelectColumns is not null && SelectColumns.Count > 0)
            {
                var parts = SelectColumns.Select(c => dialect.QuoteIdentifier(c)).ToList();
                foreach (var (exprSql, alias) in ComputedColumns)
                    parts.Add($"{exprSql} AS {dialect.QuoteIdentifier(alias)}");
                sb.Append(string.Join(", ", parts));
            }
            else if (ComputedColumns.Count > 0)
            {
                var parts = new List<string> { "*" };
                foreach (var (exprSql, alias) in ComputedColumns)
                    parts.Add($"{exprSql} AS {dialect.QuoteIdentifier(alias)}");
                sb.Append(string.Join(", ", parts));
            }
            else
            {
                // SQL Server puts TOP here
                if (Limit.HasValue && dialect is SqlServerDialect && GroupByColumns is null && OrderBy.Count == 0)
                {
                    sb.Append($"{dialect.LimitClause(Limit.Value)} ");
                }
                sb.Append('*');
            }

            // FROM
            sb.Append($" FROM {dialect.QuoteIdentifier(Table)}");

            // JOINs
            foreach (var join in Joins)
            {
                sb.Append($" {join.Keyword} {dialect.QuoteIdentifier(join.RightTable)}");
                if (join.Keyword != "CROSS JOIN")
                {
                    sb.Append($" ON {dialect.QuoteIdentifier(join.LeftTable)}.{dialect.QuoteIdentifier(join.OnColumn)}");
                    sb.Append($" = {dialect.QuoteIdentifier(join.RightTable)}.{dialect.QuoteIdentifier(join.OnColumn)}");
                }
            }

            // WHERE
            if (WhereFragments.Count > 0)
            {
                sb.Append(" WHERE ");
                sb.Append(string.Join(" AND ", WhereFragments.Select(f => $"({f})")));
            }

            // GROUP BY
            if (GroupByColumns is not null && GroupByColumns.Count > 0)
            {
                sb.Append(" GROUP BY ");
                sb.Append(string.Join(", ", GroupByColumns.Select(c => dialect.QuoteIdentifier(c))));
            }

            // ORDER BY
            if (OrderBy.Count > 0)
            {
                sb.Append(" ORDER BY ");
                sb.Append(string.Join(", ", OrderBy.Select(o =>
                    $"{dialect.QuoteIdentifier(o.Column)} {(o.Ascending ? "ASC" : "DESC")}")));
            }

            // LIMIT (non-SQL Server, or SQL Server with ORDER BY which needs OFFSET/FETCH)
            if (Limit.HasValue)
            {
                bool alreadyEmittedTop = dialect is SqlServerDialect && GroupByColumns is null && OrderBy.Count == 0;
                if (!alreadyEmittedTop)
                {
                    if (dialect is SqlServerDialect && OrderBy.Count > 0)
                        sb.Append($" {dialect.OffsetClause(0, Limit.Value)}");
                    else if (dialect is not SqlServerDialect)
                        sb.Append($" {dialect.LimitClause(Limit.Value)}");
                }
            }

            return sb.ToString();
        }
    }

    private static string? ExtractTableName(LogicalPlan plan) => plan switch
    {
        ExternalScanPlan ext when ext.SourceDescription.StartsWith("db://")
            => ext.SourceDescription["db://".Length..],
        ScanPlan => null, // in-memory scan, can't push to SQL
        SelectPlan s => ExtractTableName(s.Input),
        FilterPlan f => ExtractTableName(f.Input),
        _ => null
    };

    internal record JoinInfo(string Keyword, string OnColumn, string RightTable, string LeftTable);
}
