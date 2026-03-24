using PandaSharp.Column;

namespace PandaSharp.Expressions;

/// <summary>
/// Aggregate expression: evaluates a column and reduces to a scalar broadcast to all rows.
/// Usage: Col("Salary").Sum(), Col("Age").Mean()
/// </summary>
public enum AggExprOp { Sum, Mean, Min, Max, Count, Std, Median }

internal class AggExpr : Expr
{
    private readonly Expr _source;
    private readonly AggExprOp _op;
    internal Expr Source => _source;
    internal AggExprOp Op => _op;
    public override string Name => $"{_source.Name}.{_op.ToString().ToLower()}";

    public AggExpr(Expr source, AggExprOp op) { _source = source; _op = op; }

    public override IColumn Evaluate(DataFrame df)
    {
        var col = _source.Evaluate(df);

        // Collect non-null values as doubles
        var values = new List<double>();
        for (int i = 0; i < col.Length; i++)
            if (!col.IsNull(i)) values.Add(Convert.ToDouble(col.GetObject(i)));

        double? scalar = values.Count == 0 ? null : _op switch
        {
            AggExprOp.Sum => values.Sum(),
            AggExprOp.Mean => values.Average(),
            AggExprOp.Min => values.Min(),
            AggExprOp.Max => values.Max(),
            AggExprOp.Count => values.Count,
            AggExprOp.Std => values.Count > 1
                ? Math.Sqrt(values.Sum(v => (v - values.Average()) * (v - values.Average())) / (values.Count - 1))
                : 0,
            AggExprOp.Median => ComputeMedian(values),
            _ => null
        };

        // Broadcast scalar to all rows
        var result = new double?[col.Length];
        Array.Fill(result, scalar);
        return Column<double>.FromNullable(Name, result);
    }

    private static double ComputeMedian(List<double> vals)
    {
        vals.Sort();
        int mid = vals.Count / 2;
        return vals.Count % 2 == 0 ? (vals[mid - 1] + vals[mid]) / 2 : vals[mid];
    }
}

public static class ExprAggExtensions
{
    /// <summary>Sum of column, broadcast to all rows.</summary>
    public static Expr Sum(this Expr expr) => new AggExpr(expr, AggExprOp.Sum);
    /// <summary>Mean of column, broadcast to all rows.</summary>
    public static Expr Mean(this Expr expr) => new AggExpr(expr, AggExprOp.Mean);
    /// <summary>Min of column, broadcast to all rows.</summary>
    public static Expr Min(this Expr expr) => new AggExpr(expr, AggExprOp.Min);
    /// <summary>Max of column, broadcast to all rows.</summary>
    public static Expr Max(this Expr expr) => new AggExpr(expr, AggExprOp.Max);
    /// <summary>Count of non-null values, broadcast to all rows.</summary>
    public static Expr Count(this Expr expr) => new AggExpr(expr, AggExprOp.Count);
    /// <summary>Standard deviation, broadcast to all rows.</summary>
    public static Expr StdExpr(this Expr expr) => new AggExpr(expr, AggExprOp.Std);
    /// <summary>Median, broadcast to all rows.</summary>
    public static Expr MedianExpr(this Expr expr) => new AggExpr(expr, AggExprOp.Median);
}
