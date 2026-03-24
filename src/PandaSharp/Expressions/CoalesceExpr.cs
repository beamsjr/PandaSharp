using PandaSharp.Column;

namespace PandaSharp.Expressions;

/// <summary>
/// Returns the first non-null value from a list of expressions.
/// Usage: Expr.Coalesce(Col("A"), Col("B"), Lit(0))
/// </summary>
internal class CoalesceExpr : Expr
{
    private readonly Expr[] _exprs;
    public override string Name => $"coalesce({string.Join(", ", _exprs.Select(e => e.Name))})";

    public CoalesceExpr(Expr[] exprs) => _exprs = exprs;

    public override IColumn Evaluate(DataFrame df)
    {
        var cols = _exprs.Select(e => e.Evaluate(df)).ToArray();
        int length = cols[0].Length;

        // Try to determine if result is string or numeric
        bool isString = cols.Any(c => c is StringColumn);

        if (isString)
        {
            var result = new string?[length];
            for (int i = 0; i < length; i++)
            {
                foreach (var col in cols)
                {
                    if (!col.IsNull(i))
                    {
                        result[i] = col.GetObject(i)?.ToString();
                        break;
                    }
                }
            }
            return new StringColumn(Name, result);
        }
        else
        {
            var result = new double?[length];
            for (int i = 0; i < length; i++)
            {
                foreach (var col in cols)
                {
                    if (!col.IsNull(i))
                    {
                        result[i] = Convert.ToDouble(col.GetObject(i));
                        break;
                    }
                }
            }
            return Column<double>.FromNullable(Name, result);
        }
    }
}

public partial class Expr
{
    /// <summary>
    /// Returns the first non-null value from a list of expressions.
    /// </summary>
    public static Expr Coalesce(params Expr[] exprs) => new CoalesceExpr(exprs);
}
