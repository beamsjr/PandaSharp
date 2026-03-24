using PandaSharp.Column;

namespace PandaSharp.Expressions;

public static class ExprExtensions
{
    /// <summary>
    /// Add a computed column using an expression.
    /// df.WithColumn(Col("price") * Col("qty"), "total")
    /// </summary>
    public static DataFrame WithColumn(this DataFrame df, Expr expr, string? name = null)
    {
        var col = expr.Evaluate(df);
        if (name is not null)
            col = col.Clone(name);
        return df.AddColumn(col);
    }

    /// <summary>
    /// Select columns using expressions.
    /// df.Select(Col("name"), (Col("price") * Col("qty")).Alias("total"))
    /// </summary>
    public static DataFrame Select(this DataFrame df, params Expr[] exprs)
    {
        var columns = new List<IColumn>();
        foreach (var expr in exprs)
        {
            var col = expr.Evaluate(df);
            if (expr is AliasExpr)
                col = col.Clone(expr.Name);
            columns.Add(col);
        }
        return new DataFrame(columns);
    }

    /// <summary>
    /// Filter using a boolean expression.
    /// df.Filter(Col("age") > Lit(30))
    /// </summary>
    public static DataFrame Filter(this DataFrame df, Expr predicate)
    {
        var col = predicate.Evaluate(df);
        if (col is not Column<bool> boolCol)
            throw new ArgumentException("Filter expression must evaluate to a boolean column.");

        var mask = new bool[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
            mask[i] = boolCol[i] ?? false;

        return df.Filter(mask);
    }

    /// <summary>
    /// Add multiple computed columns at once.
    /// df.WithColumns((Col("price") * Col("qty")).Alias("total"), Col("name").Str.Upper().Alias("NAME"))
    /// </summary>
    public static DataFrame WithColumns(this DataFrame df, params Expr[] exprs)
    {
        var result = df;
        foreach (var expr in exprs)
        {
            var col = expr.Evaluate(df); // evaluate against original df
            var name = expr is AliasExpr ? expr.Name : col.Name;
            col = col.Clone(name);
            result = result.AddColumn(col);
        }
        return result;
    }
}
