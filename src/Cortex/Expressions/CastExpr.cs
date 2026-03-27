using Cortex.Column;

namespace Cortex.Expressions;

/// <summary>
/// Cast expression for type conversion: Col("Age").Cast&lt;double&gt;()
/// </summary>
public class CastExpr<TTarget> : Expr where TTarget : struct
{
    private readonly Expr _source;
    public override string Name => $"{_source.Name}.cast({typeof(TTarget).Name})";

    public CastExpr(Expr source) => _source = source;

    public override IColumn Evaluate(DataFrame df)
    {
        var col = _source.Evaluate(df);
        var result = new TTarget?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (col.IsNull(i)) { result[i] = null; continue; }
            try { result[i] = (TTarget)Convert.ChangeType(col.GetObject(i)!, typeof(TTarget)); }
            catch { result[i] = null; }
        }
        return Column<TTarget>.FromNullable(_source.Name, result);
    }
}

public static class CastExtensions
{
    /// <summary>
    /// Cast column values to a target type.
    /// Usage: Col("Price").Cast&lt;int&gt;()
    /// </summary>
    public static Expr Cast<TTarget>(this Expr expr) where TTarget : struct =>
        new CastExpr<TTarget>(expr);
}
