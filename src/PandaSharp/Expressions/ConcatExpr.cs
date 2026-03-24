using PandaSharp.Column;

namespace PandaSharp.Expressions;

/// <summary>
/// Concatenate string expressions: Expr.ConcatStr(Col("First"), Lit(" "), Col("Last"))
/// </summary>
internal class ConcatStrExpr : Expr
{
    private readonly Expr[] _parts;
    public override string Name => "concat_str";

    public ConcatStrExpr(Expr[] parts) => _parts = parts;

    public override IColumn Evaluate(DataFrame df)
    {
        var cols = _parts.Select(e => e.Evaluate(df)).ToArray();
        int length = cols[0].Length;
        var result = new string?[length];

        for (int i = 0; i < length; i++)
        {
            bool anyNull = false;
            var parts = new string[cols.Length];
            for (int c = 0; c < cols.Length; c++)
            {
                if (cols[c].IsNull(i)) { anyNull = true; break; }
                parts[c] = cols[c].GetObject(i)?.ToString() ?? "";
            }
            result[i] = anyNull ? null : string.Concat(parts);
        }

        return new StringColumn(Name, result);
    }
}

public partial class Expr
{
    /// <summary>
    /// Concatenate multiple expressions as strings.
    /// Usage: Expr.ConcatStr(Col("First"), Lit(" "), Col("Last"))
    /// </summary>
    public static Expr ConcatStr(params Expr[] parts) => new ConcatStrExpr(parts);
}
