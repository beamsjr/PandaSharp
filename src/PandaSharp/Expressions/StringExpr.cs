using PandaSharp.Column;

namespace PandaSharp.Expressions;

/// <summary>
/// String operations accessible via Col("name").Str in the expression system.
/// </summary>
public class StringExprAccessor
{
    private readonly Expr _source;

    internal StringExprAccessor(Expr source) => _source = source;

    public Expr Upper() => new StringTransformExpr(_source, "upper", s => s.ToUpperInvariant());
    public Expr Lower() => new StringTransformExpr(_source, "lower", s => s.ToLowerInvariant());
    public Expr Trim() => new StringTransformExpr(_source, "trim", s => s.Trim());
    public Expr Len() => new StringLenExpr(_source);

    public Expr Contains(string substring) => new StringPredicateExpr(_source, "contains", s => s.Contains(substring));
    public Expr StartsWith(string prefix) => new StringPredicateExpr(_source, "startswith", s => s.StartsWith(prefix));
    public Expr EndsWith(string suffix) => new StringPredicateExpr(_source, "endswith", s => s.EndsWith(suffix));

    public Expr Replace(string old, string @new) =>
        new StringTransformExpr(_source, "replace", s => s.Replace(old, @new));

    public Expr Slice(int start, int? length = null) =>
        new StringTransformExpr(_source, "slice", s =>
        {
            int actualStart = start < 0 ? Math.Max(0, s.Length + start) : Math.Min(start, s.Length);
            int actualLen = length ?? (s.Length - actualStart);
            actualLen = Math.Min(actualLen, s.Length - actualStart);
            return actualLen > 0 ? s.Substring(actualStart, actualLen) : "";
        });
}

internal class StringTransformExpr : Expr
{
    private readonly Expr _source;
    private readonly string _opName;
    private readonly Func<string, string> _transform;

    public override string Name => $"{_source.Name}.str.{_opName}";

    public StringTransformExpr(Expr source, string opName, Func<string, string> transform)
    {
        _source = source;
        _opName = opName;
        _transform = transform;
    }

    public override IColumn Evaluate(DataFrame df)
    {
        var col = _source.Evaluate(df);
        var result = new string?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            var val = col.GetObject(i);
            result[i] = val is string s ? _transform(s) : null;
        }
        return new StringColumn(Name, result);
    }
}

internal class StringPredicateExpr : Expr
{
    private readonly Expr _source;
    private readonly string _opName;
    private readonly Func<string, bool> _predicate;

    public override string Name => $"{_source.Name}.str.{_opName}";

    public StringPredicateExpr(Expr source, string opName, Func<string, bool> predicate)
    {
        _source = source;
        _opName = opName;
        _predicate = predicate;
    }

    public override IColumn Evaluate(DataFrame df)
    {
        var col = _source.Evaluate(df);
        var result = new bool?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            var val = col.GetObject(i);
            result[i] = val is string s ? _predicate(s) : null;
        }
        return Column<bool>.FromNullable(Name, result);
    }
}

internal class StringLenExpr : Expr
{
    private readonly Expr _source;
    public override string Name => $"{_source.Name}.str.len";

    public StringLenExpr(Expr source) => _source = source;

    public override IColumn Evaluate(DataFrame df)
    {
        var col = _source.Evaluate(df);
        var result = new int?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            var val = col.GetObject(i);
            result[i] = val is string s ? s.Length : null;
        }
        return Column<int>.FromNullable(Name, result);
    }
}
