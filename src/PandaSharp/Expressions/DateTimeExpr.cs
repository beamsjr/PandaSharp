using PandaSharp.Column;

namespace PandaSharp.Expressions;

/// <summary>
/// DateTime operations accessible via Col("date").Dt in the expression system.
/// </summary>
public class DateTimeExprAccessor
{
    private readonly Expr _source;

    internal DateTimeExprAccessor(Expr source) => _source = source;

    public Expr Year() => new DateTimeComponentExpr(_source, "year", dt => dt.Year);
    public Expr Month() => new DateTimeComponentExpr(_source, "month", dt => dt.Month);
    public Expr Day() => new DateTimeComponentExpr(_source, "day", dt => dt.Day);
    public Expr Hour() => new DateTimeComponentExpr(_source, "hour", dt => dt.Hour);
    public Expr Minute() => new DateTimeComponentExpr(_source, "minute", dt => dt.Minute);
    public Expr Second() => new DateTimeComponentExpr(_source, "second", dt => dt.Second);
    public Expr DayOfWeek() => new DateTimeComponentExpr(_source, "dayofweek", dt => (int)dt.DayOfWeek);
    public Expr DayOfYear() => new DateTimeComponentExpr(_source, "dayofyear", dt => dt.DayOfYear);
    public Expr Quarter() => new DateTimeComponentExpr(_source, "quarter", dt => (dt.Month - 1) / 3 + 1);
}

internal class DateTimeComponentExpr : Expr
{
    private readonly Expr _source;
    private readonly string _component;
    private readonly Func<DateTime, int> _extract;

    public override string Name => $"{_source.Name}.dt.{_component}";

    public DateTimeComponentExpr(Expr source, string component, Func<DateTime, int> extract)
    {
        _source = source;
        _component = component;
        _extract = extract;
    }

    public override IColumn Evaluate(DataFrame df)
    {
        var col = _source.Evaluate(df);
        var result = new int?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            var val = col.GetObject(i);
            result[i] = val is DateTime dt ? _extract(dt) : null;
        }
        return Column<int>.FromNullable(Name, result);
    }
}
