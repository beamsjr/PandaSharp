using PandaSharp.Column;

namespace PandaSharp.Expressions;

/// <summary>
/// Conditional expression: when(condition).then(value).otherwise(default)
/// Like SQL CASE WHEN or Polars when/then/otherwise.
/// Usage: Expr.When(Col("Age") > 30).Then(Lit("Senior")).Otherwise(Lit("Junior"))
/// </summary>
public class WhenBuilder
{
    private readonly Expr _condition;

    internal WhenBuilder(Expr condition) => _condition = condition;

    public ThenBuilder Then(Expr value) => new(_condition, value);
}

public class ThenBuilder
{
    private readonly Expr _condition;
    private readonly Expr _thenValue;

    internal ThenBuilder(Expr condition, Expr thenValue)
    {
        _condition = condition;
        _thenValue = thenValue;
    }

    public Expr Otherwise(Expr elseValue) => new WhenThenExpr(_condition, _thenValue, elseValue);
}

internal class WhenThenExpr : Expr
{
    private readonly Expr _condition;
    private readonly Expr _thenValue;
    private readonly Expr _elseValue;

    public override string Name => "when_then";

    public WhenThenExpr(Expr condition, Expr thenValue, Expr elseValue)
    {
        _condition = condition;
        _thenValue = thenValue;
        _elseValue = elseValue;
    }

    public override IColumn Evaluate(DataFrame df)
    {
        var condCol = _condition.Evaluate(df);
        var thenCol = _thenValue.Evaluate(df);
        var elseCol = _elseValue.Evaluate(df);

        // Determine output type from then/else columns
        if (thenCol is StringColumn || elseCol is StringColumn)
            return EvaluateString(condCol, thenCol, elseCol);

        return EvaluateNumeric(condCol, thenCol, elseCol);
    }

    private static IColumn EvaluateString(IColumn cond, IColumn then, IColumn @else)
    {
        var result = new string?[cond.Length];
        for (int i = 0; i < cond.Length; i++)
        {
            bool? condition = cond.IsNull(i) ? null : cond.GetObject(i) is bool b ? b : null;
            result[i] = condition == true
                ? then.GetObject(i)?.ToString()
                : @else.GetObject(i)?.ToString();
        }
        return new StringColumn("when_then", result);
    }

    private static IColumn EvaluateNumeric(IColumn cond, IColumn then, IColumn @else)
    {
        var result = new double?[cond.Length];
        for (int i = 0; i < cond.Length; i++)
        {
            bool? condition = cond.IsNull(i) ? null : cond.GetObject(i) is bool b ? b : null;
            if (condition == true)
                result[i] = then.IsNull(i) ? null : Convert.ToDouble(then.GetObject(i));
            else
                result[i] = @else.IsNull(i) ? null : Convert.ToDouble(@else.GetObject(i));
        }
        return Column<double>.FromNullable("when_then", result);
    }
}

public partial class Expr
{
    /// <summary>
    /// Start a conditional expression: Expr.When(condition).Then(value).Otherwise(default)
    /// </summary>
    public static WhenBuilder When(Expr condition) => new(condition);
}
