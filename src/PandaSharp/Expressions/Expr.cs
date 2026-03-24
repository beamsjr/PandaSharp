using PandaSharp.Column;

namespace PandaSharp.Expressions;

/// <summary>
/// Base class for deferred column expressions. Compose operations without materializing intermediates.
/// Usage: Col("price") * Col("quantity"), Col("age") > Lit(30)
/// </summary>
public abstract partial class Expr
{
    /// <summary>
    /// Evaluate this expression against a DataFrame, returning a column.
    /// </summary>
    public abstract IColumn Evaluate(DataFrame df);

    /// <summary>
    /// The output column name for this expression.
    /// </summary>
    public abstract string Name { get; }

    // -- Factory methods --
    public static ColExpr Col(string name) => new(name);
    public static LitExpr<T> Lit<T>(T value) where T : struct => new(value);
    public static LitExpr<double> Lit(double value) => new(value);
    public static LitExpr<int> Lit(int value) => new(value);
    public static StringLitExpr Lit(string value) => new(value);

    // -- Arithmetic operators --
    public static Expr operator +(Expr left, Expr right) => new BinaryExpr(left, right, BinaryOp.Add);
    public static Expr operator -(Expr left, Expr right) => new BinaryExpr(left, right, BinaryOp.Subtract);
    public static Expr operator *(Expr left, Expr right) => new BinaryExpr(left, right, BinaryOp.Multiply);
    public static Expr operator /(Expr left, Expr right) => new BinaryExpr(left, right, BinaryOp.Divide);

    // -- Comparison operators --
    public static Expr operator >(Expr left, Expr right) => new ComparisonExpr(left, right, ComparisonOp.Gt);
    public static Expr operator <(Expr left, Expr right) => new ComparisonExpr(left, right, ComparisonOp.Lt);
    public static Expr operator >=(Expr left, Expr right) => new ComparisonExpr(left, right, ComparisonOp.Gte);
    public static Expr operator <=(Expr left, Expr right) => new ComparisonExpr(left, right, ComparisonOp.Lte);

    // -- Logical operators (for combining boolean expressions) --
    public static Expr operator &(Expr left, Expr right) => new LogicalExpr(left, right, LogicalOp.And);
    public static Expr operator |(Expr left, Expr right) => new LogicalExpr(left, right, LogicalOp.Or);
    public static Expr operator !(Expr operand) => new NotExpr(operand);

    // -- Equality (methods, since == / != can't return Expr) --
    public Expr Eq(Expr other) => new ComparisonExpr(this, other, ComparisonOp.Eq);
    public Expr Neq(Expr other) => new ComparisonExpr(this, other, ComparisonOp.Neq);

    // -- Accessor chains --
    public StringExprAccessor Str => new(this);
    public DateTimeExprAccessor Dt => new(this);

    // -- Implicit conversions for ergonomics --
    public static implicit operator Expr(int value) => Lit(value);
    public static implicit operator Expr(double value) => Lit(value);
    public static implicit operator Expr(string name) => Col(name);

    // -- Alias --
    public AliasExpr Alias(string name) => new(this, name);
}

/// <summary>
/// Reference to a column by name.
/// </summary>
public class ColExpr : Expr
{
    public override string Name { get; }

    public ColExpr(string name) => Name = name;

    public override IColumn Evaluate(DataFrame df) => df[Name];
}

/// <summary>
/// A scalar literal value broadcast to column length.
/// </summary>
public class LitExpr<T> : Expr where T : struct
{
    private readonly T _value;
    public override string Name => _value.ToString() ?? "literal";

    public LitExpr(T value) => _value = value;

    public override IColumn Evaluate(DataFrame df)
    {
        var values = new T[df.RowCount];
        Array.Fill(values, _value);
        return new Column<T>("literal", values);
    }
}

public class StringLitExpr : Expr
{
    private readonly string _value;
    public override string Name => _value;

    public StringLitExpr(string value) => _value = value;

    public override IColumn Evaluate(DataFrame df)
    {
        var values = new string[df.RowCount];
        Array.Fill(values, _value);
        return new StringColumn("literal", values);
    }
}

/// <summary>
/// Renames the output of an expression.
/// </summary>
public class AliasExpr : Expr
{
    private readonly Expr _inner;
    public Expr Inner => _inner;
    public override string Name { get; }

    public AliasExpr(Expr inner, string name) { _inner = inner; Name = name; }

    public override IColumn Evaluate(DataFrame df) => _inner.Evaluate(df).Clone(Name);
}

// -- Binary arithmetic --

public enum BinaryOp { Add, Subtract, Multiply, Divide }

public class BinaryExpr : Expr
{
    private readonly Expr _left;
    private readonly Expr _right;
    private readonly BinaryOp _op;

    public Expr Left => _left;
    public Expr Right => _right;
    public BinaryOp Op => _op;

    public override string Name => $"({_left.Name} {OpSymbol} {_right.Name})";
    private string OpSymbol => _op switch
    {
        BinaryOp.Add => "+", BinaryOp.Subtract => "-",
        BinaryOp.Multiply => "*", BinaryOp.Divide => "/", _ => "?"
    };

    public BinaryExpr(Expr left, Expr right, BinaryOp op) { _left = left; _right = right; _op = op; }

    public override IColumn Evaluate(DataFrame df)
    {
        var leftCol = _left.Evaluate(df);
        var rightCol = _right.Evaluate(df);

        var leftVals = ToDoubleArray(leftCol);
        var rightVals = ToDoubleArray(rightCol);
        var result = new double?[leftVals.Length];

        for (int i = 0; i < result.Length; i++)
        {
            if (leftVals[i] is null || rightVals[i] is null) { result[i] = null; continue; }
            result[i] = _op switch
            {
                BinaryOp.Add => leftVals[i]!.Value + rightVals[i]!.Value,
                BinaryOp.Subtract => leftVals[i]!.Value - rightVals[i]!.Value,
                BinaryOp.Multiply => leftVals[i]!.Value * rightVals[i]!.Value,
                BinaryOp.Divide => rightVals[i]!.Value == 0 ? null : leftVals[i]!.Value / rightVals[i]!.Value,
                _ => null
            };
        }

        return Column<double>.FromNullable(Name, result);
    }

    private static double?[] ToDoubleArray(IColumn col)
    {
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.IsNull(i) ? null : Convert.ToDouble(col.GetObject(i));
        return result;
    }
}

// -- Comparison --

public enum ComparisonOp { Gt, Lt, Gte, Lte, Eq, Neq }

public class ComparisonExpr : Expr
{
    private readonly Expr _left;
    private readonly Expr _right;
    private readonly ComparisonOp _op;

    public Expr Left => _left;
    public Expr Right => _right;
    public ComparisonOp Op => _op;

    public override string Name => $"({_left.Name} {OpSymbol} {_right.Name})";
    private string OpSymbol => _op switch
    {
        ComparisonOp.Gt => ">", ComparisonOp.Lt => "<", ComparisonOp.Gte => ">=",
        ComparisonOp.Lte => "<=", ComparisonOp.Eq => "==", ComparisonOp.Neq => "!=", _ => "?"
    };

    public ComparisonExpr(Expr left, Expr right, ComparisonOp op) { _left = left; _right = right; _op = op; }

    public override IColumn Evaluate(DataFrame df)
    {
        var leftCol = _left.Evaluate(df);
        var rightCol = _right.Evaluate(df);
        var result = new bool?[leftCol.Length];

        for (int i = 0; i < result.Length; i++)
        {
            var lv = leftCol.GetObject(i);
            var rv = rightCol.GetObject(i);
            if (lv is null || rv is null) { result[i] = null; continue; }

            // Convert to double for cross-type numeric comparison
            int cmp;
            if (lv is IConvertible && rv is IConvertible && IsNumeric(lv) && IsNumeric(rv))
                cmp = Convert.ToDouble(lv).CompareTo(Convert.ToDouble(rv));
            else
                cmp = Comparer<object>.Default.Compare(lv, rv);
            result[i] = _op switch
            {
                ComparisonOp.Gt => cmp > 0,
                ComparisonOp.Lt => cmp < 0,
                ComparisonOp.Gte => cmp >= 0,
                ComparisonOp.Lte => cmp <= 0,
                ComparisonOp.Eq => cmp == 0,
                ComparisonOp.Neq => cmp != 0,
                _ => null
            };
        }

        return Column<bool>.FromNullable(Name, result);
    }

    private static bool IsNumeric(object val) =>
        val is int or long or float or double or decimal or short or byte;
}

// -- Logical --

public enum LogicalOp { And, Or }

public class LogicalExpr : Expr
{
    private readonly Expr _left;
    private readonly Expr _right;
    private readonly LogicalOp _op;

    public Expr Left => _left;
    public Expr Right => _right;
    public LogicalOp Op => _op;

    public override string Name => $"({_left.Name} {(_op == LogicalOp.And ? "&" : "|")} {_right.Name})";

    public LogicalExpr(Expr left, Expr right, LogicalOp op) { _left = left; _right = right; _op = op; }

    public override IColumn Evaluate(DataFrame df)
    {
        var leftCol = _left.Evaluate(df);
        var rightCol = _right.Evaluate(df);
        var result = new bool?[leftCol.Length];

        for (int i = 0; i < result.Length; i++)
        {
            var lv = leftCol.IsNull(i) ? null : leftCol.GetObject(i);
            var rv = rightCol.IsNull(i) ? null : rightCol.GetObject(i);

            bool? lb = lv is bool b1 ? b1 : null;
            bool? rb = rv is bool b2 ? b2 : null;

            result[i] = _op switch
            {
                LogicalOp.And => lb.HasValue && rb.HasValue ? lb.Value && rb.Value : null,
                LogicalOp.Or => lb.HasValue && rb.HasValue ? lb.Value || rb.Value : null,
                _ => null
            };
        }

        return Column<bool>.FromNullable(Name, result);
    }
}

public class NotExpr : Expr
{
    private readonly Expr _inner;
    public Expr Inner => _inner;
    public override string Name => $"(!{_inner.Name})";

    public NotExpr(Expr inner) => _inner = inner;

    public override IColumn Evaluate(DataFrame df)
    {
        var col = _inner.Evaluate(df);
        var result = new bool?[col.Length];
        for (int i = 0; i < result.Length; i++)
        {
            if (col.IsNull(i)) result[i] = null;
            else result[i] = col.GetObject(i) is bool b ? !b : null;
        }
        return Column<bool>.FromNullable(Name, result);
    }
}
