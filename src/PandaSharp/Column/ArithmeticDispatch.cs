using System.Numerics;
using System.Reflection;

namespace PandaSharp.Column;

/// <summary>
/// Cached delegate dispatch for arithmetic operators on Column&lt;T&gt;.
/// Because Column&lt;T&gt; constrains T to struct (not INumber&lt;T&gt;),
/// operators cannot directly call the constrained extension methods.
/// This class resolves the correct generic method once per T and caches the delegate.
/// </summary>
internal static class ArithmeticDispatch<T> where T : struct
{
    private static readonly bool IsNumeric = typeof(T).GetInterfaces()
        .Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(INumber<>));

    internal static readonly Func<Column<T>, Column<T>, Column<T>> Add = Resolve("Add");
    internal static readonly Func<Column<T>, Column<T>, Column<T>> Subtract = Resolve("Subtract");
    internal static readonly Func<Column<T>, Column<T>, Column<T>> Multiply = Resolve("Multiply");
    internal static readonly Func<Column<T>, Column<T>, Column<T>> Divide = Resolve("Divide");

    internal static readonly Func<Column<T>, T, Column<T>> AddScalar = ResolveScalar("Add");
    internal static readonly Func<Column<T>, T, Column<T>> SubtractScalar = ResolveScalar("Subtract");
    internal static readonly Func<Column<T>, T, Column<T>> MultiplyScalar = ResolveScalar("Multiply");
    internal static readonly Func<Column<T>, T, Column<T>> DivideScalar = ResolveScalar("Divide");

    internal static readonly Func<Column<T>, Column<T>> Negate = ResolveUnary("Negate");

    private static Func<Column<T>, Column<T>, Column<T>> Resolve(string name)
    {
        if (!IsNumeric)
            return (_, _) => throw new NotSupportedException(
                $"Arithmetic operator '{name}' is not supported for Column<{typeof(T).Name}>. T must implement INumber<T>.");

        var method = typeof(ColumnArithmetic)
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .First(m => m.Name == name
                        && m.GetParameters().Length == 2
                        && m.GetParameters()[0].ParameterType.GetGenericTypeDefinition() == typeof(Column<>)
                        && m.GetParameters()[1].ParameterType.GetGenericTypeDefinition() == typeof(Column<>));
        var generic = method.MakeGenericMethod(typeof(T));
        return (a, b) =>
        {
            try { return (Column<T>)generic.Invoke(null, [a, b])!; }
            catch (TargetInvocationException ex) when (ex.InnerException is not null) { throw ex.InnerException; }
        };
    }

    private static Func<Column<T>, T, Column<T>> ResolveScalar(string name)
    {
        if (!IsNumeric)
            return (_, _) => throw new NotSupportedException(
                $"Arithmetic operator '{name}' is not supported for Column<{typeof(T).Name}>. T must implement INumber<T>.");

        var method = typeof(ColumnArithmetic)
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .First(m => m.Name == name
                        && m.GetParameters().Length == 2
                        && m.GetParameters()[0].ParameterType.GetGenericTypeDefinition() == typeof(Column<>)
                        && !m.GetParameters()[1].ParameterType.IsGenericType);
        var generic = method.MakeGenericMethod(typeof(T));
        return (col, scalar) =>
        {
            try { return (Column<T>)generic.Invoke(null, [col, scalar])!; }
            catch (TargetInvocationException ex) when (ex.InnerException is not null) { throw ex.InnerException; }
        };
    }

    private static Func<Column<T>, Column<T>> ResolveUnary(string name)
    {
        if (!IsNumeric)
            return _ => throw new NotSupportedException(
                $"Arithmetic operator '{name}' is not supported for Column<{typeof(T).Name}>. T must implement INumber<T>.");

        var method = typeof(ColumnArithmetic)
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .First(m => m.Name == name && m.GetParameters().Length == 1);
        var generic = method.MakeGenericMethod(typeof(T));
        return col =>
        {
            try { return (Column<T>)generic.Invoke(null, [col])!; }
            catch (TargetInvocationException ex) when (ex.InnerException is not null) { throw ex.InnerException; }
        };
    }
}
