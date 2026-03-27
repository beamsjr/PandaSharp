using System.Reflection;
using Cortex.Column;

namespace Cortex.Interop;

public static class LinqExtensions
{
    /// <summary>
    /// Create a DataFrame from a collection of objects using reflection.
    /// df = DataFrame.FromEnumerable(new[] { new { Name = "Alice", Age = 25 } })
    /// </summary>
    public static DataFrame FromEnumerable<T>(IEnumerable<T> items)
    {
        var list = items.ToList();
        if (list.Count == 0) return new DataFrame();

        var properties = typeof(T).GetProperties(BindingFlags.Public | BindingFlags.Instance);
        var columns = new List<IColumn>();

        foreach (var prop in properties)
        {
            var values = new object?[list.Count];
            for (int i = 0; i < list.Count; i++)
                values[i] = prop.GetValue(list[i]);

            columns.Add(BuildColumn(prop.Name, prop.PropertyType, values));
        }

        return new DataFrame(columns);
    }

    /// <summary>
    /// Project DataFrame rows to a typed collection.
    /// var people = df.AsEnumerable&lt;Person&gt;();
    /// </summary>
    public static IEnumerable<T> AsEnumerable<T>(this DataFrame df) where T : new()
    {
        var properties = typeof(T).GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.CanWrite)
            .ToArray();

        // Build compiled setters once — avoids reflection per row
        var setters = new List<(Column.IColumn Col, Action<T, object> Setter, Type TargetType)>();
        foreach (var prop in properties)
        {
            if (!df.ColumnNames.Contains(prop.Name)) continue;
            var col = df[prop.Name];
            var targetType = Nullable.GetUnderlyingType(prop.PropertyType) ?? prop.PropertyType;

            // Compile a fast setter via expression tree
            var instance = System.Linq.Expressions.Expression.Parameter(typeof(T), "obj");
            var value = System.Linq.Expressions.Expression.Parameter(typeof(object), "val");
            var converted = System.Linq.Expressions.Expression.Convert(value, prop.PropertyType);
            var assign = System.Linq.Expressions.Expression.Call(instance, prop.GetSetMethod()!, converted);
            var setter = System.Linq.Expressions.Expression.Lambda<Action<T, object>>(assign, instance, value).Compile();

            setters.Add((col, setter, targetType));
        }

        for (int r = 0; r < df.RowCount; r++)
        {
            var obj = new T();
            foreach (var (col, setter, targetType) in setters)
            {
                var val = col.GetObject(r);
                if (val is not null)
                    setter(obj, Convert.ChangeType(val, targetType));
            }
            yield return obj;
        }
    }

    private static IColumn BuildColumn(string name, Type propertyType, object?[] values)
    {
        var underlying = Nullable.GetUnderlyingType(propertyType) ?? propertyType;

        if (underlying == typeof(string))
            return new StringColumn(name, values.Select(v => v as string).ToArray());
        if (underlying == typeof(int)) return BuildTyped<int>(name, values);
        if (underlying == typeof(long)) return BuildTyped<long>(name, values);
        if (underlying == typeof(double)) return BuildTyped<double>(name, values);
        if (underlying == typeof(float)) return BuildTyped<float>(name, values);
        if (underlying == typeof(bool)) return BuildTyped<bool>(name, values);
        if (underlying == typeof(DateTime)) return BuildTyped<DateTime>(name, values);

        // Fallback: convert to string
        return new StringColumn(name, values.Select(v => v?.ToString()).ToArray());
    }

    private static Column<T> BuildTyped<T>(string name, object?[] values) where T : struct
    {
        var typed = new T?[values.Length];
        for (int i = 0; i < values.Length; i++)
            typed[i] = values[i] is null ? null : (T)Convert.ChangeType(values[i]!, typeof(T));
        return Column<T>.FromNullable(name, typed);
    }
}
