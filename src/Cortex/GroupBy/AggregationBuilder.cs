namespace Cortex.GroupBy;

/// <summary>
/// Fluent builder for named aggregation in GroupBy.
/// Usage: .Agg(b => b.Sum("Salary").Mean("Age").Count("Name", alias: "HeadCount"))
/// </summary>
public class AggregationBuilder
{
    private readonly List<(string SourceColumn, string OutputName, AggFunc Func)> _specs = [];

    public AggregationBuilder Sum(string column, string? alias = null)
        => Add(column, alias, AggFunc.Sum);

    public AggregationBuilder Mean(string column, string? alias = null)
        => Add(column, alias, AggFunc.Mean);

    public AggregationBuilder Median(string column, string? alias = null)
        => Add(column, alias, AggFunc.Median);

    public AggregationBuilder Std(string column, string? alias = null)
        => Add(column, alias, AggFunc.Std);

    public AggregationBuilder Var(string column, string? alias = null)
        => Add(column, alias, AggFunc.Var);

    public AggregationBuilder Min(string column, string? alias = null)
        => Add(column, alias, AggFunc.Min);

    public AggregationBuilder Max(string column, string? alias = null)
        => Add(column, alias, AggFunc.Max);

    public AggregationBuilder Count(string column, string? alias = null)
        => Add(column, alias, AggFunc.Count);

    public AggregationBuilder First(string column, string? alias = null)
        => Add(column, alias, AggFunc.First);

    public AggregationBuilder Last(string column, string? alias = null)
        => Add(column, alias, AggFunc.Last);

    private AggregationBuilder Add(string column, string? alias, AggFunc func)
    {
        string outputName = alias ?? $"{column}_{func.ToString().ToLower()}";
        _specs.Add((column, outputName, func));
        return this;
    }

    internal List<(string SourceColumn, string OutputName, AggFunc Func)> Build() => _specs;
}
