using Cortex;

namespace Cortex.Viz;

/// <summary>
/// Extension methods to access the Viz builder from a DataFrame.
/// Usage: df.Viz.Bar("Category", "Sales").Title("Revenue").Show()
/// </summary>
public static class VizExtensions
{
    /// <summary>Access the interactive visualization builder.</summary>
    public static VizBuilder Viz(this DataFrame df) => new(df);

    /// <summary>Create a faceted grid of charts, one per unique value in the facet column.</summary>
    public static Charts.FacetGridBuilder FacetGrid(this VizBuilder viz, string facetColumn)
        => throw new NotSupportedException("Use df.Viz().FacetGrid() instead.");
}

public static class FacetGridExtensions
{
    /// <summary>Create a faceted grid of charts.</summary>
    public static Charts.FacetGridBuilder FacetGrid(this DataFrame df, string facetColumn)
        => new(df, facetColumn);
}
