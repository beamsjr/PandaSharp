namespace Cortex.Viz.Themes;

/// <summary>Built-in chart themes applied via CSS class on the SVG container.</summary>
public static class VizTheme
{
    public const string Light = "theme-light";
    public const string Dark = "theme-dark";
    public const string Minimal = "theme-minimal";
    public const string Default = "theme-light";

    /// <summary>CSS for all themes, embedded in HTML output.</summary>
    internal static string GetCss() => """
    .theme-light { --chart-bg: #ffffff; --axis-color: #333; --text-color: #333; --grid-color: #e0e0e0; --tooltip-bg: rgba(0,0,0,0.8); --tooltip-fg: #fff; }
    .theme-dark { --chart-bg: #1a1a2e; --axis-color: #aaa; --text-color: #ddd; --grid-color: #333; --tooltip-bg: rgba(255,255,255,0.9); --tooltip-fg: #111; }
    .theme-minimal { --chart-bg: #fff; --axis-color: #666; --text-color: #333; --grid-color: transparent; --tooltip-bg: rgba(0,0,0,0.8); --tooltip-fg: #fff; }
    """;
}
