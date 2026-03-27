using Cortex;
using Cortex.Column;
using Cortex.Viz;

// ============================================================
// Cortex.Viz — Interactive Visualization Examples
// ============================================================
// Generates self-contained HTML files with Plotly.js charts.
// Open the output files in any browser to view interactive charts.
// ============================================================

Console.WriteLine("Cortex.Viz Examples");
Console.WriteLine("=======================\n");

// --- Sample Data ---
var rng = new Random(42);
int n = 50;
var months = new string?[n];
var revenue = new double[n];
var costs = new double[n];
var regions = new string?[n];
var regionNames = new[] { "North", "South", "East", "West" };
for (int i = 0; i < n; i++)
{
    months[i] = $"2024-{(i % 12) + 1:D2}";
    revenue[i] = 50_000 + rng.NextDouble() * 100_000;
    costs[i] = 30_000 + rng.NextDouble() * 50_000;
    regions[i] = regionNames[rng.Next(4)];
}

var monthNums = new double[n];
for (int i = 0; i < n; i++) monthNums[i] = i + 1;

var df = new DataFrame(
    new StringColumn("Month", months),
    new Column<double>("MonthNum", monthNums),
    new Column<double>("Revenue", revenue),
    new Column<double>("Cost", costs),
    new StringColumn("Region", regions)
);

var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "viz_output");
Directory.CreateDirectory(outputDir);

// --- 1. Bar Chart ---
Console.WriteLine("1. Creating bar chart...");
df.Head(12).Viz()
    .Bar("Month", "Revenue")
    .Title("Monthly Revenue")
    .ToHtml(Path.Combine(outputDir, "bar_chart.html"));
Console.WriteLine("   -> viz_output/bar_chart.html");

// --- 2. Line Chart ---
Console.WriteLine("2. Creating line chart...");
df.Head(12).Viz()
    .Line("MonthNum", "Revenue")
    .Title("Revenue Trend")
    .ToHtml(Path.Combine(outputDir, "line_chart.html"));
Console.WriteLine("   -> viz_output/line_chart.html");

// --- 3. Scatter Plot ---
Console.WriteLine("3. Creating scatter plot...");
df.Viz()
    .Scatter("Revenue", "Cost")
    .Title("Revenue vs Cost")
    .ToHtml(Path.Combine(outputDir, "scatter.html"));
Console.WriteLine("   -> viz_output/scatter.html");

// --- 4. Histogram ---
Console.WriteLine("4. Creating histogram...");
df.Viz()
    .Histogram("Revenue")
    .Title("Revenue Distribution")
    .ToHtml(Path.Combine(outputDir, "histogram.html"));
Console.WriteLine("   -> viz_output/histogram.html");

// --- 5. Area Chart ---
Console.WriteLine("5. Creating area chart...");
df.Head(12).Viz()
    .Area("MonthNum", "Revenue")
    .Title("Revenue Area")
    .ToHtml(Path.Combine(outputDir, "area_chart.html"));
Console.WriteLine("   -> viz_output/area_chart.html");

// --- 6. Themed Chart (Dark Mode) ---
Console.WriteLine("6. Creating dark-themed bar chart...");
df.Head(12).Viz()
    .Bar("Month", "Revenue")
    .Title("Revenue (Dark Theme)")
    .Theme(Cortex.Viz.Themes.VizTheme.Dark)
    .ToHtml(Path.Combine(outputDir, "dark_theme.html"));
Console.WriteLine("   -> viz_output/dark_theme.html");

// --- 7. HTML Fragment (for embedding) ---
Console.WriteLine("7. Creating embeddable HTML fragment...");
var fragment = df.Head(12).Viz()
    .Line("MonthNum", "Revenue")
    .Title("Embeddable Chart")
    .ToHtmlFragment("my-chart-div");
File.WriteAllText(Path.Combine(outputDir, "fragment.html"),
    $"<html><head><script src='https://cdn.jsdelivr.net/npm/d3@7'></script></head><body>{fragment}</body></html>");
Console.WriteLine("   -> viz_output/fragment.html");

Console.WriteLine($"\nAll charts saved to: {outputDir}");
Console.WriteLine("Open any .html file in your browser to view interactive charts.");
