using System.Text;
using System.Text.Json;
using Cortex.Viz.Charts;

namespace Cortex.Viz.Rendering;

/// <summary>
/// Generates self-contained D3.js JavaScript from a ChartSpec.
/// Produces SVG charts with scales, axes, tooltips, legend, and responsive resize.
/// </summary>
internal static class D3Renderer
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    private static string Json<T>(T value) => JsonSerializer.Serialize(value, JsonOpts);

    private static readonly string[] Palette =
    [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"
    ];

    /// <summary>Generate the D3.js script body for a given chart spec and target div.</summary>
    public static string Render(ChartSpec spec, string divId)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"(function() {{");
        sb.AppendLine($"  const divId = '{divId}';");
        sb.AppendLine($"  const container = document.getElementById(divId);");
        sb.AppendLine($"  const W = {spec.Layout.Width}, H = {spec.Layout.Height};");

        // Determine chart type from first trace
        var primaryType = spec.Traces.Count > 0 ? spec.Traces[0].Type : "scatter";
        bool isHorizontalBar = primaryType == "bar" && spec.Traces.Any(t => t.Orientation == "h");
        bool isPie = primaryType == "pie";
        bool isHeatmap = primaryType == "heatmap";
        bool isHistogram = primaryType == "histogram";
        bool isBox = primaryType == "box";

        bool isCorrMatrix = primaryType == "corrmatrix";
        bool isParallel = primaryType == "parallel";
        bool isTreemap = primaryType == "treemap";
        bool isNetwork = primaryType == "network";

        if (isPie)
            RenderPie(sb, spec);
        else if (isHeatmap)
            RenderHeatmap(sb, spec);
        else if (isHistogram)
            RenderHistogram(sb, spec);
        else if (isBox)
            RenderBox(sb, spec);
        else if (isCorrMatrix)
            RenderCorrMatrix(sb, spec);
        else if (isParallel)
            RenderParallelCoordinates(sb, spec);
        else if (isTreemap)
            RenderTreemap(sb, spec);
        else if (isNetwork)
            RenderNetwork(sb, spec);
        else
            RenderCartesian(sb, spec, primaryType, isHorizontalBar);

        sb.AppendLine($"}})();");
        return sb.ToString();
    }

    // ═══════════════════════════════════════════════════════════
    // Cartesian charts: bar, scatter, line, area
    // ═══════════════════════════════════════════════════════════
    private static void RenderCartesian(StringBuilder sb, ChartSpec spec, string primaryType, bool isHorizontal)
    {
        var layout = spec.Layout;
        bool isBandX = !isHorizontal && spec.Traces.Any(t => t.XLabels is not null && t.Type == "bar");
        bool isBandY = isHorizontal && spec.Traces.Any(t => t.YLabels is not null);
        bool isBar = primaryType == "bar";
        bool hasArea = spec.Traces.Any(t => t.Extra.ContainsKey("fill"));

        // Margins — widen right for legend
        bool hasLegend = layout.ShowLegend && spec.Traces.Count > 1;
        int rightMargin = hasLegend ? 120 : 30;
        sb.AppendLine($"  const margin = {{top: 40, right: {rightMargin}, bottom: 50, left: 60}};");
        sb.AppendLine("  const width = W - margin.left - margin.right;");
        sb.AppendLine("  const height = H - margin.top - margin.bottom;");

        // SVG
        sb.AppendLine("  const svg = d3.select(container).append('svg')");
        sb.AppendLine("    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('max-width', W + 'px')");
        sb.AppendLine("    .style('font-family', 'system-ui, -apple-system, sans-serif')");
        sb.AppendLine("    .style('font-size', '12px');");
        // Clip path for zoom/pan
        sb.AppendLine($"  const clipId = 'clip_' + divId;");
        sb.AppendLine("  svg.append('defs').append('clipPath').attr('id', clipId)");
        sb.AppendLine("    .append('rect').attr('width', width).attr('height', height);");
        sb.AppendLine("  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);");
        sb.AppendLine("  const plotArea = g.append('g').attr('clip-path', `url(#${clipId})`);");

        // Serialize data
        EmitTraceData(sb, spec);

        // Build scales
        if (isBandX)
            BuildBandScaleX(sb, spec);
        else if (isBandY)
            BuildBandScaleY(sb, spec);

        if (!isBandX)
            BuildLinearScaleX(sb, spec, isHorizontal);
        if (!isBandY)
            BuildLinearScaleY(sb, spec, isHorizontal, isBandX);

        // Grouped bar sub-scale
        if (isBar && spec.Traces.Count > 1 && layout.Barmode != "stack")
        {
            sb.AppendLine("  const traceNames = traces.map(t => t.name);");
            if (!isHorizontal)
                sb.AppendLine("  const x1 = d3.scaleBand().domain(traceNames).range([0, x.bandwidth()]).padding(0.05);");
            else
                sb.AppendLine("  const y1 = d3.scaleBand().domain(traceNames).range([0, y.bandwidth()]).padding(0.05);");
        }

        // Color scale
        sb.AppendLine($"  const color = d3.scaleOrdinal().range({Json(Palette)});");

        // Axes
        EmitAxes(sb, layout, isBandX || !isHorizontal, isBandY || isHorizontal);

        // Tooltip
        EmitTooltip(sb);

        // Draw traces
        sb.AppendLine("  traces.forEach((trace, ti) => {");
        sb.AppendLine("    const c = color(trace.name || `trace_${ti}`);");

        if (isBar && !isHorizontal)
            EmitVerticalBars(sb, spec.Traces.Count > 1 && layout.Barmode != "stack");
        else if (isBar && isHorizontal)
            EmitHorizontalBars(sb, spec.Traces.Count > 1 && layout.Barmode != "stack");
        else if (hasArea)
            EmitArea(sb);
        else
            EmitLinesAndMarkers(sb);

        sb.AppendLine("  });");

        // Zoom/pan for continuous-axis charts (not bar charts)
        if (!isBar)
            EmitZoomPan(sb);

        // Title
        EmitTitle(sb, layout);

        // Clickable legend with show/hide
        if (hasLegend)
            EmitClickableLegend(sb);

        // Animation (if frames exist)
        if (spec.Frames.Count > 0)
            EmitAnimation(sb, spec, isBar, isHorizontal);
    }

    // ═══════════════════════════════════════════════════════════
    // Histogram
    // ═══════════════════════════════════════════════════════════
    private static void RenderHistogram(StringBuilder sb, ChartSpec spec)
    {
        var trace = spec.Traces[0];
        var layout = spec.Layout;
        bool isDensity = trace.Extra.ContainsKey("histnorm");

        sb.AppendLine("  const margin = {top: 40, right: 30, bottom: 50, left: 60};");
        sb.AppendLine("  const width = W - margin.left - margin.right;");
        sb.AppendLine("  const height = H - margin.top - margin.bottom;");
        sb.AppendLine("  const svg = d3.select(container).append('svg')");
        sb.AppendLine("    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('max-width', W + 'px')");
        sb.AppendLine("    .style('font-family', 'system-ui, -apple-system, sans-serif')");
        sb.AppendLine("    .style('font-size', '12px');");
        sb.AppendLine("  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);");

        sb.AppendLine($"  const rawData = {Json(trace.X)};");
        sb.AppendLine("  const data = rawData.filter(d => isFinite(d));");

        // X scale
        sb.AppendLine("  const x = d3.scaleLinear().domain(d3.extent(data)).nice().range([0, width]);");

        // Bin
        int bins = trace.NBinsX ?? 30;
        sb.AppendLine($"  const bins = d3.bin().domain(x.domain()).thresholds({bins})(data);");

        // Y scale
        if (isDensity)
        {
            sb.AppendLine("  const totalN = data.length;");
            sb.AppendLine("  bins.forEach(b => { b.density = b.length / totalN / (b.x1 - b.x0); });");
            sb.AppendLine("  const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.density)]).nice().range([height, 0]);");
        }
        else
        {
            sb.AppendLine("  const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.length)]).nice().range([height, 0]);");
        }

        // Axes
        EmitAxes(sb, layout, true, true);

        // Tooltip
        EmitTooltip(sb);

        // Bars
        string yAccessor = isDensity ? "d.density" : "d.length";
        sb.AppendLine("  g.selectAll('rect').data(bins).join('rect')");
        sb.AppendLine($"    .attr('x', d => x(d.x0) + 1)");
        sb.AppendLine($"    .attr('y', d => y({yAccessor}))");
        sb.AppendLine($"    .attr('width', d => Math.max(0, x(d.x1) - x(d.x0) - 1))");
        sb.AppendLine($"    .attr('height', d => height - y({yAccessor}))");
        sb.AppendLine($"    .attr('fill', '{Palette[0]}')");
        sb.AppendLine("    .attr('opacity', 0.7)");
        sb.AppendLine("    .on('mouseover', function(event, d) {");
        sb.AppendLine($"      tooltip.style('opacity', 1).html(`${{d.x0.toFixed(1)}} - ${{d.x1.toFixed(1)}}<br>Count: ${{d.length}}`)");
        sb.AppendLine("        .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');");
        sb.AppendLine("    })");
        sb.AppendLine("    .on('mouseout', function() { tooltip.style('opacity', 0); });");

        EmitTitle(sb, layout);
    }

    // ═══════════════════════════════════════════════════════════
    // Box plot
    // ═══════════════════════════════════════════════════════════
    private static void RenderBox(StringBuilder sb, ChartSpec spec)
    {
        var layout = spec.Layout;
        sb.AppendLine("  const margin = {top: 40, right: 30, bottom: 50, left: 60};");
        sb.AppendLine("  const width = W - margin.left - margin.right;");
        sb.AppendLine("  const height = H - margin.top - margin.bottom;");
        sb.AppendLine("  const svg = d3.select(container).append('svg')");
        sb.AppendLine("    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('max-width', W + 'px')");
        sb.AppendLine("    .style('font-family', 'system-ui, -apple-system, sans-serif')");
        sb.AppendLine("    .style('font-size', '12px');");
        sb.AppendLine("  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);");

        // Build box data from traces
        sb.AppendLine("  const boxData = [];");
        for (int i = 0; i < spec.Traces.Count; i++)
        {
            var t = spec.Traces[i];
            var name = t.Name ?? $"trace_{i}";
            var values = t.Y ?? t.X ?? [];
            sb.AppendLine($"  {{ const vals = {Json(values)}.filter(v => isFinite(v)).sort(d3.ascending);");
            sb.AppendLine($"    const q1 = d3.quantile(vals, 0.25), q2 = d3.quantile(vals, 0.5), q3 = d3.quantile(vals, 0.75);");
            sb.AppendLine($"    const iqr = q3 - q1, lo = Math.max(vals[0], q1 - 1.5*iqr), hi = Math.min(vals[vals.length-1], q3 + 1.5*iqr);");
            sb.AppendLine($"    const outliers = vals.filter(v => v < lo || v > hi);");
            sb.AppendLine($"    boxData.push({{name: {Json(name)}, q1, q2, q3, lo, hi, outliers}}); }}");
        }

        sb.AppendLine("  const x = d3.scaleBand().domain(boxData.map(d => d.name)).range([0, width]).padding(0.3);");
        sb.AppendLine("  const allVals = [].concat(...boxData.map(d => [d.lo, d.hi, ...d.outliers]));");
        sb.AppendLine("  const y = d3.scaleLinear().domain(d3.extent(allVals)).nice().range([height, 0]);");
        sb.AppendLine($"  const color = d3.scaleOrdinal().range({Json(Palette)});");

        EmitAxes(sb, layout, true, true);

        // Draw boxes
        sb.AppendLine("  boxData.forEach((d, i) => {");
        sb.AppendLine("    const cx = x(d.name) + x.bandwidth() / 2;");
        sb.AppendLine("    const bw = x.bandwidth() * 0.6;");
        sb.AppendLine("    const c = color(d.name);");
        // Box
        sb.AppendLine("    g.append('rect').attr('x', cx - bw/2).attr('y', y(d.q3))");
        sb.AppendLine("      .attr('width', bw).attr('height', y(d.q1) - y(d.q3))");
        sb.AppendLine("      .attr('fill', c).attr('opacity', 0.3).attr('stroke', c).attr('stroke-width', 1.5);");
        // Median
        sb.AppendLine("    g.append('line').attr('x1', cx - bw/2).attr('x2', cx + bw/2)");
        sb.AppendLine("      .attr('y1', y(d.q2)).attr('y2', y(d.q2)).attr('stroke', c).attr('stroke-width', 2);");
        // Whiskers
        sb.AppendLine("    g.append('line').attr('x1', cx).attr('x2', cx)");
        sb.AppendLine("      .attr('y1', y(d.lo)).attr('y2', y(d.q1)).attr('stroke', c).attr('stroke-dasharray', '4,2');");
        sb.AppendLine("    g.append('line').attr('x1', cx).attr('x2', cx)");
        sb.AppendLine("      .attr('y1', y(d.q3)).attr('y2', y(d.hi)).attr('stroke', c).attr('stroke-dasharray', '4,2');");
        // Whisker caps
        sb.AppendLine("    g.append('line').attr('x1', cx - bw/4).attr('x2', cx + bw/4)");
        sb.AppendLine("      .attr('y1', y(d.lo)).attr('y2', y(d.lo)).attr('stroke', c);");
        sb.AppendLine("    g.append('line').attr('x1', cx - bw/4).attr('x2', cx + bw/4)");
        sb.AppendLine("      .attr('y1', y(d.hi)).attr('y2', y(d.hi)).attr('stroke', c);");
        // Outliers
        sb.AppendLine("    d.outliers.forEach(v => {");
        sb.AppendLine("      g.append('circle').attr('cx', cx).attr('cy', y(v)).attr('r', 3)");
        sb.AppendLine("        .attr('fill', 'none').attr('stroke', c).attr('opacity', 0.6);");
        sb.AppendLine("    });");
        sb.AppendLine("  });");

        EmitTitle(sb, layout);
    }

    // ═══════════════════════════════════════════════════════════
    // Pie chart
    // ═══════════════════════════════════════════════════════════
    private static void RenderPie(StringBuilder sb, ChartSpec spec)
    {
        var trace = spec.Traces[0];
        var layout = spec.Layout;
        var labels = trace.Extra.TryGetValue("labels", out var l) ? l as string[] : [];
        var values = trace.Extra.TryGetValue("values", out var v) ? v as double[] : [];

        sb.AppendLine("  const svg = d3.select(container).append('svg')");
        sb.AppendLine("    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('max-width', W + 'px')");
        sb.AppendLine("    .style('font-family', 'system-ui, -apple-system, sans-serif')");
        sb.AppendLine("    .style('font-size', '12px');");
        sb.AppendLine("  const radius = Math.min(W, H) / 2 - 40;");
        sb.AppendLine("  const g = svg.append('g').attr('transform', `translate(${W/2},${H/2})`);");

        sb.AppendLine($"  const labels = {Json(labels)};");
        sb.AppendLine($"  const values = {Json(values)};");
        sb.AppendLine("  const data = labels.map((l, i) => ({label: l, value: values[i]}));");
        sb.AppendLine($"  const color = d3.scaleOrdinal().range({Json(Palette)});");
        sb.AppendLine("  const pie = d3.pie().value(d => d.value);");
        sb.AppendLine("  const arc = d3.arc().innerRadius(0).outerRadius(radius);");
        sb.AppendLine("  const labelArc = d3.arc().innerRadius(radius * 0.6).outerRadius(radius * 0.6);");

        EmitTooltip(sb);

        sb.AppendLine("  g.selectAll('path').data(pie(data)).join('path')");
        sb.AppendLine("    .attr('d', arc).attr('fill', d => color(d.data.label))");
        sb.AppendLine("    .attr('stroke', 'white').attr('stroke-width', 2)");
        sb.AppendLine("    .on('mouseover', function(event, d) {");
        sb.AppendLine("      tooltip.style('opacity', 1).html(`${d.data.label}<br>${d.data.value.toFixed(1)}`)");
        sb.AppendLine("        .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');");
        sb.AppendLine("    })");
        sb.AppendLine("    .on('mouseout', function() { tooltip.style('opacity', 0); });");

        // Labels
        sb.AppendLine("  g.selectAll('text').data(pie(data)).join('text')");
        sb.AppendLine("    .attr('transform', d => `translate(${labelArc.centroid(d)})`)");
        sb.AppendLine("    .attr('text-anchor', 'middle').attr('font-size', '11px')");
        sb.AppendLine("    .text(d => d.data.label);");

        EmitTitle(sb, layout);
    }

    // ═══════════════════════════════════════════════════════════
    // Heatmap
    // ═══════════════════════════════════════════════════════════
    private static void RenderHeatmap(StringBuilder sb, ChartSpec spec)
    {
        var trace = spec.Traces[0];
        var layout = spec.Layout;

        sb.AppendLine("  const margin = {top: 40, right: 80, bottom: 50, left: 60};");
        sb.AppendLine("  const width = W - margin.left - margin.right;");
        sb.AppendLine("  const height = H - margin.top - margin.bottom;");
        sb.AppendLine("  const svg = d3.select(container).append('svg')");
        sb.AppendLine("    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('max-width', W + 'px')");
        sb.AppendLine("    .style('font-family', 'system-ui, -apple-system, sans-serif')");
        sb.AppendLine("    .style('font-size', '12px');");
        sb.AppendLine("  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);");

        sb.AppendLine($"  const zFlat = {Json(trace.Z)};");
        sb.AppendLine($"  const nRows = {trace.ZRows}, nCols = {trace.ZCols};");
        sb.AppendLine($"  const xLabels = {Json(trace.XLabels ?? [])};");
        sb.AppendLine($"  const yLabels = {Json(trace.YLabels ?? [])};");

        sb.AppendLine("  const x = d3.scaleBand().domain(xLabels.length ? xLabels : d3.range(nCols)).range([0, width]).padding(0.02);");
        sb.AppendLine("  const y = d3.scaleBand().domain(yLabels.length ? yLabels : d3.range(nRows)).range([0, height]).padding(0.02);");
        sb.AppendLine("  const zMin = d3.min(zFlat.filter(isFinite)), zMax = d3.max(zFlat.filter(isFinite));");
        sb.AppendLine("  const colorScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([zMin, zMax]);");

        EmitTooltip(sb);

        sb.AppendLine("  const cells = [];");
        sb.AppendLine("  for (let r = 0; r < nRows; r++) for (let c = 0; c < nCols; c++)");
        sb.AppendLine("    cells.push({r, c, v: zFlat[r * nCols + c]});");
        sb.AppendLine("  g.selectAll('rect').data(cells).join('rect')");
        sb.AppendLine("    .attr('x', d => x(xLabels.length ? xLabels[d.c] : d.c))");
        sb.AppendLine("    .attr('y', d => y(yLabels.length ? yLabels[d.r] : d.r))");
        sb.AppendLine("    .attr('width', x.bandwidth()).attr('height', y.bandwidth())");
        sb.AppendLine("    .attr('fill', d => isFinite(d.v) ? colorScale(d.v) : '#eee')");
        sb.AppendLine("    .on('mouseover', function(event, d) {");
        sb.AppendLine("      tooltip.style('opacity', 1).html(`Row: ${d.r}, Col: ${d.c}<br>Value: ${d.v?.toFixed(2)}`)");
        sb.AppendLine("        .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');");
        sb.AppendLine("    })");
        sb.AppendLine("    .on('mouseout', function() { tooltip.style('opacity', 0); });");

        // Axes
        sb.AppendLine("  g.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x)).selectAll('text').attr('font-size','10px');");
        sb.AppendLine("  g.append('g').call(d3.axisLeft(y)).selectAll('text').attr('font-size','10px');");

        EmitTitle(sb, layout);
    }

    // ═══════════════════════════════════════════════════════════
    // Correlation matrix (diverging heatmap with value labels)
    // ═══════════════════════════════════════════════════════════
    private static void RenderCorrMatrix(StringBuilder sb, ChartSpec spec)
    {
        var trace = spec.Traces[0];
        var layout = spec.Layout;
        sb.AppendLine("  const margin = {top: 40, right: 40, bottom: 80, left: 80};");
        sb.AppendLine("  const width = W - margin.left - margin.right;");
        sb.AppendLine("  const height = H - margin.top - margin.bottom;");
        sb.AppendLine("  const svg = d3.select(container).append('svg')");
        sb.AppendLine("    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('max-width', W + 'px')");
        sb.AppendLine("    .style('font-family', 'system-ui, -apple-system, sans-serif')");
        sb.AppendLine("    .style('font-size', '12px');");
        sb.AppendLine("  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);");

        sb.AppendLine($"  const labels = {Json(trace.XLabels)};");
        sb.AppendLine($"  const zFlat = {Json(trace.Z)};");
        sb.AppendLine($"  const n = {trace.ZRows};");

        sb.AppendLine("  const x = d3.scaleBand().domain(labels).range([0, width]).padding(0.05);");
        sb.AppendLine("  const y = d3.scaleBand().domain(labels).range([0, height]).padding(0.05);");
        // Diverging color: blue (-1) → white (0) → red (+1)
        sb.AppendLine("  const colorScale = d3.scaleDiverging(d3.interpolateRdBu).domain([1, 0, -1]);");

        EmitTooltip(sb);

        sb.AppendLine("  const cells = [];");
        sb.AppendLine("  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++)");
        sb.AppendLine("    cells.push({r, c, v: zFlat[r * n + c], rl: labels[r], cl: labels[c]});");

        sb.AppendLine("  g.selectAll('rect').data(cells).join('rect')");
        sb.AppendLine("    .attr('x', d => x(d.cl)).attr('y', d => y(d.rl))");
        sb.AppendLine("    .attr('width', x.bandwidth()).attr('height', y.bandwidth())");
        sb.AppendLine("    .attr('fill', d => colorScale(d.v)).attr('rx', 2)");
        sb.AppendLine("    .on('mouseover', function(event, d) {");
        sb.AppendLine("      tooltip.style('opacity', 1).html(`${d.rl} × ${d.cl}<br>r = ${d.v.toFixed(3)}`)");
        sb.AppendLine("        .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');");
        sb.AppendLine("    }).on('mouseout', () => tooltip.style('opacity', 0));");

        // Value labels on each cell
        sb.AppendLine("  g.selectAll('.corr-label').data(cells).join('text').attr('class', 'corr-label')");
        sb.AppendLine("    .attr('x', d => x(d.cl) + x.bandwidth() / 2)");
        sb.AppendLine("    .attr('y', d => y(d.rl) + y.bandwidth() / 2)");
        sb.AppendLine("    .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')");
        sb.AppendLine("    .attr('font-size', Math.min(10, x.bandwidth() / 4) + 'px')");
        sb.AppendLine("    .attr('fill', d => Math.abs(d.v) > 0.5 ? '#fff' : '#333')");
        sb.AppendLine("    .text(d => d.v.toFixed(2));");

        // Axes
        sb.AppendLine("  g.append('g').attr('transform', `translate(0,${height})`).call(d3.axisBottom(x))");
        sb.AppendLine("    .selectAll('text').attr('transform', 'rotate(-45)').attr('text-anchor', 'end').attr('font-size', '10px');");
        sb.AppendLine("  g.append('g').call(d3.axisLeft(y)).selectAll('text').attr('font-size', '10px');");

        EmitTitle(sb, layout);
    }

    // ═══════════════════════════════════════════════════════════
    // Parallel coordinates
    // ═══════════════════════════════════════════════════════════
    private static void RenderParallelCoordinates(StringBuilder sb, ChartSpec spec)
    {
        var trace = spec.Traces[0];
        var layout = spec.Layout;
        var cols = trace.Extra["columns"] as string[] ?? [];
        var colData = trace.Extra["columnData"] as List<double[]> ?? [];

        sb.AppendLine("  const margin = {top: 50, right: 30, bottom: 30, left: 30};");
        sb.AppendLine("  const width = W - margin.left - margin.right;");
        sb.AppendLine("  const height = H - margin.top - margin.bottom;");
        sb.AppendLine("  const svg = d3.select(container).append('svg')");
        sb.AppendLine("    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('max-width', W + 'px')");
        sb.AppendLine("    .style('font-family', 'system-ui, -apple-system, sans-serif')");
        sb.AppendLine("    .style('font-size', '12px');");
        sb.AppendLine("  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);");

        sb.AppendLine($"  const dims = {Json(cols)};");
        sb.AppendLine($"  const nRows = {(colData.Count > 0 ? colData[0].Length : 0)};");

        // Serialize column data
        sb.AppendLine("  const colData = [");
        foreach (var cd in colData)
            sb.AppendLine($"    {Json(cd)},");
        sb.AppendLine("  ];");

        // Y scales per dimension
        sb.AppendLine("  const yScales = dims.map((d, i) => {");
        sb.AppendLine("    const vals = colData[i].filter(v => isFinite(v));");
        sb.AppendLine("    return d3.scaleLinear().domain(d3.extent(vals)).nice().range([height, 0]);");
        sb.AppendLine("  });");

        // X scale: evenly spaced axes
        sb.AppendLine("  const xScale = d3.scalePoint().domain(dims).range([0, width]);");

        // Draw lines (sample if too many)
        sb.AppendLine("  const maxLines = Math.min(nRows, 500);");
        sb.AppendLine("  const step = Math.max(1, Math.floor(nRows / maxLines));");
        sb.AppendLine($"  const lineColor = d3.scaleSequential(d3.interpolateViridis).domain([0, maxLines]);");

        sb.AppendLine("  for (let r = 0; r < nRows; r += step) {");
        sb.AppendLine("    const pts = dims.map((d, i) => [xScale(d), yScales[i](colData[i][r])]);");
        sb.AppendLine("    g.append('path').datum(pts)");
        sb.AppendLine("      .attr('d', d3.line()).attr('fill', 'none')");
        sb.AppendLine("      .attr('stroke', lineColor(r / step)).attr('stroke-width', 1).attr('opacity', 0.3);");
        sb.AppendLine("  }");

        // Draw axes
        sb.AppendLine("  dims.forEach((d, i) => {");
        sb.AppendLine("    const axisG = g.append('g').attr('transform', `translate(${xScale(d)},0)`);");
        sb.AppendLine("    axisG.call(d3.axisLeft(yScales[i]).ticks(6));");
        sb.AppendLine("    axisG.append('text').attr('y', -10).attr('text-anchor', 'middle')");
        sb.AppendLine("      .attr('font-size', '11px').attr('font-weight', 'bold').attr('fill', '#333').text(d);");
        sb.AppendLine("  });");

        EmitTitle(sb, layout);
    }

    // ═══════════════════════════════════════════════════════════
    // Treemap
    // ═══════════════════════════════════════════════════════════
    private static void RenderTreemap(StringBuilder sb, ChartSpec spec)
    {
        var trace = spec.Traces[0];
        var layout = spec.Layout;
        var labels = trace.Extra["hierarchy"] as string[] ?? [];
        var values = trace.Extra["values"] as double[] ?? [];

        sb.AppendLine("  const svg = d3.select(container).append('svg')");
        sb.AppendLine("    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('max-width', W + 'px')");
        sb.AppendLine("    .style('font-family', 'system-ui, -apple-system, sans-serif')");
        sb.AppendLine("    .style('font-size', '12px');");

        sb.AppendLine($"  const labels = {Json(labels)};");
        sb.AppendLine($"  const values = {Json(values)};");

        // Group by label, sum values
        sb.AppendLine("  const groups = {};");
        sb.AppendLine("  labels.forEach((l, i) => { groups[l] = (groups[l] || 0) + values[i]; });");
        sb.AppendLine("  const children = Object.entries(groups).map(([name, value]) => ({name, value}));");
        sb.AppendLine("  const rootData = {name: 'root', children};");

        sb.AppendLine($"  const color = d3.scaleOrdinal().range({Json(Palette)});");
        sb.AppendLine("  const root = d3.hierarchy(rootData).sum(d => d.value).sort((a,b) => b.value - a.value);");
        sb.AppendLine($"  d3.treemap().size([W, H]).padding(2)(root);");

        EmitTooltip(sb);

        sb.AppendLine("  const cell = svg.selectAll('g').data(root.leaves()).join('g')");
        sb.AppendLine("    .attr('transform', d => `translate(${d.x0},${d.y0})`);");
        sb.AppendLine("  cell.append('rect')");
        sb.AppendLine("    .attr('width', d => d.x1 - d.x0).attr('height', d => d.y1 - d.y0)");
        sb.AppendLine("    .attr('fill', d => color(d.data.name)).attr('rx', 3).attr('opacity', 0.85)");
        sb.AppendLine("    .on('mouseover', function(event, d) {");
        sb.AppendLine("      tooltip.style('opacity', 1).html(`${d.data.name}<br>${d.data.value.toFixed(1)}`)");
        sb.AppendLine("        .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');");
        sb.AppendLine("    }).on('mouseout', () => tooltip.style('opacity', 0));");

        // Labels (only if cell is large enough)
        sb.AppendLine("  cell.append('text').attr('x', 4).attr('y', 14)");
        sb.AppendLine("    .attr('font-size', '11px').attr('fill', '#fff').attr('font-weight', 'bold')");
        sb.AppendLine("    .text(d => (d.x1 - d.x0) > 40 ? d.data.name : '');");
        sb.AppendLine("  cell.append('text').attr('x', 4).attr('y', 28)");
        sb.AppendLine("    .attr('font-size', '10px').attr('fill', 'rgba(255,255,255,0.8)')");
        sb.AppendLine("    .text(d => (d.x1 - d.x0) > 50 ? d.data.value.toFixed(0) : '');");

        EmitTitle(sb, layout);
    }

    // ═══════════════════════════════════════════════════════════
    // Network / Force-directed graph
    // ═══════════════════════════════════════════════════════════
    private static void RenderNetwork(StringBuilder sb, ChartSpec spec)
    {
        var trace = spec.Traces[0];
        var layout = spec.Layout;
        var sources = trace.Extra["sources"] as string[] ?? [];
        var targets = trace.Extra["targets"] as string[] ?? [];
        var hasWeights = trace.Extra.ContainsKey("weights");

        sb.AppendLine("  const svg = d3.select(container).append('svg')");
        sb.AppendLine("    .attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%').style('max-width', W + 'px')");
        sb.AppendLine("    .style('font-family', 'system-ui, -apple-system, sans-serif')");
        sb.AppendLine("    .style('font-size', '12px');");

        sb.AppendLine($"  const sources = {Json(sources)};");
        sb.AppendLine($"  const targets = {Json(targets)};");
        if (hasWeights)
            sb.AppendLine($"  const weights = {Json(trace.Extra["weights"])};");

        // Build nodes and links
        sb.AppendLine("  const nodeSet = new Set([...sources, ...targets]);");
        sb.AppendLine("  const nodes = [...nodeSet].map(id => ({id}));");
        sb.AppendLine("  const links = sources.map((s, i) => ({source: s, target: targets[i]" + (hasWeights ? ", weight: weights[i]" : "") + "}));");

        sb.AppendLine($"  const color = d3.scaleOrdinal().range({Json(Palette)});");

        EmitTooltip(sb);

        // Force simulation
        sb.AppendLine("  const simulation = d3.forceSimulation(nodes)");
        sb.AppendLine("    .force('link', d3.forceLink(links).id(d => d.id).distance(80))");
        sb.AppendLine("    .force('charge', d3.forceManyBody().strength(-200))");
        sb.AppendLine("    .force('center', d3.forceCenter(W / 2, H / 2))");
        sb.AppendLine("    .force('collision', d3.forceCollide().radius(20));");

        // Zoom
        sb.AppendLine("  const zoomG = svg.append('g');");
        sb.AppendLine("  svg.call(d3.zoom().scaleExtent([0.2, 5]).on('zoom', e => zoomG.attr('transform', e.transform)));");

        // Draw links
        sb.AppendLine("  const link = zoomG.selectAll('.link').data(links).join('line')");
        sb.AppendLine("    .attr('stroke', '#999').attr('stroke-opacity', 0.6)");
        sb.AppendLine("    .attr('stroke-width', d => d.weight ? Math.max(1, d.weight) : 1.5);");

        // Draw nodes
        sb.AppendLine("  const node = zoomG.selectAll('.node').data(nodes).join('circle')");
        sb.AppendLine("    .attr('r', 8).attr('fill', d => color(d.id))");
        sb.AppendLine("    .attr('stroke', '#fff').attr('stroke-width', 1.5)");
        sb.AppendLine("    .call(d3.drag()");
        sb.AppendLine("      .on('start', (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })");
        sb.AppendLine("      .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y; })");
        sb.AppendLine("      .on('end', (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }))");
        sb.AppendLine("    .on('mouseover', function(event, d) {");
        sb.AppendLine("      tooltip.style('opacity', 1).html(`<strong>${d.id}</strong>`)");
        sb.AppendLine("        .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');");
        sb.AppendLine("    }).on('mouseout', () => tooltip.style('opacity', 0));");

        // Node labels
        sb.AppendLine("  const label = zoomG.selectAll('.label').data(nodes).join('text')");
        sb.AppendLine("    .attr('font-size', '9px').attr('text-anchor', 'middle').attr('dy', -12)");
        sb.AppendLine("    .text(d => d.id);");

        // Tick update
        sb.AppendLine("  simulation.on('tick', () => {");
        sb.AppendLine("    link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)");
        sb.AppendLine("        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);");
        sb.AppendLine("    node.attr('cx', d => d.x).attr('cy', d => d.y);");
        sb.AppendLine("    label.attr('x', d => d.x).attr('y', d => d.y);");
        sb.AppendLine("  });");

        EmitTitle(sb, layout);
    }

    // ═══════════════════════════════════════════════════════════
    // Shared helpers
    // ═══════════════════════════════════════════════════════════

    private static void EmitTraceData(StringBuilder sb, ChartSpec spec)
    {
        sb.AppendLine("  const traces = [");
        foreach (var trace in spec.Traces)
        {
            sb.Append("    {");
            sb.Append($"name: {Json(trace.Name ?? "")}, ");
            sb.Append($"type: {Json(trace.Type)}, ");
            if (trace.Mode is not null) sb.Append($"mode: {Json(trace.Mode)}, ");
            if (trace.XLabels is not null) sb.Append($"xLabels: {Json(trace.XLabels)}, ");
            if (trace.X is not null) sb.Append($"x: {Json(trace.X)}, ");
            if (trace.YLabels is not null) sb.Append($"yLabels: {Json(trace.YLabels)}, ");
            if (trace.Y is not null) sb.Append($"y: {Json(trace.Y)}, ");
            if (trace.MarkerSize is not null) sb.Append($"size: {Json(trace.MarkerSize)}, ");
            if (trace.Text is not null) sb.Append($"text: {Json(trace.Text)}, ");
            if (trace.Orientation is not null) sb.Append($"orientation: {Json(trace.Orientation)}, ");
            bool hasFill = trace.Extra.ContainsKey("fill");
            if (hasFill) sb.Append($"fill: true, ");
            sb.AppendLine("},");
        }
        sb.AppendLine("  ];");
    }

    private static void BuildBandScaleX(StringBuilder sb, ChartSpec spec)
    {
        // Collect all unique categories across traces
        sb.AppendLine("  const allXCats = [...new Set(traces.flatMap(t => t.xLabels || []))];");
        sb.AppendLine("  const x = d3.scaleBand().domain(allXCats).range([0, width]).padding(0.2);");
    }

    private static void BuildBandScaleY(StringBuilder sb, ChartSpec spec)
    {
        sb.AppendLine("  const allYCats = [...new Set(traces.flatMap(t => t.yLabels || []))];");
        sb.AppendLine("  const y = d3.scaleBand().domain(allYCats).range([height, 0]).padding(0.2);");
    }

    private static void BuildLinearScaleX(StringBuilder sb, ChartSpec spec, bool isHorizontal)
    {
        sb.AppendLine("  const allX = traces.flatMap(t => t.x || []).filter(v => isFinite(v));");
        if (isHorizontal)
            sb.AppendLine("  const x = d3.scaleLinear().domain([Math.min(0, d3.min(allX)), d3.max(allX)]).nice().range([0, width]);");
        else
            sb.AppendLine("  const x = d3.scaleLinear().domain(d3.extent(allX)).nice().range([0, width]);");
    }

    private static void BuildLinearScaleY(StringBuilder sb, ChartSpec spec, bool isHorizontal, bool isBandX)
    {
        sb.AppendLine("  const allY = traces.flatMap(t => t.y || []).filter(v => isFinite(v));");
        if (isBandX) // bar chart: y starts at 0
            sb.AppendLine("  const y = d3.scaleLinear().domain([0, d3.max(allY)]).nice().range([height, 0]);");
        else if (isHorizontal)
            sb.AppendLine("  const y = d3.scaleLinear().domain(d3.extent(allY)).nice().range([height, 0]);");
        else
            sb.AppendLine("  const y = d3.scaleLinear().domain(d3.extent(allY)).nice().range([height, 0]);");
    }

    private static void EmitAxes(StringBuilder sb, LayoutSpec layout, bool hasX, bool hasY)
    {
        sb.AppendLine("  g.append('g').attr('transform', `translate(0,${height})`)");
        sb.AppendLine("    .call(d3.axisBottom(x)).selectAll('text').attr('font-size', '10px');");
        sb.AppendLine("  g.append('g').call(d3.axisLeft(y)).selectAll('text').attr('font-size', '10px');");

        // Grid lines
        sb.AppendLine("  g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-width).tickFormat('')).selectAll('line').attr('stroke','#e0e0e0').attr('stroke-dasharray','2,2');");
        sb.AppendLine("  g.selectAll('.grid .domain').remove();");

        // Axis labels
        if (layout.XAxisTitle is not null)
        {
            sb.AppendLine($"  svg.append('text').attr('x', W/2).attr('y', H - 6)");
            sb.AppendLine($"    .attr('text-anchor', 'middle').attr('font-size', '13px')");
            sb.AppendLine($"    .text({Json(layout.XAxisTitle)});");
        }
        if (layout.YAxisTitle is not null)
        {
            sb.AppendLine($"  svg.append('text').attr('transform', 'rotate(-90)')");
            sb.AppendLine($"    .attr('x', -H/2).attr('y', 14).attr('text-anchor', 'middle').attr('font-size', '13px')");
            sb.AppendLine($"    .text({Json(layout.YAxisTitle)});");
        }
    }

    private static void EmitTooltip(StringBuilder sb)
    {
        sb.AppendLine("  const tooltip = d3.select('body').append('div')");
        sb.AppendLine("    .style('position', 'absolute').style('background', 'rgba(0,0,0,0.8)')");
        sb.AppendLine("    .style('color', '#fff').style('padding', '6px 10px').style('border-radius', '4px')");
        sb.AppendLine("    .style('font-size', '12px').style('pointer-events', 'none').style('opacity', 0)");
        sb.AppendLine("    .style('z-index', '10000');");
    }

    private static void EmitTitle(StringBuilder sb, LayoutSpec layout)
    {
        if (layout.Title is not null)
        {
            sb.AppendLine($"  svg.append('text').attr('x', W/2).attr('y', 22)");
            sb.AppendLine($"    .attr('text-anchor', 'middle').attr('font-size', '16px').attr('font-weight', 'bold')");
            sb.AppendLine($"    .text({Json(layout.Title)});");
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Zoom / Pan (d3-zoom on scatter, line, area)
    // ═══════════════════════════════════════════════════════════
    private static void EmitZoomPan(StringBuilder sb)
    {
        sb.AppendLine("  // Zoom & pan — drag to pan, scroll to zoom, double-click to reset");
        sb.AppendLine("  const xAxis = g.select('g:nth-child(1)');"); // first child is bottom axis
        sb.AppendLine("  const yAxis = g.select('g:nth-child(2)');");
        sb.AppendLine("  const gridG = g.select('.grid');");
        sb.AppendLine("  const x0 = x.copy(), y0 = y.copy();");
        sb.AppendLine("  const zoom = d3.zoom().scaleExtent([0.5, 20])");
        sb.AppendLine("    .on('zoom', function(event) {");
        sb.AppendLine("      const t = event.transform;");
        sb.AppendLine("      const nx = t.rescaleX(x0), ny = t.rescaleY(y0);");
        sb.AppendLine("      x.domain(nx.domain()); y.domain(ny.domain());");
        sb.AppendLine("      xAxis.call(d3.axisBottom(x));");
        sb.AppendLine("      yAxis.call(d3.axisLeft(y));");
        sb.AppendLine("      if (gridG.size()) gridG.call(d3.axisLeft(y).tickSize(-width).tickFormat('')).selectAll('line').attr('stroke','#e0e0e0').attr('stroke-dasharray','2,2');");
        sb.AppendLine("      if (gridG.size()) gridG.selectAll('.domain').remove();");
        // Update all data elements
        sb.AppendLine("      plotArea.selectAll('circle').attr('cx', function() { const d = d3.select(this).datum(); return d ? x(d.x) : null; }).attr('cy', function() { const d = d3.select(this).datum(); return d ? y(d.y) : null; });");
        sb.AppendLine("      plotArea.selectAll('path').each(function() {");
        sb.AppendLine("        const el = d3.select(this); const data = el.datum();");
        sb.AppendLine("        if (!data || !Array.isArray(data)) return;");
        sb.AppendLine("        if (el.attr('fill') !== 'none' && el.attr('fill')) {");
        sb.AppendLine("          el.attr('d', d3.area().defined(d => isFinite(d.x)&&isFinite(d.y)).x(d=>x(d.x)).y0(height).y1(d=>y(d.y)));");
        sb.AppendLine("        } else {");
        sb.AppendLine("          el.attr('d', d3.line().defined(d => isFinite(d.x)&&isFinite(d.y)).x(d=>x(d.x)).y(d=>y(d.y)));");
        sb.AppendLine("        }");
        sb.AppendLine("      });");
        sb.AppendLine("    });");
        sb.AppendLine("  svg.call(zoom);");
        // Double-click to reset
        sb.AppendLine("  svg.on('dblclick.zoom', function() {");
        sb.AppendLine("    svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);");
        sb.AppendLine("  });");
    }

    // ═══════════════════════════════════════════════════════════
    // Clickable legend with show/hide toggle
    // ═══════════════════════════════════════════════════════════
    private static void EmitClickableLegend(StringBuilder sb)
    {
        sb.AppendLine("  const hidden = new Set();");
        sb.AppendLine("  const legend = svg.append('g').attr('transform', `translate(${W - margin.right + 8}, ${margin.top})`);");
        sb.AppendLine("  traces.forEach((t, i) => {");
        sb.AppendLine("    const ly = i * 22;");
        sb.AppendLine("    const key = t.name || `trace_${i}`;");
        sb.AppendLine("    const lg = legend.append('g').attr('transform', `translate(0,${ly})`).style('cursor', 'pointer');");
        sb.AppendLine("    lg.append('rect').attr('width', 12).attr('height', 12).attr('rx', 2)");
        sb.AppendLine("      .attr('fill', color(key)).attr('class', 'legend-swatch');");
        sb.AppendLine("    lg.append('text').attr('x', 16).attr('y', 10).attr('font-size', '11px')");
        sb.AppendLine("      .text(key).attr('class', 'legend-label');");
        // Click handler
        sb.AppendLine("    lg.on('click', function() {");
        sb.AppendLine("      if (hidden.has(i)) { hidden.delete(i); } else { hidden.add(i); }");
        sb.AppendLine("      const vis = !hidden.has(i);");
        sb.AppendLine("      d3.select(this).select('.legend-swatch').attr('opacity', vis ? 1 : 0.2);");
        sb.AppendLine("      d3.select(this).select('.legend-label').attr('opacity', vis ? 1 : 0.4)");
        sb.AppendLine("        .style('text-decoration', vis ? 'none' : 'line-through');");
        // Toggle trace visibility
        sb.AppendLine("      plotArea.selectAll(`.trace-${i}`).attr('opacity', vis ? 0.7 : 0);");
        sb.AppendLine("      plotArea.selectAll(`path.trace-${i}`).attr('opacity', vis ? 1 : 0);");
        sb.AppendLine("    });");
        sb.AppendLine("  });");
    }

    // ═══════════════════════════════════════════════════════════
    // Animation with D3 transitions
    // ═══════════════════════════════════════════════════════════
    private static void EmitAnimation(StringBuilder sb, ChartSpec spec, bool isBar, bool isHorizontal)
    {
        // Serialize all frames as JS array
        sb.AppendLine("  // Animation frames");
        sb.AppendLine("  const frames = [");
        foreach (var frame in spec.Frames)
        {
            sb.Append($"    {{name: {Json(frame.Name)}, data: [");
            foreach (var trace in frame.Data)
            {
                sb.Append("{");
                if (trace.XLabels is not null) sb.Append($"xLabels: {Json(trace.XLabels)}, ");
                if (trace.X is not null) sb.Append($"x: {Json(trace.X)}, ");
                if (trace.YLabels is not null) sb.Append($"yLabels: {Json(trace.YLabels)}, ");
                if (trace.Y is not null) sb.Append($"y: {Json(trace.Y)}, ");
                if (trace.MarkerSize is not null) sb.Append($"size: {Json(trace.MarkerSize)}, ");
                if (trace.Text is not null) sb.Append($"text: {Json(trace.Text)}, ");
                sb.Append("},");
            }
            sb.AppendLine("]},");
        }
        sb.AppendLine("  ];");

        // Extract durations from layout Extra (set by VizBuilder.Animate)
        int frameDuration = 500;
        int transitionDuration = 300;
        if (spec.Layout.Extra.TryGetValue("sliders", out var sliders) && sliders is object[] sliderArr && sliderArr.Length > 0)
        {
            // Parse duration from the Plotly slider config
            if (sliderArr[0] is Dictionary<string, object?> sliderDict &&
                sliderDict.TryGetValue("steps", out var steps) && steps is Dictionary<string, object?>[] stepsArr && stepsArr.Length > 0 &&
                stepsArr[0].TryGetValue("args", out var args) && args is object?[] argsArr && argsArr.Length > 1 &&
                argsArr[1] is Dictionary<string, object?> animArgs &&
                animArgs.TryGetValue("frame", out var frameDict) && frameDict is Dictionary<string, int> fd)
            {
                frameDuration = fd.GetValueOrDefault("duration", 500);
            }
        }

        // Transition function: update data elements for a given frame
        sb.AppendLine("  let currentFrame = 0;");
        sb.AppendLine("  function goToFrame(fi) {");
        sb.AppendLine("    currentFrame = fi;");
        sb.AppendLine("    const frame = frames[fi];");
        sb.AppendLine("    frame.data.forEach((fd, ti) => {");
        if (isBar)
        {
            // For bar charts: update bar heights/widths
            if (!isHorizontal)
            {
                sb.AppendLine("      const data = fd.xLabels ? fd.xLabels.map((l, i) => ({label: l, value: fd.y[i]})) : [];");
                sb.AppendLine("      plotArea.selectAll(`.bar-${ti}`).data(data)");
                sb.AppendLine($"        .transition().duration({transitionDuration})");
                sb.AppendLine("        .attr('y', d => y(d.value)).attr('height', d => height - y(d.value));");
            }
            else
            {
                sb.AppendLine("      const data = fd.yLabels ? fd.yLabels.map((l, i) => ({label: l, value: fd.x[i]})) : [];");
                sb.AppendLine("      plotArea.selectAll(`.bar-${ti}`).data(data)");
                sb.AppendLine($"        .transition().duration({transitionDuration})");
                sb.AppendLine("        .attr('width', d => x(d.value));");
            }
        }
        else
        {
            // For scatter/line: update positions
            sb.AppendLine("      if (fd.x && fd.y) {");
            sb.AppendLine("        const data = fd.x.map((xv, i) => ({x: xv, y: fd.y[i], text: fd.text?.[i]}));");
            // Update circles
            sb.AppendLine("        plotArea.selectAll(`.dot-${ti}`).data(data.filter(d => isFinite(d.x) && isFinite(d.y)))");
            sb.AppendLine($"          .transition().duration({transitionDuration})");
            sb.AppendLine("          .attr('cx', d => x(d.x)).attr('cy', d => y(d.y));");
            // Update line paths
            sb.AppendLine("        plotArea.selectAll(`path.trace-${ti}`).datum(data)");
            sb.AppendLine($"          .transition().duration({transitionDuration})");
            sb.AppendLine("          .attr('d', function() {");
            sb.AppendLine("            const el = d3.select(this);");
            sb.AppendLine("            if (el.attr('fill') !== 'none' && el.attr('fill'))");
            sb.AppendLine("              return d3.area().defined(d=>isFinite(d.x)&&isFinite(d.y)).x(d=>x(d.x)).y0(height).y1(d=>y(d.y))(data);");
            sb.AppendLine("            return d3.line().defined(d=>isFinite(d.x)&&isFinite(d.y)).x(d=>x(d.x)).y(d=>y(d.y))(data);");
            sb.AppendLine("          });");
            sb.AppendLine("      }");
        }
        sb.AppendLine("    });");
        sb.AppendLine("    slider.property('value', fi);");
        sb.AppendLine("    frameLabel.text(frame.name);");
        sb.AppendLine("  }");

        // Playback controls: HTML overlay below the chart
        sb.AppendLine("  const controls = d3.select(container).append('div')");
        sb.AppendLine("    .style('display', 'flex').style('align-items', 'center').style('gap', '10px')");
        sb.AppendLine("    .style('padding', '8px 0').style('font-family', 'system-ui, sans-serif').style('font-size', '13px');");

        // Play/Pause button
        sb.AppendLine("  let playing = false, timer = null;");
        sb.AppendLine("  const playBtn = controls.append('button')");
        sb.AppendLine("    .text('▶ Play').style('padding', '4px 12px').style('cursor', 'pointer')");
        sb.AppendLine("    .style('border', '1px solid #ccc').style('border-radius', '4px').style('background', '#fff');");

        // Slider
        sb.AppendLine("  const slider = controls.append('input').attr('type', 'range')");
        sb.AppendLine("    .attr('min', 0).attr('max', frames.length - 1).attr('value', 0)");
        sb.AppendLine("    .style('flex', '1');");

        // Frame label
        sb.AppendLine("  const frameLabel = controls.append('span').text(frames[0].name).style('min-width', '60px');");

        // Wire up events
        sb.AppendLine("  slider.on('input', function() { goToFrame(+this.value); });");
        sb.AppendLine("  playBtn.on('click', function() {");
        sb.AppendLine("    playing = !playing;");
        sb.AppendLine("    playBtn.text(playing ? '⏸ Pause' : '▶ Play');");
        sb.AppendLine("    if (playing) {");
        sb.AppendLine($"      timer = setInterval(() => {{");
        sb.AppendLine("        currentFrame = (currentFrame + 1) % frames.length;");
        sb.AppendLine("        goToFrame(currentFrame);");
        sb.AppendLine($"      }}, {frameDuration});");
        sb.AppendLine("    } else { clearInterval(timer); }");
        sb.AppendLine("  });");
    }

    // ═══════════════════════════════════════════════════════════
    // Element renderers (draw into plotArea for clipping)
    // ═══════════════════════════════════════════════════════════

    private static void EmitVerticalBars(StringBuilder sb, bool grouped)
    {
        sb.AppendLine("    const data = trace.xLabels.map((label, i) => ({label, value: trace.y[i]}));");
        if (grouped)
        {
            sb.AppendLine("    plotArea.selectAll(`.bar-${ti}`).data(data).join('rect').attr('class', `bar-${ti} trace-${ti}`)");
            sb.AppendLine("      .attr('x', d => x(d.label) + x1(trace.name))");
            sb.AppendLine("      .attr('y', d => y(d.value))");
            sb.AppendLine("      .attr('width', x1.bandwidth())");
            sb.AppendLine("      .attr('height', d => height - y(d.value))");
        }
        else
        {
            sb.AppendLine("    plotArea.selectAll(`.bar-${ti}`).data(data).join('rect').attr('class', `bar-${ti} trace-${ti}`)");
            sb.AppendLine("      .attr('x', d => x(d.label))");
            sb.AppendLine("      .attr('y', d => y(d.value))");
            sb.AppendLine("      .attr('width', x.bandwidth())");
            sb.AppendLine("      .attr('height', d => height - y(d.value))");
        }
        sb.AppendLine("      .attr('fill', c).attr('opacity', 0.8)");
        sb.AppendLine("      .on('mouseover', function(event, d) {");
        sb.AppendLine("        tooltip.style('opacity', 1).html(`${d.label}<br>${d.value.toFixed(2)}`)");
        sb.AppendLine("          .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');");
        sb.AppendLine("      })");
        sb.AppendLine("      .on('mouseout', function() { tooltip.style('opacity', 0); });");
    }

    private static void EmitHorizontalBars(StringBuilder sb, bool grouped)
    {
        sb.AppendLine("    const data = trace.yLabels.map((label, i) => ({label, value: trace.x[i]}));");
        if (grouped)
        {
            sb.AppendLine("    plotArea.selectAll(`.bar-${ti}`).data(data).join('rect').attr('class', `bar-${ti} trace-${ti}`)");
            sb.AppendLine("      .attr('y', d => y(d.label) + y1(trace.name))");
            sb.AppendLine("      .attr('x', 0)");
            sb.AppendLine("      .attr('height', y1.bandwidth())");
            sb.AppendLine("      .attr('width', d => x(d.value))");
        }
        else
        {
            sb.AppendLine("    plotArea.selectAll(`.bar-${ti}`).data(data).join('rect').attr('class', `bar-${ti} trace-${ti}`)");
            sb.AppendLine("      .attr('y', d => y(d.label))");
            sb.AppendLine("      .attr('x', 0)");
            sb.AppendLine("      .attr('height', y.bandwidth())");
            sb.AppendLine("      .attr('width', d => x(d.value))");
        }
        sb.AppendLine("      .attr('fill', c).attr('opacity', 0.8)");
        sb.AppendLine("      .on('mouseover', function(event, d) {");
        sb.AppendLine("        tooltip.style('opacity', 1).html(`${d.label}<br>${d.value.toFixed(2)}`)");
        sb.AppendLine("          .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');");
        sb.AppendLine("      })");
        sb.AppendLine("      .on('mouseout', function() { tooltip.style('opacity', 0); });");
    }

    private static void EmitLinesAndMarkers(StringBuilder sb)
    {
        sb.AppendLine("    const data = trace.x.map((xv, i) => ({x: xv, y: trace.y[i], text: trace.text?.[i]}));");

        // Line path (if mode contains 'lines')
        sb.AppendLine("    if (!trace.mode || trace.mode.includes('lines')) {");
        sb.AppendLine("      plotArea.append('path').datum(data).attr('class', `trace-${ti}`)");
        sb.AppendLine("        .attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2)");
        sb.AppendLine("        .attr('d', d3.line().defined(d => isFinite(d.x) && isFinite(d.y)).x(d => x(d.x)).y(d => y(d.y)));");
        sb.AppendLine("    }");

        // Markers (if mode contains 'markers')
        sb.AppendLine("    if (trace.mode && trace.mode.includes('markers')) {");
        sb.AppendLine("      plotArea.selectAll(`.dot-${ti}`).data(data.filter(d => isFinite(d.x) && isFinite(d.y))).join('circle')");
        sb.AppendLine("        .attr('class', `dot-${ti} trace-${ti}`)");
        sb.AppendLine("        .attr('cx', d => x(d.x)).attr('cy', d => y(d.y))");
        sb.AppendLine("        .attr('r', (d, i) => trace.size ? Math.max(2, trace.size[i] / d3.max(trace.size) * 20) : 4)");
        sb.AppendLine("        .attr('fill', c).attr('opacity', 0.7)");
        sb.AppendLine("        .on('mouseover', function(event, d) {");
        sb.AppendLine("          const label = d.text || `x: ${d.x.toFixed(2)}, y: ${d.y.toFixed(2)}`;");
        sb.AppendLine("          tooltip.style('opacity', 1).html(label)");
        sb.AppendLine("            .style('left', (event.pageX + 10) + 'px').style('top', (event.pageY - 28) + 'px');");
        sb.AppendLine("        })");
        sb.AppendLine("        .on('mouseout', function() { tooltip.style('opacity', 0); });");
        sb.AppendLine("    }");
    }

    private static void EmitArea(StringBuilder sb)
    {
        sb.AppendLine("    const data = trace.x.map((xv, i) => ({x: xv, y: trace.y[i]}));");
        sb.AppendLine("    plotArea.append('path').datum(data).attr('class', `trace-${ti}`)");
        sb.AppendLine("      .attr('fill', c).attr('opacity', 0.3)");
        sb.AppendLine("      .attr('d', d3.area().defined(d => isFinite(d.x) && isFinite(d.y))");
        sb.AppendLine("        .x(d => x(d.x)).y0(height).y1(d => y(d.y)));");
        sb.AppendLine("    plotArea.append('path').datum(data).attr('class', `trace-${ti}`)");
        sb.AppendLine("      .attr('fill', 'none').attr('stroke', c).attr('stroke-width', 2)");
        sb.AppendLine("      .attr('d', d3.line().defined(d => isFinite(d.x) && isFinite(d.y)).x(d => x(d.x)).y(d => y(d.y)));");
    }
}
