using System.Diagnostics;
using System.Text;
using System.Text.Json;
using PandaSharp.ML.Models;

namespace PandaSharp.Viz.Charts;

/// <summary>
/// Renders decision tree models as interactive, collapsible D3.js tree diagrams.
/// Supports both regression and classification trees, and random forests (pick a tree by index).
/// </summary>
public static class TreeVisualizer
{
    /// <summary>Render a regression tree as an interactive HTML page.</summary>
    /// <param name="tree">A fitted DecisionTreeRegressor.</param>
    /// <param name="featureNames">Optional feature names (uses "Feature_N" if null).</param>
    /// <param name="maxDepth">Max depth to display (0 = unlimited).</param>
    public static string ToHtml(DecisionTreeRegressor tree, string[]? featureNames = null, int maxDepth = 0)
    {
        if (tree.Root is null) throw new InvalidOperationException("Tree has not been fitted.");
        var json = SerializeRegressionNode(tree.Root, featureNames, 0, maxDepth);
        return BuildHtml(json, "Regression Tree");
    }

    /// <summary>Render a classification tree as an interactive HTML page.</summary>
    public static string ToHtml(DecisionTreeClassifier tree, string[]? featureNames = null, int maxDepth = 0)
    {
        if (tree.Root is null) throw new InvalidOperationException("Tree has not been fitted.");
        var json = SerializeClassificationNode(tree.Root, featureNames, 0, maxDepth);
        return BuildHtml(json, "Classification Tree");
    }

    /// <summary>Render a specific tree from a random forest regressor.</summary>
    public static string ToHtml(RandomForestRegressor forest, int treeIndex, string[]? featureNames = null, int maxDepth = 3)
    {
        var trees = forest.Trees;
        if (trees is null || treeIndex < 0 || treeIndex >= trees.Length)
            throw new ArgumentOutOfRangeException(nameof(treeIndex));
        return ToHtml(trees[treeIndex], featureNames, maxDepth);
    }

    /// <summary>Render a regression tree as an embeddable HTML fragment (for StoryBoard).</summary>
    public static string ToFragment(DecisionTreeRegressor tree, string[]? featureNames = null, int maxDepth = 0, string divId = "tree")
    {
        if (tree.Root is null) throw new InvalidOperationException("Tree has not been fitted.");
        var json = SerializeRegressionNode(tree.Root, featureNames, 0, maxDepth);
        return BuildFragment(json, divId);
    }

    /// <summary>Render a classification tree as an embeddable HTML fragment (for StoryBoard).</summary>
    public static string ToFragment(DecisionTreeClassifier tree, string[]? featureNames = null, int maxDepth = 0, string divId = "tree")
    {
        if (tree.Root is null) throw new InvalidOperationException("Tree has not been fitted.");
        var json = SerializeClassificationNode(tree.Root, featureNames, 0, maxDepth);
        return BuildFragment(json, divId);
    }

    /// <summary>Render a forest tree as an embeddable HTML fragment (for StoryBoard).</summary>
    public static string ToFragment(RandomForestRegressor forest, int treeIndex, string[]? featureNames = null, int maxDepth = 3, string divId = "tree")
    {
        var trees = forest.Trees;
        if (trees is null || treeIndex < 0 || treeIndex >= trees.Length)
            throw new ArgumentOutOfRangeException(nameof(treeIndex));
        return ToFragment(trees[treeIndex], featureNames, maxDepth, divId);
    }

    /// <summary>Save a regression tree visualization to an HTML file.</summary>
    public static void ToHtmlFile(DecisionTreeRegressor tree, string path, string[]? featureNames = null, int maxDepth = 0)
        => File.WriteAllText(path, ToHtml(tree, featureNames, maxDepth));

    /// <summary>Save a classification tree visualization to an HTML file.</summary>
    public static void ToHtmlFile(DecisionTreeClassifier tree, string path, string[]? featureNames = null, int maxDepth = 0)
        => File.WriteAllText(path, ToHtml(tree, featureNames, maxDepth));

    /// <summary>Save a forest tree visualization to an HTML file.</summary>
    public static void ToHtmlFile(RandomForestRegressor forest, int treeIndex, string path, string[]? featureNames = null, int maxDepth = 3)
        => File.WriteAllText(path, ToHtml(forest, treeIndex, featureNames, maxDepth));

    /// <summary>Open a tree visualization in the default browser.</summary>
    public static void Show(DecisionTreeRegressor tree, string[]? featureNames = null, int maxDepth = 0)
    {
        var path = Path.Combine(Path.GetTempPath(), $"pandasharp_tree_{Guid.NewGuid():N}.html");
        ToHtmlFile(tree, path, featureNames, maxDepth);
        Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
    }

    // ═══════════════════════════════════════════════════════════
    // Serialization: tree nodes → JSON hierarchy for d3.hierarchy()
    // ═══════════════════════════════════════════════════════════

    private static string SerializeRegressionNode(RegressionTreeNode node, string[]? names, int depth, int maxDepth)
    {
        var sb = new StringBuilder();
        sb.Append('{');

        if (node.IsLeaf || (maxDepth > 0 && depth >= maxDepth))
        {
            sb.Append($"\"name\": \"value: {node.PredictedValue:F1}\", \"type\": \"leaf\", \"value\": {node.PredictedValue:F2}");
        }
        else
        {
            var fname = names is not null && node.FeatureIndex < names.Length
                ? names[node.FeatureIndex]
                : $"Feature_{node.FeatureIndex}";
            sb.Append($"\"name\": \"{Escape(fname)} <= {node.Threshold:F2}\", \"type\": \"split\", \"feature\": \"{Escape(fname)}\", \"threshold\": {node.Threshold:F2}");
            sb.Append(", \"children\": [");
            sb.Append(SerializeRegressionNode(node.Left!, names, depth + 1, maxDepth));
            sb.Append(',');
            sb.Append(SerializeRegressionNode(node.Right!, names, depth + 1, maxDepth));
            sb.Append(']');
        }

        sb.Append('}');
        return sb.ToString();
    }

    private static string SerializeClassificationNode(TreeNode node, string[]? names, int depth, int maxDepth)
    {
        var sb = new StringBuilder();
        sb.Append('{');

        if (node.IsLeaf || (maxDepth > 0 && depth >= maxDepth))
        {
            sb.Append($"\"name\": \"class: {node.PredictedClass}\", \"type\": \"leaf\", \"value\": {node.PredictedClass}");
            if (node.ClassDistribution.Length > 0)
                sb.Append($", \"distribution\": {JsonSerializer.Serialize(node.ClassDistribution)}");
        }
        else
        {
            var fname = names is not null && node.FeatureIndex < names.Length
                ? names[node.FeatureIndex]
                : $"Feature_{node.FeatureIndex}";
            sb.Append($"\"name\": \"{Escape(fname)} <= {node.Threshold:F2}\", \"type\": \"split\", \"feature\": \"{Escape(fname)}\", \"threshold\": {node.Threshold:F2}");
            sb.Append(", \"children\": [");
            sb.Append(SerializeClassificationNode(node.Left!, names, depth + 1, maxDepth));
            sb.Append(',');
            sb.Append(SerializeClassificationNode(node.Right!, names, depth + 1, maxDepth));
            sb.Append(']');
        }

        sb.Append('}');
        return sb.ToString();
    }

    /// <summary>
    /// Compute NUM_AS_ROOT importance: how many trees use each feature as the root split.
    /// Returns (featureIndex, count) pairs sorted by count descending.
    /// </summary>
    public static List<(int FeatureIndex, int Count)> NumAsRoot(RandomForestRegressor forest)
    {
        var trees = forest.Trees ?? throw new InvalidOperationException("Forest has not been fitted.");
        var counts = new Dictionary<int, int>();
        foreach (var tree in trees)
        {
            if (tree.Root is not null && !tree.Root.IsLeaf)
                counts[tree.Root.FeatureIndex] = counts.GetValueOrDefault(tree.Root.FeatureIndex) + 1;
        }
        return counts.OrderByDescending(kv => kv.Value).Select(kv => (kv.Key, kv.Value)).ToList();
    }

    private static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"");

    // ═══════════════════════════════════════════════════════════
    // HTML generation with D3.js tree layout
    // ═══════════════════════════════════════════════════════════

    private static string BuildFragment(string treeJson, string divId)
    {
        return $$"""
        <style>
            .node circle { stroke-width: 2px; cursor: pointer; }
            .node text { font-size: 11px; fill: #333; }
            .link { fill: none; stroke: #ccc; stroke-width: 1.5px; }
            .tree-tooltip { position: absolute; background: rgba(0,0,0,0.85); color: #fff; padding: 8px 12px;
                       border-radius: 4px; font-size: 12px; pointer-events: none; opacity: 0; z-index: 10000; }
        </style>
        <div id="{{divId}}" style="overflow:auto;"></div>
        <script>
        {{GetTreeScript(treeJson, divId)}}
        </script>
        """;
    }

    private static string BuildHtml(string treeJson, string title)
    {
        return $$"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{{title}} — PandaSharp</title>
            <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
            <style>
                body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background: #fafafa; overflow: auto; }
                .container { padding: 20px; }
                h2 { color: #333; margin-bottom: 10px; }
                .node circle { stroke-width: 2px; cursor: pointer; }
                .node text { font-size: 11px; }
                .link { fill: none; stroke: #ccc; stroke-width: 1.5px; }
                .tree-tooltip { position: absolute; background: rgba(0,0,0,0.85); color: #fff; padding: 8px 12px;
                           border-radius: 4px; font-size: 12px; pointer-events: none; opacity: 0; z-index: 10000; }
                .legend { font-size: 12px; fill: #666; }
            </style>
        </head>
        <body>
        <div class="container">
            <h2>{{title}}</h2>
            <p style="color:#666;font-size:13px;">Click a node to collapse/expand. Scroll to zoom. Drag to pan.</p>
            <div id="tree"></div>
        </div>
        <script>
        {{GetTreeScript(treeJson, "tree")}}
        </script>
        </body>
        </html>
        """;
    }

    private static string GetTreeScript(string treeJson, string divId) => $$"""
        (function() {
            const treeData = {{treeJson}};
            const container = document.getElementById('{{divId}}');

            const margin = {top: 30, right: 120, bottom: 30, left: 80};
            const baseW = Math.max(800, container.clientWidth || 800);
            const nodeH = 60;

            function countLeaves(node) {
                if (!node.children || node.children.length === 0) return 1;
                return node.children.reduce((s, c) => s + countLeaves(c), 0);
            }
            const nLeaves = countLeaves(treeData);
            const svgH = Math.max(400, nLeaves * nodeH + margin.top + margin.bottom);
            const svgW = baseW;

            const svg = d3.select(container).append('svg')
                .attr('width', svgW).attr('height', svgH)
                .style('font-family', 'system-ui, sans-serif');

            const zoomG = svg.append('g');
            svg.call(d3.zoom().scaleExtent([0.2, 5]).on('zoom', e => zoomG.attr('transform', e.transform)));
            const g = zoomG.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

            const tooltip = d3.select('body').append('div').attr('class', 'tree-tooltip');

            const splitColor = '#4e79a7';
            const leafColorScale = d3.scaleSequential(d3.interpolateYlGn).domain([0, 1]);

            const root = d3.hierarchy(treeData);
            const treeLayout = d3.tree().size([svgH - margin.top - margin.bottom, svgW - margin.left - margin.right - 200]);
            treeLayout(root);
            root.descendants().forEach(d => { d._children = d.children; });

            function update(source) {
                treeLayout(root);
                const nodes = root.descendants();
                const links = root.links();
                nodes.forEach(d => { d.y = d.depth * 180; });

                const link = g.selectAll('.link').data(links, d => d.target.data.name + d.target.depth);
                link.enter().append('path').attr('class', 'link')
                    .merge(link).transition().duration(300)
                    .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));
                link.exit().remove();

                const node = g.selectAll('.node').data(nodes, d => d.data.name + d.depth);
                const nodeEnter = node.enter().append('g').attr('class', 'node')
                    .attr('transform', d => `translate(${d.y},${d.x})`);
                nodeEnter.append('circle').attr('r', 8);
                nodeEnter.append('text').attr('dy', '0.35em')
                    .attr('x', d => d.children || d._children ? -14 : 14)
                    .attr('text-anchor', d => d.children || d._children ? 'end' : 'start')
                    .text(d => d.data.name);

                const nodeUpdate = nodeEnter.merge(node);
                nodeUpdate.transition().duration(300)
                    .attr('transform', d => `translate(${d.y},${d.x})`);

                nodeUpdate.select('circle')
                    .attr('fill', d => {
                        if (d.data.type === 'leaf') {
                            const allLeafVals = root.leaves().map(l => l.data.value);
                            const mn = d3.min(allLeafVals), mx = d3.max(allLeafVals);
                            const t = mx > mn ? (d.data.value - mn) / (mx - mn) : 0.5;
                            return leafColorScale(t);
                        }
                        return d._children && !d.children ? '#aaa' : splitColor;
                    })
                    .attr('stroke', d => d.data.type === 'leaf' ? '#666' : splitColor)
                    .attr('r', d => d.data.type === 'leaf' ? 6 : 8);

                nodeUpdate.style('cursor', d => d._children ? 'pointer' : 'default')
                    .on('click', function(event, d) {
                        if (d._children) { d.children = d.children ? null : d._children; update(d); }
                    });

                nodeUpdate
                    .on('mouseover', function(event, d) {
                        let html = `<strong>${d.data.name}</strong>`;
                        if (d.data.type === 'split') html += `<br>Feature: ${d.data.feature}<br>Threshold: ${d.data.threshold}`;
                        else { html += `<br>Predicted: ${d.data.value}`; if (d.data.distribution) html += `<br>Dist: [${d.data.distribution.map(v => v.toFixed(2)).join(', ')}]`; }
                        html += `<br>Depth: ${d.depth}`;
                        tooltip.style('opacity', 1).html(html)
                            .style('left', (event.pageX + 15) + 'px').style('top', (event.pageY - 10) + 'px');
                    })
                    .on('mouseout', () => tooltip.style('opacity', 0));

                nodeUpdate.select('text')
                    .attr('x', d => d.children || d._children ? -14 : 14)
                    .attr('text-anchor', d => d.children || d._children ? 'end' : 'start')
                    .text(d => d.data.name);
                node.exit().remove();
            }

            update(root);

            const lg = svg.append('g').attr('class', 'legend').attr('transform', `translate(20, ${svgH - 50})`);
            lg.append('circle').attr('r', 6).attr('fill', splitColor).attr('cx', 0).attr('cy', 0);
            lg.append('text').attr('x', 12).attr('y', 4).text('Split node').attr('fill', '#666');
            lg.append('circle').attr('r', 5).attr('fill', leafColorScale(0.5)).attr('cx', 110).attr('cy', 0).attr('stroke', '#666');
            lg.append('text').attr('x', 122).attr('y', 4).text('Leaf node (color = prediction value)').attr('fill', '#666');
        })();
        """;
}
