using PandaSharp.Vision.Transforms;

namespace PandaSharp.Vision;

/// <summary>
/// HTML visualization helpers for image datasets.
/// </summary>
public static class ImageViz
{
    /// <summary>
    /// Create an HTML grid of images from a DataFrame.
    /// </summary>
    /// <param name="df">DataFrame containing image paths.</param>
    /// <param name="pathColumn">Name of the column containing image file paths or data URIs.</param>
    /// <param name="columns">Number of columns in the grid (used for layout hints).</param>
    /// <param name="thumbWidth">Width of each thumbnail in pixels.</param>
    /// <param name="labelColumn">Optional column name to display as a label under each image.</param>
    /// <returns>An HTML string representing the image grid.</returns>
    public static string ToImageGrid(DataFrame df, string pathColumn,
        int columns = 4, int thumbWidth = 150, string? labelColumn = null)
    {
        var col = df.GetStringColumn(pathColumn);
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("<div style='display:flex;flex-wrap:wrap;gap:8px'>");
        for (int i = 0; i < col.Length; i++)
        {
            if (col[i] is null) continue;
            var label = labelColumn is not null ? df[labelColumn].GetObject(i)?.ToString() : null;
            sb.Append($"<div style='text-align:center'><img src='{col[i]}' width='{thumbWidth}'/>");
            if (label is not null)
                sb.Append($"<br/><small>{label}</small>");
            sb.AppendLine("</div>");
        }
        sb.AppendLine("</div>");
        return sb.ToString();
    }

    /// <summary>
    /// Show augmentation pipeline results side-by-side as HTML.
    /// Applies the pipeline multiple times to the same source image.
    /// </summary>
    /// <param name="imagePath">Path to the source image.</param>
    /// <param name="pipeline">The augmentation pipeline to apply.</param>
    /// <param name="nSamples">Number of augmented variants to generate.</param>
    /// <param name="columns">Number of columns in the grid (used for layout hints).</param>
    /// <returns>An HTML string showing the original and augmented images.</returns>
    public static string ShowAugmentations(string imagePath, ImagePipeline pipeline,
        int nSamples = 8, int columns = 4)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("<div style='display:flex;flex-wrap:wrap;gap:8px'>");
        sb.Append($"<div style='text-align:center'><img src='{imagePath}' width='150'/><br/><small>Original</small></div>");
        for (int i = 0; i < nSamples; i++)
        {
            var img = ImageIO.Load(imagePath);
            var augmented = pipeline.Transform(img);
            var tmpPath = Path.Combine(Path.GetTempPath(), $"aug_{i}.png");
            ImageIO.Save(augmented, tmpPath);
            sb.Append($"<div style='text-align:center'><img src='{tmpPath}' width='150'/><br/><small>Aug {i + 1}</small></div>");
        }
        sb.AppendLine("</div>");
        return sb.ToString();
    }
}
