using PuppeteerSharp;
using Cortex.Viz.Charts;

namespace Cortex.Viz.Rendering;

/// <summary>
/// Static image export for D3.js charts using PuppeteerSharp (headless Chromium).
/// Supports PNG (rasterized via screenshot) and SVG (extracted from DOM).
/// First call downloads Chromium (~130MB) if not cached.
/// </summary>
public static class StaticExporter
{
    private static readonly SemaphoreSlim _lock = new(1, 1);
    private static IBrowser? _browser;

    /// <summary>Export a chart to a static image file (PNG or SVG).</summary>
    public static void Export(ChartSpec spec, string outputPath, string format, int width, int height, int scale)
        => ExportAsync(spec, outputPath, format, width, height, scale).GetAwaiter().GetResult();

    /// <summary>Export a chart to a static image file asynchronously.</summary>
    public static async Task ExportAsync(ChartSpec spec, string outputPath, string format, int width, int height, int scale)
    {
        if (format is not "png" and not "svg")
            throw new ArgumentException("Format must be 'png' or 'svg'.", nameof(format));

        var browser = await GetBrowserAsync();
        await using var page = await browser.NewPageAsync();

        // Render using D3
        var html = BuildD3Html(spec, width, height);

        await page.SetContentAsync(html, new NavigationOptions
        {
            WaitUntil = [WaitUntilNavigation.Networkidle0]
        });

        // Wait for D3 to render (it's synchronous, but give it a moment for CDN load)
        await page.WaitForSelectorAsync("svg", new WaitForSelectorOptions { Timeout = 15_000 });

        if (format == "svg")
        {
            // Extract SVG directly from DOM
            var svgContent = await page.EvaluateFunctionAsync<string>(
                "() => document.querySelector('#chart svg').outerHTML");
            await File.WriteAllTextAsync(outputPath, svgContent);
        }
        else
        {
            // Screenshot the chart area
            await page.SetViewportAsync(new ViewPortOptions
            {
                Width = width * scale,
                Height = height * scale,
                DeviceScaleFactor = scale
            });

            var chartElement = await page.QuerySelectorAsync("#chart svg");
            if (chartElement is not null)
            {
                await chartElement.ScreenshotAsync(outputPath, new ElementScreenshotOptions
                {
                    Type = ScreenshotType.Png
                });
            }
            else
            {
                // Fallback: screenshot full page
                await page.ScreenshotAsync(outputPath, new ScreenshotOptions
                {
                    Type = ScreenshotType.Png,
                    Clip = new PuppeteerSharp.Media.Clip { X = 0, Y = 0, Width = width, Height = height }
                });
            }
        }
    }

    /// <summary>Export to bytes without writing to a file.</summary>
    public static async Task<byte[]> ExportToBytesAsync(ChartSpec spec, string format, int width, int height, int scale)
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"pandasharp_export_{Guid.NewGuid():N}.{format}");
        try
        {
            await ExportAsync(spec, tempPath, format, width, height, scale);
            return await File.ReadAllBytesAsync(tempPath);
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    /// <summary>Check if a browser is available.</summary>
    public static bool IsAvailable()
    {
        try { return new BrowserFetcher().GetInstalledBrowsers().Any(); }
        catch { return false; }
    }

    /// <summary>Pre-download Chromium to avoid delays on first export.</summary>
    public static async Task EnsureBrowserAsync()
    {
        var fetcher = new BrowserFetcher();
        if (!fetcher.GetInstalledBrowsers().Any())
            await fetcher.DownloadAsync();
    }

    private static async Task<IBrowser> GetBrowserAsync()
    {
        if (_browser is { IsClosed: false }) return _browser;
        await _lock.WaitAsync();
        try
        {
            if (_browser is { IsClosed: false }) return _browser;
            var fetcher = new BrowserFetcher();
            if (!fetcher.GetInstalledBrowsers().Any())
                await fetcher.DownloadAsync();
            _browser = await Puppeteer.LaunchAsync(new LaunchOptions
            {
                Headless = true,
                Args = ["--no-sandbox", "--disable-setuid-sandbox"]
            });
            return _browser;
        }
        finally { _lock.Release(); }
    }

    private static string BuildD3Html(ChartSpec spec, int width, int height)
    {
        // Override dimensions for export
        spec.Layout.Width = width;
        spec.Layout.Height = height;
        var script = D3Renderer.Render(spec, "chart");

        return $"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="{D3HtmlRenderer.CdnUrl}"></script>
        </head>
        <body style="margin:0;padding:0;">
            <div id="chart"></div>
            <script>{script}</script>
        </body>
        </html>
        """;
    }
}
