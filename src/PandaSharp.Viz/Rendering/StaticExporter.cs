using PuppeteerSharp;
using PandaSharp.Viz.Charts;

namespace PandaSharp.Viz.Rendering;

/// <summary>
/// Static image export for Plotly charts using PuppeteerSharp (headless Chromium).
/// No Node.js required — PuppeteerSharp manages Chromium automatically.
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

        var traces = PlotlySerializer.SerializeTraces(spec);
        var layout = PlotlySerializer.SerializeLayout(spec);
        var html = BuildHtml(traces, layout, format, width, height, scale);

        await page.SetContentAsync(html, new NavigationOptions
        {
            WaitUntil = [WaitUntilNavigation.Networkidle0]
        });

        await page.WaitForFunctionAsync("() => document.title === 'EXPORT_DONE'",
            new WaitForFunctionOptions { Timeout = 15_000 });

        var dataUrl = await page.EvaluateFunctionAsync<string>("() => window.__exportData");
        var base64 = dataUrl.Split(',')[1];
        var bytes = Convert.FromBase64String(base64);

        if (format == "svg")
            await File.WriteAllTextAsync(outputPath, System.Text.Encoding.UTF8.GetString(bytes));
        else
            await File.WriteAllBytesAsync(outputPath, bytes);
    }

    /// <summary>Export to bytes without writing to a file.</summary>
    public static async Task<byte[]> ExportToBytesAsync(ChartSpec spec, string format, int width, int height, int scale)
    {
        var browser = await GetBrowserAsync();
        await using var page = await browser.NewPageAsync();

        var traces = PlotlySerializer.SerializeTraces(spec);
        var layout = PlotlySerializer.SerializeLayout(spec);
        var html = BuildHtml(traces, layout, format, width, height, scale);

        await page.SetContentAsync(html, new NavigationOptions
        {
            WaitUntil = [WaitUntilNavigation.Networkidle0]
        });

        await page.WaitForFunctionAsync("() => document.title === 'EXPORT_DONE'",
            new WaitForFunctionOptions { Timeout = 15_000 });

        var dataUrl = await page.EvaluateFunctionAsync<string>("() => window.__exportData");
        return Convert.FromBase64String(dataUrl.Split(',')[1]);
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

    private static string BuildHtml(string traces, string layout, string format, int width, int height, int scale)
    {
        return "<!DOCTYPE html><html><head>" +
            "<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>" +
            "</head><body>" +
            $"<div id=\"chart\" style=\"width:{width}px;height:{height}px;\"></div>" +
            "<script>" +
            $"Plotly.newPlot('chart',{traces},{layout},{{responsive:false}}).then(function(gd){{" +
            $"Plotly.toImage(gd,{{format:'{format}',width:{width},height:{height},scale:{scale}}}).then(function(url){{" +
            "document.title='EXPORT_DONE';window.__exportData=url;" +
            "});" +
            "});" +
            "</script></body></html>";
    }
}
