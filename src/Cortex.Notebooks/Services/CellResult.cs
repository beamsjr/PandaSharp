namespace Cortex.Notebooks.Services;

public class CellResult
{
    public string? TextOutput { get; set; }
    public string? HtmlOutput { get; set; }
    public string? Error { get; set; }
    public double ElapsedMs { get; set; }
    public bool IsSuccess { get; set; }
}
