namespace Cortex.Notebooks.Models;

public enum CellType { Code, Markdown }
public enum CellStatus { Idle, Running, Success, Error }

public class NotebookCell
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N")[..8];
    public CellType Type { get; set; } = CellType.Code;
    public string Source { get; set; } = "";
    public string? Output { get; set; }
    public string? HtmlOutput { get; set; }
    public string? Error { get; set; }
    public CellStatus Status { get; set; } = CellStatus.Idle;
    public int ExecutionCount { get; set; }
    public double ElapsedMs { get; set; }
}
