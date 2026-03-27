using System.Diagnostics;
using System.Text;

namespace Cortex.IO;

/// <summary>
/// Read/write DataFrames to/from the system clipboard as TSV.
/// Works on macOS (pbcopy/pbpaste), Linux (xclip), and Windows (clip/powershell).
/// </summary>
public static class ClipboardIO
{
    public static DataFrame FromClipboard(CsvReadOptions? options = null)
    {
        var text = GetClipboardText();
        if (string.IsNullOrWhiteSpace(text))
            return new DataFrame();

        options ??= new CsvReadOptions { Delimiter = '\t' };
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(text));
        return CsvReader.Read(stream, options);
    }

    public static void ToClipboard(this DataFrame df, char delimiter = '\t')
    {
        var sb = new StringBuilder();
        sb.AppendLine(string.Join(delimiter, df.ColumnNames));

        for (int r = 0; r < df.RowCount; r++)
        {
            var fields = new string[df.ColumnCount];
            for (int c = 0; c < df.ColumnCount; c++)
            {
                var val = df[df.ColumnNames[c]].GetObject(r);
                fields[c] = val?.ToString() ?? "";
            }
            sb.AppendLine(string.Join(delimiter, fields));
        }

        SetClipboardText(sb.ToString());
    }

    private static string GetClipboardText()
    {
        if (OperatingSystem.IsMacOS())
            return RunProcess("pbpaste");
        if (OperatingSystem.IsLinux())
            return RunProcess("xclip", "-selection clipboard -o");
        if (OperatingSystem.IsWindows())
            return RunProcess("powershell", "-command Get-Clipboard");
        throw new PlatformNotSupportedException("Clipboard not supported on this platform.");
    }

    private static void SetClipboardText(string text)
    {
        if (OperatingSystem.IsMacOS())
            RunProcessWithInput("pbcopy", "", text);
        else if (OperatingSystem.IsLinux())
            RunProcessWithInput("xclip", "-selection clipboard", text);
        else if (OperatingSystem.IsWindows())
            RunProcessWithInput("clip", "", text);
        else
            throw new PlatformNotSupportedException("Clipboard not supported on this platform.");
    }

    private static string RunProcess(string command, string args = "")
    {
        var psi = new ProcessStartInfo(command, args)
        {
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        using var proc = Process.Start(psi)!;
        var output = proc.StandardOutput.ReadToEnd();
        proc.WaitForExit();
        return output;
    }

    private static void RunProcessWithInput(string command, string args, string input)
    {
        var psi = new ProcessStartInfo(command, args)
        {
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        using var proc = Process.Start(psi)!;
        proc.StandardInput.Write(input);
        proc.StandardInput.Close();
        proc.WaitForExit();
    }
}
