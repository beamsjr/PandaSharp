using System.Diagnostics;
using System.Reflection;
using System.Text;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;
using Cortex;
using Cortex.Viz;

namespace Cortex.Notebooks.Services;

/// <summary>
/// Roslyn-based C# scripting kernel that maintains state across cell executions.
/// Each user session (SignalR connection) gets its own scoped instance.
/// </summary>
public class NotebookKernel
{
    private ScriptState<object>? _state;
    private ScriptOptions _options;
    private int _executionCount;

    private static readonly string[] DefaultImports =
    [
        "System",
        "System.Linq",
        "System.Collections.Generic",
        "System.Text",
        "System.IO",
        "System.Threading.Tasks",
        "Cortex",
        "Cortex.Column",
        "Cortex.IO",
        "Cortex.Viz",
        "Cortex.ML.Models",
        "Cortex.ML.Transformers",
        "Cortex.ML.Splitting",
        "Cortex.ML.Metrics",
        "Cortex.Text",
        "Cortex.TimeSeries"
    ];

    public NotebookKernel()
    {
        _options = BuildScriptOptions();
    }

    private static ScriptOptions BuildScriptOptions()
    {
        var options = ScriptOptions.Default
            .WithLanguageVersion(Microsoft.CodeAnalysis.CSharp.LanguageVersion.Latest);

        // Add references to all Cortex assemblies and common framework assemblies
        var assemblies = new List<Assembly>
        {
            typeof(DataFrame).Assembly,          // Cortex
            typeof(VizBuilder).Assembly,         // Cortex.Viz
        };

        // Try to add optional assemblies that may be loaded
        TryAddAssembly(assemblies, "Cortex.ML");
        TryAddAssembly(assemblies, "Cortex.Text");
        TryAddAssembly(assemblies, "Cortex.TimeSeries");

        // Add core framework assemblies needed for scripting
        assemblies.Add(typeof(object).Assembly);
        assemblies.Add(typeof(Enumerable).Assembly);
        assemblies.Add(typeof(List<>).Assembly);
        assemblies.Add(typeof(Console).Assembly);

        // Add netstandard and runtime assemblies for full BCL access
        var runtimeDir = Path.GetDirectoryName(typeof(object).Assembly.Location)!;
        var netstandard = Path.Combine(runtimeDir, "netstandard.dll");
        if (File.Exists(netstandard))
            options = options.AddReferences(MetadataReference(netstandard));

        var runtimeDll = Path.Combine(runtimeDir, "System.Runtime.dll");
        if (File.Exists(runtimeDll))
            options = options.AddReferences(MetadataReference(runtimeDll));

        options = options
            .AddReferences(assemblies.Distinct().ToArray())
            .AddImports(DefaultImports.Where(ns => IsNamespaceAvailable(ns, assemblies)));

        return options;
    }

    private static Microsoft.CodeAnalysis.MetadataReference MetadataReference(string path) =>
        Microsoft.CodeAnalysis.MetadataReference.CreateFromFile(path);

    private static void TryAddAssembly(List<Assembly> assemblies, string name)
    {
        try
        {
            var asm = AppDomain.CurrentDomain.GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == name);
            if (asm != null)
                assemblies.Add(asm);
            else
            {
                asm = Assembly.Load(name);
                assemblies.Add(asm);
            }
        }
        catch
        {
            // Assembly not available — skip silently
        }
    }

    private static bool IsNamespaceAvailable(string ns, List<Assembly> assemblies)
    {
        return assemblies.Any(a =>
        {
            try { return a.GetTypes().Any(t => t.Namespace == ns); }
            catch { return false; }
        });
    }

    public int ExecutionCount => _executionCount;

    /// <summary>
    /// Execute a code cell and return the result. Variables persist across calls.
    /// </summary>
    public async Task<CellResult> ExecuteAsync(string code, CancellationToken ct = default)
    {
        _executionCount++;
        var sw = Stopwatch.StartNew();
        var consoleCapture = new StringWriter();
        var originalOut = Console.Out;

        try
        {
            // Redirect Console.Out to capture print statements
            Console.SetOut(consoleCapture);

            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(TimeSpan.FromSeconds(30));

            if (_state == null)
            {
                var script = CSharpScript.Create<object>(code, _options);
                _state = await script.RunAsync(cancellationToken: cts.Token);
            }
            else
            {
                _state = await _state.ContinueWithAsync<object>(code, _options, cancellationToken: cts.Token);
            }

            sw.Stop();
            Console.SetOut(originalOut);

            var textOutput = consoleCapture.ToString();
            string? htmlOutput = null;
            var returnValue = _state.ReturnValue;

            // Detect DataFrame return value and render as HTML table
            if (returnValue is DataFrame df)
            {
                htmlOutput = RenderDataFrame(df);
            }
            // Detect VizBuilder and render as inline HTML
            else if (returnValue is VizBuilder viz)
            {
                htmlOutput = viz.ToNotebookHtml();
            }
            // Detect FacetGridBuilder
            else if (returnValue is Cortex.Viz.Charts.FacetGridBuilder facet)
            {
                htmlOutput = facet.ToNotebookHtml();
            }
            // Detect SubplotBuilder
            else if (returnValue is Cortex.Viz.Charts.SubplotBuilder subplot)
            {
                htmlOutput = subplot.ToHtmlString();
            }
            // If return value is not null and not void, append its string representation
            else if (returnValue != null)
            {
                var repr = returnValue.ToString();
                if (!string.IsNullOrEmpty(repr))
                {
                    textOutput = string.IsNullOrEmpty(textOutput)
                        ? repr
                        : textOutput + "\n" + repr;
                }
            }

            return new CellResult
            {
                TextOutput = string.IsNullOrEmpty(textOutput) ? null : textOutput,
                HtmlOutput = htmlOutput,
                ElapsedMs = sw.Elapsed.TotalMilliseconds,
                IsSuccess = true
            };
        }
        catch (CompilationErrorException ex)
        {
            sw.Stop();
            Console.SetOut(originalOut);
            return new CellResult
            {
                Error = string.Join("\n", ex.Diagnostics.Select(d => d.ToString())),
                ElapsedMs = sw.Elapsed.TotalMilliseconds,
                IsSuccess = false
            };
        }
        catch (OperationCanceledException)
        {
            sw.Stop();
            Console.SetOut(originalOut);
            return new CellResult
            {
                Error = "Execution timed out after 30 seconds.",
                ElapsedMs = sw.Elapsed.TotalMilliseconds,
                IsSuccess = false
            };
        }
        catch (Exception ex)
        {
            sw.Stop();
            Console.SetOut(originalOut);
            return new CellResult
            {
                Error = FormatException(ex),
                ElapsedMs = sw.Elapsed.TotalMilliseconds,
                IsSuccess = false
            };
        }
    }

    /// <summary>
    /// Reset all kernel state. Variables, imports, and execution count are cleared.
    /// </summary>
    public void Reset()
    {
        _state = null;
        _executionCount = 0;
        _options = BuildScriptOptions();
    }

    private static string RenderDataFrame(DataFrame df)
    {
        // Use the DataFrame's built-in ToHtml which renders first 50 rows
        // with headers, truncation notice, and shape info
        return df.ToHtml(maxRows: 50);
    }

    private static string FormatException(Exception ex)
    {
        var sb = new StringBuilder();
        var current = ex is AggregateException agg && agg.InnerExceptions.Count == 1
            ? agg.InnerException!
            : ex;

        sb.AppendLine($"{current.GetType().Name}: {current.Message}");
        if (current.StackTrace != null)
        {
            // Filter out Roslyn internals from stack trace for cleaner output
            foreach (var line in current.StackTrace.Split('\n'))
            {
                if (line.Contains("Microsoft.CodeAnalysis") || line.Contains("System.Runtime"))
                    continue;
                sb.AppendLine(line.TrimEnd());
            }
        }

        return sb.ToString().TrimEnd();
    }
}
