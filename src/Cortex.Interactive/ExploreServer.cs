using System.Diagnostics;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Cortex.Column;

namespace Cortex.Interactive;

/// <summary>
/// Extension method to launch an interactive data explorer for a DataFrame in the browser.
/// </summary>
public static class ExploreExtensions
{
    /// <summary>
    /// Open an interactive web UI to explore the DataFrame.
    /// Starts a local HTTP server and opens the default browser.
    /// </summary>
    /// <param name="df">The DataFrame to explore.</param>
    /// <param name="port">Port to listen on. 0 = auto-assign.</param>
    /// <param name="openBrowser">Whether to open the default browser automatically.</param>
    public static void Explore(this DataFrame df, int port = 0, bool openBrowser = true)
    {
        var server = new ExploreServer(df, port);
        var url = server.Start();

        Console.WriteLine($"Explore server running at {url}");
        Console.WriteLine("Press Enter to stop...");

        if (openBrowser)
            OpenBrowser(url);

        Console.ReadLine();
        server.Stop();
    }

    private static void OpenBrowser(string url)
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                Process.Start("open", url);
            else
                Process.Start("xdg-open", url);
        }
        catch
        {
            // Silently ignore if browser can't be opened
        }
    }
}

/// <summary>
/// Minimal HTTP server that serves an interactive DataFrame explorer UI.
/// Uses HttpListener -- no Kestrel or other heavy dependencies.
/// </summary>
public class ExploreServer
{
    private readonly DataFrame _df;
    private readonly HttpListener _listener;
    private readonly int _port;
    private CancellationTokenSource? _cts;

    public ExploreServer(DataFrame df, int port)
    {
        _df = df;
        // If port is 0, pick an available port by trying a range
        _port = port == 0 ? FindAvailablePort() : port;
        _listener = new HttpListener();
        _listener.Prefixes.Add($"http://localhost:{_port}/");
    }

    public string Start()
    {
        _cts = new CancellationTokenSource();
        _listener.Start();
        _ = Task.Run(ListenLoop);
        return $"http://localhost:{_port}/";
    }

    public void Stop()
    {
        _cts?.Cancel();
        _listener.Stop();
    }

    private static int FindAvailablePort()
    {
        var listener = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private async Task ListenLoop()
    {
        while (_listener.IsListening && _cts is { IsCancellationRequested: false })
        {
            try
            {
                var ctx = await _listener.GetContextAsync().ConfigureAwait(false);
                _ = Task.Run(() => HandleRequest(ctx));
            }
            catch (HttpListenerException) { break; }
            catch (ObjectDisposedException) { break; }
        }
    }

    private void HandleRequest(HttpListenerContext ctx)
    {
        try
        {
            var path = ctx.Request.Url?.AbsolutePath ?? "/";

            if (path == "/" || path == "/index.html")
                ServeHtml(ctx);
            else if (path == "/api/data")
                ServeData(ctx);
            else if (path == "/api/stats")
                ServeStats(ctx);
            else
            {
                ctx.Response.StatusCode = 404;
                ctx.Response.Close();
            }
        }
        catch
        {
            try { ctx.Response.StatusCode = 500; ctx.Response.Close(); } catch { }
        }
    }

    private void ServeHtml(HttpListenerContext ctx)
    {
        var html = GenerateHtml();
        var bytes = Encoding.UTF8.GetBytes(html);
        ctx.Response.ContentType = "text/html; charset=utf-8";
        ctx.Response.ContentLength64 = bytes.Length;
        ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
        ctx.Response.Close();
    }

    private void ServeData(HttpListenerContext ctx)
    {
        var query = ctx.Request.QueryString;
        int page = int.TryParse(query["page"], out var p) ? p : 0;
        int pageSize = int.TryParse(query["pageSize"], out var ps) ? Math.Clamp(ps, 1, 1000) : 50;
        string? sortCol = query["sort"];
        string? sortDir = query["dir"]; // "asc" or "desc"
        string? filter = query["filter"];

        // Build row indices
        var indices = Enumerable.Range(0, _df.RowCount).ToList();

        // Filter
        if (!string.IsNullOrWhiteSpace(filter))
        {
            var lowerFilter = filter.ToLowerInvariant();
            indices = indices.Where(row =>
            {
                for (int c = 0; c < _df.ColumnCount; c++)
                {
                    var col = _df[_df.ColumnNames[c]];
                    if (!col.IsNull(row))
                    {
                        var val = col.GetObject(row)?.ToString();
                        if (val != null && val.ToLowerInvariant().Contains(lowerFilter))
                            return true;
                    }
                }
                return false;
            }).ToList();
        }

        // Sort
        if (!string.IsNullOrWhiteSpace(sortCol) && _df.ColumnNames.Contains(sortCol))
        {
            var col = _df[sortCol];
            bool desc = string.Equals(sortDir, "desc", StringComparison.OrdinalIgnoreCase);
            indices.Sort((a, b) =>
            {
                var va = col.GetObject(a);
                var vb = col.GetObject(b);
                if (va == null && vb == null) return 0;
                if (va == null) return desc ? 1 : -1;
                if (vb == null) return desc ? -1 : 1;
                int cmp = Comparer<object>.Default.Compare(va, vb);
                return desc ? -cmp : cmp;
            });
        }

        int totalFiltered = indices.Count;
        var pageIndices = indices.Skip(page * pageSize).Take(pageSize).ToList();

        // Serialize with Utf8JsonWriter
        using var ms = new MemoryStream();
        using (var writer = new Utf8JsonWriter(ms))
        {
            writer.WriteStartObject();
            writer.WriteNumber("total", totalFiltered);
            writer.WriteNumber("page", page);
            writer.WriteNumber("pageSize", pageSize);

            writer.WriteStartArray("columns");
            foreach (var name in _df.ColumnNames)
                writer.WriteStringValue(name);
            writer.WriteEndArray();

            writer.WriteStartArray("rows");
            foreach (var rowIdx in pageIndices)
            {
                writer.WriteStartArray();
                for (int c = 0; c < _df.ColumnCount; c++)
                {
                    var col = _df[_df.ColumnNames[c]];
                    if (col.IsNull(rowIdx))
                    {
                        writer.WriteNullValue();
                        continue;
                    }
                    var val = col.GetObject(rowIdx);
                    switch (val)
                    {
                        case int i: writer.WriteNumberValue(i); break;
                        case long l: writer.WriteNumberValue(l); break;
                        case double d: writer.WriteNumberValue(d); break;
                        case float f: writer.WriteNumberValue(f); break;
                        case bool b: writer.WriteBooleanValue(b); break;
                        case DateTime dt: writer.WriteStringValue(dt.ToString("O")); break;
                        default: writer.WriteStringValue(val?.ToString() ?? ""); break;
                    }
                }
                writer.WriteEndArray();
            }
            writer.WriteEndArray();

            writer.WriteEndObject();
        }

        var bytes = ms.ToArray();
        ctx.Response.ContentType = "application/json; charset=utf-8";
        ctx.Response.ContentLength64 = bytes.Length;
        ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
        ctx.Response.Close();
    }

    private void ServeStats(HttpListenerContext ctx)
    {
        using var ms = new MemoryStream();
        using (var writer = new Utf8JsonWriter(ms))
        {
            writer.WriteStartObject();
            foreach (var name in _df.ColumnNames)
            {
                var col = _df[name];
                writer.WriteStartObject(name);
                writer.WriteString("type", FormatDtype(col.DataType));
                writer.WriteNumber("count", col.Length);
                writer.WriteNumber("nulls", col.NullCount);

                if (IsNumeric(col.DataType))
                {
                    double sum = 0, min = double.MaxValue, max = double.MinValue;
                    int validCount = 0;
                    for (int i = 0; i < col.Length; i++)
                    {
                        if (col.IsNull(i)) continue;
                        double v = Convert.ToDouble(col.GetObject(i));
                        sum += v;
                        if (v < min) min = v;
                        if (v > max) max = v;
                        validCount++;
                    }
                    if (validCount > 0)
                    {
                        writer.WriteNumber("mean", Math.Round(sum / validCount, 4));
                        writer.WriteNumber("min", min);
                        writer.WriteNumber("max", max);
                    }
                }
                writer.WriteEndObject();
            }
            writer.WriteEndObject();
        }

        var bytes = ms.ToArray();
        ctx.Response.ContentType = "application/json; charset=utf-8";
        ctx.Response.ContentLength64 = bytes.Length;
        ctx.Response.OutputStream.Write(bytes, 0, bytes.Length);
        ctx.Response.Close();
    }

    internal string GenerateHtml()
    {
        var colNamesJson = JsonSerializer.Serialize(_df.ColumnNames.ToArray());
        var sb = new StringBuilder();
        sb.Append("<!DOCTYPE html><html lang=\"en\"><head>");
        sb.Append("<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
        sb.Append("<title>Cortex Explorer</title>");
        sb.Append("<style>");
        sb.Append("*{margin:0;padding:0;box-sizing:border-box}");
        sb.Append("body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#f5f6fa;color:#2d3436}");
        sb.Append("#header{background:#0984e3;color:#fff;padding:12px 24px;display:flex;align-items:center;gap:16px}");
        sb.Append("#header h1{font-size:18px;font-weight:600}");
        sb.Append("#header .info{font-size:13px;opacity:.85}");
        sb.Append("#toolbar{padding:12px 24px;background:#fff;border-bottom:1px solid #dfe6e9;display:flex;gap:12px;align-items:center}");
        sb.Append("#search{padding:8px 12px;border:1px solid #b2bec3;border-radius:6px;font-size:14px;width:300px;outline:none}");
        sb.Append("#search:focus{border-color:#0984e3;box-shadow:0 0 0 2px rgba(9,132,227,.2)}");
        sb.Append("#status{font-size:13px;color:#636e72}");
        sb.Append("#table-wrap{padding:0 24px;overflow:auto;max-height:calc(100vh - 160px)}");
        sb.Append("table{width:100%;border-collapse:collapse;font-size:13px}");
        sb.Append("thead th{position:sticky;top:0;background:#dfe6e9;padding:8px 12px;text-align:left;cursor:pointer;");
        sb.Append("user-select:none;white-space:nowrap;border-bottom:2px solid #b2bec3;font-weight:600}");
        sb.Append("thead th:hover{background:#c8d6e5}");
        sb.Append("thead th .dtype{font-weight:400;color:#636e72;font-size:11px}");
        sb.Append("thead th .stat{font-weight:400;color:#636e72;font-size:10px;display:block}");
        sb.Append("thead th .sort-arrow{margin-left:4px;font-size:10px}");
        sb.Append("tbody td{padding:6px 12px;border-bottom:1px solid #eee;white-space:nowrap}");
        sb.Append("tbody tr:hover{background:#dfe6e9}");
        sb.Append("tbody tr:nth-child(even){background:#f8f9fa}");
        sb.Append("td.null{color:#b2bec3;font-style:italic}");
        sb.Append("td.num{text-align:right;font-variant-numeric:tabular-nums}");
        sb.Append("#pager{padding:12px 24px;display:flex;gap:8px;align-items:center;background:#fff;border-top:1px solid #dfe6e9}");
        sb.Append("button{padding:6px 14px;border:1px solid #b2bec3;border-radius:4px;background:#fff;cursor:pointer;font-size:13px}");
        sb.Append("button:hover{background:#f0f0f0}");
        sb.Append("button:disabled{opacity:.4;cursor:default}");
        sb.Append("</style></head><body>");
        sb.Append("<div id=\"header\">");
        sb.Append("<h1>Cortex Explorer</h1>");
        sb.Append($"<span class=\"info\">{_df.RowCount:N0} rows x {_df.ColumnCount} columns</span>");
        sb.Append("</div>");
        sb.Append("<div id=\"toolbar\">");
        sb.Append("<input id=\"search\" type=\"text\" placeholder=\"Filter rows...\">");
        sb.Append("<span id=\"status\"></span>");
        sb.Append("</div>");
        sb.Append("<div id=\"table-wrap\"><table><thead id=\"thead\"><tr></tr></thead><tbody id=\"tbody\"></tbody></table></div>");
        sb.Append("<div id=\"pager\">");
        sb.Append("<button id=\"prev\" onclick=\"go(-1)\">Prev</button>");
        sb.Append("<span id=\"pageInfo\"></span>");
        sb.Append("<button id=\"next\" onclick=\"go(1)\">Next</button>");
        sb.Append("</div>");

        // JavaScript as a plain (non-interpolated) string to avoid escaping issues
        sb.Append("<script>");
        sb.Append("const COLS=").Append(colNamesJson).Append(';');
        sb.Append(JsCode);
        sb.Append("</script></body></html>");
        return sb.ToString();
    }

    // Using verbatim string (@"") for the JS code.
    // In verbatim strings, the only escape is "" for a literal double-quote.
    // The JS avoids template literals and uses string concatenation instead,
    // so no backticks or ${} that could confuse the C# parser.
    private const string JsCode = @"
let page=0, pageSize=50, total=0, sortCol=null, sortDir='asc', stats={}, filterTimer=null;

async function loadStats(){
  try{const r=await fetch('/api/stats');stats=await r.json()}catch(e){}
  renderHeader();
}

function renderHeader(){
  const tr=document.querySelector('#thead tr');
  tr.innerHTML='<th style=""width:50px"">#</th>'+COLS.map(c=>{
    let arrow='';
    if(sortCol===c) arrow='<span class=""sort-arrow"">'+(sortDir==='asc'?'\u25B2':'\u25BC')+'</span>';
    let dtype=stats[c]?'<span class=""dtype"">'+stats[c].type+'</span>':'';
    let st='';
    if(stats[c]&&stats[c].mean!==undefined)
      st='<span class=""stat"">mean:'+stats[c].mean+' min:'+stats[c].min+' max:'+stats[c].max+'</span>';
    return '<th data-col=""'+c+'"" onclick=""doSort(\''+c+'\')"">' + c + ' ' + arrow + '<br>' + dtype + st + '</th>';
  }).join('');
}

async function load(){
  var url='/api/data?page='+page+'&pageSize='+pageSize;
  if(sortCol) url+='&sort='+encodeURIComponent(sortCol)+'&dir='+sortDir;
  var f=document.getElementById('search').value;
  if(f) url+='&filter='+encodeURIComponent(f);
  var r=await fetch(url);var d=await r.json();
  total=d.total;
  var tbody=document.getElementById('tbody');
  var html='';
  for(var ri=0;ri<d.rows.length;ri++){
    var row=d.rows[ri];
    var idx=page*pageSize+ri;
    var cells='';
    for(var ci=0;ci<row.length;ci++){
      var v=row[ci];
      if(v===null){cells+='<td class=""null"">null</td>';continue;}
      var isNum=typeof v==='number';
      cells+='<td'+(isNum?' class=""num""':'')+'>'+escHtml(String(v))+'</td>';
    }
    html+='<tr><td style=""color:#636e72;font-weight:500"">'+idx+'</td>'+cells+'</tr>';
  }
  tbody.innerHTML=html;
  var totalPages=Math.ceil(total/pageSize);
  document.getElementById('pageInfo').textContent='Page '+(page+1)+' of '+(totalPages||1);
  document.getElementById('prev').disabled=page===0;
  document.getElementById('next').disabled=page>=totalPages-1;
  document.getElementById('status').textContent=total+' rows';
}

function doSort(col){
  if(sortCol===col) sortDir=sortDir==='asc'?'desc':'asc';
  else{ sortCol=col; sortDir='asc'; }
  page=0; renderHeader(); load();
}

function go(d){ page+=d; load(); }

function escHtml(s){
  var d=document.createElement('div');
  d.appendChild(document.createTextNode(s));
  return d.innerHTML;
}

document.getElementById('search').addEventListener('input',function(){
  clearTimeout(filterTimer);
  filterTimer=setTimeout(function(){ page=0; load(); }, 300);
});

loadStats(); load();
";

    private static string FormatDtype(Type type) => type switch
    {
        _ when type == typeof(int) => "int32",
        _ when type == typeof(long) => "int64",
        _ when type == typeof(double) => "float64",
        _ when type == typeof(float) => "float32",
        _ when type == typeof(bool) => "bool",
        _ when type == typeof(DateTime) => "datetime",
        _ when type == typeof(string) => "string",
        _ => type.Name
    };

    private static bool IsNumeric(Type type) =>
        type == typeof(int) || type == typeof(long) ||
        type == typeof(double) || type == typeof(float) ||
        type == typeof(decimal);
}
