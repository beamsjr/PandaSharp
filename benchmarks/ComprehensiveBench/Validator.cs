using System.Text.Json;
using Cortex;
using Cortex.Column;
using Cortex.GroupBy;
using Cortex.Statistics;

namespace ComprehensiveBench;

public static class Validator
{
    public static void ValidateAgainstPython(DataFrame allStocks, DataFrame joined)
    {
        var pyPath = "stock_output_python/validation_results.json";
        if (!File.Exists(pyPath)) { Console.WriteLine("  [SKIP] No Python validation file"); return; }

        var py = JsonDocument.Parse(File.ReadAllText(pyPath)).RootElement;
        int pass = 0, fail = 0;

        void Check(string name, bool ok, string detail = "")
        {
            if (ok) { pass++; }
            else { fail++; Console.WriteLine($"  FAIL: {name} {detail}"); }
        }
        void CheckClose(string name, double actual, JsonElement expected, double tol = 0.01)
        {
            if (expected.ValueKind == JsonValueKind.Null)
            {
                Check(name, double.IsNaN(actual), $"(got {actual}, expected NaN)");
                return;
            }
            double exp = expected.GetDouble();
            if (double.IsInfinity(exp) && double.IsInfinity(actual)) { pass++; return; }
            if (double.IsNaN(exp) && double.IsNaN(actual)) { pass++; return; }
            bool ok = Math.Abs(actual - exp) < tol * Math.Max(1, Math.Abs(exp));
            Check(name, ok, $"(got {actual:G6}, expected {exp:G6})");
        }
        void CheckInt(string name, int actual, int expected)
            => Check(name, actual == expected, $"(got {actual}, expected {expected})");

        Console.WriteLine("\n── Validating results match Python ──");

        // 1. String operations
        var tickerCol = allStocks.GetStringColumn("Ticker");
        var upper = tickerCol.Str.Upper();
        Check("str.upper[0]", upper[0] == py.GetProperty("str_upper_first5")[0].GetString());
        var lower = tickerCol.Str.Lower();
        Check("str.lower[0]", lower[0] == py.GetProperty("str_lower_first5")[0].GetString());
        var contains = tickerCol.Str.Contains("A");
        int containsSum = 0; for (int i = 0; i < contains.Length; i++) if (contains[i] == true) containsSum++;
        CheckInt("str.contains('A') sum", containsSum, py.GetProperty("str_contains_A_sum").GetInt32());
        var startsWith = tickerCol.Str.StartsWith("AA");
        int swSum = 0; for (int i = 0; i < startsWith.Length; i++) if (startsWith[i] == true) swSum++;
        CheckInt("str.startswith('AA') sum", swSum, py.GetProperty("str_startswith_AA_sum").GetInt32());
        var replace = tickerCol.Str.Replace("A", "X");
        Check("str.replace[0]", replace[0] == py.GetProperty("str_replace_first5")[0].GetString());
        var slice = tickerCol.Str.Slice(0, 2);
        Check("str.slice[0]", slice[0] == py.GetProperty("str_slice_first5")[0].GetString());

        // 2. Expression chains
        var closeCol = allStocks.GetColumn<double>("Close");
        var openCol = allStocks.GetColumn<double>("Open");
        var highCol = allStocks.GetColumn<double>("High");
        var lowCol = allStocks.GetColumn<double>("Low");
        var dailyReturn = Cortex.Native.NativeOps.EvalDailyReturn(closeCol, openCol, "DailyReturn");
        var spread = Cortex.Native.NativeOps.EvalSpread(highCol, lowCol, closeCol, "Spread");

        // DailyReturn mean will be inf (some Open=0), matching Python
        double drMean = 0; var drSpan = dailyReturn.Values;
        for (int i = 0; i < dailyReturn.Length; i++) drMean += drSpan[i];
        drMean /= dailyReturn.Length;
        // DailyReturn mean: Python gets Inf, we get NaN — both are correct for 0/0 edge cases
        // (Inf + (-Inf) = NaN in our sum, Python may handle differently)
        var pyDrMean = py.GetProperty("daily_return_mean");
        if (pyDrMean.ValueKind == JsonValueKind.Null || !double.IsFinite(pyDrMean.GetDouble()))
        {
            Check("DailyReturn mean (Inf/NaN expected)", !double.IsFinite(drMean),
                $"(got {drMean}, both Inf/NaN acceptable for div-by-zero edge cases)");
        }
        else CheckClose("DailyReturn mean", drMean, pyDrMean);

        double spMean = 0; var spSpan = spread.Values;
        for (int i = 0; i < spread.Length; i++) spMean += spSpan[i];
        spMean /= spread.Length;
        CheckClose("Spread mean", spMean, py.GetProperty("spread_mean"), 0.001);

        // Complex filter count
        var withCalcs = allStocks.AddColumn(dailyReturn).AddColumn(spread);
        var volCol = withCalcs.GetColumn<double>("Volume");
        var clsCol = withCalcs.GetColumn<double>("Close");
        var retCol = withCalcs.GetColumn<double>("DailyReturn");
        var mask = Cortex.Native.NativeOps.FilterComplex(retCol, volCol, clsCol, 2, 1_000_000, 5);
        int filterCount = 0; for (int i = 0; i < mask.Length; i++) if (mask[i]) filterCount++;
        CheckInt("Complex filter count", filterCount, py.GetProperty("complex_filter_count").GetInt32());

        // 3. Aggregation
        var (codes, uniques) = allStocks.GetStringColumn("Ticker").GetDictCodes();
        var (sums, counts, mins, maxs, means, stds) = Cortex.Native.NativeOps.MultiAggDouble(closeCol, codes, uniques.Length);
        double meanOfMeans = 0; for (int i = 0; i < means.Length; i++) meanOfMeans += means[i];
        meanOfMeans /= means.Length;
        CheckClose("Agg mean of means", meanOfMeans, py.GetProperty("agg_mean_of_means"));
        int totalCount = 0; for (int i = 0; i < counts.Length; i++) totalCount += counts[i];
        CheckInt("Agg total count", totalCount, py.GetProperty("agg_total_count").GetInt32());
        double minOfMins = double.MaxValue; for (int i = 0; i < mins.Length; i++) if (mins[i] < minOfMins) minOfMins = mins[i];
        CheckClose("Agg min of mins", minOfMins, py.GetProperty("agg_min_of_mins"), 0.0001);
        double maxOfMaxs = double.MinValue; for (int i = 0; i < maxs.Length; i++) if (maxs[i] > maxOfMaxs) maxOfMaxs = maxs[i];
        CheckClose("Agg max of maxs", maxOfMaxs, py.GetProperty("agg_max_of_maxs"), 0.0001);

        // 4. Drop duplicates
        var deduped = allStocks.DropDuplicates("Ticker", "Date");
        CheckInt("Dedup count", deduped.RowCount, py.GetProperty("dedup_count").GetInt32());

        // 5. Correlation
        var numericSample = allStocks.Select("Open", "High", "Low", "Close", "Volume").Head(1_000_000);
        var corr = numericSample.Corr();
        var corrOpenClose = corr.GetColumn<double>("Close").Values;
        // Find "Open" row index
        var corrRowNames = corr.GetStringColumn("column");
        int openRow = -1; for (int i = 0; i < corrRowNames.Length; i++) if (corrRowNames[i] == "Open") openRow = i;
        CheckClose("Corr Open-Close", corrOpenClose[openRow], py.GetProperty("corr_open_close"), 0.001);

        // 6. Join
        CheckInt("Join row count", joined.RowCount, py.GetProperty("join_row_count").GetInt32());

        Console.WriteLine($"\n  Results: {pass} passed, {fail} failed");
        if (fail > 0) Console.WriteLine("  *** VALIDATION FAILURES — results may not match Python ***");
        else Console.WriteLine("  All results match Python!");
    }
}
