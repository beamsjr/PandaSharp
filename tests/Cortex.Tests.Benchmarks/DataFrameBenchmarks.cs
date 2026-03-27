using BenchmarkDotNet.Attributes;
using Cortex;
using Cortex.Column;
using Cortex.GroupBy;
using Cortex.IO;
using Cortex.Joins;
using Cortex.Reshape;
using Cortex.Window;
using Cortex.ParallelOps;
using Cortex.Storage;
using Cortex.Statistics;
using Cortex.Lazy;

namespace Cortex.Tests.Benchmarks;

[MemoryDiagnoser]
[SimpleJob(warmupCount: 3, iterationCount: 5)]
public class DataFrameBenchmarks
{
    private DataFrame _df = null!;
    private DataFrame _dfSmall = null!;
    private DataFrame _right = null!;
    private byte[] _csvBytes = null!;

    [GlobalSetup]
    public void Setup()
    {
        int n = 100_000;
        var rng = new Random(42);

        var ids = new int[n];
        var values = new double[n];
        var categories = new string?[n];
        var cats = new[] { "Alpha", "Beta", "Gamma", "Delta", "Epsilon" };

        for (int i = 0; i < n; i++)
        {
            ids[i] = i;
            values[i] = rng.NextDouble() * 1000;
            categories[i] = cats[rng.Next(cats.Length)];
        }

        _df = new DataFrame(
            new Column<int>("Id", ids),
            new Column<double>("Value", values),
            new StringColumn("Category", categories)
        );

        _dfSmall = _df.Head(10_000);

        // Right side for join
        var rightIds = new int[1000];
        var rightLabels = new string?[1000];
        for (int i = 0; i < 1000; i++)
        {
            rightIds[i] = i * 100;
            rightLabels[i] = $"Label_{i}";
        }
        _right = new DataFrame(
            new Column<int>("Id", rightIds),
            new StringColumn("Label", rightLabels)
        );

        // CSV data for parsing benchmark
        var sb = new System.Text.StringBuilder("Id,Value,Category\n");
        var rng2 = new Random(42);
        var cats2 = new[] { "Alpha", "Beta", "Gamma" };
        for (int i = 0; i < 100_000; i++)
            sb.AppendLine($"{i},{rng2.NextDouble() * 1000:F2},{cats2[rng2.Next(3)]}");
        _csvBytes = System.Text.Encoding.UTF8.GetBytes(sb.ToString());

        // Parquet file for read benchmark
        _parquetPath = Path.Combine(Path.GetTempPath(), $"pandasharp_bench_{Guid.NewGuid():N}.parquet");
        ParquetIO.WriteParquet(_df, _parquetPath);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        if (File.Exists(_parquetPath)) File.Delete(_parquetPath);
    }

    [Benchmark]
    public DataFrame Head() => _df.Head(100);

    [Benchmark]
    public DataFrame Tail() => _df.Tail(100);

    [Benchmark]
    public DataFrame FilterMask()
    {
        var mask = _df.GetColumn<double>("Value").Gt(500.0);
        return _df.Filter(mask);
    }

    [Benchmark]
    public DataFrame FilterLambda() =>
        _df.Filter(row => (double)row["Value"]! > 500.0);

    [Benchmark]
    public DataFrame Sort() => _dfSmall.Sort("Value");

    [Benchmark]
    public DataFrame GroupBySum() => _df.GroupBy("Category").Sum();

    [Benchmark]
    public DataFrame GroupByMean() => _df.GroupBy("Category").Mean();

    [Benchmark]
    public DataFrame JoinInner() => _dfSmall.Join(_right, "Id");

    [Benchmark]
    public DataFrame Select() => _df.Select("Id", "Value");

    [Benchmark]
    public DataFrame DropDuplicates() => _dfSmall.DropDuplicates("Category");

    [Benchmark]
    public double? Sum() => _df.GetColumn<double>("Value").Sum();

    [Benchmark]
    public double? Mean() => _df.GetColumn<double>("Value").Mean();

    [Benchmark]
    public double? Median() => _df.GetColumn<double>("Value").Median();

    [Benchmark]
    public double? Std() => _df.GetColumn<double>("Value").Std();

    // -- Hot path benchmarks for optimization targets --

    [Benchmark]
    public DataFrame SortLarge() => _df.Sort("Value"); // Bug target #1: boxing per comparison

    [Benchmark]
    public Column<double> RollingMean100K() => _df.GetColumn<double>("Value").Rolling(10).Mean(); // #4

    [Benchmark]
    public Column<double> ExpandingMean100K() => _df.GetColumn<double>("Value").Expanding().Mean(); // #3

    [Benchmark]
    public Column<double> ArithmeticAdd() // #5: triple allocation
    {
        var a = _df.GetColumn<double>("Value");
        return a.Add(a);
    }

    [Benchmark]
    public Column<double> ArithmeticMultiply()
    {
        var a = _df.GetColumn<double>("Value");
        return a.Multiply(2.0);
    }

    [Benchmark]
    public DataFrame JoinLarger() // #8: join column building
    {
        // 100K left × 1K right
        return _df.Join(_right, "Id");
    }

    [Benchmark]
    public DataFrame GetDummies100K() => _df.GetDummies("Category"); // #13

    // -- CSV parsing --

    [Benchmark]
    public DataFrame CsvParse()
    {
        using var ms = new MemoryStream(_csvBytes);
        return CsvReader.Read(ms);
    }

    [Benchmark]
    public DataFrame CsvParseWithSchema()
    {
        using var ms = new MemoryStream(_csvBytes);
        return CsvReader.Read(ms, new CsvReadOptions
        {
            Schema = [("Id", typeof(int)), ("Value", typeof(double)), ("Category", typeof(string))]
        });
    }

    // -- Parquet I/O --
    private string _parquetPath = null!;

    [Benchmark]
    public DataFrame ParquetRead() => ParquetIO.ReadParquet(_parquetPath);

    // -- Pipeline benchmark: load → filter → groupby → agg --

    [Benchmark]
    public DataFrame Pipeline()
    {
        // Filter Value > 500, GroupBy Category, Sum Value
        var mask = _df.GetColumn<double>("Value").Gt(500.0);
        var filtered = _df.Filter(mask);
        return filtered.GroupBy("Category").Sum();
    }

    // -- String operations on 100K rows --
    [Benchmark]
    public Column<bool> StringContains() =>
        _df.GetStringColumn("Category").Str.Contains("Alpha");

    [Benchmark]
    public StringColumn StringUpper() =>
        _df.GetStringColumn("Category").Str.Upper();

    [Benchmark]
    public Column<int> StringLen() =>
        _df.GetStringColumn("Category").Str.Len();

    // -- Indexing operations --

    [Benchmark]
    public object? AtAccess()
    {
        // Access 1000 random scalar values via At
        object? last = null;
        for (int i = 0; i < 1000; i++)
            last = _df.At[i * 100, "Value"];
        return last;
    }

    [Benchmark]
    public object? IAtAccess()
    {
        object? last = null;
        for (int i = 0; i < 1000; i++)
            last = _df.IAt[i * 100, 1]; // column 1 = Value
        return last;
    }

    [Benchmark]
    public DataFrame XsCrossSection() => _df.Xs("Category", "Alpha");

    [Benchmark]
    public DataFrame SetIndexResetIndex()
    {
        var indexed = _df.SetIndex("Category");
        return indexed.ResetIndex();
    }

    [Benchmark]
    public DataFrame MultiIndexSetReset()
    {
        var indexed = _dfSmall.SetIndex("Id", "Category");
        return indexed.ResetIndex();
    }

    // -- SQL Generation --

    private Cortex.IO.Database.SqlGenerator _sqlGen = null!;
    private Cortex.Lazy.LogicalPlan _sqlPlan = null!;

    [GlobalSetup(Target = nameof(SqlGenerate_FullChain))]
    public void SetupSql()
    {
        Setup();
        _sqlGen = new Cortex.IO.Database.SqlGenerator(new Cortex.IO.Database.PostgresDialect());
        // Build a realistic plan: Filter → Select → Sort → Head
        var scan = new Cortex.Lazy.ScanPlan(_df);
        var filter = new Cortex.Lazy.FilterPlan(scan,
            Expressions.Expr.Col("Value") > Expressions.Expr.Lit(500));
        var select = new Cortex.Lazy.SelectPlan(filter, ["Id", "Value", "Category"]);
        var sort = new Cortex.Lazy.SortPlan(select, "Value", false);
        _sqlPlan = new Cortex.Lazy.HeadPlan(sort, 100);
    }

    [Benchmark]
    public string SqlGenerate_FullChain()
    {
        var (sql, _) = _sqlGen.Generate(_sqlPlan, "orders");
        return sql;
    }

    // -- Parallel vs Sequential --

    [Benchmark]
    public DataFrame FilterLambda_Parallel() =>
        _df.ParallelFilter(row => (double)row["Value"]! > 500.0);

    [Benchmark]
    public Column<double> ArithmeticAdd_Parallel()
    {
        var a = _df.GetColumn<double>("Value");
        return a.ParallelAdd(a);
    }

    [Benchmark]
    public Column<double> ArithmeticMultiply_Parallel()
    {
        var a = _df.GetColumn<double>("Value");
        return a.ParallelMultiply(2.0);
    }

    [Benchmark]
    public double Sum_Parallel() => _df.GetColumn<double>("Value").ParallelSum();

    // -- Geospatial --

    private Cortex.Geo.GeoColumn _geoCol = null!;
    private Cortex.Geo.GeoColumn _geoSmall = null!;

    [GlobalSetup(Targets = [nameof(GeoDistanceTo), nameof(GeoWithinDistance)])]
    public void SetupGeo()
    {
        Setup();
        var rng2 = new Random(42);
        int n = 10_000;
        var lats = new double[n];
        var lons = new double[n];
        for (int i = 0; i < n; i++)
        {
            lats[i] = 25 + rng2.NextDouble() * 25; // US lat range
            lons[i] = -125 + rng2.NextDouble() * 55; // US lon range
        }
        _geoCol = new Cortex.Geo.GeoColumn("loc", lats, lons);

        var slats = new double[100];
        var slons = new double[100];
        for (int i = 0; i < 100; i++)
        {
            slats[i] = 30 + rng2.NextDouble() * 15;
            slons[i] = -120 + rng2.NextDouble() * 50;
        }
        _geoSmall = new Cortex.Geo.GeoColumn("targets", slats, slons);
    }

    [Benchmark]
    public Cortex.Column.Column<double> GeoDistanceTo()
    {
        return _geoCol.DistanceTo(new Cortex.Geo.GeoPoint(40.7128, -74.0060));
    }

    [Benchmark]
    public bool[] GeoWithinDistance()
    {
        return _geoCol.WithinDistance(new Cortex.Geo.GeoPoint(40.7128, -74.0060), 500);
    }

    // -- R-tree spatial index --

    private Cortex.Geo.RTree _rtree = null!;

    [GlobalSetup(Targets = [nameof(RTreeNearest), nameof(RTreeQueryRadius)])]
    public void SetupRTree()
    {
        SetupGeo();
        _rtree = Cortex.Geo.RTree.Build(_geoCol);
    }

    [Benchmark]
    public (int, double) RTreeNearest()
    {
        return _rtree.Nearest(new Cortex.Geo.GeoPoint(40.7128, -74.0060));
    }

    [Benchmark]
    public List<(int, double)> RTreeQueryRadius()
    {
        return _rtree.QueryRadius(new Cortex.Geo.GeoPoint(40.7128, -74.0060), 500);
    }

    // -- Out-of-core Spill --

    private string _spillPath = null!;

    [GlobalSetup(Target = nameof(SpillToDisk))]
    public void SetupSpill()
    {
        Setup();
        _spillPath = Path.Combine(Path.GetTempPath(), $"bench_spill_{Guid.NewGuid():N}.arrow");
    }

    [GlobalCleanup(Target = nameof(SpillToDisk))]
    public void CleanupSpill()
    {
        if (File.Exists(_spillPath)) File.Delete(_spillPath);
        Cleanup();
    }

    [Benchmark]
    public void SpillToDisk()
    {
        using var spilled = _df.Spill(_spillPath);
        // Access one column to verify lazy load works
        _ = spilled["Value"];
    }

    // -- Partitioned parallel GroupBy --

    [Benchmark]
    public DataFrame PartitionedGroupBySum()
    {
        return _df.HashPartition("Category", 4)
            .ParallelGroupBy("Category").Sum();
    }

    [Benchmark]
    public DataFrame PartitionedFilterCollect()
    {
        return _df.Partition(4)
            .ParallelFilter(row => (double)row["Value"]! > 500.0)
            .Collect();
    }

    // -- DataFrame Profiling --

    [Benchmark]
    public Cortex.Statistics.DataProfile Profile() => _df.Profile();

    // -- Projection pushdown: wide DataFrame select 2 of 50 columns --

    private DataFrame _wideDf = null!;

    [GlobalSetup(Targets = [nameof(LazySelectWide), nameof(EagerSelectWide)])]
    public void SetupWide()
    {
        Setup();
        var cols = new List<Cortex.Column.IColumn>();
        var rng3 = new Random(42);
        for (int c = 0; c < 50; c++)
        {
            var values = new double[100_000];
            for (int r = 0; r < 100_000; r++)
                values[r] = rng3.NextDouble();
            cols.Add(new Column<double>($"col{c}", values));
        }
        _wideDf = new DataFrame(cols);
    }

    [Benchmark]
    public DataFrame LazySelectWide()
    {
        return _wideDf.Lazy()
            .Filter(Cortex.Expressions.Expr.Col("col25") > Cortex.Expressions.Expr.Lit(0.5))
            .Sort("col10")
            .Select("col0", "col1")
            .Head(100)
            .Collect();
    }

    [Benchmark]
    public DataFrame EagerSelectWide()
    {
        var mask = _wideDf.GetColumn<double>("col25").Gt(0.5);
        return _wideDf.Filter(mask)
            .Sort("col10")
            .Select("col0", "col1")
            .Head(100);
    }
}
