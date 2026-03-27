using FluentAssertions;
using Microsoft.Data.Sqlite;
using Cortex;
using Cortex.Column;
using Cortex.Expressions;
using Cortex.IO.Database;
using static Cortex.Expressions.Expr;

namespace Cortex.IO.Database.Tests;

public class DatabaseTests : IDisposable
{
    private readonly SqliteConnection _conn;

    public DatabaseTests()
    {
        _conn = new SqliteConnection("Data Source=:memory:");
        _conn.Open();

        // Create test table
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                salary REAL,
                active INTEGER
            );
            INSERT INTO users VALUES (1, 'Alice', 25, 50000.0, 1);
            INSERT INTO users VALUES (2, 'Bob', 30, 62000.0, 1);
            INSERT INTO users VALUES (3, 'Charlie', 35, 75000.0, 0);
            INSERT INTO users VALUES (4, 'Diana', 28, 58000.0, 1);
            INSERT INTO users VALUES (5, 'Eve', 42, 91000.0, 0);

            CREATE TABLE departments (
                id INTEGER PRIMARY KEY,
                dept_name TEXT NOT NULL,
                budget REAL
            );
            INSERT INTO departments VALUES (1, 'Engineering', 500000.0);
            INSERT INTO departments VALUES (2, 'Marketing', 300000.0);
            INSERT INTO departments VALUES (3, 'Sales', 400000.0);
        """;
        cmd.ExecuteNonQuery();
    }

    public void Dispose() => _conn.Dispose();

    // ===== Basic scanning =====

    [Fact]
    public void Scan_AllRows()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Scan();

        df.RowCount.Should().Be(5);
        df.ColumnNames.Should().Contain("name");
        df.ColumnNames.Should().Contain("age");
    }

    [Fact]
    public void Scan_SelectColumns()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Scan(selectColumns: ["name", "age"]);

        df.ColumnCount.Should().Be(2);
        df.ColumnNames.Should().Equal(["name", "age"]);
    }

    [Fact]
    public void Scan_WithFilter()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Scan(filter: Col("age") > Lit(30));

        df.RowCount.Should().Be(2); // Charlie (35), Eve (42)
    }

    [Fact]
    public void Scan_WithOrderBy()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Scan(
            selectColumns: ["name", "salary"],
            orderBy: [("salary", false)]
        );

        df.GetStringColumn("name")[0].Should().Be("Eve"); // highest salary
    }

    [Fact]
    public void Scan_WithLimit()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Scan(limit: 3);
        df.RowCount.Should().Be(3);
    }

    [Fact]
    public void Scan_Distinct()
    {
        // Add duplicate ages
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "INSERT INTO users VALUES (6, 'Frank', 25, 55000.0, 1)";
        cmd.ExecuteNonQuery();

        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Scan(selectColumns: ["age"], distinct: true);

        df.RowCount.Should().Be(5); // 25, 28, 30, 35, 42 (25 deduped)
    }

    // ===== Complex filters =====

    [Fact]
    public void Scan_CompoundFilter()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Scan(
            filter: (Col("age") >= Lit(28)) & (Col("salary") < Lit(70000))
        );

        df.RowCount.Should().Be(2); // Bob (30, 62k), Diana (28, 58k)
    }

    [Fact]
    public void Scan_OrFilter()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Scan(
            filter: (Col("age") < Lit(26)) | (Col("age") > Lit(40))
        );

        df.RowCount.Should().Be(2); // Alice (25), Eve (42)
    }

    // ===== GroupBy push-down =====

    [Fact]
    public void ScanGroupBy_Sum()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.ScanGroupBy(
            groupByColumns: ["active"],
            aggregations: [("salary", "SUM", "total_salary")]
        );

        df.RowCount.Should().Be(2); // active=0, active=1
        df.ColumnNames.Should().Contain("total_salary");
    }

    [Fact]
    public void ScanGroupBy_MultipleAggs()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.ScanGroupBy(
            groupByColumns: ["active"],
            aggregations: [
                ("salary", "AVG", "avg_salary"),
                ("id", "COUNT", "headcount")
            ]
        );

        df.ColumnNames.Should().Contain("avg_salary");
        df.ColumnNames.Should().Contain("headcount");
    }

    // ===== Raw SQL =====

    [Fact]
    public void ScanSql_CustomQuery()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.ScanSql("SELECT name, age * 2 AS double_age FROM users WHERE age > 30");

        df.RowCount.Should().Be(2);
        df.ColumnNames.Should().Contain("double_age");
    }

    // ===== Count =====

    [Fact]
    public void Count_All()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        scanner.Count().Should().Be(5);
    }

    [Fact]
    public void Count_WithFilter()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        scanner.Count(Col("age") > Lit(30)).Should().Be(2);
    }

    // ===== Write DataFrame to DB =====

    [Fact]
    public void ToSql_WritesRows()
    {
        var df = new DataFrame(
            new StringColumn("city", ["NYC", "LA", "Chicago"]),
            new Column<int>("population", [8_000_000, 4_000_000, 2_700_000])
        );

        df.ToSql(_conn, "cities", new SqliteDialect(), createTable: true);

        // Read back
        var scanner = new DatabaseScanner(_conn, "cities", new SqliteDialect());
        var loaded = scanner.Scan();
        loaded.RowCount.Should().Be(3);
        loaded.GetStringColumn("city")[0].Should().Be("NYC");
    }

    [Fact]
    public void RoundTrip_WriteAndRead()
    {
        var df = new DataFrame(
            new Column<int>("id", [1, 2, 3]),
            new Column<double>("value", [1.5, 2.5, 3.5]),
            new StringColumn("label", ["a", "b", "c"])
        );

        df.ToSql(_conn, "roundtrip", new SqliteDialect(), createTable: true);
        var loaded = new DatabaseScanner(_conn, "roundtrip", new SqliteDialect()).Scan();

        loaded.RowCount.Should().Be(3);
        loaded.GetStringColumn("label")[2].Should().Be("c");
    }

    // ===== SQL Generator unit tests =====

    [Fact]
    public void SqlGenerator_SimpleFilter()
    {
        var gen = new SqlGenerator(new PostgresDialect());
        var (sql, _) = gen.GenerateFromExpressions("users",
            selectColumns: ["name", "age"],
            filter: Col("age") > Lit(30));

        sql.Should().Contain("SELECT");
        sql.Should().Contain("\"name\"");
        sql.Should().Contain("\"age\"");
        sql.Should().Contain("WHERE");
    }

    [Fact]
    public void SqlGenerator_Dialect_Postgres()
    {
        var gen = new SqlGenerator(new PostgresDialect());
        var (sql, _) = gen.GenerateFromExpressions("users", limit: 10);
        sql.Should().Contain("LIMIT 10");
    }

    [Fact]
    public void SqlGenerator_Dialect_SqlServer()
    {
        var gen = new SqlGenerator(new SqlServerDialect());
        var (sql, _) = gen.GenerateFromExpressions("users", limit: 10);
        sql.Should().Contain("TOP 10");
    }

    [Fact]
    public void SqlGenerator_Dialect_MySql()
    {
        var gen = new SqlGenerator(new MySqlDialect());
        var (sql, _) = gen.GenerateFromExpressions("users");
        sql.Should().Contain("`users`");
    }

    // ===== Plan walking: Generate from LogicalPlan =====

    [Fact]
    public void Generate_ScanOnly_SelectStar()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.ScanPlan(df);
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, parameters) = gen.Generate(plan, "orders");

        sql.Should().Be("SELECT * FROM \"orders\"");
        parameters.Should().BeEmpty();
    }

    [Fact]
    public void Generate_Filter_PushesWhereClause()
    {
        var df = new DataFrame(new Column<int>("age", [25]));
        var plan = new Cortex.Lazy.FilterPlan(
            new Cortex.Lazy.ScanPlan(df),
            Col("age") > Lit(30));
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, parameters) = gen.Generate(plan, "users");

        sql.Should().Contain("WHERE");
        sql.Should().Contain("\"age\"");
        parameters.Should().HaveCount(1);
        parameters[0].Value.Should().Be(30);
    }

    [Fact]
    public void Generate_Select_PushesProjection()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.SelectPlan(
            new Cortex.Lazy.ScanPlan(df),
            ["name", "age"]);
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, _) = gen.Generate(plan, "users");

        sql.Should().Be("SELECT \"name\", \"age\" FROM \"users\"");
    }

    [Fact]
    public void Generate_Sort_PushesOrderBy()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.SortPlan(
            new Cortex.Lazy.ScanPlan(df),
            "salary", false);
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, _) = gen.Generate(plan, "users");

        sql.Should().Contain("ORDER BY \"salary\" DESC");
    }

    [Fact]
    public void Generate_Head_PushesLimit()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.HeadPlan(
            new Cortex.Lazy.ScanPlan(df), 10);
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, _) = gen.Generate(plan, "users");

        sql.Should().Contain("LIMIT 10");
    }

    [Fact]
    public void Generate_Head_SqlServer_UsesTop()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.HeadPlan(
            new Cortex.Lazy.ScanPlan(df), 5);
        var gen = new SqlGenerator(new SqlServerDialect());

        var (sql, _) = gen.Generate(plan, "users");

        sql.Should().Contain("TOP 5");
    }

    [Fact]
    public void Generate_FilterThenSort_CombinesClauses()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.SortPlan(
            new Cortex.Lazy.FilterPlan(
                new Cortex.Lazy.ScanPlan(df),
                Col("age") > Lit(25)),
            "salary", true);
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, parameters) = gen.Generate(plan, "employees");

        sql.Should().Contain("WHERE");
        sql.Should().Contain("ORDER BY \"salary\" ASC");
        parameters.Should().HaveCount(1);
    }

    [Fact]
    public void Generate_SelectFilterSortHead_FullChain()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.HeadPlan(
            new Cortex.Lazy.SortPlan(
                new Cortex.Lazy.FilterPlan(
                    new Cortex.Lazy.SelectPlan(
                        new Cortex.Lazy.ScanPlan(df),
                        ["name", "salary"]),
                    Col("salary") > Lit(50000)),
                "salary", false),
            5);
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, parameters) = gen.Generate(plan, "employees");

        sql.Should().Contain("SELECT \"name\", \"salary\"");
        sql.Should().Contain("WHERE");
        sql.Should().Contain("ORDER BY \"salary\" DESC");
        sql.Should().Contain("LIMIT 5");
        parameters.Should().HaveCount(1);
    }

    [Fact]
    public void Generate_MultipleFilters_CombinedWithAnd()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.FilterPlan(
            new Cortex.Lazy.FilterPlan(
                new Cortex.Lazy.ScanPlan(df),
                Col("age") > Lit(25)),
            Col("salary") < Lit(80000));
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, parameters) = gen.Generate(plan, "users");

        sql.Should().Contain("WHERE");
        sql.Should().Contain("AND");
        parameters.Should().HaveCount(2);
    }

    [Fact]
    public void Generate_GroupByCount()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.GroupByAggPlan(
            new Cortex.Lazy.ScanPlan(df),
            ["department"],
            Cortex.GroupBy.AggFunc.Count);
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, _) = gen.Generate(plan, "employees");

        sql.Should().Contain("\"department\"");
        sql.Should().Contain("COUNT(*)");
        sql.Should().Contain("GROUP BY \"department\"");
    }

    [Fact]
    public void Generate_WithColumn_AddsComputedColumn()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.WithColumnPlan(
            new Cortex.Lazy.ScanPlan(df),
            Col("price") * Col("quantity"),
            "total");
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, _) = gen.Generate(plan, "orders");

        sql.Should().Contain("\"total\"");
        sql.Should().Contain("\"price\"");
        sql.Should().Contain("\"quantity\"");
    }

    [Fact]
    public void Generate_ChainedHead_TakesSmallestLimit()
    {
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.HeadPlan(
            new Cortex.Lazy.HeadPlan(
                new Cortex.Lazy.ScanPlan(df), 100),
            10);
        var gen = new SqlGenerator(new PostgresDialect());

        var (sql, _) = gen.Generate(plan, "users");

        sql.Should().Contain("LIMIT 10");
        sql.Should().NotContain("LIMIT 100");
    }

    // ===== End-to-end: plan walking with actual SQLite execution =====

    [Fact]
    public void Generate_FilterHead_EndToEnd_SQLite()
    {
        var gen = new SqlGenerator(new SqliteDialect());
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.HeadPlan(
            new Cortex.Lazy.FilterPlan(
                new Cortex.Lazy.ScanPlan(df),
                Col("age") > Lit(28)),
            2);

        var (sql, parameters) = gen.Generate(plan, "users");

        // Execute against our test DB
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = sql;
        foreach (var (name, value) in parameters)
        {
            var p = cmd.CreateParameter();
            p.ParameterName = name;
            p.Value = value ?? DBNull.Value;
            cmd.Parameters.Add(p);
        }

        using var reader = cmd.ExecuteReader();
        int rowCount = 0;
        while (reader.Read()) rowCount++;
        rowCount.Should().Be(2); // Bob(30), Charlie(35) — limited to 2
    }

    [Fact]
    public void Generate_SelectSort_EndToEnd_SQLite()
    {
        var gen = new SqlGenerator(new SqliteDialect());
        var df = new DataFrame(new Column<int>("id", [1]));
        var plan = new Cortex.Lazy.SortPlan(
            new Cortex.Lazy.SelectPlan(
                new Cortex.Lazy.ScanPlan(df),
                ["name", "salary"]),
            "salary", false);

        var (sql, parameters) = gen.Generate(plan, "users");

        using var cmd = _conn.CreateCommand();
        cmd.CommandText = sql;
        using var reader = cmd.ExecuteReader();
        reader.Read().Should().BeTrue();
        reader.GetString(0).Should().Be("Eve"); // highest salary first
    }

    // ===== Lazy database scans with auto SQL push-down =====

    [Fact]
    public void Lazy_Collect_ReturnsAllRows()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Lazy().Collect();

        df.RowCount.Should().Be(5);
        df.ColumnNames.Should().Contain("name");
    }

    [Fact]
    public void Lazy_Filter_PushesWhereToSQL()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Lazy()
            .Filter(Col("age") > Lit(30))
            .Collect();

        df.RowCount.Should().Be(2); // Charlie (35), Eve (42)
    }

    [Fact]
    public void Lazy_Select_PushesProjection()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Lazy()
            .Select("name", "age")
            .Collect();

        df.ColumnCount.Should().Be(2);
        df.ColumnNames.Should().Equal(["name", "age"]);
        df.RowCount.Should().Be(5);
    }

    [Fact]
    public void Lazy_Sort_PushesOrderBy()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Lazy()
            .Sort("salary", ascending: false)
            .Collect();

        df.GetStringColumn("name")[0].Should().Be("Eve"); // highest salary
    }

    [Fact]
    public void Lazy_Head_PushesLimit()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Lazy()
            .Head(3)
            .Collect();

        df.RowCount.Should().Be(3);
    }

    [Fact]
    public void Lazy_FilterSortHead_FullChain()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Lazy()
            .Filter(Col("salary") > Lit(55000))
            .Sort("salary", ascending: false)
            .Head(2)
            .Collect();

        df.RowCount.Should().Be(2);
        df.GetStringColumn("name")[0].Should().Be("Eve");    // 91000
        df.GetStringColumn("name")[1].Should().Be("Charlie"); // 75000
    }

    [Fact]
    public void Lazy_SelectFilter_CombinesProjectionAndWhere()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Lazy()
            .Select("name", "salary")
            .Filter(Col("salary") >= Lit(60000))
            .Collect();

        df.ColumnCount.Should().Be(2);
        df.RowCount.Should().Be(3); // Bob (62k), Charlie (75k), Eve (91k)
    }

    [Fact]
    public void Lazy_Explain_ShowsExternalScan()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var explanation = scanner.Lazy()
            .Filter(Col("age") > Lit(30))
            .Sort("salary")
            .Explain();

        explanation.Should().Contain("ExternalScan");
        explanation.Should().Contain("db://users");
    }

    [Fact]
    public void Lazy_MultipleFilters_AllPushedDown()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.Lazy()
            .Filter(Col("age") >= Lit(28))
            .Filter(Col("salary") < Lit(70000))
            .Collect();

        df.RowCount.Should().Be(2); // Bob (30, 62k), Diana (28, 58k)
    }

    // ===== Batched write tests =====

    [Fact]
    public void ToSql_BatchedWrite_LargeDataFrame()
    {
        int n = 5000;
        var ids = new int[n];
        var values = new double[n];
        for (int i = 0; i < n; i++) { ids[i] = i; values[i] = i * 1.5; }

        var df = new DataFrame(
            new Column<int>("id", ids),
            new Column<double>("value", values)
        );

        df.ToSql(_conn, "bulk_test", new SqliteDialect(), createTable: true, batchSize: 500);

        var scanner = new DatabaseScanner(_conn, "bulk_test", new SqliteDialect());
        scanner.Count().Should().Be(5000);

        // Verify first and last rows
        var loaded = scanner.Scan(limit: 1);
        loaded.RowCount.Should().Be(1);
    }

    [Fact]
    public void ToSql_BatchedWrite_SmallBatchSize()
    {
        var df = new DataFrame(
            new Column<int>("x", [1, 2, 3, 4, 5]),
            new StringColumn("y", ["a", "b", "c", "d", "e"])
        );

        df.ToSql(_conn, "small_batch", new SqliteDialect(), createTable: true, batchSize: 2);

        var loaded = new DatabaseScanner(_conn, "small_batch", new SqliteDialect()).Scan();
        loaded.RowCount.Should().Be(5);
        loaded.GetStringColumn("y")[4].Should().Be("e");
    }

    [Fact]
    public void ToSql_BatchedWrite_NullValues()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("x", [1, null, 3]),
            new StringColumn("y", ["a", null, "c"])
        );

        df.ToSql(_conn, "null_batch", new SqliteDialect(), createTable: true);

        var loaded = new DatabaseScanner(_conn, "null_batch", new SqliteDialect()).Scan();
        loaded.RowCount.Should().Be(3);
        loaded["x"].IsNull(1).Should().BeTrue();
        loaded["y"].IsNull(1).Should().BeTrue();
    }

    // ===== Join push-down =====

    [Fact]
    public void LazyJoin_InnerJoin_PushesSQL()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.LazyJoin("departments", "id", Cortex.Joins.JoinType.Inner)
            .Collect();

        // Users 1,2,3 match departments 1,2,3
        df.RowCount.Should().Be(3);
        df.ColumnNames.Should().Contain("name");
        df.ColumnNames.Should().Contain("dept_name");
    }

    [Fact]
    public void LazyJoin_LeftJoin_IncludesUnmatched()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.LazyJoin("departments", "id", Cortex.Joins.JoinType.Left)
            .Collect();

        // All 5 users, with nulls for Diana(4) and Eve(5) who have no dept match
        df.RowCount.Should().Be(5);
    }

    [Fact]
    public void LazyJoin_WithFilter_PushesFilterAndJoin()
    {
        var scanner = new DatabaseScanner(_conn, "users", new SqliteDialect());
        var df = scanner.LazyJoin("departments", "id", Cortex.Joins.JoinType.Inner)
            .Filter(Col("age") > Lit(28))
            .Collect();

        // Users with id=1,2,3 match depts; of those, age>28 = Bob(30), Charlie(35)
        df.RowCount.Should().Be(2);
    }

    [Fact]
    public void SqlGenerator_JoinPlan_EmitsJoinSQL()
    {
        var leftPlan = new Cortex.Lazy.ExternalScanPlan(_ => null!, "db://users");
        var rightPlan = new Cortex.Lazy.ExternalScanPlan(_ => null!, "db://departments");
        var joinPlan = new Cortex.Lazy.JoinPlan(leftPlan, rightPlan, "id", Cortex.Joins.JoinType.Inner);

        var gen = new SqlGenerator(new PostgresDialect());
        var (sql, _) = gen.Generate(joinPlan, "users");

        sql.Should().Contain("INNER JOIN");
        sql.Should().Contain("\"departments\"");
        sql.Should().Contain("ON");
        sql.Should().Contain("\"id\"");
    }

    [Fact]
    public void SqlGenerator_LeftJoinPlan_EmitsLeftJoin()
    {
        var leftPlan = new Cortex.Lazy.ExternalScanPlan(_ => null!, "db://orders");
        var rightPlan = new Cortex.Lazy.ExternalScanPlan(_ => null!, "db://customers");
        var joinPlan = new Cortex.Lazy.JoinPlan(leftPlan, rightPlan, "customer_id", Cortex.Joins.JoinType.Left);

        var gen = new SqlGenerator(new PostgresDialect());
        var (sql, _) = gen.Generate(joinPlan, "orders");

        sql.Should().Contain("LEFT JOIN \"customers\"");
        sql.Should().Contain("ON \"orders\".\"customer_id\" = \"customers\".\"customer_id\"");
    }

    // ===== Connection Pool =====

    [Fact]
    public void DatabasePool_RegisterAndQuery()
    {
        try
        {
            DatabasePool.Register("test_pool", _conn, new SqliteDialect());
            DatabasePool.IsRegistered("test_pool").Should().BeTrue();

            var df = DatabasePool.Table("test_pool", "users").Scan();
            df.RowCount.Should().Be(5);
        }
        finally
        {
            DatabasePool.Unregister("test_pool");
        }
    }

    [Fact]
    public void DatabasePool_RawQuery()
    {
        try
        {
            DatabasePool.Register("test_raw", _conn, new SqliteDialect());
            var df = DatabasePool.Query("test_raw", "SELECT name FROM users WHERE age > 30");
            df.RowCount.Should().Be(2);
        }
        finally
        {
            DatabasePool.Unregister("test_raw");
        }
    }

    [Fact]
    public void DatabasePool_Unregister()
    {
        DatabasePool.Register("temp_db", _conn);
        DatabasePool.IsRegistered("temp_db").Should().BeTrue();
        DatabasePool.Unregister("temp_db").Should().BeTrue();
        DatabasePool.IsRegistered("temp_db").Should().BeFalse();
    }

    [Fact]
    public void DatabasePool_NotRegistered_Throws()
    {
        var act = () => DatabasePool.Table("nonexistent", "users");
        act.Should().Throw<KeyNotFoundException>();
    }

    [Fact]
    public void DatabasePool_RegisteredNames()
    {
        try
        {
            DatabasePool.Register("db_a", _conn);
            DatabasePool.Register("db_b", _conn);
            DatabasePool.RegisteredNames.Should().Contain("db_a");
            DatabasePool.RegisteredNames.Should().Contain("db_b");
        }
        finally
        {
            DatabasePool.Unregister("db_a");
            DatabasePool.Unregister("db_b");
        }
    }
}
