using Npgsql;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.IO.Database;
using PandaSharp.Statistics;
using static PandaSharp.Expressions.Expr;

// ============================================================
// PandaSharp + PostgreSQL — Real Database Examples
// ============================================================
// Run with: dotnet run -- "Host=localhost;Username=joe;Password=xxx;Database=postgres"
// Or set:   PANDASHARP_PG_CONN="Host=localhost;..."
// ============================================================

var connString = args.Length > 0
    ? args[0]
    : Environment.GetEnvironmentVariable("PANDASHARP_PG_CONN")
      ?? "Host=localhost;Username=postgres;Password=postgres;Database=postgres";

Console.WriteLine("PandaSharp + PostgreSQL Examples");
Console.WriteLine("=================================");
Console.WriteLine($"Connecting to: {MaskPassword(connString)}\n");

await using var adminConn = new NpgsqlConnection(connString);
await adminConn.OpenAsync();

// --- Create a temp schema to avoid polluting the database ---
var schema = $"pandasharp_demo_{DateTime.Now:yyyyMMdd_HHmmss}";
await Exec(adminConn, $"CREATE SCHEMA {schema}");
await Exec(adminConn, $"SET search_path TO {schema}, public");
Console.WriteLine($"Created temp schema: {schema}\n");

try
{
    // --- 1. Create tables and seed data ---
    Console.WriteLine("1. Creating tables and seeding data...");
    await Exec(adminConn, $"""
        CREATE TABLE employees (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary NUMERIC(10,2),
            hire_date DATE
        );
        INSERT INTO employees (name, department, salary, hire_date) VALUES
            ('Alice', 'Engineering', 95000, '2019-03-15'),
            ('Bob', 'Marketing', 72000, '2020-06-01'),
            ('Charlie', 'Engineering', 110000, '2018-11-20'),
            ('Diana', 'Sales', 68000, '2021-01-10'),
            ('Eve', 'Engineering', 125000, '2017-08-22'),
            ('Frank', 'Marketing', 78000, '2019-04-05'),
            ('Grace', 'Sales', 82000, '2020-09-14'),
            ('Henry', 'Engineering', 105000, '2018-02-28'),
            ('Ivy', 'Sales', 71000, '2022-03-01'),
            ('Jack', 'Marketing', 85000, '2019-07-16');

        CREATE TABLE departments (
            name TEXT PRIMARY KEY,
            budget NUMERIC(12,2),
            floor_num INTEGER,
            head TEXT
        );
        INSERT INTO departments VALUES
            ('Engineering', 500000, 3, 'Eve'),
            ('Marketing', 200000, 2, 'Jack'),
            ('Sales', 300000, 1, 'Grace');
    """);
    Console.WriteLine("   10 employees + 3 departments created.\n");

    // --- 2. Full scan with PostgreSQL dialect ---
    Console.WriteLine("2. Full table scan:");
    var scanner = new DatabaseScanner(adminConn, $"employees", new PostgresDialect());
    var all = scanner.Scan();
    Console.WriteLine(all);

    // --- 3. Push-down: Filter + Select ---
    Console.WriteLine("\n3. Push-down: High earners (salary > 80000):");
    var highEarners = scanner.Scan(
        selectColumns: ["name", "department", "salary"],
        filter: Col("salary") > Lit(80000),
        orderBy: [("salary", false)]
    );
    Console.WriteLine(highEarners);

    // --- 4. Push-down: Limit ---
    Console.WriteLine("\n4. Push-down: Top 3 by salary:");
    var top3 = scanner.Scan(
        selectColumns: ["name", "salary"],
        orderBy: [("salary", false)],
        limit: 3
    );
    Console.WriteLine(top3);

    // --- 5. GroupBy push-down ---
    Console.WriteLine("\n5. GroupBy push-down (department totals):");
    var deptStats = scanner.ScanGroupBy(
        groupByColumns: ["department"],
        aggregations: [
            ("salary", "SUM", "total_salary"),
            ("salary", "AVG", "avg_salary"),
            ("id", "COUNT", "headcount")
        ],
        orderBy: [("total_salary", false)]
    );
    Console.WriteLine(deptStats);

    // --- 6. Lazy evaluation → single SQL ---
    Console.WriteLine("\n6. Lazy chain → compiled to SQL:");
    var lazyResult = scanner.Lazy()
        .Filter(Col("salary") >= Lit(75000))
        .Sort("salary", ascending: false)
        .Select("name", "department", "salary")
        .Head(5)
        .Collect();
    Console.WriteLine(lazyResult);

    // --- 7. Count ---
    Console.WriteLine("\n7. Count with filter:");
    var engCount = scanner.Count(Col("department").Eq(Lit("Engineering")));
    Console.WriteLine($"   Engineering employees: {engCount}");

    // --- 8. Join via raw SQL ---
    Console.WriteLine("\n8. JOIN employees × departments:");
    var joined = DataFrameDatabaseExtensions.ReadSql(adminConn,
        $"SELECT e.name, e.department, e.salary, d.budget, d.floor_num, d.head " +
        $"FROM employees e JOIN departments d ON e.department = d.name " +
        "ORDER BY e.salary DESC LIMIT 5");
    Console.WriteLine(joined);

    // --- 9. Write a DataFrame back to Postgres ---
    Console.WriteLine("\n9. Write DataFrame to Postgres:");
    var products = new DataFrame(
        new StringColumn("product", ["Widget A", "Widget B", "Widget C", "Widget D"]),
        new Column<double>("price", [29.99, 49.99, 19.99, 79.99]),
        new Column<int>("stock", [100, 50, 200, 25])
    );

    // Create table first, then write
    await Exec(adminConn, $"""
        CREATE TABLE products (
            product TEXT,
            price DOUBLE PRECISION,
            stock INTEGER
        )
    """);
    products.ToSql(adminConn, $"products", new PostgresDialect());

    var loadedProducts = new DatabaseScanner(adminConn, $"products", new PostgresDialect()).Scan();
    Console.WriteLine(loadedProducts);

    // --- 10. Connection pool ---
    Console.WriteLine("\n10. Connection pool:");
    DatabasePool.Register("demo_pg", adminConn, new PostgresDialect());
    var pooled = DatabasePool.Table("demo_pg", $"employees").Scan(
        selectColumns: ["name", "salary"],
        limit: 3
    );
    Console.WriteLine(pooled);
    DatabasePool.Unregister("demo_pg");

    // --- 11. Schema introspection ---
    Console.WriteLine("\n11. Schema introspection:");
    var schemaInfo = DataFrameDatabaseExtensions.ReadSql(adminConn,
        $"SELECT column_name, data_type, is_nullable FROM information_schema.columns " +
        $"WHERE table_schema = '{schema}' AND table_name = 'employees' ORDER BY ordinal_position");
    Console.WriteLine(schemaInfo);

    // --- 12. Distinct values ---
    Console.WriteLine("\n12. Distinct departments:");
    var depts = scanner.Scan(selectColumns: ["department"], distinct: true);
    Console.WriteLine(depts);

    // --- 13. Profile the data ---
    Console.WriteLine("\n13. DataFrame Profile:");
    var profile = all.Profile();
    Console.WriteLine(profile);

    Console.WriteLine("All PostgreSQL examples completed successfully!");
}
finally
{
    // --- Cleanup ---
    await Exec(adminConn, $"DROP SCHEMA {schema} CASCADE");
    Console.WriteLine($"\nCleaned up schema: {schema}");
}

// --- Helpers ---

static async Task Exec(NpgsqlConnection conn, string sql)
{
    await using var cmd = new NpgsqlCommand(sql, conn);
    await cmd.ExecuteNonQueryAsync();
}

static string MaskPassword(string connString)
{
    return System.Text.RegularExpressions.Regex.Replace(
        connString, @"(?i)(password\s*=\s*)([^;]+)", "$1****");
}
