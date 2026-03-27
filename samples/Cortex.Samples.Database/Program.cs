using Microsoft.Data.Sqlite;
using Cortex;
using Cortex.Column;
using Cortex.IO.Database;
using static Cortex.Expressions.Expr;

// ============================================================
// Cortex.IO.Database — Smart Database Integration Examples
// ============================================================
// Demonstrates SQL push-down, lazy evaluation, joins, and
// connection pooling with an in-memory SQLite database.
// ============================================================

Console.WriteLine("Cortex.IO.Database Examples");
Console.WriteLine("================================\n");

// --- Setup: Create an in-memory SQLite database with sample data ---
using var conn = new SqliteConnection("Data Source=:memory:");
conn.Open();

using var cmd = conn.CreateCommand();
cmd.CommandText = """
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        salary REAL,
        hire_year INTEGER
    );
    INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 95000, 2019);
    INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 72000, 2020);
    INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 110000, 2018);
    INSERT INTO employees VALUES (4, 'Diana', 'Sales', 68000, 2021);
    INSERT INTO employees VALUES (5, 'Eve', 'Engineering', 125000, 2017);
    INSERT INTO employees VALUES (6, 'Frank', 'Marketing', 78000, 2019);
    INSERT INTO employees VALUES (7, 'Grace', 'Sales', 82000, 2020);
    INSERT INTO employees VALUES (8, 'Henry', 'Engineering', 105000, 2018);

    CREATE TABLE departments (
        name TEXT PRIMARY KEY,
        budget REAL,
        floor INTEGER
    );
    INSERT INTO departments VALUES ('Engineering', 500000, 3);
    INSERT INTO departments VALUES ('Marketing', 200000, 2);
    INSERT INTO departments VALUES ('Sales', 300000, 1);
""";
cmd.ExecuteNonQuery();

var scanner = new DatabaseScanner(conn, "employees", new SqliteDialect());

// --- 1. Basic Scan ---
Console.WriteLine("1. Full table scan:");
var allEmployees = scanner.Scan();
Console.WriteLine(allEmployees);

// --- 2. Select + Filter Push-down ---
Console.WriteLine("\n2. Push-down: SELECT name, salary WHERE salary > 80000:");
var highEarners = scanner.Scan(
    selectColumns: ["name", "salary"],
    filter: Col("salary") > Lit(80000)
);
Console.WriteLine(highEarners);

// --- 3. Sort + Limit ---
Console.WriteLine("\n3. Push-down: ORDER BY salary DESC LIMIT 3:");
var top3 = scanner.Scan(
    selectColumns: ["name", "department", "salary"],
    orderBy: [("salary", false)],
    limit: 3
);
Console.WriteLine(top3);

// --- 4. Lazy Evaluation with SQL Push-down ---
Console.WriteLine("\n4. Lazy chain → compiles to single SQL query:");
var lazyResult = scanner.Lazy()
    .Filter(Col("hire_year") >= Lit(2019))
    .Sort("salary", ascending: false)
    .Select("name", "department", "salary")
    .Head(5)
    .Collect();
Console.WriteLine(lazyResult);

// --- 5. GroupBy Push-down ---
Console.WriteLine("\n5. GroupBy push-down (SUM salary by department):");
var deptTotals = scanner.ScanGroupBy(
    groupByColumns: ["department"],
    aggregations: [("salary", "SUM", "total_salary"), ("id", "COUNT", "headcount")],
    orderBy: [("total_salary", false)]
);
Console.WriteLine(deptTotals);

// --- 6. Count with Filter ---
Console.WriteLine("\n6. Count with filter (engineering employees):");
var engCount = scanner.Count(Col("department").Eq(Lit("Engineering")));
Console.WriteLine($"   Engineering employees: {engCount}");

// --- 7. Raw SQL ---
Console.WriteLine("\n7. Raw SQL query:");
var rawResult = DataFrameDatabaseExtensions.ReadSql(conn,
    "SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department ORDER BY avg_salary DESC");
Console.WriteLine(rawResult);

// --- 8. Write DataFrame to Database ---
Console.WriteLine("\n8. Write DataFrame to database:");
var newData = new DataFrame(
    new StringColumn("product", ["Widget A", "Widget B", "Widget C"]),
    new Column<double>("price", [29.99, 49.99, 19.99]),
    new Column<int>("stock", [100, 50, 200])
);
newData.ToSql(conn, "products", new SqliteDialect(), createTable: true);
var products = new DatabaseScanner(conn, "products", new SqliteDialect()).Scan();
Console.WriteLine(products);

// --- 9. Join via Raw SQL ---
Console.WriteLine("\n9. Join (employees × departments):");
var joined = DataFrameDatabaseExtensions.ReadSql(conn,
    "SELECT e.name, e.department, e.salary, d.budget, d.floor " +
    "FROM employees e JOIN departments d ON e.department = d.name ORDER BY e.salary DESC");
Console.WriteLine(joined);

// --- 10. Connection Pool ---
Console.WriteLine("\n10. Connection pool:");
DatabasePool.Register("sample_db", conn, new SqliteDialect());
var pooledResult = DatabasePool.Table("sample_db", "employees").Scan(limit: 3);
Console.WriteLine(pooledResult);
Console.WriteLine($"   Registered databases: {string.Join(", ", DatabasePool.RegisteredNames)}");
DatabasePool.Unregister("sample_db");

Console.WriteLine("\nAll examples completed successfully!");
