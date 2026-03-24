using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Expressions;
using PandaSharp.GroupBy;
using PandaSharp.Joins;
using PandaSharp.Lazy;
using PandaSharp.Missing;
using PandaSharp.Reshape;
using PandaSharp.Statistics;
using PandaSharp.Window;
using static PandaSharp.Expressions.Expr;

// Create a DataFrame
var df = new DataFrame(
    new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana", "Eve"]),
    new Column<int>("Age", [25, 30, 35, 28, 42]),
    new Column<double>("Salary", [50_000, 62_000, 75_000, 58_000, 91_000])
);

Console.WriteLine("=== Full DataFrame ===");
Console.WriteLine(df);
Console.WriteLine();

// Head
Console.WriteLine("=== Head(3) ===");
Console.WriteLine(df.Head(3));
Console.WriteLine();

// Filter: Age > 30
Console.WriteLine("=== Filter: Age > 30 ===");
var mask = df.GetColumn<int>("Age").Gt(30);
var seniors = df.Filter(mask);
Console.WriteLine(seniors);
Console.WriteLine();

// Sort by Salary descending
Console.WriteLine("=== Sort by Salary (desc) ===");
Console.WriteLine(df.Sort("Salary", ascending: false));
Console.WriteLine();

// Select columns
Console.WriteLine("=== Select Name, Salary ===");
Console.WriteLine(df.Select("Name", "Salary"));
Console.WriteLine();

// Aggregates
var salaries = df.GetColumn<double>("Salary");
Console.WriteLine($"Sum:    {salaries.Sum()}");
Console.WriteLine($"Mean:   {salaries.Mean():F2}");
Console.WriteLine($"Median: {salaries.Median():F2}");
Console.WriteLine($"Std:    {salaries.Std():F2}");
Console.WriteLine($"Min:    {salaries.Min()}");
Console.WriteLine($"Max:    {salaries.Max()}");
Console.WriteLine();

// Describe
Console.WriteLine("=== Describe ===");
Console.WriteLine(df.Describe());
Console.WriteLine();

// Info
Console.WriteLine("=== Info ===");
Console.WriteLine(df.Info());
Console.WriteLine();

// Missing data
Console.WriteLine("=== Missing Data Demo ===");
var dfWithNulls = new DataFrame(
    new StringColumn("Name", ["Alice", "Bob", null, "Diana"]),
    Column<double>.FromNullable("Score", [95.0, null, 78.0, null])
);
Console.WriteLine("Original:");
Console.WriteLine(dfWithNulls);
Console.WriteLine();

Console.WriteLine("FillNa(forward):");
var filled = Column<double>.FromNullable("Score", [95.0, null, 78.0, null]).FillNa(FillStrategy.Forward);
var dfFilled = new DataFrame(
    new StringColumn("Name", ["Alice", "Bob", null, "Diana"]),
    filled
);
Console.WriteLine(dfFilled);
Console.WriteLine();

Console.WriteLine("DropNa:");
Console.WriteLine(dfWithNulls.DropNa());
Console.WriteLine();

// Cumulative
Console.WriteLine("=== CumSum ===");
var prices = new Column<double>("Price", [10.0, 20.0, 15.0, 25.0, 30.0]);
var cumSum = prices.CumSum();
for (int i = 0; i < cumSum.Length; i++)
    Console.Write($"{cumSum[i]} ");
Console.WriteLine();
Console.WriteLine();

// Correlation
Console.WriteLine("=== Correlation ===");
Console.WriteLine(df.Corr());
Console.WriteLine();

// GroupBy
Console.WriteLine("=== GroupBy Demo ===");
var dfDept = new DataFrame(
    new StringColumn("Dept", ["Sales", "Eng", "Sales", "Eng", "Sales"]),
    new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana", "Eve"]),
    new Column<double>("Salary", [50_000, 62_000, 75_000, 58_000, 91_000])
);
Console.WriteLine("GroupBy Dept → Mean:");
Console.WriteLine(dfDept.GroupBy("Dept").Mean());
Console.WriteLine();

// Named aggregation
Console.WriteLine("Named Agg:");
Console.WriteLine(dfDept.GroupBy("Dept").Agg(b => b
    .Sum("Salary", alias: "TotalPay")
    .Count("Name", alias: "HeadCount")
));
Console.WriteLine();

// Join
Console.WriteLine("=== Join Demo ===");
var departments = new DataFrame(
    new StringColumn("Dept", ["Sales", "Eng", "Marketing"]),
    new StringColumn("Lead", ["Alice", "Bob", "Carol"])
);
Console.WriteLine(dfDept.Select("Dept", "Name").Join(departments, "Dept", how: JoinType.Left));
Console.WriteLine();

// Expression system
Console.WriteLine("=== Expression System ===");
var dfExpr = df.WithColumn(Col("Salary") * Lit(1.1), "RaisedSalary");
Console.WriteLine(dfExpr.Select("Name", "Salary", "RaisedSalary"));
Console.WriteLine();

// Lazy evaluation
Console.WriteLine("=== Lazy Evaluation ===");
var lazyResult = df.Lazy()
    .Filter(Col("Age") > Lit(25))
    .Sort("Salary", ascending: false)
    .Select("Name", "Salary")
    .Head(3)
    .Collect();
Console.WriteLine(lazyResult);
Console.WriteLine();

// Rolling window
Console.WriteLine("=== Rolling Window ===");
var tsValues = new Column<double>("Value", [1.0, 3.0, 5.0, 2.0, 8.0, 6.0, 4.0]);
var rolling = tsValues.Rolling(3).Mean();
Console.Write("Rolling(3).Mean(): ");
for (int i = 0; i < rolling.Length; i++)
    Console.Write($"{rolling[i]?.ToString("F1") ?? "null"} ");
Console.WriteLine();
Console.WriteLine();

// Pivot
Console.WriteLine("=== Pivot ===");
var longDf = new DataFrame(
    new StringColumn("Date", ["Jan", "Jan", "Feb", "Feb"]),
    new StringColumn("Product", ["A", "B", "A", "B"]),
    new Column<double>("Sales", [100, 200, 150, 250])
);
Console.WriteLine(longDf.Pivot(index: "Date", columns: "Product", values: "Sales"));
Console.WriteLine();

// GetDummies
Console.WriteLine("=== One-Hot Encoding ===");
var catDf = new DataFrame(
    new StringColumn("Color", ["Red", "Blue", "Red", "Green"])
);
Console.WriteLine(catDf.GetDummies("Color"));
Console.WriteLine();

// Sample
Console.WriteLine("=== Sample(2, seed: 42) ===");
Console.WriteLine(df.Sample(2, seed: 42));
Console.WriteLine();

// When/Then/Otherwise
Console.WriteLine("=== When/Then/Otherwise ===");
var withCategory = df.WithColumn(
    When(Col("Age") > Lit(30))
        .Then(Lit("Senior"))
        .Otherwise(Lit("Junior")),
    "Category"
);
Console.WriteLine(withCategory.Select("Name", "Age", "Category"));
Console.WriteLine();

// Column arithmetic
Console.WriteLine("=== Column Arithmetic ===");
var ages = df.GetColumn<int>("Age");
Console.Write("Age + 10: ");
var shifted = ages.Add(10);
for (int i = 0; i < shifted.Length; i++) Console.Write($"{shifted[i]} ");
Console.WriteLine();
Console.WriteLine();

// Markdown export
Console.WriteLine("=== Markdown Export ===");
Console.WriteLine(df.Head(3).ToMarkdown());

// Shift (lag)
Console.WriteLine("=== Shift/Lag ===");
var salaryCol = df.GetColumn<double>("Salary");
var lagged = salaryCol.Shift(1);
Console.Write("Salary shifted by 1: ");
for (int i = 0; i < lagged.Length; i++) Console.Write($"{lagged[i]?.ToString() ?? "null"} ");
Console.WriteLine();
Console.WriteLine();

// Shape
Console.WriteLine($"Shape: {df.Shape}");
