using System.Text;
using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using Cortex.GroupBy;
using Cortex.IO;
using Cortex.Joins;
using Cortex.Lazy;
using Cortex.Missing;
using Cortex.Reshape;
using Cortex.Statistics;
using Cortex.Window;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit;

/// <summary>
/// End-to-end integration tests exercising realistic data pipelines.
/// </summary>
public class IntegrationTests
{
    [Fact]
    public void Pipeline_CsvLoadCleanTransformGroupExport()
    {
        // 1. Create CSV data
        var csv = """
            Name,Department,Salary,StartDate
            Alice,Engineering,95000,2020-01-15
            Bob,Sales,,2019-06-01
            Charlie,Engineering,105000,2021-03-20
            Diana,Sales,72000,2020-09-10
            Eve,Engineering,,2022-01-05
            Frank,Sales,68000,2018-11-30
            """;

        // 2. Load
        var df = CsvReader.Read(new MemoryStream(Encoding.UTF8.GetBytes(csv)));
        df.RowCount.Should().Be(6);
        df.ColumnCount.Should().Be(4);

        // 3. Check for nulls
        df["Salary"].NullCount.Should().Be(2);

        // 4. Fill missing salaries with forward fill
        // Salary may be inferred as int (values are whole numbers), so handle both types
        if (df["Salary"] is Column<int> intSalary)
        {
            var filled = intSalary.FillNa(FillStrategy.Forward);
            df = df.Assign("Salary", filled);
        }
        else
        {
            var filled = df.GetColumn<double>("Salary").FillNa(FillStrategy.Forward);
            df = df.Assign("Salary", filled);
        }
        df["Salary"].NullCount.Should().Be(0);

        // 5. Filter: only employees since 2020
        var filtered = df.Query("StartDate >= 2020-01-15");
        filtered.RowCount.Should().BeGreaterThan(0);

        // 6. GroupBy department, get mean salary
        var grouped = df.GroupBy("Department").Mean();
        grouped.RowCount.Should().Be(2);
        grouped.ColumnNames.Should().Contain("Salary");

        // 7. Export to JSON and re-import
        var json = df.ToJsonString();
        json.Should().Contain("Alice");
        var reimported = JsonReader.ReadString(json);
        reimported.RowCount.Should().Be(6);

        // 8. Export to CSV and re-import
        using var ms = new MemoryStream();
        CsvWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var csvReimported = CsvReader.Read(ms);
        csvReimported.RowCount.Should().Be(6);
    }

    [Fact]
    public void Pipeline_ExpressionBasedAnalytics()
    {
        var df = new DataFrame(
            new StringColumn("Product", ["Widget", "Gadget", "Widget", "Gadget", "Widget"]),
            new Column<double>("Price", [10.0, 25.0, 12.0, 28.0, 11.0]),
            new Column<int>("Quantity", [100, 50, 80, 60, 90])
        );

        // Add computed columns using expressions
        var result = df
            .WithColumn(Col("Price") * Col("Quantity"), "Revenue")
            .WithColumn(
                When(Col("Price") > Lit(15.0))
                    .Then(Lit("Premium"))
                    .Otherwise(Lit("Standard")),
                "Tier"
            );

        result.ColumnNames.Should().Contain("Revenue");
        result.ColumnNames.Should().Contain("Tier");
        result.GetColumn<double>("Revenue")[0].Should().Be(1000.0);
        result.GetStringColumn("Tier")[0].Should().Be("Standard");
        result.GetStringColumn("Tier")[1].Should().Be("Premium");

        // GroupBy and aggregate
        var byProduct = result.GroupBy("Product").Agg(b => b
            .Sum("Revenue", alias: "TotalRevenue")
            .Mean("Price", alias: "AvgPrice")
            .Count("Quantity", alias: "Orders")
        );
        byProduct.RowCount.Should().Be(2);
    }

    [Fact]
    public void Pipeline_LazyEvaluation()
    {
        var employees = new DataFrame(
            new Column<int>("Id", [1, 2, 3, 4, 5]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana", "Eve"]),
            new Column<int>("DeptId", [10, 20, 10, 30, 20]),
            new Column<double>("Salary", [90_000, 75_000, 105_000, 65_000, 85_000])
        );

        var departments = new DataFrame(
            new Column<int>("DeptId", [10, 20, 30]),
            new StringColumn("DeptName", ["Engineering", "Sales", "Marketing"])
        );

        // Lazy pipeline: filter → join → sort → select
        var result = employees.Lazy()
            .Filter(Col("Salary") > Lit(70_000))
            .Join(departments.Lazy(), "DeptId")
            .Sort("Salary", ascending: false)
            .Select("Name", "DeptName", "Salary")
            .Collect();

        result.RowCount.Should().Be(4);
        result.GetStringColumn("Name")[0].Should().Be("Charlie"); // highest salary
        result.ColumnNames.Should().Equal(["Name", "DeptName", "Salary"]);
    }

    [Fact]
    public void Pipeline_TimeSeriesAnalysis()
    {
        // Simulate daily stock prices
        var dates = new DateTime[30];
        var prices = new double[30];
        var rng = new Random(42);
        double price = 100.0;
        for (int i = 0; i < 30; i++)
        {
            dates[i] = new DateTime(2024, 1, 1).AddDays(i);
            price += rng.NextDouble() * 4 - 2;
            prices[i] = Math.Round(price, 2);
        }

        var df = new DataFrame(
            new Column<DateTime>("Date", dates),
            new Column<double>("Price", prices)
        );

        // Rolling average
        var rolling = df.GetColumn<double>("Price").Rolling(7).Mean();
        rolling.Length.Should().Be(30);
        rolling[0].Should().BeNull(); // not enough data for window

        // Pct change
        var pctChange = df.GetColumn<double>("Price").PctChange();
        pctChange[0].Should().BeNull();

        // Describe
        var desc = df.Describe();
        desc.RowCount.Should().Be(8);

        // Correlation
        var withRolling = df.AddColumn(new Column<double>("Rolling7",
            Enumerable.Range(0, 30).Select(i => rolling[i] ?? double.NaN).ToArray()));
        // Should have both columns
        withRolling.ColumnNames.Should().Contain("Rolling7");
    }

    [Fact]
    public void Pipeline_ReshapeAndPivot()
    {
        var sales = new DataFrame(
            new StringColumn("Region", ["East", "East", "West", "West", "East", "West"]),
            new StringColumn("Quarter", ["Q1", "Q2", "Q1", "Q2", "Q3", "Q3"]),
            new Column<double>("Revenue", [100, 150, 120, 180, 130, 160])
        );

        // Pivot table
        var pivoted = sales.PivotTable("Region", "Quarter", "Revenue");
        pivoted.RowCount.Should().Be(2);
        pivoted.ColumnNames.Should().Contain("Q1");

        // Melt back
        var melted = pivoted.Melt(idVars: ["Region"]);
        melted.ColumnNames.Should().Contain("variable");
        melted.ColumnNames.Should().Contain("value");

        // CrossTab
        var ct = sales.CrossTab("Region", "Quarter");
        ct.RowCount.Should().Be(2);
    }

    [Fact]
    public void Pipeline_ArrowIpcRoundTrip()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", null]),
            new Column<int>("Age", [25, 30, 35]),
            Column<double>.FromNullable("Score", [95.0, null, 87.0])
        );

        // Write to Arrow IPC and read back
        using var ms = new MemoryStream();
        ArrowIpcWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var loaded = ArrowIpcReader.Read(ms);

        loaded.RowCount.Should().Be(3);
        loaded.GetStringColumn("Name")[0].Should().Be("Alice");
        loaded["Name"].IsNull(2).Should().BeTrue();
        loaded.GetColumn<int>("Age")[2].Should().Be(35);
        loaded["Score"].IsNull(1).Should().BeTrue();
    }

    [Fact]
    public void Copy_CreatesIndependentDataFrame()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new StringColumn("B", ["x", "y", "z"])
        );

        var copy = df.Copy();
        copy.RowCount.Should().Be(3);
        copy.GetColumn<int>("A")[0].Should().Be(1);
        copy.ContentEquals(df).Should().BeTrue();
    }

    [Fact]
    public void ContentEquals_SameData_ReturnsTrue()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2, 3]));
        var df2 = new DataFrame(new Column<int>("A", [1, 2, 3]));
        df1.ContentEquals(df2).Should().BeTrue();
    }

    [Fact]
    public void ContentEquals_DifferentData_ReturnsFalse()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2, 3]));
        var df2 = new DataFrame(new Column<int>("A", [1, 2, 4]));
        df1.ContentEquals(df2).Should().BeFalse();
    }

    [Fact]
    public void ContentEquals_DifferentSchema_ReturnsFalse()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2]));
        var df2 = new DataFrame(new Column<int>("B", [1, 2]));
        df1.ContentEquals(df2).Should().BeFalse();
    }

    [Fact]
    public void ContentEquals_DifferentRowCount_ReturnsFalse()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2]));
        var df2 = new DataFrame(new Column<int>("A", [1, 2, 3]));
        df1.ContentEquals(df2).Should().BeFalse();
    }
}
