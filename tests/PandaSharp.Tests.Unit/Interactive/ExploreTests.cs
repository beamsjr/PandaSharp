using System.Net;
using System.Text.Json;
using FluentAssertions;
using PandaSharp.Interactive;
using Xunit;

namespace PandaSharp.Tests.Unit.Interactive;

public class ExploreTests
{
    private static DataFrame CreateTestDataFrame()
    {
        return DataFrame.FromDictionary(new()
        {
            ["Name"] = new[] { "Alice", "Bob", "Charlie" },
            ["Age"] = new[] { 25, 30, 35 },
            ["Score"] = new[] { 88.5, 92.3, 76.1 }
        });
    }

    [Fact]
    public void GenerateHtml_ContainsColumnNames()
    {
        var df = CreateTestDataFrame();
        var server = new ExploreServer(df, 0);
        var html = server.GenerateHtml();

        html.Should().Contain("Name");
        html.Should().Contain("Age");
        html.Should().Contain("Score");
        html.Should().Contain("PandaSharp Explorer");
        html.Should().Contain("3 rows"); // "3 rows x 3 columns"
    }

    [Fact]
    public void GenerateHtml_ContainsColumnNamesInJsArray()
    {
        var df = CreateTestDataFrame();
        var server = new ExploreServer(df, 0);
        var html = server.GenerateHtml();

        // The column names should appear in the JS COLS array
        html.Should().Contain(@"""Name""");
        html.Should().Contain(@"""Age""");
        html.Should().Contain(@"""Score""");
    }

    [Fact]
    public void GenerateHtml_IsSelfContained()
    {
        var df = CreateTestDataFrame();
        var server = new ExploreServer(df, 0);
        var html = server.GenerateHtml();

        // Should have inline CSS and JS, no external CDN references
        html.Should().Contain("<style>");
        html.Should().Contain("<script>");
        html.Should().NotContain("cdn");
        html.Should().NotContain("https://");
    }

    [Fact]
    public async Task DataEndpoint_ReturnsCorrectPage()
    {
        var df = CreateTestDataFrame();
        var server = new ExploreServer(df, 0);
        var url = server.Start();

        try
        {
            using var client = new HttpClient();
            var response = await client.GetAsync($"{url}api/data?page=0&pageSize=2");
            response.StatusCode.Should().Be(HttpStatusCode.OK);

            var json = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            root.GetProperty("total").GetInt32().Should().Be(3);
            root.GetProperty("page").GetInt32().Should().Be(0);
            root.GetProperty("pageSize").GetInt32().Should().Be(2);

            var rows = root.GetProperty("rows");
            rows.GetArrayLength().Should().Be(2); // pageSize=2, so only 2 rows

            var columns = root.GetProperty("columns");
            columns.GetArrayLength().Should().Be(3);
            columns[0].GetString().Should().Be("Name");
            columns[1].GetString().Should().Be("Age");
            columns[2].GetString().Should().Be("Score");

            // First row: Alice, 25, 88.5
            var firstRow = rows[0];
            firstRow[0].GetString().Should().Be("Alice");
            firstRow[1].GetInt32().Should().Be(25);
            firstRow[2].GetDouble().Should().BeApproximately(88.5, 0.01);
        }
        finally
        {
            server.Stop();
        }
    }

    [Fact]
    public async Task DataEndpoint_FilterWorks()
    {
        var df = CreateTestDataFrame();
        var server = new ExploreServer(df, 0);
        var url = server.Start();

        try
        {
            using var client = new HttpClient();
            var response = await client.GetAsync($"{url}api/data?filter=bob");
            var json = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);

            doc.RootElement.GetProperty("total").GetInt32().Should().Be(1);
            doc.RootElement.GetProperty("rows")[0][0].GetString().Should().Be("Bob");
        }
        finally
        {
            server.Stop();
        }
    }

    [Fact]
    public async Task DataEndpoint_SortWorks()
    {
        var df = CreateTestDataFrame();
        var server = new ExploreServer(df, 0);
        var url = server.Start();

        try
        {
            using var client = new HttpClient();
            var response = await client.GetAsync($"{url}api/data?sort=Age&dir=desc");
            var json = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);

            var rows = doc.RootElement.GetProperty("rows");
            // Descending by Age: Charlie(35), Bob(30), Alice(25)
            rows[0][0].GetString().Should().Be("Charlie");
            rows[1][0].GetString().Should().Be("Bob");
            rows[2][0].GetString().Should().Be("Alice");
        }
        finally
        {
            server.Stop();
        }
    }

    [Fact]
    public async Task StatsEndpoint_ReturnsColumnStats()
    {
        var df = CreateTestDataFrame();
        var server = new ExploreServer(df, 0);
        var url = server.Start();

        try
        {
            using var client = new HttpClient();
            var response = await client.GetAsync($"{url}api/stats");
            response.StatusCode.Should().Be(HttpStatusCode.OK);

            var json = await response.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            // Age stats
            var ageStat = root.GetProperty("Age");
            ageStat.GetProperty("type").GetString().Should().Be("int32");
            ageStat.GetProperty("count").GetInt32().Should().Be(3);
            ageStat.GetProperty("min").GetDouble().Should().Be(25);
            ageStat.GetProperty("max").GetDouble().Should().Be(35);
            ageStat.GetProperty("mean").GetDouble().Should().Be(30);

            // Name stats (string — no mean/min/max)
            var nameStat = root.GetProperty("Name");
            nameStat.GetProperty("type").GetString().Should().Be("string");
            nameStat.TryGetProperty("mean", out _).Should().BeFalse();
        }
        finally
        {
            server.Stop();
        }
    }
}
