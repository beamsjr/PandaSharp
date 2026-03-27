using FluentAssertions;
using Cortex.Column;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class AvroOrcTests
{
    // ===== Avro Tests =====

    [Fact]
    public void Avro_RoundTrip_FiveColumns()
    {
        var df = new DataFrame(
            new Column<int>("Id", [1, 2, 3, 4, 5]),
            new Column<double>("Value", [1.1, 2.2, 3.3, 4.4, 5.5]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie", "Dave", "Eve"]),
            new Column<bool>("Active", [true, false, true, false, true]),
            Column<double>.FromNullable("Score", [10.0, null, 30.0, null, 50.0])
        );

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.RowCount.Should().Be(5);
        df2.ColumnCount.Should().Be(5);
        df2.ColumnNames.Should().Equal("Id", "Value", "Name", "Active", "Score");

        df2.GetColumn<int>("Id")[0].Should().Be(1);
        df2.GetColumn<int>("Id")[4].Should().Be(5);
        df2.GetColumn<double>("Value")[0].Should().Be(1.1);
        df2.GetColumn<double>("Value")[4].Should().Be(5.5);
        df2.GetStringColumn("Name")[0].Should().Be("Alice");
        df2.GetStringColumn("Name")[4].Should().Be("Eve");
        df2.GetColumn<bool>("Active")[0].Should().Be(true);
        df2.GetColumn<bool>("Active")[1].Should().Be(false);
        df2.GetColumn<double>("Score")[0].Should().Be(10.0);
        df2["Score"].IsNull(1).Should().BeTrue();
        df2.GetColumn<double>("Score")[2].Should().Be(30.0);
        df2["Score"].IsNull(3).Should().BeTrue();
        df2.GetColumn<double>("Score")[4].Should().Be(50.0);
    }

    [Fact]
    public void Avro_RoundTrip_EmptyDataFrame()
    {
        var df = new DataFrame(
            new Column<int>("A", Array.Empty<int>()),
            new StringColumn("B", Array.Empty<string?>())
        );

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.RowCount.Should().Be(0);
        df2.ColumnCount.Should().Be(2);
    }

    [Fact]
    public void Avro_RoundTrip_NullStrings()
    {
        var df = new DataFrame(
            new StringColumn("Text", ["hello", null, "world", null, "!"])
        );

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.GetStringColumn("Text")[0].Should().Be("hello");
        df2["Text"].IsNull(1).Should().BeTrue();
        df2.GetStringColumn("Text")[2].Should().Be("world");
        df2["Text"].IsNull(3).Should().BeTrue();
        df2.GetStringColumn("Text")[4].Should().Be("!");
    }

    [Fact]
    public void Avro_RoundTrip_LargeStrings()
    {
        var longString = new string('X', 10_000);
        var df = new DataFrame(
            new StringColumn("Big", [longString, "short", longString])
        );

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.GetStringColumn("Big")[0].Should().Be(longString);
        df2.GetStringColumn("Big")[1].Should().Be("short");
        df2.GetStringColumn("Big")[2].Should().Be(longString);
    }

    [Fact]
    public void Avro_RoundTrip_LongType()
    {
        var df = new DataFrame(
            new Column<long>("BigNum", [long.MinValue, 0L, long.MaxValue])
        );

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.GetColumn<long>("BigNum")[0].Should().Be(long.MinValue);
        df2.GetColumn<long>("BigNum")[1].Should().Be(0L);
        df2.GetColumn<long>("BigNum")[2].Should().Be(long.MaxValue);
    }

    [Fact]
    public void Avro_RoundTrip_FloatType()
    {
        var df = new DataFrame(
            new Column<float>("F", [1.5f, -2.25f, 0f])
        );

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.GetColumn<float>("F")[0].Should().Be(1.5f);
        df2.GetColumn<float>("F")[1].Should().Be(-2.25f);
        df2.GetColumn<float>("F")[2].Should().Be(0f);
    }

    [Fact]
    public void Avro_FilePath_RoundTrip()
    {
        var df = new DataFrame(
            new Column<int>("X", [10, 20, 30])
        );

        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.avro");
        try
        {
            AvroWriter.Write(df, path);
            var df2 = AvroReader.Read(path);

            df2.RowCount.Should().Be(3);
            df2.GetColumn<int>("X")[1].Should().Be(20);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ===== ORC Tests =====

    [Fact]
    public void Orc_RoundTrip_FiveColumns()
    {
        var df = new DataFrame(
            new Column<int>("Id", [1, 2, 3, 4, 5]),
            new Column<double>("Value", [1.1, 2.2, 3.3, 4.4, 5.5]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie", "Dave", "Eve"]),
            new Column<bool>("Active", [true, false, true, false, true]),
            new Column<long>("BigNum", [100L, 200L, 300L, 400L, 500L])
        );

        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.orc");
        try
        {
            OrcWriter.Write(df, path);
            var df2 = OrcReader.Read(path);

            df2.RowCount.Should().Be(5);
            df2.ColumnCount.Should().Be(5);
            df2.ColumnNames.Should().Equal("Id", "Value", "Name", "Active", "BigNum");

            df2.GetColumn<int>("Id")[0].Should().Be(1);
            df2.GetColumn<int>("Id")[4].Should().Be(5);
            df2.GetColumn<double>("Value")[0].Should().Be(1.1);
            df2.GetColumn<double>("Value")[4].Should().Be(5.5);
            df2.GetStringColumn("Name")[0].Should().Be("Alice");
            df2.GetStringColumn("Name")[4].Should().Be("Eve");
            df2.GetColumn<bool>("Active")[0].Should().Be(true);
            df2.GetColumn<bool>("Active")[1].Should().Be(false);
            df2.GetColumn<long>("BigNum")[0].Should().Be(100L);
            df2.GetColumn<long>("BigNum")[4].Should().Be(500L);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Orc_RoundTrip_WithNulls()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3]),
            new StringColumn("B", ["x", null, "z"]),
            Column<double>.FromNullable("C", [null, 2.0, null])
        );

        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.orc");
        try
        {
            OrcWriter.Write(df, path);
            var df2 = OrcReader.Read(path);

            df2.RowCount.Should().Be(3);
            df2.GetColumn<int>("A")[0].Should().Be(1);
            df2["A"].IsNull(1).Should().BeTrue();
            df2.GetColumn<int>("A")[2].Should().Be(3);
            df2.GetStringColumn("B")[0].Should().Be("x");
            df2["B"].IsNull(1).Should().BeTrue();
            df2.GetStringColumn("B")[2].Should().Be("z");
            df2["C"].IsNull(0).Should().BeTrue();
            df2.GetColumn<double>("C")[1].Should().Be(2.0);
            df2["C"].IsNull(2).Should().BeTrue();
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Orc_RoundTrip_EmptyDataFrame()
    {
        var df = new DataFrame(
            new Column<int>("A", Array.Empty<int>()),
            new StringColumn("B", Array.Empty<string?>())
        );

        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.orc");
        try
        {
            OrcWriter.Write(df, path);
            var df2 = OrcReader.Read(path);

            df2.RowCount.Should().Be(0);
            df2.ColumnCount.Should().Be(2);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Orc_RoundTrip_LargeStrings()
    {
        var longString = new string('Y', 10_000);
        var df = new DataFrame(
            new StringColumn("Big", [longString, "small", longString])
        );

        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.orc");
        try
        {
            OrcWriter.Write(df, path);
            var df2 = OrcReader.Read(path);

            df2.GetStringColumn("Big")[0].Should().Be(longString);
            df2.GetStringColumn("Big")[1].Should().Be("small");
            df2.GetStringColumn("Big")[2].Should().Be(longString);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Orc_RoundTrip_FloatColumn()
    {
        var df = new DataFrame(
            new Column<float>("F", [1.5f, -2.25f, 0f])
        );

        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.orc");
        try
        {
            OrcWriter.Write(df, path);
            var df2 = OrcReader.Read(path);

            df2.GetColumn<float>("F")[0].Should().Be(1.5f);
            df2.GetColumn<float>("F")[1].Should().Be(-2.25f);
            df2.GetColumn<float>("F")[2].Should().Be(0f);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ===== DatabaseConnectionBuilder Tests =====

    [Fact]
    public void DatabaseConnectionBuilder_Postgres()
    {
        var connStr = DatabaseConnectionBuilder.ForPostgres()
            .Host("localhost")
            .Port(5432)
            .Database("mydb")
            .Username("admin")
            .Password("secret")
            .Build();

        connStr.Should().Contain("Host=localhost");
        connStr.Should().Contain("Port=5432");
        connStr.Should().Contain("Database=mydb");
        connStr.Should().Contain("Username=admin");
        connStr.Should().Contain("Password=secret");
    }

    [Fact]
    public void DatabaseConnectionBuilder_Postgres_Dialect()
    {
        DatabaseConnectionBuilder.ForPostgres().Dialect.Should().Be("Npgsql");
    }

    [Fact]
    public void DatabaseConnectionBuilder_SqlServer()
    {
        var connStr = DatabaseConnectionBuilder.ForSqlServer()
            .Host("db.example.com")
            .Port(1433)
            .Database("mydb")
            .Username("sa")
            .Password("p@ss")
            .Build();

        connStr.Should().Contain("Server=db.example.com,1433");
        connStr.Should().Contain("Database=mydb");
        connStr.Should().Contain("User Id=sa");
        connStr.Should().Contain("Password=p@ss");
    }

    [Fact]
    public void DatabaseConnectionBuilder_SqlServer_Dialect()
    {
        DatabaseConnectionBuilder.ForSqlServer().Dialect.Should().Be("SqlClient");
    }

    [Fact]
    public void DatabaseConnectionBuilder_Sqlite()
    {
        var connStr = DatabaseConnectionBuilder.ForSqlite()
            .Database("test.db")
            .Build();

        connStr.Should().Contain("Data Source=test.db");
    }

    [Fact]
    public void DatabaseConnectionBuilder_Sqlite_InMemory()
    {
        var connStr = DatabaseConnectionBuilder.ForSqlite().Build();
        connStr.Should().Contain("Data Source=:memory:");
    }

    [Fact]
    public void DatabaseConnectionBuilder_Sqlite_Dialect()
    {
        DatabaseConnectionBuilder.ForSqlite().Dialect.Should().Be("Sqlite");
    }

    [Fact]
    public void DatabaseConnectionBuilder_MySql()
    {
        var connStr = DatabaseConnectionBuilder.ForMySql()
            .Host("mysql.example.com")
            .Port(3306)
            .Database("appdb")
            .Username("root")
            .Password("toor")
            .Build();

        connStr.Should().Contain("Server=mysql.example.com");
        connStr.Should().Contain("Port=3306");
        connStr.Should().Contain("Database=appdb");
        connStr.Should().Contain("Uid=root");
        connStr.Should().Contain("Pwd=toor");
    }

    [Fact]
    public void DatabaseConnectionBuilder_MySql_Dialect()
    {
        DatabaseConnectionBuilder.ForMySql().Dialect.Should().Be("MySql");
    }

    [Fact]
    public void DatabaseConnectionBuilder_WithOptions()
    {
        var connStr = DatabaseConnectionBuilder.ForPostgres()
            .Host("localhost")
            .Database("mydb")
            .Option("Pooling", "true")
            .Option("Timeout", "30")
            .Build();

        connStr.Should().Contain("Pooling=true");
        connStr.Should().Contain("Timeout=30");
    }

    [Fact]
    public void DatabaseConnectionBuilder_FluentChaining()
    {
        // Verify fluent API returns same builder instance
        var builder = DatabaseConnectionBuilder.ForPostgres();
        var result = builder.Host("h").Port(1).Database("d").Username("u").Password("p");
        result.Should().BeSameAs(builder);
    }

    // ===== Extension method tests =====

    [Fact]
    public void DataFrameIO_ReadAvro_WriteAvro()
    {
        var df = new DataFrame(
            new Column<int>("X", [1, 2, 3])
        );

        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.avro");
        try
        {
            df.ToAvro(path);
            var df2 = DataFrameIO.ReadAvro(path);
            df2.RowCount.Should().Be(3);
            df2.GetColumn<int>("X")[1].Should().Be(2);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void DataFrameIO_ReadOrc_WriteOrc()
    {
        var df = new DataFrame(
            new Column<int>("X", [1, 2, 3])
        );

        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.orc");
        try
        {
            df.ToOrc(path);
            var df2 = DataFrameIO.ReadOrc(path);
            df2.RowCount.Should().Be(3);
            df2.GetColumn<int>("X")[1].Should().Be(2);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Avro_RoundTrip_AllNullColumn()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("NullCol", [null, null, null])
        );

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.RowCount.Should().Be(3);
        df2["NullCol"].IsNull(0).Should().BeTrue();
        df2["NullCol"].IsNull(1).Should().BeTrue();
        df2["NullCol"].IsNull(2).Should().BeTrue();
    }

    [Fact]
    public void Orc_RoundTrip_AllNullColumn()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("NullCol", [null, null, null])
        );

        var path = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.orc");
        try
        {
            OrcWriter.Write(df, path);
            var df2 = OrcReader.Read(path);

            df2.RowCount.Should().Be(3);
            df2["NullCol"].IsNull(0).Should().BeTrue();
            df2["NullCol"].IsNull(1).Should().BeTrue();
            df2["NullCol"].IsNull(2).Should().BeTrue();
        }
        finally
        {
            File.Delete(path);
        }
    }
}
