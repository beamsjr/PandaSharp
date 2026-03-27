using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit.Column;

public class ZipTests
{
    [Fact]
    public void Zip_AddTwoColumns()
    {
        var a = new Column<int>("A", [1, 2, 3]);
        var b = new Column<int>("B", [10, 20, 30]);

        var result = a.Zip(b, (x, y) => x + y, "Sum");

        result.Name.Should().Be("Sum");
        result[0].Should().Be(11);
        result[1].Should().Be(22);
        result[2].Should().Be(33);
    }

    [Fact]
    public void Zip_CrossType()
    {
        var prices = new Column<double>("Price", [10.0, 20.0, 30.0]);
        var quantities = new Column<int>("Qty", [5, 3, 8]);

        var result = prices.Zip(quantities, (p, q) => p * q, "Total");

        result[0].Should().Be(50.0);
        result[1].Should().Be(60.0);
        result[2].Should().Be(240.0);
    }

    [Fact]
    public void Zip_NullPropagation()
    {
        var a = Column<int>.FromNullable("A", [1, null, 3]);
        var b = new Column<int>("B", [10, 20, 30]);

        var result = a.Zip(b, (x, y) => x * y, "Product");

        result[0].Should().Be(10);
        result[1].Should().BeNull();
        result[2].Should().Be(90);
    }

    [Fact]
    public void Zip_Boolean()
    {
        var a = new Column<int>("A", [1, 5, 3]);
        var b = new Column<int>("B", [2, 4, 3]);

        var result = a.Zip(b, (x, y) => x > y, "AGtB");

        result[0].Should().Be(false);
        result[1].Should().Be(true);
        result[2].Should().Be(false);
    }

    [Fact]
    public void Zip_LengthMismatch_Throws()
    {
        var a = new Column<int>("A", [1, 2]);
        var b = new Column<int>("B", [1, 2, 3]);

        var act = () => a.Zip(b, (x, y) => x + y);
        act.Should().Throw<ArgumentException>();
    }
}
