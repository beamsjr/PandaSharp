using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit.Column;

public class ReplaceTests
{
    [Fact]
    public void Replace_Int_MapsValues()
    {
        var col = new Column<int>("X", [1, 2, 3, 4, 5]);
        var result = col.Replace(new Dictionary<int, int> { [2] = 20, [4] = 40 });

        result[0].Should().Be(1);  // unchanged
        result[1].Should().Be(20); // replaced
        result[2].Should().Be(3);  // unchanged
        result[3].Should().Be(40); // replaced
        result[4].Should().Be(5);  // unchanged
    }

    [Fact]
    public void Replace_String_MapsValues()
    {
        var col = new StringColumn("S", ["Red", "Blue", "Green", "Red"]);
        var result = col.Replace(new Dictionary<string, string> { ["Red"] = "Rojo", ["Blue"] = "Azul" });

        result[0].Should().Be("Rojo");
        result[1].Should().Be("Azul");
        result[2].Should().Be("Green"); // not in mapping
        result[3].Should().Be("Rojo");
    }

    [Fact]
    public void Replace_WithNulls_PreservesNulls()
    {
        var col = Column<int>.FromNullable("X", [1, null, 3]);
        var result = col.Replace(new Dictionary<int, int> { [1] = 10 });

        result[0].Should().Be(10);
        result[1].Should().BeNull();
        result[2].Should().Be(3);
    }

    [Fact]
    public void Replace_EmptyMapping_NoChange()
    {
        var col = new Column<int>("X", [1, 2, 3]);
        var result = col.Replace(new Dictionary<int, int>());

        result[0].Should().Be(1);
        result[1].Should().Be(2);
        result[2].Should().Be(3);
    }
}
