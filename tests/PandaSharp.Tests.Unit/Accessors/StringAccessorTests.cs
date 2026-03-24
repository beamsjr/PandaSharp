using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Accessors;

public class StringAccessorTests
{
    private static StringColumn Col(params string?[] values) => new("s", values);

    [Fact]
    public void Contains_ReturnsMatchMask()
    {
        var result = Col("hello", "world", "hello world").Str.Contains("hello");
        result[0].Should().Be(true);
        result[1].Should().Be(false);
        result[2].Should().Be(true);
    }

    [Fact]
    public void Contains_NullPropagates()
    {
        var result = Col("hello", null).Str.Contains("x");
        result[1].Should().BeNull();
    }

    [Fact]
    public void StartsWith_Works()
    {
        var result = Col("abc", "xyz").Str.StartsWith("ab");
        result[0].Should().Be(true);
        result[1].Should().Be(false);
    }

    [Fact]
    public void EndsWith_Works()
    {
        var result = Col("abc", "xyz").Str.EndsWith("yz");
        result[0].Should().Be(false);
        result[1].Should().Be(true);
    }

    [Fact]
    public void Match_Regex()
    {
        var result = Col("abc123", "xyz", "a1b2").Str.Match(@"\d+");
        result[0].Should().Be(true);
        result[1].Should().Be(false);
        result[2].Should().Be(true);
    }

    [Fact]
    public void Extract_CaptureGroup()
    {
        var result = Col("age:25", "age:30", "none").Str.Extract(@"age:(\d+)");
        result[0].Should().Be("25");
        result[1].Should().Be("30");
        result[2].Should().BeNull();
    }

    [Fact]
    public void Upper_TransformsToUpperCase()
    {
        var result = Col("hello", "World").Str.Upper();
        result[0].Should().Be("HELLO");
        result[1].Should().Be("WORLD");
    }

    [Fact]
    public void Lower_TransformsToLowerCase()
    {
        var result = Col("HELLO", "World").Str.Lower();
        result[0].Should().Be("hello");
        result[1].Should().Be("world");
    }

    [Fact]
    public void Title_TitleCases()
    {
        var result = Col("hello world", "foo bar").Str.Title();
        result[0].Should().Be("Hello World");
        result[1].Should().Be("Foo Bar");
    }

    [Fact]
    public void Capitalize_CapitalizesFirstChar()
    {
        var result = Col("hello", "WORLD").Str.Capitalize();
        result[0].Should().Be("Hello");
        result[1].Should().Be("World");
    }

    [Fact]
    public void Replace_ReplacesSubstring()
    {
        var result = Col("hello world", "foo bar").Str.Replace("o", "0");
        result[0].Should().Be("hell0 w0rld");
        result[1].Should().Be("f00 bar");
    }

    [Fact]
    public void Trim_RemovesWhitespace()
    {
        var result = Col("  hello  ", " world ").Str.Trim();
        result[0].Should().Be("hello");
        result[1].Should().Be("world");
    }

    [Fact]
    public void LStrip_RemovesLeadingWhitespace()
    {
        var result = Col("  hello  ").Str.LStrip();
        result[0].Should().Be("hello  ");
    }

    [Fact]
    public void RStrip_RemovesTrailingWhitespace()
    {
        var result = Col("  hello  ").Str.RStrip();
        result[0].Should().Be("  hello");
    }

    [Fact]
    public void Pad_PadsLeft()
    {
        var result = Col("hi", "hello").Str.Pad(5, '*');
        result[0].Should().Be("***hi");
        result[1].Should().Be("hello");
    }

    [Fact]
    public void Pad_PadsRight()
    {
        var result = Col("hi").Str.Pad(5, '*', leftPad: false);
        result[0].Should().Be("hi***");
    }

    [Fact]
    public void Slice_ExtractsSubstring()
    {
        var result = Col("hello world").Str.Slice(0, 5);
        result[0].Should().Be("hello");
    }

    [Fact]
    public void Slice_NegativeStart()
    {
        var result = Col("hello world").Str.Slice(-5);
        result[0].Should().Be("world");
    }

    [Fact]
    public void Len_ReturnsLength()
    {
        var result = Col("hi", "hello", null).Str.Len();
        result[0].Should().Be(2);
        result[1].Should().Be(5);
        result[2].Should().BeNull();
    }

    [Fact]
    public void Count_CountsOccurrences()
    {
        var result = Col("aabaa", "xyz").Str.Count("a");
        result[0].Should().Be(4);
        result[1].Should().Be(0);
    }

    [Fact]
    public void Find_ReturnsIndex()
    {
        var result = Col("hello", "world").Str.Find("lo");
        result[0].Should().Be(3);
        result[1].Should().Be(-1);
    }

    [Fact]
    public void Repeat_RepeatsString()
    {
        var result = Col("ab", null).Str.Repeat(3);
        result[0].Should().Be("ababab");
        result[1].Should().BeNull();
    }

    [Fact]
    public void Cat_ConcatenatesAll()
    {
        var result = Col("a", null, "b", "c").Str.Cat(", ");
        result.Should().Be("a, b, c");
    }
}
