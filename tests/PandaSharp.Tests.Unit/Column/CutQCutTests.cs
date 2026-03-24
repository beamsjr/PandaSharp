using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Column;

public class CutQCutTests
{
    // ===== Cut (equal-width bins) =====

    [Fact]
    public void Cut_EqualWidth_3Bins()
    {
        var col = new Column<double>("val", [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        var binned = col.Cut(3);

        binned.Length.Should().Be(9);
        binned.NullCount.Should().Be(0);

        // Values 1-3 → first bin, 4-6 → second bin, 7-9 → third bin
        // Check that we have 3 distinct bin labels
        var uniqueLabels = new HashSet<string?>();
        for (int i = 0; i < binned.Length; i++)
            uniqueLabels.Add(binned[i]);

        uniqueLabels.Count.Should().Be(3);
    }

    [Fact]
    public void Cut_WithCustomLabels()
    {
        var col = new Column<double>("val", [10, 20, 30, 40, 50]);
        var binned = col.Cut(2, ["Low", "High"]);

        // 10-30 → Low, 40-50 → High
        for (int i = 0; i < binned.Length; i++)
            binned[i].Should().BeOneOf("Low", "High");
    }

    [Fact]
    public void Cut_WithExplicitEdges()
    {
        var col = new Column<double>("val", [5, 15, 25, 35, 45]);
        var binned = col.Cut([0.0, 10.0, 20.0, 30.0, 50.0]);

        binned.Length.Should().Be(5);
        // Each value should land in a bin
        for (int i = 0; i < binned.Length; i++)
            binned[i].Should().NotBeNull();
    }

    [Fact]
    public void Cut_WithNulls_PreservesNulls()
    {
        var col = Column<double>.FromNullable("val", [1.0, null, 3.0, null, 5.0]);
        var binned = col.Cut(2);

        binned[1].Should().BeNull();
        binned[3].Should().BeNull();
        binned[0].Should().NotBeNull();
    }

    [Fact]
    public void Cut_SingleBin()
    {
        var col = new Column<double>("val", [1, 2, 3]);
        var binned = col.Cut(1);

        // All in same bin
        var label = binned[0];
        binned[1].Should().Be(label);
        binned[2].Should().Be(label);
    }

    [Fact]
    public void Cut_SameValues()
    {
        var col = new Column<double>("val", [5, 5, 5, 5]);
        var binned = col.Cut(3);

        // Should not crash — all values in one bin
        binned.Length.Should().Be(4);
    }

    // ===== QCut (quantile bins) =====

    [Fact]
    public void QCut_Quartiles()
    {
        // 100 evenly spaced values → 4 quartile bins
        var values = Enumerable.Range(0, 100).Select(i => (double)i).ToArray();
        var col = new Column<double>("val", values);
        var binned = col.QCut(4);

        binned.Length.Should().Be(100);

        // Count per bin should be ~25 each
        var binCounts = new Dictionary<string, int>();
        for (int i = 0; i < binned.Length; i++)
        {
            var label = binned[i]!;
            binCounts[label] = binCounts.GetValueOrDefault(label) + 1;
        }

        binCounts.Count.Should().Be(4);
        foreach (var count in binCounts.Values)
            count.Should().BeInRange(20, 30); // roughly equal
    }

    [Fact]
    public void QCut_WithCustomLabels()
    {
        var col = new Column<double>("val", [1, 2, 3, 4, 5, 6, 7, 8]);
        var binned = col.QCut(2, ["Bottom 50%", "Top 50%"]);

        for (int i = 0; i < binned.Length; i++)
            binned[i].Should().BeOneOf("Bottom 50%", "Top 50%");
    }

    [Fact]
    public void QCut_Deciles()
    {
        var values = Enumerable.Range(0, 1000).Select(i => (double)i).ToArray();
        var col = new Column<double>("val", values);
        var binned = col.QCut(10);

        var binCounts = new Dictionary<string, int>();
        for (int i = 0; i < binned.Length; i++)
        {
            var label = binned[i]!;
            binCounts[label] = binCounts.GetValueOrDefault(label) + 1;
        }

        binCounts.Count.Should().Be(10);
        foreach (var count in binCounts.Values)
            count.Should().BeInRange(90, 110); // roughly 100 each
    }

    [Fact]
    public void QCut_WithNulls()
    {
        var col = Column<double>.FromNullable("val", [1.0, null, 3.0, 4.0, null, 6.0]);
        var binned = col.QCut(2);

        binned[1].Should().BeNull();
        binned[4].Should().BeNull();
        binned[0].Should().NotBeNull();
    }

    [Fact]
    public void QCut_EmptyColumn()
    {
        var col = new Column<double>("val", Array.Empty<double>());
        var binned = col.QCut(4);
        binned.Length.Should().Be(0);
    }

    // ===== Cut labels contain interval notation =====

    [Fact]
    public void Cut_AutoLabels_ContainIntervalNotation()
    {
        var col = new Column<double>("val", [1, 5, 9]);
        var binned = col.Cut(3);

        // Labels should contain parentheses and brackets
        for (int i = 0; i < binned.Length; i++)
        {
            binned[i].Should().Contain("(");
            binned[i].Should().Contain("]");
        }
    }
}
