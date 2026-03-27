using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Tensors;

namespace Cortex.ML.Tests;

public class TensorTests
{
    [Fact]
    public void Tensor_Create_1D()
    {
        var t = new Tensor<double>([1.0, 2.0, 3.0], 3);
        t.Rank.Should().Be(1);
        t.Length.Should().Be(3);
        t[0].Should().Be(1.0);
        t[2].Should().Be(3.0);
    }

    [Fact]
    public void Tensor_Create_2D()
    {
        var t = new Tensor<double>([1, 2, 3, 4, 5, 6], 2, 3);
        t.Rank.Should().Be(2);
        t.Shape.Should().Equal([2, 3]);
        t[0, 0].Should().Be(1);
        t[1, 2].Should().Be(6);
    }

    [Fact]
    public void Tensor_Zeros()
    {
        var t = Tensor<double>.Zeros(3, 4);
        t.Length.Should().Be(12);
        t.Sum().Should().Be(0);
    }

    [Fact]
    public void Tensor_Ones()
    {
        var t = Tensor<double>.Ones(5);
        t.Sum().Should().Be(5);
    }

    [Fact]
    public void Tensor_Add()
    {
        var a = new Tensor<double>([1, 2, 3], 3);
        var b = new Tensor<double>([10, 20, 30], 3);
        var c = a + b;
        c[0].Should().Be(11);
        c[2].Should().Be(33);
    }

    [Fact]
    public void Tensor_MultiplyScalar()
    {
        var t = new Tensor<double>([1, 2, 3], 3);
        var r = t * 10.0;
        r[0].Should().Be(10);
        r[2].Should().Be(30);
    }

    [Fact]
    public void Tensor_Sum_Mean()
    {
        var t = new Tensor<double>([10, 20, 30], 3);
        t.Sum().Should().Be(60);
        t.Mean().Should().Be(20);
    }

    [Fact]
    public void Tensor_Transpose()
    {
        var t = new Tensor<double>([1, 2, 3, 4, 5, 6], 2, 3);
        var tr = t.Transpose();
        tr.Shape.Should().Equal([3, 2]);
        tr[0, 0].Should().Be(1);
        tr[0, 1].Should().Be(4);
        tr[2, 0].Should().Be(3);
    }

    [Fact]
    public void Tensor_ArgMax()
    {
        var t = new Tensor<double>([1, 5, 3, 8, 2, 6], 2, 3);
        var am = t.ArgMax(axis: 1);
        am[0].Should().Be(1); // row 0: max at col 1 (value 5)
        am[1].Should().Be(0); // row 1: max at col 0 (value 8)
    }

    [Fact]
    public void Column_ToTensor()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0]);
        var t = col.AsTensor();
        t.Rank.Should().Be(1);
        t.Length.Should().Be(3);
        t[0].Should().Be(1.0);
    }

    [Fact]
    public void DataFrame_ToTensor_2D()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0]),
            new Column<double>("B", [3.0, 4.0]),
            new Column<double>("C", [5.0, 6.0])
        );

        var t = df.ToTensor("A", "B", "C");
        t.Shape.Should().Equal([2, 3]);
        t[0, 0].Should().Be(1.0);
        t[1, 2].Should().Be(6.0);
    }

    [Fact]
    public void Tensor_ToDataFrame()
    {
        var t = new Tensor<double>([1, 2, 3, 4], 2, 2);
        var df = t.ToDataFrame(["X", "Y"]);
        df.RowCount.Should().Be(2);
        df.GetColumn<double>("X")[0].Should().Be(1);
        df.GetColumn<double>("Y")[1].Should().Be(4);
    }

    [Fact]
    public void Tensor_ToColumn()
    {
        var t = new Tensor<double>([1, 2, 3], 3);
        var col = t.ToColumn<double>("result");
        col.Name.Should().Be("result");
        col[1].Should().Be(2);
    }

    // ===== N-dimensional operations =====

    [Fact]
    public void Tensor3D_Slice_Axis0()
    {
        // 2×3×2 tensor
        var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var t = new Tensor<double>(data, 2, 3, 2);

        // Slice axis 0: take first "matrix" → 1×3×2
        var sliced = t.Slice(0, 0, 1);
        sliced.Shape.Should().Equal([1, 3, 2]);
        sliced.Length.Should().Be(6);
        sliced[0, 0, 0].Should().Be(1);
        sliced[0, 2, 1].Should().Be(6);
    }

    [Fact]
    public void Tensor3D_Slice_Axis1()
    {
        // 2×3×2 tensor
        var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var t = new Tensor<double>(data, 2, 3, 2);

        // Slice axis 1 (middle): take rows 1-2 → 2×2×2
        var sliced = t.Slice(1, 1, 2);
        sliced.Shape.Should().Equal([2, 2, 2]);
        sliced.Length.Should().Be(8);
        // Original [0,1,0]=3, [0,1,1]=4, [0,2,0]=5, [0,2,1]=6
        sliced[0, 0, 0].Should().Be(3);
        sliced[0, 0, 1].Should().Be(4);
    }

    [Fact]
    public void Tensor3D_Slice_Axis2()
    {
        var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var t = new Tensor<double>(data, 2, 3, 2);

        // Slice axis 2: take first column → 2×3×1
        var sliced = t.Slice(2, 0, 1);
        sliced.Shape.Should().Equal([2, 3, 1]);
        sliced[0, 0, 0].Should().Be(1);
        sliced[0, 1, 0].Should().Be(3);
        sliced[0, 2, 0].Should().Be(5);
        sliced[1, 0, 0].Should().Be(7);
    }

    [Fact]
    public void Tensor3D_SumAxis0()
    {
        // 2×3×2: sum along axis 0 → 3×2
        var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var t = new Tensor<double>(data, 2, 3, 2);

        var summed = t.SumAxis(0);
        summed.Shape.Should().Equal([3, 2]);
        // [0,0,0]+[1,0,0] = 1+7 = 8
        summed[0, 0].Should().Be(8);
        // [0,0,1]+[1,0,1] = 2+8 = 10
        summed[0, 1].Should().Be(10);
        // [0,2,1]+[1,2,1] = 6+12 = 18
        summed[2, 1].Should().Be(18);
    }

    [Fact]
    public void Tensor3D_SumAxis1()
    {
        // 2×3×2: sum along axis 1 → 2×2
        var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var t = new Tensor<double>(data, 2, 3, 2);

        var summed = t.SumAxis(1);
        summed.Shape.Should().Equal([2, 2]);
        // First "matrix" rows: [1,2],[3,4],[5,6] → col sums [9, 12]
        summed[0, 0].Should().Be(9);
        summed[0, 1].Should().Be(12);
        // Second "matrix" rows: [7,8],[9,10],[11,12] → col sums [27, 30]
        summed[1, 0].Should().Be(27);
        summed[1, 1].Should().Be(30);
    }

    [Fact]
    public void Tensor3D_SumAxis2()
    {
        var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
        var t = new Tensor<double>(data, 2, 3, 2);

        var summed = t.SumAxis(2);
        summed.Shape.Should().Equal([2, 3]);
        // [1+2, 3+4, 5+6] = [3, 7, 11]
        summed[0, 0].Should().Be(3);
        summed[0, 1].Should().Be(7);
        summed[0, 2].Should().Be(11);
    }

    [Fact]
    public void Tensor3D_ArgMax_Axis0()
    {
        // 2×2×2: argmax along axis 0
        var data = new double[] { 1, 5, 3, 4, 9, 2, 7, 8 };
        var t = new Tensor<double>(data, 2, 2, 2);

        var argmax = t.ArgMax(0);
        argmax.Length.Should().Be(4); // 2×2 result
        // Position [0,0]: max(1,9)=9 at index 1
        argmax[0].Should().Be(1);
        // Position [0,1]: max(5,2)=5 at index 0
        argmax[1].Should().Be(0);
    }

    [Fact]
    public void Tensor3D_ArgMax_Axis2()
    {
        var data = new double[] { 1, 5, 3, 4, 9, 2, 7, 8 };
        var t = new Tensor<double>(data, 2, 2, 2);

        var argmax = t.ArgMax(2);
        argmax.Length.Should().Be(4); // 2×2 result
        // [0,0]: max(1,5)=5 at index 1
        argmax[0].Should().Be(1);
        // [0,1]: max(3,4)=4 at index 1
        argmax[1].Should().Be(1);
        // [1,0]: max(9,2)=9 at index 0
        argmax[2].Should().Be(0);
    }

    [Fact]
    public void Tensor2D_Slice_StillWorks()
    {
        // Verify 2D still works after generalization
        var t = new Tensor<double>([1, 2, 3, 4, 5, 6], 2, 3);
        var sliced = t.Slice(0, 0, 1); // first row
        sliced.Shape.Should().Equal([1, 3]);
        sliced[0, 0].Should().Be(1);
        sliced[0, 2].Should().Be(3);
    }

    [Fact]
    public void Tensor2D_SumAxis_StillWorks()
    {
        var t = new Tensor<double>([1, 2, 3, 4, 5, 6], 2, 3);
        var summed = t.SumAxis(0);
        summed.Shape.Should().Equal([3]);
        summed[0].Should().Be(5);  // 1+4
        summed[1].Should().Be(7);  // 2+5
        summed[2].Should().Be(9);  // 3+6
    }
}
