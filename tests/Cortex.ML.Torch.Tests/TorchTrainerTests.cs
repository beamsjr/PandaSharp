using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Torch;
using TorchSharp;
using Xunit;

namespace Cortex.ML.Torch.Tests;

public class TorchTrainerTests
{
    /// <summary>
    /// Creates a synthetic DataFrame: y = 2*x1 + 3*x2 + 1 (with some noise).
    /// </summary>
    private static DataFrame CreateSyntheticData(int rows = 200, int? seed = 42)
    {
        var rng = new Random(seed ?? 42);
        var x1 = new double[rows];
        var x2 = new double[rows];
        var y = new double[rows];

        for (int i = 0; i < rows; i++)
        {
            x1[i] = rng.NextDouble() * 10;
            x2[i] = rng.NextDouble() * 10;
            y[i] = 2 * x1[i] + 3 * x2[i] + 1 + rng.NextDouble() * 0.1;
        }

        return new DataFrame(
            new Column<double>("x1", x1),
            new Column<double>("x2", x2),
            new Column<double>("y", y));
    }

    [Fact]
    public void Train_SimpleRegression_LossDecreases()
    {
        var df = CreateSyntheticData();
        var model = NeuralNetModels.CreateLinear(inputDim: 2, outputDim: 1);

        var config = new TrainingConfig
        {
            Epochs = 50,
            BatchSize = 32,
            LearningRate = 0.01,
            Device = "cpu",
            Seed = 42
        };

        var result = TorchTrainer.Train(
            model, df,
            featureColumns: ["x1", "x2"],
            labelColumn: "y",
            lossFunction: torch.nn.MSELoss(),
            config: config);

        result.LossHistory.Should().HaveCount(50);
        result.TotalEpochs.Should().Be(50);
        result.Duration.Should().BeGreaterThan(TimeSpan.Zero);

        // Loss should decrease over training
        double firstLoss = result.LossHistory[0];
        double finalLoss = result.FinalLoss;
        finalLoss.Should().BeLessThan(firstLoss, "training should reduce loss");
    }

    [Fact]
    public void Train_MLP_Converges()
    {
        var df = CreateSyntheticData();
        var model = NeuralNetModels.CreateMLP(
            inputDim: 2,
            hiddenDims: [16, 8],
            outputDim: 1);

        var config = new TrainingConfig
        {
            Epochs = 30,
            BatchSize = 64,
            LearningRate = 0.001,
            Device = "cpu",
            Seed = 42
        };

        var result = TorchTrainer.Train(
            model, df,
            featureColumns: ["x1", "x2"],
            labelColumn: "y",
            lossFunction: torch.nn.MSELoss(),
            config: config);

        result.FinalLoss.Should().BeLessThan(result.LossHistory[0]);
    }

    [Fact]
    public void Train_OnEpochEnd_IsCalledForEachEpoch()
    {
        var df = CreateSyntheticData(rows: 50);
        var model = NeuralNetModels.CreateLinear(2, 1);
        var epochLog = new List<(int Epoch, double Loss)>();

        var config = new TrainingConfig
        {
            Epochs = 5,
            BatchSize = 16,
            Device = "cpu",
            OnEpochEnd = (epoch, loss) => epochLog.Add((epoch, loss))
        };

        TorchTrainer.Train(
            model, df,
            featureColumns: ["x1", "x2"],
            labelColumn: "y",
            lossFunction: torch.nn.MSELoss(),
            config: config);

        epochLog.Should().HaveCount(5);
        epochLog.Select(e => e.Epoch).Should().BeEquivalentTo([0, 1, 2, 3, 4]);
    }

    [Fact]
    public void Predict_ReturnsCorrectColumnAndRowCount()
    {
        var trainDf = CreateSyntheticData(rows: 100);
        var model = NeuralNetModels.CreateLinear(2, 1);

        var config = new TrainingConfig
        {
            Epochs = 5,
            Device = "cpu"
        };

        TorchTrainer.Train(
            model, trainDf,
            featureColumns: ["x1", "x2"],
            labelColumn: "y",
            lossFunction: torch.nn.MSELoss(),
            config: config);

        // Predict on new data
        var testDf = CreateSyntheticData(rows: 20, seed: 99);
        var result = TorchTrainer.Predict(
            model, testDf,
            featureColumns: ["x1", "x2"],
            outputColumn: "pred",
            device: "cpu");

        result.RowCount.Should().Be(20);
        result.ColumnNames.Should().Contain("pred");
    }

    [Fact]
    public void Train_DeviceSelection_CpuWorks()
    {
        var df = CreateSyntheticData(rows: 30);
        var model = NeuralNetModels.CreateLinear(2, 1);

        var config = new TrainingConfig
        {
            Epochs = 2,
            Device = "cpu"
        };

        var result = TorchTrainer.Train(
            model, df,
            featureColumns: ["x1", "x2"],
            labelColumn: "y",
            lossFunction: torch.nn.MSELoss(),
            config: config);

        result.TotalEpochs.Should().Be(2);
    }
}
