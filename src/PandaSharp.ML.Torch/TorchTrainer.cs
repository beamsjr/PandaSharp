using System.Diagnostics;
using PandaSharp;
using PandaSharp.Column;
using TorchSharp;
using static TorchSharp.torch;

namespace PandaSharp.ML.Torch;

/// <summary>
/// Configuration for the training loop.
/// </summary>
public class TrainingConfig
{
    /// <summary>Number of training epochs.</summary>
    public int Epochs { get; set; } = 10;

    /// <summary>Mini-batch size.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>Optimizer learning rate.</summary>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Target device: "auto" (best available), "cuda", "mps", or "cpu".
    /// </summary>
    public string Device { get; set; } = "auto";

    /// <summary>Whether to shuffle data each epoch.</summary>
    public bool Shuffle { get; set; } = true;

    /// <summary>Optional random seed for reproducibility.</summary>
    public int? Seed { get; set; }

    /// <summary>L2 regularization weight decay.</summary>
    public double WeightDecay { get; set; } = 0;

    /// <summary>
    /// Optional callback invoked at the end of each epoch with (epoch, averageLoss).
    /// </summary>
    public Action<int, double>? OnEpochEnd { get; set; }
}

/// <summary>
/// Result of a completed training run.
/// </summary>
/// <param name="LossHistory">Average loss per epoch.</param>
/// <param name="FinalLoss">Loss of the last epoch.</param>
/// <param name="TotalEpochs">Number of epochs completed.</param>
/// <param name="Duration">Wall-clock training time.</param>
public record TrainingResult(double[] LossHistory, double FinalLoss, int TotalEpochs, TimeSpan Duration);

/// <summary>
/// High-level training and prediction utilities that integrate PandaSharp DataFrames
/// with TorchSharp models, including automatic device management and batching.
/// </summary>
public static class TorchTrainer
{
    /// <summary>
    /// Train a TorchSharp module on DataFrame data with automatic batching and device management.
    /// </summary>
    /// <param name="model">The model to train. Must accept a 2D float tensor and return a tensor.</param>
    /// <param name="trainDf">Training data as a DataFrame.</param>
    /// <param name="featureColumns">Column names to use as input features.</param>
    /// <param name="labelColumn">Column name for the target/label.</param>
    /// <param name="lossFunction">Loss function module (e.g., MSELoss, CrossEntropyLoss).</param>
    /// <param name="config">Training configuration. Uses defaults if null.</param>
    /// <returns>A <see cref="TrainingResult"/> with loss history and timing information.</returns>
    public static TrainingResult Train(
        nn.Module<Tensor, Tensor> model,
        DataFrame trainDf,
        string[] featureColumns,
        string labelColumn,
        nn.Module<Tensor, Tensor, Tensor> lossFunction,
        TrainingConfig? config = null)
    {
        config ??= new TrainingConfig();
        var device = TorchDevice.Resolve(config.Device);

        if (config.Seed.HasValue)
            torch.manual_seed(config.Seed.Value);

        // Move model to target device
        model.to(device);

        var optimizer = torch.optim.Adam(
            model.parameters(),
            lr: config.LearningRate,
            weight_decay: config.WeightDecay);

        var lossHistory = new List<double>();
        var sw = Stopwatch.StartNew();

        for (int epoch = 0; epoch < config.Epochs; epoch++)
        {
            model.train();
            double epochLoss = 0;
            int batchCount = 0;

            foreach (var (featureBatch, labelBatch) in BatchIterator(
                         trainDf, featureColumns, labelColumn,
                         config.BatchSize, config.Shuffle, config.Seed))
            {
                using var features = featureBatch.to(device);
                using var labels = labelBatch.to(device);

                optimizer.zero_grad();

                using var predictions = model.call(features);

                // Reshape labels to match predictions if needed
                using var labelsReshaped = labels.dim() == 1 && predictions.dim() == 2 && predictions.shape[1] == 1
                    ? labels.unsqueeze(1)
                    : labels.alias();

                using var loss = lossFunction.call(predictions, labelsReshaped);
                loss.backward();
                optimizer.step();

                epochLoss += loss.item<float>();
                batchCount++;
            }

            double avgLoss = batchCount > 0 ? epochLoss / batchCount : 0;
            lossHistory.Add(avgLoss);
            config.OnEpochEnd?.Invoke(epoch, avgLoss);
        }

        sw.Stop();

        var history = lossHistory.ToArray();
        return new TrainingResult(
            history,
            history.Length > 0 ? history[^1] : 0,
            config.Epochs,
            sw.Elapsed);
    }

    /// <summary>
    /// Run prediction on a DataFrame using a trained model, returning results as a new DataFrame.
    /// </summary>
    /// <param name="model">The trained model.</param>
    /// <param name="df">Input data.</param>
    /// <param name="featureColumns">Feature column names matching training.</param>
    /// <param name="outputColumn">Name for the prediction output column.</param>
    /// <param name="device">Device string ("auto", "cuda", "mps", "cpu"). Null means auto.</param>
    /// <returns>A DataFrame containing the prediction column.</returns>
    public static DataFrame Predict(
        nn.Module<Tensor, Tensor> model,
        DataFrame df,
        string[] featureColumns,
        string outputColumn = "prediction",
        string? device = null)
    {
        var dev = TorchDevice.Resolve(device);
        model.to(dev);
        model.eval();

        using var noGrad = torch.no_grad();
        using var featureTensor = df.ToTorchTensor(featureColumns).to(dev);
        using var output = model.call(featureTensor);
        using var cpuOutput = output.cpu().to(ScalarType.Float64);

        int rows = (int)cpuOutput.shape[0];

        if (cpuOutput.dim() == 1 || (cpuOutput.dim() == 2 && cpuOutput.shape[1] == 1))
        {
            // Single output column
            using var flat = cpuOutput.reshape(rows);
            var data = flat.data<double>().ToArray();
            var col = new Column<double>(outputColumn, data);
            return new DataFrame([col]);
        }
        else
        {
            // Multiple output columns
            int cols = (int)cpuOutput.shape[1];
            var columns = new List<IColumn>();
            for (int c = 0; c < cols; c++)
            {
                using var colTensor = cpuOutput[.., c];
                var data = colTensor.data<double>().ToArray();
                columns.Add(new Column<double>($"{outputColumn}_{c}", data));
            }
            return new DataFrame(columns);
        }
    }

    /// <summary>
    /// Iterates over the DataFrame in batches, yielding (features, labels) tensor pairs.
    /// </summary>
    private static IEnumerable<(Tensor Features, Tensor Labels)> BatchIterator(
        DataFrame df, string[] featureColumns, string labelColumn,
        int batchSize, bool shuffle, int? seed)
    {
        int rows = df.RowCount;
        int[] indices = Enumerable.Range(0, rows).ToArray();

        if (shuffle)
        {
            var rng = seed.HasValue ? new Random(seed.Value) : new Random();
            // Fisher-Yates shuffle
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        int featureCols = featureColumns.Length;

        for (int start = 0; start < rows; start += batchSize)
        {
            int end = Math.Min(start + batchSize, rows);
            int count = end - start;

            var featureData = new float[count * featureCols];
            var labelData = new float[count];

            for (int i = 0; i < count; i++)
            {
                int rowIdx = indices[start + i];

                for (int c = 0; c < featureCols; c++)
                {
                    var col = df[featureColumns[c]];
                    featureData[i * featureCols + c] = col.IsNull(rowIdx)
                        ? 0f
                        : (float)TypeHelpers.GetDouble(col, rowIdx);
                }

                var labelCol = df[labelColumn];
                labelData[i] = labelCol.IsNull(rowIdx)
                    ? 0f
                    : (float)TypeHelpers.GetDouble(labelCol, rowIdx);
            }

            yield return (
                torch.tensor(featureData, [count, featureCols]),
                torch.tensor(labelData, [count])
            );
        }
    }
}
