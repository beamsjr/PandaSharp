using Cortex;
using Cortex.Column;
using Cortex.ML.Splitting;
using Cortex.ML.Tensors;
using Cortex.ML.Transformers;
using Cortex.ML.Pipeline;
using Cortex.ML.Metrics;

// ============================================================
// Cortex.ML — Machine Learning Pipeline Examples
// ============================================================

Console.WriteLine("Cortex.ML Examples");
Console.WriteLine("======================\n");

// --- Sample Data ---
var rng = new Random(42);
int n = 100;
var sepalLength = new double[n];
var sepalWidth = new double[n];
var petalLength = new double[n];
var labels = new string?[n];
var numericLabels = new int[n];

for (int i = 0; i < n; i++)
{
    int species = i % 3;
    sepalLength[i] = 4.5 + species * 1.5 + rng.NextDouble() * 0.8;
    sepalWidth[i] = 2.0 + species * 0.5 + rng.NextDouble() * 0.5;
    petalLength[i] = 1.0 + species * 2.0 + rng.NextDouble() * 0.6;
    labels[i] = species switch { 0 => "setosa", 1 => "versicolor", _ => "virginica" };
    numericLabels[i] = species;
}

var df = new DataFrame(
    new Column<double>("SepalLength", sepalLength),
    new Column<double>("SepalWidth", sepalWidth),
    new Column<double>("PetalLength", petalLength),
    new StringColumn("Species", labels),
    new Column<int>("Label", numericLabels)
);

Console.WriteLine("Dataset:");
Console.WriteLine(df.Head(5));
Console.WriteLine($"Shape: {df.Shape}\n");

// --- 1. Tensor Operations ---
Console.WriteLine("1. Tensor Operations:");
var tensor = df.ToTensor<double>("SepalLength", "SepalWidth", "PetalLength");
Console.WriteLine($"   Shape: [{string.Join(", ", tensor.Shape)}]");
Console.WriteLine($"   Sum: {tensor.Sum():F2}, Mean: {tensor.Mean():F2}");

var colSums = tensor.SumAxis(0);
Console.Write("   Column sums: ");
for (int i = 0; i < colSums.Length; i++) Console.Write($"{colSums[i]:F1} ");
Console.WriteLine("\n");

// --- 2. Train/Test Split ---
Console.WriteLine("2. Train/Test Split (80/20):");
var (train, test) = DataSplitting.TrainTestSplit(df, testFraction: 0.2, seed: 42);
Console.WriteLine($"   Train: {train.RowCount}, Test: {test.RowCount}\n");

// --- 3. StandardScaler ---
Console.WriteLine("3. StandardScaler:");
var numericDf = df.Select("SepalLength", "SepalWidth", "PetalLength");
var scaler = new StandardScaler();
var scaled = scaler.FitTransform(numericDf);
var sl = scaled.GetColumn<double>("SepalLength");
Console.WriteLine($"   Scaled Mean: {sl.Mean():F4}, Std: {sl.Std():F4}\n");

// --- 4. MinMaxScaler ---
Console.WriteLine("4. MinMaxScaler:");
var minmax = new MinMaxScaler();
var mmScaled = minmax.FitTransform(df.Select("SepalLength"));
var mmCol = mmScaled.GetColumn<double>("SepalLength");
Console.WriteLine($"   Min: {mmCol.Min():F4}, Max: {mmCol.Max():F4}\n");

// --- 5. LabelEncoder ---
Console.WriteLine("5. LabelEncoder:");
var encoder = new LabelEncoder();
var encoded = encoder.FitTransform(df.Select("Species"));
Console.Write("   Encoded columns: ");
Console.WriteLine(string.Join(", ", encoded.ColumnNames));
Console.Write("   Values: ");
for (int i = 0; i < Math.Min(6, encoded.RowCount); i++)
    Console.Write($"{encoded[encoded.ColumnNames[0]].GetObject(i)} ");
Console.WriteLine("\n");

// --- 6. OneHotEncoder ---
Console.WriteLine("6. OneHotEncoder:");
var onehot = new OneHotEncoder();
var oheResult = onehot.FitTransform(df.Select("Species"));
Console.WriteLine($"   Columns: {string.Join(", ", oheResult.ColumnNames)}\n");

// --- 7. Feature Pipeline ---
Console.WriteLine("7. Feature Pipeline:");
var pipeline = new FeaturePipeline(new Imputer(ImputeStrategy.Mean), new StandardScaler());
var pipeResult = pipeline.FitTransform(numericDf);
Console.WriteLine($"   Result: {pipeResult.RowCount} rows × {pipeResult.ColumnCount} cols\n");

// --- 8. KFold Cross-Validation ---
Console.WriteLine("8. KFold (5 folds):");
foreach (var (fold, foldTrain, foldVal) in DataSplitting.KFold(df, k: 5, seed: 42))
    Console.WriteLine($"   Fold {fold}: train={foldTrain.RowCount}, val={foldVal.RowCount}");
Console.WriteLine();

// --- 9. Multi-class Classification Metrics ---
Console.WriteLine("9. Classification Metrics:");
var yTrue = new Column<int>("true", [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]);
var yPred = new Column<int>("pred", [0, 0, 1, 2, 2, 2, 0, 1, 1, 0]);
var metrics = MetricsCalculator.MultiClass(yTrue, yPred);
Console.WriteLine($"   {metrics}\n");

// --- 10. Tensor MatMul ---
Console.WriteLine("10. Tensor MatMul (2×3) @ (3×2):");
var a = new Tensor<double>([1, 2, 3, 4, 5, 6], 2, 3);
var b = new Tensor<double>([7, 8, 9, 10, 11, 12], 3, 2);
var product = a.MatMul(b);
Console.Write("    Result: ");
for (int i = 0; i < product.Length; i++) Console.Write($"{product.Span[i]} ");
Console.WriteLine("\n");

// --- 11. RobustScaler ---
Console.WriteLine("11. RobustScaler:");
var robust = new RobustScaler();
var robustScaled = robust.FitTransform(df.Select("SepalLength"));
Console.WriteLine($"    Median after scaling: {robustScaled.GetColumn<double>("SepalLength").Median():F4}");

Console.WriteLine("\nAll ML examples completed!");
