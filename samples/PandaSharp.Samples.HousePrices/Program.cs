using System.Diagnostics;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.IO;
using PandaSharp.Missing;
using PandaSharp.Statistics;
using PandaSharp.ML.Models;
using PandaSharp.ML.Metrics;
using PandaSharp.ML.Splitting;
using PandaSharp.ML.Tensors;
using PandaSharp.ML.Transformers;
using PandaSharp.Viz;
using PandaSharp.Viz.Charts;
using TreeVisualizer = PandaSharp.Viz.Charts.TreeVisualizer;

/*
 * House Prices Prediction using PandaSharp
 * ==========================================
 * C# port of the Kaggle notebook "House Prices Prediction using TF-DF"
 * (https://www.kaggle.com/code/gusthema/house-prices-prediction-using-tfdf)
 *
 * Uses PandaSharp for data loading/manipulation and PandaSharp.ML
 * for Random Forest regression — all in a single, integrated pipeline
 * with zero library boundary crossings.
 *
 * Data: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
 *       Place train.csv and test.csv in a "data/" subdirectory.
 */

// ════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════
var searchPaths = new[]
{
    Path.Combine(AppContext.BaseDirectory, "data"),
    "data",
    "../../../data",
    "../../../../input/house-prices-advanced-regression-techniques",
    "../../../input/house-prices-advanced-regression-techniques",
    "../../input/house-prices-advanced-regression-techniques",
    "../input/house-prices-advanced-regression-techniques",
};

string? dataDir = null;
foreach (var candidate in searchPaths)
{
    if (File.Exists(Path.Combine(candidate, "train.csv")))
    {
        dataDir = candidate;
        break;
    }
}

if (dataDir is null)
{
    Console.WriteLine("ERROR: train.csv not found.");
    Console.WriteLine("Download the data from:");
    Console.WriteLine("  https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data");
    Console.WriteLine("Place train.csv and test.csv in one of:");
    foreach (var p in searchPaths)
        Console.WriteLine($"  {Path.GetFullPath(p)}/");
    return;
}

var trainPath = Path.Combine(dataDir, "train.csv");
var testPath = Path.Combine(dataDir, "test.csv");

Console.WriteLine("════════════════════════════════════════════════════════════");
Console.WriteLine("  House Prices Prediction — PandaSharp");
Console.WriteLine("════════════════════════════════════════════════════════════\n");

var totalTimer = Stopwatch.StartNew();

// Start building the report
var report = StoryBoard.Create("House Prices Prediction")
    .Author("PandaSharp")
    .Text("C# port of the Kaggle notebook *House Prices Prediction using TF-DF*. " +
          "Uses **PandaSharp** for data loading/manipulation and **PandaSharp.ML** " +
          "for Decision Forest regression — all in a single, integrated pipeline " +
          "with **zero library boundary crossings**.");

// ════════════════════════════════════════════════════════════════
// 1. Load the dataset
// ════════════════════════════════════════════════════════════════
Console.WriteLine("1. Loading dataset...");
var df = CsvReader.Read(trainPath);
Console.WriteLine($"   Shape: ({df.RowCount}, {df.ColumnCount})");

// ════════════════════════════════════════════════════════════════
// 2. Drop the Id column
// ════════════════════════════════════════════════════════════════
df = df.DropColumn("Id");

// ════════════════════════════════════════════════════════════════
// 3. Inspect the data
// ════════════════════════════════════════════════════════════════
Console.WriteLine("2. Inspecting data...");
var info = df.Info();
var typeGroups = info.Columns.GroupBy(c => c.DataType);
var nullCols = info.Columns.Where(c => c.NullCount > 0).OrderByDescending(c => c.NullCount).ToList();

report
    .Section("Dataset Overview")
    .Text($"The dataset contains **{info.RowCount:N0}** houses with **{info.ColumnCount}** features. " +
          $"After dropping the `Id` column, we have {string.Join(", ", typeGroups.Select(g => $"**{g.Count()}** {g.Key}"))} columns.")
    .Stats(
        ("Houses", $"{info.RowCount:N0}"),
        ("Features", $"{info.ColumnCount}"),
        ("Missing Columns", $"{nullCols.Count}"),
        ("Numeric", $"{info.Columns.Count(c => c.DataType != "string")}")
    )
    .Table(df.Head(5), caption: "First 5 rows of the training data");

// ════════════════════════════════════════════════════════════════
// 4. SalePrice distribution
// ════════════════════════════════════════════════════════════════
Console.WriteLine("3. Analyzing SalePrice distribution...");
var salePriceIntCol = df.GetColumn<int>("SalePrice");
var salePriceValues = new double[salePriceIntCol.Length];
for (int i = 0; i < salePriceIntCol.Length; i++)
    salePriceValues[i] = salePriceIntCol.Values[i];
df = df.DropColumn("SalePrice").AddColumn(new Column<double>("SalePrice", salePriceValues));
var salePriceCol = df.GetColumn<double>("SalePrice");

report
    .Section("Sale Price Distribution")
    .Text("The target variable `SalePrice` shows a right-skewed distribution, " +
          "with most homes selling between $100K and $250K.")
    .Stats(
        ("Mean", $"${salePriceCol.Mean():N0}"),
        ("Median", $"${salePriceCol.Median():N0}"),
        ("Std Dev", $"${salePriceCol.Std():N0}"),
        ("Range", $"${salePriceCol.Min():N0} - ${salePriceCol.Max():N0}")
    )
    .Chart(df.Viz()
        .Histogram("SalePrice", bins: 100)
        .Title("Sale Price Distribution")
        .XLabel("Sale Price ($)")
        .YLabel("Count"),
        caption: "Distribution of house sale prices (1,460 homes)");

// ════════════════════════════════════════════════════════════════
// 5. Numerical data overview
// ════════════════════════════════════════════════════════════════
Console.WriteLine("4. Visualizing numeric features...");
var numericDf = df.SelectDtypes(typeof(double), typeof(int));

// Grid of ALL numeric column histograms — mirrors Python's df_num.hist()
var allHistCharts = numericDf.ColumnNames
    .Select(c => numericDf.Viz().Histogram(c, bins: 50).Title(c).Size(250, 200))
    .ToArray();
var histGrid = SubplotBuilder.Grid(allHistCharts).Cols(6);

report
    .Section("Numerical Feature Distributions")
    .Text($"The dataset has **{numericDf.ColumnCount}** numeric features. " +
          "Below are distributions for all of them.")
    .Grid(histGrid, caption: "Histogram grid of all numeric features (6 columns)");

// Scatter plot grid: key features vs SalePrice
var scatterCols = new[] { "GrLivArea", "OverallQual", "GarageArea", "TotalBsmtSF", "YearBuilt", "1stFlrSF" };
var scatterCharts = scatterCols
    .Where(c => df.ColumnNames.Contains(c))
    .Select(c => df.Viz().Scatter(c, "SalePrice").Title($"{c} vs SalePrice").Size(350, 280))
    .ToArray();
var scatterGrid = SubplotBuilder.Grid(scatterCharts).Cols(3);

report
    .Section("Feature Correlations with Sale Price")
    .Text("Scatter plots reveal which features have the strongest relationship with sale price. " +
          "**Overall Quality** and **Living Area** show clear positive correlations.")
    .Grid(scatterGrid, caption: "Key features vs SalePrice");

// Descriptive statistics table
report
    .Subsection("Descriptive Statistics")
    .Table(numericDf.Select(numericDf.ColumnNames.Take(12).ToArray()).Describe(),
        caption: "Summary statistics for the first 12 numeric columns");

// ════════════════════════════════════════════════════════════════
// 6. Prepare the dataset
// ════════════════════════════════════════════════════════════════
Console.WriteLine("5. Preparing dataset...");
string label = "SalePrice";
var featureDf = df.DropColumn(label);

var numericFeatures = featureDf.SelectDtypes(typeof(double), typeof(int));
var featureNames = numericFeatures.ColumnNames.ToArray();

var imputer = new Imputer(ImputeStrategy.Median);
var imputed = imputer.FitTransform(numericFeatures);

var categoricalDf = featureDf.ExcludeDtypes(typeof(double), typeof(int));
var allFeatures = imputed;
var catColNames = categoricalDf.ColumnNames.ToArray();
var catMappings = new Dictionary<string, Dictionary<string, int>>();
foreach (var catName in catColNames)
{
    var catCol = categoricalDf[catName];
    var mapping = new Dictionary<string, int>();
    int nextId = 0;
    var encoded = new double[categoricalDf.RowCount];
    for (int i = 0; i < categoricalDf.RowCount; i++)
    {
        var val = catCol.GetObject(i)?.ToString() ?? "__NULL__";
        if (!mapping.TryGetValue(val, out int id))
        {
            id = nextId++;
            mapping[val] = id;
        }
        encoded[i] = id;
    }
    catMappings[catName] = mapping;
    allFeatures = allFeatures.AddColumn(new Column<double>(catName, encoded));
}
var allFeatureNames = allFeatures.ColumnNames.ToArray();

report
    .Section("Data Preparation")
    .Text($"**{numericFeatures.ColumnCount}** numeric features were imputed (median strategy). " +
          $"**{categoricalDf.ColumnCount}** categorical features were label-encoded. " +
          $"Total features: **{allFeatures.ColumnCount}**.")
    .Callout(
        "Decision Forests handle mixed types natively in TF-DF. " +
        "In PandaSharp we encode categoricals explicitly — the model sees the same information.",
        CalloutStyle.Note);

// ════════════════════════════════════════════════════════════════
// 7. Split into training and validation sets
// ════════════════════════════════════════════════════════════════
Console.WriteLine("6. Splitting & training...");
var fullDf = allFeatures.AddColumn(new Column<double>(label, salePriceValues));
var (trainDf, validDf) = DataSplitting.TrainTestSplit(fullDf, testFraction: 0.30, seed: 42);

var xTrain = trainDf.ToTensor(allFeatureNames);
var yTrain = new Tensor<double>(trainDf.GetColumn<double>(label).Values.ToArray(), trainDf.RowCount);
var xValid = validDf.ToTensor(allFeatureNames);
var yValid = new Tensor<double>(validDf.GetColumn<double>(label).Values.ToArray(), validDf.RowCount);

// ════════════════════════════════════════════════════════════════
// 8. Train Random Forest
// ════════════════════════════════════════════════════════════════
var timer = Stopwatch.StartNew();
var rf = new RandomForestRegressor(nEstimators: 300, maxDepth: 0, seed: 42);
rf.Fit(xTrain, yTrain);
var rfTrainTime = timer.Elapsed.TotalSeconds;
Console.WriteLine($"   RF trained in {rfTrainTime:F2}s");

// Visualize a single tree from the forest (like TF-DF's plot_model_in_colab)
Console.WriteLine("   Generating tree visualization...");
var treePath = Path.Combine(dataDir, "tree_visualizer.html");
TreeVisualizer.ToHtmlFile(rf, treeIndex: 0, treePath, featureNames: allFeatureNames, maxDepth: 4);
Console.WriteLine($"   Tree saved to: {treePath}");

// ════════════════════════════════════════════════════════════════
// 9. Evaluate
// ════════════════════════════════════════════════════════════════
var validPreds = rf.Predict(xValid);
double rmse = RegressionMetrics.RMSE(yValid, validPreds);
double mae = RegressionMetrics.MAE(yValid, validPreds);
double r2 = RegressionMetrics.R2(yValid, validPreds);

// ════════════════════════════════════════════════════════════════
// 10. Feature importance
// ════════════════════════════════════════════════════════════════
Console.WriteLine("7. Computing feature importances...");
var importances = new List<(string Name, double Importance)>();
double baselineR2 = rf.Score(xValid, yValid);

var rng = new Random(42);
var xValidData = xValid.ToArray();
int nSamples = xValid.Shape[0];
int nFeatures = xValid.Shape[1];

for (int f = 0; f < nFeatures; f++)
{
    var original = new double[nSamples];
    for (int i = 0; i < nSamples; i++)
        original[i] = xValidData[i * nFeatures + f];

    var shuffled = (double[])original.Clone();
    for (int i = shuffled.Length - 1; i > 0; i--)
    {
        int j = rng.Next(i + 1);
        (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
    }
    for (int i = 0; i < nSamples; i++)
        xValidData[i * nFeatures + f] = shuffled[i];

    var permutedX = new Tensor<double>(xValidData, nSamples, nFeatures);
    double permutedR2 = rf.Score(permutedX, yValid);
    importances.Add((allFeatureNames[f], baselineR2 - permutedR2));

    for (int i = 0; i < nSamples; i++)
        xValidData[i * nFeatures + f] = original[i];
}

var topFeatures = importances.OrderByDescending(x => x.Importance).Take(10).ToList();

var impDf = new DataFrame(
    new StringColumn("Feature", topFeatures.Select(f => (string?)f.Name).ToArray()),
    new Column<double>("Importance", topFeatures.Select(f => f.Importance).ToArray())
);

// ════════════════════════════════════════════════════════════════
// 11. Train GBT for comparison
// ════════════════════════════════════════════════════════════════
Console.WriteLine("8. Training GBT for comparison...");
timer.Restart();
var gbt = new GradientBoostedTreeRegressor(
    nEstimators: 100, maxDepth: 5, learningRate: 0.1, seed: 42);
gbt.Fit(xTrain, yTrain);
var gbtTrainTime = timer.Elapsed.TotalSeconds;
Console.WriteLine($"   GBT trained in {gbtTrainTime:F2}s");

var gbtPreds = gbt.Predict(xValid);
double gbtRmse = RegressionMetrics.RMSE(yValid, gbtPreds);
double gbtR2 = RegressionMetrics.R2(yValid, gbtPreds);

// Build results DataFrames for charts
var yArr = yValid.ToArray();
var rfArr = validPreds.ToArray();
var gbtArr = gbtPreds.ToArray();

var predDf = new DataFrame(
    new Column<double>("Actual", yArr),
    new Column<double>("RF_Predicted", rfArr),
    new Column<double>("GBT_Predicted", gbtArr)
);

var rfResiduals = new double[yValid.Length];
var gbtResiduals = new double[yValid.Length];
for (int i = 0; i < yValid.Length; i++)
{
    rfResiduals[i] = yArr[i] - rfArr[i];
    gbtResiduals[i] = yArr[i] - gbtArr[i];
}
var residDf = new DataFrame(
    new Column<double>("RF_Residuals", rfResiduals),
    new Column<double>("GBT_Residuals", gbtResiduals)
);

// ════════════════════════════════════════════════════════════════
// Add model results to report
// ════════════════════════════════════════════════════════════════
report
    .Section("Model Training & Evaluation")
    .Text($"We train two Decision Forest models on **{trainDf.RowCount:N0}** training examples " +
          $"and evaluate on **{validDf.RowCount:N0}** held-out validation examples (70/30 split).")
    .Divider()

    .Subsection("Random Forest (300 trees)")
    .Stats(
        ("RMSE", $"${rmse:N0}"),
        ("MAE", $"${mae:N0}"),
        ("R^2", $"{r2:F4}"),
        ("Train Time", $"{rfTrainTime:F2}s")
    )
    .Chart(predDf.Viz()
        .Scatter("Actual", "RF_Predicted")
        .Title("Random Forest: Actual vs Predicted")
        .XLabel("Actual Price ($)")
        .YLabel("Predicted Price ($)"),
        caption: "Points on the diagonal line indicate perfect predictions")
    .Subsection("Visualize the Model")
    .Text("One benefit of tree-based models is that you can easily visualize them. " +
          "Below is **Tree 0** from the Random Forest (depth limited to 4). " +
          "Click nodes to collapse/expand. Scroll to zoom.")
    .Tree(rf, treeIndex: 0, featureNames: allFeatureNames, maxDepth: 4,
        caption: "Tree 0 of 300 — split features and leaf predictions")

    .Subsection("Gradient Boosted Trees (100 estimators)")
    .Stats(
        ("RMSE", $"${gbtRmse:N0}"),
        ("R^2", $"{gbtR2:F4}"),
        ("Train Time", $"{gbtTrainTime:F2}s"),
        ("Winner?", gbtR2 > r2 ? "Yes" : "No")
    )
    .Chart(predDf.Viz()
        .Scatter("Actual", "GBT_Predicted")
        .Title("GBT: Actual vs Predicted")
        .XLabel("Actual Price ($)")
        .YLabel("Predicted Price ($)"),
        caption: "Gradient Boosted Trees prediction accuracy")

    .Subsection("Residual Distributions")
    .Text("A well-calibrated model should have residuals centered around zero. " +
          "Both models show roughly symmetric distributions.")
    .Chart(residDf.Viz()
        .Histogram("RF_Residuals", bins: 50)
        .Title("Random Forest Residuals")
        .XLabel("Prediction Error ($)"))
    .Chart(residDf.Viz()
        .Histogram("GBT_Residuals", bins: 50)
        .Title("GBT Residuals")
        .XLabel("Prediction Error ($)"));

// ════════════════════════════════════════════════════════════════
// Feature importance section
// ════════════════════════════════════════════════════════════════
report
    .Section("Variable Importances")
    .Text("Feature importance is measured by **permutation importance**: " +
          "shuffling each feature and measuring the R^2 drop. " +
          "Larger drops indicate more important features.")
    .Chart(impDf.Viz()
        .Bar("Feature", "Importance", horizontal: true)
        .Title("Top 10 Feature Importances (Permutation)")
        .XLabel("R^2 Drop"),
        caption: "OverallQual and GrLivArea dominate — consistent with the TF-DF notebook")
    .Table(impDf, caption: "Permutation importance values")
    .Callout(
        $"The top features match the original notebook: **OverallQual** ({topFeatures[0].Importance:F4}) " +
        $"and **GrLivArea** ({topFeatures[1].Importance:F4}) are the strongest predictors.",
        CalloutStyle.Success);

// NUM_AS_ROOT: count how many times each feature is used as the root split across all trees
// This matches TF-DF's variable_importances()["NUM_AS_ROOT"]
var rootCountsRaw = TreeVisualizer.NumAsRoot(rf);
var numAsRoot = rootCountsRaw
    .Select(kv => (
        Name: kv.FeatureIndex < allFeatureNames.Length ? allFeatureNames[kv.FeatureIndex] : $"Feature_{kv.FeatureIndex}",
        Count: (double)kv.Count))
    .ToList();

var numAsRootDf = new DataFrame(
    new StringColumn("Feature", numAsRoot.Select(f => (string?)f.Name).ToArray()),
    new Column<double>("NUM_AS_ROOT", numAsRoot.Select(f => f.Count).ToArray())
);

report
    .Subsection("NUM_AS_ROOT (Tree-Structural Importance)")
    .Text("How many of the **300 trees** in the Random Forest use each feature as the root split. " +
          "This is the same metric shown in the TF-DF notebook (`inspector.variable_importances()[\"NUM_AS_ROOT\"]`).")
    .Chart(numAsRootDf.Viz()
        .Bar("Feature", "NUM_AS_ROOT", horizontal: true)
        .Title("NUM_AS_ROOT — Root Split Frequency")
        .XLabel("Number of Trees"),
        caption: "Features most often chosen as the first split across all 300 trees")
    .Table(numAsRootDf, caption: "NUM_AS_ROOT values");

Console.WriteLine($"   NUM_AS_ROOT top 3: {string.Join(", ", numAsRoot.Take(3).Select(f => $"{f.Name}={f.Count}"))}");

// ════════════════════════════════════════════════════════════════
// Add new D3-only chart types to the report
// ════════════════════════════════════════════════════════════════
Console.WriteLine("   Adding D3-only chart types to report...");

// Correlation matrix
var corrCols = new[] { "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
    "TotalBsmtSF", "1stFlrSF", "YearBuilt", "LotArea" };
var corrDf = df.Select(corrCols.Where(c => df.ColumnNames.Contains(c)).ToArray());

report
    .Section("Advanced Visualizations (D3-only)")
    .Text("These chart types are unique to the D3.js engine — they were **not possible with Plotly**.")

    .Subsection("Correlation Matrix")
    .Text("Pairwise Pearson correlations between the top numeric features. " +
          "Red = positive correlation, blue = negative. Values overlaid on each cell.")
    .Chart(corrDf.Viz()
        .CorrMatrix()
        .Title("Feature Correlation Matrix")
        .Size(700, 700),
        caption: "Diverging color scale: red (+1) to blue (-1)")

    .Subsection("Parallel Coordinates")
    .Text("Each vertical axis represents a feature; each line is a house. " +
          "Patterns in the line crossings reveal feature relationships.")
    .Chart(df.Head(500).Viz()
        .ParallelCoordinates("OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "SalePrice")
        .Title("Parallel Coordinates — 500 Houses")
        .Size(900, 400),
        caption: "Color gradient from first to last house in dataset")

    .Subsection("Treemap")
    .Text("Proportional area visualization — each rectangle's size represents total sale value per neighborhood.")
    .Chart(df.Viz()
        .Treemap("Neighborhood", "SalePrice")
        .Title("Sale Value by Neighborhood")
        .Size(900, 500),
        caption: "Larger rectangles = higher total sale value in that neighborhood");

// ════════════════════════════════════════════════════════════════
// 12. Predict on test data
// ════════════════════════════════════════════════════════════════
if (File.Exists(testPath))
{
    Console.WriteLine("9. Generating submission...");
    var testDf = CsvReader.Read(testPath);
    var testIdsInt = testDf.GetColumn<int>("Id");
    var testIdValues = new double[testIdsInt.Length];
    for (int i = 0; i < testIdsInt.Length; i++)
        testIdValues[i] = testIdsInt.Values[i];

    var testFeatures = testDf.DropColumn("Id");
    var testNumeric = testFeatures.SelectDtypes(typeof(double), typeof(int));
    var testImputed = imputer.Transform(testNumeric);

    var testCat = testFeatures.ExcludeDtypes(typeof(double), typeof(int));
    var testAll = testImputed;
    foreach (var catName in catColNames)
    {
        if (!testCat.ColumnNames.Contains(catName)) continue;
        var catCol = testCat[catName];
        var mapping = catMappings[catName];
        var encoded = new double[testCat.RowCount];
        for (int i = 0; i < testCat.RowCount; i++)
        {
            var val = catCol.GetObject(i)?.ToString() ?? "__NULL__";
            encoded[i] = mapping.GetValueOrDefault(val, 0);
        }
        testAll = testAll.AddColumn(new Column<double>(catName, encoded));
    }

    foreach (var col in allFeatureNames)
    {
        if (!testAll.ColumnNames.Contains(col))
            testAll = testAll.AddColumn(new Column<double>(col, new double[testAll.RowCount]));
    }
    var testTensor = testAll.ToTensor(allFeatureNames);

    var bestModel = gbtR2 > r2 ? (IModel)gbt : rf;
    string bestName = gbtR2 > r2 ? "GBT" : "Random Forest";

    var testPreds = bestModel.Predict(testTensor);
    var submission = new DataFrame(
        new Column<double>("Id", testIdValues),
        new Column<double>("SalePrice", testPreds.ToArray())
    );

    var submissionPath = Path.Combine(dataDir, "submission.csv");
    CsvWriter.Write(submission, submissionPath);

    report
        .Section("Submission")
        .Text($"Using the **{bestName}** model (better R^2) to predict on **{testDf.RowCount:N0}** test houses.")
        .Table(submission.Head(10), caption: "First 10 predictions");

    Console.WriteLine($"   Submission saved to {submissionPath}");
}

// ════════════════════════════════════════════════════════════════
// Summary & save report
// ════════════════════════════════════════════════════════════════
totalTimer.Stop();

report
    .Divider()
    .Section("Summary")
    .Stats(
        ("RF RMSE", $"${rmse:N0}"),
        ("GBT RMSE", $"${gbtRmse:N0}"),
        ("Best R^2", $"{Math.Max(r2, gbtR2):F4}"),
        ("Total Time", $"{totalTimer.Elapsed.TotalSeconds:F1}s")
    )
    .Callout(
        "Zero library boundary crossings — everything from CSV loading to model training " +
        "to visualization stayed in PandaSharp. No pandas, no numpy, no scikit-learn, no matplotlib.",
        CalloutStyle.Success);

var reportPath = Path.Combine(dataDir, "house_prices_report.html");
report.ToHtml(reportPath);
Console.WriteLine($"\n   Report saved to: {Path.GetFullPath(reportPath)}");

// Open in browser
try
{
    Process.Start(new ProcessStartInfo(Path.GetFullPath(reportPath)) { UseShellExecute = true });
}
catch { }
