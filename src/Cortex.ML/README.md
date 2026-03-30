# Cortex.ML

Classical machine learning toolkit for Cortex with 25+ models, transformers, metrics, and BLAS acceleration.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **25+ ML models** — linear/logistic regression, decision trees, SVM, k-NN, random forest, gradient boosted trees, KMeans, PCA, t-SNE, and more
- **Feature engineering** — encoders, scalers, imputers, and polynomial features
- **Pipeline API** — compose preprocessing and model steps into reproducible workflows
- **Evaluation metrics** — accuracy, precision, recall, F1, ROC-AUC, RMSE, R², and more
- **BLAS/LAPACK-accelerated** tensor operations for fast training
- **Tensor API** — convert DataFrames to typed tensors for numeric computation

## Installation

```bash
dotnet add package Cortex.ML
```

## Quick Start

```csharp
using Cortex;
using Cortex.ML;

var df = DataFrame.ReadCsv("iris.csv");
var (train, test) = df.TrainTestSplit(testFraction: 0.2);

var model = new RandomForestClassifier(nTrees: 100);
model.Fit(train.Drop("species"), train["species"]);

var predictions = model.Predict(test.Drop("species"));
Console.WriteLine($"Accuracy: {Metrics.Accuracy(test["species"], predictions):P2}");
```

## Regression

```csharp
using Cortex.ML.Models;
using Cortex.ML.Tensors;

var X = df.ToTensor<double>("Feature1", "Feature2", "Feature3");
var y = df.ToTensor<double>("Target");

var model = new RandomForestRegressor(nEstimators: 100, maxDepth: 10);
model.Fit(X, y);

var predictions = model.Predict(X_test);
var r2 = model.Score(X_test, y_test);
```

## Clustering and Dimensionality Reduction

```csharp
var kmeans = new KMeans(nClusters: 5);
var labels = kmeans.FitPredict(X);

var pca = new PCA(nComponents: 2);
var reduced = pca.FitTransform(X);
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.ML.Torch** | GPU training with TorchSharp neural networks |
| **Cortex.ML.Onnx** | ONNX Runtime model inference |
| **Cortex.ML.MLNet** | ML.NET IDataView bridge |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
