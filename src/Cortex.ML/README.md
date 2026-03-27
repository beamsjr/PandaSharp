# Cortex.ML

Classical machine learning toolkit for Cortex with 25+ models, transformers, metrics, and BLAS acceleration.

## Features

- **25+ ML models** — linear/logistic regression, decision trees, SVM, k-NN, random forest, and more
- **Feature engineering** — encoders, scalers, imputers, and polynomial features
- **Pipeline API** — compose preprocessing and model steps into reproducible workflows
- **Evaluation metrics** — accuracy, precision, recall, F1, ROC-AUC, RMSE, and more
- **BLAS-accelerated** tensor operations for fast training

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

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
