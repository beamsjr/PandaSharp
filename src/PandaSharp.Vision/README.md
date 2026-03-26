# PandaSharp.Vision

Image and video processing for PandaSharp with transforms, augmentation pipelines, and ONNX model inference.

## Features

- **Image transforms** — resize, crop, normalize, color conversion, and augmentation
- **Video processing** — frame extraction and batch processing via FFmpeg
- **Augmentation pipelines** — composable random transforms for training data
- **ONNX model inference** — run vision models directly on image DataFrames
- **Data loaders** — batched, parallel image loading from directories

## Installation

```bash
dotnet add package PandaSharp.Vision
```

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.Vision;

var images = ImageDataFrame.LoadFromDirectory("dataset/train",
    resize: (224, 224), normalize: true);

var augmented = images.Transform(
    new RandomHorizontalFlip(),
    new RandomRotation(degrees: 15),
    new ColorJitter(brightness: 0.2));

var model = new OnnxVisionModel("resnet50.onnx");
var predictions = model.Predict(images);
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
