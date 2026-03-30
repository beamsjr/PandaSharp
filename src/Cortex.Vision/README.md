# Cortex.Vision

Image and video processing for Cortex with transforms, augmentation pipelines, and ONNX model inference.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package. Video processing requires FFmpeg (or Apple AVFoundation on macOS).

## Features

- **Image transforms** — resize, crop, normalize, color conversion, and augmentation
- **Video processing** — frame extraction and batch processing via FFmpeg and Apple AVFoundation
- **Augmentation pipelines** — composable random transforms for training data
- **ONNX model inference** — run vision models (classification, embeddings) directly on image DataFrames
- **Data loaders** — batched, parallel image loading from directories

## Installation

```bash
dotnet add package Cortex.Vision
```

## Quick Start

```csharp
using Cortex;
using Cortex.Vision;

var images = ImageDataFrame.LoadFromDirectory("dataset/train",
    resize: (224, 224), normalize: true);

var augmented = images.Transform(
    new RandomHorizontalFlip(),
    new RandomRotation(degrees: 15),
    new ColorJitter(brightness: 0.2));

var model = new OnnxVisionModel("resnet50.onnx");
var predictions = model.Predict(images);
```

## Image Pipeline

```csharp
var pipeline = new ImagePipeline()
    .Add(new Resize(224, 224))
    .Add(new Normalize(ImageNet.Mean, ImageNet.Std));
var processed = pipeline.Transform(images);
```

## Video Frame Extraction

```csharp
var frames = VideoIO.ExtractFrames("video.mp4", fps: 1);
var classified = model.Predict(frames);
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.ML.Onnx** | General-purpose ONNX model inference |
| **Cortex.ML.Torch** | TorchSharp training for custom vision models |
| **Cortex.GPU** | GPU-accelerated image processing |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
