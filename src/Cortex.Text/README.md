# Cortex.Text

NLP pipeline for Cortex: tokenizers, stemming, embeddings, and string distance.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **Tokenizers** — whitespace, word-piece, BPE, and regex-based tokenization
- **Text preprocessing** — stemming, lemmatization, stop-word removal
- **Embeddings** — TF-IDF, bag-of-words, and pre-trained word vector loading
- **String distance** — Levenshtein, Jaro-Winkler, cosine similarity
- **N-grams** — unigram, bigram, and configurable n-gram generation
- **Column-level NLP** — apply text pipelines directly to DataFrame string columns

## Installation

```bash
dotnet add package Cortex.Text
```

## Quick Start

```csharp
using Cortex;
using Cortex.Text;

var df = DataFrame.ReadCsv("reviews.csv");

df["tokens"] = df["text"].Text.Tokenize();
df["stems"] = df["tokens"].Text.Stem();

var tfidf = new TfIdfVectorizer(maxFeatures: 500);
var matrix = tfidf.FitTransform(df["text"]);
```

## String Similarity

```csharp
var distance = StringDistance.Levenshtein("kitten", "sitting");
var similarity = StringDistance.JaroWinkler("martha", "marhta");
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.ML** | ML models for text classification |
| **Cortex.ML.Torch** | Deep learning for NLP tasks |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
