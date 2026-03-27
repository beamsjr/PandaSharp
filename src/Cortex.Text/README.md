# Cortex.Text

NLP pipeline for Cortex: tokenizers, stemming, embeddings, and string distance.

## Features

- **Tokenizers** — whitespace, word-piece, and regex-based tokenization
- **Text preprocessing** — stemming, lemmatization, stop-word removal
- **Embeddings** — TF-IDF, bag-of-words, and pre-trained word vector loading
- **String distance** — Levenshtein, Jaro-Winkler, cosine similarity
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

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
