using Cortex;
using Cortex.Column;
using Cortex.ML.Tensors;

namespace Cortex.Text.Embeddings;

/// <summary>
/// Semantic search using cosine similarity over embedding vectors.
/// Ranks documents by similarity to a query vector.
/// </summary>
public static class SemanticSearch
{
    /// <summary>
    /// Search a corpus of document embeddings for the most similar documents to a query.
    /// </summary>
    /// <param name="query">Query embedding vector.</param>
    /// <param name="corpus">2D tensor of corpus embeddings [N, D].</param>
    /// <param name="documents">Document text/labels corresponding to each corpus row.</param>
    /// <param name="topK">Number of top results to return.</param>
    /// <returns>DataFrame with columns: rank, document, score.</returns>
    public static DataFrame Search(double[] query, Tensor<double> corpus, string[] documents, int topK)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(corpus);
        ArgumentNullException.ThrowIfNull(documents);
        if (topK < 1) throw new ArgumentOutOfRangeException(nameof(topK), "topK must be >= 1.");
        if (corpus.Rank != 2)
            throw new ArgumentException("Corpus must be a 2D tensor [N, D].");
        if (documents.Length != corpus.Shape[0])
            throw new ArgumentException("Documents length must match number of corpus rows.");
        if (query.Length != corpus.Shape[1])
            throw new ArgumentException("Query dimension must match corpus embedding dimension.");

        int n = corpus.Shape[0];
        int d = corpus.Shape[1];
        var data = corpus.ToArray();

        // Compute query norm
        double queryNorm = 0;
        for (int k = 0; k < d; k++)
            queryNorm += query[k] * query[k];
        queryNorm = Math.Sqrt(queryNorm);

        // Score each document
        var scores = new (int Index, double Score)[n];
        for (int i = 0; i < n; i++)
        {
            double dot = 0, docNorm = 0;
            int offset = i * d;
            for (int k = 0; k < d; k++)
            {
                dot += query[k] * data[offset + k];
                docNorm += data[offset + k] * data[offset + k];
            }
            docNorm = Math.Sqrt(docNorm);
            double denom = queryNorm * docNorm;
            scores[i] = (i, denom == 0 ? 0.0 : dot / denom);
        }

        // Sort by score descending and take top K
        Array.Sort(scores, (a, b) => b.Score.CompareTo(a.Score));
        int resultCount = Math.Min(topK, n);

        var ranks = new int[resultCount];
        var docs = new string?[resultCount];
        var sims = new double[resultCount];

        for (int i = 0; i < resultCount; i++)
        {
            ranks[i] = i + 1;
            docs[i] = documents[scores[i].Index];
            sims[i] = scores[i].Score;
        }

        return new DataFrame(
            new Column<int>("rank", ranks),
            new StringColumn("document", docs),
            new Column<double>("score", sims));
    }
}
