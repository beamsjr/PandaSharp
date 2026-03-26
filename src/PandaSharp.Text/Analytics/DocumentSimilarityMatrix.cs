using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Text.Embeddings;

namespace PandaSharp.Text.Analytics;

/// <summary>
/// Builds document similarity matrices using TF-IDF cosine similarity.
/// Returns a symmetric DataFrame where each cell [i,j] is the similarity between documents i and j.
/// </summary>
public static class DocumentSimilarityMatrix
{
    /// <summary>
    /// Build a TF-IDF cosine similarity matrix for a collection of documents.
    /// </summary>
    /// <param name="documents">Array of document texts.</param>
    /// <param name="labels">Optional labels for documents. Uses "doc_0", "doc_1", etc. if null.</param>
    /// <returns>DataFrame representing the symmetric similarity matrix.</returns>
    public static DataFrame Build(string[] documents, string[]? labels = null)
    {
        ArgumentNullException.ThrowIfNull(documents);
        if (documents.Length == 0)
            throw new ArgumentException("At least one document is required.", nameof(documents));

        labels ??= Enumerable.Range(0, documents.Length).Select(i => $"doc_{i}").ToArray();
        if (labels.Length != documents.Length)
            throw new ArgumentException("Labels length must match documents length.", nameof(labels));

        int n = documents.Length;

        // Tokenize documents
        var docTokens = new List<string[]>(n);
        for (int i = 0; i < n; i++)
        {
            docTokens.Add(documents[i].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
        }

        // Build vocabulary
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var tokens in docTokens)
        {
            foreach (var token in tokens)
            {
                if (!vocab.ContainsKey(token))
                    vocab[token] = vocab.Count;
            }
        }

        int vocabSize = vocab.Count;
        if (vocabSize == 0)
        {
            // All empty documents — return identity-like matrix
            return BuildEmptyMatrix(labels);
        }

        // Compute TF-IDF
        // Term frequency: count / total words in document
        var tfidf = new double[n * vocabSize];

        // Document frequency for IDF
        var df = new int[vocabSize];
        foreach (var tokens in docTokens)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var token in tokens)
            {
                if (seen.Add(token))
                    df[vocab[token]]++;
            }
        }

        // Compute IDF: log(N / df)
        var idf = new double[vocabSize];
        for (int j = 0; j < vocabSize; j++)
        {
            idf[j] = df[j] > 0 ? Math.Log((double)n / df[j]) : 0;
        }

        // Compute TF-IDF vectors
        for (int i = 0; i < n; i++)
        {
            var tokens = docTokens[i];
            if (tokens.Length == 0) continue;

            // Term frequency
            var tf = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var token in tokens)
            {
                tf.TryGetValue(token, out int count);
                tf[token] = count + 1;
            }

            int offset = i * vocabSize;
            foreach (var (token, count) in tf)
            {
                int j = vocab[token];
                tfidf[offset + j] = ((double)count / tokens.Length) * idf[j];
            }
        }

        // Compute cosine similarity matrix
        var matrix = new double[n, n];

        // Pre-compute norms
        var norms = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            int offset = i * vocabSize;
            for (int j = 0; j < vocabSize; j++)
                sum += tfidf[offset + j] * tfidf[offset + j];
            norms[i] = Math.Sqrt(sum);
        }

        for (int i = 0; i < n; i++)
        {
            matrix[i, i] = 1.0;
            for (int j = i + 1; j < n; j++)
            {
                double dot = 0;
                int offsetI = i * vocabSize;
                int offsetJ = j * vocabSize;
                for (int k = 0; k < vocabSize; k++)
                    dot += tfidf[offsetI + k] * tfidf[offsetJ + k];

                double denom = norms[i] * norms[j];
                double sim = denom == 0 ? 0.0 : dot / denom;
                matrix[i, j] = sim;
                matrix[j, i] = sim;
            }
        }

        // Build DataFrame
        var columns = new List<IColumn>();
        columns.Add(new StringColumn("document", labels));

        for (int j = 0; j < n; j++)
        {
            var colValues = new double[n];
            for (int i = 0; i < n; i++)
                colValues[i] = matrix[i, j];
            columns.Add(new Column<double>(labels[j], colValues));
        }

        return new DataFrame(columns);
    }

    private static DataFrame BuildEmptyMatrix(string[] labels)
    {
        int n = labels.Length;
        var columns = new List<IColumn>();
        columns.Add(new StringColumn("document", labels));

        for (int j = 0; j < n; j++)
        {
            var colValues = new double[n];
            for (int i = 0; i < n; i++)
                colValues[i] = i == j ? 1.0 : 0.0;
            columns.Add(new Column<double>(labels[j], colValues));
        }

        return new DataFrame(columns);
    }
}
