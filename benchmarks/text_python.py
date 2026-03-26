import time, json, os, re, numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

os.makedirs("text_bench_output", exist_ok=True)
results = []
def lap(cat, name, start):
    ms = round((time.perf_counter() - start) * 1000)
    results.append({"category": cat, "op": name, "ms": ms})
    print(f"  {name:<55} {ms:>6} ms")

# Generate corpus: 10K sentences
corpus = [f"The quick brown fox jumps over the lazy dog number {i} with extra words for padding and testing tokenization" for i in range(10000)]

print("=== Python Text/NLP Benchmark ===\n")

# Tokenization
print("── Tokenization ──")
t = time.perf_counter()
tokens = [s.lower().split() for s in corpus]
lap("Tokenize", "Whitespace tokenize (10K)", t)

t = time.perf_counter()
tokens = [re.findall(r'\w+', s.lower()) for s in corpus]
lap("Tokenize", "Regex tokenize (10K)", t)

# Preprocessing
print("\n── Preprocessing ──")
stop_words = set(stopwords.words('english'))
t = time.perf_counter()
cleaned = [[w for w in s.lower().split() if w not in stop_words] for s in corpus]
lap("Preprocess", "Stop word removal (10K)", t)

stemmer = PorterStemmer()
t = time.perf_counter()
stemmed = [[stemmer.stem(w) for w in s.lower().split()] for s in corpus]
lap("Preprocess", "Porter stemming (10K)", t)

# N-grams
t = time.perf_counter()
for s in corpus:
    words = s.lower().split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
lap("Preprocess", "Bigram extraction (10K)", t)

# Cosine similarity
print("\n── Similarity ──")
dim = 384
emb_a = np.random.randn(1000, dim).astype(np.float64)
emb_b = np.random.randn(1000, dim).astype(np.float64)

t = time.perf_counter()
norms_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
norms_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
sim_matrix = (emb_a / norms_a) @ (emb_b / norms_b).T
lap("Similarity", "Pairwise cosine (1Kx1K, 384-dim)", t)

# Summary
print(f"\n{'═'*70}")
cats = {}
for r in results: cats[r["category"]] = cats.get(r["category"], 0) + r["ms"]
for c, m in sorted(cats.items(), key=lambda x: -x[1]): print(f"  {c:<30} {m:>8,} ms")
total = sum(r["ms"] for r in results)
print(f"  {'TOTAL':<30} {total:>8,} ms")
with open("text_bench_output/python_text_results.json", "w") as f: json.dump(results, f, indent=2)
