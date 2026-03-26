# PandaSharp — Killer Features Roadmap

> **Performance mandate:** Every feature below MUST target best-in-class performance.
> Use SIMD via `Vector<T>` for bulk numeric loops, Span<T>/stackalloc for short-lived buffers,
> and P/Invoke to hand-tuned C/C++ kernels for any hot path where managed code cannot match
> native throughput (e.g., string distance algorithms, sort-heavy analytics, rolling windows).
> All benchmarks must compare against the equivalent pandas/NumPy/SciPy operation.

## 1. DataFrame Introspection & Profiling

- [x] **`DataFrame.Explain()` — Execution plan visualization**
  Show column types, memory layout, null counts, Arrow chunk boundaries, and estimated cost
  for pending operations (lazy eval hint). Returns a structured `ExecutionPlan` record AND
  a pretty-printed string.
  - [x] Test: Verify output on a 5-column mixed-type DataFrame; assert correct types, null counts, chunk counts
  - [ ] Benchmark: Profile `Explain()` on 10M-row / 50-column DataFrame — must complete < 5 ms (metadata scan only, no data copy)
  - **Perf notes:** Walk Arrow chunk metadata pointers only; never materialise column data. Use `Unsafe.As` for type checks. O(columns), not O(rows).

- [x] **`DataFrame.Benchmark(Action<DataFrame>)` — Operation timing / profiling**
  Run a user-supplied lambda N times, return min / median / p95 / p99 / allocations.
  Automatically warms up JIT (3 iterations discarded). Returns `BenchmarkResult` record
  with `.ToDataFrame()`.
  - [x] Test: Benchmark a trivial Select; verify result contains all stat fields and iteration count
  - [ ] Benchmark: Ensure overhead of the harness itself is < 1 µs per iteration (measure empty lambda)
  - **Perf notes:** Use `Stopwatch.GetTimestamp()` (no allocation). Track GC counts via `GC.CollectionCount()`. Use `ArrayPool<double>` for timing storage.

- [x] **`DataFrame.Profile()` — Unified statistical profile across modules**
  Single call returns count, mean, std, min, quartiles, max, null%, unique count, top-N
  frequent values, dtype, memory bytes — for every column. Returns a new DataFrame
  (one row per stat, one column per source column).
  - [x] Test: Profile a 4-column DataFrame (int, double, string, bool); verify all stat rows present and correct
  - [ ] Benchmark: Profile on 1M rows / 20 columns — target < 200 ms. Compare vs `pandas.DataFrame.describe()`
  - **Perf notes:** Compute min/max/mean/count in a single SIMD pass per numeric column. Use HyperLogLog for approximate distinct counts on large columns. Histograms via fixed-width binning with `Vector<T>` comparisons.

## 2. Advanced Statistics

- [x] **Spearman rank correlation**
  `df.CorrSpearman()` — rank-transform then Pearson. Returns a correlation-matrix DataFrame.
  - [x] Test: Verify against known Spearman coefficients for a 3-column dataset
  - [ ] Benchmark: 1M rows × 10 columns — target < 500 ms. Compare vs `scipy.stats.spearmanr`
  - **Perf notes:** Rank-transform in-place using an `int[]` index sort (IntroSort). Pearson on ranks uses SIMD dot-product. For ties, use average-rank with a single scan. Consider C kernel for the sort if > 5M rows.

- [x] **Kendall tau correlation**
  `df.CorrKendall()` — count concordant/discordant pairs. Returns correlation-matrix DataFrame.
  - [x] Test: Verify against known Kendall tau values including tie correction
  - [ ] Benchmark: 100K rows × 5 columns — target < 2 s. Compare vs `scipy.stats.kendalltau`
  - **Perf notes:** Use Knight's O(n log n) merge-sort algorithm, not naïve O(n²). Implement in C for the inner merge-count loop — managed overhead on the tight comparison loop is measurable at >100K rows.

## 3. GroupBy Extensions

- [x] **`GroupBy.Shift(int periods)` — Lag/lead within groups**
  Shift values up or down within each group, filling with null. Returns a full-length column
  aligned to the original DataFrame index.
  - [x] Test: Shift(1) and Shift(-1) on a 3-group dataset; verify nulls at group boundaries
  - [ ] Benchmark: 5M rows, 10K groups — target < 100 ms. Compare vs `pandas.GroupBy.shift()`
  - **Perf notes:** Pre-compute group offsets via a single sort pass. Use `Buffer.BlockCopy` / `Span.CopyTo` for bulk moves within each group segment. Avoid per-row dictionary lookups.

- [x] **`GroupBy.PctChange(int periods)` — Percentage change within groups**
  `(current - previous) / previous` within each group. Returns full-length Column<double>.
  - [x] Test: PctChange(1) on known data; verify NaN at group starts and correct percentages
  - [ ] Benchmark: 5M rows, 10K groups — target < 150 ms. Compare vs `pandas.GroupBy.pct_change()`
  - **Perf notes:** Fuse with Shift — compute shift and division in a single pass over each group segment. Use SIMD `Vector<double>` for the division loop.

## 4. Rolling / Window Extensions

- [x] **Rolling with `center=true` alignment**
  Center the window so the result aligns with the middle of the window, not the right edge.
  Applies to all rolling aggregations (Mean, Sum, Std, Min, Max).
  - [x] Test: Rolling(5, center=true).Mean() on 10 values; verify alignment offset and NaN placement
  - [ ] Benchmark: 10M rows rolling(100, center=true).Mean() — target < 80 ms. Compare vs `pandas.rolling(center=True).mean()`
  - **Perf notes:** Center alignment is just an index offset on the output — do NOT re-sort or re-copy data. Compute using the same SIMD sliding-window kernel as non-centered, then shift the output pointer.

## 5. Null Handling

- [x] **`DataFrame.CombineFirst(DataFrame other)` — Priority null filling**
  For each cell, use `this` value if non-null, else fall back to `other`. Column-aligned by name.
  Like pandas `combine_first`.
  - [x] Test: Two DataFrames with complementary nulls; verify merged result has no nulls where either had a value
  - [ ] Benchmark: 5M rows × 10 columns — target < 50 ms. Compare vs `pandas.DataFrame.combine_first()`
  - **Perf notes:** Operate on the null bitmap directly — `AND NOT` the validity bitmaps, then `Vector.ConditionalSelect` for value arrays. Pure bitwise ops, no per-cell branching.

## 6. ML Transformers (PandaSharp.ML)

- [x] **QuantileTransformer — Map to uniform or normal distribution**
  Fits quantiles from training data; transforms by interpolating into the target distribution.
  Robust to outliers.
  - [x] Test: Transform a skewed column; verify output distribution is approximately uniform (KS-test p > 0.05)
  - [ ] Benchmark: Fit + Transform on 1M rows — target < 300 ms. Compare vs `sklearn.preprocessing.QuantileTransformer`
  - **Perf notes:** Sort once during Fit (use IntroSort). Transform uses binary search on quantile edges — implement in C for cache-friendly binary search on large arrays. Use SIMD linear interpolation between quantile boundaries.

- [x] **PowerTransformer (Box-Cox & Yeo-Johnson)**
  Stabilize variance and make data more Gaussian-like. Box-Cox requires positive data;
  Yeo-Johnson handles negatives.
  - [x] Test: Transform a log-normal column with Box-Cox; verify output skewness ≈ 0
  - [ ] Benchmark: Fit + Transform on 1M rows — target < 200 ms. Compare vs `sklearn.preprocessing.PowerTransformer`
  - **Perf notes:** Lambda estimation via Brent's method (bounded scalar optimization). Transform loop uses `Math.Pow` — for large arrays, batch via a C kernel using `pow()` from libm, which can auto-vectorize on modern compilers.

## 7. ML Ensembles (PandaSharp.ML)

- [x] **VotingEnsemble — Hard/soft voting across models**
  Combine predictions from multiple `IPredictor` instances. Hard voting = majority class;
  soft voting = average predicted probabilities.
  - [x] Test: 3 dummy predictors with known outputs; verify hard vote = majority, soft vote = weighted average
  - [ ] Benchmark: Voting over 5 models × 100K rows — target < 50 ms (excluding individual model inference)
  - **Perf notes:** Predict all models in parallel (`Parallel.For` over models). Accumulate votes in a `Span<int>` (hard) or `Span<double>` (soft). Final argmax via SIMD comparison scan.

- [x] **StackingEnsemble — Meta-learner on base model outputs**
  Train K base models, generate out-of-fold predictions, train a meta-model on stacked features.
  Supports arbitrary `IPredictor` as base and meta learners.
  - [x] Test: Stack 2 base models with a linear meta-learner; verify meta-model trains on correct fold structure
  - [ ] Benchmark: 5-fold stacking with 3 base models on 50K rows — target < 2 s (excluding base model training)
  - **Perf notes:** Out-of-fold predictions are embarrassingly parallel — train folds concurrently. Use `ArrayPool<double>` for fold prediction buffers. Minimize DataFrame copies by using column views.

## 8. ML Metrics → DataFrame (PandaSharp.ML)

- [x] **Confusion matrix as DataFrame**
  `ClassificationResult.ToDataFrame()` and `MultiClassResult.ToDataFrame()` — return the
  confusion matrix as a labeled DataFrame with row/column headers = class labels.
  - [x] Test: Verify 2×2 and 4×4 confusion matrices produce correctly labeled DataFrames
  - [ ] Benchmark: N/A (matrix is small) — verify zero allocation beyond the DataFrame itself
  - **Perf notes:** Trivial hot path — just ensure no unnecessary copies. Flatten `int[,]` to `int[]` for Column<int> construction.

## 9. Text / NLP (PandaSharp.ML)

- [x] **Fuzzy string matching — Levenshtein distance**
  `StringColumn.Levenshtein(string target)` → Column<int> of edit distances.
  `StringColumn.LevenshteinMatch(string target, int maxDistance)` → Column<bool>.
  - [x] Test: Known string pairs with expected edit distances (including empty strings, equal strings, Unicode)
  - [ ] Benchmark: 1M strings × average length 20 — target < 500 ms. Compare vs `python-Levenshtein`
  - **Perf notes:** **Must use C native kernel.** Classic DP algorithm with a single-row buffer (2×n allocation). For batch operation, use a thread-per-chunk partitioning. The inner loop is pure integer ops — a C implementation will be 3-5× faster than managed due to bounds-check elimination and auto-vectorization.

- [x] **Fuzzy string matching — Jaro-Winkler similarity**
  `StringColumn.JaroWinkler(string target)` → Column<double> of similarity scores [0, 1].
  - [x] Test: Known pairs with expected Jaro-Winkler scores; verify winkler prefix bonus
  - [ ] Benchmark: 1M strings × average length 15 — target < 400 ms. Compare vs `jellyfish.jaro_winkler_similarity`
  - **Perf notes:** **Must use C native kernel.** The matching-character scan and transposition count are branch-heavy — C compilers optimize this significantly better than JIT. Use `stackalloc bool[]` for the match flags (strings < 256 chars) to avoid heap allocation.

## 10. DataFrame Comparison

- [x] **`DataFrame.Compare(DataFrame other)` — Structural diff**
  Return a DataFrame showing only cells that differ, with columns `{col}_self` and `{col}_other`.
  Optionally align by index column. Like pandas `DataFrame.compare()`.
  - [x] Test: Two DataFrames differing in 3 cells; verify output contains exactly those 3 differences
  - [ ] Benchmark: 1M rows × 10 columns with 1% differences — target < 200 ms. Compare vs `pandas.DataFrame.compare()`
  - **Perf notes:** Compare columns element-wise using SIMD `Vector.Equals` for numeric types, producing a bool mask. Only materialise diff rows (sparse output). For strings, use `MemoryExtensions.Equals` with `Ordinal` comparison.

## 11. String Operations

- [x] **Case-insensitive string operations**
  `StringColumn.ContainsIgnoreCase(string)`, `.StartsWithIgnoreCase()`, `.EndsWithIgnoreCase()`,
  `.ReplaceIgnoreCase()`. All return Column<bool> or StringColumn.
  - [x] Test: Mixed-case strings with case-insensitive Contains; verify matches regardless of case
  - [ ] Benchmark: 1M strings × average length 30 — target < 150 ms. Compare vs `pandas.str.contains(case=False)`
  - **Perf notes:** Use `StringComparison.OrdinalIgnoreCase` which is already optimized in .NET. For bulk search, consider pre-lowering the target once and using `Span<char>` scanning with `MemoryExtensions.Contains`. For regex-backed case-insensitive, use `RegexOptions.Compiled | RegexOptions.IgnoreCase`.

- [x] **Unicode normalization**
  `StringColumn.NormalizeUnicode(NormalizationForm form)` — NFC, NFD, NFKC, NFKD.
  Critical for internationalized data pipelines.
  - [x] Test: Composed vs decomposed forms (é = e+combining accent); verify NFC round-trip
  - [ ] Benchmark: 500K strings with mixed Unicode — target < 300 ms. Compare vs `unicodedata.normalize` in Python
  - **Perf notes:** Use `string.Normalize()` from BCL (backed by ICU). For large batches, parallelize with `Parallel.ForEach` over 64K-row chunks. Pre-check ASCII-only strings (skip normalization) using `Ascii.IsValid(span)` from .NET 8+.

## 12. I/O Formats

- [x] **Apache Avro reader/writer**
  `DataFrame.ReadAvro(path)` / `df.WriteAvro(path)`. Support schema evolution (added/removed columns).
  - [x] Test: Round-trip a 5-column DataFrame through Avro; verify data integrity and schema
  - [ ] Benchmark: Read/Write 1M rows × 10 columns — target < 1 s. Compare vs `pandas.read_avro` (via fastavro)
  - **Perf notes:** Use the Apache Avro C# library (`Apache.Avro`). For write, batch-encode using `BinaryEncoder` with a pre-allocated `MemoryStream` (avoid per-record allocation). For read, use `GenericDatumReader` with pre-allocated reusable record objects.

- [x] **Apache ORC reader/writer**
  `DataFrame.ReadOrc(path)` / `df.WriteOrc(path)`. Columnar format with built-in compression.
  - [x] Test: Round-trip with multiple column types; verify data, null handling, and compression
  - [ ] Benchmark: Read/Write 5M rows × 10 columns — target < 2 s. Compare vs `pandas.read_orc` (via pyarrow)
  - **Perf notes:** Use `ApacheOrc.net` or P/Invoke to the C++ ORC library for maximum throughput. ORC's columnar layout maps naturally to Arrow columns — use zero-copy where buffer formats align. Stripe-level parallelism for reads.

## 13. Database I/O

- [x] **Connection string builder with dialect support**
  `DatabaseConnectionBuilder` fluent API: `.ForPostgres()`, `.ForSqlServer()`, `.ForSqlite()`,
  `.ForMySql()`. Generates correct connection strings and selects the right ADO.NET provider.
  - [x] Test: Build connection strings for each dialect; verify format matches expected patterns
  - [ ] Benchmark: N/A (configuration utility) — verify zero allocation beyond the final string
  - **Perf notes:** Use `StringBuilder` with pre-sized capacity. Cache provider factory lookups in a `ConcurrentDictionary`.

## 14. Cloud Storage Resilience

- [x] **Retry with exponential backoff + circuit breaker for cloud I/O**
  Wrap S3/Azure/GCS operations in a resilience pipeline: retry transient failures with jitter,
  circuit-break after N consecutive failures. Configurable via `CloudStorageOptions`.
  - [x] Test: Mock a failing S3 client; verify retry count, backoff delays, and circuit-breaker trip
  - [ ] Benchmark: Measure overhead of resilience wrapper on a no-op call — target < 5 µs per invocation
  - **Perf notes:** Use `Polly.Core` (v8+) for zero-allocation resilience pipeline. Pre-build the pipeline once and reuse. Jitter via `Random.Shared` (thread-safe, no lock).

## 15. Interactive Exploration

- [x] **`DataFrame.Explore()` — Interactive web UI**
  Launch a local Kestrel server serving a single-page app: sortable/filterable table,
  column histograms, scatter plots, and correlation heatmap. Opens default browser.
  - [x] Test: Verify server starts, returns HTML, table contains correct data for a 5-row DataFrame
  - [ ] Benchmark: Serve a 100K-row DataFrame — page load < 500 ms (virtual scrolling, only render visible rows)
  - **Perf notes:** Serialize to JSON using `System.Text.Json.Utf8JsonWriter` (zero-copy for double[] columns). Use virtual scrolling (only send visible page to browser). Histograms computed server-side via SIMD binning. WebSocket for live filtering. Consider `MessagePack` for binary transport if JSON serialization becomes the bottleneck.

## 16. Improvements

- [x] CrossValidation.SliceRows() creates new double[rowIndices.Length * cols] — could use index-based views over the original DataFrame instead of materializing a copy
- [x] GridSearchCV stores results internally as List<HyperparameterResult> then rebuilds arrays via .Select().ToArray() in the AllResults property — should build the DataFrame incrementally or cache it
- [x] TextVectorizer.Transform() creates a flat new double[rows * vocabularyLength] term-document matrix — for sparse text data this is extremely wasteful, should use a sparse representation or build columns directly
- [x] Backtesting.Evaluate() calls TypeHelpers.GetDoubleArray(df[valueColumn]) to pull the whole series into a double[] before windowing — could slice the column directly
- [x] GeoExtensions.ToGeoColumn() calls latCol.Values.ToArray() and lonCol.Values.ToArray() — copies both coordinate arrays when it could reference the underlying Arrow buffers
- [x] ImageClassifier.Predict() creates new string?[] and new double[] intermediaries before building the result DataFrame — could build columns directly
- [x] CosineSimilarity returns raw arrays from cosine operations instead of Column/DataFrame — reviewed, API is already appropriate (Compute returns double, PairwiseDataFrame returns DataFrame)

## 17. Code Review — Bugs (14)

- [x] DescribeViaDouble null corruption — null values default to 0.0 in double array, biasing min/max/mean. Needs compaction before stats.
- [x] DescribeViaDouble quantile bounds — QuickSelect receives full array including nulls but k computed from non-null count. Potential OOB.
- [x] Rank average formula (RankExtensions.cs:57-65) — `(i + 1 + j) / 2.0` doesn't correctly average 1-indexed positions for tied groups.
- [x] Sigmoid epsilon inconsistency (SGDClassifier + LogisticRegression) — `Math.Log(prob + 1e-15)` but Sigmoid can produce exact 0.0/1.0. Need `Math.Clamp(Sigmoid(z), 1e-15, 1-1e-15)`.
- [x] CSV hardcoded quote character (CsvReader.cs:520) — uses `'"'` instead of configurable `options.QuoteChar`.
- [x] CSV quote escaping (CsvWriter.cs:70) — doesn't handle backslash-quote `\"` in input.
- [x] Parquet row count accumulation (ParquetIO.cs:59-60) — counts rows inside field loop, over-counting across row groups.
- [x] ARIMA double-differencing integration (ARIMA.cs:259-268) — cumulative sum not maintained correctly across differencing levels.
- [x] SARIMA non-seasonal integration (SARIMA.cs:230-237) — doesn't respect original series for known portion, causing forecast divergence.
- [x] SafeTensor memory leak (SafeTensorReader.cs:44-49) — AcquirePointer() never calls ReleasePointer() on exception paths.
- [x] SafeTensor header size validation (SafeTensorReader.cs:65-66) — negative headerSize passes unsigned comparison on malformed files.
- [x] ImageIO parallel race condition (ImageIO.cs:53-74) — shared batchData array written without synchronization in Parallel.For.
- [x] Normalize division by zero (Normalize.cs:56-58) — `1f / _std[c]` produces Infinity if std is zero. No clamp.
- [x] KNN division by zero (KNearestNeighborsRegressor.cs:178-190) — all distances exactly 0, weighted average silently returns 0 instead of mean.

## 18. Code Review — Numerical Stability (7)

- [x] LinearRegression singular matrix tolerance (1e-15) — doesn't scale with matrix magnitude. Ill-conditioned problems fail unnecessarily.
- [x] PCA negative eigenvalues — clamps to 0 silently, losing variance info. SVD on data directly would avoid this.
- [x] Levinson-Durbin hard threshold (ARIMA.cs:305) — breaks at `den < 1e-15` absolute. Should be relative to acf[0].
- [x] ExponentialSmoothing zero seasonal — multiplicative mode clamps near-zero seasonality to 1e-15, permanently altering component.
- [x] ExponentialSmoothing zero initial level — if `_level == 0`, seasonal components silently become 1.0.
- [x] SeasonalDecompose division by near-zero mean — multiplicative normalization doesn't check for mean ≈ 0.
- [x] TF-IDF term frequency not normalized (TextVectorizer.cs:106-107) — uses raw counts instead of `count / words.Length`.

## 19. Code Review — Performance (11)

- [x] Rolling window LINQ overhead (RollingWindow.cs:29-31) — `vals.Sum()/Min()/Max()` via LINQ called millions of times.
- [x] DataFrame.Filter double allocation (DataFrame.cs:170-189) — allocates full RowCount then copies to exact-size.
- [x] Reshape triple-scan (ReshapeExtensions.cs:18-50) — Pivot iterates RowCount 3 times. Could be one pass.
- [x] DescribeExtensions parallel allocation — `double[][]` results for wide DataFrames.
- [x] StringColumn dict caching race — `Volatile.Read/Write` isn't atomic. Two threads can both compute encoding.
- [x] CSV gzip memory doubling (CsvReader.cs:68-72) — reads entire file into MemoryStream for gzip. Should stream.
- [x] DocumentSimilarityMatrix O(n²) — computes full matrix but doesn't exploit symmetry. 50% wasted.
- [x] SafeTensor JSON triple-GetProperty — calls GetProperty() 3 times per tensor entry. Should cache.
- [x] NGramExtractor flat array — allocates dense `rows * vocab` array for sparse n-gram data.
- [x] CrossValidation.SliceRows intermediate arrays — creates new double[] copies instead of index-based views.
- [x] GridSearchCV List→Array rebuilds — `.Select().ToArray()` rebuilds on every access. Should cache.

## 20. Code Review — Edge Cases (15)

- [x] QuickSelect OOB — k can equal count when percent=1.0, no bounds clamp.
- [x] NaN min/max — all-NaN column returns MaxValue/MinValue instead of NaN.
- [x] ILocAccessor no bounds check — indices ≥ RowCount give unhelpful IndexOutOfRangeException.
- [x] Row count int overflow (ConcatExtensions.cs:34-35) — int totalRows overflows for >2.1B rows.
- [x] GroupedDataFrame missing column check — throws KeyNotFoundException instead of ArgumentException.
- [x] KNN empty neighbors — MaxBy() throws on empty array if K > training size.
- [x] DBSCAN empty input — no validation for 0-row input.
- [x] AutoARIMA skips (0,d,0) — `if (p==0 && q==0) continue` skips valid random walk model.
- [x] ARIMA insufficient data check — doesn't catch all cases where lags exceed available data after differencing.
- [x] Antimeridian not handled (GeoPolygon.cs:120-124) — ray-casting wrong near ±180° longitude.
- [x] Pole area calculation (GeoPolygon.cs:79) — Shoelace in degrees inaccurate at high latitudes.
- [x] RTree antimeridian clamp — Math.Clamp fails when MinLon > MaxLon across dateline.
- [x] ImageTensor rank-5 accessors — Height/Width assume rank ≤ 4, rank-5 throws OOB.
- [x] Resize bilinear boundary — clamping before interpolation weights loses precision at large scale factors.
- [x] WordPiece unknown handling — emits [UNK] for entire remaining word instead of character-by-character.

## 21. Code Review — Consistency (8)

- [x] DictEncoding null = empty string — null and "" indistinguishable after encoding.
- [x] CategoricalColumn null count methods — different calculation across Slice/Filter/TakeRows.
- [x] JsonReader int.Parse missing CultureInfo — uses current thread culture, others use InvariantCulture.
- [x] Coordinate order mismatch (CoordinateTransform.cs:35-37) — ProjNet (x=lon,y=lat) vs GeoPoint (lat,lon). No bounds validation.
- [x] GeoOps KmToDegrees asymmetric — returns max of lat/lon degrees, over-selecting. Should return both.
- [x] SafeTensor endianness — little-endian assumed but not validated. Breaks on big-endian.
- [x] TextEmbedder all NotImplementedException — stub code crashes at runtime. Should be abstract.
- [x] GridSearchCV silent param ignore — reflection silently skips nonexistent property names.

## 22. The Big 4 — Ergonomics

- [x] **Boolean indexing** — `df[df["age"].Gt(30)]` should work via indexer overload on `Column<bool>`.
- [x] **Arithmetic operators on columns** — `priceCol * 1.1` and `colA + colB` via operator overloads.
- [x] **Implicit numeric type widening** — `Column<int> + Column<double>` should work. Add `.AsDouble()` helper.
- [x] **Namespace consolidation** — collapse GroupBy, Joins, Reshape, Missing, Statistics, Window into fewer namespaces or add global using aggregation.

## 23. High-Value Ergonomics

- [x] **Multi-column aggregation** — `df.GroupBy("cat").Agg(("price", Sum), ("qty", Mean))` syntax.
- [x] **Actionable error messages** — include column names, lengths, types in error messages.
- [x] **DataFrame.Pipe()** — functional composition: `df.Pipe(normalize).Pipe(encode).Pipe(split)`.
- [x] **LazyFrame parity** — add Distinct(), Rename(), Take(), Skip() on LazyFrame.
- [x] **GroupBy null behavior** — add NullGroupingMode parameter (Include/Exclude/Own group).
- [x] **RenameColumn() / RenameColumns()** — direct column rename without clone.

## 24. Polish

- [x] **Factory methods** — `DataFrame.FromMatrix(double[,])`, `FromTuples(...)`.
- [x] **Batch(int size)** — yield DataFrames of given size for ML data loading.
- [x] **ToString() pretty-print** — wire ConsoleFormatter to DataFrame.ToString().
- [x] **Missing string column methods** — ToUpper(), ToLower(), Trim(), StartsWith(), Substring() directly on StringColumn.
- [x] **DataFrame.Equals()** — value equality comparison for testing.
- [x] **XML doc coverage** — document public API.
