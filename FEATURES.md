# PandaSharp Feature Tracker

## Implemented Features

### Core DataFrame (Phase 1-10)
- [x] Column<T>, StringColumn, CategoricalColumn, IColumn
- [x] Arrow-backed columnar storage, NullBitmask, zero-copy slicing
- [x] Head, Tail, Select, Slice, Sample, SampleFrac, Copy
- [x] Shape, IsEmpty, Dtypes, Memory(), Summary(), Info(), Describe(), DescribeAll()
- [x] Loc/ILoc indexing, At/IAt scalar access, SetIndex/ResetIndex (single + multi)
- [x] MultiIndex, Xs cross-section
- [x] Transpose (.T), Iterrows, Itertuples, Itercolumns
- [x] FromDictionary, FromRecords, FromEnumerable, ToRecords, ToDictionary(orient), ToRecordDicts
- [x] Query (string-based with AND/OR), Where, Filter (mask/lambda/expression), Eval(string)
- [x] Sort (single/multi, typed struct comparers), SortValues, Nlargest, Nsmallest
- [x] DropDuplicates (bucketed hash), Duplicated
- [x] AddColumn, DropColumn, DropColumns, RenameColumn, RenameColumns
- [x] Assign, UpdateColumn, ReplaceColumn, ReorderColumns, CastColumn(name, targetType)
- [x] Pipe, Apply<T>, Apply(string), ApplyMap, ApplyColumns
- [x] Combine, Clip, Agg, SelectDtypes, ExcludeDtypes, NumericOnly, Idxmin, Idxmax

### Column Operations
- [x] Arithmetic: Add, Subtract, Multiply, Divide, Negate (SIMD-accelerated, zero-copy output)
- [x] Comparison: Gt, Gte, Lt, Lte, Eq, Between, IsIn
- [x] Math: Abs, Round, Sqrt, Log, Log10, Pow
- [x] Aggregates: Sum (SIMD), Mean (SIMD), Median (quickselect), Std, Var, Quantile, Mode, Skew, Kurtosis, Sem, Min, Max, Count
- [x] Cumulative: CumSum, CumProd, CumMin, CumMax, Cumcount
- [x] Transform: Map, Where, Replace, Clip, Shift, Unique, Normalize (MinMax, ZScore)
- [x] ArgMin, ArgMax, Zip, ValueCounts, NUnique, Rank
- [x] Bool: Any, All, SumTrue
- [x] Mask: And, Or, Not, Xor, CountTrue
- [x] Prod (column product), Cast&lt;TSource, TTarget&gt; (typed column conversion)
- [x] Cut (equal-width binning), QCut (quantile binning) — custom labels, explicit edges

### Missing Data
- [x] IsNa, NotNa, FillNa (scalar/forward/backward), DropNa (axis, threshold)
- [x] Interpolate (linear, polynomial, cubic spline)

### Statistics
- [x] Describe, DescribeAll, Info, Corr, Cov, PctChange, Diff
- [x] df.Profile() — automated EDA: per-column stats (mean/std/quartiles/skew/kurtosis), value counts, correlation matrix, missing data analysis, duplicate detection, string length stats, HTML report generation

### GroupBy
- [x] Single/multi-key, typed accumulators (double/int fast paths)
- [x] Sum, Mean, Median, Std, Var, Min, Max, Count, First, Last, Nth
- [x] Named Agg builder, Transform, Filter, Apply
- [x] Cumcount, Ngroup, SumParallel, MeanParallel
- [x] Key reuse optimization (single alloc per new group)

### Joins & Concat
- [x] Inner, Left, Right, Outer, Cross, Anti joins (typed int fast path)
- [x] Multi-key joins, suffix handling, Merge alias
- [x] AsOf join (backward/forward/nearest, with by-group)
- [x] Concat (rows/columns), DataFrame.Concat static, Compare

### Reshaping
- [x] Pivot (O(1) lookup), Melt, PivotTable (O(1) lookup, custom agg)
- [x] Stack, Unstack, CrossTab (O(1) lookup)
- [x] Explode, GetDummies (single-pass dictionary)

### Window Functions
- [x] Rolling (O(n) sliding mean, reused buffer for generic Apply)
- [x] Expanding (O(n) single-pass Mean/Sum/Min/Max/Std)
- [x] EWM (exponentially weighted mean)

### String Accessor (25+ methods)
- [x] Contains, StartsWith, EndsWith, Match, Extract, Replace, Split
- [x] Upper, Lower, Title, Capitalize, Trim, LStrip, RStrip
- [x] Pad, Slice, Len, Count, Find, Repeat, Cat
- [x] ZFill, Center, LJust, RJust (string padding variants)

### DateTime Accessor
- [x] Year, Month, Day, Hour, Minute, Second, Microsecond
- [x] DayOfWeek, DayOfYear, Quarter
- [x] IsMonthStart/End, IsQuarterStart/End, IsYearStart/End, IsLeapYear
- [x] Floor, Ceil, Round, Date, AddDays/Months/Years/Hours
- [x] DateRange

### Expression System
- [x] Col, Lit, arithmetic (+,-,*,/), comparison (>,<,>=,<=)
- [x] Logical (&, |, !), Eq, Neq
- [x] String chains: Col("x").Str.Upper/Lower/Contains/Replace/Slice/Len
- [x] DateTime chains: Col("d").Dt.Year/Month/Day/Quarter
- [x] When/Then/Otherwise, Coalesce, ConcatStr, Cast<T>
- [x] Aggregates: Sum/Mean/Min/Max/Count/StdExpr/MedianExpr (broadcast)
- [x] Alias
- [x] df.Eval("price * quantity > 1000") — string expression parser with full operator support
- [x] ExprParser: recursive descent, arithmetic/comparison/logical/not, parentheses, identifiers, literals

### Lazy Evaluation
- [x] LazyFrame with query plan DAG
- [x] Select, Filter, Sort, WithColumn, WithColumns, Head, GroupBy, Join
- [x] Predicate pushdown optimizer (Filter past Sort)
- [x] Projection pushdown optimizer (Select past Sort/Filter/Head with column dependency tracking, 2x memory reduction on wide DataFrames)
- [x] Explain, Collect()
- [x] ExternalScanPlan for database-backed lazy evaluation with auto SQL push-down

### I/O
- [x] CSV (inferred + schema fast path, span-based numeric parsing, strict quoting, gzip compression)
- [x] JSON (records + columns + JSON Lines/.jsonl)
- [x] Arrow IPC (zero-copy)
- [x] Parquet (read/write, column pruning, Hive-style partitioned read/write)
- [x] Excel (read/write, multi-sheet, ListSheets)
- [x] HTML table reading (multi-table, type inference)
- [x] Clipboard (macOS/Linux/Windows)
- [x] Universal Save/Load: df.Save(path) / DataFrame.Load(path) with auto-format detection (.csv, .csv.gz, .tsv, .json, .jsonl, .parquet, .arrow, .xlsx)

### Display
- [x] Console (Unicode box-drawing), HTML, Markdown, LaTeX
- [x] .NET Interactive formatter (styled HTML with dtype annotations)

### Interop
- [x] LINQ: FromEnumerable<T>, AsEnumerable<T> (compiled setters)
- [x] ADO.NET: FromDataTable, ToDataTable, FromDataReader
- [x] Schema validation (type, null, unique, range, regex, allowed values, custom checks)
- [x] DataFrameSchema fluent builder: Column rules, HasColumns, NoExtraColumns, MinRows, MaxRows, NoDuplicateRows, Check()
- [x] df.ValidateSchema(schema) — throws on failure, returns df for chaining

### Plotting (PandaSharp.Plot)
- [x] ScottPlot 5: Scatter, Line, Bar, Histogram

### Copy-on-Write & Out-of-Core
- [x] MutableDataFrame with CoW semantics (type-guarded SetValue)
- [x] SpilledDataFrame: Arrow IPC spill-to-disk with lazy column loading, eviction, Select/Filter to new spill

### Time Series
- [x] Resample with frequency parsing

### PandaSharp.ML (Phase 1)
- [x] Tensor<T> (1D/2D, SIMD arithmetic, Sum, Mean, ArgMax, Transpose)
- [x] DataFrame ↔ Tensor zero-copy bridge
- [x] StandardScaler, MinMaxScaler (Fit/Transform)
- [x] LabelEncoder, OneHotEncoder (Fit/Transform)
- [x] Imputer (Mean/Median/Mode/Constant)
- [x] FeaturePipeline composition
- [x] TrainTestSplit (random + stratified), TrainValTestSplit, KFold
- [x] Classification metrics (Accuracy/Precision/Recall/F1/ConfusionMatrix)
- [x] Regression metrics (MSE/RMSE/MAE/R²/MAPE)

### Performance Optimizations
- [x] SIMD Vector<T> arithmetic (zero-copy output to Arrow buffer)
- [x] Quickselect O(n) Median and Quantile (was O(n log n) sort-based)
- [x] Typed GroupBy accumulators (double/int/count fast paths)
- [x] Typed Sort (struct IComparer, Array.Sort(keys,items))
- [x] Branchless filter index construction
- [x] Typed inner join (Dictionary<int, List<int>>, TakeRows)
- [x] ArrayPool in TakeRows/Filter buffers → replaced with direct byte[] write
- [x] NullBitmask PopCount + single-pass FromNullables
- [x] Column<T>.Clone() deep copy
- [x] CSV span-based parsing, two-phase infer+schema, fast-path ReadLine for non-quoted fields
- [x] GetDummies single-pass dictionary
- [x] Pivot/CrossTab O(1) dictionary lookups
- [x] Expanding O(n) single-pass (was O(n²) → 18,300x faster)
- [x] Rolling O(n) sliding window mean
- [x] Local parallelism: row-range partitioned Filter/Apply/Where/Arithmetic/Sum (PandaSharp.ParallelOps)

## TODO / Known Gaps
- [x] RobustScaler (median/IQR based)
- [x] OrdinalEncoder (explicit ordering)
- [x] PolynomialFeatures (degree N, interactions)
- [x] StratifiedKFold
- [x] TimeSeriesSplit (expanding window)
- [x] TargetEncoder (with smoothing)
- [x] Discretizer (Uniform + Quantile binning)
- [x] Tensor MatMul (cache-friendly ikj loop) + Dot product
- [x] TextVectorizer (Count + TF-IDF, vocabulary, maxFeatures, tokenizer)
- [x] Pipeline serialization (SerializeToJson, Serialize to bytes, step order preserved)
- [x] Ranking metrics (NDCG, MRR, MAP, @K support)
- [x] ML.NET IDataView bridge (PandaSharp.ML.MLNet) — df.ToDataView(mlContext), dataView.ToDataFrame()
- [x] TorchSharp tensor bridge (PandaSharp.ML.Torch) — df.ToTorchTensor(), tensor.ToDataFrame(), ToTorchDataLoader()
- [x] ONNX Runtime scorer (PandaSharp.ML.Onnx) — OnnxScorer.Predict(df), PredictBatched(df, batchSize)
- [x] DataFrameDataLoader (batched, shuffled, deterministic seed, pre-extracted columns)
- [x] Feature importance (permutation importance with configurable repeats, sorted output)
- [x] README.md (quick start, I/O, expressions, lazy eval, joins, ML, benchmarks, install)
- [x] XML doc comments on Tensor<T> core properties (Shape, Rank, Length, Span)
- [x] Parquet performance optimization (7.6x faster: 13.5ms→1.8ms, 4.8x less memory: 28MB→5.9MB via zero-boxing typed array reads)
- [x] Chunked CSV reading without header (auto-generates Column0..ColumnN, buffers first line)
- [x] Interpolation methods: Polynomial (Neville's algorithm), Cubic/Spline/Pchip (natural cubic spline)
- [x] Tensor N-dimensional operations: Slice, SumAxis, ArgMax generalized for any rank (stride-based indexing)

## PandaSharp.IO.Database — Smart Database Integration

### Phase 1: Core Engine ✅
- [x] DatabaseScanner: Scan, ScanSql, ScanGroupBy, Schema, Count
- [x] SqlGenerator: expression-to-SQL compilation (parameterized)
- [x] Expression push-down: Filter, Select, Sort, Limit, GroupBy, Distinct
- [x] SqlDialect abstraction: PostgreSQL, SQL Server, MySQL, SQLite
- [x] Identifier quoting per dialect
- [x] DataFrame.ToSql: write DataFrames to database tables
- [x] Auto-dialect detection from connection type
- [x] LogicalPlan → SQL compilation (full plan walking: Filter, Select, Sort, Head, GroupBy, WithColumn, Join)
- [x] SQL generation benchmark: 374 ns / 2.3 KB per full query plan

### Phase 2: Advanced Push-down (TODO)
- [x] Full Join push-down: scanner.LazyJoin(rightTable, onColumn, joinType) emits proper SQL JOIN with table-qualified ON clause
- [x] Batched multi-row INSERT writes (configurable batchSize, default 1000 rows per statement)
- [x] Connection pooling: DatabasePool.Register/Table/Query/Unregister/DisposeAll
- [x] Lazy database scans: scanner.Lazy().Filter(...).Select(...).Sort(...).Head(...).Collect() with auto SQL push-down + fallback
- [x] Arrow Flight RPC: FlightDataClient for distributed DataFrame get/put/list (PandaSharp.Flight)

## PandaSharp.Viz — Interactive Visualization

### Phase 1: Core Engine ✅
- [x] VizBuilder fluent API: df.Viz().Bar/Line/Scatter/Histogram/Box/Heatmap/Pie/Area
- [x] PlotlySerializer: DataFrame columns → Plotly.js JSON traces + layout
- [x] HtmlRenderer: self-contained HTML + fragment mode + Plotly CDN
- [x] Customization: Title, XLabel, YLabel, Width, Height, Size, Theme, Legend
- [x] Export: .ToHtml(path), .ToHtmlString(), .ToHtmlFragment(divId), .Show()
- [x] Color/Size/Text encoding via automatic group splitting

### Phase 2: Chart Types ✅
- [x] Bar (vertical/horizontal, grouped by color, barmode)
- [x] Line (multi-series, markers)
- [x] Scatter (color/size encoding, WebGL for 50K+)
- [x] Histogram (bins, density)
- [x] Box (grouped)
- [x] Heatmap
- [x] Pie
- [x] Area (filled, stacked)

### Phase 3: Advanced Features
- [x] FacetGrid: df.FacetGrid("Region").Line("Month", "Revenue")
- [x] Subplots: SubplotBuilder.Grid(chart1, chart2).Cols(2)
- [x] Theme system: VizTheme.Light/Dark/Minimal + .Theme()
- [x] .NET Interactive Viz: ToNotebookHtml() for VizBuilder + FacetGrid
- [x] Large data: WebGL auto-switch for 50K+ points in Scatter
- [x] Animation: .Animate("Year") for time-based transitions (Plotly frames, slider, play/pause)
- [x] .ToPng() / .ToSvg() static export via PuppeteerSharp (headless Chromium, no Node.js required)

## PandaSharp.ML Code Review Issues (2026-03-24)
- [x] Tensor constructor doc: fixed misleading "zero-copy" comment
- [x] Pipeline serialization v2: serializes learned params for all transformers via reflection
- [x] DataViewBridge.ToDataView: proper type mapping (Double, Int32, Int64, Bool, Text) + typed ToDataFrame builders (zero boxing)
- [x] Tensor.ArgMax axis=0: implemented per-column argmax
- [x] GetObject() boxing in transformers: replaced with TypeHelpers.GetDouble() typed fast paths
- [x] TF-IDF smoothing: log((N+1)/(df+1))+1
- [x] Imputer.ComputeMedian: copies list before sorting
- [x] DataFrameDataLoader: increments seed per epoch
- [x] PolynomialFeatures interaction: documented as correct (reads original values intentionally)
- [x] OneHotEncoder unseen categories: UnknownCategoryHandling (Ignore/Error/Indicator)
- [x] Multi-class classification: confusion matrix, per-class P/R/F1, macro/weighted F1
- [x] OnnxScorer dimensionality: uses Dimensions.Length for 2D detection
- [x] TypeHelpers.IsNumeric: shared helper with decimal+short support
- [x] RankingResult.PrecisionAtK: removed broken instance method
- [x] Tensor.Slice bounds-check: throws ArgumentOutOfRangeException
- [x] FeaturePipeline.Fit: documented re-fit behavior in XML doc

## Ecosystem Roadmap (from PandaSharp_Ecosystem_Roadmap.docx)

### PandaSharp.Streaming — Real-Time Event Processing

#### Phase 1: Core Engine ✅
- [x] StreamFrame fluent API: StreamFrame.From(source).Window(...).Agg(...).OnEmit(...).Start()
- [x] IStreamSource interface + EnumerableSource (testing) + ChannelSource (in-process pub/sub)
- [x] TumblingWindow: fixed-size non-overlapping time windows
- [x] SlidingWindow: overlapping windows with configurable slide interval
- [x] SessionWindow: dynamic windows based on activity gaps
- [x] Aggregations: Sum, Mean, Min, Max, Count per window
- [x] Watermarks: WithWatermark(allowedLateness) for out-of-order events
- [x] Backpressure: bounded Channel with configurable capacity
- [x] Micro-batch: events accumulated into DataFrames with window metadata
- [x] Collect() for testing: collects all emitted DataFrames into a list

#### Phase 2: External Sources ✅
- [x] KafkaSource: consumer group, JSON deserialization, manual offset tracking (PandaSharp.Streaming.Kafka)
- [x] KafkaSink: produce DataFrame rows as JSON messages to Kafka topics
- [x] RedisSource: Redis Streams XREAD/XREADGROUP with consumer groups (PandaSharp.Streaming.Redis)
- [x] RedisSink: XADD DataFrame rows to Redis Streams with optional maxLen
- [x] WebSocketSource: real-time WebSocket event ingestion (System.Net.WebSockets, JSON parsing, auto-timestamp)
- [x] Exactly-once semantics via Kafka manual offset commit after processing

### PandaSharp.Geo — Geospatial Operations

#### Phase 1: Core Engine ✅
- [x] GeoPoint record, BoundingBox with Contains/Intersects
- [x] GeoColumn: parallel lat/lon arrays with DistanceTo, Within, WithinDistance, Buffer, Bounds, Centroid, BearingTo, Filter, TakeRows
- [x] GeoOps: HaversineKm/Miles, EuclideanDegrees, KmToDegrees, Bearing, Destination
- [x] SpatialJoin.NearestJoin: nearest-neighbor with optional maxDistanceKm
- [x] SpatialJoin.WithinJoin: all-within-radius with bounding box pre-filter
- [x] GeoExtensions: df.WithDistance, df.FilterByDistance, df.FilterByBounds, df.SpatialJoinNearest, df.SpatialJoinWithin
- [x] df.ToGeoColumn(latCol, lonCol) for DataFrame interop

#### Phase 2: Geometry & Indexing ✅
- [x] R-tree spatial index (STR bulk-loaded): Query, QueryRadius, Nearest (63x faster), KNearest
- [x] GeoPolygon: ray-casting point-in-polygon, AreaKm2, PerimeterKm, Centroid, Bounds
- [x] GeoLineString: LengthKm, Interpolate, Bounds
- [x] GeoColumn.Within(polygon), df.FilterByPolygon()
- [x] SpatialJoin upgraded to R-tree indexed (O(n log m) nearest, O(n log m + k) radius)

- [x] GeoParquet read/write: WKB hex encoding, geo metadata sidecar, ReadAsDataFrame with lat/lon extraction
- [x] WKB encoding: EncodePoint/DecodePoint, EncodePoints/DecodePoints, DecodePointArray
- [x] Dissolve aggregation: group by key, aggregate values, compute group centroids

#### Phase 3: Advanced (Planned)
- [x] Coordinate system reprojection via ProjNet: Reproject(source, target), EPSG codes, WGS84/WebMercator/UTM built-in

### PandaSharp.Distributed — Parallel & Distributed Compute (Planned)
- [x] Phase 1: Local parallelism — ParallelFilter (1.6x on 100K rows), ParallelApply, ParallelWhere, ParallelAdd/Multiply, ParallelSum
- [x] Phase 2: Out-of-core via SpilledDataFrame — df.Spill(path), lazy column loading, eviction, Select/Filter to new spill, auto-cleanup
- [x] Phase 3: Distributed via Arrow Flight RPC (PandaSharp.Flight — get/put/list DataFrames over gRPC)
- [x] Hash/range partitioning: df.Partition(n), df.HashPartition(col, n), ParallelGroupBy/Filter/Where/Map, auto-tune to core count
- [x] Partitioned Parquet (Hive-style): ReadPartitioned, WritePartitioned with multi-level key=value directories
- [x] Cloud storage adapters: S3Storage, AzureStorage, GcsStorage with auto-format DataFrame read/write (PandaSharp.Cloud)

### PandaSharp.ML.Models — Classical ML Algorithms

#### Linear Models

- [x] LinearRegression (OLS via normal equations, Ridge/L2 penalty option)
- [x] LogisticRegression (binary + multi-class, gradient descent solver)
- [x] SGDClassifier (stochastic gradient descent with configurable loss)
- [x] SGDRegressor (stochastic gradient descent for regression)
- [x] ElasticNet (combined L1/L2 regularization)
- [x] Lasso (L1 regularization, coordinate descent solver)

#### Tree-Based Models

- [x] DecisionTreeClassifier (CART algorithm, Gini/entropy split criteria)
- [x] DecisionTreeRegressor (MSE/MAE split criteria)
- [x] RandomForestClassifier (bagged decision trees, configurable n_estimators/max_depth)
- [x] RandomForestRegressor (bagged trees for regression)
- [x] GradientBoostedTreeClassifier (sequential boosting, learning rate, subsample)
- [x] GradientBoostedTreeRegressor (sequential boosting for regression)

#### Distance-Based Models

- [x] KNearestNeighborsClassifier (brute-force + KD-tree, configurable k/distance metric)
- [x] KNearestNeighborsRegressor (weighted average of k nearest)

#### Clustering

- [x] KMeans (Lloyd's algorithm, k-means++ initialization, configurable n_clusters/max_iter)
- [x] DBSCAN (density-based, configurable eps/min_samples)
- [x] AgglomerativeClustering (single/complete/average/ward linkage)
- [x] Silhouette score, Davies-Bouldin index, Calinski-Harabasz index (cluster metrics)

#### Dimensionality Reduction

- [x] PCA (eigendecomposition of covariance matrix, explained variance ratio)
- [x] TruncatedSVD (for sparse data, no centering required)
- [x] UMAP (approximate nearest neighbors + graph layout)
- [x] t-SNE (Barnes-Hut approximation for large datasets)

#### Model Infrastructure

- [x] IModel interface: Fit(X, y), Predict(X), Score(X, y) — unified API for all models
- [x] Cross-validation: CrossValScore(model, df, features, label, nFolds, scorer)
- [x] GridSearchCV (exhaustive hyperparameter search, returns best params + scores DataFrame)
- [x] HyperparameterResult record: BestParams, BestScore, AllResults DataFrame
- [x] Model serialization: model.Save(path), Model.Load(path) via JSON/binary
- [x] LearningCurve: train/val score vs training set size (returns DataFrame for plotting)
- [x] ConfusionMatrixDisplay: confusion matrix → DataFrame for Viz heatmap

### PandaSharp.TimeSeries — Forecasting & Analysis

#### Statistical Models

- [x] ARIMA (p, d, q) with auto-differencing and configurable order
- [x] SARIMA (seasonal ARIMA with seasonal period P, D, Q, m)
- [x] AutoARIMA (AIC/BIC-based order selection, stepwise search)
- [x] ExponentialSmoothing (Simple, Double/Holt, Triple/Holt-Winters, additive/multiplicative)
- [x] SimpleMovingAverageForecast (baseline forecaster using rolling mean)

#### Decomposition

- [x] SeasonalDecompose (additive/multiplicative, STL decomposition into trend + seasonal + residual)
- [x] FFT-based periodogram (dominant frequency detection)

#### Stationarity & Diagnostics

- [x] AugmentedDickeyFuller test (unit root test for stationarity)
- [x] KPSS test (stationarity around deterministic trend)
- [x] ACF (autocorrelation function, returns DataFrame with lags + values)
- [x] PACF (partial autocorrelation function)
- [x] Ljung-Box test (residual autocorrelation significance)

#### Forecasting API

- [x] IForecaster interface: Fit(DataFrame, dateColumn, valueColumn), Forecast(horizon), ForecastWithInterval(horizon, alpha)
- [x] ForecastResult record: Dates, Values, LowerBound, UpperBound (as DataFrame)
- [x] Backtesting: expanding/sliding window evaluation, returns metrics per fold DataFrame
- [x] Multi-step forecast (recursive and direct strategies)
- [x] Changepoint detection (PELT algorithm or Bayesian online changepoint)

#### Feature Engineering for Time Series

- [x] LagFeatures transformer: auto-generate lag_1, lag_2, ..., lag_n columns
- [x] DateTimeFeatures transformer: extract day_of_week, month, quarter, is_holiday, etc.
- [x] RollingFeatures transformer: rolling mean/std/min/max as new columns with configurable windows
- [x] FourierFeatures transformer: sin/cos components for seasonal modeling


### PandaSharp.Text — NLP Pipeline

#### Tokenization

- [x] WhitespaceTokenizer (split on whitespace, configurable lowercasing)
- [x] RegexTokenizer (configurable pattern, handles punctuation/contractions)
- [x] BPETokenizer (byte-pair encoding, train from corpus or load pre-trained vocab)
- [x] WordPieceTokenizer (BERT-style subword tokenization, load from HuggingFace vocab.txt)
- [x] SentencePieceTokenizer (unigram model, load pre-trained .model files)
- [x] TokenizerResult: token IDs, attention mask, token-to-word mapping

#### Text Preprocessing

- [x] StopWordRemover (built-in English/Spanish/French/German lists + custom)
- [x] Stemmer (Porter stemmer, Snowball stemmer)
- [x] Lemmatizer (dictionary-based English lemmatization)
- [x] NGramExtractor (unigram, bigram, trigram, configurable n + range)
- [x] TextCleaner transformer: lowercase, strip HTML, remove URLs/emails/numbers, normalize whitespace
- [x] SentenceSplitter (rule-based sentence boundary detection)

#### Embeddings

- [x] TextEmbedder (ONNX-based sentence embeddings, wraps OnnxScorer with tokenizer preprocessing)
- [x] TextEmbedder.MiniLM() preset (all-MiniLM-L6-v2, 384-dim)
- [x] TextEmbedder.E5() preset (e5-small-v2, 384-dim)
- [x] CosineSimilarity: pairwise similarity between embedding columns
- [x] SemanticSearch: query embedding vs corpus embeddings, returns top-k DataFrame with scores

#### Text Analytics

- [x] TextColumn accessor: df["text"].Text.TokenCount(), .WordFrequency(), .SentenceCount()
- [x] NamedEntityRecognition via ONNX (stub — load NER model, returns DataFrame with entity spans + labels)
- [x] TextClassifier via ONNX (stub — sentiment, topic, etc. — wraps OnnxScorer with tokenizer)
- [x] DocumentSimilarityMatrix: TF-IDF or embedding cosine sim → DataFrame/heatmap


### PandaSharp.Vision — Image & Video Processing

#### Core

- [x] ImageTensor wrapper (Tensor<float> with [H,W,C] / [N,H,W,C] shape enforcement)
- [x] ImageIO.Load(path) / Load(paths) / Load(stream) → ImageTensor
- [x] ImageIO.Save(tensor, path) (PNG/JPEG based on extension)
- [x] ImageIO.LoadFromColumn(df, pathColumn, resizeWidth, resizeHeight) → ImageTensor batch
- [x] ChannelOrder enum (RGB, BGR, Grayscale)
- [x] CorruptImageHandling enum (Skip, ZeroFill, Error)

#### Transforms (IImageTransformer)

- [x] Resize (Bilinear/NearestNeighbor/Bicubic interpolation)
- [x] CenterCrop (pure tensor slice)
- [x] RandomCrop (with optional padding, seeded Random)
- [x] RandomHorizontalFlip (probability p, default 0.5)
- [x] RandomVerticalFlip (probability p)
- [x] Normalize (channel-wise mean/std, ImageNet/CIFAR10 presets)
- [x] ColorJitter (random brightness/contrast/saturation/hue)
- [x] GaussianBlur (configurable sigma range)
- [x] RandomRotation (degree range via ImageSharp affine)
- [x] RandomErasing (cutout with configurable area ratio/aspect ratio)
- [x] Grayscale (RGB → single-channel, ITU-R BT.601 weights)
- [x] ToTensorTransform (Image<Rgb24> → ImageTensor, bytes → [0,1] float)

#### Pipeline & Data Loading

- [x] ImagePipeline (composable IImageTransformer chain with builder API)
- [x] ImageDataLoader (DataFrame path column → augmented ImageTensor batches, shuffle per epoch)
- [x] ImageColumn (lazy-load image references, LoadAll/LoadAt, base64 thumbnails for display)

#### Pre-trained Models

- [x] ImageEmbedder (ONNX wrapper with preprocessing, ResNet50/MobileNetV3 presets)
- [x] ImageClassifier (ONNX wrapper, Predict/PredictTopK, returns DataFrame with predictions)

#### Video

- [x] VideoReader (frame extraction to DataFrame [FrameIndex, Timestamp, ImagePath] or ImageTensor)
- [x] VideoReader.Frames() lazy enumeration
- [x] VideoFrameDataLoader (DataFrame of video paths → frame clip batches)

#### Utilities

- [x] ImageStats.ComputeNormalization (per-channel mean/std across dataset)
- [x] ImageStats.DatasetSummary (image count, size distribution, aspect ratios, format breakdown)
- [x] ImageStats.ValidateImages (find corrupt/unreadable images)
- [x] ImageViz.ToImageGrid (HTML grid of images from DataFrame)
- [x] ImageViz.ShowAugmentations (side-by-side augmentation results)


### PandaSharp.SafeTensors — HuggingFace Weight Loading

- [x] SafeTensorReader: parse header JSON + memory-map tensor data
- [x] SafeTensorReader.Open(path) / Open(stream)
- [x] GetTensorNames() → string[]
- [x] GetTensor<T>(name) → Tensor<T> (zero-copy via memory-mapped span)
- [x] GetMetadata() → Dictionary<string, string>
- [x] SafeTensorWriter: create SafeTensors files from PandaSharp Tensor<T>
- [x] Support dtypes: float16, float32, float64, int32, int64, bfloat16 (read as float32)
- [x] TorchSharp integration: LoadSafeTensors(path) → populate TorchSharp module parameters


### PandaSharp.ML Review Fixes (from Code Review 2 & 3)
### Review 2 — Carried Forward

- [x] OnnxScorer: replace Convert.ToSingle(col.GetObject(r)) with TypeHelpers fast path (Predict line 50, PredictBatched line 91)
- [x] TorchBridge: replace Convert.ToSingle(col.GetObject(r)) with TypeHelpers fast path (ToTorchTensor line 31, ToTorchTensor1D line 45)
- [x] TorchBridge: deduplicate IsNumeric helper (line 113 duplicates TensorExtensions line 77)
- [x] PipelineSerialization: implement Deserialize() or update doc comment
- [x] PipelineSerialization: add TargetEncoder to ExtractParams()
- [x] PipelineSerialization: add comment for stateless PolynomialFeatures
- [x] TorchBridge: document precision loss in Tensor<double> → float conversion

#### Review 3 — New Findings

- [x] DataViewBridge: replace GetObject() boxing in int/long getters with typed fast paths
- [x] Extract shared IsNumeric helper to TypeHelpers (currently copy-pasted in 6 files)
- [x] OnnxScorer: remove unused _inputWidth field (dead code)
- [x] MinMaxScaler: handle all-null columns (min > max edge case)
- [x] LabelEncoder: add UnknownCategoryHandling for consistency with OneHotEncoder
- [x] DataFrameDataLoader: document _epochCount behavior when shuffle is toggled
- [x] Tensor.Random: add RandomNormal factory for standard-normal distribution
- [x] ClassificationMetrics.MultiClass: warn or throw on null values instead of treating as class 0

### Code Review Fixes Applied
- [x] DropDuplicates composite key collision fix (<<20 → *nUniques2)
- [x] DropDuplicates 3+ string hash collision resolution (bucket-based equality check)
- [x] CumMin/CumMax empty column crash fix
- [x] JoinMany negative key / BuildMappedColumnIndirect OOB fix
- [x] DescribeDouble null value handling (compact non-null values)
- [x] Thread safety: Random locks in Vision transforms, Volatile for _cachedDict, Interlocked for _epochCount
- [x] Error handling: bare catch blocks narrowed to catch (Exception), idempotent Dispose, SafeTensorReader stream dispose
- [x] Performance: hoisted inner loop allocations (JacobiEigen, t-SNE, SGD), cached BLAS availability
- [x] Null guards on all public APIs across ML Models, TimeSeries, Text, Vision, SafeTensors (~100 methods)
- [x] Dimension validation (y.Length == X.Shape[0]) in all ML model Fit methods
- [x] Negative label guards in DecisionTree/GradientBoosted classifiers
- [x] Empty input guards in Score methods
- [x] Triple ES multiplicative division-by-zero epsilon guard
- [x] PartialSort/ComputeNumericAggregate/GetGroupDoubles boxing elimination (typed fast paths)

### TODO: Test Coverage (from Code Review)
- [x] Tests for PandaSharp.ML.Models (21 tests — LinearRegression, LogisticRegression, DecisionTree, RandomForest, KNN, KMeans, PCA, ElasticNet, CrossVal, null guards, dimension mismatch)
- [x] Tests for PandaSharp.TimeSeries (23 tests — SMA, ETS, ARIMA, AutoARIMA, SeasonalDecompose, ACF, ADF/KPSS, Periodogram, Features, Changepoint, Backtesting, null guards)
- [x] Tests for PandaSharp.Text (27 tests — all tokenizers, preprocessors, CosineSimilarity, SemanticSearch, TextColumnAccessor, null guards)
- [x] Tests for PandaSharp.SafeTensors (7 tests — round-trip float/double, multiple tensors, metadata, GetTensorNames, null guards)
- [x] Tests for PandaSharp.Vision (19 tests — ImageTensor, all transforms, pipeline composition, builder API)

