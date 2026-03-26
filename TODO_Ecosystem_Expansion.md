# PandaSharp Ecosystem Expansion — Feature Tracker

## PandaSharp.ML.Models — Classical ML Algorithms

### Linear Models
- [ ] LinearRegression (OLS via normal equations, Ridge/L2 penalty option)
- [ ] LogisticRegression (binary + multi-class, gradient descent solver)
- [ ] SGDClassifier (stochastic gradient descent with configurable loss)
- [ ] SGDRegressor (stochastic gradient descent for regression)
- [ ] ElasticNet (combined L1/L2 regularization)
- [ ] Lasso (L1 regularization, coordinate descent solver)

### Tree-Based Models
- [ ] DecisionTreeClassifier (CART algorithm, Gini/entropy split criteria)
- [ ] DecisionTreeRegressor (MSE/MAE split criteria)
- [ ] RandomForestClassifier (bagged decision trees, configurable n_estimators/max_depth)
- [ ] RandomForestRegressor (bagged trees for regression)
- [ ] GradientBoostedTreeClassifier (sequential boosting, learning rate, subsample)
- [ ] GradientBoostedTreeRegressor (sequential boosting for regression)

### Distance-Based Models
- [ ] KNearestNeighborsClassifier (brute-force + KD-tree, configurable k/distance metric)
- [ ] KNearestNeighborsRegressor (weighted average of k nearest)

### Clustering
- [ ] KMeans (Lloyd's algorithm, k-means++ initialization, configurable n_clusters/max_iter)
- [ ] DBSCAN (density-based, configurable eps/min_samples)
- [ ] AgglomerativeClustering (single/complete/average/ward linkage)
- [ ] Silhouette score, Davies-Bouldin index, Calinski-Harabasz index (cluster metrics)

### Dimensionality Reduction
- [ ] PCA (eigendecomposition of covariance matrix, explained variance ratio)
- [ ] TruncatedSVD (for sparse data, no centering required)
- [ ] UMAP (approximate nearest neighbors + graph layout)
- [ ] t-SNE (Barnes-Hut approximation for large datasets)

### Model Infrastructure
- [ ] IModel interface: Fit(X, y), Predict(X), Score(X, y) — unified API for all models
- [ ] Cross-validation: CrossValScore(model, df, features, label, nFolds, scorer)
- [ ] GridSearchCV (exhaustive hyperparameter search, returns best params + scores DataFrame)
- [ ] HyperparameterResult record: BestParams, BestScore, AllResults DataFrame
- [ ] Model serialization: model.Save(path), Model.Load(path) via JSON/binary
- [ ] LearningCurve: train/val score vs training set size (returns DataFrame for plotting)
- [ ] ConfusionMatrixDisplay: confusion matrix → DataFrame for Viz heatmap

---

## PandaSharp.TimeSeries — Forecasting & Analysis

### Statistical Models
- [ ] ARIMA (p, d, q) with auto-differencing and configurable order
- [ ] SARIMA (seasonal ARIMA with seasonal period P, D, Q, m)
- [ ] AutoARIMA (AIC/BIC-based order selection, stepwise search)
- [ ] ExponentialSmoothing (Simple, Double/Holt, Triple/Holt-Winters, additive/multiplicative)
- [ ] SimpleMovingAverageForecast (baseline forecaster using rolling mean)

### Decomposition
- [ ] SeasonalDecompose (additive/multiplicative, STL decomposition into trend + seasonal + residual)
- [ ] FFT-based periodogram (dominant frequency detection)

### Stationarity & Diagnostics
- [ ] AugmentedDickeyFuller test (unit root test for stationarity)
- [ ] KPSS test (stationarity around deterministic trend)
- [ ] ACF (autocorrelation function, returns DataFrame with lags + values)
- [ ] PACF (partial autocorrelation function)
- [ ] Ljung-Box test (residual autocorrelation significance)

### Forecasting API
- [ ] IForecaster interface: Fit(DataFrame, dateColumn, valueColumn), Forecast(horizon), ForecastWithInterval(horizon, alpha)
- [ ] ForecastResult record: Dates, Values, LowerBound, UpperBound (as DataFrame)
- [ ] Backtesting: expanding/sliding window evaluation, returns metrics per fold DataFrame
- [ ] Multi-step forecast (recursive and direct strategies)
- [ ] Changepoint detection (PELT algorithm or Bayesian online changepoint)

### Feature Engineering for Time Series
- [ ] LagFeatures transformer: auto-generate lag_1, lag_2, ..., lag_n columns
- [ ] DateTimeFeatures transformer: extract day_of_week, month, quarter, is_holiday, etc.
- [ ] RollingFeatures transformer: rolling mean/std/min/max as new columns with configurable windows
- [ ] FourierFeatures transformer: sin/cos components for seasonal modeling

---

## PandaSharp.Text — NLP Pipeline

### Tokenization
- [ ] WhitespaceTokenizer (split on whitespace, configurable lowercasing)
- [ ] RegexTokenizer (configurable pattern, handles punctuation/contractions)
- [ ] BPETokenizer (byte-pair encoding, train from corpus or load pre-trained vocab)
- [ ] WordPieceTokenizer (BERT-style subword tokenization, load from HuggingFace vocab.txt)
- [ ] SentencePieceTokenizer (unigram model, load pre-trained .model files)
- [ ] TokenizerResult: token IDs, attention mask, token-to-word mapping

### Text Preprocessing
- [ ] StopWordRemover (built-in English/Spanish/French/German lists + custom)
- [ ] Stemmer (Porter stemmer, Snowball stemmer)
- [ ] Lemmatizer (dictionary-based English lemmatization)
- [ ] NGramExtractor (unigram, bigram, trigram, configurable n + range)
- [ ] TextCleaner transformer: lowercase, strip HTML, remove URLs/emails/numbers, normalize whitespace
- [ ] SentenceSplitter (rule-based sentence boundary detection)

### Embeddings
- [ ] TextEmbedder (ONNX-based sentence embeddings, wraps OnnxScorer with tokenizer preprocessing)
- [ ] TextEmbedder.MiniLM() preset (all-MiniLM-L6-v2, 384-dim)
- [ ] TextEmbedder.E5() preset (e5-small-v2, 384-dim)
- [ ] CosineSimilarity: pairwise similarity between embedding columns
- [ ] SemanticSearch: query embedding vs corpus embeddings, returns top-k DataFrame with scores

### Text Analytics
- [ ] TextColumn accessor: df["text"].Text.TokenCount(), .WordFrequency(), .SentenceCount()
- [ ] NamedEntityRecognition via ONNX (load NER model, returns DataFrame with entity spans + labels)
- [ ] TextClassifier via ONNX (sentiment, topic, etc. — wraps OnnxScorer with tokenizer)
- [ ] DocumentSimilarityMatrix: TF-IDF or embedding cosine sim → DataFrame/heatmap

---

## PandaSharp.Vision — Image & Video Processing

### Core
- [ ] ImageTensor wrapper (Tensor<float> with [H,W,C] / [N,H,W,C] shape enforcement)
- [ ] ImageIO.Load(path) / Load(paths) / Load(stream) → ImageTensor
- [ ] ImageIO.Save(tensor, path) (PNG/JPEG based on extension)
- [ ] ImageIO.LoadFromColumn(df, pathColumn, resizeWidth, resizeHeight) → ImageTensor batch
- [ ] ChannelOrder enum (RGB, BGR, Grayscale)
- [ ] CorruptImageHandling enum (Skip, ZeroFill, Error)

### Transforms (IImageTransformer)
- [ ] Resize (Bilinear/NearestNeighbor/Bicubic interpolation)
- [ ] CenterCrop (pure tensor slice)
- [ ] RandomCrop (with optional padding, seeded Random)
- [ ] RandomHorizontalFlip (probability p, default 0.5)
- [ ] RandomVerticalFlip (probability p)
- [ ] Normalize (channel-wise mean/std, ImageNet/CIFAR10 presets)
- [ ] ColorJitter (random brightness/contrast/saturation/hue)
- [ ] GaussianBlur (configurable sigma range)
- [ ] RandomRotation (degree range via ImageSharp affine)
- [ ] RandomErasing (cutout with configurable area ratio/aspect ratio)
- [ ] Grayscale (RGB → single-channel, ITU-R BT.601 weights)
- [ ] ToTensorTransform (Image<Rgb24> → ImageTensor, bytes → [0,1] float)

### Pipeline & Data Loading
- [ ] ImagePipeline (composable IImageTransformer chain with builder API)
- [ ] ImageDataLoader (DataFrame path column → augmented ImageTensor batches, shuffle per epoch)
- [ ] ImageColumn (lazy-load image references, LoadAll/LoadAt, base64 thumbnails for display)

### Pre-trained Models
- [ ] ImageEmbedder (ONNX wrapper with preprocessing, ResNet50/MobileNetV3 presets)
- [ ] ImageClassifier (ONNX wrapper, Predict/PredictTopK, returns DataFrame with predictions)

### Video
- [ ] VideoReader (frame extraction to DataFrame [FrameIndex, Timestamp, ImagePath] or ImageTensor)
- [ ] VideoReader.Frames() lazy enumeration
- [ ] VideoFrameDataLoader (DataFrame of video paths → frame clip batches)

### Utilities
- [ ] ImageStats.ComputeNormalization (per-channel mean/std across dataset)
- [ ] ImageStats.DatasetSummary (image count, size distribution, aspect ratios, format breakdown)
- [ ] ImageStats.ValidateImages (find corrupt/unreadable images)
- [ ] ImageViz.ToImageGrid (HTML grid of images from DataFrame)
- [ ] ImageViz.ShowAugmentations (side-by-side augmentation results)

---

## PandaSharp.SafeTensors — HuggingFace Weight Loading

- [ ] SafeTensorReader: parse header JSON + memory-map tensor data
- [ ] SafeTensorReader.Open(path) / Open(stream)
- [ ] GetTensorNames() → string[]
- [ ] GetTensor<T>(name) → Tensor<T> (zero-copy via memory-mapped span)
- [ ] GetMetadata() → Dictionary<string, string>
- [ ] SafeTensorWriter: create SafeTensors files from PandaSharp Tensor<T>
- [ ] Support dtypes: float16, float32, float64, int32, int64, bfloat16 (read as float32)
- [ ] TorchSharp integration: LoadSafeTensors(path) → populate TorchSharp module parameters

---

## PandaSharp.ML Review Fixes (from Code Review 2 & 3)

### Review 2 — Carried Forward
- [ ] `[Medium]` OnnxScorer: replace Convert.ToSingle(col.GetObject(r)) with TypeHelpers fast path (Predict line 50, PredictBatched line 91)
- [ ] `[Medium]` TorchBridge: replace Convert.ToSingle(col.GetObject(r)) with TypeHelpers fast path (ToTorchTensor line 31, ToTorchTensor1D line 45)
- [ ] `[Low]` TorchBridge: deduplicate IsNumeric helper (line 113 duplicates TensorExtensions line 77)
- [ ] `[Low]` PipelineSerialization: implement Deserialize() or update doc comment
- [ ] `[Low]` PipelineSerialization: add TargetEncoder to ExtractParams()
- [ ] `[Low]` PipelineSerialization: add comment for stateless PolynomialFeatures
- [ ] `[Info]` TorchBridge: document precision loss in Tensor<double> → float conversion

### Review 3 — New Findings
- [ ] `[Medium]` DataViewBridge: replace GetObject() boxing in int/long getters with typed fast paths
- [ ] `[Medium]` Extract shared IsNumeric helper to TypeHelpers (currently copy-pasted in 6 files)
- [ ] `[Low]` OnnxScorer: remove unused _inputWidth field (dead code)
- [ ] `[Low]` MinMaxScaler: handle all-null columns (min > max edge case)
- [ ] `[Low]` LabelEncoder: add UnknownCategoryHandling for consistency with OneHotEncoder
- [ ] `[Low]` DataFrameDataLoader: document _epochCount behavior when shuffle is toggled
- [ ] `[Info]` Tensor.Random: add RandomNormal factory for standard-normal distribution
- [ ] `[Info]` ClassificationMetrics.MultiClass: warn or throw on null values instead of treating as class 0
