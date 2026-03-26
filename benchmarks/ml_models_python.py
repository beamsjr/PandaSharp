import time, json, os, numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Lasso, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.manifold import TSNE

os.makedirs("ml_bench_output", exist_ok=True)

results = []
def lap(cat, name, start):
    ms = round((time.perf_counter() - start) * 1000)
    results.append({"category": cat, "op": name, "ms": ms})
    print(f"  {name:<55} {ms:>6} ms")

# Generate data
np.random.seed(42)
N = 50000  # 50K samples
P = 20     # 20 features
X_reg = np.random.randn(N, P)
y_reg = X_reg @ np.random.randn(P) + np.random.randn(N) * 0.1
X_cls = np.random.randn(N, P)
y_cls = (X_cls[:, 0] + X_cls[:, 1] > 0).astype(int)

print("=== Python ML Models Benchmark ===")
print(f"  Data: {N} samples, {P} features\n")

# Linear Models
print("── Linear Models ──")
for name, model in [("LinearRegression", LinearRegression()), ("ElasticNet", ElasticNet(alpha=0.1)), ("Lasso", Lasso(alpha=0.1))]:
    t = time.perf_counter(); model.fit(X_reg, y_reg); pred = model.predict(X_reg); lap("Linear", f"{name} fit+predict", t)

for name, model in [("LogisticRegression", LogisticRegression(max_iter=200)), ("SGDClassifier", SGDClassifier(max_iter=100))]:
    t = time.perf_counter(); model.fit(X_cls, y_cls); pred = model.predict(X_cls); lap("Linear", f"{name} fit+predict", t)

t = time.perf_counter(); SGDRegressor(max_iter=100).fit(X_reg, y_reg); lap("Linear", "SGDRegressor fit+predict", t)

# Tree Models
print("\n── Tree Models ──")
for name, model in [("DecisionTreeClassifier", DecisionTreeClassifier(max_depth=10)), ("DecisionTreeRegressor", DecisionTreeRegressor(max_depth=10))]:
    t = time.perf_counter(); model.fit(X_cls if "Classifier" in name else X_reg, y_cls if "Classifier" in name else y_reg); model.predict(X_cls if "Classifier" in name else X_reg); lap("Tree", f"{name} fit+predict", t)

for name, model in [("RandomForestClassifier(100)", RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)), ("RandomForestRegressor(100)", RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1))]:
    t = time.perf_counter(); model.fit(X_cls if "Classifier" in name else X_reg, y_cls if "Classifier" in name else y_reg); model.predict(X_cls if "Classifier" in name else X_reg); lap("Tree", f"{name} fit+predict", t)

for name, model in [("GBTClassifier(100)", GradientBoostingClassifier(n_estimators=100, max_depth=3)), ("GBTRegressor(100)", GradientBoostingRegressor(n_estimators=100, max_depth=3))]:
    t = time.perf_counter(); model.fit(X_cls if "Classifier" in name else X_reg, y_cls if "Classifier" in name else y_reg); model.predict(X_cls if "Classifier" in name else X_reg); lap("Tree", f"{name} fit+predict", t)

# Distance Models
print("\n── Distance Models ──")
# Use smaller dataset for KNN (brute force is O(N*k))
X_knn = X_cls[:5000]; y_knn = y_cls[:5000]
for name, model in [("KNN Classifier(k=5)", KNeighborsClassifier(n_neighbors=5)), ("KNN Regressor(k=5)", KNeighborsRegressor(n_neighbors=5))]:
    t = time.perf_counter(); model.fit(X_knn, y_knn); model.predict(X_knn); lap("Distance", f"{name} fit+predict (5K)", t)

# Clustering
print("\n── Clustering ──")
X_cluster = X_reg[:10000]
t = time.perf_counter(); KMeans(n_clusters=5, n_init=10, max_iter=300).fit(X_cluster); lap("Cluster", "KMeans(5) fit (10K)", t)
t = time.perf_counter(); DBSCAN(eps=3.0, min_samples=5).fit(X_cluster[:5000]); lap("Cluster", "DBSCAN fit (5K)", t)
t = time.perf_counter(); AgglomerativeClustering(n_clusters=5).fit(X_cluster[:2000]); lap("Cluster", "AgglomerativeClustering fit (2K)", t)

# Dimensionality Reduction
print("\n── Dimensionality Reduction ──")
t = time.perf_counter(); PCA(n_components=5).fit_transform(X_reg); lap("DimReduce", "PCA(5) fit+transform (50K)", t)
t = time.perf_counter(); TruncatedSVD(n_components=5).fit_transform(X_reg); lap("DimReduce", "TruncatedSVD(5) fit+transform (50K)", t)
t = time.perf_counter(); TSNE(n_components=2, perplexity=30, max_iter=250).fit_transform(X_reg[:1000]); lap("DimReduce", "t-SNE(2) fit+transform (1K)", t)

# Model Selection
print("\n── Model Selection ──")
t = time.perf_counter(); scores = cross_val_score(LinearRegression(), X_reg[:5000], y_reg[:5000], cv=5); lap("ModelSel", "CrossValScore(5-fold, 5K)", t)

# Summary
print(f"\n{'═'*70}")
cats = {}
for r in results: cats[r["category"]] = cats.get(r["category"], 0) + r["ms"]
for c, m in sorted(cats.items(), key=lambda x: -x[1]): print(f"  {c:<30} {m:>8,} ms")
total = sum(r["ms"] for r in results)
print(f"  {'TOTAL':<30} {total:>8,} ms")
with open("ml_bench_output/python_ml_results.json", "w") as f: json.dump(results, f, indent=2)
