"""
PandaSharp.Vision vs Python (PIL + torchvision) — Image Processing Benchmark
=============================================================================
"""
import time, os, json, numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T
import torch

# ════════════════════════════════════════════════════════
# SETUP: Generate test images
# ════════════════════════════════════════════════════════
IMG_DIR = "vision_bench_images"
os.makedirs(IMG_DIR, exist_ok=True)
N_IMAGES = 500
IMG_SIZE = 256

print("=== Python Vision Benchmark ===\n")

# Generate random test images if not present
if len([f for f in os.listdir(IMG_DIR) if f.endswith('.png')]) < N_IMAGES:
    print(f"  Generating {N_IMAGES} test images ({IMG_SIZE}x{IMG_SIZE})...")
    rng = np.random.default_rng(42)
    for i in range(N_IMAGES):
        arr = rng.integers(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(IMG_DIR, f"img_{i:04d}.png"))
    print("  Done.")

image_paths = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.png')])[:N_IMAGES]

results = []
def lap(cat, name, start):
    ms = round((time.perf_counter() - start) * 1000)
    results.append({"category": cat, "op": name, "ms": ms})
    print(f"  {name:<55} {ms:>6} ms")
    return ms

# ════════════════════════════════════════════════════════
# 1. IMAGE LOADING
# ════════════════════════════════════════════════════════
print("\n── Image Loading ──")

t = time.perf_counter()
imgs = [Image.open(p).convert('RGB') for p in image_paths]
lap("Load", f"Load {N_IMAGES} images (PIL)", t)

t = time.perf_counter()
arrays = [np.array(img).astype(np.float32) / 255.0 for img in imgs]
lap("Load", f"Convert to float32 arrays ({N_IMAGES})", t)

t = time.perf_counter()
batch = np.stack(arrays)  # [N, H, W, 3]
lap("Load", f"Stack to batch tensor ({N_IMAGES}x{IMG_SIZE}x{IMG_SIZE}x3)", t)

# ════════════════════════════════════════════════════════
# 2. SINGLE IMAGE TRANSFORMS
# ════════════════════════════════════════════════════════
print("\n── Single Image Transforms (applied to all images) ──")
test_img = imgs[0]
test_arr = arrays[0]

# Resize
t = time.perf_counter()
for img in imgs:
    img.resize((224, 224), Image.BILINEAR)
lap("Transform", f"Resize to 224x224 ({N_IMAGES} images)", t)

# Center crop
t = time.perf_counter()
crop = T.CenterCrop(128)
for img in imgs:
    crop(img)
lap("Transform", f"CenterCrop 128x128 ({N_IMAGES} images)", t)

# Horizontal flip
t = time.perf_counter()
for arr in arrays:
    np.flip(arr, axis=1).copy()  # .copy() to materialize
lap("Transform", f"Horizontal flip ({N_IMAGES} images)", t)

# Vertical flip
t = time.perf_counter()
for arr in arrays:
    np.flip(arr, axis=0).copy()
lap("Transform", f"Vertical flip ({N_IMAGES} images)", t)

# Normalize (ImageNet)
t = time.perf_counter()
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
for arr in arrays:
    (arr - mean) / std
lap("Transform", f"Normalize ImageNet ({N_IMAGES} images)", t)

# Grayscale
t = time.perf_counter()
weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
for arr in arrays:
    np.dot(arr, weights)
lap("Transform", f"Grayscale ({N_IMAGES} images)", t)

# Color jitter (brightness)
t = time.perf_counter()
jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
for img in imgs:
    jitter(img)
lap("Transform", f"ColorJitter ({N_IMAGES} images)", t)

# Gaussian blur
t = time.perf_counter()
for img in imgs:
    img.filter(ImageFilter.GaussianBlur(radius=2))
lap("Transform", f"GaussianBlur ({N_IMAGES} images)", t)

# Random erasing on tensor
t = time.perf_counter()
erase = T.RandomErasing(p=1.0)
for arr in arrays:
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # CHW
    erase(tensor)
lap("Transform", f"RandomErasing ({N_IMAGES} images)", t)

# ════════════════════════════════════════════════════════
# 3. BATCH TRANSFORMS (numpy)
# ════════════════════════════════════════════════════════
print("\n── Batch Transforms (numpy on full batch) ──")

# Batch normalize
t = time.perf_counter()
normalized = (batch - mean) / std
lap("Batch", f"Batch normalize ({N_IMAGES} images)", t)

# Batch horizontal flip
t = time.perf_counter()
flipped = np.flip(batch, axis=2).copy()
lap("Batch", f"Batch horizontal flip ({N_IMAGES} images)", t)

# Batch grayscale
t = time.perf_counter()
gray = np.dot(batch, weights)
lap("Batch", f"Batch grayscale ({N_IMAGES} images)", t)

# ════════════════════════════════════════════════════════
# 4. TORCHVISION PIPELINE
# ════════════════════════════════════════════════════════
print("\n── Torchvision Pipeline ──")

pipeline = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

t = time.perf_counter()
for img in imgs:
    pipeline(img)
lap("Pipeline", f"Resize+Flip+ToTensor+Normalize ({N_IMAGES} images)", t)

# Full augmentation pipeline
aug_pipeline = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.5),
])

t = time.perf_counter()
for img in imgs:
    aug_pipeline(img)
lap("Pipeline", f"Full augmentation pipeline ({N_IMAGES} images)", t)

# ════════════════════════════════════════════════════════
# 5. IMAGE SAVING
# ════════════════════════════════════════════════════════
print("\n── Image Saving ──")

os.makedirs("vision_bench_output", exist_ok=True)
t = time.perf_counter()
for i, img in enumerate(imgs[:100]):
    img.save(f"vision_bench_output/out_{i:04d}.png")
lap("Save", f"Save 100 images (PNG)", t)

# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
categories = {}
for r in results:
    cat = r["category"]
    categories[cat] = categories.get(cat, 0) + r["ms"]

for cat, ms in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat:<30} {ms:>8,} ms")
total = sum(r["ms"] for r in results)
print(f"  {'TOTAL':<30} {total:>8,} ms")
print(f"{'═'*70}")

with open("vision_bench_output/python_vision_results.json", "w") as f:
    json.dump(results, f, indent=2)
