"""
PandaSharp.Vision vs Python — Extended Image Processing Benchmark
=================================================================
Tests all transform operations, pipeline compositions, data loader patterns,
and image statistics computation.
"""
import time, os, json, numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms as T
import torch

IMG_DIR = "vision_bench_images"
N_IMAGES = 500
IMG_SIZE = 256

print("=== Python Vision Benchmark v2 ===\n")

# Ensure images exist
image_paths = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.png')])[:N_IMAGES]
assert len(image_paths) == N_IMAGES, f"Need {N_IMAGES} images in {IMG_DIR}. Run vision_python.py first."

results = []
def lap(cat, name, start):
    ms = round((time.perf_counter() - start) * 1000)
    results.append({"category": cat, "op": name, "ms": ms})
    print(f"  {name:<55} {ms:>6} ms")
    return ms

# Pre-load images
imgs = [Image.open(p).convert('RGB') for p in image_paths]
arrays = [np.array(img).astype(np.float32) / 255.0 for img in imgs]
batch = np.stack(arrays)

# ════════════════════════════════════════════════════════
# 1. IMAGE LOADING
# ════════════════════════════════════════════════════════
print("── Image Loading ──")

t = time.perf_counter()
loaded = [Image.open(p).convert('RGB') for p in image_paths]
lap("Load", f"Load {N_IMAGES} images (PIL)", t)

t = time.perf_counter()
arrs = [np.array(img).astype(np.float32) / 255.0 for img in loaded]
lap("Load", f"Convert to float32 ({N_IMAGES})", t)

t = time.perf_counter()
b = np.stack(arrs)
lap("Load", f"Stack to batch ({N_IMAGES}x{IMG_SIZE}x{IMG_SIZE}x3)", t)

# ════════════════════════════════════════════════════════
# 2. INDIVIDUAL TRANSFORMS
# ════════════════════════════════════════════════════════
print("\n── Individual Transforms ──")

t = time.perf_counter()
for img in imgs: img.resize((224, 224), Image.BILINEAR)
lap("Transform", f"Resize 224x224 ({N_IMAGES})", t)

t = time.perf_counter()
crop = T.CenterCrop(128)
for img in imgs: crop(img)
lap("Transform", f"CenterCrop 128 ({N_IMAGES})", t)

t = time.perf_counter()
rcrop = T.RandomCrop(128, padding=4)
for img in imgs: rcrop(img)
lap("Transform", f"RandomCrop 128 pad=4 ({N_IMAGES})", t)

t = time.perf_counter()
for arr in arrays: np.flip(arr, axis=1).copy()
lap("Transform", f"HorizontalFlip ({N_IMAGES})", t)

t = time.perf_counter()
for arr in arrays: np.flip(arr, axis=0).copy()
lap("Transform", f"VerticalFlip ({N_IMAGES})", t)

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
t = time.perf_counter()
for arr in arrays: (arr - mean) / std
lap("Transform", f"Normalize ImageNet ({N_IMAGES})", t)

weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
t = time.perf_counter()
for arr in arrays: np.dot(arr, weights)
lap("Transform", f"Grayscale ({N_IMAGES})", t)

t = time.perf_counter()
jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
for img in imgs: jitter(img)
lap("Transform", f"ColorJitter ({N_IMAGES})", t)

t = time.perf_counter()
for img in imgs: img.filter(ImageFilter.GaussianBlur(radius=2))
lap("Transform", f"GaussianBlur ({N_IMAGES})", t)

t = time.perf_counter()
rot = T.RandomRotation(10)
for img in imgs: rot(img)
lap("Transform", f"RandomRotation ±10° ({N_IMAGES})", t)

t = time.perf_counter()
erase = T.RandomErasing(p=1.0)
for arr in arrays:
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    erase(tensor)
lap("Transform", f"RandomErasing ({N_IMAGES})", t)

# ════════════════════════════════════════════════════════
# 3. BATCH TRANSFORMS
# ════════════════════════════════════════════════════════
print("\n── Batch Transforms ──")

t = time.perf_counter()
_ = (batch - mean) / std
lap("Batch", f"Batch normalize ({N_IMAGES})", t)

t = time.perf_counter()
_ = np.flip(batch, axis=2).copy()
lap("Batch", f"Batch HFlip ({N_IMAGES})", t)

t = time.perf_counter()
_ = np.dot(batch, weights)
lap("Batch", f"Batch grayscale ({N_IMAGES})", t)

t = time.perf_counter()
_ = np.flip(batch, axis=1).copy()
lap("Batch", f"Batch VFlip ({N_IMAGES})", t)

# ════════════════════════════════════════════════════════
# 4. PIPELINE COMPOSITIONS
# ════════════════════════════════════════════════════════
print("\n── Pipelines ──")

p1 = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
t = time.perf_counter()
for img in imgs: p1(img)
lap("Pipeline", f"Resize+Flip+Normalize ({N_IMAGES})", t)

p2 = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ColorJitter(0.2,0.2,0.2), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), T.RandomErasing(p=0.5)])
t = time.perf_counter()
for img in imgs: p2(img)
lap("Pipeline", f"Full augmentation ({N_IMAGES})", t)

p3 = T.Compose([T.Resize((224,224)), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
t = time.perf_counter()
for img in imgs: p3(img)
lap("Pipeline", f"Inference pipeline ({N_IMAGES})", t)

# ════════════════════════════════════════════════════════
# 5. DATA LOADER SIMULATION
# ════════════════════════════════════════════════════════
print("\n── DataLoader Simulation ──")

# Simulate loading batches of 32 with augmentation
t = time.perf_counter()
batch_size = 32
for start in range(0, N_IMAGES, batch_size):
    end = min(start + batch_size, N_IMAGES)
    batch_imgs = [Image.open(image_paths[i]).convert('RGB') for i in range(start, end)]
    batch_tensors = torch.stack([p1(img) for img in batch_imgs])
lap("DataLoader", f"Load+augment batches of 32 ({N_IMAGES})", t)

# ════════════════════════════════════════════════════════
# 6. IMAGE STATISTICS
# ════════════════════════════════════════════════════════
print("\n── Image Statistics ──")

t = time.perf_counter()
all_pixels = batch.reshape(-1, 3)
ch_mean = all_pixels.mean(axis=0)
ch_std = all_pixels.std(axis=0)
lap("Stats", f"Compute dataset mean/std ({N_IMAGES} images)", t)

# ════════════════════════════════════════════════════════
# 7. IMAGE SAVING
# ════════════════════════════════════════════════════════
print("\n── Image Saving ──")

os.makedirs("vision_bench_output", exist_ok=True)
t = time.perf_counter()
for i in range(100):
    imgs[i].save(f"vision_bench_output/out_{i:04d}.png")
lap("Save", f"Save 100 PNG", t)

# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
categories = {}
for r in results:
    categories[r["category"]] = categories.get(r["category"], 0) + r["ms"]
for cat, ms in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat:<30} {ms:>8,} ms")
total = sum(r["ms"] for r in results)
print(f"  {'TOTAL':<30} {total:>8,} ms")
print(f"{'═'*70}")

with open("vision_bench_output/python_vision_v2_results.json", "w") as f:
    json.dump(results, f, indent=2)
