"""
PandaSharp.Vision vs Python — Video + ONNX Model Benchmark
===========================================================
"""
import time, os, json, numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as T

VIDEO_PATH = "vision_bench_models/test_video.mp4"
MODEL_PATH = "vision_bench_models/mobilenetv2.onnx"
IMG_DIR = "vision_bench_images"

print("=== Python Video + ONNX Benchmark ===\n")

results = []
def lap(cat, name, start):
    ms = round((time.perf_counter() - start) * 1000)
    results.append({"category": cat, "op": name, "ms": ms})
    print(f"  {name:<55} {ms:>6} ms")
    return ms

# ════════════════════════════════════════════════════════
# 1. VIDEO FRAME EXTRACTION
# ════════════════════════════════════════════════════════
print("── Video Frame Extraction ──")

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"  Video: {total_frames} frames, {fps:.0f} fps")
cap.release()

# Extract all frames to memory
t = time.perf_counter()
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
lap("Video", f"Extract all frames to memory ({len(frames)})", t)

# Extract every 10th frame
t = time.perf_counter()
cap = cv2.VideoCapture(VIDEO_PATH)
sampled_frames = []
idx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if idx % 10 == 0:
        sampled_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    idx += 1
cap.release()
lap("Video", f"Extract every 10th frame ({len(sampled_frames)})", t)

# Extract + resize frames
t = time.perf_counter()
cap = cv2.VideoCapture(VIDEO_PATH)
resized_frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    resized = cv2.resize(frame, (224, 224))
    resized_frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
cap.release()
lap("Video", f"Extract + resize to 224x224 ({len(resized_frames)})", t)

# Extract frames to disk
os.makedirs("vision_bench_output/video_frames_py", exist_ok=True)
t = time.perf_counter()
cap = cv2.VideoCapture(VIDEO_PATH)
fidx = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if fidx % 10 == 0:
        cv2.imwrite(f"vision_bench_output/video_frames_py/frame_{fidx:06d}.png", frame)
    fidx += 1
cap.release()
lap("Video", f"Extract every 10th to disk ({fidx // 10} PNGs)", t)

# Convert frames to float batch
t = time.perf_counter()
float_frames = np.stack([f.astype(np.float32) / 255.0 for f in resized_frames])
lap("Video", f"Frames to float batch ({float_frames.shape})", t)

# ════════════════════════════════════════════════════════
# 2. ONNX MODEL INFERENCE
# ════════════════════════════════════════════════════════
print("\n── ONNX Model Inference ──")

# Load model
t = time.perf_counter()
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
lap("ONNX", f"Load MobileNetV2 model", t)
print(f"  Input: {input_name} shape={input_shape}")

# Preprocessing pipeline
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Single image inference
image_paths = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.png')])[:100]

t = time.perf_counter()
for p in image_paths[:10]:
    img = Image.open(p).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).numpy()  # [1, 3, 224, 224]
    session.run(None, {input_name: tensor})
lap("ONNX", f"Single image inference (10 images)", t)

# Batched inference
t = time.perf_counter()
batch_tensors = []
for p in image_paths[:32]:
    img = Image.open(p).convert('RGB')
    batch_tensors.append(preprocess(img))
import torch
batch = torch.stack(batch_tensors).numpy()  # [32, 3, 224, 224]
session.run(None, {input_name: batch})
lap("ONNX", f"Batch inference (32 images)", t)

# Full dataset inference in batches
t = time.perf_counter()
all_preds = []
for start in range(0, len(image_paths), 32):
    end = min(start + 32, len(image_paths))
    batch_tensors = []
    for p in image_paths[start:end]:
        img = Image.open(p).convert('RGB')
        batch_tensors.append(preprocess(img))
    batch = torch.stack(batch_tensors).numpy()
    output = session.run(None, {input_name: batch})
    preds = np.argmax(output[0], axis=1)
    all_preds.extend(preds.tolist())
lap("ONNX", f"Full inference pipeline (100 images, batch=32)", t)

# Embedding extraction (use second-to-last layer output)
t = time.perf_counter()
for p in image_paths[:32]:
    img = Image.open(p).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).numpy()
    output = session.run(None, {input_name: tensor})
    embedding = output[0].flatten()  # Use logits as "embedding"
lap("ONNX", f"Embedding extraction (32 images)", t)

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

with open("vision_bench_output/python_advanced_results.json", "w") as f:
    json.dump(results, f, indent=2)
