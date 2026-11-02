# VinUni Buildings Detection

**Instance segmentation of VinUniversity's Building K using YOLOv8-seg.**

A COMP4040 final project at VinUniversity. The pipeline records campus walkaround videos, extracts and annotates frames, trains a YOLOv8 segmentation model, and runs real-time polygon-mask inference on live video.

---

## Results

| Model | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| V1 – YOLOv8n-seg (baseline) | 0.997 | 0.503 | 0.592 | 0.489 |
| V2 – YOLOv8s-seg (larger backbone) | — | — | — | — |
| V3 – YOLOv8s-seg + tuned hyperparams | — | ~0.65 | ~0.697 | — |

Recall improved from **50% → 65%** and mAP@50 reached **0.697** across iterative versions.

---

## Repository Structure

```
VinUni-Buildings-Detection/
├── notebooks/
│   ├── 01_extract_frames.ipynb      # Extract & filter frames from raw videos (blur rejection)
│   ├── 02_train_and_evaluate.ipynb  # YOLOv8-seg training, evaluation, and visualization
│   └── 03_detect_video.ipynb        # Run V2/V3 models on a video, save annotated output
├── models/
│   ├── building_k_v1_nano.pt        # V1 trained weights (YOLOv8n-seg, 6.7 MB)
│   ├── building_k_v2_best.pt        # V2 trained weights (YOLOv8s-seg, 23 MB)
│   └── building_k_v3_best.pt        # V3 trained weights (YOLOv8s-seg, 23 MB)
├── docs/
│   ├── Milestone1_report.pdf        # Full project report (dataset, pipeline, results)
│   └── tag_distribution.png         # Dataset metadata tag distribution chart
├── data.yaml                        # Roboflow dataset config (train/val/test paths + class)
├── README.dataset.txt               # Roboflow dataset export notes
└── README.md
```

> **Dataset not included.** The annotated image dataset (512 images, YOLOv8 segmentation format) is hosted on Roboflow:
> [universe.roboflow.com/tuan-trinh-bfldh/vinuni_buildings_dataset](https://universe.roboflow.com/tuan-trinh-bfldh/vinuni_buildings_dataset/dataset/1)

---

## Dataset

- **Source:** 12 walkaround videos of Building K recorded on campus
- **Frames extracted:** 579 total, filtered for blur (Laplacian variance < 100 threshold)
- **Annotated images:** 512 (polygon masks, YOLOv8 segmentation format via Roboflow)
- **Split:** 422 train / 60 val / 30 test
- **Class:** `building_k` (1 class)
- **Preprocessing:** auto-orient, resize to 640×640
- **Augmentation:** random 90° rotations (4 variants per image)
- **Metadata tags engineered:** lighting condition, occlusion level, camera orientation — used for systematic error analysis across conditions

---

## Notebooks

### `01_extract_frames.ipynb`
Iterates over raw `.mp4` videos, samples 1 frame/second, rejects blurry frames (Laplacian variance threshold), and saves clean frames for annotation. Run on Google Colab with Drive mounted.

### `02_train_and_evaluate.ipynb`
Main pipeline: loads dataset from Roboflow, trains YOLOv8-seg on Google Colab (Tesla T4 GPU), evaluates on val/test sets, and generates visualization outputs. Open in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1syTwRbco7AIkmFUKP-jzgmpOF6m-Cb33)

### `03_detect_video.ipynb`
Loads V2 (red mask) and V3 (blue mask) models, runs inference on a source video frame-by-frame, overlays polygon masks and confidence scores, and saves annotated `.mp4` output to Drive.

---

## Setup

All notebooks are designed to run on **Google Colab** with GPU acceleration. No local installation required.

For local inference:

```bash
pip install ultralytics opencv-python
```

```python
from ultralytics import YOLO

model = YOLO("models/building_k_v3_best.pt")
results = model("path/to/video.mp4", stream=True, conf=0.90)
```

---

## Key Findings

- **High precision, moderate recall** in V1: the model rarely produces false positives, but misses detections in challenging conditions (occlusion, unusual angles, low light).
- **Recall bottleneck** traced to training images lacking diversity in lighting and orientation — addressed in subsequent versions by adding more varied frames.
- **V3 improvements** came from switching to a larger backbone (YOLOv8s vs. YOLOv8n), adjusting learning rate, and enabling mixup augmentation.
- **Live video deployment** validated that the model generalizes to non-static camera motion at 3840×2160 resolution.

---

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — instance segmentation
- [Roboflow](https://roboflow.com) — dataset annotation and management
- Google Colab — training (Tesla T4 / A100 GPU)
- OpenCV — frame extraction and video rendering

---

## Course

COMP4040 · VinUniversity · 2026  
Author: Hung Tuan Trinh
