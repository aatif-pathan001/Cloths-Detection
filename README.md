# Clothes Detection with YOLOv8

![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00BFFF?style=flat-square&logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

> **Custom object detection for clothing items using YOLOv8** — Fine-tuned on a custom-labeled dataset to detect and classify clothing categories in real-time from images and video.

---

## 🎯 Overview

This project trains a **YOLOv8 object detection model** on a custom clothing dataset to detect and localize garment types in images. The model learns to draw bounding boxes around clothing items and classify them into categories such as lower-wear, top-wear, dress, and more.

This project demonstrates the full custom object detection workflow — from raw data collection and annotation to training, evaluation, and inference.

---

## 📚 Learning Objectives / Goals

- [x] Understand YOLO architecture and anchor-based detection
- [x] Build and structure a custom dataset with proper annotations
- [x] Train YOLOv8 on a domain-specific object detection task
- [x] Evaluate model performance using mAP, Precision, Recall
- [x] Run inference on new images and visualize detections
- [ ] Deploy model via Gradio / FastAPI for live demo
- [ ] Export model to ONNX for cross-platform inference

---

## 🗂️ Repository Structure

```
Cloths-Detection/
├── README.md
├── requirements.txt
├── dataset/
│   ├── images/
│   │   ├── train/              # Training images
│   │   └── val/                # Validation images
│   ├── labels/
│   │   ├── train/              # YOLO-format .txt annotations
│   │   └── val/
│   └── data.yaml               # Dataset config (classes, paths)
├── notebooks/
│   └── clothes_detection.ipynb # Full training & inference notebook
└── runs/
    └── detect/
        └── train/              # YOLOv8 training outputs (auto-generated)
            ├── weights/
            │   ├── best.pt     # Best checkpoint (use for inference)
            │   └── last.pt     # Final epoch checkpoint
            ├── results.png     # Training curves
            ├── confusion_matrix.png
            └── val_batch*.jpg  # Validation prediction samples
```

---

## 📖 Project Modules

### 1. Dataset Preparation
**Status**: ✅ Complete

**Objective**: Collect, annotate, and structure a custom clothing image dataset in YOLO format.

**Approach**:
- Collected clothing images covering multiple garment categories
- Annotated bounding boxes using [Roboflow](https://roboflow.com) / LabelImg
- Exported annotations in YOLO format (normalized `[class x_center y_center width height]`)
- Split into train/val sets and defined `dataset/data.yaml` config

**Key Learnings**:
- YOLO annotation format: one `.txt` per image, one row per detected object
- Balanced class distribution is critical for avoiding biased detections
- Data augmentation (flip, mosaic, HSV shift) significantly reduces overfitting on small datasets

---

### 2. YOLOv8 Model Training
**Status**: ✅ Complete

**Objective**: Fine-tune a pretrained YOLOv8 model on the custom clothing dataset using transfer learning.

**Approach**:
- Loaded `yolov8n.pt` (nano) pretrained on COCO as starting weights
- Fine-tuned with Ultralytics `model.train()` API
- Monitored loss curves, mAP, and validation batch predictions per epoch

**Training Configuration**:

| Parameter | Value |
|-----------|-------|
| Base Model | `yolov8m.pt` (COCO pretrained) |
| Epochs | 50 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Augmentations | Mosaic, Flip, HSV Jitter |

---

### 3. Evaluation & Results
**Status**: ✅ Complete

**Objective**: Measure model performance on the held-out validation set.

**Approach**:
- Computed mAP@0.5 and mAP@0.5:0.95 on validation split
- Generated confusion matrix to identify per-class errors
- Visualized predicted vs ground-truth bounding boxes on val batches

---

### 4. Inference Pipeline
**Status**: ✅ Complete

**Objective**: Run the trained model on new, unseen images with bounding box visualization.

**Approach**:
- Load `runs/detect/train/weights/best.pt`
- Run predictions with configurable confidence threshold
- Output images with class labels and confidence scores overlaid

---

## 📊 Results

### Performance Summary

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.70 |
| mAP@0.5:0.95 | 0.59 |
| Precision | 0.78 |
| Recall | 0.6 |
| Inference Speed | 2.3ms / per image (GPU) |

> 📁 Full training curves, confusion matrix, and val batch samples are saved in `runs/detect/train/`.

### Sample Detection Output


---

## 🛠️ Tech Stack

| Category | Tool / Library |
|----------|---------------|
| Language | Python 3.9+ |
| Detection Framework | Ultralytics YOLOv8 |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Annotation Tool | Roboflow / LabelImg |
| Notebook | Jupyter Notebook |
| Visualization | Matplotlib |
| Experiment Output | Ultralytics built-in (`runs/detect/`) |

---


## 📈 Roadmap

**Completed**:
- [x] Dataset annotation in YOLO format
- [x] YOLOv8 fine-tuning on custom clothing classes
- [x] mAP evaluation and confusion matrix
- [x] Inference pipeline on new images

**Future Enhancements**:
- [ ] Gradio UI for live image / webcam upload and detection
- [ ] Export model to ONNX for edge/mobile deployment
- [ ] Expand dataset with more classes (accessories, footwear, bags)
- [ ] Benchmark across YOLOv8 model sizes (n → s → m) for accuracy vs speed tradeoff
- [ ] Real-time webcam detection demo script

---

## 🤝 Connect

- **GitHub**: [@aatif-pathan001](https://github.com/aatif-pathan001)
- **LinkedIn**: [Aatif Khan Pathan](https://linkedin.com/in/aatif-khan-pathan)

---

**Author**: Aatif Khan Pathan  
