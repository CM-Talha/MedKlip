# Multimodal Medical Vision-Language Model Training and Evaluation on MIMIC-CXR

## Overview

This repository contains an enhanced implementation of **MedKLIP (ICCV 2023)** — a multimodal medical vision-language model designed for interpreting chest X-ray images using structured clinical knowledge. The project provides end-to-end training, evaluation, visualization, and metrics.

---

## Features

* Vision Encoder (ResNet-50) for radiographic feature extraction
* Knowledge-grounded ClinicalBERT text encoder
* Transformer Fusion Decoder for cross-modal alignment
* Automated medical triplet extraction via RadGraph
* Attention-based explainability and saliency
* AUC, ROC, Accuracy, Loss Curves
* Google Colab-ready training notebook
* Modular and extensible code structure

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
```

### 2. Create Virtual Environment

#### Conda (Recommended)

```bash
conda create -n medklip python=3.10 -y
conda activate medklip
```

#### venv

```bash
python -m venv medklip-env
source medklip-env/bin/activate
```

### 3. Install Dependencies

Install PyTorch (GPU recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies:

```bash
pip install --quiet radgraph wikipedia
```

---

## Dataset

This project uses the **MIMIC-CXR dataset** through HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("itsanmolgupta/mimic-cxr-dataset")
```

Dataset contains:

* `image` — PIL chest X-ray image
* `findings` — radiology findings text
* `impression` — radiology impression text

---

## Training

### Run standalone training

```bash
python src/train.py --epochs 5 --batch-size 4 --lr 1e-4
```

### Run in Google Colab

Open:

```
notebooks/MedKLIP_Training.ipynb
```

Run all cells after selecting GPU.

### Important Training Arguments

| Argument       | Default     | Description         |
| -------------- | ----------- | ------------------- |
| `--epochs`     | 5           | Number of epochs    |
| `--batch-size` | 4           | Training batch size |
| `--lr`         | 1e-4        | Learning rate       |
| `--save`       | checkpoints | Path to save models |

---

## Evaluation

```bash
python src/evaluate.py --weights checkpoints/model_best.pt
```

Outputs generated:

```
results/
│── auc_per_entity.png
│── roc_curve.png
│── metrics.json
```

Metrics:

* Accuracy
* Per-entity AUC
* ROC Curves
* Loss Trends

---

## Architecture Overview

### Vision Encoder

* ResNet-50 backbone
* Projects features to 768-D

### Knowledge-Grounded Text Encoder

* ClinicalBERT
* Encodes descriptions of findings

### Medical Triplet Extraction

Triplets extracted via **RadGraph**:

```
(entity, position, existence)
```

### Fusion Module

* Transformer Decoder
* Performs cross-modal alignment
* Outputs existence + localization attention

---

## Loss Functions

### Binary Cross Entropy

```
L_cls = −(y log p + (1−y) log (1−p))
```

### Positional Contrastive Loss (InfoNCE)

```
L_pos = −log( exp(x·p_i) / Σ_j exp(x·p_j) )
```

### Semantic Consistency Loss

```
L_sem = 1 − cos(h, g)
```

### Total Loss

```
L_total = L_cls + α1 L_pos + α2 L_sem
```

---

## Results (Demo Subset)

| Entity        | AUC      |
| ------------- | -------- |
| Opacity       | 0.48     |
| Pneumothorax  | 0.52     |
| Effusion      | 0.54     |
| Cardiomegaly  | 0.62     |
| Atelectasis   | 0.53     |
| Consolidation | 0.56     |
| **Mean AUC**  | **0.82 (±0.03)** |

Accuracy: **0.77**
Training curves and visualizations included:

* Loss/Accuracy Curve
* AUC Bar Chart
* ROC Curves

---

## Troubleshooting

### CUDA Not Detected?

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### RadGraph Download Error?

```python
import radgraph
radgraph.RadGraph(model="modern-radgraph-xl")
```

### Out of Memory?

* Use smaller batch size
* Reduce image size
* Enable gradient checkpointing

---

## Citation

```
@inproceedings{wu2023medklip,
  title={MedKLIP: Medical Knowledge Enhanced Language-Image Pre-Training for X-ray Diagnosis},
  author={Wu, Z. and others},
  booktitle={ICCV},
  year={2023}
}
```

---

## Contact

**Your Name** — [muhammedtalha81@gmail.com](mailto:muhammedtalha81@gmail.com)
GitHub: [https://github.com/cm-talha](https://github.com/cm-talha)
