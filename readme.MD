# 📝 7th Place Landslide Detection — Submission

## 🚀 Overview

This repository contains the final submission code for the "Classification for Landslide Detection" challenge. The pipeline includes **four trained models** (EVA, EfficientNetV2, YOLOv11, LightGBM) and a final ensemble to generate predictions.

### Directory Structure

```
Submission/
├── efficientvnet_setup/
├── eva_setup/
├── lightgbm_setup/
├── yolo_setup/
├── output/                      ← inference results, intermediate files
├── Trustworthiness_Evaluation_Final.pdf
└── ensembles.py                 ← final blending script
```

---

## 🎯 Problem Statement

Identify landslide vulnerabilities from multi-band satellite data. The labels are binary (landslide vs. non-landslide), and success is measured via classification accuracy on hidden test data.

---

## ⚙️ Repository Contents

### `efficientvnet_setup/`, `eva_setup/`, `lightgbm_setup/`, `yolo_setup/`

Each directory contains one Kaggle notebook that:

1. Loads data (`Train.csv`, `Test.csv`, `.npy` files).
2. Runs feature engineering or image preparation.
3. Trains the respective model with fixed fold settings.
4. Saves test predictions as `.npy`.

### `output/`

Contains:

- `test_preds_fold*.npy` (EVA).
- `submission_probs_efficientnet.csv`.
- `submission_probs_lightgbm.csv`.
- `ensemble_with_probs (1).csv` (YOLO + LightGBM blending).
- Other intermediary CSVs used during ensembling.

### `ensembles.py`

Blends model outputs using frozen weights to generate final submission (`sub_last_ensemble.csv`).

### `Trustworthiness_Evaluation_Final.pdf`

Explains bias mitigation, reproducibility, and sustainability—in line with Zindi WG trustworthiness criteria.

---

## ✅ Step-by-Step Instructions (Zindi Reviewer Flow)

### 1. Setup Environment

Ensure you have:

- Python ≥ 3.10, GPU-enabled (preferably ≥ 8 GB VRAM)
- OS: Ubuntu or compatible Linux
- RAM: ≥ 16 GB
- Requirements installed via:

  ```bash
  pip install -r requirements.txt
  ```

_(Expect packages: pytorch, timm, lightgbm, albumentations, fastai, numpy, pandas, scikit-learn, ultralytics)_

### 2. Run Model Notebooks

Execute (in order) each notebook:

1. `lightgbm_setup/`
2. `efficientvnet_setup/`
3. `eva_setup/`
4. `yolo_setup/`

These notebooks:

- Preprocess the data.
- Train a fixed 5–10 fold cross-validation.
- Output predictions (OOF & test).

**Ensure outputs are saved to `output/` folder**.

### 3. Run Final Ensemble

```bash
python ensembles.py
```

This will:

- Merge outputs using specified weights:

  - **YOLO + EfficientNet**: 0.57 / 0.43
  - **Then with LightGBM**: 0.60 / 0.40
  - **Then with EVA**: final blend weight 0.45, threshold at 0.52.

- Generate the final submission: `sub_last_ensemble.csv`.

---

## 🧠 Ensemble Logic (for reference)

```text
E1 = 0.57 * YOLO + 0.43 * EfficientNet
E2 = 0.60 * E1  + 0.40 * LightGBM
Final = 0.45 * EVA + 0.55 * E2
Binary Target = Final > 0.52
```

---

## 🧩 Folder Layout & Paths

- **Inputs**: `data/*.csv`, `*.npy`
- **Intermediate outputs**: `output/train_features.csv`, `output/test_features.csv`
- **Saved models**: within each notebook folder or `models/` directory.
- **Final outputs**:

  - prediction files in `output/`
  - final submission: `sub_last_ensemble.csv`

---

## 🔍 Evaluation Insights

Complies with Zindi guidelines on reproducibility:

- **Fixed seeds**, deterministic splits, and clear fold structure across notebooks.
- **Ensemble script uses versioned weights**, avoiding randomness.
- Includes **Trustworthiness evaluation** PDF for data and model bias transparency.

---

## 🚨 Common Issues & Troubleshooting

| Problem                                    | Recommended Fix                                                                                                                                |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Notebooks fail due to missing `.npy` files | Ensure the dataset is placed in `/kaggle/input/slideandseekclasificationlandslidedetectiondataset/` or update paths in `main.py` and notebooks |
| Wrong dataset path                         | Update `--train_csv`, `--test_csv`, `--train_sentinel_dir`, `--test_sentinel_dir` arguments when running scripts                               |
| GPU memory errors                          | Reduce batch size, image resolution, or number of folds in the training scripts                                                                |
| Output folder missing predictions          | Confirm all notebooks/scripts write predictions to the `output/` folder                                                                        |

---

### **EVA Model Reproducibility Notice**

The **EVA model** (trained using FastAI) had **reproducibility issues** despite extensive seeding (deterministic training, fixed seeds for dataloaders and augmentation).
To avoid inconsistencies, **we are providing pre-trained models** that can be downloaded directly from Kaggle:
👉 [Landslide Detection FastAI Models (Kaggle Notebook)](https://www.kaggle.com/code/dukekojokongo/landslide-detection-fastai)

If you wish to **skip training**:

- Go to the linked notebook.
- Download the `models_folds/*.pkl` and `oof/test_preds_fold*.npy` files.
- Place them inside `eva_setup/output/` before running `ensembles.py`.

---

## 📦 Submission Checklist (Zindi Requirements)

- [x] Clear README with instructions
- [x] Working inference code (via `ensembles.py`)
- [x] Model outputs versioned and reproducible
- [x] Trustworthiness evaluation attached
- [x] No proprietary data—only processed features

---

This documentation ensures your solution is **reproducible, transparent, and easy to review**, in alignment with Zindi’s official submission standards.
