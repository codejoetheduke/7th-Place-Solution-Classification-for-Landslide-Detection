# src/models/train_yolo.py
import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from src.utils.config_yolo import CFG, seed_everything

seed_everything(CFG.seed)

# === Helper functions ===
def prepare_classification_dir(df, base_dir):
    """Copy images into YOLO classification folder structure."""
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Preparing {base_dir.name}'):
        img_path = Path(row['image_path'])
        class_name = str(row['new_class'])
        class_dir = base_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, class_dir / img_path.name)

def prepare_test_dir(df, base_dir):
    """Copy test images into a flat directory."""
    for img_path in tqdm(df['image_path'].unique(), desc=f'Preparing test'):
        shutil.copy2(img_path, base_dir / Path(img_path).name)

# === Main training function ===
def train_yolo(
    train_csv,
    test_csv,
    images_dir,
    model_dir,
    folds=CFG.folds,
    epochs=CFG.epochs,
    imgsz=CFG.imgsz,
    device=CFG.device,
    batch=CFG.batch,
    optimizer=CFG.optimizer,
    lr0=CFG.lr0,
    momentum=CFG.momentum,
    weight_decay=CFG.weight_decay,
    close_mosaic=CFG.close_mosaic,
    seed=CFG.seed,
    patience=CFG.patience
):
    # Load train/test
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    # Map class labels to indices
    unique_classes = sorted(train['label'].unique())
    full_label_dict = {cls: idx for idx, cls in enumerate(unique_classes)}
    train['new_class'] = train['label'].map(full_label_dict)

    # Group multilabels
    grouped = train.groupby('ID')['new_class'].apply(list).reset_index()
    for cls in unique_classes:
        grouped[cls] = -1
    reverse_mapping = {v: k for k, v in full_label_dict.items()}
    for idx, labels in enumerate(grouped['new_class']):
        for lbl in set(labels):
            grouped.loc[idx, reverse_mapping[lbl]] = 1

    # Multilabel stratified folds
    grouped['fold'] = -1
    mskf = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for i, (_, val_idx) in enumerate(mskf.split(grouped[['ID']], grouped[unique_classes])):
        grouped.loc[val_idx, 'fold'] = i

    # Add image paths
    grouped['image_path'] = grouped['ID'].apply(lambda x: Path(images_dir) / f"{x}.png")
    test['image_path'] = test['ID'].apply(lambda x: Path(images_dir.replace('train', 'test')) / f"{x}.png")

    # Loop over folds
    for fold in range(folds):
        print(f"\n🚀 Training Fold {fold + 1}/{folds}")

        # Define directories
        fold_root = Path(model_dir) / f'fold_{fold + 1}'
        train_dir = fold_root / 'train'
        val_dir = fold_root / 'val'
        test_dir = Path(model_dir) / 'test/images'

        # Clean dirs
        for d in [train_dir, val_dir, test_dir]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        # Prepare data for YOLO classification
        prepare_classification_dir(grouped[grouped['fold'] != fold], train_dir)
        prepare_classification_dir(grouped[grouped['fold'] == fold], val_dir)
        prepare_test_dir(test, test_dir)

        # Train YOLO
        model = YOLO("yolo11l-cls.pt")
        model.train(
            data=str(fold_root),
            epochs=epochs,
            imgsz=imgsz,
            device=device,
            batch=batch,
            optimizer=optimizer,
            lr0=lr0,
            momentum=momentum,
            weight_decay=weight_decay,
            close_mosaic=close_mosaic,
            seed=seed,
            patience=patience,
            project=str(model_dir),
            name=f"fold_{fold + 1}",
            exist_ok=True
        )

        # Save best weights to fold-specific directory
        src = Path(model_dir) / f"fold_{fold + 1}/weights/best.pt"
        dst = fold_root / "best.pt"
        shutil.copy(src, dst)
        print(f"✅ Fold {fold + 1} model saved at {dst}")