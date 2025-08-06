import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import timm
from torch.nn.functional import sigmoid
from src.utils.config_efficientnet import CFG, seed_everything
import random
import warnings
warnings.filterwarnings('ignore')

seed_everything(CFG.seed)

# === Dataset ===
class SentinelDataset(Dataset):
    def __init__(self, dataframe, transform=None, to_train=True):
        self.dataframe = dataframe
        self.transform = transform
        self.to_train = to_train

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['path']
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        if self.transform:
            image = self.transform(image=image)["image"]
        if self.to_train:
            target = torch.tensor(row["label"], dtype=torch.float32)
            return image, target
        return image

# === Model ===
class EfficientNetV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnetv2_rw_m", pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features, 1)
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# === Transforms ===
train_transforms = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=5, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
test_transforms = A.Compose([
    A.Resize(CFG.img_size, CFG.img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# === Training ===
def train_one_fold(fold, train_loader, val_loader, model_dir):
    seed_everything(42)
    model = EfficientNetV2Classifier().cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)  # <-- FIX: match script
    scaler = torch.amp.GradScaler('cuda')
    best_loss = float("inf")
    fold_path = Path(model_dir) / f"fold_{fold+1}"
    fold_path.mkdir(parents=True, exist_ok=True)
    best_model_path = fold_path / "best.pt"

    for epoch in range(CFG.epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{CFG.epochs} - Train"):
            images, targets = images.cuda(), targets.cuda().unsqueeze(1).float()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for images, targets in val_loader:
                images, targets = images.cuda(), targets.cuda().unsqueeze(1).float()
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Fold {fold+1} Epoch {epoch+1}: Train Loss {epoch_loss/len(train_loader):.4f}, Val Loss {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        scheduler.step()
    return best_model_path

# === Main Training ===
def train_efficientnet(train_csv, test_csv, train_imgs, test_imgs, model_dir):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    train_df = train_df.groupby("ID").sum().reset_index()[["ID", "label"]]
    train_df["path"] = train_imgs + "/" + train_df["ID"] + ".png"
    test_df["path"] = test_imgs + "/" + test_df["ID"] + ".png"
    skf = StratifiedKFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
    train_df["stratify_label"] = train_df[["label"]].sum(axis=1)
    for fold, (_, valid_idx) in enumerate(skf.split(train_df, train_df["stratify_label"])):
        train_df.loc[valid_idx, "fold"] = fold
    for fold in range(CFG.folds):
        train_data = train_df[train_df["fold"] != fold].reset_index(drop=True)
        valid_data = train_df[train_df["fold"] == fold].reset_index(drop=True)
        train_loader = DataLoader(SentinelDataset(train_data, transform=train_transforms), batch_size=CFG.batch_size, shuffle=True, num_workers=os.cpu_count())
        val_loader = DataLoader(SentinelDataset(valid_data, transform=test_transforms), batch_size=CFG.batch_size, shuffle=False)
        train_one_fold(fold, train_loader, val_loader, model_dir)

# === Inference (Ensemble of folds) ===
def infer_efficientnet(test_csv, test_imgs, model_dir, out_csv):
    test_df = pd.read_csv(test_csv)
    test_df["path"] = test_imgs + "/" + test_df["ID"] + ".png"
    test_loader = DataLoader(SentinelDataset(test_df, transform=test_transforms, to_train=False), batch_size=CFG.batch_size, shuffle=False)
    probs = np.zeros(len(test_df))
    model = EfficientNetV2Classifier().cuda()
    for fold in range(CFG.folds):
        fold_model = Path(model_dir) / f"fold_{fold+1}" / "best.pt"
        model.load_state_dict(torch.load(fold_model))
        model.eval()
        fold_preds = []
        with torch.inference_mode():
            for images in tqdm(test_loader, desc=f"Inference Fold {fold+1}"):
                images = images.cuda()
                logits = model(images)
                preds = sigmoid(logits).squeeze(1).cpu().numpy()
                fold_preds.append(preds)
        probs += np.concatenate(fold_preds)
    probs /= CFG.folds
    pd.DataFrame({"ID": test_df["ID"], "probability": probs}).to_csv(out_csv, index=False)
    print(f"✅ Predictions saved to {out_csv}")