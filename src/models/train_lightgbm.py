# src/models/train_lightgbm.py
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from src.features.feature_engineering_lightgbm import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
from pathlib import Path

# === LightGBM Training ===
def train_lgb(train_path, model_dir, n_splits=10):
    df = pd.read_csv(train_path)
    df = engineer_features(df)
    X = df.drop(columns=['ID', 'sentinel_path','label'])
    y = df['label']

    classes = np.unique(y)
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    sample_weights = class_weights[y]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models, f1_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        sw_train = sample_weights[train_idx]

        train_data = lgb.Dataset(X_train, label=y_train, weight=sw_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'random_state': 42,
            'early_stopping_rounds':100,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
        )

        y_proba = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = (y_proba > 0.5).astype(int)
        fold_f1 = f1_score(y_val, y_pred)
        f1_scores.append(fold_f1)
        print(f"Fold {fold+1} F1: {fold_f1:.4f}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, f"{model_dir}/lgb_fold{fold}.pkl")
        models.append(model)

    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    return models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="output/train_features.csv")
    parser.add_argument("--model_dir", type=str, default="models/lightgbm")
    args = parser.parse_args()
    train_lgb(args.train_path, args.model_dir)