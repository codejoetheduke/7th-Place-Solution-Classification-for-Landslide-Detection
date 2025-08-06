# src/models/infer_lightgbm.py
from src.features.feature_engineering_lightgbm import *
import argparse
import pandas as pd
import numpy as np
import joblib

def predict_lgb(test_path, model_dir, output_path, threshold=0.55):
    df = pd.read_csv(test_path)
    df = engineer_features(df)
    X = df.drop(columns=['ID', 'sentinel_path'])
    ids = df['ID']
    
    models = [joblib.load(f"{model_dir}/lgb_fold{i}.pkl") for i in range(10)]
    
    probs = np.zeros(X.shape[0])
    for model in models:
        probs += model.predict(X, num_iteration=model.best_iteration)
    probs /= len(models)
    preds = (probs > threshold).astype(int)
    
    pd.DataFrame({"ID": ids, "Probs": probs}).to_csv(output_path.replace(".csv", "_probs.csv"), index=False)
    pd.DataFrame({"ID": ids, "Target": preds}).to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    predict_lgb(args.test_path, args.model_dir, args.output_path)