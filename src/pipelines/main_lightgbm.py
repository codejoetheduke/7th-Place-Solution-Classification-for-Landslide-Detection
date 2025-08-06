import argparse
import os
from pathlib import Path

# Import the pipeline functions
from src.features.extract_band_stats import extract_features
from src.models.train_lightgbm import train_lgb
from src.models.infer_lightgbm import predict_lgb
import pandas as pd

def main(mode, train_csv, test_csv, train_sentinel_dir, test_sentinel_dir, output_dir, model_dir, submission_path):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    if mode in ["preprocess", "all"]:
        print("=== STEP 1: Feature Extraction ===")
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        train_df['sentinel_path'] = train_df['ID'].apply(lambda id: os.path.join(train_sentinel_dir, f"{id}.npy"))
        test_df['sentinel_path'] = test_df['ID'].apply(lambda id: os.path.join(test_sentinel_dir, f"{id}.npy"))
        extract_features(train_df, test_df, output_dir=output_dir)

    if mode in ["train", "all"]:
        print("=== STEP 2: Training LightGBM ===")
        train_lgb(os.path.join(output_dir, "train_features.csv"), model_dir=model_dir)

    if mode in ["inference", "all"]:
        print("=== STEP 3: Inference ===")
        predict_lgb(
            test_path=os.path.join(output_dir, "test_features.csv"),
            model_dir=model_dir,
            output_path=submission_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["preprocess", "train", "inference", "all"], required=True,
                        help="Which step to run: preprocess | train | inference | all")
    parser.add_argument("--train_csv", type=str, default="/kaggle/input/slideandseekclasificationlandslidedetectiondataset/Train.csv")
    parser.add_argument("--test_csv", type=str, default="/kaggle/input/slideandseekclasificationlandslidedetectiondataset/Test.csv")
    parser.add_argument("--train_sentinel_dir", type=str, default="/kaggle/input/slideandseekclasificationlandslidedetectiondataset/train_data/train_data/")
    parser.add_argument("--test_sentinel_dir", type=str, default="/kaggle/input/slideandseekclasificationlandslidedetectiondataset/test_data/test_data/")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--model_dir", type=str, default="models/lightgbm")
    parser.add_argument("--submission_path", type=str, default="output/lgbm_submission.csv")
    args = parser.parse_args()

    main(
        mode=args.mode,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        train_sentinel_dir=args.train_sentinel_dir,
        test_sentinel_dir=args.test_sentinel_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        submission_path=args.submission_path
    )