import argparse
from pathlib import Path
from src.features.convert_npy_to_png import convert_npy_to_png
from src.models.train_yolo import train_yolo
from src.models.infer_yolo import infer_yolo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--train_csv", type=str, default="/kaggle/input/slideandseekclasificationlandslidedetectiondataset/Train.csv")
    parser.add_argument("--test_csv", type=str, default="/kaggle/input/slideandseekclasificationlandslidedetectiondataset/Test.csv")
    parser.add_argument("--train_npy_dir", type=str, default="/kaggle/input/slideandseekclasificationlandslidedetectiondataset/train_data/train_data")
    parser.add_argument("--test_npy_dir", type=str, default="/kaggle/input/slideandseekclasificationlandslidedetectiondataset/test_data/test_data")
    parser.add_argument("--image_dir", type=str, default="train_data_sentinel_png")
    parser.add_argument("--test_dir", type=str, default="test_data_sentinel_png")
    parser.add_argument("--model_dir", type=str, default="models/yolo")
    parser.add_argument("--output_path", type=str, default="output/yolo_submission.csv")
    args = parser.parse_args()

    if args.mode == "train":
        # Step 1: Convert npy → png (if not already done)
        if not Path(args.image_dir).exists() or len(list(Path(args.image_dir).glob("*.png"))) == 0:
            print(f"⚙️ Converting training npy files from {args.train_npy_dir} to PNG at {args.image_dir}")
            convert_npy_to_png(args.train_npy_dir, args.image_dir)
        if not Path(args.test_dir).exists() or len(list(Path(args.test_dir).glob("*.png"))) == 0:
            print(f"⚙️ Converting test npy files from {args.test_npy_dir} to PNG at {args.test_dir}")
            convert_npy_to_png(args.test_npy_dir, args.test_dir)

        # Step 2: Train YOLO models
        train_yolo(args.train_csv, args.test_csv, args.image_dir, args.model_dir)

    elif args.mode == "infer":
        infer_yolo(args.test_csv,args.test_dir, args.model_dir, args.output_path)