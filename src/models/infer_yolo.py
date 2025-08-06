#src/models/infer_yolo.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from src.utils.config_yolo import seed_everything, CFG
from tqdm import tqdm

def infer_yolo(test_csv,test_dir, model_dir, output_path, folds=CFG.folds, threshold=0.5):
    test = pd.read_csv(test_csv)
    test_images_dir = Path(test_dir)
    image_files = sorted(os.listdir(test_images_dir))
    
    all_probs = []
    for fold in range(1, folds+1):
        model = YOLO(str(Path(model_dir) / f'fold_{fold}/best.pt'))
        fold_probs = []
        for img in tqdm(image_files, desc=f"Fold {fold} inference"):
            results = model(test_images_dir / img, verbose=False, imgsz=CFG.imgsz)
            probs = results[0].probs.data.cpu().numpy()
            fold_probs.append(float(probs[1]))  # Probability of class 1
        all_probs.append(fold_probs)

    # Average probabilities
    probs_matrix = np.array(all_probs).T
    avg_probs = probs_matrix.mean(axis=1)
    preds = (avg_probs > threshold).astype(int)
    
    # Save
    df = pd.DataFrame({
        'ID': [f.replace('.png', '') for f in image_files],
        'Probs': avg_probs,
        'Predicted': preds
    })
    df.to_csv(output_path, index=False)
    print(f"✅ Saved predictions to {output_path}")