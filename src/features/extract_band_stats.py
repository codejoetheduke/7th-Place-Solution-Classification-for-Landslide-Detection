import pandas as pd
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew

# CONFIG
BANDS = [f"B{i}" for i in range(1, 13)]
N_JOBS = 4

def process_npy_file(row):
    try:
        file_path = Path(row["sentinel_path"])
        ID = row["ID"]

        if not file_path.exists():
            logging.warning(f"[MISSING] {file_path}")
            return None

        arr = np.load(file_path)  # (H, W, 12)
        if arr.shape[-1] != 12:
            logging.warning(f"[SKIPPED] {ID}: got {arr.shape}")
            return None

        H, W = arr.shape[:2]
        flat_pixels = arr.reshape(-1, 12)
        valid = ~np.all(flat_pixels == 0, axis=1)
        flat_pixels = flat_pixels[valid]

        if flat_pixels.shape[0] == 0:
            logging.warning(f"[EMPTY] {ID}")
            return None

        df = pd.DataFrame(flat_pixels, columns=BANDS)
        df["ID"] = ID
        return df
    except Exception as e:
        logging.error(f"[ERROR] {file_path}: {e}")
        return None

def summarize_id_stats(group_tuple):
    ID, df = group_tuple
    stats = {"ID": ID}
    for band in BANDS:
        vals = df[band].values
        finite = vals[np.isfinite(vals)]
        if finite.size == 0: continue
        stats.update({
            f"{band}_mean": np.mean(finite),
            f"{band}_median": np.median(finite),
            f"{band}_min": np.min(finite),
            f"{band}_max": np.max(finite),
            f"{band}_std": np.std(finite),
            f"{band}_kurtosis": kurtosis(finite),
            f"{band}_skew": skew(finite),
            f"{band}_q25": np.percentile(finite, 25),
            f"{band}_q75": np.percentile(finite, 75),
        })
    return stats

def extract_features(train_df, test_df, output_dir="output"):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    logging.info("=== Extracting Sentinel-2 features ===")

    df_all = pd.concat([train_df.assign(split='train'), test_df.assign(split='test')])

    with Parallel(n_jobs=N_JOBS) as parallel:
        results = parallel(delayed(process_npy_file)(row) for _, row in tqdm(df_all.iterrows(), total=len(df_all)))

    valid_results = [r for r in results if r is not None]
    if not valid_results:
        raise RuntimeError("No valid .npy files processed.")

    pxdf = pd.concat(valid_results, ignore_index=True)
    logging.info(f"Pixel data shape: {pxdf.shape}")

    id_groups = pxdf.groupby("ID")
    with Parallel(n_jobs=N_JOBS) as parallel:
        stats_list = parallel(delayed(summarize_id_stats)(group) for group in tqdm(id_groups, total=len(id_groups)))

    agg_df = pd.DataFrame(stats_list)
    agg_df.to_csv(Path(output_dir) / "agg_df.csv", index=False)

    train_features = train_df.merge(agg_df, on="ID", how="left")
    test_features = test_df.merge(agg_df, on="ID", how="left")
    train_features.to_csv(Path(output_dir) / "train_features.csv", index=False)
    test_features.to_csv(Path(output_dir) / "test_features.csv", index=False)

    return train_features, test_features
