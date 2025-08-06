import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from PIL import Image

def normalize_to_uint8(img):
    img = np.clip(img, 0, 10000)
    img = img / 10000.0
    img = (img * 255).astype(np.uint8)
    return img

def convert_npy_to_png(npy_folder, output_folder, bands=[3,2,1]):
    npy_folder = Path(npy_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for filename in tqdm(os.listdir(npy_folder), desc=f"Converting {npy_folder.name}"):
        if filename.endswith(".npy"):
            path = npy_folder / filename
            data = np.load(path)
            rgb = data[:, :, bands]
            rgb = normalize_to_uint8(rgb)
            image_pil = Image.fromarray(rgb)
            out_path = output_folder / filename.replace(".npy", ".png")
            image_pil.save(out_path)