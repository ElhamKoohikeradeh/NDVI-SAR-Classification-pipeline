from pathlib import Path

import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier


def load_masks(period_key, mask_files):
    with rasterio.open(mask_files[period_key]["soil"]) as s:
        soil = s.read(1)
        prof = s.profile

    with rasterio.open(mask_files[period_key]["residue"]) as s:
        residue = s.read(1)

    with rasterio.open(mask_files[period_key]["crop"]) as s:
        crop = s.read(1)

    labels = np.zeros_like(soil, dtype=np.uint8)
    labels[soil == 1] = 1
    labels[residue == 1] = 2
    labels[crop == 1] = 3
    return labels, prof


def stack_indices(date_list, ndvi_dir, ndti_dir, ndri_dir):
    stacks = []
    for d in date_list:
        with rasterio.open(Path(ndvi_dir) / f"NDVI_{d}.tif") as s:
            ndvi = s.read(1)
        with rasterio.open(Path(ndti_dir) / f"NDTI_{d}.tif") as s:
            ndti = s.read(1)
        with rasterio.open(Path(ndri_dir) / f"NDRI_{d}.tif") as s:
            ndri = s.read(1)

        arr3d = np.dstack([ndvi, ndti, ndri])
        stacks.append(arr3d)

    return stacks


def extract_xy(arr3d, labels2d):
    rows, cols, bands = arr3d.shape
    X = arr3d.reshape(rows * cols, bands)
    y = labels2d.reshape(rows * cols)

    valid = (y > 0) & (~np.isnan(X).any(axis=1))
    return X[valid], y[valid]


def majority(maps):
    stacked = np.stack(maps, axis=0)
    return mode(stacked, axis=0, keepdims=False).mode


def predict_chunks(arr3d, model, chunk_size=500_000):
    rows, cols, bands = arr3d.shape
    flat = np.nan_to_num(arr3d.reshape(rows * cols, bands), nan=0.0)
    out = np.empty(flat.shape[0], dtype=np.uint8)

    for i in range(0, flat.shape[0], chunk_size):
        out[i:i + chunk_size] = model.predict(flat[i:i + chunk_size])

    return out.reshape(rows, cols)


def save_tif(arr2d, template_profile, out_fp):
    profile = template_profile.copy()
    profile.update(count=1, dtype=arr2d.dtype)

    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(arr2d, 1)


def train_period_classifier(period_key, mask_files, dates, ndvi_dir, ndti_dir, ndri_dir):
    labels, prof = load_masks(period_key, mask_files)
    stacks = stack_indices(dates[period_key], ndvi_dir, ndti_dir, ndri_dir)

    X_list, y_list = [], []
    for arr3d in stacks:
        X, y = extract_xy(arr3d, labels)
        X_list.append(X)
        y_list.append(y)

    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, prof, stacks


def classify_period(period_key, model, stacks, prof, out_dir):
    pred_maps = [predict_chunks(arr3d, model) for arr3d in stacks]
    fused = majority(pred_maps)
    out_fp = Path(out_dir) / f"{period_key}_classified.tif"
    save_tif(fused.astype("uint8"), prof, out_fp)
    return out_fp