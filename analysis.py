import pandas as pd
import numpy as np


def restore_full_id(x):
    try:
        return str(int(float(x)))
    except Exception:
        return ""


def flatten_feature(group):
    flat = {
        "ROI": group["ROI"].iloc[0],
        "id": group["id"].iloc[0],
    }
    for i, row in enumerate(group.itertuples(), 1):
        flat[f"NDVI_{i}"] = row.ndvi
        flat[f"Date_{i}"] = row.date.strftime("%Y-%m-%d")
    return pd.Series(flat)


def group_ndvi_per_feature(csv_path, out_path):
    df = pd.read_csv(csv_path, dtype={"id": str})
    df["date"] = pd.to_datetime(df["date"])
    df["id"] = df["id"].apply(restore_full_id)

    df_clean = df[["ROI", "id", "date", "ndvi"]].dropna(subset=["id", "ndvi"])
    df_clean = df_clean.sort_values(["ROI", "id", "date"])

    grouped_df = (
        df_clean.groupby(["ROI", "id"], sort=False)
        .apply(flatten_feature)
        .reset_index(drop=True)
    )

    grouped_df.to_csv(out_path, index=False)
    return out_path


def build_q_profile(df, n_bins=100, roll_window=5):
    q_min, q_max = df["q"].min(), df["q"].max()
    df = df.copy()
    df["q_norm"] = (df["q"] - q_min) / (q_max - q_min)

    bins = np.linspace(0, 1, n_bins + 1)
    df["q_bin"] = pd.cut(df["q_norm"], bins=bins, include_lowest=True)

    prof = (
        df.groupby("q_bin")[["Theta_c", "mc", "Hc"]]
        .median()
        .reset_index()
    )
    prof["q_center"] = prof["q_bin"].apply(lambda iv: 0.5 * (iv.left + iv.right))

    if roll_window > 1:
        for col in ["Theta_c", "mc", "Hc"]:
            prof[col] = prof[col].rolling(
                roll_window, min_periods=1, center=True
            ).median()

    prof = prof.dropna(subset=["Theta_c", "mc", "Hc"], how="all").reset_index(drop=True)
    return prof