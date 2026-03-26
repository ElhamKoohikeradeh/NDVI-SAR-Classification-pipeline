from pathlib import Path

import pandas as pd

from src.config import (
    BASE_SHAPE_DIR,
    YEARS,
    MONTH_RANGES,
    NDVI_EXPORT_DIR,
    PLOTS_DIR,
    MASK_FILES,
    DATES,
    NDVI_DIR,
    NDTI_DIR,
    NDRI_DIR,
    CLASSIFIED_DIR,
)
from src.geometry import build_roi_dict
from src.gee_utils import init_ee, to_ee_geometry
from src.extract import run_ndvi_summary_export, run_ndvi_tiff_export
from src.classification import train_period_classifier, classify_period
from src.analysis import group_ndvi_per_feature, build_q_profile
from src.plot import plot_daily_ndvi, plot_q_profile, plot_sar_timeseries


def run_pipeline(
    do_ndvi_extraction=True,
    do_ndvi_tiffs=False,
    do_classification=False,
    do_analysis=False,
    do_plots=True,
):
    # 1) Init EE
    init_ee()

    # 2) Load ROIs
    roi_shapes, roi_geojson = build_roi_dict(BASE_SHAPE_DIR)
    roi_ee = {name: to_ee_geometry(gj) for name, gj in roi_geojson.items()}

    # 3) Extraction
    ndvi_csv = None
    if do_ndvi_extraction:
        ndvi_csv = run_ndvi_summary_export(roi_ee)

    if do_ndvi_tiffs:
        run_ndvi_tiff_export(roi_ee)

    # 4) Classification
    if do_classification:
        for period_key in MASK_FILES.keys():
            model, prof, stacks = train_period_classifier(
                period_key=period_key,
                mask_files=MASK_FILES,
                dates=DATES,
                ndvi_dir=NDVI_DIR,
                ndti_dir=NDTI_DIR,
                ndri_dir=NDRI_DIR,
            )
            classify_period(period_key, model, stacks, prof, CLASSIFIED_DIR)

    # 5) Analysis examples
    if do_analysis:
        # Example 1: group NDVI per feature
        per_feature_csv = NDVI_EXPORT_DIR / "NDVI_per_feature_all.csv"
        if per_feature_csv.exists():
            group_ndvi_per_feature(
                per_feature_csv,
                NDVI_EXPORT_DIR / "NDVI_grouped.csv",
            )

        # Example 2: q-profile analysis
        q_fp = NDVI_EXPORT_DIR / "dual_pol_descriptors_q.xlsx"
        if q_fp.exists():
            q_df = pd.read_excel(q_fp)
            prof = build_q_profile(q_df)
            plot_q_profile(prof, PLOTS_DIR / "q_profile.png")

        # Example 3: SAR timeseries
        sar_fp = NDVI_EXPORT_DIR / "SARPlot.csv"
        if sar_fp.exists():
            sar_df = pd.read_csv(sar_fp)
            plot_sar_timeseries(sar_df, PLOTS_DIR / "sar_timeseries")

    # 6) Plots
    if do_plots and ndvi_csv is not None:
        plot_daily_ndvi(ndvi_csv, PLOTS_DIR / "daily_ndvi")


if __name__ == "__main__":
    run_pipeline()