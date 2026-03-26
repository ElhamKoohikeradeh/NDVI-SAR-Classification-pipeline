from pathlib import Path

import pandas as pd

from src.config import YEARS, MONTH_RANGES, NDVI_EXPORT_DIR
from src.gee_utils import (
    get_s2_sr_cld_col,
    add_cld_shdw_mask,
    mask_clouds,
    mask_snow,
    add_ndvi,
    mosaic_by_date,
    download_and_unzip,
    ee,
)


def build_ndvi_daily_collection(geom, start_date, end_date):
    col = (
        get_s2_sr_cld_col(geom, start_date, end_date)
        .map(add_cld_shdw_mask)
        .map(mask_clouds)
        .map(mask_snow)
        .map(add_ndvi)
    )
    return mosaic_by_date(col).select("ndvi")


def export_daily_ndvi_csv(roi_name, geom, year, month_name, start_mmdd, end_mmdd):
    daily = build_ndvi_daily_collection(geom, f"{year}-{start_mmdd}", f"{year}-{end_mmdd}")
    n = daily.size().getInfo()
    images = daily.toList(n)

    summary = []
    for i in range(n):
        img = ee.Image(images.get(i))
        date_str = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()

        mean_ndvi = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10,
            bestEffort=True
        ).get("ndvi").getInfo()

        summary.append(
            {
                "ROI": roi_name,
                "year": year,
                "month": month_name,
                "date": date_str,
                "ndvi_mean": mean_ndvi,
            }
        )

    return pd.DataFrame(summary)


def run_ndvi_summary_export(roi_ee: dict):
    frames = []
    for name, geom in roi_ee.items():
        for year in YEARS:
            for month_name, (sd, ed) in MONTH_RANGES.items():
                frames.append(export_daily_ndvi_csv(name, geom, year, month_name, sd, ed))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_fp = NDVI_EXPORT_DIR / "DailyNDVIValues.csv"
    out.to_csv(out_fp, index=False)
    return out_fp


def run_ndvi_tiff_export(roi_ee: dict):
    """
    Median NDVI TIFF export per ROI/year/month.
    """
    for name, geom in roi_ee.items():
        region_geojson = geom.toGeoJSONString()
        roi_folder = NDVI_EXPORT_DIR / name
        roi_folder.mkdir(parents=True, exist_ok=True)

        for year in YEARS:
            for mon, (sd, ed) in MONTH_RANGES.items():
                daily = build_ndvi_daily_collection(geom, f"{year}-{sd}", f"{year}-{ed}")
                if daily.size().getInfo() == 0:
                    continue

                img = daily.median().clip(geom)
                out_fp = roi_folder / f"NDVI_{name}_{year}_{mon}.tif"
                download_and_unzip(img, region_geojson, out_fp)


def add_sar_indicators(image):
    vv = image.select("VV").multiply(0.01)
    vh = image.select("VH").multiply(0.01)

    alpha = vv.atan2(vh).rename("Alpha")
    entropy = vv.subtract(vh).abs().divide(vv.add(vh)).rename("Entropy")
    anisotropy = vv.subtract(vh).divide(vv.add(vh)).pow(2).rename("Anisotropy")
    dprvi = vh.pow(2).multiply(4).divide(vv.pow(2).add(vh.pow(2))).rename("DpRVI")
    vsi = vh.divide(vv).rename("VSI")

    return image.addBands([alpha, entropy, anisotropy, dprvi, vsi])