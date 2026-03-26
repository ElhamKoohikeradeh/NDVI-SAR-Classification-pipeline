import os
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union


def load_single_roi(shp_path: Path):
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"Empty shapefile: {shp_path}")

    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.buffer(0)
    merged = unary_union(gdf.geometry)
    return merged


def build_roi_dict(base_shape_dir: Path):
    """
    Returns:
        roi_shapes: dict[str, shapely geometry]
        roi_geojson: dict[str, dict]
    """
    roi_shapes = {}
    roi_geojson = {}

    for name in os.listdir(base_shape_dir):
        sub = Path(base_shape_dir) / name
        if not sub.is_dir():
            continue

        shp_files = list(sub.glob("*.shp"))
        if not shp_files:
            continue

        merged = load_single_roi(shp_files[0])
        roi_shapes[name] = merged
        roi_geojson[name] = mapping(merged)

    return roi_shapes, roi_geojson


def load_feature_collection_shapefiles(shapefile_dir: Path):
    """
    For per-feature plotting / SAR joins.
    Creates shapefile_name from base name + FID/index.
    """
    import pandas as pd

    gdf_list = []
    for shp in shapefile_dir.rglob("*.shp"):
        try:
            gdf = gpd.read_file(shp)
            if gdf.empty:
                continue

            gdf = gdf.to_crs(epsg=4326)
            if "FID" in gdf.columns:
                id_series = gdf["FID"].astype(str)
            else:
                id_series = gdf.index.astype(str)

            base = shp.stem
            gdf["shapefile_name"] = base + id_series
            gdf_list.append(gdf)
        except Exception:
            continue

    if not gdf_list:
        raise ValueError("No valid shapefiles loaded.")

    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs="EPSG:4326")
    return merged_gdf