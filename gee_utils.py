import io
import json
import os
import zipfile
from pathlib import Path

import requests

from src.config import (
    EE_PROJECT,
    CLOUD_FILTER,
    CLD_PRB_THRESH,
    NIR_DRK_THRESH,
    CLD_PRJ_DIST,
    BUFFER,
    NDWI_THRESH,
)

try:
    import ee
except ImportError:
    ee = None


def init_ee():
    if ee is None:
        raise ImportError("earthengine-api is not installed.")
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)
    return ee


def to_ee_geometry(geojson_obj):
    geom_type = geojson_obj["type"]
    coords = geojson_obj["coordinates"]

    if geom_type == "MultiPolygon":
        return ee.Geometry.MultiPolygon(coords)
    if geom_type == "Polygon":
        return ee.Geometry.Polygon(coords)
    return ee.Geometry(geojson_obj)


def get_s2_sr_col(aoi, start_date, end_date, cloud_pct=CLOUD_FILTER):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    )


def get_s2_sr_cld_col(aoi, start_date, end_date):
    s2_sr = get_s2_sr_col(aoi, start_date, end_date)
    s2_cloud = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )
    join_cond = ee.Filter.equals(leftField="system:index", rightField="system:index")
    joined = ee.Join.saveFirst("s2cloudless").apply(
        primary=s2_sr,
        secondary=s2_cloud,
        condition=join_cond,
    )
    return ee.ImageCollection(joined)


def add_cld_shdw_mask(img):
    prb = ee.Image(img.get("s2cloudless")).select("probability")
    cloud = prb.gt(CLD_PRB_THRESH).rename("clouds")
    not_water = img.select("SCL").neq(6)

    dark = (
        img.select("B8")
        .lt(NIR_DRK_THRESH * 1e4)
        .multiply(not_water)
        .rename("dark_pixels")
    )

    az = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
    proj = (
        cloud.directionalDistanceTransform(az, CLD_PRJ_DIST * 10)
        .reproject(crs=img.select(0).projection(), scale=100)
        .select("distance")
        .mask()
        .rename("cloud_transform")
    )

    shadows = proj.multiply(dark).rename("shadows")
    mask = (
        cloud.add(shadows)
        .gt(0)
        .focalMin(2)
        .focalMax(BUFFER * 2 / 20)
        .reproject(crs=img.select(0).projection(), scale=20)
        .rename("cloudmask")
    )

    return img.addBands([prb, cloud, dark, proj, shadows, mask])


def mask_clouds(img):
    return img.updateMask(img.select("cloudmask").Not())


def mask_clouds_simple(img):
    return img.updateMask(img.select("MSK_CLDPRB").lte(CLOUD_FILTER))


def mask_snow(img):
    ndsi = img.normalizedDifference(["B3", "B11"]).rename("ndsi")
    snow = ndsi.gt(0).And(img.select("B3").gt(1000))
    return img.updateMask(snow.Not())


def add_ndwi(img):
    return img.addBands(img.normalizedDifference(["B3", "B8"]).rename("ndwi"))


def mask_fully_moist(img):
    return img.updateMask(img.select("ndwi").lt(NDWI_THRESH))


def add_ndvi(img):
    return img.addBands(img.normalizedDifference(["B8", "B4"]).rename("ndvi"))


def add_ndti(img):
    return img.addBands(img.normalizedDifference(["B11", "B12"]).rename("ndti"))


def add_ndri(img):
    return img.addBands(img.normalizedDifference(["B11", "B4"]).rename("ndri"))


def mosaic_by_date(col):
    times = col.aggregate_array("system:time_start")
    uniq = ee.List(times).distinct()

    def per_day(t):
        d = ee.Date(t)
        return col.filterDate(d, d.advance(1, "day")).mosaic().set("system:time_start", d.millis())

    return ee.ImageCollection(uniq.map(per_day))


def download_and_unzip(img, region_geojson, out_path, scale=10, crs="EPSG:4326"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "name": out_path.stem,
        "scale": scale,
        "crs": crs,
        "region": region_geojson,
        "fileFormat": "GeoTIFF",
    }
    url = img.getDownloadURL(params)
    r = requests.get(url)
    r.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(r.content))
    tif_name = next(n for n in z.namelist() if n.lower().endswith(".tif"))
    with open(out_path, "wb") as f:
        f.write(z.read(tif_name))