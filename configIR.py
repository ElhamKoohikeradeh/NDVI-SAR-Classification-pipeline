from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, OUTPUTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Earth Engine / ROI config ----------
EE_PROJECT = "elhamkoohi2024"   # change if needed

BASE_SHAPE_DIR = RAW_DIR / "roi_shapefiles"

YEARS = list(range(2019, 2026))
MONTH_RANGES = {
    "May": ("05-15", "05-31"),
    "June": ("06-01", "06-30"),
    "July": ("07-01", "07-31"),
}

CLOUD_FILTER = 20
CLD_PRB_THRESH = 50
NIR_DRK_THRESH = 0.15
CLD_PRJ_DIST = 1
BUFFER = 50
NDWI_THRESH = 0.0

# ---------- Classification config ----------
CLASSIFICATION_DIR = RAW_DIR / "classification"
MASK_DIR = CLASSIFICATION_DIR / "masks"
NDVI_DIR = CLASSIFICATION_DIR / "ndvi"
NDTI_DIR = CLASSIFICATION_DIR / "ndti"
NDRI_DIR = CLASSIFICATION_DIR / "ndri"

CLASSIFIED_DIR = OUTPUTS_DIR / "classified_maps"
CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

MASK_FILES = {
    # example structure — replace with your real names
    # "Period1": {"soil": "...tif", "residue": "...tif", "crop": "...tif"}
}

DATES = {
    # example structure — replace with your real dates
    # "Period1": ["2024-05-20", "2024-05-30", "2024-06-10"]
}

# ---------- Outputs ----------
NDVI_EXPORT_DIR = OUTPUTS_DIR / "ndvi_exports"
NDVI_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)