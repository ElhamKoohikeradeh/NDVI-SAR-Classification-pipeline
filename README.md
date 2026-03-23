# NDVI-SAR-Classification-pipeline

This repository organizes a remote sensing workflow into a clean `src/`-based pipeline for:

- loading and preparing ROI shapefiles
- initializing and using Google Earth Engine
- extracting Sentinel-2 NDVI products
- exporting daily and aggregated NDVI summaries
- generating NDVI TIFF outputs
- running raster-based classification using NDVI, NDTI, and NDRI
- analyzing SAR descriptor tables
- generating time-series and profile plots

The project was refactored from several script-style files into a more modular structure.

---

## Project structure

```text
project/
├─ src/
│  ├─ config.py
│  ├─ geometry.py
│  ├─ gee_utils.py
│  ├─ extract.py
│  ├─ classification.py
│  ├─ analysis.py
│  ├─ plot.py
│  └─ pipeline.py
├─ scripts/
│  └─ run_pipeline.py
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ outputs/
└─ README.md
