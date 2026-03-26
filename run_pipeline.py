from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(
        do_ndvi_extraction=True,
        do_ndvi_tiffs=False,
        do_classification=False,
        do_analysis=False,
        do_plots=True,
    )