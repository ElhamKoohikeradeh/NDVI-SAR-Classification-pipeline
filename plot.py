from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_daily_ndvi(csv_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[df["date"].dt.strftime("%m-%d").between("05-15", "07-31")]
    df = df.sort_values(by=["ROI", "year", "date"])

    for (roi, year), group in df.groupby(["ROI", "year"]):
        fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=300)
        ax.plot(group["date"], group["ndvi_mean"], marker="o", linestyle="-")
        ax.set_title(f"Daily NDVI for {roi} — {year}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean NDVI")
        ax.grid(True)
        ax.set_xticks(group["date"])
        ax.set_xticklabels([d.strftime("%b %d") for d in group["date"]], rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(output_dir / f"{roi}_{year}_DailyNDVI.png")
        plt.close(fig)


def plot_q_profile(prof_df, out_fp):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(prof_df["q_center"], prof_df["Theta_c"], marker="o", label="Theta_c")
    ax2.plot(prof_df["q_center"], prof_df["Hc"], marker="o", label="Hc")
    ax2.plot(prof_df["q_center"], prof_df["mc"], marker="o", label="mc")

    for y in [0.30, 0.60, 0.85]:
        ax2.axhline(y, linestyle="--", alpha=0.5)

    ax1.set_xlabel("Normalized q")
    ax1.set_ylabel("Theta_c")
    ax2.set_ylabel("Hc / mc")
    plt.tight_layout()
    fig.savefig(out_fp, dpi=300)
    plt.close(fig)


def plot_sar_timeseries(df, out_dir, group_col="shapefile_name"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    sar_params = ["DpRVI", "Hc", "Theta_c", "VH", "VV", "angle", "mc", "ratio"]

    for shapefile, group in df.groupby(group_col):
        group = group.sort_values("date")
        fig, ax = plt.subplots(figsize=(14, 6))

        for param in sar_params:
            if param in group.columns:
                ax.plot(group["date"], group[param], marker="o", label=param)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        ax.set_title(f"SAR Parameters Time Series - {shapefile}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        fig.savefig(out_dir / f"{shapefile}_SAR_timeseries.png", dpi=300)
        plt.close(fig)