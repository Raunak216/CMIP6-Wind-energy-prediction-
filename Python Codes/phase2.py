from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import traceback


# Configuration

BASE = Path(r"D:\CMIP6")

PHASE2_DIR = BASE / "PHASE2"
SUMMARY_DIR = PHASE2_DIR / "summaries"

SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "CNRM-CM6-1",
    "GFDL-CM4",
    "GFDL-ESM4",
]

EXPERIMENTS = [
    "historical",
    "ssp245",
    "ssp585",
]

PERIODS = {
    "historical": (1950, 2014),
    "ssp245": (2015, 2100),
    "ssp585": (2015, 2100),
}

REFERENCE_HEIGHT = 10.0

HUB_HEIGHTS = [
    80.0,
    100.0,
    120.0,
]

# Conventional 1/7 power-law exponent
ALPHA = 1.0 / 7.0


# Print header

print()
print("=" * 90)
print("PHASE 2 — HUB-HEIGHT WIND SPEED EXTRAPOLATION")
print("=" * 90)
print()
print(f"Reference height : {REFERENCE_HEIGHT} m")
print(f"Hub heights      : {HUB_HEIGHTS}")
print(f"Alpha            : {ALPHA:.6f} (1/7)")
print(f"Datasets         : {len(MODELS)} × {len(EXPERIMENTS)} = "
      f"{len(MODELS) * len(EXPERIMENTS)}")
print()


# File finder

def find_phase1_file(model, experiment):
    """
    Locate the Phase 1 daily wind-speed NetCDF.

    Expected structure:

    D:\\CMIP6\\
        MODEL\\
            day\\
                processed\\
                    MODEL_experiment_daily_wind_speed.nc
    """

    model_dir = BASE / model

    # Handle both "day" and "Day"
    possible_dirs = [
        model_dir / "day" / "processed",
        model_dir / "Day" / "processed",
    ]

    expected_name = f"{model}_{experiment}_daily_wind_speed.nc"

    for processed_dir in possible_dirs:

        exact_file = processed_dir / expected_name

        if exact_file.exists():
            return exact_file

    # Fallback: recursive search inside model folder
    candidates = list(
        model_dir.rglob(f"{model}_{experiment}_daily_wind_speed.nc")
    )

    if candidates:
        return candidates[0]

    return None


# Identify wind variable

def find_wind_variable(ds):

    preferred_names = [
        "wind_speed",
        "windspeed",
        "ws",
        "wind_speed_10m",
        "sfcWind",
    ]

    for name in preferred_names:
        if name in ds.data_vars:
            return name

    # Fallback: look for a variable containing "wind"
    wind_candidates = [
        name for name in ds.data_vars
        if "wind" in name.lower()
    ]

    if len(wind_candidates) == 1:
        return wind_candidates[0]

    # If dataset has exactly one variable
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise ValueError(
        f"Could not uniquely identify wind-speed variable. "
        f"Variables found: {list(ds.data_vars)}"
    )


# Time decoder

time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)


# Process one dataset

def process_dataset(model, experiment):

    input_file = find_phase1_file(model, experiment)

    if input_file is None:
        raise FileNotFoundError(
            f"Phase 1 file not found: {model} | {experiment}"
        )

    print()
    print("-" * 90)
    print(f"Processing: {model} | {experiment}")
    print(f"Input     : {input_file}")

    start_year, end_year = PERIODS[experiment]

    ds = xr.open_dataset(
        input_file,
        decode_times=time_coder
    )

    try:

        wind_var = find_wind_variable(ds)

        print(f"Wind variable : {wind_var}")

        ws = ds[wind_var]

        # Check time coverage

        actual_start = int(ws["time"].dt.year.min().item())
        actual_end = int(ws["time"].dt.year.max().item())

        print(
            f"Available time: {actual_start} → {actual_end}"
        )

        # Restrict to requested period

        ws = ws.sel(
            time=slice(
                f"{start_year}-01-01",
                f"{end_year}-12-31"
            )
        )

        if ws.sizes.get("time", 0) == 0:
            raise ValueError(
                f"No data available for requested period "
                f"{start_year}-{end_year}"
            )

        print(
            f"Processing time: "
            f"{start_year} → {end_year}"
        )

        # Basic validation

        sample = ws.isel(time=0)

        if not np.isfinite(sample.values).any():
            raise ValueError(
                "First wind-speed timestep contains no finite values."
            )

        # Spatial dimensions

        spatial_dims = [
            d for d in ws.dims
            if d not in ["time"]
        ]

        print(f"Spatial dimensions: {spatial_dims}")

        # Create result containers

        annual_records = []
        monthly_records = []

        summary_records = []

        # Hub height calculation

        for hub_height in HUB_HEIGHTS:

            factor = (
                hub_height / REFERENCE_HEIGHT
            ) ** ALPHA

            print(
                f"  Hub height {hub_height:.0f} m "
                f"| multiplier = {factor:.6f}"
            )

            hub_ws = ws * factor

            # Annual mean

            annual = (
                hub_ws
                .groupby("time.year")
                .mean(
                    dim=["time"] + spatial_dims,
                    skipna=True
                )
            )

            for year in annual["year"].values:

                annual_records.append({
                    "model": model,
                    "experiment": experiment,
                    "year": int(year),
                    "hub_height_m": hub_height,
                    "alpha": ALPHA,
                    "mean_wind_speed_ms": float(
                        annual.sel(year=year).values
                    ),
                })

            # Monthly climatology

            monthly = (
                hub_ws
                .groupby("time.month")
                .mean(
                    dim=["time"] + spatial_dims,
                    skipna=True
                )
            )

            for month in monthly["month"].values:

                monthly_records.append({
                    "model": model,
                    "experiment": experiment,
                    "month": int(month),
                    "hub_height_m": hub_height,
                    "alpha": ALPHA,
                    "mean_wind_speed_ms": float(
                        monthly.sel(month=month).values
                    ),
                })

            # Overall summary

            overall_mean = float(
                hub_ws.mean(
                    dim=["time"] + spatial_dims,
                    skipna=True
                ).values
            )

            overall_std = float(
                hub_ws.std(
                    dim=["time"] + spatial_dims,
                    skipna=True
                ).values
            )

            overall_min = float(
                hub_ws.min(
                    dim=["time"] + spatial_dims,
                    skipna=True
                ).values
            )

            overall_max = float(
                hub_ws.max(
                    dim=["time"] + spatial_dims,
                    skipna=True
                ).values
            )

            summary_records.append({
                "model": model,
                "experiment": experiment,
                "hub_height_m": hub_height,
                "alpha": ALPHA,
                "start_year": start_year,
                "end_year": end_year,
                "mean_wind_speed_ms": overall_mean,
                "std_wind_speed_ms": overall_std,
                "min_wind_speed_ms": overall_min,
                "max_wind_speed_ms": overall_max,
            })

        # Save per-dataset summaries

        annual_df = pd.DataFrame(annual_records)
        monthly_df = pd.DataFrame(monthly_records)
        summary_df = pd.DataFrame(summary_records)

        annual_file = (
            SUMMARY_DIR /
            f"PHASE2_{model}_{experiment}_annual.csv"
        )

        monthly_file = (
            SUMMARY_DIR /
            f"PHASE2_{model}_{experiment}_monthly.csv"
        )

        summary_file = (
            SUMMARY_DIR /
            f"PHASE2_{model}_{experiment}_summary.csv"
        )

        annual_df.to_csv(
            annual_file,
            index=False
        )

        monthly_df.to_csv(
            monthly_file,
            index=False
        )

        summary_df.to_csv(
            summary_file,
            index=False
        )

        print(
            f"  ✓ Annual summary : {annual_file.name}"
        )

        print(
            f"  ✓ Monthly summary: {monthly_file.name}"
        )

        print(
            f"  ✓ Summary        : {summary_file.name}"
        )

        print("  ✓ SUCCESS")

        return {
            "model": model,
            "experiment": experiment,
            "status": "SUCCESS",
            "input_file": str(input_file),
            "annual_rows": len(annual_df),
            "monthly_rows": len(monthly_df),
        }

    finally:

        ds.close()


# Main processing

results = []

for model in MODELS:

    for experiment in EXPERIMENTS:

        try:

            result = process_dataset(
                model,
                experiment
            )

            results.append(result)

        except Exception as e:

            print()
            print(
                f"❌ FAILED: {model} | {experiment}"
            )

            print(
                f"   Error: {e}"
            )

            traceback.print_exc()

            results.append({
                "model": model,
                "experiment": experiment,
                "status": "FAILED",
                "input_file": "",
                "annual_rows": 0,
                "monthly_rows": 0,
            })


# Combine master files

print()
print("=" * 90)
print("COMBINING MASTER PHASE 2 SUMMARIES")
print("=" * 90)
print()


annual_files = sorted(
    SUMMARY_DIR.glob(
        "PHASE2_*_*_annual.csv"
    )
)

monthly_files = sorted(
    SUMMARY_DIR.glob(
        "PHASE2_*_*_monthly.csv"
    )
)

summary_files = sorted(
    SUMMARY_DIR.glob(
        "PHASE2_*_*_summary.csv"
    )
)

# Remove any potential master files from matching
annual_files = [
    f for f in annual_files
    if "FINAL" not in f.name
]

monthly_files = [
    f for f in monthly_files
    if "FINAL" not in f.name
]

summary_files = [
    f for f in summary_files
    if "FINAL" not in f.name
]


# Annual

if annual_files:

    annual_master = pd.concat(
        [
            pd.read_csv(f)
            for f in annual_files
        ],
        ignore_index=True
    )

    annual_master_file = (
        SUMMARY_DIR /
        "PHASE2_FINAL_annual_hub_height_wind.csv"
    )

    annual_master.to_csv(
        annual_master_file,
        index=False
    )

    print(
        f"✓ Annual master: {annual_master_file}"
    )


# Monthly

if monthly_files:

    monthly_master = pd.concat(
        [
            pd.read_csv(f)
            for f in monthly_files
        ],
        ignore_index=True
    )

    monthly_master_file = (
        SUMMARY_DIR /
        "PHASE2_FINAL_monthly_hub_height_wind.csv"
    )

    monthly_master.to_csv(
        monthly_master_file,
        index=False
    )

    print(
        f"✓ Monthly master: {monthly_master_file}"
    )


# Summary

if summary_files:

    summary_master = pd.concat(
        [
            pd.read_csv(f)
            for f in summary_files
        ],
        ignore_index=True
    )

    summary_master_file = (
        SUMMARY_DIR /
        "PHASE2_FINAL_hub_height_summary.csv"
    )

    summary_master.to_csv(
        summary_master_file,
        index=False
    )

    print(
        f"✓ Summary master: {summary_master_file}"
    )


# Validation

print()
print("=" * 90)
print("PHASE 2 VALIDATION")
print("=" * 90)

expected = len(MODELS) * len(EXPERIMENTS)

successful = sum(
    1 for r in results
    if r["status"] == "SUCCESS"
)

failed = expected - successful

print(f"Expected datasets : {expected}")
print(f"Successful         : {successful}")
print(f"Failed             : {failed}")

if failed == 0:

    print()
    print("🎉 PHASE 2 COMPLETE — ALL 21 DATASETS PROCESSED")

    # Validate annual row count

    final_annual = (
        SUMMARY_DIR /
        "PHASE2_FINAL_annual_hub_height_wind.csv"
    )

    if final_annual.exists():

        df = pd.read_csv(final_annual)

        expected_annual_rows = (
            7 * (
                65 + 86 + 86
            ) * 3
        )

        print()
        print(
            f"Annual rows      : {len(df)}"
        )

        print(
            f"Expected annual  : {expected_annual_rows}"
        )

        if len(df) == expected_annual_rows:
            print(
                "✓ Annual row-count validation PASSED"
            )
        else:
            print(
                "⚠ Annual row-count differs from expected"
            )

    # Validate monthly row count

    final_monthly = (
        SUMMARY_DIR /
        "PHASE2_FINAL_monthly_hub_height_wind.csv"
    )

    if final_monthly.exists():

        df = pd.read_csv(final_monthly)

        expected_monthly_rows = (
            7 * 3 * 3 * 12
        )

        print(
            f"Monthly rows     : {len(df)}"
        )

        print(
            f"Expected monthly : {expected_monthly_rows}"
        )

        if len(df) == expected_monthly_rows:
            print(
                "✓ Monthly row-count validation PASSED"
            )
        else:
            print(
                "⚠ Monthly row-count differs from expected"
            )

else:

    print()
    print(
        "⚠ PHASE 2 INCOMPLETE — CHECK FAILED DATASETS"
    )


print()
print(
    "Daily hub-height NetCDF files were NOT created."
)

print(
    "Hub-height wind speed will be calculated on-the-fly "
    "from Phase 1 during Phase 3 to conserve disk space."
)

print()
print("DONE.")