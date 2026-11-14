# CMIP6 daily wind speed processing across Historical, SSP245, and SSP585 for 7 models.

import os
import glob
import re
import gc
import shutil
import warnings

import numpy as np
import pandas as pd
import xarray as xr


# Configuration

BASE_DIR = r"D:\CMIP6"

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "CNRM-CM6-1",
    "GFDL-CM4",
    "GFDL-ESM4",
]

EXPERIMENTS = {
    "historical": {
        "start": "1950-01-01",
        "end": "2014-12-31",
        "start_year": 1950,
        "end_year": 2014,
    },

    "ssp245": {
        "start": "2015-01-01",
        "end": "2100-12-31",
        "start_year": 2015,
        "end_year": 2100,
    },

    "ssp585": {
        "start": "2015-01-01",
        "end": "2100-12-31",
        "start_year": 2015,
        "end_year": 2100,
    },
}


# Storage settings

PHASE1_SUMMARY_DIR = os.path.join(
    BASE_DIR,
    "PHASE1",
    "summaries"
)

MAX_PHASE1_GB = 45.0

COMPRESSION_LEVEL = 4

TIME_CHUNK = 365


# IMPORTANT: New xarray API for CF calendars

TIME_CODER = xr.coders.CFDatetimeCoder(
    use_cftime=True
)


# Basic utilities

def get_free_space_gb(path):

    total, used, free = shutil.disk_usage(path)

    return free / (1024 ** 3)


def get_folder_size_gb(folder):

    if not os.path.exists(folder):
        return 0.0

    total_bytes = 0

    for root, dirs, files in os.walk(folder):

        for filename in files:

            path = os.path.join(
                root,
                filename
            )

            try:
                total_bytes += os.path.getsize(
                    path
                )

            except OSError:
                pass

    return total_bytes / (1024 ** 3)


def find_nc_files(folder):

    if not os.path.isdir(folder):
        return []

    files = []

    for pattern in [
        "*.nc",
        "*.nc4",
        "*.cdf"
    ]:

        files.extend(
            glob.glob(
                os.path.join(
                    folder,
                    pattern
                )
            )
        )

    return sorted(
        set(files)
    )


# Experiment-specific file selection

def select_experiment_files(
    folder,
    experiment
):
    """
    Select ONLY files belonging to the requested
    experiment.

    This is the most important correction compared
    with the previous production script.
    """

    all_files = find_nc_files(
        folder
    )

    selected = []

    for path in all_files:

        filename = os.path.basename(
            path
        ).lower()

        # Historical

        if experiment == "historical":

            if (
                "_historical_" in filename
                or "_historical." in filename
                or "_historical-" in filename
            ):
                selected.append(path)

        # Ssp245

        elif experiment == "ssp245":

            if (
                "_ssp245_" in filename
                or "_ssp245." in filename
                or "_ssp245-" in filename
            ):
                selected.append(path)

        # Ssp585

        elif experiment == "ssp585":

            if (
                "_ssp585_" in filename
                or "_ssp585." in filename
                or "_ssp585-" in filename
            ):
                selected.append(path)

    return sorted(
        set(selected)
    )


# Print selected files

def print_selected_files(
    model,
    variable,
    experiment,
    files
):

    print("\n")
    print("-" * 75)

    print(
        f"{model} | "
        f"{experiment} | "
        f"{variable.upper()}"
    )

    print(
        f"Selected files: {len(files)}"
    )

    for path in files:

        print(
            "  "
            + os.path.basename(path)
        )


# Open component

def open_component(
    files,
    variable
):

    if len(files) == 0:

        raise FileNotFoundError(
            f"No files selected for "
            f"{variable}."
        )

    ds = xr.open_mfdataset(
        files,

        combine="by_coords",

        parallel=False,

        chunks={
            "time": TIME_CHUNK
        },

        # NEW xarray API

        decode_times=TIME_CODER,

        data_vars="minimal",

        coords="minimal",

        compat="override",

        combine_attrs="override"
    )

    if variable not in ds.data_vars:

        raise KeyError(
            f"Variable '{variable}' "
            f"not found in selected files.\n"
            f"Available variables: "
            f"{list(ds.data_vars)}"
        )

    return ds


# Time validation

def validate_time(
    da,
    experiment,
    config
):

    if "time" not in da.coords:

        raise ValueError(
            f"{experiment}: no time coordinate."
        )

    start = da.time.isel(
        time=0
    ).values

    end = da.time.isel(
        time=-1
    ).values

    print(
        f"Original time coverage: "
        f"{start} → {end}"
    )

    # Select project period

    da = da.sel(
        time=slice(
            config["start"],
            config["end"]
        )
    )

    if da.sizes["time"] == 0:

        raise ValueError(
            f"{experiment}: "
            f"no data after requested "
            f"time selection."
        )

    final_start = da.time.isel(
        time=0
    ).values

    final_end = da.time.isel(
        time=-1
    ).values

    print(
        f"Selected time coverage: "
        f"{final_start} → {final_end}"
    )

    print(
        f"Time points: "
        f"{da.sizes['time']:,}"
    )

    return da


# Grid information

def print_grid_info(
    model,
    experiment,
    uas,
    vas
):

    print("\n")
    print(
        f"GRID: {model} | {experiment}"
    )

    print(
        f"UAS: "
        f"{uas.sizes.get('lat')} × "
        f"{uas.sizes.get('lon')}"
    )

    print(
        f"VAS: "
        f"{vas.sizes.get('lat')} × "
        f"{vas.sizes.get('lon')}"
    )

    print(
        f"UAS lat: "
        f"{float(uas.lat.min()):.6f} → "
        f"{float(uas.lat.max()):.6f}"
    )

    print(
        f"VAS lat: "
        f"{float(vas.lat.min()):.6f} → "
        f"{float(vas.lat.max()):.6f}"
    )

    print(
        f"UAS lon: "
        f"{float(uas.lon.min()):.6f} → "
        f"{float(uas.lon.max()):.6f}"
    )

    print(
        f"VAS lon: "
        f"{float(vas.lon.min()):.6f} → "
        f"{float(vas.lon.max()):.6f}"
    )


# Spatial alignment

def align_spatial(
    uas,
    vas,
    model,
    experiment
):
    """
    Align UAS and VAS.

    If grids are identical:
        no interpolation.

    If grids differ:
        determine spatial overlap,
        restrict UAS to overlap,
        interpolate VAS onto the restricted UAS grid.

    NO EXTRAPOLATION.
    """

    # Check coordinates

    for da, name in [
        (uas, "UAS"),
        (vas, "VAS")
    ]:

        if "lat" not in da.coords:
            raise ValueError(
                f"{model} {experiment}: "
                f"{name} has no latitude."
            )

        if "lon" not in da.coords:
            raise ValueError(
                f"{model} {experiment}: "
                f"{name} has no longitude."
            )

    # Check if coordinates are already identical

    same_lat = (
        uas.lat.size == vas.lat.size
        and
        np.allclose(
            uas.lat.values,
            vas.lat.values,
            rtol=0,
            atol=1e-8
        )
    )

    same_lon = (
        uas.lon.size == vas.lon.size
        and
        np.allclose(
            uas.lon.values,
            vas.lon.values,
            rtol=0,
            atol=1e-8
        )
    )

    if same_lat and same_lon:

        print(
            "\nUAS/VAS grids are identical."
        )

        return uas, vas

    # Different grid

    print(
        "\nUAS/VAS grids differ."
    )

    print(
        "Using spatial-overlap alignment."
    )

    # Find overlap

    lat_min = max(
        float(uas.lat.min()),
        float(vas.lat.min())
    )

    lat_max = min(
        float(uas.lat.max()),
        float(vas.lat.max())
    )

    lon_min = max(
        float(uas.lon.min()),
        float(vas.lon.min())
    )

    lon_max = min(
        float(uas.lon.max()),
        float(vas.lon.max())
    )

    if lat_min >= lat_max:

        raise ValueError(
            f"{model} {experiment}: "
            f"no latitude overlap."
        )

    if lon_min >= lon_max:

        raise ValueError(
            f"{model} {experiment}: "
            f"no longitude overlap."
        )

    print(
        f"Overlap latitude: "
        f"{lat_min:.6f} → {lat_max:.6f}"
    )

    print(
        f"Overlap longitude: "
        f"{lon_min:.6f} → {lon_max:.6f}"
    )

    # Restrict UAS to overlapping coordinates

    target_lat = uas.lat.where(
        (
            uas.lat >= lat_min
        )
        &
        (
            uas.lat <= lat_max
        ),
        drop=True
    )

    target_lon = uas.lon.where(
        (
            uas.lon >= lon_min
        )
        &
        (
            uas.lon <= lon_max
        ),
        drop=True
    )

    uas_target = uas.sel(
        lat=target_lat,
        lon=target_lon
    )

    # Interpolate VAS to UAS target grid IMPORTANT: Target grid is already restricted to overlap, so interpolation does not...

    vas_target = vas.interp(
        lat=uas_target.lat,
        lon=uas_target.lon,
        method="linear"
    )

    print(
        f"Aligned grid: "
        f"{uas_target.sizes['lat']} × "
        f"{uas_target.sizes['lon']}"
    )

    return uas_target, vas_target


# Wind speed

def calculate_wind_speed(
    uas,
    vas
):

    print(
        "\nCalculating:"
    )

    print(
        "V = sqrt(UAS² + VAS²)"
    )

    wind_speed = np.hypot(
        uas.astype("float32"),
        vas.astype("float32")
    )

    wind_speed.name = (
        "wind_speed"
    )

    wind_speed.attrs = {
        "long_name":
            "Daily near-surface wind speed",

        "units":
            "m s-1",

        "formula":
            "sqrt(uas^2 + vas^2)",

        "source":
            "CMIP6 daily uas and vas",

        "processing":
            "Computed after time and spatial alignment",
    }

    return wind_speed


# Statistics

def calculate_statistics(
    wind_speed
):

    print(
        "\nCalculating statistics..."
    )

    mean_value = float(
        wind_speed
        .mean(skipna=True)
        .compute()
    )

    min_value = float(
        wind_speed
        .min(skipna=True)
        .compute()
    )

    max_value = float(
        wind_speed
        .max(skipna=True)
        .compute()
    )

    std_value = float(
        wind_speed
        .std(skipna=True)
        .compute()
    )

    # Latitude-weighted mean

    weights = np.cos(
        np.deg2rad(
            wind_speed.lat
        )
    )

    weighted_mean = float(
        wind_speed
        .weighted(weights)
        .mean(
            dim=(
                "time",
                "lat",
                "lon"
            )
        )
        .compute()
    )

    # Missing values

    missing = int(
        wind_speed
        .isnull()
        .sum()
        .compute()
    )

    total = wind_speed.size

    missing_percent = (
        100.0 * missing / total
    )

    print(
        f"Mean                 : "
        f"{mean_value:.6f} m/s"
    )

    print(
        f"Lat-weighted mean    : "
        f"{weighted_mean:.6f} m/s"
    )

    print(
        f"Minimum              : "
        f"{min_value:.6f} m/s"
    )

    print(
        f"Maximum              : "
        f"{max_value:.6f} m/s"
    )

    print(
        f"Standard deviation   : "
        f"{std_value:.6f} m/s"
    )

    print(
        f"Missing              : "
        f"{missing:,} / {total:,} "
        f"({missing_percent:.6f}%)"
    )

    return {
        "mean_wind_speed_mps":
            mean_value,

        "latitude_weighted_mean_mps":
            weighted_mean,

        "min_wind_speed_mps":
            min_value,

        "max_wind_speed_mps":
            max_value,

        "std_wind_speed_mps":
            std_value,

        "missing_count":
            missing,

        "missing_percent":
            missing_percent,
    }


# Annual summary

def calculate_annual_summary(
    wind_speed,
    model,
    experiment
):

    weights = np.cos(
        np.deg2rad(
            wind_speed.lat
        )
    )

    spatial_mean = (
        wind_speed
        .weighted(weights)
        .mean(
            dim=("lat", "lon")
        )
    )

    annual = (
        spatial_mean
        .groupby("time.year")
        .mean("time")
        .compute()
    )

    return pd.DataFrame({
        "model":
            model,

        "experiment":
            experiment,

        "year":
            annual.year.values,

        "mean_wind_speed_mps":
            annual.values,
    })


# Monthly climatology

def calculate_monthly_summary(
    wind_speed,
    model,
    experiment
):

    weights = np.cos(
        np.deg2rad(
            wind_speed.lat
        )
    )

    spatial_mean = (
        wind_speed
        .weighted(weights)
        .mean(
            dim=("lat", "lon")
        )
    )

    monthly = (
        spatial_mean
        .groupby("time.month")
        .mean("time")
        .compute()
    )

    return pd.DataFrame({
        "model":
            model,

        "experiment":
            experiment,

        "month":
            monthly.month.values,

        "mean_wind_speed_mps":
            monthly.values,
    })


# Save daily output

def save_daily_output(
    wind_speed,
    output_file,
    model,
    experiment
):

    print(
        "\nSaving compressed NetCDF..."
    )

    time_chunk = min(
        TIME_CHUNK,
        wind_speed.sizes["time"]
    )

    encoding = {
        "wind_speed": {
            "dtype": "float32",
            "zlib": True,
            "complevel":
                COMPRESSION_LEVEL,
            "shuffle": True,

            "chunksizes": (
                time_chunk,
                wind_speed.sizes["lat"],
                wind_speed.sizes["lon"]
            ),
        }
    }

    output_ds = (
        wind_speed
        .to_dataset()
    )

    output_ds.attrs = {
        "project":
            "Climate-Driven Capacity Factor Prediction "
            "and Future Energy Yield Assessment of "
            "Wind Farms Using CMIP6 Climate Projections",

        "phase":
            "Phase 1",

        "model":
            model,

        "experiment":
            experiment,

        "variable":
            "daily wind speed",

        "formula":
            "sqrt(uas^2 + vas^2)",

        "historical_period":
            "1950-2014",

        "future_period":
            "2015-2100",

        "spatial_alignment":
            "UAS/VAS overlap; VAS linearly interpolated "
            "to UAS grid where required",

        "extrapolation":
            "None",

        "storage":
            "Float32 NetCDF4 with zlib compression",
    }

    output_ds.to_netcdf(
        output_file,
        mode="w",
        format="NETCDF4",
        engine="netcdf4",
        encoding=encoding
    )

    output_ds.close()

    size_gb = (
        os.path.getsize(
            output_file
        )
        / (1024 ** 3)
    )

    print(
        f"Saved: "
        f"{output_file}"
    )

    print(
        f"File size: "
        f"{size_gb:.3f} GB"
    )

    return size_gb


# Process one model + experiment

def process_one(
    model,
    experiment,
    config
):

    print("\n")
    print("=" * 80)

    print(
        f"PROCESSING "
        f"{model} | {experiment}"
    )

    print("=" * 80)

    model_day = os.path.join(
        BASE_DIR,
        model,
        "Day"
    )

    uas_dir = os.path.join(
        model_day,
        "uas"
    )

    vas_dir = os.path.join(
        model_day,
        "vas"
    )

    output_dir = os.path.join(
        model_day,
        "processed"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        f"{model}_{experiment}_daily_wind_speed.nc"
    )

    # Don't silently accept old files

    if os.path.exists(output_file):

        raise FileExistsError(
            f"Output already exists:\n"
            f"{output_file}\n\n"
            f"Delete old processed outputs before "
            f"starting a clean production run."
        )

    # Disk check

    free_gb = get_free_space_gb(
        BASE_DIR
    )

    print(
        f"Free disk space: "
        f"{free_gb:.2f} GB"
    )

    if free_gb < 10:

        raise RuntimeError(
            "Less than 10 GB free. "
            "Stopping."
        )

    # Select files

    uas_files = select_experiment_files(
        uas_dir,
        experiment
    )

    vas_files = select_experiment_files(
        vas_dir,
        experiment
    )

    if len(uas_files) == 0:

        raise FileNotFoundError(
            f"No UAS files found for "
            f"{model} {experiment}"
        )

    if len(vas_files) == 0:

        raise FileNotFoundError(
            f"No VAS files found for "
            f"{model} {experiment}"
        )

    print_selected_files(
        model,
        "uas",
        experiment,
        uas_files
    )

    print_selected_files(
        model,
        "vas",
        experiment,
        vas_files
    )

    # Open UAS

    print("\nOpening UAS...")

    uas_ds = open_component(
        uas_files,
        "uas"
    )

    uas = uas_ds["uas"]

    # Open VAS

    print("\nOpening VAS...")

    vas_ds = open_component(
        vas_files,
        "vas"
    )

    vas = vas_ds["vas"]

    # Time selection

    print(
        "\nSelecting requested period..."
    )

    uas = validate_time(
        uas,
        experiment,
        config
    )

    vas = validate_time(
        vas,
        experiment,
        config
    )

    # Time alignment

    print(
        "\nAligning UAS/VAS time..."
    )

    uas, vas = xr.align(
        uas,
        vas,
        join="inner"
    )

    if uas.sizes["time"] == 0:

        raise ValueError(
            f"{model} {experiment}: "
            f"no common time points "
            f"between UAS and VAS."
        )

    print(
        f"Common time points: "
        f"{uas.sizes['time']:,}"
    )

    # Grid information

    print_grid_info(
        model,
        experiment,
        uas,
        vas
    )

    # Spatial alignment

    uas, vas = align_spatial(
        uas,
        vas,
        model,
        experiment
    )

    # Wind speed

    wind_speed = calculate_wind_speed(
        uas,
        vas
    )

    # Statistics

    stats = calculate_statistics(
        wind_speed
    )

    # Safety check

    if (
        stats["missing_percent"]
        > 0.000001
    ):

        warnings.warn(
            f"{model} {experiment}: "
            f"missing values remain: "
            f"{stats['missing_percent']:.6f}%"
        )

    # Save daily output

    output_size = save_daily_output(
        wind_speed,
        output_file,
        model,
        experiment
    )

    # Annual summary

    print(
        "\nCalculating annual summary..."
    )

    annual = calculate_annual_summary(
        wind_speed,
        model,
        experiment
    )

    # Monthly summary

    print(
        "Calculating monthly summary..."
    )

    monthly = calculate_monthly_summary(
        wind_speed,
        model,
        experiment
    )

    # Close

    uas_ds.close()
    vas_ds.close()

    del uas
    del vas
    del wind_speed

    gc.collect()

    # Disk

    free_after = get_free_space_gb(
        BASE_DIR
    )

    phase1_size = get_folder_size_gb(
        os.path.join(
            BASE_DIR,
            "PHASE1"
        )
    )

    print(
        f"\nFree space after: "
        f"{free_after:.2f} GB"
    )

    print(
        f"Phase 1 folder size: "
        f"{phase1_size:.3f} GB"
    )

    if phase1_size > MAX_PHASE1_GB:

        raise RuntimeError(
            f"Phase 1 output exceeded "
            f"{MAX_PHASE1_GB} GB."
        )

    return {
        "model":
            model,

        "experiment":
            experiment,

        **stats,

        "output_file":
            output_file,

        "output_size_gb":
            output_size,
    }, annual, monthly


# Main

def main():

    print("\n")
    print("=" * 80)
    print("PHASE 1 — FINAL PRODUCTION RUN")
    print("=" * 80)

    print(
        "\nModels:"
    )

    for model in MODELS:

        print(
            f"  {model}"
        )

    print(
        "\nExpected datasets:"
    )

    print(
        f"  {len(MODELS)} models × "
        f"{len(EXPERIMENTS)} experiments "
        f"= {len(MODELS) * len(EXPERIMENTS)}"
    )

    print(
        "\nHistorical: 1950–2014"
    )

    print(
        "SSP245:     2015–2100"
    )

    print(
        "SSP585:     2015–2100"
    )

    print(
        "\nRaw inputs are ONLY:"
    )

    print(
        r"  Day\uas"
    )

    print(
        r"  Day\vas"
    )

    print(
        "\nLoose Day files are NOT used."
    )

    print(
        "TAS is NOT used in Phase 1."
    )

    print(
        "\nEach experiment is selected "
        "separately from filenames."
    )

    print(
        "\nNo spatial extrapolation."
    )

    print(
        "ACCESS staggered grids use "
        "spatial overlap + interpolation."
    )

    # Create summary directory

    os.makedirs(
        PHASE1_SUMMARY_DIR,
        exist_ok=True
    )

    results = []
    annual_results = []
    monthly_results = []

    failures = []

    expected_total = (
        len(MODELS)
        * len(EXPERIMENTS)
    )

    # Process

    for model in MODELS:

        for experiment, config in (
            EXPERIMENTS.items()
        ):

            try:

                result = process_one(
                    model,
                    experiment,
                    config
                )

                summary, annual, monthly = (
                    result
                )

                results.append(
                    summary
                )

                annual_results.append(
                    annual
                )

                monthly_results.append(
                    monthly
                )

                print(
                    f"\nSUCCESS: "
                    f"{model} | "
                    f"{experiment}"
                )

            except Exception as e:

                print("\n")
                print("!" * 80)

                print(
                    f"FAILED: "
                    f"{model} | "
                    f"{experiment}"
                )

                print(
                    "!" * 80
                )

                print(
                    f"\nError: "
                    f"{repr(e)}"
                )

                failures.append({
                    "model":
                        model,

                    "experiment":
                        experiment,

                    "error":
                        repr(e),
                })

                # IMPORTANT: Continue so we get complete failure report, but final validation will fail.

                continue

    # Save summary

    if results:

        summary_df = pd.DataFrame(
            results
        )

        summary_file = os.path.join(
            PHASE1_SUMMARY_DIR,
            "PHASE1_daily_wind_speed_summary.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False
        )

    # Annual summary

    if annual_results:

        annual_df = pd.concat(
            annual_results,
            ignore_index=True
        )

        annual_file = os.path.join(
            PHASE1_SUMMARY_DIR,
            "PHASE1_annual_wind_speed_summary.csv"
        )

        annual_df.to_csv(
            annual_file,
            index=False
        )

    # Monthly summary

    if monthly_results:

        monthly_df = pd.concat(
            monthly_results,
            ignore_index=True
        )

        monthly_file = os.path.join(
            PHASE1_SUMMARY_DIR,
            "PHASE1_monthly_wind_speed_summary.csv"
        )

        monthly_df.to_csv(
            monthly_file,
            index=False
        )

    # Failure report

    failure_df = pd.DataFrame(
        failures
    )

    failure_file = os.path.join(
        PHASE1_SUMMARY_DIR,
        "PHASE1_failures.csv"
    )

    failure_df.to_csv(
        failure_file,
        index=False
    )

    # Final validation

    success_count = len(
        results
    )

    failure_count = len(
        failures
    )

    print("\n")
    print("=" * 80)
    print("PHASE 1 FINAL VALIDATION")
    print("=" * 80)

    print(
        f"Expected datasets : "
        f"{expected_total}"
    )

    print(
        f"Successful        : "
        f"{success_count}"
    )

    print(
        f"Failed            : "
        f"{failure_count}"
    )

    # Check exact expected set

    expected_pairs = {
        (
            model,
            experiment
        )

        for model in MODELS

        for experiment in EXPERIMENTS
    }

    successful_pairs = {
        (
            row["model"],
            row["experiment"]
        )

        for row in results
    }

    missing_pairs = (
        expected_pairs
        - successful_pairs
    )

    # Final decision

    if (
        success_count == expected_total
        and
        failure_count == 0
        and
        len(missing_pairs) == 0
    ):

        print("\n")
        print(
            "PHASE 1 COMPLETE — VALIDATED"
        )

        print(
            "\nAll 21 model/experiment "
            "datasets were successfully processed."
        )

    else:

        print("\n")
        print(
            "PHASE 1 INCOMPLETE — "
            "DO NOT PROCEED TO PHASE 2"
        )

        print(
            "\nMissing/failed pairs:"
        )

        for pair in sorted(
            missing_pairs
        ):

            print(
                f"  {pair[0]} | {pair[1]}"
            )

    # Output locations

    print("\n")
    print("=" * 80)

    print(
        "Daily outputs:"
    )

    print(
        r"D:\CMIP6\<MODEL>\Day\processed"
    )

    print(
        "\nSummary outputs:"
    )

    print(
        r"D:\CMIP6\PHASE1\summaries"
    )

    print(
        "\nFailure report:"
    )

    print(
        failure_file
    )

    print("\n")


# RUN

if __name__ == "__main__":
    main()