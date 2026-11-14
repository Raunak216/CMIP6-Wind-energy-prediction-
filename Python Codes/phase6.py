from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json
import math
import traceback
import warnings
import gc

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# CONFIGURATION — keep consistent with Phases 1, 3B, 4 and 5
BASE = Path(r"D:\CMIP6")

MODELS = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "BCC-CSM2-MR",
    "CanESM5",
    "CNRM-CM6-1",
    "GFDL-CM4",
    "GFDL-ESM4",
]
EXPERIMENTS = ["historical", "ssp245", "ssp585"]
PERIODS = {
    "historical": (1950, 2014),
    "ssp245": (2015, 2100),
    "ssp585": (2015, 2100),
}
HUB_HEIGHTS_M = [80.0, 100.0, 120.0]
REFERENCE_HEIGHT_M = 10.0
ALPHA = 1.0 / 7.0

RATED_POWER_MW = 3.0
CUT_IN_MS = 4.0
RATED_SPEED_MS = 15.0
CUT_OUT_MS = 25.0

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
# Fixed representative seasonal durations for comparability across 360_day, no-leap, and Gregorian model calendars.
STANDARD_SEASON_DAYS = {"DJF": 90.25, "MAM": 92.0, "JJA": 92.0, "SON": 91.0}
MIN_CELL_DAILY_COVERAGE = 0.95

# Phase 4 common target grid: cell centres at 1.25 degree spacing.
GRID_TAG = "1p25deg"
TARGET_LAT = np.arange(-89.375, 89.375 + 1e-8, 1.25, dtype=np.float64)
TARGET_LON = np.arange(0.625, 359.375 + 1e-8, 1.25, dtype=np.float64)

PHASE6_DIR = BASE / "PHASE6"
MODEL_DIR = PHASE6_DIR / "model_seasonal_1p25deg"
ENSEMBLE_DIR = PHASE6_DIR / "ensemble"
CHANGE_DIR = PHASE6_DIR / "changes"
FIGURE_DIR = PHASE6_DIR / "figures"
TABLE_DIR = PHASE6_DIR / "tables"
for folder in (MODEL_DIR, ENSEMBLE_DIR, CHANGE_DIR, FIGURE_DIR, TABLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Set True only when intentionally rebuilding all model-seasonal files.
FORCE_REPROCESS_MODEL_OUTPUTS = False
# Maps are produced at 100 m to keep figure counts manageable.
MAP_HEIGHT_M = 100.0
SCRIPT_VERSION = "1.0.0"

MODEL_METRICS = {
    "capacity_factor": "capacity_factor_seasonal_mean",
    "aep_gwh_per_year": "aep_gwh_per_year_seasonal_mean",
    "seasonal_energy_gwh_per_season": "seasonal_energy_gwh_per_season_mean",
    "mean_wind_speed_hubheight_ms": "mean_wind_speed_hubheight_ms",
}


# Small helpers
def input_path(model: str, experiment: str) -> Path:
    return BASE / model / "Day" / "processed" / f"{model}_{experiment}_daily_wind_speed.nc"


def model_output_path(model: str, experiment: str) -> Path:
    return MODEL_DIR / f"{model}_{experiment}_seasonal_{GRID_TAG}.nc"


def safe_attr(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def find_wind_variable(ds: xr.Dataset) -> str:
    for candidate in ("wind_speed", "wind_speed_10m", "ws10", "ws"):
        if candidate in ds.data_vars:
            return candidate
    candidates = [
        name for name, da in ds.data_vars.items()
        if "time" in da.dims and {"lat", "lon"}.issubset(set(da.dims) | set(da.coords))
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(
        f"Could not uniquely identify daily wind-speed variable. Variables: {list(ds.data_vars)}"
    )


def normalize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "lat" not in ds.coords and "latitude" in ds.variables:
        rename["latitude"] = "lat"
    if "lon" not in ds.coords and "longitude" in ds.variables:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)
    for coord in ("time", "lat", "lon"):
        if coord not in ds.coords:
            raise ValueError(f"Input dataset is missing required coordinate {coord!r}.")
    return ds


def turbine_power_curve(wind_ms: np.ndarray) -> np.ndarray:
    """Representative 3-MW curve matching the current Phase 3B settings."""
    v = np.asarray(wind_ms, dtype=np.float32)
    p = np.zeros_like(v, dtype=np.float32)
    finite = np.isfinite(v) & (v >= 0)
    ramp = finite & (v >= CUT_IN_MS) & (v < RATED_SPEED_MS)
    p[ramp] = RATED_POWER_MW * (
        (v[ramp] ** 3 - CUT_IN_MS ** 3) /
        (RATED_SPEED_MS ** 3 - CUT_IN_MS ** 3)
    )
    rated = finite & (v >= RATED_SPEED_MS) & (v <= CUT_OUT_MS)
    p[rated] = RATED_POWER_MW
    p[~finite] = np.nan
    # Above cut-out remains zero, as intended, not NaN.
    above_cutout = finite & (v > CUT_OUT_MS)
    p[above_cutout] = 0.0
    return p


def season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def required_month_years(season: str, season_year: int):
    if season == "DJF":
        return [(season_year - 1, 12), (season_year, 1), (season_year, 2)]
    return [(season_year, m) for m in SEASON_MONTHS[season]]


def make_season_groups(time_coord: xr.DataArray, start_year: int, end_year: int):
    """Return only complete season-year groups; supports xarray-decoded CF calendars."""
    try:
        years = np.asarray(time_coord.dt.year.values, dtype=int)
        months = np.asarray(time_coord.dt.month.values, dtype=int)
        days = np.asarray(time_coord.dt.day.values, dtype=int)
        days_in_month = np.asarray(time_coord.dt.days_in_month.values, dtype=int)
    except Exception as exc:
        raise RuntimeError(
            "Could not decode the input time coordinate. Ensure xarray/netCDF4/cftime can decode its CF calendar."
        ) from exc

    keep = (years >= start_year) & (years <= end_year)
    original_indices = np.where(keep)[0]
    years, months, days, days_in_month = (
        years[keep], months[keep], days[keep], days_in_month[keep]
    )
    # Translate filtered time indices back to indices in the original DataArray.
    by_month = defaultdict(list)
    group_indices = defaultdict(list)
    group_lookup = {}
    for j, (y, m, d, nd) in enumerate(zip(years, months, days, days_in_month)):
        idx_original = int(original_indices[j])
        by_month[(int(y), int(m))].append((int(d), int(nd), idx_original))
        season = season_for_month(int(m))
        season_year = int(y) + (1 if int(m) == 12 else 0)
        group_indices[(season, season_year)].append(idx_original)
        group_lookup[(season, season_year)] = True

    complete_groups = {s: [] for s in SEASONS}
    incomplete_counts = {s: 0 for s in SEASONS}
    for (season, season_year), indices in sorted(
        group_indices.items(), key=lambda item: (item[0][1], SEASONS.index(item[0][0]))
    ):
        required = required_month_years(season, season_year)
        expected_days = 0
        complete = True
        for ym in required:
            records = by_month.get(ym, [])
            if not records:
                complete = False
                break
            n_days_month = int(records[0][1])
            seen_days = sorted(d for d, _, _ in records)
            if len(records) != n_days_month or seen_days != list(range(1, n_days_month + 1)):
                complete = False
                break
            expected_days += n_days_month
        if len(indices) != expected_days:
            complete = False
        if complete:
            complete_groups[season].append((int(season_year), np.asarray(sorted(indices), dtype=np.int64), int(expected_days)))
        else:
            incomplete_counts[season] += 1

    # Strict completeness check: don't silently publish a seasonal climatology if an interior season-year is absent.
    for s in SEASONS:
        expected_count = (end_year - start_year) if s == "DJF" else (end_year - start_year + 1)
        actual_count = len(complete_groups[s])
        if actual_count != expected_count:
            raise ValueError(
                f"Incomplete temporal coverage for {s}: found {actual_count} complete season-years, "
                f"expected {expected_count} for {start_year}-{end_year}. "
                "Check the daily input for missing dates or a truncated experiment period. "
                "The script stopped rather than silently averaging incomplete data."
            )
    metadata = {
        "time_steps_in_period": int(len(original_indices)),
        "first_year_in_period": int(years.min()) if years.size else None,
        "last_year_in_period": int(years.max()) if years.size else None,
        "complete_season_years": {s: len(complete_groups[s]) for s in SEASONS},
        "incomplete_season_groups_dropped": incomplete_counts,
    }
    return complete_groups, metadata


def mean_and_std(sum_arr, sumsq_arr, count_arr):
    mean = np.full(sum_arr.shape, np.nan, dtype=np.float64)
    std = np.full(sum_arr.shape, np.nan, dtype=np.float64)
    good = count_arr > 0
    mean[good] = sum_arr[good] / count_arr[good]
    variance = np.zeros(sum_arr.shape, dtype=np.float64)
    variance[good] = sumsq_arr[good] / count_arr[good] - mean[good] ** 2
    variance = np.maximum(variance, 0.0)
    std[good] = np.sqrt(variance[good])
    return mean, std


# Compute one model / experiment from daily wind speed
def process_model_experiment(model: str, experiment: str):
    source = input_path(model, experiment)
    outpath = model_output_path(model, experiment)

    if outpath.exists() and not FORCE_REPROCESS_MODEL_OUTPUTS:
        try:
            with xr.open_dataset(outpath) as existing:
                if (
                    existing.attrs.get("phase6_script_version") == SCRIPT_VERSION
                    and existing.sizes.get("season") == 4
                    and existing.sizes.get("hub_height") == 3
                    and existing.sizes.get("lat") == len(TARGET_LAT)
                    and existing.sizes.get("lon") == len(TARGET_LON)
                ):
                    stored_audit = json.loads(existing.attrs.get("time_audit_json", "{}"))
                    print(f"[REUSE] {model} / {experiment}: {outpath.name}")
                    return {
                        "model": model, "experiment": experiment, "input_file": str(source),
                        "output_file": str(outpath), "status": "REUSED",
                        **stored_audit,
                    }
        except Exception:
            pass
        raise RuntimeError(
            f"An existing Phase 6 output is present but is not recognized as version {SCRIPT_VERSION}:\n{outpath}\n"
            "Move the old file aside or set FORCE_REPROCESS_MODEL_OUTPUTS = True before rerunning."
        )

    if not source.exists():
        raise FileNotFoundError(f"Daily wind-speed input not found: {source}")

    start_year, end_year = PERIODS[experiment]
    print("\n" + "=" * 84)
    print(f"PHASE 6 | {model} | {experiment.upper()} | {start_year}-{end_year}")
    print(f"Input : {source}")
    print(f"Output: {outpath}")

    with xr.open_dataset(source, decode_times=True) as raw:
        ds = normalize_lat_lon(raw)
        var_name = find_wind_variable(ds)
        wind = ds[var_name]
        if "time" not in wind.dims or "lat" not in wind.dims or "lon" not in wind.dims:
            raise ValueError(f"Unexpected wind variable dimensions in {source}: {wind.dims}")
        extra_dims = set(wind.dims) - {"time", "lat", "lon"}
        if extra_dims:
            raise ValueError(f"Unexpected extra dimensions {extra_dims} in {source}")
        wind = wind.transpose("time", "lat", "lon")
        source_lat = np.asarray(ds.lat.values, dtype=np.float64)
        source_lon = np.asarray(ds.lon.values, dtype=np.float64)
        source_time = ds["time"]
        calendar = str(source_time.encoding.get("calendar", source_time.attrs.get("calendar", "standard")))
        units = str(wind.attrs.get("units", "unknown"))
        if units not in ("m s-1", "m/s", "m s**-1", "m s^-1", "unknown"):
            warnings.warn(f"Unusual wind-speed units {units!r} in {source}. Expected m s-1.")

        groups, time_audit = make_season_groups(source_time, start_year, end_year)
        # Process only the requested years. The source file remains open while seasonal blocks are read.
        t_years = np.asarray(source_time.dt.year.values, dtype=int)
        in_period = (t_years >= start_year) & (t_years <= end_year)
        period_indices = np.where(in_period)[0]
        time_values = source_time.values
        time_audit.update({
            "source_calendar": calendar,
            "source_variable": var_name,
            "source_units": units,
            "source_time_steps_total": int(wind.sizes["time"]),
            "period_time_steps": int(len(period_indices)),
            "period_first_timestamp": str(time_values[period_indices[0]]) if len(period_indices) else "",
            "period_last_timestamp": str(time_values[period_indices[-1]]) if len(period_indices) else "",
            "native_lat_count": int(len(source_lat)),
            "native_lon_count": int(len(source_lon)),
            "native_lat_min": float(np.nanmin(source_lat)),
            "native_lat_max": float(np.nanmax(source_lat)),
            "native_lon_min": float(np.nanmin(source_lon)),
            "native_lon_max": float(np.nanmax(source_lon)),
        })

        n_seasons = len(SEASONS)
        n_heights = len(HUB_HEIGHTS_M)
        spatial_shape = (n_seasons, n_heights, len(source_lat), len(source_lon))
        sum_metrics = {
            name: np.zeros(spatial_shape, dtype=np.float64)
            for name in ("cf", "aep", "energy", "speed")
        }
        sumsq_metrics = {
            name: np.zeros(spatial_shape, dtype=np.float64)
            for name in ("cf", "aep", "energy", "speed")
        }
        valid_year_count = np.zeros(spatial_shape, dtype=np.int16)

        group_total = sum(len(groups[s]) for s in SEASONS)
        group_done = 0
        for season_index, season in enumerate(SEASONS):
            seasonal_groups = groups[season]
            for season_year, indices, expected_days in seasonal_groups:
                group_done += 1
                # Read one season-year only (about 90 days) to keep RAM usage controlled.
                values = np.asarray(wind.isel(time=indices).values, dtype=np.float32)
                values[~np.isfinite(values) | (values < 0)] = np.nan
                for hi, height in enumerate(HUB_HEIGHTS_M):
                    scale = (float(height) / REFERENCE_HEIGHT_M) ** ALPHA
                    hub_wind = values * np.float32(scale)
                    power = turbine_power_curve(hub_wind)
                    n_valid_days = np.isfinite(hub_wind).sum(axis=0)
                    min_days = int(math.ceil(expected_days * MIN_CELL_DAILY_COVERAGE))
                    good_cell = n_valid_days >= min_days

                    sum_power = np.nansum(power, axis=0, dtype=np.float64)
                    mean_power = np.full(n_valid_days.shape, np.nan, dtype=np.float64)
                    np.divide(sum_power, n_valid_days, out=mean_power, where=n_valid_days > 0)
                    mean_power[~good_cell] = np.nan

                    sum_wind = np.nansum(hub_wind, axis=0, dtype=np.float64)
                    mean_wind = np.full(n_valid_days.shape, np.nan, dtype=np.float64)
                    np.divide(sum_wind, n_valid_days, out=mean_wind, where=n_valid_days > 0)
                    mean_wind[~good_cell] = np.nan

                    cf_year = mean_power / RATED_POWER_MW
                    aep_year = mean_power * 8760.0 / 1000.0
                    energy_year = mean_power * STANDARD_SEASON_DAYS[season] * 24.0 / 1000.0
                    values_by_metric = {
                        "cf": cf_year,
                        "aep": aep_year,
                        "energy": energy_year,
                        "speed": mean_wind,
                    }
                    for metric, metric_values in values_by_metric.items():
                        finite = np.isfinite(metric_values)
                        sum_metrics[metric][season_index, hi] += np.where(finite, metric_values, 0.0)
                        sumsq_metrics[metric][season_index, hi] += np.where(finite, metric_values ** 2, 0.0)
                    valid_year_count[season_index, hi] += good_cell.astype(np.int16)

                del values
                if group_done == 1 or group_done % 40 == 0 or group_done == group_total:
                    print(f"  Complete season-years: {group_done}/{group_total}")

    # Build the seasonal climatology and interannual variability on the native grid.
    means = {}
    stds = {}
    for metric in sum_metrics:
        means[metric], stds[metric] = mean_and_std(
            sum_metrics[metric], sumsq_metrics[metric], valid_year_count
        )

    coords = {
        "season": np.asarray(SEASONS, dtype=str),
        "hub_height": np.asarray(HUB_HEIGHTS_M, dtype=np.float32),
        "lat": source_lat,
        "lon": source_lon,
    }
    dims = ("season", "hub_height", "lat", "lon")
    native = xr.Dataset(
        {
            "capacity_factor_seasonal_mean": (dims, means["cf"].astype(np.float32)),
            "capacity_factor_interannual_std": (dims, stds["cf"].astype(np.float32)),
            "aep_gwh_per_year_seasonal_mean": (dims, means["aep"].astype(np.float32)),
            "aep_gwh_per_year_interannual_std": (dims, stds["aep"].astype(np.float32)),
            "seasonal_energy_gwh_per_season_mean": (dims, means["energy"].astype(np.float32)),
            "seasonal_energy_interannual_std_gwh": (dims, stds["energy"].astype(np.float32)),
            "mean_wind_speed_hubheight_ms": (dims, means["speed"].astype(np.float32)),
            "wind_speed_interannual_std_ms": (dims, stds["speed"].astype(np.float32)),
            "valid_season_year_count": (dims, valid_year_count.astype(np.int16)),
            "season_year_count_available": (("season",), np.asarray([len(groups[s]) for s in SEASONS], dtype=np.int16)),
        },
        coords=coords,
    )

    # Align longitudes to 0..360 and interpolate summary fields to the exact Phase 4 grid.
    native = normalize_spatial_dataset(native)
    regridded = regrid_seasonal_dataset(native)
    regridded.attrs.update({
        "title": f"Seasonal CMIP6 wind-energy fields: {model} / {experiment}",
        "phase": "Phase 6",
        "phase6_script_version": SCRIPT_VERSION,
        "model": model,
        "experiment": experiment,
        "analysis_period": f"{start_year}-{end_year}",
        "models_expected_in_ensemble": len(MODELS),
        "models": ", ".join(MODELS),
        "reference_height_m": REFERENCE_HEIGHT_M,
        "wind_shear_alpha": ALPHA,
        "rated_power_mw": RATED_POWER_MW,
        "cut_in_wind_speed_ms": CUT_IN_MS,
        "rated_wind_speed_ms": RATED_SPEED_MS,
        "cut_out_wind_speed_ms": CUT_OUT_MS,
        "target_grid_resolution_degrees": 1.25,
        "longitude_convention": "0 to 360 degrees east",
        "regridding_method": "linear interpolation in lat/lon; periodic longitude; no latitude extrapolation",
        "daily_wind_power_note": "Power curve applied to daily wind-speed values; sub-daily variability is not resolved.",
        "seasonal_energy_standard_days_json": json.dumps(STANDARD_SEASON_DAYS),
        "minimum_cell_daily_coverage_fraction": MIN_CELL_DAILY_COVERAGE,
        "time_audit_json": json.dumps(time_audit),
        "complete_season_years_json": json.dumps({s: [int(t[0]) for t in groups[s]] for s in SEASONS}),
        "incomplete_season_groups_dropped_json": json.dumps(time_audit["incomplete_season_groups_dropped"]),
    })
    regridded["capacity_factor_seasonal_mean"].attrs.update({
        "long_name": "Climatological seasonal capacity factor",
        "units": "1",
        "definition": "mean power over each season-year / rated power; then mean across complete season-years",
    })
    regridded["capacity_factor_interannual_std"].attrs.update({
        "long_name": "Interannual standard deviation of seasonal capacity factor",
        "units": "1",
    })
    regridded["aep_gwh_per_year_seasonal_mean"].attrs.update({
        "long_name": "Seasonal mean power expressed as annualized energy yield",
        "units": "GWh year-1 per representative 3-MW turbine",
        "definition": "seasonal mean power * 8760 hours / 1000; useful for comparison to Phase 4 AEP",
    })
    regridded["seasonal_energy_gwh_per_season_mean"].attrs.update({
        "long_name": "Energy production over a standardized representative season",
        "units": "GWh per season per representative 3-MW turbine",
        "definition": "seasonal-year mean power * standardized season hours / 1000; standard durations are in dataset attributes",
    })
    regridded["mean_wind_speed_hubheight_ms"].attrs.update({
        "long_name": "Mean hub-height wind speed over each season",
        "units": "m s-1",
    })
    regridded["valid_season_year_count"].attrs.update({
        "long_name": "Number of seasonal years with sufficient daily data per grid cell",
        "units": "season-years",
    })

    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_netcdf(regridded, outpath)
    regridded.close()
    native.close()
    gc.collect()

    audit_row = {
        "model": model,
        "experiment": experiment,
        "input_file": str(source),
        "output_file": str(outpath),
        "status": "PROCESSED",
        **time_audit,
    }
    print(f"  SAVED seasonal fields: {outpath}")
    return audit_row


def normalize_spatial_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Normalize source spatial coordinates and deduplicate the seam before regridding."""
    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise ValueError("Seasonal dataset missing lat/lon coordinates before regridding.")
    ds = ds.assign_coords(lon=np.mod(np.asarray(ds.lon.values, dtype=np.float64), 360.0))
    ds = ds.sortby("lon").sortby("lat")
    lon_vals = np.asarray(ds.lon.values, dtype=np.float64)
    _, lon_idx = np.unique(np.round(lon_vals, 8), return_index=True)
    ds = ds.isel(lon=np.sort(lon_idx))
    lat_vals = np.asarray(ds.lat.values, dtype=np.float64)
    _, lat_idx = np.unique(np.round(lat_vals, 8), return_index=True)
    ds = ds.isel(lat=np.sort(lat_idx))
    ds = ds.sortby("lat").sortby("lon")
    lon_vals = np.asarray(ds.lon.values, dtype=np.float64)
    if len(lon_vals) < 3:
        raise ValueError("Not enough source longitude cells for periodic interpolation.")
    median_step = float(np.median(np.diff(lon_vals)))
    seam_gap = float(lon_vals[0] + 360.0 - lon_vals[-1])
    if not np.isfinite(median_step) or median_step <= 0:
        raise ValueError("Invalid longitude spacing in the source grid.")
    if seam_gap > max(3.0 * median_step, 5.0):
        raise ValueError(
            f"Source longitude does not look global: seam gap={seam_gap:.3f}°, typical step={median_step:.3f}°."
        )
    return ds


def regrid_one(da: xr.DataArray, method: str = "linear") -> xr.DataArray:
    da = da.sortby("lat").sortby("lon")
    lon = np.asarray(da.lon.values, dtype=np.float64)
    left = da.isel(lon=slice(-1, None)).assign_coords(lon=[lon[-1] - 360.0])
    right = da.isel(lon=slice(0, 1)).assign_coords(lon=[lon[0] + 360.0])
    extended = xr.concat([left, da, right], dim="lon").sortby("lon")
    result = extended.interp(
        lat=xr.DataArray(TARGET_LAT, dims=("lat",), name="lat"),
        lon=xr.DataArray(TARGET_LON, dims=("lon",), name="lon"),
        method=method,
        kwargs={"fill_value": np.nan},
    )
    return result.assign_coords(lat=TARGET_LAT, lon=TARGET_LON)


def regrid_seasonal_dataset(ds: xr.Dataset) -> xr.Dataset:
    spatial_vars = {}
    non_spatial_vars = {}
    for name, da in ds.data_vars.items():
        if "lat" in da.dims and "lon" in da.dims:
            method = "nearest" if name == "valid_season_year_count" else "linear"
            spatial_vars[name] = regrid_one(da, method=method)
        else:
            non_spatial_vars[name] = da
    out = xr.Dataset(spatial_vars)
    for name, da in non_spatial_vars.items():
        out[name] = da
    out = out.assign_coords(season=np.asarray(SEASONS, dtype=str), hub_height=np.asarray(HUB_HEIGHTS_M, dtype=np.float32),
                            lat=TARGET_LAT, lon=TARGET_LON)
    # Keep coverage counts as float because out-of-range cells may be NaN.
    if "valid_season_year_count" in out:
        out["valid_season_year_count"] = out["valid_season_year_count"].astype(np.float32)
    return out


# Ensemble and paired change outputs
def build_ensemble(experiment: str) -> Path:
    paths = [model_output_path(m, experiment) for m in MODELS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Cannot build ensemble; missing model-seasonal files:\n" + "\n".join(map(str, missing)))

    model_datasets = []
    for model, path in zip(MODELS, paths):
        with xr.open_dataset(path) as ds:
            if ds.attrs.get("phase6_script_version") != SCRIPT_VERSION:
                raise ValueError(f"Unexpected Phase 6 output version in {path}")
            model_datasets.append(ds.load())

    output = xr.Dataset(coords={
        "season": np.asarray(SEASONS, dtype=str),
        "hub_height": np.asarray(HUB_HEIGHTS_M, dtype=np.float32),
        "lat": TARGET_LAT,
        "lon": TARGET_LON,
    })
    for metric, source_var in MODEL_METRICS.items():
        stack = xr.concat(
            [ds[source_var] for ds in model_datasets],
            dim=xr.IndexVariable("model", MODELS),
            join="exact", coords="minimal", compat="override",
        )
        output[f"{metric}_ensemble_mean"] = stack.mean("model", skipna=True).astype(np.float32)
        output[f"{metric}_ensemble_median"] = stack.median("model", skipna=True).astype(np.float32)
        output[f"{metric}_ensemble_std"] = stack.std("model", skipna=True, ddof=0).astype(np.float32)
        output[f"{metric}_ensemble_min"] = stack.min("model", skipna=True).astype(np.float32)
        output[f"{metric}_ensemble_max"] = stack.max("model", skipna=True).astype(np.float32)
        output[f"{metric}_ensemble_range"] = (stack.max("model", skipna=True) - stack.min("model", skipna=True)).astype(np.float32)
        output[f"{metric}_valid_model_count"] = stack.count("model").astype(np.int8)
        del stack

    # Ensemble-average within-model interannual variability, distinct from model spread.
    for source_var, out_var in [
        ("capacity_factor_interannual_std", "capacity_factor_interannual_std_ensemble_mean"),
        ("aep_gwh_per_year_interannual_std", "aep_interannual_std_ensemble_mean"),
        ("seasonal_energy_interannual_std_gwh", "seasonal_energy_interannual_std_ensemble_mean"),
        ("wind_speed_interannual_std_ms", "wind_speed_interannual_std_ensemble_mean"),
    ]:
        stack = xr.concat(
            [ds[source_var] for ds in model_datasets],
            dim=xr.IndexVariable("model", MODELS),
            join="exact", coords="minimal", compat="override",
        )
        output[out_var] = stack.mean("model", skipna=True).astype(np.float32)
        del stack

    output.attrs.update({
        "title": f"Seven-model seasonal CMIP6 wind-energy ensemble: {experiment}",
        "phase": "Phase 6",
        "phase6_script_version": SCRIPT_VERSION,
        "experiment": experiment,
        "models": ", ".join(MODELS),
        "model_count_expected": len(MODELS),
        "target_grid_resolution_degrees": 1.25,
        "longitude_convention": "0 to 360 degrees east",
        "season_definitions": "DJF, MAM, JJA, SON; incomplete season-years dropped",
        "seasonal_energy_standard_days_json": json.dumps(STANDARD_SEASON_DAYS),
        "method": "Calculate daily power on each native grid; aggregate season-years; regrid seasonal means; compute unweighted cellwise ensemble statistics.",
        "caveat": "1.25-degree target grid standardizes comparison but does not increase native climate model resolution.",
    })
    path = ENSEMBLE_DIR / f"ensemble_seasonal_{experiment}_7models_{GRID_TAG}.nc"
    save_netcdf(output, path)
    for ds in model_datasets:
        ds.close()
    return path


def build_change(scenario: str) -> Path:
    future_paths = [model_output_path(m, scenario) for m in MODELS]
    hist_paths = [model_output_path(m, "historical") for m in MODELS]
    if any(not p.exists() for p in future_paths + hist_paths):
        raise FileNotFoundError(f"Cannot build {scenario} changes: some model-seasonal inputs are missing.")

    future_ds_list = []
    hist_ds_list = []
    for fp, hp in zip(future_paths, hist_paths):
        with xr.open_dataset(fp) as ds:
            future_ds_list.append(ds.load())
        with xr.open_dataset(hp) as ds:
            hist_ds_list.append(ds.load())

    out = xr.Dataset(coords={
        "season": np.asarray(SEASONS, dtype=str),
        "hub_height": np.asarray(HUB_HEIGHTS_M, dtype=np.float32),
        "lat": TARGET_LAT,
        "lon": TARGET_LON,
    })
    for metric, source_var in MODEL_METRICS.items():
        differences = [f[source_var] - h[source_var] for f, h in zip(future_ds_list, hist_ds_list)]
        stack = xr.concat(
            differences,
            dim=xr.IndexVariable("model", MODELS),
            join="exact", coords="minimal", compat="override",
        )
        count = stack.count("model")
        prefix = f"{metric}_change"
        out[f"{prefix}_ensemble_mean"] = stack.mean("model", skipna=True).astype(np.float32)
        out[f"{prefix}_ensemble_median"] = stack.median("model", skipna=True).astype(np.float32)
        out[f"{prefix}_ensemble_std"] = stack.std("model", skipna=True, ddof=0).astype(np.float32)
        out[f"{prefix}_ensemble_min"] = stack.min("model", skipna=True).astype(np.float32)
        out[f"{prefix}_ensemble_max"] = stack.max("model", skipna=True).astype(np.float32)
        out[f"{prefix}_ensemble_range"] = (stack.max("model", skipna=True) - stack.min("model", skipna=True)).astype(np.float32)
        out[f"{prefix}_valid_model_count"] = count.astype(np.int8)
        out[f"{metric}_increase_agreement_fraction"] = ((stack > 0).where(stack.notnull()).sum("model", skipna=True) / count.where(count > 0)).astype(np.float32)
        del stack, differences

    out.attrs.update({
        "title": f"Seasonal wind-energy changes: {scenario} minus historical",
        "phase": "Phase 6",
        "phase6_script_version": SCRIPT_VERSION,
        "scenario": scenario,
        "baseline": "historical",
        "models": ", ".join(MODELS),
        "model_count_expected": len(MODELS),
        "target_grid_resolution_degrees": 1.25,
        "change_definition": "Future seasonal climatological mean minus historical seasonal climatological mean, paired by model.",
        "agreement_definition": "Fraction of valid paired models with a positive change at each grid cell.",
    })
    path = CHANGE_DIR / f"change_seasonal_{scenario}_minus_historical_7models_{GRID_TAG}.nc"
    save_netcdf(out, path)
    for ds in future_ds_list + hist_ds_list:
        ds.close()
    return path


def save_netcdf(ds: xr.Dataset, path: Path):
    """Write compressed NetCDF when a supporting engine is installed; otherwise use SciPy."""
    encoding = {}
    engine = None
    try:
        import netCDF4  # noqa: F401
        engine = "netcdf4"
    except ImportError:
        try:
            import h5netcdf  # noqa: F401
            engine = "h5netcdf"
        except ImportError:
            engine = "scipy"
    if engine != "scipy":
        for name, var in ds.data_vars.items():
            if np.issubdtype(var.dtype, np.floating):
                encoding[name] = {"zlib": True, "complevel": 4, "dtype": "float32"}
            elif np.issubdtype(var.dtype, np.integer):
                encoding[name] = {"zlib": True, "complevel": 4}
        ds.to_netcdf(path, engine=engine, encoding=encoding)
    else:
        # The SciPy backend supports NetCDF3 but not compression options.
        ds.to_netcdf(path, engine=engine)
    print(f"Saved NetCDF: {path} (engine={engine})")


# Summaries and maps
def area_weighted_global_mean(da: xr.DataArray) -> float:
    if "lat" not in da.dims or "lon" not in da.dims:
        return float(da.mean(skipna=True))
    weights = xr.DataArray(
        np.cos(np.deg2rad(da.lat.values)).clip(min=0),
        dims=("lat",), coords={"lat": da.lat},
    )
    return float(da.weighted(weights).mean(dim=("lat", "lon"), skipna=True))


def save_global_summaries():
    rows = []
    change_rows = []
    metric_vars = {
        "capacity_factor": ("capacity_factor_ensemble_mean", "fraction", 100.0),
        "aep_gwh_per_year": ("aep_gwh_per_year_ensemble_mean", "GWh/year per 3-MW turbine", 1.0),
        "seasonal_energy_gwh_per_season": ("seasonal_energy_gwh_per_season_ensemble_mean", "GWh/standardized season per 3-MW turbine", 1.0),
        "mean_wind_speed_hubheight_ms": ("mean_wind_speed_hubheight_ms_ensemble_mean", "m/s", 1.0),
    }
    for exp in EXPERIMENTS:
        path = ENSEMBLE_DIR / f"ensemble_seasonal_{exp}_7models_{GRID_TAG}.nc"
        with xr.open_dataset(path) as ds:
            for season in SEASONS:
                for h in HUB_HEIGHTS_M:
                    for metric, (var, units, factor) in metric_vars.items():
                        da = ds[var].sel(season=season, hub_height=h)
                        value = area_weighted_global_mean(da) * factor
                        rows.append({
                            "experiment": exp, "season": season, "hub_height_m": h,
                            "metric": metric, "area_weighted_global_mean": value,
                            "reported_units": "percent" if metric == "capacity_factor" else units,
                            "ensemble_spatial_mean_std": area_weighted_global_mean(
                                ds[f"{metric}_ensemble_std"].sel(season=season, hub_height=h)
                            ) * factor,
                            "mean_valid_model_count": area_weighted_global_mean(
                                ds[f"{metric}_valid_model_count"].sel(season=season, hub_height=h)
                            ),
                        })

    for scenario in ("ssp245", "ssp585"):
        path = CHANGE_DIR / f"change_seasonal_{scenario}_minus_historical_7models_{GRID_TAG}.nc"
        with xr.open_dataset(path) as ds:
            for season in SEASONS:
                for h in HUB_HEIGHTS_M:
                    for metric, (_, units, factor) in metric_vars.items():
                        change_var = f"{metric}_change_ensemble_mean"
                        da = ds[change_var].sel(season=season, hub_height=h)
                        agreement = ds[f"{metric}_increase_agreement_fraction"].sel(season=season, hub_height=h)
                        change_rows.append({
                            "scenario": scenario, "baseline": "historical", "season": season,
                            "hub_height_m": h, "metric": metric,
                            "area_weighted_mean_change": area_weighted_global_mean(da) * factor,
                            "change_units": "percentage points" if metric == "capacity_factor" else units,
                            "area_weighted_mean_positive_change_agreement_percent": area_weighted_global_mean(agreement) * 100.0,
                            "mean_valid_model_count": area_weighted_global_mean(
                                ds[f"{metric}_change_valid_model_count"].sel(season=season, hub_height=h)
                            ),
                        })

    pd.DataFrame(rows).to_csv(TABLE_DIR / "PHASE6_seasonal_global_summary.csv", index=False)
    pd.DataFrame(change_rows).to_csv(TABLE_DIR / "PHASE6_seasonal_change_summary.csv", index=False)
    print(f"Saved CSV: {TABLE_DIR / 'PHASE6_seasonal_global_summary.csv'}")
    print(f"Saved CSV: {TABLE_DIR / 'PHASE6_seasonal_change_summary.csv'}")


def robust_color_map():
    """Shared map scales across seasons/scenarios for visual comparability."""
    cf_values = []
    energy_values = []
    cf_change_values = []
    energy_change_values = []
    for exp in EXPERIMENTS:
        p = ENSEMBLE_DIR / f"ensemble_seasonal_{exp}_7models_{GRID_TAG}.nc"
        with xr.open_dataset(p) as ds:
            for season in SEASONS:
                cf = ds["capacity_factor_ensemble_mean"].sel(season=season, hub_height=MAP_HEIGHT_M).values * 100.0
                energy = ds["seasonal_energy_gwh_per_season_ensemble_mean"].sel(season=season, hub_height=MAP_HEIGHT_M).values
                cf_values.append(cf[np.isfinite(cf)])
                energy_values.append(energy[np.isfinite(energy)])
    for sc in ("ssp245", "ssp585"):
        p = CHANGE_DIR / f"change_seasonal_{sc}_minus_historical_7models_{GRID_TAG}.nc"
        with xr.open_dataset(p) as ds:
            for season in SEASONS:
                cf = ds["capacity_factor_change_ensemble_mean"].sel(season=season, hub_height=MAP_HEIGHT_M).values * 100.0
                energy = ds["seasonal_energy_gwh_per_season_change_ensemble_mean"].sel(season=season, hub_height=MAP_HEIGHT_M).values
                cf_change_values.append(cf[np.isfinite(cf)])
                energy_change_values.append(energy[np.isfinite(energy)])
    def p98_abs(chunks, fallback):
        usable = [x for x in chunks if x.size]
        if not usable:
            return fallback
        return float(np.nanpercentile(np.abs(np.concatenate(usable)), 98)) or fallback
    energy_pool = [x for x in energy_values if x.size]
    energy_max = float(np.nanpercentile(np.concatenate(energy_pool), 98.5)) if energy_pool else 1.0
    return {
        "energy_vmax": max(energy_max, 0.1),
        "cf_change_limit": max(p98_abs(cf_change_values, 0.05), 0.01),
        "energy_change_limit": max(p98_abs(energy_change_values, 0.05), 0.01),
    }


def plot_map(da: xr.DataArray, title: str, cbar_label: str, filename: str,
             vmin=None, vmax=None, diverging=False):
    fig, ax = plt.subplots(figsize=(14, 6.2), constrained_layout=True)
    cmap_name = "RdBu_r" if diverging else "viridis"
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgrey")
    arr = np.asarray(da.values, dtype=np.float64)
    mesh = ax.pcolormesh(
        da.lon.values, da.lat.values, np.ma.masked_invalid(arr),
        shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set(xlim=(0, 360), ylim=(-90, 90), xlabel="Longitude (degrees east)",
           ylabel="Latitude (degrees north)", title=title)
    ax.set_xticks(np.arange(0, 361, 50))
    ax.set_yticks(np.arange(-80, 81, 20))
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(cbar_label)
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")


def generate_maps():
    scales = robust_color_map()
    # Scenario climatology maps: 4 seasons x 3 scenarios x 2 mapped metrics at 100 m.
    for exp in EXPERIMENTS:
        path = ENSEMBLE_DIR / f"ensemble_seasonal_{exp}_7models_{GRID_TAG}.nc"
        with xr.open_dataset(path) as ds:
            for season in SEASONS:
                cf = ds["capacity_factor_ensemble_mean"].sel(season=season, hub_height=MAP_HEIGHT_M) * 100.0
                energy = ds["seasonal_energy_gwh_per_season_ensemble_mean"].sel(season=season, hub_height=MAP_HEIGHT_M)
                plot_map(
                    cf, f"Seven-model seasonal mean capacity factor — {exp.upper()} — {season} — {int(MAP_HEIGHT_M)} m",
                    "Capacity factor (%)", f"CF_seasonal_{exp}_{season}_{int(MAP_HEIGHT_M)}m_{GRID_TAG}.png",
                    vmin=0, vmax=100,
                )
                plot_map(
                    energy, f"Seasonal energy per representative season — {exp.upper()} — {season} — {int(MAP_HEIGHT_M)} m",
                    "Energy (GWh per standardized season per representative 3-MW turbine)",
                    f"seasonal_energy_{exp}_{season}_{int(MAP_HEIGHT_M)}m_{GRID_TAG}.png",
                    vmin=0, vmax=scales["energy_vmax"],
                )

    # Paired future-minus-historical maps and model-agreement maps.
    for sc in ("ssp245", "ssp585"):
        path = CHANGE_DIR / f"change_seasonal_{sc}_minus_historical_7models_{GRID_TAG}.nc"
        with xr.open_dataset(path) as ds:
            for season in SEASONS:
                cf_change = ds["capacity_factor_change_ensemble_mean"].sel(season=season, hub_height=MAP_HEIGHT_M) * 100.0
                en_change = ds["seasonal_energy_gwh_per_season_change_ensemble_mean"].sel(season=season, hub_height=MAP_HEIGHT_M)
                cf_agree = ds["capacity_factor_increase_agreement_fraction"].sel(season=season, hub_height=MAP_HEIGHT_M) * 100.0
                en_agree = ds["seasonal_energy_gwh_per_season_increase_agreement_fraction"].sel(season=season, hub_height=MAP_HEIGHT_M) * 100.0
                plot_map(
                    cf_change, f"Seasonal capacity-factor change — {sc.upper()} minus historical — {season} — {int(MAP_HEIGHT_M)} m",
                    "Change in capacity factor (percentage points)",
                    f"CF_change_seasonal_{sc}_{season}_{int(MAP_HEIGHT_M)}m_{GRID_TAG}.png",
                    vmin=-scales["cf_change_limit"], vmax=scales["cf_change_limit"], diverging=True,
                )
                plot_map(
                    en_change, f"Seasonal energy change — {sc.upper()} minus historical — {season} — {int(MAP_HEIGHT_M)} m",
                    "Change in energy (GWh per standardized season per representative 3-MW turbine)",
                    f"seasonal_energy_change_{sc}_{season}_{int(MAP_HEIGHT_M)}m_{GRID_TAG}.png",
                    vmin=-scales["energy_change_limit"], vmax=scales["energy_change_limit"], diverging=True,
                )
                plot_map(
                    cf_agree, f"Model agreement for increasing seasonal CF — {sc.upper()} — {season} — {int(MAP_HEIGHT_M)} m",
                    "Models projecting CF increase (%)",
                    f"CF_increase_agreement_{sc}_{season}_{int(MAP_HEIGHT_M)}m_{GRID_TAG}.png",
                    vmin=0, vmax=100,
                )
                plot_map(
                    en_agree, f"Model agreement for increasing seasonal energy — {sc.upper()} — {season} — {int(MAP_HEIGHT_M)} m",
                    "Models projecting energy increase (%)",
                    f"seasonal_energy_increase_agreement_{sc}_{season}_{int(MAP_HEIGHT_M)}m_{GRID_TAG}.png",
                    vmin=0, vmax=100,
                )


def write_readme_and_manifest(audit_rows):
    audit_path = TABLE_DIR / "PHASE6_input_and_season_coverage_audit.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    manifest = {
        "phase": 6,
        "phase6_script_version": SCRIPT_VERSION,
        "project": "Climate-Driven Capacity Factor Prediction and Future Energy Yield Assessment of Wind Farms Using CMIP6 Climate Projections",
        "base_directory": str(BASE),
        "daily_input_pattern": str(BASE / "<MODEL>" / "Day" / "processed" / "<MODEL>_<experiment>_daily_wind_speed.nc"),
        "phase6_output_directory": str(PHASE6_DIR),
        "models": MODELS,
        "experiments": PERIODS,
        "hub_heights_m": HUB_HEIGHTS_M,
        "reference_height_m": REFERENCE_HEIGHT_M,
        "wind_shear_alpha": ALPHA,
        "rated_power_mw": RATED_POWER_MW,
        "cut_in_ms": CUT_IN_MS,
        "rated_speed_ms": RATED_SPEED_MS,
        "cut_out_ms": CUT_OUT_MS,
        "target_grid_degrees": 1.25,
        "target_grid_shape_lat_lon": [len(TARGET_LAT), len(TARGET_LON)],
        "seasons": SEASONS,
        "season_definition": "DJF season-year identified by Jan/Feb year; December belongs to following season-year. Incomplete season-years are omitted.",
        "standard_season_days": STANDARD_SEASON_DAYS,
        "minimum_cell_daily_coverage_fraction": MIN_CELL_DAILY_COVERAGE,
        "daily_power_curve_caveat": "Daily wind-speed fields do not resolve sub-daily variability; power is evaluated at daily wind speed.",
        "aep_definition": "Seasonal mean power converted to annualized GWh/year per representative 3-MW turbine, using 8760 h/year.",
        "seasonal_energy_definition": "Seasonal mean power multiplied by standardized season duration, yielding GWh per representative season per turbine.",
        "interannual_std_definition": "Standard deviation of seasonal-year means within each model and season; separate from between-model ensemble spread.",
        "figure_map_height_m": MAP_HEIGHT_M,
        "run_timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "audit_csv": str(audit_path),
    }
    with open(PHASE6_DIR / "PHASE6_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme = f"""PHASE 6 — SEASONAL CMIP6 WIND-ENERGY ANALYSIS

Inputs: daily Phase 1 wind-speed files for {len(MODELS)} models and three experiments.
Outputs are written below {PHASE6_DIR}.

Folders:
  model_seasonal_1p25deg/  Per-model seasonal climatologies, regridded to 1.25 degrees.
  ensemble/                Seven-model seasonal means, medians, spread and valid-model counts.
  changes/                 SSP245/SSP585 minus historical changes and positive-change agreement.
  figures/                 Global maps at {int(MAP_HEIGHT_M)} m; NetCDF retains all three hub heights.
  tables/                  Input/season coverage audit and global spatial summaries.

Methods:
  Wind speed at hub height = daily reference-height speed * (hub/reference height)^(1/7).
  Turbine curve: 3 MW; cut-in {CUT_IN_MS} m/s; rated {RATED_SPEED_MS} m/s; cut-out {CUT_OUT_MS} m/s.
  Seasons: DJF, MAM, JJA, SON. Incomplete edge seasons are dropped.
  Cells require at least {MIN_CELL_DAILY_COVERAGE:.0%} valid days in a season-year.
  Model seasonal means are calculated on each model native grid, then linearly regridded
  to the common {GRID_TAG} comparison grid with periodic longitude and no latitude extrapolation.
  Seasonal energy is standardized to common season durations: {STANDARD_SEASON_DAYS} days.

Interpretation caveats:
  * The daily wind-speed data do not resolve sub-daily wind variability.
  * A 1.25-degree target grid standardizes comparisons but is not new native climate detail.
  * Ensemble valid-model counts may be lower than seven at polar edges where source model
    latitude coverage differs. Consult NetCDF valid-model-count variables and audit CSV.
  * AEP is annualized seasonal mean power for one representative 3-MW turbine per cell;
    standardized seasonal energy is not a total regional/farm generation estimate.
"""
    with open(PHASE6_DIR / "README_PHASE6.txt", "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"Saved CSV: {audit_path}")
    print(f"Saved manifest: {PHASE6_DIR / 'PHASE6_run_manifest.json'}")
    print(f"Saved readme: {PHASE6_DIR / 'README_PHASE6.txt'}")


def preflight():
    missing = [str(input_path(m, e)) for m in MODELS for e in EXPERIMENTS if not input_path(m, e).is_file()]
    if missing:
        raise FileNotFoundError(
            "Phase 6 preflight failed. These daily Phase 1 inputs are missing:\n" + "\n".join(missing)
        )
    print("PHASE 6 PREFLIGHT PASSED")
    print(f"Daily source files found: {len(MODELS) * len(EXPERIMENTS)} / {len(MODELS) * len(EXPERIMENTS)}")
    print(f"Target grid: {len(TARGET_LAT)} lat x {len(TARGET_LON)} lon = 1.25 degrees")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Outputs: {PHASE6_DIR}")


def main():
    print("=" * 86)
    print("PHASE 6 — SEASONAL CMIP6 WIND-ENERGY ANALYSIS")
    print("Daily wind speed -> hub-height wind -> turbine power -> seasonal CF/AEP/energy")
    print("=" * 86)
    preflight()

    audit_rows = []
    audit_path = TABLE_DIR / "PHASE6_input_and_season_coverage_audit.csv"
    for model in MODELS:
        for exp in EXPERIMENTS:
            row = process_model_experiment(model, exp)
            audit_rows.append(row)
            # Save the audit incrementally so completed work remains documented if interrupted.
            pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    print("\nBuilding seasonal ensembles...")
    for exp in EXPERIMENTS:
        build_ensemble(exp)

    print("\nBuilding paired future-minus-historical changes and model agreement...")
    for sc in ("ssp245", "ssp585"):
        build_change(sc)

    print("\nWriting summary tables...")
    save_global_summaries()

    print("\nGenerating maps...")
    generate_maps()

    write_readme_and_manifest(audit_rows)
    print("\n" + "=" * 86)
    print("PHASE 6 COMPLETE")
    print(f"Model seasonal files: {MODEL_DIR}")
    print(f"Ensemble files:       {ENSEMBLE_DIR}")
    print(f"Change files:         {CHANGE_DIR}")
    print(f"Figures:              {FIGURE_DIR}")
    print(f"Tables:               {TABLE_DIR}")
    print("Review PHASE6_input_and_season_coverage_audit.csv before interpreting results.")
    print("=" * 86)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nPHASE 6 FAILED — completed model files are retained for resume.")
        traceback.print_exc()
        raise
