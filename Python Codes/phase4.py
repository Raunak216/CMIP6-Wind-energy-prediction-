
from pathlib import Path
import traceback
import warnings

import numpy as np
import pandas as pd
import xarray as xr

# Configuration: match this project’s existing Phase 3B layout and seven models
BASE = Path(r"D:\CMIP6")
INPUT_DIR = BASE / "PHASE3B" / "native_spatial"
OUTPUT_DIR = BASE / "PHASE4" / "ensemble"
SUMMARY_DIR = BASE / "PHASE4" / "summaries"

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
FUTURES = ["ssp245", "ssp585"]
HUB_HEIGHTS = np.array([80.0, 100.0, 120.0], dtype=np.float32)

# Global 1.25-degree cell-centre grid: 144 lat x 288 lon.
TARGET_LAT = np.arange(-89.375, 89.375 + 1e-8, 1.25, dtype=np.float64)
TARGET_LON = np.arange(0.625, 359.375 + 1e-8, 1.25, dtype=np.float64)

CF_VAR = "capacity_factor"
AEP_VAR = "aep_gwh_per_year"
REGIONS = {
    # Longitudes are given as degrees east in the 0-360 convention.
    "Mexico_Central_America": {"lat": (5.0, 35.0), "lon": (235.0, 300.0)},
    "Northern_South_America": {"lat": (-5.0, 15.0), "lon": (278.0, 320.0)},
    "Caribbean_Gulf_of_Mexico": {"lat": (8.0, 32.0), "lon": (255.0, 305.0)},
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def ensure_source_files():
    missing = []
    for model in MODELS:
        for exp in EXPERIMENTS:
            p = INPUT_DIR / f"{model}_{exp}_spatial_cf_aep.nc"
            if not p.is_file():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            "Phase 3B inputs are missing; no ensemble outputs were written:\n" + "\n".join(missing)
        )


def _deduplicate_coordinate(da, coord):
    vals = np.asarray(da[coord].values, dtype=np.float64)
    _, idx = np.unique(np.round(vals, 8), return_index=True)
    da = da.isel({coord: np.sort(idx)})
    return da.sortby(coord)


def normalise_lon_and_extend(da):
    """Put a 1-D global lon coordinate on 0..360 and add cyclic seam copies."""
    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(f"Expected coordinates named lat/lon; found {list(da.coords)}")
    if da["lat"].ndim != 1 or da["lon"].ndim != 1:
        raise ValueError("This script expects 1-D latitude/longitude coordinates, as in Phase 3B outputs.")
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Expected lat/lon dimensions; got {da.dims}")

    # Convert e.g. -180..180 to 0..360 if any source uses that convention.
    da = da.assign_coords(lon=np.mod(np.asarray(da.lon.values, dtype=np.float64), 360.0))
    da = da.sortby("lon").sortby("lat")
    da = _deduplicate_coordinate(da, "lon")
    da = _deduplicate_coordinate(da, "lat")

    lon = np.asarray(da.lon.values, dtype=np.float64)
    if lon.size < 3 or not np.all(np.diff(lon) > 0):
        raise ValueError("Longitude coordinate is not strictly increasing after normalization.")

    # Periodic copies prevent interp() from returning NaN at the 0/360 seam.
    left = da.isel(lon=slice(-1, None)).assign_coords(lon=[lon[-1] - 360.0])
    right = da.isel(lon=slice(0, 1)).assign_coords(lon=[lon[0] + 360.0])
    extended = xr.concat([left, da, right], dim="lon", coords="minimal", compat="override").sortby("lon")
    return extended


def regrid_model_field(da):
    """Linear interpolation to the 1.25° global target grid, with periodic longitude."""
    da = normalise_lon_and_extend(da)
    result = da.interp(
        lat=xr.DataArray(TARGET_LAT, dims=("lat",)),
        lon=xr.DataArray(TARGET_LON, dims=("lon",)),
        method="linear",
    )
    result = result.assign_coords(lat=TARGET_LAT, lon=TARGET_LON)
    result = result.transpose("hub_height", "lat", "lon").astype("float32")
    return result


def regional_finite_fraction(da, bounds):
    lat_lo, lat_hi = bounds["lat"]
    lon_lo, lon_hi = bounds["lon"]
    subset = da.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
    vals = np.asarray(subset.values)
    return float(np.isfinite(vals).mean()) if vals.size else 0.0


def validate_regridded_field(model, exp, variable, da):
    if da.sizes.get("lat") != 144 or da.sizes.get("lon") != 288:
        raise ValueError(f"{model}/{exp}/{variable}: wrong target shape {da.shape}; expected (3,144,288).")
    if not np.allclose(da.lat.values, TARGET_LAT) or not np.allclose(da.lon.values, TARGET_LON):
        raise ValueError(f"{model}/{exp}/{variable}: output coordinates do not match the intended 1.25° grid.")

    # Critical check: mid-latitude data from 180° to 360° must not disappear.
    mid = da.sel(lat=slice(-55.0, 75.0), lon=slice(180.625, 359.375))
    frac = float(np.isfinite(mid.values).mean()) if mid.size else 0.0
    if frac < 0.995:
        raise ValueError(
            f"{model}/{exp}/{variable}: only {frac:.2%} finite in mid-latitude longitude 180–360°. "
            "Stopping before saving this experiment; inspect source longitude coordinates."
        )

    # Explicit regional tests aimed at the observed blank part of the map.
    for region, bounds in REGIONS.items():
        frac = regional_finite_fraction(da.sel(hub_height=100.0), bounds)
        if frac < 0.99:
            raise ValueError(
                f"{model}/{exp}/{variable}: regional coverage in {region} is {frac:.2%}; "
                "expected >=99% finite. No ensemble file written for this experiment."
            )
    return True


def load_and_regrid(model, exp, variable):
    path = INPUT_DIR / f"{model}_{exp}_spatial_cf_aep.nc"
    with xr.open_dataset(path) as ds:
        if variable not in ds.data_vars:
            raise KeyError(f"{variable} missing in {path}; found {list(ds.data_vars)}")
        if "hub_height" not in ds.coords:
            raise KeyError(f"hub_height coordinate missing in {path}")
        available = np.asarray(ds.hub_height.values, dtype=float)
        if not all(np.any(np.isclose(available, h)) for h in HUB_HEIGHTS):
            raise ValueError(f"One or more hub heights {HUB_HEIGHTS.tolist()} missing from {path}")
        da = ds[variable].sel(hub_height=HUB_HEIGHTS).load()
    if set(("hub_height", "lat", "lon")) != set(da.dims):
        if set(("hub_height", "lat", "lon")).issubset(set(da.dims)):
            da = da.transpose("hub_height", "lat", "lon")
        else:
            raise ValueError(f"Unexpected dimensions in {path.name}/{variable}: {da.dims}")
    out = regrid_model_field(da)
    validate_regridded_field(model, exp, variable, out)
    return out


def stats_over_models(model_fields, variable):
    stacked = xr.concat(
        [model_fields[m].expand_dims(model=[m]) for m in MODELS],
        dim="model",
        join="exact",
    )
    count = stacked.count("model").astype("int8")
    stats = xr.Dataset({
        f"{variable}_ensemble_mean": stacked.mean("model", skipna=True),
        f"{variable}_ensemble_median": stacked.median("model", skipna=True),
        f"{variable}_ensemble_std": stacked.std("model", skipna=True, ddof=0),
        f"{variable}_ensemble_min": stacked.min("model", skipna=True),
        f"{variable}_ensemble_max": stacked.max("model", skipna=True),
        f"{variable}_ensemble_range": stacked.max("model", skipna=True) - stacked.min("model", skipna=True),
        "valid_model_count": count,
    })
    return stats


def make_change_dataset(future_model_fields, historical_model_fields, variable, exp):
    changes = {
        m: future_model_fields[m] - historical_model_fields[m]
        for m in MODELS
    }
    stack = xr.concat([changes[m].expand_dims(model=[m]) for m in MODELS], dim="model", join="exact")
    count = stack.count("model").astype("int8")
    if variable == CF_VAR:
        prefix = "capacity_factor_change"
    else:
        prefix = "aep_change"
    out = xr.Dataset({
        f"{prefix}_ensemble_mean": stack.mean("model", skipna=True),
        f"{prefix}_ensemble_median": stack.median("model", skipna=True),
        f"{prefix}_ensemble_std": stack.std("model", skipna=True, ddof=0),
        f"{prefix}_ensemble_min": stack.min("model", skipna=True),
        f"{prefix}_ensemble_max": stack.max("model", skipna=True),
        f"{prefix}_ensemble_range": stack.max("model", skipna=True) - stack.min("model", skipna=True),
        "valid_model_count": count,
    })
    # Retain both the absolute model-level change and the ensemble count.
    out.attrs.update({
        "model_count_expected": len(MODELS),
        "models": ", ".join(MODELS),
        "experiment": exp,
        "baseline_experiment": "historical",
        "target_grid_resolution_degrees": 1.25,
        "longitude_convention": "0 to 360 degrees east",
        "method": "Future-minus-historical differences calculated per model after regridding, then summarized across models.",
    })
    return out


def save_dataset(ds, path):
    ds = ds.assign_coords(hub_height=HUB_HEIGHTS, lat=TARGET_LAT, lon=TARGET_LON)
    for name, var in ds.data_vars.items():
        if "hub_height" in var.dims and set(var.dims) == {"hub_height", "lat", "lon"}:
            ds[name] = var.transpose("hub_height", "lat", "lon")
    ds.attrs.update({
        "target_grid_resolution_degrees": 1.25,
        "longitude_convention": "0 to 360 degrees east",
        "latitude_convention": "degrees north, ascending",
        "model_count_expected": len(MODELS),
        "models": ", ".join(MODELS),
        "source_directory": str(INPUT_DIR),
        "note": "Interpolated 1.25-degree target grid; interpolation does not increase native model resolution.",
    })
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name, var in ds.data_vars.items()
        if np.issubdtype(var.dtype, np.floating)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=encoding)
    print(f"  SAVED {path}")


def save_ensemble_files(exp, cf_stats, aep_stats):
    cf_path = OUTPUT_DIR / f"CF_ensemble_{exp}_1p25deg.nc"
    aep_path = OUTPUT_DIR / f"AEP_ensemble_{exp}_1p25deg.nc"
    save_dataset(cf_stats, cf_path)
    save_dataset(aep_stats, aep_path)


def save_agreement_file(exp, future_cf, hist_cf, future_aep, hist_aep, metric):
    if metric == "CF":
        future = future_cf
        hist = hist_cf
        var_name = "capacity_factor"
    else:
        future = future_aep
        hist = hist_aep
        var_name = "aep_gwh_per_year"
    changes = {m: future[m] - hist[m] for m in MODELS}
    stack = xr.concat([changes[m].expand_dims(model=[m]) for m in MODELS], dim="model", join="exact")
    count = stack.count("model")
    increase = (stack > 0).where(stack.notnull()).sum("model", skipna=True)
    decrease = (stack < 0).where(stack.notnull()).sum("model", skipna=True)
    out = xr.Dataset({
        "increase_agreement_fraction": (increase / count.where(count > 0)).astype("float32"),
        "decrease_agreement_fraction": (decrease / count.where(count > 0)).astype("float32"),
        "models_increasing": increase.astype("int8"),
        "models_decreasing": decrease.astype("int8"),
        "valid_model_count": count.astype("int8"),
    })
    out.attrs.update({"variable_assessed": var_name, "scenario": exp, "baseline": "historical", "model_count_expected": len(MODELS)})
    path = OUTPUT_DIR / f"{metric}_model_agreement_{exp}_1p25deg.nc"
    save_dataset(out, path)


def main():
    print("=" * 84)
    print("PHASE 4 REPAIR — SEVEN-MODEL GLOBAL ENSEMBLE ON A 1.25-DEGREE GRID")
    print(f"Inputs : {INPUT_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Grid   : {len(TARGET_LAT)} latitude x {len(TARGET_LON)} longitude cells")
    print("Longitude convention: 0–360°E; periodic extension enabled")
    print("Existing *_2deg.nc outputs will not be touched.")
    print("=" * 84)

    if len(TARGET_LAT) != 144 or len(TARGET_LON) != 288:
        raise RuntimeError(f"Internal target grid error: {len(TARGET_LAT)}x{len(TARGET_LON)}")
    ensure_source_files()

    all_fields = {CF_VAR: {}, AEP_VAR: {}}
    summary_rows = []

    for exp in EXPERIMENTS:
        print(f"\nProcessing {exp} from existing native Phase 3B files...")
        exp_fields = {CF_VAR: {}, AEP_VAR: {}}
        for variable in (CF_VAR, AEP_VAR):
            for model in MODELS:
                print(f"  {model}: {variable}")
                field = load_and_regrid(model, exp, variable)
                exp_fields[variable][model] = field
                finite = np.isfinite(field.sel(hub_height=100.0).values)
                mid = field.sel(hub_height=100.0, lat=slice(-55.0, 75.0), lon=slice(180.625, 359.375)).values
                america_fraction = float(np.isfinite(mid).mean())
                summary_rows.append({
                    "model": model,
                    "experiment": exp,
                    "variable": variable,
                    "target_lat_count": int(field.sizes["lat"]),
                    "target_lon_count": int(field.sizes["lon"]),
                    "finite_cells_100m": int(finite.sum()),
                    "total_cells_100m": int(finite.size),
                    "finite_fraction_100m": float(finite.mean()),
                    "finite_fraction_midlat_lon180_360": america_fraction,
                    **{f"finite_fraction_{region}": regional_finite_fraction(field.sel(hub_height=100.0), bounds)
                       for region, bounds in REGIONS.items()},
                })

        cf_stats = stats_over_models(exp_fields[CF_VAR], CF_VAR)
        aep_stats = stats_over_models(exp_fields[AEP_VAR], AEP_VAR)

        # Fail-safe check on the combined ensemble before writing outputs.
        for variable, stats in ((CF_VAR, cf_stats), (AEP_VAR, aep_stats)):
            mean = stats[f"{variable}_ensemble_mean"].sel(hub_height=100.0)
            mid = mean.sel(lat=slice(-55.0, 75.0), lon=slice(180.625, 359.375))
            frac = float(np.isfinite(mid.values).mean())
            if frac < 0.995:
                raise RuntimeError(
                    f"Ensemble {exp}/{variable} still has only {frac:.2%} finite cells over longitudes 180–360° "
                    "at mid-latitudes. Nothing from this experiment was saved."
                )
            for region, bounds in REGIONS.items():
                region_frac = regional_finite_fraction(mean, bounds)
                if region_frac < 0.99:
                    raise RuntimeError(f"Ensemble {exp}/{variable} coverage for {region} is {region_frac:.2%}; stopping.")

        save_ensemble_files(exp, cf_stats, aep_stats)
        all_fields[CF_VAR][exp] = exp_fields[CF_VAR]
        all_fields[AEP_VAR][exp] = exp_fields[AEP_VAR]
        del exp_fields

    # Save per-model differences and model agreement for Phase 5.
    for future in FUTURES:
        for variable in (CF_VAR, AEP_VAR):
            change_ds = make_change_dataset(
                all_fields[variable][future], all_fields[variable]["historical"], variable, future
            )
            prefix = "CF" if variable == CF_VAR else "AEP"
            path = OUTPUT_DIR / f"{prefix}_change_{future}_minus_historical_1p25deg.nc"
            save_dataset(change_ds, path)

        save_agreement_file(future, all_fields[CF_VAR][future], all_fields[CF_VAR]["historical"],
                            all_fields[AEP_VAR][future], all_fields[AEP_VAR]["historical"], "CF")
        save_agreement_file(future, all_fields[CF_VAR][future], all_fields[CF_VAR]["historical"],
                            all_fields[AEP_VAR][future], all_fields[AEP_VAR]["historical"], "AEP")

    report_path = SUMMARY_DIR / "PHASE4_ensemble_summary_1p25deg.csv"
    pd.DataFrame(summary_rows).to_csv(report_path, index=False)
    print(f"\nSaved grid/coverage audit: {report_path}")
    print("\nPHASE 4 REPAIR COMPLETE.")
    print("Now rerun the existing Phase 5 script with GRID_TAG = '1p25deg'.")
    print("The Phase 5 output maps should use the full 0–360° longitude domain.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nPHASE 4 REPAIR FAILED — no claim of completion. Traceback:")
        traceback.print_exc()
        raise
