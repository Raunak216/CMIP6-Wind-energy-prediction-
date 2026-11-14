"""Phase 7: Multi-criteria composite screening and robust candidate region identification."""
from __future__ import annotations

import json
import os
import time
import warnings
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration
BASE = Path(os.environ.get("CMIP6_BASE", r"D:\CMIP6"))
PHASE4_DIR = BASE / "PHASE4" / "ensemble"
PHASE6_DIR = BASE / "PHASE6"
PHASE6_ENSEMBLE_DIR = PHASE6_DIR / "ensemble"
PHASE6_MODEL_DIR = PHASE6_DIR / "model_seasonal_1p25deg"
PHASE7_DIR = BASE / "PHASE7"
FIELD_DIR = PHASE7_DIR / "fields"
TABLE_DIR = PHASE7_DIR / "tables"
FIGURE_DIR = PHASE7_DIR / "figures"
for folder in (FIELD_DIR, TABLE_DIR, FIGURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

MODELS = [
    "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
    "CNRM-CM6-1", "GFDL-CM4", "GFDL-ESM4",
]
EXPERIMENTS = ["historical", "ssp245", "ssp585"]
FUTURES = ["ssp245", "ssp585"]
SEASONS = ["DJF", "MAM", "JJA", "SON"]
HUB_HEIGHTS = [80.0, 100.0, 120.0]
GRID_TAG = "1p25deg"
MIN_VALID_MODELS = 5  # strict Tier-1 candidate requirement
MIN_MODELS_FOR_SCORE = 2  # score all cells with at least two contributing models; never discard valid fields
HISTORICAL_CF_FOR_RETENTION_MIN = 0.01
HIGH_POTENTIAL_PERCENTILE = 75.0
SEASONAL_FLOOR_PERCENTILE = 50.0
ROBUST_COMPOSITE_PERCENTILE = 90.0
MIN_MEAN_SEASONAL_NONDECLINE_AGREEMENT = 0.60
TOP_N_PER_CLASS_HEIGHT = 100
MIN_CELLS_PER_REGION_TILE = 4
REGION_TILE_DEGREES = 5.0
MAP_HEIGHT_M = 100.0
USE_NATURAL_EARTH_LAND_MASK = os.environ.get("PHASE7_USE_LAND_MASK", "1").strip() not in {"0", "false", "False"}
LAND_GEOJSON = PHASE7_DIR / "ne_110m_land.geojson"
LAND_GEOJSON_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson"

# Configurable component weights. Keep each group summing to 1.0.
POTENTIAL_WEIGHTS = {"future_annual_cf": 0.55, "future_seasonal_cf_floor": 0.30, "low_interannual_cv": 0.15}
RESILIENCE_WEIGHTS = {"annual_cf_change": 0.40, "low_model_spread_cv": 0.25,
                      "scenario_consistency": 0.15, "seasonal_nondecline_agreement": 0.20}
COMPOSITE_POTENTIAL_WEIGHT = 0.65
COMPOSITE_RESILIENCE_WEIGHT = 0.35

TARGET_LAT = np.arange(-89.375, 89.375 + 1e-8, 1.25, dtype=np.float64)
TARGET_LON = np.arange(0.625, 359.375 + 1e-8, 1.25, dtype=np.float64)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 4/6 input is missing:\n{path}")


def load_da(path: Path, var: str) -> xr.DataArray:
    require_file(path)
    with xr.open_dataset(path) as ds:
        if var not in ds:
            raise KeyError(f"Variable {var!r} not in {path}. Available: {list(ds.data_vars)}")
        da = ds[var].load()
    return da


def check_spatial_grid(da: xr.DataArray, label: str) -> None:
    for coord in ("lat", "lon"):
        if coord not in da.coords:
            raise ValueError(f"{label}: required coordinate {coord!r} missing")
    if da.sizes.get("lat") != 144 or da.sizes.get("lon") != 288:
        raise ValueError(f"{label}: expected 144x288 1.25-degree grid; got {da.shape}")
    if not np.allclose(da.lat.values, TARGET_LAT, atol=1e-5):
        raise ValueError(f"{label}: latitude coordinates differ from the Phase 4 target grid")
    if not np.allclose(da.lon.values, TARGET_LON, atol=1e-5):
        raise ValueError(f"{label}: longitude coordinates differ from the Phase 4 target grid")


def finite_model_count(path: Path, expected_var: str = "valid_model_count") -> xr.DataArray:
    da = load_da(path, expected_var)
    check_spatial_grid(da.isel(hub_height=0) if "hub_height" in da.dims else da, str(path))
    return da


def get_phase4_data():
    cf_mean, cf_std, cf_count = {}, {}, {}
    aep_mean, aep_std, aep_count = {}, {}, {}
    audit = []
    for exp in EXPERIMENTS:
        cf_path = PHASE4_DIR / f"CF_ensemble_{exp}_{GRID_TAG}.nc"
        aep_path = PHASE4_DIR / f"AEP_ensemble_{exp}_{GRID_TAG}.nc"
        for path in (cf_path, aep_path):
            require_file(path)
        for metric, path, mean_name, std_name, target_mean, target_std, target_count in [
            ("CF", cf_path, "capacity_factor_ensemble_mean", "capacity_factor_ensemble_std", cf_mean, cf_std, cf_count),
            ("AEP", aep_path, "aep_gwh_per_year_ensemble_mean", "aep_gwh_per_year_ensemble_std", aep_mean, aep_std, aep_count),
        ]:
            ds = xr.open_dataset(path)
            try:
                for name in (mean_name, std_name, "valid_model_count"):
                    if name not in ds:
                        raise KeyError(f"{name} missing from {path}; available {list(ds.data_vars)}")
                mean = ds[mean_name].load()
                std = ds[std_name].load()
                count = ds["valid_model_count"].load()
                check_spatial_grid(mean, f"{metric}/{exp}")
                if not np.allclose(mean.hub_height.values, HUB_HEIGHTS):
                    raise ValueError(f"Unexpected hub heights in {path}")
                target_mean[exp], target_std[exp], target_count[exp] = mean, std, count
                for h in HUB_HEIGHTS:
                    arr = mean.sel(hub_height=h).values
                    n = count.sel(hub_height=h).values
                    audit.append({
                        "phase": 4, "metric": metric, "experiment": exp, "hub_height_m": h,
                        "path": str(path), "finite_fraction": float(np.isfinite(arr).mean()),
                        "minimum_valid_model_count": int(np.nanmin(n)),
                        "fraction_cells_with_7_models": float((n == len(MODELS)).mean()),
                        "grid_lat": int(mean.sizes["lat"]), "grid_lon": int(mean.sizes["lon"]),
                    })
            finally:
                ds.close()
    return cf_mean, cf_std, cf_count, aep_mean, aep_std, aep_count, audit


def get_phase6_data():
    means, interannual_std, counts, energy_counts, model_fields = {}, {}, {}, {}, {}
    audit = []
    for exp in EXPERIMENTS:
        path = PHASE6_ENSEMBLE_DIR / f"ensemble_seasonal_{exp}_7models_{GRID_TAG}.nc"
        require_file(path)
        with xr.open_dataset(path) as ds:
            required = ["capacity_factor_ensemble_mean", "capacity_factor_ensemble_std",
                        "capacity_factor_interannual_std_ensemble_mean", "capacity_factor_valid_model_count",
                        "seasonal_energy_gwh_per_season_ensemble_mean",
                        "seasonal_energy_gwh_per_season_valid_model_count"]
            missing = [n for n in required if n not in ds]
            if missing:
                raise KeyError(f"{path} missing required Phase 6 variables: {missing}")
            for var in required:
                check_spatial_grid(ds[var].sel(season="DJF", hub_height=HUB_HEIGHTS[0]), f"{exp}/{var}")
            means[exp] = ds["capacity_factor_ensemble_mean"].load()
            interannual_std[exp] = ds["capacity_factor_interannual_std_ensemble_mean"].load()
            counts[exp] = ds["capacity_factor_valid_model_count"].load()
            energy_count = ds["seasonal_energy_gwh_per_season_valid_model_count"].load()
            energy_counts[exp] = energy_count
            for h in HUB_HEIGHTS:
                for season in SEASONS:
                    a = means[exp].sel(season=season, hub_height=h).values
                    n = counts[exp].sel(season=season, hub_height=h).values
                    e = energy_count.sel(season=season, hub_height=h).values
                    audit.append({
                        "phase": 6, "metric": "seasonal_CF_and_energy", "experiment": exp,
                        "season": season, "hub_height_m": h, "path": str(path),
                        "finite_fraction": float(np.isfinite(a).mean()),
                        "minimum_valid_model_count_CF": int(np.nanmin(n)),
                        "minimum_valid_model_count_energy": int(np.nanmin(e)),
                        "fraction_cells_with_7_models_CF": float((n == len(MODELS)).mean()),
                        "grid_lat": int(means[exp].sizes["lat"]), "grid_lon": int(means[exp].sizes["lon"]),
                    })
    # Load per-model CF seasonal means for paired non-decline agreement.
    for model in MODELS:
        model_fields[model] = {}
        for exp in EXPERIMENTS:
            path = PHASE6_MODEL_DIR / f"{model}_{exp}_seasonal_{GRID_TAG}.nc"
            cf = load_da(path, "capacity_factor_seasonal_mean")
            check_spatial_grid(cf.sel(season="DJF", hub_height=HUB_HEIGHTS[0]), f"{model}/{exp}")
            model_fields[model][exp] = cf
    return means, interannual_std, counts, energy_counts, model_fields, audit


def approx_land_fraction(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, str]:
    """Approximate land fraction of 1.25-degree cells using 9 sample points.

    Attempts to download Natural Earth 1:110m land polygons if not found locally.
    If masking cannot be built, continues with an explicit unclassified global screen.
    """
    if not USE_NATURAL_EARTH_LAND_MASK:
        warnings.warn("PHASE7_USE_LAND_MASK=0: producing global land+ocean-unclassified screening.")
        return np.full((len(lat), len(lon)), np.nan, dtype=np.float32), "all_global_unclassified"
    try:
        try:
            from shapely.geometry import shape
            from shapely import contains_xy, union_all
        except ImportError:
            warnings.warn("Shapely not installed. Land mask unavailable; install shapely>=2.0 to separate land/ocean.")
            return np.full((len(lat), len(lon)), np.nan, dtype=np.float32), "all_global_unclassified"

        if not LAND_GEOJSON.exists():
            print("Natural Earth land-mask file not found; attempting one-time download...")
            try:
                req = urllib.request.Request(LAND_GEOJSON_URL, headers={"User-Agent": "CMIP6-Wind-Phase7/1.0"})
                with urllib.request.urlopen(req, timeout=20) as response:
                    payload = response.read()
                LAND_GEOJSON.write_bytes(payload)
                print(f"Saved Natural Earth land polygons: {LAND_GEOJSON}")
            except Exception as exc:
                warnings.warn(f"Could not download Natural Earth mask ({exc}). Continuing without land/ocean separation.")
                return np.full((len(lat), len(lon)), np.nan, dtype=np.float32), "all_global_unclassified"

        with LAND_GEOJSON.open("r", encoding="utf-8") as f:
            geo = json.load(f)
        polys = [shape(feat["geometry"]) for feat in geo.get("features", []) if feat.get("geometry")]
        if not polys:
            warnings.warn("Natural Earth GeoJSON contains no polygons. Continuing without surface classification.")
            return np.full((len(lat), len(lon)), np.nan, dtype=np.float32), "all_global_unclassified"
        land = union_all(polys)

        lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
        samples = []
        offsets = (-0.45, 0.0, 0.45)  # inside the 1.25-degree cell; coarse fractional estimate
        for dy in offsets:
            for dx in offsets:
                sample_lat = np.clip(lat2d + dy, -89.999999, 89.999999)
                sample_lon_360 = np.mod(lon2d + dx, 360.0)
                sample_lon = np.where(sample_lon_360 > 180.0, sample_lon_360 - 360.0, sample_lon_360)
                samples.append(contains_xy(land, sample_lon, sample_lat))
        fraction = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
        print("Natural Earth coarse land-fraction mask created (9 sample points per cell).")
        return fraction, "natural_earth_110m_land_fraction_9_point_approximation"
    except Exception as exc:
        warnings.warn(f"Land-mask processing failed ({exc}). Continuing without land/ocean separation.")
        return np.full((len(lat), len(lon)), np.nan, dtype=np.float32), "all_global_unclassified"


def surface_classes(land_fraction: np.ndarray, mask_method: str) -> np.ndarray:
    cls = np.full(land_fraction.shape, "all_global_unclassified", dtype=object)
    if mask_method == "all_global_unclassified":
        return cls
    cls[land_fraction >= (2.0 / 3.0)] = "land_dominant"
    cls[land_fraction <= (1.0 / 3.0)] = "ocean_dominant"
    middle = (land_fraction > (1.0 / 3.0)) & (land_fraction < (2.0 / 3.0))
    cls[middle] = "coastal_mixed"
    return cls


def rank_pct(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    r = series.rank(method="average", pct=True)
    return r if higher_is_better else 1.0 - r


def calculate_nondecline_agreement(model_fields: dict[str, dict[str, xr.DataArray]]) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Mean/min fraction of paired model-season cases with future CF >= historical CF."""
    condition_fractions = []
    condition_counts = []
    for scenario in FUTURES:
        for season in SEASONS:
            n_valid = None
            n_nondecline = None
            for model in MODELS:
                hist = model_fields[model]["historical"].sel(season=season)
                fut = model_fields[model][scenario].sel(season=season)
                delta = fut - hist
                valid = np.isfinite(hist) & np.isfinite(fut)
                nondecline = (delta >= -1e-7) & valid
                n_valid = valid.astype(np.int16) if n_valid is None else n_valid + valid.astype(np.int16)
                n_nondecline = nondecline.astype(np.int16) if n_nondecline is None else n_nondecline + nondecline.astype(np.int16)
            fraction = (n_nondecline / n_valid.where(n_valid > 0)).where(n_valid >= MIN_MODELS_FOR_SCORE)
            condition_fractions.append(fraction.expand_dims(condition=[f"{scenario}_{season}"]))
            condition_counts.append(n_valid.expand_dims(condition=[f"{scenario}_{season}"]))
    stack = xr.concat(condition_fractions, dim="condition", join="exact", coords="minimal", compat="override")
    count_stack = xr.concat(condition_counts, dim="condition", join="exact", coords="minimal", compat="override")
    mean_agree = stack.mean("condition", skipna=True).astype(np.float32)
    min_agree = stack.min("condition", skipna=True).astype(np.float32)
    robust_conditions = (stack >= 0.70).sum("condition").astype(np.int8)
    robust_conditions = robust_conditions.where(stack.count("condition") == len(FUTURES) * len(SEASONS))
    return mean_agree, min_agree, robust_conditions


def maybe_select_height(da: xr.DataArray, height: float) -> xr.DataArray:
    return da.sel(hub_height=height) if "hub_height" in da.dims else da


def build_screening_fields():
    print("=" * 82)
    print("PHASE 7 — ROBUST HIGH-POTENTIAL WIND-ENERGY REGION SCREENING")
    print(f"BASE: {BASE}")
    print(f"Phase 4: {PHASE4_DIR}")
    print(f"Phase 6: {PHASE6_DIR}")
    print("=" * 82)

    cf_mean, cf_std, cf_count, aep_mean, aep_std, aep_count, audit4 = get_phase4_data()
    sea_cf, sea_interannual_std, sea_count, sea_energy_count, model_fields, audit6 = get_phase6_data()
    land_frac, land_method = approx_land_fraction(TARGET_LAT, TARGET_LON)
    classes = surface_classes(land_frac, land_method)

    # Conservative future resources: use the lower CF/AEP between SSP245 and SSP585.
    hist_cf = cf_mean["historical"]
    s245_cf, s585_cf = cf_mean["ssp245"], cf_mean["ssp585"]
    conservative_cf = xr.concat([s245_cf, s585_cf], dim="future_scenario").min("future_scenario", skipna=True)
    mean_future_cf = xr.concat([s245_cf, s585_cf], dim="future_scenario").mean("future_scenario", skipna=True)
    conservative_aep = xr.concat([aep_mean["ssp245"], aep_mean["ssp585"]], dim="future_scenario").min("future_scenario", skipna=True)
    # Retention is retained for interpretation only.
    retention_pct = (100.0 * conservative_cf / hist_cf.where(hist_cf >= HISTORICAL_CF_FOR_RETENTION_MIN)).astype(np.float32)
    # Use an absolute change in capacity-factor percentage points for resilience scoring.
    annual_cf_change_pp = (100.0 * (conservative_cf - hist_cf)).astype(np.float32)
    scenario_spread_rel = (abs(s585_cf - s245_cf) / ((s585_cf + s245_cf) / 2.0).where(((s585_cf + s245_cf) / 2.0) > 0)).astype(np.float32)

    # Between-model CF spread CV, averaged over SSP245/SSP585.
    model_cvs = []
    for sc in FUTURES:
        model_cvs.append((cf_std[sc] / cf_mean[sc].where(cf_mean[sc] > 0)).expand_dims(future_scenario=[sc]))
    annual_model_cv = xr.concat(model_cvs, dim="future_scenario").mean("future_scenario", skipna=True).astype(np.float32)

    # Future seasonal floor, mean seasonal CF, and interannual CF coefficient of variation.
    future_season_arrays = []
    future_interannual_cv_arrays = []
    for sc in FUTURES:
        future_season_arrays.append(sea_cf[sc].expand_dims(future_scenario=[sc]))
        inter_std = sea_interannual_std[sc]
        inter_cv = (inter_std / sea_cf[sc].where(sea_cf[sc] > 0)).astype(np.float32)
        future_interannual_cv_arrays.append(inter_cv.expand_dims(future_scenario=[sc]))
    sea_stack = xr.concat(future_season_arrays, dim="future_scenario")
    season_floor = sea_stack.min(dim=("future_scenario", "season"), skipna=True).astype(np.float32)
    season_mean = sea_stack.mean(dim=("future_scenario", "season"), skipna=True).astype(np.float32)
    inter_cv_stack = xr.concat(future_interannual_cv_arrays, dim="future_scenario")
    interannual_cv = inter_cv_stack.mean(dim=("future_scenario", "season"), skipna=True).astype(np.float32)

    nondecline_mean, nondecline_min, robust_conditions = calculate_nondecline_agreement(model_fields)
    for label, da in [("hist CF", hist_cf), ("future conservative CF", conservative_cf),
                      ("seasonal floor", season_floor), ("absolute annual CF change", annual_cf_change_pp),
                      ("retention diagnostic", retention_pct), ("model CV", annual_model_cv), ("scenario spread", scenario_spread_rel),
                      ("non-decline agreement", nondecline_mean)]:
        check_spatial_grid(maybe_select_height(da, HUB_HEIGHTS[0]), label)

    # Most conservative valid-model count across required Phase 4 and Phase 6 products.
    counts_to_min = []
    for sc in EXPERIMENTS:
        counts_to_min.extend([cf_count[sc], aep_count[sc]])
        counts_to_min.append(sea_count[sc].min("season"))
        counts_to_min.append(sea_energy_count[sc].min("season"))
    min_count = xr.concat(counts_to_min, dim="count_source").min("count_source", skipna=True).astype(np.int8)

    fields = xr.Dataset(coords={
        "hub_height": np.asarray(HUB_HEIGHTS, dtype=np.float32),
        "lat": TARGET_LAT,
        "lon": TARGET_LON,
    })
    named = {
        "historical_capacity_factor": hist_cf,
        "ssp245_capacity_factor": s245_cf,
        "ssp585_capacity_factor": s585_cf,
        "mean_future_capacity_factor": mean_future_cf,
        "conservative_future_capacity_factor": conservative_cf,
        "conservative_future_aep_gwh_per_year": conservative_aep,
        "future_seasonal_capacity_factor_floor": season_floor,
        "mean_future_seasonal_capacity_factor": season_mean,
        "future_interannual_capacity_factor_cv": interannual_cv,
        "annual_capacity_factor_retention_percent_diagnostic": retention_pct,
        "annual_capacity_factor_change_percentage_points": annual_cf_change_pp,
        "future_annual_model_capacity_factor_cv": annual_model_cv,
        "relative_ssp_scenario_difference": scenario_spread_rel,
        "mean_seasonal_nondecline_model_agreement": nondecline_mean,
        "minimum_seasonal_nondecline_model_agreement": nondecline_min,
        "number_of_8_future_season_cases_with_at_least_70pct_nondecline_agreement": robust_conditions,
        "minimum_valid_model_count_across_required_inputs": min_count,
    }
    for name, da in named.items():
        fields[name] = da.astype(np.float32) if da.dtype.kind == "f" else da
    fields["approx_land_fraction"] = xr.DataArray(land_frac, dims=("lat", "lon"), coords={"lat": TARGET_LAT, "lon": TARGET_LON})
    class_to_code = {"all_global_unclassified": 0, "land_dominant": 1, "ocean_dominant": 2, "coastal_mixed": 3}
    code_grid = np.vectorize(class_to_code.get)(classes).astype(np.int8)
    fields["surface_class_code"] = xr.DataArray(code_grid, dims=("lat", "lon"), coords={"lat": TARGET_LAT, "lon": TARGET_LON})
    fields["annual_capacity_factor_retention_percent_diagnostic"].attrs.update({
        "long_name": "Diagnostic annual CF retention relative to historical",
        "note": "Masked where historical CF < 0.01; diagnostic only, not used for scoring or eligibility."
    })
    fields["annual_capacity_factor_change_percentage_points"].attrs.update({
        "long_name": "Conservative future minus historical annual capacity-factor change",
        "units": "percentage points",
        "definition": "100 * (min(SSP245 mean CF, SSP585 mean CF) - historical mean CF)"
    })

    # Calculate class-relative percentile scores for each hub height.
    cell_frames = []
    score_arrays = {k: np.full((len(HUB_HEIGHTS), len(TARGET_LAT), len(TARGET_LON)), np.nan, dtype=np.float32)
                    for k in ["potential_score", "resilience_score", "composite_score", "high_potential_candidate", "robust_candidate"]}
    score_fields = {
        "future_annual_cf_percentile": np.full_like(score_arrays["potential_score"], np.nan),
        "future_seasonal_floor_percentile": np.full_like(score_arrays["potential_score"], np.nan),
        "interannual_stability_percentile": np.full_like(score_arrays["potential_score"], np.nan),
        "annual_cf_change_percentile": np.full_like(score_arrays["potential_score"], np.nan),
        "low_model_spread_percentile": np.full_like(score_arrays["potential_score"], np.nan),
        "scenario_consistency_percentile": np.full_like(score_arrays["potential_score"], np.nan),
        "nondecline_agreement_percentile": np.full_like(score_arrays["potential_score"], np.nan),
    }
    lat2d, lon2d = np.meshgrid(TARGET_LAT, TARGET_LON, indexing="ij")
    class_flat = classes.ravel()
    land_flat = land_frac.ravel()
    for hi, h in enumerate(HUB_HEIGHTS):
        flat = lambda da: np.asarray(maybe_select_height(da, h).values).ravel()
        df = pd.DataFrame({
            "hub_height_m": h,
            "lat": lat2d.ravel(),
            "lon_0_360": lon2d.ravel(),
            "lon_minus180_180": ((lon2d.ravel() + 180.0) % 360.0) - 180.0,
            "surface_class": class_flat,
            "approx_land_fraction": land_flat,
            "historical_cf": flat(hist_cf),
            "ssp245_cf": flat(s245_cf),
            "ssp585_cf": flat(s585_cf),
            "conservative_future_cf": flat(conservative_cf),
            "mean_future_cf": flat(mean_future_cf),
            "conservative_future_aep_gwh_per_year_per_3mw_turbine": flat(conservative_aep),
            "future_seasonal_cf_floor": flat(season_floor),
            "mean_future_seasonal_cf": flat(season_mean),
            "future_interannual_cf_cv": flat(interannual_cv),
            "annual_cf_retention_percent_diagnostic": flat(retention_pct),
            "annual_cf_change_percentage_points": flat(annual_cf_change_pp),
            "future_annual_model_cf_cv": flat(annual_model_cv),
            "relative_ssp_scenario_difference": flat(scenario_spread_rel),
            "mean_seasonal_nondecline_model_agreement": flat(nondecline_mean),
            "minimum_seasonal_nondecline_model_agreement": flat(nondecline_min),
            "future_season_cases_with_ge70pct_nondecline_agreement_of_8": flat(robust_conditions),
            "minimum_valid_model_count": flat(min_count),
        })
        # Record each eligibility condition separately.
        df["eligible_model_count"] = df["minimum_valid_model_count"] >= MIN_MODELS_FOR_SCORE
        df["tier1_model_count_qualified"] = df["minimum_valid_model_count"] >= MIN_VALID_MODELS
        df["eligible_finite_conservative_cf"] = np.isfinite(df["conservative_future_cf"])
        df["eligible_finite_seasonal_cf_floor"] = np.isfinite(df["future_seasonal_cf_floor"])
        df["eligible_finite_annual_cf_change"] = np.isfinite(df["annual_cf_change_percentage_points"])
        df["eligible_finite_interannual_cv"] = np.isfinite(df["future_interannual_cf_cv"])
        df["eligible_finite_model_spread_cv"] = np.isfinite(df["future_annual_model_cf_cv"])
        df["eligible_finite_scenario_difference"] = np.isfinite(df["relative_ssp_scenario_difference"])
        df["eligible_finite_nondecline_agreement"] = np.isfinite(df["mean_seasonal_nondecline_model_agreement"])
        df["historical_cf_below_retention_threshold"] = (
            ~np.isfinite(df["historical_cf"]) | (df["historical_cf"] < HISTORICAL_CF_FOR_RETENTION_MIN)
        )
        eligibility_flags = [
            "eligible_model_count", "eligible_finite_conservative_cf",
            "eligible_finite_seasonal_cf_floor", "eligible_finite_annual_cf_change",
            "eligible_finite_interannual_cv", "eligible_finite_model_spread_cv",
            "eligible_finite_scenario_difference", "eligible_finite_nondecline_agreement",
        ]
        df["eligible_for_screening"] = df[eligibility_flags].all(axis=1)
        df["future_annual_cf_percentile"] = np.nan
        df["future_seasonal_floor_percentile"] = np.nan
        df["interannual_stability_percentile"] = np.nan
        df["annual_cf_change_percentile"] = np.nan
        df["low_model_spread_percentile"] = np.nan
        df["scenario_consistency_percentile"] = np.nan
        df["nondecline_agreement_percentile"] = np.nan
        df["potential_score_0_100"] = np.nan
        df["resilience_score_0_100"] = np.nan
        df["composite_score_0_100"] = np.nan
        df["high_potential_candidate"] = False
        df["robust_candidate"] = False

        for cls in sorted(df["surface_class"].unique()):
            ix = df.index[(df["surface_class"] == cls) & df["eligible_for_screening"]]
            if len(ix) < 100:
                warnings.warn(f"Only {len(ix)} eligible cells for class {cls} at {h:.0f}m; percentile rankings may be unstable.")
            if len(ix) == 0:
                continue
            sub = df.loc[ix]
            p_cf = rank_pct(sub["conservative_future_cf"], True) * 100
            p_floor = rank_pct(sub["future_seasonal_cf_floor"], True) * 100
            p_inter_stable = rank_pct(sub["future_interannual_cf_cv"], False) * 100
            p_cf_change = rank_pct(sub["annual_cf_change_percentage_points"], True) * 100
            p_model_stable = rank_pct(sub["future_annual_model_cf_cv"], False) * 100
            p_scenario_stable = rank_pct(sub["relative_ssp_scenario_difference"], False) * 100
            p_agreement = rank_pct(sub["mean_seasonal_nondecline_model_agreement"], True) * 100
            potential = (
                POTENTIAL_WEIGHTS["future_annual_cf"] * p_cf
                + POTENTIAL_WEIGHTS["future_seasonal_cf_floor"] * p_floor
                + POTENTIAL_WEIGHTS["low_interannual_cv"] * p_inter_stable
            )
            resilience = (
                RESILIENCE_WEIGHTS["annual_cf_change"] * p_cf_change
                + RESILIENCE_WEIGHTS["low_model_spread_cv"] * p_model_stable
                + RESILIENCE_WEIGHTS["scenario_consistency"] * p_scenario_stable
                + RESILIENCE_WEIGHTS["seasonal_nondecline_agreement"] * p_agreement
            )
            composite = COMPOSITE_POTENTIAL_WEIGHT * potential + COMPOSITE_RESILIENCE_WEIGHT * resilience
            class_q75_cf = float(sub["conservative_future_cf"].quantile(HIGH_POTENTIAL_PERCENTILE / 100.0))
            class_q50_floor = float(sub["future_seasonal_cf_floor"].quantile(SEASONAL_FLOOR_PERCENTILE / 100.0))
            class_q90_score = float(composite.quantile(ROBUST_COMPOSITE_PERCENTILE / 100.0))
            high = (sub["conservative_future_cf"] >= class_q75_cf) & (sub["future_seasonal_cf_floor"] >= class_q50_floor)
            robust = (high & (composite >= class_q90_score)
                      & (sub["mean_seasonal_nondecline_model_agreement"] >= MIN_MEAN_SEASONAL_NONDECLINE_AGREEMENT)
                      & (sub["minimum_valid_model_count"] >= MIN_VALID_MODELS)
                      & (sub["future_season_cases_with_ge70pct_nondecline_agreement_of_8"] == 8))
            df.loc[ix, "future_annual_cf_percentile"] = p_cf
            df.loc[ix, "future_seasonal_floor_percentile"] = p_floor
            df.loc[ix, "interannual_stability_percentile"] = p_inter_stable
            df.loc[ix, "annual_cf_change_percentile"] = p_cf_change
            df.loc[ix, "low_model_spread_percentile"] = p_model_stable
            df.loc[ix, "scenario_consistency_percentile"] = p_scenario_stable
            df.loc[ix, "nondecline_agreement_percentile"] = p_agreement
            df.loc[ix, "potential_score_0_100"] = potential
            df.loc[ix, "resilience_score_0_100"] = resilience
            df.loc[ix, "composite_score_0_100"] = composite
            df.loc[ix, "high_potential_candidate"] = high.to_numpy(dtype=bool)
            df.loc[ix, "robust_candidate"] = robust.to_numpy(dtype=bool)
            # Store rank/score fields back into lat/lon rasters.
            for key, series in [
                ("potential_score", potential), ("resilience_score", resilience),
                ("composite_score", composite),
                ("high_potential_candidate", high.astype(float)), ("robust_candidate", robust.astype(float)),
            ]:
                flat_arr = score_arrays[key][hi].ravel()
                flat_arr[ix.to_numpy()] = series.to_numpy(dtype=np.float32)
            for key, series in [
                ("future_annual_cf_percentile", p_cf), ("future_seasonal_floor_percentile", p_floor),
                ("interannual_stability_percentile", p_inter_stable), ("annual_cf_change_percentile", p_cf_change),
                ("low_model_spread_percentile", p_model_stable), ("scenario_consistency_percentile", p_scenario_stable),
                ("nondecline_agreement_percentile", p_agreement),
            ]:
                score_fields[key][hi].ravel()[ix.to_numpy()] = series.to_numpy(dtype=np.float32)
        cell_frames.append(df)

    all_cells = pd.concat(cell_frames, ignore_index=True)

    # Explicit cell eligibility audit: counts are non-exclusive because a cell can fail multiple conditions.
    audit_reason_cols = [
        ("too_few_valid_models", "eligible_model_count"),
        ("undefined_conservative_future_cf", "eligible_finite_conservative_cf"),
        ("undefined_seasonal_cf_floor", "eligible_finite_seasonal_cf_floor"),
        ("undefined_annual_cf_change", "eligible_finite_annual_cf_change"),
        ("undefined_interannual_cv", "eligible_finite_interannual_cv"),
        ("undefined_annual_model_spread_cv", "eligible_finite_model_spread_cv"),
        ("undefined_ssp_scenario_difference", "eligible_finite_scenario_difference"),
        ("undefined_seasonal_nondecline_agreement", "eligible_finite_nondecline_agreement"),
    ]
    eligibility_rows = []
    for (height, cls), g in all_cells.groupby(["hub_height_m", "surface_class"], dropna=False):
        base_row = {
            "hub_height_m": float(height),
            "surface_class": str(cls),
            "total_grid_cells": int(len(g)),
            "eligible_grid_cells": int(g["eligible_for_screening"].sum()),
            "unscored_grid_cells": int((~g["eligible_for_screening"]).sum()),
            "historical_cf_below_0p01_cells_diagnostic_only": int(g["historical_cf_below_retention_threshold"].sum()),
            "low_baseline_threshold_used_to_exclude_cells": False,
            "finite_phase_inputs_expected": True,
            "minimum_models_for_scoring": MIN_MODELS_FOR_SCORE,
            "minimum_models_for_tier1": MIN_VALID_MODELS,
        }
        for reason, flag in audit_reason_cols:
            base_row[reason] = int((~g[flag].astype(bool)).sum())
        eligibility_rows.append(base_row)
    eligibility_audit_path = TABLE_DIR / "PHASE7_score_eligibility_audit.csv"
    pd.DataFrame(eligibility_rows).to_csv(eligibility_audit_path, index=False)
    print(f"Saved score-eligibility audit: {eligibility_audit_path}")

    all_cells["region_lat_bin_min"] = -90.0 + np.floor((all_cells["lat"] + 90.0) / REGION_TILE_DEGREES) * REGION_TILE_DEGREES
    all_cells["region_lat_bin_max"] = all_cells["region_lat_bin_min"] + REGION_TILE_DEGREES
    all_cells["region_lon_bin_min_0_360"] = np.floor(all_cells["lon_0_360"] / REGION_TILE_DEGREES) * REGION_TILE_DEGREES
    all_cells["region_lon_bin_max_0_360"] = all_cells["region_lon_bin_min_0_360"] + REGION_TILE_DEGREES
    all_cells["region_tile_id"] = (
        "lat_" + all_cells["region_lat_bin_min"].map(lambda x: f"{x:g}") + "_to_" + all_cells["region_lat_bin_max"].map(lambda x: f"{x:g}")
        + "_lon_" + all_cells["region_lon_bin_min_0_360"].map(lambda x: f"{x:g}") + "_to_" + all_cells["region_lon_bin_max_0_360"].map(lambda x: f"{x:g}")
    )
    all_cells["priority_tier"] = np.select(
        [all_cells["robust_candidate"], all_cells["high_potential_candidate"]],
        ["Tier 1 - robust high potential", "Tier 2 - high potential; robustness criteria not all met"],
        default="Not shortlisted",
    )

    # Build all-field NetCDF. Percentile scores are class-relative by design.
    for name, arr in score_arrays.items():
        fields[name] = xr.DataArray(arr, dims=("hub_height", "lat", "lon"),
                                    coords={"hub_height": HUB_HEIGHTS, "lat": TARGET_LAT, "lon": TARGET_LON})
    for name, arr in score_fields.items():
        fields[name] = xr.DataArray(arr, dims=("hub_height", "lat", "lon"),
                                    coords={"hub_height": HUB_HEIGHTS, "lat": TARGET_LAT, "lon": TARGET_LON})
    fields.attrs.update({
        "title": "Phase 7 global high-potential and climate-resilience screening",
        "project": "Climate-Driven Capacity Factor Prediction and Future Energy Yield Assessment of Wind Farms Using CMIP6 Climate Projections",
        "phase": "Phase 7",
        "target_grid_resolution_degrees": 1.25,
        "models_expected": ", ".join(MODELS),
        "scenarios": "historical, ssp245, ssp585",
        "screening_minimum_valid_models_for_continuous_score": MIN_MODELS_FOR_SCORE,
        "minimum_valid_models_for_tier1_candidates": MIN_VALID_MODELS,
        "surface_class_method": land_method,
        "surface_class_code": "0=all_global_unclassified; 1=land_dominant; 2=ocean_dominant; 3=coastal_mixed",
        "land_fraction_method_note": "Optional Natural Earth 1:110m polygons and 9 sample points per 1.25-degree cell; screening only, not precise coastline or siting classification.",
        "potential_score_weights": json.dumps(POTENTIAL_WEIGHTS),
        "resilience_score_weights": json.dumps(RESILIENCE_WEIGHTS),
        "composite_score_weights": json.dumps({"potential": COMPOSITE_POTENTIAL_WEIGHT, "resilience": COMPOSITE_RESILIENCE_WEIGHT}),
        "robust_candidate_rule": f"Top {ROBUST_COMPOSITE_PERCENTILE}th percentile composite within surface class and height; CF >= class Q{HIGH_POTENTIAL_PERCENTILE:g}; seasonal floor >= class Q{SEASONAL_FLOOR_PERCENTILE:g}; mean non-decline agreement >= {MIN_MEAN_SEASONAL_NONDECLINE_AGREEMENT:.0%}; minimum valid models >= {MIN_VALID_MODELS}; all 8 scenario-season cases must be computable and >=70% agreement.",
        "resilience_scoring_note": "Uses conservative future-minus-historical annual CF change in percentage points instead of a ratio. Historical CF retention remains a diagnostic only, avoiding exclusions where historical CF is below 0.01.",

        "interpretation_warning": "Screening candidates only. Not a wind-farm siting recommendation; no terrain, land-use, protected-area, bathymetry, grid-access, cost, or observations constraints are included.",
    })
    field_path = FIELD_DIR / f"PHASE7_screening_fields_{GRID_TAG}.nc"
    try:
        import netCDF4  # noqa: F401
        engine = "netcdf4"
        enc = {name: {"zlib": True, "complevel": 4, "dtype": "float32"}
               for name, var in fields.data_vars.items() if var.dtype.kind == "f"}
        enc.update({name: {"zlib": True, "complevel": 4}
                    for name, var in fields.data_vars.items() if var.dtype.kind in "iu"})
        fields.to_netcdf(field_path, engine=engine, encoding=enc)
    except ImportError:
        # Scipy's NetCDF3 backend doesn't support NetCDF4 compression encodings.
        fields.to_netcdf(field_path, engine="scipy")
    print(f"Saved screening fields: {field_path}")

    # CSV exports: all grid cells have a score record, but only valid cells have scores.
    all_cells.to_csv(TABLE_DIR / "PHASE7_all_grid_cell_scores.csv", index=False)
    all_cells[all_cells["high_potential_candidate"]].sort_values(
        ["surface_class", "hub_height_m", "composite_score_0_100"], ascending=[True, True, False]
    ).groupby(["surface_class", "hub_height_m"], group_keys=False).head(TOP_N_PER_CLASS_HEIGHT).to_csv(
        TABLE_DIR / "PHASE7_high_potential_cells.csv", index=False)
    robust_cells = all_cells[all_cells["robust_candidate"]].sort_values(
        ["surface_class", "hub_height_m", "composite_score_0_100"], ascending=[True, True, False]
    )
    robust_cells.to_csv(TABLE_DIR / "PHASE7_robust_candidates.csv", index=False)

    # Summarize 5-degree screening tiles. Cos(latitude) weights reduce polar over-weighting.
    elig = all_cells[all_cells["eligible_for_screening"]].copy()
    elig["area_weight"] = np.cos(np.deg2rad(elig["lat"])).clip(lower=0)
    region_rows = []
    group_cols = ["surface_class", "hub_height_m", "region_tile_id", "region_lat_bin_min", "region_lat_bin_max",
                  "region_lon_bin_min_0_360", "region_lon_bin_max_0_360"]
    for keys, g in elig.groupby(group_cols, dropna=False):
        if len(g) < MIN_CELLS_PER_REGION_TILE:
            continue
        weights = g["area_weight"].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(g), dtype=float)
        row = dict(zip(group_cols, keys))
        row.update({
            "eligible_grid_cell_count": int(len(g)),
            "high_potential_cell_count": int(g["high_potential_candidate"].sum()),
            "robust_candidate_cell_count": int(g["robust_candidate"].sum()),
            "area_weighted_mean_composite_score": float(np.average(g["composite_score_0_100"], weights=weights)),
            "area_weighted_mean_potential_score": float(np.average(g["potential_score_0_100"], weights=weights)),
            "area_weighted_mean_resilience_score": float(np.average(g["resilience_score_0_100"], weights=weights)),
            "area_weighted_conservative_future_cf_percent": float(100 * np.average(g["conservative_future_cf"], weights=weights)),
            "area_weighted_future_seasonal_cf_floor_percent": float(100 * np.average(g["future_seasonal_cf_floor"], weights=weights)),
            "area_weighted_annual_cf_change_percentage_points": float(np.average(g["annual_cf_change_percentage_points"], weights=weights)),
            "area_weighted_nondecline_agreement_percent": float(100 * np.average(g["mean_seasonal_nondecline_model_agreement"], weights=weights)),
            "area_weighted_future_aep_gwh_per_year_per_3mw_turbine": float(np.average(g["conservative_future_aep_gwh_per_year_per_3mw_turbine"], weights=weights)),
            "min_valid_model_count": int(g["minimum_valid_model_count"].min()),
        })
        region_rows.append(row)
    regions = pd.DataFrame(region_rows)
    if not regions.empty:
        regions["rank_within_surface_height"] = regions.groupby(["surface_class", "hub_height_m"])[
            "area_weighted_mean_composite_score"].rank(method="first", ascending=False).astype(int)
        regions = regions.sort_values(["surface_class", "hub_height_m", "rank_within_surface_height"])
    regions.to_csv(TABLE_DIR / "PHASE7_region_tiles_5deg.csv", index=False)

    # Input and coverage audit includes counts and file names for traceability.
    pd.DataFrame(audit4 + audit6).to_csv(TABLE_DIR / "PHASE7_input_audit.csv", index=False)

    # Maps at 100 m.
    def plot_raster(data: np.ndarray, title: str, cbar_label: str, filename: str,
                    vmin=None, vmax=None, cmap="viridis", note=None):
        fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
        masked_color = "#9e9e9e"
        chosen_cmap = plt.get_cmap(cmap).copy()
        chosen_cmap.set_bad(color=masked_color)
        plot_data = np.ma.masked_invalid(np.asarray(data, dtype=float))
        ax.set_facecolor(masked_color)
        mesh = ax.pcolormesh(TARGET_LON, TARGET_LAT, plot_data, shading="auto", cmap=chosen_cmap,
                             vmin=vmin, vmax=vmax, rasterized=True)
        ax.set(xlim=(0, 360), ylim=(-90, 90), xlabel="Longitude (degrees east, 0–360)",
               ylabel="Latitude (degrees north)", title=title)
        ax.set_xticks(np.arange(0, 361, 60)); ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(True, linewidth=0.35, alpha=0.25)
        if note:
            ax.text(0.01, 0.015, note, transform=ax.transAxes, fontsize=8.5, va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3})
        cb = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.9); cb.set_label(cbar_label)
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
        print(f"Saved map: {path}")

    hi = HUB_HEIGHTS.index(MAP_HEIGHT_M)
    base_annual = maybe_select_height(conservative_cf, MAP_HEIGHT_M).values
    base_floor = maybe_select_height(season_floor, MAP_HEIGHT_M).values
    plot_raster(base_annual * 100, f"Conservative future capacity factor — lower of SSP245/SSP585 — {int(MAP_HEIGHT_M)} m",
                "Capacity factor (%)", f"PHASE7_conservative_future_CF_{int(MAP_HEIGHT_M)}m.png", vmin=0, vmax=100,
                note="Grey appears only where the underlying Phase 4/6 metric is genuinely undefined; candidate eligibility is shown separately.")
    plot_raster(base_floor * 100, f"Future seasonal CF floor — minimum across scenarios/seasons — {int(MAP_HEIGHT_M)} m",
                "Minimum seasonal CF (%)", f"PHASE7_future_seasonal_CF_floor_{int(MAP_HEIGHT_M)}m.png", vmin=0, vmax=100,
                note="Grey appears only where the underlying Phase 4/6 metric is genuinely undefined; candidate eligibility is shown separately.")
    # Separate eligibility map: green = scored, grey = not scored.
    by_h = all_cells[all_cells["hub_height_m"] == MAP_HEIGHT_M].sort_values(["lat", "lon_0_360"])
    elig_100m = by_h["eligible_for_screening"].to_numpy(dtype=bool).reshape(len(TARGET_LAT), len(TARGET_LON))
    eligibility_display = np.where(elig_100m, 1.0, np.nan)
    plot_raster(eligibility_display, f"Phase 7 score eligibility — {int(MAP_HEIGHT_M)} m",
                "Green = scored; grey = not scored", f"PHASE7_score_eligibility_{int(MAP_HEIGHT_M)}m.png",
                vmin=0, vmax=1, cmap="Greens",
                note="Green = scored using available model data (>=2 models); Tier 1 still requires >=5 models and all 8 seasonal agreement cases.")

    score_map = score_arrays["composite_score"][hi]
    plot_raster(score_map, f"Composite screening score (within surface class) — {int(MAP_HEIGHT_M)} m",
                "Composite score (0–100 percentile score)", f"PHASE7_composite_score_{int(MAP_HEIGHT_M)}m.png", vmin=0, vmax=100,
                note="Scores are calculated within each surface class; grey is reserved for cells with genuinely undefined score inputs.")

    # Candidate maps use the conservative future-CF raster as a continuous background and overlay only shortlisted cells in...
    candidate_map = score_arrays["robust_candidate"][hi]
    candidate_mask = np.isfinite(candidate_map) & (candidate_map >= 0.5)

    def plot_candidate_overlay(background_cf_percent: np.ndarray, candidate_mask: np.ndarray,
                               title: str, filename: str, note: str):
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        fig, ax = plt.subplots(figsize=(15, 6.5), constrained_layout=True)
        bg = np.ma.masked_invalid(np.asarray(background_cf_percent, dtype=float))
        mesh = ax.pcolormesh(TARGET_LON, TARGET_LAT, bg, shading="auto",
                             cmap="viridis", vmin=0, vmax=100, rasterized=True)
        overlay = np.ma.masked_where(~np.asarray(candidate_mask, dtype=bool),
                                     np.ones(candidate_mask.shape, dtype=np.float32))
        candidate_cmap = ListedColormap(["#00e676"])
        candidate_cmap.set_bad((0, 0, 0, 0))
        ax.pcolormesh(TARGET_LON, TARGET_LAT, overlay, shading="auto",
                      cmap=candidate_cmap, vmin=0.5, vmax=1.5,
                      rasterized=True, alpha=0.95)
        ax.set(xlim=(0, 360), ylim=(-90, 90),
               xlabel="Longitude (degrees east, 0–360)",
               ylabel="Latitude (degrees north)", title=title)
        ax.set_xticks(np.arange(0, 361, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(True, linewidth=0.35, alpha=0.25)
        cb = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.9)
        cb.set_label("Conservative future capacity factor (%)")
        ax.legend(handles=[Patch(facecolor="#00e676", edgecolor="black",
                                 label="Tier 1 screening candidate")],
                  loc="lower left", framealpha=0.95)
        ax.text(0.99, 0.015, note, transform=ax.transAxes, fontsize=8,
                va="bottom", ha="right",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3})
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved map: {path}")

    plot_candidate_overlay(
        base_annual * 100.0, candidate_mask,
        f"Tier 1 robust candidates over conservative future CF — all surface classes — {int(MAP_HEIGHT_M)} m",
        f"PHASE7_robust_candidate_cells_{int(MAP_HEIGHT_M)}m.png",
        "Background = conservative future CF; green = selected cells."
    )

    for cls, tag in [("land_dominant", "land"),
                     ("ocean_dominant", "ocean"),
                     ("coastal_mixed", "coastal_mixed")]:
        class_mask = candidate_mask & (classes == cls)
        plot_candidate_overlay(
            base_annual * 100.0, class_mask,
            f"Tier 1 robust {cls.replace('_', ' ')} candidates over conservative future CF — {int(MAP_HEIGHT_M)} m",
            f"PHASE7_robust_candidate_cells_{tag}_{int(MAP_HEIGHT_M)}m.png",
            f"Green = Tier 1 {cls.replace('_', ' ')} candidate cells."
        )

    # Remove obsolete class-only score rasters from earlier versions.
    for tag in ("land", "ocean", "coastal_mixed"):
        obsolete = FIGURE_DIR / f"PHASE7_composite_score_{tag}_{int(MAP_HEIGHT_M)}m.png"
        if obsolete.exists():
            obsolete.unlink()

    manifest = {
        "phase": 7,
        "title": "Robust high-potential wind-energy region screening",
        "base_directory": str(BASE),
        "phase4_input_directory": str(PHASE4_DIR),
        "phase6_ensemble_directory": str(PHASE6_ENSEMBLE_DIR),
        "phase6_model_directory": str(PHASE6_MODEL_DIR),
        "phase7_output_directory": str(PHASE7_DIR),
        "grid": {"resolution_degrees": 1.25, "lat_count": 144, "lon_count": 288,
                 "longitude_centres": "0.625 to 359.375 degrees east"},
        "models": MODELS,
        "experiments": EXPERIMENTS,
        "hub_heights_m": HUB_HEIGHTS,
        "seasons": SEASONS,
        "minimum_valid_models_for_tier1_candidates": MIN_VALID_MODELS,
        "minimum_valid_models_for_continuous_scoring": MIN_MODELS_FOR_SCORE,
        "surface_classification": land_method,
        "weights": {"potential": POTENTIAL_WEIGHTS, "resilience": RESILIENCE_WEIGHTS,
                    "composite": {"potential": COMPOSITE_POTENTIAL_WEIGHT, "resilience": COMPOSITE_RESILIENCE_WEIGHT}},
        "screening_thresholds": {"high_potential_cf_percentile": HIGH_POTENTIAL_PERCENTILE,
                                 "seasonal_floor_percentile": SEASONAL_FLOOR_PERCENTILE,
                                 "composite_percentile": ROBUST_COMPOSITE_PERCENTILE,
                                 "minimum_mean_seasonal_nondecline_agreement": MIN_MEAN_SEASONAL_NONDECLINE_AGREEMENT},
        "resilience_scoring_note": "The resilience component uses percentile rank of conservative future-minus-historical annual CF change in percentage points. Historical CF retention is a diagnostic only and is not an eligibility gate; this avoids masking cells with a low historical baseline.",
        "score_comparability_note": "Scores and percentile gates are calculated within each surface class and hub height; do not compare onshore score directly with offshore score.",
        "aep_note": "Annualized energy yield per representative 3-MW turbine per grid cell; AEP not separately weighted in composite because it is proportional to CF for the same turbine.",
        "limitations": [
            "This is a climate-resource screening, not a project-ready wind-farm site selection.",
            "No terrain, protected-area, land use, bathymetry/depth, transmission-grid distance, logistics, costs, or observed-wind validation layers are included.",
            "The Natural Earth 1:110m land mask is a coarse land/ocean classifier only and should not be used for parcel-level siting.",
            "Continuous scoring accepts >=2 valid models to avoid broad unscored regions; Tier-1 candidate qualification remains strict at >=5 models and all 8 future scenario-season agreement cases.",
            "The 1.25-degree common grid does not increase native climate-model resolution.",
            "Score weights and screening thresholds are analyst-defined and should be sensitivity-tested.",
        ],
    }
    (PHASE7_DIR / "PHASE7_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (PHASE7_DIR / "README_PHASE7.txt").write_text(
        "Phase 7 output guide\n\n"
        "PHASE7_all_grid_cell_scores.csv: score/metric record for every grid cell and hub height.\n"
        "PHASE7_high_potential_cells.csv: top-ranked cells passing potential (CF percentile + seasonal-floor) gates.\n"
        "PHASE7_robust_candidates.csv: Tier 1 subset passing high-potential, composite-score and seasonal non-decline agreement rules.\n"
        "PHASE7_region_tiles_5deg.csv: ranked 5-degree x 5-degree screening tiles, separated by land/ocean/coastal class where the Natural Earth mask is available.\n"
        "PHASE7_input_audit.csv: Phase 4/6 input completeness and valid-model coverage.\n"
        "PHASE7_score_eligibility_audit.csv: counts of why cells were not assigned screening scores.\n"
        "PHASE7_score_eligibility_100m.png: green = scored with at least 2 valid models; grey = genuinely undefined inputs only.\n"
        "fields/PHASE7_screening_fields_1p25deg.nc: gridded inputs, indices and score fields.\n\n"
        "Historical CF retention is a diagnostic only; the composite resilience score uses absolute annual CF change in percentage points to avoid low-baseline ratio masking.\n"
        "Candidate maps show conservative future CF as the background and highlight selected cells in green; non-candidates remain visible in the background.\n"
        "IMPORTANT: candidate cells/tiles are preliminary climate-resource screens only. No terrain, land-use, protected-area, grid-access, bathymetry, cost or measured-wind filters are included. Score weights and thresholds are transparent analyst choices and should be sensitivity-tested before reporting final recommendations.\n",
        encoding="utf-8")

    print("\nPHASE 7 COMPLETE")
    print(f"Fields: {field_path}")
    print(f"Tables: {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"High-potential rows: {int(all_cells['high_potential_candidate'].sum())}")
    print(f"Robust candidate rows: {int(all_cells['robust_candidate'].sum())}")
    if land_method == "all_global_unclassified":
        print("NOTE: Land/ocean mask unavailable; results are global mixed-surface screens. See run manifest.")
    print("Interpret candidates as screening locations, not confirmed wind-farm sites.")
    return field_path


def main():
    start = time.time()
    try:
        build_screening_fields()
    except Exception:
        print("\nPHASE 7 FAILED — traceback follows; no success claim is made.")
        raise
    print(f"Elapsed: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
