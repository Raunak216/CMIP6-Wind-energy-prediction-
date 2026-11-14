from pathlib import Path
import warnings
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Configuration — edit only if your phase 4 filenames/output paths differ
BASE = Path(r"D:\CMIP6")
PHASE4_DIR = BASE / "PHASE4" / "ensemble"
PHASE5_DIR = BASE / "PHASE5"
CSV_DIR = PHASE5_DIR / "tables"
FIG_DIR = PHASE5_DIR / "figures"
NETCDF_DIR = PHASE5_DIR / "netcdf"

# Phase 5: Evaluates climate resiliency, yield risk, and inter-model agreement from 1.25-degree ensemble.
GRID_TAG = "1p25deg"

MODELS = [
    "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
    "CNRM-CM6-1", "GFDL-CM4", "GFDL-ESM4",
]
HUB_HEIGHTS_M = [80.0, 100.0, 120.0]
SCENARIOS = ["ssp245", "ssp585"]
PERIODS = {
    "historical": "1950–2014 period mean",
    "ssp245": "2015–2100 period mean under SSP245",
    "ssp585": "2015–2100 period mean under SSP585",
}

# Analysis choices
MIN_VALID_MODELS_FOR_RANKING = 5
TOP_N_CELLS_PER_SCENARIO_HEIGHT = 100
# Prevent unstable ratio indices where historical baseline is near zero.
MIN_BASELINE_CF_FOR_INDEX = 0.01
MIN_BASELINE_AEP_FOR_INDEX = 0.01  # GWh/year per representative 3-MW turbine

for folder in (PHASE5_DIR, CSV_DIR, FIG_DIR, NETCDF_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# File / data helpers
def phase4_file(prefix, scenario):
    return PHASE4_DIR / f"{prefix}_ensemble_{scenario}_{GRID_TAG}.nc"


def read_required(path, variable):
    if not path.exists():
        raise FileNotFoundError(
            f"Required Phase 4 file not found:\n{path}\n\n"
            "Run Phase 4 first and ensure its output filename suffix matches "
            f"GRID_TAG={GRID_TAG!r}. Do not rename a 2-degree file to claim "
            "it is a 1.25-degree product."
        )
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise KeyError(f"Variable {variable!r} missing from {path}. Found: {list(ds.data_vars)}")
        out = ds.load()
    return out


def require_coordinates(ds, path):
    for coord in ("hub_height", "lat", "lon"):
        if coord not in ds.coords:
            raise ValueError(f"Coordinate {coord!r} missing from {path}")
    if not np.all(np.diff(ds.lat.values) > 0):
        raise ValueError(f"Latitude is not strictly increasing in {path}")
    if not np.all(np.diff(ds.lon.values) > 0):
        raise ValueError(f"Longitude is not strictly increasing in {path}")
    missing_heights = [h for h in HUB_HEIGHTS_M if not np.any(np.isclose(ds.hub_height.values, h))]
    if missing_heights:
        raise ValueError(f"Hub heights {missing_heights} missing in {path}")


def weighted_spatial_mean(da):
    """Cos(latitude)-weighted mean; valid for regular lon/lat grids."""
    weights = xr.DataArray(
        np.cos(np.deg2rad(da.lat.values)), dims=("lat",), coords={"lat": da.lat}
    ).clip(min=0)
    return da.weighted(weights).mean(dim=("lat", "lon"), skipna=True)


def write_netcdf(ds, filename):
    path = NETCDF_DIR / filename
    encoding = {}
    for name, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.integer):
            encoding[name] = {"zlib": True, "complevel": 4}
        elif np.issubdtype(var.dtype, np.floating):
            encoding[name] = {"zlib": True, "complevel": 4, "dtype": "float32"}
    ds.to_netcdf(path, encoding=encoding)
    print(f"Saved NetCDF: {path}")
    return path


def save_csv(df, filename):
    path = CSV_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved CSV: {path}")
    return path


def make_change(hist, future, name, baseline_threshold):
    absolute = future - hist
    percent = (absolute / hist) * 100.0
    percent = percent.where(np.abs(hist) >= baseline_threshold)
    return absolute.rename(f"{name}_absolute_change"), percent.rename(f"{name}_percentage_change")


def make_ratio_index(future, historical, name, threshold):
    """100 × future / historical; mask only where the baseline is non-positive.

    Positive low-baseline cells are retained to avoid large blank regions on the
    global map. They are flagged separately in the index NetCDF because ratios
    can be unstable when the historical baseline is very small.
    The threshold argument is retained so existing calls remain compatible.
    """
    denominator = historical.where(historical > 0)
    return (100.0 * future / denominator).rename(name)


def make_map(da, title, cbar_label, filename, cmap="viridis", vmin=None, vmax=None, diverging=False, note=None):
    """Simple lon/lat raster map; uses the coordinates in the source NetCDF."""
    arr = da.values
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    chosen_cmap = "RdBu_r" if diverging else cmap
    # Make any remaining mathematically undefined cells a visible neutral grey, rather than allowing transparent NaNs to lo...
    chosen_cmap = plt.get_cmap(chosen_cmap).copy()
    chosen_cmap.set_bad("#d9d9d9")
    mesh = ax.pcolormesh(
        da.lon.values, da.lat.values, arr,
        shading="auto", cmap=chosen_cmap, vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title(title)
    if note:
        ax.text(
            0.01, 0.015, note, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.5},
        )
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(cbar_label)
    ax.set_xlim(float(da.lon.min()), float(da.lon.max()))
    ax.set_ylim(float(da.lat.min()), float(da.lat.max()))
    out = FIG_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out}")

# Load and validate phase 4 ensembles
def load_ensemble(prefix, scenario, value_var):
    path = phase4_file(prefix, scenario)
    ds = read_required(path, f"{value_var}_ensemble_mean")
    require_coordinates(ds, path)
    return ds


def validate_ensemble(ds, variable, scenario):
    mean_name = f"{variable}_ensemble_mean"
    mean = ds[mean_name]
    issues = []
    finite = mean.values[np.isfinite(mean.values)]
    if finite.size == 0:
        issues.append("No finite values in ensemble mean")
    if variable == "capacity_factor" and finite.size:
        if finite.min() < -1e-5 or finite.max() > 1.00001:
            issues.append(f"CF outside expected [0,1] range: min={finite.min()}, max={finite.max()}")
    if "valid_model_count" in ds:
        counts = ds["valid_model_count"].values
        if counts.size and (np.nanmin(counts) < 0 or np.nanmax(counts) > len(MODELS)):
            issues.append("valid_model_count outside expected range")
    return {
        "scenario": scenario,
        "variable": variable,
        "lat_count": int(ds.sizes["lat"]),
        "lon_count": int(ds.sizes["lon"]),
        "lat_min": float(ds.lat.min()),
        "lat_max": float(ds.lat.max()),
        "lon_min": float(ds.lon.min()),
        "lon_max": float(ds.lon.max()),
        "finite_cell_count": int(np.isfinite(mean.values).sum()),
        "total_cell_count": int(mean.size),
        "mean_of_ensemble_mean": float(mean.mean(skipna=True)),
        "min_of_ensemble_mean": float(mean.min(skipna=True)),
        "max_of_ensemble_mean": float(mean.max(skipna=True)),
        "validation_status": "PASS" if not issues else "CHECK",
        "issues": "; ".join(issues),
    }

# Main analysis
def main():
    print("=" * 78)
    print("PHASE 5 — CLIMATE RESILIENCE AND FUTURE YIELD ANALYSIS")
    print(f"Phase 4 directory: {PHASE4_DIR}")
    print(f"Grid filename tag: {GRID_TAG}")
    print("=" * 78)

    validation_rows = []
    summaries = []
    top_cells_frames = []
    scenario_data = {"capacity_factor": {}, "aep_gwh_per_year": {}}

    # Load CF and AEP period means for historical + both future scenarios.
    for variable, prefix in [("capacity_factor", "CF"), ("aep_gwh_per_year", "AEP")]:
        for scenario in ["historical", *SCENARIOS]:
            ds = load_ensemble(prefix, scenario, variable)
            validation_rows.append(validate_ensemble(ds, variable, scenario))
            scenario_data[variable][scenario] = ds
            print(f"Loaded {prefix} {scenario}: {ds.sizes['lat']} lat × {ds.sizes['lon']} lon")

    # Confirm all six datasets use exactly the same coordinates before comparison.
    reference = scenario_data["capacity_factor"]["historical"]
    for variable in scenario_data:
        for scenario, ds in scenario_data[variable].items():
            for coord in ("hub_height", "lat", "lon"):
                if not np.array_equal(reference[coord].values, ds[coord].values):
                    raise ValueError(
                        f"Coordinate mismatch: {variable}/{scenario} differs from historical in {coord}. "
                        "Do not compare fields until Phase 4 is regenerated on one common grid."
                    )

    save_csv(pd.DataFrame(validation_rows), "PHASE5_input_validation.csv")

    # Spatial statistics, changes, CRI/FYRI, agreement files, and maps.
    for variable, prefix, index_name, threshold in [
        ("capacity_factor", "CF", "CRI", MIN_BASELINE_CF_FOR_INDEX),
        ("aep_gwh_per_year", "AEP", "FYRI", MIN_BASELINE_AEP_FOR_INDEX),
    ]:
        hist_ds = scenario_data[variable]["historical"]
        hist = hist_ds[f"{variable}_ensemble_mean"]

        # Report ensemble uncertainty and historical baseline summaries.
        for scenario in ["historical", *SCENARIOS]:
            ds = scenario_data[variable][scenario]
            mean = ds[f"{variable}_ensemble_mean"]
            median_name = f"{variable}_ensemble_median"
            std_name = f"{variable}_ensemble_std"
            min_name = f"{variable}_ensemble_min"
            max_name = f"{variable}_ensemble_max"
            for h in HUB_HEIGHTS_M:
                row = {
                    "variable": variable,
                    "scenario": scenario,
                    "period": PERIODS[scenario],
                    "hub_height_m": h,
                    "spatial_mean_ensemble_mean": float(weighted_spatial_mean(mean.sel(hub_height=h))),
                    "spatial_median_ensemble_mean": float(mean.sel(hub_height=h).median(skipna=True)),
                    "spatial_mean_ensemble_median": float(weighted_spatial_mean(ds[median_name].sel(hub_height=h))) if median_name in ds else np.nan,
                    "spatial_mean_model_std": float(weighted_spatial_mean(ds[std_name].sel(hub_height=h))) if std_name in ds else np.nan,
                    "minimum_ensemble_mean": float(mean.sel(hub_height=h).min(skipna=True)),
                    "maximum_ensemble_mean": float(mean.sel(hub_height=h).max(skipna=True)),
                }
                summaries.append(row)

        for scenario in SCENARIOS:
            future_ds = scenario_data[variable][scenario]
            future = future_ds[f"{variable}_ensemble_mean"]
            abs_change, pct_change = make_change(hist, future, variable, threshold)
            index = make_ratio_index(future, hist, index_name, threshold)

            change_ds = xr.Dataset({abs_change.name: abs_change, pct_change.name: pct_change})
            change_ds.attrs.update({
                "analysis": "future period mean minus historical period mean",
                "historical_period": "1950-2014",
                "future_period": "2015-2100",
                "scenario": scenario,
                "grid_tag": GRID_TAG,
                "note": "Period-mean comparison; not a year-2100 projection.",
            })
            write_netcdf(change_ds, f"{prefix}_phase5_change_{scenario}_minus_historical_{GRID_TAG}.nc")

            index_ds = index.to_dataset()
            # Flag low historical baselines so they are visible and auditable, even though positive low-baseline ratios are retaine...
            index_ds["low_baseline_mask"] = (hist < threshold).astype("int8")
            index_ds["low_baseline_mask"].attrs["long_name"] = "1 where historical baseline is below the usual ratio-stability threshold"
            index_ds["low_baseline_mask"].attrs["threshold"] = float(threshold)
            if variable == "capacity_factor":
                index_ds.attrs["index_definition"] = "CRI = 100 × future-period mean CF / historical-period mean CF"
                index_ds.attrs["interpretation"] = "100 = unchanged; above 100 = higher CF; below 100 = lower CF"
                index_ds.attrs["baseline_mask"] = "Only non-positive historical CF is masked (CRI undefined at zero baseline). Positive historical CF below the usual 0.01 threshold is retained and flagged by low_baseline_mask."
                index_ds.attrs["low_baseline_warning"] = "CRI can be unstable when historical CF is close to zero. Map color scale is capped at 200%; values above 200% display at the top color."
            else:
                index_ds.attrs["index_definition"] = "FYRI = 100 × future-period mean AEP / historical-period mean AEP"
                index_ds.attrs["interpretation"] = "100 = retained baseline yield; above 100 = higher annualized yield; below 100 = lower annualized yield"
                index_ds.attrs["baseline_mask"] = "Only non-positive historical AEP is masked (FYRI undefined at zero baseline). Positive historical AEP below the usual 0.01 GWh/year threshold is retained and flagged by low_baseline_mask."
                index_ds.attrs["low_baseline_warning"] = "FYRI can be unstable when historical AEP is close to zero. Map color scale is capped at 200%; values above 200% display at the top color."
            index_ds.attrs["grid_tag"] = GRID_TAG
            index_ds.attrs["note"] = "Operational definitions for this analysis; the project proposal names these indices but does not specify their equations. AEP is per representative 3-MW turbine per grid cell. Positive low-baseline cells are included in the plotted ratio and flagged; interpret them cautiously."
            write_netcdf(index_ds, f"{index_name}_{scenario}_{GRID_TAG}.nc")

            # Maps at each hub height.
            for h in HUB_HEIGHTS_M:
                hlabel = f"{int(h)}m"
                if variable == "capacity_factor":
                    make_map(hist.sel(hub_height=h), f"Historical mean CF — {hlabel}", "Capacity factor (0–1)", f"CF_historical_{hlabel}_{GRID_TAG}.png", vmin=0, vmax=1)
                    make_map(future.sel(hub_height=h), f"{scenario.upper()} mean CF — {hlabel}", "Capacity factor (0–1)", f"CF_{scenario}_{hlabel}_{GRID_TAG}.png", vmin=0, vmax=1)
                    make_map(abs_change.sel(hub_height=h), f"CF absolute change: {scenario.upper()} − historical — {hlabel}", "Absolute CF change", f"CF_change_{scenario}_{hlabel}_{GRID_TAG}.png", diverging=True)
                    make_map(
                        index.sel(hub_height=h),
                        f"Climate Resilience Index (CRI) — {scenario.upper()} — {hlabel}",
                        "CRI (%)", f"CRI_{scenario}_{hlabel}_{GRID_TAG}.png",
                        cmap="viridis", vmin=0, vmax=200,
                        note=f"Low-baseline cells included; interpret cautiously where historical CF < {threshold:g}. Color scale capped at 200%.",
                    )
                else:
                    make_map(hist.sel(hub_height=h), f"Historical annualized AEP — {hlabel}", "AEP (GWh/year per 3-MW turbine)", f"AEP_historical_{hlabel}_{GRID_TAG}.png")
                    make_map(future.sel(hub_height=h), f"{scenario.upper()} annualized AEP — {hlabel}", "AEP (GWh/year per 3-MW turbine)", f"AEP_{scenario}_{hlabel}_{GRID_TAG}.png")
                    make_map(abs_change.sel(hub_height=h), f"AEP absolute change: {scenario.upper()} − historical — {hlabel}", "AEP change (GWh/year)", f"AEP_change_{scenario}_{hlabel}_{GRID_TAG}.png", diverging=True)
                    make_map(
                        index.sel(hub_height=h),
                        f"Future Yield Retention Index (FYRI) — {scenario.upper()} — {hlabel}",
                        "FYRI (%)", f"FYRI_{scenario}_{hlabel}_{GRID_TAG}.png",
                        cmap="viridis", vmin=0, vmax=200,
                        note=f"Low-baseline cells included; interpret cautiously where historical AEP < {threshold:g} GWh/year. Color scale capped at 200%.",
                    )

            # Top grid cells by future CF, with yield and resilience context.
            cf_future = scenario_data["capacity_factor"][scenario]["capacity_factor_ensemble_mean"]
            aep_future = scenario_data["aep_gwh_per_year"][scenario]["aep_gwh_per_year_ensemble_mean"]
            cri_future = make_ratio_index(
                cf_future,
                scenario_data["capacity_factor"]["historical"]["capacity_factor_ensemble_mean"],
                "CRI", MIN_BASELINE_CF_FOR_INDEX
            )
            fyri_future = make_ratio_index(
                aep_future,
                scenario_data["aep_gwh_per_year"]["historical"]["aep_gwh_per_year_ensemble_mean"],
                "FYRI", MIN_BASELINE_AEP_FOR_INDEX
            )
            valid_count_name = "valid_model_count"
            valid_count = future_ds[valid_count_name] if valid_count_name in future_ds else xr.full_like(future, len(MODELS))
            for h in HUB_HEIGHTS_M:
                frame = pd.DataFrame({
                    "lat": cf_future.sel(hub_height=h).lat.values.repeat(cf_future.sizes["lon"]),
                    "lon": np.tile(cf_future.lon.values, cf_future.sizes["lat"]),
                    "future_capacity_factor": cf_future.sel(hub_height=h).values.ravel(),
                    "future_aep_gwh_per_year_per_3mw_turbine": aep_future.sel(hub_height=h).values.ravel(),
                    "cri_percent": cri_future.sel(hub_height=h).values.ravel(),
                    "fyri_percent": fyri_future.sel(hub_height=h).values.ravel(),
                    "valid_model_count": valid_count.sel(hub_height=h).values.ravel(),
                })
                frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["future_capacity_factor", "future_aep_gwh_per_year_per_3mw_turbine"])
                frame = frame[frame["valid_model_count"] >= MIN_VALID_MODELS_FOR_RANKING]
                frame = frame.sort_values(
                    ["future_capacity_factor", "future_aep_gwh_per_year_per_3mw_turbine"],
                    ascending=[False, False]
                ).head(TOP_N_CELLS_PER_SCENARIO_HEIGHT)
                frame.insert(0, "rank", np.arange(1, len(frame) + 1))
                frame.insert(0, "hub_height_m", h)
                frame.insert(0, "scenario", scenario)
                frame["interpretation_note"] = "Candidate high-CF grid cell for screening only; not a siting recommendation. Check land, bathymetry, protected areas, grid access, terrain, and local wind observations separately."
                top_cells_frames.append(frame)

    # Save summary tables.
    summary_df = pd.DataFrame(summaries)
    save_csv(summary_df, "PHASE5_spatial_summary_by_scenario_height.csv")
    if top_cells_frames:
        save_csv(pd.concat(top_cells_frames, ignore_index=True), "PHASE5_top_future_grid_cells.csv")

    # Existing Phase 4 model-agreement files are summarized if present.
    agreement_rows = []
    for prefix, variable in [("CF", "capacity_factor"), ("AEP", "aep_gwh_per_year")]:
        for scenario in SCENARIOS:
            path = PHASE4_DIR / f"{prefix}_model_agreement_{scenario}_{GRID_TAG}.nc"
            if not path.exists():
                warnings.warn(f"Model agreement file missing; skipping: {path}")
                continue
            with xr.open_dataset(path) as ds:
                ds = ds.load()
            for h in HUB_HEIGHTS_M:
                row = {"variable": variable, "scenario": scenario, "hub_height_m": h}
                for field in ("increase_agreement_fraction", "decrease_agreement_fraction", "models_increasing", "models_decreasing", "valid_model_count"):
                    if field in ds:
                        arr = ds[field].sel(hub_height=h)
                        row[f"spatial_mean_{field}"] = float(weighted_spatial_mean(arr)) if "fraction" in field else float(weighted_spatial_mean(arr))
                agreement_rows.append(row)
    if agreement_rows:
        save_csv(pd.DataFrame(agreement_rows), "PHASE5_model_agreement_summary.csv")

    # Machine-readable methods and run manifest.
    manifest = {
        "phase": 5,
        "grid_tag": GRID_TAG,
        "phase4_input_directory": str(PHASE4_DIR),
        "phase5_output_directory": str(PHASE5_DIR),
        "models_expected": MODELS,
        "hub_heights_m": HUB_HEIGHTS_M,
        "historical_period": "1950-2014 period mean",
        "future_period": "2015-2100 period mean",
        "cri_definition": "100 * future period-mean capacity factor / historical period-mean capacity factor; masked only where historical CF <= 0; positive low-baseline cells are retained and flagged in low_baseline_mask",
        "fyri_definition": "100 * future period-mean AEP / historical period-mean AEP; masked only where historical AEP <= 0; positive low-baseline cells are retained and flagged in low_baseline_mask",
        "ratio_caution": "Ratios can be unstable for low positive historical baselines. Displayed map scale is 0–200%; values above 200% are color-capped.",
        "aep_note": "Annualized energy yield of one representative 3-MW turbine at each grid cell; not total regional energy.",
        "seasonal_analysis": "Not computed: Phase 4 period-mean spatial fields do not retain monthly/seasonal fields.",
        "year_2100_maps": "Not computed: future inputs represent the 2015-2100 period mean, not year 2100 alone.",
    }
    with open(PHASE5_DIR / "PHASE5_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 78)
    print("PHASE 5 COMPLETE")
    print(f"Tables : {CSV_DIR}")
    print(f"Figures: {FIG_DIR}")
    print(f"NetCDF : {NETCDF_DIR}")
    print(f"Manifest: {PHASE5_DIR / 'PHASE5_run_manifest.json'}")
    print("Remember: CRI/FYRI are explicitly defined operational indices; seasonal and year-2100 outputs need additional time-resolved data.")
    print("=" * 78)


if __name__ == "__main__":
    main()
