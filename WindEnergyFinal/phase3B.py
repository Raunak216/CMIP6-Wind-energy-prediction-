from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd
import gc
import traceback
import time


# Configuration

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

# Wind shear

REFERENCE_HEIGHT_M = 10.0
HUB_HEIGHTS_M = [80.0, 100.0, 120.0]

ALPHA = 1.0 / 7.0


# Representative 3 MW turbine

RATED_POWER_MW = 3.0

CUT_IN_MS = 4.0
RATED_SPEED_MS = 15.0
CUT_OUT_MS = 25.0


# Memory control One year at a time is intentionally conservative.

CHUNK_YEARS = 1


# Output directories

OUTPUT_DIR = BASE / "PHASE3B" / "native_spatial"
SUMMARY_DIR = BASE / "PHASE3B" / "summaries"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


# Power curve

def turbine_power_curve(wind_ms):
    """
    Representative 3 MW turbine power curve.

    P = 0 below cut-in
    Cubic normalized ramp between cut-in and rated
    P = rated power between rated speed and cut-out
    P = 0 above cut-out

    Input
    -----
    wind_ms : numpy array

    Output
    ------
    power : numpy array, MW
    """

    wind_ms = np.asarray(wind_ms, dtype=np.float32)

    power = np.zeros_like(wind_ms, dtype=np.float32)

    # Cubic region

    ramp = (
        (wind_ms >= CUT_IN_MS)
        & (wind_ms < RATED_SPEED_MS)
    )

    numerator = (
        wind_ms[ramp] ** 3
        - CUT_IN_MS ** 3
    )

    denominator = (
        RATED_SPEED_MS ** 3
        - CUT_IN_MS ** 3
    )

    power[ramp] = (
        RATED_POWER_MW
        * numerator
        / denominator
    )

    # Rated region

    rated = (
        (wind_ms >= RATED_SPEED_MS)
        & (wind_ms <= CUT_OUT_MS)
    )

    power[rated] = RATED_POWER_MW

    return power


# Input file

def get_input_file(model, experiment):

    return (
        BASE
        / model
        / "day"
        / "processed"
        / f"{model}_{experiment}_daily_wind_speed.nc"
    )


# Find wind variable

def find_wind_variable(ds):

    preferred = [
        "wind_speed",
        "wind_speed_10m",
        "ws10",
        "ws",
    ]

    for name in preferred:

        if name in ds.data_vars:
            return name

    # Fallback
    candidates = []

    for name, da in ds.data_vars.items():

        if (
            "time" in da.dims
            and "lat" in da.dims
            and "lon" in da.dims
        ):
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]

    raise ValueError(
        f"Could not uniquely identify wind-speed variable. "
        f"Available variables: {list(ds.data_vars)}"
    )


# Process one model / experiment

def process_dataset(model, experiment):

    start_year, end_year = PERIODS[experiment]

    input_file = get_input_file(
        model,
        experiment
    )

    print()
    print("=" * 90)
    print(f"MODEL      : {model}")
    print(f"EXPERIMENT : {experiment}")
    print(f"PERIOD     : {start_year}-{end_year}")
    print(f"INPUT      : {input_file}")
    print("=" * 90)

    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{input_file}"
        )

    # CFDatetimeCoder avoids the deprecated use_cftime=True argument.

    time_coder = xr.coders.CFDatetimeCoder(
        use_cftime=True
    )

    ds = xr.open_dataset(
        input_file,
        decode_times=time_coder
    )

    try:

        wind_var = find_wind_variable(ds)

        wind = ds[wind_var]

        print(f"Wind variable : {wind_var}")
        print(f"Dimensions    : {wind.dims}")
        print(f"Shape         : {wind.shape}")

        if "lat" not in wind.dims:
            raise ValueError("Latitude dimension not found.")

        if "lon" not in wind.dims:
            raise ValueError("Longitude dimension not found.")

        if "time" not in wind.dims:
            raise ValueError("Time dimension not found.")

        # Restrict explicitly to requested period.

        years = wind["time"].dt.year

        wind = wind.where(
            (years >= start_year)
            & (years <= end_year),
            drop=True
        )

        if wind.sizes["time"] == 0:

            raise ValueError(
                f"No time steps found for "
                f"{start_year}-{end_year}"
            )

        actual_start = int(
            wind["time"].dt.year.min().values
        )

        actual_end = int(
            wind["time"].dt.year.max().values
        )

        print(
            f"Actual years  : "
            f"{actual_start}-{actual_end}"
        )

        print(
            f"Time steps    : "
            f"{wind.sizes['time']:,}"
        )

        lat_values = wind["lat"].values
        lon_values = wind["lon"].values

        nlat = len(lat_values)
        nlon = len(lon_values)

        print(
            f"Native grid   : "
            f"{nlat} × {nlon} "
            f"({nlat*nlon:,} cells)"
        )

        # OUTPUT ARRAYS Shape: [hub_height, lat, lon] These are tiny compared with daily arrays.

        shape = (
            len(HUB_HEIGHTS_M),
            nlat,
            nlon
        )

        # Float64 accumulators improve numerical stability.

        wind_sum = np.zeros(
            shape,
            dtype=np.float64
        )

        power_sum = np.zeros(
            shape,
            dtype=np.float64
        )

        valid_count = np.zeros(
            shape,
            dtype=np.int64
        )

        # Year chunks

        for chunk_start in range(
            start_year,
            end_year + 1,
            CHUNK_YEARS
        ):

            chunk_end = min(
                chunk_start + CHUNK_YEARS - 1,
                end_year
            )

            print(
                f"  Processing "
                f"{chunk_start}-{chunk_end} ..."
            )

            chunk = wind.where(
                (
                    wind["time"].dt.year
                    >= chunk_start
                )
                &
                (
                    wind["time"].dt.year
                    <= chunk_end
                ),
                drop=True
            )

            if chunk.sizes["time"] == 0:

                print(
                    "    No data in chunk - skipping."
                )

                continue

            # IMPORTANT: Only this one-year chunk becomes NumPy.

            v10 = chunk.values.astype(
                np.float32,
                copy=False
            )

            # Expected: time × lat × lon

            if v10.ndim != 3:

                raise ValueError(
                    f"Expected 3-D wind array, "
                    f"got shape {v10.shape}"
                )

            # Process hub heights ONE AT A TIME.

            for h_index, hub_height in enumerate(
                HUB_HEIGHTS_M
            ):

                shear_factor = (
                    hub_height
                    / REFERENCE_HEIGHT_M
                ) ** ALPHA

                # Hub-height wind

                vhub = (
                    v10 * shear_factor
                ).astype(
                    np.float32,
                    copy=False
                )

                valid = np.isfinite(vhub)

                # Power

                power = turbine_power_curve(
                    vhub
                )

                # Ensure missing wind remains missing.
                power[~valid] = np.nan

                # Spatial accumulators Sum over TIME ONLY.

                wind_sum[h_index] += (
                    np.nansum(
                        vhub,
                        axis=0,
                        dtype=np.float64
                    )
                )

                power_sum[h_index] += (
                    np.nansum(
                        power,
                        axis=0,
                        dtype=np.float64
                    )
                )

                valid_count[h_index] += (
                    np.sum(
                        valid,
                        axis=0,
                        dtype=np.int64
                    )
                )

                del vhub
                del valid
                del power

                gc.collect()

            del v10
            del chunk

            gc.collect()

        # Final statistics

        print()
        print("  Calculating spatial statistics...")

        with np.errstate(
            divide="ignore",
            invalid="ignore"
        ):

            mean_wind = (
                wind_sum
                / valid_count
            )

            mean_power = (
                power_sum
                / valid_count
            )

        # Cells with zero valid observations -> NaN

        mean_wind[
            valid_count == 0
        ] = np.nan

        mean_power[
            valid_count == 0
        ] = np.nan

        # Capacity factor CF = mean power / rated power

        capacity_factor = (
            mean_power
            / RATED_POWER_MW
        )

        # Annualized energy yield MW × 8760 h / 1000 = GWh/year This represents one 3-MW turbine located at each grid cell.

        aep_gwh_per_year = (
            mean_power
            * 8760.0
            / 1000.0
        )

        # Build output dataset

        output = xr.Dataset(

            data_vars={

                "mean_wind_speed_ms": (
                    (
                        "hub_height",
                        "lat",
                        "lon"
                    ),
                    mean_wind.astype(
                        np.float32
                    )
                ),

                "mean_power_mw": (
                    (
                        "hub_height",
                        "lat",
                        "lon"
                    ),
                    mean_power.astype(
                        np.float32
                    )
                ),

                "capacity_factor": (
                    (
                        "hub_height",
                        "lat",
                        "lon"
                    ),
                    capacity_factor.astype(
                        np.float32
                    )
                ),

                "aep_gwh_per_year": (
                    (
                        "hub_height",
                        "lat",
                        "lon"
                    ),
                    aep_gwh_per_year.astype(
                        np.float32
                    )
                ),

                "valid_day_count": (
                    (
                        "hub_height",
                        "lat",
                        "lon"
                    ),
                    valid_count.astype(
                        np.int32
                    )
                ),
            },

            coords={

                "hub_height": np.array(
                    HUB_HEIGHTS_M,
                    dtype=np.float32
                ),

                "lat": lat_values,

                "lon": lon_values,
            }
        )

        # Attributes

        output.attrs.update({

            "title":
                "CMIP6 spatial wind-power assessment",

            "model":
                model,

            "experiment":
                experiment,

            "analysis_start_year":
                int(start_year),

            "analysis_end_year":
                int(end_year),

            "reference_height_m":
                float(REFERENCE_HEIGHT_M),

            "wind_shear_alpha":
                float(ALPHA),

            "rated_power_mw":
                float(RATED_POWER_MW),

            "cut_in_wind_speed_ms":
                float(CUT_IN_MS),

            "rated_wind_speed_ms":
                float(RATED_SPEED_MS),

            "cut_out_wind_speed_ms":
                float(CUT_OUT_MS),

            "power_curve":
                (
                    "Cubic normalized power ramp "
                    "between cut-in and rated speed; "
                    "rated output from rated speed "
                    "through cut-out."
                ),

            "spatial_processing":
                (
                    "Power conversion performed on "
                    "the model native grid before "
                    "multi-model regridding."
                ),

            "aep_definition":
                (
                    "Annualized energy yield of one "
                    "representative 3-MW turbine at "
                    "each grid cell, calculated from "
                    "period-mean power using 8760 h/year."
                ),
        })


        # Variable metadata

        output["hub_height"].attrs = {
            "long_name": "Turbine hub height",
            "units": "m"
        }

        output["lat"].attrs.update({
            "standard_name": "latitude",
            "units": "degrees_north"
        })

        output["lon"].attrs.update({
            "standard_name": "longitude",
            "units": "degrees_east"
        })

        output[
            "mean_wind_speed_ms"
        ].attrs = {

            "long_name":
                "Period mean hub-height wind speed",

            "units":
                "m s-1"
        }

        output[
            "mean_power_mw"
        ].attrs = {

            "long_name":
                "Period mean turbine power",

            "units":
                "MW"
        }

        output[
            "capacity_factor"
        ].attrs = {

            "long_name":
                "Period mean turbine capacity factor",

            "units":
                "1",

            "definition":
                "mean_power_mw / rated_power_mw"
        }

        output[
            "aep_gwh_per_year"
        ].attrs = {

            "long_name":
                "Annualized turbine energy yield",

            "units":
                "GWh year-1",

            "definition":
                (
                    "mean_power_mw * "
                    "8760 hours / 1000"
                )
        }

        output[
            "valid_day_count"
        ].attrs = {

            "long_name":
                "Number of valid daily wind observations",

            "units":
                "days"
        }


        # Save

        output_file = (
            OUTPUT_DIR
            / f"{model}_{experiment}_spatial_cf_aep.nc"
        )

        encoding = {

            "mean_wind_speed_ms": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32"
            },

            "mean_power_mw": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32"
            },

            "capacity_factor": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32"
            },

            "aep_gwh_per_year": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32"
            },

            "valid_day_count": {
                "zlib": True,
                "complevel": 4,
                "dtype": "int32"
            },
        }

        output.to_netcdf(
            output_file,
            encoding=encoding
        )

        print()
        print(
            f"  Saved: {output_file}"
        )

        # Sanity statistics

        cf_min = float(
            np.nanmin(capacity_factor)
        )

        cf_max = float(
            np.nanmax(capacity_factor)
        )

        cf_mean = float(
            np.nanmean(capacity_factor)
        )

        power_mean_global = float(
            np.nanmean(mean_power)
        )

        aep_mean_global = float(
            np.nanmean(aep_gwh_per_year)
        )

        print(
            f"  CF range : "
            f"{cf_min:.4f} → {cf_max:.4f}"
        )

        print(
            f"  CF mean  : "
            f"{cf_mean:.4f}"
        )

        print(
            f"  Mean P   : "
            f"{power_mean_global:.4f} MW"
        )

        print(
            f"  Mean AEP : "
            f"{aep_mean_global:.4f} GWh/year"
        )

        # Summary record

        result = {

            "model":
                model,

            "experiment":
                experiment,

            "start_year":
                start_year,

            "end_year":
                end_year,

            "nlat":
                nlat,

            "nlon":
                nlon,

            "grid_cells":
                nlat * nlon,

            "cf_min":
                cf_min,

            "cf_max":
                cf_max,

            "cf_spatial_mean":
                cf_mean,

            "mean_power_spatial_mean_mw":
                power_mean_global,

            "aep_spatial_mean_gwh_per_year":
                aep_mean_global,

            "output_file":
                str(output_file),

            "status":
                "SUCCESS"
        }

        # Cleanup

        output.close()

        del output
        del mean_wind
        del mean_power
        del capacity_factor
        del aep_gwh_per_year

        del wind_sum
        del power_sum
        del valid_count

        gc.collect()

        return result

    finally:

        ds.close()

        gc.collect()


# Main

def main():

    print()
    print("#" * 90)
    print("PHASE 3B")
    print("NATIVE-GRID SPATIAL CAPACITY FACTOR AND AEP")
    print("#" * 90)

    print()
    print(
        f"Models      : {len(MODELS)}"
    )

    print(
        f"Experiments : {len(EXPERIMENTS)}"
    )

    print(
        f"Datasets    : "
        f"{len(MODELS) * len(EXPERIMENTS)}"
    )

    print(
        f"Hub heights : {HUB_HEIGHTS_M}"
    )

    print(
        f"Alpha       : {ALPHA:.6f}"
    )

    print(
        f"Turbine     : {RATED_POWER_MW} MW"
    )

    print(
        f"Power curve : "
        f"{CUT_IN_MS}/{RATED_SPEED_MS}/{CUT_OUT_MS} m/s"
    )

    print()

    results = []
    failures = []

    total_start = time.time()

    # Loop

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
                print("!" * 90)
                print(
                    f"FAILED: "
                    f"{model} / {experiment}"
                )
                print(str(e))
                print("!" * 90)

                traceback.print_exc()

                failures.append({

                    "model":
                        model,

                    "experiment":
                        experiment,

                    "error":
                        str(e)
                })

            gc.collect()


    # Save master summary

    summary_df = pd.DataFrame(
        results
    )

    summary_file = (
        SUMMARY_DIR
        / "PHASE3B_native_spatial_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )


    # Save failures

    failure_df = pd.DataFrame(
        failures,
        columns=[
            "model",
            "experiment",
            "error"
        ]
    )

    failure_file = (
        SUMMARY_DIR
        / "PHASE3B_failures.csv"
    )

    failure_df.to_csv(
        failure_file,
        index=False
    )


    # Validation

    expected = (
        len(MODELS)
        * len(EXPERIMENTS)
    )

    successful = len(results)
    failed = len(failures)

    elapsed_minutes = (
        time.time() - total_start
    ) / 60.0


    print()
    print("#" * 90)
    print("PHASE 3B COMPLETE")
    print("#" * 90)

    print(
        f"Expected datasets : {expected}"
    )

    print(
        f"Successful        : {successful}"
    )

    print(
        f"Failed            : {failed}"
    )

    print(
        f"Elapsed time      : "
        f"{elapsed_minutes:.2f} minutes"
    )

    print()

    print(
        f"Summary file:"
    )

    print(
        summary_file
    )

    print()

    print(
        f"Failure file:"
    )

    print(
        failure_file
    )

    print()

    if successful == expected:

        print(
            "VALIDATION PASSED: "
            "all 21 model/experiment datasets completed."
        )

    else:

        print(
            "VALIDATION WARNING: "
            "not all datasets completed."
        )

    print()

    if successful > 0:

        print(
            "Overall CF range:"
        )

        print(
            f"{summary_df['cf_min'].min():.6f}"
            f" → "
            f"{summary_df['cf_max'].max():.6f}"
        )

    print()


# RUN

if __name__ == "__main__":

    main()