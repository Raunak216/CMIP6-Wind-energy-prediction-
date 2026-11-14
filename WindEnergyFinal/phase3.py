from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import traceback
import gc


# Phase 3: Wind speed to turbine power, capacity factor (CF), and annual energy production (AEP).


# Configuration

BASE = Path(r"D:\CMIP6")

PHASE3_DIR = BASE / "PHASE3"
SUMMARY_DIR = PHASE3_DIR / "summaries"

SUMMARY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Models

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


# Hub height

REFERENCE_HEIGHT = 10.0

HUB_HEIGHTS = [
    80.0,
    100.0,
    120.0,
]

ALPHA = 1.0 / 7.0


# Turbine

# Representative Vestas V90-based 3 MW turbine
RATED_POWER_MW = 3.0

CUT_IN_MS = 4.0
RATED_SPEED_MS = 15.0
CUT_OUT_MS = 25.0


# Memory control and garbage collection

CHUNK_YEARS = 3


# Time decoder

time_coder = xr.coders.CFDatetimeCoder(
    use_cftime=True
)


# Header

print()
print("=" * 90)
print("PHASE 3 — TURBINE POWER / CAPACITY FACTOR / AEP")
print("=" * 90)
print()

print(f"Reference height : {REFERENCE_HEIGHT} m")
print(f"Hub heights      : {HUB_HEIGHTS}")
print(f"Alpha            : {ALPHA:.6f} (1/7)")
print()
print("TURBINE")
print(f"Rated power      : {RATED_POWER_MW} MW")
print(f"Cut-in speed     : {CUT_IN_MS} m/s")
print(f"Rated speed      : {RATED_SPEED_MS} m/s")
print(f"Cut-out speed    : {CUT_OUT_MS} m/s")
print()
print(f"Chunk size       : {CHUNK_YEARS} years")
print(f"Datasets         : {len(MODELS)} × {len(EXPERIMENTS)} = "
      f"{len(MODELS) * len(EXPERIMENTS)}")
print()


# Find phase 1 file

def find_phase1_file(model, experiment):

    model_dir = BASE / model

    possible_dirs = [
        model_dir / "day" / "processed",
        model_dir / "Day" / "processed",
    ]

    expected_name = (
        f"{model}_{experiment}_daily_wind_speed.nc"
    )

    # First try exact expected locations
    for directory in possible_dirs:

        path = directory / expected_name

        if path.exists():
            return path

    # Fallback
    candidates = list(
        model_dir.rglob(expected_name)
    )

    if candidates:
        return candidates[0]

    return None


# Find wind variable

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

    candidates = [
        name
        for name in ds.data_vars
        if "wind" in name.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]

    raise ValueError(
        "Unable to identify wind-speed variable. "
        f"Variables: {list(ds.data_vars)}"
    )


# Turbine power curve

def turbine_power_mw(wind_speed):
    """
    Vectorized representative turbine power curve.

    0 below cut-in
    Cubic ramp from cut-in to rated
    Rated power from rated to cut-out
    0 above cut-out
    """

    # Start with zero power

    power = np.zeros_like(
        wind_speed,
        dtype=np.float32
    )

    # Cubic region

    ramp = (
        (wind_speed >= CUT_IN_MS)
        &
        (wind_speed < RATED_SPEED_MS)
    )

    power[ramp] = (
        RATED_POWER_MW
        *
        (
            (
                wind_speed[ramp] ** 3
                -
                CUT_IN_MS ** 3
            )
            /
            (
                RATED_SPEED_MS ** 3
                -
                CUT_IN_MS ** 3
            )
        )
    )

    # Rated region

    rated = (
        (wind_speed >= RATED_SPEED_MS)
        &
        (wind_speed <= CUT_OUT_MS)
    )

    power[rated] = RATED_POWER_MW

    # Above cut-out remains zero

    return power


# Process one time chunk

def process_chunk(
    ws,
    chunk_start_year,
    chunk_end_year,
    model,
    experiment,
    hub_height,
):

    chunk = ws.sel(
        time=slice(
            f"{chunk_start_year}-01-01",
            f"{chunk_end_year}-12-31"
        )
    )

    if chunk.sizes.get("time", 0) == 0:
        return None

    # Hub height

    factor = (
        hub_height / REFERENCE_HEIGHT
    ) ** ALPHA

    hub_ws = chunk * factor

    # CRITICAL MEMORY STEP Only NOW convert this small time chunk to NumPy.

    wind = hub_ws.values

    # Calculate turbine power

    power = turbine_power_mw(
        wind
    )

    # Clean the temporary wind array as soon as possible

    del wind

    # Dimensions Expected: time × lat × lon But we determine spatial size generically.

    if power.ndim == 1:

        spatial_axes = ()

    else:

        spatial_axes = tuple(
            range(1, power.ndim)
        )

    # Daily spatial mean power IMPORTANT: We reduce spatial dimensions immediately.

    if spatial_axes:

        daily_power = np.nanmean(
            power,
            axis=spatial_axes
        )

    else:

        daily_power = power

    # Daily spatial mean wind

    if spatial_axes:

        # hub_ws is still an xarray object.
        wind_chunk = hub_ws.values

        daily_wind = np.nanmean(
            wind_chunk,
            axis=spatial_axes
        )

        del wind_chunk

    else:

        daily_wind = hub_ws.values

    # Daily spatial CF

    daily_cf = (
        daily_power /
        RATED_POWER_MW
    )

    # Daily energy CMIP6 daily values represent daily mean wind speed.

    daily_energy_mwh = (
        daily_power * 24.0
    )

    # Extract time

    times = chunk.time.values

    years = np.asarray(
        [
            int(t.year)
            for t in times
        ]
    )

    # Annual accumulation

    annual_records = []

    for year in np.unique(years):

        mask = (
            years == year
        )

        year_power = daily_power[
            mask
        ]

        year_wind = daily_wind[
            mask
        ]

        year_cf = daily_cf[
            mask
        ]

        year_energy = daily_energy_mwh[
            mask
        ]

        # Number of valid days

        valid_power = np.isfinite(
            year_power
        )

        valid_energy = np.isfinite(
            year_energy
        )

        valid_wind = np.isfinite(
            year_wind
        )

        power_valid_values = (
            year_power[
                valid_power
            ]
        )

        energy_valid_values = (
            year_energy[
                valid_energy
            ]
        )

        wind_valid_values = (
            year_wind[
                valid_wind
            ]
        )

        # Annual statistics

        if len(power_valid_values) == 0:
            continue

        mean_power = float(
            np.mean(
                power_valid_values
            )
        )

        mean_wind = float(
            np.mean(
                wind_valid_values
            )
        )

        annual_energy = float(
            np.sum(
                energy_valid_values
            ) / 1000.0
        )
        # MWh → GWh

        # Capacity factor Use actual number of valid days.

        valid_days = len(
            energy_valid_values
        )

        theoretical_energy_gwh = (
            RATED_POWER_MW
            *
            valid_days
            *
            24.0
            /
            1000.0
        )

        if theoretical_energy_gwh > 0:

            capacity_factor = (
                annual_energy
                /
                theoretical_energy_gwh
            )

        else:

            capacity_factor = np.nan

        annual_records.append({

            "model": model,

            "experiment": experiment,

            "year": int(year),

            "hub_height_m": hub_height,

            "alpha": ALPHA,

            "turbine_rated_power_mw":
                RATED_POWER_MW,

            "mean_wind_speed_ms":
                mean_wind,

            "mean_power_mw":
                mean_power,

            "annual_energy_gwh":
                annual_energy,

            "capacity_factor":
                capacity_factor,

            "valid_days":
                valid_days,

        })

    # Delete large arrays immediately

    del power
    del daily_power
    del daily_cf
    del daily_energy_mwh
    del daily_wind
    del hub_ws
    del chunk

    gc.collect()

    return annual_records


# Process one dataset

def process_dataset(
    model,
    experiment
):

    input_file = find_phase1_file(
        model,
        experiment
    )

    if input_file is None:

        raise FileNotFoundError(
            f"Phase 1 file not found: "
            f"{model} | {experiment}"
        )

    print()
    print("-" * 90)

    print(
        f"Processing: "
        f"{model} | {experiment}"
    )

    print(
        f"Input: "
        f"{input_file}"
    )

    start_year, end_year = (
        PERIODS[experiment]
    )

    # Open lazily

    ds = xr.open_dataset(
        input_file,
        decode_times=time_coder
    )

    try:

        wind_var = find_wind_variable(
            ds
        )

        print(
            f"Wind variable: "
            f"{wind_var}"
        )

        ws = ds[wind_var]

        # Actual coverage

        actual_start = int(
            ws.time.dt.year.min().item()
        )

        actual_end = int(
            ws.time.dt.year.max().item()
        )

        print(
            f"Available time: "
            f"{actual_start} → {actual_end}"
        )

        # Select requested period

        ws = ws.sel(
            time=slice(
                f"{start_year}-01-01",
                f"{end_year}-12-31"
            )
        )

        if ws.sizes.get("time", 0) == 0:

            raise ValueError(
                "No data available for requested period."
            )

        print(
            f"Processing period: "
            f"{start_year} → {end_year}"
        )

        # Check dimensions

        print(
            f"Dimensions: "
            f"{dict(ws.sizes)}"
        )

        # Hub heights

        all_records = []

        for hub_height in HUB_HEIGHTS:

            factor = (
                hub_height / REFERENCE_HEIGHT
            ) ** ALPHA

            print()
            print(
                f"Hub height: "
                f"{hub_height:.0f} m"
            )

            print(
                f"Wind multiplier: "
                f"{factor:.6f}"
            )

            # Process time chunks

            dataset_records = []

            current_year = start_year

            while current_year <= end_year:

                chunk_end = min(
                    current_year
                    + CHUNK_YEARS
                    - 1,
                    end_year
                )

                print(
                    f"  Chunk: "
                    f"{current_year} → "
                    f"{chunk_end}"
                )

                records = process_chunk(
                    ws=ws,
                    chunk_start_year=current_year,
                    chunk_end_year=chunk_end,
                    model=model,
                    experiment=experiment,
                    hub_height=hub_height,
                )

                if records:

                    dataset_records.extend(
                        records
                    )

                current_year = (
                    chunk_end + 1
                )

                # Explicit cleanup
                gc.collect()

            all_records.extend(
                dataset_records
            )

            print(
                f"  ✓ "
                f"{len(dataset_records)} annual "
                f"records generated"
            )

        # Save annual data

        df = pd.DataFrame(
            all_records
        )

        if df.empty:

            raise ValueError(
                "No annual records were produced."
            )

        df = (
            df
            .sort_values(
                [
                    "hub_height_m",
                    "year",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        output_file = (
            SUMMARY_DIR /
            f"PHASE3_{model}_{experiment}_annual.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print()
        print(
            f"✓ Saved: "
            f"{output_file}"
        )

        print(
            f"  Rows: "
            f"{len(df)}"
        )

        print(
            f"  Expected: "
            f"{(
                end_year - start_year + 1
            ) * len(HUB_HEIGHTS)}"
        )

        # Small period summary

        summary_records = []

        for hub_height in HUB_HEIGHTS:

            sub = df[
                df["hub_height_m"]
                == hub_height
            ]

            if sub.empty:
                continue

            summary_records.append({

                "model": model,

                "experiment": experiment,

                "hub_height_m":
                    hub_height,

                "alpha": ALPHA,

                "turbine_rated_power_mw":
                    RATED_POWER_MW,

                "period_start":
                    start_year,

                "period_end":
                    end_year,

                "mean_wind_speed_ms":
                    sub[
                        "mean_wind_speed_ms"
                    ].mean(),

                "mean_power_mw":
                    sub[
                        "mean_power_mw"
                    ].mean(),

                "mean_annual_energy_gwh":
                    sub[
                        "annual_energy_gwh"
                    ].mean(),

                "mean_capacity_factor":
                    sub[
                        "capacity_factor"
                    ].mean(),

                "min_capacity_factor":
                    sub[
                        "capacity_factor"
                    ].min(),

                "max_capacity_factor":
                    sub[
                        "capacity_factor"
                    ].max(),

            })

        summary_df = pd.DataFrame(
            summary_records
        )

        summary_file = (
            SUMMARY_DIR /
            f"PHASE3_{model}_{experiment}_summary.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False
        )

        print(
            f"✓ Saved summary: "
            f"{summary_file}"
        )

        print(
            "✓ DATASET SUCCESS"
        )

        return {
            "model": model,
            "experiment": experiment,
            "status": "SUCCESS",
            "rows": len(df),
        }

    finally:

        ds.close()

        gc.collect()


# Process all 21 datasets

results = []


for model in MODELS:

    for experiment in EXPERIMENTS:

        try:

            result = process_dataset(
                model,
                experiment
            )

            results.append(
                result
            )

        except Exception as e:

            print()
            print(
                f"❌ FAILED: "
                f"{model} | {experiment}"
            )

            print(
                f"Error: {e}"
            )

            traceback.print_exc()

            results.append({

                "model": model,

                "experiment":
                    experiment,

                "status":
                    "FAILED",

                "rows": 0,

            })

            # Make sure failed dataset memory is released
            gc.collect()


# Combine master files

print()
print("=" * 90)
print("COMBINING MASTER PHASE 3 OUTPUTS")
print("=" * 90)
print()


annual_files = []

summary_files = []


for model in MODELS:

    for experiment in EXPERIMENTS:

        annual_file = (
            SUMMARY_DIR /
            f"PHASE3_{model}_{experiment}_annual.csv"
        )

        summary_file = (
            SUMMARY_DIR /
            f"PHASE3_{model}_{experiment}_summary.csv"
        )

        if annual_file.exists():

            annual_files.append(
                annual_file
            )

        if summary_file.exists():

            summary_files.append(
                summary_file
            )


# Master annual

if annual_files:

    annual_master = pd.concat(
        [
            pd.read_csv(
                file
            )
            for file in annual_files
        ],
        ignore_index=True
    )

    annual_master = (
        annual_master
        .sort_values(
            [
                "model",
                "experiment",
                "hub_height_m",
                "year",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    annual_master_file = (
        SUMMARY_DIR /
        "PHASE3_FINAL_annual_power_cf_aep.csv"
    )

    annual_master.to_csv(
        annual_master_file,
        index=False
    )

    print(
        f"✓ Annual master:"
    )

    print(
        f"  {annual_master_file}"
    )

    print(
        f"  Rows: "
        f"{len(annual_master)}"
    )


# Master summary

if summary_files:

    summary_master = pd.concat(
        [
            pd.read_csv(
                file
            )
            for file in summary_files
        ],
        ignore_index=True
    )

    summary_master = (
        summary_master
        .sort_values(
            [
                "model",
                "experiment",
                "hub_height_m",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary_master_file = (
        SUMMARY_DIR /
        "PHASE3_FINAL_power_cf_aep_summary.csv"
    )

    summary_master.to_csv(
        summary_master_file,
        index=False
    )

    print(
        f"✓ Summary master:"
    )

    print(
        f"  {summary_master_file}"
    )

    print(
        f"  Rows: "
        f"{len(summary_master)}"
    )


# Final validation

print()
print("=" * 90)
print("PHASE 3 VALIDATION")
print("=" * 90)


expected_datasets = (
    len(MODELS)
    *
    len(EXPERIMENTS)
)


successful = sum(
    1
    for result in results
    if result["status"]
    == "SUCCESS"
)


failed = (
    expected_datasets
    -
    successful
)


print(
    f"Expected datasets : "
    f"{expected_datasets}"
)

print(
    f"Successful         : "
    f"{successful}"
)

print(
    f"Failed             : "
    f"{failed}"
)


# Expected annual rows

# Historical: 65 years SSP245: 86 years SSP585: 86 years × 3 hub heights × 7 models


expected_annual_rows = (
    7
    *
    (65 + 86 + 86)
    *
    3
)


final_annual = (
    SUMMARY_DIR /
    "PHASE3_FINAL_annual_power_cf_aep.csv"
)


if final_annual.exists():

    df = pd.read_csv(
        final_annual
    )

    print()
    print(
        f"Annual rows        : "
        f"{len(df)}"
    )

    print(
        f"Expected annual    : "
        f"{expected_annual_rows}"
    )

    if len(df) == expected_annual_rows:

        print(
            "✓ Annual row-count "
            "validation PASSED"
        )

    else:

        print(
            "⚠ Annual row-count "
            "does not match expected"
        )


# Physical sanity checks

if final_annual.exists():

    df = pd.read_csv(
        final_annual
    )

    print()
    print(
        "PHYSICAL SANITY CHECKS"
    )

    # Capacity factor must be 0–1

    cf_min = df[
        "capacity_factor"
    ].min()

    cf_max = df[
        "capacity_factor"
    ].max()

    print(
        f"Capacity factor range: "
        f"{cf_min:.6f} → "
        f"{cf_max:.6f}"
    )

    if (
        cf_min >= 0
        and cf_max <= 1
    ):

        print(
            "✓ Capacity factor range valid"
        )

    else:

        print(
            "❌ Capacity factor outside 0–1"
        )

    # Power must be between 0 and rated power

    power_min = df[
        "mean_power_mw"
    ].min()

    power_max = df[
        "mean_power_mw"
    ].max()

    print(
        f"Mean power range: "
        f"{power_min:.6f} → "
        f"{power_max:.6f} MW"
    )

    if (
        power_min >= 0
        and power_max <= RATED_POWER_MW
    ):

        print(
            "✓ Power range valid"
        )

    else:

        print(
            "❌ Power outside turbine limits"
        )

    # Energy must be non-negative

    energy_min = df[
        "annual_energy_gwh"
    ].min()

    print(
        f"Minimum annual energy: "
        f"{energy_min:.6f} GWh"
    )

    if energy_min >= 0:

        print(
            "✓ Energy values valid"
        )

    else:

        print(
            "❌ Negative energy detected"
        )


# Final result

print()

if (
    successful == expected_datasets
    and
    final_annual.exists()
):

    print(
        "🎉 PHASE 3A COMPLETE"
    )

    print()
    print(
        "All 21 model/experiment datasets "
        "were processed."
    )

    print()
    print(
        "No daily hub-height or power NetCDF "
        "files were created."
    )

    print()
    print(
        "Next step: PHASE 3B — spatial "
        "CF/AEP calculation and maps."
    )

else:

    print(
        "⚠ PHASE 3A INCOMPLETE"
    )

    print(
        "Check the failed datasets above."
    )


print()
print("DONE.")