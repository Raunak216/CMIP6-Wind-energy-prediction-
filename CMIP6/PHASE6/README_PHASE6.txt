PHASE 6 — SEASONAL CMIP6 WIND-ENERGY ANALYSIS

Inputs: daily Phase 1 wind-speed files for 7 models and three experiments.
Outputs are written below D:\CMIP6\PHASE6.

Folders:
  model_seasonal_1p25deg/  Per-model seasonal climatologies, regridded to 1.25 degrees.
  ensemble/                Seven-model seasonal means, medians, spread and valid-model counts.
  changes/                 SSP245/SSP585 minus historical changes and positive-change agreement.
  figures/                 Global maps at 100 m; NetCDF retains all three hub heights.
  tables/                  Input/season coverage audit and global spatial summaries.

Methods:
  Wind speed at hub height = daily reference-height speed * (hub/reference height)^(1/7).
  Turbine curve: 3 MW; cut-in 4.0 m/s; rated 15.0 m/s; cut-out 25.0 m/s.
  Seasons: DJF, MAM, JJA, SON. Incomplete edge seasons are dropped.
  Cells require at least 95% valid days in a season-year.
  Model seasonal means are calculated on each model native grid, then linearly regridded
  to the common 1p25deg comparison grid with periodic longitude and no latitude extrapolation.
  Seasonal energy is standardized to common season durations: {'DJF': 90.25, 'MAM': 92.0, 'JJA': 92.0, 'SON': 91.0} days.

Interpretation caveats:
  * The daily wind-speed data do not resolve sub-daily wind variability.
  * A 1.25-degree target grid standardizes comparisons but is not new native climate detail.
  * Ensemble valid-model counts may be lower than seven at polar edges where source model
    latitude coverage differs. Consult NetCDF valid-model-count variables and audit CSV.
  * AEP is annualized seasonal mean power for one representative 3-MW turbine per cell;
    standardized seasonal energy is not a total regional/farm generation estimate.
