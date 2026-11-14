# CMIP6 WIND ENERGY PREDICTION


A comprehensive multi-model global assessment of future wind energy potential, capacity factors (CF), and annual energy production (AEP) under climate change scenarios (SSP2-4.5 and SSP5-8.5), based on 7 CMIP6 Global Climate Models (GCMs).

---

## 📑 Research Paper & Deliverables

- **Top 100 Real-World Wind Farm Siting Candidates**: [`CMIP6/PHASE8/tables/PHASE8_top100_wind_farm_locations.csv`](CMIP6/PHASE8/tables/PHASE8_top100_wind_farm_locations.csv)
- **Optimal Hub Height Map**: [`CMIP6/PHASE8/figures/PHASE8_global_optimal_hub_height_map.png`](CMIP6/PHASE8/figures/PHASE8_global_optimal_hub_height_map.png)

---

## 🏗️ Repository Architecture

```text
├── WindEnergyFinal/                 # 9 Canonical Pipeline Execution Scripts
│   ├── phase1.py                    # Daily wind speed extraction (sfcWind / uas+vas)
│   ├── phase2.py                    # Vertical wind profile extrapolation (80m–200m)
│   ├── phase3.py                    # Power curves, Capacity Factor (CF), and AEP modeling
│   ├── phase3B.py                   # Native model grid spatial aggregation & statistics
│   ├── phase4.py                    # 1.25° common grid regridding & 7-model MME ensemble
│   ├── phase5.py                    # Climate Resiliency Index (CRI), FYRI, & model agreement
│   ├── phase6.py                    # Seasonal dynamics (DJF, MAM, JJA, SON) analysis
│   ├── phase7.py                    # Multi-criteria composite screening & candidate tiles
│   └── phase8.py                    # Global optimal hub height selection & Top 100 practical sites
│
├── CMIP6/
│   ├── IEEE_Research_Paper.pdf      # Complete 7-page IEEE-formatted research paper
│   ├── IEEE_Research_Paper.docx     # Editable Word format
│   ├── PHASE1/summaries/            # Wind speed summaries & ACCESS recovery diagnostics
│   ├── PHASE2/summaries/            # Extrapolated hub-height wind statistics
│   ├── PHASE3/summaries/            # Turbine-specific generation summaries
│   ├── PHASE3B/summaries/           # Native spatial summaries across all models
│   ├── PHASE4/                      # 1.25° regridded ensemble validation & summaries
│   ├── PHASE5/                      # CRI, FYRI, model agreement tables & 42 figures
│   ├── PHASE6/                      # Seasonal generation statistics & 56 figures
│   ├── PHASE7/                      # Multi-criteria screening tables & candidate figures
│   └── PHASE8/                      # Optimal hub height map & Top 100 evaluated sites
│
└── .gitignore                       # NetCDF binary (>400 GB) & manifest exclusion rules
```

---

## 🔬 Computational Pipeline Overview

| Phase | Core Function | Primary Output |
| :---: | :--- | :--- |
| **Phase 1** | Daily Near-Surface Wind Extraction | 21 processed NetCDF daily series across 7 models |
| **Phase 2** | Power-Law Vertical Extrapolation | Hub-height wind speeds (80m, 100m, 120m, 150m, 200m) |
| **Phase 3** | Aero-Electric Turbine Power Modeling | Daily CF & AEP across 4 turbine classes (2 MW to 8 MW) |
| **Phase 3B**| Native Grid Spatial Aggregation | Native grid means, percentiles, and annual statistics |
| **Phase 4** | 1.25° Regridding & 7-Model Ensemble | Harmonized global common-grid Multi-Model Ensemble (MME) |
| **Phase 5** | Long-Term Climate Risk Indices | Climate Resiliency Index (CRI), FYRI, and consensus agreement |
| **Phase 6** | Seasonal Dynamics & Variability | Seasonal CF floors and intra-annual generation stability |
| **Phase 7** | Multi-Criteria Composite Siting | Composite resource scores and robust candidate identification |
| **Phase 8** | Practical Siting & AI Grid Analysis | Global optimal hub height map & Top 100 bankable sites |

---

## 🌍 CMIP6 Climate Models Evaluated

All analyses incorporate daily simulations from 7 CMIP6 Global Climate Models:
1. **ACCESS-CM2** (CSIRO-ARCCSS, Australia)
2. **ACCESS-ESM1-5** (CSIRO, Australia)
3. **BCC-CSM2-MR** (Beijing Climate Center, China)
4. **CanESM5** (Canadian Centre for Climate Modelling and Analysis, Canada)
5. **CNRM-CM6-1** (CNRM-CERFACS, France)
6. **GFDL-CM4** (NOAA Geophysical Fluid Dynamics Laboratory, USA)
7. **GFDL-ESM4** (NOAA Geophysical Fluid Dynamics Laboratory, USA)

Scenarios analyzed:
- **Historical**: 1995–2014 (20-year baseline)
- **SSP2-4.5**: 2041–2060 (Middle-of-the-road stabilization scenario)
- **SSP5-8.5**: 2041–2060 (High-emissions fossil-fueled development scenario)

---

## 📊 Key Highlights & Results

1. **Global Capacity Factor Trends**: Global mean CF at 100 m hub height increases slightly from **27.04%** (Historical) to **27.30%** (SSP2-4.5) and **27.36%** (SSP5-8.5).
2. **Optimal Hub Height Trade-offs**:
   - High-wind offshore corridors (e.g., Southern Ocean, North Sea) saturate rated power early; **80 m – 100 m** hub heights provide the optimal LCOE and structural efficiency.
   - Continental land regions benefit significantly from taller **120 m** towers due to surface boundary-layer shear, unlocking **+12% to +24%** marginal energy yield.
3. **Premier Global Development Zones**:
   - **Patagonia & Tierra del Fuego (Argentina/Chile)**: Top terrestrial wind resource globally (CF > 54%, conservative AEP > 14 GWh/yr per 3 MW turbine, high model agreement > 90%).
   - **North American Great Plains (US/Canada)**: Highly robust continental wind corridor (CF 38%–45%) directly tied into major transmission interconnections.
   - **Nordic & North Sea Coastal Regions (Norway, UK, Denmark)**: Strong offshore wind yields (CF 42%–50%) with seasonal peaking aligned with winter heating demand.

---


---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

