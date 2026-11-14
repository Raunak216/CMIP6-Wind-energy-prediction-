Phase 7 output guide

PHASE7_all_grid_cell_scores.csv: score/metric record for every grid cell and hub height.
PHASE7_high_potential_cells.csv: top-ranked cells passing potential (CF percentile + seasonal-floor) gates.
PHASE7_robust_candidates.csv: Tier 1 subset passing high-potential, composite-score and seasonal non-decline agreement rules.
PHASE7_region_tiles_5deg.csv: ranked 5-degree x 5-degree screening tiles, separated by land/ocean/coastal class where the Natural Earth mask is available.
PHASE7_input_audit.csv: Phase 4/6 input completeness and valid-model coverage.
PHASE7_score_eligibility_audit.csv: counts of why cells were not assigned screening scores.
PHASE7_score_eligibility_100m.png: green = scored with at least 2 valid models; grey = genuinely undefined inputs only.
fields/PHASE7_screening_fields_1p25deg.nc: gridded inputs, indices and score fields.

Historical CF retention is a diagnostic only; the composite resilience score uses absolute annual CF change in percentage points to avoid low-baseline ratio masking.
Candidate maps show conservative future CF as the background and highlight selected cells in green; non-candidates remain visible in the background.
IMPORTANT: candidate cells/tiles are preliminary climate-resource screens only. No terrain, land-use, protected-area, grid-access, bathymetry, cost or measured-wind filters are included. Score weights and thresholds are transparent analyst choices and should be sensitivity-tested before reporting final recommendations.
