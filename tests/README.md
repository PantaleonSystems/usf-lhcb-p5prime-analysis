# Validation suite

This suite is deliberately **failing**. It encodes the claims the manuscript
makes about its own pipeline, so that each defect is named by a test rather
than found by inspection. Fixing the analysis means turning these green.

```bash
conda env update -f environment.yaml   # adds flavio, wilson, pytest
pytest
```

Current status: **23 failed, 3 passed**.

## What passes

| Test | Meaning |
| --- | --- |
| `test_committed_csv_matches_a_table2_reconstruction` | The seven measured P5' values are correct — a faithful Table2 reconstruction. The *data* are not the problem. |
| `test_published_p5p_is_used_where_available` | LHCb's own P5' values are present in Table1 and available to anchor against. |
| `test_sm_limit_is_recovered` | `kappa = 0` returns the SM curve exactly. |

## What fails, by severity

### Fatal — the model has no content

| Test | Defect |
| --- | --- |
| `test_f_geo_actually_varies_with_q2` | `f_geo(q^2) == 1.0` identically. Two unit errors leave `f_geo - 1 ~ 1.6e-33`, which underflows against 1.0. |
| `test_usf_is_distinguishable_from_a_constant_shift` | Consequently the fitted model *is* `P5'_SM(q^2) - 0.3*kappa`. A plain constant offset recovers the same kappa to 8 significant figures and the same chi2 to 12. |
| `test_f_geo_decreases_with_q2` | The low- to high-q^2 contrast the manuscript's Figure 1 shows does not exist in the code. That figure is drawn from a hard-coded `illustrative_amplitude = 0.8` in `plot_p5p.py`, not from `geometric_factor_usf`. |

### Fatal — the reference curve manufactures the anomaly

| Test | Defect |
| --- | --- |
| `test_sm_curve_matches_flavio` | The hard-coded SM grid is off by up to **32.6 theory sigma** (bin 11–12.5: −0.139 vs −0.822). |
| `test_sm_high_q2_anchor` | `P5'_SM(16 GeV^2) = -0.119`; the accepted value is ≈ −0.67. |
| `test_reference_grid_covers_the_analysis_range` | The 18 GeV² bin is blind cubic **extrapolation** (`_q_ref` stops at 15.0), and contributes 19.7 of the 70.4 chi2. |
| `test_anomaly_has_the_sign_reported_in_the_literature` | With the correct SM, data sit *above* the SM in 4–8 GeV² (the real anomaly). With the hard-coded curve they sit below, inverting the sign — which is why the fit prefers `kappa = +1.44` while global fits prefer `Delta C9 < 0`. |
| `test_delta_chi2_survives_a_correct_sm_curve` | chi2_SM collapses **70.4 → 17.2** (7 dof, p = 0.016 ≈ 2.4σ) once flavio supplies the SM and theory errors are included. |

### Serious — units and dimensions

| Test | Defect |
| --- | --- |
| `test_planck_energy_is_expressed_in_gev` | `utils.E_P` is in joules; `E_P/1e9` is used as if GeV, giving 1.956 instead of 1.22e19 — a factor ~1e19. |
| `test_planck_suppression_is_not_saturated` | The correct `tanh(E_cms²/E_P²)` at 14 TeV is 1.3e-30. The code returns 1.0, i.e. it *accidentally reproduces the manuscript's incorrect claim* via the joule/GeV error. |
| `test_geometric_prefactor_is_dimensionless_and_order_unity` | `G_LQG * R_AdS = 1.63e-33` and carries dimension of **length**; the manuscript calls it a dimensionless O(1) constant. |
| `test_collider_energy_matches_the_fitted_dataset` | `collision_energy` defaults to 14 TeV; the fitted Run-1 data are 7 and 8 TeV. |

### Serious — statistics

| Test | Defect |
| --- | --- |
| `test_theory_uncertainty_enters_the_chi2` | chi2 uses experimental errors only. Theory errors are 0.03–0.11 here, comparable to the experimental ones at high q². |
| `test_bin_integrated_not_bin_centre` | P5' is evaluated at bin centres. It is a ratio of integrals and must be bin-averaged, notably for the 4 GeV-wide bins. |
| `test_correlation_matrices_are_available_and_used` | HEPData ships a likelihood correlation matrix per q² bin; **they are already vendored in `data/raw/` and no script reads them**. Sec. 3.2's "at most 10%" estimate was unnecessary. |
| `test_significance_is_not_taken_as_sqrt_delta_chi2` | 7.5σ is `sqrt(56.5)`. Wilks is asymptotic; with 7 points and a bounded prior it needs toy-MC calibration, and a p-value should be quoted. |

### Reproducibility

| Test | Defect |
| --- | --- |
| `test_fl_and_s5_are_read_from_the_same_table` | `generate_p5p_csv.py` reads F_L from Table1 (2 bins) and S_5 from Table2 (8 bins), then zips them positionally. |
| `test_extraction_selects_the_seven_analysis_bins` | The resulting bin centres (3.55, 17.0) match no target within tolerance, so the script writes a **zero-row CSV** without raising. The committed CSV was produced some other way. |
| `test_mcmc_is_seeded` | No seed; the credibility interval changes every run. |
| `test_reported_interval_matches_the_manuscript` | json `[1.247, 1.640]` vs manuscript `[1.26, 1.64]`. |
| `test_readme_numbers_match_fit_results` | README `[1.2493, 1.6293]` vs json `[1.2468, 1.6401]` — pasted from a third run. |
| `test_provenance_is_recorded` / `test_convergence_diagnostics_are_reported` | No seed, step count, autocorrelation time or acceptance fraction in the results file. |

## Note on the two "expected to fail forever" tests

`test_planck_suppression_is_not_saturated` and
`test_geometric_prefactor_is_dimensionless_and_order_unity` assert the
*physically correct* values, not the manuscript's claims. They cannot be made
to pass by fixing arithmetic: making the tanh correct drives `f_geo → 1`
exactly, and the model loses its geometric prediction. They mark the point
where the framework has to be either reformulated or restated as a
phenomenological ansatz with a free normalisation.
