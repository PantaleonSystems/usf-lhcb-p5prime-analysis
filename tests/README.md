# Validation suite

This suite encodes the claims the analysis makes about itself, so that defects
are named by a test rather than found by inspection. The original pipeline had
no test that could fail: every check was a figure the authors inspected.

```bash
make test
```

Current status: **45 passed, 5 xfailed**.

Every remaining `xfail` is marked `strict=True` and carries a reason. They are
**results, not unfixed defects** — each asserts something the model would have
to do in order to have content, and records that it does not. If any of them
ever passes, the suite fails, because that would mean the physics changed.

## The five xfails

| Test | What it asserts | Measured |
| --- | --- | --- |
| `test_geometry.py::test_geometric_prefactor_is_dimensionless_and_order_unity` | $G_{\rm LQG} R_{\rm AdS}$ is dimensionless and O(1), as the manuscript states | $1.63\times10^{-33}$, carrying a dimension of length |
| `test_geometry.py::test_f_geo_actually_varies_with_q2` | $f_{\rm geo}$ depends on $q^2$ | Predicted amplitude $\approx 7\times10^{-64}$, so $f_{\rm geo} \equiv 1$ |
| `test_geometry.py::test_f_geo_decreases_with_q2` | $f_{\rm geo}$ is larger at low $q^2$ (manuscript Fig. 1) | Flat; the falling curve exists only at a hand-supplied amplitude |
| `test_baseline.py::test_usf_does_not_beat_a_free_c9` | The USF improves on a plain constant $\Delta C_9$ | $\Delta\chi^2 = -0.018$ at equal parameter count |
| `test_baseline.py::test_q2_shape_earns_its_parameter` | Freeing the amplitude justifies the extra parameter | AIC worsens, 7.23 → 9.01 |

The first three are the referee's central objection, confirmed. The last two are
its consequence for the fit.

## Coverage by file

| File | Subject |
| --- | --- |
| `test_geometry.py` | Units, dimensions and the geometric coupling. The repaired unit errors are pinned here so they cannot return. |
| `test_sm.py` | The SM reference: bin integration, theory covariance, the anomaly's sign, and how far the superseded curve was off (quarantined, not deleted). |
| `test_response.py` | $P_5'$ against the Wilson coefficients: the interpolator is validated against exact flavio calls, and the constant-slope assumption is measured against the true bin-dependent one. |
| `test_covariance.py` | The LHCb correlation matrices: table-to-bin mapping checked against `submission.yaml`, and the effect of the previously dropped cross-term measured at 1.3%. |
| `test_data_extraction.py` | HEPData → CSV, including a cross-check of the reconstruction against LHCb's own published $P_5'$. |
| `test_baseline.py` | Model comparison against the naive alternative — the check whose absence let the original result stand. |
| `test_global_consistency.py` | The fixed $\Delta C_{10}/\Delta C_9 = -0.2$ prediction against $B_s\to\mu\mu$, $R_K$, $R_{K^*}$, under both lepton-flavour readings. |
| `test_pipeline.py` | Seeding, provenance, input hashing and convergence diagnostics. |

## What the suite established

Beyond the two referee reports:

- The SM curve deviated by up to **26 theory sigma**, and inverted the sign of
  the discrepancy in the anomaly region.
- $dP_5'/dC_9$ varies by a factor of **14** across the bins, so the constant
  $-0.3$ gave the high-$q^2$ bins — exactly where the fake anomaly sat — an
  order of magnitude too much leverage. The two errors compounded.
- `generate_p5p_csv.py` could not have produced the committed CSV.
- The correlation matrices were vendored in `data/raw/` and unread.

And two findings in the analysis's favour, recorded rather than buried:

- The measured $P_5'$ values are correct, reproducing LHCb's own published
  values to better than 0.001.
- The manuscript's estimate that neglecting the $F_L$–$S_5$ correlation matters
  "at most 10%" was right; the true effect is 1.3%.
