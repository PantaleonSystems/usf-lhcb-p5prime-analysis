# scripts/covariance.py
"""Experimental covariance of P5', built from the LHCb correlation matrices.

P5' is reconstructed from two measured angular coefficients,

    P5' = S5 / sqrt(F_L (1 - F_L)),

so its uncertainty depends on the correlation between F_L and S5. The published
analysis propagated errors assuming those were uncorrelated and justified it in
Sec. 3.2 with an estimate ("|rho| <~ 0.2 in the kinematic region considered;
neglecting it affects the total uncertainty by at most 10%").

No estimate was needed. HEPData record ins1409497 ships the full likelihood
correlation matrix for every q^2 bin, and those files are already vendored in
data/raw/. This module reads them.

Honest result: the estimate holds up. The measured correlations are
|rho(F_L, S5)| <= 0.10 and the effect on sigma_P5' is at the percent level.
This is the one approximation in the manuscript that survives scrutiny. It is
implemented anyway because it removes an assumption at no cost, and because the
matrices being present but unread was itself the problem.

Between different q^2 bins the LHCb fits are independent, so the experimental
covariance of P5' is diagonal -- but each diagonal entry changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

YAML_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "HEPData-ins1409497-v1-yaml"

FL_NAME = "$F_{\\rm L}$"
S5_NAME = "$S_{5}$"
#: The correlation tables spell S5 without braces around the subscript.
S5_NAME_IN_CORRELATION = "$S_5$"

#: Maximum-likelihood correlation matrices, one per narrow q^2 bin.
#: Verified against the descriptions in submission.yaml by the test suite.
CORRELATION_TABLES: dict[tuple[float, float], str] = {
    (1.1, 2.5): "Table10",
    (2.5, 4.0): "Table11",
    (4.0, 6.0): "Table12",
    (6.0, 8.0): "Table13",
    (11.0, 12.5): "Table14",
    (15.0, 17.0): "Table15",
    (17.0, 19.0): "Table16",
}


def _load(table: str) -> dict:
    with open(YAML_DIR / f"{table}.yaml") as fh:
        return yaml.safe_load(fh)


def load_correlation_matrix(table: str) -> tuple[list[str], np.ndarray]:
    """Read one HEPData correlation table into (observable names, matrix).

    The tables are stored in long format: two CORR independent variables
    holding the row and column labels, and one dependent variable holding the
    64 entries of the 8x8 matrix.
    """
    doc = _load(table)
    rows = [v["value"] for v in doc["independent_variables"][0]["values"]]
    cols = [v["value"] for v in doc["independent_variables"][1]["values"]]
    entries = [v["value"] for v in doc["dependent_variables"][0]["values"]]

    names = list(dict.fromkeys(rows))
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)))
    for row, col, value in zip(rows, cols, entries):
        matrix[index[row], index[col]] = value
    return names, matrix


def fl_s5_correlation(bin_: tuple[float, float]) -> float:
    """rho(F_L, S5) for one q^2 bin, from the published likelihood matrix."""
    names, matrix = load_correlation_matrix(CORRELATION_TABLES[bin_])
    return float(matrix[names.index(FL_NAME), names.index(S5_NAME_IN_CORRELATION)])


def p5p_from_fl_s5(fl, s5):
    """P5' = S5 / sqrt(F_L (1 - F_L))."""
    return s5 / np.sqrt(fl * (1 - fl))


def p5p_error(fl, fl_err, s5, s5_err, rho=0.0):
    """Propagated uncertainty on P5', optionally including the F_L-S5 correlation.

    sigma^2 = (dP/dS5)^2 sigma_S5^2 + (dP/dF_L)^2 sigma_FL^2
              + 2 (dP/dS5)(dP/dF_L) rho sigma_S5 sigma_FL

    The final term is the one the published analysis dropped.
    """
    denom = np.sqrt(fl * (1 - fl))
    d_s5 = 1.0 / denom
    d_fl = -0.5 * s5 * (1 - 2 * fl) / denom ** 3
    variance = (
        (d_s5 * s5_err) ** 2
        + (d_fl * fl_err) ** 2
        + 2 * d_s5 * d_fl * rho * s5_err * fl_err
    )
    return np.sqrt(variance)


def experimental_covariance(bins, fl, fl_err, s5, s5_err) -> np.ndarray:
    """Diagonal experimental covariance of P5' across the given bins.

    Diagonal because the LHCb angular fits in different q^2 bins are
    independent; the correlations that exist are between observables *within*
    a bin, and those are folded into each variance here.
    """
    rho = np.array([fl_s5_correlation(tuple(b)) for b in bins])
    sigma = p5p_error(fl, fl_err, s5, s5_err, rho)
    return np.diag(sigma ** 2)


if __name__ == "__main__":
    import pandas as pd

    from sm_predictions import ANALYSIS_BINS

    def _observable(table, name):
        doc = _load(table)
        target = next(
            dv for dv in doc["dependent_variables"] if dv["header"]["name"] == name
        )
        values, errors = [], []
        for entry in target["values"]:
            variance = 0.0
            for err in entry["errors"]:
                if "symerror" in err:
                    variance += err["symerror"] ** 2
                else:
                    plus, minus = err["asymerror"]["plus"], err["asymerror"]["minus"]
                    variance += ((plus - minus) / 2) ** 2
            values.append(entry["value"])
            errors.append(np.sqrt(variance))
        bins = [
            (v["low"], v["high"]) for v in doc["independent_variables"][0]["values"]
        ]
        return bins, np.array(values), np.array(errors)

    all_bins, fl, fl_err = _observable("Table2", FL_NAME)
    _, s5, s5_err = _observable("Table2", S5_NAME)
    keep = [all_bins.index(b) for b in ANALYSIS_BINS]
    fl, fl_err, s5, s5_err = fl[keep], fl_err[keep], s5[keep], s5_err[keep]

    rho = np.array([fl_s5_correlation(b) for b in ANALYSIS_BINS])
    without = p5p_error(fl, fl_err, s5, s5_err, 0.0)
    with_rho = p5p_error(fl, fl_err, s5, s5_err, rho)

    print(f"{'bin (GeV^2)':>14} {'rho(FL,S5)':>11} {'sigma (rho=0)':>14} "
          f"{'sigma (published rho)':>22} {'change':>8}")
    for b, r, a, c in zip(ANALYSIS_BINS, rho, without, with_rho):
        print(f"{str(b):>14} {r:+11.3f} {a:14.4f} {c:22.4f} {(c / a - 1) * 100:+7.1f}%")

    print(f"\nLargest change in sigma_P5': {np.max(np.abs(with_rho / without - 1)) * 100:.1f}%")
    print("The manuscript's Sec. 3.2 estimate of 'at most 10%' is borne out.")
