# scripts/generate_p5p_csv.py
"""Build data/p5p_observables.csv from the raw HEPData YAML files.

The previous version read F_L from Table1 (two wide bins: 1.1-6 and 15-19) and
S5 from Table2 (eight narrow bins), then zipped the two lists positionally --
pairing F_L(1.1-6) with S5(0.1-0.98). The bin centres it derived from Table1
(3.55 and 17.0 GeV^2) matched none of the seven targets within tolerance, so
`selected` came out empty and the script wrote a zero-row CSV without raising.
The committed CSV was therefore produced some other way, which is why the
manuscript's claim of full reproducibility from the raw files did not hold.

Both coefficients now come from Table2, the extraction asserts that it found
all seven bins, and the F_L-S5 correlations from the published likelihood
matrices are applied (see scripts/covariance.py).

LHCb also publishes P5' directly for the two wide bins in Table1. Those values
are emitted alongside as a cross-check, since they carry the collaboration's
own treatment of the correlations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from covariance import (
    FL_NAME,
    S5_NAME,
    fl_s5_correlation,
    p5p_error,
    p5p_from_fl_s5,
)
from sm_predictions import ANALYSIS_BINS

YAML_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "HEPData-ins1409497-v1-yaml"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "p5p_observables.csv"
CROSS_CHECK = Path(__file__).resolve().parent.parent / "data" / "p5p_published_wide_bins.csv"

#: Narrow-binned CP-averaged angular observables. Both F_L and S5 live here.
SOURCE_TABLE = "Table2"

#: Wide-binned table, which additionally carries the published P5'.
PUBLISHED_TABLE = "Table1"
P5P_NAME = "$P'_{5}$"


def load_observable(table: str, name: str):
    """Extract values, total errors and bins for one observable.

    Statistical and systematic uncertainties are summed in quadrature;
    asymmetric errors are symmetrised as half the full span.
    """
    with open(YAML_DIR / f"{table}.yaml") as fh:
        doc = yaml.safe_load(fh)

    matches = [
        dv for dv in doc["dependent_variables"] if dv["header"]["name"] == name
    ]
    if not matches:
        available = [dv["header"]["name"] for dv in doc["dependent_variables"]]
        raise ValueError(f"{name!r} not found in {table}.yaml; available: {available}")

    values, errors = [], []
    for entry in matches[0]["values"]:
        variance = 0.0
        for err in entry["errors"]:
            if "symerror" in err:
                variance += err["symerror"] ** 2
            elif "asymerror" in err:
                plus, minus = err["asymerror"]["plus"], err["asymerror"]["minus"]
                variance += ((plus - minus) / 2) ** 2
        values.append(entry["value"])
        errors.append(np.sqrt(variance))

    bins = [
        (v["low"], v["high"]) for v in doc["independent_variables"][0]["values"]
    ]
    return bins, np.array(values), np.array(errors)


def build_analysis_csv() -> pd.DataFrame:
    """Reconstruct P5' for the seven analysis bins from F_L and S5."""
    bins, fl, fl_err = load_observable(SOURCE_TABLE, FL_NAME)
    bins_s5, s5, s5_err = load_observable(SOURCE_TABLE, S5_NAME)
    if bins != bins_s5:
        raise ValueError("F_L and S5 are binned differently within the same table")

    missing = [b for b in ANALYSIS_BINS if b not in bins]
    if missing:
        raise ValueError(f"{SOURCE_TABLE} does not contain the bins {missing}")

    keep = [bins.index(b) for b in ANALYSIS_BINS]
    fl, fl_err = fl[keep], fl_err[keep]
    s5, s5_err = s5[keep], s5_err[keep]

    rho = np.array([fl_s5_correlation(b) for b in ANALYSIS_BINS])
    value = p5p_from_fl_s5(fl, s5)

    frame = pd.DataFrame({
        "q2_min": [b[0] for b in ANALYSIS_BINS],
        "q2_max": [b[1] for b in ANALYSIS_BINS],
        "q2_center": [np.mean(b) for b in ANALYSIS_BINS],
        "value": value,
        # Correlated propagation is the one used by the fit.
        "error": p5p_error(fl, fl_err, s5, s5_err, rho),
        # Retained so the effect of the correlation stays auditable, and so the
        # published numbers remain reproducible from this file.
        "error_uncorrelated": p5p_error(fl, fl_err, s5, s5_err, 0.0),
        "rho_fl_s5": rho,
        "FL": fl,
        "FL_error": fl_err,
        "S5": s5,
        "S5_error": s5_err,
    })

    if len(frame) != len(ANALYSIS_BINS):
        raise AssertionError(
            f"expected {len(ANALYSIS_BINS)} bins, built {len(frame)}"
        )
    if not np.all(np.isfinite(frame["value"])) or not np.all(frame["error"] > 0):
        raise AssertionError("non-finite P5' or non-positive uncertainty produced")
    return frame


def build_cross_check_csv() -> pd.DataFrame:
    """LHCb's own P5' for the two wide bins, with its own error treatment."""
    bins, value, error = load_observable(PUBLISHED_TABLE, P5P_NAME)
    _, fl, fl_err = load_observable(PUBLISHED_TABLE, FL_NAME)
    _, s5, s5_err = load_observable(PUBLISHED_TABLE, S5_NAME)

    return pd.DataFrame({
        "q2_min": [b[0] for b in bins],
        "q2_max": [b[1] for b in bins],
        "value_published": value,
        "error_published": error,
        "value_reconstructed": p5p_from_fl_s5(fl, s5),
        "error_reconstructed_uncorrelated": p5p_error(fl, fl_err, s5, s5_err, 0.0),
    })


def main() -> None:
    frame = build_analysis_csv()
    frame.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT.relative_to(OUTPUT.parents[1])} with {len(frame)} bins.")
    print(frame[["q2_min", "q2_max", "value", "error", "error_uncorrelated"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    cross = build_cross_check_csv()
    cross.to_csv(CROSS_CHECK, index=False)
    print(f"\nCross-check against LHCb's published P5' ({PUBLISHED_TABLE}):")
    print(cross.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
