# tests/test_covariance.py
"""Tests on the experimental covariance built from the LHCb correlation matrices.

Sec. 3.2 of the manuscript estimated that neglecting the F_L-S5 correlation
"affects the total uncertainty by at most 10%". The estimate was unnecessary --
the matrices ship with the HEPData record and are vendored in data/raw/ -- but
it was also correct. These tests record both facts.
"""
import re

import numpy as np
import pytest
import yaml

from conftest import ANALYSIS_BINS, YAML_DIR


@pytest.fixture(scope="module")
def cov():
    import covariance

    return covariance


def test_table_mapping_matches_submission_metadata(cov):
    """The bin each correlation table describes must come from the record.

    Guards the hard-coded CORRELATION_TABLES mapping against a silent mismatch,
    which would attach the wrong correlation to the wrong bin.
    """
    with open(YAML_DIR / "submission.yaml") as fh:
        docs = [d for d in yaml.safe_load_all(fh) if d and "data_file" in d]

    declared = {}
    for doc in docs:
        description = doc.get("description", "")
        if "correlation" not in description.lower():
            continue
        match = re.search(r"([\d.]+)\s*<\s*q\^2\s*<\s*([\d.]+)", description)
        if match:
            table = doc["data_file"].replace(".yaml", "")
            declared[table] = (float(match.group(1)), float(match.group(2)))

    for bin_, table in cov.CORRELATION_TABLES.items():
        assert declared[table] == bin_, (
            f"{table} is mapped to {bin_} but the record describes it as "
            f"{declared[table]}"
        )


def test_matrices_are_well_formed(cov):
    """Each table must be a genuine correlation matrix."""
    for bin_, table in cov.CORRELATION_TABLES.items():
        names, matrix = cov.load_correlation_matrix(table)
        assert len(names) == 8, f"{table}: expected 8 observables, got {len(names)}"
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-9)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(matrix) > -1e-9), (
            f"{table} is not positive semi-definite"
        )


def test_fl_s5_correlations_are_read_for_every_analysis_bin(cov):
    """All seven bins must resolve to a published correlation."""
    rho = np.array([cov.fl_s5_correlation(b) for b in ANALYSIS_BINS])
    assert len(rho) == len(ANALYSIS_BINS)
    assert np.all(np.abs(rho) <= 1.0)
    # The manuscript asserted |rho| <~ 0.2 without checking. It was right.
    assert np.abs(rho).max() <= 0.2, f"measured correlations {rho.round(3)}"


def test_correlation_term_changes_the_errors_only_slightly(cov):
    """The dropped cross-term matters at the percent level, not more.

    Recorded so the correction is not oversold. This is the one approximation
    in the published analysis that survives scrutiny; the objection was that it
    was estimated rather than computed from data already in the repository.
    """
    import pandas as pd

    from conftest import DATA_DIR

    frame = pd.read_csv(DATA_DIR / "p5p_observables.csv")
    published_errors = frame["error_uncorrelated"].values

    fl, fl_err, s5, s5_err = _table2_inputs()
    rho = np.array([cov.fl_s5_correlation(b) for b in ANALYSIS_BINS])

    without = cov.p5p_error(fl, fl_err, s5, s5_err, 0.0)
    with_rho = cov.p5p_error(fl, fl_err, s5, s5_err, rho)

    # rho = 0 must reproduce exactly what the committed CSV holds.
    np.testing.assert_allclose(without, published_errors, rtol=1e-10)

    change = np.abs(with_rho / without - 1).max()
    assert change < 0.10, f"correlation changes sigma by {change:.1%}"
    assert change > 1e-4, "correlation term appears not to be applied at all"


def test_experimental_covariance_is_diagonal_and_positive(cov):
    """Different q^2 bins are independent fits, so the matrix is diagonal."""
    fl, fl_err, s5, s5_err = _table2_inputs()
    matrix = cov.experimental_covariance(ANALYSIS_BINS, fl, fl_err, s5, s5_err)

    n = len(ANALYSIS_BINS)
    assert matrix.shape == (n, n)
    np.testing.assert_allclose(matrix[~np.eye(n, dtype=bool)], 0.0)
    assert np.all(np.diag(matrix) > 0)


def _table2_inputs():
    """F_L and S5 for the seven analysis bins, from HEPData Table2."""
    import covariance

    def observable(name):
        doc = covariance._load("Table2")
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
        keep = [bins.index(b) for b in ANALYSIS_BINS]
        return np.array(values)[keep], np.array(errors)[keep]

    fl, fl_err = observable(covariance.FL_NAME)
    s5, s5_err = observable(covariance.S5_NAME)
    return fl, fl_err, s5, s5_err
