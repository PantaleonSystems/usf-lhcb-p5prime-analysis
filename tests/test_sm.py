# tests/test_sm.py
"""Tests on the Standard Model reference for P5'.

The published analysis hard-coded eight (q^2, P5'_SM) pairs in scripts/utils.py
and cubic-interpolated them. The source comment attributed them to "Fig. 5 of
arXiv:1505.07814" while the manuscript cites LHCb 2016a (arXiv:1512.04442).

That curve was wrong in the direction that manufactures the claimed anomaly:
it puts P5'_SM near -0.12 at high q^2 where the true value is near -0.67, and
roughly 66 of the 70.4 units of chi2_SM came from bins at q^2 >= 7 GeV^2 where
no anomaly is known to exist.

scripts/sm_predictions.py replaces it with flavio. These tests pin the new
module's behaviour and keep the old curve from creeping back.
"""
import inspect

import numpy as np
import pytest

from conftest import ANALYSIS_BINS, ANALYSIS_BIN_CENTRES


def test_prediction_is_bin_integrated(sm):
    """P5' must be requested for a bin, not for a point.

    P5' is a ratio of integrals of angular coefficients, so <P5'>_bin differs
    from P5'(q^2_centre) -- materially for the wide bins used here. The
    interface takes (q2min, q2max) and offers no way to ask for a centre.
    """
    params = set(inspect.signature(sm.p5p_sm).parameters)
    assert {"q2min", "q2max"} <= params, f"p5p_sm accepts {sorted(params)}"


def test_covers_the_full_analysis_range(sm):
    """Every analysis bin must be predictable, none by extrapolation.

    The superseded grid stopped at 15.0 GeV^2, so the 17-19 bin was blind cubic
    extrapolation returning -0.034 and contributing 19.7 units of chi2 on its
    own. flavio has no such grid.
    """
    values = sm.p5p_sm_binned(ANALYSIS_BINS)
    assert len(values) == len(ANALYSIS_BINS)
    assert np.all(np.isfinite(values))
    assert np.all(np.abs(values) <= 1.0), "P5' is bounded by construction"


def test_high_q2_anchor(sm):
    """P5'_SM in 15-17 GeV^2 is close to -0.67.

    An anchor against an externally known value. The high-q^2 region is
    theoretically clean (OPE in 1/m_b), so this prediction is not controversial
    and the measurement agrees with it.
    """
    value = sm.p5p_sm(15.0, 17.0)
    assert value == pytest.approx(-0.67, abs=0.15), (
        f"P5'_SM(15-17) = {value:+.3f}; the accepted value is ~-0.67"
    )


def test_low_q2_anchor(sm):
    """P5'_SM rises towards positive values at the bottom of the range."""
    assert sm.p5p_sm(1.1, 2.5) > 0.0


def test_theory_covariance_is_available_and_correlated(sm):
    """Theory errors must be supplied as a covariance, not per-bin sigmas.

    Uncertainties on P5' are dominated by form factors shared across q^2, so
    adjacent bins are correlated at up to rho = 0.97. Treating them as diagonal
    is wrong in both directions -- it is not a conservative approximation.
    """
    cov = sm.p5p_sm_covariance(ANALYSIS_BINS)
    n = len(ANALYSIS_BINS)
    assert cov.shape == (n, n)

    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0), "theory covariance is not positive definite"

    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    off_diagonal = corr[~np.eye(n, dtype=bool)]
    assert off_diagonal.max() > 0.5, (
        f"largest off-diagonal theory correlation is {off_diagonal.max():.2f}; "
        "expected strong correlation from shared form factors"
    )


def test_theory_uncertainties_are_comparable_to_experimental(sm, observed):
    """Theory errors are not negligible, so omitting them inflates significance.

    The published chi2 divided by the experimental error alone.
    """
    _, _, exp_err = observed
    theory_err = sm.p5p_sm_uncertainty(ANALYSIS_BINS)
    ratio = theory_err / exp_err
    assert ratio.max() > 0.2, (
        f"theory/experimental error ratios {ratio.round(2)}; "
        "if these were negligible the omission would be harmless"
    )


def test_anomaly_has_the_sign_reported_in_the_literature(sm, observed):
    """In 4-8 GeV^2 the measurement must lie *above* the SM prediction.

    This is the actual, well-documented P5' anomaly. The superseded curve put
    the data *below* the SM there, inverting the sign of the discrepancy --
    which is why the published fit preferred kappa = +1.44 (shifting P5' down)
    while global fits prefer Delta C9 < 0.
    """
    _, values, _ = observed
    anomaly_bins = [(4.0, 6.0), (6.0, 8.0)]
    indices = [ANALYSIS_BINS.index(b) for b in anomaly_bins]
    residual = values[indices] - sm.p5p_sm_binned(anomaly_bins)
    assert np.all(residual > 0), (
        f"data-minus-SM in the anomaly bins is {residual.round(3)}; "
        "the measured P5' is known to sit above the SM there"
    )


def test_legacy_curve_is_quarantined(sm, utils):
    """The superseded curve must be retained but unused for fitting.

    It stays available so tests and figures can quantify the error, and so the
    provenance of the published numbers is not lost.
    """
    assert "DEPRECATED" in sm.p5p_sm_legacy.__doc__

    legacy = sm.p5p_sm_legacy(ANALYSIS_BIN_CENTRES)
    correct = sm.p5p_sm_binned(ANALYSIS_BINS)
    sigma = sm.p5p_sm_uncertainty(ANALYSIS_BINS)
    deviation = np.abs(legacy - correct) / sigma

    # Pins how wrong it was, so the regression cannot be silently reintroduced.
    assert deviation.max() > 20.0, (
        f"legacy curve deviates by at most {deviation.max():.1f} theory sigma"
    )
    worst = int(np.argmax(deviation))
    assert ANALYSIS_BINS[worst] == (11.0, 12.5)
