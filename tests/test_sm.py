# tests/test_sm.py
"""Tests on the Standard Model reference curve for P5'.

scripts/utils.py hard-codes eight (q^2, P5'_SM) pairs and cubic-interpolates
them. The comment attributes them to "Fig. 5 of arXiv:1505.07814", while the
manuscript cites LHCb 2016a (arXiv:1512.04442) as the source.

The curve is wrong, and it is wrong in the direction that manufactures the
claimed anomaly: it puts P5'_SM near -0.12 at high q^2 where the true SM value
is near -0.67. Roughly 66 of the 70.4 units of chi2_SM reported in the
manuscript come from bins at q^2 >= 7 GeV^2, where no anomaly is known to
exist.
"""
import numpy as np
import pytest

from conftest import ANALYSIS_BINS, ANALYSIS_BIN_CENTRES


def test_reference_grid_covers_the_analysis_range(utils):
    """No prediction may be produced by extrapolation.

    DEFECT: the reference grid _q_ref ends at 15.0 GeV^2, but the analysis uses
    a bin centred at 18.0 GeV^2. With fill_value='extrapolate', that point is a
    blind cubic extrapolation returning -0.034, and it contributes 19.7 units
    of chi2 -- 28% of the headline chi2_SM.
    """
    q_ref = utils._q_ref
    assert q_ref.min() <= ANALYSIS_BIN_CENTRES.min(), "grid starts above the first bin"
    assert q_ref.max() >= ANALYSIS_BIN_CENTRES.max(), (
        f"reference grid ends at {q_ref.max()} GeV^2 but the analysis needs "
        f"{ANALYSIS_BIN_CENTRES.max()} GeV^2; predictions there are extrapolated"
    )


def test_sm_high_q2_anchor(utils):
    """P5'_SM in 15-17 GeV^2 is close to -0.67, not -0.12.

    An anchor against an externally known value. The high-q^2 region is
    theoretically clean (OPE in 1/m_b), so the SM prediction there is not
    controversial and the measurement agrees with it.
    """
    value = utils.p5p_sm(16.0)
    assert value == pytest.approx(-0.67, abs=0.15), (
        f"P5'_SM(16 GeV^2) = {value:+.3f}; the accepted SM value is ~-0.67"
    )


def test_sm_curve_matches_flavio(utils, flavio_sm):
    """The hard-coded curve must agree with a real SM calculation.

    Tolerance is 3x the flavio theory uncertainty per bin, which is generous.
    """
    central, sigma = flavio_sm
    hard_coded = utils.p5p_sm(ANALYSIS_BIN_CENTRES)
    deviations = np.abs(hard_coded - central) / sigma
    worst = int(np.argmax(deviations))
    assert np.all(deviations < 3.0), (
        f"worst bin {ANALYSIS_BINS[worst]}: hard-coded {hard_coded[worst]:+.3f} "
        f"vs flavio {central[worst]:+.3f} +- {sigma[worst]:.3f} "
        f"({deviations[worst]:.1f} theory sigma)"
    )


def test_bin_integrated_not_bin_centre(utils):
    """P5' must be integrated over each bin, not evaluated at its centre.

    P5' is a ratio of integrals of angular coefficients, so <P5'>_bin differs
    from P5'(q^2_centre) -- materially so for the 4 GeV-wide bins used here
    (15-19 in Table 1, and the 11-12.5 / 17-19 bins where the curve turns).

    DEFECT: utils.p5p_sm takes a scalar q^2 and has no notion of a bin. This
    test documents a missing capability rather than a wrong number.
    """
    import inspect

    params = set(inspect.signature(utils.p5p_sm).parameters)
    assert {"q2min", "q2max"} <= params, (
        f"p5p_sm accepts {sorted(params)}; a bin-integrated interface "
        "(q2min, q2max) is required"
    )


def test_anomaly_has_the_sign_reported_in_the_literature(utils, observed):
    """In 4-8 GeV^2 the measurement must lie *above* the SM prediction.

    This is the actual, well-documented P5' anomaly: LHCb measures P5' larger
    (less negative) than the SM in the 4-6 and 6-8 GeV^2 bins.

    DEFECT: with the hard-coded curve the data fall *below* the SM in those
    bins, inverting the sign of the discrepancy. This is why the fit prefers
    kappa = +1.44, which shifts P5' downwards -- the opposite of the direction
    global fits require (they prefer Delta C9 < 0).
    """
    q2, values, _ = observed
    mask = (q2 > 4.0) & (q2 < 9.0)
    residual = values[mask] - utils.p5p_sm(q2[mask])
    assert np.all(residual > 0), (
        f"data-minus-SM in the anomaly bins is {residual.round(3)}; "
        "the measured P5' is known to sit above the SM there"
    )


def test_theory_uncertainty_enters_the_chi2(utils):
    """The SM prediction must carry an uncertainty.

    DEFECT: the chi2 in scripts/fit_usf.py divides by the experimental error
    only. Form-factor and charm-loop uncertainties on P5' are 0.03-0.11 in the
    bins used here -- comparable to the experimental errors in the high-q^2
    bins -- so omitting them inflates every significance quoted.
    """
    assert hasattr(utils, "p5p_sm_uncertainty"), (
        "utils exposes no SM theory uncertainty; chi2 is computed with "
        "experimental errors alone"
    )
