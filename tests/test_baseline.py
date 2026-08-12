# tests/test_baseline.py
"""The null-benchmark tests.

A new model must be compared against a naive baseline, not only against the
Standard Model. The relevant baseline here is a plain constant shift in C9,
which the b->s l l literature has fitted for a decade.

If the USF cannot be distinguished from that baseline, its geometric content
is empty regardless of how large Delta chi2 is. This is the single check whose
absence allowed the result to stand.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def _chi2(pred, values, errors):
    return float((((values - pred) / errors) ** 2).sum())


def test_usf_is_distinguishable_from_a_constant_shift(utils, observed):
    """The USF must not be numerically identical to a constant C9 shift.

    DEFECT: because f_geo == 1, P5'_USF(q^2, kappa) = P5'_SM(q^2) - 0.3*kappa
    exactly. Fitting a plain constant offset recovers the same kappa to 8
    significant figures and the same chi2 to 12. The "geometric coupling"
    contributes nothing that a single constant does not.
    """
    q2, values, errors = observed
    sm = utils.p5p_sm(q2)

    usf = minimize_scalar(
        lambda k: _chi2(utils.p5p_usf(k, q2), values, errors),
        bounds=(-2, 2), method="bounded",
    )
    constant = minimize_scalar(
        lambda k: _chi2(sm - 0.3 * k, values, errors),
        bounds=(-2, 2), method="bounded",
    )

    assert abs(usf.fun - constant.fun) > 0.5, (
        f"USF chi2 = {usf.fun:.6f}, constant-shift chi2 = {constant.fun:.6f}; "
        "the two models are the same model"
    )


def test_delta_chi2_survives_a_correct_sm_curve(utils, observed, flavio_sm):
    """Most of the claimed Delta chi2 must not be an artefact of the SM curve.

    The manuscript reports chi2_SM = 70.4. Recomputed against flavio with
    experimental and theory uncertainties, chi2_SM = 17.2 for 7 dof (p = 0.016),
    which is the ~2-3 sigma tension the literature reports -- not a 7.5 sigma
    rejection.
    """
    q2, values, errors = observed
    central, sigma = flavio_sm
    total = np.hypot(errors, sigma)

    chi2_hard_coded = _chi2(utils.p5p_sm(q2), values, errors)
    chi2_flavio = _chi2(central, values, total)

    assert chi2_flavio > 0.5 * chi2_hard_coded, (
        f"chi2_SM collapses from {chi2_hard_coded:.1f} (hard-coded curve, "
        f"experimental errors only) to {chi2_flavio:.1f} (flavio, exp+theory); "
        "the headline discrepancy is an artefact of the SM reference"
    )


def test_significance_is_not_taken_as_sqrt_delta_chi2(utils):
    """sqrt(Delta chi2) is an asymptotic result that must be validated.

    DEFECT: scripts/fit_usf.py reports Delta chi2 and the manuscript converts
    it to 7.5 sigma via sqrt(56.5). Wilks' theorem is asymptotic; with seven
    points and a bounded prior on kappa it needs a toy-MC check, and a p-value
    should be quoted rather than a sigma count.
    """
    assert hasattr(utils, "significance_from_toys"), (
        "no toy-MC calibration of the test statistic is available; "
        "significance is being read off sqrt(Delta chi2)"
    )
