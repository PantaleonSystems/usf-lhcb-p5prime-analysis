# tests/test_response.py
"""Tests on the P5' response to the Wilson coefficients.

The published analysis wrote

    P5'(q^2, kappa) = P5'_SM(q^2) - 0.3 * kappa * f_geo(q^2)

taking dP5'/dC9 = -0.3 as a constant and dP5'/dC10 = 0. Both the constancy and
the value are wrong, and the error compounds with the SM-curve error: the true
slope is largest exactly where the anomaly is (low q^2) and nearly vanishes at
high q^2, whereas the hard-coded SM curve placed its fake discrepancy at high
q^2. Assuming a constant slope let those high-q^2 bins drive the fit.
"""
import numpy as np
import pytest

from conftest import ANALYSIS_BINS


@pytest.fixture(scope="module")
def response():
    pytest.importorskip("flavio")
    pytest.importorskip("wilson")
    from response import P5pResponse

    return P5pResponse()


def test_sm_limit_matches_sm_predictions(response, sm):
    """Zero Wilson coefficient shifts must reproduce the SM prediction."""
    np.testing.assert_allclose(
        response(0.0, 0.0), sm.p5p_sm_binned(ANALYSIS_BINS), atol=2e-3
    )


def test_interpolator_matches_exact_flavio(response):
    """The interpolated grid must agree with exact calls.

    The published analysis approximated the response and never validated the
    approximation. This check exists so the same mistake is not repeated one
    level down, where it would be harder to spot.
    """
    worst = response.validate(n_points=25)
    assert worst < 1e-3, f"worst interpolation deviation is {worst:.2e}"


def test_slope_is_not_constant_across_bins():
    """dP5'/dC9 varies by more than an order of magnitude across the bins.

    Measured at the SM point: -0.278 in 2.5-4 GeV^2 down to -0.020 in 17-19.
    The analysis used -0.3 everywhere, which is roughly right in the first two
    bins and wrong by a factor of ~14 in the last three.
    """
    from response import local_slope

    slopes = np.abs(local_slope())
    assert slopes.max() / slopes.min() > 5.0, (
        f"slopes {slopes.round(3)} do not vary enough to matter"
    )
    assert abs(slopes[-1]) < 0.1, (
        f"high-q^2 slope is {slopes[-1]:.3f}; the analysis assumed -0.3, which "
        "gives those bins far more leverage on kappa than they have"
    )


def test_response_is_not_linear_in_dc9(response):
    """A single slope cannot describe the range the fit explores.

    In 4-6 GeV^2: Delta C9 from 0 to -1 shifts P5' by +0.232, but from -1 to -2
    by a further +0.352. The response is also asymmetric about zero.
    """
    index = ANALYSIS_BINS.index((4.0, 6.0))
    at = {d: response(d, 0.0)[index] for d in (-2.0, -1.0, 0.0, 1.0)}

    first = at[-1.0] - at[0.0]
    second = at[-2.0] - at[-1.0]
    assert abs(second - first) > 0.05, (
        f"successive unit steps in Delta C9 give {first:+.3f} and {second:+.3f}; "
        "a constant slope would give equal steps"
    )

    forward = at[1.0] - at[0.0]
    assert abs(abs(forward) - abs(first)) > 0.05, "response is symmetric about zero"


def test_dc10_dependence_is_weak_but_present(response):
    """dP5'/dC10 was set to exactly zero; check that is a fair approximation.

    Unlike the other approximations this one holds up, and the test records
    that it was checked rather than assumed.
    """
    base = response(0.0, 0.0)
    shifted = response(0.0, 1.0)
    assert np.max(np.abs(shifted - base)) < 0.15, (
        "C10 dependence is larger than the analysis assumed"
    )


def test_negative_dc9_moves_p5p_towards_the_data(response, observed):
    """The data prefer Delta C9 < 0, the direction global fits report.

    The published fit preferred kappa = +1.44, i.e. Delta C9 > 0, because the
    incorrect SM curve inverted the sign of the discrepancy.
    """
    _, values, _ = observed
    index = ANALYSIS_BINS.index((4.0, 6.0))
    measured = values[index]

    sm_value = response(0.0, 0.0)[index]
    negative = response(-1.0, 0.2)[index]
    positive = response(+1.0, -0.2)[index]

    assert abs(negative - measured) < abs(sm_value - measured), (
        "Delta C9 < 0 should improve agreement in the anomaly bin"
    )
    assert abs(positive - measured) > abs(sm_value - measured), (
        "Delta C9 > 0, the published direction, should make it worse"
    )
