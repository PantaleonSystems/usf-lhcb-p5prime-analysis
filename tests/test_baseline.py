# tests/test_baseline.py
"""The null-benchmark tests.

A new model must be compared against a naive baseline, not only against the
Standard Model. The relevant baseline here is a free constant shift in C9,
which the b -> s l l literature has fitted for a decade.

If the USF cannot be distinguished from that baseline, its geometric content is
empty regardless of how large Delta chi2 against the SM looks. The absence of
this single comparison is what allowed the published result to stand.
"""
import json

import numpy as np
import pytest

from conftest import RESULTS_DIR

#: Published claims, for comparison against what the corrected pipeline gives.
PUBLISHED_CHI2_SM = 70.4
PUBLISHED_KAPPA = 1.44
PUBLISHED_SIGNIFICANCE = 7.5


@pytest.fixture(scope="module")
def fit():
    path = RESULTS_DIR / "fit_results.json"
    if not path.exists():
        pytest.skip("run scripts/fit_usf.py first")
    with open(path) as fh:
        return json.load(fh)


@pytest.mark.xfail(
    strict=True,
    reason="RESULT, not a bug: with f_geo == 1 the USF differs from a free "
           "Delta C9 only by the imposed Delta C10 = -0.2 Delta C9 ratio, which "
           "P5' is insensitive to. It lands marginally worse at equal parameter "
           "count. Turning this green requires a geometric factor that actually "
           "varies with q^2 -- see test_f_geo_actually_varies_with_q2.",
)
def test_usf_does_not_beat_a_free_c9(fit):
    """The USF must improve on a plain constant C9 shift to have content.

    It does not. With f_geo == 1 the two models differ only by the imposed
    Delta C10 = -0.2 Delta C9 ratio, which P5' is insensitive to, so the USF
    lands marginally *worse* than the one-parameter baseline at equal cost.
    """
    gain = fit["comparisons"]["delta_chi2_c9_minus_usf"]
    assert gain > 1.0, (
        f"USF improves on the free-C9 baseline by Delta chi2 = {gain:+.3f} "
        "at the same parameter count; it is not a distinct model"
    )


@pytest.mark.xfail(
    strict=True,
    reason="RESULT, not a bug: freeing the f_geo amplitude buys only 0.216 in "
           "chi2 for one extra parameter, so AIC worsens. Seven Run-1 bins do "
           "not ask for q^2 dependence in Delta C9. This is the measurement "
           "that decides whether the phenomenological salvage route is viable.",
)
def test_q2_shape_earns_its_parameter(fit):
    """Freeing the f_geo amplitude must justify the extra parameter.

    The USF ansatz f_geo = 1 + A/(1 + q^2/q0^2) with A free gains only 0.216 in
    chi2 for one more parameter, so AIC gets worse (9.01 against 7.23). The
    data do not ask for q^2 dependence in Delta C9.
    """
    models = fit["models"]
    assert models["USF ansatz (kappa, A)"]["aic"] < models["USF (kappa)"]["aic"], (
        f"AIC {models['USF ansatz (kappa, A)']['aic']:.2f} against "
        f"{models['USF (kappa)']['aic']:.2f}; the q^2 shape does not pay for itself"
    )


def test_chi2_sm_is_not_the_published_value(fit):
    """chi2_SM must not reproduce 70.4, which came from the wrong SM curve."""
    chi2_sm = fit["models"]["SM"]["chi2"]
    assert chi2_sm < 0.5 * PUBLISHED_CHI2_SM, (
        f"chi2_SM = {chi2_sm:.2f} against the published {PUBLISHED_CHI2_SM}"
    )
    # The SM is in mild tension with the data, which is the known anomaly.
    assert 0.01 < fit["models"]["SM"]["p_goodness_of_fit"] < 0.2


def test_significance_is_toy_calibrated_not_sqrt_delta_chi2(fit):
    """Significance must come from pseudo-experiments, not sqrt(Delta chi2)."""
    significance = fit["significance"]
    assert significance["n_toys"] >= 1000
    assert "p_value_toys" in significance

    # Here Wilks happens to hold; the point is that it was checked.
    assert significance["significance_toys"] == pytest.approx(
        significance["significance_wilks_naive"], abs=0.5
    )
    assert significance["significance_toys"] < 0.5 * PUBLISHED_SIGNIFICANCE, (
        f"{significance['significance_toys']:.2f} sigma against the published "
        f"{PUBLISHED_SIGNIFICANCE} sigma"
    )


def test_preferred_shift_has_the_sign_the_literature_reports(fit):
    """The data prefer Delta C9 < 0, opposite to the published kappa = +1.44.

    The published fit found kappa = +1.44 because the incorrect SM curve
    inverted the sign of the discrepancy. Against a correct SM the sign flips.
    """
    kappa = fit["posterior"]["kappa_median"]
    assert kappa < 0, f"kappa = {kappa:+.3f}"
    assert fit["posterior"]["kappa_upper"] < 0, "zero is inside the credible interval"

    assert np.sign(kappa) != np.sign(PUBLISHED_KAPPA)

    c9 = fit["models"]["Delta C9 free"]["best_fit"][0]
    assert -2.5 < c9 < -0.5, (
        f"Delta C9 = {c9:+.3f}; global fits report roughly -1"
    )
