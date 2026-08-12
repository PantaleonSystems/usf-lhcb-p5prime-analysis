# tests/test_geometry.py
"""Tests on the geometric effective coupling f_geo(q^2).

The manuscript makes four checkable claims:

  1. f_geo is larger than 1 at low q^2 and decays to 1 at high q^2
     ("the shape is fixed by the theory");
  2. G_LQG * R_AdS is "a constant of order unity";
  3. the hyperbolic tangent "saturates the correction at LHC energies";
  4. the Standard Model is recovered for kappa = 0.

Claims 2 and 3 are false as physics and cannot be repaired by fixing code, so
the tests asserting them are marked xfail(strict=True) with the measured
values. Claim 1 follows from them and fails for the same reason. Claim 4 holds.

The unit errors that hid all this *have* been repaired, and those tests now
pass: the Planck energy is in GeV, the collider energy matches the dataset, and
the dimensional structure is explicit rather than buried.
"""
import numpy as np
import pytest

from conftest import E_PLANCK_GEV

Q2_GRID = np.linspace(0.1, 19.0, 60)


# ---------------------------------------------------------------------------
# Repaired: unit and provenance errors
# ---------------------------------------------------------------------------

def test_planck_energy_is_expressed_in_gev(utils):
    """The Planck energy entering the tanh must be ~1.22e19 GeV.

    Previously utils.E_P held joules (1.956e9) and was divided by 1e9 as though
    that converted to GeV, giving 1.956 GeV -- an error of ~1e19 that made the
    tanh saturate spuriously.
    """
    assert utils.PLANCK_ENERGY_GEV == pytest.approx(E_PLANCK_GEV, rel=0.01)


def test_collider_energy_matches_the_fitted_dataset(utils):
    """f_geo must not be evaluated at an energy the dataset was not taken at.

    The published analysis used 14 TeV, the LHC design energy, while fitting
    Run-1 data taken at 7 and 8 TeV.
    """
    assert utils.DEFAULT_COLLISION_ENERGY_GEV in utils.RUN1_ENERGIES_GEV
    assert max(utils.RUN1_ENERGIES_GEV) <= 8.0e3


def test_dimensional_structure_is_explicit(utils):
    """Each geometric quantity must declare what it carries.

    G_LQG is sqrt(area), i.e. a length; R_AdS is dimensionless. Their product
    is a length, and the code says so rather than implying otherwise.
    """
    assert "length" in utils.lqg_operator.__doc__
    assert "dimensionless" in utils.ads_curvature.__doc__
    assert utils.ads_curvature() == pytest.approx(38.3, rel=0.01)


def test_planck_suppression_is_computed_correctly(utils):
    """tanh(E_cms^2/E_P^2) must evaluate to its true, tiny value.

    ~4.3e-31 at 8 TeV. The manuscript asserts this saturates; it does not, and
    the code no longer pretends otherwise.
    """
    assert utils.planck_activation(8.0e3) == pytest.approx(4.29e-31, rel=0.05)
    assert utils.planck_activation(14.0e3) == pytest.approx(1.31e-30, rel=0.05)


def test_amplitude_can_be_freed_for_a_phenomenological_fit(utils):
    """Overriding the amplitude must produce the shape the manuscript draws.

    With A = 1 the coupling runs from ~1.94 at low q^2 to ~1.61 at 18 GeV^2 --
    the falling curve of Figure 1. That figure was produced from a hard-coded
    illustrative amplitude, not from the theory value.
    """
    f_geo = utils.geometric_factor_usf(Q2_GRID, amplitude=1.0)
    assert np.all(np.diff(f_geo) < 0)
    assert f_geo[0] > 1.9
    assert f_geo[-1] < 1.7


def test_sm_limit_is_recovered(utils):
    """A zero amplitude leaves f_geo at exactly 1. (Manuscript claim 4.)"""
    np.testing.assert_array_equal(
        utils.geometric_factor_usf(Q2_GRID, amplitude=0.0), np.ones_like(Q2_GRID)
    )


# ---------------------------------------------------------------------------
# Not repairable: the physics claims themselves
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason="RESULT, not a bug: G_LQG * R_AdS = 1.63e-33 and carries a dimension "
           "of length, against the manuscript's 'dimensionless constant of order "
           "unity'. No code change fixes this -- it is what the stated "
           "construction evaluates to.",
)
def test_geometric_prefactor_is_dimensionless_and_order_unity(utils):
    """G_LQG * R_AdS must be dimensionless and O(1). (Manuscript claim 2.)"""
    prefactor = np.abs(utils.lqg_operator()) * utils.ads_curvature()
    assert 0.01 < prefactor < 100.0, f"G_LQG * R_AdS = {prefactor:.3g}"


@pytest.mark.xfail(
    strict=True,
    reason="RESULT, not a bug: the predicted amplitude is ~7e-64 at Run-1 "
           "energies (1.63e-33 length times a 4.3e-31 Planck suppression), so "
           "f_geo - 1 underflows against 1.0. This is the referee's central "
           "objection and it is correct. Turning it green requires either a "
           "different derivation or admitting the normalisation is fitted.",
)
def test_f_geo_actually_varies_with_q2(utils):
    """f_geo must depend on q^2, else the model is a constant offset.

    At the theory amplitude it does not, so the fitted model reduces to
    P5' = P5'_SM(q^2) - 0.3*kappa, with no geometric content. See
    tests/test_baseline.py::test_usf_does_not_beat_a_free_c9 for the
    consequence.
    """
    f_geo = utils.geometric_factor_usf(Q2_GRID)
    assert np.ptp(f_geo) > 1e-3, f"f_geo is constant at {f_geo[0]!r}"


@pytest.mark.xfail(
    strict=True,
    reason="RESULT, not a bug: follows from the amplitude being ~7e-64. The "
           "falling shape of Figure 1 exists only when the normalisation is "
           "supplied by hand.",
)
def test_f_geo_decreases_with_q2(utils):
    """f_geo must be largest at low q^2, where the anomaly sits. (Claim 1.)"""
    f_geo = utils.geometric_factor_usf(Q2_GRID)
    assert np.all(np.diff(f_geo) < 0)
    assert f_geo[0] > f_geo[-1] * 1.05
