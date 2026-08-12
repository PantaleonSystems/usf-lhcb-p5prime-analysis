# tests/test_geometry.py
"""Tests on the geometric effective coupling f_geo(q^2).

The manuscript makes four checkable claims about f_geo:

  1. it is larger than 1 at low q^2 and decays to 1 at high q^2
     ("the shape is fixed by the theory");
  2. the product G_LQG * R_AdS is "a constant of order unity";
  3. the hyperbolic tangent "saturates the correction at LHC energies";
  4. the Standard Model is recovered for kappa = 0.

Claims 1-3 are false for the code as written. Claim 3 is also false as
*physics* -- tanh(E_cms^2/E_P^2) is ~1e-30, as the referee notes -- so the
test below asserts the physically correct value, not the manuscript's claim.
"""
import numpy as np
import pytest

from conftest import E_PLANCK_GEV

Q2_GRID = np.linspace(0.1, 19.0, 60)


def test_sm_limit_is_recovered(utils):
    """kappa = 0 must reproduce the SM prediction exactly. (Manuscript claim 4.)"""
    q2 = np.array([1.8, 5.0, 18.0])
    np.testing.assert_allclose(utils.p5p_usf(0.0, q2), utils.p5p_sm(q2), rtol=0, atol=0)


def test_f_geo_actually_varies_with_q2(utils):
    """f_geo must depend on q^2, otherwise the model is a constant offset.

    DEFECT: f_geo(q^2) == 1.0 identically at double precision. Two unit errors
    (see the two tests below) leave f_geo - 1 ~ 1.6e-33, which underflows the
    ~1e-16 resolution of a float near 1.0. The fitted model therefore reduces
    to P5' = P5'_SM(q^2) - 0.3*kappa, a one-parameter constant shift with no
    geometric content whatsoever.
    """
    f_geo = utils.geometric_factor_usf(Q2_GRID)
    assert np.ptp(f_geo) > 1e-3, (
        f"f_geo is constant at {f_geo[0]!r} across q^2 in [0.1, 19] GeV^2; "
        "the geometric structure is numerically inert"
    )


def test_f_geo_decreases_with_q2(utils):
    """f_geo must be largest at low q^2, where the anomaly sits. (Claim 1.)"""
    f_geo = utils.geometric_factor_usf(Q2_GRID)
    assert np.all(np.diff(f_geo) < 0), "f_geo is not monotonically decreasing in q^2"
    assert f_geo[0] > f_geo[-1] * 1.05, "no meaningful low- to high-q^2 contrast"


def test_planck_energy_is_expressed_in_gev(utils):
    """The Planck energy used in the tanh must be ~1.22e19 GeV.

    DEFECT: utils.E_P is in joules (1.956e9 J). geometric_factor_usf divides it
    by 1e9 and uses the result as if it were GeV, giving 1.956 GeV instead of
    1.22e19 GeV -- an error of ~1e19.
    """
    e_planck_as_used_gev = utils.E_P / 1e9
    assert e_planck_as_used_gev == pytest.approx(E_PLANCK_GEV, rel=0.01), (
        f"E_P is used as {e_planck_as_used_gev:.3g} GeV; expected "
        f"{E_PLANCK_GEV:.3g} GeV (utils.E_P is stored in joules)"
    )


def test_planck_suppression_is_not_saturated(utils):
    """tanh(E_cms^2/E_P^2) must be ~1.3e-30 at 14 TeV, not ~1.

    This asserts the *physically correct* value. The manuscript claims the
    tanh saturates; it does not. The current code returns 1.0 only because of
    the joule/GeV error above, so this test fails in the opposite direction
    from the referee's complaint: the code accidentally reproduces the
    manuscript's (incorrect) claim.
    """
    e_cms_gev = 14e3
    expected = np.tanh((e_cms_gev / E_PLANCK_GEV) ** 2)
    as_used = np.tanh((e_cms_gev ** 2) / (utils.E_P / 1e9) ** 2)
    assert as_used == pytest.approx(expected, rel=0.01), (
        f"Planck activation evaluates to {as_used:.3g}; the correct value at "
        f"E_cms = 14 TeV is {expected:.3g}"
    )


def test_geometric_prefactor_is_dimensionless_and_order_unity(utils):
    """G_LQG * R_AdS must be dimensionless and O(1). (Manuscript claim 2.)

    DEFECT: |lqg_operator()| is sqrt(quantised area), i.e. a *length*
    (4.27e-35 m), and R_AdS = L_AdS^2/alpha' is dimensionless (38.3). Their
    product is 1.63e-33 metres -- neither dimensionless nor order unity.
    """
    prefactor = np.abs(utils.lqg_operator()) * (utils.L_AdS ** 2 / utils.alpha_prime)
    assert 0.01 < prefactor < 100.0, (
        f"G_LQG * R_AdS = {prefactor:.3g} (carries dimension of length); "
        "the manuscript asserts a dimensionless constant of order unity"
    )


def test_collider_energy_matches_the_fitted_dataset(utils):
    """f_geo must not be evaluated at an energy the dataset was not taken at.

    DEFECT: geometric_factor_usf defaults to collision_energy=14e3 GeV, but the
    fitted data are LHCb Run-1 at sqrt(s) = 7 and 8 TeV. (Referee point 2.)
    Beyond the mismatch, no EFT justification is given for low-energy Wilson
    coefficients depending on the pp collider energy at all.
    """
    import inspect

    default = inspect.signature(utils.geometric_factor_usf).parameters[
        "collision_energy"
    ].default
    assert default <= 8.0e3, (
        f"default collision_energy is {default} GeV; the Run-1 dataset was "
        "collected at 7 and 8 TeV"
    )
