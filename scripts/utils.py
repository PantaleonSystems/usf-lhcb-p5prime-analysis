# scripts/utils.py
"""The USF geometric effective coupling, with its units made explicit.

This module now holds only the geometry. The P5' predictions that used to live
here moved to scripts/sm_predictions.py (Standard Model) and scripts/response.py
(Wilson-coefficient dependence), both of which call real theory codes instead of
interpolating hard-coded numbers.

Two unit errors in the previous version cancelled to within thirty orders of
magnitude of each other, which is why they went unnoticed:

  * ``E_P`` held the Planck energy in *joules* (1.956e9 J) and was divided by
    1e9 as though converting to GeV, giving 1.956 GeV instead of 1.22e19 GeV.
    That inflated the tanh argument by ~1e38 and made it saturate at 1.0.
  * ``G_LQG * R_AdS`` is 1.63e-33 and carries a dimension of *length*, while
    the manuscript describes it as a dimensionless constant of order unity.

Net effect: ``f_geo - 1 ~ 1.6e-33``, which underflows against 1.0 in double
precision, so ``f_geo`` evaluated to exactly 1 and the model reduced to a
constant shift in C9. With the units corrected the suppression is worse, not
better -- the honest amplitude is ~7e-64 -- so ``f_geo`` is still 1 to machine
precision. The geometric factor cannot produce an order-one effect at collider
energies, which is the referee's central objection and is correct.

The amplitude is therefore exposed as a parameter. Passing the theory value
gives no effect; freeing it turns f_geo into a phenomenological ansatz whose
normalisation is fitted rather than predicted. scripts/fit_usf.py fits both and
reports that the data do not ask for the q^2 shape either way.
"""
from __future__ import annotations

import numpy as np

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================
HBAR_SI = 1.0545718e-34          # J s
C_SI = 299792458.0               # m/s
G_SI = 6.67430e-11               # m^3 kg^-1 s^-2
GEV_IN_JOULES = 1.602176634e-10  # J/GeV

#: Planck length in metres, 1.616e-35 m.
PLANCK_LENGTH_M = np.sqrt(HBAR_SI * G_SI / C_SI ** 3)

#: Planck energy in GeV, 1.221e19 GeV. The quantity the tanh needs.
PLANCK_ENERGY_GEV = np.sqrt(HBAR_SI * C_SI ** 5 / G_SI) / GEV_IN_JOULES

# ---------------------------------------------------------------------------
# USF parameters
# ---------------------------------------------------------------------------
IMMIRZI_GAMMA = 0.37
L_ADS_M = 1e-34                       # m
ALPHA_PRIME_M2 = PLANCK_LENGTH_M ** 2  # string length squared, m^2

#: LHCb Run-1 collision energies in GeV. The fitted dataset is 7 and 8 TeV.
#: The published analysis used 14 TeV, the LHC design energy, which is neither
#: the energy of these data nor -- absent an effective-theory argument the
#: manuscript does not give -- obviously relevant to a low-energy Wilson
#: coefficient at all.
RUN1_ENERGIES_GEV = (7.0e3, 8.0e3)
DEFAULT_COLLISION_ENERGY_GEV = 8.0e3

#: Natural scale of the decay, q0 = m_B.
M_B_GEV = 5.279
Q0_SQUARED_GEV2 = M_B_GEV ** 2  # ~27.87 GeV^2


# ============================================================================
# LQG + AdS GEOMETRIC FUNCTIONS
# ============================================================================

def lqg_operator(j_v: float = 0.5, K_v: float = 1.0) -> complex:
    """Loop Quantum Gravity area-quantisation operator.

    Returns sqrt(quantised area) times a phase, so the modulus carries a
    dimension of **length** (~4.3e-35 m). This is the quantity the manuscript
    calls G_LQG and describes as dimensionless; it is not.
    """
    quantised_area_m2 = (
        8 * np.pi * IMMIRZI_GAMMA * PLANCK_LENGTH_M ** 2 * j_v * (j_v + 1)
    )
    return np.sqrt(quantised_area_m2) * np.exp(1j * IMMIRZI_GAMMA * K_v)


def ads_curvature() -> float:
    """R_AdS = L_AdS^2 / alpha', dimensionless, ~38.3."""
    return L_ADS_M ** 2 / ALPHA_PRIME_M2


def planck_activation(collision_energy_gev: float = DEFAULT_COLLISION_ENERGY_GEV) -> float:
    """tanh(E_cms^2 / E_P^2), the factor the manuscript says saturates.

    It does not saturate. At Run-1 energies it is ~4e-31, and even at the 14 TeV
    design energy only ~1.3e-30. Both are indistinguishable from their argument,
    since tanh(x) = x for x this small.
    """
    return float(np.tanh((collision_energy_gev / PLANCK_ENERGY_GEV) ** 2))


def geometric_amplitude(
    collision_energy_gev: float = DEFAULT_COLLISION_ENERGY_GEV,
) -> float:
    """The amplitude the theory predicts for the geometric correction.

    G_LQG * R_AdS * tanh(E_cms^2/E_P^2). With the manuscript's own constants
    this is ~7e-64 at Run-1 energies. For the correction to reach order unity,
    G_LQG * R_AdS would have to be ~1e30 rather than ~1, at which point the
    normalisation is a free phenomenological parameter and the framework makes
    no prediction about the size of the effect.
    """
    return (
        float(np.abs(lqg_operator()))
        * ads_curvature()
        * planck_activation(collision_energy_gev)
    )


def geometric_factor_usf(
    q2,
    collision_energy: float = DEFAULT_COLLISION_ENERGY_GEV,
    amplitude: float | None = None,
):
    """Geometric effective coupling f_geo(q^2) = 1 + A / (1 + q^2/q0^2).

    Parameters
    ----------
    q2:
        Dilepton invariant mass squared, in GeV^2.
    collision_energy:
        Centre-of-mass energy in GeV, used only when ``amplitude`` is None.
        Defaults to the Run-1 energy of the fitted dataset.
    amplitude:
        Overall size A of the correction. ``None`` uses the value the theory
        predicts, which is ~7e-64 and therefore returns exactly 1.0 in double
        precision. Pass a value explicitly to treat f_geo as a phenomenological
        ansatz with a fitted normalisation.
    """
    if amplitude is None:
        amplitude = geometric_amplitude(collision_energy)
    return 1.0 + amplitude / (1.0 + np.asarray(q2, dtype=float) / Q0_SQUARED_GEV2)


if __name__ == "__main__":
    print(f"Planck length            {PLANCK_LENGTH_M:.4e} m")
    print(f"Planck energy            {PLANCK_ENERGY_GEV:.4e} GeV")
    print(f"|G_LQG|                  {np.abs(lqg_operator()):.4e} m   (a length)")
    print(f"R_AdS                    {ads_curvature():.4e}      (dimensionless)")
    print(f"G_LQG * R_AdS            {np.abs(lqg_operator()) * ads_curvature():.4e} m")
    print()
    for energy in (*RUN1_ENERGIES_GEV, 14.0e3):
        print(f"tanh at {energy / 1e3:>4.0f} TeV        "
              f"{planck_activation(energy):.4e}")
    print()
    print(f"Predicted amplitude      {geometric_amplitude():.4e}")

    q2 = np.array([1.8, 5.0, 11.75, 18.0])
    print(f"\nf_geo at theory amplitude: {geometric_factor_usf(q2)}")
    print(f"f_geo - 1                : {geometric_factor_usf(q2) - 1}")
    print(f"\nf_geo with A = 1 (ansatz): {geometric_factor_usf(q2, amplitude=1.0)}")
