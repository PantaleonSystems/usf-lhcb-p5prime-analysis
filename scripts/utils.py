# scripts/utils.py
import numpy as np
from scipy.interpolate import interp1d

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================
hbar = 1.0545718e-34      # J·s
c = 299792458             # m/s
G = 6.67430e-11           # m³/kg/s²
ell_P = np.sqrt(hbar * G / c**3)          # Planck length (1.616e-35 m)
E_P = np.sqrt(hbar * c**5 / G)            # Planck energy (1.956e9 J = 1.22e19 GeV)

# USF parameters
gamma_Immirzi = 0.37
L_AdS = 1e-34             # m
alpha_prime = ell_P**2    # ℓ_s²

# Standard Model Wilson coefficients
C9_SM = 4.1
C10_SM = -4.2

# Natural scale of the B -> K* mu mu decay
m_B = 5.279                # GeV (B meson mass)
q0_2 = m_B**2              # ≈ 27.86 GeV²

# ============================================================================
# LQG + AdS GEOMETRIC FUNCTIONS
# ============================================================================
def lqg_operator(j_v=0.5, K_v=1.0):
    """Loop Quantum Gravity operator (area quantization)."""
    quantized_area = 8 * np.pi * gamma_Immirzi * ell_P**2 * j_v * (j_v + 1)
    return np.sqrt(quantized_area) * np.exp(1j * gamma_Immirzi * K_v)

def geometric_factor_usf(q2, collision_energy=14e3):
    """
    Quantum‑geometric coupling that modifies the decay amplitude.
    - q2 in GeV²
    - collision_energy in GeV (14 TeV = 14000 GeV)
    """
    G_LQG = np.abs(lqg_operator())
    R_AdS = (L_AdS**2) / alpha_prime
    energy_scale = (collision_energy**2) / (E_P / 1e9)**2   # dimensionless
    planck_activation = np.tanh(energy_scale)
    # Natural scale q0_2 = m_B^2 (avoid ad‑hoc tuning)
    f_geo = 1 + G_LQG * R_AdS * planck_activation / (1 + q2 / q0_2)
    return f_geo

# ============================================================================
# AMPLITUDE AND BRANCHING RATIOS (kept for completeness, not used for P5')
# ============================================================================
def usf_amplitude(q2, kappa):
    f_geo = geometric_factor_usf(q2)
    delta_C9 = kappa * f_geo
    delta_C10 = -0.2 * kappa * f_geo
    C9_eff = C9_SM + delta_C9
    C10_eff = C10_SM + delta_C10
    return C9_eff**2 + C10_eff**2

def branching_ratio_sm(q2):
    return 1e-7 * (1 - 0.1 * (q2 - 10)**2 / 100)

def branching_ratio_usf(q2, kappa, norm=1e-7):
    amp2 = usf_amplitude(q2, kappa)
    amp2_sm = usf_amplitude(q2, 0.0)
    return norm * (amp2 / amp2_sm) * branching_ratio_sm(q2)

# ============================================================================
# USF PREDICTION FOR P_5'
# ============================================================================

# SM values from Fig. 5 of arXiv:1505.07814
_q_ref = np.array([1.55, 2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 15.0])
_p5_sm_ref = np.array([0.12, 0.08, 0.04, 0.01, -0.02, -0.05, -0.09, -0.14])
_p5_sm_interp = interp1d(_q_ref, _p5_sm_ref, kind='cubic', fill_value='extrapolate')

def p5p_sm(q2):
    """Standard Model prediction for P5' (interpolated from literature)."""
    if np.isscalar(q2):
        return float(_p5_sm_interp(q2))
    else:
        return _p5_sm_interp(q2)

_dP5_dC9 = -0.3   # sensitivity dP5'/dC9 (constant approximation)

def delta_p5p(kappa, q2):
    """USF‑induced change in P5'."""
    f_geo = geometric_factor_usf(q2)
    return _dP5_dC9 * kappa * f_geo

def p5p_usf(kappa, q2):
    """Full USF prediction for P5'."""
    return p5p_sm(q2) + delta_p5p(kappa, q2)