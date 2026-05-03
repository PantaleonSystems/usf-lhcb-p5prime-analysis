# scripts/plot_p5p.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from utils import p5p_sm, p5p_usf, geometric_factor_usf

plt.rcParams.update({'font.size': 12})

# ============================================================================
# Load experimental data and fit results
# ============================================================================
df = pd.read_csv("data/p5p_observables.csv")
q2 = df['q2_center'].values
obs = df['value'].values
err = df['error'].values

with open("results/fit_results.json") as f:
    kappa = json.load(f)['kappa_best']

# Predictions
sm_pred = p5p_sm(q2)
usf_pred = p5p_usf(kappa, q2)

# ============================================================================
# Figure 1: Spectrum (data vs. SM vs. USF)
# ============================================================================
plt.figure(figsize=(7,5))
plt.errorbar(q2, obs, yerr=err, fmt='o', color='red', capsize=3,
             label='LHCb data (P5\')')
plt.plot(q2, sm_pred, 'b--', label='SM (literature)')
plt.plot(q2, usf_pred, 'g-', linewidth=2, label=f'USF ($\\kappa={kappa:.3f}$)')
plt.xlabel('$q^2$ (GeV$^2$)')
plt.ylabel("$P_5'$")
plt.title("Spectrum of $P_5'$: data vs. SM vs. USF")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("results/spectrum_p5p.pdf")
plt.savefig("results/spectrum_p5p.png", dpi=150)

# ============================================================================
# Figure 2: Normalised residuals (SM vs. USF)
# ============================================================================
res_sm = (obs - sm_pred) / err
res_usf = (obs - usf_pred) / err

plt.figure(figsize=(9,5))  # wider to avoid x‑axis crowding
plt.errorbar(q2, res_sm, yerr=0.05, fmt='s', color='blue', 
             label='SM', alpha=0.7)
plt.errorbar(q2, res_usf, yerr=0.05, fmt='o', color='green', 
             label='USF', alpha=0.7)
plt.axhline(0, color='black', linewidth=1)
plt.axhline(2, linestyle='--', color='gray')
plt.axhline(-2, linestyle='--', color='gray')
plt.xlabel('$q^2$ (GeV$^2$)')
plt.ylabel('Residual ($\\sigma$)')
plt.title("Normalised residuals: SM vs. USF")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
# Set x-axis limits to avoid cutting off the first/last point labels
plt.xlim(min(q2)-0.5, max(q2)+0.5)
plt.tight_layout()
plt.savefig("results/residuals_p5p.pdf")
plt.savefig("results/residuals_p5p.png", dpi=150)

# ============================================================================
# Figure 3: Geometric coupling factor f_geo(q²)
# ============================================================================
q2_fine = np.linspace(0, 20, 300)
f_geo = geometric_factor_usf(q2_fine)

plt.figure(figsize=(6,4))
plt.plot(q2_fine, f_geo, 'purple', linewidth=2)
plt.xlabel('$q^2$ (GeV$^2$)')
plt.ylabel(r'$f_{\rm geo}(q^2)$')
plt.title('Geometric coupling (LQG + AdS)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.axhline(1, color='gray', linestyle='--', alpha=0.5)
plt.text(12, 1.02, 'High $q^2$: $f \\approx 1$', fontsize=9)
plt.text(2, 1.08, 'Anomaly region: $f > 1$', fontsize=9, color='purple')
plt.tight_layout()
plt.savefig('results/fator_geometrico.pdf')
plt.savefig('results/fator_geometrico.png', dpi=150)

# ============================================================================
# Figure 4: HL-LHC projection (errors reduced by factor 5)
# ============================================================================
err_hllhc = err / 5
obs_hllhc = obs   # central values unchanged (same as current data)

plt.figure(figsize=(7,5))
# Current LHCb data (semi‑transparent)
plt.errorbar(q2, obs, yerr=err, fmt='ro', capsize=3, alpha=0.4,
             label='LHCb current')
# Projected HL-LHC data (smaller errors, red squares)
plt.errorbar(q2, obs_hllhc, yerr=err_hllhc, fmt='rs', capsize=3,
             markersize=6, label='HL-LHC (projected)')
plt.plot(q2, sm_pred, 'b--', label='SM')
plt.plot(q2, usf_pred, 'g-', linewidth=2, label=f'USF ($\\kappa={kappa:.3f}$)')
plt.xlabel('$q^2$ (GeV$^2$)')
plt.ylabel("$P_5'$")
plt.title('Future test: HL-LHC (5× smaller errors)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("results/hllhc_projection.pdf")
plt.savefig("results/hllhc_projection.png", dpi=150)

print("All figures saved in results/")