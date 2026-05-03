# scripts/fit_usf.py
import numpy as np
import pandas as pd
import emcee
import corner
import json
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path
from utils import p5p_sm, p5p_usf

# =========================================================
# 1. LOAD REAL LHCb DATA (P5')
# =========================================================
data_file = Path("data/p5p_observables.csv")
df = pd.read_csv(data_file)
q2 = df['q2_center'].values
obs_p5p = df['value'].values      # measured P5'
err_p5p = df['error'].values

print(f"Loaded {len(q2)} q² bins (P5') from LHCb.")

# =========================================================
# 2. LIKELIHOOD FUNCTIONS
# =========================================================
def log_likelihood(kappa, q2, obs, err):
    pred = p5p_usf(kappa, q2)
    chi2 = np.sum(((obs - pred) / err)**2)
    return -0.5 * chi2

def log_prior(kappa):
    # Uniform prior between -2 and 2
    if -2.0 < kappa < 2.0:
        return 0.0
    return -np.inf

def log_posterior(kappa, q2, obs, err):
    lp = log_prior(kappa)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(kappa, q2, obs, err)

# =========================================================
# 3. BEST-FIT OPTIMIZATION (scipy)
# =========================================================
result = minimize(lambda k: -log_likelihood(k[0], q2, obs_p5p, err_p5p),
                  x0=[0.0], bounds=[(-2.0, 2.0)])
kappa_best = result.x[0]
print(f"Best-fit kappa: {kappa_best:.4f}")

# =========================================================
# 4. MCMC (emcee)
# =========================================================
ndim = 1
nwalkers = 32
nsteps = 2000

# Initial walker positions around the best-fit value
initial = np.random.normal(kappa_best, 0.01, size=(nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior,
                                args=(q2, obs_p5p, err_p5p))
sampler.run_mcmc(initial, nsteps, progress=True)

# Discard burn-in (first half) and flatten the chain
samples = sampler.get_chain(discard=nsteps//2, flat=True)
kappa_median = np.median(samples)
kappa_lower = np.percentile(samples, 16)
kappa_upper = np.percentile(samples, 84)

# =========================================================
# 5. CHI2 COMPUTATION AND RESULTS
# =========================================================
chi2_sm = np.sum(((obs_p5p - p5p_sm(q2)) / err_p5p)**2)
chi2_usf = np.sum(((obs_p5p - p5p_usf(kappa_best, q2)) / err_p5p)**2)
delta_chi2 = chi2_sm - chi2_usf
print(f"chi2_SM = {chi2_sm:.2f}, chi2_USF = {chi2_usf:.2f}, Delta = {delta_chi2:.2f}")

results = {
    "kappa_best": float(kappa_best),
    "kappa_median": float(kappa_median),
    "kappa_lower": float(kappa_lower),
    "kappa_upper": float(kappa_upper),
    "chi2_SM": float(chi2_sm),
    "chi2_USF": float(chi2_usf),
    "delta_chi2": float(delta_chi2)
}

with open("results/fit_results.json", "w") as f:
    json.dump(results, f, indent=2)

# =========================================================
# 6. SAVE MCMC CHAINS (HDF5)
# =========================================================
with h5py.File("results/mcmc_chains.h5", "w") as f:
    f.create_dataset("samples", data=samples)

# =========================================================
# 7. CORNER PLOT (POSTERIOR OF κ)
# =========================================================
plt.rcParams.update({'font.size': 12})
if samples.shape[1] == 1:
    fig, ax = plt.subplots(figsize=(6,5))
    # Draw histogram first (background)
    ax.hist(samples, bins=50, density=True, alpha=0.7, color='gray', edgecolor='black',
            label='Posterior')
    # Shaded 68% credibility region
    ax.axvspan(kappa_lower, kappa_upper, alpha=0.3, color='gray', label='68% CL')
    # Then lines on top
    ax.axvline(kappa_median, color='green', linestyle='-', linewidth=3, label='median')
    ax.axvline(kappa_best, color='red', linestyle='--', linewidth=1, label='best-fit')
    ax.set_xlabel(r'$\kappa$')
    ax.set_ylabel('Probability density')
    ax.set_title(r'Posterior distribution of $\kappa$ (USF)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    fig.savefig("results/corner_kappa.pdf", dpi=300)
    fig.savefig("results/corner_kappa.png", dpi=300)
    plt.close(fig)
else:
    # Multi-dimensional case (not used here, but kept for completeness)
    fig = corner.corner(samples, labels=[r"$\kappa$"], truths=[kappa_best],
                        quantiles=[0.16, 0.84], show_titles=True)
    fig.savefig("results/corner_kappa.pdf")
    fig.savefig("results/corner_kappa.png")
    plt.close(fig)

print("MCMC done. Results saved in results/")