# scripts/sm_predictions.py
"""Standard Model predictions for P5' in B0 -> K*0 mu+ mu-.

Replaces the eight hard-coded (q^2, P5'_SM) pairs that scripts/utils.py
interpolated. Those values were wrong by up to 32.6 theory sigma and were
blindly extrapolated above 15 GeV^2; roughly 66 of the 70.4 units of chi2_SM
in the original analysis came from that error rather than from any physical
discrepancy.

Two things change here beyond correctness:

*Bin integration.* P5' is a ratio of integrals of angular coefficients, so the
bin average <P5'> is not P5'(q^2_centre). flavio's ``<P5p>(...)`` observable is
the bin-integrated one; the interface below takes (q2min, q2max) and has no way
to ask for a bin centre.

*Covariance, not per-bin errors.* Theory uncertainties on P5' are dominated by
form factors shared across q^2, so they are strongly correlated between bins --
up to rho = 0.98 for adjacent bins here. A diagonal treatment is wrong in both
directions and must not be used.

The covariance is estimated by Monte Carlo over the theory parameters, so it is
cached on disk and the global RNG is seeded to keep results reproducible.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

OBSERVABLE = "<P5p>(B0->K*mumu)"

#: The seven q^2 bins used in the analysis, as (low, high) in GeV^2.
#: These are the LHCb Run-1 narrow bins (HEPData ins1409497, Table2) with the
#: photon-pole bin (0.1-0.98) and the charmonium regions (8-11, 12.5-15) removed.
ANALYSIS_BINS: list[tuple[float, float]] = [
    (1.1, 2.5),
    (2.5, 4.0),
    (4.0, 6.0),
    (6.0, 8.0),
    (11.0, 12.5),
    (15.0, 17.0),
    (17.0, 19.0),
]

#: Number of theory-parameter draws for the covariance estimate.
COVARIANCE_SAMPLES = 500

#: Seed for the covariance Monte Carlo. flavio samples from the global NumPy
#: RNG, so this is applied with np.random.seed rather than a Generator.
COVARIANCE_SEED = 20260812

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "sm_cache.npz"


def _flavio():
    """Import flavio lazily so that importing this module stays cheap."""
    import flavio

    return flavio


def _cache_key(bins, n_samples: int) -> str:
    flavio = _flavio()
    payload = f"{OBSERVABLE}|{list(map(tuple, bins))}|{n_samples}|{COVARIANCE_SEED}|{flavio.__version__}"
    return "k" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def _cache_load(key: str):
    if not CACHE_PATH.exists():
        return None
    with np.load(CACHE_PATH) as store:
        if key not in store:
            return None
        return store[key]


def _cache_store(key: str, value: np.ndarray) -> None:
    existing = {}
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH) as store:
            existing = {k: store[k] for k in store.files}
    existing[key] = value
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, **existing)


# ---------------------------------------------------------------------------
# Central values
# ---------------------------------------------------------------------------

def p5p_sm(q2min: float, q2max: float) -> float:
    """Bin-integrated SM prediction for P5' over [q2min, q2max] in GeV^2."""
    return float(_flavio().sm_prediction(OBSERVABLE, q2min, q2max))


def p5p_sm_binned(bins=None) -> np.ndarray:
    """SM predictions for a list of (q2min, q2max) bins."""
    bins = ANALYSIS_BINS if bins is None else bins
    return np.array([p5p_sm(lo, hi) for lo, hi in bins])


# ---------------------------------------------------------------------------
# Theory covariance
# ---------------------------------------------------------------------------

def p5p_sm_covariance(bins=None, n_samples: int = COVARIANCE_SAMPLES,
                      threads: int = 4, use_cache: bool = True) -> np.ndarray:
    """Theory covariance matrix of P5' across the given bins.

    Estimated by flavio from the uncertainties on form factors, CKM elements
    and the remaining hadronic inputs. Off-diagonal entries are large and must
    be kept: neglecting them is not a conservative approximation.
    """
    bins = ANALYSIS_BINS if bins is None else bins
    key = _cache_key(bins, n_samples)

    if use_cache:
        cached = _cache_load(key)
        if cached is not None:
            return cached

    flavio = _flavio()
    observables = [(OBSERVABLE, lo, hi) for lo, hi in bins]
    # flavio samples from the global NumPy RNG. Seed it for reproducibility,
    # but restore the caller's state afterwards: otherwise whether this cache
    # was warm or cold would silently change every later random draw.
    state = np.random.get_state()
    try:
        np.random.seed(COVARIANCE_SEED)
        cov = np.asarray(
            flavio.sm_covariance(observables, N=n_samples, threads=threads)
        )
    finally:
        np.random.set_state(state)

    if use_cache:
        _cache_store(key, cov)
    return cov


def p5p_sm_uncertainty(bins=None) -> np.ndarray:
    """Per-bin theory standard deviations (the diagonal of the covariance).

    Provided for reporting and plotting only. Do not build a chi2 from these --
    use :func:`p5p_sm_covariance`.
    """
    return np.sqrt(np.diag(p5p_sm_covariance(bins)))


# ---------------------------------------------------------------------------
# The superseded curve, retained for regression comparisons
# ---------------------------------------------------------------------------

#: Hard-coded grid formerly used by scripts/utils.py. The source comment
#: attributed it to "Fig. 5 of arXiv:1505.07814" while the manuscript cites
#: LHCb 2016a (arXiv:1512.04442). It is retained only so that tests and the
#: figures can quantify how far off it was; it must not be used for fitting.
_LEGACY_Q2 = np.array([1.55, 2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 15.0])
_LEGACY_P5P = np.array([0.12, 0.08, 0.04, 0.01, -0.02, -0.05, -0.09, -0.14])


def p5p_sm_legacy(q2):
    """DEPRECATED. The incorrect SM curve used in the published analysis.

    Kept for provenance and for the regression tests that quantify the error.
    Note the extrapolation beyond 15 GeV^2, which produced the -0.034 value at
    18 GeV^2 that alone contributed 19.7 units of chi2.
    """
    from scipy.interpolate import interp1d

    interp = interp1d(_LEGACY_Q2, _LEGACY_P5P, kind="cubic", fill_value="extrapolate")
    return float(interp(q2)) if np.isscalar(q2) else interp(q2)


if __name__ == "__main__":
    centres = np.array([np.mean(b) for b in ANALYSIS_BINS])
    sm = p5p_sm_binned()
    cov = p5p_sm_covariance()
    sd = np.sqrt(np.diag(cov))
    legacy = p5p_sm_legacy(centres)

    print(f"{'bin (GeV^2)':>14} {'SM (flavio)':>13} {'sigma_th':>9} "
          f"{'legacy curve':>13} {'deviation':>11}")
    for b, s, e, l in zip(ANALYSIS_BINS, sm, sd, legacy):
        print(f"{str(b):>14} {s:+13.4f} {e:9.4f} {l:+13.4f} {abs(s - l) / e:10.1f}s")

    corr = cov / np.outer(sd, sd)
    print(f"\nLargest off-diagonal theory correlation: {np.max(corr - np.eye(len(sd))):.2f}")
    print(f"Cache: {CACHE_PATH}")
