# scripts/response.py
"""P5' as a function of the b -> s mu mu Wilson coefficients.

Replaces the linearised response used in the published analysis,

    P5'(q^2, kappa) = P5'_SM(q^2) + (dP5'/dC9) * Delta C9,   dP5'/dC9 = -0.3

which is inadequate for two reasons.

*It is not linear.* In the 4-6 GeV^2 bin, moving Delta C9 from 0 to -1 shifts
P5' by +0.232, while moving from -1 to -2 shifts it by a further +0.352. The
response is also asymmetric about zero: +1 gives only -0.110. A single constant
slope cannot represent this over the range the fit explores.

*It is not bin independent.* The dependence on C9, C10, C7, form factors and
non-local charm-loop amplitudes varies with q^2, so one number cannot serve all
seven bins.

Here P5' is computed directly from the Wilson coefficients with flavio. An
exact evaluation costs ~20 ms per bin, which is too slow for MCMC, so a grid is
precomputed and interpolated -- and the interpolator is validated against exact
calls rather than trusted.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from sm_predictions import ANALYSIS_BINS, OBSERVABLE

#: Wilson coefficient grid. The range covers the published uniform prior on
#: kappa ([-2, 2]) with margin, in both signs, since the sign of the preferred
#: shift is one of the things under test.
GRID_DC9 = np.linspace(-4.0, 3.0, 43)
GRID_DC10 = np.linspace(-2.0, 2.0, 21)

#: Scale at which the coefficients are defined, mu ~ m_b.
WC_SCALE = 4.8

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "response_grid.npz"


def _wilson(dc9: float, dc10: float):
    from wilson import Wilson

    return Wilson(
        {"C9_bsmumu": float(dc9), "C10_bsmumu": float(dc10)},
        scale=WC_SCALE, eft="WET", basis="flavio",
    )


def p5p_np_exact(dc9: float, dc10: float, bins=None) -> np.ndarray:
    """Exact bin-integrated P5' for the given Wilson coefficient shifts."""
    import flavio

    bins = ANALYSIS_BINS if bins is None else bins
    wc = _wilson(dc9, dc10)
    return np.array([
        flavio.np_prediction(OBSERVABLE, wc, lo, hi) for lo, hi in bins
    ])


def _cache_key(bins) -> str:
    import flavio

    payload = (
        f"{OBSERVABLE}|{list(map(tuple, bins))}|{WC_SCALE}"
        f"|{GRID_DC9[0]},{GRID_DC9[-1]},{len(GRID_DC9)}"
        f"|{GRID_DC10[0]},{GRID_DC10[-1]},{len(GRID_DC10)}"
        f"|{flavio.__version__}"
    )
    return "g" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def build_grid(bins=None, verbose: bool = False) -> np.ndarray:
    """Evaluate P5' on the (Delta C9, Delta C10) grid for every bin.

    Returns an array of shape (n_bins, len(GRID_DC9), len(GRID_DC10)).
    Cached on disk; the key includes the grid definition and flavio version.
    """
    bins = ANALYSIS_BINS if bins is None else bins
    key = _cache_key(bins)

    if CACHE_PATH.exists():
        with np.load(CACHE_PATH) as store:
            if key in store:
                return store[key]

    grid = np.empty((len(bins), len(GRID_DC9), len(GRID_DC10)))
    for i, dc9 in enumerate(GRID_DC9):
        for j, dc10 in enumerate(GRID_DC10):
            grid[:, i, j] = p5p_np_exact(dc9, dc10, bins)
        if verbose:
            print(f"  Delta C9 = {dc9:+.3f}  ({i + 1}/{len(GRID_DC9)})", flush=True)

    existing = {}
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH) as store:
            existing = {k: store[k] for k in store.files}
    existing[key] = grid
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, **existing)
    return grid


class P5pResponse:
    """Fast interpolated P5'(Delta C9, Delta C10) for a fixed set of bins.

    Always check :meth:`validate` before relying on it in a fit.
    """

    def __init__(self, bins=None, verbose: bool = False):
        self.bins = list(ANALYSIS_BINS if bins is None else bins)
        self._grid = build_grid(self.bins, verbose=verbose)
        self._interpolators = [
            RegularGridInterpolator(
                (GRID_DC9, GRID_DC10), self._grid[i],
                method="cubic", bounds_error=True,
            )
            for i in range(len(self.bins))
        ]

    def __call__(self, dc9: float, dc10: float) -> np.ndarray:
        point = np.array([[dc9, dc10]])
        return np.array([float(interp(point)[0]) for interp in self._interpolators])

    def validate(self, n_points: int = 40, seed: int = 20260812) -> float:
        """Compare the interpolator with exact flavio calls at random points.

        Returns the largest absolute deviation found. The published analysis
        approximated the response and never checked it; this exists so the same
        mistake is not repeated one level down.
        """
        rng = np.random.default_rng(seed)
        # Stay inside the grid so the comparison tests interpolation, not
        # extrapolation, which bounds_error=True forbids anyway.
        dc9 = rng.uniform(GRID_DC9[0] * 0.95, GRID_DC9[-1] * 0.95, n_points)
        dc10 = rng.uniform(GRID_DC10[0] * 0.95, GRID_DC10[-1] * 0.95, n_points)

        worst = 0.0
        for a, b in zip(dc9, dc10):
            deviation = np.max(np.abs(self(a, b) - p5p_np_exact(a, b, self.bins)))
            worst = max(worst, float(deviation))
        return worst


def local_slope(dc9: float = 0.0, dc10: float = 0.0, step: float = 0.05,
                bins=None) -> np.ndarray:
    """Numerical dP5'/dC9 per bin, for comparison with the assumed -0.3."""
    plus = p5p_np_exact(dc9 + step, dc10, bins)
    minus = p5p_np_exact(dc9 - step, dc10, bins)
    return (plus - minus) / (2 * step)


if __name__ == "__main__":
    print("Building the response grid "
          f"({len(GRID_DC9)} x {len(GRID_DC10)} points x {len(ANALYSIS_BINS)} bins)...")
    response = P5pResponse(verbose=True)

    print("\nValidating the interpolator against exact flavio calls...")
    worst = response.validate()
    print(f"  worst absolute deviation: {worst:.2e}")
    assert worst < 1e-3, f"interpolator is not accurate enough ({worst:.2e})"

    print("\ndP5'/dC9 at the SM point, per bin "
          "(the analysis assumed a constant -0.3):")
    for b, slope in zip(ANALYSIS_BINS, local_slope()):
        print(f"  {str(b):>14}  {slope:+.3f}")

    print("\nNon-linearity in the 4-6 GeV^2 bin:")
    index = ANALYSIS_BINS.index((4.0, 6.0))
    for dc9 in (-3, -2, -1, 0, 1):
        print(f"  Delta C9 = {dc9:+d}  P5' = {p5p_np_exact(dc9, 0.0)[index]:+.4f}")
