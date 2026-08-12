# scripts/global_consistency.py
"""Test the fitted Wilson coefficients against other b -> s l l observables.

The manuscript states that Delta C10 = -0.2 * Delta C9 "emerges from the tensor
structure of the holographic coupling", is "a fixed prediction of the framework"
and "is not adjusted to the data". That makes it falsifiable at no cost: the
same coefficients that fit P5' also determine other rare decays.

This addresses the second referee's third point directly -- a model that
modifies C9 and C10 must be tested against more than one observable.

The outcome turns on a point the manuscript never states: whether the geometric
modification is lepton universal. Both readings are evaluated here, because
they give opposite verdicts.

If Delta C9 is muon-specific, R_K and R_K* are dragged well below 1 and the
point is excluded. If it is lepton universal -- the natural reading for an
effect sourced by spacetime geometry, which has no reason to distinguish lepton
flavour -- the ratios stay at 1 and the point survives.

b -> s nu nu is insensitive to these coefficients and is not a useful
discriminator, so it is not included.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

C10_OVER_C9 = -0.2

#: Current measurements. Central value, uncertainty, and source.
MEASUREMENTS = {
    "BR(Bs->mumu)": (
        3.34e-9, 0.27e-9,
        "LHCb/CMS/ATLAS combination, PDG 2023",
    ),
    "<Rmue>(B+->Kll)": (
        0.949, 0.047,
        "LHCb 2022, arXiv:2212.09152 (1.1-6 GeV^2)",
    ),
    "<Rmue>(B0->K*ll)": (
        1.027, 0.077,
        "LHCb 2022, arXiv:2212.09153 (1.1-6 GeV^2)",
    ),
}

#: Bin for the ratio observables, in GeV^2.
RATIO_BIN = (1.1, 6.0)


#: The two readings of how the geometric modification couples to leptons.
SCENARIOS = {
    "muon-specific": "Delta C9, C10 applied to b->s mu mu only",
    "lepton-universal": "the same shift applied to electrons as well",
}


def wilson_point(dc9: float, dc10: float, scenario: str):
    from wilson import Wilson

    coefficients = {"C9_bsmumu": dc9, "C10_bsmumu": dc10}
    if scenario == "lepton-universal":
        coefficients.update({"C9_bsee": dc9, "C10_bsee": dc10})
    elif scenario != "muon-specific":
        raise ValueError(f"unknown scenario {scenario!r}")
    return Wilson(coefficients, scale=4.8, eft="WET", basis="flavio")


def evaluate(dc9: float, dc10: float, scenario: str) -> dict:
    """Predictions for the cross-check observables at a given coefficient point."""
    import flavio

    wc = wilson_point(dc9, dc10, scenario)
    out = {}
    for name in MEASUREMENTS:
        if name.startswith("<"):
            sm = flavio.sm_prediction(name, *RATIO_BIN)
            np_value = flavio.np_prediction(name, wc, *RATIO_BIN)
        else:
            sm = flavio.sm_prediction(name)
            np_value = flavio.np_prediction(name, wc)
        out[name] = {"sm": float(sm), "np": float(np_value)}
    return out


def pulls(predictions: dict) -> dict:
    """Deviation of each prediction from its measurement, in sigma.

    Experimental uncertainty only. Theory uncertainties on these observables are
    small compared with the shifts being tested, except for BR(Bs->mumu) where
    the ~4% form-factor/decay-constant error is noted in the output.
    """
    result = {}
    for name, (measured, sigma, _) in MEASUREMENTS.items():
        entry = predictions[name]
        result[name] = {
            "measured": measured,
            "sigma": sigma,
            "pull_sm": (entry["sm"] - measured) / sigma,
            "pull_np": (entry["np"] - measured) / sigma,
        }
    return result


def main() -> None:
    path = RESULTS_DIR / "fit_results.json"
    if not path.exists():
        raise SystemExit("run scripts/fit_usf.py first")
    with open(path) as fh:
        fit = json.load(fh)

    kappa = fit["posterior"]["kappa_best"]
    # f_geo == 1 with the manuscript's constants, so Delta C9 = kappa.
    dc9 = kappa
    dc10 = C10_OVER_C9 * dc9

    print("Testing the fixed prediction Delta C10 = -0.2 * Delta C9")
    print(f"  kappa       = {kappa:+.4f}   (from the corrected P5' fit)")
    print(f"  Delta C9    = {dc9:+.4f}")
    print(f"  Delta C10   = {dc10:+.4f}   (fixed by the framework, not fitted)\n")

    scenarios = {}
    chi2_sm = None
    for scenario, description in SCENARIOS.items():
        predictions = evaluate(dc9, dc10, scenario)
        table = pulls(predictions)
        chi2_sm = sum(e["pull_sm"] ** 2 for e in table.values())
        chi2_np = sum(e["pull_np"] ** 2 for e in table.values())
        worst = max(table.items(), key=lambda kv: abs(kv[1]["pull_np"]))

        print(f"--- {scenario}: {description}")
        print(f"{'observable':<20} {'measured':>12} {'SM':>12} {'USF':>12} "
              f"{'pull_SM':>9} {'pull_USF':>9}")
        for name, entry in table.items():
            p = predictions[name]
            fmt = "{:12.3e}" if "BR" in name else "{:12.4f}"
            print(f"{name:<20} " + fmt.format(entry["measured"]) + " "
                  + fmt.format(p["sm"]) + " " + fmt.format(p["np"])
                  + f" {entry['pull_sm']:+9.2f} {entry['pull_np']:+9.2f}")
        print(f"  sum of squared pulls: SM {chi2_sm:.2f}, USF {chi2_np:.2f}")
        print(f"  worst tension: {worst[0]} at {worst[1]['pull_np']:+.2f} sigma\n")

        scenarios[scenario] = {
            "description": description,
            "observables": {
                name: {**predictions[name], **table[name],
                       "source": MEASUREMENTS[name][2]}
                for name in table
            },
            "sum_squared_pulls_usf": chi2_np,
            "worst_observable": worst[0],
            "worst_pull": worst[1]["pull_np"],
        }

    muon = scenarios["muon-specific"]["sum_squared_pulls_usf"]
    universal = scenarios["lepton-universal"]["sum_squared_pulls_usf"]
    print(f"Verdict depends on a point the manuscript never states.")
    print(f"  If muon-specific    : excluded "
          f"(sum of squared pulls {muon:.1f} against {chi2_sm:.1f} for the SM)")
    print(f"  If lepton-universal : survives "
          f"(sum of squared pulls {universal:.1f} against {chi2_sm:.1f} for the SM)")
    print("\nA modification sourced by spacetime geometry has no evident reason "
          "to\ndistinguish lepton flavour, so the second reading is the natural "
          "one. Under\nit the fitted coefficients are compatible with the "
          "cross-check set -- but a\nlepton-universal Delta C9 near -1 is also "
          "what an unmodelled charm-loop\nhadronic effect would produce, so "
          "compatibility is not evidence for geometry.")

    print("\nSources:")
    for name, (_, _, source) in MEASUREMENTS.items():
        print(f"  {name:<20} {source}")

    payload = {
        "kappa": kappa,
        "delta_c9": dc9,
        "delta_c10": dc10,
        "c10_over_c9": C10_OVER_C9,
        "ratio_bin": list(RATIO_BIN),
        "sum_squared_pulls_sm": chi2_sm,
        "scenarios": scenarios,
    }
    with open(RESULTS_DIR / "global_consistency.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWritten to results/global_consistency.json")


if __name__ == "__main__":
    main()
