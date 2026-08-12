# tests/test_data_extraction.py
"""Tests on the HEPData -> CSV extraction step.

The manuscript states (Sec. 3.5) that the pipeline allows "full reproduction of
the analysis" from the raw YAML files. It does not: scripts/generate_p5p_csv.py
reads F_L and S_5 from different tables with different binnings and would emit
an empty CSV. The committed data/p5p_observables.csv was produced some other
way.

The numbers in the committed CSV are nevertheless correct -- they match a
Table2-only reconstruction exactly -- so the tests separate "the values are
right" from "the script that claims to produce them works".
"""
import numpy as np
import pytest

from conftest import ANALYSIS_BINS, YAML_DIR

FL = "$F_{\\rm L}$"
S5 = "$S_{5}$"
P5P = "$P'_{5}$"


@pytest.fixture(scope="module")
def generator():
    import generate_p5p_csv

    return generate_p5p_csv


def test_fl_and_s5_are_read_from_the_same_table(generator):
    """Both angular coefficients must come from one consistently binned table.

    The previous version read F_L from Table1 (two wide bins) and S_5 from
    Table2 (eight narrow bins), then zipped them positionally.
    """
    bins_fl, _, _ = generator.load_observable(generator.SOURCE_TABLE, FL)
    bins_s5, _, _ = generator.load_observable(generator.SOURCE_TABLE, S5)
    assert bins_fl == bins_s5


def test_extraction_selects_the_seven_analysis_bins(generator):
    """The extraction must find all seven bins, or fail loudly.

    The previous version derived candidate centres from Table1 (3.55 and 17.0
    GeV^2), matched none of the targets within tolerance, and wrote a zero-row
    CSV without raising -- which is why the committed CSV could not have come
    from this script.
    """
    frame = generator.build_analysis_csv()
    assert len(frame) == len(ANALYSIS_BINS)
    built = list(zip(frame.q2_min, frame.q2_max))
    assert built == ANALYSIS_BINS


def test_missing_bins_raise_instead_of_writing_an_empty_file(generator, monkeypatch):
    """Silent failure is what allowed the broken extraction to go unnoticed."""
    monkeypatch.setattr(generator, "ANALYSIS_BINS", [(1.1, 2.5), (99.0, 100.0)])
    with pytest.raises(ValueError, match="does not contain the bins"):
        generator.build_analysis_csv()


def test_reconstruction_agrees_with_lhcb_published_p5p(generator):
    """The F_L/S5 reconstruction must reproduce LHCb's own P5' where published.

    Validates the method itself: in the two wide bins where the collaboration
    publishes P5' directly, the reconstruction agrees to better than 0.001.
    """
    cross = generator.build_cross_check_csv()
    deviation = np.abs(cross.value_published - cross.value_reconstructed)
    assert deviation.max() < 0.002, (
        f"reconstruction deviates from the published values by "
        f"{deviation.max():.4f}"
    )


def test_committed_csv_matches_a_table2_reconstruction(hepdata):
    """The committed values must be reproducible from the raw YAML.

    Pins the numbers so a change to the extraction cannot silently move them.
    The comparison is against the *uncorrelated* error column, which is what
    the published analysis used and what this reconstruction reproduces; the
    `error` column now additionally carries the F_L-S5 correlation.
    """
    import pandas as pd

    from conftest import DATA_DIR

    frame = pd.read_csv(DATA_DIR / "p5p_observables.csv")
    values = frame["value"].values
    errors = frame["error_uncorrelated"].values

    bins, fl, fl_err = hepdata("Table2", FL)
    _, s5, s5_err = hepdata("Table2", S5)

    keep = [i for i, b in enumerate(bins) if b in ANALYSIS_BINS]
    fl, fl_err = fl[keep], fl_err[keep]
    s5, s5_err = s5[keep], s5_err[keep]

    denom = np.sqrt(fl * (1 - fl))
    p5p = s5 / denom
    d_s5 = 1 / denom
    d_fl = -0.5 * s5 * (1 - 2 * fl) / denom ** 3
    sigma = np.sqrt((d_s5 * s5_err) ** 2 + (d_fl * fl_err) ** 2)

    np.testing.assert_allclose(values, p5p, rtol=1e-10)
    np.testing.assert_allclose(errors, sigma, rtol=1e-10)


def test_published_p5p_is_used_where_available(hepdata):
    """LHCb publishes P5' directly; reconstructing it is unnecessary.

    Table1 contains a measured $P'_{5}$ column for the 1.1-6 and 15-19 GeV^2
    bins, with the experimental correlations handled by the collaboration.
    Rebuilding P5' from F_L and S_5 under a diagonal-error assumption discards
    that treatment for no gain.
    """
    bins, values, errors = hepdata("Table1", P5P)
    assert len(values) == 2 and errors[0] > 0
    # The reference values the analysis should be anchored to.
    assert values[0] == pytest.approx(-0.049, abs=0.01)
    assert values[1] == pytest.approx(-0.684, abs=0.01)


def test_correlation_matrices_are_available_and_used():
    """The per-bin correlation matrices are in the repository but unused.

    Sec. 3.2 estimates that neglecting the F_L-S_5 correlation "affects the
    total uncertainty by at most 10%". No estimate is needed: HEPData ships the
    full likelihood correlation matrix for every q^2 bin, and those files are
    already vendored in data/raw/.

    DEFECT: no module in scripts/ reads them.
    """
    import yaml

    with open(YAML_DIR / "submission.yaml") as fh:
        docs = [d for d in yaml.safe_load_all(fh) if d and "data_file" in d]
    corr_tables = [
        d["data_file"] for d in docs
        if "correlation" in d.get("description", "").lower()
    ]
    assert len(corr_tables) >= 7, "expected one correlation matrix per q^2 bin"

    sources = (YAML_DIR.parent.parent.parent / "scripts").glob("*.py")
    users = [p.name for p in sources if "correlation" in p.read_text().lower()]
    assert users, (
        f"{len(corr_tables)} correlation matrices are vendored "
        f"({corr_tables[0]} ...) but no script in scripts/ reads them"
    )
