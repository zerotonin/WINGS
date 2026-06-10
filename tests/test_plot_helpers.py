"""
W.I.N.G.S. — plotting helper tests.

Covers the pure helpers in plot_wings (label building, mechanic
exclusion parsing/filtering, style lookup).  No figures are rendered;
the matplotlib backend is forced to Agg at import time.
"""

import pytest

from wings.analysis import plot_wings as pw


def test_combo_label_matches_wfm_convention():
    assert pw.combo_label(True, False, True, False) == "CI+ER"
    assert pw.combo_label(False, False, False, False) == "None"


def test_parse_exclude_valid():
    assert pw.parse_exclude("MK") == {"MK"}
    assert pw.parse_exclude("mk, ie") == {"MK", "IE"}
    assert pw.parse_exclude("") == set()
    assert pw.parse_exclude(None) == set()


def test_parse_exclude_invalid_exits():
    with pytest.raises(SystemExit):
        pw.parse_exclude("XX")


def test_filter_subset_by_exclusion_drops_mk():
    filtered = pw.filter_subset_by_exclusion(pw.SUBSET_A, {"MK"})
    mk_index = pw.MECHANIC_INDEX["MK"]
    assert all(not combo[mk_index] for combo in filtered)
    # The MK-only and all-effects combos must be gone.
    assert (False, True, False, False) not in filtered
    assert (True, True, True, True) not in filtered


def test_filter_subset_no_exclusion_is_identity():
    assert pw.filter_subset_by_exclusion(pw.SUBSET_B, set()) == pw.SUBSET_B


def test_get_style_returns_triplet():
    colour, dash, lw = pw.get_style("CI")
    assert isinstance(colour, str) and colour.startswith("#")
    assert isinstance(lw, (int, float))


def test_get_style_unknown_falls_back():
    style = pw.get_style("definitely-not-a-combo")
    assert style == (pw._TOL_GREY, pw._SOLID, 1.5)


def test_heatmap_exclusion_prunes_rows_and_cols():
    rows = [(False, False, "—"), (False, True, "MK"),
            (True, False, "ER"), (True, True, "MK+ER")]
    cols = [(False, False, "—"), (False, True, "IE"),
            (True, False, "CI"), (True, True, "CI+IE")]
    fr, fc = pw.filter_heatmap_configs(rows, cols, {"MK"})
    assert all(not mk for _, mk, _ in fr)
    assert fc == cols  # MK exclusion touches rows only
