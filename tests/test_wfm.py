"""
W.I.N.G.S. — Wright-Fisher model tests.

Covers the combo bit-encoding (shared contract with the SLURM submit
scripts and the ingest filename parser), determinism under a fixed
seed, history padding, and the expected CI fixation dynamics.
"""

import numpy as np
import pytest

from wings.analysis import ingest
from wings.models import wfm

# ======================================================================
#  Combo encoding — shared contract across Python and SLURM bash
# ======================================================================

def test_combo_id_bit_order():
    # CI<<3 | MK<<2 | ER<<1 | IE  (must match slurm/submit_abm.sh)
    assert wfm.combo_id(True, False, False, False) == 8
    assert wfm.combo_id(False, True, False, False) == 4
    assert wfm.combo_id(False, False, True, False) == 2
    assert wfm.combo_id(False, False, False, True) == 1
    assert wfm.combo_id(True, True, True, True) == 15
    assert wfm.combo_id(False, False, False, False) == 0


def test_combo_label():
    assert wfm.combo_label(True, False, True, False) == "CI+ER"
    assert wfm.combo_label(False, False, False, False) == "None"
    assert wfm.combo_label(True, True, True, True) == "CI+MK+ER+IE"


def test_filename_roundtrips_through_ingest():
    """wfm output filenames must be parseable by the ingest stage."""
    fname = wfm.make_filename(True, False, True, False, 7)
    parsed = ingest.parse_conditions_from_filename(fname)
    assert parsed == (True, False, True, False, 7)


# ======================================================================
#  Simulation behaviour
# ======================================================================

def test_simulate_is_deterministic():
    a = wfm.simulate(seed=42, ci=True)
    b = wfm.simulate(seed=42, ci=True)
    assert a == b


def test_simulate_history_length_is_padded():
    """History is always max_generations + 1 entries, even on early fixation."""
    for gens in (5, 15, 20):
        history = wfm.simulate(seed=1, ci=True, max_generations=gens)
        assert len(history) == gens + 1


def test_simulate_records_pop_and_rate():
    history = wfm.simulate(seed=3)
    pop, rate = history[0]
    assert pop == 50            # default N
    assert 0.0 <= rate <= 1.0


def test_full_ci_drives_fixation():
    """Full CI from p0=0.5 should fix in the large majority of replicates."""
    finals = [
        wfm.simulate(seed=s, ci=True, ci_strength=1.0,
                     initial_infection_freq=0.5)[-1][1]
        for s in range(12)
    ]
    assert np.mean(finals) >= 0.9
    assert sum(f >= 0.99 for f in finals) >= 9


def test_neutral_run_does_not_fix_deterministically():
    """With no effects, Wolbachia should not reliably reach fixation."""
    finals = [wfm.simulate(seed=s)[-1][1] for s in range(12)]
    # At least one replicate should fail to fix (drift, not selection).
    assert any(f < 0.99 for f in finals)


def test_mu_zero_reproduces_default():
    """mu=0.0 must reproduce the perfect-transmission recursion exactly."""
    assert wfm.simulate(seed=7, ci=True, mu=0.0) == wfm.simulate(seed=7, ci=True)


def test_leakage_drives_infection_down_without_ci():
    """With no CI to sustain it, strong leakage erodes the infection."""
    finals = [wfm.simulate(seed=s, mu=0.3, initial_infection_freq=0.5)[-1][1]
              for s in range(12)]
    assert np.mean(finals) < 0.4


@pytest.mark.parametrize("freq", [0.0, 0.25, 0.75, 1.0])
def test_initial_frequency_respected(freq):
    pop0, rate0 = wfm.simulate(seed=0, initial_infection_freq=freq)[0]
    # Initial rate is the rounded count / N
    expected = round(50 * freq) / 50
    assert rate0 == pytest.approx(expected, abs=1e-9)
