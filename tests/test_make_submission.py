import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from make_submission import apply_risk_budget, count_preserving_cycles, count_rule, em_weights, main

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("PACKET_FLOW_DATASET", ROOT / "application-inference-packet-flows-public" / "dataset"))
SUBMITTED_SHA256 = "75193d24f87d77507a45d13689c171dbe01dba5a9dad36b3d75ba2071256a63f"
CLASSES = ["facebook", "google", "instagram", "mozilla", "x", "youtube"]


def test_count_rule_moves_the_row_that_loses_least():
    # Both rows prefer column 0 but only one may have it. Moving row 1 costs log(0.6 / 0.4),
    # moving row 0 costs log(0.9 / 0.1), so row 1 moves.
    P = np.array([[0.9, 0.1], [0.6, 0.4]])
    assert count_rule(P, [1, 1]).tolist() == [0, 1]


def test_count_rule_never_uses_a_class_with_zero_count():
    P = np.array([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1], [0.5, 0.4, 0.1], [0.1, 0.5, 0.4]])
    assert np.bincount(count_rule(P, [2, 0, 3]), minlength=3).tolist() == [2, 0, 3]


def test_count_rule_rejects_counts_that_do_not_match_the_rows():
    with pytest.raises(ValueError):
        count_rule(np.full((3, 2), 0.5), [1, 1])


def test_em_gives_all_weight_to_a_model_that_is_better_on_every_row():
    y = np.array([0, 1, 0, 1])
    sharp = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    flat = np.full((4, 2), 0.5)
    np.testing.assert_allclose(em_weights(np.stack([sharp, flat]), y), [1.0, 0.0])


def test_em_finds_the_log_loss_optimum_between_two_models():
    # Model a gives the true class 0.9 on rows 0-2 and 0.1 on row 3; model b the reverse.
    # Maximising 3 log(0.1 + 0.8 w) + log(0.9 - 0.8 w) gives w = 2.6 / 3.2 = 0.8125.
    y = np.zeros(4, int)
    a = np.array([[0.9, 0.1]] * 3 + [[0.1, 0.9]])
    b = np.array([[0.1, 0.9]] * 3 + [[0.9, 0.1]])
    np.testing.assert_allclose(em_weights(np.stack([a, b]), y), [0.8125, 0.1875], atol=1e-6)


def test_cycles_split_the_changed_rows_into_count_preserving_loops():
    base = ["x", "mozilla", "google", "x", "youtube", "google"]
    new = ["mozilla", "x", "google", "youtube", "google", "x"]
    cycles = count_preserving_cycles(base, new)
    assert sorted(i for c in cycles for i in c) == [0, 1, 3, 4, 5]
    assert sorted(len(c) for c in cycles) == [2, 3]
    for c in cycles:
        assert sorted(base[i] for i in c) == sorted(new[i] for i in c)


def test_cycles_reject_label_sets_with_different_counts():
    with pytest.raises(ValueError):
        count_preserving_cycles(["x", "x"], ["x", "google"])


def _two_swaps():
    # Rows 0-1 swap x and mozilla (risk 1.0 + 1.0, log-probability gain 0.5 + 0.5).
    # Rows 2-3 swap google and youtube (risk 0.5 + 0.5, gain 0.25 + 0.25).
    base = ["x", "mozilla", "google", "youtube"]
    new = ["mozilla", "x", "youtube", "google"]
    col = {c: i for i, c in enumerate(CLASSES)}
    logp = np.zeros((4, len(CLASSES)))
    logp[0, col["x"]], logp[0, col["mozilla"]] = -1.0, -0.5
    logp[1, col["mozilla"]], logp[1, col["x"]] = -1.0, -0.5
    logp[2, col["google"]], logp[2, col["youtube"]] = -0.5, -0.25
    logp[3, col["youtube"]], logp[3, col["google"]] = -0.5, -0.25
    value = {"facebook": 4.0, "google": 0.5, "instagram": 2.0, "mozilla": 1.0, "x": 1.0, "youtube": 0.5}
    return base, new, logp, value


@pytest.mark.parametrize("headroom, want", [
    (3.0, ["mozilla", "x", "youtube", "google"]),  # both swaps fit
    (2.5, ["mozilla", "x", "google", "youtube"]),  # only the bigger gain fits
    (1.5, ["x", "mozilla", "youtube", "google"]),  # bigger gain is too risky, the smaller one still fits
    (0.5, ["x", "mozilla", "google", "youtube"]),  # nothing fits
])
def test_risk_budget_takes_whole_swaps_by_gain_while_they_fit(headroom, want):
    base, new, logp, value = _two_swaps()
    assert apply_risk_budget(base, new, logp, value, headroom) == want


@pytest.mark.skipif(not DATASET.is_dir(), reason="public dataset not found, see README")
def test_rebuilds_the_submitted_entry_byte_for_byte(tmp_path):
    main(["--dataset", str(DATASET), "--out", str(tmp_path)])
    assert hashlib.sha256((tmp_path / "submission.json").read_bytes()).hexdigest() == SUBMITTED_SHA256
