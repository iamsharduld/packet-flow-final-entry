"""Rebuild our final entry to the Application Inference from Packet Flows challenge.

Reads the saved model predictions in predictions/ and, from the public dataset, the test flow order,
the released train and validation labels and the published test class counts. Writes submission.json.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

CLASSES = ["facebook", "google", "instagram", "mozilla", "x", "youtube"]
TASK1, TASK2 = "task1_static", "task2_realtime_prefix5"
RARE = {"facebook", "instagram"}
REFERENCE_TASK2_F1 = 0.951640  # Final Test macro F1 of the reference Task 2 labels, from the leaderboard
SUBMITTED_SHA256 = "75193d24f87d77507a45d13689c171dbe01dba5a9dad36b3d75ba2071256a63f"
PREDICTIONS = Path(__file__).resolve().parent / "predictions"


def normalise(P):
    return P / P.sum(-1, keepdims=True)


def count_rule(P, counts):
    """Row labels with the highest total log-probability such that column c is used counts[c] times."""
    slots = [c for c, n in enumerate(counts) for _ in range(int(n))]
    if len(slots) != len(P):
        raise ValueError(f"counts sum to {len(slots)} but there are {len(P)} rows")
    rows, cols = linear_sum_assignment(-np.log(np.clip(P, 1e-12, 1))[:, slots])
    labels = np.empty(len(P), int)
    labels[rows] = np.array(slots)[cols]
    return labels


def em_weights(P, y, iters=400, tol=1e-9, floor=1e-4):
    """Weights for the mixture of models P[k] that minimise log loss on labels y, fitted by EM."""
    lik = np.clip(P[:, np.arange(len(y)), y], 1e-12, None)
    w = np.full(len(P), 1.0 / len(P))
    for _ in range(iters):
        new = np.clip(w * (lik / np.clip(w @ lik, 1e-300, None)).mean(1), 0, None)
        new /= new.sum()
        converged = np.max(np.abs(new - w)) < tol
        w = new
        if converged:
            break
    w[w < floor] = 0.0
    return w / w.sum()


def count_preserving_cycles(base, new):
    """Split the rows where `new` differs from `base` into closed loops of class moves.

    With equal class counts, the moves (base class -> new class) form loops. A loop is the smallest
    change that keeps the counts, so it is kept or undone as a whole.
    """
    if Counter(base) != Counter(new):
        raise ValueError("base and new must have the same class counts")
    moves = {}
    for i, (b, n) in enumerate(zip(base, new)):
        if b != n:
            moves.setdefault(b, []).append(i)
    cycles = []
    while any(moves.values()):
        start = next(c for c, rows in moves.items() if rows)
        cycle, cls = [], start
        while True:
            i = moves[cls].pop()
            cycle.append(i)
            cls = new[i]
            if cls == start:
                break
        cycles.append(cycle)
    return cycles


def apply_risk_budget(base, new, logp, value, headroom):
    """Keep the loops that gain most log-probability while their summed worst case fits the headroom.

    A loop's worst case is that every row it moves had a correct label, costing value[class] each.
    """
    scored = []
    for cycle in count_preserving_cycles(base, new):
        gain = sum(logp[i, CLASSES.index(new[i])] - logp[i, CLASSES.index(base[i])] for i in cycle)
        scored.append((gain, sum(value[base[i]] for i in cycle), cycle))
    out, spent = list(base), 0.0
    for gain, risk, cycle in sorted(scored, key=lambda s: -s[0]):
        if spent + risk <= headroom + 1e-9:
            spent += risk
            for i in cycle:
                out[i] = new[i]
    return out


def task1_labels(pred, counts):
    """Mean of five Task 1 models, times the Task 2 probabilities, under the count rule."""
    P = normalise(normalise(pred["test"].mean(0)) * pred["task2_test"])
    return [CLASSES[k] for k in count_rule(P, counts)]


def task2_labels(pred, counts, y):
    """EM-weighted blend of ten Task 2 models under the count rule, with two safety policies."""
    w = em_weights(normalise(pred["oof"]), y)
    P = normalise(np.tensordot(w, normalise(pred["test"]), axes=1))
    reference = [CLASSES[k] for k in count_rule(normalise(pred["reference_test"]), counts)]

    # 1. Keep the reference's facebook and instagram rows; let the blend re-decide only the rest.
    frozen = Counter(c for c in reference if c in RARE)
    free = [i for i, c in enumerate(reference) if c not in RARE]
    new = list(reference)
    for i, k in zip(free, count_rule(P[free], [n - frozen[c] for c, n in zip(CLASSES, counts)])):
        new[i] = CLASSES[k]

    # 2. Risk budget: the worst case may not exceed what Task 2 has left to win. Under the count rule
    # one correct row of class c is worth 50 / (6 * n_c) points of the 50-point task score.
    value = {c: 50 / (6 * n) for c, n in zip(CLASSES, counts)}
    headroom = 50 * (1 - REFERENCE_TASK2_F1)
    return apply_risk_budget(reference, new, np.log(np.clip(P, 1e-12, 1)), value, headroom)


def load(dataset, task):
    """Test flow order, published test counts and saved predictions (rows in that order) for a task."""
    with open(dataset / task / "sample_submission.csv", newline="") as f:
        ids = [row["flow_id"] for row in csv.DictReader(f)]
    counts = json.loads((dataset / task / "metadata.json").read_text())["label_counts"]["test"]
    with np.load(PREDICTIONS / f"{task}.npz") as z:
        pred = dict(z)
    assert pred["classes"].tolist() == CLASSES
    pos = {f: i for i, f in enumerate(pred["test_flow_id"].tolist())}
    order = [pos[f] for f in ids]
    for key in ("test", "task2_test", "reference_test"):
        if key in pred:
            pred[key] = pred[key][..., order, :]
    return ids, [counts[c] for c in CLASSES], pred


def released_labels(dataset, task, flow_ids):
    labels = {}
    for split in ("train", "validation"):
        with open(dataset / task / f"{split}_labels.csv", newline="") as f:
            labels.update((row["flow_id"], row["app"]) for row in csv.DictReader(f))
    return np.array([CLASSES.index(labels[f]) for f in flow_ids])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild the final entry from saved predictions.")
    parser.add_argument("--dataset", type=Path, default=Path("application-inference-packet-flows-public/dataset"),
                        help="the dataset/ folder of the organizers' repository")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args(argv)

    ids1, counts1, pred1 = load(args.dataset, TASK1)
    ids2, counts2, pred2 = load(args.dataset, TASK2)
    y = released_labels(args.dataset, TASK2, pred2["oof_flow_id"].tolist())
    submission = {TASK1: dict(zip(ids1, task1_labels(pred1, counts1))),
                  TASK2: dict(zip(ids2, task2_labels(pred2, counts2, y)))}

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "submission.json"
    path.write_text(json.dumps(submission, indent=1))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path}, sha256 {digest}")
    print("matches the submitted entry" if digest == SUBMITTED_SHA256 else "does NOT match the submitted entry")


if __name__ == "__main__":
    main()
