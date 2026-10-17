import csv
import json
from pathlib import Path

import numpy as np

from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS


GROUPS = {
    "all23": tuple(range(NUM_LANDMARKS)),
    "core20": tuple(CORE20),
    "hard3": tuple(HARD3),
    "lm0": (0,),
    "lm21": (21,),
    "lm22": (22,),
}


def error_metrics(errors):
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    return {
        "ale": float(values.mean()),
        "median": float(np.median(values)),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "sdr2": float(np.mean(values <= 2.0)),
        "sdr3": float(np.mean(values <= 3.0)),
        "sdr4": float(np.mean(values <= 4.0)),
        "max": float(values.max()),
    }


def paired_comparison(expert, incoming, candidate, sample_ids, seed=42, iterations=2000):
    expert = np.asarray(expert, dtype=np.float64)
    incoming = np.asarray(incoming, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if expert.shape != incoming.shape or expert.shape != candidate.shape:
        raise ValueError("Paired coordinates have different shapes")
    if expert.ndim != 3 or expert.shape[1:] != (NUM_LANDMARKS, 3):
        raise ValueError("Expected [sample, 23, 3] coordinates")
    if len(sample_ids) != len(expert) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Participant identifiers must be unique and complete")
    baseline = np.linalg.norm(incoming - expert, axis=-1)
    proposed = np.linalg.norm(candidate - expert, axis=-1)
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, len(expert), size=(iterations, len(expert)))
    result = {}
    for name, landmarks in GROUPS.items():
        old = baseline[:, landmarks]
        new = proposed[:, landmarks]
        paired_gain = (old - new).mean(axis=1)
        bootstrap = paired_gain[boot_indices].mean(axis=1)
        change = new - old
        result[name] = {
            "incoming": error_metrics(old),
            "candidate": error_metrics(new),
            "gain_mm": float(paired_gain.mean()),
            "gain_ci95": [float(value) for value in np.percentile(bootstrap, [2.5, 97.5])],
            "fraction_improved": float(np.mean(change < -1e-6)),
            "fraction_harmed": float(np.mean(change > 1e-6)),
            "fraction_unchanged": float(np.mean(np.abs(change) <= 1e-6)),
            "count": int(old.size),
        }
    result["sample_count"] = len(sample_ids)
    return result


def _coordinates(row, prefix):
    return [float(row[f"{prefix}_{axis}"]) for axis in "xyz"]


def read_archived_predictions(root, split):
    rows = {}
    for fold in range(1, 6):
        path = Path(root) / f"fold_{fold}" / f"predictions_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["sample_id"], int(row["landmark"]))
                if key in rows:
                    raise ValueError(f"Duplicate participant/landmark: {key}")
                rows[key] = row
    sample_ids = sorted({key[0] for key in rows})
    expected = {(sample_id, index) for sample_id in sample_ids for index in range(NUM_LANDMARKS)}
    if set(rows) != expected:
        raise ValueError("Archived predictions do not contain exactly 23 landmarks per participant")
    arrays = {}
    for name, prefix in (
        ("expert", "expert"),
        ("stage1", "stage1_coarse"),
        ("fusion", "coarse"),
        ("local", "refined"),
        ("token_gate", "neural_prediction"),
    ):
        arrays[name] = np.asarray(
            [[_coordinates(rows[(sample_id, index)], prefix) for index in range(NUM_LANDMARKS)]
             for sample_id in sample_ids], dtype=np.float64
        )
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Non-finite coordinates in {name}")
    return sample_ids, arrays


def audit_archived(root, split="test", seed=42, iterations=2000):
    sample_ids, arrays = read_archived_predictions(root, split)
    stages = {}
    for name in ("stage1", "fusion", "local", "token_gate"):
        errors = np.linalg.norm(arrays[name] - arrays["expert"], axis=-1)
        stages[name] = {
            group: error_metrics(errors[:, landmarks])
            for group, landmarks in GROUPS.items()
        }
    transitions = {}
    for incoming, candidate in (
        ("stage1", "fusion"),
        ("fusion", "local"),
        ("local", "token_gate"),
    ):
        transitions[f"{incoming}_to_{candidate}"] = paired_comparison(
            arrays["expert"], arrays[incoming], arrays[candidate], sample_ids,
            seed=seed, iterations=iterations,
        )
    return {
        "source": str(root), "split": split, "sample_count": len(sample_ids),
        "note": "Historical audit; no new gate or independent validation is implied.",
        "stages": stages, "transitions": transitions,
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
