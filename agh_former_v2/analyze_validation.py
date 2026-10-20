#!/usr/bin/env python3
"""Validation-only diagnostics comparing AGH-Former v2 with archived vNext.

Reads only fold_k/predictions_val.csv and fold_k/all23_mvsc/all23_policy.json;
test predictions are never opened. Rows are fold-predictions keyed by
(fold, sample_id); confidence intervals resample whole participants because a
participant can be in the validation partition of several folds.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agh_former_v2.run_agh_v2 import cluster_bootstrap_delta
from all23_mvsc_refinement.anatomy import LANDMARK_TYPES
from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, LANDMARK_NAMES, NUM_LANDMARKS
from all23_rgb_geodesic_cascade.metrics import write_csv


V2_STAGES = ("stage_a", "prior", "iter1", "iter2", "prediction")
VNEXT_STAGES = (
    "coarse",
    "refined",
    "neural_prediction",
    "pre_hard3_prediction",
    "pre_core20_prediction",
    "prediction",
)
SETS = {
    "all23": tuple(range(NUM_LANDMARKS)),
    "core20": tuple(CORE20),
    "hard3": tuple(HARD3),
    **LANDMARK_TYPES,
}
ALPHA_BINS = ((0.0, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def open_validation_csv(path):
    """Open a validation prediction file; any test file is refused."""
    if "test" in Path(path).name:
        raise AssertionError(f"Diagnostics must never read test predictions: {path}")
    return open(path, newline="", encoding="utf-8")


class ValidationRun:
    """Validation predictions of one run, rows keyed by (fold, sample_id)."""

    def __init__(self, run_dir, folds):
        self.run_dir = Path(run_dir)
        values = {}
        for fold in folds:
            path = self.run_dir / f"fold_{fold}" / "predictions_val.csv"
            with open_validation_csv(path) as handle:
                for row in csv.DictReader(handle):
                    values.setdefault((int(fold), row["sample_id"]), {})[
                        int(row["landmark"])
                    ] = row
        self.keys = sorted(values)
        incomplete = [key for key in self.keys if len(values[key]) != NUM_LANDMARKS]
        if incomplete:
            raise RuntimeError(f"Incomplete validation rows in {run_dir}: {incomplete[:3]}")
        self.rows = [values[key] for key in self.keys]
        self.participants = [sample_id for _, sample_id in self.keys]
        self.folds = np.asarray([fold for fold, _ in self.keys])
        self.columns = set(self.rows[0][0])

    def has(self, prefix):
        return f"{prefix}_x" in self.columns

    def coordinates(self, prefix):
        return np.asarray(
            [
                [
                    [float(row[landmark][f"{prefix}_{axis}"]) for axis in "xyz"]
                    for landmark in range(NUM_LANDMARKS)
                ]
                for row in self.rows
            ],
            dtype=np.float64,
        )

    def scalar(self, name):
        return np.asarray(
            [[float(row[landmark][name]) for landmark in range(NUM_LANDMARKS)] for row in self.rows],
            dtype=np.float64,
        )

    def errors(self, prefix):
        return np.linalg.norm(self.coordinates(prefix) - self.coordinates("expert"), axis=-1)


def stage_rows(run, model, stages):
    rows = []
    for stage in stages:
        if not run.has(stage):
            continue
        errors = run.errors(stage)
        row = {"model": model, "stage": stage}
        row.update({name: float(errors[:, list(landmarks)].mean()) for name, landmarks in SETS.items()})
        rows.append(row)
    return rows


def landmark_stage_rows(run, model, stages):
    rows = []
    present = [stage for stage in stages if run.has(stage)]
    errors = {stage: run.errors(stage) for stage in present}
    for landmark in range(NUM_LANDMARKS):
        row = {"model": model, "landmark": landmark, "landmark_name": LANDMARK_NAMES[landmark]}
        row.update({stage: float(errors[stage][:, landmark].mean()) for stage in present})
        rows.append(row)
    return rows


def head_to_head(v2, vnext, iterations, seed):
    v2_errors = v2.errors("prediction")
    vnext_errors = vnext.errors("prediction")
    rows = []
    groups = [(name, list(landmarks)) for name, landmarks in SETS.items()] + [
        (f"LM{landmark} {LANDMARK_NAMES[landmark]}", [landmark]) for landmark in range(NUM_LANDMARKS)
    ]
    for name, columns in groups:
        delta = cluster_bootstrap_delta(
            vnext_errors[:, columns], v2_errors[:, columns], v2.participants, iterations, seed
        )
        rows.append(
            {
                "set": name,
                "v2_ale": float(v2_errors[:, columns].mean()),
                "vnext_ale": float(vnext_errors[:, columns].mean()),
                "v2_minus_vnext": delta["delta_ale"],
                "ci95_low": delta["ci95"][0],
                "ci95_high": delta["ci95"][1],
                "probability_v2_better": delta["probability_improved"],
            }
        )
    return rows


def _bin_mask(alpha, low, high):
    if low == high:
        return alpha <= low + 1e-9
    return (alpha > low + 1e-9) & (alpha <= high + 1e-9)


def gate_rows(v2):
    if not (v2.has("prior") and "effective_alpha" in v2.columns):
        return []
    alpha = v2.scalar("effective_alpha")
    incoming = v2.errors("prior")
    final = v2.errors("prediction")
    rows = []
    for name, landmarks in LANDMARK_TYPES.items():
        columns = list(landmarks)
        for low, high in ALPHA_BINS:
            mask = _bin_mask(alpha[:, columns], low, high)
            label = f"({low},{high}]" if low != high else "0"
            count = int(mask.sum())
            if count == 0:
                empty = ("incoming_ale", "final_ale", "gain_mm", "fraction_improved")
                rows.append(
                    {"type": name, "alpha_bin": label, "count": 0, **dict.fromkeys(empty, "")}
                )
                continue
            incoming_values = incoming[:, columns][mask]
            final_values = final[:, columns][mask]
            rows.append(
                {
                    "type": name,
                    "alpha_bin": label,
                    "count": count,
                    "incoming_ale": float(incoming_values.mean()),
                    "final_ale": float(final_values.mean()),
                    "gain_mm": float((incoming_values - final_values).mean()),
                    "fraction_improved": float(np.mean(final_values < incoming_values)),
                }
            )
    return rows


def policy_rows(v2, folds):
    rows = []
    selected_iterations = {}
    for fold in folds:
        path = v2.run_dir / f"fold_{fold}" / "all23_mvsc" / "all23_policy.json"
        if not path.exists():
            continue
        policy = json.loads(path.read_text(encoding="utf-8"))
        iterations = policy["iterations"]
        if not isinstance(iterations, dict):
            iterations = {name: int(iterations) for name in LANDMARK_TYPES}
        selected_iterations[fold] = {name: int(value) for name, value in iterations.items()}
        rows.append(
            {
                "fold": fold,
                "accepted": bool(policy["accepted"]),
                **{f"iterations_{name}": value for name, value in iterations.items()},
                **{f"cap_{name}": value for name, value in policy["alpha_caps"].items()},
                **{f"proposed_cap_{name}": value for name, value in policy["proposed_alpha_caps"].items()},
                "base_ale": policy["base"]["ale"],
                "ungated_proposal_ale": policy["ungated_proposal"]["ale"],
                "proposed_ale": policy["proposed"]["ale"],
                "selected_ale": policy["selected"]["ale"],
            }
        )
    return rows, selected_iterations


def ungated_rows(v2, selected_iterations):
    """Selected-iteration proposal (alpha = 1) versus the gated output."""
    counts = {value for by_type in selected_iterations.values() for value in by_type.values()}
    if not selected_iterations or not all(v2.has(f"iter{count}") for count in counts):
        return []
    proposal = np.zeros((len(v2.keys), NUM_LANDMARKS))
    errors_by_iteration = {count: v2.errors(f"iter{count}") for count in counts}
    for row, fold in enumerate(v2.folds):
        for name, landmarks in LANDMARK_TYPES.items():
            columns = list(landmarks)
            count = selected_iterations[int(fold)][name]
            proposal[row, columns] = errors_by_iteration[count][row, columns]
    final = v2.errors("prediction")
    incoming = v2.errors("prior")
    return [
        {
            "set": name,
            "incoming_ale": float(incoming[:, list(landmarks)].mean()),
            "ungated_proposal_ale": float(proposal[:, list(landmarks)].mean()),
            "gated_final_ale": float(final[:, list(landmarks)].mean()),
            "gate_gain_vs_ungated_mm": float(
                proposal[:, list(landmarks)].mean() - final[:, list(landmarks)].mean()
            ),
        }
        for name, landmarks in SETS.items()
    ]


def _table(rows, columns):
    if not rows:
        return "_(veri yok)_\n"
    header = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = ""
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        body += "| " + " | ".join(cells) + " |\n"
    return header + body


def write_markdown(path, report):
    sets = list(SETS)
    lines = [
        "# AGH-Former v2 — validasyon tanı raporu\n",
        f"Fold-tahmini: {report['predictions']}, farklı katılımcı: {report['unique_participants']}. "
        "Test dosyası okunmadı. Güven aralıkları katılımcı kümesi bootstrap'ıdır.\n",
        "## Aşama tablosu (ALE, mm)\n",
        _table(report["stages"], ["model", "stage", *sets]),
        "## v2 − vNext (final)\n",
        _table(
            report["head_to_head"],
            ["set", "v2_ale", "vnext_ale", "v2_minus_vnext", "ci95_low", "ci95_high", "probability_v2_better"],
        ),
        "## Fold politikaları\n",
        _table(report["policies"], list(report["policies"][0]) if report["policies"] else []),
        "## Gate'siz öneri ve gate'li final\n",
        _table(
            report["ungated"],
            ["set", "incoming_ale", "ungated_proposal_ale", "gated_final_ale", "gate_gain_vs_ungated_mm"],
        ),
        "## Etkin alfa aralıkları\n",
        _table(
            report["alpha_bins"],
            ["type", "alpha_bin", "count", "incoming_ale", "final_ale", "gain_mm", "fraction_improved"],
        ),
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def analyze(v2_run, vnext_run, folds, output_dir, iterations=2000, seed=42):
    v2 = ValidationRun(v2_run, folds)
    vnext = ValidationRun(vnext_run, folds)
    if v2.keys != vnext.keys:
        raise RuntimeError("v2 and vNext validation fold-predictions differ")
    expert_gap = float(np.abs(v2.coordinates("expert") - vnext.coordinates("expert")).max())
    if expert_gap > 1e-2:
        raise RuntimeError(f"Expert coordinates differ between runs ({expert_gap:.4f} mm)")
    policies, selected_iterations = policy_rows(v2, folds)
    report = {
        "folds": list(folds),
        "predictions": len(v2.keys),
        "unique_participants": len(set(v2.participants)),
        "stages": stage_rows(v2, "v2", V2_STAGES) + stage_rows(vnext, "vnext", VNEXT_STAGES),
        "head_to_head": head_to_head(v2, vnext, iterations, seed),
        "policies": policies,
        "ungated": ungated_rows(v2, selected_iterations),
        "alpha_bins": gate_rows(v2),
        "test_files_read": False,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics_val.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output_dir / "stages_val.csv", report["stages"])
    write_csv(output_dir / "landmark_stages_v2_val.csv", landmark_stage_rows(v2, "v2", V2_STAGES))
    write_csv(
        output_dir / "landmark_stages_vnext_val.csv",
        landmark_stage_rows(vnext, "vnext", VNEXT_STAGES),
    )
    write_csv(output_dir / "head_to_head_val.csv", report["head_to_head"])
    write_csv(output_dir / "policies_val.csv", report["policies"])
    write_csv(output_dir / "alpha_bins_val.csv", report["alpha_bins"])
    write_markdown(output_dir / "diagnostics_val.md", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v2-run", required=True)
    parser.add_argument("--vnext-run", required=True)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    output_dir = args.output_dir or Path(args.v2_run) / "validation_diagnostics"
    report = analyze(args.v2_run, args.vnext_run, folds, output_dir, args.bootstrap_iters, args.seed)
    print(
        f"Wrote {output_dir}/diagnostics_val.md "
        f"({report['predictions']} fold-predictions, {report['unique_participants']} participants)",
        flush=True,
    )


if __name__ == "__main__":
    main()
