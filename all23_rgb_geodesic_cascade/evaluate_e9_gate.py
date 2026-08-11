#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_metrics(root, fold):
    root = Path(root)
    candidates = [root / f"fold_{fold}" / "metrics_val.json", root / f"fold{fold}" / "metrics_val.json"]
    for path in candidates:
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("Validation metrics not found:\n- " + "\n- ".join(map(str, candidates)))


def main():
    parser = argparse.ArgumentParser(description="Decide whether E9 is worth a full five-fold run")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-hard3-gain-mm", type=float, default=0.75)
    parser.add_argument("--min-overall-gain-mm", type=float, default=0.10)
    parser.add_argument("--max-core20-regression-mm", type=float, default=0.05)
    parser.add_argument("--max-candidate-overall-mm", type=float, default=2.25)
    parser.add_argument("--max-candidate-hard3-mm", type=float, default=4.5)
    args = parser.parse_args()

    baseline_path, baseline = load_metrics(args.baseline_root, args.fold)
    candidate_path, candidate = load_metrics(args.candidate_root, args.fold)
    changes = {
        "overall_gain_mm": baseline["overall"]["ale"] - candidate["overall"]["ale"],
        "core20_change_mm": candidate["core20"]["ale"] - baseline["core20"]["ale"],
        "hard3_gain_mm": baseline["hard3"]["ale"] - candidate["hard3"]["ale"],
        "p95_change_mm": candidate["overall"]["p95"] - baseline["overall"]["p95"],
        "max_change_mm": candidate["overall"]["max"] - baseline["overall"]["max"],
    }
    checks = {
        "overall_gain": changes["overall_gain_mm"] >= args.min_overall_gain_mm,
        "hard3_gain": changes["hard3_gain_mm"] >= args.min_hard3_gain_mm,
        "core20_preserved": changes["core20_change_mm"] <= args.max_core20_regression_mm,
        "p95_not_worse": changes["p95_change_mm"] <= 0.0,
        "max_not_worse": changes["max_change_mm"] <= 0.0,
        "absolute_overall_viability": (
            candidate["overall"]["ale"] <= args.max_candidate_overall_mm
        ),
        "absolute_hard3_viability": (
            candidate["hard3"]["ale"] <= args.max_candidate_hard3_mm
        ),
    }
    report = {
        "fold": args.fold,
        "baseline_metrics": str(baseline_path),
        "candidate_metrics": str(candidate_path),
        "baseline": {
            "overall_ale": baseline["overall"]["ale"],
            "core20_ale": baseline["core20"]["ale"],
            "hard3_ale": baseline["hard3"]["ale"],
            "p95": baseline["overall"]["p95"],
            "max": baseline["overall"]["max"],
        },
        "candidate": {
            "overall_ale": candidate["overall"]["ale"],
            "core20_ale": candidate["core20"]["ale"],
            "hard3_ale": candidate["hard3"]["ale"],
            "p95": candidate["overall"]["p95"],
            "max": candidate["overall"]["max"],
        },
        "changes": changes,
        "checks": checks,
        "run_full_cv": all(checks.values()),
        "two_mm_target_plausible": (
            candidate["overall"]["ale"] <= 2.10
            and candidate["core20"]["ale"] <= 1.95
            and candidate["hard3"]["ale"] <= 3.20
        ),
    }
    output = Path(args.output) if args.output else Path(args.candidate_root) / "e9_acceptance_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(
        "\nDecision:",
        "RUN e9_cv" if report["run_full_cv"] else "STOP; do not spend four more folds",
        flush=True,
    )


if __name__ == "__main__":
    main()
