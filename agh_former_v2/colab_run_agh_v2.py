#!/usr/bin/env python3
"""Colab presets for AGH-Former v2 (phase 1).

The global-stage flags are taken verbatim from the AGH-Former vNext CV preset so
that the shared preprocessing root (alignment, mesh cache, Stage-1 OOF centres)
and the archived vNext Stage-2 fusion outputs are reused unchanged.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path("/content/comparative-study")
sys.path.insert(0, str(CODE_ROOT))

from agh_former_vnext_orthodontic_comparison.colab_run_aghformer_vnext import (  # noqa: E402
    DATA_ROOT,
    RUN_ROOT as VNEXT_RUN_ROOT,
    command_for as vnext_command_for,
)

RUN_ROOT = Path("/content/drive/MyDrive/orthodontic/agh_v2_runs")
SCRATCH_ROOT = Path("/content/agh_v2_scratch")
STAGE_A_RUN = VNEXT_RUN_ROOT / "publication_cv_seed42"
SCRIPT = CODE_ROOT / "agh_former_v2/run_agh_v2.py"


def require_validation_gate(output_dir):
    gate_path = Path(output_dir) / "validation_gate.json"
    if not gate_path.exists():
        raise RuntimeError(
            f"Validation gate is missing. Run --preset v2_val_all_folds first: {gate_path}"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("run_final_test", False):
        failed = [name for name, passed in gate.get("checks", {}).items() if not passed]
        raise RuntimeError(f"Validation gate failed; the test split stays sealed: {failed}")
    return gate


def _replace_value(command, flag, value):
    command = list(command)
    command[command.index(flag) + 1] = str(value)
    return command


def command_for(preset, seed, stage_a_run, fold_indices=None):
    base = vnext_command_for("cv", seed, None)
    base[2] = str(SCRIPT)
    output_dir = RUN_ROOT / (
        f"smoke_seed{seed}" if preset == "v2_smoke" else f"publication_cv_seed{seed}"
    )
    command = _replace_value(base, "--output-dir", output_dir)
    command += [
        "--stage-a-run",
        str(stage_a_run),
        "--scratch-dir",
        str(SCRATCH_ROOT / preset),
    ]
    if preset == "v2_smoke":
        command += [
            "--fold-indices",
            "1",
            "--validation-only",
            "--all23-epochs",
            "2",
            "--all23-min-epochs",
            "1",
            "--all23-patience",
            "1",
            "--all23-gate-folds",
            "2",
            "--all23-gate-epochs",
            "40",
            "--all23-image-size",
            "48",
            "--bootstrap-iters",
            "200",
        ]
    elif preset == "v2_val_all_folds":
        command.append("--validation-only")
    elif preset == "v2_final_test":
        command.append("--final-test")
    else:
        raise ValueError(f"Unknown preset: {preset}")
    if fold_indices and preset != "v2_smoke":
        command += ["--fold-indices", fold_indices]
    return command, output_dir


def main():
    parser = argparse.ArgumentParser(description="Colab runner for AGH-Former v2")
    parser.add_argument(
        "--preset",
        choices=("v2_smoke", "v2_val_all_folds", "v2_final_test"),
        default="v2_smoke",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-indices", default=None)
    parser.add_argument("--stage-a-run", default=str(STAGE_A_RUN))
    args = parser.parse_args()
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")
    if not Path(args.stage_a_run, "fold_1", "predictions_val.csv").exists():
        raise FileNotFoundError(f"Archived vNext run not found: {args.stage_a_run}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    command, output_dir = command_for(
        args.preset, args.seed, args.stage_a_run, args.fold_indices
    )
    if args.preset == "v2_final_test":
        require_validation_gate(output_dir)
    print("Running:", " ".join(map(str, command)), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=str(CODE_ROOT), check=True, env=environment)


if __name__ == "__main__":
    main()
