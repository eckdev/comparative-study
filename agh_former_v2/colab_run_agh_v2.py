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
            f"Validation gate is missing. Run the matching *_val_all_folds preset first: {gate_path}"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    decision = Path(output_dir) / "final_model_decision.json"
    if not gate.get("run_final_test", False) and not decision.exists():
        failed = [name for name, passed in gate.get("checks", {}).items() if not passed]
        raise RuntimeError(
            "Validation gate failed and no final-model decision was committed; "
            f"the test split stays sealed: {failed}"
        )
    # run_agh_v2.py re-checks the decision against the locked validation policies.
    return gate


def _replace_value(command, flag, value):
    command = list(command)
    command[command.index(flag) + 1] = str(value)
    return command


# v2.1 reuses the refiners trained by the v2 run and starts from the vNext
# Stage-2 gated output instead of the Stage-2 fusion centre.
VARIANTS = {
    "v2": {"output": "publication_cv_seed{seed}", "flags": []},
    "v21": {
        "output": "publication_cv_v21_seed{seed}",
        "flags": [
            "--stage-a-column",
            "neural_prediction",
            "--refiner-source-run",
            str(RUN_ROOT / "publication_cv_seed{seed}"),
        ],
    },
    # Final model: v2.1 without the OOF gate (see PREREGISTRATION_TR.md).
    "v2final": {
        "output": "publication_cv_v2final_seed{seed}",
        "flags": [
            "--stage-a-column",
            "neural_prediction",
            "--refiner-source-run",
            str(RUN_ROOT / "publication_cv_seed{seed}"),
            "--no-all23-gate",
        ],
    },
}
ACTIONS = {
    "val_all_folds": ["--validation-only"],
    "final_test": ["--final-test"],
    "gate_only": ["--gate-only"],
    "commit": ["--commit-final-model"],
}
SMOKE_FLAGS = [
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
    "--all23-mc-samples",
    "2",
    "--bootstrap-iters",
    "200",
]


def _split_preset(preset):
    variant, _, action = preset.partition("_")
    if variant not in VARIANTS or (action not in ACTIONS and preset != "v2_smoke"):
        raise ValueError(f"Unknown preset: {preset}")
    return variant, action


def command_for(preset, seed, stage_a_run, fold_indices=None):
    variant, action = _split_preset(preset)
    base = vnext_command_for("cv", seed, None)
    base[2] = str(SCRIPT)
    if preset == "v2_smoke":
        output_dir = RUN_ROOT / f"smoke_seed{seed}"
    else:
        output_dir = RUN_ROOT / VARIANTS[variant]["output"].format(seed=seed)
    command = _replace_value(base, "--output-dir", output_dir)
    command += [
        "--stage-a-run",
        str(stage_a_run),
        "--scratch-dir",
        str(SCRATCH_ROOT / preset),
    ]
    command += [flag.format(seed=seed) for flag in VARIANTS[variant]["flags"]]
    if preset == "v2_smoke":
        return command + SMOKE_FLAGS, output_dir
    command += ACTIONS[action]
    if fold_indices:
        command += ["--fold-indices", fold_indices]
    return command, output_dir


def diagnostics_commands(seed, stage_a_run, variant="v2"):
    """CPU-only: recompute the validation gate, then write the diagnostics report."""
    gate, output_dir = command_for(f"{variant}_gate_only", seed, stage_a_run)
    analysis = [
        sys.executable,
        "-u",
        str(CODE_ROOT / "agh_former_v2/analyze_validation.py"),
        "--v2-run",
        str(output_dir),
        "--vnext-run",
        str(stage_a_run),
    ]
    return [gate, analysis], output_dir


def main():
    parser = argparse.ArgumentParser(description="Colab runner for AGH-Former v2")
    parser.add_argument(
        "--preset",
        choices=(
            "v2_smoke",
            "v2_val_all_folds",
            "v2_diagnostics",
            "v2_final_test",
            "v21_val_all_folds",
            "v21_diagnostics",
            "v21_final_test",
            "v2final_val_all_folds",
            "v2final_diagnostics",
            "v2final_commit",
            "v2final_final_test",
        ),
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
    if args.preset.endswith("_diagnostics"):
        commands, _ = diagnostics_commands(
            args.seed, args.stage_a_run, variant=args.preset.split("_")[0]
        )
    else:
        command, output_dir = command_for(
            args.preset, args.seed, args.stage_a_run, args.fold_indices
        )
        if args.preset.endswith("_final_test"):
            require_validation_gate(output_dir)
        commands = [command]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    for command in commands:
        print("Running:", " ".join(map(str, command)), flush=True)
        subprocess.run(command, cwd=str(CODE_ROOT), check=True, env=environment)


if __name__ == "__main__":
    main()
