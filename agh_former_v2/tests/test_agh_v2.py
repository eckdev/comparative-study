from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from agh_former_v2.run_agh_v2 import (
    policy_digest,
    read_stage_a,
    validation_gate,
    verify_against_shared_manifest,
)
from all23_rgb_geodesic_cascade.anatomy import NUM_LANDMARKS


def _write_predictions(path, sample_ids, errors_by_id, with_coarse=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id in sample_ids:
        for landmark in range(NUM_LANDMARKS):
            error = float(errors_by_id[sample_id][landmark])
            row = {"sample_id": sample_id, "landmark": landmark}
            for axis in "xyz":
                row[f"expert_{axis}"] = 0.0
                row[f"prediction_{axis}"] = error if axis == "x" else 0.0
                if with_coarse:
                    row[f"coarse_{axis}"] = landmark + (1.0 if axis == "y" else 0.0)
            row["error"] = error
            rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_read_stage_a_returns_requested_columns_only(tmp_path):
    path = tmp_path / "predictions_test.csv"
    ids = ["A", "B"]
    _write_predictions(path, ids, {key: np.ones(NUM_LANDMARKS) for key in ids})
    coordinates = read_stage_a(path, ["B"])
    assert set(coordinates) == {"B"}
    np.testing.assert_allclose(coordinates["B"][5], [5.0, 6.0, 5.0])
    with pytest.raises(RuntimeError):
        read_stage_a(path, ["missing"])


def test_policy_digest_detects_edits():
    policy = {"iterations": 2, "alpha_caps": {"corner": 1.0}, "accepted": True, "sweep": []}
    edited = {**policy, "alpha_caps": {"corner": 0.75}}
    assert policy_digest(policy) == policy_digest({**policy, "sweep": [1]})
    assert policy_digest(policy) != policy_digest(edited)


def test_validation_gate_compares_pooled_validation(tmp_path):
    ids = ["A", "B", "C"]
    good = {key: np.full(NUM_LANDMARKS, 1.0) for key in ids}
    worse = {key: np.full(NUM_LANDMARKS, 2.0) for key in ids}
    _write_predictions(tmp_path / "v2/fold_1/predictions_val.csv", ids, good)
    _write_predictions(tmp_path / "vnext/fold_1/predictions_val.csv", ids, worse)
    gate = validation_gate(tmp_path / "v2", tmp_path / "vnext", [1], seed=1, iterations=50)
    assert gate["run_final_test"] and gate["participants"] == 3
    _write_predictions(tmp_path / "v2/fold_1/predictions_val.csv", ids, {key: v * 3 for key, v in good.items()})
    gate = validation_gate(tmp_path / "v2", tmp_path / "vnext", [1], seed=1, iterations=50)
    assert not gate["run_final_test"]
    assert json.loads((tmp_path / "v2/validation_gate.json").read_text())["test_labels_consumed"] is False


def test_manifest_mismatch_is_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"folds": [{"fold": 1, "train": ["A"], "val": ["B"], "test": ["C"]}]})
    )
    assert verify_against_shared_manifest(1, {"train": ["A"], "val": ["B"], "test": ["C"]}, manifest)[
        "checked"
    ]
    with pytest.raises(RuntimeError):
        verify_against_shared_manifest(1, {"train": ["A"], "val": ["C"], "test": ["B"]}, manifest)


def test_colab_presets_build_parseable_commands():
    path = Path(__file__).resolve().parents[1] / "colab_run_agh_v2.py"
    spec = importlib.util.spec_from_file_location("colab_run_agh_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from agh_former_v2.run_agh_v2 import build_parser

    for preset, flag in (("v2_val_all_folds", "validation_only"), ("v2_final_test", "final_test")):
        command, output_dir = module.command_for(preset, 42, "/archive/run")
        args = build_parser().parse_args([str(value) for value in command[3:]])
        assert getattr(args, flag)
        assert args.stage_a_run == "/archive/run"
        assert Path(args.output_dir) == output_dir
        assert args.roi_points == 1024 and args.coarse_source == "stage1_oof"
    command, _ = module.command_for("v2_smoke", 42, "/archive/run")
    args = build_parser().parse_args([str(value) for value in command[3:]])
    assert args.validation_only and args.fold_indices == "1" and args.all23_epochs == 2
