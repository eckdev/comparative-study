from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from agh_former_v2.analyze_validation import analyze, open_validation_csv
from agh_former_v2.run_agh_v2 import (
    check_locked_validation,
    cluster_bootstrap_delta,
    policy_digest,
    pooled_csv_errors,
    read_stage_a,
    validation_gate,
    verify_against_shared_manifest,
)
from all23_rgb_geodesic_cascade.anatomy import NUM_LANDMARKS


def _write_predictions(path, sample_ids, errors_by_id, with_coarse=True, extra=None):
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
            for name, value in (extra or {}).items():
                for axis in "xyz":
                    row[f"{name}_{axis}"] = value(error) if axis == "x" else 0.0
            if extra is not None:
                row["effective_alpha"] = 0.5
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
    assert gate["run_final_test"] and gate["predictions"] == 3 and gate["unique_participants"] == 3
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


def test_participant_in_two_folds_counts_as_two_predictions(tmp_path):
    first = {"A": np.full(NUM_LANDMARKS, 1.0), "B": np.full(NUM_LANDMARKS, 2.0)}
    second = {"A": np.full(NUM_LANDMARKS, 5.0), "C": np.full(NUM_LANDMARKS, 3.0)}
    _write_predictions(tmp_path / "fold_1/predictions_val.csv", list(first), first)
    _write_predictions(tmp_path / "fold_2/predictions_val.csv", list(second), second)
    keys, matrix, participants = pooled_csv_errors(
        {1: tmp_path / "fold_1/predictions_val.csv", 2: tmp_path / "fold_2/predictions_val.csv"}
    )
    assert keys == [(1, "A"), (1, "B"), (2, "A"), (2, "C")]
    assert participants == ["A", "B", "A", "C"]
    # The fold-1 prediction of A is no longer overwritten by its fold-2 row.
    assert matrix.mean() == pytest.approx((1 + 2 + 5 + 3) / 4)


def test_cluster_bootstrap_keeps_participant_rows_together():
    base = np.zeros((4, 2))
    final = np.asarray([[1.0, 1.0], [1.0, 1.0], [-3.0, -3.0], [0.0, 0.0]])
    result = cluster_bootstrap_delta(base, final, ["A", "A", "B", "C"], iterations=2000, seed=3)
    assert result["clusters"] == 3 and result["rows"] == 4
    assert result["delta_ale"] == pytest.approx(final.mean())
    # Only three distinct participant draws exist, so A's two rows always move together:
    # every bootstrap mean is a weighted mean of the per-participant totals.
    single = cluster_bootstrap_delta(base[:1], final[:1], ["A"], iterations=10, seed=3)
    assert single["ci95"] == [1.0, 1.0]


def test_gate_only_reads_validation_files_only(tmp_path, monkeypatch):
    ids = ["A"]
    errors = {"A": np.full(NUM_LANDMARKS, 1.0)}
    for run in ("v2", "vnext"):
        _write_predictions(tmp_path / run / "fold_1/predictions_val.csv", ids, errors)
        (tmp_path / run / "fold_1/predictions_test.csv").write_text("sealed")
    opened = []
    real_open = open

    def tracking_open(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", tracking_open)
    validation_gate(tmp_path / "v2", tmp_path / "vnext", [1], seed=1, iterations=10)
    assert "predictions_test.csv" not in opened
    with pytest.raises(AssertionError):
        open_validation_csv(tmp_path / "v2/fold_1/predictions_test.csv")


def test_analyze_validation_builds_stage_gate_and_head_to_head_tables(tmp_path):
    first = {"A": np.full(NUM_LANDMARKS, 1.0), "B": np.full(NUM_LANDMARKS, 2.0)}
    second = {"A": np.full(NUM_LANDMARKS, 1.5), "C": np.full(NUM_LANDMARKS, 0.5)}
    stages = {
        "stage_a": lambda error: error + 2.0,
        "prior": lambda error: error + 1.0,
        "iter1": lambda error: error + 0.5,
        "iter2": lambda error: error + 0.25,
    }
    for fold, errors in ((1, first), (2, second)):
        _write_predictions(
            tmp_path / f"v2/fold_{fold}/predictions_val.csv", list(errors), errors, extra=stages
        )
        _write_predictions(
            tmp_path / f"vnext/fold_{fold}/predictions_val.csv",
            list(errors),
            {key: value + 0.1 for key, value in errors.items()},
        )
        policy_dir = tmp_path / f"v2/fold_{fold}/all23_mvsc"
        policy_dir.mkdir(parents=True)
        caps = {"corner": 1.0, "contour": 0.5, "texture_boundary": 0.0}
        summary = {"ale": 1.0}
        (policy_dir / "all23_policy.json").write_text(
            json.dumps(
                {
                    "accepted": True,
                    "iterations": 2,
                    "alpha_caps": caps,
                    "proposed_alpha_caps": caps,
                    "base": summary,
                    "ungated_proposal": summary,
                    "proposed": summary,
                    "selected": summary,
                }
            )
        )
    report = analyze(tmp_path / "v2", tmp_path / "vnext", [1, 2], tmp_path / "out", iterations=50)
    assert report["predictions"] == 4 and report["unique_participants"] == 3
    v2_stages = [row["stage"] for row in report["stages"] if row["model"] == "v2"]
    assert v2_stages == ["stage_a", "prior", "iter1", "iter2", "prediction"]
    all23 = next(row for row in report["head_to_head"] if row["set"] == "all23")
    assert all23["v2_minus_vnext"] == pytest.approx(-0.1)
    assert {row["alpha_bin"] for row in report["alpha_bins"] if row["count"]} == {"(0.25,0.5]"}
    ungated = next(row for row in report["ungated"] if row["set"] == "all23")
    assert ungated["gate_gain_vs_ungated_mm"] == pytest.approx(0.25)
    assert (tmp_path / "out/diagnostics_val.md").exists()


def test_final_test_requires_reproduced_validation(tmp_path):
    path = tmp_path / "locked.npz"
    prediction = np.zeros((2, NUM_LANDMARKS, 3), dtype=np.float32)
    np.savez_compressed(path, sample_ids=np.asarray(["A", "B"]), prediction=prediction)
    assert check_locked_validation(path, ["A", "B"], prediction + 0.001) < 0.01
    with pytest.raises(RuntimeError, match="drifted"):
        check_locked_validation(path, ["A", "B"], prediction + 0.5)
    with pytest.raises(RuntimeError, match="participants differ"):
        check_locked_validation(path, ["B", "A"], prediction)
