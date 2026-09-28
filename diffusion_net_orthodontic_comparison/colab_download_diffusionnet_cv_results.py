#!/usr/bin/env python3
"""Create a compact, analysis-ready archive from DiffusionNet CV results.

The default bundle intentionally excludes checkpoints and operator/point caches.
It includes fold predictions, metrics, training histories, split/provenance files,
and derived outlier tables. Run with ``%run ... --download`` in Google Colab to
trigger a browser download after the archive has been verified.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RUN_DIR = Path(
    "/content/drive/MyDrive/orthodontic/diffusion_runs/"
    "diffusionnet_publication_cv_seed42"
)
DEFAULT_PREPROCESSING_ROOT = Path(
    "/content/drive/MyDrive/orthodontic/all23_rgb_geodesic_runs/"
    "publication_cv_stage1_v4_seed42"
)
LIGHTWEIGHT_SUFFIXES = {".csv", ".json", ".log", ".md", ".txt"}
ERROR_COLUMNS = ("error", "localization_error")
PREPROCESSING_FILES = (
    "split_and_leakage_report.json",
    "normalization.json",
    "candidate_oracle_pretrain.json",
    "alignment/alignment_report.json",
    "alignment/mesh_only_transforms.npz",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def error_of(row):
    for key in ERROR_COLUMNS:
        value = row.get(key)
        if value not in (None, ""):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"Non-finite prediction error: {value!r}")
            return result
    raise KeyError(f"Prediction row has no error column; expected one of {ERROR_COLUMNS}")


def percentile(values, q):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values):
    values = [float(value) for value in values]
    if not values:
        return {}
    return {
        "n": len(values),
        "ale": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "sdr_at_2mm": sum(value <= 2.0 for value in values) / len(values),
        "sdr_at_3mm": sum(value <= 3.0 for value in values) / len(values),
        "sdr_at_4mm": sum(value <= 4.0 for value in values) / len(values),
    }


def csv_bytes(rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    output = io.StringIO(newline="")
    if fieldnames:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def json_bytes(payload):
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def reported_ale(metrics_path):
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in ("diffusionnet_heatmap", "pooled", "overall"):
        value = payload.get(key)
        if isinstance(value, dict) and value.get("ale") is not None:
            return float(value["ale"])
    if payload.get("ale") is not None:
        return float(payload["ale"])
    return None


def collect_prediction_rows(run_dir, folds, allow_partial):
    all_rows = []
    fold_reports = []
    missing_required = []
    completed = []
    for fold in range(1, folds + 1):
        fold_dir = run_dir / f"fold_{fold}"
        metrics_path = fold_dir / "metrics.json"
        predictions_path = fold_dir / "predictions_test.csv"
        missing = [
            str(path.relative_to(run_dir))
            for path in (metrics_path, predictions_path)
            if not path.exists()
        ]
        if missing:
            missing_required.extend(missing)
            if allow_partial:
                continue
            raise FileNotFoundError(
                f"Fold {fold} is incomplete; missing: {', '.join(missing)}"
            )

        with open(predictions_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["fold"] = fold
            row["error_mm"] = error_of(row)
        all_rows.extend(rows)
        completed.append(fold)
        values = [row["error_mm"] for row in rows]
        sample_ids = {row.get("sample_id", "") for row in rows}
        computed = summarize(values)
        source_ale = reported_ale(metrics_path)
        fold_reports.append(
            {
                "fold": fold,
                "samples": len(sample_ids),
                "prediction_rows": len(rows),
                "reported_ale": source_ale,
                "recomputed_ale": computed.get("ale"),
                "ale_absolute_difference": (
                    abs(source_ale - computed["ale"])
                    if source_ale is not None and computed
                    else None
                ),
                "median": computed.get("median"),
                "p95": computed.get("p95"),
                "max": computed.get("max"),
            }
        )
    return all_rows, fold_reports, completed, missing_required


def build_sample_summary(rows):
    grouped = defaultdict(list)
    metadata = {}
    for row in rows:
        key = (int(row["fold"]), row["sample_id"])
        grouped[key].append(row)
        metadata[key] = row

    result = []
    for key, current in grouped.items():
        errors = [row["error_mm"] for row in current]
        core20 = [
            row["error_mm"] for row in current if 1 <= int(row["landmark"]) <= 20
        ]
        hard3 = [
            row["error_mm"] for row in current if int(row["landmark"]) in (0, 21, 22)
        ]
        worst = max(current, key=lambda row: row["error_mm"])
        meta = metadata[key]
        result.append(
            {
                "fold": key[0],
                "sample_id": key[1],
                "class": meta.get("class", ""),
                "gender": meta.get("gender", ""),
                "subject_id": meta.get("subject_id", ""),
                "n_landmarks": len(current),
                "ale": statistics.fmean(errors),
                "median": statistics.median(errors),
                "p90": percentile(errors, 0.90),
                "p95": percentile(errors, 0.95),
                "max": max(errors),
                "core20_ale": statistics.fmean(core20) if core20 else "",
                "hard3_ale": statistics.fmean(hard3) if hard3 else "",
                "worst_landmark": int(worst["landmark"]),
                "worst_error_mm": worst["error_mm"],
            }
        )
    return sorted(result, key=lambda row: float(row["ale"]), reverse=True)


def build_integrity_report(
    run_dir,
    folds,
    rows,
    fold_reports,
    completed,
    missing_required,
    preprocessing_root,
    skip_preprocessing,
):
    sample_counts = Counter((int(row["fold"]), row["sample_id"]) for row in rows)
    invalid_landmark_counts = [
        {"fold": fold, "sample_id": sample_id, "rows": count}
        for (fold, sample_id), count in sorted(sample_counts.items())
        if count != 23
    ]
    folds_by_sample = defaultdict(set)
    for row in rows:
        folds_by_sample[row["sample_id"]].add(int(row["fold"]))
    duplicate_outer_test_samples = {
        sample_id: sorted(values)
        for sample_id, values in sorted(folds_by_sample.items())
        if len(values) > 1
    }
    metric_mismatches = [
        report
        for report in fold_reports
        if report["ale_absolute_difference"] is not None
        and report["ale_absolute_difference"] > 1e-4
    ]

    preprocessing_checks = []
    if not skip_preprocessing:
        for fold in completed:
            alignment_path = (
                preprocessing_root / f"fold_{fold}" / "alignment" / "alignment_report.json"
            )
            check = {
                "fold": fold,
                "alignment_report_exists": alignment_path.exists(),
                "uses_expert_landmarks": None,
                "scale": None,
                "label_free_scale_preserving": False,
            }
            if alignment_path.exists():
                payload = json.loads(alignment_path.read_text(encoding="utf-8"))
                check["uses_expert_landmarks"] = payload.get("uses_expert_landmarks")
                check["scale"] = payload.get("scale")
                check["label_free_scale_preserving"] = (
                    payload.get("uses_expert_landmarks") is False
                    and payload.get("scale") is False
                )
            preprocessing_checks.append(check)

    worst = max(rows, key=lambda row: row["error_mm"]) if rows else None
    complete = len(completed) == folds and not missing_required
    valid = (
        complete
        and not invalid_landmark_counts
        and not duplicate_outer_test_samples
        and not metric_mismatches
    )
    if preprocessing_checks:
        valid = valid and all(
            check["label_free_scale_preserving"] for check in preprocessing_checks
        )
    return {
        "schema_version": 1,
        "source_run_dir": str(run_dir),
        "requested_folds": folds,
        "completed_folds": completed,
        "complete": complete,
        "integrity_passed": valid,
        "unique_outer_test_samples": len(folds_by_sample),
        "prediction_rows": len(rows),
        "missing_required_files": missing_required,
        "invalid_landmark_row_counts": invalid_landmark_counts,
        "duplicate_outer_test_samples": duplicate_outer_test_samples,
        "fold_metric_mismatches": metric_mismatches,
        "preprocessing_checks": preprocessing_checks,
        "recomputed_pooled_metrics": summarize([row["error_mm"] for row in rows]),
        "worst_prediction": (
            {
                "fold": int(worst["fold"]),
                "sample_id": worst["sample_id"],
                "landmark": int(worst["landmark"]),
                "error_mm": worst["error_mm"],
            }
            if worst
            else None
        ),
    }


def add_source_file(archive, source, archive_name, manifest_files):
    archive.write(source, archive_name)
    manifest_files.append(
        {
            "archive_path": archive_name,
            "source_path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    )


def lightweight_run_files(run_dir, completed, include_checkpoints):
    files = []
    files.extend(
        path
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.suffix.lower() in LIGHTWEIGHT_SUFFIXES
    )
    for fold in completed:
        fold_dir = run_dir / f"fold_{fold}"
        files.extend(
            path
            for path in sorted(fold_dir.iterdir())
            if path.is_file() and path.suffix.lower() in LIGHTWEIGHT_SUFFIXES
        )
        if include_checkpoints:
            checkpoint = fold_dir / "best_model.pth"
            if checkpoint.exists():
                files.append(checkpoint)
    return files


def create_bundle(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    preprocessing_root = Path(args.preprocessing_root).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"DiffusionNet run directory not found: {run_dir}")
    if not args.skip_preprocessing and not preprocessing_root.exists():
        raise FileNotFoundError(
            f"Preprocessing directory not found: {preprocessing_root}. "
            "Use --skip-preprocessing only if provenance files are unavailable."
        )

    output = Path(args.output).expanduser() if args.output else Path(
        "/content" if Path("/content").exists() else Path.cwd()
    ) / f"{run_dir.name}_analysis_bundle.zip"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    rows, fold_reports, completed, missing_required = collect_prediction_rows(
        run_dir, args.folds, args.allow_partial
    )
    integrity = build_integrity_report(
        run_dir,
        args.folds,
        rows,
        fold_reports,
        completed,
        missing_required,
        preprocessing_root,
        args.skip_preprocessing,
    )
    sample_summary = build_sample_summary(rows)
    worst_rows = sorted(rows, key=lambda row: row["error_mm"], reverse=True)[
        : args.top_errors
    ]

    bundle_root = f"{run_dir.name}_analysis"
    manifest_files = []
    generated = {}
    generated["analysis/fold_completion.csv"] = csv_bytes(fold_reports)
    generated["analysis/sample_error_summary.csv"] = csv_bytes(sample_summary)
    worst_fieldnames = ["fold"] + [
        key for key in rows[0] if key not in ("fold", "error_mm")
    ] + ["error_mm"] if rows else []
    generated["analysis/worst_landmark_predictions.csv"] = csv_bytes(
        worst_rows, worst_fieldnames
    )
    generated["analysis/integrity_report.json"] = json_bytes(integrity)

    readme = f"""DiffusionNet 5-fold analysis bundle

Source run: {run_dir}
Created (UTC): {datetime.now(timezone.utc).isoformat()}
Completed folds: {len(completed)}/{args.folds}
Unique outer-test samples: {integrity['unique_outer_test_samples']}
Prediction rows: {integrity['prediction_rows']}
Integrity passed: {integrity['integrity_passed']}

Start with:
  analysis/integrity_report.json
  analysis/fold_completion.csv
  analysis/sample_error_summary.csv
  analysis/worst_landmark_predictions.csv
  run/summary_metrics.json
  run/summary_landmark_metrics.csv
  run/pooled_predictions_test.csv

The default archive excludes patient meshes, point/operator caches, and model
checkpoints. Expert and predicted coordinates in prediction CSV files are kept
because they are required for error auditing. Checkpoints are included only
when --include-checkpoints is explicitly supplied.
"""
    generated["README_TR.txt"] = readme.encode("utf-8")

    source_files = lightweight_run_files(
        run_dir, completed, args.include_checkpoints
    )
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(output, "w", compression=compression, compresslevel=6) as archive:
        for source in source_files:
            relative = source.relative_to(run_dir)
            archive_name = f"{bundle_root}/run/{relative.as_posix()}"
            add_source_file(archive, source, archive_name, manifest_files)

        if not args.skip_preprocessing:
            for fold in completed:
                fold_root = preprocessing_root / f"fold_{fold}"
                for relative_name in PREPROCESSING_FILES:
                    source = fold_root / relative_name
                    if not source.exists():
                        continue
                    archive_name = (
                        f"{bundle_root}/preprocessing/fold_{fold}/{relative_name}"
                    )
                    add_source_file(archive, source, archive_name, manifest_files)

        for relative_name, content in generated.items():
            archive.writestr(f"{bundle_root}/{relative_name}", content)

        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_dir": str(run_dir),
            "source_preprocessing_root": (
                None if args.skip_preprocessing else str(preprocessing_root)
            ),
            "include_checkpoints": args.include_checkpoints,
            "completed_folds": completed,
            "source_files": manifest_files,
            "generated_files": sorted(generated),
            "integrity_report": integrity,
        }
        archive.writestr(f"{bundle_root}/MANIFEST.json", json_bytes(manifest))

    print(f"Archive created: {output}", flush=True)
    print(f"Archive size: {output.stat().st_size / (1024 * 1024):.2f} MiB", flush=True)
    print(
        f"Integrity: {'PASS' if integrity['integrity_passed'] else 'CHECK REQUIRED'}; "
        f"folds={len(completed)}/{args.folds}, "
        f"samples={integrity['unique_outer_test_samples']}, "
        f"rows={integrity['prediction_rows']}",
        flush=True,
    )
    worst = integrity.get("worst_prediction")
    if worst:
        print(
            "Worst prediction: "
            f"fold={worst['fold']} sample={worst['sample_id']} "
            f"landmark={worst['landmark']} error={worst['error_mm']:.4f} mm",
            flush=True,
        )
    return output


def trigger_colab_download(path):
    try:
        from google.colab import files  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "--download is available only inside a Google Colab notebook. "
            f"The archive was still created at: {path}"
        ) from exc
    files.download(str(path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Package DiffusionNet CV results for local detailed analysis."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--preprocessing-root", type=Path, default=DEFAULT_PREPROCESSING_ROOT
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--top-errors", type=int, default=250)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Trigger google.colab.files.download after creating the archive.",
    )
    args = parser.parse_args(argv)
    if args.folds < 1:
        parser.error("--folds must be positive")
    if args.top_errors < 1:
        parser.error("--top-errors must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = create_bundle(args)
    if args.download:
        trigger_colab_download(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
