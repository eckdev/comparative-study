import csv
from pathlib import Path

import numpy as np

from .data import apply_matrix, read_landmarks
from .utils import write_json


def rigid_transform_matrix(source, target, reflection=False):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src = source - src_centroid
    tgt = target - tgt_centroid
    h = src.T @ tgt
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if not reflection and np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = tgt_centroid - r @ src_centroid
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = r.astype(np.float32)
    matrix[:3, 3] = t.astype(np.float32)
    return matrix


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def build_fold_alignment(samples, split_indices, output_dir, reflection=False):
    output_dir = Path(output_dir)
    transform_dir = output_dir / "transforms"
    transform_dir.mkdir(parents=True, exist_ok=True)
    train_landmarks = [read_landmarks(samples[idx].landmark_path) for idx in split_indices["train"]]
    template = np.asarray(train_landmarks, dtype=np.float64).mean(axis=0)
    np.save(output_dir / "train_template_landmarks.npy", template.astype(np.float32))
    transforms = {}
    rows = []
    before_all = []
    after_all = []
    for idxs in split_indices.values():
        for idx in idxs:
            sample = samples[idx]
            landmarks = read_landmarks(sample.landmark_path)
            matrix = rigid_transform_matrix(landmarks, template, reflection=reflection)
            transformed = apply_matrix(landmarks, matrix)
            before = np.linalg.norm(landmarks - template, axis=1)
            after = np.linalg.norm(transformed - template, axis=1)
            before_all.extend(before.tolist())
            after_all.extend(after.tolist())
            rel_parent = sample.mesh_path.parent.name
            class_dir = sample.class_name
            sample_dir = transform_dir / class_dir / rel_parent
            sample_dir.mkdir(parents=True, exist_ok=True)
            matrix_path = sample_dir / f"{sample.mesh_path.stem}_transformation_matrix.npy"
            np.save(matrix_path, matrix.astype(np.float32))
            transforms[sample.sample_id] = matrix.astype(np.float32)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "class": sample.class_name,
                    "gender": sample.gender,
                    "subject_id": sample.subject_id,
                    "split": next(name for name, ids in split_indices.items() if idx in ids),
                    "matrix_path": str(matrix_path),
                    "template_error_before": float(before.mean()),
                    "template_error_after": float(after.mean()),
                }
            )
    with open(output_dir / "alignment_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "method": "train_template_rigid_kabsch",
        "scale": False,
        "reflection": bool(reflection),
        "template_fit_samples": [samples[idx].sample_id for idx in split_indices["train"]],
        "uses_validation_or_test_landmarks_for_template": False,
        "uses_validation_or_test_landmarks_for_individual_rigid_alignment": True,
        "label_leakage_note": "Val/test landmarks are not used for template fitting, but are used to compute per-sample rigid transforms. Use mesh-only registration for strictly label-free inference.",
        "template_error_before": summarize(before_all),
        "template_error_after": summarize(after_all),
    }
    write_json(output_dir / "alignment_report.json", report)
    return transforms, report
