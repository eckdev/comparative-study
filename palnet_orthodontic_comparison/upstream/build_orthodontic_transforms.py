import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from trimesh.registration import procrustes
from trimesh.transformations import transform_points

from src.datasets.orthodontic_dataset import OrthodonticDataset


def compute_template(landmarks, mode, reference_index):
    if mode == "mean":
        return landmarks.mean(axis=0)
    if mode == "reference":
        return landmarks[reference_index]
    raise ValueError(f"Unknown template mode: {mode}")


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create PAL-Net-style transformation_matrix.npy files for the orthodontic PLY dataset using trimesh."
    )
    parser.add_argument("--data-root", default="../../data/dataset")
    parser.add_argument("--output-dir", default="../transforms/orthodontic_procrustes_rigid")
    parser.add_argument("--template", choices=["mean", "reference"], default="mean")
    parser.add_argument("--reference-index", type=int, default=0)
    parser.add_argument("--scale", action="store_true", help="Allow similarity scaling. Off by default to preserve mm-like units.")
    parser.add_argument("--reflection", action="store_true", help="Allow reflections. Off by default for anatomical consistency.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OrthodonticDataset(args.data_root, cache_dir=None, num_surface_points=1000)
    landmarks = []
    vertices = []
    for idx in tqdm(range(len(dataset)), desc="loading"):
        _, lm, raw_vertices = dataset[idx]
        landmarks.append(lm.numpy())
        vertices.append(raw_vertices.numpy())
    landmarks = np.asarray(landmarks, dtype=np.float64)

    template = compute_template(landmarks, args.template, args.reference_index)
    np.save(output_dir / "template_landmarks.npy", template.astype(np.float32))

    rows = []
    all_before_template = []
    all_after_template = []
    all_surface_after = []

    for idx, sample in enumerate(tqdm(dataset.samples, desc="writing transforms")):
        matrix, transformed_landmarks, cost = procrustes(
            landmarks[idx],
            template,
            reflection=args.reflection,
            translation=True,
            scale=args.scale,
            return_cost=True,
        )
        rel_parent = sample.mesh_path.relative_to(dataset.root_dir).parent
        sample_out_dir = output_dir / rel_parent
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = sample_out_dir / f"{sample.mesh_path.stem}_transformation_matrix.npy"
        np.save(matrix_path, matrix.astype(np.float32))

        transformed_vertices = transform_points(vertices[idx], matrix)
        tree = cKDTree(transformed_vertices[:, :3])
        surface_distances, _ = tree.query(transformed_landmarks, k=1)

        before_template = np.linalg.norm(landmarks[idx] - template, axis=1)
        after_template = np.linalg.norm(transformed_landmarks - template, axis=1)
        all_before_template.extend(before_template.tolist())
        all_after_template.extend(after_template.tolist())
        all_surface_after.extend(surface_distances.tolist())

        rows.append(
            {
                "sample_id": sample.sample_id,
                "class": sample.class_name,
                "gender": sample.gender,
                "subject_id": sample.subject_id,
                "matrix_path": str(matrix_path),
                "procrustes_cost": float(cost),
                "template_error_before": float(before_template.mean()),
                "template_error_after": float(after_template.mean()),
                "surface_distance_after": float(surface_distances.mean()),
            }
        )

    with open(output_dir / "transform_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "method": "trimesh.registration.procrustes",
        "template": args.template,
        "reference_index": args.reference_index,
        "scale": args.scale,
        "reflection": args.reflection,
        "n_samples": len(dataset),
        "missing_landmark_meshes": [str(path) for path in dataset.missing_landmarks],
        "template_error_before": summarize(all_before_template),
        "template_error_after": summarize(all_after_template),
        "landmark_to_transformed_surface": summarize(all_surface_after),
        "note": "Matrices map each sample's original PLY/landmark coordinates into the common template coordinate frame.",
    }
    (output_dir / "transform_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
