import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.datasets.orthodontic_dataset import OrthodonticDataset


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
    parser = argparse.ArgumentParser(description="Check whether the orthodontic dataset matches PAL-Net assumptions.")
    parser.add_argument("--data-root", default="../../data/dataset")
    parser.add_argument("--surface-points", type=int, default=1000)
    parser.add_argument("--output", default="../runs/dataset_diagnostics.json")
    args = parser.parse_args()

    dataset = OrthodonticDataset(args.data_root, cache_dir=None, num_surface_points=args.surface_points)
    landmark_to_surface = []
    centroids = []
    bboxes = []
    landmarks = []
    sample_ids = []

    for idx in tqdm(range(len(dataset)), desc="diagnosing"):
        points, lm, vertices = dataset[idx]
        lm_np = lm.numpy()
        v_np = vertices.numpy()
        tree = cKDTree(v_np[:, :3])
        distances, _ = tree.query(lm_np, k=1)
        landmark_to_surface.append(distances)
        centroids.append(lm_np.mean(axis=0))
        bboxes.append(v_np[:, :3].max(axis=0) - v_np[:, :3].min(axis=0))
        landmarks.append(lm_np)
        sample_ids.append(dataset.samples[idx].sample_id)

    landmark_to_surface = np.asarray(landmark_to_surface)
    centroids = np.asarray(centroids)
    bboxes = np.asarray(bboxes)
    landmarks = np.asarray(landmarks)

    mean_lm = landmarks.mean(axis=0)
    mean_template_errors = np.linalg.norm(landmarks - mean_lm[None, :, :], axis=-1)

    worst_surface_flat = np.argsort(landmark_to_surface.ravel())[-10:][::-1]
    worst_surface = []
    for flat_idx in worst_surface_flat:
        sample_idx, landmark_idx = divmod(int(flat_idx), landmark_to_surface.shape[1])
        worst_surface.append(
            {
                "sample_id": sample_ids[sample_idx],
                "landmark": landmark_idx,
                "distance": float(landmark_to_surface[sample_idx, landmark_idx]),
            }
        )

    report = {
        "paired_samples": len(dataset),
        "missing_landmark_meshes": [str(path) for path in dataset.missing_landmarks],
        "landmark_count": 23,
        "landmark_to_nearest_vertex": summarize(landmark_to_surface.ravel()),
        "mean_template_landmark_error": summarize(mean_template_errors.ravel()),
        "centroid_std_xyz": centroids.std(axis=0).tolist(),
        "bbox_mean_xyz": bboxes.mean(axis=0).tolist(),
        "bbox_median_xyz": np.median(bboxes, axis=0).tolist(),
        "worst_landmark_to_surface": worst_surface,
        "interpretation": [
            "PAL-Net LA-FAS loader applies transformation_matrix.npy to mesh and landmarks.",
            "This dataset has no transformation_matrix.npy, so raw PLY scans must already be in a common coordinate frame or be aligned before PAL-Net can match paper-level ALE.",
            "Large centroid_std_xyz or large mean_template_landmark_error means mean-landmark patch centers are weak initializers.",
            "Large landmark_to_nearest_vertex values indicate annotation/mesh coordinate mismatches or off-surface landmarks.",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
