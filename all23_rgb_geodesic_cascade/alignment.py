"""Label-free, scale-preserving mesh registration utilities."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def apply_transform(points, matrix):
    points = np.asarray(points, dtype=np.float32)
    homogeneous = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    return (homogeneous @ np.asarray(matrix, dtype=np.float32).T)[:, :3]


def rotate_vectors(vectors, matrix):
    values = np.asarray(vectors, dtype=np.float32) @ np.asarray(matrix, dtype=np.float32)[:3, :3].T
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-8, None)


def deterministic_surface(mesh, count, seed):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if len(vertices) <= count:
        return vertices
    rng = np.random.default_rng(seed)
    return vertices[rng.choice(len(vertices), count, replace=False)]


def mesh_descriptor(points):
    centered = points - np.median(points, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    extents = np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
    return np.concatenate([np.sqrt(np.maximum(eigenvalues, 0)), np.sort(extents)[::-1]])


def select_train_medoid(samples, mesh_loader, sample_points=2048, seed=42):
    descriptors = []
    for offset, sample in enumerate(samples):
        mesh = mesh_loader(sample.mesh_path)
        points = deterministic_surface(mesh, sample_points, seed + offset)
        descriptors.append(mesh_descriptor(points))
    descriptors = np.asarray(descriptors, dtype=np.float64)
    scale = np.maximum(np.median(np.abs(descriptors - np.median(descriptors, axis=0)), axis=0), 1e-6)
    normalized = (descriptors - np.median(descriptors, axis=0)) / scale
    return int(np.argmin(np.linalg.norm(normalized, axis=1)))


def _pca_basis(points):
    centered = points - np.median(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def pca_initial_transform(source, target, max_rotation_degrees=30.0):
    source_center = np.median(source, axis=0)
    target_center = np.median(target, axis=0)
    source_basis = _pca_basis(source)
    target_basis = _pca_basis(target)
    target_tree = cKDTree(target)
    # Orthodontic scanners usually share a camera frame. Preserve that orientation
    # when simple robust centering beats a PCA solution.
    identity_rotation = np.eye(3, dtype=np.float64)
    identity_translation = target_center - source_center
    identity_moved = source + identity_translation
    best = (
        float(np.median(target_tree.query(identity_moved, k=1, workers=-1)[0])),
        identity_rotation,
        identity_translation,
    )
    for signs in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        signed = source_basis @ np.diag(signs)
        rotation = target_basis @ signed.T
        if np.linalg.det(rotation) < 0:
            continue
        if np.degrees(Rotation.from_matrix(rotation).magnitude()) > float(max_rotation_degrees):
            continue
        translation = target_center - rotation @ source_center
        transformed = source @ rotation.T + translation
        distances = target_tree.query(transformed, k=1, workers=-1)[0]
        score = float(np.median(distances))
        if best is None or score < best[0]:
            best = (score, rotation, translation)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = best[1].astype(np.float32)
    matrix[:3, 3] = best[2].astype(np.float32)
    return matrix


def point_to_plane_icp(
    source,
    target,
    target_normals,
    initial,
    iterations=30,
    rejection_quantile=0.8,
    max_distance_mm=25.0,
    max_rotation_degrees=30.0,
    max_step_rotation_degrees=2.0,
    max_step_translation_mm=3.0,
):
    matrix = np.asarray(initial, dtype=np.float64).copy()
    tree = cKDTree(target)
    history = []
    for _ in range(int(iterations)):
        moved = apply_transform(source, matrix).astype(np.float64)
        distances, indices = tree.query(moved, k=1, workers=-1)
        threshold = min(float(np.quantile(distances, rejection_quantile)), float(max_distance_mm))
        keep = distances <= max(threshold, 1e-6)
        if keep.sum() < 12:
            break
        p = moved[keep]
        q = target[indices[keep]]
        n = target_normals[indices[keep]]
        a = np.concatenate([np.cross(p, n), n], axis=1)
        b = -np.sum(n * (p - q), axis=1)
        delta, *_ = np.linalg.lstsq(a, b, rcond=None)
        rotation_norm = np.linalg.norm(delta[:3])
        max_rotation = np.radians(float(max_step_rotation_degrees))
        if rotation_norm > max_rotation:
            delta[:3] *= max_rotation / rotation_norm
        translation_norm = np.linalg.norm(delta[3:])
        if translation_norm > float(max_step_translation_mm):
            delta[3:] *= float(max_step_translation_mm) / translation_norm
        rotation = Rotation.from_rotvec(delta[:3]).as_matrix()
        update = np.eye(4, dtype=np.float64)
        update[:3, :3] = rotation
        update[:3, 3] = delta[3:]
        candidate_matrix = update @ matrix
        if np.degrees(Rotation.from_matrix(candidate_matrix[:3, :3]).magnitude()) > float(max_rotation_degrees):
            break
        matrix = candidate_matrix
        history.append(float(np.median(distances[keep])))
        if np.linalg.norm(delta[:3]) < 1e-5 and np.linalg.norm(delta[3:]) < 1e-3:
            break
    return matrix.astype(np.float32), history


def build_label_free_alignment(
    samples,
    train_ids,
    output_dir,
    mesh_loader,
    sample_points=4096,
    icp_iterations=30,
    seed=42,
):
    """Fit the reference on train meshes and register every mesh without labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_file = output_dir / "mesh_only_transforms.npz"
    report_file = output_dir / "alignment_report.json"
    if transform_file.exists() and report_file.exists():
        report = json.loads(report_file.read_text(encoding="utf-8"))
        if report.get("algorithm_version") == 3:
            stored = np.load(transform_file)
            transforms = {key: stored[key].astype(np.float32) for key in stored.files}
            return transforms, report

    by_id = {sample.sample_id: sample for sample in samples}
    train_samples = [by_id[sample_id] for sample_id in train_ids]
    medoid_position = select_train_medoid(train_samples, mesh_loader, min(sample_points, 2048), seed)
    medoid = train_samples[medoid_position]
    target_mesh = mesh_loader(medoid.mesh_path)
    target = deterministic_surface(target_mesh, sample_points, seed)
    target_normals_all = np.asarray(target_mesh.vertex_normals, dtype=np.float32)
    target_vertices = np.asarray(target_mesh.vertices, dtype=np.float32)
    normal_tree = cKDTree(target_vertices)
    target_normals = target_normals_all[normal_tree.query(target, k=1, workers=-1)[1]]

    transforms = {}
    rows = []
    for offset, sample in enumerate(samples):
        mesh = mesh_loader(sample.mesh_path)
        source = deterministic_surface(mesh, sample_points, seed + 10_000 + offset)
        initial = pca_initial_transform(source, target)
        matrix, history = point_to_plane_icp(
            source,
            target,
            target_normals,
            initial,
            iterations=icp_iterations,
        )
        moved = apply_transform(source, matrix)
        residual = cKDTree(target).query(moved, k=1, workers=-1)[0]
        transforms[sample.sample_id] = matrix
        rows.append(
            {
                "sample_id": sample.sample_id,
                "median_surface_residual_mm": float(np.median(residual)),
                "p95_surface_residual_mm": float(np.percentile(residual, 95)),
                "icp_iterations": len(history),
            }
        )
    np.savez_compressed(transform_file, **transforms)
    report = {
        "algorithm_version": 3,
        "method": "train_medoid_pca_point_to_plane_icp",
        "scale": False,
        "reflection": False,
        "uses_expert_landmarks": False,
        "train_medoid_sample_id": medoid.sample_id,
        "train_template_sample_ids": list(train_ids),
        "sample_points": int(sample_points),
        "icp_iterations": int(icp_iterations),
        "samples": rows,
    }
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return transforms, report
