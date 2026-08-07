"""Label-free, scale-preserving mesh registration utilities."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def mesh_source_fingerprint(samples):
    digest = hashlib.sha1()
    for sample in sorted(samples, key=lambda value: value.sample_id):
        path = Path(sample.mesh_path)
        stat = path.stat()
        digest.update(
            f"{sample.sample_id}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


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


def deterministic_surface_with_normals(mesh, count, seed):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if len(vertices) <= count:
        indices = np.arange(len(vertices), dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(vertices), count, replace=False)
    selected_normals = normals[indices]
    selected_normals /= np.clip(np.linalg.norm(selected_normals, axis=1, keepdims=True), 1e-8, None)
    return vertices[indices], selected_normals


def mesh_descriptor(points):
    centered = points - np.median(points, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    extents = np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
    return np.concatenate([np.sqrt(np.maximum(eigenvalues, 0)), np.sort(extents)[::-1]])


def rank_train_medoid_candidates(samples, mesh_loader, sample_points=2048, seed=42):
    descriptors = []
    for offset, sample in enumerate(samples):
        mesh = mesh_loader(sample.mesh_path)
        points = deterministic_surface(mesh, sample_points, seed + offset)
        descriptors.append(mesh_descriptor(points))
    descriptors = np.asarray(descriptors, dtype=np.float64)
    scale = np.maximum(np.median(np.abs(descriptors - np.median(descriptors, axis=0)), axis=0), 1e-6)
    normalized = (descriptors - np.median(descriptors, axis=0)) / scale
    return np.argsort(np.linalg.norm(normalized, axis=1)).astype(np.int64)


def select_train_medoid(samples, mesh_loader, sample_points=2048, seed=42):
    return int(rank_train_medoid_candidates(samples, mesh_loader, sample_points, seed)[0])


def _pca_basis(points):
    centered = points - np.median(points, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def pca_initial_transforms(source, target, max_rotation_degrees=30.0):
    """Return deterministic rigid starts ordered by label-free surface fit."""
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
    candidates = [
        (
            float(np.median(target_tree.query(identity_moved, k=1, workers=-1)[0])),
            identity_rotation,
            identity_translation,
            "center_only",
        )
    ]
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
        candidates.append((score, rotation, translation, f"pca_{''.join(map(str, signs))}"))
    ordered = []
    seen = set()
    for score, rotation, translation, name in sorted(candidates, key=lambda row: row[0]):
        signature = np.round(np.concatenate([rotation.reshape(-1), translation]), 5).tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation.astype(np.float32)
        matrix[:3, 3] = translation.astype(np.float32)
        ordered.append({"name": name, "initial_score": score, "matrix": matrix})
    return ordered


def pca_initial_transform(source, target, max_rotation_degrees=30.0):
    return pca_initial_transforms(source, target, max_rotation_degrees)[0]["matrix"]


def robust_symmetric_surface_score(source, target, trim_quantile=0.8):
    """Score a rigid fit without labels while reducing crop/boundary influence."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_to_target = cKDTree(target).query(source, k=1, workers=-1)[0]
    target_to_source = cKDTree(source).query(target, k=1, workers=-1)[0]
    distances = np.concatenate([source_to_target, target_to_source])
    distances = distances[np.isfinite(distances)]
    if not len(distances):
        return {
            "score": float("inf"),
            "median_mm": float("inf"),
            "trimmed_mean_mm": float("inf"),
            "p90_mm": float("inf"),
        }
    quantile = float(np.clip(trim_quantile, 0.5, 1.0))
    cutoff = float(np.quantile(distances, quantile))
    trimmed = distances[distances <= cutoff]
    median = float(np.median(distances))
    trimmed_mean = float(np.mean(trimmed))
    return {
        "score": float(0.5 * median + 0.5 * trimmed_mean),
        "median_mm": median,
        "trimmed_mean_mm": trimmed_mean,
        "p90_mm": float(np.percentile(distances, 90)),
    }


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


def build_robust_atlas(
    train_samples,
    mesh_loader,
    sample_points=4096,
    atlas_size=8,
    atlas_iterations=2,
    icp_iterations=30,
    seed=42,
    correspondence_limit_mm=20.0,
):
    """Build a topology-independent atlas from central train meshes only."""
    ranking = rank_train_medoid_candidates(
        train_samples,
        mesh_loader,
        min(int(sample_points), 2048),
        seed,
    )
    selected = [train_samples[int(index)] for index in ranking[: max(1, int(atlas_size))]]
    anchor = selected[0]
    target, target_normals = deterministic_surface_with_normals(
        mesh_loader(anchor.mesh_path), int(sample_points), seed
    )
    iteration_reports = []
    for atlas_iteration in range(max(1, int(atlas_iterations))):
        point_rows = [target.astype(np.float64)]
        normal_rows = [target_normals.astype(np.float64)]
        member_reports = []
        for member_index, sample in enumerate(selected):
            mesh = mesh_loader(sample.mesh_path)
            source, source_normals = deterministic_surface_with_normals(
                mesh,
                int(sample_points),
                seed + 50_000 + atlas_iteration * 10_000 + member_index,
            )
            initial = pca_initial_transform(source, target)
            matrix, history = point_to_plane_icp(
                source,
                target,
                target_normals,
                initial,
                iterations=icp_iterations,
            )
            moved = apply_transform(source, matrix).astype(np.float64)
            moved_normals = rotate_vectors(source_normals, matrix).astype(np.float64)
            distances, nearest = cKDTree(moved).query(target, k=1, workers=-1)
            valid = distances <= float(correspondence_limit_mm)
            corresponding_points = moved[nearest]
            corresponding_normals = moved_normals[nearest]
            corresponding_points[~valid] = np.nan
            corresponding_normals[~valid] = np.nan
            point_rows.append(corresponding_points)
            normal_rows.append(corresponding_normals)
            member_reports.append(
                {
                    "sample_id": sample.sample_id,
                    "median_correspondence_mm": float(np.median(distances)),
                    "p95_correspondence_mm": float(np.percentile(distances, 95)),
                    "valid_fraction": float(np.mean(valid)),
                    "icp_iterations": len(history),
                }
            )
        target = np.nanmedian(np.stack(point_rows), axis=0).astype(np.float32)
        combined_normals = np.nanmean(np.stack(normal_rows), axis=0)
        norm = np.linalg.norm(combined_normals, axis=1, keepdims=True)
        invalid = ~np.isfinite(norm[:, 0]) | (norm[:, 0] < 1e-8)
        combined_normals[invalid] = target_normals[invalid]
        target_normals = (combined_normals / np.clip(np.linalg.norm(combined_normals, axis=1, keepdims=True), 1e-8, None)).astype(np.float32)
        iteration_reports.append({"iteration": atlas_iteration + 1, "members": member_reports})
    return target, target_normals, anchor, selected, iteration_reports


def build_label_free_alignment(
    samples,
    train_ids,
    output_dir,
    mesh_loader,
    sample_points=4096,
    icp_iterations=30,
    atlas_size=8,
    atlas_iterations=2,
    registration_candidates=3,
    registration_restarts=2,
    registration_trim_quantile=0.8,
    seed=42,
):
    """Fit the reference on train meshes and register every mesh without labels."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_file = output_dir / "mesh_only_transforms.npz"
    report_file = output_dir / "alignment_report.json"
    expected_config = {
        "algorithm_version": 5,
        "train_template_sample_ids": list(train_ids),
        "mesh_source_fingerprint": mesh_source_fingerprint(samples),
        "sample_points": int(sample_points),
        "icp_iterations": int(icp_iterations),
        "atlas_size": int(atlas_size),
        "atlas_iterations": int(atlas_iterations),
        "registration_candidates": int(registration_candidates),
        "registration_restarts": int(registration_restarts),
        "registration_trim_quantile": float(registration_trim_quantile),
    }
    if transform_file.exists() and report_file.exists():
        report = json.loads(report_file.read_text(encoding="utf-8"))
        if all(report.get(key) == value for key, value in expected_config.items()):
            stored = np.load(transform_file)
            transforms = {key: stored[key].astype(np.float32) for key in stored.files}
            return transforms, report

    by_id = {sample.sample_id: sample for sample in samples}
    train_samples = [by_id[sample_id] for sample_id in train_ids]
    target, target_normals, medoid, atlas_samples, atlas_report = build_robust_atlas(
        train_samples,
        mesh_loader,
        sample_points=sample_points,
        atlas_size=atlas_size,
        atlas_iterations=atlas_iterations,
        icp_iterations=icp_iterations,
        seed=seed,
    )
    np.savez_compressed(
        output_dir / "train_multimesh_atlas.npz",
        points=target.astype(np.float32),
        normals=target_normals.astype(np.float32),
    )

    # Keep several train-only atlas members in the same canonical frame. A scan
    # can then avoid a poor local optimum against the fused atlas without using
    # validation/test landmarks to select its transform.
    references = [
        {
            "reference_id": "fused_atlas",
            "points": target.astype(np.float32),
            "normals": target_normals.astype(np.float32),
        }
    ]
    for member_index, atlas_sample in enumerate(atlas_samples):
        mesh = mesh_loader(atlas_sample.mesh_path)
        member_points, member_normals = deterministic_surface_with_normals(
            mesh,
            sample_points,
            seed + 70_000 + member_index,
        )
        starts = pca_initial_transforms(member_points, target)
        member_matrix, _ = point_to_plane_icp(
            member_points,
            target,
            target_normals,
            starts[0]["matrix"],
            iterations=icp_iterations,
        )
        references.append(
            {
                "reference_id": atlas_sample.sample_id,
                "points": apply_transform(member_points, member_matrix),
                "normals": rotate_vectors(member_normals, member_matrix),
            }
        )

    transforms = {}
    rows = []
    for offset, sample in enumerate(samples):
        mesh = mesh_loader(sample.mesh_path)
        source = deterministic_surface(mesh, sample_points, seed + 10_000 + offset)
        ranked_references = []
        for reference in references:
            starts = pca_initial_transforms(source, reference["points"])
            moved = apply_transform(source, starts[0]["matrix"])
            initial_score = robust_symmetric_surface_score(
                moved,
                reference["points"],
                registration_trim_quantile,
            )["score"]
            ranked_references.append((initial_score, reference, starts))
        ranked_references.sort(key=lambda row: row[0])
        selected_references = ranked_references[: max(1, int(registration_candidates))]
        candidate_rows = []
        for _, reference, starts in selected_references:
            for start in starts[: max(1, int(registration_restarts))]:
                candidate_matrix, candidate_history = point_to_plane_icp(
                    source,
                    reference["points"],
                    reference["normals"],
                    start["matrix"],
                    iterations=icp_iterations,
                )
                candidate_moved = apply_transform(source, candidate_matrix)
                candidate_score = robust_symmetric_surface_score(
                    candidate_moved,
                    reference["points"],
                    registration_trim_quantile,
                )
                candidate_rows.append(
                    {
                        "reference_id": reference["reference_id"],
                        "initialization": start["name"],
                        "matrix": candidate_matrix,
                        "history": candidate_history,
                        **candidate_score,
                    }
                )
        best = min(candidate_rows, key=lambda row: row["score"])
        matrix = best["matrix"]
        history = best["history"]
        moved = apply_transform(source, matrix)
        residual = cKDTree(target).query(moved, k=1, workers=-1)[0]
        transforms[sample.sample_id] = matrix
        rows.append(
            {
                "sample_id": sample.sample_id,
                "median_surface_residual_mm": float(np.median(residual)),
                "p95_surface_residual_mm": float(np.percentile(residual, 95)),
                "robust_symmetric_score": float(best["score"]),
                "robust_symmetric_median_mm": float(best["median_mm"]),
                "selected_reference_id": best["reference_id"],
                "selected_initialization": best["initialization"],
                "icp_iterations": len(history),
                "candidates": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key not in ("matrix", "history")
                    }
                    for candidate in sorted(candidate_rows, key=lambda row: row["score"])
                ],
            }
        )
    np.savez_compressed(transform_file, **transforms)
    report = {
        "algorithm_version": 5,
        "method": "train_only_multi_reference_robust_pca_point_to_plane_icp",
        "scale": False,
        "reflection": False,
        "uses_expert_landmarks": False,
        "train_medoid_sample_id": medoid.sample_id,
        "train_template_sample_ids": list(train_ids),
        "mesh_source_fingerprint": expected_config["mesh_source_fingerprint"],
        "atlas_sample_ids": [sample.sample_id for sample in atlas_samples],
        "atlas_size": int(atlas_size),
        "atlas_iterations": int(atlas_iterations),
        "registration_candidates": int(registration_candidates),
        "registration_restarts": int(registration_restarts),
        "registration_trim_quantile": float(registration_trim_quantile),
        "atlas_build_report": atlas_report,
        "sample_points": int(sample_points),
        "icp_iterations": int(icp_iterations),
        "samples": rows,
    }
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return transforms, report
