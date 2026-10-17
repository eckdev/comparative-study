import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from all23_rgb_geodesic_cascade.alignment import apply_transform, mesh_source_fingerprint
from all23_rgb_geodesic_cascade.anatomy import NUM_LANDMARKS, mirror_permutation
from all23_rgb_geodesic_cascade.data import RGBGeodesicDataset, assert_disjoint_splits, read_landmarks
from all23_rgb_geodesic_cascade.run_all23_rgb_geodesic import label_free_registration_roi_scales
from all23_rgb_geodesic_cascade.stage1 import _read_predictions
from all23_rgb_geodesic_cascade.train import (
    autocast_context, inverse_predictions, move_batch, transform_batch,
)
from .gate import Evidence


def _expected_record_path(sample, matrix, cache_dir, include_landmarks):
    matrix_hash = hashlib.sha1(np.asarray(matrix, dtype=np.float32).tobytes()).hexdigest()[:10]
    paths = [sample.mesh_path]
    if include_landmarks:
        paths.append(sample.landmark_path)
    digest = hashlib.sha1()
    for path in paths:
        path = Path(path)
        stat = path.stat()
        digest.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8"))
    tag = "labels" if include_landmarks else "geometry"
    return Path(cache_dir) / f"{sample.sample_id}.{matrix_hash}.{digest.hexdigest()[:10]}.{tag}.npz"


def load_outer_preprocessing(samples, splits, preprocessing_dir, roi_expansion=0.75):
    """Load the original outer-fold artifacts without modifying that directory."""
    assert_disjoint_splits(splits)
    preprocessing_dir = Path(preprocessing_dir)
    alignment_dir = preprocessing_dir / "alignment"
    report = json.loads((alignment_dir / "alignment_report.json").read_text(encoding="utf-8"))
    if report.get("uses_expert_landmarks", False):
        raise ValueError("Outer preprocessing must use label-free alignment")
    if report.get("train_template_sample_ids") != splits["train"]:
        raise ValueError("Outer atlas was not fitted on this outer-train split")
    selected = {sample.sample_id: sample for sample in samples if sample.sample_id in set(sum(splits.values(), []))}
    fingerprint = mesh_source_fingerprint(list(selected.values()))
    if report.get("mesh_source_fingerprint") != fingerprint:
        raise ValueError("Mesh source fingerprint changed since outer-fold preprocessing")
    with np.load(alignment_dir / "mesh_only_transforms.npz") as stored:
        transforms = {key: stored[key].copy() for key in stored.files}
    if not set(selected).issubset(transforms):
        raise ValueError("Outer alignment lacks sample transforms")
    normalizer = json.loads((preprocessing_dir / "normalization.json").read_text(encoding="utf-8"))
    if normalizer.get("fit_sample_ids") != splits["train"]:
        raise ValueError("Outer normalization was not fitted on this outer-train split")
    stage1_dir = preprocessing_dir / "stage1_global_coarse"
    complete = json.loads((stage1_dir / "complete.json").read_text(encoding="utf-8"))
    if complete.get("config", {}).get("splits") not in (None, splits):
        raise ValueError("Stage 1 cache split does not match outer fold")
    coarse = {}
    for filename in ("oof_predictions_train.csv", "predictions_val.csv", "predictions_test.csv"):
        coarse.update(_read_predictions(stage1_dir / filename))
    if set(coarse) != set(selected):
        raise ValueError("Stage 1 coordinates are incomplete for the outer fold")
    records = {}
    cache_dir = preprocessing_dir / "mesh_cache"
    for sample_id, sample in selected.items():
        record = _expected_record_path(
            sample, transforms[sample_id], cache_dir,
            include_landmarks=sample_id not in splits["test"],
        )
        if not record.exists():
            raise FileNotFoundError(
                f"Original mesh cache missing or dataset changed: {record}"
            )
        records[sample_id] = record
    scales, _ = label_free_registration_roi_scales(
        report, splits["train"], max_expansion=roi_expansion
    )
    return transforms, records, normalizer, coarse, scales


def inference_dataset(
    samples, sample_ids, records, transforms, coarse, normalizer, scales,
    args, cache_dir,
):
    return RGBGeodesicDataset(
        samples=samples,
        sample_ids=sample_ids,
        records=records,
        transforms=transforms,
        coarse_predictions=coarse,
        legacy_transform_dir=None,
        normalizer=normalizer,
        cache_dir=cache_dir,
        roi_points=args.roi_points,
        roi_radius_scale=args.roi_radius_scale,
        roi_mode=args.roi_mode,
        roi_euclidean_scale=args.roi_euclidean_scale,
        roi_multi_seeds=args.roi_multi_seeds,
        training=False,
        use_rgb=True,
        coarse_in_target_space=True,
        sample_radius_scales=scales,
        memory_cache=not args.no_memory_cache,
        include_expert=False,
        seed=args.seed,
    )


@torch.no_grad()
def collect_evidence(model, loader, device, args, normalizer, *, use_tta):
    model.eval()
    result = {name: [] for name in ("center", "local", "diagnostics", "token_alpha")}
    sample_ids = []
    variants = [(0.0, False)]
    if use_tta:
        variants.extend([(-5.0, False), (5.0, False), (0.0, True)])
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        heatmaps, variances, alphas, centers, globals_ = [], [], [], [], []
        for angle, mirror in variants:
            variant, rotation, origin = transform_batch(batch, normalizer, angle, mirror)
            with autocast_context(device, args.mixed_precision, args.amp_dtype):
                output = model(variant, coordinate_mode=args.coordinate_mode)
            logits = output["local_logits"].float()
            variance = output["log_var"].float()
            alpha = output["refinement_alpha"].float()
            center = inverse_predictions(output["gate_coarse"].float(), rotation, origin, mirror)
            global_coordinate = inverse_predictions(
                output["global_heatmap_coordinates"].float(), rotation, origin, mirror
            )
            if mirror:
                permutation = torch.tensor(mirror_permutation(), device=device)
                logits = logits[:, permutation]
                variance = variance[:, permutation]
                alpha = alpha[:, permutation]
            heatmaps.append(logits)
            variances.append(variance)
            alphas.append(alpha)
            centers.append(center)
            globals_.append(global_coordinate)
        heatmap = torch.stack(heatmaps).mean(dim=0)
        center = torch.stack(centers).mean(dim=0)
        global_coordinate = torch.stack(globals_).mean(dim=0)
        variance = torch.logsumexp(torch.stack(variances), dim=0) - math.log(len(variances))
        candidates = batch["points"][batch["roi_index"]]
        local = model.coordinates_from_logits(
            heatmap, candidates, batch["roi_mask"], args.coordinate_mode
        )
        mask = batch["roi_mask"]
        probability = torch.softmax(heatmap.masked_fill(~mask, -torch.inf), dim=-1)
        probability = torch.where(mask, probability, torch.zeros_like(probability))
        entropy = -(probability * torch.log(probability.clamp_min(1e-8))).sum(dim=-1)
        entropy = entropy / torch.log(mask.sum(dim=-1).float().clamp_min(2.0))
        top = torch.topk(probability, k=min(2, probability.shape[-1]), dim=-1).values
        margin = top[..., 0] - top[..., 1] if top.shape[-1] > 1 else top[..., 0]
        radius = model.roi_radii[None, :] * batch["sample_radius_scale"][:, None]
        radius = radius.clamp_min(1e-6)
        diagnostics = torch.stack(
            [
                variance,
                torch.linalg.norm(local - center, dim=-1) / radius,
                entropy,
                margin,
                torch.linalg.norm(global_coordinate - center, dim=-1) / radius,
            ], dim=-1,
        )
        sample_ids.extend(raw_batch["sample_id"])
        for name, value in (
            ("center", center), ("local", local),
            ("diagnostics", diagnostics),
            ("token_alpha", torch.stack(alphas).mean(dim=0)),
        ):
            result[name].append(value.cpu().numpy().astype(np.float32))
        if args.no_tqdm and len(sample_ids) % 10 == 0:
            print(f"Gate evidence {len(sample_ids)}/{len(loader.dataset)}", flush=True)
    return Evidence(
        sample_ids=sample_ids,
        **{name: np.concatenate(chunks, axis=0) for name, chunks in result.items()},
    ).validate()


def attach_aligned_expert(evidence, samples, transforms):
    by_id = {sample.sample_id: sample for sample in samples}
    expert = {
        sample_id: apply_transform(read_landmarks(by_id[sample_id].landmark_path), transforms[sample_id])
        for sample_id in evidence.sample_ids
    }
    return evidence.attach_expert(expert)
