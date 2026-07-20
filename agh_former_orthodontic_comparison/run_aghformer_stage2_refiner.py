import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from run_orthodontic_aghformer import (
    AGHFormer,
    AGHFormerDataset,
    analysis_from_eval_rows,
    build_anatomical_adjacency,
    compute_train_templates,
    ids_to_indices,
    limit_samples_balanced,
    parse_indices,
    parse_pairs,
    predict_landmarks,
    summarize_errors,
    write_analysis_csvs,
)
from shared_metrics.orthodontic_analysis import build_error_analysis


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_from_config(config, key, fallback):
    return config[key] if key in config and config[key] is not None else fallback


def build_stage1_dataset(args, stage1_config):
    dataset = AGHFormerDataset(
        root_dir=args.data_root,
        cache_dir=Path(args.output_dir) / "stage1_point_cache",
        num_points=args.surface_points,
        heatmap_sigma=float(resolve_from_config(stage1_config, "heatmap_sigma_start", args.heatmap_sigma_start)),
        use_normals=bool(resolve_from_config(stage1_config, "use_normals", True)),
        use_local_geometry=bool(resolve_from_config(stage1_config, "use_local_geometry", True)),
        local_geometry_k=int(resolve_from_config(stage1_config, "local_geometry_k", 16)),
        transformation_dir=args.transformation_dir,
        seed=args.seed,
    )
    if args.max_samples is not None:
        dataset.samples = limit_samples_balanced(dataset.samples, args.max_samples, args.seed)

    if args.splits_json and args.max_samples is None:
        split_source = load_json(args.splits_json)
        train_idx = ids_to_indices(dataset, split_source["train"])
        val_idx = ids_to_indices(dataset, split_source["val"])
        test_idx = ids_to_indices(dataset, split_source["test"])
        source_splits_json = str(Path(args.splits_json))
    else:
        from run_orthodontic_aghformer import make_splits

        train_idx, val_idx, test_idx = make_splits(dataset, args.test_size, args.val_size, args.seed)
        source_splits_json = None if not args.splits_json else f"{Path(args.splits_json)} ignored because --max-samples was used"

    templates = compute_train_templates(dataset, train_idx)
    dataset.template_mode = args.template_mode
    dataset.template_landmarks = templates
    return dataset, train_idx, val_idx, test_idx, source_splits_json, templates


def build_stage1_model(args, stage1_config, input_dim, device):
    symmetry_pairs = parse_pairs(str(resolve_from_config(stage1_config, "symmetry_pairs", args.symmetry_pairs)))
    midline_indices = parse_indices(str(resolve_from_config(stage1_config, "midline_indices", args.midline_indices)))
    graph = build_anatomical_adjacency(23, symmetry_pairs, midline_indices)
    model = AGHFormer(
        input_dim=input_dim,
        num_landmarks=23,
        width=int(resolve_from_config(stage1_config, "width", args.stage1_width)),
        blocks=int(resolve_from_config(stage1_config, "blocks", args.stage1_blocks)),
        heads=int(resolve_from_config(stage1_config, "heads", args.stage1_heads)),
        mlp_ratio=float(resolve_from_config(stage1_config, "mlp_ratio", args.stage1_mlp_ratio)),
        dropout=float(resolve_from_config(stage1_config, "dropout", args.stage1_dropout)),
        graph_adjacency=graph,
        residual_scale=float(resolve_from_config(stage1_config, "residual_scale", args.stage1_residual_scale)),
    ).to(device)
    model.load_state_dict(torch.load(args.stage1_model_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def generate_stage1_predictions(model, dataset, indices, args, device, output_csv):
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=args.stage1_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(
        f"Generating Stage1 predictions: {Path(output_csv).name} "
        f"n={len(indices)} batch_size={args.stage1_batch_size}",
        flush=True,
    )
    pred_by_idx = {}
    rows = []
    for batch in tqdm(loader, desc=f"stage1 {Path(output_csv).stem}", leave=False, disable=args.no_tqdm):
        points = batch["points_norm"].to(device)
        features = batch["features"].to(device)
        template_norm = batch["template_norm"].to(device)
        outputs = model(points, features, template_norm=template_norm)
        raw, snapped = predict_landmarks(
            outputs,
            batch["points_world"],
            batch["center"],
            batch["scale"],
            postprocess=args.stage1_postprocess,
            temperature=args.stage1_temperature,
            topk=args.stage1_topk,
            prediction_mode=args.stage1_prediction_mode,
            snap=True,
        )
        experts = batch["landmarks_world"].cpu().numpy()
        raw_errors = np.linalg.norm(raw - experts, axis=-1)
        snapped_errors = np.linalg.norm(snapped - experts, axis=-1)
        sample_indices = batch["sample_index"].cpu().numpy().astype(int)
        for row_i, sample_idx in enumerate(sample_indices):
            chosen = snapped[row_i] if args.stage1_center == "snapped" else raw[row_i]
            pred_by_idx[int(sample_idx)] = {
                "raw": raw[row_i].astype(np.float32),
                "snapped": snapped[row_i].astype(np.float32),
                "center": chosen.astype(np.float32),
                "expert": experts[row_i].astype(np.float32),
                "raw_errors": raw_errors[row_i].astype(np.float32),
                "snapped_errors": snapped_errors[row_i].astype(np.float32),
            }
            meta = dataset.metadata(int(sample_idx))
            for lm_idx in range(23):
                rows.append(
                    {
                        "sample_id": meta.sample_id,
                        "class": meta.class_name,
                        "gender": meta.gender,
                        "subject_id": meta.subject_id,
                        "landmark": lm_idx,
                        "expert_x": experts[row_i, lm_idx, 0],
                        "expert_y": experts[row_i, lm_idx, 1],
                        "expert_z": experts[row_i, lm_idx, 2],
                        "stage1_raw_x": raw[row_i, lm_idx, 0],
                        "stage1_raw_y": raw[row_i, lm_idx, 1],
                        "stage1_raw_z": raw[row_i, lm_idx, 2],
                        "stage1_snapped_x": snapped[row_i, lm_idx, 0],
                        "stage1_snapped_y": snapped[row_i, lm_idx, 1],
                        "stage1_snapped_z": snapped[row_i, lm_idx, 2],
                        "raw_localization_error": raw_errors[row_i, lm_idx],
                        "snapped_localization_error": snapped_errors[row_i, lm_idx],
                    }
                )
    write_dict_rows(output_csv, rows)
    return pred_by_idx


def stage1_landmark_weights(pred_by_idx, mode="train_error"):
    if mode == "none":
        return np.ones(23, dtype=np.float32)
    errors = np.stack([entry["snapped_errors"] for entry in pred_by_idx.values()], axis=0)
    means = errors.mean(axis=0)
    weights = means / max(float(means.mean()), 1e-6)
    return np.clip(weights, 0.75, 2.5).astype(np.float32)


class Stage2PatchDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        sample_indices,
        stage1_predictions,
        patch_points=512,
        patch_radius_mm=12.0,
        center_jitter_mm=0.0,
        point_noise_mm=0.0,
        point_dropout=0.0,
        heatmap_sigma_mm=3.0,
        seed=42,
    ):
        self.base_dataset = base_dataset
        self.sample_indices = list(sample_indices)
        self.stage1_predictions = stage1_predictions
        self.patch_points = int(patch_points)
        self.patch_radius_mm = float(patch_radius_mm)
        self.center_jitter_mm = float(center_jitter_mm)
        self.point_noise_mm = float(point_noise_mm)
        self.point_dropout = float(point_dropout)
        self.heatmap_sigma_mm = float(heatmap_sigma_mm)
        self.seed = int(seed)

    def __len__(self):
        return len(self.sample_indices) * 23

    def __getitem__(self, item_idx):
        sample_pos = item_idx // 23
        lm_idx = item_idx % 23
        sample_idx = self.sample_indices[sample_pos]
        data = self.base_dataset[sample_idx]
        points_world = data["points_world"].numpy().astype(np.float32)
        features = data["features"].numpy().astype(np.float32)
        expert = data["landmarks_world"].numpy().astype(np.float32)[lm_idx]
        center = self.stage1_predictions[sample_idx]["center"][lm_idx].astype(np.float32)
        rng = np.random.default_rng(self.seed + item_idx)
        if self.center_jitter_mm > 0:
            center = center + rng.normal(0.0, self.center_jitter_mm, size=3).astype(np.float32)

        dists = np.linalg.norm(points_world - center[None, :], axis=1)
        within = np.where(dists <= self.patch_radius_mm)[0]
        if len(within) >= self.patch_points:
            local_order = within[np.argpartition(dists[within], self.patch_points - 1)[: self.patch_points]]
        else:
            local_order = np.argpartition(dists, min(self.patch_points, len(dists)) - 1)[: self.patch_points]
        if len(local_order) < self.patch_points:
            extra = rng.choice(local_order, self.patch_points - len(local_order), replace=True)
            local_order = np.concatenate([local_order, extra], axis=0)
        points_patch = points_world[local_order].astype(np.float32)
        expert_dists = np.linalg.norm(points_patch - expert[None, :], axis=1)
        sigma = max(self.heatmap_sigma_mm, 1e-6)
        patch_heatmap = np.exp(-(expert_dists**2) / (2.0 * sigma**2)).astype(np.float32)
        local_xyz = (points_patch - center[None, :]) / max(self.patch_radius_mm, 1e-6)
        if self.point_noise_mm > 0:
            local_xyz = local_xyz + rng.normal(0.0, self.point_noise_mm / max(self.patch_radius_mm, 1e-6), size=local_xyz.shape).astype(np.float32)
        normals = features[local_order, 3:6].astype(np.float32) if features.shape[1] >= 6 else np.zeros_like(local_xyz, dtype=np.float32)
        local_dist = np.linalg.norm(local_xyz, axis=1, keepdims=True).astype(np.float32)
        patch_features = np.concatenate([local_xyz.astype(np.float32), normals, local_dist], axis=1)
        if self.point_dropout > 0:
            keep = (rng.random(self.patch_points) > self.point_dropout).astype(np.float32)[:, None]
            patch_features = patch_features * keep

        stage1_center = self.stage1_predictions[sample_idx]["center"][lm_idx].astype(np.float32)
        target_delta = (expert - stage1_center).astype(np.float32)
        return {
            "patch_features": torch.tensor(patch_features, dtype=torch.float32),
            "patch_points_world": torch.tensor(points_patch, dtype=torch.float32),
            "patch_heatmap": torch.tensor(patch_heatmap, dtype=torch.float32),
            "stage1_center": torch.tensor(stage1_center, dtype=torch.float32),
            "expert": torch.tensor(expert, dtype=torch.float32),
            "target_delta": torch.tensor(target_delta, dtype=torch.float32),
            "landmark": torch.tensor(lm_idx, dtype=torch.long),
            "sample_index": torch.tensor(sample_idx, dtype=torch.long),
        }


class LocalPatchResidualRefiner(nn.Module):
    def __init__(self, input_dim=7, width=128, landmark_dim=32, dropout=0.1, residual_limit_mm=12.0):
        super().__init__()
        self.residual_limit_mm = float(residual_limit_mm)
        self.landmark_embedding = nn.Embedding(23, landmark_dim)
        self.point_mlp = nn.Sequential(
            nn.Conv1d(input_dim, width // 2, 1, bias=False),
            nn.BatchNorm1d(width // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(width // 2, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.Conv1d(width, width, 1, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(width * 2 + landmark_dim, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv1d(width, 1, 1)
        self.delta_head = nn.Linear(width // 2, 3)
        self.log_var_head = nn.Linear(width // 2, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, patch_features, landmark):
        x = patch_features.transpose(1, 2).contiguous()
        point_features = self.point_mlp(x)
        point_logits = self.heatmap_head(point_features).squeeze(1)
        pooled_max = point_features.max(dim=2).values
        pooled_mean = point_features.mean(dim=2)
        lm_emb = self.landmark_embedding(landmark)
        hidden = self.head(torch.cat([pooled_max, pooled_mean, lm_emb], dim=1))
        delta = torch.tanh(self.delta_head(hidden)) * self.residual_limit_mm
        log_var = torch.clamp(self.log_var_head(hidden).squeeze(1), min=-6.0, max=6.0)
        return delta, log_var, point_logits


def differentiable_patch_coordinate(point_logits, patch_points_world, temperature=1.0):
    weights = torch.softmax(point_logits / max(float(temperature), 1e-6), dim=1)
    return torch.einsum("bp,bpd->bd", weights, patch_points_world)


def patch_heatmap_loss(point_logits, patch_heatmap, positive_weight=20.0, ce_weight=0.05):
    weights = 1.0 + float(positive_weight) * patch_heatmap
    mse = ((torch.sigmoid(point_logits) - patch_heatmap).pow(2.0) * weights).sum() / weights.sum().clamp_min(1.0)
    if ce_weight <= 0:
        return mse
    target_idx = patch_heatmap.argmax(dim=1)
    ce = F.cross_entropy(point_logits, target_idx)
    return mse + float(ce_weight) * ce


def clinical_loss(pred, expert, threshold_mm=2.0, margin_mm=0.5):
    err = torch.linalg.norm(pred - expert, dim=1)
    return F.softplus((err - float(threshold_mm)) / max(float(margin_mm), 1e-6)).mean()


def refiner_loss(delta, log_var, point_logits, batch, landmark_weights, args):
    heatmap_coord = differentiable_patch_coordinate(point_logits, batch["patch_points_world"], args.heatmap_temperature)
    if args.final_mode == "center_delta":
        final = batch["stage1_center"] + delta
    elif args.final_mode == "heatmap_only":
        final = heatmap_coord
    else:
        final = batch["stage1_center"] + delta + float(args.heatmap_refine_weight) * (heatmap_coord - batch["stage1_center"])
    expert = batch["expert"]
    per_item = F.smooth_l1_loss(final, expert, reduction="none").mean(dim=1)
    weights = landmark_weights[batch["landmark"]]
    coord = (per_item * weights).mean()
    heatmap = patch_heatmap_loss(
        point_logits,
        batch["patch_heatmap"],
        positive_weight=args.patch_heatmap_positive_weight,
        ce_weight=args.patch_heatmap_ce_weight,
    )
    clinical = clinical_loss(final, expert, args.clinical_threshold_mm)
    delta_reg = torch.linalg.norm(delta, dim=1).mean()
    err_detached = torch.linalg.norm(final - expert, dim=1).detach()
    uncertain = (torch.exp(-log_var) * err_detached + log_var).mean()
    loss = (
        coord
        + args.patch_heatmap_weight * heatmap
        + args.clinical_weight * clinical
        + args.delta_reg_weight * delta_reg
        + args.uncertainty_weight * uncertain
    )
    return loss, {
        "coord_loss": float(coord.detach().cpu()),
        "patch_heatmap_loss": float(heatmap.detach().cpu()),
        "clinical_loss": float(clinical.detach().cpu()),
        "delta_reg": float(delta_reg.detach().cpu()),
        "uncertainty_loss": float(uncertain.detach().cpu()),
    }


def train_refiner_epoch(model, loader, optimizer, device, landmark_weights, args):
    model.train()
    total = 0.0
    parts_total = {}
    for batch in tqdm(loader, desc="stage2 train", leave=False, disable=args.no_tqdm):
        batch = {k: v.to(device) for k, v in batch.items()}
        delta, log_var, point_logits = model(batch["patch_features"], batch["landmark"])
        loss, parts = refiner_loss(delta, log_var, point_logits, batch, landmark_weights, args)
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu()) * batch["patch_features"].shape[0]
        for key, value in parts.items():
            parts_total[key] = parts_total.get(key, 0.0) + value * batch["patch_features"].shape[0]
    n = max(1, len(loader.dataset))
    return total / n, {k: v / n for k, v in parts_total.items()}


def numpy_topk_patch_coordinate(logits, patch_points, topk=30, temperature=1.0):
    k = min(int(topk), len(logits))
    idx = np.argpartition(logits, -k)[-k:]
    scores = logits[idx] / max(float(temperature), 1e-6)
    scores = scores - scores.max()
    weights = np.exp(scores)
    weights = weights / max(float(weights.sum()), 1e-12)
    return np.sum(patch_points[idx] * weights[:, None], axis=0).astype(np.float32)


def project_to_patch_surface(raw, patch_points, mode="topk_distance", topk=5):
    if mode == "none":
        return raw.astype(np.float32)
    dists = np.linalg.norm(patch_points - raw[None, :], axis=1)
    if mode == "nearest" or topk <= 1:
        return patch_points[dists.argmin()].astype(np.float32)
    k = min(int(topk), len(dists))
    idx = np.argpartition(dists, k - 1)[:k]
    weights = 1.0 / np.clip(dists[idx], 1e-6, None)
    weights = weights / max(float(weights.sum()), 1e-12)
    return np.sum(patch_points[idx] * weights[:, None], axis=0).astype(np.float32)


@torch.no_grad()
def evaluate_refiner(model, loader, base_dataset, split_indices, device, args):
    model.eval()
    n_samples = len(split_indices)
    raw_pred = np.zeros((n_samples, 23, 3), dtype=np.float32)
    snapped_pred = np.zeros((n_samples, 23, 3), dtype=np.float32)
    experts = np.zeros((n_samples, 23, 3), dtype=np.float32)
    log_vars = np.zeros((n_samples, 23), dtype=np.float32)
    index_to_pos = {sample_idx: pos for pos, sample_idx in enumerate(split_indices)}
    stage1_centers = np.zeros((n_samples, 23, 3), dtype=np.float32)
    for batch in tqdm(loader, desc="stage2 eval", leave=False, disable=args.no_tqdm):
        device_batch = {k: v.to(device) for k, v in batch.items()}
        delta, log_var, point_logits = model(device_batch["patch_features"], device_batch["landmark"])
        heatmap_coord = differentiable_patch_coordinate(point_logits, device_batch["patch_points_world"], args.heatmap_temperature)
        if args.final_mode == "center_delta":
            final_t = device_batch["stage1_center"] + delta
        elif args.final_mode == "heatmap_only":
            final_t = heatmap_coord
        else:
            final_t = device_batch["stage1_center"] + delta + float(args.heatmap_refine_weight) * (
                heatmap_coord - device_batch["stage1_center"]
            )
        final = final_t.cpu().numpy().astype(np.float32)
        logits_np = point_logits.cpu().numpy().astype(np.float32)
        patch_points = batch["patch_points_world"].numpy().astype(np.float32)
        sample_indices = batch["sample_index"].numpy().astype(int)
        landmarks = batch["landmark"].numpy().astype(int)
        expert_np = batch["expert"].numpy().astype(np.float32)
        centers_np = batch["stage1_center"].numpy().astype(np.float32)
        for row_i, sample_idx in enumerate(sample_indices):
            pos = index_to_pos[int(sample_idx)]
            lm_idx = int(landmarks[row_i])
            raw_pred[pos, lm_idx] = final[row_i]
            experts[pos, lm_idx] = expert_np[row_i]
            stage1_centers[pos, lm_idx] = centers_np[row_i]
            log_vars[pos, lm_idx] = float(log_var[row_i].cpu())
            if args.eval_coordinate_mode == "topk_heatmap":
                projected_source = numpy_topk_patch_coordinate(
                    logits_np[row_i],
                    patch_points[row_i],
                    topk=args.eval_topk,
                    temperature=args.heatmap_temperature,
                )
            else:
                projected_source = final[row_i]
            snapped_pred[pos, lm_idx] = project_to_patch_surface(
                projected_source,
                patch_points[row_i],
                mode=args.projection_mode,
                topk=args.projection_topk,
            )
    raw_errors = np.linalg.norm(raw_pred - experts, axis=-1)
    snapped_errors = np.linalg.norm(snapped_pred - experts, axis=-1)
    stage1_errors = np.linalg.norm(stage1_centers - experts, axis=-1)
    rows = []
    for pos, sample_idx in enumerate(split_indices):
        rows.append(
            (
                int(sample_idx),
                raw_pred[pos],
                snapped_pred[pos],
                experts[pos],
                raw_errors[pos],
                snapped_errors[pos],
                log_vars[pos],
            )
        )
    return rows, raw_errors, snapped_errors, stage1_errors


def write_dict_rows(path, rows):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_stage2_predictions(path, dataset, eval_rows):
    rows = []
    for sample_idx, raw_pred, snapped_pred, expert, raw_errors, snapped_errors, log_vars in eval_rows:
        meta = dataset.metadata(sample_idx)
        for lm_idx in range(23):
            rows.append(
                {
                    "sample_id": meta.sample_id,
                    "class": meta.class_name,
                    "gender": meta.gender,
                    "subject_id": meta.subject_id,
                    "landmark": lm_idx,
                    "expert_x": expert[lm_idx, 0],
                    "expert_y": expert[lm_idx, 1],
                    "expert_z": expert[lm_idx, 2],
                    "stage2_raw_x": raw_pred[lm_idx, 0],
                    "stage2_raw_y": raw_pred[lm_idx, 1],
                    "stage2_raw_z": raw_pred[lm_idx, 2],
                    "stage2_snapped_x": snapped_pred[lm_idx, 0],
                    "stage2_snapped_y": snapped_pred[lm_idx, 1],
                    "stage2_snapped_z": snapped_pred[lm_idx, 2],
                    "raw_localization_error": raw_errors[lm_idx],
                    "snapped_localization_error": snapped_errors[lm_idx],
                    "uncertainty": float(np.exp(log_vars[lm_idx])),
                }
            )
    write_dict_rows(path, rows)


def write_group_metrics(path, dataset, eval_rows):
    groups = {}
    for sample_idx, _, _, _, _, snapped_errors, _ in eval_rows:
        meta = dataset.metadata(sample_idx)
        groups.setdefault((meta.class_name, meta.gender), []).extend(snapped_errors.tolist())
    rows = []
    for (class_name, gender), errors in sorted(groups.items()):
        arr = np.asarray(errors, dtype=np.float64)
        rows.append(
            {
                "class": class_name,
                "gender": gender,
                "n_samples": int(len(arr) / 23),
                "ale": float(arr.mean()),
                "std": float(arr.std()),
                "median": float(np.median(arr)),
            }
        )
    write_dict_rows(path, rows)


def save_refiner_outputs(output_dir, args, dataset, split_indices, eval_rows, raw_errors, snapped_errors, stage1_errors, suffix="test"):
    samples = [dataset.metadata(i) for i in split_indices]
    analysis = build_error_analysis(samples, snapped_errors)
    write_stage2_predictions(output_dir / f"refined_predictions_{suffix}.csv", dataset, eval_rows)
    write_group_metrics(output_dir / f"group_metrics_{suffix}.csv", dataset, eval_rows)
    write_analysis_csvs(output_dir, analysis, suffix=suffix)
    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "model": "AGH-Former Stage 2 local patch residual refiner",
        "stage1_center": args.stage1_center,
        "patch_points": args.patch_points,
        "patch_radius_mm": args.patch_radius_mm,
        "patch_heatmap_sigma_mm": args.patch_heatmap_sigma_mm,
        "final_mode": args.final_mode,
        "heatmap_refine_weight": args.heatmap_refine_weight,
        "eval_coordinate_mode": args.eval_coordinate_mode,
        "eval_topk": args.eval_topk,
        "projection_mode": args.projection_mode,
        "projection_topk": args.projection_topk,
        "stage2_raw": summarize_errors(raw_errors),
        "stage2_snapped": summarize_errors(snapped_errors),
        "stage1_center_baseline": summarize_errors(stage1_errors),
    }
    metrics.update(analysis)
    (output_dir / f"metrics_refined_{suffix}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if suffix == "test":
        (output_dir / "metrics_refined.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train AGH-Former Stage 2 local residual refiner.")
    parser.add_argument("--data-root", default="../data/dataset")
    parser.add_argument("--splits-json", default=None)
    parser.add_argument("--transformation-dir", default=None)
    parser.add_argument("--stage1-run-dir", required=True)
    parser.add_argument("--stage1-model-path", default=None)
    parser.add_argument("--stage1-batch-size", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--surface-points", type=int, default=None)
    parser.add_argument("--heatmap-sigma-start", type=float, default=5.0)
    parser.add_argument("--template-mode", choices=["global", "class", "gender", "class_gender"], default="class_gender")
    parser.add_argument("--stage1-center", choices=["raw", "snapped"], default="snapped")
    parser.add_argument("--stage1-postprocess", choices=["softmax", "topk_softmax", "argmax"], default="topk_softmax")
    parser.add_argument("--stage1-prediction-mode", choices=["direct", "heatmap_residual"], default="direct")
    parser.add_argument("--stage1-temperature", type=float, default=1.0)
    parser.add_argument("--stage1-topk", type=int, default=30)
    parser.add_argument("--stage1-width", type=int, default=192)
    parser.add_argument("--stage1-blocks", type=int, default=4)
    parser.add_argument("--stage1-heads", type=int, default=6)
    parser.add_argument("--stage1-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--stage1-dropout", type=float, default=0.1)
    parser.add_argument("--stage1-residual-scale", type=float, default=0.18)
    parser.add_argument("--symmetry-pairs", default="1-2,3-4,7-8,10-11,12-13,14-15,16-17,19-20,21-22")
    parser.add_argument("--midline-indices", default="0,5,6,9,18")
    parser.add_argument("--patch-points", type=int, default=512)
    parser.add_argument("--patch-radius-mm", type=float, default=12.0)
    parser.add_argument("--patch-heatmap-sigma-mm", type=float, default=3.0)
    parser.add_argument("--center-jitter-mm", type=float, default=1.5)
    parser.add_argument("--point-noise-mm", type=float, default=0.1)
    parser.add_argument("--point-dropout", type=float, default=0.05)
    parser.add_argument("--refiner-width", type=int, default=128)
    parser.add_argument("--landmark-embedding-dim", type=int, default=32)
    parser.add_argument("--refiner-dropout", type=float, default=0.1)
    parser.add_argument("--residual-limit-mm", type=float, default=12.0)
    parser.add_argument("--final-mode", choices=["heatmap_delta", "center_delta", "heatmap_only"], default="center_delta")
    parser.add_argument("--heatmap-refine-weight", type=float, default=0.25)
    parser.add_argument("--heatmap-temperature", type=float, default=1.0)
    parser.add_argument("--patch-heatmap-weight", type=float, default=0.25)
    parser.add_argument("--patch-heatmap-positive-weight", type=float, default=20.0)
    parser.add_argument("--patch-heatmap-ce-weight", type=float, default=0.05)
    parser.add_argument("--eval-coordinate-mode", choices=["raw_final", "topk_heatmap"], default="raw_final")
    parser.add_argument("--eval-topk", type=int, default=30)
    parser.add_argument("--projection-mode", choices=["nearest", "topk_distance", "none"], default="topk_distance")
    parser.add_argument("--projection-topk", type=int, default=5)
    parser.add_argument("--selection-metric", choices=["raw", "snapped"], default="snapped")
    parser.add_argument("--landmark-weighting", choices=["none", "train_error"], default="train_error")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clinical-weight", type=float, default=0.05)
    parser.add_argument("--clinical-threshold-mm", type=float, default=2.0)
    parser.add_argument("--delta-reg-weight", type=float, default=0.002)
    parser.add_argument("--uncertainty-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-tqdm", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_run_dir = Path(args.stage1_run_dir)
    stage1_config_path = stage1_run_dir / "config.json"
    stage1_config = load_json(stage1_config_path) if stage1_config_path.exists() else {}
    args.stage1_model_path = args.stage1_model_path or str(stage1_run_dir / "best_model.pth")
    if args.surface_points is None:
        args.surface_points = int(resolve_from_config(stage1_config, "surface_points", 12000))
    config_payload = vars(args).copy()
    (output_dir / "config_stage2.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    dataset, train_idx, val_idx, test_idx, source_splits_json, templates = build_stage1_dataset(args, stage1_config)
    split_payload = {
        "train": [dataset.samples[i].sample_id for i in train_idx],
        "val": [dataset.samples[i].sample_id for i in val_idx],
        "test": [dataset.samples[i].sample_id for i in test_idx],
        "source_splits_json": source_splits_json,
        "stage1_run_dir": str(stage1_run_dir),
        "stage1_model_path": str(args.stage1_model_path),
    }
    (output_dir / "splits.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
    (output_dir / "template_landmarks.json").write_text(
        json.dumps({key: value.astype(float).tolist() for key, value in sorted(templates.items())}, indent=2),
        encoding="utf-8",
    )
    input_dim = 3 + (3 if bool(resolve_from_config(stage1_config, "use_normals", True)) else 0) + (
        2 if bool(resolve_from_config(stage1_config, "use_local_geometry", True)) else 0
    )
    stage1_model = build_stage1_model(args, stage1_config, input_dim, device)
    print(f"Device: {device}", flush=True)
    print(f"Samples train/val/test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}", flush=True)
    print(f"Stage1 model: {args.stage1_model_path}", flush=True)

    stage1_train = generate_stage1_predictions(stage1_model, dataset, train_idx, args, device, output_dir / "stage1_predictions_train.csv")
    stage1_val = generate_stage1_predictions(stage1_model, dataset, val_idx, args, device, output_dir / "stage1_predictions_val.csv")
    stage1_test = generate_stage1_predictions(stage1_model, dataset, test_idx, args, device, output_dir / "stage1_predictions_test.csv")
    stage1_all = {}
    stage1_all.update(stage1_train)
    stage1_all.update(stage1_val)
    stage1_all.update(stage1_test)

    weights_np = stage1_landmark_weights(stage1_train, args.landmark_weighting)
    (output_dir / "landmark_weights.json").write_text(json.dumps(weights_np.astype(float).tolist(), indent=2), encoding="utf-8")
    landmark_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)

    train_ds = Stage2PatchDataset(
        dataset,
        train_idx,
        stage1_all,
        patch_points=args.patch_points,
        patch_radius_mm=args.patch_radius_mm,
        heatmap_sigma_mm=args.patch_heatmap_sigma_mm,
        center_jitter_mm=args.center_jitter_mm,
        point_noise_mm=args.point_noise_mm,
        point_dropout=args.point_dropout,
        seed=args.seed,
    )
    val_ds = Stage2PatchDataset(
        dataset,
        val_idx,
        stage1_all,
        args.patch_points,
        args.patch_radius_mm,
        0.0,
        0.0,
        0.0,
        args.patch_heatmap_sigma_mm,
        args.seed,
    )
    test_ds = Stage2PatchDataset(
        dataset,
        test_idx,
        stage1_all,
        args.patch_points,
        args.patch_radius_mm,
        0.0,
        0.0,
        0.0,
        args.patch_heatmap_sigma_mm,
        args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    refiner = LocalPatchResidualRefiner(
        input_dim=7,
        width=args.refiner_width,
        landmark_dim=args.landmark_embedding_dim,
        dropout=args.refiner_dropout,
        residual_limit_mm=args.residual_limit_mm,
    ).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)

    history = []
    best_val_ale = math.inf
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_parts = train_refiner_epoch(refiner, train_loader, optimizer, device, landmark_weights, args)
        val_rows, val_raw_errors, val_snapped_errors, val_stage1_errors = evaluate_refiner(refiner, val_loader, dataset, val_idx, device, args)
        val_raw_ale = float(val_raw_errors.mean())
        val_snapped_ale = float(val_snapped_errors.mean())
        val_ale = val_raw_ale if args.selection_metric == "raw" else val_snapped_ale
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_parts": train_parts,
                "val_raw_ale": val_raw_ale,
                "val_snapped_ale": val_snapped_ale,
                "selection_metric": args.selection_metric,
                "val_selected_ale": val_ale,
                "val_stage1_center_ale": float(val_stage1_errors.mean()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.5f} "
            f"val_stage1={float(val_stage1_errors.mean()):.4f} val_raw={val_raw_ale:.4f} "
            f"val_snapped={val_snapped_ale:.4f} selected_{args.selection_metric}={val_ale:.4f}",
            flush=True,
        )
        if val_ale < best_val_ale:
            best_val_ale = val_ale
            epochs_no_improve = 0
            torch.save(refiner.state_dict(), output_dir / "best_refiner.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    (output_dir / "history_stage2.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    refiner.load_state_dict(torch.load(output_dir / "best_refiner.pth", map_location=device))
    val_rows, val_raw_errors, val_snapped_errors, val_stage1_errors = evaluate_refiner(refiner, val_loader, dataset, val_idx, device, args)
    save_refiner_outputs(output_dir, args, dataset, val_idx, val_rows, val_raw_errors, val_snapped_errors, val_stage1_errors, suffix="val")
    test_rows, test_raw_errors, test_snapped_errors, test_stage1_errors = evaluate_refiner(refiner, test_loader, dataset, test_idx, device, args)
    metrics = save_refiner_outputs(output_dir, args, dataset, test_idx, test_rows, test_raw_errors, test_snapped_errors, test_stage1_errors, suffix="test")

    print("\nEvaluation against expert orthodontist landmarks", flush=True)
    print(f"Stage1 center ALE: {metrics['stage1_center_baseline']['ale']:.4f}", flush=True)
    print(f"AGH-Former Stage2 raw ALE: {metrics['stage2_raw']['ale']:.4f}", flush=True)
    print(f"AGH-Former Stage2 snapped ALE: {metrics['stage2_snapped']['ale']:.4f}", flush=True)
    print(f"AGH-Former Stage2 snapped median: {metrics['stage2_snapped']['median']:.4f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
