"""Leakage-safe OOF training for curve-first Hard3 refinement."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agh_former_vnext_orthodontic_comparison.hard3_structured import _splitter
from all23_rgb_geodesic_cascade.anatomy import HARD3
from all23_rgb_geodesic_cascade.metrics import summarize
from hard3_anatomical_context_refinement.patches import extract_dual_view_set

from .annotations import CurveAnnotationStore
from .model import CurveFirstHard3Net, probability_coordinate
from .targets import build_curve_targets


@dataclass(frozen=True)
class CurveHard3Config:
    folds: int = 5
    epochs: int = 100
    min_epochs: int = 30
    patience: int = 15
    batch_size: int = 8
    image_size: int = 64
    width: int = 48
    blocks: int = 2
    dropout: float = 0.10
    radius_scale: float = 1.25
    neighbor_count: int = 12
    lr: float = 5e-4
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    curve_sigma_mm: float = 2.0
    point_sigma_lm0_mm: float = 2.5
    point_sigma_gonion_mm: float = 3.0
    pseudo_radius_mm: float = 12.0
    pseudo_half_length_mm: float = 20.0
    pseudo_curve_weight: float = 0.20
    curve_bce_weight: float = 0.75
    curve_dice_weight: float = 0.25
    listwise_weight: float = 1.0
    coordinate_weight: float = 0.25
    clinical_weight: float = 0.25
    bilateral_weight: float = 0.10
    confidence_weight: float = 0.05
    temperature: float = 0.75
    annotation_manifest: str | None = None
    minimum_annotated_samples: int = 0
    allow_pseudo_curves: bool = True
    final_model_policy: str = "inner_fold_ensemble"
    maximum_step_lm0: float = 12.0
    maximum_step_gonion: float = 15.0
    bootstrap_iters: int = 2000
    minimum_overall_gain_mm: float = 0.03
    minimum_hard3_gain_mm: float = 0.20
    minimum_improvement_probability: float = 0.90
    maximum_p95_regression_mm: float = 0.10
    target_hard3_ale: float = 4.0
    seed: int = 42


def _annotation_store(config):
    if config.annotation_manifest:
        return CurveAnnotationStore.load(config.annotation_manifest)
    return CurveAnnotationStore.empty()


def _batch(candidate_set, targets, indices, device):
    selected = np.asarray(indices, dtype=np.int64)
    batch = {
        "images": torch.from_numpy(
            candidate_set.images[selected].astype(np.float32)
        ).to(device),
        "canonical": torch.from_numpy(candidate_set.canonical[selected]).to(device),
        "neighbor_index": torch.from_numpy(candidate_set.neighbor_index[selected]).to(
            device
        ),
        "neighbor_mask": torch.from_numpy(candidate_set.neighbor_mask[selected]).to(
            device
        ),
        "mask": torch.from_numpy(candidate_set.mask[selected]).to(device),
        "points": torch.from_numpy(candidate_set.points[selected]).to(device),
        "shape_context": torch.from_numpy(candidate_set.shape_context[selected]).to(
            device
        ),
    }
    if targets is not None:
        batch.update(
            {
                "curve_distance": torch.from_numpy(targets.curve_distance[selected]).to(
                    device
                ),
                "point_distance": torch.from_numpy(targets.point_distance[selected]).to(
                    device
                ),
                "curve_source": torch.from_numpy(targets.source[selected]).to(device),
                "expert": torch.from_numpy(candidate_set.expert[selected]).to(device),
            }
        )
    return batch


def _forward(model, batch):
    return model(
        batch["images"],
        batch["canonical"],
        batch["neighbor_index"],
        batch["neighbor_mask"],
        batch["mask"],
        batch["shape_context"],
    )


def _masked_mean(values, mask):
    valid = mask.to(values.dtype)
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def _curve_loss(output, batch, config):
    mask = batch["mask"]
    curve_target = torch.exp(
        -batch["curve_distance"].float().square()
        / (2.0 * max(float(config.curve_sigma_mm), 1e-4) ** 2)
    ).masked_fill(~mask, 0.0)
    curve_logits = output["curve_logits"].float().masked_fill(~mask, 0.0)
    source_weight = torch.where(
        batch["curve_source"].bool(),
        torch.ones_like(batch["curve_source"], dtype=torch.float32),
        torch.full_like(
            batch["curve_source"],
            float(config.pseudo_curve_weight),
            dtype=torch.float32,
        ),
    )
    point_weight = 1.0 + 4.0 * curve_target
    bce = F.binary_cross_entropy_with_logits(
        curve_logits, curve_target, reduction="none"
    )
    bce = (bce * point_weight * mask * source_weight[..., None]).sum() / (
        mask * source_weight[..., None]
    ).sum().clamp_min(1.0)
    predicted = torch.sigmoid(curve_logits) * mask
    intersection = (predicted * curve_target).sum(dim=-1)
    denominator = predicted.sum(dim=-1) + curve_target.sum(dim=-1)
    dice_rows = 1.0 - (2.0 * intersection + 1e-4) / (denominator + 1e-4)
    dice = (dice_rows * source_weight).sum() / source_weight.sum().clamp_min(1.0)
    return bce, dice, curve_target


def _loss(output, batch, config):
    mask = batch["mask"]
    bce, dice, _ = _curve_loss(output, batch, config)
    sigma = batch["point_distance"].new_tensor(
        [
            config.point_sigma_lm0_mm,
            config.point_sigma_gonion_mm,
            config.point_sigma_gonion_mm,
        ]
    )[None, :, None]
    target_energy = -batch["point_distance"].float().square() / (
        2.0 * sigma.square().clamp_min(1e-6)
    )
    target_energy = target_energy.masked_fill(~mask, -torch.inf)
    target_probability = torch.softmax(target_energy, dim=-1)
    log_probability = torch.log_softmax(
        output["final_logits"].float().masked_fill(~mask, -torch.inf)
        / max(float(config.temperature), 1e-4),
        dim=-1,
    ).masked_fill(~mask, 0.0)
    listwise = (
        target_probability
        * (torch.log(target_probability.clamp_min(1e-8)) - log_probability)
    ).masked_fill(~mask, 0.0)
    listwise = listwise.sum(dim=-1).mean()
    coordinate, probability = probability_coordinate(
        output["final_logits"], batch["points"], mask, config.temperature
    )
    coordinate_loss = F.smooth_l1_loss(coordinate, batch["expert"].float(), beta=1.0)
    finite_distance = batch["point_distance"].float().masked_fill(~mask, 0.0)
    clinical = (probability * finite_distance).sum(dim=-1).mean() / 10.0
    predicted_midpoint = coordinate[:, 1:3].mean(dim=1)
    expert_midpoint = batch["expert"][:, 1:3].float().mean(dim=1)
    predicted_width = torch.linalg.norm(coordinate[:, 1] - coordinate[:, 2], dim=-1)
    expert_width = torch.linalg.norm(
        batch["expert"][:, 1].float() - batch["expert"][:, 2].float(), dim=-1
    )
    bilateral = F.smooth_l1_loss(
        predicted_midpoint, expert_midpoint, beta=1.0
    ) + F.smooth_l1_loss(predicted_width, expert_width, beta=1.0)
    squared_error = (
        torch.linalg.norm(coordinate - batch["expert"].float(), dim=-1).square() / 16.0
    )
    log_variance = output["log_variance"].float()
    confidence = 0.5 * (torch.exp(-log_variance) * squared_error + log_variance).mean()
    total = (
        config.curve_bce_weight * bce
        + config.curve_dice_weight * dice
        + config.listwise_weight * listwise
        + config.coordinate_weight * coordinate_loss
        + config.clinical_weight * clinical
        + config.bilateral_weight * bilateral
        + config.confidence_weight * confidence
    )
    return total, {
        "curve_bce": bce,
        "curve_dice": dice,
        "listwise": listwise,
        "coordinate": coordinate_loss,
        "clinical": clinical,
        "bilateral": bilateral,
        "confidence": confidence,
    }


@torch.no_grad()
def _predict_model(model, candidate_set, targets, indices, config, device):
    model.eval()
    chunks = {"final_logits": [], "landmark_logits": [], "log_variance": []}
    indices = list(indices)
    for start in range(0, len(indices), config.batch_size):
        selected = indices[start : start + config.batch_size]
        batch = _batch(candidate_set, targets, selected, device)
        output = _forward(model, batch)
        for key in chunks:
            chunks[key].append(output[key].float().cpu().numpy())
    result = {key: np.concatenate(rows) for key, rows in chunks.items()}
    result["coordinate"] = _decode_numpy(
        result["final_logits"],
        candidate_set.points[np.asarray(indices)],
        candidate_set.mask[np.asarray(indices)],
        config.temperature,
        False,
    )
    return result


def _decode_numpy(logits, points, mask, temperature=1.0, argmax=False):
    safe = np.where(mask, logits, -np.inf).astype(np.float64)
    if argmax:
        index = np.argmax(safe, axis=-1)
        return np.take_along_axis(points, index[..., None, None], axis=2)[
            ..., 0, :
        ].astype(np.float32)
    maximum = np.max(safe, axis=-1, keepdims=True)
    probability = np.exp((safe - maximum) / max(float(temperature), 1e-4)) * mask
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
    return np.sum(probability[..., None] * points, axis=-2).astype(np.float32)


def _new_model(candidate_set, config):
    return CurveFirstHard3Net(
        candidate_set.images.shape[3],
        candidate_set.canonical.shape[-1],
        candidate_set.shape_context.shape[-1],
        config.width,
        config.blocks,
        config.dropout,
    )


def _fit_one(candidate_set, targets, train_indices, val_indices, config, device, fold):
    torch.manual_seed(config.seed + fold * 1009)
    model = _new_model(candidate_set, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    rng = np.random.default_rng(config.seed + fold * 7919)
    best_score, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals, seen = {}, 0
        for start in range(0, len(train_indices), config.batch_size):
            if start == 0:
                order = rng.permutation(train_indices)
            selected = order[start : start + config.batch_size]
            batch = _batch(candidate_set, targets, selected, device)
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            loss, components = _loss(output, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite curve-supervised Hard3 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            count = len(selected)
            totals["total"] = totals.get("total", 0.0) + float(loss.detach()) * count
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
            seen += count
        prediction = _predict_model(
            model, candidate_set, targets, val_indices, config, device
        )["coordinate"]
        error = np.linalg.norm(prediction - candidate_set.expert[val_indices], axis=-1)
        score = float(error.mean())
        scheduler.step(score)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
            "val_hard3_ale": score,
            "val_lm0_ale": float(error[:, 0].mean()),
            "val_lm21_ale": float(error[:, 1].mean()),
            "val_lm22_ale": float(error[:, 2].mean()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        if score < best_score - 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        elif epoch >= config.min_epochs:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Curve-H3 OOF fold {fold} epoch {epoch:03d}/{config.epochs} "
                f"train={row['train_total']:.4f} val={score:.4f}",
                flush=True,
            )
        if epoch >= config.min_epochs and stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Curve-supervised Hard3 did not produce a checkpoint")
    model.load_state_dict(best_state)
    output = _predict_model(model, candidate_set, targets, val_indices, config, device)
    return (
        model,
        output,
        best_epoch,
        best_score,
        history,
        {key: value.detach().cpu() for key, value in best_state.items()},
    )


def _signature(candidate_set, config, annotation_store):
    payload = {
        "version": 1,
        "sample_ids": list(candidate_set.sample_ids),
        "centers_sha256": hashlib.sha256(
            np.asarray(candidate_set.centers, dtype=np.float32).tobytes()
        ).hexdigest(),
        "annotation_sha256": annotation_store.source_hash,
        "config": asdict(config),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


class FittedCurveHard3Refiner:
    def __init__(self, models, report, config, device):
        self.models = [model.to(device).eval() for model in models]
        self.report = dict(report)
        self.config = config
        self.device = device

    def predict(self, dataset, baseline_outputs, label="Curve-H3 inference"):
        centers = {
            sample_id: np.asarray(
                baseline_outputs["prediction"][index], dtype=np.float32
            )
            for index, sample_id in enumerate(baseline_outputs["sample_ids"])
        }
        candidates = extract_dual_view_set(
            dataset,
            self.config.image_size,
            self.config.radius_scale,
            centers,
            label,
            neighbor_count=self.config.neighbor_count,
            include_contour_features=True,
        )
        indices = np.arange(len(candidates), dtype=np.int64)
        member = [
            _predict_model(model, candidates, None, indices, self.config, self.device)
            for model in self.models
        ]
        final_logits = np.mean(
            np.stack([row["final_logits"] for row in member]), axis=0
        )
        landmark_logits = np.mean(
            np.stack([row["landmark_logits"] for row in member]), axis=0
        )
        member_coordinate = np.stack([row["coordinate"] for row in member])
        curve_soft = _decode_numpy(
            final_logits,
            candidates.points,
            candidates.mask,
            self.config.temperature,
            False,
        )
        curve_argmax = _decode_numpy(
            final_logits,
            candidates.points,
            candidates.mask,
            self.config.temperature,
            True,
        )
        point_only = _decode_numpy(
            landmark_logits,
            candidates.points,
            candidates.mask,
            self.config.temperature,
            False,
        )
        baseline_by_id = {
            sample_id: baseline_outputs["prediction"][index, list(HARD3)]
            for index, sample_id in enumerate(baseline_outputs["sample_ids"])
        }
        baseline = np.stack([baseline_by_id[value] for value in candidates.sample_ids])
        spread = np.linalg.norm(
            member_coordinate - member_coordinate.mean(axis=0)[None], axis=-1
        ).mean(axis=0)
        predicted_variance = np.exp(
            np.mean(np.stack([row["log_variance"] for row in member]), axis=0)
        )
        scale = np.maximum(
            np.asarray(self.report["reliability_scale_mm"], dtype=np.float32), 0.25
        )
        reliability = 1.0 / (1.0 + (spread / scale[None]) ** 2)
        reliability *= 1.0 / (1.0 + predicted_variance / 4.0)
        variants = {
            "neural_policy": curve_soft,
            "curve_policy": curve_soft,
            "curve_argmax": curve_argmax,
            "curve_point_only": point_only,
            "atlas_direct": baseline.astype(np.float32),
        }
        oracle = np.min(candidates.target_distance, axis=-1)
        return {
            "sample_ids": candidates.sample_ids,
            "prediction": curve_soft,
            "variant_predictions": variants,
            "expert": candidates.expert,
            "reliability": np.clip(reliability, 0.05, 1.0).astype(np.float32),
            "oracle_error": oracle,
            "validation_diagnostics": {
                "curve_supervision": self.report["annotation_report"],
                "candidate_oracle": {
                    "hard3_ale": float(oracle.mean()),
                    "hard3_p95": float(np.percentile(oracle, 95)),
                    "hard3_sdr_at_2mm": float((oracle <= 2.0).mean()),
                },
            },
        }


def fit_or_load_curve_hard3_refiner(dataset, output_dir, config, device):
    if config.final_model_policy != "inner_fold_ensemble":
        raise ValueError("Curve-H3 currently requires inner_fold_ensemble")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_sample_ids = [sample.sample_id for sample in dataset.samples]
    annotation_store = _annotation_store(config).subset(train_sample_ids)
    candidates = extract_dual_view_set(
        dataset,
        config.image_size,
        config.radius_scale,
        None,
        "Curve-H3 training patches",
        neighbor_count=config.neighbor_count,
        include_contour_features=True,
    )
    targets = build_curve_targets(
        dataset,
        candidates,
        annotation_store,
        config.allow_pseudo_curves,
        config.pseudo_radius_mm,
        config.pseudo_half_length_mm,
    )
    annotated = int(targets.annotation_report["fully_curve_annotated"])
    if annotated < int(config.minimum_annotated_samples):
        raise RuntimeError(
            f"Curve-H3 requires at least {config.minimum_annotated_samples} fully "
            f"annotated training samples; found {annotated}"
        )
    signature = _signature(candidates, config, annotation_store)
    checkpoint_path = output_dir / "curve_hard3_model.pth"
    report_path = output_dir / "curve_hard3_training_report.json"
    if checkpoint_path.exists() and report_path.exists():
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("signature") == signature:
            models = []
            for state in checkpoint["model_states"]:
                model = _new_model(candidates, config)
                model.load_state_dict(state)
                models.append(model)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print("Curve-supervised Hard3 refiner cached", flush=True)
            return FittedCurveHard3Refiner(models, report, config, device)

    started = time.time()
    splits = list(_splitter(candidates.strata, config.folds, config.seed))
    coverage = np.zeros(len(candidates), dtype=np.int64)
    oof_prediction = np.zeros_like(candidates.expert, dtype=np.float32)
    oof_log_variance = np.zeros((len(candidates), 3), dtype=np.float32)
    models, states, fold_reports = [], [], []
    for fold, (train_indices, val_indices) in enumerate(splits, start=1):
        coverage[np.asarray(val_indices, dtype=np.int64)] += 1
        model, output, best_epoch, best_score, history, state = _fit_one(
            candidates,
            targets,
            train_indices,
            val_indices,
            config,
            device,
            fold,
        )
        oof_prediction[np.asarray(val_indices)] = output["coordinate"]
        oof_log_variance[np.asarray(val_indices)] = output["log_variance"]
        models.append(model)
        states.append(state)
        fold_reports.append(
            {
                "fold": fold,
                "train_sample_ids": [candidates.sample_ids[i] for i in train_indices],
                "validation_sample_ids": [
                    candidates.sample_ids[i] for i in val_indices
                ],
                "best_epoch": best_epoch,
                "best_validation_hard3_ale": best_score,
                "history": history,
            }
        )
    if not np.all(coverage == 1):
        raise RuntimeError("Curve-H3 requires exactly one OOF prediction per sample")
    oof_error = np.linalg.norm(oof_prediction - candidates.expert, axis=-1)
    member_prediction = np.stack(
        [
            _predict_model(
                model,
                candidates,
                targets,
                np.arange(len(candidates)),
                config,
                device,
            )["coordinate"]
            for model in models
        ]
    )
    spread = np.linalg.norm(
        member_prediction - member_prediction.mean(axis=0)[None], axis=-1
    ).mean(axis=0)
    reliability_scale = np.maximum(np.percentile(spread, 75, axis=0), 0.25)
    parameter_count = sum(parameter.numel() for parameter in models[0].parameters())
    report = {
        "version": "Curve-H3-v1",
        "method": (
            "curve-support segmentation followed by landmark-on-curve ranking with "
            "bilateral Gonion context"
        ),
        "uses_validation_labels_for_model_fit": False,
        "uses_test_labels": False,
        "publication_ready": bool(
            annotated >= int(config.minimum_annotated_samples)
            and int(config.minimum_annotated_samples) > 0
        ),
        "annotation_report": targets.annotation_report,
        "sample_count": len(candidates),
        "parameter_count_per_member": int(parameter_count),
        "ensemble_members": len(models),
        "reliability_scale_mm": reliability_scale.tolist(),
        "oof": {
            "hard3": summarize(oof_error),
            "lm0": summarize(oof_error[:, 0]),
            "lm21": summarize(oof_error[:, 1]),
            "lm22": summarize(oof_error[:, 2]),
            "candidate_oracle": summarize(np.min(candidates.target_distance, axis=-1)),
        },
        "folds": fold_reports,
        "training_seconds": float(time.time() - started),
        "config": asdict(config),
    }
    torch.save(
        {
            "signature": signature,
            "model_states": states,
            "input_channels": int(candidates.images.shape[3]),
            "feature_dim": int(candidates.canonical.shape[-1]),
        },
        checkpoint_path,
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return FittedCurveHard3Refiner(models, report, config, device)
