"""Training, inference and validation policy for Core20-MVSC Stage 4."""

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
from torch.utils.data import DataLoader, Dataset

from all23_rgb_geodesic_cascade.anatomy import HARD3, NUM_LANDMARKS
from all23_rgb_geodesic_cascade.metrics import bootstrap_delta, summarize, write_csv

from .anatomy import CORE20_GROUPS, CORE20_INDICES, GROUP_NAMES, core20_group_index
from .model import Core20MVSCNet
from .patches import (
    GEOMETRY_FEATURES,
    IMAGE_CHANNELS,
    Core20CandidateSet,
    extract_core20_set,
    group_indices,
)
from .spatial_prior import ConditionalSpatialPrior, SpatialPriorConfig


@dataclass
class Core20MVSCConfig:
    enabled: bool = True
    ablation: str = "C4"
    folds: int = 5
    epochs: int = 120
    min_epochs: int = 40
    patience: int = 25
    batch_size: int = 32
    image_size: int = 96
    width: int = 48
    dropout: float = 0.10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    coordinate_topk: int = 8
    coordinate_temperature: float = 0.5
    color_noise: float = 0.02
    candidate_dropout: float = 0.05
    heatmap_weight: float = 1.0
    coordinate_weight: float = 1.0
    hard_negative_weight: float = 0.25
    base_regret_weight: float = 0.25
    anatomy_weight: float = 0.05
    confidence_weight: float = 0.02
    hard_negative_count: int = 16
    hard_negative_margin: float = 0.5
    prior_folds: int = 5
    prior_l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    target_core20_ale: float = 1.70
    stretch_core20_ale: float = 1.647
    minimum_improvement_probability: float = 0.95
    maximum_group_regression_mm: float = 0.10
    blend_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    bootstrap_iters: int = 2000
    mixed_precision: bool = True
    amp_dtype: str = "bfloat16"
    seed: int = 42


def _config_signature(config, image_channels=None, geometry_dim=None):
    payload = {
        **asdict(config),
        "image_channels": image_channels,
        "geometry_dim": geometry_dim,
        "pipeline_version": 2,
    }
    encoded = json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dataset_shapes(dataset):
    shapes, sample_ids, classes, genders = [], [], [], []
    for sample in dataset.samples:
        with np.load(dataset.records[sample.sample_id]) as record:
            if "landmarks" not in record.files:
                raise RuntimeError(
                    f"Core20 outer-train record has no landmarks: {sample.sample_id}"
                )
            shapes.append(record["landmarks"].astype(np.float32))
        sample_ids.append(sample.sample_id)
        classes.append(sample.class_name)
        genders.append(sample.gender)
    return np.stack(shapes), sample_ids, classes, genders


def _prior_covariance_rows(prior, count):
    row = np.stack(
        [prior.parameters[str(landmark)]["covariance"] for landmark in CORE20_INDICES]
    ).astype(np.float32)
    return np.repeat(row[None], count, axis=0)


def _training_centers(dataset, shapes, seed):
    """Expert+jitter centers using the requested empirical/Gaussian mixture."""
    result = {}
    rng = np.random.default_rng(seed)
    for index, sample in enumerate(dataset.samples):
        expert = shapes[index]
        oof_coarse = dataset._coarse(sample)
        centers = oof_coarse.copy()
        for landmark in CORE20_INDICES:
            draw = rng.random()
            if draw < 0.50:
                noise = oof_coarse[landmark] - expert[landmark]
            elif draw < 0.85:
                noise = rng.normal(0.0, 1.5, size=3)
            else:
                noise = rng.normal(0.0, 3.0, size=3)
            noise = np.asarray(noise, dtype=np.float32)
            length = float(np.linalg.norm(noise))
            if length > 6.0:
                noise *= 6.0 / length
            centers[landmark] = expert[landmark] + noise
        result[sample.sample_id] = centers.astype(np.float32)
    return result


class _FlatCore20Dataset(Dataset):
    def __init__(self, candidates, sample_indices, config, training=False):
        self.candidates = candidates
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.config = config
        self.training = bool(training)
        self.rows = [
            (int(sample_index), local_landmark)
            for sample_index in self.sample_indices
            for local_landmark in range(20)
        ]
        self.groups = group_indices()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sample, landmark = self.rows[index]
        images = torch.from_numpy(
            self.candidates.images[sample, landmark].astype(np.float32)
        )
        features = torch.from_numpy(self.candidates.features[sample, landmark].copy())
        mask = self.candidates.mask[sample, landmark].copy()
        if self.training:
            if self.config.color_noise > 0:
                images[:, :3] = torch.clamp(
                    images[:, :3]
                    + torch.randn_like(images[:, :3]) * self.config.color_noise,
                    0.0,
                    1.0,
                )
            # Geometry-only fallback is learned even when most meshes have RGB.
            if torch.rand(()) < 0.10:
                images[:, :3] = 0.0
                images[:, 9:12] = 0.0
                features[:, 12:18] = 0.0
                features[:, 21:24] = 0.0
            if self.config.candidate_dropout > 0:
                rng = np.random.default_rng(
                    self.config.seed * 1009
                    + index
                    + int(torch.randint(0, 1_000_000, ()).item())
                )
                dropped = rng.random(len(mask)) < self.config.candidate_dropout
                closest = int(
                    np.argmin(self.candidates.target_distance[sample, landmark])
                )
                dropped[closest] = False
                mask &= ~dropped
        return {
            "images": images,
            "grids": torch.from_numpy(self.candidates.grids[sample, landmark]),
            "points": torch.from_numpy(self.candidates.points[sample, landmark]),
            "features": features,
            "mask": torch.from_numpy(mask),
            "heatmap_target": torch.from_numpy(
                self.candidates.heatmap_target[sample, landmark]
            ),
            "distance": torch.from_numpy(
                self.candidates.target_distance[sample, landmark]
            ),
            "expert": torch.from_numpy(self.candidates.expert[sample, landmark]),
            "base": torch.from_numpy(self.candidates.base[sample, landmark]),
            "prior_mean": torch.from_numpy(
                self.candidates.prior_mean[sample, landmark]
            ),
            "prior_covariance": torch.from_numpy(
                self.candidates.prior_covariance[sample, landmark]
            ),
            "prior_score": torch.from_numpy(
                self.candidates.prior_score[sample, landmark]
            ),
            "anatomy_anchors": torch.from_numpy(
                self.candidates.anatomy_anchors[sample, landmark]
            ),
            "anatomy_distances": torch.from_numpy(
                self.candidates.anatomy_distances[sample, landmark]
            ),
            "anatomy_mask": torch.from_numpy(
                self.candidates.anatomy_mask[sample, landmark]
            ),
            "local_landmark": torch.tensor(landmark, dtype=torch.long),
            "actual_landmark": torch.tensor(CORE20_INDICES[landmark], dtype=torch.long),
            "group": torch.tensor(self.groups[landmark], dtype=torch.long),
            "sample_index": torch.tensor(sample, dtype=torch.long),
        }


def _move(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _autocast(config, device):
    enabled = bool(config.mixed_precision and device.type == "cuda")
    dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _hard_negative_loss(logits, distance, mask, count, margin):
    nearest = distance.masked_fill(~mask, torch.inf).argmin(dim=-1)
    positive = torch.gather(logits, 1, nearest[:, None]).squeeze(1)
    negative_mask = mask & (distance > 3.0)
    safe = logits.masked_fill(~negative_mask, -torch.inf)
    chosen = min(max(1, int(count)), logits.shape[-1])
    values, indices = torch.topk(safe, chosen, dim=-1)
    valid = torch.gather(negative_mask, 1, indices)
    losses = F.softplus(values - positive[:, None] + float(margin))
    losses = torch.where(valid, losses, torch.zeros_like(losses))
    return (losses.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)).mean()


def _loss(outputs, batch, config):
    mask = batch["mask"].bool()
    target = batch["heatmap_target"].float().masked_fill(~mask, 0.0)
    missing = target.sum(dim=-1, keepdim=True) <= 1e-8
    fallback = torch.exp(-batch["distance"].float().square() / (2.0 * 2.5**2))
    target = torch.where(missing, fallback.masked_fill(~mask, 0.0), target)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_probability = torch.log_softmax(outputs["logits"].float(), dim=-1)
    safe_log_probability = torch.where(
        mask, log_probability, torch.zeros_like(log_probability)
    )
    heatmap = (
        (target * (torch.log(target.clamp_min(1e-8)) - safe_log_probability))
        .sum(dim=-1)
        .mean()
    )
    coordinate = F.smooth_l1_loss(outputs["final"], batch["expert"].float(), beta=1.0)
    hard_negative = _hard_negative_loss(
        outputs["logits"].float(),
        batch["distance"].float(),
        mask,
        config.hard_negative_count,
        config.hard_negative_margin,
    )

    final_error = torch.linalg.norm(outputs["final"] - batch["expert"].float(), dim=-1)
    proposal_error = torch.linalg.norm(
        outputs["proposal"] - batch["expert"].float(), dim=-1
    )
    base_error = torch.linalg.norm(
        batch["base"].float() - batch["expert"].float(), dim=-1
    )
    regret = F.relu(final_error - base_error).square().mean() / 4.0
    gate_target = (proposal_error < base_error).float()
    gate_loss = (
        F.binary_cross_entropy_with_logits(
            outputs["gate_logit"].float(), gate_target
        )
        if config.ablation == "C4"
        else torch.zeros((), device=final_error.device)
    )
    base_regret = regret + 0.25 * gate_loss

    anchor_distance = torch.linalg.norm(
        outputs["final"][:, None] - batch["anatomy_anchors"].float(), dim=-1
    )
    anchor_loss = F.smooth_l1_loss(
        anchor_distance / 10.0,
        batch["anatomy_distances"].float() / 10.0,
        reduction="none",
    )
    anchor_mask = batch["anatomy_mask"].float()
    anatomy = (anchor_loss * anchor_mask).sum() / anchor_mask.sum().clamp_min(1.0)

    normalized_squared_error = (final_error / 2.0).square()
    confidence = (
        0.5
        * (
            torch.exp(-outputs["log_variance"]) * normalized_squared_error
            + outputs["log_variance"]
        ).mean()
    )
    total = (
        config.heatmap_weight * heatmap
        + config.coordinate_weight * coordinate
        + config.hard_negative_weight * hard_negative
        + config.base_regret_weight * base_regret
        + config.anatomy_weight * anatomy
        + config.confidence_weight * confidence
    )
    components = {
        "heatmap_kl": heatmap,
        "coordinate": coordinate,
        "hard_negative": hard_negative,
        "base_regret": base_regret,
        "anatomy": anatomy,
        "confidence_nll": confidence,
    }
    return total, components, final_error


def _inner_split(candidates, seed):
    rng = np.random.default_rng(seed)
    validation = []
    strata = np.asarray(
        [f"{c}|{g}" for c, g in zip(candidates.classes, candidates.genders)],
        dtype=object,
    )
    for value in sorted(set(strata.tolist())):
        rows = np.flatnonzero(strata == value)
        rng.shuffle(rows)
        count = max(1, int(round(0.20 * len(rows))))
        validation.extend(rows[:count].tolist())
    validation = np.asarray(sorted(set(validation)), dtype=np.int64)
    train = np.setdiff1d(np.arange(len(candidates)), validation)
    if len(train) == 0:
        train, validation = np.arange(len(candidates) - 1), np.asarray(
            [len(candidates) - 1]
        )
    return train, validation


@torch.no_grad()
def _evaluate(model, loader, device, config):
    model.eval()
    errors, landmarks = [], []
    for batch in loader:
        batch = _move(batch, device)
        with _autocast(config, device):
            outputs = model(batch)
        errors.append(
            torch.linalg.norm(
                outputs["final"].float() - batch["expert"].float(), dim=-1
            )
            .cpu()
            .numpy()
        )
        landmarks.append(batch["local_landmark"].cpu().numpy())
    errors = np.concatenate(errors)
    landmarks = np.concatenate(landmarks)
    per_landmark = [float(errors[landmarks == index].mean()) for index in range(20)]
    return float(errors.mean()), per_landmark


def _fit_network(candidates, output_dir, config, device):
    # Upstream checkpoint/cache reuse must not change Stage 4 initialization.
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    train_index, validation_index = _inner_split(candidates, config.seed)
    train_dataset = _FlatCore20Dataset(candidates, train_index, config, training=True)
    validation_dataset = _FlatCore20Dataset(
        candidates, validation_index, config, training=False
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = Core20MVSCNet(
        image_channels=candidates.images.shape[3],
        geometry_dim=candidates.features.shape[-1],
        width=config.width,
        dropout=config.dropout,
        coordinate_topk=config.coordinate_topk,
        coordinate_temperature=config.coordinate_temperature,
        use_images=config.ablation in ("C2", "C3", "C4"),
        use_spatial_prior=config.ablation in ("C3", "C4"),
        use_confidence_gate=config.ablation == "C4",
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, config.patience // 4),
        min_lr=1e-6,
    )
    use_scaler = bool(
        config.mixed_precision
        and device.type == "cuda"
        and config.amp_dtype == "float16"
    )
    scaler = _grad_scaler(use_scaler)
    checkpoint_path = Path(output_dir) / "best_model.pth"
    last_checkpoint_path = Path(output_dir) / "last_model.pth"
    history_path = Path(output_dir) / "history.json"
    best_ale, best_epoch, stale = float("inf"), 0, 0
    history = []
    start_epoch = 1
    previous_training_seconds = 0.0
    signature = _config_signature(
        config, candidates.images.shape[3], candidates.features.shape[-1]
    )
    if last_checkpoint_path.exists():
        resume = _torch_load(last_checkpoint_path, device)
        if resume.get("signature") == signature:
            model.load_state_dict(resume["model"])
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            if resume.get("scaler"):
                scaler.load_state_dict(resume["scaler"])
            best_ale = float(resume["best_ale"])
            best_epoch = int(resume["best_epoch"])
            stale = int(resume["stale"])
            history = list(resume["history"])
            previous_training_seconds = float(resume.get("training_seconds", 0.0))
            start_epoch = int(resume["epoch"]) + 1
            generator.set_state(resume["loader_generator_state"])
            torch.set_rng_state(resume["torch_rng_state"].cpu())
            if device.type == "cuda" and resume.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
            print(
                f"Resuming Core20-MVSC after epoch {start_epoch - 1}; "
                f"best epoch={best_epoch}",
                flush=True,
            )
    started = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        totals = []
        component_rows = {}
        per_landmark_sum = np.zeros(20, dtype=np.float64)
        per_landmark_count = np.zeros(20, dtype=np.int64)
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                outputs = model(batch)
                loss, components, errors = _loss(outputs, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite Core20-MVSC loss at epoch={epoch} "
                    f"batch={batch_index}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            if not torch.isfinite(gradient_norm):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f"Non-finite Core20-MVSC gradient at epoch={epoch} "
                    f"batch={batch_index}"
                )
            scaler.step(optimizer)
            scaler.update()
            totals.append(float(loss.detach().cpu()))
            for name, value in components.items():
                component_rows.setdefault(name, []).append(float(value.detach().cpu()))
            landmark_rows = batch["local_landmark"].detach().cpu().numpy()
            error_rows = errors.detach().cpu().numpy()
            for landmark in np.unique(landmark_rows):
                selected = landmark_rows == landmark
                per_landmark_sum[landmark] += float(error_rows[selected].sum())
                per_landmark_count[landmark] += int(selected.sum())
        validation_ale, validation_landmarks = _evaluate(
            model, validation_loader, device, config
        )
        scheduler.step(validation_ale)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "validation_core20_ale": validation_ale,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "components": {
                name: float(np.mean(values)) for name, values in component_rows.items()
            },
            "train_landmark_ale": (
                per_landmark_sum / np.maximum(per_landmark_count, 1)
            ).tolist(),
            "validation_landmark_ale": validation_landmarks,
        }
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Core20-MVSC epoch {epoch:03d}/{config.epochs} "
            f"train={row['train_loss']:.5f} val_ALE={validation_ale:.4f}",
            flush=True,
        )
        if validation_ale < best_ale - 1e-5:
            best_ale, best_epoch, stale = validation_ale, epoch, 0
            _atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "image_channels": candidates.images.shape[3],
                    "geometry_dim": candidates.features.shape[-1],
                    "config": asdict(config),
                    "signature": _config_signature(
                        config,
                        candidates.images.shape[3],
                        candidates.features.shape[-1],
                    ),
                },
                checkpoint_path,
            )
        else:
            stale += 1
        _atomic_torch_save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_ale": best_ale,
                "best_epoch": best_epoch,
                "stale": stale,
                "history": history,
                "loader_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
                "training_seconds": previous_training_seconds
                + float(time.time() - started),
                "signature": signature,
            },
            last_checkpoint_path,
        )
        if epoch >= config.min_epochs and stale >= config.patience:
            print(
                f"Core20-MVSC early stopping at epoch {epoch}; best epoch={best_epoch}",
                flush=True,
            )
            break
    checkpoint = _torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    report = {
        "version": "Core20-MVSC-v1",
        "ablation": config.ablation,
        "best_epoch": best_epoch,
        "best_inner_validation_core20_ale": best_ale,
        "training_seconds": previous_training_seconds + float(time.time() - started),
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "image_channels": list(IMAGE_CHANNELS),
        "geometry_features": list(GEOMETRY_FEATURES),
        "inner_train_sample_ids": [
            candidates.sample_ids[index] for index in train_index
        ],
        "inner_validation_sample_ids": [
            candidates.sample_ids[index] for index in validation_index
        ],
        "uses_outer_validation_for_training": False,
        "uses_test_labels": False,
        "signature": signature,
        "training_complete": True,
    }
    return model, report


def _model_from_checkpoint(checkpoint, device):
    config = Core20MVSCConfig(**checkpoint["config"])
    model = Core20MVSCNet(
        image_channels=checkpoint["image_channels"],
        geometry_dim=checkpoint["geometry_dim"],
        width=config.width,
        dropout=config.dropout,
        coordinate_topk=config.coordinate_topk,
        coordinate_temperature=config.coordinate_temperature,
        use_images=config.ablation in ("C2", "C3", "C4"),
        use_spatial_prior=config.ablation in ("C3", "C4"),
        use_confidence_gate=config.ablation == "C4",
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, config


@torch.no_grad()
def _predict_set(model, candidates, config, device):
    dataset = _FlatCore20Dataset(
        candidates, np.arange(len(candidates)), config, training=False
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    rows = {
        name: []
        for name in (
            "final",
            "proposal",
            "confidence",
            "gate_alpha",
            "entropy",
            "margin",
            "prior_disagreement",
            "view_weights",
        )
    }
    model.eval()
    for batch in loader:
        batch = _move(batch, device)
        with _autocast(config, device):
            outputs = model(batch)
        for name in rows:
            rows[name].append(outputs[name].float().cpu().numpy())
    stacked = {name: np.concatenate(values, axis=0) for name, values in rows.items()}
    samples = len(candidates)
    for name in ("final", "proposal"):
        stacked[name] = stacked[name].reshape(samples, 20, 3)
    for name in ("confidence", "gate_alpha", "entropy", "margin", "prior_disagreement"):
        stacked[name] = stacked[name].reshape(samples, 20)
    stacked["view_weights"] = stacked["view_weights"].reshape(samples, 20, 3)
    return stacked


class FittedCore20MVSCRefiner:
    def __init__(self, model, prior, config, report, device):
        self.model = model
        self.prior = prior
        self.config = config
        self.report = report
        self.device = device

    def predict(self, dataset, baseline_outputs, label="Core20-MVSC inference"):
        sample_ids = list(baseline_outputs["sample_ids"])
        lookup = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        expected = [sample.sample_id for sample in dataset.samples]
        missing = [sample_id for sample_id in expected if sample_id not in lookup]
        if missing:
            raise KeyError(f"Core20 baseline outputs miss samples: {missing[:5]}")
        order = np.asarray(
            [lookup[sample_id] for sample_id in expected], dtype=np.int64
        )
        base_full = np.asarray(baseline_outputs["prediction"], dtype=np.float32)[order]
        classes = [baseline_outputs["classes"][index] for index in order]
        genders = [baseline_outputs["genders"][index] for index in order]
        prior_mean, prior_covariance = self.prior.predict(base_full, classes, genders)
        candidates = extract_core20_set(
            dataset,
            {sample_id: base_full[index] for index, sample_id in enumerate(expected)},
            {sample_id: prior_mean[index] for index, sample_id in enumerate(expected)},
            {
                sample_id: prior_covariance[index]
                for index, sample_id in enumerate(expected)
            },
            image_size=self.config.image_size,
            label=label,
        )
        prediction = _predict_set(self.model, candidates, self.config, self.device)
        oracle = np.where(candidates.mask, candidates.target_distance, np.inf).min(
            axis=-1
        )
        return {
            "sample_ids": expected,
            "prediction": prediction["final"],
            "proposal": prediction["proposal"],
            "confidence": prediction["confidence"],
            "gate_alpha": prediction["gate_alpha"],
            "entropy": prediction["entropy"],
            "margin": prediction["margin"],
            "prior_disagreement": prediction["prior_disagreement"],
            "view_weights": prediction["view_weights"],
            "oracle": oracle,
            "has_rgb": candidates.has_rgb,
        }


def fit_or_load_core20_refiner(dataset, output_dir, config, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    prior_path = output_dir / "spatial_prior.json"
    report_path = output_dir / "training_report.json"
    if checkpoint_path.exists() and prior_path.exists() and report_path.exists():
        checkpoint = _torch_load(checkpoint_path, device)
        expected_signature = _config_signature(
            config, checkpoint["image_channels"], checkpoint["geometry_dim"]
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("signature") == expected_signature
            and report.get("signature") == expected_signature
            and report.get("training_complete", False)
        ):
            model, loaded_config = _model_from_checkpoint(checkpoint, device)
            print("Core20-MVSC refiner cached", flush=True)
            return FittedCore20MVSCRefiner(
                model,
                ConditionalSpatialPrior.load(prior_path),
                loaded_config,
                report,
                device,
            )

    shapes, sample_ids, classes, genders = _dataset_shapes(dataset)
    prior = ConditionalSpatialPrior(
        SpatialPriorConfig(
            folds=config.prior_folds,
            l2_grid=tuple(config.prior_l2_grid),
            seed=config.seed,
        )
    ).fit(shapes, sample_ids, classes, genders)
    prior.save(prior_path)
    centers = _training_centers(dataset, shapes, config.seed)
    training_shape_context = np.stack([centers[sample_id] for sample_id in sample_ids])
    prior_means = prior.oof_predict_from_shapes(
        training_shape_context, sample_ids, classes, genders
    )
    prior_covariances = _prior_covariance_rows(prior, len(sample_ids))
    candidates = extract_core20_set(
        dataset,
        centers,
        {sample_id: prior_means[index] for index, sample_id in enumerate(sample_ids)},
        {
            sample_id: prior_covariances[index]
            for index, sample_id in enumerate(sample_ids)
        },
        image_size=config.image_size,
        label="Core20-MVSC outer-train OOF patches",
    )
    model, report = _fit_network(candidates, output_dir, config, device)
    report["spatial_prior"] = prior.report()
    report["jitter_mixture"] = {
        "stage1_oof_residual_fraction": 0.50,
        "gaussian_1p5mm_fraction": 0.35,
        "gaussian_3mm_fraction": 0.15,
        "maximum_norm_mm": 6.0,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return FittedCore20MVSCRefiner(model, prior, config, report, device)


def _ordered(candidate_result, outputs, field):
    lookup = {
        sample_id: index
        for index, sample_id in enumerate(candidate_result["sample_ids"])
    }
    missing = [
        sample_id for sample_id in outputs["sample_ids"] if sample_id not in lookup
    ]
    if missing:
        raise KeyError(f"Core20 candidate output misses samples: {missing[:5]}")
    order = [lookup[sample_id] for sample_id in outputs["sample_ids"]]
    return np.asarray(candidate_result[field])[order]


def _anatomical_group_rows(errors):
    rows = []
    for group, landmarks in CORE20_GROUPS.items():
        values = errors[:, [CORE20_INDICES.index(index) for index in landmarks]]
        rows.append(
            {
                "group": group,
                "landmarks": ",".join(map(str, landmarks)),
                **summarize(values),
            }
        )
    return rows


def calibrate_core20_policy(outputs, candidate_result, config):
    base_full = np.asarray(outputs["prediction"], dtype=np.float32)
    expert_full = np.asarray(outputs["expert"], dtype=np.float32)
    base = base_full[:, list(CORE20_INDICES)]
    expert = expert_full[:, list(CORE20_INDICES)]
    candidate = _ordered(candidate_result, outputs, "prediction").astype(np.float32)
    base_error = np.linalg.norm(base - expert, axis=-1)
    group_alpha = {}
    sweep = []
    final = base.copy()
    for group, landmarks in CORE20_GROUPS.items():
        local = [CORE20_INDICES.index(index) for index in landmarks]
        rows = []
        for alpha in config.blend_grid:
            prediction = base[:, local] + float(alpha) * (
                candidate[:, local] - base[:, local]
            )
            error = np.linalg.norm(prediction - expert[:, local], axis=-1)
            row = {"group": group, "alpha": float(alpha), **summarize(error)}
            rows.append(row)
            sweep.append(row)
        selected = min(rows, key=lambda row: (row["ale"], row["alpha"]))
        group_alpha[group] = selected["alpha"]
        final[:, local] = base[:, local] + selected["alpha"] * (
            candidate[:, local] - base[:, local]
        )
    final_error = np.linalg.norm(final - expert, axis=-1)
    base_group = {row["group"]: row for row in _anatomical_group_rows(base_error)}
    proposed_group = {row["group"]: row for row in _anatomical_group_rows(final_error)}
    proposed_group_regression = {
        group: proposed_group[group]["ale"] - base_group[group]["ale"]
        for group in CORE20_GROUPS
    }
    bootstrap = bootstrap_delta(
        base_error, final_error, config.bootstrap_iters, config.seed
    )
    accepted = bool(
        final_error.mean() < base_error.mean()
        and bootstrap["probability_improved"] >= config.minimum_improvement_probability
        and max(proposed_group_regression.values())
        <= config.maximum_group_regression_mm
    )
    if not accepted:
        group_alpha = {group: 0.0 for group in CORE20_GROUPS}
        final = base.copy()
        final_error = base_error.copy()
    final_group = {row["group"]: row for row in _anatomical_group_rows(final_error)}
    group_regression = {
        group: final_group[group]["ale"] - base_group[group]["ale"]
        for group in CORE20_GROUPS
    }
    core20_ale = float(final_error.mean())
    run_full_cv = bool(
        accepted
        and core20_ale <= config.target_core20_ale
        and bootstrap["probability_improved"] >= config.minimum_improvement_probability
        and max(group_regression.values()) <= config.maximum_group_regression_mm
    )
    oracle = _ordered(candidate_result, outputs, "oracle")
    return {
        "version": "Core20-MVSC-v1",
        "accepted": accepted,
        "selected_group_alpha": group_alpha,
        "base_core20": summarize(base_error),
        "candidate_core20": summarize(np.linalg.norm(candidate - expert, axis=-1)),
        "selected_core20": summarize(final_error),
        "candidate_oracle_core20": summarize(oracle),
        "base_anatomical_groups": list(base_group.values()),
        "selected_anatomical_groups": list(final_group.values()),
        "proposed_anatomical_groups": list(proposed_group.values()),
        "proposed_group_regression_mm": proposed_group_regression,
        "group_regression_mm": group_regression,
        "bootstrap_vs_base": bootstrap,
        "targets": {
            "development_core20_ale": config.target_core20_ale,
            "stretch_core20_ale": config.stretch_core20_ale,
            "minimum_improvement_probability": config.minimum_improvement_probability,
            "maximum_group_regression_mm": config.maximum_group_regression_mm,
        },
        "checks": {
            "core20_at_or_below_development_target": core20_ale
            <= config.target_core20_ale,
            "bootstrap_probability_at_or_above_gate": bootstrap["probability_improved"]
            >= config.minimum_improvement_probability,
            "no_anatomical_group_regression_over_limit": max(group_regression.values())
            <= config.maximum_group_regression_mm,
        },
        "run_full_cv": run_full_cv,
        "validation_labels_used_for_group_blend_only": True,
        "test_labels_used": False,
        "sweep": sweep,
    }


def apply_core20_refinement(outputs, candidate_result, policy):
    result = dict(outputs)
    base = np.asarray(outputs["prediction"])
    candidate = _ordered(candidate_result, outputs, "prediction").astype(
        base.dtype, copy=False
    )
    hard3_before = base[:, list(HARD3)].copy()
    selected = base[:, list(CORE20_INDICES)].copy()
    effective_alpha = np.zeros((len(base), 20), dtype=np.float32)
    if policy.get("accepted", False):
        for group, landmarks in CORE20_GROUPS.items():
            local = [CORE20_INDICES.index(index) for index in landmarks]
            alpha = float(policy["selected_group_alpha"][group])
            selected[:, local] += alpha * (candidate[:, local] - selected[:, local])
            effective_alpha[:, local] = alpha
    prediction = base.copy()
    prediction[:, list(CORE20_INDICES)] = selected
    if not np.array_equal(prediction[:, list(HARD3)], hard3_before):
        raise AssertionError("Core20 Stage 4 changed frozen Hard3 coordinates")
    full_candidate = base.copy()
    full_candidate[:, list(CORE20_INDICES)] = candidate
    result["pre_core20_prediction"] = base.copy()
    result["core20_candidate"] = full_candidate
    result["prediction"] = prediction
    result["errors"] = np.linalg.norm(prediction - result["expert"], axis=-1)
    for source, destination in (
        ("confidence", "core20_confidence"),
        ("gate_alpha", "core20_gate_alpha"),
        ("entropy", "core20_entropy"),
        ("margin", "core20_margin"),
        ("prior_disagreement", "core20_prior_disagreement"),
    ):
        values = _ordered(candidate_result, outputs, source)
        full = np.full((len(base), NUM_LANDMARKS), np.nan, dtype=np.float32)
        full[:, list(CORE20_INDICES)] = values
        result[destination] = full
    full_alpha = np.zeros((len(base), NUM_LANDMARKS), dtype=np.float32)
    full_alpha[:, list(CORE20_INDICES)] = effective_alpha
    result["core20_effective_alpha"] = full_alpha
    return result


def write_core20_reports(output_dir, policy, split_name="val"):
    output_dir = Path(output_dir)
    (output_dir / "core20_feasibility.json").write_text(
        json.dumps(
            {
                "candidate_oracle_core20": policy["candidate_oracle_core20"],
                "target_core20_ale": policy["targets"]["development_core20_ale"],
                "candidate_coverage_is_sufficient": policy["candidate_oracle_core20"][
                    "ale"
                ]
                < 1.0,
                "interpretation": "The remaining error is candidate ranking, not ROI coverage.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "core20_stage4_decision.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    write_csv(
        output_dir / f"anatomical_group_metrics_{split_name}.csv",
        policy["selected_anatomical_groups"],
    )
    write_csv(output_dir / "core20_blend_sweep.csv", policy["sweep"])


def write_core20_split_report(
    output_dir, split_name, outputs, candidate_result, bootstrap_iters=2000, seed=42
):
    """Report a locked Stage 4 policy without tuning anything on this split."""
    output_dir = Path(output_dir)
    selected = np.asarray(outputs["prediction"], dtype=np.float32)[
        :, list(CORE20_INDICES)
    ]
    base = np.asarray(
        outputs.get("pre_core20_prediction", outputs["prediction"]),
        dtype=np.float32,
    )[:, list(CORE20_INDICES)]
    expert = np.asarray(outputs["expert"], dtype=np.float32)[:, list(CORE20_INDICES)]
    candidate = _ordered(candidate_result, outputs, "prediction").astype(np.float32)
    oracle = _ordered(candidate_result, outputs, "oracle").astype(np.float32)
    confidence = _ordered(candidate_result, outputs, "confidence").astype(np.float32)
    gate_alpha = _ordered(candidate_result, outputs, "gate_alpha").astype(np.float32)
    entropy = _ordered(candidate_result, outputs, "entropy").astype(np.float32)
    margin = _ordered(candidate_result, outputs, "margin").astype(np.float32)
    base_error = np.linalg.norm(base - expert, axis=-1)
    candidate_error = np.linalg.norm(candidate - expert, axis=-1)
    selected_error = np.linalg.norm(selected - expert, axis=-1)
    rows = []
    for group, landmarks in CORE20_GROUPS.items():
        local = [CORE20_INDICES.index(index) for index in landmarks]
        base_summary = summarize(base_error[:, local])
        candidate_summary = summarize(candidate_error[:, local])
        selected_summary = summarize(selected_error[:, local])
        rows.append(
            {
                "group": group,
                "landmarks": ",".join(map(str, landmarks)),
                "base_ale": base_summary["ale"],
                "candidate_ale": candidate_summary["ale"],
                "selected_ale": selected_summary["ale"],
                "selected_median": selected_summary["median"],
                "selected_std": selected_summary["std"],
                "gain_vs_base_mm": base_summary["ale"] - selected_summary["ale"],
                "oracle_ale": float(oracle[:, local].mean()),
                "mean_confidence": float(confidence[:, local].mean()),
                "mean_gate_alpha": float(gate_alpha[:, local].mean()),
                "mean_entropy": float(entropy[:, local].mean()),
                "mean_peak_margin": float(margin[:, local].mean()),
            }
        )
    payload = {
        "split": split_name,
        "policy_locked_before_split_evaluation": split_name == "test",
        "base_core20": summarize(base_error),
        "candidate_core20": summarize(candidate_error),
        "selected_core20": summarize(selected_error),
        "candidate_oracle_core20": summarize(oracle),
        "bootstrap_vs_base": bootstrap_delta(
            base_error, selected_error, bootstrap_iters, seed
        ),
        "anatomical_groups": rows,
    }
    (output_dir / f"core20_metrics_{split_name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / f"anatomical_group_metrics_{split_name}.csv", rows)
    return payload


def run_core20_preflight(
    train_dataset, validation_dataset, output_dir, config, device, sample_limit=2
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes, sample_ids, classes, genders = _dataset_shapes(train_dataset)
    prior = ConditionalSpatialPrior(
        SpatialPriorConfig(
            folds=config.prior_folds,
            l2_grid=tuple(config.prior_l2_grid),
            seed=config.seed,
        )
    ).fit(shapes, sample_ids, classes, genders)
    prior.save(output_dir / "spatial_prior.json")
    subset = copy.copy(train_dataset)
    subset.samples = subset.samples[: min(sample_limit, len(subset.samples))]
    subset_ids = [sample.sample_id for sample in subset.samples]
    all_centers = _training_centers(train_dataset, shapes, config.seed)
    all_center_shapes = np.stack([all_centers[sample_id] for sample_id in sample_ids])
    all_oof = prior.oof_predict_from_shapes(
        all_center_shapes, sample_ids, classes, genders
    )
    center_lookup = {sample_id: all_centers[sample_id] for sample_id in subset_ids}
    oof_lookup = {
        sample_id: all_oof[sample_ids.index(sample_id)] for sample_id in subset_ids
    }
    covariance = _prior_covariance_rows(prior, len(subset_ids))
    candidates = extract_core20_set(
        subset,
        center_lookup,
        oof_lookup,
        {sample_id: covariance[index] for index, sample_id in enumerate(subset_ids)},
        image_size=min(config.image_size, 48),
        label="Core20-MVSC preflight patches",
    )
    model = Core20MVSCNet(
        candidates.images.shape[3],
        candidates.features.shape[-1],
        width=min(config.width, 24),
        coordinate_topk=min(config.coordinate_topk, candidates.points.shape[2]),
        use_images=config.ablation in ("C2", "C3", "C4"),
        use_spatial_prior=config.ablation in ("C3", "C4"),
        use_confidence_gate=config.ablation == "C4",
    ).to(device)
    loader = DataLoader(
        _FlatCore20Dataset(candidates, np.arange(len(candidates)), config),
        batch_size=min(4, config.batch_size),
    )
    batch = _move(next(iter(loader)), device)
    with torch.no_grad():
        outputs = model(batch)
        geometry_fallback = dict(batch)
        geometry_fallback["images"] = batch["images"].clone()
        geometry_fallback["images"][:, :, :3] = 0.0
        geometry_fallback["images"][:, :, 9:12] = 0.0
        geometry_fallback["features"] = batch["features"].clone()
        geometry_fallback["features"][:, :, 12:18] = 0.0
        geometry_fallback["features"][:, :, 21:24] = 0.0
        fallback = model(geometry_fallback)
    train_ids = set(sample_ids)
    validation_ids = {sample.sample_id for sample in validation_dataset.samples}
    checks = {
        "prior_fit_is_outer_train_only": set(prior.fit_sample_ids) == train_ids,
        "prior_fit_excludes_validation": not bool(train_ids & validation_ids),
        "twenty_landmarks_present": candidates.points.shape[1] == 20,
        "three_views_present": candidates.images.shape[2] == 3,
        "all_model_outputs_finite": bool(torch.isfinite(outputs["final"]).all()),
        "geometry_only_fallback_finite": bool(torch.isfinite(fallback["final"]).all()),
        "hard3_disjoint_from_stage4": set(CORE20_INDICES).isdisjoint(HARD3),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "image_shape": list(candidates.images.shape),
        "candidate_shape": list(candidates.points.shape),
        "candidate_oracle_core20_ale": float(
            np.where(candidates.mask, candidates.target_distance, np.inf)
            .min(axis=-1)
            .mean()
        ),
        "spatial_prior": prior.report(),
        "test_labels_consumed": False,
    }
    (output_dir / "preflight_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Core20-MVSC preflight failed: {failed}")
    print("Core20-MVSC preflight passed", flush=True)
    return report
