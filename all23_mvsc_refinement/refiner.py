"""Training, cross-fitted gating and validation policy of the all-23 refiner.

Pipeline per outer fold (all fitting uses outer-train participants only):
1. ``fit_refiner`` trains one All23MVSCNet on jittered expert-centred
   candidates (as Core20-MVSC did, generalized to 23 landmarks/four views).
2. ``build_crossfit_evidence`` trains K inner refiners, each on (K-1)/K of the
   outer-train participants, and runs them for 1..max_iterations on the
   remaining participants starting from their genuine Stage-1 OOF centres.
3. ``fit_oof_gate`` (gate.py) learns per-prediction alpha from that evidence.
4. A refiner fitted on all outer-train participants is used for validation and
   test; ``calibrate_policy`` picks the iteration count and per-type alpha caps
   on the outer validation split, after which the policy is locked.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from all23_rgb_geodesic_cascade.anatomy import CORE20, HARD3, NUM_LANDMARKS
from all23_rgb_geodesic_cascade.metrics import bootstrap_delta, summarize, write_csv

from .anatomy import (
    ALL23_INDICES,
    ANATOMICAL_GROUPS,
    LANDMARK_TYPES,
    landmark_type,
)
from .gate import (
    GateEvidence,
    diagnostics_from_iterations,
    fit_oof_gate,
    load_oof_gate,
    predict_gate_alpha,
)
from .model import All23MVSCNet
from .patches import (
    GEOMETRY_FEATURES,
    IMAGE_CHANNELS,
    RGB_GEOMETRY_FEATURES,
    RGB_IMAGE_CHANNELS,
    VIEW_NAMES,
    extract_all23_set,
    group_indices,
    type_indices,
)
from .spatial_prior import ConditionalSpatialPrior, SpatialPriorConfig, _folds_from_strata


VERSION = "All23-MVSC-v1.1"
# Part of every trained-refiner signature. Inference-only changes must not bump
# it, otherwise refiners trained by an earlier run can no longer be reused.
MODEL_VERSION = "All23-MVSC-v1"


@dataclass
class All23MVSCConfig:
    folds: int = 5
    gate_folds: int = 5
    max_iterations: int = 2
    epochs: int = 120
    min_epochs: int = 30
    patience: int = 20
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
    geometry_only_probability: float = 0.10
    heatmap_weight: float = 1.0
    coordinate_weight: float = 1.0
    hard_negative_weight: float = 0.25
    anatomy_weight: float = 0.05
    confidence_weight: float = 0.02
    hard_negative_count: int = 16
    use_images: bool = True
    use_spatial_prior: bool = True
    use_jaw_profile: bool = True
    prior_folds: int = 5
    prior_l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    # Jitter mixture: fraction using the genuine Stage-1 OOF residual, then two
    # Gaussian scales; norms are clipped per landmark type.
    oof_residual_fraction: float = 0.50
    small_noise_fraction: float = 0.35
    corner_noise_mm: tuple[float, float] = (1.5, 3.0)
    wide_noise_mm: tuple[float, float] = (2.5, 5.0)
    corner_clip_mm: float = 6.0
    wide_clip_mm: float = 15.0
    alpha_cap_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    minimum_improvement_probability: float = 0.95
    maximum_type_regression_mm: float = 0.05
    bootstrap_iters: int = 2000
    gate_epochs: int = 300
    gate_patience: int = 30
    # Inference-only settings: averaging the cross-fit inner refiners with the
    # full-fit refiner, and MC-dropout passes that feed the gate's spread signal.
    ensemble: bool = True
    mc_samples: int = 8
    mixed_precision: bool = True
    amp_dtype: str = "bfloat16"
    seed: int = 42


# Hyper-parameters that do not change a fitted network (policy-only settings).
_POLICY_ONLY_FIELDS = (
    "alpha_cap_grid",
    "minimum_improvement_probability",
    "maximum_type_regression_mm",
    "bootstrap_iters",
    "gate_epochs",
    "gate_patience",
    "max_iterations",
    "gate_folds",
    "ensemble",
    "mc_samples",
)
EVIDENCE_VERSION = 2


def config_signature(config, fit_sample_ids=()):
    payload = {
        key: value
        for key, value in asdict(config).items()
        if key not in _POLICY_ONLY_FIELDS
    }
    payload.update(
        {
            "image_channels": list(IMAGE_CHANNELS),
            "geometry_features": len(GEOMETRY_FEATURES),
            "views": list(VIEW_NAMES),
            "fit_sample_ids": sorted(fit_sample_ids),
            "version": MODEL_VERSION,
        }
    )
    encoded = json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def subset_dataset(dataset, sample_ids):
    lookup = {sample.sample_id: sample for sample in dataset.samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in lookup]
    if missing:
        raise KeyError(f"Dataset misses requested samples: {missing[:5]}")
    subset = copy.copy(dataset)
    subset.samples = [lookup[sample_id] for sample_id in sample_ids]
    subset._roi_memory = {}
    subset._record_memory = {}
    return subset


def dataset_shapes(dataset):
    shapes, sample_ids, classes, genders = [], [], [], []
    for sample in dataset.samples:
        with np.load(dataset.records[sample.sample_id]) as record:
            if "landmarks" not in record.files:
                raise RuntimeError(f"Training record has no landmarks: {sample.sample_id}")
            shapes.append(record["landmarks"].astype(np.float32))
        sample_ids.append(sample.sample_id)
        classes.append(sample.class_name)
        genders.append(sample.gender)
    return np.stack(shapes), sample_ids, classes, genders


def _prior_covariance_rows(prior, count):
    row = np.stack(
        [prior.parameters[str(landmark)]["covariance"] for landmark in ALL23_INDICES]
    ).astype(np.float32)
    return np.repeat(row[None], count, axis=0)


def training_centers(dataset, shapes, config):
    """Expert + jitter centres; jitter scale and clipping depend on type."""
    result = {}
    rng = np.random.default_rng(config.seed)
    for index, sample in enumerate(dataset.samples):
        expert = shapes[index]
        oof_coarse = dataset._coarse(sample)
        centers = np.empty_like(expert)
        for landmark in ALL23_INDICES:
            corner = landmark_type(landmark) == "corner"
            small, large = config.corner_noise_mm if corner else config.wide_noise_mm
            clip = config.corner_clip_mm if corner else config.wide_clip_mm
            draw = rng.random()
            if draw < config.oof_residual_fraction:
                noise = oof_coarse[landmark] - expert[landmark]
            elif draw < config.oof_residual_fraction + config.small_noise_fraction:
                noise = rng.normal(0.0, small, size=3)
            else:
                noise = rng.normal(0.0, large, size=3)
            noise = np.asarray(noise, dtype=np.float32)
            length = float(np.linalg.norm(noise))
            if length > clip:
                noise *= clip / length
            centers[landmark] = expert[landmark] + noise
        result[sample.sample_id] = centers.astype(np.float32)
    return result


class FlatAll23Dataset(Dataset):
    """One row per (participant, landmark)."""

    def __init__(self, candidates, sample_indices, config, training=False):
        self.candidates = candidates
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.config = config
        self.training = bool(training)
        self.rows = [
            (int(sample_index), landmark)
            for sample_index in self.sample_indices
            for landmark in ALL23_INDICES
        ]
        self.groups = group_indices()
        self.types = type_indices()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sample, landmark = self.rows[index]
        candidates = self.candidates
        images = torch.from_numpy(
            np.asarray(candidates.images[sample, landmark], dtype=np.float32)
        )
        features = torch.from_numpy(
            np.asarray(candidates.features[sample, landmark], dtype=np.float32)
        )
        mask = np.asarray(candidates.mask[sample, landmark]).copy()
        if self.training:
            if self.config.color_noise > 0:
                images[:, :3] = torch.clamp(
                    images[:, :3] + torch.randn_like(images[:, :3]) * self.config.color_noise,
                    0.0,
                    1.0,
                )
            if torch.rand(()) < self.config.geometry_only_probability:
                images[:, list(RGB_IMAGE_CHANNELS)] = 0.0
                features[:, list(RGB_GEOMETRY_FEATURES)] = 0.0
            if self.config.candidate_dropout > 0:
                rng = np.random.default_rng(
                    self.config.seed * 1009
                    + index
                    + int(torch.randint(0, 1_000_000, ()).item())
                )
                dropped = rng.random(len(mask)) < self.config.candidate_dropout
                closest = int(np.argmin(candidates.target_distance[sample, landmark]))
                dropped[closest] = False
                mask &= ~dropped
        return {
            "images": images,
            "grids": torch.from_numpy(
                np.asarray(candidates.grids[sample, landmark], dtype=np.float32)
            ),
            "points": torch.from_numpy(candidates.points[sample, landmark]),
            "features": features,
            "mask": torch.from_numpy(mask),
            "heatmap_target": torch.from_numpy(candidates.heatmap_target[sample, landmark]),
            "distance": torch.from_numpy(candidates.target_distance[sample, landmark]),
            "expert": torch.from_numpy(candidates.expert[sample, landmark]),
            "base": torch.from_numpy(candidates.base[sample, landmark]),
            "prior_mean": torch.from_numpy(candidates.prior_mean[sample, landmark]),
            "prior_covariance": torch.from_numpy(
                candidates.prior_covariance[sample, landmark]
            ),
            "prior_score": torch.from_numpy(candidates.prior_score[sample, landmark]),
            "anatomy_anchors": torch.from_numpy(
                candidates.anatomy_anchors[sample, landmark]
            ),
            "anatomy_distances": torch.from_numpy(
                candidates.anatomy_distances[sample, landmark]
            ),
            "anatomy_mask": torch.from_numpy(candidates.anatomy_mask[sample, landmark]),
            "landmark": torch.tensor(landmark, dtype=torch.long),
            "group": torch.tensor(self.groups[landmark], dtype=torch.long),
            "type": torch.tensor(self.types[landmark], dtype=torch.long),
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


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _type_scale(batch, device):
    scale = torch.tensor([2.0, 4.0, 5.0], device=device)
    return scale[batch["type"].long()]


def _hard_negative_loss(logits, distance, mask, radius, count, margin=0.5):
    nearest = distance.masked_fill(~mask, torch.inf).argmin(dim=-1)
    positive = torch.gather(logits, 1, nearest[:, None]).squeeze(1)
    negative_mask = mask & (distance > radius[:, None])
    safe = logits.masked_fill(~negative_mask, -torch.inf)
    chosen = min(max(1, int(count)), logits.shape[-1])
    values, indices = torch.topk(safe, chosen, dim=-1)
    valid = torch.gather(negative_mask, 1, indices)
    losses = F.softplus(values - positive[:, None] + float(margin))
    losses = torch.where(valid, losses, torch.zeros_like(losses))
    return (losses.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)).mean()


def compute_loss(outputs, batch, config):
    mask = batch["mask"].bool()
    scale = _type_scale(batch, mask.device)
    target = batch["heatmap_target"].float().masked_fill(~mask, 0.0)
    missing = target.sum(dim=-1, keepdim=True) <= 1e-8
    sigma = (scale * 1.25)[:, None]
    fallback = torch.exp(-batch["distance"].float().square() / (2.0 * sigma.square()))
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
    expert = batch["expert"].float()
    proposal = outputs["proposal"].float()
    coordinate = F.smooth_l1_loss(proposal, expert, beta=1.0)
    hard_negative = _hard_negative_loss(
        outputs["logits"].float(),
        batch["distance"].float(),
        mask,
        1.5 * scale,
        config.hard_negative_count,
    )
    anchor_distance = torch.linalg.norm(
        proposal[:, None] - batch["anatomy_anchors"].float(), dim=-1
    )
    anchor_loss = F.smooth_l1_loss(
        anchor_distance / 10.0,
        batch["anatomy_distances"].float() / 10.0,
        reduction="none",
    )
    anchor_mask = batch["anatomy_mask"].float()
    anatomy = (anchor_loss * anchor_mask).sum() / anchor_mask.sum().clamp_min(1.0)
    error = torch.linalg.norm(proposal - expert, dim=-1)
    normalized_squared_error = (error / scale).square()
    confidence = (
        0.5
        * (
            torch.exp(-outputs["log_variance"].float()) * normalized_squared_error
            + outputs["log_variance"].float()
        ).mean()
    )
    total = (
        config.heatmap_weight * heatmap
        + config.coordinate_weight * coordinate
        + config.hard_negative_weight * hard_negative
        + config.anatomy_weight * anatomy
        + config.confidence_weight * confidence
    )
    components = {
        "heatmap_kl": heatmap,
        "coordinate": coordinate,
        "hard_negative": hard_negative,
        "anatomy": anatomy,
        "confidence_nll": confidence,
    }
    return total, components, error


def inner_split(classes, genders, seed, fraction=0.20):
    rng = np.random.default_rng(seed)
    validation = []
    strata = np.asarray([f"{c}|{g}" for c, g in zip(classes, genders)], dtype=object)
    for value in sorted(set(strata.tolist())):
        rows = np.flatnonzero(strata == value)
        rng.shuffle(rows)
        count = max(1, int(round(fraction * len(rows))))
        validation.extend(rows[:count].tolist())
    validation = np.asarray(sorted(set(validation)), dtype=np.int64)
    train = np.setdiff1d(np.arange(len(strata)), validation)
    if len(train) == 0:
        train, validation = np.arange(len(strata) - 1), np.asarray([len(strata) - 1])
    return train, validation


def build_model(config, image_channels, geometry_dim):
    return All23MVSCNet(
        image_channels=image_channels,
        geometry_dim=geometry_dim,
        width=config.width,
        dropout=config.dropout,
        coordinate_topk=config.coordinate_topk,
        coordinate_temperature=config.coordinate_temperature,
        use_images=config.use_images,
        use_spatial_prior=config.use_spatial_prior,
    )


@torch.no_grad()
def _evaluate(model, loader, device, config):
    model.eval()
    errors, landmarks = [], []
    for batch in loader:
        batch = _move(batch, device)
        with _autocast(config, device):
            outputs = model(batch)
        errors.append(
            torch.linalg.norm(outputs["proposal"].float() - batch["expert"].float(), dim=-1)
            .cpu()
            .numpy()
        )
        landmarks.append(batch["landmark"].cpu().numpy())
    errors = np.concatenate(errors)
    landmarks = np.concatenate(landmarks)
    per_landmark = [float(errors[landmarks == index].mean()) for index in ALL23_INDICES]
    return float(errors.mean()), per_landmark


def fit_network(candidates, output_dir, config, device, signature):
    """Train with early stopping on an inner split of the fitting participants."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    train_index, validation_index = inner_split(
        candidates.classes, candidates.genders, config.seed
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        FlatAll23Dataset(candidates, train_index, config, training=True),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        FlatAll23Dataset(candidates, validation_index, config, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    image_channels = candidates.images.shape[3]
    geometry_dim = candidates.features.shape[-1]
    model = build_model(config, image_channels, geometry_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, config.patience // 4), min_lr=1e-6
    )
    scaler = _grad_scaler(
        bool(config.mixed_precision and device.type == "cuda" and config.amp_dtype == "float16")
    )
    checkpoint_path = output_dir / "best_model.pth"
    last_checkpoint_path = output_dir / "last_model.pth"
    history_path = output_dir / "history.json"
    best_ale, best_epoch, stale = float("inf"), 0, 0
    history = []
    start_epoch = 1
    previous_seconds = 0.0
    if last_checkpoint_path.exists():
        resume = torch_load(last_checkpoint_path, device)
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
            previous_seconds = float(resume.get("training_seconds", 0.0))
            start_epoch = int(resume["epoch"]) + 1
            generator.set_state(resume["loader_generator_state"])
            torch.set_rng_state(resume["torch_rng_state"].cpu())
            if device.type == "cuda" and resume.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
            print(f"Resuming All23-MVSC after epoch {start_epoch - 1}", flush=True)
    started = time.time()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        totals, component_rows = [], {}
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                outputs = model(batch)
                loss, components, _ = compute_loss(outputs, batch, config)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite All23-MVSC loss at epoch={epoch} batch={batch_index}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"Non-finite All23-MVSC gradient at epoch={epoch} batch={batch_index}"
                )
            scaler.step(optimizer)
            scaler.update()
            totals.append(float(loss.detach().cpu()))
            for name, value in components.items():
                component_rows.setdefault(name, []).append(float(value.detach().cpu()))
        validation_ale, validation_landmarks = _evaluate(
            model, validation_loader, device, config
        )
        scheduler.step(validation_ale)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(totals)),
            "inner_validation_ale": validation_ale,
            "inner_validation_core20_ale": float(
                np.mean([validation_landmarks[index] for index in CORE20])
            ),
            "inner_validation_hard3_ale": float(
                np.mean([validation_landmarks[index] for index in HARD3])
            ),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "components": {
                name: float(np.mean(values)) for name, values in component_rows.items()
            },
            "inner_validation_landmark_ale": validation_landmarks,
        }
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"All23-MVSC epoch {epoch:03d}/{config.epochs} train={row['train_loss']:.5f} "
            f"inner_val_ALE={validation_ale:.4f} core20={row['inner_validation_core20_ale']:.4f} "
            f"hard3={row['inner_validation_hard3_ale']:.4f}",
            flush=True,
        )
        if validation_ale < best_ale - 1e-5:
            best_ale, best_epoch, stale = validation_ale, epoch, 0
            _atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "image_channels": image_channels,
                    "geometry_dim": geometry_dim,
                    "config": asdict(config),
                    "signature": signature,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        stop = epoch >= config.min_epochs and stale >= config.patience
        _atomic_torch_save(
            {
                "epoch": config.epochs if stop else epoch,
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
                "training_seconds": previous_seconds + float(time.time() - started),
                "signature": signature,
            },
            last_checkpoint_path,
        )
        if stop:
            print(f"All23-MVSC early stopping at epoch {epoch}; best={best_epoch}", flush=True)
            break
    checkpoint = torch_load(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"])
    report = {
        "version": VERSION,
        "best_epoch": best_epoch,
        "best_inner_validation_ale": best_ale,
        "training_seconds": previous_seconds + float(time.time() - started),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "image_channels": list(IMAGE_CHANNELS),
        "geometry_features": list(GEOMETRY_FEATURES),
        "views": list(VIEW_NAMES),
        "inner_train_sample_ids": [candidates.sample_ids[index] for index in train_index],
        "inner_validation_sample_ids": [
            candidates.sample_ids[index] for index in validation_index
        ],
        "signature": signature,
        "training_complete": True,
    }
    return model, report


def _mc_dropout_spread(model, loader, config, device, samples):
    """RMS deviation (mm) of MC-dropout proposals around their mean.

    The network has no batch normalization (GroupNorm/LayerNorm only), so
    train mode changes nothing except the dropout layers.
    """
    passes = []
    devices = [device] if device.type == "cuda" else []
    # A fixed, isolated seed keeps the gate inputs (and hence the locked
    # validation predictions) reproducible without touching the global RNG.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 2027)
        model.train()
        try:
            for _ in range(int(samples)):
                rows = []
                for batch in loader:
                    batch = _move(batch, device)
                    with _autocast(config, device):
                        rows.append(model(batch)["proposal"].float().cpu().numpy())
                passes.append(np.concatenate(rows, axis=0))
        finally:
            model.eval()
    stacked = np.stack(passes)
    deviation = np.linalg.norm(stacked - stacked.mean(axis=0, keepdims=True), axis=-1)
    return np.sqrt(np.mean(deviation**2, axis=0))


@torch.no_grad()
def predict_candidates(model, candidates, config, device, mc_samples=0):
    loader = DataLoader(
        FlatAll23Dataset(candidates, np.arange(len(candidates)), config, training=False),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    names = (
        "proposal",
        "confidence",
        "entropy",
        "margin",
        "prior_disagreement",
        "log_variance",
        "view_weights",
    )
    rows = {name: [] for name in names}
    model.eval()
    for batch in loader:
        batch = _move(batch, device)
        with _autocast(config, device):
            outputs = model(batch)
        for name in names:
            rows[name].append(outputs[name].float().cpu().numpy())
    samples = len(candidates)
    stacked = {name: np.concatenate(values, axis=0) for name, values in rows.items()}
    stacked["proposal"] = stacked["proposal"].reshape(samples, NUM_LANDMARKS, 3)
    for name in ("confidence", "entropy", "margin", "prior_disagreement", "log_variance"):
        stacked[name] = stacked[name].reshape(samples, NUM_LANDMARKS)
    stacked["view_weights"] = stacked["view_weights"].reshape(
        samples, NUM_LANDMARKS, len(VIEW_NAMES)
    )
    if mc_samples and mc_samples > 1:
        stacked["mc_spread"] = _mc_dropout_spread(
            model, loader, config, device, mc_samples
        ).reshape(samples, NUM_LANDMARKS)
    if candidates.has_expert:
        stacked["oracle"] = np.where(
            candidates.mask, candidates.target_distance, np.inf
        ).min(axis=-1)
    return stacked


_AVERAGED_OUTPUTS = ("confidence", "entropy", "margin", "prior_disagreement", "log_variance")


def predict_members(models, candidates, config, device, mc_samples=0):
    """Average member proposals; MC-dropout spread comes from the first member.

    The gate is fitted on single-refiner evidence, so its spread signal is
    always that of one network (the full-fit refiner at inference).
    """
    member_outputs = [
        predict_candidates(
            model, candidates, config, device, mc_samples=mc_samples if index == 0 else 0
        )
        for index, model in enumerate(models)
    ]
    proposals = np.stack([output["proposal"] for output in member_outputs])
    result = {
        "proposal": proposals.mean(axis=0),
        "ensemble_spread": np.sqrt(
            np.mean(
                np.linalg.norm(proposals - proposals.mean(axis=0, keepdims=True), axis=-1)
                ** 2,
                axis=0,
            )
        ),
        "members": len(models),
    }
    for name in _AVERAGED_OUTPUTS:
        result[name] = np.mean([output[name] for output in member_outputs], axis=0)
    result["view_weights"] = member_outputs[0]["view_weights"]
    for name in ("mc_spread", "oracle"):
        if name in member_outputs[0]:
            result[name] = member_outputs[0][name]
    return result


class RefinerEnsemble:
    """One or more trained networks sharing one spatial prior and candidate set."""

    def __init__(self, models, prior, config, device):
        self.models = list(models)
        self.prior = prior
        self.config = config
        self.device = device

    def iterate(
        self,
        dataset,
        base_by_id,
        iterations,
        scratch_dir=None,
        cache_dir=None,
        label="All23-MVSC inference",
        mc_samples=0,
    ):
        """Run 1..iterations proposals, re-centring the ROI on each proposal."""
        sample_ids = [sample.sample_id for sample in dataset.samples]
        classes = [sample.class_name for sample in dataset.samples]
        genders = [sample.gender for sample in dataset.samples]
        centers = np.stack([base_by_id[sample_id] for sample_id in sample_ids]).astype(
            np.float32
        )
        results = []
        for iteration in range(1, int(iterations) + 1):
            prior_mean, prior_covariance = self.prior.predict(centers, classes, genders)
            storage = (
                Path(scratch_dir) / f"{label.replace(' ', '_')}_iter{iteration}"
                if scratch_dir
                else None
            )
            candidates = extract_all23_set(
                dataset,
                {sample_id: centers[index] for index, sample_id in enumerate(sample_ids)},
                {sample_id: prior_mean[index] for index, sample_id in enumerate(sample_ids)},
                {
                    sample_id: prior_covariance[index]
                    for index, sample_id in enumerate(sample_ids)
                },
                image_size=self.config.image_size,
                use_jaw_profile=self.config.use_jaw_profile,
                cache_dir=cache_dir,
                storage_dir=storage,
                label=f"{label} iteration {iteration}",
            )
            prediction = predict_members(
                self.models, candidates, self.config, self.device, mc_samples
            )
            prediction["center"] = centers.copy()
            results.append(prediction)
            centers = prediction["proposal"].astype(np.float32)
            del candidates
            if storage is not None:
                shutil.rmtree(storage, ignore_errors=True)
        return sample_ids, results


class FittedRefiner:
    """A trained network together with the spatial prior of its fit set."""

    def __init__(self, model, prior, config, report, device):
        self.model = model
        self.prior = prior
        self.config = config
        self.report = report
        self.device = device

    def iterate(self, dataset, base_by_id, iterations, **kwargs):
        return RefinerEnsemble([self.model], self.prior, self.config, self.device).iterate(
            dataset, base_by_id, iterations, **kwargs
        )


def fit_refiner(
    dataset,
    output_dir,
    config,
    device,
    scratch_dir=None,
    label="All23-MVSC",
    require_cached=False,
):
    """Fit (or load) one refiner on all participants of ``dataset``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shapes, sample_ids, classes, genders = dataset_shapes(dataset)
    signature = config_signature(config, sample_ids)
    checkpoint_path = output_dir / "best_model.pth"
    prior_path = output_dir / "spatial_prior.json"
    report_path = output_dir / "training_report.json"
    if checkpoint_path.exists() and prior_path.exists() and report_path.exists():
        checkpoint = torch_load(checkpoint_path, device)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("signature") == signature
            and report.get("signature") == signature
            and report.get("training_complete", False)
        ):
            model = build_model(
                config, checkpoint["image_channels"], checkpoint["geometry_dim"]
            ).to(device)
            model.load_state_dict(checkpoint["model"])
            print(f"{label} cached: {output_dir}", flush=True)
            return FittedRefiner(
                model, ConditionalSpatialPrior.load(prior_path), config, report, device
            )

    if require_cached:
        raise RuntimeError(
            f"{label}: no compatible trained refiner in {output_dir}; "
            "the source run must have been trained with the same configuration"
        )
    prior = ConditionalSpatialPrior(
        SpatialPriorConfig(
            folds=config.prior_folds, l2_grid=tuple(config.prior_l2_grid), seed=config.seed
        )
    ).fit(shapes, sample_ids, classes, genders)
    prior.save(prior_path)
    centers = training_centers(dataset, shapes, config)
    context = np.stack([centers[sample_id] for sample_id in sample_ids])
    prior_means = prior.oof_predict_from_shapes(context, sample_ids, classes, genders)
    prior_covariances = _prior_covariance_rows(prior, len(sample_ids))
    storage = Path(scratch_dir) / f"{output_dir.name}_train_candidates" if scratch_dir else None
    candidates = extract_all23_set(
        dataset,
        centers,
        {sample_id: prior_means[index] for index, sample_id in enumerate(sample_ids)},
        {sample_id: prior_covariances[index] for index, sample_id in enumerate(sample_ids)},
        image_size=config.image_size,
        use_jaw_profile=config.use_jaw_profile,
        cache_dir=roi_cache_dir(scratch_dir, output_dir.parent / "roi_cache"),
        storage_dir=storage,
        label=f"{label} training patches",
    )
    oracle = np.where(candidates.mask, candidates.target_distance, np.inf).min(axis=-1)
    model, report = fit_network(candidates, output_dir, config, device, signature)
    report["fit_sample_ids"] = list(sample_ids)
    report["training_candidate_oracle"] = {
        "all23": float(oracle.mean()),
        "core20": float(oracle[:, list(CORE20)].mean()),
        "hard3": float(oracle[:, list(HARD3)].mean()),
    }
    report["spatial_prior"] = prior.report()
    report["jitter"] = {
        "oof_residual_fraction": config.oof_residual_fraction,
        "small_noise_fraction": config.small_noise_fraction,
        "corner_noise_mm": list(config.corner_noise_mm),
        "wide_noise_mm": list(config.wide_noise_mm),
        "corner_clip_mm": config.corner_clip_mm,
        "wide_clip_mm": config.wide_clip_mm,
    }
    report["uses_outer_validation_labels"] = False
    report["uses_test_labels"] = False
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    del candidates
    if storage is not None:
        shutil.rmtree(storage, ignore_errors=True)
    return FittedRefiner(model, prior, config, report, device)


def roi_cache_dir(scratch_dir, fallback):
    """ROI caches are rebuilt cheaply; keep them on local scratch when available."""
    return Path(scratch_dir) / "roi_cache" if scratch_dir else Path(fallback)


def _expert_by_id(dataset):
    shapes, sample_ids, _, _ = dataset_shapes(dataset)
    return {sample_id: shapes[index] for index, sample_id in enumerate(sample_ids)}


def _incoming_transform(fit_dataset, prior_config):
    """Shape prior refitted on the inner-fit participants with the outer config.

    Gate evidence starts from Stage-1 OOF centres; applying the same shape-prior
    step as the outer pipeline brings that incoming estimate closer to the one
    the gate sees at inference. No held-out label is used.
    """
    if not prior_config:
        return None
    from .shape_prior import ProcrustesShapePrior

    shapes, sample_ids, _, _ = dataset_shapes(fit_dataset)
    prior = ProcrustesShapePrior().fit(shapes, sample_ids)
    prior.config = {
        key: prior_config[key] for key in ("components", "l2", "alpha_core", "alpha_hard")
    }
    return prior


def evidence_signature(config, sample_ids, prior_config):
    payload = {
        "refiner": config_signature(config, sample_ids),
        "evidence_version": EVIDENCE_VERSION,
        "gate_folds": config.gate_folds,
        "max_iterations": config.max_iterations,
        "mc_samples": config.mc_samples,
        "incoming_prior": prior_config or None,
    }
    encoded = json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_crossfit_evidence(
    train_dataset,
    output_dir,
    config,
    device,
    scratch_dir=None,
    model_dir=None,
    incoming_prior_config=None,
    require_cached_models=False,
):
    """Out-of-fold refiner evidence for every outer-train participant.

    Each inner refiner is trained without its held-out participants, and those
    participants are refined starting from their Stage-1 OOF centres (produced
    by Stage-1 models that never saw them), optionally passed through the shape
    prior. Inner refiners are read from ``model_dir`` (defaults to output_dir).
    Returns ({iterations: GateEvidence}, [inner FittedRefiner]).
    """
    output_dir = Path(output_dir)
    model_dir = Path(model_dir) if model_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_paths = {
        iterations: output_dir / f"evidence_iter{iterations}.npz"
        for iterations in range(1, config.max_iterations + 1)
    }
    sample_ids = [sample.sample_id for sample in train_dataset.samples]
    manifest_path = output_dir / "crossfit_manifest.json"
    signature = evidence_signature(config, sample_ids, incoming_prior_config)
    strata = [f"{sample.class_name}|{sample.gender}" for sample in train_dataset.samples]
    folds = _folds_from_strata(strata, config.gate_folds, config.seed + 7)

    def inner_refiner(fold_number, fit_ids):
        return fit_refiner(
            subset_dataset(train_dataset, fit_ids),
            model_dir / f"inner_fold_{fold_number}",
            config,
            device,
            scratch_dir=scratch_dir,
            label=f"All23-MVSC inner fold {fold_number}",
            require_cached=require_cached_models,
        )

    split_rows = []
    for fold_number, held_out in enumerate(folds, start=1):
        held_ids = [sample_ids[index] for index in held_out]
        fit_ids = [sample_id for sample_id in sample_ids if sample_id not in set(held_ids)]
        if set(held_ids) & set(fit_ids):
            raise AssertionError("Cross-fit evidence participants leaked into the fit set")
        split_rows.append((fold_number, held_ids, fit_ids))

    if manifest_path.exists() and all(path.exists() for path in evidence_paths.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("signature") == signature:
            print("All23-MVSC cross-fit evidence cached", flush=True)
            refiners = [inner_refiner(number, fit_ids) for number, _, fit_ids in split_rows]
            return (
                {iterations: GateEvidence.load(path) for iterations, path in evidence_paths.items()},
                refiners,
            )

    expert = _expert_by_id(train_dataset)
    stage1 = {
        sample.sample_id: train_dataset._coarse(sample) for sample in train_dataset.samples
    }
    collected = {iterations: [] for iterations in evidence_paths}
    fold_rows, refiners = [], []
    for fold_number, held_ids, fit_ids in split_rows:
        refiner = inner_refiner(fold_number, fit_ids)
        refiners.append(refiner)
        held_dataset = subset_dataset(train_dataset, held_ids)
        incoming = {sample_id: stage1[sample_id] for sample_id in held_ids}
        transform = _incoming_transform(
            subset_dataset(train_dataset, fit_ids), incoming_prior_config
        )
        if transform is not None:
            transformed = transform.transform(np.stack([incoming[key] for key in held_ids]))
            incoming = {key: transformed[index] for index, key in enumerate(held_ids)}
        ordered_ids, results = refiner.iterate(
            held_dataset,
            incoming,
            config.max_iterations,
            scratch_dir=scratch_dir,
            cache_dir=roi_cache_dir(scratch_dir, output_dir / "roi_cache"),
            label=f"crossfit fold {fold_number}",
            mc_samples=config.mc_samples,
        )
        base = np.stack([incoming[sample_id] for sample_id in ordered_ids])
        experts = np.stack([expert[sample_id] for sample_id in ordered_ids])
        for iterations in evidence_paths:
            used = results[:iterations]
            collected[iterations].append(
                GateEvidence(
                    sample_ids=list(ordered_ids),
                    base=base,
                    proposal=used[-1]["proposal"],
                    diagnostics=diagnostics_from_iterations(base, used),
                    expert=experts,
                )
            )
        fold_rows.append(
            {
                "fold": fold_number,
                "fit_sample_ids": fit_ids,
                "evidence_sample_ids": list(ordered_ids),
                "best_epoch": refiner.report.get("best_epoch"),
                "proposal_ale_by_iteration": [
                    float(np.linalg.norm(row["proposal"] - experts, axis=-1).mean())
                    for row in results
                ],
                "base_ale": float(np.linalg.norm(base - experts, axis=-1).mean()),
            }
        )
        print(
            f"Cross-fit fold {fold_number}: base={fold_rows[-1]['base_ale']:.4f} "
            f"proposals={[round(value, 4) for value in fold_rows[-1]['proposal_ale_by_iteration']]}",
            flush=True,
        )
    evidence = {}
    for iterations, rows in collected.items():
        merged = GateEvidence.concatenate(rows)
        if sorted(merged.sample_ids) != sorted(sample_ids):
            raise RuntimeError("Cross-fit evidence does not cover every outer-train participant")
        merged.save(evidence_paths[iterations])
        evidence[iterations] = merged
    manifest_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "gate_folds": config.gate_folds,
                "model_dir": str(model_dir),
                "base_source": (
                    "stage1_nested_oof_centres_plus_inner_shape_prior"
                    if incoming_prior_config
                    else "stage1_nested_oof_centres"
                ),
                "incoming_prior_config": incoming_prior_config or None,
                "mc_samples": config.mc_samples,
                "folds": fold_rows,
                "uses_outer_validation_labels": False,
                "uses_test_labels": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return evidence, refiners


class All23Refinement:
    """Refiner ensemble plus one OOF gate per iteration count."""

    def __init__(self, ensemble, gates, reports, config, device):
        self.ensemble = ensemble
        self.gates = gates
        self.reports = reports
        self.config = config
        self.device = device

    def predict(self, dataset, base_by_id, scratch_dir=None, cache_dir=None, label="All23-MVSC"):
        sample_ids, results = self.ensemble.iterate(
            dataset,
            base_by_id,
            self.config.max_iterations,
            scratch_dir=scratch_dir,
            cache_dir=cache_dir,
            label=label,
            mc_samples=self.config.mc_samples,
        )
        base = np.stack([base_by_id[sample_id] for sample_id in sample_ids]).astype(np.float32)
        by_iterations = {}
        for iterations in range(1, self.config.max_iterations + 1):
            used = results[:iterations]
            diagnostics = diagnostics_from_iterations(base, used)
            model, mean, std = self.gates[iterations]
            by_iterations[iterations] = {
                "proposal": used[-1]["proposal"],
                "gate_alpha": predict_gate_alpha(model, mean, std, diagnostics, self.device),
                "diagnostics": diagnostics,
                "confidence": used[-1]["confidence"],
                "entropy": used[-1]["entropy"],
                "ensemble_spread": used[-1]["ensemble_spread"],
                "mc_spread": used[-1].get(
                    "mc_spread", np.zeros_like(used[-1]["entropy"])
                ),
                "oracle": used[-1].get("oracle"),
            }
        return {"sample_ids": sample_ids, "base": base, "iterations": by_iterations}


def fit_or_load_all23_refinement(
    train_dataset,
    output_dir,
    config,
    device,
    scratch_dir=None,
    model_dir=None,
    incoming_prior_config=None,
):
    """Cross-fit evidence, OOF gates and the inference ensemble of one fold.

    ``model_dir`` points at an earlier run whose refiners are reused unchanged
    (no training happens then); evidence and gates are rebuilt in output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reuse = model_dir is not None and Path(model_dir) != output_dir
    model_root = Path(model_dir) if model_dir else output_dir
    evidence, inner_refiners = build_crossfit_evidence(
        train_dataset,
        output_dir / "crossfit",
        config,
        device,
        scratch_dir=scratch_dir,
        model_dir=model_root / "crossfit",
        incoming_prior_config=incoming_prior_config,
        require_cached_models=reuse,
    )
    gates, reports = {}, {"gates": {}}
    for iterations, rows in evidence.items():
        gate_dir = output_dir / f"gate_iter{iterations}"
        gate_path = gate_dir / "oof_gate.pth"
        report_path = gate_dir / "gate_report.json"
        signature = (
            evidence_signature(config, rows.sample_ids, incoming_prior_config)
            + f"|gate{iterations}"
        )
        cached = False
        if gate_path.exists() and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("signature") == signature:
                gates[iterations] = load_oof_gate(gate_path, device)
                cached = True
        if not cached:
            model, mean, std, report = fit_oof_gate(
                rows,
                gate_dir,
                device,
                seed=config.seed,
                epochs=config.gate_epochs,
                patience=config.gate_patience,
            )
            report["signature"] = signature
            alpha = predict_gate_alpha(model, mean, std, rows.diagnostics, device)
            gated = rows.base + alpha[..., None] * (rows.proposal - rows.base)
            report["oof_evidence_ale"] = {
                "base": float(np.linalg.norm(rows.base - rows.expert, axis=-1).mean()),
                "proposal": float(np.linalg.norm(rows.proposal - rows.expert, axis=-1).mean()),
                "gated_in_sample": float(np.linalg.norm(gated - rows.expert, axis=-1).mean()),
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            gates[iterations] = (model, mean, std)
        reports["gates"][iterations] = json.loads(report_path.read_text(encoding="utf-8"))
    full = fit_refiner(
        train_dataset,
        model_root / "full_model",
        config,
        device,
        scratch_dir=scratch_dir,
        label="All23-MVSC full outer-train",
        require_cached=reuse,
    )
    reports["full_model"] = full.report
    # The full-fit refiner leads: its prior renders the candidates and its
    # MC-dropout spread feeds the gate. Inner refiners never saw val/test.
    members = [full.model] + (
        [refiner.model for refiner in inner_refiners] if config.ensemble else []
    )
    reports["ensemble_members"] = len(members)
    reports["model_dir"] = str(model_root)
    ensemble = RefinerEnsemble(members, full.prior, config, device)
    return All23Refinement(ensemble, gates, reports, config, device)


def _type_columns():
    return {name: list(landmarks) for name, landmarks in LANDMARK_TYPES.items()}


def _group_rows(errors):
    rows = []
    for name, landmarks in {**LANDMARK_TYPES, **ANATOMICAL_GROUPS}.items():
        rows.append(
            {
                "set": name,
                "landmarks": ",".join(map(str, landmarks)),
                **summarize(errors[:, list(landmarks)]),
            }
        )
    return rows


def _policy_iterations(policy):
    """Per-type iteration counts; v1 policies stored one shared count."""
    iterations = policy["iterations"]
    if isinstance(iterations, dict):
        return {name: int(value) for name, value in iterations.items()}
    return {name: int(iterations) for name in LANDMARK_TYPES}


_COMPOSED_FIELDS = ("gate_alpha", "confidence", "entropy", "ensemble_spread", "mc_spread")


def compose_policy(result, iterations, caps):
    base = result["base"]
    final = base.copy()
    alpha = np.zeros(base.shape[:2], dtype=np.float32)
    fields = {name: np.zeros(base.shape[:2], dtype=np.float32) for name in _COMPOSED_FIELDS}
    proposal = base.copy()
    for name, columns in _type_columns().items():
        values = result["iterations"][int(iterations[name])]
        alpha[:, columns] = float(caps[name]) * values["gate_alpha"][:, columns]
        proposal[:, columns] = values["proposal"][:, columns]
        final[:, columns] = base[:, columns] + alpha[:, columns, None] * (
            values["proposal"][:, columns] - base[:, columns]
        )
        for field in _COMPOSED_FIELDS:
            if field in values:
                fields[field][:, columns] = values[field][:, columns]
    return final, alpha, proposal, fields



def calibrate_policy(result, expert, config):
    """Choose iteration count and alpha cap per landmark type on outer validation."""
    base = result["base"]
    base_error = np.linalg.norm(base - expert, axis=-1)
    sweep, iterations, caps = [], {}, {}
    for name, columns in _type_columns().items():
        rows = []
        for count, values in result["iterations"].items():
            for cap in config.alpha_cap_grid:
                alpha = float(cap) * values["gate_alpha"][:, columns]
                final = base[:, columns] + alpha[..., None] * (
                    values["proposal"][:, columns] - base[:, columns]
                )
                error = np.linalg.norm(final - expert[:, columns], axis=-1)
                row = {
                    "iterations": int(count),
                    "type": name,
                    "alpha_cap": float(cap),
                    **summarize(error),
                }
                rows.append(row)
                sweep.append(row)
        best = min(rows, key=lambda row: (row["ale"], row["alpha_cap"], row["iterations"]))
        iterations[name] = int(best["iterations"])
        caps[name] = float(best["alpha_cap"])
    final, _, proposal, _ = compose_policy(result, iterations, caps)
    final_error = np.linalg.norm(final - expert, axis=-1)
    type_regression = {
        name: float(final_error[:, columns].mean() - base_error[:, columns].mean())
        for name, columns in _type_columns().items()
    }
    bootstrap = bootstrap_delta(base_error, final_error, config.bootstrap_iters, config.seed)
    accepted = bool(
        final_error.mean() < base_error.mean()
        and bootstrap["probability_improved"] >= config.minimum_improvement_probability
        and max(type_regression.values()) <= config.maximum_type_regression_mm
    )
    ungated_error = np.linalg.norm(proposal - expert, axis=-1)
    return {
        "version": VERSION,
        "accepted": accepted,
        "iterations": iterations,
        "alpha_caps": caps if accepted else {name: 0.0 for name in LANDMARK_TYPES},
        "proposed_alpha_caps": caps,
        "base": summarize(base_error),
        "ungated_proposal": summarize(ungated_error),
        "selected": summarize(final_error if accepted else base_error),
        "proposed": summarize(final_error),
        "type_regression_mm": type_regression,
        "bootstrap_vs_base": bootstrap,
        "base_sets": _group_rows(base_error),
        "proposed_sets": _group_rows(final_error),
        "validation_labels_used_for_policy_only": True,
        "test_labels_used": False,
        "sweep": sweep,
    }


def apply_policy(result, policy):
    iterations = _policy_iterations(policy)
    final, alpha, proposal, fields = compose_policy(result, iterations, policy["alpha_caps"])
    return {
        "sample_ids": list(result["sample_ids"]),
        "base": result["base"],
        "proposal": proposal,
        "prediction": final.astype(np.float32),
        "effective_alpha": alpha.astype(np.float32),
        **fields,
        "iteration_proposals": {
            key: value["proposal"] for key, value in result["iterations"].items()
        },
    }


def write_policy(output_dir, policy):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "all23_policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    write_csv(output_dir / "all23_policy_sweep.csv", policy["sweep"])
    write_csv(output_dir / "all23_policy_sets_val.csv", policy["proposed_sets"])
