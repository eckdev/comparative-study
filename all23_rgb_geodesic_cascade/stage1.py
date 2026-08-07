import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .anatomy import HARD3, NUM_LANDMARKS
from .data import (
    GlobalSurfaceDataset,
    collate_surface_graphs,
    fit_feature_normalizer,
)
from .model import GlobalCoarseNetwork
from .train import autocast_context, grad_scaler, move_batch


def _loader(dataset, batch_size, shuffle, args, seed):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(args.num_workers > 0),
        collate_fn=collate_surface_graphs,
        generator=torch.Generator().manual_seed(int(seed)),
    )


def _nearest_vertex_ce(logits, batch, expert):
    terms = []
    landmark_weight = expert.new_ones(NUM_LANDMARKS)
    landmark_weight[list(HARD3)] = 1.5
    for batch_index in range(expert.shape[0]):
        selected = (batch["batch"] == batch_index) & batch["vertex_mask"]
        points = batch["points"][selected].float()
        local_logits = logits[selected].float().transpose(0, 1)
        nearest = torch.argmin(torch.cdist(expert[batch_index].float(), points), dim=1)
        terms.append(
            (F.cross_entropy(local_logits, nearest, reduction="none") * landmark_weight).mean()
        )
    return torch.stack(terms).mean()


def stage1_loss(outputs, batch):
    expert = batch["expert"].float()
    prediction = outputs["coordinates"].float()
    distance = torch.linalg.norm(prediction - expert, dim=-1)
    landmark_weight = distance.new_ones(NUM_LANDMARKS)
    landmark_weight[list(HARD3)] = 1.5
    coordinate = F.smooth_l1_loss(prediction, expert, beta=2.0, reduction="none").mean(dim=-1)
    coordinate = (coordinate * landmark_weight).mean()
    classification = _nearest_vertex_ce(outputs["coarse_logits"], batch, expert)
    log_var = outputs["log_var"].float().clamp(-6.0, 6.0)
    uncertainty = (
        0.5 * torch.exp(-log_var) * distance.pow(2) + 0.5 * log_var
    ).mean()
    clinical = F.softplus((distance - 4.0) / 1.0).mean()
    total = coordinate + 0.1 * classification + 0.01 * uncertainty + 0.03 * clinical
    return total, distance, {
        "total": total,
        "coordinate": coordinate,
        "classification": classification,
        "uncertainty": uncertainty,
        "clinical": clinical,
    }


@torch.no_grad()
def collect_stage1(model, loader, device, args):
    model.eval()
    sample_ids, predictions, errors = [], [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with autocast_context(device, args.mixed_precision, args.amp_dtype):
            outputs = model(batch)
        prediction = outputs["coordinates"].float()
        sample_ids.extend(raw_batch["sample_id"])
        predictions.append(prediction.cpu().numpy())
        if "expert" in batch:
            errors.append(
                torch.linalg.norm(prediction - batch["expert"].float(), dim=-1).cpu().numpy()
            )
    result = {
        "sample_ids": sample_ids,
        "prediction": np.concatenate(predictions, axis=0),
    }
    if errors:
        result["errors"] = np.concatenate(errors, axis=0)
    return result


def _set_warmup_lr(optimizer, base_lr, epoch, warmup_epochs):
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return
    fraction = epoch / max(int(warmup_epochs), 1)
    lr = float(base_lr) * (0.2 + 0.8 * fraction)
    for group in optimizer.param_groups:
        group["lr"] = lr


def _set_fixed_oof_lr(optimizer, base_lr, epoch, total_epochs, warmup_epochs):
    """Keep OOF optimization active, then use a short label-free cosine tail."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        fraction = epoch / max(int(warmup_epochs), 1)
        lr = float(base_lr) * (0.2 + 0.8 * fraction)
    else:
        decay_start = max(int(warmup_epochs) + 1, int(round(total_epochs * 0.8)))
        if epoch <= decay_start:
            lr = float(base_lr)
        else:
            progress = (epoch - decay_start) / max(total_epochs - decay_start, 1)
            minimum = max(float(base_lr) * 0.05, 1e-6)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = minimum + (float(base_lr) - minimum) * cosine
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_stage1_model(
    train_dataset,
    val_dataset,
    normalizer,
    output_dir,
    device,
    args,
    seed,
    cache_signature,
    fixed_epochs=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if fixed_epochs is not None and int(fixed_epochs) < 1:
        raise ValueError("fixed_epochs must be positive")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    train_loader = _loader(train_dataset, args.stage1_batch_size, True, args, seed)
    fixed_mode = fixed_epochs is not None
    total_epochs = int(fixed_epochs) if fixed_mode else int(args.stage1_epochs)
    val_loader = (
        None
        if fixed_mode
        else _loader(val_dataset, args.stage1_eval_batch_size, False, args, seed)
    )
    model = GlobalCoarseNetwork(
        input_dim=len(normalizer["mean"]),
        width=args.stage1_width,
        global_blocks=args.stage1_blocks,
        heads=args.heads,
        dropout=args.dropout,
        coordinate_topk=args.stage1_topk,
        coordinate_temperature=args.stage1_temperature,
    ).to(device)
    checkpoint_path = output_dir / "best_model.pth"
    completion_path = output_dir / "training_complete.json"
    if completion_path.exists() and checkpoint_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("cache_signature") == cache_signature:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            report = dict(completion["report"])
            report["loaded_from_cache"] = True
            score_text = (
                f"best validation ALE={report['best_validation_ale']:.4f}"
                if report.get("best_validation_ale") is not None
                else f"fixed epochs={report['best_epoch']}"
            )
            print(f"Stage 1 model cached: {output_dir.name}; {score_text}", flush=True)
            return model, report
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.stage1_lr, weight_decay=args.weight_decay
    )
    scheduler = (
        None
        if fixed_mode
        else torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.stage1_scheduler_patience,
            min_lr=1e-6,
        )
    )
    scaler = grad_scaler(args.mixed_precision, args.amp_dtype, args.amp_init_scale)
    best_score, best_epoch, stale = float("inf"), 0, 0
    history = []
    started = time.time()
    for epoch in range(1, total_epochs + 1):
        train_dataset.set_epoch(epoch)
        if fixed_mode:
            _set_fixed_oof_lr(
                optimizer,
                args.stage1_lr,
                epoch,
                total_epochs,
                args.stage1_lr_warmup_epochs,
            )
        else:
            _set_warmup_lr(
                optimizer, args.stage1_lr, epoch, args.stage1_lr_warmup_epochs
            )
        model.train()
        totals = {}
        count = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.mixed_precision, args.amp_dtype):
                outputs = model(batch)
            loss, _, components = stage1_loss(outputs, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Stage 1 produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise RuntimeError("Stage 1 produced non-finite gradients")
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(raw_batch["sample_id"])
            count += batch_size
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
        if count == 0:
            raise RuntimeError("Stage 1 completed an epoch without an optimizer update")
        if fixed_mode:
            score = None
            row = {
                "epoch": epoch,
                "training_mode": "fixed_epoch_oof",
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train_{name}": value / count for name, value in totals.items()},
            }
        else:
            validation = collect_stage1(model, val_loader, device, args)
            score = float(validation["errors"].mean())
            if not np.isfinite(score):
                raise RuntimeError("Stage 1 validation ALE is non-finite")
            if epoch >= args.stage1_scheduler_start_epoch:
                scheduler.step(score)
            row = {
                "epoch": epoch,
                "validation_ale": score,
                "validation_median": float(np.median(validation["errors"])),
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train_{name}": value / count for name, value in totals.items()},
            }
        history.append(row)
        if fixed_mode:
            print(
                f"Stage1 OOF epoch {epoch:04d}/{total_epochs} "
                f"train={row['train_total']:.5f}",
                flush=True,
            )
            best_epoch = epoch
            if epoch == total_epochs:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "validation_ale": None,
                        "normalizer": normalizer,
                        "stage1_config": {
                            "width": args.stage1_width,
                            "blocks": args.stage1_blocks,
                            "topk": args.stage1_topk,
                            "training_mode": "fixed_epoch_oof",
                            "lr_schedule": "constant_then_final_20pct_cosine",
                        },
                    },
                    checkpoint_path,
                )
        else:
            print(
                f"Stage1 epoch {epoch:04d}/{total_epochs} "
                f"train={row['train_total']:.5f} val_ALE={score:.4f}",
                flush=True,
            )
        if not fixed_mode and score < best_score - args.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_ale": score,
                    "normalizer": normalizer,
                    "stage1_config": {
                        "width": args.stage1_width,
                        "blocks": args.stage1_blocks,
                        "topk": args.stage1_topk,
                    },
                },
                checkpoint_path,
            )
        elif not fixed_mode and epoch >= args.stage1_min_epochs:
            stale += 1
        elif not fixed_mode:
            stale = 0
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if not fixed_mode and epoch >= args.stage1_min_epochs and stale >= args.stage1_patience:
            print(
                f"Stage1 early stopping at epoch {epoch}; best epoch={best_epoch}",
                flush=True,
            )
            break
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    report = {
        "best_epoch": best_epoch,
        "best_validation_ale": None if fixed_mode else best_score,
        "training_seconds": float(time.time() - started),
        "checkpoint": str(checkpoint_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loaded_from_cache": False,
        "training_mode": "fixed_epoch_oof" if fixed_mode else "validation_early_stop",
        "lr_schedule": (
            "constant_then_final_20pct_cosine"
            if fixed_mode
            else "validation_reduce_on_plateau"
        ),
        "fit_sample_count": len(train_dataset),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    completion_path.write_text(
        json.dumps(
            {
                "complete": True,
                "cache_signature": cache_signature,
                "report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model, report


def _strata(sample_by_id, ids):
    return np.asarray(
        [f"{sample_by_id[sample_id].class_name}_{sample_by_id[sample_id].gender}" for sample_id in ids]
    )


def _inner_partitions(
    sample_by_id,
    train_ids,
    folds,
    val_fraction,
    seed,
    fixed_training=False,
):
    ids = np.asarray(train_ids)
    strata = _strata(sample_by_id, ids)
    counts = Counter(strata.tolist())
    use_stratified = len(counts) > 1 and min(counts.values()) >= int(folds)
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        if use_stratified
        else KFold(n_splits=folds, shuffle=True, random_state=seed)
    )
    outer_iterator = splitter.split(ids, strata) if use_stratified else splitter.split(ids)
    result = []
    for fold_index, (remaining, holdout) in enumerate(outer_iterator, start=1):
        remaining_ids = ids[remaining]
        if fixed_training:
            result.append(
                {
                    "fold": fold_index,
                    "train": remaining_ids.tolist(),
                    "val": [],
                    "holdout": ids[holdout].tolist(),
                }
            )
            continue
        remaining_strata = strata[remaining]
        remaining_counts = Counter(remaining_strata.tolist())
        can_stratify = len(remaining_counts) > 1 and min(remaining_counts.values()) >= 2
        if can_stratify:
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_fraction,
                random_state=seed + fold_index,
            )
            train_local, val_local = next(inner.split(remaining_ids, remaining_strata))
        else:
            rng = np.random.default_rng(seed + fold_index)
            order = rng.permutation(len(remaining_ids))
            val_count = max(1, int(round(len(order) * val_fraction)))
            val_local, train_local = order[:val_count], order[val_count:]
        result.append(
            {
                "fold": fold_index,
                "train": remaining_ids[train_local].tolist(),
                "val": remaining_ids[val_local].tolist(),
                "holdout": ids[holdout].tolist(),
            }
        )
    return result


def _dataset(samples, ids, records, transforms, normalizer, args, training, include_expert, seed):
    return GlobalSurfaceDataset(
        samples=samples,
        sample_ids=ids,
        records=records,
        transforms=transforms,
        normalizer=normalizer,
        training=training,
        include_expert=include_expert,
        rotation_degrees=args.stage1_rotation_degrees if training else 0.0,
        point_noise_mm=args.point_noise_mm if training else 0.0,
        rgb_noise=args.rgb_noise if training else 0.0,
        point_dropout=args.point_dropout if training else 0.0,
        use_rgb=args.use_rgb,
        memory_cache=not args.no_memory_cache,
        seed=seed,
    )


def _predict(model, dataset, device, args, seed):
    loader = _loader(dataset, args.stage1_eval_batch_size, False, args, seed)
    values = collect_stage1(model, loader, device, args)
    return {
        sample_id: values["prediction"][index].astype(np.float32)
        for index, sample_id in enumerate(values["sample_ids"])
    }


def _train_template(ids, records):
    rows = []
    for sample_id in ids:
        with np.load(records[sample_id]) as stored:
            rows.append(stored["landmarks"].astype(np.float32))
    return np.mean(np.stack(rows), axis=0).astype(np.float32)


def _calibrate_template_blend(predictions, validation_ids, template, records):
    expert = []
    predicted = []
    for sample_id in validation_ids:
        with np.load(records[sample_id]) as stored:
            expert.append(stored["landmarks"].astype(np.float32))
        predicted.append(predictions[sample_id])
    expert = np.stack(expert)
    predicted = np.stack(predicted)
    candidates = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        blended = template[None] + alpha * (predicted - template[None])
        ale = float(np.linalg.norm(blended - expert, axis=-1).mean())
        candidates.append({"alpha": alpha, "validation_ale": ale})
    best = min(candidates, key=lambda row: (row["validation_ale"], row["alpha"]))
    return float(best["alpha"]), candidates


def _configured_template_alpha(value):
    if isinstance(value, str) and value.strip().lower() == "auto":
        return None
    alpha = float(value)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--stage1-oof-template-alpha must be auto or a value in [0, 1]")
    return alpha


def _calibrate_oof_template_blend(predictions, templates, train_ids, records):
    """Fit one coarse calibration parameter using outer-train OOF labels only."""
    expert = []
    predicted = []
    template_rows = []
    for sample_id in train_ids:
        with np.load(records[sample_id]) as stored:
            expert.append(stored["landmarks"].astype(np.float32))
        predicted.append(predictions[sample_id])
        template_rows.append(templates[sample_id])
    expert = np.stack(expert)
    predicted = np.stack(predicted)
    template_rows = np.stack(template_rows)
    candidates = []
    for alpha in np.linspace(0.0, 1.0, 41):
        blended = template_rows + float(alpha) * (predicted - template_rows)
        errors = np.linalg.norm(blended - expert, axis=-1)
        candidates.append(
            {
                "alpha": float(alpha),
                "oof_ale": float(errors.mean()),
                "oof_p95": float(np.percentile(errors, 95)),
            }
        )
    best = min(candidates, key=lambda row: (row["oof_ale"], row["oof_p95"]))
    return float(best["alpha"]), candidates


def _apply_template_blend(predictions, template, alpha):
    return {
        sample_id: (template + alpha * (values - template)).astype(np.float32)
        for sample_id, values in predictions.items()
    }


def _apply_sample_template_blend(predictions, templates, alpha):
    return {
        sample_id: (
            templates[sample_id] + alpha * (values - templates[sample_id])
        ).astype(np.float32)
        for sample_id, values in predictions.items()
    }


def _write_predictions(path, predictions, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "landmark", "coarse_x", "coarse_y", "coarse_z", "provenance"],
        )
        writer.writeheader()
        for sample_id in sorted(predictions):
            for landmark in range(NUM_LANDMARKS):
                xyz = predictions[sample_id][landmark]
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "landmark": landmark,
                        "coarse_x": float(xyz[0]),
                        "coarse_y": float(xyz[1]),
                        "coarse_z": float(xyz[2]),
                        "provenance": provenance[sample_id],
                    }
                )


def _read_predictions(path):
    grouped = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(
                row["sample_id"], np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            )
            grouped[row["sample_id"]][int(row["landmark"])] = [
                float(row[f"coarse_{axis}"]) for axis in ("x", "y", "z")
            ]
    return grouped


def _stage1_cache_key(splits, records, transforms, args):
    transform_digest = hashlib.sha1()
    for sample_id in sorted(transforms):
        transform_digest.update(sample_id.encode("utf-8"))
        transform_digest.update(np.asarray(transforms[sample_id], dtype=np.float32).tobytes())
    payload = {
        "pipeline_version": 6,
        "splits": splits,
        "record_files": {
            sample_id: Path(records[sample_id]).name
            for sample_id in sorted(set(splits["train"] + splits["val"] + splits["test"]))
        },
        "transform_digest": transform_digest.hexdigest(),
        "oof_folds": args.stage1_oof_folds,
        "oof_mode": args.stage1_oof_mode,
        "oof_fixed_epochs": args.stage1_oof_fixed_epochs,
        "oof_template_alpha": args.stage1_oof_template_alpha,
        "inner_val_fraction": args.stage1_inner_val_fraction,
        "width": args.stage1_width,
        "blocks": args.stage1_blocks,
        "heads": args.heads,
        "dropout": args.dropout,
        "topk": args.stage1_topk,
        "temperature": args.stage1_temperature,
        "epochs": args.stage1_epochs,
        "min_epochs": args.stage1_min_epochs,
        "patience": args.stage1_patience,
        "scheduler_patience": args.stage1_scheduler_patience,
        "scheduler_start_epoch": args.stage1_scheduler_start_epoch,
        "lr_warmup_epochs": args.stage1_lr_warmup_epochs,
        "lr": args.stage1_lr,
        "batch_size": args.stage1_batch_size,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "min_delta": args.min_delta,
        "rotation_degrees": args.stage1_rotation_degrees,
        "point_noise_mm": args.point_noise_mm,
        "rgb_noise": args.rgb_noise,
        "point_dropout": args.point_dropout,
        "use_rgb": args.use_rgb,
        "mixed_precision": args.mixed_precision,
        "amp_dtype": args.amp_dtype,
        "seed": args.seed,
        "alignment": args.alignment,
        "icp_points": args.icp_points,
        "icp_iterations": args.icp_iterations,
        "atlas_size": args.atlas_size,
        "atlas_iterations": args.atlas_iterations,
        "max_vertices": args.max_vertices,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest(), payload


def _stage1_quality_status(metrics, args):
    signed_gap = (
        metrics["train_oof"]["overall_ale"]
        - metrics["validation"]["overall_ale"]
    )
    gap = abs(signed_gap)
    checks = {
        "validation": metrics["validation"]["overall_ale"] <= args.max_stage1_val_ale,
        "train_oof": metrics["train_oof"]["overall_ale"] <= args.max_stage1_oof_ale,
        "train_oof_p95": metrics["train_oof"]["p95"] <= args.max_stage1_oof_p95,
        "oof_validation_gap": gap <= args.max_stage1_oof_val_gap,
    }
    return {
        **checks,
        "gap_mm": float(gap),
        "signed_gap_mm": float(signed_gap),
        "passed": all(checks.values()),
    }


def _enforce_stage1_quality(metrics, args, cached=False):
    prefix = "Cached Stage 1" if cached else "Stage 1"
    if metrics["validation"]["overall_ale"] > args.max_stage1_val_ale:
        raise RuntimeError(
            f"{prefix} validation ALE={metrics['validation']['overall_ale']:.4f} exceeds "
            f"--max-stage1-val-ale={args.max_stage1_val_ale:.4f}. "
            "Completed predictions remain cached for inspection or a revised gate."
        )
    if metrics["train_oof"]["overall_ale"] > args.max_stage1_oof_ale:
        raise RuntimeError(
            f"{prefix} train OOF ALE={metrics['train_oof']['overall_ale']:.4f} exceeds "
            f"--max-stage1-oof-ale={args.max_stage1_oof_ale:.4f}. "
            "Completed predictions remain cached for inspection or a revised gate."
        )
    if metrics["train_oof"]["p95"] > args.max_stage1_oof_p95:
        raise RuntimeError(
            f"{prefix} train OOF p95={metrics['train_oof']['p95']:.4f} exceeds "
            f"--max-stage1-oof-p95={args.max_stage1_oof_p95:.4f}."
        )
    signed_gap = (
        metrics["train_oof"]["overall_ale"]
        - metrics["validation"]["overall_ale"]
    )
    gap = abs(signed_gap)
    if gap > args.max_stage1_oof_val_gap:
        raise RuntimeError(
            f"{prefix} absolute OOF-validation ALE gap={gap:.4f} "
            f"(signed={signed_gap:.4f}) exceeds "
            f"--max-stage1-oof-val-gap={args.max_stage1_oof_val_gap:.4f}. "
            "Stage 2 would otherwise train on a mismatched coarse-center distribution."
        )


def _print_stage1_summary(metrics, cached=False):
    source = "cached" if cached else "completed"
    signed_gap = (
        metrics["train_oof"]["overall_ale"]
        - metrics["validation"]["overall_ale"]
    )
    print(
        f"Stage 1 {source}: train OOF ALE={metrics['train_oof']['overall_ale']:.4f}, "
        f"validation ALE={metrics['validation']['overall_ale']:.4f}, "
        f"absolute gap={abs(signed_gap):.4f} (signed={signed_gap:.4f})",
        flush=True,
    )


def _prediction_metrics(predictions, ids, records):
    errors = []
    for sample_id in ids:
        with np.load(records[sample_id]) as stored:
            expert = stored["landmarks"].astype(np.float32)
        errors.append(np.linalg.norm(predictions[sample_id] - expert, axis=1))
    matrix = np.stack(errors)
    core = np.delete(matrix, list(HARD3), axis=1)
    hard = matrix[:, list(HARD3)]
    return {
        "overall_ale": float(matrix.mean()),
        "overall_median": float(np.median(matrix)),
        "core20_ale": float(core.mean()),
        "hard3_ale": float(hard.mean()),
        "p95": float(np.percentile(matrix, 95)),
        "max": float(matrix.max()),
        "per_landmark_ale": matrix.mean(axis=0).tolist(),
    }


def generate_oof_stage1_predictions(
    samples,
    splits,
    records,
    transforms,
    outer_normalizer,
    fold_dir,
    device,
    args,
):
    """Return OOF train and outer-train-only validation/test coarse centers."""
    stage1_dir = Path(fold_dir) / "stage1_global_coarse"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    cache_key, config = _stage1_cache_key(splits, records, transforms, args)
    complete_path = stage1_dir / "complete.json"
    prediction_paths = {
        "train": stage1_dir / "oof_predictions_train.csv",
        "val": stage1_dir / "predictions_val.csv",
        "test": stage1_dir / "predictions_test.csv",
    }
    if complete_path.exists() and all(path.exists() for path in prediction_paths.values()):
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("cache_key") == cache_key:
            metrics = complete.get("metrics")
            if not metrics:
                raise RuntimeError(f"Stage 1 cache has no metrics: {complete_path}")
            _print_stage1_summary(metrics, cached=True)
            _enforce_stage1_quality(metrics, args, cached=True)
            merged = {}
            for path in prediction_paths.values():
                merged.update(_read_predictions(path))
            return merged, complete

    sample_by_id = {sample.sample_id: sample for sample in samples}
    partitions = _inner_partitions(
        sample_by_id,
        splits["train"],
        args.stage1_oof_folds,
        args.stage1_inner_val_fraction,
        args.seed,
        fixed_training=args.stage1_oof_mode == "fixed_epoch",
    )
    oof_predictions = {}
    oof_templates = {}
    provenance = {}
    inner_reports = []
    for partition in partitions:
        inner_dir = stage1_dir / f"inner_fold_{partition['fold']}"
        inner_normalizer = fit_feature_normalizer(
            samples,
            records,
            partition["train"],
            inner_dir / "normalization.json",
            seed=args.seed + partition["fold"],
        )
        train_dataset = _dataset(
            samples,
            partition["train"],
            records,
            transforms,
            inner_normalizer,
            args,
            True,
            True,
            args.seed + partition["fold"] * 100,
        )
        val_dataset = (
            None
            if args.stage1_oof_mode == "fixed_epoch"
            else _dataset(
                samples,
                partition["val"],
                records,
                transforms,
                inner_normalizer,
                args,
                False,
                True,
                args.seed,
            )
        )
        model, report = train_stage1_model(
            train_dataset,
            val_dataset,
            inner_normalizer,
            inner_dir,
            device,
            args,
            args.seed + partition["fold"] * 1009,
            f"{cache_key}:inner:{partition['fold']}",
            fixed_epochs=(
                args.stage1_oof_fixed_epochs
                if args.stage1_oof_mode == "fixed_epoch"
                else None
            ),
        )
        holdout_dataset = _dataset(
            samples,
            partition["holdout"],
            records,
            transforms,
            inner_normalizer,
            args,
            False,
            False,
            args.seed,
        )
        template = _train_template(partition["train"], records)
        if args.stage1_oof_mode == "fixed_epoch":
            alpha = _configured_template_alpha(args.stage1_oof_template_alpha)
            candidates = []
            raw_predicted = _predict(
                model, holdout_dataset, device, args, args.seed
            )
            predicted = raw_predicted
            for sample_id in partition["holdout"]:
                oof_templates[sample_id] = template
        else:
            calibration_dataset = _dataset(
                samples,
                partition["val"],
                records,
                transforms,
                inner_normalizer,
                args,
                False,
                False,
                args.seed,
            )
            calibration_prediction = _predict(
                model, calibration_dataset, device, args, args.seed
            )
            alpha, candidates = _calibrate_template_blend(
                calibration_prediction, partition["val"], template, records
            )
            predicted = _apply_template_blend(
                _predict(model, holdout_dataset, device, args, args.seed),
                template,
                alpha,
            )
        overlap = set(oof_predictions) & set(predicted)
        if overlap:
            raise RuntimeError(f"Duplicate Stage 1 OOF predictions: {sorted(overlap)[:5]}")
        oof_predictions.update(predicted)
        for sample_id in predicted:
            provenance[sample_id] = f"inner_fold_{partition['fold']}_out_of_fold"
        inner_reports.append(
            {
                "partition": partition,
                "training": report,
                "template_blend": {"alpha": alpha, "candidates": candidates},
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if set(oof_predictions) != set(splits["train"]):
        missing = set(splits["train"]) - set(oof_predictions)
        raise RuntimeError(f"Stage 1 OOF coverage is incomplete: {sorted(missing)[:5]}")

    oof_blend_report = None
    if args.stage1_oof_mode == "fixed_epoch":
        configured_alpha = _configured_template_alpha(args.stage1_oof_template_alpha)
        if configured_alpha is None:
            oof_alpha, oof_candidates = _calibrate_oof_template_blend(
                oof_predictions,
                oof_templates,
                splits["train"],
                records,
            )
            selection = "outer_train_oof_labels_only"
        else:
            oof_alpha = configured_alpha
            oof_candidates = [
                {
                    "alpha": oof_alpha,
                    "oof_ale": None,
                    "oof_p95": None,
                }
            ]
            selection = "preconfigured_without_prediction_labels"
        oof_predictions = _apply_sample_template_blend(
            oof_predictions, oof_templates, oof_alpha
        )
        oof_blend_report = {
            "alpha": oof_alpha,
            "selection": selection,
            "candidates": oof_candidates,
        }
        for report in inner_reports:
            report["template_blend"] = {
                "alpha": oof_alpha,
                "selection": selection,
                "scope": "shared_across_outer_train_oof_predictions",
            }
        for sample_id in provenance:
            provenance[sample_id] += f"_template_blend_alpha_{oof_alpha:g}"

    final_dir = stage1_dir / "outer_train_model"
    final_train = _dataset(
        samples,
        splits["train"],
        records,
        transforms,
        outer_normalizer,
        args,
        True,
        True,
        args.seed + 90_000,
    )
    final_val = _dataset(
        samples,
        splits["val"],
        records,
        transforms,
        outer_normalizer,
        args,
        False,
        True,
        args.seed,
    )
    final_model, final_report = train_stage1_model(
        final_train,
        final_val,
        outer_normalizer,
        final_dir,
        device,
        args,
        args.seed + 99_991,
        f"{cache_key}:outer_train",
    )
    val_dataset = _dataset(
        samples,
        splits["val"],
        records,
        transforms,
        outer_normalizer,
        args,
        False,
        False,
        args.seed,
    )
    test_dataset = _dataset(
        samples,
        splits["test"],
        records,
        transforms,
        outer_normalizer,
        args,
        False,
        False,
        args.seed,
    )
    val_predictions = _predict(final_model, val_dataset, device, args, args.seed)
    test_predictions = _predict(final_model, test_dataset, device, args, args.seed)
    outer_template = _train_template(splits["train"], records)
    if args.stage1_oof_mode == "fixed_epoch":
        outer_alpha = oof_blend_report["alpha"]
        outer_candidates = [
            {
                "alpha": outer_alpha,
                "validation_ale": None,
                "selection": "matched_to_outer_train_oof_calibration",
            }
        ]
    else:
        outer_alpha, outer_candidates = _calibrate_template_blend(
            val_predictions, splits["val"], outer_template, records
        )
    val_predictions = _apply_template_blend(
        val_predictions, outer_template, outer_alpha
    )
    test_predictions = _apply_template_blend(
        test_predictions, outer_template, outer_alpha
    )
    val_provenance = {sample_id: "outer_train_model_validation" for sample_id in val_predictions}
    test_provenance = {sample_id: "outer_train_model_test_blind" for sample_id in test_predictions}
    _write_predictions(prediction_paths["train"], oof_predictions, provenance)
    _write_predictions(prediction_paths["val"], val_predictions, val_provenance)
    _write_predictions(prediction_paths["test"], test_predictions, test_provenance)
    stage1_metrics = {
        "train_oof": _prediction_metrics(oof_predictions, splits["train"], records),
        "validation": _prediction_metrics(val_predictions, splits["val"], records),
    }
    (stage1_dir / "metrics_train_val.json").write_text(
        json.dumps(stage1_metrics, indent=2), encoding="utf-8"
    )
    quality = _stage1_quality_status(stage1_metrics, args)
    complete = {
        "complete": True,
        "quality_gate": quality,
        "cache_key": cache_key,
        "config": config,
        "inner_models": inner_reports,
        "oof_template_blend": oof_blend_report,
        "outer_train_model": {
            **final_report,
            "template_blend": {
                "alpha": outer_alpha,
                "candidates": outer_candidates,
            },
        },
        "metrics": stage1_metrics,
        "provenance": {
            "train": (
                "fixed-epoch out-of-fold; each sample excluded from model fit; one shared "
                "template blend is calibrated on outer-train OOF labels only, with no outer "
                "validation or test label use"
                if args.stage1_oof_mode == "fixed_epoch"
                else "nested out-of-fold; each sample excluded from model fit and checkpoint validation"
            ),
            "validation": "outer-train model; validation labels used only for checkpoint selection",
            "test": "outer-train model inference without loading test expert landmarks",
        },
        "prediction_files": {key: str(path) for key, path in prediction_paths.items()},
    }
    complete_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    _print_stage1_summary(stage1_metrics)
    _enforce_stage1_quality(stage1_metrics, args)
    merged = dict(oof_predictions)
    merged.update(val_predictions)
    merged.update(test_predictions)
    return merged, complete
