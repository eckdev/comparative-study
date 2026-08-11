import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .anatomy import CORE20, HARD3, NUM_LANDMARKS, mirror_permutation
from .losses import compute_loss, sample_specific_gate_loss


def amp_torch_dtype(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def autocast_context(device, enabled, amp_dtype):
    dtype = amp_torch_dtype(amp_dtype) if enabled else torch.float16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled and device.type == "cuda",
    )


def grad_scaler(enabled, amp_dtype, init_scale):
    # BF16 has FP32-like exponent range and does not need dynamic loss scaling.
    scaler_enabled = bool(enabled and amp_dtype == "float16")
    try:
        return torch.amp.GradScaler("cuda", enabled=scaler_enabled, init_scale=float(init_scale))
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=scaler_enabled, init_scale=float(init_scale))


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_epoch(model, loader, optimizer, scaler, device, args, loss_weights):
    model.train()
    running = {}
    count = 0
    landmark_sum = np.zeros(NUM_LANDMARKS, dtype=np.float64)
    landmark_count = 0
    skipped_nonfinite = 0
    amp_overflows = 0
    total_batches = 0
    for batch in tqdm(loader, desc="train", leave=False, disable=args.no_tqdm):
        total_batches += 1
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.mixed_precision, args.amp_dtype):
            outputs = model(batch, coordinate_mode=args.coordinate_mode)
        if getattr(args, "separate_gate_training", False):
            outputs["final_coordinates"] = outputs["refined_coordinates"]
        elif (
            getattr(model, "use_refinement_gate", False)
            and getattr(loader.dataset, "epoch", 0) <= getattr(args, "gate_warmup_epochs", 0)
        ):
            # Let the local heatmap/refiner learn before alpha can attenuate its
            # coordinate gradient. The gate loss still trains in parallel.
            outputs["final_coordinates"] = outputs["refined_coordinates"]
        loss, errors, components = compute_loss(
            outputs,
            batch,
            loss_weights,
            args.region_positive_weight,
            hard_rank_mode=getattr(args, "hard_rank_mode", "nearest_ce"),
            hard_rank_sigma_lm0=getattr(args, "hard_rank_sigma_lm0", 3.0),
            hard_rank_sigma_gonion=getattr(args, "hard_rank_sigma_gonion", 4.0),
            hard_negative_count=getattr(args, "hard_negative_count", 16),
            hard_negative_margin=getattr(args, "hard_negative_margin", 0.5),
            hard_negative_weight=getattr(args, "hard_negative_weight", 0.25),
        )
        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            bad = [name for name, value in components.items() if not torch.isfinite(value)]
            print(
                f"Warning: skipped non-finite loss batch {skipped_nonfinite}; components={bad}",
                flush=True,
            )
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_limit = args.grad_clip if args.grad_clip > 0 else float("inf")
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_limit)
        if not torch.isfinite(grad_norm):
            if scaler.is_enabled():
                amp_overflows += 1
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)  # GradScaler records the overflow and skips the update.
                scaler.update()
                current_scale = scaler.get_scale()
                print(
                    f"Warning: AMP overflow batch {amp_overflows}; "
                    f"scale {previous_scale:g}->{current_scale:g}",
                    flush=True,
                )
            else:
                skipped_nonfinite += 1
                print(
                    f"Warning: skipped non-finite gradient batch {skipped_nonfinite}",
                    flush=True,
                )
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.step(optimizer)
        scaler.update()
        batch_size = len(batch["sample_id"])
        count += batch_size
        for name, value in components.items():
            running[name] = running.get(name, 0.0) + float(value.detach()) * batch_size
        landmark_sum += errors.detach().sum(dim=0).cpu().numpy()
        landmark_count += batch_size
    if count == 0:
        raise RuntimeError("Every training batch was non-finite; no optimizer update was applied")
    nonfinite_fraction = skipped_nonfinite / max(total_batches, 1)
    amp_overflow_fraction = amp_overflows / max(total_batches, 1)
    if nonfinite_fraction > args.max_nonfinite_fraction:
        raise RuntimeError(
            f"Non-finite batch fraction {nonfinite_fraction:.3f} exceeds "
            f"--max-nonfinite-fraction={args.max_nonfinite_fraction:.3f}"
        )
    if amp_overflow_fraction > args.max_amp_overflow_fraction:
        raise RuntimeError(
            f"AMP overflow fraction {amp_overflow_fraction:.3f} exceeds "
            f"--max-amp-overflow-fraction={args.max_amp_overflow_fraction:.3f}"
        )
    running["skipped_nonfinite_batches"] = float(skipped_nonfinite)
    running["nonfinite_batch_fraction"] = float(nonfinite_fraction)
    running["amp_overflow_batches"] = float(amp_overflows)
    running["amp_overflow_fraction"] = float(amp_overflow_fraction)
    return (
        {
            name: value
            if name in (
                "skipped_nonfinite_batches",
                "nonfinite_batch_fraction",
                "amp_overflow_batches",
                "amp_overflow_fraction",
            )
            else value / max(count, 1)
            for name, value in running.items()
        },
        (landmark_sum / max(landmark_count, 1)).tolist(),
    )


def _rigid_matrix(angle_degrees=0.0, mirror=False, device=None, dtype=torch.float32):
    angle = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    if mirror:
        rotation = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=dtype)) @ rotation
    return rotation


def transform_batch(batch, normalizer, angle_degrees=0.0, mirror=False):
    transformed = {key: value for key, value in batch.items()}
    transformed = copy.copy(transformed)
    points = batch["points"].clone()
    features = batch["features"].clone()
    mean = torch.as_tensor(normalizer["mean"], device=points.device, dtype=points.dtype)
    std = torch.as_tensor(normalizer["std"], device=points.device, dtype=points.dtype)
    raw = features * std + mean
    center = mean[:3]
    rotation = _rigid_matrix(angle_degrees, mirror, points.device, points.dtype)
    points = (points - center) @ rotation.T + center
    raw[:, :3] = points
    raw[:, 9:12] = raw[:, 9:12] @ rotation.T
    transformed["points"] = points
    transformed["features"] = (raw - mean) / std
    transformed["coarse"] = (batch["coarse"] - center) @ rotation.T + center
    transformed["expert"] = (batch["expert"] - center) @ rotation.T + center
    if mirror:
        permutation = torch.tensor(mirror_permutation(), device=points.device)
        for name in ("coarse", "expert", "roi_index", "roi_mask", "heatmap_target", "region_target", "oracle_error"):
            transformed[name] = transformed[name][:, permutation]
    return transformed, rotation, center


def inverse_predictions(prediction, rotation, center, mirror=False):
    restored = (prediction - center) @ rotation + center
    if mirror:
        permutation = torch.tensor(mirror_permutation(), device=prediction.device)
        restored = restored[:, permutation]
    return restored


@torch.no_grad()
def collect_outputs(
    model, loader, device, args, normalizer, use_tta=False, force_refined=False
):
    model.eval()
    collected = {
        "sample_ids": [], "classes": [], "genders": [], "subject_ids": [],
        "prediction": [], "refined": [], "expert": [], "coarse": [],
        "log_var": [], "refinement_alpha": [], "oracle": [],
    }
    variants = [(0.0, False)]
    if use_tta:
        variants.extend([(-5.0, False), (5.0, False), (0.0, True)])
    for raw_batch in tqdm(loader, desc="eval", leave=False, disable=args.no_tqdm):
        batch = move_batch(raw_batch, device)
        heatmaps, variances, refinement_alphas = [], [], []
        for angle, mirror in variants:
            variant, rotation, center = transform_batch(batch, normalizer, angle, mirror)
            with autocast_context(device, args.mixed_precision, args.amp_dtype):
                outputs = model(variant, coordinate_mode=args.coordinate_mode)
            logits = outputs["local_logits"].float()
            log_var = outputs["log_var"].float()
            refinement_alpha = outputs["refinement_alpha"].float()
            if mirror:
                permutation = torch.tensor(mirror_permutation(), device=device)
                logits = logits[:, permutation]
                log_var = log_var[:, permutation]
                refinement_alpha = refinement_alpha[:, permutation]
            heatmaps.append(logits)
            variances.append(log_var)
            refinement_alphas.append(refinement_alpha)
        averaged_heatmap = torch.stack(heatmaps).mean(dim=0)
        candidates = batch["points"][batch["roi_index"]]
        refined = model.coordinates_from_logits(
            averaged_heatmap,
            candidates,
            batch["roi_mask"],
            args.coordinate_mode,
        )
        refinement_alpha = torch.stack(refinement_alphas).mean(dim=0)
        if force_refined:
            refinement_alpha = torch.ones_like(refinement_alpha)
        prediction = refined if force_refined else (
            batch["coarse"]
            + refinement_alpha[..., None] * (refined - batch["coarse"])
            if model.use_refinement_gate
            else refined
        )
        log_var = torch.logsumexp(torch.stack(variances), dim=0) - math.log(len(variances))
        expert = batch["expert"]
        errors = torch.linalg.norm(prediction - expert, dim=-1)
        collected["sample_ids"].extend(raw_batch["sample_id"])
        collected["classes"].extend(raw_batch["class"])
        collected["genders"].extend(raw_batch["gender"])
        collected["subject_ids"].extend(raw_batch["subject_id"].numpy().tolist())
        for name, value in (
            ("prediction", prediction), ("refined", refined),
            ("expert", expert), ("coarse", batch["coarse"]),
            ("log_var", log_var), ("refinement_alpha", refinement_alpha),
            ("oracle", batch["oracle_error"]),
        ):
            collected[name].append(value.detach().cpu().numpy())
    for name in (
        "prediction", "refined", "expert", "coarse", "log_var",
        "refinement_alpha", "oracle",
    ):
        collected[name] = np.concatenate(collected[name], axis=0)
    collected["subject_ids"] = np.asarray(collected["subject_ids"], dtype=np.int64)
    collected["errors"] = np.linalg.norm(collected["prediction"] - collected["expert"], axis=-1)
    collected["refined_errors"] = np.linalg.norm(
        collected["refined"] - collected["expert"], axis=-1
    )
    return collected


def validation_selection_score(validation, args):
    overall = float(validation["errors"].mean())
    core20 = float(validation["errors"][:, CORE20].mean())
    hard3 = float(validation["errors"][:, HARD3].mean())
    if args.checkpoint_metric == "balanced":
        hard_weight = float(args.checkpoint_hard3_weight)
        score = (1.0 - hard_weight) * core20 + hard_weight * hard3
    else:
        score = overall
    return score, overall, core20, hard3


def fit_refinement_calibration(outputs, mode="none"):
    calibration = {"mode": mode, "groups": {}}
    if mode == "none":
        return calibration
    groups = {"core20": CORE20, "hard3": HARD3}
    scales = np.linspace(0.0, 1.25, 26)
    for name, indices in groups.items():
        indices = list(indices)
        best = None
        for scale in scales:
            alpha = np.clip(outputs["refinement_alpha"][:, indices] * scale, 0.0, 1.0)
            prediction = outputs["coarse"][:, indices] + alpha[..., None] * (
                outputs["refined"][:, indices] - outputs["coarse"][:, indices]
            )
            ale = float(
                np.linalg.norm(prediction - outputs["expert"][:, indices], axis=-1).mean()
            )
            candidate = (ale, abs(float(scale) - 1.0), float(scale))
            if best is None or candidate < best:
                best = candidate
        calibration["groups"][name] = {
            "scale": best[2],
            "validation_ale": best[0],
        }
    return calibration


def apply_refinement_calibration(outputs, calibration):
    if calibration.get("mode") == "none":
        return outputs
    outputs = dict(outputs)
    outputs["uncalibrated_prediction"] = outputs["prediction"].copy()
    outputs["uncalibrated_refinement_alpha"] = outputs["refinement_alpha"].copy()
    alpha = outputs["refinement_alpha"].copy()
    for name, indices in (("core20", CORE20), ("hard3", HARD3)):
        scale = float(calibration["groups"][name]["scale"])
        alpha[:, indices] = np.clip(alpha[:, indices] * scale, 0.0, 1.0)
    prediction = outputs["coarse"] + alpha[..., None] * (
        outputs["refined"] - outputs["coarse"]
    )
    outputs["refinement_alpha"] = alpha
    outputs["prediction"] = prediction
    outputs["errors"] = np.linalg.norm(prediction - outputs["expert"], axis=-1)
    return outputs


def _atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state(train_loader):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    generator = getattr(train_loader, "generator", None)
    if generator is not None:
        state["loader_generator"] = generator.get_state()
    return state


def _restore_rng_state(state, train_loader):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator = getattr(train_loader, "generator", None)
    if generator is not None and "loader_generator" in state:
        generator.set_state(state["loader_generator"])


def fit_separate_refinement_gate(
    model, train_loader, val_loader, device, args, normalizer, output_dir
):
    """Freeze the refiner and fit only the sample-specific coarse/refined gate."""
    if not getattr(model, "use_refinement_gate", False):
        raise ValueError("Separate gate training requires use_refinement_gate=True")
    output_dir = Path(output_dir)
    best_path = output_dir / "best_gate.pth"
    history_path = output_dir / "gate_history.json"
    result_path = output_dir / "gate_training.json"
    if (
        getattr(args, "resume_stage2", True)
        and not getattr(args, "force_stage2_retrain", False)
        and result_path.exists()
    ):
        cached_result = json.loads(result_path.read_text(encoding="utf-8"))
        expected_signature = getattr(args, "stage2_signature", None)
        if cached_result.get("stage2_signature") == expected_signature:
            if not cached_result.get("accepted", False):
                print("Separate gate cached: refined-only fallback selected", flush=True)
                return cached_result
            if best_path.exists():
                try:
                    cached_checkpoint = torch.load(
                        best_path, map_location=device, weights_only=False
                    )
                except TypeError:
                    cached_checkpoint = torch.load(best_path, map_location=device)
                if cached_checkpoint.get("stage2_signature") == expected_signature:
                    model.load_refinement_gate_state_dict(cached_checkpoint["gate_state"])
                    print(
                        f"Separate gate cached: best epoch={cached_result['best_epoch']}",
                        flush=True,
                    )
                    return cached_result
    original_trainable = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.set_refinement_gate_trainable(True)
    gate_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        gate_parameters,
        lr=float(getattr(args, "gate_stage_lr", 1e-3)),
        weight_decay=float(getattr(args, "gate_stage_weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(int(getattr(args, "gate_stage_scheduler_patience", 4)), 1),
        min_lr=1e-6,
    )
    scaler = grad_scaler(args.mixed_precision, args.amp_dtype, args.amp_init_scale)
    epochs = int(getattr(args, "gate_stage_epochs", 40))
    minimum_epochs = int(getattr(args, "gate_stage_min_epochs", 10))
    patience = int(getattr(args, "gate_stage_patience", 10))
    baseline_validation = collect_outputs(
        model,
        val_loader,
        device,
        args,
        normalizer,
        use_tta=args.tta_validation,
        force_refined=True,
    )
    baseline_score, baseline_overall, baseline_core20, baseline_hard3 = (
        validation_selection_score(baseline_validation, args)
    )
    best_score, best_epoch, stale = baseline_score, 0, 0
    improved_checkpoint = False
    history = []
    started = time.time()
    print(
        f"Gate baseline refined-only: val_ALE={baseline_overall:.4f} "
        f"core20={baseline_core20:.4f} hard3={baseline_hard3:.4f} "
        f"selected={baseline_score:.4f}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.eval()
        for module in model.refinement_gate_modules():
            module.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        running, seen = {}, 0
        per_landmark_sum = np.zeros(NUM_LANDMARKS, dtype=np.float64)
        for batch in tqdm(
            train_loader,
            desc="gate train",
            leave=False,
            disable=args.no_tqdm,
        ):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.mixed_precision, args.amp_dtype):
                outputs = model(batch, coordinate_mode=args.coordinate_mode)
                loss, errors, components = sample_specific_gate_loss(
                    outputs,
                    batch,
                    hard_landmark_weight=getattr(args, "hard_landmark_weight", 4.0),
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Separate gate training produced a non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(
                gate_parameters, float(getattr(args, "grad_clip", 1.0))
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("Separate gate training produced non-finite gradients")
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(batch["sample_id"])
            seen += batch_size
            per_landmark_sum += errors.detach().sum(dim=0).cpu().numpy()
            for name, value in components.items():
                running[name] = running.get(name, 0.0) + float(value.detach()) * batch_size

        validation = collect_outputs(
            model,
            val_loader,
            device,
            args,
            normalizer,
            use_tta=args.tta_validation,
            force_refined=False,
        )
        score, overall, core20, hard3 = validation_selection_score(validation, args)
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "validation_selection_score": score,
            "validation_ale": overall,
            "validation_core20_ale": core20,
            "validation_hard3_ale": hard3,
            "validation_median": float(np.median(validation["errors"])),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value / max(seen, 1) for name, value in running.items()},
            **{
                f"train_lm{index}_ale": value
                for index, value in enumerate(per_landmark_sum / max(seen, 1))
            },
        }
        history.append(row)
        print(
            f"Gate epoch {epoch:03d}/{epochs} train={row['train_total']:.5f} "
            f"val_ALE={overall:.4f} core20={core20:.4f} hard3={hard3:.4f} "
            f"selected={score:.4f}",
            flush=True,
        )
        if score < best_score - float(getattr(args, "min_delta", 1e-4)):
            best_score, best_epoch, stale = score, epoch, 0
            improved_checkpoint = True
            _atomic_torch_save(
                {
                    "gate_state": model.refinement_gate_state_dict(),
                    "epoch": epoch,
                    "validation_ale": overall,
                    "validation_core20_ale": core20,
                    "validation_hard3_ale": hard3,
                    "validation_selection_score": score,
                    "stage2_signature": getattr(args, "stage2_signature", None),
                },
                best_path,
            )
        elif epoch >= minimum_epochs:
            stale += 1
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch >= minimum_epochs and stale >= patience:
            print(
                f"Gate early stopping at epoch {epoch}; best epoch={best_epoch}",
                flush=True,
            )
            break

    if improved_checkpoint:
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_path, map_location=device)
        model.load_refinement_gate_state_dict(checkpoint["gate_state"])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_trainable[name])
    model.eval()
    result = {
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "accepted": bool(improved_checkpoint),
        "baseline_refined_only": {
            "selection_score": baseline_score,
            "overall_ale": baseline_overall,
            "core20_ale": baseline_core20,
            "hard3_ale": baseline_hard3,
        },
        "selection_metric": args.checkpoint_metric,
        "training_seconds": float(time.time() - started),
        "checkpoint": str(best_path) if improved_checkpoint else None,
        "stage2_signature": getattr(args, "stage2_signature", None),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def fit_model(model, train_loader, val_loader, device, args, loss_weights, normalizer, output_dir):
    output_dir = Path(output_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience, min_lr=1e-6
    )
    scaler = grad_scaler(args.mixed_precision, args.amp_dtype, args.amp_init_scale)
    checkpoint_path = output_dir / "best_model.pth"
    last_checkpoint_path = output_dir / "last_model.pth"
    best_score, best_validation_ale, best_epoch, stale = float("inf"), float("inf"), 0, 0
    history = []
    start_epoch = 1
    elapsed_before = 0.0
    training_complete = False
    if args.resume_stage2 and not args.force_stage2_retrain and last_checkpoint_path.exists():
        try:
            resume = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            resume = torch.load(last_checkpoint_path, map_location=device)
        expected_signature = getattr(args, "stage2_signature", None)
        if resume.get("stage2_signature") != expected_signature:
            raise RuntimeError(
                "Stage 2 resume checkpoint does not match the current model/data configuration. "
                "Use a new --output-dir or pass --force-stage2-retrain."
            )
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        if resume.get("scaler_state"):
            scaler.load_state_dict(resume["scaler_state"])
        best_score = float(resume["best_score"])
        best_validation_ale = float(resume["best_validation_ale"])
        best_epoch = int(resume["best_epoch"])
        stale = int(resume["stale"])
        history = list(resume.get("history", []))
        start_epoch = int(resume["epoch"]) + 1
        elapsed_before = float(resume.get("training_seconds", 0.0))
        training_complete = bool(resume.get("training_complete", False))
        _restore_rng_state(resume.get("rng_state"), train_loader)
        print(
            f"Resuming Stage 2 after epoch {start_epoch - 1}; "
            f"best epoch={best_epoch}, selection={best_score:.4f}",
            flush=True,
        )
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if training_complete:
            break
        if args.lr_warmup_epochs > 0 and epoch <= args.lr_warmup_epochs:
            fraction = epoch / max(int(args.lr_warmup_epochs), 1)
            warmup_lr = float(args.lr) * (0.2 + 0.8 * fraction)
            for group in optimizer.param_groups:
                group["lr"] = warmup_lr
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        components, per_landmark = train_epoch(
            model, train_loader, optimizer, scaler, device, args, loss_weights
        )
        validation = collect_outputs(
            model,
            val_loader,
            device,
            args,
            normalizer,
            use_tta=args.tta_validation,
            force_refined=getattr(args, "separate_gate_training", False),
        )
        score, overall_ale, core20_ale, hard3_ale = validation_selection_score(
            validation, args
        )
        if epoch >= args.scheduler_start_epoch:
            scheduler.step(score)
        row = {
            "epoch": epoch,
            "validation_selection_score": score,
            "validation_ale": overall_ale,
            "validation_core20_ale": core20_ale,
            "validation_hard3_ale": hard3_ale,
            "validation_median": float(np.median(validation["errors"])),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value for name, value in components.items()},
            **{f"train_lm{index}_ale": value for index, value in enumerate(per_landmark)},
        }
        history.append(row)
        print(
            f"Epoch {epoch:04d}/{args.epochs} train={components['total']:.5f} "
            f"val_ALE={overall_ale:.4f} core20={core20_ale:.4f} hard3={hard3_ale:.4f} "
            f"selected={score:.4f}",
            flush=True,
        )
        if score < best_score - args.min_delta:
            best_score, best_validation_ale, best_epoch, stale = score, overall_ale, epoch, 0
            _atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_ale": overall_ale,
                    "validation_core20_ale": core20_ale,
                    "validation_hard3_ale": hard3_ale,
                    "validation_selection_score": score,
                    "stage2_signature": getattr(args, "stage2_signature", None),
                    "args": vars(args),
                    "normalizer": normalizer,
                },
                checkpoint_path,
            )
        elif epoch >= args.min_epochs:
            stale += 1
        else:
            stale = 0
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        should_stop = epoch >= args.min_epochs and stale >= args.patience
        training_complete = bool(should_stop or epoch >= args.epochs)
        if (
            epoch % max(int(args.checkpoint_every), 1) == 0
            or training_complete
            or epoch == best_epoch
        ):
            _atomic_torch_save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "best_validation_ale": best_validation_ale,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "history": history,
                    "training_seconds": elapsed_before + float(time.time() - started),
                    "training_complete": training_complete,
                    "stage2_signature": getattr(args, "stage2_signature", None),
                    "rng_state": _rng_state(train_loader),
                },
                last_checkpoint_path,
            )
        if should_stop:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
            break
    if not checkpoint_path.exists():
        raise RuntimeError("Stage 2 did not produce a best checkpoint")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return {
        "best_epoch": best_epoch,
        "best_validation_ale": best_validation_ale,
        "best_selection_score": best_score,
        "selection_metric": args.checkpoint_metric,
        "training_seconds": elapsed_before + float(time.time() - started),
        "checkpoint": str(checkpoint_path),
        "resume_checkpoint": str(last_checkpoint_path),
    }
