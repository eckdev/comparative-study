import numpy as np
import torch
from tqdm import tqdm

from .losses import atlas_loss


def train_one_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    total = 0.0
    parts_total = {}
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    for batch in tqdm(loader, desc="train", leave=False, disable=args.no_tqdm):
        points = batch["points_norm"].to(device)
        features = batch["features"].to(device)
        landmarks = batch["landmarks_norm"].to(device)
        scale = batch["scale"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(points, features)
            loss, parts = atlas_loss(outputs, landmarks, scale, args)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * points.shape[0]
        for key, value in parts.items():
            if key == "per_landmark_coord_loss":
                arr = np.asarray(value, dtype=np.float64) * points.shape[0]
                parts_total[key] = parts_total.get(key, np.zeros_like(arr)) + arr
            else:
                parts_total[key] = parts_total.get(key, 0.0) + value * points.shape[0]
    n = max(1, len(loader.dataset))
    out = {}
    for key, value in parts_total.items():
        out[key] = (value / n).tolist() if key == "per_landmark_coord_loss" else value / n
    return total / n, out


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    total = 0.0
    parts_total = {}
    preds = []
    experts = []
    confidences = []
    sample_indices = []
    amp_enabled = bool(args.mixed_precision and device.type == "cuda")
    for batch in tqdm(loader, desc="eval", leave=False, disable=args.no_tqdm):
        points = batch["points_norm"].to(device)
        features = batch["features"].to(device)
        landmarks = batch["landmarks_norm"].to(device)
        scale = batch["scale"].to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(points, features)
            loss, parts = atlas_loss(outputs, landmarks, scale, args)
        total += float(loss.detach().cpu()) * points.shape[0]
        for key, value in parts.items():
            if key == "per_landmark_coord_loss":
                arr = np.asarray(value, dtype=np.float64) * points.shape[0]
                parts_total[key] = parts_total.get(key, np.zeros_like(arr)) + arr
            else:
                parts_total[key] = parts_total.get(key, 0.0) + value * points.shape[0]
        center = np.zeros((points.shape[0], 1, 3), dtype=np.float32)
        # Dataset points are already aligned-mm in points_world. Convert normalized preds by nearest local affine using normalizer scale.
        pred_norm = outputs["pred_norm"].detach().float().cpu().numpy()
        scale_np = batch["scale"].numpy().reshape(-1, 1, 1)
        norm_center = batch.get("normalizer_center")
        if norm_center is None:
            # run_atlas_spnet injects world conversion through dataset-normalizer fields in batch by default collate omission;
            # use points/landmarks relation: landmarks_world = landmarks_norm * scale + center. Center is recovered from first landmark.
            landmark_norm_np = batch["landmarks_norm"].numpy()
            landmark_world_np = batch["landmarks_world"].numpy()
            center = landmark_world_np[:, :1, :] - landmark_norm_np[:, :1, :] * scale_np
        pred_world = pred_norm * scale_np + center
        expert_world = batch["landmarks_world"].numpy()
        preds.append(pred_world.astype(np.float32))
        experts.append(expert_world.astype(np.float32))
        confidences.append(outputs["confidence"].detach().float().cpu().numpy())
        sample_indices.extend(batch["sample_index"].numpy().astype(int).tolist())
    n = max(1, len(loader.dataset))
    out_parts = {}
    for key, value in parts_total.items():
        out_parts[key] = (value / n).tolist() if key == "per_landmark_coord_loss" else value / n
    pred = np.concatenate(preds, axis=0)
    expert = np.concatenate(experts, axis=0)
    confidence = np.concatenate(confidences, axis=0)
    errors = np.linalg.norm(pred - expert, axis=-1)
    return total / n, out_parts, pred, expert, errors, confidence, sample_indices
