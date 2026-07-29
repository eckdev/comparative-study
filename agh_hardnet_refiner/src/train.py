import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import HARD_LANDMARKS, HardLandmarkDataset
from .metrics import combined_metrics
from .model import AGHHardNet, hardnet_loss


def set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    total = 0.0
    parts_total = {}
    amp = bool(args.mixed_precision and device.type == "cuda")
    for batch in tqdm(loader, desc="train", leave=False, disable=args.no_tqdm):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(
                batch["candidate_features"],
                batch["candidate_points"],
                batch["landmark"],
                temperature=args.temperature,
                topk=args.topk,
            )
            loss, parts = hardnet_loss(outputs, batch, args)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        bsz = batch["landmark"].shape[0]
        total += float(loss.detach().cpu()) * bsz
        for key, value in parts.items():
            parts_total[key] = parts_total.get(key, 0.0) + float(value) * bsz
    n = max(1, len(loader.dataset))
    return total / n, {k: v / n for k, v in parts_total.items()}


@torch.no_grad()
def evaluate_hard(model, loader, device, args):
    model.eval()
    total = 0.0
    parts_total = {}
    rows = []
    amp = bool(args.mixed_precision and device.type == "cuda")
    for batch in tqdm(loader, desc="eval", leave=False, disable=args.no_tqdm):
        batch_device = {k: v.to(device) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(
                batch_device["candidate_features"],
                batch_device["candidate_points"],
                batch_device["landmark"],
                temperature=args.temperature,
                topk=args.topk,
            )
            loss, parts = hardnet_loss(outputs, batch_device, args)
        bsz = batch["landmark"].shape[0]
        total += float(loss.detach().cpu()) * bsz
        for key, value in parts.items():
            parts_total[key] = parts_total.get(key, 0.0) + float(value) * bsz
        pred = outputs["pred"].detach().float().cpu().numpy()
        weighted = outputs["weighted"].detach().float().cpu().numpy()
        confidence = outputs["confidence"].detach().float().cpu().numpy()
        for i in range(bsz):
            rows.append(
                {
                    "sample_index": int(batch["sample_index"][i]),
                    "landmark": int(batch["landmark"][i]),
                    "pred": pred[i],
                    "weighted": weighted[i],
                    "expert": batch["expert"][i].numpy(),
                    "base": batch["base"][i].numpy(),
                    "confidence": float(confidence[i]),
                }
            )
    n = max(1, len(loader.dataset))
    return total / n, {k: v / n for k, v in parts_total.items()}, rows


def build_loader(samples, args, split_name, shuffle=False):
    dataset = HardLandmarkDataset(
        samples,
        data_root=args.data_root,
        point_cache_dir=args.point_cache_dir,
        landmarks=HARD_LANDMARKS,
        radius_mm=args.radius_mm,
        trichion_radius_mm=args.trichion_radius_mm,
        patch_points=args.patch_points,
        max_surface_points=args.max_surface_points,
        sigma_mm=args.sigma_mm,
        seed=args.seed + (0 if split_name == "train" else 10000),
        max_items=args.max_items if split_name == "train" else None,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())


def train_hardnet(train_samples, val_samples, args, output_dir):
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_loader = build_loader(train_samples, args, "train", shuffle=True)
    val_loader = build_loader(val_samples, args, "val", shuffle=False)
    sample_item = train_loader.dataset[0]
    input_dim = sample_item["candidate_features"].shape[-1]
    model = AGHHardNet(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        landmark_embedding_dim=args.landmark_embedding_dim,
        residual_limit_mm=args.residual_limit_mm,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.03)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.mixed_precision and device.type == "cuda"))
    history = []
    best_val = float("inf")
    no_improve = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_parts = train_epoch(model, train_loader, optimizer, scaler, device, args)
        val_loss, val_parts, _ = evaluate_hard(model, val_loader, device, args)
        val_error = float(val_parts.get("mean_error", val_loss))
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_hard_ale": val_error,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_parts": train_parts,
            "val_parts": val_parts,
        }
        history.append(row)
        print(f"Epoch {epoch:04d}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f} val_hard_ALE={val_error:.4f}", flush=True)
        if val_error < best_val:
            best_val = val_error
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_hard_ale": val_error, "input_dim": input_dim, "args": vars(args)}, output_dir / "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    return model, history, time.time() - start, int(checkpoint["epoch"]), float(checkpoint["val_hard_ale"])


def combine_predictions(samples, hard_rows, mode="pred"):
    pred = np.stack([sample.base.copy() for sample in samples], axis=0)
    expert = np.stack([sample.expert for sample in samples], axis=0)
    confidence = np.zeros((len(samples), 23), dtype=np.float32)
    for row in hard_rows:
        pred[row["sample_index"], row["landmark"]] = row[mode]
        confidence[row["sample_index"], row["landmark"]] = row["confidence"]
    errors = np.linalg.norm(pred - expert, axis=-1)
    return pred, expert, confidence, errors, combined_metrics(errors)
