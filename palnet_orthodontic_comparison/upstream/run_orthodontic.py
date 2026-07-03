import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.datasets.orthodontic_dataset import OrthodonticDataset
from src.datasets.patch_dataset import PatchDataset
from src.models.loss import CombinedLoss, localizationLoss
from src.models.model import PALNET, PLNET_noatt

for parent in Path(__file__).resolve().parents:
    if (parent / "shared_metrics" / "orthodontic_analysis.py").exists():
        sys.path.append(str(parent))
        break

from shared_metrics.orthodontic_analysis import build_error_analysis, write_analysis_csvs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_splits(dataset, test_size, val_size, seed):
    indices = np.arange(len(dataset))
    strata = np.array([f"{s.class_name}_{s.gender}" for s in dataset.samples])

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=strata,
    )
    train_val_strata = strata[train_val_idx]
    val_fraction = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_val_strata,
    )
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def ids_to_indices(dataset, sample_ids):
    index_by_id = {sample.sample_id: idx for idx, sample in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"Split file references samples not found in dataset: {missing[:10]}")
    return [index_by_id[sample_id] for sample_id in sample_ids]


def mean_landmarks(base_dataset, subset_indices):
    total = torch.zeros(23, 3)
    for idx in subset_indices:
        _, landmarks, _ = base_dataset[idx]
        total += landmarks
    return total / len(subset_indices)


def nearest_surface_predictions(point_clouds, predictions, k=1):
    fixed = []
    for pc, pred in zip(point_clouds, predictions):
        tree = cKDTree(pc[:, :3])
        _, idx = tree.query(pred, k=k)
        if k == 1:
            fixed.append(pc[idx, :3])
        else:
            fixed.append(pc[idx, :3].mean(axis=1))
    return np.asarray(fixed, dtype=np.float32)


def localization_errors(y_true, y_pred):
    return np.linalg.norm(y_pred - y_true, axis=-1)


def ale_summary(y_true, y_pred):
    errors = localization_errors(y_true, y_pred)
    summary = {
        "ale": float(errors.mean()),
        "std": float(errors.std()),
        "median": float(np.median(errors)),
        "max": float(errors.max()),
        "per_landmark_ale": errors.mean(axis=0).tolist(),
        "per_sample_ale": errors.mean(axis=1).tolist(),
    }
    for threshold in (2.0, 2.5, 3.0):
        key = ("%g" % threshold).replace(".", "_")
        summary[f"pck_at_{key}mm"] = float((errors <= threshold).mean())
    return summary


def template_baseline(train_mean, y_true, point_clouds=None):
    pred = np.repeat(train_mean[None, :, :], repeats=len(y_true), axis=0).astype(np.float32)
    if point_clouds is not None:
        pred = nearest_surface_predictions(point_clouds, pred, k=1)
    return pred


def inverse_normalize_arrays(dataset, indices, *arrays):
    restored = [np.empty_like(array, dtype=np.float32) for array in arrays]
    for row, dataset_idx in enumerate(indices):
        center, scale = dataset.normalization_params(dataset_idx)
        center = center.reshape(1, 3)
        for out, array in zip(restored, arrays):
            out[row] = array[row] * scale + center
    return restored


def collect_predictions(model, loader, device, snap_k):
    model.eval()
    preds = []
    truths = []
    point_clouds = []
    with torch.no_grad():
        for patches, landmarks, sampled_points in loader:
            patches = patches.to(device, non_blocking=True)
            outputs = model(patches).cpu().numpy()
            preds.append(outputs)
            truths.append(landmarks.numpy())
            point_clouds.append(sampled_points.numpy())

    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)
    point_clouds = np.concatenate(point_clouds, axis=0)
    snapped = nearest_surface_predictions(point_clouds, preds.copy(), k=snap_k)
    return preds, snapped, truths, point_clouds


def write_prediction_csv(path, samples, y_true, y_pred):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sample_id",
                "class",
                "gender",
                "subject_id",
                "landmark",
                "expert_x",
                "expert_y",
                "expert_z",
                "palnet_x",
                "palnet_y",
                "palnet_z",
                "localization_error",
            ]
        )
        errors = localization_errors(y_true, y_pred)
        for sample, truth, pred, sample_errors in zip(samples, y_true, y_pred, errors):
            for lm_idx in range(truth.shape[0]):
                writer.writerow(
                    [
                        sample.sample_id,
                        sample.class_name,
                        sample.gender,
                        sample.subject_id,
                        lm_idx,
                        *truth[lm_idx].tolist(),
                        *pred[lm_idx].tolist(),
                        float(sample_errors[lm_idx]),
                    ]
                )


def write_group_metrics(path, samples, y_true, y_pred):
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault((sample.class_name, sample.gender), []).append(i)

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "gender", "n_samples", "ale", "std", "median"])
        for (class_name, gender), idxs in sorted(groups.items()):
            summary = ale_summary(y_true[idxs], y_pred[idxs])
            writer.writerow([class_name, gender, len(idxs), summary["ale"], summary["std"], summary["median"]])


def main():
    parser = argparse.ArgumentParser(description="Train PAL-Net on the 23-point orthodontic dataset and report ALE.")
    parser.add_argument("--data-root", default="../../data/dataset", help="Path to Class*/ mesh and landmark folders.")
    parser.add_argument("--output-dir", default="../runs/orthodontic_palnet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=250)
    parser.add_argument("--surface-points", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--splits-json", default=None, help="Shared split JSON with train/val/test sample_id lists.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--snap-k", type=int, default=1, help="Nearest sampled surface points used to snap PAL-Net output.")
    parser.add_argument("--model", choices=["PALNET", "PLNET_noatt"], default="PALNET")
    parser.add_argument("--loss", choices=["combined", "localization"], default="combined")
    parser.add_argument("--normalize", action="store_true", help="Normalize each face to unit scale before training.")
    parser.add_argument(
        "--transformation-dir",
        default=None,
        help="Directory containing PAL-Net-style *_transformation_matrix.npy files.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OrthodonticDataset(
        args.data_root,
        cache_dir=output_dir / "mesh_cache",
        num_surface_points=args.surface_points,
        normalize=args.normalize,
        transformation_dir=args.transformation_dir,
    )
    print(f"Paired samples: {len(dataset)}")
    print(f"Meshes without matching landmark file: {len(dataset.missing_landmarks)}")

    source_splits_json = None
    if args.splits_json:
        source_splits_json = str(Path(args.splits_json))
        split_source = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
        train_idx = ids_to_indices(dataset, split_source["train"])
        val_idx = ids_to_indices(dataset, split_source["val"])
        test_idx = ids_to_indices(dataset, split_source["test"])
    else:
        train_idx, val_idx, test_idx = make_splits(dataset, args.test_size, args.val_size, args.seed)
    split_payload = {
        "train": [dataset.samples[i].sample_id for i in train_idx],
        "val": [dataset.samples[i].sample_id for i in val_idx],
        "test": [dataset.samples[i].sample_id for i in test_idx],
        "source_splits_json": source_splits_json,
        "missing_landmarks": [str(p) for p in dataset.missing_landmarks],
    }
    (output_dir / "splits.json").write_text(json.dumps(split_payload, indent=2), encoding="utf-8")

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_mean = mean_landmarks(dataset, train_idx)
    np.save(output_dir / "train_mean_landmarks.npy", train_mean.numpy())

    train_patches = PatchDataset(train_ds, train_mean, args.patch_size, output_dir / "patch_cache_train")
    val_patches = PatchDataset(val_ds, train_mean, args.patch_size, output_dir / "patch_cache_val")
    test_patches = PatchDataset(test_ds, train_mean, args.patch_size, output_dir / "patch_cache_test")

    print("Pre-caching train/val/test patches...")
    for patch_ds in (train_patches, val_patches, test_patches):
        for i in tqdm(range(len(patch_ds)), leave=False):
            _ = patch_ds[i]

    train_loader = DataLoader(train_patches, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_patches, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_patches, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    first_patch, first_landmark, _ = train_patches[0]
    input_shape = first_patch.shape
    output_shape = first_landmark.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = PALNET if args.model == "PALNET" else PLNET_noatt
    model = model_cls(input_shape, output_shape, seed=args.seed).to(device)
    criterion = CombinedLoss(alpha=0.6, beta=0.4) if args.loss == "combined" else localizationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_val = float("inf")
    epochs_no_improve = 0
    best_path = output_dir / "best_model.pth"
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for patches, landmarks, _ in train_loader:
            patches = patches.to(device, non_blocking=True)
            landmarks = landmarks.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(landmarks, outputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * patches.size(0)
        train_loss /= len(train_patches)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, landmarks, _ in val_loader:
                patches = patches.to(device, non_blocking=True)
                landmarks = landmarks.to(device, non_blocking=True)
                outputs = model(patches)
                val_loss += criterion(landmarks, outputs).item() * patches.size(0)
        val_loss /= len(val_patches)
        scheduler.step(val_loss)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch + 1:04d}/{args.epochs} train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    model.load_state_dict(torch.load(best_path, map_location=device))

    raw_pred, snapped_pred, y_test, point_clouds = collect_predictions(model, test_loader, device, args.snap_k)
    test_samples = [dataset.samples[i] for i in test_idx]
    baseline_raw_pred = template_baseline(train_mean.numpy(), y_test)
    baseline_snapped_pred = template_baseline(train_mean.numpy(), y_test, point_clouds)

    if args.normalize:
        raw_pred, snapped_pred, y_test, point_clouds, baseline_raw_pred, baseline_snapped_pred = inverse_normalize_arrays(
            dataset,
            test_idx,
            raw_pred,
            snapped_pred,
            y_test,
            point_clouds,
            baseline_raw_pred,
            baseline_snapped_pred,
        )

    palnet_raw = ale_summary(y_test, raw_pred)
    palnet_snapped = ale_summary(y_test, snapped_pred)
    baseline_raw = ale_summary(y_test, baseline_raw_pred)
    baseline_snapped = ale_summary(y_test, baseline_snapped_pred)
    advanced_analysis = build_error_analysis(test_samples, localization_errors(y_test, snapped_pred))

    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "unit": "dataset coordinate unit",
        "clinical_threshold_unit": "mm",
        "model": args.model,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "palnet_raw": palnet_raw,
        "palnet_snapped": palnet_snapped,
        "mean_shape_baseline_raw": baseline_raw,
        "mean_shape_baseline_snapped": baseline_snapped,
        **advanced_analysis,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_prediction_csv(output_dir / "predictions_test.csv", test_samples, y_test, snapped_pred)
    write_group_metrics(output_dir / "group_metrics_test.csv", test_samples, y_test, snapped_pred)
    write_analysis_csvs(output_dir, advanced_analysis, suffix="test")

    print("\nEvaluation against expert orthodontist landmarks")
    print(f"PAL-Net raw ALE:      {palnet_raw['ale']:.4f}")
    print(f"PAL-Net snapped ALE:  {palnet_snapped['ale']:.4f}")
    print(f"Mean-template ALE:    {baseline_snapped['ale']:.4f}")
    print(f"Results saved to:     {output_dir}")


if __name__ == "__main__":
    main()
