import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from run_orthodontic import (
    ale_summary,
    collect_predictions,
    template_baseline,
    write_group_metrics,
    write_prediction_csv,
)
from src.datasets.orthodontic_dataset import OrthodonticDataset
from src.datasets.patch_dataset import PatchDataset
from src.models.model import PALNET, PLNET_noatt


def ids_to_indices(dataset, sample_ids):
    index = {sample.sample_id: i for i, sample in enumerate(dataset.samples)}
    return [index[sample_id] for sample_id in sample_ids]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved PAL-Net orthodontic run.")
    parser.add_argument("--data-root", default="../../data/dataset")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--transformation-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--surface-points", type=int, default=100000)
    parser.add_argument("--snap-k", type=int, default=1)
    parser.add_argument("--model", choices=["PALNET", "PLNET_noatt"], default="PALNET")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    dataset = OrthodonticDataset(
        args.data_root,
        cache_dir=run_dir / "mesh_cache",
        num_surface_points=args.surface_points,
        transformation_dir=args.transformation_dir,
    )
    splits = json.loads((run_dir / "splits.json").read_text(encoding="utf-8"))
    test_idx = ids_to_indices(dataset, splits["test"])
    test_ds = Subset(dataset, test_idx)
    train_mean = torch.from_numpy(np.load(run_dir / "train_mean_landmarks.npy")).float()

    test_patches = PatchDataset(test_ds, train_mean, args.patch_size, run_dir / "patch_cache_test")
    test_loader = DataLoader(
        test_patches,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    first_patch, first_landmark, _ = test_patches[0]
    model_cls = PALNET if args.model == "PALNET" else PLNET_noatt
    model = model_cls(first_patch.shape, first_landmark.shape, seed=42)
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location="cpu"))

    raw_pred, snapped_pred, y_test, point_clouds = collect_predictions(
        model,
        test_loader,
        torch.device("cpu"),
        args.snap_k,
    )
    test_samples = [dataset.samples[i] for i in test_idx]

    baseline_raw_pred = template_baseline(train_mean.numpy(), y_test)
    baseline_snapped_pred = template_baseline(train_mean.numpy(), y_test, point_clouds)

    metrics = {
        "metric": "Average Localization Error (mean Euclidean distance over 23 landmarks)",
        "unit": "dataset coordinate unit",
        "model": args.model,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(test_idx),
        "palnet_raw": ale_summary(y_test, raw_pred),
        "palnet_snapped": ale_summary(y_test, snapped_pred),
        "mean_shape_baseline_raw": ale_summary(y_test, baseline_raw_pred),
        "mean_shape_baseline_snapped": ale_summary(y_test, baseline_snapped_pred),
        "note": "Evaluated from saved best_model.pth after the long run was interrupted.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_prediction_csv(run_dir / "predictions_test.csv", test_samples, y_test, snapped_pred)
    write_group_metrics(run_dir / "group_metrics_test.csv", test_samples, y_test, snapped_pred)

    print(f"PAL-Net raw ALE:     {metrics['palnet_raw']['ale']:.4f}")
    print(f"PAL-Net snapped ALE: {metrics['palnet_snapped']['ale']:.4f}")
    print(f"Baseline ALE:        {metrics['mean_shape_baseline_snapped']['ale']:.4f}")
    print(f"Results saved to:    {run_dir}")


if __name__ == "__main__":
    main()
