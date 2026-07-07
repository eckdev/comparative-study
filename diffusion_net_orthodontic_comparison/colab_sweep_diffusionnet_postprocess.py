import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

for parent in Path(__file__).resolve().parents:
    if (parent / "shared_metrics" / "orthodontic_analysis.py").exists():
        sys.path.append(str(parent))
        break

from shared_metrics.orthodontic_analysis import write_analysis_csvs


TOPK_RE = re.compile(r"^topk(?P<k>\d+)(?:_t(?P<t>[0-9]+(?:p[0-9]+)?|[0-9]+(?:\.[0-9]+)?))?$")


def assert_path(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} bulunamadi: {path}")


def auto_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_mlp_hidden_dims(value):
    if value in (None, "", []):
        return None
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_variant(token):
    token = token.strip()
    if token == "argmax":
        return {"name": "argmax", "postprocess": "argmax", "refine_topk": 1, "refine_temperature": 1.0}
    if token == "softmax":
        return {"name": "softmax", "postprocess": "softmax", "refine_topk": 1, "refine_temperature": 1.0}
    if token == "sigmoid_weighted":
        return {
            "name": "sigmoid_weighted",
            "postprocess": "sigmoid_weighted",
            "refine_topk": 1,
            "refine_temperature": 1.0,
        }
    match = TOPK_RE.match(token)
    if match:
        k = int(match.group("k"))
        temp_text = match.group("t") or "1"
        temperature = float(temp_text.replace("p", "."))
        temp_name = ("%g" % temperature).replace(".", "p")
        return {
            "name": f"topk{k}_t{temp_name}",
            "postprocess": "topk_softmax",
            "refine_topk": k,
            "refine_temperature": temperature,
        }
    raise ValueError(f"Unsupported postprocess variant: {token}")


def load_metrics(run_dir):
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def get_config(args, metrics, key, default=None):
    value = getattr(args, key.replace("-", "_"), None)
    if value is not None:
        return value
    return metrics.get(key, default)


def main():
    parser = argparse.ArgumentParser(description="Sweep DiffusionNet postprocess settings without retraining.")
    parser.add_argument(
        "--run-dir",
        default="/content/drive/MyDrive/orthodontic/diffusion_runs/diffusionnet_shared_metrics_p12000_k96_w192_b8_e220_topk30",
        help="Training run directory containing best_model.pth, metrics.json, point_cache and op_cache.",
    )
    parser.add_argument("--data-root", default="/content/drive/MyDrive/orthodontic/data/dataset")
    parser.add_argument(
        "--splits-json",
        default="/content/comparative-study/shared_splits/orthodontic_180_60_60_seed42.json",
    )
    parser.add_argument(
        "--transformation-dir",
        default="/content/drive/MyDrive/orthodontic/transforms/orthodontic_procrustes_rigid_20260627_143801",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--variants",
        default="argmax,topk5_t1,topk10_t1,topk20_t1,topk30_t1,topk40_t1,topk20_t0p5,topk30_t0p5,topk30_t2,softmax,sigmoid_weighted",
    )
    parser.add_argument("--surface-points", type=int, default=None)
    parser.add_argument("--k-eig", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--mask-radius", type=float, default=None)
    parser.add_argument("--loss-mode", choices=["ce", "mask_bce"], default=None)
    parser.add_argument("--positive-weight", type=float, default=None)
    parser.add_argument("--input-features", choices=["xyz", "hks", "xyz_hks"], default=None)
    parser.add_argument("--hks-dim", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--mlp-hidden-dims", default=None)
    parser.add_argument("--use-mesh-vertices", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    import run_orthodontic_diffusion as diffusion_run

    run_dir = Path(args.run_dir)
    model_path = Path(args.model_path) if args.model_path else run_dir / "best_model.pth"
    output_root = Path(args.output_root) if args.output_root else run_dir / "postprocess_sweep"

    assert_path(run_dir, "Run directory")
    assert_path(model_path, "Model checkpoint")
    assert_path(args.data_root, "Dataset")
    assert_path(args.splits_json, "Shared split JSON")
    if args.transformation_dir:
        assert_path(args.transformation_dir, "Transformation directory")

    metrics = load_metrics(run_dir)
    seed = int(args.seed if args.seed is not None else metrics.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    surface_points = int(get_config(args, metrics, "surface_points", 12000))
    k_eig = int(get_config(args, metrics, "k_eig", 96))
    sigma = float(get_config(args, metrics, "sigma", 0.04))
    mask_radius = float(get_config(args, metrics, "mask_radius", 3.5))
    loss_mode = get_config(args, metrics, "loss_mode", "mask_bce")
    positive_weight = float(get_config(args, metrics, "positive_weight", 0.0))
    input_features = get_config(args, metrics, "input_features", "xyz")
    hks_dim = int(get_config(args, metrics, "hks_dim", 16))
    width = int(get_config(args, metrics, "width", 192))
    blocks = int(get_config(args, metrics, "blocks", 8))
    mlp_hidden_dims = parse_mlp_hidden_dims(args.mlp_hidden_dims if args.mlp_hidden_dims else metrics.get("mlp_hidden_dims"))
    use_mesh_vertices = bool(args.use_mesh_vertices or metrics.get("use_mesh_vertices", False))

    output_root.mkdir(parents=True, exist_ok=True)
    dataset = diffusion_run.DiffusionOrthodonticDataset(
        root_dir=args.data_root,
        cache_dir=run_dir / "point_cache",
        op_cache_dir=run_dir / "op_cache",
        num_points=surface_points,
        k_eig=k_eig,
        sigma=sigma,
        mask_radius=mask_radius,
        transformation_dir=args.transformation_dir,
        use_mesh_vertices=use_mesh_vertices,
        seed=seed,
    )
    split_source = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
    test_idx = diffusion_run.ids_to_indices(dataset, split_source["test"])
    test_ds = Subset(dataset, test_idx)

    print(f"Paired samples: {len(dataset)}", flush=True)
    print(f"Test samples: {len(test_ds)}", flush=True)
    print("Pre-caching test point clouds/operators from source run cache...", flush=True)
    for idx in range(len(test_ds)):
        _ = test_ds[idx]

    device = auto_device(args.device)
    print(f"Device: {device}", flush=True)
    feature_dims = {"xyz": 3, "hks": hks_dim, "xyz_hks": 3 + hks_dim}
    model = diffusion_run.diffusion_net.layers.DiffusionNet(
        C_in=feature_dims[input_features],
        C_out=23,
        C_width=width,
        N_block=blocks,
        outputs_at="vertices",
        mlp_hidden_dims=mlp_hidden_dims,
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    variants = [parse_variant(token) for token in args.variants.split(",") if token.strip()]
    summary_rows = []
    best = None
    for variant in variants:
        print(
            f"\nEvaluating {variant['name']} "
            f"postprocess={variant['postprocess']} topk={variant['refine_topk']} temp={variant['refine_temperature']}",
            flush=True,
        )
        test_loss, test_rows, test_errors = diffusion_run.evaluate(
            model,
            test_ds,
            device,
            input_features,
            hks_dim,
            loss_mode=loss_mode,
            positive_weight=positive_weight,
            refine_topk=variant["refine_topk"],
            refine_temperature=variant["refine_temperature"],
            postprocess=variant["postprocess"],
        )
        analysis = diffusion_run.analysis_from_eval_rows(dataset, test_rows)
        heatmap = diffusion_run.summarize_errors(test_errors)
        variant_dir = output_root / variant["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_run_dir": str(run_dir),
            "model_path": str(model_path),
            "test_loss": test_loss,
            "postprocess": variant["postprocess"],
            "refine_topk": variant["refine_topk"],
            "refine_temperature": variant["refine_temperature"],
            "diffusionnet_heatmap": heatmap,
            **analysis,
        }
        (variant_dir / "metrics_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        diffusion_run.write_predictions(variant_dir / "predictions_eval_test.csv", dataset, test_rows)
        diffusion_run.write_group_metrics(variant_dir / "group_metrics_eval_test.csv", dataset, test_rows)
        write_analysis_csvs(variant_dir, analysis, suffix="eval_test")

        row = {
            "name": variant["name"],
            "postprocess": variant["postprocess"],
            "refine_topk": variant["refine_topk"],
            "refine_temperature": variant["refine_temperature"],
            "ale": heatmap["ale"],
            "median": heatmap["median"],
            "std": heatmap["std"],
            "max": heatmap["max"],
            "pck_at_2mm": heatmap["pck_at_2mm"],
            "pck_at_2_5mm": heatmap["pck_at_2_5mm"],
            "pck_at_3mm": heatmap["pck_at_3mm"],
            "output_dir": str(variant_dir),
        }
        summary_rows.append(row)
        if best is None or row["ale"] < best["ale"]:
            best = row
        print(
            f"{variant['name']}: ALE={row['ale']:.4f} median={row['median']:.4f} "
            f"PCK@2={row['pck_at_2mm']:.4f}",
            flush=True,
        )

    summary_rows.sort(key=lambda row: row["ale"])
    summary_path = output_root / "postprocess_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output_root / "best_postprocess.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("\nBest postprocess:", flush=True)
    print(json.dumps(best, indent=2), flush=True)
    print("Summary saved to:", summary_path, flush=True)


if __name__ == "__main__":
    main()
