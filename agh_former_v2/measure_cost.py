"""Measure parameters, per-scan inference time and peak GPU memory (Colab, GPU).

The four methods are rebuilt from their trained fold checkpoints on Drive and
run one scan at a time (batch size 1) on the test partition of one fold, with
all on-disk and in-memory caches disabled, so that every scan is processed
from its mesh file. The shared label-free rigid alignment is precomputed for
all methods and is not timed.

Each scan's time is split into a CPU part (mesh reading, sampling, operators,
geodesic regions, rendering) and a network part (GPU forward passes and
coordinate decoding). The coordinates produced here are compared with the
archived test predictions of the same fold, so the timed pipeline is the
reported one.

Colab:
    python agh_former_v2/measure_cost.py --fold 1 \
        --out /content/drive/MyDrive/orthodontic/cost/cost_fold1.json
Local code-path check (random weights, CPU, two scans, comparators only):
    python agh_former_v2/measure_cost.py --smoke
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DRIVE = Path("/content/drive/MyDrive/orthodontic")
COMPARATOR_RUNS = {
    "pointnet2": DRIVE / "pointnet2_runs/pointnet2_publication_cv_seed42",
    "diffusionnet": DRIVE / "diffusion_runs/diffusionnet_publication_cv_seed42",
    "palnet": DRIVE / "palnet_runs/palnet_publication_cv_seed42",
}
PREDICTION_PREFIX = {"pointnet2": "pointnet2", "diffusionnet": "prediction", "palnet": "palnet"}
PROPOSED_RUN = DRIVE / "agh_v2_runs/publication_cv_v2final_meshonly_seed42"
PROPOSED_PRESET = "v2final_meshonly_val_all_folds"
WARMUP_SCANS = 3


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def timed(store, key, device):
    """Add the wall-clock seconds of the block to store[key]."""
    synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        store[key] = store.get(key, 0.0) + time.perf_counter() - started


@contextlib.contextmanager
def no_array_cache():
    """Keep comparator datasets from writing their per-scan .npz caches."""
    original = np.savez_compressed
    np.savez_compressed = lambda *args, **kwargs: None
    try:
        yield
    finally:
        np.savez_compressed = original


def count_parameters(*modules):
    return int(sum(parameter.numel() for module in modules for parameter in module.parameters()))


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / 2**20)


def read_archived(path, sample_ids, prefix):
    """Archived coordinates {sample_id: [23, 3]} for the given column prefix."""
    wanted = set(sample_ids)
    values = {sample_id: np.zeros((23, 3), dtype=np.float64) for sample_id in sample_ids}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["sample_id"] in wanted:
                values[row["sample_id"]][int(row["landmark"])] = [
                    float(row[f"{prefix}_{axis}"]) for axis in "xyz"
                ]
    return values


def max_difference(predicted, archived):
    return float(max(
        np.linalg.norm(np.asarray(predicted[key], dtype=np.float64) - archived[key], axis=-1).max()
        for key in predicted
    ))


def summarize_times(rows):
    keys = sorted({key for row in rows for key in row if key != "sample_id"})
    summary = {}
    for key in keys:
        values = np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float64)
        summary[key] = {
            "median_s": float(np.median(values)),
            "q1_s": float(np.percentile(values, 25)),
            "q3_s": float(np.percentile(values, 75)),
            "mean_s": float(values.mean()),
        }
    return summary


def run_scans(sample_ids, process_scan, warmup):
    """Warm up on the first scans, then time every scan once."""
    for sample_id in sample_ids[:warmup]:
        process_scan(sample_id, {})
    rows, predictions = [], {}
    for sample_id in sample_ids:
        times = {}
        predictions[sample_id] = process_scan(sample_id, times)
        times["total"] = sum(times.values())
        rows.append({"sample_id": sample_id, **times})
    return rows, predictions


def fold_test_ids(fold):
    """Test participants of a fold from the shared five-fold manifest."""
    manifest = json.loads(
        (ROOT / "shared_splits/orthodontic_5fold_192_48_60_seed42.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in manifest["folds"] if int(item["fold"]) == int(fold))
    return list(entry["test"])


# ----------------------------------------------------------------------------
# Comparators
# ----------------------------------------------------------------------------
def comparator_config(name, fold, smoke):
    if smoke:
        return {"data_root": str(ROOT / "data/dataset"), "transformation_npz": None,
                "transformation_dir": None, "seed": 42}
    run_dir = COMPARATOR_RUNS[name] / f"fold_{fold}"
    return json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))


def load_weights(model, name, fold, device, smoke, loader):
    if not smoke:
        model.load_state_dict(loader(COMPARATOR_RUNS[name] / f"fold_{fold}" / "best_model.pth", device))
    return model.eval()


def measure_pointnet2(fold, sample_ids, device, smoke, warmup):
    sys.path.insert(0, str(ROOT / "pointnet2_orthodontic_comparison"))
    import run_orthodontic_pointnet2 as module

    defaults = {"surface_points": 4096, "eval_surface_points": 4096, "mask_radius": 3.5,
                "target_mode": "gaussian", "heatmap_sigma": 3.5, "use_normals": True,
                "sa1_points": 1024, "sa2_points": 256, "sa3_points": 64, "nsample": 32,
                "postprocess": "topk_softmax", "temperature": 1.0, "topk": 20}
    cfg = {**defaults, **comparator_config("pointnet2", fold, smoke)}
    scratch = Path(tempfile.mkdtemp(prefix="pointnet2_"))
    dataset = module.OrthodonticPointCloudDataset(
        root_dir=cfg["data_root"], cache_dir=scratch,
        num_points=cfg["eval_surface_points"] or cfg["surface_points"],
        mask_radius=cfg["mask_radius"], target_mode=cfg["target_mode"],
        heatmap_sigma=cfg["heatmap_sigma"], use_normals=cfg["use_normals"],
        transformation_dir=cfg["transformation_dir"],
        transformation_npz=cfg["transformation_npz"], seed=cfg["seed"],
    )
    index = {sample.sample_id: i for i, sample in enumerate(dataset.samples)}
    model = module.PointNet2LandmarkSeg(
        num_landmarks=23, input_dim=6 if cfg["use_normals"] else 3,
        sa1_points=cfg["sa1_points"], sa2_points=cfg["sa2_points"],
        sa3_points=cfg["sa3_points"], nsample=cfg["nsample"],
    ).to(device)
    load_weights(model, "pointnet2", fold, device, smoke, module.load_torch)

    @torch.no_grad()
    def process(sample_id, times):
        with timed(times, "cpu_preprocessing", device), no_array_cache():
            points_norm, features, points_world = dataset._load_arrays(index[sample_id])[:3]
        with timed(times, "network", device):
            logits = model(
                torch.from_numpy(points_norm)[None].to(device),
                torch.from_numpy(features)[None].to(device),
            )
            return module.predict_landmarks(
                logits, torch.from_numpy(points_world)[None], postprocess=cfg["postprocess"],
                temperature=cfg["temperature"], topk=cfg["topk"],
            )[0]

    reset_peak_memory(device)
    rows, predictions = run_scans(sample_ids, process, warmup)
    shutil.rmtree(scratch, ignore_errors=True)
    return {"parameters": count_parameters(model), "rows": rows, "predictions": predictions,
            "peak_gpu_mb": peak_memory_mb(device)}


def measure_diffusionnet(fold, sample_ids, device, smoke, warmup):
    sys.path.insert(0, str(ROOT / "diffusion_net_orthodontic_comparison"))
    import run_orthodontic_diffusion as module

    defaults = {"surface_points": 12000, "k_eig": 96, "sigma": 0.04, "mask_radius": 3.5,
                "use_mesh_vertices": False, "width": 192, "blocks": 8, "input_features": "xyz",
                "hks_dim": 16, "refine_topk": 30, "refine_temperature": 1.0,
                "postprocess": "topk_softmax", "mlp_hidden_dims": "384"}
    cfg = {**defaults, **comparator_config("diffusionnet", fold, smoke)}
    if smoke:
        cfg.update({"surface_points": 1500, "k_eig": 16, "width": 32, "blocks": 2})
    scratch = Path(tempfile.mkdtemp(prefix="diffusionnet_"))
    dataset = module.DiffusionOrthodonticDataset(
        root_dir=cfg["data_root"], cache_dir=scratch / "points", op_cache_dir=scratch / "ops",
        num_points=cfg["surface_points"], k_eig=cfg["k_eig"], sigma=cfg["sigma"],
        mask_radius=cfg["mask_radius"], transformation_dir=cfg["transformation_dir"],
        transformation_npz=cfg["transformation_npz"],
        use_mesh_vertices=cfg["use_mesh_vertices"], seed=cfg["seed"],
    )
    index = {sample.sample_id: i for i, sample in enumerate(dataset.samples)}
    feature_dims = {"xyz": 3, "hks": cfg["hks_dim"], "xyz_hks": 3 + cfg["hks_dim"]}
    hidden = [int(part) for part in str(cfg["mlp_hidden_dims"]).split(",") if part.strip()]
    model = module.diffusion_net.layers.DiffusionNet(
        C_in=feature_dims[cfg["input_features"]], C_out=23, C_width=cfg["width"],
        N_block=cfg["blocks"], outputs_at="vertices", mlp_hidden_dims=hidden or None,
        dropout=True, with_gradient_features=True, with_gradient_rotations=False,
    ).to(device)
    load_weights(model, "diffusionnet", fold, device, smoke, module.load_torch)

    @torch.no_grad()
    def process(sample_id, times):
        with timed(times, "cpu_preprocessing", device), no_array_cache():
            points_world, _, points_norm, faces_np = dataset._load_arrays(index[sample_id])[:4]
        with timed(times, "cpu_operators", device):
            verts = torch.tensor(points_norm, dtype=torch.float32)
            faces = torch.tensor(faces_np, dtype=torch.long)
            _, mass, laplacian, evals, evecs, grad_x, grad_y = (
                module.diffusion_net.geometry.get_operators(
                    verts, faces, k_eig=dataset.k_eig, op_cache_dir=None
                )
            )
        with timed(times, "network", device):
            data = module.move_data(
                {"verts": verts, "faces": faces, "mass": mass, "L": laplacian, "evals": evals,
                 "evecs": evecs, "gradX": grad_x, "gradY": grad_y},
                device,
            )
            features = module.build_features(data, cfg["input_features"], cfg["hks_dim"])
            scores = model(
                features, data["mass"], L=data["L"], evals=data["evals"], evecs=data["evecs"],
                gradX=data["gradX"], gradY=data["gradY"], faces=data["faces"],
            )
            return module.predict_landmarks(
                scores.cpu().numpy(), points_world, cfg["refine_topk"],
                cfg["refine_temperature"], cfg["postprocess"],
            )

    reset_peak_memory(device)
    rows, predictions = run_scans(sample_ids, process, warmup)
    shutil.rmtree(scratch, ignore_errors=True)
    return {"parameters": count_parameters(model), "rows": rows, "predictions": predictions,
            "peak_gpu_mb": peak_memory_mb(device)}


def measure_palnet(fold, sample_ids, device, smoke, warmup):
    upstream = ROOT / "palnet_orthodontic_comparison/upstream"
    sys.path.insert(0, str(upstream))
    import run_orthodontic as module
    from src.datasets.orthodontic_dataset import OrthodonticDataset
    from src.models.model import PALNET
    from src.utils.utils import get_patches

    defaults = {"surface_points": 100000, "patch_size": 1000, "normalize": False, "snap_k": 1}
    cfg = {**defaults, **comparator_config("palnet", fold, smoke)}
    if smoke:
        cfg.update({"surface_points": 5000, "patch_size": 100})
    dataset = OrthodonticDataset(
        cfg["data_root"], cache_dir=None, num_surface_points=cfg["surface_points"],
        normalize=cfg["normalize"], transformation_dir=cfg["transformation_dir"],
        transformation_npz=cfg["transformation_npz"], seed=cfg["seed"],
    )
    index = {sample.sample_id: i for i, sample in enumerate(dataset.samples)}
    if smoke:
        train_mean = np.stack([dataset[i][1].numpy() for i in range(3)]).mean(axis=0)
    else:
        train_mean = np.load(COMPARATOR_RUNS["palnet"] / f"fold_{fold}" / "train_mean_landmarks.npy")
    model = PALNET((23, cfg["patch_size"], 3), (23, 3), seed=42).to(device)
    load_weights(model, "palnet", fold, device, smoke, module.load_torch)

    @torch.no_grad()
    def process(sample_id, times):
        with timed(times, "cpu_preprocessing", device):
            points, _, raw_vertices = dataset[index[sample_id]]
            patch = get_patches(
                np.array([raw_vertices.numpy()]), np.array(train_mean[np.newaxis, ...]),
                cfg["patch_size"],
            )[0]
        with timed(times, "network", device):
            output = model(torch.from_numpy(np.asarray(patch, dtype=np.float32))[None].to(device))
            return module.nearest_surface_predictions(
                points.numpy()[None], output.cpu().numpy(), k=cfg["snap_k"]
            )[0]

    reset_peak_memory(device)
    rows, predictions = run_scans(sample_ids, process, warmup)
    return {"parameters": count_parameters(model), "rows": rows, "predictions": predictions,
            "peak_gpu_mb": peak_memory_mb(device)}


# ----------------------------------------------------------------------------
# Proposed model: Stage 1 -> Stage 2 (TTA) -> shape prior -> six refiners
# ----------------------------------------------------------------------------
def proposed_args(device):
    """Arguments of the reported proposed-model run, checked against its saved config."""
    from agh_former_v2 import colab_run_agh_v2 as colab
    from agh_former_v2.run_agh_v2 import build_parser
    from agh_former_vnext_orthodontic_comparison.run_aghformer_vnext import configure_vnext
    from all23_rgb_geodesic_cascade.run_all23_rgb_geodesic import resolve_precision, set_seed

    command, _ = colab.command_for(PROPOSED_PRESET, 42, colab.STAGE_A_RUN)
    args = configure_vnext(build_parser().parse_args([str(item) for item in command[3:]]))
    # The arguments must equal those saved by the reported test run.
    saved = json.loads((PROPOSED_RUN / "experiment_config_test.json").read_text(encoding="utf-8"))
    if not args.preprocessing_root:
        args.preprocessing_root = saved["preprocessing_root"]
    ignored = {"validation_only", "final_test", "scratch_dir", "version"}
    mismatched = {
        key: [getattr(args, key), value]
        for key, value in saved.items()
        if key not in ignored and hasattr(args, key)
        and json.dumps(getattr(args, key), default=str) != json.dumps(value, default=str)
    }
    if mismatched:
        raise RuntimeError(f"Arguments differ from the reported run: {mismatched}")
    args.no_tqdm = True
    set_seed(args.seed)
    resolve_precision(args, device)
    return args


def measure_proposed(fold, sample_ids, device, warmup):
    from torch.utils.data import DataLoader, Subset

    from agh_former_v2.run_agh_v2 import all23_config_from_args
    from agh_former_vnext_orthodontic_comparison.run_aghformer_vnext import model_from_args
    import all23_mvsc_refinement.refiner as refiner_module
    from all23_mvsc_refinement import ProcrustesShapePrior, apply_policy, fit_or_load_all23_refinement
    from all23_mvsc_refinement.refiner import dataset_shapes, subset_dataset
    from all23_rgb_geodesic_cascade.data import (
        collate_graphs,
        collate_surface_graphs,
        discover_samples,
        prepare_mesh_record,
    )
    from all23_rgb_geodesic_cascade.model import GlobalCoarseNetwork
    from all23_rgb_geodesic_cascade.run_all23_rgb_geodesic import (
        fold_output_dir,
        make_cv_splits,
        prepare_fold,
        validate_preprocessing_root,
    )
    from all23_rgb_geodesic_cascade.stage1 import _dataset as stage1_dataset
    from all23_rgb_geodesic_cascade.stage1 import _train_template, collect_stage1
    from all23_rgb_geodesic_cascade.train import collect_outputs

    args = proposed_args(device)
    samples = discover_samples(args.data_root)
    preprocessing_root = validate_preprocessing_root(args.preprocessing_root, args)
    splits = make_cv_splits(samples, args.folds, args.val_fraction, args.seed)[fold - 1]
    if sorted(splits["test"]) != sorted(sample_ids):
        raise RuntimeError("Fold test partition differs from the shared manifest")
    preprocessing_dir = fold_output_dir(preprocessing_root, args, 1, fold)
    scratch = Path(tempfile.mkdtemp(prefix="proposed_"))
    datasets, normalizer = prepare_fold(
        samples, splits, args, scratch / "fold", device,
        enforce_oracle_gate=False, preprocessing_dir=preprocessing_dir,
    )
    test_dataset = datasets["test"]
    records, transforms = test_dataset.records, test_dataset.transforms
    by_id = {sample.sample_id: sample for sample in samples}
    test_position = {sample.sample_id: i for i, sample in enumerate(test_dataset.samples)}

    # Stage 1: global network plus the train-template blend of the archived run.
    stage1_dir = Path(preprocessing_dir) / "stage1_global_coarse"
    complete = json.loads((stage1_dir / "complete.json").read_text(encoding="utf-8"))
    blend_alpha = float(complete["outer_train_model"]["template_blend"]["alpha"])
    template = _train_template(splits["train"], records)
    stage1 = GlobalCoarseNetwork(
        input_dim=len(normalizer["mean"]), width=args.stage1_width,
        global_blocks=args.stage1_blocks, heads=args.heads, dropout=args.dropout,
        coordinate_topk=args.stage1_topk, coordinate_temperature=args.stage1_temperature,
    ).to(device)
    stage1.load_state_dict(torch.load(
        stage1_dir / "outer_train_model" / "best_model.pth", map_location=device, weights_only=False
    )["model_state"])
    stage1.eval()

    # Stage 2: anatomical/geodesic network with its learned combination layer.
    stage_a_dir = Path(args.stage_a_run) / f"fold_{fold}"
    stage2 = model_from_args(args, normalizer, device)
    stage2.load_state_dict(torch.load(
        stage_a_dir / "best_model.pth", map_location=device, weights_only=False
    )["model_state"])
    stage2.load_refinement_gate_state_dict(torch.load(
        stage_a_dir / "best_gate.pth", map_location=device, weights_only=False
    )["gate_state"])
    stage2.eval()
    force_refined = not json.loads(
        (stage_a_dir / "gate_training.json").read_text(encoding="utf-8")
    )["accepted"]

    # Shape prior: refitted on the training partition with the locked settings.
    final_dir = PROPOSED_RUN / f"fold_{fold}"
    lock = json.loads((final_dir / "v2_lock.json").read_text(encoding="utf-8"))
    shapes, train_ids, _, _ = dataset_shapes(datasets["train"])
    shape_prior = ProcrustesShapePrior().fit(shapes, train_ids)
    shape_prior.config = dict(lock["shape_prior"]["selected"])
    if not lock["shape_prior"]["accepted"]:
        shape_prior.reject()

    # Refiner ensemble (six members) with the locked per-group policy.
    config = all23_config_from_args(args)
    refinement = fit_or_load_all23_refinement(
        datasets["train"], scratch / "all23_mvsc", config, device,
        scratch_dir=scratch / "refiner_scratch",
        model_dir=Path(args.refiner_source_run) / f"fold_{fold}" / "all23_mvsc",
    )
    policy = json.loads((final_dir / "all23_mvsc" / "all23_policy.json").read_text(encoding="utf-8"))
    refiners = refinement.ensemble.models

    original_extract = refiner_module.extract_all23_set
    original_members = refiner_module.predict_members
    current = {}

    def timed_extract(*a, **k):
        with timed(current, "refiner_cpu_rendering", device):
            return original_extract(*a, **k)

    def timed_members(*a, **k):
        with timed(current, "refiner_network", device):
            return original_members(*a, **k)

    stage_outputs = {"stage1": {}, "stage2": {}, "prior": {}}

    @torch.no_grad()
    def process(sample_id, times):
        work = Path(tempfile.mkdtemp(dir=scratch))
        sample = by_id[sample_id]
        with timed(times, "cpu_mesh_record", device):
            record = prepare_mesh_record(
                sample, transforms[sample_id], work / "mesh_cache", args.max_vertices,
                include_landmarks=False,
            )
        local_records = {**records, sample_id: record}
        with timed(times, "stage1_network", device):
            dataset = stage1_dataset(
                samples, [sample_id], local_records, transforms, normalizer, args,
                False, False, args.seed,
            )
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                collate_fn=collate_surface_graphs)
            coarse = collect_stage1(stage1, loader, device, args)["prediction"][0]
            coarse = template + blend_alpha * (coarse - template)
        # Stage 2 starts from the archived Stage 1 centres, as in the reported run.
        test_dataset.cache_dir = work / "stage2_cache"
        test_dataset._roi_memory.pop(sample_id, None)
        test_dataset._record_memory.pop(sample_id, None)
        position = test_position[sample_id]
        with timed(times, "stage2_cpu_region", device):
            test_dataset[position]
        with timed(times, "stage2_network", device):
            loader = DataLoader(Subset(test_dataset, [position]), batch_size=1, shuffle=False,
                                num_workers=0, collate_fn=collate_graphs)
            stage2_out = collect_outputs(stage2, loader, device, args, normalizer,
                                         use_tta=args.tta, force_refined=force_refined)
            base_a = stage2_out["prediction"][0]
        with timed(times, "shape_prior", device):
            base = shape_prior.transform(base_a[None])[0].astype(np.float32)
        single = subset_dataset(test_dataset, [sample_id])
        single.include_expert = False
        current.clear()
        refiner_module.extract_all23_set = timed_extract
        refiner_module.predict_members = timed_members
        try:
            with timed(times, "refiner_total", device):
                result = refinement.predict(
                    single, {sample_id: base}, scratch_dir=work / "render",
                    cache_dir=work / "refiner_cache", label="cost",
                )
                final = apply_policy(result, policy)["prediction"][0]
        finally:
            refiner_module.extract_all23_set = original_extract
            refiner_module.predict_members = original_members
        times["refiner_cpu_rendering"] = current.get("refiner_cpu_rendering", 0.0)
        times["refiner_network"] = current.get("refiner_network", 0.0)
        times["refiner_other"] = (
            times.pop("refiner_total") - times["refiner_cpu_rendering"] - times["refiner_network"]
        )
        stage_outputs["stage1"][sample_id] = coarse
        stage_outputs["stage2"][sample_id] = base_a
        stage_outputs["prior"][sample_id] = base
        shutil.rmtree(work, ignore_errors=True)
        return final

    reset_peak_memory(device)
    rows, predictions = run_scans(sample_ids, process, warmup)
    checks = {
        "stage1_vs_archived_mm": max_difference(
            stage_outputs["stage1"],
            read_archived(stage1_dir / "predictions_test.csv", sample_ids, "coarse"),
        ),
        "stage2_vs_archived_mm": max_difference(
            stage_outputs["stage2"],
            read_archived(stage_a_dir / "predictions_test.csv", sample_ids, args.stage_a_column),
        ),
        "shape_prior_vs_archived_mm": max_difference(
            stage_outputs["prior"],
            read_archived(final_dir / "predictions_test.csv", sample_ids, "prior"),
        ),
    }
    shutil.rmtree(scratch, ignore_errors=True)
    return {
        "parameters": count_parameters(stage1, stage2, *refiners),
        "parameters_by_stage": {
            "stage1": count_parameters(stage1),
            "stage2": count_parameters(stage2),
            "refiners": count_parameters(*refiners),
            "refiner_members": len(refiners),
        },
        "rows": rows,
        "predictions": predictions,
        "peak_gpu_mb": peak_memory_mb(device),
        "stage_checks": checks,
        "archived_csv": str(final_dir / "predictions_test.csv"),
    }


# ----------------------------------------------------------------------------
def environment(device):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--models", default="pointnet2,diffusionnet,palnet,proposed")
    parser.add_argument("--out", default=str(DRIVE / "cost" / "cost_fold1.json"))
    parser.add_argument("--warmup", type=int, default=WARMUP_SCANS)
    parser.add_argument("--smoke", action="store_true",
                        help="CPU, random weights, two local scans, comparators only")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    if args.smoke:
        device = torch.device("cpu")
        from all23_rgb_geodesic_cascade.data import discover_samples
        sample_ids = [sample.sample_id for sample in discover_samples(ROOT / "data/dataset")][:2]
        models = [name for name in args.models.split(",") if name != "proposed"]
        warmup = 1
        out = Path(tempfile.gettempdir()) / "cost_smoke.json"
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Timing must run on the GPU used for the experiments")
        device = torch.device("cuda")
        sample_ids = fold_test_ids(args.fold)
        models = [name for name in args.models.split(",") if name]
        warmup = args.warmup
        out = Path(args.out)

    measures = {
        "pointnet2": lambda: measure_pointnet2(args.fold, sample_ids, device, args.smoke, warmup),
        "diffusionnet": lambda: measure_diffusionnet(args.fold, sample_ids, device, args.smoke, warmup),
        "palnet": lambda: measure_palnet(args.fold, sample_ids, device, args.smoke, warmup),
        "proposed": lambda: measure_proposed(args.fold, sample_ids, device, warmup),
    }
    report = {"fold": args.fold, "n_scans": len(sample_ids), "warmup_scans": warmup,
              "batch_size": 1, "environment": environment(device), "models": {}}
    for name in models:
        print(f"Measuring {name} on {len(sample_ids)} scans...", flush=True)
        result = measures[name]()
        entry = {
            "parameters": result["parameters"],
            "peak_gpu_mb": result["peak_gpu_mb"],
            "times": summarize_times(result["rows"]),
            "per_scan": result["rows"],
        }
        for key in ("parameters_by_stage", "stage_checks"):
            if key in result:
                entry[key] = result[key]
        if not args.smoke:
            archived_csv = (
                Path(result["archived_csv"]) if "archived_csv" in result
                else COMPARATOR_RUNS[name] / f"fold_{args.fold}" / "predictions_test.csv"
            )
            prefix = "prediction" if name == "proposed" else PREDICTION_PREFIX[name]
            entry["max_difference_to_archived_mm"] = max_difference(
                result["predictions"], read_archived(archived_csv, sample_ids, prefix)
            )
            metrics_path = archived_csv.parent / "metrics.json"
            if name != "proposed" and metrics_path.exists():
                archived_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                entry["archived_parameter_count"] = archived_metrics.get("parameter_count")
        report["models"][name] = entry
        total = entry["times"]["total"]
        print(
            f"  {name}: {entry['parameters'] / 1e6:.2f} M parameters, "
            f"{total['median_s']:.3f} s/scan (IQR {total['q1_s']:.3f}-{total['q3_s']:.3f}), "
            f"peak GPU {entry['peak_gpu_mb']} MB, "
            f"max difference to archived {entry.get('max_difference_to_archived_mm')}",
            flush=True,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
