import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class OrthodonticSample:
    mesh_path: Path
    landmark_path: Path
    class_name: str
    gender: str
    subject_id: int

    @property
    def sample_id(self):
        prefix = "M" if self.gender == "men" else "F"
        return f"{self.class_name}_{prefix}{self.subject_id}"


def discover_samples(root_dir):
    root_dir = Path(root_dir)
    samples = []
    missing = []
    for class_dir in sorted(root_dir.glob("Class*")):
        if not class_dir.is_dir():
            continue
        landmark_root = class_dir / f"{class_dir.name}-Landmark"
        for gender, prefix in (("men", "M"), ("women", "F")):
            mesh_dir = class_dir / gender
            landmark_dir = landmark_root / gender
            if not mesh_dir.exists():
                continue
            for mesh_path in sorted(mesh_dir.glob("*.ply"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
                if not mesh_path.stem.isdigit():
                    continue
                subject_id = int(mesh_path.stem)
                landmark_path = landmark_dir / f"{class_dir.name}_{prefix}{subject_id}.txt"
                if landmark_path.exists():
                    samples.append(OrthodonticSample(mesh_path, landmark_path, class_dir.name, gender, subject_id))
                else:
                    missing.append(str(mesh_path))
    return samples, missing


def make_splits(samples, test_size, val_size, seed):
    indices = np.arange(len(samples))
    strata = np.array([f"{sample.class_name}_{sample.gender}" for sample in samples])
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


def group_counts(samples, split_indices):
    counts = defaultdict(Counter)
    for split_name, indices in split_indices.items():
        for idx in indices:
            sample = samples[idx]
            counts[f"{sample.class_name}_{sample.gender}"][split_name] += 1
    return {group: dict(counter) for group, counter in sorted(counts.items())}


def main():
    parser = argparse.ArgumentParser(description="Create one shared train/val/test split for all orthodontic models.")
    parser.add_argument("--data-root", default="data/dataset")
    parser.add_argument("--output", default="shared_splits/orthodontic_180_60_60_seed42.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    samples, missing = discover_samples(args.data_root)
    if not samples:
        raise RuntimeError(f"No paired samples found under {args.data_root}")

    train_idx, val_idx, test_idx = make_splits(samples, args.test_size, args.val_size, args.seed)
    payload = {
        "name": f"orthodontic_{len(train_idx)}_{len(val_idx)}_{len(test_idx)}_seed{args.seed}",
        "data_root": str(Path(args.data_root)),
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "stratification": "class_gender",
        "n_total": len(samples),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train": [samples[i].sample_id for i in train_idx],
        "val": [samples[i].sample_id for i in val_idx],
        "test": [samples[i].sample_id for i in test_idx],
        "groups": group_counts(samples, {"train": train_idx, "val": val_idx, "test": test_idx}),
        "missing_landmarks": missing,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Total/train/val/test: {len(samples)}/{len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    if missing:
        print(f"Missing landmarks: {len(missing)}")


if __name__ == "__main__":
    main()
