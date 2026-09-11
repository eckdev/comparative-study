#!/usr/bin/env python3
"""Create a blank, versioned curve-annotation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from all23_rgb_geodesic_cascade.data import discover_samples

from curve_supervised_hard3_refinement.annotations import blank_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    samples = discover_samples(args.data_root)
    payload = blank_manifest(sample.sample_id for sample in samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Blank manifest for {len(samples)} samples: {output}", flush=True)


if __name__ == "__main__":
    main()
