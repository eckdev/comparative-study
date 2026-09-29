#!/usr/bin/env python3
"""Create and download a compact, analysis-ready PointNet++ CV archive."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_net_orthodontic_comparison.colab_download_diffusionnet_cv_results import (  # noqa: E402
    main as package_cv_results,
)


DEFAULT_RUN_DIR = Path(
    "/content/drive/MyDrive/orthodontic/pointnet2_runs/"
    "pointnet2_publication_cv_seed42"
)


def main(argv=None):
    user_args = list(sys.argv[1:] if argv is None else argv)
    defaults = [
        "--model-name",
        "PointNet++",
        "--metric-key",
        "pointnet2",
        "--run-dir",
        str(DEFAULT_RUN_DIR),
    ]
    return package_cv_results(defaults + user_args)


if __name__ == "__main__":
    sys.exit(main())
