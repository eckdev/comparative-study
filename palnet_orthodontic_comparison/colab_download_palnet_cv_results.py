#!/usr/bin/env python3
"""Create and download a compact, analysis-ready PAL-Net CV archive."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_net_orthodontic_comparison.colab_download_diffusionnet_cv_results import (  # noqa: E402
    main as package_cv_results,
)


DEFAULT_RUN_DIR = Path(
    "/content/drive/MyDrive/orthodontic/palnet_runs/"
    "palnet_publication_cv_seed42"
)


def main(argv=None):
    user_args = list(sys.argv[1:] if argv is None else argv)
    defaults = [
        "--model-name",
        "PAL-Net",
        "--metric-key",
        "palnet_snapped",
        "--run-dir",
        str(DEFAULT_RUN_DIR),
    ]
    return package_cv_results(defaults + user_args)


if __name__ == "__main__":
    sys.exit(main())
