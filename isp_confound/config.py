"""Path configuration for the toolkit scripts.

Scripts are run from the project root and, by default, read datasets from
``./data``. Relocate the data root with an environment variable::

    set ISP_DATA_ROOT=D:\\datasets\\scperturb      # Windows
    export ISP_DATA_ROOT=/mnt/data/scperturb        # Unix

The Geneformer clone is expected at ``./Geneformer`` (or installed as the
``geneformer`` package). Full per-script path configuration is intentionally
out of scope; this is the minimal hook for relocating the data root.
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("ISP_DATA_ROOT", "data"))
