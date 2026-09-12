"""Download the canonical PlantVillage dataset (color/grayscale/segmented) from HuggingFace.

Writes: data/raw/PlantVillage/data.zip (then extracted to data/raw/PlantVillage/color)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_plantvillage")

REPO_ID = "mohanty/PlantVillage"
FILENAME = "data.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/PlantVillage"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s/%s -> %s", REPO_ID, FILENAME, args.out)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=args.out,
        repo_type="dataset",
        resume_download=True,
    )
    log.info("Downloaded to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())