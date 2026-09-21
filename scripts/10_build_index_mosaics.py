#!/usr/bin/env python3
"""AOI-wide spectral-index mosaics, for change-detection analysis.

Purpose
-------
The continuous-index counterpart of Stage 8: crops every tile's monthly
index composite (Stage 9) down to that tile's own unbuffered
``search_bbox`` and places the crops side by side into one AOI-wide,
multi-band raster per period. Stored as ``.npz`` like every other
intermediate array in this pipeline; use
``scripts/export_gui.py`` on demand for a GIS-readable file. Output:
``outputs/index_mosaics/<year>-<month>/indices.npz``.

Usage
-----
    python scripts/10_build_index_mosaics.py --config configs/config.yaml
    python scripts/10_build_index_mosaics.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.config import load_config  # noqa: E402
from landscape_change_detection_pipeline.change.index_mosaic import build_all_period_index_mosaics  # noqa: E402
from landscape_change_detection_pipeline.tiles.registry import read_registry  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--env-file", default=None, help="Path to a .env file")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv if argv is not None else sys.argv[1:]))
    config = load_config(args.config, args.env_file)

    registry = read_registry(config.tiles.registry_path)

    written = build_all_period_index_mosaics(
        index_composites_root=config.index_composites.output_root,
        output_root=config.index_mosaic.output_root,
        registry=registry,
        overwrite=args.overwrite,
    )
    for path in written:
        print(f"[index_mosaic] wrote {path}")
    print(f"[index_mosaic] {len(written)} index mosaics written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
