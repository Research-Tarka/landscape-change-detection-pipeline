#!/usr/bin/env python3
"""Generate configs/maskforge_session.json (MaskForge session) from configs/classes.yaml.

The session file holds local absolute paths, so it is gitignored and must be
regenerated on each machine.

Usage
-----
    python scripts/export_maskforge_palette.py
    python scripts/export_maskforge_palette.py --classes configs/classes.yaml \
        --output configs/maskforge_session.json
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from landscape_change_detection_pipeline.classes.class_config import (  # noqa: E402
    load_class_config,
    to_maskforge_palette,
)


def build_session(palette: dict, name: str) -> dict:
    return {
        "schema_version": 1,
        "id": str(uuid.uuid4()),
        "name": name,
        "discovery": {
            "source_root": str(ROOT / "data" / "tiles"),
            "scan_rule": {
                "name": "default",
                "rgb_patterns": [],
                "mask_patterns": [],
                "inference_patterns": ["*class_map*", "*inference*"],
                "max_depth": 4,
                "file_extensions": [".tif", ".png", ".zarr"],
            },
            "exclude_globs": [],
            "inference_root": str(ROOT / "outputs" / "inference"),
        },
        "active_palette_id": palette["id"],
        "active_class_ids": [c["id"] for c in palette["classes"] if c.get("active_by_default", True)],
        "save_config": {
            "output_format": "geotiff_rgba",
            "output_root": "",
            "folder_structure_template": "{scene_id}/mask.tif",
            "copy_raw": False,
            "copy_shadow": False,
            "preserve_georef": True,
            "resolution_mode": "native",
            "target_resolution": None,
            "compress": "deflate",
        },
        "training_save_config": None,
        "ui_state": {},
        "qa_state": {},
        "recent_sessions": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", default="configs/classes.yaml", help="Path to classes.yaml")
    parser.add_argument("--output", default="configs/maskforge_session.json", help="Output session JSON")
    parser.add_argument("--name", default="landscape-change-detection-pipeline", help="Session name")
    args = parser.parse_args(argv)

    config = load_class_config(args.classes)
    palette = to_maskforge_palette(config, args.name)
    session = build_session(palette, args.name)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    print(f"Wrote session with {len(session['active_class_ids'])} classes to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
