"""Convert existing per-scene ``features.npz`` caches to the memmap layout.

Each scene dir gets ``features.npy`` + ``labels.npy`` (raw, memory-mappable)
and its ``features.npz`` is rewritten to hold only the metadata. The
``.json`` signature sidecar is untouched, so nothing is regenerated.
Already-converted scenes are skipped; safe to re-run or interrupt.
Scenes are converted in parallel (decompression is CPU-bound).

No CLI: converts ``features.train_root`` and ``training.pseudo_label.pseudo_label_root``
using ``features.convert_workers`` (all from configs/config.yaml).
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from landscape_change_detection_pipeline.features.training_cache import (  # noqa: E402
    CACHE_FILENAME,
    FEATURES_NPY,
    LABELS_NPY,
)

_DATA_KEYS = ("features", "labels")


def convert_scene(scene_dir: Path) -> bool:
    npz_path = scene_dir / CACHE_FILENAME
    with np.load(npz_path, allow_pickle=False) as data:
        if "features" not in data:
            return False  # already converted
        features = data["features"]
        labels = data["labels"]
        meta = {k: data[k] for k in data.files if k not in _DATA_KEYS}
    np.save(scene_dir / FEATURES_NPY, features)
    np.save(scene_dir / LABELS_NPY, labels)
    del features, labels
    tmp = scene_dir / "features.tmp.npz"
    np.savez(tmp, **meta)
    os.replace(tmp, npz_path)
    return True


def main() -> None:
    from landscape_change_detection_pipeline.config import load_config

    config = load_config()
    roots = [Path(config.features.train_root), Path(config.training.pseudo_label.pseudo_label_root)]
    workers = max(1, config.features.convert_workers)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for root in roots:
            dirs = sorted(p.parent for p in root.rglob(CACHE_FILENAME))
            done = 0
            for i, converted in enumerate(pool.map(convert_scene, dirs, chunksize=1), 1):
                done += bool(converted)
                print(f"[convert] {root} [{i}/{len(dirs)}] converted={done}", end="\r", flush=True)
            print(f"\n[convert] {root}: {done} converted, {len(dirs) - done} already done")


if __name__ == "__main__":
    main()
