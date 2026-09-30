"""Export everything the pipeline generated, in one go.

Purpose
-------
:mod:`export.geotiff` and :mod:`export.csv_export` convert one file at a time;
this module walks a whole output folder and applies them to every ``.npz``
(and every per-tile DEM zarr): each raster product becomes a GeoTIFF, each
table a CSV. A file that is only a raster gets no CSV and a file that is only
a table gets no GeoTIFF -- that is a fact about the file, recorded as
``"skipped"`` rather than an error. Used by the "Export everything" panel of
``scripts/export_gui.py``.

Layout of the result: ``<out_dir>/<same relative path as the source>``, the
``.npz`` stem becoming ``<stem>.tif`` (or ``<stem>__<dtype>.tif`` when a file
mixes dtypes) and ``<stem>.csv`` (or ``<stem>__<table>.csv`` when a file holds
several tables).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from landscape_change_detection_pipeline.export.csv_export import list_csv_tables, write_table_csv
from landscape_change_detection_pipeline.export.geotiff import (
    GeoTiffExportError,
    build_dem_export_payload,
    export_to_geotiff,
    write_geotiff,
)

#: Folders that hold temporary working files, never products.
_SKIP_DIRS = ("_work",)


def find_npz_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    return sorted(p for p in root_path.rglob("*.npz") if not any(part in _SKIP_DIRS for part in p.parts))


def find_dem_tiles(root: str | Path) -> list[tuple[str, str]]:
    """Every ``<tile_id>.zarr`` under ``root`` that has a ``dem`` group, as ``(zarr_dir, tile_id)``."""
    import zarr

    root_path = Path(root)
    if not root_path.is_dir():
        return []
    found = []
    for zarr_dir in sorted(root_path.rglob("*.zarr")):
        try:
            store = zarr.open_group(str(zarr_dir), mode="r")
        except Exception:  # noqa: BLE001 -- a partially written/foreign store must not crash the scan
            continue
        if "dem" in store:
            found.append((str(zarr_dir.parent), zarr_dir.stem))
    return found


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", name).strip("_")[:80] or "table"


def _as_list(value) -> list[Path]:
    return [Path(p) for p in (value if isinstance(value, list) else [value])]


def _tif_exists(target: Path) -> bool:
    """A GeoTIFF export exists as ``<stem>.tif`` or, for a file mixing dtypes, ``<stem>__<dtype>.tif``."""
    return target.is_file() or (target.parent.is_dir() and any(target.parent.glob(f"{target.stem}__*.tif")))


def export_one(npz_path: Path, out_base: Path, geotiff: bool = True, csv: bool = True, skip_existing: bool = False) -> list[dict]:
    """Export one ``.npz`` to ``out_base`` (no suffix). Returns one record per
    attempted output: ``{"file", "kind", "status" ok|skipped|error, "paths"|"detail"}``."""
    records: list[dict] = []
    if geotiff:
        target = out_base.with_suffix(".tif")
        try:
            if skip_existing and _tif_exists(target):
                records.append({"file": str(npz_path), "kind": "geotiff", "status": "skipped", "detail": "exists"})
            else:
                written = export_to_geotiff(npz_path, target)
                records.append({"file": str(npz_path), "kind": "geotiff", "status": "ok", "paths": [str(p) for p in _as_list(written)]})
        except GeoTiffExportError as exc:
            records.append({"file": str(npz_path), "kind": "geotiff", "status": "skipped", "detail": f"no raster: {exc}"})
        except Exception as exc:  # noqa: BLE001 -- one broken file must not stop the batch
            records.append({"file": str(npz_path), "kind": "geotiff", "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
    if csv:
        try:
            tables = list_csv_tables(npz_path)
        except Exception as exc:  # noqa: BLE001
            tables = []
            records.append({"file": str(npz_path), "kind": "csv", "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
        else:
            if not tables:
                records.append({"file": str(npz_path), "kind": "csv", "status": "skipped", "detail": "no table"})
        paths = []
        for table in tables:
            target = out_base.with_suffix(".csv") if len(tables) == 1 else out_base.with_name(f"{out_base.name}__{_safe(table['name'])}.csv")
            try:
                if skip_existing and target.is_file():
                    continue
                paths.append(str(write_table_csv(table, target)))
            except Exception as exc:  # noqa: BLE001
                records.append({"file": str(npz_path), "kind": "csv", "status": "error", "detail": f"{table['name']}: {type(exc).__name__}: {exc}"})
        if paths:
            records.append({"file": str(npz_path), "kind": "csv", "status": "ok", "paths": paths})
    return records


def export_all(
    root: str | Path,
    out_dir: str | Path,
    geotiff: bool = True,
    csv: bool = True,
    include_dem: bool = True,
    name_filter: str = "",
    skip_existing: bool = False,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> list[dict]:
    """Export every ``.npz`` under ``root`` (whose relative path contains
    ``name_filter``) -- and, if ``include_dem``, every tile DEM -- into ``out_dir``.
    ``progress(done, total, label)`` is called after each file."""
    root_path, out_path = Path(root), Path(out_dir)
    files = [p for p in find_npz_files(root_path) if name_filter.lower() in str(p.relative_to(root_path)).lower()]
    dems = find_dem_tiles(root_path) if include_dem and geotiff else []
    dems = [(d, t) for d, t in dems if name_filter.lower() in t.lower()]
    total = len(files) + len(dems)
    records: list[dict] = []
    done = 0
    for npz in files:
        rel = npz.relative_to(root_path)
        records += export_one(npz, out_path / rel.with_suffix(""), geotiff, csv, skip_existing)
        done += 1
        if progress:
            progress(done, total, str(rel))
    for tile_dir, tile_id in dems:
        target = out_path / "dem" / f"{tile_id}_dem.tif"
        try:
            if skip_existing and _tif_exists(target):
                records.append({"file": f"dem::{tile_id}", "kind": "geotiff", "status": "skipped", "detail": "exists"})
            else:
                written = write_geotiff(build_dem_export_payload(tile_dir, tile_id), target)
                records.append({"file": f"dem::{tile_id}", "kind": "geotiff", "status": "ok", "paths": [str(p) for p in _as_list(written)]})
        except Exception as exc:  # noqa: BLE001
            records.append({"file": f"dem::{tile_id}", "kind": "geotiff", "status": "error", "detail": f"{type(exc).__name__}: {exc}"})
        done += 1
        if progress:
            progress(done, total, f"[DEM] {tile_id}")
    return records
