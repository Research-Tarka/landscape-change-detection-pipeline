"""Per-tile DEM extraction orchestration.

Purpose
-------
For each tile in the registry (:mod:`landscape_change_detection_pipeline.tiles.registry`):
fetch the best available source DEM over the tile's buffered analysis window
(MRDEM-30 DTM, else Copernicus GLO-30 -- :mod:`.sources`), orient/mask/derive
slope and aspect (:mod:`.processing`), and write everything into the tile's
zarr store (:mod:`.zarr_store`).

Inputs
------
- The tile registry (:mod:`landscape_change_detection_pipeline.tiles.registry`).

Outputs
-------
- ``<data_root>/tiles/<tile_id>.zarr`` -- ``dem`` group holding ``elevation``,
  ``slope``, ``aspect`` (float32, EPSG:26910, NaN nodata) plus provenance
  attrs.
- ``<data_root>/tiles/<tile_id>.dem.done`` -- completion flag (enables resume).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np

from .processing import build_tile_dem_products, elevation_stats, valid_fraction_pct
from .sources import TARGET_EPSG, load_best_dem
from .zarr_store import write_tile_dem

DONE_SUFFIX = ".dem.done"


def process_one_tile(
    tile_dir: Path,
    tile_id: str,
    window_bounds: tuple[float, float, float, float],
    overwrite: bool = False,
    copernicus_fallback: bool = True,
    force_copernicus: bool = False,
) -> tuple[str, str]:
    """Extract and write the DEM products for one tile.

    Returns ``(tile_id, status)`` where status is ``"ok"``, ``"skip (...)"``
    or ``"error (...)"``. Never raises: failures are reported per tile so a
    sweep over ~1000+ tiles is not aborted by one bad window.
    """
    tile_dir = Path(tile_dir)
    done_flag = tile_dir / f"{tile_id}{DONE_SUFFIX}"
    try:
        if not overwrite and done_flag.exists():
            return tile_id, "skip (already done)"

        dem, source_label, base_res, valid_pct = load_best_dem(
            window_bounds,
            src_crs=TARGET_EPSG,
            use_copernicus_fallback=copernicus_fallback,
            force_copernicus=force_copernicus,
        )
        if dem is None:
            return tile_id, "skip (no DEM coverage from MRDEM-30 or Copernicus GLO-30)"

        products = build_tile_dem_products(dem, window_bounds)
        elev_valid_pct = valid_fraction_pct(products["elevation"])
        if elev_valid_pct < 1.0:
            return tile_id, f"skip (masked window is empty: {elev_valid_pct:.2f}% valid)"

        transform = products["elevation"].rio.transform(recalc=False)
        crs_wkt = products["elevation"].rio.crs.to_wkt()

        arrays = {name: np.asarray(da.values, dtype=np.float32) for name, da in products.items()}

        sidecar = {
            "tile_id": tile_id,
            "source_dem": source_label,
            "source_base_resolution_m": base_res,
            "crs": f"EPSG:{TARGET_EPSG}",
            "window_bounds": list(window_bounds),
            "source_window_valid_pct": round(valid_pct, 2),
            "elevation_valid_pct": round(elev_valid_pct, 2),
            "elevation_stats": elevation_stats(products["elevation"]),
        }
        write_tile_dem(tile_dir, tile_id, arrays, transform, crs_wkt, attrs=sidecar)

        done_flag.write_text(
            json.dumps({"source_dem": source_label, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}),
            encoding="utf-8",
        )
        return tile_id, "ok"

    except Exception as exc:  # noqa: BLE001 - reported per tile, never fatal
        tb = traceback.format_exc().splitlines()
        where = tb[-2].strip() if len(tb) >= 2 else "?"
        return tile_id, f"error ({type(exc).__name__}: {exc} @ {where})"


def run_dem_extraction(
    registry,
    tile_dir: Path,
    tile_ids: Optional[list[str]] = None,
    overwrite: bool = False,
    copernicus_fallback: bool = True,
    force_copernicus: bool = False,
) -> list[tuple[str, str]]:
    """Run DEM extraction over every tile in ``registry`` (a tile_registry DataFrame).

    ``tile_ids`` restricts the run to a subset (e.g. a pilot run), otherwise
    every row in ``registry`` is processed. Sequential by default: each fetch
    is one or two remote HTTP range-request reads, not a hot loop, and the
    STAC/GEE/AWS backends already retry internally (see
    ``sources.configure_gdal_for_public_cogs``); parallelising is left to the
    caller (e.g. via a thread pool) rather than built in here, since a pilot
    run over a handful of tiles does not need it.
    """
    from tqdm import tqdm

    rows = registry if tile_ids is None else registry[registry["tile_id"].isin(tile_ids)]
    results = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="DEM extraction"):
        bounds = (
            float(row["window_minx"]),
            float(row["window_miny"]),
            float(row["window_maxx"]),
            float(row["window_maxy"]),
        )
        results.append(
            process_one_tile(
                tile_dir,
                str(row["tile_id"]),
                bounds,
                overwrite=overwrite,
                copernicus_fallback=copernicus_fallback,
                force_copernicus=force_copernicus,
            )
        )
    return results
