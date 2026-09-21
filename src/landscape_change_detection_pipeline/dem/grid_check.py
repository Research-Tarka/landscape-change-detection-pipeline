"""Grid-alignment invariant: every later scene for a tile must match its DEM grid.

Purpose
-------
The DEM stage fixes a tile's pixel grid (shape, transform, CRS). Every later
per-tile array (sensor scenes, feature stacks, labels) must land on that
*exact* same grid -- not "close enough" -- so bands stack without an implicit
resample. This module makes that invariant checkable rather than assumed:
an unstated assumption about
grid alignment is exactly the kind of bug that stays invisible until a model
trains on misregistered inputs.

This pipeline's DEM is fetched directly on the tile's buffered
analysis window -- the same window every later stage requests. So the
invariant here is strict and simple: shape and transform must match
*exactly*, not just fall within a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .zarr_store import read_tile_dem


@dataclass
class GridCheckResult:
    """One tile's grid-alignment report against its DEM."""

    tile_id: str
    dem_crs_ok: bool = True
    checked_crs_ok: dict[str, bool] = field(default_factory=dict)
    shape_mismatch: list[str] = field(default_factory=list)
    transform_mismatch: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if self.errors or self.shape_mismatch or self.transform_mismatch:
            return False
        if not self.dem_crs_ok or not all(self.checked_crs_ok.values()):
            return False
        return True

    def report(self) -> str:
        lines = [f"Grid check for {self.tile_id}: {'OK' if self.ok else 'FAILED'}"]
        lines.append(f"  DEM CRS is EPSG:26910: {self.dem_crs_ok}")
        for name, ok in sorted(self.checked_crs_ok.items()):
            lines.append(f"  {name} CRS matches DEM: {ok}")
        if self.shape_mismatch:
            lines.append(f"  Shape mismatches: {self.shape_mismatch}")
        if self.transform_mismatch:
            lines.append(f"  Transform mismatches: {self.transform_mismatch}")
        if self.errors:
            lines.append(f"  Errors: {self.errors}")
        return "\n".join(lines)


def check_grid_alignment(
    tile_id: str,
    tile_dir,
    other_arrays: Optional[dict[str, tuple[tuple[int, int], object, str]]] = None,
) -> GridCheckResult:
    """Check that ``other_arrays`` share the tile's DEM grid exactly.

    Parameters
    ----------
    other_arrays
        ``{name: (shape, transform, crs_wkt)}`` for every later array to
        check against the DEM (e.g. a scene's TOA grid). Empty/``None``
        checks only that the DEM itself is on the working CRS.
    """
    result = GridCheckResult(tile_id=tile_id)

    try:
        dem_arr, dem_transform, dem_crs = read_tile_dem(tile_dir, tile_id, "elevation")
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"could not read DEM: {exc}")
        return result

    result.dem_crs_ok = "26910" in dem_crs
    dem_shape = tuple(dem_arr.shape)

    for name, (shape, transform, crs_wkt) in (other_arrays or {}).items():
        result.checked_crs_ok[name] = "26910" in crs_wkt
        if tuple(shape) != dem_shape:
            result.shape_mismatch.append(f"{name}: {shape} != DEM {dem_shape}")
            continue
        if transform is not None and dem_transform is not None:
            if not _transforms_equal(transform, dem_transform):
                result.transform_mismatch.append(f"{name}: transform differs from DEM")

    return result


def _transforms_equal(a, b, tol: float = 1e-6) -> bool:
    return all(abs(getattr(a, attr) - getattr(b, attr)) <= tol for attr in "abcdef")
