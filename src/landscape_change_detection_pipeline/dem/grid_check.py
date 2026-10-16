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

#: CRS assumed when a caller does not pass the tile's own (the original AOI's UTM zone).
DEFAULT_CRS = "EPSG:26910"


@dataclass
class GridCheckResult:
    """One tile's grid-alignment report against its DEM."""

    tile_id: str
    expected_crs: Optional[str] = None
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
        lines.append(f"  DEM CRS is the tile CRS ({self.expected_crs or DEFAULT_CRS}): {self.dem_crs_ok}")
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
    expected_crs: Optional[str] = None,
) -> GridCheckResult:
    """Check that ``other_arrays`` share the tile's DEM grid exactly.

    Parameters
    ----------
    other_arrays
        ``{name: (shape, transform, crs_wkt)}`` for every later array to
        check against the DEM (e.g. a scene's TOA grid). Empty/``None``
        checks only that the DEM itself is on the tile's CRS.
    expected_crs
        The tile's CRS (registry ``crs`` column, e.g. ``"EPSG:26909"``); ``None``
        = the original AOI's ``EPSG:26910``.
    """
    result = GridCheckResult(tile_id=tile_id, expected_crs=expected_crs)

    try:
        dem_arr, dem_transform, dem_crs = read_tile_dem(tile_dir, tile_id, "elevation")
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"could not read DEM: {exc}")
        return result

    reference_crs = expected_crs or DEFAULT_CRS
    result.dem_crs_ok = _same_crs(dem_crs, reference_crs)
    dem_shape = tuple(dem_arr.shape)

    for name, (shape, transform, crs_wkt) in (other_arrays or {}).items():
        result.checked_crs_ok[name] = _same_crs(crs_wkt, reference_crs)
        if tuple(shape) != dem_shape:
            result.shape_mismatch.append(f"{name}: {shape} != DEM {dem_shape}")
            continue
        if transform is not None and dem_transform is not None:
            if not _transforms_equal(transform, dem_transform):
                result.transform_mismatch.append(f"{name}: transform differs from DEM")

    return result


def _same_crs(a: str, b: str) -> bool:
    """Whether two CRS strings (WKT, ``EPSG:xxxx``, ...) name the same CRS."""
    if str(a) == str(b):
        return True
    try:
        from pyproj import CRS

        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:  # noqa: BLE001 - not parseable: fall back to the EPSG code appearing in the text
        code = str(b).split(":")[-1].strip()
        return code.isdigit() and code in str(a)


def _transforms_equal(a, b, tol: float = 1e-6) -> bool:
    return all(abs(getattr(a, attr) - getattr(b, attr)) <= tol for attr in "abcdef")
