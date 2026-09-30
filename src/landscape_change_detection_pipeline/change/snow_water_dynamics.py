"""Per-pixel, per-water-year snow, ice and open-water summaries.

Purpose
-------
:mod:`change.change_detection` (Stage 11) deliberately excludes snow/ice/
water observations from its vegetation-trend model (mixing "is this pixel
under snow this month" into a NBR/NDVI harmonic-trend fit would read every
snow onset/melt as a spurious break -- see that module's own docstring).
That does not mean snow/ice/water are discarded: they are a real, different
kind of signal, tracked here directly from the classifier's own monthly
class -- the thing this pipeline already computes to decide "is this pixel
snow this month," not re-derived from a spectral index.

Source: Stage 7's monthly class composites
--------------------------------------------
Reads ``composites.output_root``'s monthly per-tile class composites (Stage
7), never raw scenes -- Stage 7 already reduces a month's scenes to one
class per pixel with the right per-class rule for exactly this purpose
(``configs/config.yaml``'s ``composites.class_rules``: ``ice_cover``/
``snow_cover`` use ``fallback_occurrence``, so a single snowy scene does not
wrongly cover a mostly-snow-free month, and a mostly-snow month is not
masked by one rare clear scene).

Water year, not calendar year
--------------------------------
Grouped into water years starting ``water_year_start_month`` (default
October, Water Survey of Canada's standard water-year start) so one winter
is never split across two summary rows -- a calendar-year grouping would
put a season's December in one year and its February in the next.

What is computed, per pixel per water year
----------------------------------------------
From every stored month of that water year (a tile can have gaps -- no
scene that month, or Stage 7 not yet run on it -- so ``n_months_observed``
records the real denominator, never assumed to be 12):

- ``n_months_snow``/``n_months_ice``/``n_months_water`` -- how many observed
  months this pixel's dominant class was ``snow_cover``/``ice_cover``/
  ``open_water``.
- ``first_snow_month``/``last_snow_month`` -- the water-year-relative month
  index (1 = the water year's first month) of the first/last observed snow
  month, ``-1`` if none observed. A snow-free-season proxy follows directly
  from these two plus ``n_months_observed``.

Grid
----
A tile's monthly composites can each sit on a slightly different grid (the
month's own finest-available-sensor grid -- see ``inference.composites``),
almost always the same unified 10 m grid in practice but not guaranteed for
data predating that revision. Every month is reprojected (nearest-neighbour
-- categorical class ids) onto the first available month's grid for the
tile, matching the categorical-reprojection discipline used everywhere else
in this pipeline (``mosaic.mosaic``, ``change.change_detection``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from landscape_change_detection_pipeline.classes.class_config import ClassConfig
from landscape_change_detection_pipeline.inference.composites import composite_output_path, read_composite

NODATA_MONTH = -1  # sentinel for "no snow observed this water year" in first/last_snow_month


def water_year_of(year: int, month: int, start_month: int) -> int:
    """The water year a calendar ``(year, month)`` belongs to, labelled by
    its starting calendar year (e.g. water year 2020 with
    ``start_month=10`` runs October 2020 - September 2021)."""
    return year if month >= start_month else year - 1


def water_year_month_index(month: int, start_month: int) -> int:
    """1-indexed position of calendar ``month`` within its water year (1 =
    ``start_month`` itself)."""
    return (month - start_month) % 12 + 1


def discover_water_years(periods: list[str], start_month: int) -> dict[int, list[str]]:
    """Group ``"YYYY-MM"`` periods by water year."""
    groups: dict[int, list[str]] = {}
    for period in periods:
        year, month = (int(v) for v in period.split("-"))
        wy = water_year_of(year, month, start_month)
        groups.setdefault(wy, []).append(period)
    return {wy: sorted(months) for wy, months in groups.items()}


def _reproject_nearest(array: np.ndarray, src_transform, src_crs_wkt: str, dst_transform, dst_crs_wkt: str,
                       dst_shape: tuple[int, int], fill_value: int) -> np.ndarray:
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    destination = np.full(dst_shape, fill_value, dtype=np.uint8)
    reproject(
        source=array,
        destination=destination,
        src_transform=Affine(*src_transform[:6]),
        src_crs=CRS.from_user_input(src_crs_wkt),
        dst_transform=Affine(*dst_transform[:6]),
        dst_crs=CRS.from_user_input(dst_crs_wkt),
        resampling=Resampling.nearest,
    )
    return destination


def snow_water_dynamics_output_path(output_root: str | Path, tile_id: str) -> Path:
    """``<output_root>/<tile_id>/snow_water_dynamics.npz``."""
    return Path(output_root) / tile_id / "snow_water_dynamics.npz"


def build_tile_snow_water_dynamics(
    composites_root: str | Path,
    output_root: str | Path,
    tile_id: str,
    class_config: ClassConfig,
    water_year_start_month: int = 10,
    overwrite: bool = False,
) -> "Path | None":
    """Build one tile's per-water-year snow/ice/water summary from its
    Stage 7 monthly class composites. Returns the written path, or ``None``
    if already built and ``overwrite`` is false, or if the tile has no
    stored composites at all."""
    from landscape_change_detection_pipeline.change.spectral_composites import discover_periods_for_tile

    out_path = snow_water_dynamics_output_path(output_root, tile_id)
    if out_path.is_file() and not overwrite:
        return None

    periods = discover_periods_for_tile(composites_root, tile_id, filename="composite.npz")
    if not periods:
        return None

    snow_id = class_config.by_name("snow_cover").id
    ice_id = class_config.by_name("ice_cover").id
    water_id = class_config.by_name("open_water").id

    reference = read_composite(composite_output_path(composites_root, tile_id, periods[0]))
    dst_transform, dst_crs_wkt = reference["transform"], reference["crs_wkt"]
    dst_shape = reference["composite"].shape

    by_water_year = discover_water_years(periods, water_year_start_month)
    water_years = sorted(by_water_year)
    n_years = len(water_years)
    height, width = dst_shape

    n_observed = np.zeros((n_years, height, width), dtype=np.uint8)
    n_snow = np.zeros((n_years, height, width), dtype=np.uint8)
    n_ice = np.zeros((n_years, height, width), dtype=np.uint8)
    n_water = np.zeros((n_years, height, width), dtype=np.uint8)
    first_snow = np.full((n_years, height, width), NODATA_MONTH, dtype=np.int8)
    last_snow = np.full((n_years, height, width), NODATA_MONTH, dtype=np.int8)

    for yi, wy in enumerate(water_years):
        for period in by_water_year[wy]:
            year, month = (int(v) for v in period.split("-"))
            composite = read_composite(composite_output_path(composites_root, tile_id, period))
            class_map = composite["composite"]
            if class_map.shape != dst_shape or composite["transform"] != dst_transform:
                class_map = _reproject_nearest(
                    class_map, composite["transform"], composite["crs_wkt"],
                    dst_transform, dst_crs_wkt, dst_shape, fill_value=composite["nodata"],
                )
            valid = class_map != composite["nodata"]
            n_observed[yi] += valid

            is_snow = valid & (class_map == snow_id)
            n_snow[yi] += is_snow
            n_ice[yi] += valid & (class_map == ice_id)
            n_water[yi] += valid & (class_map == water_id)

            month_idx = water_year_month_index(month, water_year_start_month)
            first_unset = is_snow & (first_snow[yi] == NODATA_MONTH)
            first_snow[yi][first_unset] = month_idx
            last_snow[yi][is_snow] = month_idx  # periods within a water year are visited in order

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        water_years=np.array(water_years, dtype=np.int32),
        n_months_observed=n_observed,
        n_months_snow=n_snow,
        n_months_ice=n_ice,
        n_months_water=n_water,
        first_snow_month=first_snow,
        last_snow_month=last_snow,
        water_year_start_month=np.array(water_year_start_month, dtype=np.int8),
        transform=np.array(list(dst_transform)[:6], dtype=np.float64),
        crs_wkt=np.array(dst_crs_wkt),
        resolution_m=np.array(abs(dst_transform[0]), dtype=np.float64),
    )
    return out_path


def read_snow_water_dynamics(path: str | Path) -> dict:
    """Read a tile's ``snow_water_dynamics.npz`` back."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {
            "water_years": np.array(data["water_years"]),
            "n_months_observed": np.array(data["n_months_observed"]),
            "n_months_snow": np.array(data["n_months_snow"]),
            "n_months_ice": np.array(data["n_months_ice"]),
            "n_months_water": np.array(data["n_months_water"]),
            "first_snow_month": np.array(data["first_snow_month"]),
            "last_snow_month": np.array(data["last_snow_month"]),
            "water_year_start_month": int(data["water_year_start_month"]),
            "transform": tuple(float(v) for v in data["transform"]),
            "crs_wkt": str(data["crs_wkt"]),
            "resolution_m": float(data["resolution_m"]),
        }
