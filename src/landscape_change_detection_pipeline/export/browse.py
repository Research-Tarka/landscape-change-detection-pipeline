"""Cheap, lazy helpers for ``scripts/export_gui.py``: classify a stored ``.npz``
by name, list its arrays *without loading them*, and render a georeferenced
preview of one band.

Purpose
-------
The GUI must never read a whole output tree into RAM just to show a list.
Here a file is only described from its ``.npy`` headers (a few bytes per
array); a band is only decoded when previewed, and always decimated to screen
size first.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

#: Band-name stems that hold *dense* class ids (position in ``classes.yaml``).
CLASS_BAND_STEMS = ("class_map", "composite", "mosaic", "from_class", "to_class")

_MAX_PREVIEW_PX = 1400



def product_type(path: str | Path) -> str:
    """Product type of a stored file, from its name only (no I/O)."""
    stem = Path(path).stem
    if stem.startswith("monthly_anomaly_"):
        return "monthly_anomaly"
    if stem.startswith("regrowth_severity"):
        return "regrowth_severity"
    if stem.startswith("block_"):
        return "block"
    return stem


def describe_npz(path: str | Path) -> list[dict]:
    """``[{"key", "shape", "dtype", "mb"}]`` for every array of an ``.npz``, read from
    the ``.npy`` headers only -- nothing is decompressed."""
    out = []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with z.open(info) as f:
                version = np.lib.format.read_magic(f)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _fortran, dtype = reader(f)
            out.append({
                "key": info.filename[:-4], "shape": tuple(int(v) for v in shape),
                "dtype": str(dtype), "mb": round(int(np.prod(shape, dtype=np.int64)) * dtype.itemsize / 2**20, 2),
            })
    return out


def load_palette(classes_yaml: str | Path) -> Optional[dict[int, tuple[int, int, int]]]:
    """``{dense_id: (r, g, b)}`` and names from ``classes.yaml``, or ``None`` if unreadable."""
    try:
        from landscape_change_detection_pipeline.classes.class_config import load_class_config

        cfg = load_class_config(str(classes_yaml))
        return {i: (c.color, c.display_name) for i, c in enumerate(cfg.classes)}
    except Exception:  # noqa: BLE001 -- previews fall back to a generic palette
        return None


def _decimate(array: np.ndarray) -> tuple[np.ndarray, int]:
    step = max(1, int(np.ceil(max(array.shape[-2:]) / _MAX_PREVIEW_PX)))
    return array[::step, ::step], step


# ── Labels (English) ──────────────────────────────────────────────────────────
#: Event types (order of ``event_typing.EVENT_NAMES``).
EVENT_LABELS = ("None", "Fire", "Cutblock", "Permanent clearing", "Canopy decline", "Other loss", "Gain")


def type_label(product: str) -> str:
    """Readable name of a result type."""
    return "DEM" if product == "[DEM]" else product.replace("_", " ").capitalize()


def band_label(band: str) -> str:
    """Readable band / table / column name: ``latest_event_type`` -> "Latest event type",
    ``ndvi__2019`` -> "Ndvi - 2019"."""
    parts = band.split("__")
    head = parts[0].replace("_", " ")
    head = head[:1].upper() + head[1:]
    return head if len(parts) == 1 else f"{head} - {' '.join(parts[1:])}"


def _unit(band: str) -> str:
    low = band.lower()
    for key, unit in (("yyyymm", "YYYYMM"), ("year", "year"), ("duration_months", "months"), ("n_segments", "number of segments"),
                      ("n_months", "number of months"), ("n_intervals", "number of intervals"), ("n_", "count"),
                      ("month", "month"), ("dnbr", "dNBR"), ("ndvi", "NDVI"), ("slope", "slope"), ("elevation", "m")):
        if key in low:
            return unit
    return "value"


_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
           "October", "November", "December")
_FLAG_BASES = ("truncated", "cropland", "still_active")
_FLAG_PREFIXES = ("has_", "changed_", "is_")


def _short(name: str) -> str:
    """Class names in classes.yaml carry long explanations: keep the part before ' (' / ' --'."""
    return name.split(" (")[0].split(" --")[0].strip()


def _named_labels(band: str, palette: Optional[dict]) -> Optional[dict[int, tuple]]:
    """``{id: (rgb or None, libellé)}`` when the band has a known meaning, else ``None``
    (the band is then shown as a gradient with its unit)."""
    base = band.split("__")[0]
    if base in CLASS_BAND_STEMS and palette:
        return {i: (tuple(c), _short(str(n))) for i, (c, n) in palette.items()}
    if "event_type" in base:
        return {i: (None, n) for i, n in enumerate(EVENT_LABELS)}
    if "recovery_state" in base:
        return {i: (None, n) for i, n in enumerate(("No event", "Recovered", "Not recovered yet", "No baseline"))}
    if base.endswith("_month") and "snow" in base:
        return {**{-1: (None, "No month")}, **{i + 1: (None, m) for i, m in enumerate(_MONTHS)}}
    if base in _FLAG_BASES or band.endswith("__still_active") or base.startswith(_FLAG_PREFIXES):
        return {0: ((200, 200, 200), "No"), 1: ((214, 39, 40), "Yes")}
    return None


def render_band(array: np.ndarray, band: str, nodata=None, palette: Optional[dict] = None):
    """Decimate one band and colour it. Returns ``(rgba uint8 (h, w, 4), step, legend, values)``.
    ``legend`` is ``{"kind": "categories", "items": [(libellé, (r, g, b))], "labels": {id: libellé}}``
    or ``{"kind": "continuous", "vmin", "vmax", "unit", "stops": [(r, g, b) x 5]}``; ``values`` is the
    decimated float32 grid (NaN = no data) used for the hover read-out."""
    small, step = _decimate(array)
    valid = np.isfinite(small) if small.dtype.kind == "f" else np.ones(small.shape, bool)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        valid &= small != nodata
    rgba = np.zeros((*small.shape, 4), np.uint8)
    values = np.where(valid, small, np.nan).astype(np.float32)
    ids = np.unique(small[valid]) if valid.any() else np.array([], small.dtype)
    named = None if small.dtype.kind == "f" else _named_labels(band, palette)
    continuous = named is None

    if continuous:
        import matplotlib as mpl

        vals = small[valid]
        lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1e-9
        cmap = mpl.colormaps["viridis"]
        rgba[:] = (cmap(np.clip((small - lo) / (hi - lo), 0, 1)) * 255).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 255, 0)
        stops = [tuple(int(v) for v in np.array(cmap(t)[:3]) * 255) for t in np.linspace(0, 1, 5)]
        legend = {"kind": "continuous", "vmin": float(lo), "vmax": float(hi), "unit": _unit(band), "stops": stops}
        return rgba, step, legend, values

    import matplotlib as mpl

    tab = mpl.colormaps["tab20"]
    items, labels = [], {}
    for k, i in enumerate(int(v) for v in ids):
        colour, name = named.get(i, (None, f"Value {i}"))
        colour = colour or tuple(int(v) for v in np.array(tab(k % 20)[:3]) * 255)
        sel = valid & (small == i)
        rgba[sel, :3] = colour
        rgba[sel, 3] = 255
        items.append((f"{i} — {name}", tuple(int(c) for c in colour)))
        labels[i] = name
    return rgba, step, {"kind": "categories", "items": items, "labels": labels}, values


def legend_html(legend: dict, title: str) -> str:
    """HTML legend: colour bar with min / middle / max, or one swatch per class."""
    if legend["kind"] == "continuous":
        grad = ", ".join(f"rgb{c}" for c in legend["stops"])
        mid = (legend["vmin"] + legend["vmax"]) / 2
        return (
            f"<div><b>{title}</b><br><span style='font-size:12px'>Unit: {legend['unit']}</span>"
            f"<div style='height:14px;width:200px;margin:4px 0;background:linear-gradient(to right,{grad});"
            f"border:1px solid #888'></div>"
            f"<div style='display:flex;justify-content:space-between;width:200px;font-size:12px'>"
            f"<span>{legend['vmin']:.4g}</span><span>{mid:.4g}</span><span>{legend['vmax']:.4g}</span></div>"
            f"<span style='font-size:11px;color:#666'>2–98 % stretch (extremes clipped)</span></div>"
        )
    rows = "".join(
        f"<div><span style='display:inline-block;width:12px;height:12px;background:rgb{c};margin-right:6px;"
        f"border:1px solid #888'></span>{n}</div>"
        for n, c in legend["items"]
    )
    return f"<div><b>{title}</b>{rows or '<div>No values</div>'}</div>"


def wgs84_overlay(rgba: np.ndarray, transform: tuple, crs_wkt: str, step: int, values: Optional[np.ndarray] = None):
    """Reproject a decimated RGBA preview (and optionally its float32 value grid) to EPSG:4326 ->
    ``(rgba, [[south, west], [north, east]], values_wgs84)`` for a web-map image overlay."""
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    src_crs = CRS.from_user_input(crs_wkt)
    height, width = rgba.shape[:2]
    src_transform = Affine(*transform) * Affine.scale(step)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, "EPSG:4326", width, height, *rasterio_bounds(src_transform, width, height)
    )
    out = np.zeros((dst_h, dst_w, 4), np.uint8)
    for band in range(4):
        reproject(
            rgba[..., band], out[..., band], src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs="EPSG:4326", resampling=Resampling.nearest,
        )
    out_vals = None
    if values is not None:
        out_vals = np.full((dst_h, dst_w), np.nan, np.float32)
        reproject(
            values, out_vals, src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform,
            dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan,
        )
    west, north = dst_transform * (0, 0)
    east, south = dst_transform * (dst_w, dst_h)
    return out, [[south, west], [north, east]], out_vals


def rasterio_bounds(transform, width: int, height: int) -> tuple[float, float, float, float]:
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    return min(left, right), min(top, bottom), max(left, right), max(top, bottom)
