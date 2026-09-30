"""Convert this pipeline's sparse, non-rasterisable tables to CSV.

Purpose
-------
Some of this pipeline's stored ``.npz`` tables are genuinely not grids --
CCDC's segment table (0..N rows per pixel) and the NDVI regrowth
trajectory (0..N (pixel, month) rows per pixel) -- so
:mod:`landscape_change_detection_pipeline.export.geotiff` cannot turn them
into a GeoTIFF (it only derives a dense per-pixel *summary* grid from
CCDC's table, e.g. "latest break date"; the full table is lost in that
derivation). This module is the export path for the full table itself, for
analysis (plotting a trajectory, inspecting segment magnitudes) rather than
cartographic display.

Supported inputs
-----------------
- **CCDC segment table** (``change.ccdc.write_ccdc_result``): one row per
  detected segment, ``row``/``col`` plus its per-band magnitude columns
  (``magnitude__{band}``, expanded from the stored ``(N, n_bands)`` array
  using ``band_order``).
- **Regrowth NDVI trajectory** (``change.regrowth_severity.write_regrowth_severity_result``):
  one row per (pixel, month) observation after a detected break.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CsvExportError(Exception):
    """Raised when an ``.npz`` file's contents cannot be recognised as one
    of this converter's supported sparse-table formats."""


def _load_npz_allow_object(path: str | Path) -> dict:
    """Unlike :func:`geotiff._load_npz`, this loader keeps ``object``-dtype
    string arrays (``pre_month``/``post_month`` in regrowth-severity files)
    -- a CSV, unlike a GeoTIFF, has no trouble with a text column. Pickle
    loading is scoped to this pipeline's own trusted output files, never
    user-supplied ones."""
    with np.load(Path(path), allow_pickle=True) as data:
        out = {}
        for key in data.files:
            array = data[key]  # one decompression per array (it used to be read twice)
            out[key] = np.array(array, dtype=object) if array.dtype == object else np.asarray(array)
        return out


def ccdc_segments_to_rows(data: dict) -> tuple[list[str], list[list]]:
    """Build (header, rows) for a CCDC segment table."""
    band_order = [str(b) for b in data["band_order"]]
    header = ["row", "col", "t_start", "t_end", "t_break", "num_obs", "change_prob"] + [f"magnitude__{b}" for b in band_order]
    rows = []
    magnitude = data["magnitude"]
    for i in range(len(data["row"])):
        rows.append(
            [
                int(data["row"][i]), int(data["col"][i]), int(data["t_start"][i]), int(data["t_end"][i]),
                int(data["t_break"][i]), int(data["num_obs"][i]), int(data["change_prob"][i]),
                *[float(v) for v in magnitude[i]],
            ]
        )
    return header, rows


def regrowth_trajectory_to_rows(data: dict) -> tuple[list[str], list[list]]:
    """Build (header, rows) for a regrowth-severity NDVI trajectory table."""
    header = ["row", "col", "month", "ndvi"]
    if "trajectory_month_index" in data:  # months stored as an index into a name table
        months = [str(data["trajectory_month_names"][j]) for j in data["trajectory_month_index"]]
    else:
        months = [str(m) for m in data["trajectory_month"]]
    rows = [
        [int(data["trajectory_row"][i]), int(data["trajectory_col"][i]), months[i], float(data["trajectory_ndvi"][i])]
        for i in range(len(data["trajectory_row"]))
    ]
    return header, rows


def build_csv_rows(npz_path: str | Path) -> tuple[str, list[str], list[list]]:
    """Load an ``.npz`` file and return ``(kind, header, rows)``, detecting
    which of this converter's supported sparse-table formats it is."""
    data = _load_npz_allow_object(npz_path)

    if "band_order" in data:
        header, rows = ccdc_segments_to_rows(data)
        return "ccdc_segments", header, rows

    if "trajectory_row" in data:
        header, rows = regrowth_trajectory_to_rows(data)
        return "regrowth_trajectory", header, rows

    raise CsvExportError(
        f"'{npz_path}' does not match any recognised sparse-table format "
        f"(expected a ccdc segment table or a regrowth-severity trajectory "
        f"table). Found keys: {sorted(data)}"
    )


def write_csv(header: list[str], rows: list[list], out_path: str | Path) -> Path:
    """Write ``header``/``rows`` to a plain CSV file."""
    import csv

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return out


def export_to_csv(npz_path: str | Path, out_path: str | Path) -> Path:
    """Convert one stored sparse-table ``.npz`` to CSV. Entry point
    ``scripts/export_gui.py`` calls."""
    _kind, header, rows = build_csv_rows(npz_path)
    return write_csv(header, rows, out_path)


# ---------------------------------------------------------------------------
# Generic tables: every sparse / tabular array group of any pipeline .npz
# ---------------------------------------------------------------------------

#: Keys that describe the file (georeferencing, axis labels, name lists) rather than being a column.
_CSV_META_KEYS = frozenset({
    "transform", "crs_wkt", "resolution_m", "shape", "nodata", "product", "year0", "water_year_start_month",
    "feature_names", "event_names", "masked_class_ids", "succession_class_ids", "succession_class_names",
    "growing_season", "years", "months", "water_years",
})
_TABLE_KINDS = "biufUS"
_CHUNK_ROWS = 100_000


def _is_table_candidate(key: str, array: np.ndarray) -> bool:
    return not (
        key in _CSV_META_KEYS or key.startswith(("labels__", "trajectory_")) or array.dtype.kind not in _TABLE_KINDS
    )


def _expand_columns(key: str, col_name: str, array: np.ndarray, data: dict) -> tuple[list[str], list[np.ndarray]]:
    """One column for a 1-D array; one per position of the second axis for a 2-D array
    (named from ``labels__<key>``, ``feature_names`` for per-feature arrays, else the index)."""
    if array.ndim == 1:
        return [col_name], [array]
    labels = data.get(f"labels__{key}")
    if labels is None and "feature_names" in data and len(data["feature_names"]) == array.shape[1]:
        labels = data["feature_names"]
    if labels is None or len(labels) != array.shape[1]:
        labels = [str(i) for i in range(array.shape[1])]
    return [f"{col_name}__{lab}" for lab in labels], [array[:, j] for j in range(array.shape[1])]


def _make_table(name: str, items: list[tuple[str, str, np.ndarray]], data: dict) -> dict:
    header: list[str] = []
    columns: list[np.ndarray] = []
    for key, col_name, array in items:
        h, c = _expand_columns(key, col_name, array, data)
        header += h
        columns += c
    return {"name": name, "header": header, "columns": columns, "n_rows": len(columns[0]) if columns else 0}


def _table_name(length: int, keys: list[str]) -> str:
    """A short name for an unnamed table: the columns' common prefix (``cropland_row``,
    ``cropland_col`` -> ``cropland``), else ``table_<n rows>``."""
    import os

    prefix = os.path.commonprefix(keys).rstrip("_")
    if len(keys) > 1 and len(prefix) >= 3 and all(k.startswith(prefix + "_") for k in keys):
        return prefix
    return f"table_{length}_rows"


def _generic_tables(data: dict) -> list[dict]:
    from landscape_change_detection_pipeline.export.geotiff import _grid_shape_of

    tables: list[dict] = []
    named: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    for key, array in data.items():
        if key.startswith("tbl_") and "__" in key and array.dtype.kind in _TABLE_KINDS and array.ndim in (1, 2):
            table_name, col = key[4:].split("__", 1)
            named.setdefault(table_name, []).append((key, col, array))
    for table_name, items in named.items():
        length = len(items[0][2])
        items = [it for it in items if len(it[2]) == length]
        tables.append(_make_table(table_name, items, data))

    grid = _grid_shape_of(data) if "transform" in data else None
    groups: dict[int, list[tuple[str, str, np.ndarray]]] = {}
    for key, array in data.items():
        if key.startswith("tbl_") or not _is_table_candidate(key, array):
            continue
        if array.ndim == 1 and len(array) >= 1:
            groups.setdefault(len(array), []).append((key, key, array))
    for length, items in groups.items():
        for key, array in data.items():  # 2-D per-row arrays (e.g. magnitude (N, n_features))
            if (
                array.ndim == 2 and len(array) == length and not key.startswith("tbl_")
                and _is_table_candidate(key, array) and (grid is None or tuple(array.shape) != tuple(grid))
            ):
                items.append((key, key, array))
    for length, items in groups.items():
        if length < 2 and len(items) < 2:
            continue
        tables.append(_make_table(_table_name(length, [it[1] for it in items]), items, data))
    return [t for t in tables if t["n_rows"] > 0]


def _legacy_table(name: str, header: list[str], rows: list[list]) -> dict:
    columns = [np.array(col) for col in zip(*rows)] if rows else [np.array([]) for _ in header]
    return {"name": name, "header": header, "columns": columns, "n_rows": len(rows)}


def list_csv_tables(npz_path: str | Path) -> list[dict]:
    """Every table a stored ``.npz`` holds, ready to preview / write as CSV:
    ``[{"name", "header", "columns", "n_rows"}]`` (empty list = nothing tabular).
    Covers the segment table, event tables, landcover-persistence intervals, the
    regrowth trajectory, and every named ``tbl_*`` table of scripts/14."""
    data = _load_npz_allow_object(npz_path)
    return tables_from_data(data)


def tables_from_data(data: dict) -> list[dict]:
    """Same as :func:`list_csv_tables` on an already-loaded ``.npz`` dict (columnar, no row loops)."""
    tables: list[dict] = []
    if "band_order" in data:
        band_order = [str(b) for b in data["band_order"]]
        header = ["row", "col", "t_start", "t_end", "t_break", "num_obs", "change_prob"] + [f"magnitude__{b}" for b in band_order]
        columns = [data[k] for k in ("row", "col", "t_start", "t_end", "t_break", "num_obs", "change_prob")]
        columns += [data["magnitude"][:, j] for j in range(data["magnitude"].shape[1])]
        tables.append({"name": "ccdc_segments", "header": header, "columns": columns, "n_rows": len(data["row"])})
    if "trajectory_row" in data:
        if "trajectory_month_index" in data:
            names = np.array([str(n) for n in data["trajectory_month_names"]])
            months = names[data["trajectory_month_index"].astype(np.int64)]
        else:
            months = np.array([str(m) for m in data["trajectory_month"]])
        tables.append({
            "name": "regrowth_trajectory", "header": ["row", "col", "month", "ndvi"],
            "columns": [data["trajectory_row"], data["trajectory_col"], months, data["trajectory_ndvi"]],
            "n_rows": len(data["trajectory_row"]),
        })
    if "band_order" not in data:
        tables += _generic_tables(data)
    return tables


def table_preview(table: dict, n: int = 200) -> tuple[list[str], list[list]]:
    """First ``n`` rows as plain Python lists, for display."""
    cols = [c[:n].tolist() for c in table["columns"]]
    return table["header"], [list(r) for r in zip(*cols)]


def write_table_csv(table: dict, out_path: str | Path) -> Path:
    """Stream one table to CSV in chunks (no full list-of-rows in memory)."""
    import csv

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(table["header"])
        for start in range(0, table["n_rows"], _CHUNK_ROWS):
            chunk = [c[start:start + _CHUNK_ROWS].tolist() for c in table["columns"]]
            writer.writerows(zip(*chunk))
    return out
