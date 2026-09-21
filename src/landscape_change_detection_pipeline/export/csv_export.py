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
        return {key: np.array(data[key], dtype=object) if data[key].dtype == object else np.array(data[key]) for key in data.files}


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
    rows = [
        [int(data["trajectory_row"][i]), int(data["trajectory_col"][i]), str(data["trajectory_month"][i]), float(data["trajectory_ndvi"][i])]
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
