#!/usr/bin/env python3
"""Browse this pipeline's stored rasters/tables, preview them, and export
any of them to GeoTIFF or CSV on demand -- a GUI front end for
:mod:`landscape_change_detection_pipeline.export.geotiff` and
:mod:`landscape_change_detection_pipeline.export.csv_export`.

Purpose
-------
The only front end for this pipeline's exporters -- there is no CLI to
remember a file path for. On-demand, one-file-at-a-time export, never a
batch/automatic pass over a whole output tree, since some of this
pipeline's rasters (AOI-wide index mosaics in particular) are large enough
that exporting every period at once would be slow and wasteful for outputs
nobody asked to see. A folder browser, a quicklook of every band before
exporting, and a button.

What's browsable
-----------------
- Every ``.npz`` raster this pipeline stores (class map, composite,
  mosaic, index composite/mosaic, change map, bfast, dNBR) -> GeoTIFF.
- ``ccdc.npz`` -> either a derived per-pixel summary GeoTIFF (latest break
  date, segment count) or the full ragged segment table as CSV.
- ``regrowth_severity_*.npz`` -> either the dense dNBR GeoTIFF or the full
  NDVI trajectory table as CSV.
- Per-tile DEM products (``<tile_id>.zarr/dem``) -> GeoTIFF. Zarr stores
  are folders, not files, so they are listed separately from the ``.npz``
  files found under the same root.

Usage
-----
    streamlit run scripts/export_gui.py

Running it plainly (``python scripts/export_gui.py``, e.g. VS Code's "Run
Python File" button) re-execs itself under ``streamlit run`` instead of
failing -- see ``_relaunch_under_streamlit`` below.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _relaunch_under_streamlit() -> None:
    """If this file is being executed directly (no Streamlit
    ``ScriptRunContext``, e.g. a plain ``python export_gui.py`` or VS
    Code's "Run Python File" button), re-exec it as ``streamlit run
    <this file>`` in a subprocess and exit -- so the one button people
    actually reach for always works, without needing to remember this
    script is special."""
    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

    if get_script_run_ctx() is not None:
        return

    import subprocess

    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(result.returncode)


_relaunch_under_streamlit()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import streamlit as st

from landscape_change_detection_pipeline.export.geotiff import (
    GeoTiffExportError,
    build_dem_export_payload,
    build_export_payload,
    write_geotiff,
)
from landscape_change_detection_pipeline.export.csv_export import CsvExportError, build_csv_rows, write_csv

st.set_page_config(page_title="Pipeline export", layout="wide")

#: .npz key that marks a file as a sparse table (also exportable as CSV)
#: rather than a plain dense raster -- checked before building the GeoTIFF
#: payload so the UI can offer both export kinds without paying the
#: GeoTIFF-payload cost twice.
_SPARSE_TABLE_MARKER_KEYS = ("band_order", "trajectory_row")


@st.cache_data(show_spinner=False)
def _find_npz_files(root: str) -> list[str]:
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    return sorted(str(p) for p in root_path.rglob("*.npz"))


@st.cache_data(show_spinner=False)
def _find_dem_tiles(root: str) -> list[tuple[str, str]]:
    """Every ``<tile_id>.zarr`` under ``root`` that has a ``dem`` group --
    returns ``(zarr_dir, tile_id)`` pairs. A zarr store is a folder, so
    this is a separate scan from :func:`_find_npz_files`, not a glob
    extension of it."""
    import zarr

    root_path = Path(root)
    if not root_path.is_dir():
        return []
    found = []
    for zarr_dir in sorted(root_path.rglob("*.zarr")):
        try:
            store = zarr.open_group(str(zarr_dir), mode="r")
        except Exception:  # noqa: BLE001 -- a partially written/foreign store must not crash the browser
            continue
        if "dem" in store:
            found.append((str(zarr_dir.parent), zarr_dir.stem))
    return found


@st.cache_data(show_spinner=False)
def _is_sparse_table(npz_path: str) -> bool:
    with np.load(npz_path, allow_pickle=False) as data:
        return any(key in data.files for key in _SPARSE_TABLE_MARKER_KEYS)


@st.cache_data(show_spinner=False)
def _load_geotiff_payload(npz_path: str):
    """Cached so re-selecting a file already previewed this session does
    not re-read and re-detect it. Returns (payload, error)."""
    try:
        return build_export_payload(npz_path), None
    except GeoTiffExportError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _load_dem_payload(tile_dir: str, tile_id: str):
    try:
        return build_dem_export_payload(tile_dir, tile_id), None
    except GeoTiffExportError as exc:
        return None, str(exc)


def _band_dtype_kind(array: np.ndarray) -> str:
    if array.dtype == np.uint8 and set(np.unique(array)) <= {0, 1}:
        return "boolean"
    if np.issubdtype(array.dtype, np.integer):
        return "categorical"
    return "continuous"


def _preview_bands(bands: dict[str, np.ndarray], key_prefix: str) -> None:
    band_keys = list(bands.keys())
    st.subheader(f"{len(band_keys)} band(s) detected")
    selected_band = st.selectbox("Preview band", band_keys, key=f"{key_prefix}_band")
    array = bands[selected_band]
    kind = _band_dtype_kind(array)

    col_preview, col_info = st.columns([3, 1])
    with col_info:
        st.metric("Shape", f"{array.shape[0]} x {array.shape[1]}")
        st.metric("Dtype", str(array.dtype))
        st.metric("Kind", kind)
        finite = array[np.isfinite(array.astype(np.float64))] if array.dtype.kind == "f" else array
        if finite.size:
            st.metric("Min / Max", f"{finite.min():.3g} / {finite.max():.3g}")

    with col_preview:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = "tab20" if kind == "categorical" else ("gray" if kind == "boolean" else "viridis")
        display_array = array.astype(np.float64) if array.dtype.kind in "iu" else array
        im = ax.imshow(display_array, cmap=cmap)
        ax.set_title(selected_band)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)
        plt.close(fig)


st.title("Pipeline output browser and exporter")
st.caption(
    "Browse this pipeline's stored outputs, preview them, and export the one you actually want "
    "-- as GeoTIFF for a GIS tool, or CSV for a sparse table -- on demand, one file at a time."
)

default_root = str(Path.cwd() / "outputs")
root = st.text_input("Output folder to scan (recursive)", value=default_root)

npz_files = _find_npz_files(root)
dem_tiles = _find_dem_tiles(root)

if not npz_files and not dem_tiles:
    st.warning(f"No .npz files or DEM zarr stores found under '{root}'.")
    st.stop()

rel_root = Path(root)
npz_labels = {f: str(Path(f).relative_to(rel_root)) for f in npz_files}
dem_labels = {f"dem::{d}::{t}": f"[DEM] {t}" for d, t in dem_tiles}

all_keys = npz_files + list(dem_labels.keys())
all_labels = {**npz_labels, **dem_labels}
st.write(f"{len(npz_files)} .npz file(s), {len(dem_tiles)} DEM tile(s) found.")

selected = st.selectbox("File", all_keys, format_func=lambda f: all_labels[f])

if selected.startswith("dem::"):
    _, tile_dir, tile_id = selected.split("::", 2)
    payload, error = _load_dem_payload(tile_dir, tile_id)
    if error is not None:
        st.error(f"Could not read DEM for '{tile_id}': {error}")
        st.stop()

    _preview_bands(payload["bands"], key_prefix="dem")
    st.divider()
    out_default = str(Path(tile_dir) / f"{tile_id}_dem.tif")
    out_path_str = st.text_input("Export destination (.tif)", value=out_default)
    if st.button("Export DEM to GeoTIFF", type="primary"):
        with st.spinner("Writing GeoTIFF..."):
            written = write_geotiff(payload, out_path_str)
        st.success("Exported:")
        st.code(str(written))
    st.stop()

# A plain .npz file: sparse table (CSV) or dense raster (GeoTIFF), or both.
if _is_sparse_table(selected):
    st.info(
        "This file is a sparse table (variable rows per pixel), not a dense grid -- "
        "export it as CSV for the full table, or as GeoTIFF for a derived per-pixel summary grid."
    )
    export_kind = st.radio("Export as", ["CSV (full table)", "GeoTIFF (derived summary)"], horizontal=True)
else:
    export_kind = "GeoTIFF (derived summary)"

if export_kind.startswith("CSV"):
    try:
        kind, header, rows = build_csv_rows(selected)
    except CsvExportError as exc:
        st.error(f"Not a recognised table format: {exc}")
        st.stop()

    st.write(f"{len(rows)} row(s), columns: {', '.join(header)}")
    st.dataframe(rows[:200], column_config=None, use_container_width=True)
    if len(rows) > 200:
        st.caption(f"Showing first 200 of {len(rows)} rows.")

    st.divider()
    out_default = str(Path(selected).with_suffix(".csv"))
    out_path_str = st.text_input("Export destination (.csv)", value=out_default)
    if st.button("Export to CSV", type="primary"):
        with st.spinner("Writing CSV..."):
            written = write_csv(header, rows, out_path_str)
        st.success("Exported:")
        st.code(str(written))

else:
    payload, error = _load_geotiff_payload(selected)
    if error is not None:
        st.error(f"Not a recognised export format: {error}")
        st.stop()

    _preview_bands(payload["bands"], key_prefix="raster")
    st.divider()
    out_default = str(Path(selected).with_suffix(".tif"))
    out_path_str = st.text_input("Export destination (.tif)", value=out_default)

    if "per_band_dtype" in payload:
        st.info(
            "This file mixes dtypes across bands -- it will be written as more than one "
            "GeoTIFF (one per dtype), each suffixed with that dtype."
        )

    if st.button("Export to GeoTIFF", type="primary"):
        from landscape_change_detection_pipeline.export.geotiff import export_to_geotiff

        with st.spinner("Writing GeoTIFF..."):
            written = export_to_geotiff(selected, out_path_str)
        written_list = written if isinstance(written, list) else [written]
        st.success("Exported:")
        for p in written_list:
            st.code(str(p))
