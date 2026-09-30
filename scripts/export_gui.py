#!/usr/bin/env python3
"""Browse this pipeline's stored rasters/tables, preview them on a map, and export
what you pick to GeoTIFF or CSV -- a GUI front end for
:mod:`landscape_change_detection_pipeline.export`.

Purpose
-------
Four steps, top to bottom, nothing is read until you ask for it:

1. **Folder** -- type or browse a path, press *Scan* (lists file names only).
2. **Filter** -- by result type (class map, change map, segments ...) and by name.
3. **Select** -- tick the files to export (table with sizes).
4. **Act** -- either *inspect one file* (its arrays, choose which bands / tables to
   export, georeferenced preview on a satellite basemap) or *export the selection*
   (every raster -> GeoTIFF, every table -> CSV, one file at a time, so RAM stays flat).

Only one file is ever held in memory (the one being inspected); a preview decodes a
single band, decimated to screen size.

Usage
-----
    streamlit run scripts/export_gui.py

Running it plainly (``python scripts/export_gui.py``, e.g. VS Code's "Run
Python File" button) re-execs itself under ``streamlit run``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _relaunch_under_streamlit() -> None:
    """Re-exec as ``streamlit run <this file>`` when run as a plain script."""
    from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

    if get_script_run_ctx() is not None:
        return

    import subprocess

    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(result.returncode)


_relaunch_under_streamlit()

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402  -- before rasterio (DLL clash on some conda envs)

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from landscape_change_detection_pipeline.export.batch import export_one, find_dem_tiles, find_npz_files  # noqa: E402
from landscape_change_detection_pipeline.export.browse import (  # noqa: E402
    band_label, describe_npz, legend_html, load_palette, product_type, render_band, type_label, wgs84_overlay,
)
from landscape_change_detection_pipeline.export.csv_export import (  # noqa: E402
    _load_npz_allow_object, table_preview, tables_from_data, write_table_csv,
)
from landscape_change_detection_pipeline.export.geotiff import (  # noqa: E402
    GeoTiffExportError, build_dem_export_payload, build_export_payload, write_geotiff,
)

st.set_page_config(page_title="Results export", layout="wide")


@st.cache_resource
def _palette():
    return load_palette(REPO / "configs" / "classes.yaml")


def _browse_folder(initial: str) -> str | None:
    """Native folder dialog, in a subprocess (Tk must own its main thread)."""
    import subprocess

    code = (
        "import tkinter as tk, tkinter.filedialog as fd, sys\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
        "print(fd.askdirectory(initialdir=sys.argv[1] or None, title='Results folder'))"
    )
    out = subprocess.run([sys.executable, "-c", code, initial], capture_output=True, text=True).stdout.strip()
    return out or None


def _ss(key, default):
    return st.session_state.setdefault(key, default)


def _load_file(path: str) -> tuple[dict | None, str | None, list]:
    """Read a file once for the raster payload and the tables, both in parallel threads
    (zlib decompression releases the GIL, so the two reads overlap)."""
    from concurrent.futures import ThreadPoolExecutor

    def tables():
        try:
            return tables_from_data(_load_npz_allow_object(path))
        except Exception:  # noqa: BLE001
            return []

    def raster():
        try:
            return build_export_payload(path), None
        except GeoTiffExportError as exc:
            return None, str(exc)

    with ThreadPoolExecutor(2) as pool:
        f_tables, f_raster = pool.submit(tables), pool.submit(raster)
        payload, err = f_raster.result()
        return payload, err, f_tables.result()


@st.cache_data(show_spinner="Preparing preview…", max_entries=8)
def _preview(path: str, band: str, _payload: dict):
    """Colour + reproject one band; cached so that changing a widget does not redo it."""
    import base64

    array = _payload["bands"][band]
    nodata = _payload.get("per_band_nodata", {}).get(band, _payload.get("nodata"))
    rgba, step, legend, values = render_band(array, band, nodata, _palette())
    overlay, bounds, vals = wgs84_overlay(rgba, _payload["transform"], _payload["crs_wkt"], step, values)
    stride = max(1, int(np.ceil(max(vals.shape) / 900)))  # keep the embedded value grid light
    vals = np.ascontiguousarray(vals[::stride, ::stride], dtype="<f4")
    return {
        "overlay": overlay, "bounds": bounds, "legend": legend, "shape": array.shape, "dtype": str(array.dtype),
        "vals_b64": base64.b64encode(vals.tobytes()).decode("ascii"), "vals_hw": vals.shape,
    }


_HOVER_JS = """
window.addEventListener("load", function () { try {
  var map = %(map)s;
  var raw = atob("%(b64)s"), bytes = new Uint8Array(raw.length);
  for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  var V = new Float32Array(bytes.buffer), H = %(h)d, W = %(w)d;
  var south = %(south)s, west = %(west)s, north = %(north)s, east = %(east)s;
  var labels = %(labels)s, unit = "%(unit)s";
  function merc(l) { return Math.log(Math.tan(Math.PI / 4 + l * Math.PI / 360)); }
  var box = L.control({position: "bottomleft"});
  box.onAdd = function () {
    var d = L.DomUtil.create("div");
    d.style.cssText = "background:rgba(255,255,255,.92);color:#111;padding:4px 8px;border:1px solid #888;font:12px sans-serif";
    d.innerHTML = "Hover a pixel to read its value";
    this._d = d; return d;
  };
  box.addTo(map);
  map.on("mousemove", function (e) {
    var x = (e.latlng.lng - west) / (east - west), y = (merc(north) - merc(e.latlng.lat)) / (merc(north) - merc(south));
    var out = "Outside the image";
    if (x >= 0 && x < 1 && y >= 0 && y < 1) {
      var v = V[Math.floor(y * H) * W + Math.floor(x * W)];
      if (isNaN(v)) out = "No data";
      else {
        var t = labels[String(Math.round(v))];
        out = (t !== undefined ? t + " (" + Math.round(v) + ")" : unit + " : " + (Math.abs(v) >= 1000 || Number.isInteger(v) ? v : v.toPrecision(4)));
      }
    }
    box._d.innerHTML = out + "<br>Lat " + e.latlng.lat.toFixed(5) + " · Lon " + e.latlng.lng.toFixed(5);
  });
} catch (err) { console.error(err); } });
"""


def _build_map(prev: dict, band: str) -> str:
    import json

    import folium

    bounds = prev["bounds"]
    fmap = folium.Map(location=[np.mean([bounds[0][0], bounds[1][0]]), np.mean([bounds[0][1], bounds[1][1]])],
                      zoom_start=11, tiles=None, zoom_control=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(fmap)
    folium.raster_layers.ImageOverlay(
        prev["overlay"], bounds=bounds, opacity=0.75, name=band_label(band),
    ).add_to(fmap)
    fmap.fit_bounds(bounds)
    legend = prev["legend"]
    fmap.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;top:10px;right:10px;z-index:9999;background:rgba(255,255,255,.92);color:#111;"
        "padding:6px 10px;border:1px solid #888;font:12px sans-serif;max-height:70%;overflow:auto'>"
        + legend_html(legend, band_label(band)) + "</div>"
    ))
    south, west = bounds[0]
    north, east = bounds[1]
    h, w = prev["vals_hw"]
    fmap.get_root().script.add_child(folium.Element(_HOVER_JS % {
        "map": fmap.get_name(), "b64": prev["vals_b64"], "h": h, "w": w, "south": south, "west": west,
        "north": north, "east": east, "labels": json.dumps({str(k): v for k, v in legend.get("labels", {}).items()}),
        "unit": legend.get("unit", "value"),
    }))
    return fmap.get_root().render()


# ── Interface theme (white by default) ──────────────────────────────────
_theme = st.sidebar.radio("Interface background", ["White", "Black"], horizontal=True, key="ui_theme")
if _theme == "Black":
    # Dark mode by inversion: covers every widget (tables included); maps / images are re-inverted to keep their colours.
    st.markdown(
        "<style>.stApp{background:#fff;filter:invert(1) hue-rotate(180deg)}"
        ".stApp iframe,.stApp img,.stApp canvas.mapboxgl-canvas{filter:invert(1) hue-rotate(180deg)}</style>",
        unsafe_allow_html=True,
    )
else:
    st.markdown("<style>.stApp{background:#ffffff;color:#111}</style>", unsafe_allow_html=True)

# ── 1. Dossier ────────────────────────────────────────────────────────────────
st.title("Pipeline results export")
st.subheader("1. Folder")
_ss("root", str(REPO / "outputs"))
c_path, c_browse, c_scan = st.columns([6, 1, 1], vertical_alignment="bottom")
if c_browse.button("Browse…"):
    picked = _browse_folder(st.session_state["root"])
    if picked:
        st.session_state["root"] = picked
        st.rerun()
c_path.text_input("Results folder path", key="root")
if c_scan.button("Scan", type="primary"):
    root_path = Path(st.session_state["root"])
    with st.spinner("Looking for files (names only)…"):
        files = find_npz_files(root_path)
        dems = find_dem_tiles(root_path)
    st.session_state["scan"] = {
        "root": str(root_path),
        "rows": [
            {"path": str(p), "rel": str(p.relative_to(root_path)), "type": product_type(p),
             "mb": round(p.stat().st_size / 2**20, 1)}
            for p in files
        ],
        "dems": dems,
    }
    st.session_state["selected"] = set()
    st.session_state["loaded"] = None

scan = st.session_state.get("scan")
if not scan:
    st.info("Pick a folder then click **Scan**. Nothing is loaded until you ask for it.")
    st.stop()
if not scan["rows"] and not scan["dems"]:
    st.warning(f"No .npz file or DEM found in '{scan['root']}'.")
    st.stop()

# ── 2. Filtres ────────────────────────────────────────────────────────────────
st.subheader("2. Result type")
type_counts: dict[str, int] = {}
for r in scan["rows"]:
    type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
if scan["dems"]:
    type_counts["[DEM]"] = len(scan["dems"])
st.caption("Click to toggle one or more types — all types stay visible, the chosen ones are highlighted. "
           "No choice = all types.")
chosen_types = st.pills(
    "Types", sorted(type_counts), selection_mode="multi", key="f_types",
    format_func=lambda t: f"{'DEM' if t == '[DEM]' else type_label(t)}  ({type_counts[t]})",
) or []
name_filter = st.text_input("Path contains", key="f_name").lower()
max_mb = st.slider("Max size (MB)", 0, int(max([r["mb"] for r in scan["rows"]] + [1])) + 1,
                   int(max([r["mb"] for r in scan["rows"]] + [1])) + 1, key="f_mb")

rows = [
    r for r in scan["rows"]
    if (not chosen_types or r["type"] in chosen_types) and name_filter in r["rel"].lower() and r["mb"] <= max_mb
]
dems = [d for d in scan["dems"] if (not chosen_types or "[DEM]" in chosen_types) and name_filter in d[1].lower()]

# ── 3. Sélection ──────────────────────────────────────────────────────────────
st.subheader("3. Files to export")
selected: set[str] = st.session_state["selected"]
b1, b2, b3 = st.columns([1, 1, 6])
if b1.button("Select all"):
    selected.update(r["path"] for r in rows)
    st.session_state["ed_ver"] = st.session_state.get("ed_ver", 0) + 1
if b2.button("Clear all"):
    selected.clear()
    st.session_state["ed_ver"] = st.session_state.get("ed_ver", 0) + 1
b3.caption(f"{len(rows)} file(s) and {len(dems)} DEM match the filters — {len(selected)} ticked.")

SHOW_MAX = 500
shown = rows[:SHOW_MAX]
if len(rows) > SHOW_MAX:
    st.caption(f"Table limited to the first {SHOW_MAX} files ({len(rows)} in total): refine the filters, "
               f"or 'Select all' ticks all {len(rows)}.")
table = [{"Export": r["path"] in selected, "Type": type_label(r["type"]), "File": r["rel"], "MB": r["mb"]} for r in shown]
edited = st.data_editor(
    table, hide_index=True, width="stretch", height=min(420, 40 + 35 * max(len(table), 1)),
    disabled=["Type", "File", "MB"], key=f"ed_{st.session_state.get('ed_ver', 0)}",
)
for r, e in zip(shown, edited):
    (selected.add if e["Export"] else selected.discard)(r["path"])

# ── 4. Action ─────────────────────────────────────────────────────────────────
st.subheader("4. Action")
tab_inspect, tab_batch = st.tabs(["Inspect / export one file", f"Export the selection ({len(selected)})"])

# --- Export the selection -------------------------------------------------
with tab_batch:
    root_path = Path(scan["root"])
    out_dir = st.text_input("Output folder", value=str(root_path.parent / (root_path.name + "_export")), key="b_out")
    c1, c2, c3 = st.columns(3)
    do_tif = c1.checkbox("Rasters → GeoTIFF", value=True, key="b_tif")
    do_csv = c2.checkbox("Tables → CSV", value=True, key="b_csv")
    skip = c3.checkbox("Skip if already exported", value=True, key="b_skip")
    do_dem = bool(dems) and st.checkbox(f"Include the {len(dems)} DEM (GeoTIFF)", value=False, key="b_dem")
    todo = sorted(selected)
    if st.button("Run export", type="primary", disabled=not (todo or do_dem), key="b_go"):
        bar, status = st.progress(0.0), st.empty()
        records: list[dict] = []
        total = len(todo) + (len(dems) if do_dem else 0)
        for i, path in enumerate(todo, 1):
            rel = Path(path).relative_to(root_path)
            status.write(f"{i}/{total} — {rel}")
            records += export_one(Path(path), Path(out_dir) / rel.with_suffix(""), do_tif, do_csv, skip)
            bar.progress(i / total)
        if do_dem:
            for j, (tile_dir, tile_id) in enumerate(dems, len(todo) + 1):
                status.write(f"{j}/{total} — [DEM] {tile_id}")
                try:
                    write_geotiff(build_dem_export_payload(tile_dir, tile_id), Path(out_dir) / "dem" / f"{tile_id}_dem.tif")
                    records.append({"file": f"dem::{tile_id}", "kind": "geotiff", "status": "ok"})
                except Exception as exc:  # noqa: BLE001
                    records.append({"file": f"dem::{tile_id}", "kind": "geotiff", "status": "error", "detail": str(exc)})
                bar.progress(j / total)
        errs = [r for r in records if r["status"] == "error"]
        st.success(f"{sum(r['status'] == 'ok' for r in records)} export(s) written to {out_dir} — "
                   f"{sum(r['status'] == 'skipped' for r in records)} skipped, {len(errs)} error(s).")
        if errs:
            st.dataframe([{"file": r["file"], "type": r["kind"], "detail": r.get("detail", "")} for r in errs])

# --- Inspect one file ------------------------------------------------
with tab_inspect:
    candidates = [r["path"] for r in shown]
    labels = {r["path"]: f"[{type_label(r['type'])}] {r['rel']}  ({r['mb']} MB)" for r in shown}
    if not candidates:
        st.info("No .npz file matches the filters.")
        st.stop()
    path = st.selectbox("File", candidates, format_func=labels.get, key="i_file")

    with st.expander("File contents (headers only, nothing is loaded)"):
        st.dataframe([{"array": band_label(d["key"]), "shape": str(d["shape"]), "data type": d["dtype"], "MB": d["mb"]} for d in describe_npz(path)],
                     width="stretch", hide_index=True)

    loaded = st.session_state.get("loaded")
    if st.button("Load this file", type="primary"):
        st.session_state["loaded"] = None  # free the previous file before reading the next
        with st.spinner("Reading file…"):
            payload, err, tables = _load_file(path)
        st.session_state["loaded"] = loaded = {"path": path, "payload": payload, "err": err, "tables": tables}

    if not loaded or loaded["path"] != path:
        st.stop()

    payload, tables = loaded["payload"], loaded["tables"]
    if payload is None and not tables:
        st.error("Nothing exportable in this file: no georeferenced grid or recognised table.")
        if loaded["err"]:
            st.caption(f"Technical detail: {loaded['err']}")
        st.stop()

    kept_bands, kept_tables = [], []
    if payload is not None:
        st.markdown("**Raster bands** (GeoTIFF)")
        band_names = list(payload["bands"])
        st.caption("All bands stay visible; highlighted ones will be exported.")
        kept_bands = st.pills("Bands to export", band_names, selection_mode="multi", default=band_names,
                              format_func=band_label, key=f"i_bands_{path}") or []
        prev_band = st.selectbox("Band to display", band_names, format_func=band_label, key=f"i_prev_{path}")
        prev = _preview(path, prev_band, payload)

        left, right = st.columns([4, 1])
        with right:
            st.metric("Size (rows × columns)", f"{prev['shape'][0]} × {prev['shape'][1]}")
            st.metric("Data type", prev["dtype"])
            st.markdown(legend_html(prev["legend"], band_label(prev_band)), unsafe_allow_html=True)
        with left:
            st.caption("Hover the map: the pixel value is shown at the bottom left.")
            components.html(_build_map(prev, prev_band), height=620)

    if tables:
        st.markdown("**Tables** (CSV)")
        tnames = {i: f"{band_label(t['name'])}  ({t['n_rows']} rows, {len(t['header'])} columns)" for i, t in enumerate(tables)}
        kept_tables = st.pills("Tables to export", list(tnames), selection_mode="multi", default=list(tnames),
                               format_func=tnames.get, key=f"i_tables_{path}") or []
        peek = st.selectbox("Table to preview", list(tnames), format_func=tnames.get, key=f"i_tpeek_{path}")
        header, preview_rows = table_preview(tables[peek], 200)
        st.dataframe(preview_rows, column_config={i: band_label(h) for i, h in enumerate(header)}, width="stretch")

    st.divider()
    dest = st.text_input("Output folder", value=str(Path(path).parent), key=f"i_out_{path}")
    if st.button("Export this file's selection", type="primary", key="i_go"):
        stem = Path(path).stem
        written: list[Path] = []
        if payload is not None and kept_bands:
            sub = dict(payload)
            sub["bands"] = {k: v for k, v in payload["bands"].items() if k in kept_bands}
            for meta in ("per_band_dtype", "per_band_nodata"):
                if meta in payload:
                    sub[meta] = {k: v for k, v in payload[meta].items() if k in kept_bands}
            with st.spinner("Writing GeoTIFF…"):
                out = write_geotiff(sub, Path(dest) / f"{stem}.tif")
            written += out if isinstance(out, list) else [out]
        for i in kept_tables:
            suffix = "" if len(tables) == 1 else "__" + "".join(c if c.isalnum() else "_" for c in tables[i]["name"])[:60]
            with st.spinner(f"Writing {band_label(tables[i]['name'])}…"):
                written.append(write_table_csv(tables[i], Path(dest) / f"{stem}{suffix}.csv"))
        if written:
            st.success("Exported:")
            for p in written:
                st.code(str(p))
        else:
            st.warning("Nothing ticked.")
