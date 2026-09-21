# landscape-change-detection-pipeline

A satellite-imagery land-cover and change-detection pipeline. Satellite scenes (Landsat 5/7/8/9 + Sentinel-2) are downloaded over a study area, manually annotated to build training data, used to train a model that classifies every pixel into a land-cover class, run at inference over every downloaded scene, summarized into monthly composites, and finally mined for breakpoints and changes over time (disturbance, regrowth, etc.) at the pixel level.

The pipeline itself grew out of an earlier project doing glacier change detection from satellite imagery, and was generalized from there: the class taxonomy (`configs/classes.yaml`) is fully swappable, so the same tiling/download/training/inference/change-detection machinery applies to any land-cover or land-surface classification problem, not just one fixed set of classes.

This document explains how the pieces fit together, not just the list of commands.

This repo's architecture is built around proven solutions to the structural problems involved: multi-sensor download with resume, zarr storage, leak-free train/val/test splitting, a U-Net with per-sensor normalization bands, Optuna HPO, bootstrap ensembling, robust GEE authentication.

## Setup

```
conda env create -f environment.yml
conda activate landscape-change-detection-pipeline
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # GPU build, see requirements-torch.txt
pip install -r requirements-torch.txt
pip install -e .
```

Copy `.env.example` -> `.env` and `configs/config.example.yaml` -> `configs/config.yaml`, then fill in the values (local paths, GEE projects). Both files are gitignored -- they contain machine-specific information (local paths, real GEE project list) and must never be committed. `configs/config.example.yaml` is the only config file tracked by git; it serves as a generic template with no personal information.

Google Earth Engine authentication (once):
```
earthengine authenticate
```

## The pipeline, step by step

Fourteen numbered scripts in `scripts/`, each run **with no arguments** -- they read everything from `configs/config.yaml` (and `.env` is loaded automatically if present in the working directory). The numbers give the execution order; each step depends on the previous one's actual output. `scripts/export_gui.py` is not numbered -- it is a standalone, on-demand GUI utility, never part of the sequential run (see below).

```
python scripts/01_build_tiles.py
python scripts/02_download_dem.py
python scripts/03_download_scenes.py --split all --parallel
python scripts/04_export_training_cache.py
python scripts/05_train_model.py
python scripts/06_run_inference.py
python scripts/07_build_monthly_composites.py
python scripts/08_build_mosaics.py
python scripts/09_build_index_composites.py
python scripts/10_build_index_mosaics.py
python scripts/11_run_ccdc.py
python scripts/12_run_bfast.py
python scripts/13_build_regrowth_severity.py
python scripts/14_build_change_maps.py

python scripts/export_gui.py   # on demand, never automatic, run any time
```

Every script also accepts options to restrict its scope (useful for testing without rerunning everything) -- see `--help` on each.

### 01 -- Tile grid (`01_build_tiles.py`)

The study area (`tiles.aoi_path`, a GeoPackage) is cut into a grid of square tiles (`tiles.tile_size_m`, 8 km by default), each with a small margin (`tiles.buffer_m`). `tiles.pilot_bbox` can restrict processing to a sub-area for testing without touching the rest of the config.

The result (`data/tiles/tile_registry.parquet`) is the reference table every downstream step uses -- each tile has a stable id (`tile_0000_0000`, etc.) used as a key everywhere (zarr filename, mask folder, inference output folder).

The script also splits tiles into `tiles.n_splits` balanced batches (by count + area), written to `split_assignment.parquet`. Each batch maps to a distinct GEE project to parallelize downloading (see step 03).

### 02 -- DEM (`02_download_dem.py`)

For each tile, fetches the best available digital elevation model over its window (MRDEM-30 first, Copernicus GLO-30 fallback where MRDEM has no coverage), computes slope and aspect, and stores everything in the tile's zarr store (`data/tiles/<tile_id>.zarr/dem`). The DEM keeps its source resolution (~30 m) -- it is **not** resampled at this stage (see "Working resolutions and why the DEM is separate" below).

### 03 -- Scene download (`03_download_scenes.py`)

For each tile, searches and downloads scenes from every sensor (Landsat 5/7/8/9, Sentinel-2) across the whole available year range, filtered by cloud cover and AOI coverage. Each sensor is brought to its "working resolution" (see below), with pansharpening for Landsat 7/8/9 and bilinear resampling for Sentinel-2's 20 m bands.

Two layers of parallelism, both real OS processes (never threads -- a real crash was observed with `ThreadPoolExecutor` on GEE calls):
- **Across batches** (`--split all --parallel`): one subprocess per batch, each with its own GEE project -- this is what actually parallelizes the GEE quota rather than sharing it.
- **Across tiles within a batch** (`scene_download.tile_workers`): real `multiprocessing.Process` workers, with timeout and retry.

For a quick test without re-downloading everything, `--first-year`/`--until-year`/`--sensors` restrict the scope.

**Important: Landsat 7 is excluded after 2002.** The Scan Line Corrector failed permanently in May 2003 and was never repaired -- every scene after that date has ~22% of pixels missing in diagonal stripes. The pipeline has no mechanism to fill these gaps, so post-2003 L7 is simply never downloaded.

Result: `data/tiles/<tile_id>.zarr`, one group per sensor (`l5`/`l7`/`l8`/`l9`/`s2`), holding quantized TOA bands plus two RGB composites (Raw and Shadow) for visual annotation.

### Annotation (MaskForge)

This is where the workflow leaves this repo -- annotation happens in MaskForge, a separate tool. See the dedicated section further below.

### 04 -- Training cache export (`04_export_training_cache.py`)

Once annotated masks are available (written by MaskForge under `data/masks/<tile_id>_<sensor>_<scene_id>/mask.tif`), this script assembles, for each annotated scene, a `(C, H, W)` feature stack -- spectral indices (NDVI, NDSI, NBR, etc.) + DEM layers (elevation, slope, aspect) -- aligned to the scene's grid, and caches it at `data/train_cache/<tile_id>/<sensor>/<scene_id>/features.npz`.

The DEM (stored at its source resolution, ~30 m) is resampled on the fly onto the scene's sensor working grid at this point -- this is where the two resolutions meet. The cache is invalidated automatically if the source (zarr or mask) changes (size+mtime hash), so rerunning this script is always safe and only redoes the necessary work.

Each cached scene's `features.npz` is self-contained (features, labels, feature names, and the scene's own georeferencing -- transform + CRS), so it can be handed to someone else without also sending the raw zarr scene store or the MaskForge mask. To merge received training data into your own: copy their `<tile_id>/` folder(s) directly into your `data/train_cache/`. The only requirement is that `tile_id` values never collide between contributors -- see `tiles.tile_id_prefix` in the config, which you should set **before** running `01_build_tiles.py` if you ever intend to share or merge training data (it cannot be changed retroactively without redoing every stage). `tile_id` is otherwise just a row/col index into each contributor's own AOI grid, so two different AOIs will independently produce the same `tile_0000_0000`.

To inspect a shared (or your own) training cache visually -- e.g. to check a mask against its scene in QGIS -- use `scripts/export_georeferenced_cache.py`, which reads a `train_root` and writes plain georeferenced GeoTIFFs (`features.tif` + `labels.tif` per scene) that need no knowledge of this pipeline to open.

### 05 -- Split + training (`05_train_model.py`)

Splits annotated scenes into train/val/test **by whole tile** (never two scenes from the same tile in different partitions, to avoid leakage), with repair passes so every partition sees every sensor/class. Then trains the model selected by `model.type` (defaults to `unet`).

The model and its training hyperparameters (epochs, patience, patch size, etc.) live in the `model`/`training` config sections. The final checkpoint (weights + all metadata needed to rebuild the architecture) is written to `models/checkpoints/<model_type>_best.pt`.

`model.type` is the single switch selecting which of the seven supported model technologies is used; every other subsection of `model` holds that one type's own hyperparameters and is simply ignored when a different type is active:

- `threshold` -- spectral-index thresholds tuned by Optuna (`model.threshold`); `sampler="tpe"` for Bayesian search or `sampler="grid"` for a plain exhaustive grid search.
- `random_forest` -- `sklearn.ensemble.RandomForestClassifier` wrapper (`model.random_forest`), with `class_weight="balanced"` for the uneven class frequencies typical of land cover.
- `catboost` -- CatBoost classifier (`model.catboost`), GPU by default with early stopping.
- `unet` -- U-Net segmentation model (`model.unet`), with optional FiLM spatial/sensor conditioning, bottleneck self-attention, and deep supervision.
- `deeplabv3plus` -- DeepLabv3+ (`model.deeplabv3plus`), with a configurable ResNet-family backbone, optionally ImageNet-pretrained.
- `segformer` -- SegFormer (`model.segformer`), with a configurable MiT encoder variant (`mit-b0` through `mit-b5`), optionally pretrained.

### 06 -- Inference (`06_run_inference.py`)

Runs the trained checkpoint (by default the one written at step 05) over **every** scene stored in the zarr stores -- not just annotated ones. Sliding window with Hann-window weighting to avoid edge artifacts between patches. Result: one class map per scene, written to `outputs/inference/<tile_id>/<sensor>/<scene_id>/class_map.npz`. Idempotent -- an already-processed scene is skipped unless `--overwrite`.

### 07 -- Monthly composites (`07_build_monthly_composites.py`)

Groups class maps by tile and month, reprojects them all onto the finest sensor grid available that month (priority S2 > L9 > L8 > L7 > L5), then reduces them into a single composite per a rule configurable by class (`composites.class_rules`): `median` (per-pixel majority vote -- for stable classes like forest/water/wetland/built-up), `any_occurrence` (the class wins outright if it appears even once in the month, even against a majority -- for durable disturbance flags like a cutblock or burn that must survive an outvote), or `fallback_occurrence` (only fills in where no class won a median majority -- for transient, naturally-fluctuating states like snow/ice, so it doesn't mask a real land-cover class the majority of scenes agree on). A `priority` value breaks ties between classes under the same rule on the same pixel. Result: `outputs/composites/<tile_id>/<year>-<month>/composite.npz`.

### 08 -- Mosaicking (`08_build_mosaics.py`)

Assembles per-tile monthly composites (step 7, still fragmented and in `.npz`) into a single georeferenced continuous raster per month across the whole AOI. Each tile has two bboxes in the registry: `search_*` (its own share of the AOI grid, unbuffered -- neighboring tiles' `search_bbox` fit exactly, zero gap zero overlap) and `window_*` (with the buffer, used throughout upstream processing to give the model context). Mosaicking crops each tile composite to its own `search_bbox` (removing the buffer), then places the crops side by side -- a plain crop-and-place, never blending.

Since neighboring tiles can have different resolutions in the same month (one with S2 10 m, its neighbor only Landsat 15 m), output resolution is chosen per month at the AOI scale: the finest resolution achieved anywhere that month, with nearest-neighbor resampling of coarser tiles up to that grid. Result: `outputs/mosaics/<year>-<month>/mosaic.npz`, one continuous raster per month across the whole AOI. Like every intermediate step in this pipeline, this is `.npz`, not a GeoTIFF -- use `export_gui.py` on demand to get a GIS-readable file for a specific period.

### 09 -- Monthly spectral index composites (`09_build_index_composites.py`)

The continuous-reflectance counterpart to step 7 (which only keeps classes): for each tile and month, computes NDVI/NDSI/NBR/NDWI/Tasseled-Cap/etc. per scene (never used by the classification model, only for change analysis), excludes cloud/shadow pixels via each scene's own classification, then reduces each index into median/min/max/observation-count per tile-month. Result: `outputs/index_composites/<tile_id>/<year>-<month>/indices.npz`.

**Not used by CCDC or BFAST** (steps 11/12), which fit directly on raw per-scene reflectance bands instead -- see the note under step 11. These monthly index composites (and their AOI-wide mosaics from step 10) are consumed only by step 13 (dNBR severity and NDVI regrowth trajectory).

### 10 -- AOI-wide index mosaics (`10_build_index_mosaics.py`)

Step 8's counterpart for continuous indices instead of classes -- same crop-and-place geometry. Result: `outputs/index_mosaics/<year>-<month>/indices.npz`.

### 11 -- Local CCDC/COLD (`11_run_ccdc.py`)

Per-pixel breakpoint detection (date + per-band magnitude) via `pyxccd` (COLD), run **entirely locally** on this pipeline's own already-pansharpened stored data -- never against raw GEE collections (which would produce breaks on different pixels than the ones the model sees). The cloud/shadow/snow/water mask comes from this pipeline's own classification, not a GEE cloud percentage. LandTrendr was evaluated and dropped (no local alternative exists).

CCDC is fit on the six normalized reflectance bands every sensor is aligned to (blue/green/red/nir/swir1/swir2) -- the same per-pixel, multi-decade time series built once per tile on a single shared grid (the finest resolution any sensor ever achieved for that tile). This is **per-scene**, not the monthly index composites from step 9/10: COLD needs one observation per actual scene date to fit its time-series model, so feeding it a pre-aggregated monthly value would throw away the temporal resolution breakpoint detection depends on.

### 12 -- BFAST-Monitor (`12_run_bfast.py`)

A fire-specific cross-check (BFAST is ~96% accurate for fire vs. ~73% for CCDC/LandTrendr-family methods per the 2025 comparison cited in the initial design), hand-reimplemented in pure Python (the `bfast` pip package has been dead since 2021, R was explicitly ruled out to stay on a single stack). Harmonic regression over a stable history period + a closed-boundary CUSUM statistical test on the monitored period's residuals.

Unlike CCDC (fit directly on the six raw bands), BFAST is applied to a single derived index: NBR (normalized burn ratio), computed from that same aligned band stack -- the standard fire-sensitive index, and what the univariate `bfastmonitor` formulation this module reimplements expects.

### 13 -- NDVI regrowth and dNBR severity (`13_build_regrowth_severity.py`)

For every pixel with a detected break (CCDC or BFAST): dNBR between the composite just before and just after the break (continuous burn severity, complementary to the model's categorical `burned_bare_disturbed` class), and month-by-month NDVI trajectory since the break (vegetation regrowth -- no model class can give this, it is a trajectory over time, not a point-in-time state).

### 14 -- Combined change maps (`14_build_change_maps.py`)

Combines classification + CCDC/BFAST + dNBR into a final output per pair of periods (annual: same month, consecutive years; month-to-month: consecutive stored months). Three classification-change confidence bands, kept deliberately separate (never merged): `raw` (noisy immediate comparison), `persistent` (confirmed by the following month), `corroborated` (confirmed by a nearby CCDC/BFAST break). Cloud/shadow pixels in either compared period are excluded, never counted as change.

### On-demand export (`export_gui.py`)

Converts any of this pipeline's stored outputs into GeoTIFF or CSV. The only place in the pipeline that pays the GeoTIFF write cost -- never called automatically by another step, and not numbered since it is not part of the sequential run: it can be run at any point once the file it targets exists.

A GUI, not a CLI: `python scripts/export_gui.py` (it relaunches itself under `streamlit run` automatically). Point it at an `outputs/...` (or `data/tiles/...`, for DEM) folder, pick a file from the list, preview it, then export. Still on demand and one file at a time -- it never scans and exports in bulk.

- **Dense rasters** (class map, composite, mosaic, index composite/mosaic, change map, dNBR, bfast, per-tile DEM elevation/slope/aspect) -> GeoTIFF (COG), previewed as a quicklook + shape/dtype/min-max per band before export. Logic in `landscape_change_detection_pipeline.export.geotiff`.
- **Sparse tables** that are not grids -- CCDC's per-pixel segment table (0..N segments per pixel) and the NDVI regrowth trajectory (0..N (pixel, month) rows per pixel) -- offer a choice: export the full table as CSV (`landscape_change_detection_pipeline.export.csv_export`), or a derived per-pixel *summary* GeoTIFF (for CCDC: latest break date, segment count, latest change probability/observation count -- the full segment table itself is not a grid and is lost in that derivation).

## Config: `config.yaml` vs `config.example.yaml`

- `configs/config.example.yaml`: tracked by git, serves as the template. No personal information in it (no local path, no real GEE project name).
- `configs/config.yaml`: gitignored, holds your real values (paths, GEE projects). This is what every script reads by default.

GEE projects are declared once in `gee.projects` (the full list of Earth Engine projects available, to spread load). `scene_download.ee_projects` can stay empty -- in that case download automatically round-robins over `gee.projects`. The number of parallel batches is set via `tiles.n_splits`.

## Working resolutions and why the DEM is separate

Each sensor has its own uniform "working resolution" (all its bands stored on a single grid -- a CNN takes one tensor at a time, resolutions cannot be mixed within a single pass):

| Sensor | Working resolution | How it's reached |
|---|---|---|
| Landsat 5 | 30 m | native, no processing (TM never had a panchromatic band) |
| Landsat 7/8/9 | 15 m | Gram-Schmidt Adaptive pansharpening on blue/green/red/NIR (a real detail gain); SWIR1/SWIR2 simply resampled from 30 m to 15 m (alignment only, no invented information) |
| Sentinel-2 | 10 m | native 10 m bands; 20 m bands (red-edge/SWIR) resampled to 10 m via bilinear interpolation |

The DEM is downloaded **once per tile**, at its source resolution (~30 m) -- there is no per-tile "5 versions of the DEM." It is resampled on the fly onto the relevant sensor's grid when building the training/inference feature stack (`features/training_cache.py::build_feature_stack`), with a provenance tag (`native` or `resampled_for_alignment`) that honestly states whether those DEM pixels are really at that resolution or just resampled from a coarser DEM. This is deliberate: duplicating the DEM into 3 versions per tile would add no real information and would only complicate storage for nothing.

## MaskForge: annotating from the zarr stores

MaskForge is the annotation tool -- a separate Tauri/React project with a Python sidecar, not a dependency of this repo. It already reads this pipeline's zarr stores natively (a plugin already written and tested on the MaskForge side, `sidecar/maskforge_core/plugins/zarr_reader.py`).

**The addressing problem.** A zarr store is a *folder*, not a file, and a single folder holds every scene from every sensor for one tile. So you cannot just "open a file" the way you would a plain GeoTIFF. MaskForge uses a composite path:

```
<tile_id>.zarr!<sensor>/<scene_id_or_index>/<composite>
```

where `<composite>` is `rgb_raw`, `rgb_shadow`, or `toa:B4,B3,B2` (any band combination by name).

**To start an annotation session:**

1. In MaskForge's scene discovery config (`DiscoveryConfig`), point `source_root` at `data/tiles/` (the folder holding every `<tile_id>.zarr`).
2. Add `.zarr` to `scan_rule.file_extensions` -- **without this MaskForge will not detect the stores** (by default it only scans `.tif/.tiff/.png/.jpg/.jpeg`).
3. MaskForge then lists every `(sensor, scene)` in every store as its own entry to annotate, with an id `{tile_id}_{sensor}_{scene_id}` -- this exact id is what script 04 later parses back to find which tile/sensor/scene a mask came from.
4. Load the class palette: generated from `configs/classes.yaml` (single source of truth, shared between this pipeline and MaskForge) via `to_maskforge_palette()`.
5. Annotate normally -- navigate scenes, switch between RGB Raw and RGB Shadow (the latter brightens shadowed areas in dense forest, useful in mountainous terrain), paint the mask with the palette.
6. Save. The mask is written as an RGBA GeoTIFF under `<mask_root>/{tile_id}_{sensor}_{scene_id}/mask.tif`, exactly the layout script 04 expects (configured via `features.mask_root` in `config.yaml`, which must point at the same folder on both sides).

No need to convert anything to GeoTIFF beforehand -- MaskForge reads the zarr directly, and writes the final mask as GeoTIFF (a standard format, easy to inspect with any GIS tool).
