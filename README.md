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

For each tile, fetches the best available digital elevation model over its window (MRDEM-30 first, Copernicus GLO-30 fallback where MRDEM has no coverage), computes slope and aspect, and stores everything in the tile's zarr store (`data/tiles/<tile_id>.zarr/dem`). The DEM is reprojected onto the pipeline's uniform 10 m reference grid (`dem.target_resolution_m`) at this stage -- it **is** the grid every later sensor scene must land on exactly (see "Working resolution: one unified 10 m grid, pinned to the DEM" below).

### 03 -- Scene download (`03_download_scenes.py`)

For each tile, searches and downloads scenes from every sensor (Landsat 5/7/8/9, Sentinel-2) across the whole available year range, filtered by cloud cover and AOI coverage. Every sensor is resampled (bilinear) directly onto the tile's DEM grid (see below) -- there is no pansharpening anywhere in this pipeline: Landsat Collection 2 Level-2 Surface Reflectance carries no panchromatic band at all.

Two layers of parallelism, both real OS processes (never threads -- a real crash was observed with `ThreadPoolExecutor` on GEE calls):
- **Across batches** (`--split all --parallel`): one subprocess per batch, each with its own GEE project -- this is what actually parallelizes the GEE quota rather than sharing it.
- **Across tiles within a batch** (`scene_download.tile_workers`): real `multiprocessing.Process` workers, with timeout and retry.

For a quick test without re-downloading everything, `--first-year`/`--until-year`/`--sensors` restrict the scope.

**Important: Landsat 7 is excluded after 2002.** The Scan Line Corrector failed permanently in May 2003 and was never repaired -- every scene after that date has ~22% of pixels missing in diagonal stripes. The pipeline has no mechanism to fill these gaps, so post-2003 L7 is simply never downloaded.

Result: `data/tiles/<tile_id>.zarr`, one group per sensor (`l5`/`l7`/`l8`/`l9`/`s2`), holding quantized TOA bands plus RGB composites for visual annotation. Four views are available (`rgb_composites` config), each independently toggleable: `true_color` and `true_color_shadow` (asinh-compressed + gamma-boosted, recovering shadow detail) are on by default; `natural_color` (SWIR2/NIR/red) and `color_infrared` (NIR/red/green, the standard vegetation false-color composite) are off by default until visually validated.

### Annotation (MaskForge)

This is where the workflow leaves this repo -- annotation happens in MaskForge, a separate tool. See the dedicated section further below.

### 04 -- Training cache export (`04_export_training_cache.py`)

Once annotated masks are available (written by MaskForge under `data/masks/<tile_id>_<sensor>_<scene_id>/mask.tif`), this script assembles, for each annotated scene, a `(C, H, W)` feature stack -- spectral indices (NDVI, NDSI, NBR, etc.) + DEM layers (elevation, slope, aspect) -- aligned to the scene's grid, and caches it at `data/train_cache/<tile_id>/<sensor>/<scene_id>/features.npz`.

Since every sensor scene already lands exactly on the tile's DEM grid at fetch time (step 03), the DEM layers normally need no further resampling here -- the on-the-fly resample path only fires as a fallback if a shape mismatch is actually detected. The cache is invalidated automatically if the source (zarr or mask) changes (size+mtime hash), so rerunning this script is always safe and only redoes the necessary work.

Each cached scene's `features.npz` is self-contained (features, labels, feature names, and the scene's own georeferencing -- transform + CRS), so it can be handed to someone else without also sending the raw zarr scene store or the MaskForge mask. To merge received training data into your own: copy their `<tile_id>/` folder(s) directly into your `data/train_cache/`. The only requirement is that `tile_id` values never collide between contributors -- see `tiles.tile_id_prefix` in the config, which you should set **before** running `01_build_tiles.py` if you ever intend to share or merge training data (it cannot be changed retroactively without redoing every stage). `tile_id` is otherwise just a row/col index into each contributor's own AOI grid, so two different AOIs will independently produce the same `tile_0000_0000`.

To inspect a shared (or your own) training cache visually -- e.g. to check a mask against its scene in QGIS -- use `scripts/export_georeferenced_cache.py`, which reads a `train_root` and writes plain georeferenced GeoTIFFs (`features.tif` + `labels.tif` per scene) that need no knowledge of this pipeline to open.

### 05 -- Split + training (`05_train_model.py`)

Splits annotated scenes into train/val/test **by whole tile** (never two scenes from the same tile in different partitions, to avoid leakage), with repair passes so every partition sees every sensor/class. Then trains the model selected by `model.type` (defaults to `unet`).

The model and its training hyperparameters (epochs, patience, patch size, etc.) live in the `model`/`training` config sections. The final checkpoint (weights + all metadata needed to rebuild the architecture) is written to `models/checkpoints/<model_type>_best.pt`.

`model.type` is the single switch selecting which of the eight supported model technologies is used; every other subsection of `model` holds that one type's own hyperparameters and is simply ignored when a different type is active:

- `threshold` -- spectral-index thresholds tuned by Optuna (`model.threshold`); `sampler="tpe"` for Bayesian search or `sampler="grid"` for a plain exhaustive grid search.
- `random_forest` -- `sklearn.ensemble.RandomForestClassifier` wrapper (`model.random_forest`), with `class_weight="balanced"` for the uneven class frequencies typical of land cover. No incremental fit, so the full dense pixel matrix must fit in RAM at once -- prefer `lightgbm` on a large training corpus.
- `catboost` -- CatBoost classifier (`model.catboost`), GPU by default with early stopping.
- `lightgbm` -- LightGBM classifier (`model.lightgbm`), CPU by default with early stopping; the RAM-conscious alternative to `random_forest` on a large corpus (histogram-binned `Dataset`, no full dense matrix kept resident).
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

Per-pixel breakpoint detection (date + per-band magnitude) via `pyxccd` (COLD), run **entirely locally** on this pipeline's own already-resampled-onto-the-DEM's-grid stored data -- never against raw GEE collections (which would produce breaks on different pixels than the ones the model sees). The cloud/shadow/snow/water mask comes from this pipeline's own classification, not a GEE cloud percentage. LandTrendr was evaluated and dropped (no local alternative exists).

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

## Working resolution: one unified 10 m grid, pinned to the DEM

Every sensor -- Landsat 5/7/8/9 and Sentinel-2 alike -- shares one project-wide working resolution, 10 m (`dem.target_resolution_m`), pinned to the exact transform/shape of the tile's own stored DEM:

| Sensor | How it reaches the DEM's 10 m grid |
|---|---|
| Landsat 5/7/8/9 | native 30 m Surface Reflectance bands, bilinear-resampled directly onto the DEM's grid |
| Sentinel-2 | native 10 m bands reprojected onto the DEM's grid (no resolution change); 20 m bands (red-edge/SWIR) bilinear-resampled onto the same grid |

10 m is Sentinel-2's native ceiling -- nothing in the stack genuinely resolves finer than that -- so this is the best achievable *uniform* resolution, not an arbitrary choice. Every Landsat-derived pixel carries a resolution-semantics caveat: it is genuinely ~30 m information smoothed onto a finer grid, real detail is not being invented, only alignment; never read it as true 10 m resolving power. There is no pansharpening anywhere in this pipeline any more -- Landsat Collection 2 Level-2 Surface Reflectance (the collection this pipeline now uses, see below) carries no panchromatic band at all.

The DEM is fetched **once per tile**, reprojected directly onto this 10 m reference grid (`dem/sources.py`) -- it *is* the grid, not a separate resolution that needs reconciling later. Every later array (every sensor's scenes, every feature stack, every label) is checked against it via `dem/grid_check.py::check_grid_alignment`, called right after each scene is written (`scenes/process_scene.py::process_and_store_scene`); a mismatch is a hard per-scene failure, never a silent pass-through. Band provenance (`native` or `resampled_for_alignment`) still records whether a given band is genuinely native at 10 m (only Sentinel-2's four 10 m bands) or resampled onto the grid.

Every collection is now atmospherically corrected **Surface Reflectance** (SR), not TOA: `LANDSAT/LT05|LE07|LC08|LC09/C02/T1_L2` and `COPERNICUS/S2_SR_HARMONIZED`. See `docs/decisions/unified_10m_grid.md` for the full rationale.

## MaskForge: annotating from the zarr stores

MaskForge is the annotation tool -- a separate Tauri/React project with a Python sidecar, not a dependency of this repo. It already reads this pipeline's zarr stores natively (a plugin already written and tested on the MaskForge side, `sidecar/maskforge_core/plugins/zarr_reader.py`).

**The addressing problem.** A zarr store is a *folder*, not a file, and a single folder holds every scene from every sensor for one tile. So you cannot just "open a file" the way you would a plain GeoTIFF. MaskForge uses a composite path:

```
<tile_id>.zarr!<sensor>/<scene_id_or_index>/<composite>
```

where `<composite>` is `rgb_true_color`, `rgb_true_color_shadow`, `rgb_natural_color`, `rgb_color_infrared` (whichever are enabled in `rgb_composites`), or `toa:B4,B3,B2` (any band combination by name).

**To start an annotation session:**

1. In MaskForge's scene discovery config (`DiscoveryConfig`), point `source_root` at `data/tiles/` (the folder holding every `<tile_id>.zarr`).
2. Add `.zarr` to `scan_rule.file_extensions` -- **without this MaskForge will not detect the stores** (by default it only scans `.tif/.tiff/.png/.jpg/.jpeg`).
3. MaskForge then lists every `(sensor, scene)` in every store as its own entry to annotate, with an id `{tile_id}_{sensor}_{scene_id}` -- this exact id is what script 04 later parses back to find which tile/sensor/scene a mask came from.
4. Load the class palette: generated from `configs/classes.yaml` (single source of truth, shared between this pipeline and MaskForge) via `to_maskforge_palette()`.
5. Annotate normally -- navigate scenes, switch between RGB Raw and RGB Shadow (the latter brightens shadowed areas in dense forest, useful in mountainous terrain), paint the mask with the palette.
6. Save. The mask is written as an RGBA GeoTIFF under `<mask_root>/{tile_id}_{sensor}_{scene_id}/mask.tif`, exactly the layout script 04 expects (configured via `features.mask_root` in `config.yaml`, which must point at the same folder on both sides).

No need to convert anything to GeoTIFF beforehand -- MaskForge reads the zarr directly, and writes the final mask as GeoTIFF (a standard format, easy to inspect with any GIS tool).
