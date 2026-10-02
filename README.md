# landscape-change-detection-pipeline

Take a study area, download every Landsat 5/7/8/9 and Sentinel-2 scene ever acquired over it, hand-label a few of them, train a model that classifies every pixel (forest, water, snow, burn...), run it on all scenes, and then use the resulting time series to find **what changed, when, and how the land recovered**.

It started as a glacier project and was generalized: the list of classes lives in one file (`configs/classes.yaml`), so the same machinery works for any land-cover or land-surface problem, not just the one it is currently applied to. Nothing in the code should assume a particular study area or a particular class list. If you find something that does, that is a bug.

This README explains how the pieces fit together, what each script does, and the few rules that keep it from breaking.

---

## 1. Setup

```
conda env create -f environment.yml
conda activate landscape-change-detection-pipeline
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # GPU build, see requirements-torch.txt
pip install -r requirements-torch.txt
pip install -e .
```

Copy `.env.example` to `.env` and `configs/config.example.yaml` to `configs/config.yaml`, then fill in your paths and Earth Engine projects. Both files are gitignored (machine-specific). **`config.example.yaml` is the only config tracked by git**: whenever you add a setting, add it to `config.py` (pydantic), `config.example.yaml` **and** your own `config.yaml`.

Earth Engine, once:

```
earthengine authenticate
```

## 2. The one rule: everything goes through the config

You run a script with **no options**. Every parameter (tiles, workers, thresholds, what is enabled) is read from `configs/config.yaml`, so a run can be reproduced from the config alone. A few scripts keep a flag to say *which* files to read (`--config`, `--env-file`, `--classes`), and the downloader / pseudo-labeler keep scoping flags (`--split`, `--checkpoint`...). That is all.

Two more habits shared by every stage:

- **Each element has its own `enabled`** in the config. Phenology off, SHAP off, recovery curves off: no code change.
- **Live logs.** A long job prints progress as it goes (`[i/N]`), it never goes silent for an hour.

## 3. The pipeline at a glance

```
01 tiles -> 02 DEM -> 03 scenes -> [annotate in MaskForge] -> 04 training cache
   -> 05 train -> (05b pseudo-labels) -> (05c feature importance)
   -> 06 inference -> 07 monthly composites -> 08 mosaics
   -> 09 break detection -> 10 snow/ice/water -> 11 regrowth/severity
   -> 12 persistence + named events -> 13 change maps -> 14 vegetation dynamics + recovery

export_gui.py   <- on demand, turns any output into GeoTIFF / CSV
```

Run them in order: each step needs the real output of the previous one.

```
python scripts/01_build_tiles.py
python scripts/02_download_dem.py
python scripts/03_download_scenes.py --split all --parallel
python scripts/04_export_training_cache.py
python scripts/05_train_model.py
python scripts/05b_generate_pseudo_labels.py   # optional (training.pseudo_label.enabled)
python scripts/05c_feature_importance.py       # optional, analysis only
python scripts/06_run_inference.py
python scripts/07_build_monthly_composites.py
python scripts/08_build_mosaics.py
python scripts/09_run_change_detection.py
python scripts/10_build_snow_water_dynamics.py
python scripts/11_build_regrowth_severity.py
python scripts/12_build_landcover_persistence.py
python scripts/13_build_change_maps.py
python scripts/14_build_vegetation_dynamics.py
streamlit run scripts/export_gui.py            # or plain `python scripts/export_gui.py`
```

Intermediate arrays are `.npz`, not GeoTIFF: writing GeoTIFFs everywhere is slow and fills the disk. Anything that needs to leave the pipeline goes through `export_gui.py`.

---

## 4. The scripts, one by one

### 01 Tiles: `01_build_tiles.py`
Cuts the study area (`tiles.aoi_path`) into square tiles (`tiles.tile_size_m`, 8 km by default) with a small buffer (`tiles.buffer_m`). `tiles.pilot_bbox` restricts everything to a sub-area for testing. Writes `data/tiles/tile_registry.parquet`, the table every later step reads, and splits the tiles into `tiles.n_splits` balanced batches (one per Earth Engine project, to spread the quota).
Set `tiles.tile_id_prefix` **before** running it if you will ever merge training data with someone else's: tile ids are otherwise just row/col indices and two studies will collide.

### 02 DEM: `02_download_dem.py`
Fetches the best DEM per tile (MRDEM-30, Copernicus GLO-30 as fallback), derives slope and aspect, and stores them in the tile's zarr. The DEM is reprojected onto the pipeline's **10 m reference grid**; this grid is the one every sensor will be forced onto.

### 03 Scenes: `03_download_scenes.py`
Searches and downloads every scene of every sensor, filtered by cloud cover and AOI coverage, straight onto the DEM grid (bilinear for Landsat's 30 m). There is no pansharpening, even though the Landsat TOA product has a 15 m panchromatic band (`B8`): every sensor is kept on the same footing (see `docs/decisions/toa_rollback.md`). Two layers of real OS-level parallelism (never threads, GEE crashed with them): one subprocess per batch, and worker processes across tiles within a batch.
All collections are top-of-atmosphere (TOA) reflectance, not surface reflectance (`LANDSAT/.../C02/T1_TOA`, `COPERNICUS/S2_HARMONIZED`).
**Landsat 7 after 2002 is not downloaded**: the scan-line corrector failed in 2003 and about a fifth of every later scene is missing in stripes.
Result: `data/tiles/<tile_id>.zarr`, one group per sensor, plus RGB composites for annotation (`rgb_composites`: true color, true color with shadow detail, natural color, color infrared; each switchable).

### Annotation (MaskForge, outside this repo)
See section 8. Short version: point MaskForge at the folder that contains the `.zarr` stores and it finds them.

### 04 Training cache: `04_export_training_cache.py`
For every mask MaskForge wrote (`data/masks/<tile>_<sensor>_<scene>/mask.tif`), builds the `(C, H, W)` feature stack (spectral indices + DEM layers + optional day-of-year / lat-lon) and caches it under `data/train_cache/`: memory-mappable `features.npy` (float16) + `labels.npy`, plus metadata. Safe to re-run: a scene is only redone if its source changed. Each cached scene is self-contained (features, labels, names, georeferencing), so you can hand a folder to a colleague or merge theirs into yours.
Which indices go in the model is `features.index_names` in the config. **Changing it changes the model's input size: re-run 04, then 05.** `scripts/export_georeferenced_cache.py` writes the cache back out as plain GeoTIFFs for a visual check in QGIS.

### 05 Split + training: `05_train_model.py`
Splits annotated scenes into train / val / test (`split.split_by`: `tile` is leakage-safe, `scene` is looser and lets two dates of the same tile land in different partitions, so the validation score is then optimistic), with repair passes so each partition sees every sensor and class. Then trains `model.type`.

**The two supported models are `unet` and `catboost`.** The other types (`threshold`, `random_forest`, `lightgbm`, `deeplabv3plus`, `segformer`) are still in the code but have not been set up or tuned; if you switch to one, expect to do that work yourself. New analysis tools (05c) refuse them explicitly.

- `unet`: the main model. Optional per-sensor conditioning (FiLM), attention at the bottleneck, deep supervision. Checkpoint: `models/checkpoints/unet_best.pt`, self-describing (weights, class names, feature names, normalization), so inference needs nothing else.
- `catboost`: pixel-wise gradient boosting on GPU with Optuna search and early stopping. Fast, no spatial context.

For the rare, confusable classes, the `training` section has levers that act on different things; try them roughly in this order:
- `scene_oversampling`: draw scenes holding rare classes more often (gentle, try first);
- `class_weighting` + `focal_gamma` + `dice_weight`: reweight the loss (too aggressive weights destabilize training);
- `class_merge`: fold a class into another for training and evaluation (`built_up_infrastructure: bare_ground` today);
- `confusion_penalties`: extra loss on one specific (true, predicted) pair seen in a confusion matrix;
- `radiometric_augmentation`: brightness / contrast / noise on the index channels only;
- `pseudo_label`: see 05b.

A report with the config, per-epoch history, per-class IoU and confusion matrices of train / val / test is written next to the checkpoint (`<model>_run_report.json`).

### 05b Pseudo-labels: `05b_generate_pseudo_labels.py` (optional)
Scores every un-annotated scene with a trained checkpoint, keeps only confident pixels, and writes them in the same cache layout so that 05 can mix them in at a reduced weight (`training.pseudo_label`). Streams one scene at a time, since all of `data/tiles` is far bigger than RAM. Beware: it also propagates the model's own mistakes, which matters most on weak classes (burned, ice).

### 05c Feature importance: `05c_feature_importance.py` (optional, analysis only)
Answers "which input features does the model really need, and are my classes separable at all?". Everything is configured in the `feature_importance:` section; the script takes no options. **`unet` and `catboost` only.**

| Method | What it tells you |
|---|---|
| `permutation` | Drop in mIoU when a feature (or a group such as Tasseled Cap) is destroyed, repeated several times (mean +/- std). UNet: the channel is swapped with the same channel of another patch, on GPU-resident 256 px patches, so it takes minutes. |
| `shap` | Per-class mean absolute SHAP (CatBoost TreeSHAP): which feature separates which class. |
| `dropcolumn` | Retrain without the feature and compare mIoU. The only method that says "as good without it", and it is not fooled by a correlated twin. |
| `separability` | Model-free Jeffries-Matusita distance between every pair of annotated classes (0 identical, 2 separable), and the best single features for the weak pairs. Use it **before annotating a new class**. |

How to read it: a trained network gets worse whenever you break *any* channel, so permutation alone can never call a feature useless; it is informative only. A feature is a **drop candidate** when drop-column (change within the fit-to-fit noise, `drop_threshold`) *and* SHAP (at most 10 % of the top feature) agree. For a UNet run, SHAP and drop-column use a CatBoost **proxy** trained on a pixel sample (spectral signal only, no spatial context). Always retrain the real model without a candidate before removing it, one at a time, and never remove both members of a correlated pair together. JM is a Gaussian, in-sample estimate: optimistic, a floor for what a pixel-wise model can do, not what a UNet can recover from context.
Output: `outputs/feature_importance/feature_importance.npz` (tables, exportable to CSV from `export_gui.py`) and a `.json` summary.

### 06 Inference: `06_run_inference.py`
Runs the checkpoint over **every** stored scene (not just annotated ones) with a sliding window and Hann weighting to hide patch edges. Writes `outputs/inference/<tile>/<sensor>/<scene>/class_map.npz`. Idempotent. The model works in **dense** class ids (position in `classes.yaml`); the file is written back in **raw** ids (the `id` field), because every downstream tool and MaskForge speak raw ids. A merged-away class is folded onto its target.
`inference.ambiguity_threshold` (off by default) turns on a tie-break between the top-two classes using `class_priority_order`; empty means the name-based default in `inference/engine.py`.

### 07 Monthly composites: `07_build_monthly_composites.py`
Groups the scene class maps by tile and month, puts them on the finest grid available (S2 > L9 > L8 > L7 > L5) and reduces them to one map per month with a rule per class (`composites.class_rules`, which uses **raw ids**):

- `median`: per-pixel majority vote (default; stable classes);
- `any_occurrence`: wins if seen even once (durable flags such as a burn);
- `fallback_occurrence`: fills in only where nothing won a majority (transient states). `priority` decides between several of them, **lower wins**.

Today: snow (1) > firn (2) > ice (3) > any other class. `composites.ignore_classes` (default cloud, shadow) lists classes that cast no vote, so a month is decided by its clear observations; set it to `[]` and a cloud-majority pixel becomes cloud in the composite. The vote arrays are sized by the highest raw id, not by the class count (ids are not contiguous: 11/12 retired, 16/17 appended).
Output: `outputs/composites/<tile>/<year>-<month>/composite.npz`.

### 08 Mosaics: `08_build_mosaics.py`
Crops every tile composite to its own unbuffered area and places the crops side by side: one continuous raster per month for the whole AOI. Plain crop-and-place, no blending. Resolution per month is the finest achieved anywhere that month.

### 09 Break detection: `09_run_change_detection.py`
Per-pixel temporal segmentation (CCDC-style) on the pixel's full multi-decade series of NBR, NDVI, NDWI and TC brightness, run **locally** on the stored data (never on raw GEE collections, which would put breaks on different pixels than the model sees). Observations whose scene class is in `change_detection.masked_classes` (cloud, shadow, snow, firn, ice, water) are ignored: snow onset would otherwise register as a break every winter. Output: `segments.npz`, one row per segment (dates, per-feature magnitude, level, trend, seasonal amplitude, class before and at the break). Much of the behaviour is in the `min_magnitude`, `persist_days`, `season_window` filters.

### 10 Snow, firn, ice, open water: `10_build_snow_water_dynamics.py`
Per pixel and per water year (October to September by default): how many observed months were snow / firn / ice / open water, plus the first and last snow month. Categorical, read from the classifier; kept apart from step 09 on purpose. Older files without firn are read with `n_months_firn` = 0.

### 11 Regrowth and severity: `11_build_regrowth_severity.py`
For each pixel with a break: dNBR across the break (continuous burn severity) and the monthly NDVI trajectory afterwards (regrowth).

### 12 Persistence and named events: `12_build_landcover_persistence.py`
Turns persistent classes (burned today) from a monthly state into dated intervals (appearance, end), with a sliding majority vote over observed months. The same run **names the events** (`event_typing`): fire, cutblock (with duration and forest recovery), permanent clearing, canopy decline, cropland. Cutblock, crop and built-up are *uses*, not classifier states: they are read from how the states behave over time. Output: `outputs/landcover_persistence/` and `outputs/change_events/`.

### 13 Change maps: `13_build_change_maps.py`
Annual (same month, consecutive years) and month-to-month change maps with three separate confidence bands: `raw`, `persistent` (confirmed the next month), `corroborated` (backed by a break from step 09), plus dNBR. Cloud / shadow pixels are never counted as change.

### 14 Vegetation dynamics and recovery: `14_build_vegetation_dynamics.py`
New *data*, not maps. From one temporary monthly NDVI cube per tile:
- **phenology** (peak, amplitude, start / end / length of the growing season), **trends** (Theil-Sen + Mann-Kendall, also since the last break), **anomalies** (monthly z-score against the pixel's own climatology), **variability**;
- **recovery analysis** per named event: recovery curves (right-censoring explicit, never a fake "did not recover"), succession (dominant class by year, Markov and class-share tables), terrain factors (elevation, slope, aspect).
Every element has its own `enabled`.

### Export: `export_gui.py`
A Streamlit GUI, the only place that pays the GeoTIFF cost. Point it at a folder (`outputs/...`, `data/tiles/...` for DEM), press *Scan* (names only, nothing is loaded), pick a file, preview it on a map, export. **Export everything** writes every raster (GeoTIFF) and every table (CSV) found under the folder, in one go, one file at a time so RAM stays flat. Any `.npz` with `tbl_<table>__<column>` arrays and a `product` key is exported as a CSV table automatically; that is how the 05c results come out.

### Helpers
- `export_georeferenced_cache.py`: training cache to GeoTIFF (see 04).
- `gen_synthetic_masks.py`: dev-only fake annotations to smoke-test 04 to 07 without a real MaskForge session.

---

## 5. Classes

`configs/classes.yaml` is the single source of truth, shared with MaskForge. Currently 16: forest, bare ground, grassland / herbaceous, cultivated agriculture, wetland / marsh, open water, cutblock, burned / disturbed, snow, ice, rock (natural bare), built-up, cloud, shadow, **firn**, **sand / gravel**. Cloud and shadow are imaging artifacts (`change_eligible: false`).

Two id spaces, do not mix them up:

| | Where it is used | Example |
|---|---|---|
| **raw id** (`id:` in the yaml) | painted masks, `class_map.npz`, composites, `composites.class_rules`, MaskForge | firn = 16 |
| **dense id** (position in the file, 0..N-1) | the model's output channels, the loss, `rare_classes`, `confusion_penalties`, `inference.class_priority_order`, `scene_oversampling` | firn = 14 |

Where you can, the config uses class **names** (`masked_classes`, `ignore_classes`, `event_typing`...), which survive renumbering.

### Adding a class (checklist)
1. Append it at the **end** of `classes.yaml` with a **new raw id**. Never renumber or reuse an id: ids are burned into every painted mask, and dense ids of existing classes stay put when you append.
2. If it is a transient or snow-like state, give it a rule in `composites.class_rules` (raw id) and add its name to the `masked_classes` lists of `change_detection`, `landcover_persistence`, `event_typing` and `vegetation_dynamics` (anything that is not the land surface).
3. If it should win ties, add its name to `DEFAULT_CLASS_PRIORITY_NAMES` (`inference/engine.py`).
4. Check `training.class_merge`, `scene_oversampling.rare_classes`, `pseudo_label.rare_classes`, `confusion_penalties` (dense ids).
5. Annotate a few polygons, re-run 04, and run 05c's `separability` to see whether the class is distinguishable from the others at all.
6. **Retrain**: the model's output size changed, an older checkpoint will not load.

A study that does not have `firn` can simply leave it out of its `classes.yaml`; remove its `class_rules` entry and its name from the `masked_classes` lists in that study's config (an unknown name is an error, on purpose, so a typo does not go unnoticed).

---

## 6. Working resolution: one 10 m grid pinned to the DEM

Every sensor shares one grid: 10 m, the exact transform and shape of the tile's own DEM. Sentinel-2's 10 m bands are reprojected, its 20 m bands and all of Landsat's 30 m bands are bilinear-resampled onto it. 10 m is the best uniform resolution the stack can offer (it is Sentinel-2's ceiling), but a Landsat pixel is still ~30 m information on a finer grid: alignment, not new detail. Every scene is checked against the DEM grid right after it is written (`dem/grid_check.py`); a mismatch is a hard failure, never silently passed through. Rationale in `docs/decisions/unified_10m_grid.md`.

This also bounds what the model can do at **boundaries**: at 10 m, a 1-2 pixel offset between annotation and image is the same size as the error you are trying to measure.

## 7. Config: `config.yaml` vs `config.example.yaml`

- `config.example.yaml`: tracked, generic template, no personal information.
- `config.yaml`: gitignored, your real values. What every script reads by default.

Earth Engine projects are listed once in `gee.projects`; `scene_download.ee_projects` can stay empty and download round-robins over them. `tiles.n_splits` sets the number of parallel batches. Worker settings (`workers`, `tile_workers`, `cube_workers`...) are memory traps on Windows (each process is a full interpreter): the comments in the config say what froze or exhausted RAM; raise them gradually.

## 8. MaskForge: annotating from the zarr stores

MaskForge is a separate tool (Tauri/React + Python sidecar), not a dependency of this repo. It reads this pipeline's zarr stores natively.

A zarr store is a folder holding every scene of every sensor for one tile, so MaskForge addresses a scene with a composite path: `<tile_id>.zarr!<sensor>/<scene_id_or_index>/<composite>`, where the composite is `rgb_true_color`, `rgb_true_color_shadow`, `rgb_natural_color`, `rgb_color_infrared` (whichever are enabled) or `toa:B4,B3,B2` (any bands by name).

**To start annotating:** give MaskForge the folder that contains the `.zarr` stores (`data/tiles/`) as its source root. It detects the stores automatically and lists every `(sensor, scene)` as its own entry, with the id `{tile_id}_{sensor}_{scene_id}`: that exact id is what script 04 parses back. Then:

1. Load the class palette, generated from `classes.yaml` (`to_maskforge_palette()`), so both tools always share the same classes and colors.
2. Paint the masks. Switch between the normal and the shadow-boosted RGB when working in dense forest or steep terrain.
3. Save. The mask is written as an RGBA GeoTIFF to `<mask_root>/{tile_id}_{sensor}_{scene_id}/mask.tif`, the layout script 04 expects (`features.mask_root` must point at the same folder on both sides).

Nothing to convert beforehand.
