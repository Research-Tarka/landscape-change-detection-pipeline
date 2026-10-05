"""Configuration loading and validation for landscape-change-detection-pipeline.

Decision: use pydantic (v2 BaseModel) rather than omegaconf or a hand-written
PyYAML validation pass: the config will grow into
many nested sections with mixed types (paths, floats, ints, bools, nested
lists of dicts for tile registries, sensor specs, and HPO search spaces).
Pydantic gives per-field type coercion and a single ValidationError listing
every offending field (path + expected type + given value) with no extra
boilerplate beyond the model definitions themselves. ${VAR} substitution is
handled explicitly against the environment/.env before YAML parsing (see
`_substitute_env_vars`), so a plain "the fields raise cleanly" story from
pydantic is simpler than adopting an extra dependency (e.g. omegaconf) whose
main selling point (interpolation, merging) is not otherwise needed.

This file starts minimal (scaffolding only, no pipeline logic).
Later sections are added (tiles, DEM, scene_download, features, split,
model, train, ...) following the same pattern documented here rather than
inventing a new one each time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field, field_validator

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

CONFIG_ENV_VAR = "LANDSCAPE_CHANGE_DETECTION_CONFIG"


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed, or invalid."""


class PathsConfig(BaseModel):
    data_root: str
    output_root: str
    model_root: str


class GeeConfig(BaseModel):
    """Google Earth Engine project configuration.

    ``project`` is the single fallback project id (used when a step has no
    per-split/per-worker project pool configured, or as the value
    ``initialize_ee``'s ``config_default`` falls back to). ``projects`` is
    the full pool of GEE project ids available to this pipeline -- e.g. the
    18 Earth Engine Cloud projects the user maintains for quota spreading.
    Any step that can spread work across multiple GEE projects (scene
    download's per-split ``ee_projects``, a future features/DEM step) should
    default to round-robining over this pool rather than each inventing its
    own list, so all 18 projects stay in one place. ``scene_download.ee_projects``
    remains the authoritative *per-split* assignment (index-aligned with
    ``tiles.n_splits``) since download splits need a stable, explicit
    project-to-split mapping; ``gee.projects`` is the source pool a config can
    derive that list from.
    """

    project: str
    projects: list[str] = Field(default_factory=list)


class TilesConfig(BaseModel):
    """AOI tiling parameters (see :mod:`landscape_change_detection_pipeline.tiles.registry`).

    ``aoi_path``/``aoi_layer`` locate the source polygon. ``pilot_bbox`` is an
    optional ``[minx, miny, maxx, maxy]`` sub-window in the AOI's own CRS
    (``aoi_crs``), letting a pilot run cover a small area without touching any
    code. ``tile_size_m``/``buffer_m`` are pipeline parameters, not constants,
    so tile granularity and edge overlap can be tuned per run.

    ``tile_id_prefix`` is prepended to every generated ``tile_id``
    (``"tile_0000_0000"`` -> ``"<prefix>_tile_0000_0000"``). ``tile_id`` is
    just a row/col index into *this* AOI's own grid, so two contributors
    tiling two different AOIs independently produce identical ``tile_id``
    values -- harmless until their training caches
    (``config.features.train_root``) are merged into one shared
    ``data/train_cache/`` tree, at which point one contributor's tile
    silently overwrites the other's. Set this to something unique per
    contributor/project (e.g. your initials or the project name) before
    running ``scripts/01_build_tiles.py``, if you intend to ever share or
    merge training data with someone else -- there is no way to relabel
    ``tile_id`` after the fact without redoing every downstream stage.
    """

    aoi_path: str = "CHANGE_ME.gpkg"
    aoi_layer: str = "output"
    aoi_crs: str = "EPSG:26910"
    tile_size_m: float = 8000.0
    buffer_m: float = 250.0
    pilot_bbox: Optional[list[float]] = None
    # If False, a pilot_bbox outside the AOI polygon is tiled as-is instead
    # of raising -- for pilot zones deliberately outside the tracked AOI.
    pilot_bbox_requires_aoi_overlap: bool = True
    tile_id_prefix: str = ""
    registry_path: str = "data/tiles/tile_registry.parquet"
    n_splits: int = 1
    split_assignment_path: str = "data/tiles/split_assignment.parquet"


class DemConfig(BaseModel):
    """Per-tile DEM acquisition parameters (see :mod:`landscape_change_detection_pipeline.dem`).

    ``tile_dir`` holds each tile's ``<tile_id>.zarr`` store (the same one
    later steps add sensor-scene groups into). ``use_copernicus_fallback``
    lets a run disable the fallback entirely (fail loud on MRDEM gaps rather
    than silently substituting a canopy-biased source); ``force_copernicus``
    is a debug/test switch to exercise the fallback path deliberately.
    ``target_resolution_m`` is the pipeline's single reference grid
    resolution -- the DEM is no longer fetched at its own native 30 m;
    instead every source DEM is reprojected onto this resolution, and every
    later per-tile array (every sensor's scenes, every feature stack, every
    label) must land on this exact grid (see ``dem/grid_check.py`` and
    ``docs/decisions/unified_10m_grid.md``).
    """

    tile_dir: str = "data/tiles"
    use_copernicus_fallback: bool = True
    force_copernicus: bool = False
    target_resolution_m: float = 10.0


class SceneDownloadConfig(BaseModel):
    """Scene search/download parameters (see :mod:`landscape_change_detection_pipeline.scenes`).

    ``date_start_mmdd``/``date_end_mmdd`` default to the full calendar year
    (whole-year coverage, not a fixed ablation season). ``max_cloud_pct`` is
    checked against each sensor's own cloud-cover property (``CLOUD_COVER``
    for Landsat, ``CLOUDY_PIXEL_PERCENTAGE`` for Sentinel-2; see
    :mod:`landscape_change_detection_pipeline.scenes.sensors`). ``max_scenes_per_tile_month``
    caps how many scenes per sensor are kept per tile per calendar month
    (least-cloudy first), to bound download volume while still keeping every
    month represented across the year; ``null`` means unlimited.
    ``min_aoi_coverage_pct``
    rejects scenes whose footprint barely clips the tile's search bbox (e.g.
    a swath edge). ``min_plausible_reflectance`` is the
    ``reduceRegion(minMax())`` sanity-check threshold below which a scene is
    assumed to be edge-of-swath fill rather than real surface reflectance
    (see :mod:`landscape_change_detection_pipeline.scenes.download`). ``cache_dir`` holds
    the per-split SQLite-WAL resume caches (see
    :mod:`landscape_change_detection_pipeline.scenes.cache`).

    ``ee_projects`` lists one GEE project id per split (index-aligned with
    ``tiles.n_splits``), each used by its own subprocess for real quota
    spreading (see
    :mod:`landscape_change_detection_pipeline.scenes.orchestration`); ``tile_workers``
    caps how many tiles are processed concurrently *within* one split, via
    real ``multiprocessing.Process`` workers, never threads; ``tile_work_timeout_s`` is
    the per-tile wall-clock ceiling before a stalled worker is killed and
    reassigned.

    ``run_all_splits`` drives :data:`scripts/03_download_scenes.py`'s default
    behaviour with no ``--split``/``--parallel`` CLI flags: when true, every
    configured split (``tiles.n_splits``) is launched at once, one subprocess
    each against its own ``ee_projects``/``gee.projects`` entry, matching
    ``--split all --parallel``. An explicit ``--split``/``--parallel`` CLI
    flag still overrides this.
    """

    date_start_mmdd: str = "01-01"
    date_end_mmdd: str = "12-31"
    max_cloud_pct: float = 20.0
    max_scenes_per_tile_month: Optional[int] = None
    min_aoi_coverage_pct: float = 50.0
    min_plausible_reflectance: float = 0.15
    cache_dir: str = "data/scenes/cache"
    ee_projects: list[str] = Field(default_factory=list)
    tile_workers: int = 1
    tile_work_timeout_s: int = 1800
    run_all_splits: bool = False


class TopographicCorrectionConfig(BaseModel):
    """Illumination-normalization parameters (see
    :mod:`landscape_change_detection_pipeline.features.topographic_correction`).

    Corrects the reflectance difference between a sun-facing and a shaded
    slope of the same land cover, using each tile's own slope/aspect (DEM
    stage) plus each scene's solar illumination geometry at acquisition
    time. ``method`` selects the correction formula; ``"scs_c"``
    (Sun-Canopy-Sensor + C correction, Soenen et al. 2005) is the default --
    a standard, moderate-terrain-safe choice in the current topographic-
    correction literature that avoids over-correction on near-flat pixels
    (unlike plain Cosine correction) without SCS's own tendency to
    over-correct steep slopes. ``min_sun_elevation_deg`` skips correction
    entirely for a scene with implausibly low sun angle (near sunrise/sunset
    geometry makes the correction numerically unstable, dividing by a
    near-zero cosine term).
    """

    enabled: bool = True
    method: str = "scs_c"
    min_sun_elevation_deg: float = 5.0
    #: Band label the scene-wide ``C`` parameter is fit against, then reused
    #: for every band (see ``features/topographic_correction.py``'s module
    #: docstring for why one shared fit replaces an independent per-band
    #: fit -- fixes a color-balance drift confirmed to appear in the RGB
    #: composites when each band was corrected independently).
    reference_band: str = "nir"
    #: Bounds on the per-pixel correction ratio -- see
    #: ``features/topographic_correction.py::DEFAULT_RATIO_CLIP_MIN``/
    #: ``DEFAULT_RATIO_CLIP_MAX`` for why (bounds an outlier pixel's
    #: correction regardless of how well-behaved the scene-wide fit is).
    ratio_clip_min: float = 0.2
    ratio_clip_max: float = 5.0

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        if value not in ("scs_c",):
            raise ValueError(f"topographic_correction.method must be 'scs_c', got {value!r}")
        return value


class BandpassCoefficient(BaseModel):
    """One band's linear bandpass-adjustment coefficient (see
    :mod:`landscape_change_detection_pipeline.scenes.harmonize`):
    ``y = scale * x + offset``."""

    scale: float = 1.0
    offset: float = 0.0


#: Claverie et al. 2018 / HLS v1.4's published MSI->OLI coefficients
#: (S2A), algebraically inverted to OLI->MSI (Landsat onto Sentinel-2's
#: convention, this pipeline's harmonization direction -- see
#: scenes/harmonize.py's module docstring). Same table for all four Landsat
#: sensors (TM/ETM+/OLI share one equivalent-band mapping in the published
#: coefficients).
_LANDSAT_TO_S2_BANDPASS_COEFFICIENTS: dict[str, BandpassCoefficient] = {
    "blue": BandpassCoefficient(scale=1.022704, offset=0.004091),
    "green": BandpassCoefficient(scale=0.994728, offset=0.000895),
    "red": BandpassCoefficient(scale=1.024066, offset=-0.000922),
    "nir": BandpassCoefficient(scale=1.001703, offset=0.000100),
    "swir1": BandpassCoefficient(scale=1.001302, offset=0.001101),
    "swir2": BandpassCoefficient(scale=0.997009, offset=0.001196),
}


class HarmonizationConfig(BaseModel):
    """Cross-sensor radiometric bandpass adjustment (see
    :mod:`landscape_change_detection_pipeline.scenes.harmonize`).

    Brings Landsat's stored bands onto Sentinel-2's radiometric convention
    via a per-band linear transform (``y = scale*x + offset``), the
    bandpass-adjustment piece of NASA's HLS method (Claverie et al. 2018) --
    BRDF normalization (HLS's other major component) is out of scope, see
    ``docs/decisions/cross_sensor_harmonization.md``.
    ``coefficients`` is ``{sensor_key: {band_label: BandpassCoefficient}}``
    (band *labels* -- blue/green/red/nir/swir1/swir2 -- not raw EE band
    names, so one table works across L5/L7/L8/L9's differing band-name
    layouts); Sentinel-2 is the harmonization target and is never itself
    adjusted. Applied at scene-storage time (``scenes/process_scene.py``),
    after reflectance scale/offset conversion and before topographic
    correction -- see the decision doc for why storage-time was chosen over
    training-cache-export-time.

    ``enabled`` defaults to ``False``: the published coefficients were fit
    on Surface Reflectance (SR-to-SR spectral response differences), and
    this pipeline runs on TOA (``docs/decisions/toa_rollback.md``) --
    applying an SR-derived linear correction to TOA reflectance is not
    radiometrically justified (TOA still carries the atmospheric path-
    radiance contribution the SR fit implicitly assumes is already
    removed). Left here as an explicit, documented opt-in rather than
    removed outright, in case TOA-appropriate coefficients are derived
    later -- see ``docs/decisions/cross_sensor_harmonization.md`` for the
    full reasoning. Out of scope for the current TOA-rollback work.
    """

    enabled: bool = False
    coefficients: dict[str, dict[str, BandpassCoefficient]] = Field(
        default_factory=lambda: {
            sensor_key: dict(_LANDSAT_TO_S2_BANDPASS_COEFFICIENTS)
            for sensor_key in ("L5", "L7", "L8", "L9")
        }
    )


class RgbCompositesConfig(BaseModel):
    """Which RGB visualization composites to build per scene (see
    :mod:`landscape_change_detection_pipeline.scenes.composites`), and their
    shared stretch tunables.

    Four selectable views, each independently toggleable so a view nobody
    has validated yet does not cost compute/storage by default:

    - ``true_color_enabled`` (``rgb_true_color``, was ``rgb_raw``):
      percentile-stretched real red/green/blue.
    - ``true_color_shadow_enabled`` (``rgb_true_color_shadow``, was
      ``rgb_shadow``): asinh-compressed + gamma-boosted real red/green/blue,
      recovering shadow detail on this mountainous AOI's valley shadow.
    - ``natural_color_enabled`` (``rgb_natural_color``, new): SWIR2/NIR/red
      as R/G/B, percentile stretched -- a moisture/burn-scar-sensitive
      interpretation composite. Off by default until visually validated.
    - ``color_infrared_enabled`` (``rgb_color_infrared``, new): NIR/red/green
      as R/G/B, percentile stretched -- the standard vegetation-vigor false-
      color composite. Off by default until visually validated.

    ``asinh_k``/``gamma`` tune ``true_color_shadow``'s asinh-compression
    steepness and gamma boost (see
    ``scenes/composites.py::asinh_shadow_stretch``) -- exposed here rather
    than hardcoded so they can be tuned by eye without a code change.
    """

    true_color_enabled: bool = True
    true_color_shadow_enabled: bool = True
    natural_color_enabled: bool = False
    color_infrared_enabled: bool = False
    asinh_k: float = 8.0
    gamma: float = 1.0 / 2.2


class FeaturesConfig(BaseModel):
    """Spectral-index computation and annotation-to-training-cache export
    (see :mod:`landscape_change_detection_pipeline.features`).

    ``train_root`` holds one ``.npz``/``.json`` cache pair per annotated
    scene. ``mask_root`` is where MaskForge writes annotated masks
    back to (``SaveConfig.output_root`` on the MaskForge side), keyed by its
    ``{tile_id}_{sensor}_{scene_id}/mask.tif`` folder convention.

    ``index_names``/``dem_layer_names`` select which of the available
    spectral indices (see
    :data:`landscape_change_detection_pipeline.features.spectral_indices.INDEX_NAMES`)
    and DEM layers (:data:`...spectral_indices.DEM_LAYER_NAMES`) go into the
    feature stack, in the given order -- ``null`` (the default) means "every
    available name", so a config only needs to list a subset here to disable
    the rest (e.g. ``dem_layer_names: []`` to train on indices only).

    ``geotiff_export_root`` is where
    ``scripts/export_georeferenced_cache.py`` writes standalone georeferenced
    GeoTIFFs (``features.tif`` + ``labels.tif`` per scene) reconstructed from
    ``train_root`` -- a folder you can hand to someone else (or open in QGIS
    yourself) without them needing anything from this pipeline.

    ``include_doy_features``/``include_latlon_features`` add cyclical
    day-of-year (``doy_sin``/``doy_cos``) and normalized scene-centroid
    latitude/longitude (``lat_norm``/``lon_norm``) channels to the feature
    stack (see
    :func:`landscape_change_detection_pipeline.features.training_cache.build_feature_stack`).
    Both default to ``True``: land-cover classes look spectrally different
    by season (e.g. bare ground vs. snow-covered ground) and by region, and
    giving the model that context directly as input channels lets one
    taxonomy (``configs/classes.yaml``) and one model stay season/region-
    agnostic instead of needing separate seasonal classes or separate
    regional models. Changing either flag changes the model's input channel
    count -- re-export the training cache (``scripts/04_export_training_cache.py``)
    and retrain (``scripts/05_train_model.py``) after changing it.
    """

    train_root: str = "data/train_cache"
    mask_root: str = "data/masks"
    geotiff_export_root: str = "data/train_geotiff"
    pseudo_geotiff_export_root: str = "data/pseudo_label_geotiff"
    index_names: Optional[list[str]] = None
    dem_layer_names: Optional[list[str]] = None
    include_doy_features: bool = True
    include_latlon_features: bool = True

    @field_validator("index_names")
    @classmethod
    def _validate_index_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        from landscape_change_detection_pipeline.features.spectral_indices import MODEL_INDEX_NAMES as INDEX_NAMES

        unknown = [name for name in value if name not in INDEX_NAMES]
        if unknown:
            raise ValueError(
                f"features.index_names contains unknown name(s) {unknown}; "
                f"expected a subset of {list(INDEX_NAMES)}."
            )
        return value

    @field_validator("dem_layer_names")
    @classmethod
    def _validate_dem_layer_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        from landscape_change_detection_pipeline.features.spectral_indices import DEM_LAYER_NAMES

        unknown = [name for name in value if name not in DEM_LAYER_NAMES]
        if unknown:
            raise ValueError(
                f"features.dem_layer_names contains unknown name(s) {unknown}; "
                f"expected a subset of {list(DEM_LAYER_NAMES)}."
            )
        return value


def resolved_feature_names(features: FeaturesConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(index_names, dem_layer_names)`` a run should actually use: the
    config's own lists if set, else every available name (see
    ``FeaturesConfig.index_names``/``dem_layer_names``).

    Does **not** include the DOY/lat-lon channel names -- those are
    conditionally appended by
    :func:`~landscape_change_detection_pipeline.features.training_cache.build_feature_stack`
    itself based on ``features.include_doy_features``/``include_latlon_features``.
    Use :func:`resolved_extra_feature_names` for those, or
    :func:`all_resolved_feature_names` for the full, model-input-channel-count
    list in the exact order ``build_feature_stack`` assembles them.
    """
    from landscape_change_detection_pipeline.features.spectral_indices import DEM_LAYER_NAMES, INDEX_NAMES

    index_names = tuple(features.index_names) if features.index_names is not None else INDEX_NAMES
    dem_layer_names = tuple(features.dem_layer_names) if features.dem_layer_names is not None else DEM_LAYER_NAMES
    return index_names, dem_layer_names


def resolved_extra_feature_names(features: FeaturesConfig) -> tuple[str, ...]:
    """DOY/lat-lon channel names actually enabled by ``features``, in the
    same order :func:`~landscape_change_detection_pipeline.features.training_cache.build_feature_stack`
    appends them (DOY first, then lat/lon)."""
    from landscape_change_detection_pipeline.features.spectral_indices import (
        DOY_FEATURE_NAMES,
        LATLON_FEATURE_NAMES,
    )

    names: tuple[str, ...] = ()
    if features.include_doy_features:
        names += DOY_FEATURE_NAMES
    if features.include_latlon_features:
        names += LATLON_FEATURE_NAMES
    return names


def all_resolved_feature_names(features: FeaturesConfig) -> tuple[str, ...]:
    """Every feature channel name a run's feature stack will actually have,
    in ``build_feature_stack``'s own order: spectral indices, DEM layers,
    then (if enabled) DOY, then lat/lon. This is the list whose length is
    the model's real input channel count."""
    from landscape_change_detection_pipeline.features.spectral_indices import expand_dem_feature_names

    index_names, dem_layer_names = resolved_feature_names(features)
    return (*index_names, *expand_dem_feature_names(dem_layer_names), *resolved_extra_feature_names(features))


class SplitConfig(BaseModel):
    """Train/val/test split parameters (see
    :mod:`landscape_change_detection_pipeline.training.dataset`).

    ``ratios`` is ``[train, val, test]`` and must sum to 1.0. ``split_seed``
    makes the stratified shuffle reproducible. ``drift_tolerance`` is the
    fraction-of-target threshold beyond which a partition's actual
    scene-count ratio triggers a (non-fatal) drift warning after coverage
    repair. ``split_assignment_path`` records the resulting
    ``{tile_id: partition}`` assignment (unused when ``split_by="scene"``,
    since there is then no tile-level assignment to record).

    ``split_by`` selects the leakage-preventing unit: ``"tile"`` (default)
    keeps every scene from one tile in the same partition -- the safer
    choice once there is enough annotated data that tile-level stratified
    coverage repair can actually find a valid split. ``"scene"`` splits at
    the individual scene level instead, ignoring which tile a scene came
    from -- appropriate only while the annotated corpus is still small
    enough that tile-level splitting can't guarantee every class appears in
    every partition (few tiles means few whole units to redistribute, so
    the class-coverage repair passes can run out of tiles to move). Scene-
    level splitting accepts a real risk of leakage (two scenes of the same
    tile, correlated by footprint/season, can land in different
    partitions) in exchange for finer-grained control over per-partition
    class representation. Applies uniformly to every model type (torch and
    non-torch alike), since both paths through ``scripts/05_train_model.py``
    call the same :func:`~landscape_change_detection_pipeline.training.dataset.split_scenes`.
    """

    ratios: list[float] = Field(default_factory=lambda: [0.7, 0.15, 0.15])
    split_seed: int = 0
    drift_tolerance: float = 0.25
    split_assignment_path: str = "data/train_cache/split_assignment.json"
    split_by: str = "tile"

    @field_validator("split_by")
    @classmethod
    def _validate_split_by(cls, value: str) -> str:
        if value not in ("tile", "scene"):
            raise ValueError(f"split.split_by must be 'tile' or 'scene', got {value!r}")
        return value


#: The eight selectable model "technologies" (see
#: :mod:`landscape_change_detection_pipeline.models.registry`). ``grid_search`` is folded
#: into ``threshold`` (both are handled by the Optuna-tuned spectral-index
#: thresholder; a plain grid search is just a non-Bayesian ``sampler``
#: choice on the same search), so only seven distinct dispatch targets exist.
#: ``lightgbm`` is a separate, additive option alongside ``random_forest`` --
#: not a replacement -- added specifically to fix ``random_forest``'s RAM
#: blowup on this project's real training corpus (see
#: ``models/lightgbm_model.py``'s module docstring).
MODEL_TYPES: tuple[str, ...] = (
    "threshold",
    "random_forest",
    "catboost",
    "lightgbm",
    "unet",
    "deeplabv3plus",
    "segformer",
)


class ThresholdConfig(BaseModel):
    """Optuna search settings for the spectral-index threshold model (see
    :mod:`landscape_change_detection_pipeline.models.threshold`).

    ``n_trials`` bounds the Optuna study; ``sampler`` selects the search
    strategy (``"tpe"`` for Bayesian search, ``"grid"`` for an exhaustive
    grid search -- this is how ``model.type == "grid_search"`` from the
    project brief is expressed, as a sampler choice rather than a separate
    model type). ``timeout_s`` is an optional wall-clock cap on the whole
    study, independent of ``n_trials``. ``study_storage`` is an optional
    Optuna storage URL (e.g. a SQLite file) so a study can be resumed or
    inspected after the run; ``null`` keeps the study in-memory only.
    """

    n_trials: int = 200
    sampler: str = "tpe"
    timeout_s: Optional[int] = None
    study_storage: Optional[str] = None


class RandomForestConfig(BaseModel):
    """Hyperparameters for the random-forest classifier (see
    :mod:`landscape_change_detection_pipeline.models.random_forest`, a thin wrapper
    around ``sklearn.ensemble.RandomForestClassifier``).

    ``class_weight="balanced"`` reweights classes inversely to their
    frequency, which matters here since land-cover classes are far from
    evenly represented. ``n_jobs=-1`` uses all available cores.
    """

    n_estimators: int = 300
    max_depth: Optional[int] = None
    n_jobs: int = -1
    class_weight: Optional[str] = "balanced"
    random_state: int = 0


class CatBoostConfig(BaseModel):
    """Hyperparameters for the CatBoost classifier (see
    :mod:`landscape_change_detection_pipeline.models.catboost_model`).

    ``task_type="GPU"`` matches this project's CPU-only dev environment
    being a secondary target; runs on machines without a GPU should
    override this to ``"CPU"`` in their own config. ``early_stopping_rounds``
    stops training once the validation metric stops improving for that many
    rounds, so ``iterations`` is an upper bound rather than a fixed cost.
    """

    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 8
    task_type: str = "GPU"
    random_state: int = 0
    early_stopping_rounds: int = 50

    #: Inverse-frequency class weights (see
    #: :func:`landscape_change_detection_pipeline.training.losses.compute_class_weights`,
    #: the same formula ``random_forest``'s ``class_weight="balanced"`` and
    #: the torch models' weighted cross-entropy already use), passed as
    #: CatBoost's own ``class_weights=`` at fit time -- without it, the
    #: model has no reason to spend capacity on rare classes, which show up
    #: as the near-zero IoU per-class entries. Default ``True`` since a
    #: land-cover class distribution this skewed makes an unweighted fit
    #: the wrong default, not a deliberate choice.
    class_weighting: bool = True

    #: Optuna search over (learning_rate, depth, l2_leaf_reg) -- see
    #: :func:`landscape_change_detection_pipeline.models.catboost_model.search_catboost`.
    #: ``n_trials<=0`` skips the search and fits once at this section's own
    #: fixed hyperparameters above (mirrors ``ThresholdConfig``/``HpoConfig``).
    n_trials: int = 0
    sampler: str = "tpe"
    #: ``"hyperband"`` prunes a trial early from its intermediate validation
    #: score reported every ``metric_period`` boosting rounds (see
    #: :func:`landscape_change_detection_pipeline.models.catboost_model.search_catboost`),
    #: using ``search_iterations`` as Hyperband's resource axis. Mirrors
    #: ``HpoConfig.pruner``. Ignored (with a logged warning) when
    #: ``task_type="GPU"`` -- CatBoost supports neither pruning callbacks nor
    #: resumable fitting on GPU, so every trial always runs to completion.
    pruner: str = "none"
    timeout_s: Optional[int] = None
    study_storage: Optional[str] = None

    #: Iteration budget used only during the Optuna search itself -- kept
    #: low (versus the final refit's ``iterations``) so exploring many
    #: candidate (learning_rate, depth, l2_leaf_reg) combinations stays
    #: cheap; the winning trial is then refit once at ``iterations`` (the
    #: real, full budget) before being returned. Separate from
    #: ``iterations`` so "try lots of cheap trials, then one real training
    #: run on the best params" is one knob apart, not one shared value.
    search_iterations: int = 500
    search_early_stopping_rounds: int = 30

    @field_validator("pruner")
    @classmethod
    def _validate_pruner(cls, value: str) -> str:
        if value not in ("none", "median", "hyperband"):
            raise ValueError(f"catboost.pruner must be 'none', 'median', or 'hyperband', got {value!r}")
        return value


class LightGBMConfig(BaseModel):
    """Hyperparameters for the LightGBM classifier (see
    :mod:`landscape_change_detection_pipeline.models.lightgbm_model`).

    Added specifically to fix ``random_forest``'s RAM blowup on this
    project's real training corpus (confirmed live: sklearn's
    ``RandomForestClassifier`` saturating 32 GB, since it has no incremental
    fit and must hold the full dense pixel matrix -- plus every bootstrap
    tree's own bookkeeping -- resident at once). LightGBM's histogram-binned
    ``Dataset`` avoids that; see the model module's own docstring for the
    full memory-management reasoning.

    ``device="cpu"`` is the default, not ``"gpu"``/``"cuda"``: confirmed
    live on this project's own machine (RTX 5070 Laptop GPU) that the
    standard ``pip install lightgbm`` wheel has neither GPU nor CUDA tree
    learners enabled (``LightGBMError: ... Tree Learner was not enabled in
    this build``) -- unlike CatBoost, which only defaults to GPU because
    that was already confirmed working here. Switching to ``"gpu"``/
    ``"cuda"`` requires a from-source LightGBM build with the matching CMake
    flag, not currently set up for this project; only change this default
    after re-confirming a GPU build actually works live, not from this
    comment alone. ``early_stopping_rounds`` only takes effect when a
    validation set is available (mirrors ``CatBoostConfig``'s own field).
    """

    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 1000
    max_depth: int = -1
    min_data_in_leaf: int = 20
    device: str = "cpu"
    random_state: int = 0
    early_stopping_rounds: int = 50

    #: Inverse-frequency class weights (see
    #: :func:`landscape_change_detection_pipeline.training.losses.compute_class_weights`,
    #: the same formula ``random_forest``'s ``class_weight="balanced"`` and
    #: the torch models' weighted cross-entropy already use), applied as a
    #: per-pixel sample weight at fit time -- LightGBM's native API has no
    #: CatBoost-style ``class_weights=`` shortcut, so each row's weight is
    #: its label's class weight. Default ``True``, matching every other
    #: model type in this project: an unweighted fit on this skewed a
    #: class distribution is the wrong default, not a deliberate choice.
    class_weighting: bool = True

    #: Optuna search over (learning_rate, num_leaves, min_data_in_leaf) --
    #: see :func:`landscape_change_detection_pipeline.models.lightgbm_model.search_lightgbm`.
    #: ``n_trials<=0`` skips the search and fits once at this section's own
    #: fixed hyperparameters above (mirrors ``ThresholdConfig``/``HpoConfig``).
    n_trials: int = 0
    sampler: str = "tpe"
    #: ``"hyperband"`` prunes a trial early from its intermediate validation
    #: score reported every boosting round (see
    #: :func:`landscape_change_detection_pipeline.models.lightgbm_model.search_lightgbm`),
    #: using ``search_n_estimators`` as Hyperband's resource axis. Mirrors
    #: ``HpoConfig.pruner``.
    pruner: str = "none"
    timeout_s: Optional[int] = None
    study_storage: Optional[str] = None

    #: Estimator budget used only during the Optuna search itself -- kept
    #: low (versus the final refit's ``n_estimators``) so exploring many
    #: candidate (learning_rate, num_leaves, min_data_in_leaf) combinations
    #: stays cheap; the winning trial is then refit once at ``n_estimators``
    #: (the real, full budget) before being returned.
    search_n_estimators: int = 500
    search_early_stopping_rounds: int = 30

    @field_validator("pruner")
    @classmethod
    def _validate_pruner(cls, value: str) -> str:
        if value not in ("none", "median", "hyperband"):
            raise ValueError(f"lightgbm.pruner must be 'none', 'median', or 'hyperband', got {value!r}")
        return value


class UnetConfig(BaseModel):
    """Architecture options for the U-Net segmentation model (see
    :mod:`landscape_change_detection_pipeline.models.unet`, mirroring ``build_unet``'s
    keyword arguments and defaults exactly).

    ``base_ch`` is the first encoder stage's channel width (doubling at each
    subsequent stage); ``depth`` is the number of levels (default 4 = three
    poolings; ``training.patch_size`` / ``inference.patch_size`` must be divisible
    by ``2 ** (depth - 1)``). ``film_sensor`` / ``film_doy`` /
    ``film_latlon`` switch on FiLM conditioning of the encoder and decoder
    stages, each independently, on the scene's sensor (learned embedding of
    L5/L7/L8/L9/S2), its acquisition day-of-year (sin/cos) and its centroid
    position -- all derived from the scene alone (see
    ``features/scene_context.py``). All off = a plain U-Net.
    ``num_sensors`` is only the size of the optional per-sensor normalisation
    bank (leave at 1; FiLM does not need it). ``norm_type="group"`` is incompatible with
    ``num_sensors > 1`` (see ``UNet``'s own validation). ``bottleneck_attention``
    inserts self-attention over the bottleneck grid; ``deep_supervision``
    attaches auxiliary decoder-level logit heads used only during training.
    """

    base_ch: int = 48
    dropout_p: float = 0.0
    depth: int = 4
    num_sensors: int = 1
    film_sensor: bool = False
    film_doy: bool = False
    film_latlon: bool = False
    norm_type: str = "batch"
    bottleneck_attention: bool = False
    bottleneck_attention_heads: int = 8
    deep_supervision: bool = False


class Deeplabv3PlusConfig(BaseModel):
    """Architecture options for the DeepLabv3+ segmentation model (see
    :mod:`landscape_change_detection_pipeline.models.deeplabv3plus`).

    ``backbone`` selects the encoder (e.g. a ResNet variant);
    ``pretrained`` loads ImageNet-pretrained backbone weights before
    replacing the input stem for this project's non-RGB channel count.
    """

    backbone: str = "resnet50"
    pretrained: bool = True
    dropout_p: float = 0.1


class SegformerConfig(BaseModel):
    """Architecture options for the SegFormer segmentation model (see
    :mod:`landscape_change_detection_pipeline.models.segformer`).

    ``variant`` selects the MiT encoder size (e.g. ``"mit-b0"`` through
    ``"mit-b5"``); ``pretrained`` loads pretrained encoder weights before
    replacing the input stem for this project's non-RGB channel count.
    """

    variant: str = "mit-b0"
    pretrained: bool = True
    dropout_p: float = 0.1


class ModelConfig(BaseModel):
    """Land-cover model selection (see
    :mod:`landscape_change_detection_pipeline.models.registry`).

    ``type`` is the single switch that selects which of the eight supported
    approaches (spectral-index thresholds tuned by Optuna, folding in a
    plain grid search as a ``threshold.sampler`` choice; random forest;
    CatBoost; LightGBM; U-Net; DeepLabv3+; SegFormer) inference and training
    actually use; :func:`landscape_change_detection_pipeline.models.registry.build_model`
    dispatches on it. Every other field here is a nested, always-present
    sub-section holding that one model type's own hyperparameters --
    unused sub-sections are simply ignored rather than omitted, so a config
    can keep tuned settings for more than one model type around at once
    while only one is active.
    """

    type: str
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)
    lightgbm: LightGBMConfig = Field(default_factory=LightGBMConfig)
    unet: UnetConfig = Field(default_factory=UnetConfig)
    deeplabv3plus: Deeplabv3PlusConfig = Field(default_factory=Deeplabv3PlusConfig)
    segformer: SegformerConfig = Field(default_factory=SegformerConfig)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value not in MODEL_TYPES:
            raise ValueError(
                f"model.type={value!r} is not a supported model technology; "
                f"expected one of {MODEL_TYPES!r}."
            )
        return value


class ConfusionPenaltyConfig(BaseModel):
    """One directional confusion penalty (see
    :class:`landscape_change_detection_pipeline.training.losses.ConfusionPenalty`).

    ``true_class=None`` means "any class other than predicted_class" (see
    :func:`~landscape_change_detection_pipeline.training.losses.directional_penalty`).
    """

    true_class: Optional[int] = None
    predicted_class: int
    beta: float = 1.0


class RadiometricAugmentationConfig(BaseModel):
    """Radiometric jitter applied to the spectral-index channels of a
    training patch, alongside the existing geometric augmentation (see
    :class:`landscape_change_detection_pipeline.training.train.PatchDataset`).

    Only the leading ``len(features.index_names)`` channels of the feature
    stack (spectral indices / chromaticity ratios -- see
    ``features/spectral_indices.py::INDEX_NAMES``) are perturbed; DEM,
    day-of-year, and lat/lon channels are left untouched since brightness/
    contrast/noise have no physical meaning for elevation, slope, aspect, or
    the acquisition-date/location encodings. Applied only to the training
    split -- never validation/test.

    ``brightness_std``/``contrast_std`` are the standard deviation of a
    per-patch multiplicative jitter (``contrast``) and additive jitter
    (``brightness``) sampled once per patch and applied uniformly across its
    spectral channels: ``x' = (x - mean) * (1 + contrast) + mean +
    brightness``, using each channel's own train-corpus mean so contrast
    scales around the channel's real center rather than zero.
    ``noise_std`` is the standard deviation of i.i.d. Gaussian noise added
    per pixel per channel, in the same normalized (post train-mean/std)
    units the model actually trains on.
    """

    enabled: bool = False
    brightness_std: float = 0.05
    contrast_std: float = 0.05
    noise_std: float = 0.02


class SceneOversamplingConfig(BaseModel):
    """Class-driven per-scene oversampling for :class:`PatchDataset`'s random
    scene draw (see
    :class:`landscape_change_detection_pipeline.training.train.PatchDataset`).

    Unlike per-pixel class weighting (``training.class_weighting``) or a
    hard-rebalanced sampler, this only changes *how often each scene is
    drawn* -- the pixels within a drawn patch keep their natural class mix,
    so a rare class gets more exposure through more varied crops of the
    scenes that contain it, without inflating its per-pixel loss weight or
    swamping the batch with synthetic repeats. Meant to be gentler than
    per-pixel rebalancing when that has destabilized training (see the
    :class:`TrainingConfig` docstring).

    A scene's sampling weight is
    ``1 + boost_factor * (rare-class pixel share in that scene)``, where the
    "rare-class pixel share" is the fraction of the scene's labeled pixels
    belonging to a class whose corpus-wide frequency is below
    ``rare_class_threshold``. ``boost_factor=0`` (or ``enabled=False``)
    reproduces the previous uniform-over-scenes behaviour exactly.
    """

    enabled: bool = False
    rare_class_threshold: float = 0.02
    boost_factor: float = 5.0
    # Explicit dense class ids (position in classes.yaml, NOT the raw `id`) to
    # treat as rare; when non-empty it replaces the rare_class_threshold rule.
    rare_classes: list[int] = Field(default_factory=list)
    # A scene's rare share is otherwise its rare-pixel fraction, which is tiny
    # even for a scene that does contain the class. With a value > 0, a scene
    # holding >= this many rare pixels counts as fully rare (share 1.0), and
    # fewer pixels count proportionally.
    presence_min_pixels: int = 0


class PseudoLabelConfig(BaseModel):
    """Semi-supervised self-training over the unlabeled scene pool under
    ``dem.tile_dir`` (see
    :mod:`landscape_change_detection_pipeline.training.pseudo_label`).

    Only scenes with no corresponding annotated mask are eligible (the
    annotated ones are already used directly). Because the unlabeled pool is
    far larger than what fits in RAM or the caches already built (see the
    module docstring), pseudo-labeling is a **separate, streaming** step run
    with ``scripts/05b_generate_pseudo_labels.py``, not something the main
    training loop does inline: it loads one unlabeled scene at a time,
    predicts with an already-trained checkpoint, keeps only
    confidence-thresholded pixels, and writes the result straight to disk in
    the same per-scene ``.npz`` cache schema
    :mod:`landscape_change_detection_pipeline.features.training_cache` uses for
    real annotations -- never holding more than one scene's feature stack in
    memory at a time, and never touching VRAM beyond one inference batch.

    The resulting cache tree (``pseudo_label_root``) is then mixed into
    training as a second, separately-weighted :class:`PatchDataset` --
    labeled scenes keep full weight, pseudo-labeled scenes are drawn at
    ``pseudo_label_weight`` of a real scene's sampling probability, and
    ``pseudo_label_loss_weight`` scales their contribution to the loss, so a
    wrong pseudo-label costs less than a wrong real one.
    """

    enabled: bool = False
    pseudo_label_root: str = "data/pseudo_label_cache"
    confidence_threshold: float = 0.9
    max_scenes: Optional[int] = None
    sensors: Optional[list[str]] = None
    inference_batch_size: int = 32
    # Sliding-window stride used only for pseudo-labeling. null = patch_size
    # (no overlap, ~4x faster than the usual stride = patch_size / 2).
    inference_stride: Optional[int] = None
    use_amp: bool = True
    # Threads preloading the next scenes' feature stacks while the GPU infers.
    prefetch_workers: int = 3
    # Rare-class handling: pixels predicted as one of ``rare_classes`` are
    # kept at the (lower) ``rare_confidence_threshold`` instead of
    # ``confidence_threshold``, and a scene is kept if it has at least
    # ``min_rare_pixels`` such pixels even when its overall kept ratio is low.
    rare_classes: list[int] = []
    rare_confidence_threshold: float = 0.7
    min_rare_pixels: int = 500
    pseudo_label_weight: float = 0.3
    pseudo_label_loss_weight: float = 0.5


class TrainingConfig(BaseModel):
    """Training-loop hyperparameters (see
    :mod:`landscape_change_detection_pipeline.training.train`).

    AdamW + cosine annealing, mixed precision with fp32 loss computation, early
    stopping on ``main_iou_metric`` (see
    :func:`landscape_change_detection_pipeline.training.metrics.select_main_iou_metric`)
    with ``patience`` epochs of no improvement (rounded to 3 decimals, so an
    epoch must gain >= 0.001 to count). ``augment_flips``/``augment_rotate90``
    default to **on**, since the rotation-invariance argument for leaving
    them off does not hold as strongly for this project's land-cover classes
    as it might elsewhere.
    """

    epochs: int = 100
    batch_size: int = 32
    #: Forward/backward on chunks of this many samples, gradients accumulated
    #: over the whole ``batch_size`` -- same optimisation step, far less VRAM
    #: (activations dominate). null = whole batch at once.
    micro_batch_size: Optional[int] = None
    #: Random training patches drawn per epoch. ``None`` = one per training scene, which is a
    #: tiny epoch (e.g. 123 scenes = ~4 optimiser steps at batch 32) with a full validation pass
    #: after each. Set it to a few dozen steps' worth (e.g. 1024 = 32 steps at batch 32) so that
    #: ``epochs``, ``patience`` and ``hpo.trial_epochs`` count meaningful units.
    samples_per_epoch: Optional[int] = None
    #: DataLoader RAM knobs. Each worker is a full Python process holding ``prefetch_factor``
    #: prefetched batches (a batch of 32 x 256 px x 21 channels is ~180 MB), so training RAM is roughly
    #: ``(num_workers + val_num_workers) x (process ~0.5 GB + prefetch_factor x batch)``.
    #: ``val_num_workers: None`` = same as ``num_workers``.
    prefetch_factor: int = 4
    val_num_workers: Optional[int] = None
    patch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15
    num_workers: int = 16
    seed: int = 0
    deterministic: bool = False
    amp: bool = True
    grad_clip_norm: float = 1.0
    main_iou_metric: str = "miou_macro"
    augment_flips: bool = True
    augment_rotate90: bool = True
    confusion_penalties: list[ConfusionPenaltyConfig] = Field(default_factory=list)
    checkpoint_dir: str = "models/checkpoints"

    #: Inverse-frequency class weights in the training loss (see
    #: :func:`landscape_change_detection_pipeline.training.losses.compute_class_weights`
    #: and ``weighted_cross_entropy``) -- the same rebalancing
    #: ``random_forest``/``catboost``/``lightgbm`` apply via their own
    #: ``class_weighting``/``class_weight`` fields. Default ``True``: this
    #: project's land-cover classes are far from evenly represented, and an
    #: unweighted loss leaves the rarest ones under-learned.
    class_weighting: bool = True

    #: Softens ``class_weighting``: weight = (inverse frequency) ** power.
    #: 1.0 = plain inverse frequency (can reach hundreds for a very rare class
    #: and destabilise training); 0.5 = square root (recommended); 0 = none.
    class_weight_power: float = 0.5

    #: Hard cap on any single class weight (before renormalisation); null = no cap.
    class_weight_max: Optional[float] = 10.0

    #: Focal-loss exponent applied to the cross-entropy: 0 = off, 2 = usual.
    #: Down-weights easy pixels so hard/rare classes drive the gradient.
    focal_gamma: float = 0.0

    #: Weight of a soft-Dice term added to the loss: 0 = off, 0.5-1.0 typical.
    #: Every class counts equally in Dice, which helps rare classes directly.
    dice_weight: float = 0.0

    #: Radiometric jitter (brightness/contrast/noise) on top of the existing
    #: geometric augmentation -- see :class:`RadiometricAugmentationConfig`.
    radiometric_augmentation: RadiometricAugmentationConfig = Field(
        default_factory=RadiometricAugmentationConfig
    )

    #: Merge classes for training/evaluation: ``{source: target}`` class names
    #: from ``classes.yaml`` (e.g. ``built_up_infrastructure: bare_ground``).
    #: Source pixels are relabelled as the target whenever a cached scene is
    #: read (the caches on disk are untouched), and predictions of a source
    #: class are folded onto the target at evaluation, pseudo-labeling and
    #: inference. The model keeps its full output size; a source class simply
    #: has no support. Remove merged classes from ``scene_oversampling`` /
    #: ``pseudo_label`` ``rare_classes`` (a warning is printed otherwise).
    class_merge: dict[str, str] = Field(default_factory=dict)

    #: Class-driven per-scene oversampling -- see
    #: :class:`SceneOversamplingConfig`.
    scene_oversampling: SceneOversamplingConfig = Field(default_factory=SceneOversamplingConfig)

    #: Semi-supervised pseudo-labeling over the unlabeled scene pool -- see
    #: :class:`PseudoLabelConfig`.
    pseudo_label: PseudoLabelConfig = Field(default_factory=PseudoLabelConfig)


class HpoSearchSpaceEntry(BaseModel):
    """One hyperparameter's Optuna search distribution (see
    :mod:`landscape_change_detection_pipeline.training.hpo`).

    ``type="log_uniform"`` samples in log space (appropriate for a
    multiplicative-scale hyperparameter like ``lr``/``weight_decay``, where a
    linear search would over-sample the high end); ``"uniform"`` samples
    linearly (appropriate for ``dropout_p``, which is already on a bounded
    linear [0, 1) scale). ``bounds`` is ``[low, high]``, required for both.
    """

    type: str = "log_uniform"
    bounds: list[float] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value not in ("log_uniform", "uniform"):
            raise ValueError(f"hpo search-space entry type must be 'log_uniform' or 'uniform', got {value!r}")
        return value


class HpoConfig(BaseModel):
    """Optuna hyperparameter search settings for the torch model types
    (unet/deeplabv3plus/segformer -- see :mod:`landscape_change_detection_pipeline.training.hpo`).

    ``trials<=0`` (the default) skips the search entirely and trains once at
    ``training``'s own configured hyperparameters -- appropriate once those
    values are already tuned, or for a quick smoke test. ``trials>0`` runs
    that many short trials (``trial_epochs`` each, cut short of the full
    ``training.epochs``) over ``search_space``, then retrains once at full
    length with the winning hyperparameters.

    ``search_space`` intentionally excludes structural hyperparameters
    (``patch_size``, ``stride``, ``batch_size``) -- varying those changes
    what a trial is even measuring (different effective receptive field /
    tile population), making trials incomparable; only continuous
    hyperparameters that don't change the tile population (``lr``,
    ``weight_decay``, ``dropout_p``) are swept.
    """

    trials: int = 0
    trial_epochs: int = 25
    #: Validation cost per trial epoch. Each trial scores the same ``val_patches_per_scene``
    #: evenly spaced patches of every val scene (only used when ``training.samples_per_epoch``
    #: is set; ``None`` = every patch, the slow exact score). The final retrain always
    #: validates on everything.
    val_patches_per_scene: Optional[int] = 4
    #: ``>= 2``: score every trial as the mean best val metric over this many tile-grouped
    #: folds of train+val (far less noisy than one small val split, ``k`` times the cost per
    #: trial, no pruning). ``0`` = the single configured train/val split.
    cv_folds: int = 0
    sampler: str = "tpe"
    pruner: str = "none"
    storage: Optional[str] = None
    search_space: dict[str, HpoSearchSpaceEntry] = Field(
        default_factory=lambda: {
            "lr": HpoSearchSpaceEntry(type="log_uniform", bounds=[1e-5, 1e-2]),
            "weight_decay": HpoSearchSpaceEntry(type="log_uniform", bounds=[1e-6, 1e-2]),
            "dropout_p": HpoSearchSpaceEntry(type="uniform", bounds=[0.0, 0.5]),
        }
    )

    @field_validator("sampler")
    @classmethod
    def _validate_sampler(cls, value: str) -> str:
        if value not in ("tpe", "cmaes", "random"):
            raise ValueError(f"hpo.sampler must be 'tpe', 'cmaes', or 'random', got {value!r}")
        return value

    @field_validator("pruner")
    @classmethod
    def _validate_pruner(cls, value: str) -> str:
        if value not in ("none", "median", "hyperband"):
            raise ValueError(f"hpo.pruner must be 'none', 'median', or 'hyperband', got {value!r}")
        return value


class BootstrapConfig(BaseModel):
    """Multi-seed bootstrap ensembling settings (see
    :mod:`landscape_change_detection_pipeline.training.bootstrap`).

    Trains the same configuration ``n_seeds`` times, varying only ``seed``
    (never ``split_seed`` -- see the module docstring for why holding the
    split fixed across seeds is what makes the spread interpretable as
    initialization variance rather than a mix of that and split variance).
    ``create_ensemble`` additionally writes every seed's weights side by
    side (never averaged -- independently initialized networks are not in
    parameter correspondence) for softmax-averaged ensemble inference.
    """

    enabled: bool = False
    n_seeds: int = 5
    seed_base: int = 0
    create_ensemble: bool = True


class InferenceConfig(BaseModel):
    """Sliding-window inference parameters (see
    :mod:`landscape_change_detection_pipeline.inference.engine`).

    ``patch_size``/``stride`` default to the training patch size with 50%
    overlap, and are config-driven rather than hardcoded.
    ``scene_precheck_min_valid_ratio`` mirrors
    ``scene_passes_precheck`` -- a scene below this finite-pixel fraction is
    marked unprocessable rather than run through the model.

    ``checkpoint_path``/``tiles``/``sensors``/``device``/``workers`` are
    ``scripts/06_run_inference.py``'s own run parameters -- config-driven,
    like every other run parameter in this project, rather than CLI flags
    (see this project's own convention: everything a run needs to be
    reproduced from ``config.yaml`` alone, not from a shell history).
    ``checkpoint_path=null`` keeps that script's own default
    (``<training.checkpoint_dir>/<model.type>_best.<pt|joblib>``);
    ``tiles``/``sensors`` empty means every tile/sensor found.
    ``workers<=1`` runs sequentially in the main process; ``workers>1``
    spreads scenes across that many worker processes (see
    ``06_run_inference.py``'s own module docstring for why this should stay
    small for a GPU-backed model type, e.g. ``catboost`` with
    ``task_type=GPU``, versus a CPU-only one).
    """

    patch_size: int = 256
    stride: int = 128
    batch_size: int = 32
    scene_precheck_min_valid_ratio: float = 0.30
    overwrite: bool = False
    output_root: str = "outputs/inference"
    ambiguity_threshold: Optional[float] = None
    class_priority_order: list[int] = Field(default_factory=list)
    checkpoint_path: Optional[str] = None
    tiles: list[str] = Field(default_factory=list)
    sensors: list[str] = Field(default_factory=list)
    device: Optional[str] = None
    workers: int = 1


class CompositeClassRule(BaseModel):
    """Per-class reduction rule for one month's stack of per-scene class maps
    (see :mod:`landscape_change_detection_pipeline.inference.composites`).

    - ``rule="median"`` for stable classes (forest, water, wetland,
      built-up): the per-pixel majority vote.
    - ``rule="any_occurrence"`` for durable-but-easily-outvoted disturbance
      flags (e.g. a cutblock or burn that must survive and win outright
      even if seen in just one scene that month).
    - ``rule="fallback_occurrence"`` for transient, naturally-fluctuating
      states that should only fill in where no class won a median majority
      (e.g. snow/ice: must not mask a real land-cover class the majority of
      the month's scenes agree on, since that would erase exactly the
      melt/freeze signal a snow/ice trajectory analysis needs to see).

    ``priority`` breaks ties when more than one class under the *same* rule
    would otherwise fire on the same pixel -- lower value wins.
    """

    class_id: int
    rule: str = "median"
    priority: int = 0

    @field_validator("rule")
    @classmethod
    def _validate_rule(cls, value: str) -> str:
        if value not in ("median", "any_occurrence", "fallback_occurrence"):
            raise ValueError(
                f"composite class rule must be 'median', 'any_occurrence', or 'fallback_occurrence', got {value!r}"
            )
        return value


class CompositesConfig(BaseModel):
    """Monthly composite parameters.

    ``sensor_resolution_priority`` is the finest-available-grid priority
    order used to pick each tile-month's composite resolution: the first
    sensor in this list with any scene that tile-month sets the resolution
    every other sensor's classification raster is reprojected onto
    (nearest-neighbour, since these are categorical labels). Default matches
    this project's separate ``l7``/``l8``/``l9`` groups (``s2 > l9 > l8 > l7
    > l5``).
    """

    sensor_resolution_priority: list[str] = Field(
        default_factory=lambda: ["s2", "l9", "l8", "l7", "l5"]
    )
    default_rule: str = "median"
    class_rules: list[CompositeClassRule] = Field(default_factory=list)
    ignore_classes: list[str] = Field(default_factory=lambda: ["cloud", "shadow"])
    """Class names (classes.yaml) that cast no vote in a monthly composite, so a
    month is decided by its clear observations only. ``[]`` lets them vote (a
    cloud-majority pixel then becomes ``cloud`` in the composite)."""
    output_root: str = "outputs/composites"

    #: scripts/07_build_monthly_composites.py's own run parameters --
    #: config-driven rather than CLI flags (this project's convention: a run
    #: must be reproducible from config.yaml alone). tiles=[] means every
    #: tile in the registry. workers>1 builds separate tiles' composites in
    #: parallel worker processes -- tiles are fully independent (each reads
    #: only its own inference output and writes its own output subtree) and
    #: this stage is CPU/numpy-only (no GPU contention to worry about, unlike
    #: scripts/06_run_inference.py's own ``inference.workers``), so this can
    #: reasonably scale close to the machine's core count.
    tiles: list[str] = Field(default_factory=list)
    overwrite: bool = False
    workers: int = 1


class ChangeDetectionConfig(BaseModel):
    """Per-pixel temporal segmentation parameters (see
    :mod:`landscape_change_detection_pipeline.change.segmentation` for the
    method and :mod:`landscape_change_detection_pipeline.change.change_detection`
    for the tile driver).

    ``features`` are the spectral indices each pixel's series is built from
    (names from :data:`change.spectral_composites.ALL_INDEX_NAMES`); indices
    are normalised ratios or fixed linear combinations of the bands, so they
    stay comparable across Landsat 5/7/8/9 and Sentinel-2. ``NBR`` and
    ``NDVI`` are required: later stages read burn severity from the first
    and vegetation state from the second. ``masked_classes`` are class names
    whose observations are not used at all (states that are not the land
    surface: cloud, shadow, snow, ice, open water).

    ``p_change`` is the chi-square probability level of the per-observation
    change test; ``conse`` how many consecutive observations must all exceed
    it to confirm a break -- lower it for sparser data, raise it for denser
    data. ``init_obs``/``init_min_span_days`` size the initial window of each
    segment; ``stability_threshold`` (in residual standard deviations) is how
    much trend/end-residual that window may show; ``min_rmse`` floors each
    feature's residual scale (index units) so a near-perfect fit cannot make
    every observation look anomalous; ``max_harmonics`` (1-3) caps the
    seasonal model; ``outlier_p`` is the probability level above which an
    isolated observation in the initial window is discarded; ``max_segments``
    caps segments per pixel (a pixel that would exceed it is flagged
    ``truncated``).

    ``start_year``/``end_year`` bound the time series (``end_year=null``: to
    the latest scene). ``tiles=[]`` means every tile in the registry.
    ``block_rows`` is the row-block height of the temporary scene cube (memory
    per block scales with it); ``io_threads`` scenes are decoded at once;
    ``numba_threads=0`` uses every core for the per-pixel computation.
    """

    output_root: str = "outputs/change_detection"
    tiles: list[str] = Field(default_factory=list)
    overwrite: bool = False

    start_year: int = 1984
    end_year: Optional[int] = None
    features: list[str] = Field(default_factory=lambda: ["NBR", "NDVI", "NDWI_GAO", "TC_BRIGHTNESS"])
    masked_classes: list[str] = Field(
        default_factory=lambda: ["cloud", "shadow", "snow_cover", "ice_cover", "open_water"]
    )

    p_change: float = 0.999
    conse: int = 4
    init_obs: int = 18
    init_min_span_days: int = 365
    stability_threshold: float = 3.0
    min_rmse: float = 0.03
    max_harmonics: int = 3
    outlier_p: float = 0.999999
    max_segments: int = 16
    use_slope_interaction: bool = False
    """Adds, per non-reference sensor, one extra model column of
    ``terrain_slope * sensor_indicator`` alongside the plain per-sensor
    fixed effect (see change.segmentation's module docstring) -- lets each
    sensor's own radiometric offset scale with this pixel's own DEM slope,
    since cross-sensor TOA residuals are known to be worse on steep terrain
    (see scenes/harmonize.py). Experimental: evaluate against the plain
    per-sensor offset (``use_slope_interaction: false``) before relying on
    it, since it roughly doubles the sensor-related columns fit per pixel."""

    min_magnitude: float = 0.1
    """A statistically confirmed break is kept only if the shift (median of
    the confirming observations minus median of the last pre-break ones)
    reaches this size on at least one of ``magnitude_features``; a smaller
    shift is absorbed into the segment's model. 0 disables the filter."""
    magnitude_features: list[str] = Field(default_factory=lambda: ["NBR", "NDVI"])
    min_break_span_days: int = 45
    """The confirming run of exceedances must also span this many days,
    whatever ``conse`` -- dense Sentinel-2 revisits otherwise confirm a break
    from a couple of weeks of seasonal misfit. 0 disables."""
    refit_every: int = 3
    """Refit a segment's model every this many accepted observations once it
    has 30 (each fit is the costliest step of monitoring). 1 = every one."""
    aggregate_days: int = 15
    """Aggregate observations into bins of this many days (per sensor, per-
    pixel median of the usable ones) before segmenting: faster, less noisy,
    and a run of ``conse`` exceedances then means a sustained change. 0 = use
    every scene as is (slow, and dense Sentinel-2 gives false breaks)."""

    season_window: Optional[list[int]] = Field(default_factory=lambda: [152, 258])
    """``[first_doy, last_doy]``: only observations whose calendar day of year
    falls inside are used (default 152-258, about 1 June - 15 September, the
    growing season). Snowmelt, leaf fall and low sun are what the annual
    harmonics model badly, and they produced most of the seasonal false
    breaks. Break dates are then only as precise as the season. The
    harmonic count is forced to 1 when a window is active. ``null``/``[]`` =
    use the whole year."""
    min_magnitude_same_class: float = 0.2
    """Stricter ``min_magnitude`` for a break after which the land-cover class
    is unchanged (thinning, phenology, sensor drift look like that). 0 = same
    threshold as ``min_magnitude``."""
    persist_days: int = 300
    """A break must still be visible (same sign, at least half of
    ``min_magnitude`` on one of ``magnitude_features``) in the first
    observations at least this many days after it. Breaks too close to the end
    of the series to verify are kept. 0 disables."""
    min_obs_break_year: int = 8
    """A break needs at least this many usable observations within a year of
    its date (sparse years make any offset look like a break). 0 disables."""

    block_rows: int = 50
    io_threads: int = 4
    numba_threads: int = 0

    @field_validator("season_window")
    @classmethod
    def _validate_season_window(cls, value):
        if value:
            if len(value) != 2 or not (1 <= value[0] < value[1] <= 366):
                raise ValueError(f"change_detection.season_window must be [first_doy, last_doy] within 1-366, got {value}")
        return value

    @field_validator("magnitude_features")
    @classmethod
    def _validate_magnitude_features(cls, value: list[str], info) -> list[str]:
        feats = info.data.get("features")
        if feats is not None:
            unknown = [v for v in value if v not in feats]
            if unknown:
                raise ValueError(f"change_detection.magnitude_features {unknown} must be among features {feats}")
        return value

    @field_validator("features")
    @classmethod
    def _validate_features(cls, value: list[str]) -> list[str]:
        for required in ("NBR", "NDVI"):
            if required not in value:
                raise ValueError(f"change_detection.features must include {required}, got {value!r}")
        if len(set(value)) != len(value):
            raise ValueError(f"change_detection.features has duplicates: {value!r}")
        return value

    @field_validator("p_change", "outlier_p")
    @classmethod
    def _validate_probability(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError(f"probability must be strictly between 0 and 1, got {value}")
        return value

    @field_validator("conse")
    @classmethod
    def _validate_conse(cls, value: int) -> int:
        if value < 2:
            raise ValueError(f"change_detection.conse must be >= 2, got {value}")
        return value

    @field_validator("max_harmonics")
    @classmethod
    def _validate_harmonics(cls, value: int) -> int:
        if value not in (1, 2, 3):
            raise ValueError(f"change_detection.max_harmonics must be 1, 2 or 3, got {value}")
        return value

    @field_validator("init_obs")
    @classmethod
    def _validate_init_obs(cls, value: int) -> int:
        if value < 8:
            raise ValueError(f"change_detection.init_obs must be >= 8, got {value}")
        return value


class SnowWaterDynamicsConfig(BaseModel):
    """Per-pixel, per-water-year snow/ice/open-water summary parameters (see
    :mod:`landscape_change_detection_pipeline.change.snow_water_dynamics`).

    Reads Stage 7's monthly class composites directly (``composites.output_root``)
    -- the classifier's own snow_cover/ice_cover/open_water class, never a
    spectral-index reading of the same thing (this is deliberately a
    different, categorical signal from ``change_detection``'s continuous
    vegetation-index trend model, which excludes these classes for the
    opposite reason -- see that module's own docstring).

    ``water_year_start_month`` (1-12, default 10 = October, Water Survey of
    Canada's standard) groups months into water years so one winter is
    never split across two summary rows.

    ``tiles``/``overwrite``/``workers`` are
    ``scripts/10_build_snow_water_dynamics.py``'s own run parameters --
    config-driven rather than CLI flags (this project's convention: a run
    must be reproducible from ``config.yaml`` alone). ``tiles=[]`` means
    every tile in the registry. ``workers>1`` builds separate tiles in
    parallel worker processes -- tiles are fully independent here and this
    stage does no internal multi-core work of its own (unlike
    ``change_detection``'s Numba pass), so it benefits from inter-tile
    parallelism the way ``mosaic`` does.
    """

    output_root: str = "outputs/snow_water_dynamics"
    tiles: list[str] = Field(default_factory=list)
    overwrite: bool = False
    water_year_start_month: int = 10
    workers: int = 1

    @field_validator("water_year_start_month")
    @classmethod
    def _validate_water_year_start_month(cls, value: int) -> int:
        if not 1 <= value <= 12:
            raise ValueError(f"snow_water_dynamics.water_year_start_month must be 1-12, got {value}")
        return value


class RegrowthSeverityConfig(BaseModel):
    """NDVI regrowth / dNBR burn severity parameters (see
    :mod:`landscape_change_detection_pipeline.change.regrowth_severity`).

    Reads each pixel's most recent break date from
    ``change_detection.output_root``'s segment table (Stage 09). No NDSI
    snow filter here by design -- the trained classifier's own
    ``snow_cover`` class already answers that question from the same
    composites this module reads; see :mod:`change.regrowth_severity`'s own
    module docstring for why dNBR and NDVI regrowth earn a separate
    representation but snow does not.

    ``tiles``/``overwrite``/``workers`` are
    ``scripts/11_build_regrowth_severity.py``'s own run parameters --
    config-driven rather than CLI flags (this project's convention: a run
    must be reproducible from ``config.yaml`` alone). ``tiles=[]`` means
    every tile in the registry. ``workers>1`` builds separate tiles in
    parallel worker processes -- tiles are fully independent here and this
    stage does no internal multi-core work of its own (unlike
    ``change_detection``'s Numba pass), so it benefits from inter-tile
    parallelism the way ``mosaic`` does.
    """

    output_root: str = "outputs/regrowth_severity"
    tiles: list[str] = Field(default_factory=list)
    overwrite: bool = False
    workers: int = 1
    month_threads: int = 4
    """Monthly composites (the expensive part: reading each month's scenes and
    reducing them) are computed this many months ahead in threads inside one
    tile. Multiply by ``workers`` for the total; memory is one month's
    composite per thread. 1 = sequential."""


class EventTypingConfig(BaseModel):
    """Naming of change events from the detected surface states (see
    :mod:`landscape_change_detection_pipeline.change.event_typing`).

    Cultivated agriculture, cutblock and built-up are *not* classes of the
    classifier (they are uses, not surface states): they are read from the
    monthly class series of each pixel (Stage 07 composites), only within a
    growing season, and -- for the abrupt cutblock/fire events -- from Stage
    09's breaks. Cloud/shadow/snow/ice months (``masked_classes``) are
    unobserved.

    Cropland: within ``crop_season`` (month range, default May-October) a year
    cycles when bare ground and grassland each show at least
    ``crop_min_months_per_year`` observed months; cultivation must then cycle every
    observed year from its first year to the end of the series, for at least
    ``crop_min_run_years`` years (``crop_allowed_gaps`` non-cycling observed years
    tolerated), with at most ``crop_max_forest_fraction`` forest.

    Cutblock: a break with forest in the ``pre_seasons`` observed seasons
    before (at least ``min_fraction`` forest), then at least
    ``min_cleared_years`` consecutive observed seasons of ``cutblock_season``
    (default May-October) that are not forest (at most ``end_fraction``
    forest) and not burned, then forest again (``min_fraction``) -- the
    duration and recovery year are kept. A season needs ``min_season_obs``
    observed months. ``min_drop`` is the NBR/NDVI fall that makes a break a
    loss; ``burn_min_fraction`` the burned share that makes it a fire;
    ``permanent_years``/``permanent_bare_fraction`` a clearing that never
    recovers for that long and is mostly bare (road/pad/mine/built candidate).
    """

    output_root: str = "outputs/change_events"
    tiles: list[str] = Field(default_factory=list)
    overwrite: bool = False
    forest_class: str = "forest"
    grass_class: str = "grassland_herbaceous"
    bare_class: str = "bare_ground"
    burned_class: str = "burned_disturbed"
    masked_classes: list[str] = Field(default_factory=lambda: ["cloud", "shadow", "snow_cover", "ice_cover"])
    min_drop: float = 0.15
    min_fraction: float = 0.8
    end_fraction: float = 0.2
    burn_min_fraction: float = 0.4
    cutblock_season: list[int] = Field(default_factory=lambda: [5, 10])
    min_season_obs: int = 2
    pre_seasons: int = 2
    min_cleared_years: int = 3
    permanent_years: float = 8.0
    permanent_bare_fraction: float = 0.6
    crop_season: list[int] = Field(default_factory=lambda: [6, 9])
    crop_min_run_years: int = 8
    crop_max_forest_fraction: float = 0.2
    crop_open_fraction: float = 0.8
    crop_min_bare_fraction: float = 0.03
    crop_min_bare_year_fraction: float = 0.15
    crop_end_gap_years: int = 2
    crop_since_start_years: int = 1
    fire_season: list[int] = Field(default_factory=lambda: [6, 8])
    cleared_gap_seasons: int = 1
    crop_min_patch_px: int = 300
    crop_min_width_px: int = 5
    pre_min_fraction: float = 0.5
    object_fire_min_pixels: int = 2000
    object_fire_burned: float = 0.10
    object_fire_dnbr: float = 0.45
    object_fire_dnbr_burned: float = 0.05
    object_fire_local_burned: float = 0.03
    linear_max_width_px: int = 3
    linear_min_extent_px: int = 15
    linear_gap_px: int = 3

    @field_validator("cutblock_season", "crop_season", "fire_season")
    @classmethod
    def _validate_season(cls, value: list[int]) -> list[int]:
        if len(value) != 2 or not (1 <= value[0] <= value[1] <= 12):
            raise ValueError(f"event_typing season must be [first_month, last_month] within 1-12, got {value}")
        return value


class LandcoverPersistenceConfig(BaseModel):
    """Persistent land-cover object tracking parameters (see
    :mod:`landscape_change_detection_pipeline.change.landcover_persistence`).

    Reads Stage 7's monthly class composites (``composites.output_root``)
    directly, plus Stage 09's segment table (``change_detection.output_root``)
    for ``burned_disturbed``'s appearance date specifically -- see that
    module's own docstring for why ``burned_disturbed`` reuses Stage 09's
    break date instead of being detected purely from the monthly-composite
    majority vote every other tracked class uses.

    ``tracked_classes`` names must exist in ``configs/classes.yaml``.

    ``window_size``/``min_fraction`` control the sliding-window majority
    vote that confirms a class transition (see the module docstring for why
    a proportional vote over *observed* months, not a strict N-consecutive
    rule): within the last ``window_size`` observed months, at least
    ``min_fraction`` of them must show the tracked class to confirm it
    present. No single "right" default exists here -- tune against real
    output (too many short-lived spurious intervals means raising
    ``min_fraction`` or ``window_size``; missed real transitions means
    lowering either).

    ``tiles``/``overwrite``/``workers`` are
    ``scripts/12_build_landcover_persistence.py``'s own run parameters --
    config-driven rather than CLI flags (this project's convention: a run
    must be reproducible from ``config.yaml`` alone). ``tiles=[]`` means
    every tile in the registry. ``workers>1`` builds separate tiles in
    parallel worker processes -- tiles are fully independent here and this
    stage does no internal multi-core work of its own (unlike
    ``change_detection``'s Numba pass), so it benefits from inter-tile
    parallelism the way ``mosaic`` does.
    """

    output_root: str = "outputs/landcover_persistence"
    tiles: list[str] = Field(default_factory=list)
    overwrite: bool = False
    workers: int = 1
    tracked_classes: list[str] = Field(
        default_factory=lambda: [
            "burned_disturbed",
        ]
    )
    window_size: int = 5
    min_fraction: float = 0.8
    end_fraction: float = 0.2
    """The class ends once at most this fraction of the last ``window_size``
    observed months shows it (must be below ``min_fraction``: hysteresis, one
    odd month never closes an interval)."""
    masked_classes: list[str] = Field(default_factory=lambda: ["cloud", "shadow", "snow_cover", "ice_cover"])
    """Classes whose months are *unobserved* for the vote (neither for nor
    against the tracked class): cloud/shadow/snow/ice cover the ground, they
    say nothing about what is under them."""
    default_min_duration_months: int = 12
    min_duration_months: dict[str, int] = Field(default_factory=dict)
    """Intervals shorter than this many months are dropped (per-class
    override in ``min_duration_months``, else the default)."""
    require_break_classes: list[str] = Field(default_factory=lambda: ["burned_disturbed"])
    """Abrupt-event classes: an interval is kept only if Stage 09 found a
    break with an NBR/NDVI fall of at least ``min_break_drop`` between
    ``break_tolerance_months`` before and ``break_after_months`` after its
    start (the break month then becomes the start date)."""
    break_tolerance_months: int = 18
    break_after_months: int = 6
    min_break_drop: float = 0.1

    @field_validator("window_size")
    @classmethod
    def _validate_window_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"landcover_persistence.window_size must be >= 1, got {value}")
        return value

    @field_validator("min_fraction")
    @classmethod
    def _validate_min_fraction(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"landcover_persistence.min_fraction must be in (0, 1], got {value}")
        return value


class PhenologyConfig(BaseModel):
    """Per-year phenology of every pixel (peak month, start/end/length of the season)."""

    enabled: bool = True
    sos_fraction: float = 0.5
    """Season start/end = the date the (interpolated) monthly curve crosses
    ``low + sos_fraction * (peak - low)``, ``low`` being the pixel's
    ``baseline_percentile`` of all its observations."""
    baseline_percentile: float = 10.0
    min_amplitude: float = 0.08
    """Peak-minus-low below this = no seasonal cycle that year (nothing written)."""
    max_gap_months: int = 2
    """Interpolating across more consecutive unobserved months than this is refused."""
    trends: bool = True
    """Also fit a trend (Theil-Sen + Mann-Kendall) on the yearly start/end/length series."""

    @field_validator("sos_fraction")
    @classmethod
    def _validate_fraction(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError(f"vegetation_dynamics.phenology.sos_fraction must be in (0, 1), got {value}")
        return value


class TrendConfig(BaseModel):
    """Slow greening / browning: Theil-Sen slope + Mann-Kendall test on the yearly series."""

    enabled: bool = True
    series: list[str] = Field(default_factory=lambda: ["growing_season_mean", "annual_max"])
    min_years: int = 8
    since_last_break: bool = True
    """Also fit the trend on the years after each pixel's latest Stage 09 break
    (a trend fitted across a clear-cut says nothing about slow change)."""

    @field_validator("series")
    @classmethod
    def _validate_series(cls, value: list[str]) -> list[str]:
        bad = [v for v in value if v not in ("growing_season_mean", "annual_max")]
        if bad:
            raise ValueError(f"vegetation_dynamics.trend.series: unknown {bad}; use growing_season_mean / annual_max")
        return value


class AnomalyConfig(BaseModel):
    """Monthly z-score against the pixel's own climatology (same calendar month)."""

    enabled: bool = True
    robust: bool = True
    """median / MAD instead of mean / standard deviation: a disturbance in the
    baseline then does not inflate the scale."""
    baseline_years: Optional[list[int]] = None
    """[first_year, last_year] the climatology is computed on; null = every year."""
    min_clim_obs: int = 5
    min_scale: float = 0.02
    extreme_z: float = 2.0
    store_monthly: bool = False
    """Also write the full monthly z-score cube (n_months x H x W float16) to its own file. Large."""

    @field_validator("baseline_years")
    @classmethod
    def _validate_baseline(cls, value):
        if value is not None and (len(value) != 2 or value[0] > value[1]):
            raise ValueError(f"vegetation_dynamics.anomalies.baseline_years must be [first, last], got {value}")
        return value


class VariabilityConfig(BaseModel):
    """Interannual variability of the yearly series (coefficient of variation, year-to-year change)."""

    enabled: bool = True
    min_years: int = 5


class VegetationDynamicsConfig(BaseModel):
    """Per-pixel vegetation dynamics from the monthly index series (see
    :mod:`landscape_change_detection_pipeline.change.vegetation_dynamics`).

    Builds, once per tile, a monthly cube of each index in ``indices`` on the
    Stage 09 grid (cloud/shadow already excluded, plus the months whose
    class composite is one of ``mask_classes`` -- snow, ice, water -- since a
    vegetation index under snow is not vegetation), stored as a temporary
    on-disk cube deleted at the end. Every element below can be switched off
    on its own with ``enabled``.

    ``workers>1`` runs tiles in parallel processes; memory per worker is the
    yearly output arrays (a few hundred MB for an 8 km tile), the cube itself
    is on disk.
    """

    enabled: bool = True
    output_root: str = "outputs/vegetation_dynamics"
    tiles: list[str] = Field(default_factory=list)
    overwrite: bool = False
    workers: int = 1
    cube_workers: int = 1
    month_threads: int = 4
    block_rows: int = 64
    indices: list[str] = Field(default_factory=lambda: ["NDVI"])
    growing_season: list[int] = Field(default_factory=lambda: [5, 10])
    min_gs_obs: int = 2
    """Observed growing-season months a year needs for its yearly mean/max."""
    min_year_obs: int = 5
    """Observed months (whole year) a year needs for a phenology fit."""
    mask_classes: list[str] = Field(default_factory=lambda: ["cloud", "shadow", "snow_cover", "ice_cover", "open_water"])
    phenology: PhenologyConfig = Field(default_factory=PhenologyConfig)
    trend: TrendConfig = Field(default_factory=TrendConfig)
    anomalies: AnomalyConfig = Field(default_factory=AnomalyConfig)
    variability: VariabilityConfig = Field(default_factory=VariabilityConfig)

    @field_validator("growing_season")
    @classmethod
    def _validate_season(cls, value: list[int]) -> list[int]:
        if len(value) != 2 or not (1 <= value[0] <= value[1] <= 12):
            raise ValueError(f"vegetation_dynamics.growing_season must be [first_month, last_month] within 1-12, got {value}")
        return value

    @field_validator("indices")
    @classmethod
    def _validate_indices(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("vegetation_dynamics.indices must list at least one index (e.g. [NDVI])")
        return value


class RecoveryCurvesConfig(BaseModel):
    """Yearly index after each event, relative to the pre-event level."""

    enabled: bool = True
    max_years: int = 20
    """Years after the event's break year kept in the curve (0..max_years)."""
    pre_years: int = 3
    """Years before the break year whose median is the pre-event baseline."""
    min_pre_years: int = 2
    recovery_threshold: float = 0.8
    """Recovered = the yearly value is back to this share of the baseline..."""
    sustain_years: int = 2
    """... for this many observed years in a row (a one-year spike is not recovery)."""


class SuccessionConfig(BaseModel):
    """Dominant class of the growing season, year by year after each event."""

    enabled: bool = True
    max_years: int = 20
    min_season_obs: int = 2
    forest_class: str = "forest"


class RecoveryFactorsConfig(BaseModel):
    """What explains the recovery: terrain, event type, severity, pre-event level (tables + binned summary)."""

    enabled: bool = True
    elevation_bins_m: list[float] = Field(default_factory=lambda: [0, 500, 1000, 1500, 2000, 4000])
    slope_bins_deg: list[float] = Field(default_factory=lambda: [0, 5, 15, 30, 90])
    aspect_sectors: int = 8
    min_group_size: int = 5
    """Groups smaller than this are left out of the binned summary."""


class RecoveryAnalysisConfig(BaseModel):
    """Recovery / succession analysis of every named change event (see
    :mod:`landscape_change_detection_pipeline.change.recovery_analysis`).

    Needs Stage 09 (grid), Stage 12 (``event_typing.output_root``: the events)
    and the same monthly index cube as ``vegetation_dynamics`` (built in the
    same run of ``scripts/14_build_vegetation_dynamics.py``). The DEM
    (``dem.tile_dir``) is optional: without it the terrain columns are NaN.
    Every element has its own ``enabled``.
    """

    enabled: bool = True
    output_root: str = "outputs/recovery_analysis"
    overwrite: bool = False
    index: str = "NDVI"
    """The index (one of ``vegetation_dynamics.indices``) the curves are built on."""
    event_types: list[str] = Field(
        default_factory=lambda: ["fire", "cutblock", "permanent_clearing", "canopy_decline", "other_loss", "recent_clearing", "linear_feature", "open_land_disturbance", "regrowth_change", "cropland_change"]
    )
    curves: RecoveryCurvesConfig = Field(default_factory=RecoveryCurvesConfig)
    succession: SuccessionConfig = Field(default_factory=SuccessionConfig)
    factors: RecoveryFactorsConfig = Field(default_factory=RecoveryFactorsConfig)


class FeatureImportanceMethodConfig(BaseModel):
    enabled: bool = True


class FeatureImportanceConfig(BaseModel):
    """Which input features / classes matter (``scripts/05c_feature_importance.py``).

    Only ``model.type`` = ``unet`` or ``catboost`` is supported: the analysis was
    not set up or tuned for any other model type, and the script refuses them.
    Every method has its own ``enabled``. Results are one ``.npz`` of ``tbl_*``
    tables under ``output_root`` (exportable as CSV from ``scripts/export_gui.py``).
    """

    enabled: bool = True
    output_root: str = "outputs/feature_importance"
    checkpoint_path: Optional[str] = None
    """null = ``<training.checkpoint_dir>/<model.type>_best.<pt|joblib>``."""
    device: Optional[str] = None
    split: str = "val"
    """Scenes the importance is measured on: train | val | test."""
    max_scenes: int = 30
    """Scenes the unet permutation draws patches from (0 = all of the split)."""
    patches_per_scene: int = 8
    """unet permutation: random labelled patches (training patch_size) cut per scene."""
    batch_size: int = 32
    """unet permutation: patches per forward pass."""
    pixels_per_scene: int = 3000
    """Labelled pixels sampled per scene for the CatBoost-based methods."""
    proxy_iterations: int = 300
    """Max CatBoost iterations of the proxy / drop-column fits."""
    drop_threshold: float = 0.01
    """mIoU drop at or below this = unimportant (drop-column; ~ the fit-to-fit noise)."""
    corr_threshold: float = 0.95
    feature_groups: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "chromaticity(r,g,b)": ["r", "g", "b"],
            "tasseled_cap": ["TC_BRIGHTNESS", "TC_GREENNESS", "TC_WETNESS"],
            "dem(elevation,slope)": ["elevation", "slope"],
            "doy(sin,cos)": ["doy_sin", "doy_cos"],
            "swir_indices": ["NDSI", "NDWI_GAO", "NBR", "ND_SWIR1_SWIR2"],
            "blue/green_ratios": ["ND_BLUE_RED", "ND_BLUE_NIR", "ND_GREEN_RED"],
        }
    )
    """Channels also destroyed together as one extra row ({} = no group rows)."""

    class PermutationConfig(FeatureImportanceMethodConfig):
        repeats: int = 3

    class ShapConfig(FeatureImportanceMethodConfig):
        per_class: int = 3000

    class SeparabilityConfig(FeatureImportanceMethodConfig):
        pixels_per_class: int = 20000
        weak_below: float = 1.5

    permutation: PermutationConfig = Field(default_factory=PermutationConfig)
    shap: ShapConfig = Field(default_factory=ShapConfig)
    dropcolumn: FeatureImportanceMethodConfig = Field(default_factory=FeatureImportanceMethodConfig)
    separability: SeparabilityConfig = Field(default_factory=SeparabilityConfig)


class ChangeMapsConfig(BaseModel):
    """Combined annual/month-to-month change map parameters (see
    :mod:`landscape_change_detection_pipeline.change.change_maps`).

    ``persistence_periods`` is how many stored follow-up months must also
    show the *to*-period class for a raw classification change to count as
    "persistent" -- kept deliberately short (default 1) so a real but brief
    disturbance is not filtered out the way a long persistence requirement
    would (explicit project direction: short-lived real events matter here,
    not just multi-month ones). ``corroboration_window_days`` is how close
    a CCDC/BFAST break date must fall to a compared period boundary to
    count as corroborating a classification change (see the module's own
    docstring for why a window rather than an exact-date match). Both
    ``ccdc``/``bfast`` results are read from their own already-configured
    ``ccdc.output_root``/``bfast.output_root`` -- nothing new to configure
    for where those come from.

    ``tiles``/``overwrite``/``workers`` are ``scripts/13_build_change_maps.py``'s
    own run parameters -- config-driven rather than CLI flags (this
    project's convention: a run must be reproducible from ``config.yaml``
    alone). ``tiles=[]`` means every tile in the registry. ``workers>1``
    builds separate tiles in parallel worker processes (every comparison
    pair for one tile still runs sequentially within its own worker) --
    tiles are fully independent here and this stage does no internal
    multi-core work of its own (unlike ``change_detection``'s Numba pass),
    so it benefits from inter-tile parallelism the way ``mosaic`` does.
    """

    persistence_periods: int = 1
    corroboration_window_days: int = 45
    output_root: str = "outputs/change_maps"
    tiles: list[str] = Field(default_factory=list)
    overwrite: bool = False
    workers: int = 1


class MosaicConfig(BaseModel):
    """AOI-wide mosaicking parameters (see :mod:`landscape_change_detection_pipeline.mosaic.mosaic`).

    Each period's output resolution is decided automatically -- the finest
    ``resolution_m`` achieved by any tile that period, with coarser tiles
    nearest-neighbour-resampled up to it -- so there is no
    resolution field to configure here. ``output_root`` holds one
    ``<period>/mosaic.tif`` Cloud-Optimized GeoTIFF per period.

    ``periods``/``overwrite``/``workers`` are ``scripts/08_build_mosaics.py``'s
    own run parameters -- config-driven rather than CLI flags (this
    project's convention: a run must be reproducible from ``config.yaml``
    alone). ``periods=[]`` means every period with at least one tile
    composite. ``workers>1`` builds separate periods' mosaics in parallel
    worker processes -- periods are fully independent (each reads only that
    period's tile composites and writes its own output subtree) and this
    stage is CPU/numpy-only (no GPU contention), so this can reasonably
    scale close to the machine's core count.
    """

    output_root: str = "outputs/mosaics"
    periods: list[str] = Field(default_factory=list)
    overwrite: bool = False
    workers: int = 1


class PipelineConfig(BaseModel):
    """Root configuration model for the landscape classification and
    change detection pipeline."""

    paths: PathsConfig
    gee: GeeConfig
    tiles: TilesConfig = Field(default_factory=TilesConfig)
    dem: DemConfig = Field(default_factory=DemConfig)
    scene_download: SceneDownloadConfig = Field(default_factory=SceneDownloadConfig)
    harmonization: HarmonizationConfig = Field(default_factory=HarmonizationConfig)
    topographic_correction: TopographicCorrectionConfig = Field(default_factory=TopographicCorrectionConfig)
    rgb_composites: RgbCompositesConfig = Field(default_factory=RgbCompositesConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(type="unet"))
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    hpo: HpoConfig = Field(default_factory=HpoConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    composites: CompositesConfig = Field(default_factory=CompositesConfig)
    mosaic: MosaicConfig = Field(default_factory=MosaicConfig)
    change_detection: ChangeDetectionConfig = Field(default_factory=ChangeDetectionConfig)
    snow_water_dynamics: SnowWaterDynamicsConfig = Field(default_factory=SnowWaterDynamicsConfig)
    regrowth_severity: RegrowthSeverityConfig = Field(default_factory=RegrowthSeverityConfig)
    landcover_persistence: LandcoverPersistenceConfig = Field(default_factory=LandcoverPersistenceConfig)
    event_typing: EventTypingConfig = Field(default_factory=EventTypingConfig)
    change_maps: ChangeMapsConfig = Field(default_factory=ChangeMapsConfig)
    vegetation_dynamics: VegetationDynamicsConfig = Field(default_factory=VegetationDynamicsConfig)
    recovery_analysis: RecoveryAnalysisConfig = Field(default_factory=RecoveryAnalysisConfig)
    feature_importance: FeatureImportanceConfig = Field(default_factory=FeatureImportanceConfig)


def _substitute_env_vars(raw_text: str) -> str:
    """Replace ${VAR} placeholders in raw_text with values from the environment.

    Raises ConfigError naming the missing variable if a referenced ${VAR} is not
    set in the environment (or in a .env file already loaded via load_dotenv).
    """

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.environ.get(var_name)
        if value is None:
            raise ConfigError(
                f"Config references ${{{var_name}}} but no such environment "
                f"variable is set (check your .env file or environment)."
            )
        return value

    return _ENV_VAR_PATTERN.sub(_replace, raw_text)


def load_config(config_path: Optional[str] = None, env_file: Optional[str] = None) -> PipelineConfig:
    """Load, substitute, and validate the pipeline configuration.

    Resolution order for the config path: explicit `config_path` argument,
    then the LANDSCAPE_CHANGE_DETECTION_CONFIG environment variable, then
    "configs/config.yaml" relative to the current working directory.

    A .env file (default: ".env" in the current working directory) is loaded
    first via python-dotenv so its values are available for ${VAR} substitution.
    """
    load_dotenv(dotenv_path=env_file)  # no-op if the file does not exist

    resolved_path = config_path or os.environ.get(CONFIG_ENV_VAR) or "configs/config.yaml"
    path = Path(resolved_path)
    if not path.is_file():
        raise ConfigError(
            f"Config file not found at '{path}'. Pass --config, set "
            f"{CONFIG_ENV_VAR}, or create configs/config.yaml (see "
            f"configs/config.example.yaml)."
        )

    raw_text = path.read_text(encoding="utf-8")
    substituted_text = _substitute_env_vars(raw_text)

    try:
        raw_data = yaml.safe_load(substituted_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML in '{path}': {exc}") from exc

    try:
        return PipelineConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in '{path}':\n{exc}") from exc
