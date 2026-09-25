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
    """

    train_root: str = "data/train_cache"
    mask_root: str = "data/masks"
    geotiff_export_root: str = "data/train_geotiff"
    index_names: Optional[list[str]] = None
    dem_layer_names: Optional[list[str]] = None

    @field_validator("index_names")
    @classmethod
    def _validate_index_names(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        from landscape_change_detection_pipeline.features.spectral_indices import INDEX_NAMES

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
    ``FeaturesConfig.index_names``/``dem_layer_names``)."""
    from landscape_change_detection_pipeline.features.spectral_indices import DEM_LAYER_NAMES, INDEX_NAMES

    index_names = tuple(features.index_names) if features.index_names is not None else INDEX_NAMES
    dem_layer_names = tuple(features.dem_layer_names) if features.dem_layer_names is not None else DEM_LAYER_NAMES
    return index_names, dem_layer_names


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


class UnetConfig(BaseModel):
    """Architecture options for the U-Net segmentation model (see
    :mod:`landscape_change_detection_pipeline.models.unet`, mirroring ``build_unet``'s
    keyword arguments and defaults exactly).

    ``base_ch`` is the first encoder stage's channel width (doubling at each
    of the three subsequent stages). ``use_spatial_context`` enables FiLM
    conditioning of the first encoder stage on ``(x, y, year_norm,
    area_norm)``; ``use_sensor_film`` additionally folds a learned sensor
    embedding into that conditioning and only has an effect when
    ``num_sensors > 1``. ``norm_type="group"`` is incompatible with
    ``num_sensors > 1`` (see ``UNet``'s own validation). ``bottleneck_attention``
    inserts self-attention over the bottleneck grid; ``deep_supervision``
    attaches auxiliary decoder-level logit heads used only during training.
    """

    base_ch: int = 48
    dropout_p: float = 0.0
    use_spatial_context: bool = False
    num_sensors: int = 1
    use_sensor_film: bool = False
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
    batch_size: int = 16
    patch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15
    num_workers: int = 4
    seed: int = 0
    deterministic: bool = False
    amp: bool = True
    grad_clip_norm: float = 1.0
    main_iou_metric: str = "miou_macro"
    augment_flips: bool = True
    augment_rotate90: bool = True
    confusion_penalties: list[ConfusionPenaltyConfig] = Field(default_factory=list)
    checkpoint_dir: str = "models/checkpoints"


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
    """

    patch_size: int = 256
    stride: int = 128
    batch_size: int = 32
    scene_precheck_min_valid_ratio: float = 0.30
    overwrite: bool = False
    output_root: str = "outputs/inference"
    ambiguity_threshold: Optional[float] = None
    class_priority_order: list[int] = Field(default_factory=list)


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
    output_root: str = "outputs/composites"


class CcdcConfig(BaseModel):
    """Local CCDC/COLD change-detection parameters (see
    :mod:`landscape_change_detection_pipeline.change.ccdc`).

    ``lam``/``p_cg``/``conse`` mirror pyxccd's ``cold_detect_flex`` kwargs of
    the same names (Lasso regularization weight, change-magnitude
    probability threshold, consecutive-observation count to confirm a
    break); defaults match pyxccd's own and GEE CCDC's documented defaults.
    ``min_clear_obs`` guards against fitting a pixel with too little usable
    history (permanently cloud/shadow-flagged, or too few scenes) --
    pyxccd's own COLD initialization needs a minimum window of clear
    observations to fit meaningfully. ``n_workers`` spreads pixels across
    local CPU cores via ``multiprocessing`` (never GPU -- no CUDA path
    exists for this algorithm).
    """

    lam: float = 20.0
    p_cg: float = 0.99
    conse: int = 6
    min_clear_obs: int = 12
    n_workers: int = 1
    output_root: str = "outputs/ccdc"


class BfastConfig(BaseModel):
    """BFAST-Monitor fire cross-check parameters (see
    :mod:`landscape_change_detection_pipeline.change.bfast`).

    ``monitor_start`` (``"YYYY-MM-DD"``) splits each tile's time series into
    the stable history period (before) and the monitoring period being
    tested (on/after) -- the same split for every tile/pixel, since BFAST-
    Monitor's near-real-time framing assumes one fixed "as of" boundary
    rather than a per-pixel one. ``order``/``alpha`` are the harmonic
    regression order and CUSUM significance level (see the module's own
    docstring for the exact formulas). ``n_workers`` mirrors
    ``ccdc.n_workers`` -- CPU processes, never GPU (no CUDA path exists for
    this algorithm either).
    """

    monitor_start: str = "2020-01-01"
    order: int = 3
    alpha: float = 0.05
    min_history_obs: int = 12
    min_monitoring_obs: int = 3
    n_workers: int = 1
    output_root: str = "outputs/bfast"


class RegrowthSeverityConfig(BaseModel):
    """NDVI regrowth / dNBR burn severity parameters (see
    :mod:`landscape_change_detection_pipeline.change.regrowth_severity`).

    ``break_source`` selects which break-detection output supplies each
    pixel's disturbance date -- ``"ccdc"`` (general breaks) or ``"bfast"``
    (fire-specific cross-check). No NDSI snow filter here by design -- the
    trained classifier's own ``snow_cover`` class already answers that
    question from the same composites this module reads; see
    :mod:`change.regrowth_severity`'s own module docstring for why dNBR and
    NDVI regrowth earn a separate representation but snow does not.
    """

    break_source: str = "ccdc"
    output_root: str = "outputs/regrowth_severity"

    @field_validator("break_source")
    @classmethod
    def _validate_break_source(cls, value: str) -> str:
        if value not in ("ccdc", "bfast"):
            raise ValueError(f"regrowth_severity.break_source must be 'ccdc' or 'bfast', got {value!r}")
        return value


class IndexCompositesConfig(BaseModel):
    """Monthly per-tile spectral-index composite parameters (see
    :mod:`landscape_change_detection_pipeline.change.spectral_composites`) --
    analysis-only NDVI/NDSI/NBR/Tasseled-Cap/etc. statistics, never fed to
    the land-cover model. No reduction-rule knobs here (unlike
    ``composites``): every index always keeps median/min/max/n_obs
    unconditionally, since which statistic a given change-detection layer
    needs varies by layer, not by a single global config choice."""

    output_root: str = "outputs/index_composites"


class IndexMosaicConfig(BaseModel):
    """AOI-wide spectral-index mosaic parameters (see
    :mod:`landscape_change_detection_pipeline.change.index_mosaic`) -- the continuous-
    index counterpart of ``mosaic`` (Stage 8's categorical class mosaic)."""

    output_root: str = "outputs/index_mosaics"


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
    """

    persistence_periods: int = 1
    corroboration_window_days: int = 45
    output_root: str = "outputs/change_maps"


class MosaicConfig(BaseModel):
    """AOI-wide mosaicking parameters (see :mod:`landscape_change_detection_pipeline.mosaic.mosaic`).

    Each period's output resolution is decided automatically -- the finest
    ``resolution_m`` achieved by any tile that period, with coarser tiles
    nearest-neighbour-resampled up to it -- so there is no
    resolution field to configure here. ``output_root`` holds one
    ``<period>/mosaic.tif`` Cloud-Optimized GeoTIFF per period.
    """

    output_root: str = "outputs/mosaics"


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
    index_composites: IndexCompositesConfig = Field(default_factory=IndexCompositesConfig)
    index_mosaic: IndexMosaicConfig = Field(default_factory=IndexMosaicConfig)
    ccdc: CcdcConfig = Field(default_factory=CcdcConfig)
    bfast: BfastConfig = Field(default_factory=BfastConfig)
    regrowth_severity: RegrowthSeverityConfig = Field(default_factory=RegrowthSeverityConfig)
    change_maps: ChangeMapsConfig = Field(default_factory=ChangeMapsConfig)


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
