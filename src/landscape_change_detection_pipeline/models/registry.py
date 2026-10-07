"""Single dispatch point from ``model.type`` to a constructed model instance.

:func:`build_model` reads
:class:`landscape_change_detection_pipeline.config.ModelConfig`'s ``type`` field and calls
the matching ``build_<name>`` constructor in that model's own module (see
:mod:`landscape_change_detection_pipeline.models`). Each sibling module is imported
lazily, inside its own branch, rather than at the top of this file: importing
this module (e.g. to read ``MODEL_TYPES``) must not require ``catboost`` or
``transformers`` to be installed unless the corresponding branch is actually
taken.

Config fields are projected onto each ``build_<name>`` call explicitly,
field-by-field, rather than via ``model_cfg.<section>.model_dump()`` --
matching this project's existing config-projection convention (e.g.
``landscape_change_detection_pipeline.training.dataset``'s explicit dataclasses over raw
config passthrough) -- so a renamed or removed config field fails loudly at
the call site instead of silently changing a constructor's kwargs.
"""

from __future__ import annotations

from typing import Any

from landscape_change_detection_pipeline.config import MODEL_TYPES, ModelConfig

__all__ = ["MODEL_TYPES", "build_model"]


def build_model(
    model_cfg: ModelConfig,
    in_channels: int,
    num_classes: int,
    num_sensors: int = 1,
) -> Any:
    """Construct the model selected by ``model_cfg.type``.

    Args:
        model_cfg: the validated ``model`` section of the pipeline config.
        in_channels: input feature channel count (only used by the
            pixel/patch-based models: unet, deeplabv3plus, segformer).
        num_classes: number of land-cover classes (see
            :mod:`landscape_change_detection_pipeline.classes.class_config`); required by
            every model type.
        num_sensors: number of distinct sensors in play, passed through to
            models that support per-sensor handling (currently only unet).

    Returns:
        The constructed model: an ``nn.Module`` for unet/deeplabv3plus/segformer,
        or a plain estimator object for threshold/random_forest/catboost.

    Raises:
        ValueError: if ``model_cfg.type`` is not one of ``MODEL_TYPES``.
            The pydantic validator on ``ModelConfig.type`` should make this
            unreachable in practice, but this function defends anyway since
            it may be called with a ``ModelConfig`` built by hand (e.g. in
            tests) rather than loaded from YAML.
    """
    model_type = model_cfg.type

    if model_type == "threshold":
        from landscape_change_detection_pipeline.models.threshold import build_threshold

        cfg = model_cfg.threshold
        return build_threshold(
            num_classes=num_classes,
            n_trials=cfg.n_trials,
            sampler=cfg.sampler,
            timeout_s=cfg.timeout_s,
            study_storage=cfg.study_storage,
        )

    if model_type == "random_forest":
        from landscape_change_detection_pipeline.models.random_forest import build_random_forest

        cfg = model_cfg.random_forest
        return build_random_forest(
            num_classes=num_classes,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            n_jobs=cfg.n_jobs,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
        )

    if model_type == "catboost":
        from landscape_change_detection_pipeline.models.catboost_model import build_catboost

        cfg = model_cfg.catboost
        return build_catboost(
            num_classes=num_classes,
            iterations=cfg.iterations,
            learning_rate=cfg.learning_rate,
            depth=cfg.depth,
            task_type=cfg.task_type,
            random_state=cfg.random_state,
            early_stopping_rounds=cfg.early_stopping_rounds,
            class_weighting=cfg.class_weighting,
        )

    if model_type == "lightgbm":
        from landscape_change_detection_pipeline.models.lightgbm_model import build_lightgbm

        cfg = model_cfg.lightgbm
        return build_lightgbm(
            num_classes=num_classes,
            num_leaves=cfg.num_leaves,
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_data_in_leaf=cfg.min_data_in_leaf,
            device=cfg.device,
            random_state=cfg.random_state,
            early_stopping_rounds=cfg.early_stopping_rounds,
            class_weighting=cfg.class_weighting,
        )

    if model_type == "unet":
        from landscape_change_detection_pipeline.models.unet import build_unet

        cfg = model_cfg.unet
        return build_unet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_ch=cfg.base_ch,
            dropout_p=cfg.dropout_p,
            num_sensors=num_sensors,
            film_sensor=cfg.film_sensor,
            film_doy=cfg.film_doy,
            film_latlon=cfg.film_latlon,
            norm_type=cfg.norm_type,
            bottleneck_attention=cfg.bottleneck_attention,
            bottleneck_attention_heads=cfg.bottleneck_attention_heads,
            deep_supervision=cfg.deep_supervision,
            depth=cfg.depth,
        )

    if model_type == "deeplabv3plus":
        from landscape_change_detection_pipeline.models.deeplabv3plus import build_deeplabv3plus

        cfg = model_cfg.deeplabv3plus
        return build_deeplabv3plus(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone=cfg.backbone,
            pretrained=cfg.pretrained,
            dropout_p=cfg.dropout_p,
        )

    if model_type == "segformer":
        from landscape_change_detection_pipeline.models.segformer import build_segformer

        cfg = model_cfg.segformer
        return build_segformer(
            in_channels=in_channels,
            num_classes=num_classes,
            variant=cfg.variant,
            pretrained=cfg.pretrained,
            dropout_p=cfg.dropout_p,
        )

    # Unreachable given ModelConfig.type's pydantic validator, but defended
    # anyway (e.g. a ModelConfig constructed via model_construct(), bypassing
    # validation).
    raise ValueError(
        f"Unknown model.type={model_type!r}; expected one of {MODEL_TYPES!r}."
    )
