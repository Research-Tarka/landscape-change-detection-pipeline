"""Land-cover segmentation/classification models.

This package holds one module per selectable model "technology"
(``model.type`` in :class:`landscape_change_detection_pipeline.config.ModelConfig`):

- ``threshold`` -- spectral-index thresholds tuned by Optuna (also covers
  the ``grid_search`` case, expressed as a ``threshold.sampler`` choice
  rather than a separate model type; see
  :mod:`landscape_change_detection_pipeline.models.threshold`)
- ``random_forest`` -- :mod:`landscape_change_detection_pipeline.models.random_forest`
- ``catboost`` -- :mod:`landscape_change_detection_pipeline.models.catboost_model`
- ``unet`` -- :mod:`landscape_change_detection_pipeline.models.unet`
- ``deeplabv3plus`` -- :mod:`landscape_change_detection_pipeline.models.deeplabv3plus`
- ``segformer`` -- :mod:`landscape_change_detection_pipeline.models.segformer`

:mod:`landscape_change_detection_pipeline.models.registry` is the single dispatch point:
its ``build_model(model_cfg, ...)`` reads ``model_cfg.type`` and constructs
the corresponding model, importing that one module lazily so that an
environment without e.g. ``catboost`` or ``transformers`` installed can
still use the other six model types.
"""

from __future__ import annotations
