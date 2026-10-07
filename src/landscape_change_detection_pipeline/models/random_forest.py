"""Pixel-level Random Forest classifier for land-cover classification.

Unlike the patch-based models in this package (``unet``, ``deeplabv3plus``,
``segformer``), a Random Forest is fit on individual pixels: every valid
(non-``IGNORE_INDEX``) pixel across every training scene becomes one row,
with its ``C`` feature-channel values as columns and its class label as the
target. This uses the standard pixel-level tabular ML
pattern: CPU-only (no GPU RF available on Windows
without RAPIDS/cuML), and a single ``scikit-learn`` ``RandomForestClassifier``
fit.

Memory tradeoff
----------------
:meth:`RandomForestModel.fit` materialises every valid pixel from every
training scene as one ``(N_valid_pixels, C)`` array before calling
``sklearn``'s ``.fit`` (sklearn's ``RandomForestClassifier`` has no partial/
incremental fit, so the whole training matrix must exist in memory at once
regardless of how it is assembled). This module still avoids the cheapest-
to-write approach -- appending each scene's ``(n_i, C)`` array to a Python
list and calling ``np.concatenate`` once at the end, which briefly holds two
full copies of the data (the per-scene list and the concatenated output) --
by making two passes over the scene caches: a first pass to count total
valid pixels per scene (cheap: only ``labels`` needs loading), then a single
pre-allocated ``(N_total, C)``/``(N_total,)`` pair filled scene-by-scene.
This costs one extra ``labels``-only load per scene but keeps peak memory at
roughly one copy of the full pixel matrix instead of two. For a corpus large
enough that even one copy does not fit in memory, this whole approach would
need revisiting (e.g. streaming/subsampling as the reference script's
``stratified_subsample`` does for its hyperparameter-search stage) -- out of
scope here.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from landscape_change_detection_pipeline.features.training_cache import load_scene_cache
from landscape_change_detection_pipeline.training.dataset import SceneRecord
from landscape_change_detection_pipeline.training.losses import IGNORE_INDEX

__all__ = ["RandomForestModel", "build_random_forest"]


class RandomForestModel:
    """Thin wrapper around ``sklearn.ensemble.RandomForestClassifier`` for
    pixel-level land-cover classification.

    Holds ``num_classes`` explicitly (independent of how many distinct
    classes actually appear in the training pixels) so that
    :meth:`predict_proba` can always return a full ``(num_classes, H, W)``
    array -- see that method's docstring for why this matters.
    """

    def __init__(
        self,
        num_classes: int,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        n_jobs: int = -1,
        class_weight: Optional[str] = "balanced",
        random_state: int = 0,
    ) -> None:
        self.num_classes = int(num_classes)
        self._sklearn_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            class_weight=class_weight,
            random_state=random_state,
        )

    # -- fitting -----------------------------------------------------------

    def fit(self, train_records: Sequence[SceneRecord]) -> "RandomForestModel":
        """Fit on every valid pixel of every scene in ``train_records``.

        Loads each scene's cached ``(C, H, W)`` feature stack and ``(H, W)``
        label map (see :mod:`landscape_change_detection_pipeline.features.training_cache`),
        drops pixels labelled ``IGNORE_INDEX`` (unannotated), and flattens the
        remainder into rows of a single ``(N_valid_pixels, C)`` design matrix.
        See the module docstring for the two-pass memory strategy used here.
        """
        records = sorted(train_records, key=lambda r: r.sort_key)
        if not records:
            raise ValueError("RandomForestModel.fit requires at least one train scene.")

        # First pass: count valid pixels per scene (labels only) so the
        # design matrix can be pre-allocated once instead of grown.
        n_channels: Optional[int] = None
        valid_counts: list[int] = []
        for record in records:
            cache = load_scene_cache(record.cache_dir, label_remap=record.label_remap)
            labels = cache["labels"]
            valid_counts.append(int(np.count_nonzero(labels != IGNORE_INDEX)))
            c = cache["features"].shape[0]
            if n_channels is None:
                n_channels = c
            elif c != n_channels:
                raise ValueError(
                    f"Scene '{record.scene_key}' has {c} feature channels, expected "
                    f"{n_channels} (every scene must share the same feature stack)."
                )

        total_valid = sum(valid_counts)
        if total_valid == 0:
            raise ValueError("RandomForestModel.fit found no valid (non-ignore-index) pixels.")
        assert n_channels is not None  # for type-checkers; guaranteed by the loop above

        X = np.empty((total_valid, n_channels), dtype=np.float32)
        y = np.empty((total_valid,), dtype=np.int64)

        # Second pass: fill the pre-allocated arrays scene-by-scene.
        offset = 0
        for record, n_valid in zip(records, valid_counts):
            if n_valid == 0:
                continue
            cache = load_scene_cache(record.cache_dir, label_remap=record.label_remap)
            features = cache["features"].astype(np.float32)  # (C, H, W)
            labels = cache["labels"]  # (H, W)
            mask = labels != IGNORE_INDEX

            # (C, H, W) -> (H, W, C) -> (n_valid, C), selecting only valid pixels.
            scene_X = np.moveaxis(features, 0, -1)[mask]
            scene_y = labels[mask].astype(np.int64)

            X[offset : offset + n_valid] = scene_X
            y[offset : offset + n_valid] = scene_y
            offset += n_valid

        self._sklearn_model.fit(X, y)
        return self

    # -- inference -----------------------------------------------------------

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class ids for one scene's ``(C, H, W)`` feature stack.

        Returns an ``(H, W)`` array of predicted class ids. Raises
        sklearn's ``NotFittedError`` if called before :meth:`fit`.
        """
        c, h, w = features.shape
        flat = np.moveaxis(features, 0, -1).reshape(h * w, c)
        pred = self._sklearn_model.predict(flat)
        return pred.reshape(h, w)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict per-class probabilities for one scene's ``(C, H, W)``
        feature stack.

        Returns an ``(num_classes, H, W)`` array. Raises sklearn's
        ``NotFittedError`` if called before :meth:`fit`.

        sklearn's ``RandomForestClassifier.predict_proba`` only returns
        columns for the classes actually seen during ``.fit`` (in the order
        given by ``self._sklearn_model.classes_``) -- a class entirely absent
        from the training pixels has **no column** in its output, rather than
        a column of zeros. Naively reshaping that output would silently
        misalign class ids whenever any class was missing from training (or
        put a rare class's probabilities under the wrong index). To avoid
        this, sklearn's output columns are explicitly scattered into a full
        ``(num_classes,)``-wide array using ``classes_`` as the column ->
        class-id mapping, leaving every absent class's probability at 0.0.
        """
        c, h, w = features.shape
        flat = np.moveaxis(features, 0, -1).reshape(h * w, c)
        sklearn_proba = self._sklearn_model.predict_proba(flat)  # (N, n_seen_classes)

        full_proba = np.zeros((h * w, self.num_classes), dtype=np.float64)
        seen_classes = self._sklearn_model.classes_  # (n_seen_classes,), column -> class id
        full_proba[:, seen_classes.astype(np.int64)] = sklearn_proba

        return np.moveaxis(full_proba.reshape(h, w, self.num_classes), -1, 0)


def build_random_forest(
    num_classes: int,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    n_jobs: int = -1,
    class_weight: Optional[str] = "balanced",
    random_state: int = 0,
    train_records: Optional[Sequence[SceneRecord]] = None,
    **_ignored,
) -> RandomForestModel:
    """Construct a :class:`RandomForestModel`, called by
    :mod:`landscape_change_detection_pipeline.models.registry` (see
    :func:`landscape_change_detection_pipeline.models.registry.build_model`).

    If ``train_records`` is ``None``, returns an unfitted model rather than
    fitting -- the registry constructs models before training happens, so it
    never passes ``train_records`` itself. Calling ``.predict``/
    ``.predict_proba`` on an unfitted model raises sklearn's own
    ``NotFittedError``, which is the expected/accepted behaviour here rather
    than a bug to guard against in this constructor. If ``train_records`` is
    provided (e.g. a caller that wants a ready-to-use model in one call), the
    model is fit before being returned.

    ``**_ignored`` absorbs any config fields the registry's other model
    branches project (e.g. fields specific to a different model type) so
    that a shared/looser call site cannot crash this constructor on an
    unexpected keyword.
    """
    model = RandomForestModel(
        num_classes=num_classes,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        class_weight=class_weight,
        random_state=random_state,
    )
    if train_records is not None:
        model.fit(train_records)
    return model
