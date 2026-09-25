"""Single source-of-truth land-cover class definitions (``configs/classes.yaml``).

Purpose
-------
Both this pipeline's training code and MaskForge (the annotation tool) need the same 14-class
land-cover taxonomy with the same integer ids and colors. Rather than define
the classes twice and risk drift, this module is the single reader of
``configs/classes.yaml``, and provides ``to_maskforge_palette`` to derive a
MaskForge ``ClassPalette`` JSON (matching the schema in
``Maskforge/sidecar/maskforge_core/class_config.py``) from it on demand.

Stability
---------
``ClassDef.id`` is a stable integer, assigned once in ``classes.yaml`` and
never reused or renumbered -- it is burned into annotation rasters and
trained-model output channels. Adding a class (e.g. splitting
"wetland/riparian" into finer subclasses) means appending a new
id at the end of ``classes.yaml``, never editing or renumbering an existing
one; see the rules documented at the top of that file.

``ClassDef.change_eligible`` marks classes that must never count in
change-detection comparisons (``cloud``, ``shadow`` -- these are imaging
artifacts, not land cover).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

DEFAULT_CLASSES_PATH = Path("configs/classes.yaml")


class ClassConfigError(Exception):
    """Raised when ``configs/classes.yaml`` is missing, malformed, or invalid."""


@dataclass(frozen=True)
class ClassDef:
    """One land-cover class: a stable id, names, a display color, and change eligibility."""

    id: int
    name: str
    display_name: str
    color: tuple[int, int, int]
    change_eligible: bool = True


@dataclass(frozen=True)
class ClassConfig:
    """The full ordered set of land-cover classes loaded from ``classes.yaml``.

    ``ClassDef.id`` (the "raw" id) is stable and never renumbered, so any
    contributor can append or retire classes over time without disturbing
    ids already burned into other contributors' annotation rasters -- but
    that means raw ids are not guaranteed contiguous from 0 (this pipeline
    has shipped with gaps, e.g. ids 11/12 absent from a 14-class file whose
    highest id is 15). Every place a class id sizes or indexes a tensor
    (the loss, a model's output channel count, a one-hot encoding) needs a
    **dense** id space (0..N-1, no gaps) instead, or a raw id past
    ``len(classes)`` indexes out of bounds. :attr:`num_model_classes` /
    :meth:`raw_to_dense` / :meth:`dense_to_raw` provide that mapping
    (dense index == position in ``classes``, i.e. file order) without
    requiring raw ids to be contiguous. Model-facing code should use dense
    ids throughout and only convert back to raw ids at the boundary
    (reading a painted mask, writing/interpreting a prediction raster).
    """

    classes: tuple[ClassDef, ...]

    def by_id(self, class_id: int) -> ClassDef:
        for c in self.classes:
            if c.id == class_id:
                return c
        raise KeyError(f"No class with id={class_id}")

    def by_name(self, name: str) -> ClassDef:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(f"No class with name={name!r}")

    def change_eligible_ids(self) -> tuple[int, ...]:
        return tuple(c.id for c in self.classes if c.change_eligible)

    @property
    def num_model_classes(self) -> int:
        """The dense class count -- always ``len(self.classes)``, since a
        dense id is a class's position in this tuple, not its raw id."""
        return len(self.classes)

    @property
    def _raw_to_dense_map(self) -> dict[int, int]:
        return {c.id: i for i, c in enumerate(self.classes)}

    @property
    def _dense_to_raw_map(self) -> dict[int, int]:
        return {i: c.id for i, c in enumerate(self.classes)}

    def raw_to_dense(self, raw_id: int) -> int:
        """Map a raw ``ClassDef.id`` (as painted in a mask) to its dense
        index (0..N-1, this class's position in ``classes``)."""
        try:
            return self._raw_to_dense_map[raw_id]
        except KeyError:
            raise KeyError(f"No class with raw id={raw_id}") from None

    def dense_to_raw(self, dense_id: int) -> int:
        """Inverse of :meth:`raw_to_dense`: a model's dense output channel
        index back to the class's stable raw ``ClassDef.id``."""
        try:
            return self._dense_to_raw_map[dense_id]
        except KeyError:
            raise KeyError(f"No class with dense id={dense_id}") from None

    def remap_raw_to_dense(self, labels: np.ndarray, nodata_value: int = 255) -> np.ndarray:
        """Vectorized :meth:`raw_to_dense` over a whole label array.

        Any value not a known raw id (including an existing ``nodata_value``
        sentinel) maps to ``nodata_value`` in the output -- never silently
        aliased onto a real dense class id.
        """
        raw_to_dense = self._raw_to_dense_map
        out = np.full(labels.shape, nodata_value, dtype=np.uint8)
        for raw_id, dense_id in raw_to_dense.items():
            out[labels == raw_id] = dense_id
        return out


def load_class_config(path: Optional[str] = None) -> ClassConfig:
    """Load and validate the land-cover class definitions from YAML.

    Resolution order for the path: explicit `path` argument, then
    "configs/classes.yaml" relative to the current working directory.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_CLASSES_PATH
    if not resolved_path.is_file():
        raise ClassConfigError(f"Class config file not found at '{resolved_path}'.")

    try:
        raw_data = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ClassConfigError(f"Failed to parse YAML in '{resolved_path}': {exc}") from exc

    raw_classes = (raw_data or {}).get("classes")
    if not raw_classes:
        raise ClassConfigError(f"'{resolved_path}' defines no classes.")

    classes: list[ClassDef] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for entry in raw_classes:
        try:
            class_id = int(entry["id"])
            name = str(entry["name"])
            display_name = str(entry["display_name"])
            color = tuple(int(v) for v in entry["color"])
            change_eligible = bool(entry.get("change_eligible", True))
        except (KeyError, TypeError, ValueError) as exc:
            raise ClassConfigError(f"Invalid class entry in '{resolved_path}': {entry!r} ({exc})") from exc

        if len(color) != 3:
            raise ClassConfigError(f"Class '{name}' has a non-RGB color: {color!r}")
        if class_id in seen_ids:
            raise ClassConfigError(f"Duplicate class id {class_id} in '{resolved_path}'.")
        if name in seen_names:
            raise ClassConfigError(f"Duplicate class name '{name}' in '{resolved_path}'.")
        seen_ids.add(class_id)
        seen_names.add(name)

        classes.append(
            ClassDef(
                id=class_id,
                name=name,
                display_name=display_name,
                color=color,  # type: ignore[arg-type]
                change_eligible=change_eligible,
            )
        )

    return ClassConfig(classes=tuple(classes))


def to_maskforge_palette(config: ClassConfig, palette_name: str = "landscape-change-detection-pipeline") -> dict:
    """Convert a :class:`ClassConfig` into a MaskForge ``ClassPalette`` JSON dict.

    Matches the schema in ``Maskforge/sidecar/maskforge_core/class_config.py``
    (``ClassPalette{id, name, classes: [ClassDef{id, name, color, value,
    active_by_default}]}``). MaskForge's ``ClassDef.id`` is a palette-scoped
    UUID string (distinct from this module's stable integer ``ClassDef.id``,
    which MaskForge stores as ``value``); a fresh UUID is generated per class
    here since round-tripping through MaskForge does not need it to be
    deterministic, only ``value`` (the integer class id) does.
    """
    return {
        "id": str(uuid.uuid4()),
        "name": palette_name,
        "classes": [
            {
                "id": str(uuid.uuid4()),
                "name": c.name,
                "color": list(c.color),
                "value": c.id,
                "active_by_default": True,
            }
            for c in config.classes
        ],
    }


def write_maskforge_palette(
    config: ClassConfig, output_path: str, palette_name: str = "landscape-change-detection-pipeline"
) -> Path:
    """Write the MaskForge palette JSON for ``config`` to ``output_path``."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_maskforge_palette(config, palette_name), indent=2), encoding="utf-8")
    return path
