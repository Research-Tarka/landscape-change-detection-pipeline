"""Per-sensor band tables for scene fetch: which bands, and how each one
reaches the pipeline's uniform 10 m reference grid.

Purpose
-------
:mod:`.sensors` already knows each sensor's collection id and one flat
``native_resolution_m``. This module adds the finer-grained per-*band*
picture needed: not every band of every sensor is natively at the working
resolution, so each band is tagged with a :class:`BandRole` describing how
it reaches the grid.

Working resolution
-------------------
Every sensor -- including Landsat 5 -- shares one project-wide working
resolution, ``10.0`` m, pinned to the DEM's own grid (not just "some 10 m
grid": the exact transform/shape of the tile's stored DEM, see
``dem/zarr_store.py::read_tile_dem`` and
``docs/decisions/unified_10m_grid.md``). 10 m is Sentinel-2's native
ceiling -- nothing in the stack genuinely resolves finer than that -- so
this is the best achievable *uniform* resolution, not an arbitrary choice.

There is no panchromatic band anywhere in this table
(:class:`BandRole.PAN`/``PANSHARPEN`` have been removed): even though the
Landsat TOA collections used here (``docs/decisions/toa_rollback.md``) do
carry a 15 m panchromatic band, this pipeline deliberately does not
pansharpen against it -- every sensor, Landsat included, stays on the one
uniform 10 m grid pinned to the DEM via plain bilinear resample, a decision
kept unchanged across the SR->TOA rollback (see
``docs/decisions/unified_10m_grid.md``). Every Landsat band is genuinely
~30 m information resampled (bilinear) onto the finer 10 m grid -- real
detail is not being invented, only alignment; this is a resolution-*semantics*
caveat, never to be read as true 10 m resolving power for any
Landsat-derived pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BandRole(str, Enum):
    """How one band reaches the pipeline's uniform 10 m reference grid."""

    #: Already at 10 m; stored unchanged (Sentinel-2's four 10 m bands only).
    NATIVE = "native"
    #: Coarser than 10 m; brought up via plain bilinear resampling onto the
    #: DEM's exact grid (alignment only, never a detail gain).
    RESAMPLE = "resample"


@dataclass(frozen=True)
class Band:
    """One spectral band as fetched from Earth Engine."""

    name: str  # EE band name, e.g. "B4"
    label: str  # human-readable, e.g. "red"
    native_resolution_m: float
    role: BandRole


@dataclass(frozen=True)
class SensorBandSpec:
    """The full band table for one sensor, plus its working resolution."""

    sensor_key: str
    working_resolution_m: float
    bands: tuple[Band, ...]
    #: RGB band names (in R, G, B order) for visualization composites.
    rgb_bands: tuple[str, str, str] = ()

    def band(self, name: str) -> Band:
        for b in self.bands:
            if b.name == name:
                return b
        raise KeyError(f"Band '{name}' is not defined for sensor '{self.sensor_key}'")

    def toa_band_names(self) -> tuple[str, ...]:
        """Band names stored in the reflectance array."""
        return tuple(b.name for b in self.bands)

    def resample_band_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bands if b.role == BandRole.RESAMPLE)


# ---------------------------------------------------------------------------
# Landsat 5 TM / 7 ETM+ / 8 OLI / 9 OLI-2: Collection 2 Tier 1 TOA. All bands
# native 30 m, resampled (bilinear) onto the DEM's exact 10 m grid -- the 15 m
# panchromatic band (B8, L7/L8/L9 only) exists in the TOA product but is
# deliberately not fetched/used (no pansharpening, see module docstring).
# Band-name layout differs between L5/L7 (TM/ETM+) and L8/L9 (OLI): the
# TM/ETM+ blue/green/red/nir/swir1/swir2 bands are B1-B5,B7; OLI's are B2-B7.
# ---------------------------------------------------------------------------
L5_BANDS = SensorBandSpec(
    sensor_key="L5",
    working_resolution_m=10.0,
    bands=(
        Band("B1", "blue", 30.0, BandRole.RESAMPLE),
        Band("B2", "green", 30.0, BandRole.RESAMPLE),
        Band("B3", "red", 30.0, BandRole.RESAMPLE),
        Band("B4", "nir", 30.0, BandRole.RESAMPLE),
        Band("B5", "swir1", 30.0, BandRole.RESAMPLE),
        Band("B7", "swir2", 30.0, BandRole.RESAMPLE),
    ),
    rgb_bands=("B3", "B2", "B1"),
)

L7_BANDS = SensorBandSpec(
    sensor_key="L7",
    working_resolution_m=10.0,
    bands=(
        Band("B1", "blue", 30.0, BandRole.RESAMPLE),
        Band("B2", "green", 30.0, BandRole.RESAMPLE),
        Band("B3", "red", 30.0, BandRole.RESAMPLE),
        Band("B4", "nir", 30.0, BandRole.RESAMPLE),
        Band("B5", "swir1", 30.0, BandRole.RESAMPLE),
        Band("B7", "swir2", 30.0, BandRole.RESAMPLE),
    ),
    rgb_bands=("B3", "B2", "B1"),
)

L8_BANDS = SensorBandSpec(
    sensor_key="L8",
    working_resolution_m=10.0,
    bands=(
        Band("B2", "blue", 30.0, BandRole.RESAMPLE),
        Band("B3", "green", 30.0, BandRole.RESAMPLE),
        Band("B4", "red", 30.0, BandRole.RESAMPLE),
        Band("B5", "nir", 30.0, BandRole.RESAMPLE),
        Band("B6", "swir1", 30.0, BandRole.RESAMPLE),
        Band("B7", "swir2", 30.0, BandRole.RESAMPLE),
    ),
    rgb_bands=("B4", "B3", "B2"),
)

L9_BANDS = SensorBandSpec(
    sensor_key="L9",
    working_resolution_m=10.0,
    bands=(
        Band("B2", "blue", 30.0, BandRole.RESAMPLE),
        Band("B3", "green", 30.0, BandRole.RESAMPLE),
        Band("B4", "red", 30.0, BandRole.RESAMPLE),
        Band("B5", "nir", 30.0, BandRole.RESAMPLE),
        Band("B6", "swir1", 30.0, BandRole.RESAMPLE),
        Band("B7", "swir2", 30.0, BandRole.RESAMPLE),
    ),
    rgb_bands=("B4", "B3", "B2"),
)

# ---------------------------------------------------------------------------
# Sentinel-2 MSI (COPERNICUS/S2_SR_HARMONIZED): working resolution 10 m.
# Band names are unchanged from the TOA collection (still B1...B12/B8A).
# 10 m bands stored unchanged; the six 20 m red-edge/SWIR bands are
# resampled to 10 m (not DSen2-super-resolved).
# ---------------------------------------------------------------------------
S2_BANDS = SensorBandSpec(
    sensor_key="S2",
    working_resolution_m=10.0,
    bands=(
        Band("B2", "blue", 10.0, BandRole.NATIVE),
        Band("B3", "green", 10.0, BandRole.NATIVE),
        Band("B4", "red", 10.0, BandRole.NATIVE),
        Band("B8", "nir", 10.0, BandRole.NATIVE),
        Band("B5", "red_edge1", 20.0, BandRole.RESAMPLE),
        Band("B6", "red_edge2", 20.0, BandRole.RESAMPLE),
        Band("B7", "red_edge3", 20.0, BandRole.RESAMPLE),
        Band("B8A", "red_edge4", 20.0, BandRole.RESAMPLE),
        Band("B11", "swir1", 20.0, BandRole.RESAMPLE),
        Band("B12", "swir2", 20.0, BandRole.RESAMPLE),
    ),
    rgb_bands=("B4", "B3", "B2"),
)

SENSOR_BAND_SPECS: dict[str, SensorBandSpec] = {
    "L5": L5_BANDS,
    "L7": L7_BANDS,
    "L8": L8_BANDS,
    "L9": L9_BANDS,
    "S2": S2_BANDS,
}


def get_band_spec(sensor_key: str) -> SensorBandSpec:
    """Look up a sensor's band table, with a helpful error listing valid keys."""
    try:
        return SENSOR_BAND_SPECS[sensor_key.upper()]
    except KeyError:
        raise KeyError(
            f"Unknown sensor '{sensor_key}'. Valid sensors: {', '.join(SENSOR_BAND_SPECS)}"
        ) from None
