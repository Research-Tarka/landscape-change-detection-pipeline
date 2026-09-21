"""Per-sensor band tables for scene fetch: which bands, at what native
resolution, and how each one reaches its sensor's uniform working grid.

Purpose
-------
:mod:`.sensors` already knows each sensor's collection id and one flat
``native_resolution_m``. This module adds the finer-grained per-*band*
picture needed: not every band of Landsat 7/8/9 or Sentinel-2 is
natively at that sensor's working resolution, so each band is tagged with a
:class:`BandRole` describing how it gets there (Sentinel-2's 20 m bands are
resampled, not DSen2-super-resolved, per the project lead's call).

Working resolution per sensor
------------------------------
=========  ==================  ========================================
Sensor     Working resolution  Notes
=========  ==================  ========================================
L5         30 m                native only, no pan band, never enhanced
L7/L8/L9   15 m                pan-sharpened RGB+NIR, resampled SWIR
S2         10 m                native 10 m bands, resampled 20 m bands
=========  ==================  ========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BandRole(str, Enum):
    """How one band reaches its sensor's uniform working-resolution grid."""

    #: Already at the sensor's working resolution; stored unchanged.
    NATIVE = "native"
    #: The panchromatic band itself (Landsat 7/8/9 only) -- not stored as a
    #: TOA channel, only used as the pansharpening reference.
    PAN = "pan"
    #: Coarser than the working resolution; sharpened against the pan band
    #: (Gram-Schmidt Adaptive for the analytic stack, Brovey for RGB visuals).
    PANSHARPEN = "pansharpen"
    #: Coarser than the working resolution; brought up via plain
    #: bilinear/cubic resampling only (alignment, not detail gain).
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
    pan_band: str | None = None  # EE band name of the pan band, if any
    #: RGB band names (in R, G, B order) for visualization composites.
    rgb_bands: tuple[str, str, str] = ()

    def band(self, name: str) -> Band:
        for b in self.bands:
            if b.name == name:
                return b
        raise KeyError(f"Band '{name}' is not defined for sensor '{self.sensor_key}'")

    def toa_band_names(self) -> tuple[str, ...]:
        """Band names actually stored in the TOA array (excludes the pan band)."""
        return tuple(b.name for b in self.bands if b.role != BandRole.PAN)

    def pansharpen_band_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bands if b.role == BandRole.PANSHARPEN)

    def resample_band_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.bands if b.role == BandRole.RESAMPLE)


# ---------------------------------------------------------------------------
# Landsat 5 TM: native 30 m throughout, no pan band, never enhanced.
# ---------------------------------------------------------------------------
L5_BANDS = SensorBandSpec(
    sensor_key="L5",
    working_resolution_m=30.0,
    bands=(
        Band("B1", "blue", 30.0, BandRole.NATIVE),
        Band("B2", "green", 30.0, BandRole.NATIVE),
        Band("B3", "red", 30.0, BandRole.NATIVE),
        Band("B4", "nir", 30.0, BandRole.NATIVE),
        Band("B5", "swir1", 30.0, BandRole.NATIVE),
        Band("B7", "swir2", 30.0, BandRole.NATIVE),
    ),
    rgb_bands=("B3", "B2", "B1"),
)

# ---------------------------------------------------------------------------
# Landsat 7 ETM+ / 8 OLI / 9 OLI-2: working resolution 15 m via the 15 m pan
# band. Band-name layout differs between L7 (ETM+) and L8/L9 (OLI): L7's
# blue/green/red/NIR are B1-B4, its pan band is B8; OLI's are B2-B5, pan B8.
# ---------------------------------------------------------------------------
L7_BANDS = SensorBandSpec(
    sensor_key="L7",
    working_resolution_m=15.0,
    pan_band="B8",
    bands=(
        Band("B1", "blue", 30.0, BandRole.PANSHARPEN),
        Band("B2", "green", 30.0, BandRole.PANSHARPEN),
        Band("B3", "red", 30.0, BandRole.PANSHARPEN),
        Band("B4", "nir", 30.0, BandRole.PANSHARPEN),
        Band("B5", "swir1", 30.0, BandRole.RESAMPLE),
        Band("B7", "swir2", 30.0, BandRole.RESAMPLE),
        Band("B8", "pan", 15.0, BandRole.PAN),
    ),
    rgb_bands=("B3", "B2", "B1"),
)

L8_BANDS = SensorBandSpec(
    sensor_key="L8",
    working_resolution_m=15.0,
    pan_band="B8",
    bands=(
        Band("B2", "blue", 30.0, BandRole.PANSHARPEN),
        Band("B3", "green", 30.0, BandRole.PANSHARPEN),
        Band("B4", "red", 30.0, BandRole.PANSHARPEN),
        Band("B5", "nir", 30.0, BandRole.PANSHARPEN),
        Band("B6", "swir1", 30.0, BandRole.RESAMPLE),
        Band("B7", "swir2", 30.0, BandRole.RESAMPLE),
        Band("B8", "pan", 15.0, BandRole.PAN),
    ),
    rgb_bands=("B4", "B3", "B2"),
)

L9_BANDS = SensorBandSpec(
    sensor_key="L9",
    working_resolution_m=15.0,
    pan_band="B8",
    bands=(
        Band("B2", "blue", 30.0, BandRole.PANSHARPEN),
        Band("B3", "green", 30.0, BandRole.PANSHARPEN),
        Band("B4", "red", 30.0, BandRole.PANSHARPEN),
        Band("B5", "nir", 30.0, BandRole.PANSHARPEN),
        Band("B6", "swir1", 30.0, BandRole.RESAMPLE),
        Band("B7", "swir2", 30.0, BandRole.RESAMPLE),
        Band("B8", "pan", 15.0, BandRole.PAN),
    ),
    rgb_bands=("B4", "B3", "B2"),
)

# ---------------------------------------------------------------------------
# Sentinel-2 MSI: working resolution 10 m. 10 m bands stored unchanged; the
# six 20 m red-edge/SWIR bands are resampled to 10 m (not DSen2-super-resolved).
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
