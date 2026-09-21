"""Change detection: analysis-only spectral indices, break detection, and combined change layers.

Nothing here feeds the land-cover model (:mod:`landscape_change_detection_pipeline.models`,
:mod:`landscape_change_detection_pipeline.features`) -- nothing in this package is
read by training or inference. It exists purely for the change-detection
layers built on top of an already-trained classification (NDVI regrowth,
dNBR, CCDC/LandTrendr breaks, BFAST fire cross-check, combined change maps).
"""
