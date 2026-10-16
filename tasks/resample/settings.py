# ============================================================================
# tasks/resample/settings.py — resample a volume to an isotropic voxel size
# ============================================================================

DEFAULT_VOXEL_MM = 1.0
MIN_VOXEL_MM, MAX_VOXEL_MM = 0.1, 10.0

# Interpolation choices; "Auto" picks by content (service.auto_method).
METHODS = ("Auto", "Nearest", "Linear", "B-spline")
DEFAULT_METHOD = "Auto"

# Auto treats a volume as LABELS (-> nearest neighbour) when its data type is
# integer and it holds at most this many distinct values; anything else is an
# image (-> linear, anti-aliased when the voxels grow).
MAX_LABELS = 256

# Reorient to RAS (as isocond loads images) unless the user unticks it.
DEFAULT_REORIENT_RAS = True

# The preview: this many axial slices of the result, evenly spread over the
# slices that carry signal, in one row.
MOSAIC_SLICES = 5
MOSAIC_SLICE_INCHES = 2.4          # width of one slice in the figure
MOSAIC_DPI = 100
WINDOW_PERCENTILES = (1.0, 99.0)   # grey window of an image (labels: own colours)
