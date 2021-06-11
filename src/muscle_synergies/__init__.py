"""Determine muscle synergies on the data outputted by Vicon Nexus."""

__version__ = "0.0.1"

from .vicon_data import *
from .analysis import *

__all__ = (
    "load_vicon_file",
    "plot_signal",
    "synergy_heatmap",
    "plot_fft",
    "fft_spectrum",
    "zero_center",
    "linear_envelope",
    "digital_filter",
    "rms",
    "normalize",
    "subsample",
    "time_normalize",
    "vaf",
    "find_synergies",
)
