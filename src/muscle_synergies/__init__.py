"""Determine muscle synergies on the data outputted by Vicon Nexus."""

__version__ = "0.0.1"

from .vicon_data import *

__all__ = ("load_vicon_file",
           "butterworth_filter",
           "find_synergies",
           "plot_emg_signal",
           "plot_spectrum",
           "positive_spectrum",
           "rms_envelope",
           "synergy_heatmap",
)
