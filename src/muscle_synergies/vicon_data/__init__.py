"""Load the data outputted by the Vicon Nexus machine.

Users of this package should mainly be concerned with the high-level function
:py:func:`~muscle_synergies.vicon_data.load_vicon_file`.

To provide an overview of how the internals work, that function reads the CSV
file line-by-line and feeds it to
:py:class:`~muscle_synergies.vicon_data.reader.Reader`, which parses the lines
and send the data to
:py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator`.  `Aggregator`
stores it as the file is being parsed.  When the file ends,
:py:class:`~muscle_synergies.vicon_data.user_data.Builder` is used to get a
:py:class:`~muscle_synergies.vicon_data.ViconNexusData` from it that is then
delivered to the user.
"""

from .aggregator import *
from .load_csv import *
from .reader import *
from .user_data import *

__all__ = (
    "load_vicon_file",
    "ViconNexusData",
    "DeviceData",
)
