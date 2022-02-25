"""Basic types used to parse Vicon CSV file.

These double as a way to define a vocabulary to talk about the different parts
of the file.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, NewType, TypeVar

# type variables used to define generic types
# pylint: disable=invalid-name
T = TypeVar("T")
X = TypeVar("X")
Y = TypeVar("Y")
# pylint: enable=invalid-name


Row = NewType("Row", List[str])
"""a row from the CSV file is simply a list of strings, corresponding to
different values"""


class SectionType(Enum):
    """Type of a section of a Vicon Nexus CSV file.

    The files outputted by experiments are split into sections, each containing
    different types of data. The sections are separated by a single blank line.
    Each section contains a header plus several rows of measurements. The
    header spans 5 lines including, among other things, an identification of
    its type. See the docs for :py:class:`ViconCSVLines` for a full description
    of the meaning of the different lines.
    """

    FORCES_EMG = 1
    """a section that begins with the single word `Devices` and
    contains measurements from force plates and an EMG device.
    """

    TRAJECTORIES = 2
    """a section that begins with the single word `Trajectories` and
    contains kinemetric measurements of joint position.
    """


class ViconCSVLines(Enum):
    """Lines in the CSV file with experimental data.

    The members refer to lines in the CSV file.
    """

    SECTION_TYPE_LINE = 1
    """the first line in a section, which contains either the word `Devices`
    for the section with force plate and EMG data) or `Trajectories` for the
    section with kinematic data.
    """

    SAMPLING_FREQUENCY_LINE = 2
    """the second line in a section, which contains a single integer
    representing the sampling frequency.
    """

    DEVICE_NAMES_LINE = 3
    """the third line in a section, which contains the names of measuring
    devices (such as `Angelica:HV`). The exact strings occurring in this line
    are referred to as "device headers". A single force plate can be
    represented as several device headers.
    """

    COORDINATES_LINE = 4
    """the fourth line in a section, which contains headers like `X`, `Y` and
    `Z` referring to the different coordinates of vector-valued measurements
    made by different devices.
    """

    UNITS_LINE = 5
    """the fifth line in a section, which describes the physical units of
    different measurements.
    """

    DATA_LINE = 6
    """lines from the sixth one until a blank line or EOF is found. They
    represent measurements over time.
    """

    BLANK_LINE = 7
    """a blank line that occurs between sections."""


class DeviceType(Enum):
    """Type of a measurement device.

    Measurement devices are named in the third line of each section of the CSV
    file outputted by Vicon Nexus. Each of the nonempty strings occurring in
    that line is referred to as a "device header". Each device header by
    definition spans a single column, but the data for each device in the
    following lines usually span more than 1 column:

    + one column per muscle in the case of EMG data.
    + 3 columns (1 per spatial coordinate) in the case of trajectory markers.
    + 9 columns per force plate (see :py:class:`ForcePlateMeasurement`).
    """

    FORCE_PLATE = 1
    """a force plate."""

    EMG = 2
    """EMG measurements."""

    TRAJECTORY_MARKER = 3
    """a trajectory marker."""

    @staticmethod
    def from_str(device_type: str) -> "DeviceType":
        """Parse string returning a DeviceType object.

        Args:
            device_type: one of `"emg"`, `"traj"` (or `"marker"`) or
                `"forcepl"` (or `"fp"` or `"force plate"`). Case is ignored.
        """
        if device_type.upper() == "EMG":
            return DeviceType.EMG
        if device_type.upper() in {"FORCE PLATE", "FP", "FORCEPL"}:
            return DeviceType.FORCE_PLATE
        if device_type.upper() in {"TRAJ", "MARKER"}:
            return DeviceType.TRAJECTORY_MARKER
        raise ValueError(f"device type not understood: {device_type}")

    def section_type(self) -> SectionType:
        """Section type in which device occurs"""
        if self in {DeviceType.EMG, DeviceType.FORCE_PLATE}:
            return SectionType.FORCES_EMG
        return SectionType.TRAJECTORIES


class ForcePlateMeasurement(Enum):
    """The type of a measurement from a force plate.

    If we define a "measurement device" as a non-blank entry in the
    `DevicesLine` (see :py:class:`ViconCSVLines`), a single force plate is
    actually represented as 3 different device headers, for example:

    + `Imported AMTI OR6 Series Force Plate #1 - Force`
    + `Imported AMTI OR6 Series Force Plate #1 - Moment`
    + `Imported AMTI OR6 Series Force Plate #1 - CoP`

    These 3 clearly refer actually to different measurements (force, moment,
    origin) for the same experimental device (Force Plate #1). All of the
    measured quantities are spatial vectors, so each of those 3 "devices" is
    represented in 3 columns. So a force plate will in total contain 9 columns
    of data.
    """

    FORCE = 1
    """the forces measured by a force plate."""

    MOMENT = 2
    """the moment measured by a force plate."""

    COP = 3
    """the origin of the quantities measured by a force plate."""


@dataclass
class SamplingFreq:
    """The sampling rates of different measurement devices.

    All of the devices in each section (see :py:class:`SectionType`) have the
    same sampling rate but the sampling rates may differ between sections. For
    that reason, the i-th data line of the `FORCES_EMG` section will not have
    been measured at the same time as the i-th data line of the `TRAJECTORIES`
    section. In order to create a unified way of referring to both types of
    measurement, "frames" and "subframes" are used.

    Each data line in the `TRAJECTORIES` section correspond to a different
    frame, all of which occur at subframe 0. In the case of the `FORCES_EMG`
    section, however, each sequence of `num_subframes` measurements correspond
    to a single frame. By hypothesis, `freq_forces_emg` is greater than
    `freq_traj` and `num_subframes` is the ratio between the 2.

    Args:
        freq_forces_emg: the sampling rate of the `FORCES_EMG` section.

        freq_traj: the sampling rate of the `TRAJECTORIES` section.

        num_frames: the number of frames.

    Attributes
        num_subframes: the number of subframes contained in every frame.
    """

    freq_forces_emg: int
    freq_traj: int
    num_frames: int

    @property
    def num_subframes(self) -> int:
        num_subframes = self.freq_forces_emg / self.freq_traj
        assert num_subframes == int(num_subframes)
        return int(num_subframes)
