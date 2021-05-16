"""Basic types used to parse Vicon CSV file.

These double as a way to define a vocabulary to talk about the different parts
of the file.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, NewType, TypeVar

# type variables used to define generic types
T = TypeVar("T")
X = TypeVar("X")
Y = TypeVar("Y")

# a row from the CSV file is simply a list of strings,
# corresponding to different values
Row = NewType("Row", List[str])


class SectionType(Enum):
    """Type of a section of a Vicon Nexus CSV file.

    The files outputted by experiments are split into sections, each containing
    different types of data. The sections are separated by a single blank line.
    Each section contains a header plus several rows of measurements. The
    header spans 5 lines including, among other things, an identification of
    its type See the docs for :py:class:`ViconCSVLines` for a full description
    of the meaning of the different lines.
    """

    FORCES_EMG = (
        "refers to a section that begins with the single word "
        + '"Devices" and contains measurements from force plates and an EMG '
        + "device."
    )
    TRAJECTORIES = (
        'refers to a section that begins with the single word "Trajectories" '
        + "and contains kinemetric measurements of joint position."
    )


class ViconCSVLines(Enum):
    """Lines in the CSV file with experimental data.

    The members refer to lines in the CSV file.
    """

    SECTION_TYPE_LINE = (
        'the first line in a section, which contains either the word "Devices" '
        + '(for the section with force plate and EMG data) or "Trajectories" '
        + "(for the section with kinematic data)."
    )
    SAMPLING_FREQUENCY_LINE = (
        "the second line in a section, which contains a single integer "
        + "representing the sampling frequency."
    )
    DEVICE_NAMES_LINE = (
        "the third line in a section, which contains the "
        + 'names of measuring devices (such as "Angelica:HV").'
    )

    COORDINATES_LINE = (
        "the fourth line in a section, which contains headers "
        + 'like "X", "Y" and "Z" referring to the different coordinates of '
        + "vector-valued measurements made by different devices."
    )
    UNITS_LINE = (
        "the fifth line in a section, which describes the physical"
        + "units of different measurements."
    )
    DATA_LINE = (
        "lines from the sixth one until a blank line or EOF is found. "
        + "They represent measurements over time."
    )
    BLANK_LINE = "a blank line that occurs between sections."


class DeviceType(Enum):
    """Type of a measurement device.

    Measurement devices are named in the third line of each section of the CSV
    file outputted by Vicon Nexus. Each name is included in a single column,
    but the data for each device in the following lines usually span more than
    1 column:

    + one column per muscle in the case of EMG data.
    + 3 columns (1 per spatial coordinate) in the case of trajectory markers.
    + 9 columns per force plate (see :py:class:`ForcePlateMeasurement`).
    """

    FORCE_PLATE = "a force plate."
    EMG = "EMG measurement."
    TRAJECTORY_MARKER = "a trajectory marker."

    def section_type(self) -> SectionType:
        if self in {DeviceType.EMG, DeviceType.FORCE_PLATE}:
            return SectionType.FORCES_EMG
        return SectionType.TRAJECTORIES


@dataclass
class ForcePlateMeasurement:
    """The type of a measurement from a force plate.

    If we define a "measurement device" as a non-blank entry in the
    `DevicesLine` (see :py:class:`ViconCSVLines`), a single force plate is
    actually represented as 3 different devices, for example:

    + Imported AMTI OR6 Series Force Plate #1 - Force
    + Imported AMTI OR6 Series Force Plate #1 - Moment
    + Imported AMTI OR6 Series Force Plate #1 - CoP

    These 3 clearly refer actually to different measurements (force, moment,
    origin) for the same experimental device (Force Plate #1). All of the
    measured quantities are spatial vectors, so each of those 3 "devices" is
    represented in 3 columns. So a force plate will in total contain 9 columns
    of data.
    """

    FORCE = "the forces measured by a force plate."
    MOMENT = "the moment measured by a force plate."
    COP = "the origin of the quantities measured by a force plate."


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
