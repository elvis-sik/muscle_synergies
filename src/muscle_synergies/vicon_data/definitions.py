from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, Union, List, NewType

import pint
import pint_pandas

# a single registry for the physical units used across the program
ureg = pint.UnitRegistry()
pint_pandas.PintType.ureg = ureg

# type variables used to define generic types
T = TypeVar('T')
X = TypeVar('X')
Y = TypeVar('Y')

# a row from the CSV file is simply a list of strings,
# corresponding to different values
Row = NewType('Row', List[str])

# unified types for several different representations of a device header
DeviceHeaderRepresentation = Union['ColOfHeader', 'DeviceHeaderCols',
                                   'DeviceHeaderPair', 'DeviceHeaderData']
ForcePlateRepresentation = Union['ForcePlateDevices', 'ForcePlateData']


class SectionType(Enum):
    """Type of a section of a Vicon Nexus CSV file.

    The files outputted by experiments are split into sections, each containing
    different types of data. The sections are separated by a single blank line.
    Each section contains a header plus several rows of measurements. The header
    spans 5 lines including, among other things, an identification of its type
    See the docs for py:class:ViconCSVLines for a full description of the meaning
    of the different lines.

    Members:
    + FORCES_EMG refers to a section that begins with the single word "Devices"
      and contains measurements from force plates and an EMG device.
    + TRAJECTORIES refers to a section that begins with the single word
      "Trajectories" and contains kinemetric measurements of joint position.
    """
    FORCES_EMG = 1
    TRAJECTORIES = 2


class ViconCSVLines(Enum):
    """Lines in the CSV file with experimental data.

    The members refer to lines in the CSV file.
    + SECTION_TYPE_LINE is the first line in a section, which contains either the
      word "Devices" (for the section with force plate and EMG data) or
      "Trajectories" (for the section with kinematic data).

    + SAMPLING_FREQUENCY_LINE is the second line in a section, which contains a
      single integer representing the sampling frequency.

    + DEVICE_NAMES_LINE is the third line in a section, which contains the names
      of measuring devices (such as "Angelica:HV").

    + COORDINATES_LINE is the fourth line in a section, which contains headers
      like "X", "Y" and "Z" referring to the different coordinates of
      vector-valued measurements made by different devices.

    + UNITS_LINE is the fifth line in a section, which describes the physical
      units of different measurements.

    + DATA_LINE are lines from the sixth until a blank line or EOF is found.
      They represent measurements over time.

    + BLANK_LINE is a line happenening between sections.
    """
    SECTION_TYPE_LINE = 1
    SAMPLING_FREQUENCY_LINE = 2
    DEVICE_NAMES_LINE = 3
    COORDINATES_LINE = 4
    UNITS_LINE = 5
    DATA_LINE = 6
    BLANK_LINE = 7


class DeviceType(Enum):
    """Type of a measurement device.

    Measurement devices are named in the third line of each section of the CSV
    file outputted by Vicon Nexus. Each name is included in a single column, but
    the data for each device in the following lines usually span more than 1
    column:
    + one column per muscle in the case of EMG data.
    + 3 columns (1 per spatial coordinate) in the case of trajectory markers.
    + several columns per force plate (see below).

    Force plates are complicated by the fact that a single one of them is
    represented as several "devices". For example, take a look at the following:
    + Imported AMTI OR6 Series Force Plate #1 - Force
    + Imported AMTI OR6 Series Force Plate #1 - Moment
    + Imported AMTI OR6 Series Force Plate #1 - CoP

    These 3 refer actually to different measurements (force, moment, origin) for
    the same experimental device (Force Plate #1). All of the measured
    quantities are vectors, so each of those 3 "devices" is represented in 3
    columns.

    At any rate, all 3 of the different columns (Force, Moment and CoP) for each
    force plate is treated as if they were actually different devices when they
    are first read by the parsing functions of :py:module:vicon_data, and then
    later they are unified.

    Members:
    + FORCE_PLATE is a force plate. The data

    + EMG is EMG measurement.

    + TRAJECTORY_MARKER is a trajectory marker used with kinemetry.
    """
    FORCE_PLATE = 1
    EMG = 2
    TRAJECTORY_MARKER = 3

    def section_type(self) -> SectionType:
        if self in {DeviceType.EMG, DeviceType.FORCE_PLATE}:
            return SectionType.FORCES_EMG
        return SectionType.TRAJECTORIES


@dataclass
class ForcePlateMeasurement:
    FORCE = 1
    MOMENT = 2
    COP = 3


@dataclass
class ColOfHeader:
    """The string describing a device and the column in which it occurs.

    Args:
        col_index: the index of the column in the CSV file in which the
            device header is described.

        header_str: the exact string occurring in that column.
    """
    col_index: int
    header_str: str
