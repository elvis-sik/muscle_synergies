"""Load data from the csv outputted by Vicon Nexus."""

import abc
import csv
from dataclasses import dataclass
from enum import Enum
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator)

# a row from the CSV file is simply a list of strings,
# corresponding to different values
Row = List[str]


class SectionType(Enum):
    """Type of a section of a Vicon Nexus CSV file.

    The files outputted by experiments are split into sections, each containing
    different types of data. The sections are separated by a single blank line.
    Each section contains a header plus several rows of measurements. The header
    spans 5 lines including, among other things, an identification of its type
    See the docs for py:class:ReaderState for a full description of the meaning
    of the different lines.

    Members:
    + FORCES_EMG refers to a section that begins with the single word "Devices"
      and contains measurements from force plates and an EMG device.
    + TRAJECTORIES refers to a section that begins with the single word
      "Trajectories" and contains kinemetric measurements of joint position.
    """
    FORCES_EMG = 1
    TRAJECTORIES = 2


class ReaderState(Enum):
    """State of a reader going through a CSV file with experimental data.

    The members refer to successive states in the reader:
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


class DeviceHeaderCols:
    """Intermediate representation of the data read in the device names line.

    This class is used as a way of communicating data from the :py:class:Reader
    to the :py:class:DataBuilder. For more information on where the data that
    is held by this class, see the docs for :py:class:ReaderState.

    Args:
        device_name: the name of the device, as read from the CSV file

        device_type: the type of the device

        first_col_index: the index in a Row in which the data for the device
            begins
    """
    device_type: DeviceType
    device_name: str
    first_col_index: int
    num_of_cols: Optional[int]

    def __init__(self, device_type: DeviceType, device_name: str,
                 first_col_index: int):
        self.device_name = device_name
        self.device_type = device_type
        self.first_col_index: int
        self._initialize_num_of_cols

    def slice(self):
        if self.num_of_cols is None:
            raise TypeError('add_num_of_cols should be called before slice')

        return slice(self.first_col_index, self.first_col_index + num_of_cols)

    def add_num_of_cols(self, num_of_cols: int):
        if self.num_of_cols is not None:
            raise TypeError(
                'tried to set num_of_cols with the variable already set')

        if self.device_type is not DeviceType.EMG:
            raise TypeError(
                "tried to set num_of_cols for a device the type of which isn't EMG"
            )

        self.num_of_cols = num_of_cols

    def _initialize_num_of_cols(self):
        if self.device_type is DeviceType.EMG:
            self.num_of_cols = None
        else:
            self.num_of_cols = 3



class SectionReaderState(abc.ABC):
    # TODO Idea is to obsolete TestViconNexusCSVReader
    # include all check/read/etc functions as State methods.
    # The BLANK_LINE state can have a side effect so that
    # the second time it is their turn to act
    # they instead end the whole process.

    # Also, the idea is for the DataBuilder, in turn, to also be
    # stateful. In particular, it should have 2 states, corresponding
    # to the 2 section types.

    # TODO could rename all funcs below if want to
    # should add signature to them
    def feed_row():
        pass

    def _get_validator():
        pass

    def _get_data_builder():
        pass

    def _set_new_state():
        pass

    @abc.abstractmethod
    def _check_row():
        pass

    @abc.abstractmethod
    def _parse_row():
        pass

    @abc.abstractmethod
    def _change_state():
        pass


class SectionReader:
    """Reader for a single section of the CSV file outputted by Vicon Nexus.

    Initialize it
    """
    state: ReaderState
    data_builder: 'DataBuilder'
    validator: 'Validator'

    def __init__(self, section_data_builder: 'DataBuilder',
                 validator: 'Validator'):
        self.state = ReaderState.SECTION_TYPE_LINE
        self.data_builder = section_data_builder
        self.validator = validator

    def feed_line(self, row: Row):
        self._raise_if_ended()

        check_function = self._get_check_function()
        read_function = self._get_read_function()
        aggregation_function = self._get_build_function()

        self._call_validator(check_function(row))
        read_line = read_function(row)
        aggregation_function(read_line)

        self._transition()

    def _get_check_function(self):
        if self.state == ReaderState.SECTION_TYPE_LINE:
            return check_section_type_line
        elif self.state == ReaderState.SAMPLING_FREQUENCY_LINE:
            return check_sampling_frequency_line
        elif self.state == ReaderState.DEVICE_NAMES_LINE:
            return self.devices_reader.check_device_names_line

    def _get_read_function(self):
        if self.state == ReaderState.SECTION_TYPE_LINE:
            return read_section_type_line
        elif self.state == ReaderState.SAMPLING_FREQUENCY_LINE:
            return read_sampling_frequency_line
        elif self.state == ReaderState.DEVICE_NAMES_LINE:
            return self.devices_reader.read_device_names_line

    def _get_build_function(self):
        if self.state == ReaderState.SECTION_TYPE_LINE:
            return self.data_builder.add_section_type
        elif self.state == ReaderState.SAMPLING_FREQUENCY_LINE:
            return self.data_builder.add_frequency
        elif self.state == ReaderState.DEVICE_NAMES_LINE:
            # TODO Essa é a próxima linha
            return

    def _transition(self):
        if self.state == ReaderState.SECTION_TYPE_LINE:
            self.state = ReaderState.SAMPLING_FREQUENCY_LINE
        elif self.state == ReaderState.SAMPLING_FREQUENCY_LINE:
            self.state = ReaderState.DEVICE_NAMES_LINE
        elif self.state == ReaderState.DEVICE_NAMES_LINE:
            return

    def _raise_if_ended(self):
        if self.state == ReaderState.BLANK_LINE:
            raise EOFError(
                'tried to read another line from a section that has already been completely readd'
            )

    def _call_validator(self, data_check_result: 'DataCheck'):
        self.validator.validate(data_check_result)


class SectionDataBuilder:
    section_type: SectionType
    frequency: int

    def add_section_type(self, section_type: SectionType):
        self.section_type = section_type

    def add_frequency(self, frequency: int):
        self.frequency = frequency


class AllDevicesDataBuilder:
    pass


class DeviceHeaderDataBuilder:
    pass


class TimeSeriesDataBuilder:
    pass


# A data check is a dict representing the result of validating the data of a
# single line of the CSV data. Its keys:
#   'is_valid' maps to a bool that answers
#
#   'error_message' maps to a str containing an error message to be
#   displayed in case there are problems with the data. The str is appended to
#   the prefix "error parsing line {line_number} of file {filename}: "
DataCheck = Mapping[str, Any]


class Validator:
    current_line: int
    csv_filename: str
    should_raise: bool = True

    def __init__(self, csv_filename: str, should_raise: bool = True):
        self.current_line = 1
        self.csv_filename = csv_filename
        self.should_raise = should_raise

    def validate(self, data_check_result: DataCheck):
        if self.should_raise:
            self._raise_if_invalid(data_check_result)

        self.current_line += 1

    def _raise_if_invalid(self, data_check_result: DataCheck):
        is_valid = data_check_result['is_valid']
        error_description = data_check_result['error_message']

        if not is_valid:
            raise ValueError(self._build_error_message(error_description))

    def _build_error_message(self, error_description: str) -> str:
        return f'error parsing line {self.current_line} of file {self.csv_filename}: {error_message}'


def check_section_type_line(row: Row) -> DataCheck:
    has_acceptable_value = row[0] in {'Devices', 'Trajectories'}
    is_valid = has_acceptable_value and columns_are_empty(row[1:])
    message = ('this line should contain either "Devices" or "Trajectories"'
               ' in its first column and nothing else')
    return {'is_valid': is_valid, 'error_message': message}


def read_section_type_line(row: Row) -> SectionType:
    section_type_str = row[0]
    str_to_enum_mapping = {
        'Devices': SectionType.FORCES_EMG,
        'Trajectories': SectionType.TRAJECTORIES
    }

    return str_to_enum_mapping[section_type_str]


def check_sampling_frequency_line(row: Row) -> DataCheck:
    try:
        is_valid = int(row[0])
    except ValueError:
        is_valid = False

    is_valid = is_valid and columns_are_empty(row[1:])
    message = ('this line should contain an integer representing'
               ' sampling frequency in its first column and nothing else')
    return {'is_valid': is_valid, 'error_message': message}


def read_sampling_frequency_line(row: Row) -> int:
    return int(row[0])


class DevicesReader:
    """Reader for device names, coordinates and units lines of the CSV file."""
    dev_names: List[str]
    section_type: SectionType

    def add_section_type(self, section_type: SectionType):
        self.section_type = section_type

    def read_device_names_line(self, row: Row) -> List[str]:
        names_str = [row[i] for i in range(2, len(row), 3)]
        self.dev_names = dev_names
        return names_str

    def check_device_names_line(row: Row) -> DataCheck:
        def col_should_contain_name(col_num):
            return (col_num - 2) % 3 == 0

        device_names_line_checks = (
            self._check_device_names_line_has_section_type,
            self._check_device_names_line_blanks_and_names,
            self._check_device_names_line_consistent_with_section_type)

        for check in device_names_line_checks:
            check_result = check(row)

            if not check_result['is_valid']:
                return check_result

        return {'is_valid': True, 'error_message': 'no error'}

    def _check_device_names_line_has_section_type(self, row: Row) -> DataCheck:
        return {
            'is_valid':
            self.section_type in SectionType,
            'error_message':
            'tried to read device names without first calling add_section_type'
        }

    def _check_device_names_line_blanks_and_names(self, row: Row) -> DataCheck:
        is_valid = not row[0] and not row[1]

        for col_num, col_val in enumerate(row[2:], start=2):
            if col_should_contain_name(col_num):
                current_is_correct = col_val
            else:
                current_is_correct = not col_val

            is_valid = is_valid and current_is_correct

        message = ('this line should contain two blank columns '
                   'then one device name every 3 columns')
        return {'is_valid': is_valid, 'error_message': message}

    def _check_device_names_line_consistent_with_section_type(self, row: Row
                                                              ) -> DataCheck:
        device_names = self.read_device_names_line(row)

        if self.section_type is SectionType.FORCES_EMG:
            force_plates_until_second_last = all('Force Plate' in name
                                                 for name in row[:-1])
            last_is_emg = 'EMG' in row[-1]

            is_valid = force_plates_until_second_last and last_is_emg
            error_message = (
                'since this section began with "Devices" two lines above, '
                'expected to see a series of "Force Plate" devices and then an '
                'EMG one.')

        elif self.section_type is SectionType.TRAJECTORIES:
            no_force_plates = all('Force Plate' not in name for name in row)
            no_emg = all('EMG' not in name for name in row)

            is_valid = no_force_plates and no_emg
            error_message = (
                'since this secton began with "Trajectories" two lines above, '
                'expected to not see "Force Plate" and "EMG"')

        return {'is_valid': is_valid, 'error_message': error_message}


class ViconNexusCSVReader:
    pass


def csv_lines_stream(filename) -> Iterator[Row]:
    """Yields lines from a CSV file as a stream.

    Args:
        filename: the name of the CSV file which should be read. This argument
                  is passed to :py:func:open, so it can be a str among other
                  things.
    """
    with open(filename) as csvfile:
        data_reader = csv.reader(csvfile)
        yield from data_reader


def initialize_vicon_nexus_reader():
    pass


def load_vicon_file():
    pass
