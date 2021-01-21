"""Load data from the csv outputted by Vicon Nexus."""

import abc
import csv
from enum import Enum
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping)


class SingleDeviceDataBuilder:
    def add_name(self):
        pass


class ForcePlateDataBuilder:
    # TODO "ForcePlate" might be a misnomer since this is
    # very similar to the Angelica machines

    # TODO basic idea with this class is for it to propagate down
    # each piece of info to each of the VectorBuilder classes
    # might need to rethink

    # go back to how the parser works (split row into parts for each machine)
    # and try to figure it out
    def __init__(self, force_data_builder: SingleDeviceDataBuilder,
                 moment_data_builder: SingleDeviceDataBuilder,
                 origin_data_builder: SingleDeviceDataBuilder):
        pass

    def add_name(self):
        # TODO machine name
        pass

    def add_sampling_frequency(self):
        # TODO probably rename this
        # TODO idea is to let the builder know how many subframes per frame
        # there are
        pass

    def add_units(self):
        pass


class ViconOutputData:
    pass


# csvfile = open('../../tests/test_data.csv')
# data_reader = csv.reader(csvfile)

# a row from the CSV file is simply a list of strings,
# corresponding to different values
Row = List[str]


class LineByLineParser(abc.ABC):
    """A parser that is fed one line (a list of columns) at a time."""
    def __init__(self, check_data: bool = True):
        self.current_line = 0
        self.check_data = check_data

    def feed(self, row: Row):
        """Feed one line to the parser."""
        self.current_line += 1
        self.check(row)
        feed_function = self.get_current_feed_function()
        feed_function(row)

    def check(self, row: Row):
        if not self.check_data:
            return

        check_function = self.get_current_check_function()
        check_result = check_function(row)
        is_valid = check_result['is_valid']
        error_message = check_result['error_message']

        if not is_valid:
            raise ValueError(
                f'error in line {self.current_line}: {error_message}')

    @abc.abstractmethod
    def get_current_feed_function(self) -> Callable[[Row], None]:
        pass

    @abc.abstractmethod
    def get_current_check_function(self) -> Callable[[Row], bool]:
        pass


class ViconCSVSectionParser(LineByLineParser):
    """Parses one of the sections of CSV file outputted by Vicon Nexus.

    Suppose an experiment generates data for all of the following:
    + trajectories (using kinemetry)
    + floor reactions (using force plates)
    + muscle activation (EMG)

    Then the CSV file will have 2 big sections:
    1. force plate + EMG data
    2. trajectory data

    Each of those 2 sections have a very similar structure, so this class
    provides functionality common to both of them.
    """
    def __init__(self,
                 header_parser: 'SectionHeaderParser',
                 measurement_parser: 'SectionMeasurementParser',
                 check_data: bool = True):
        super.__init__(check_data=check_data)
        self.header_parser = header_parser
        self.measurement_parser = measurement_parser

    def get_current_feed_function(self):
        return self.get_current_parser().feed

    def get_current_check_function(self):
        return self.get_current_parser().check

    def get_current_parser(self):
        if self.current_line <= 2:
            return self.header_parser
        else:
            return self.measurement_parser


class ParserState(Enum):
    """State of a parser going through a CSV file with experimental data.

    The members refer to successive states in the parser:
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


class SectionType(Enum):
    FORCES_EMG = 1
    TRAJECTORIES = 2


class DeviceType:
    FORCE_PLATE = 1
    EMG = 2
    TRAJECTORY_MARKER = 3


class Device:
    devtype: DeviceType
    str_name: str


class Validator:
    current_line: int
    csv_filename: str
    should_raise: bool = True

    def __init__(self, csv_filename: str, should_raise: bool = True):
        self.current_line = 1
        self.csv_filename = csv_filename
        self.should_raise = should_raise

    def validate(self, data_check_result: 'DataCheck'):
        if self.should_raise:
            self._raise_if_invalid(data_check_result)

        self.current_line += 1

    def _raise_if_invalid(self, data_check_result: 'DataCheck'):
        is_valid = data_check_result['is_valid']
        error_description = data_check_result['error_message']

        if not is_valid:
            raise ValueError(self._build_error_message(error_description))

    def _build_error_message(self, error_description: str) -> str:
        return f'error parsing line {self.current_line} of file {self.csv_filename}: {error_message}'


class SectionParser:
    state: ParserState
    aggregator: 'Aggregator'
    devices_parser: 'DevicesParser'
    checker: Validator

    def __init__(self, aggregator: 'Aggregator',
                 devices_parser: 'DevicesParser', validator: Validator):
        self.state = SECTION_TYPE_LINE
        self.devices_parser = devices_parser
        self.aggregator = aggregator
        self.validator = Validator

    def feed_line(self, row: Row):
        self._raise_if_ended()

        check_function = self.get_check_function()
        parse_function = self.get_parse_function()
        aggregation_function = self.get_build_function()

        self._call_validator(check_function(row))
        parsed_line = parse_function(row)
        aggregation_function(parsed_line)

        self.transition()

    def get_check_function(self):
        if self.state == ParserState.SECTION_TYPE_LINE:
            return check_section_type_line
        elif self.state == ParserState.SAMPLING_FREQUENCY_LINE:
            return check_sampling_frequency_line
        elif self.state == ParserState.DEVICE_NAMES_LINE:
            return self.devices_parser.check_device_names_line

    def get_parse_function(self):
        if self.state == ParserState.SECTION_TYPE_LINE:
            return parse_section_type_line
        elif self.state == ParserState.SAMPLING_FREQUENCY_LINE:
            return parse_sampling_frequency_line
        elif self.state == ParserState.DEVICE_NAMES_LINE:
            return self.devices_parser.parse_device_names_line

    def get_build_function(self):
        if self.state == ParserState.SECTION_TYPE_LINE:
            return self.aggregator.add_section_type
        elif self.state == ParserState.SAMPLING_FREQUENCY_LINE:
            return self.aggregator.add_frequency
        elif self.state == ParserState.DEVICE_NAMES_LINE:
            # TODO Essa é a próxima linha
            return

    def transition(self):
        if self.state == ParserState.SECTION_TYPE_LINE:
            self.state = ParserState.SAMPLING_FREQUENCY_LINE
        elif self.state == ParserState.SAMPLING_FREQUENCY_LINE:
            self.state = ParserState.DEVICE_NAMES_LINE
        elif self.state == ParserState.DEVICE_NAMES_LINE:
            return

    def _raise_if_ended(self):
        if self.state == ParserState.BLANK_LINE:
            raise EOFError(
                'tried to parse another line from a section that has already been completely parsed'
            )

    def _call_validator(self, data_check_result: 'DataCheck'):
        self.validator.validate(data_check_result)


class SectionAggregator:
    section_type: SectionType
    frequency: int

    def add_section_type(self, section_type: SectionType):
        self.section_type = section_type

    def add_frequency(self, frequency: int):
        self.frequency = frequency


class DeviceAggregator:
    pass


class DevicesAggregator:
    def add_data():
        pass


# A data check is a dict representing the result of validating the data of a
# single line of the CSV data. Its keys:
#   'is_valid' maps to a bool that answers
#
#   'error_message' maps to a str containing an error message to be
#   displayed in case there are problems with the data. The str is appended to
#   the prefix "error parsing line {line_number} of file {filename}: "
DataCheck = Mapping[str, Any]


def check_section_type_line(row: Row) -> DataCheck:
    has_acceptable_value = row[0] in {'Devices', 'Trajectories'}
    is_valid = has_acceptable_value and columns_are_empty(row[1:])
    message = ('this line should contain either "Devices" or "Trajectories"'
               ' in its first column and nothing else')
    return {'is_valid': is_valid, 'error_message': message}


def parse_section_type_line(row: Row) -> SectionType:
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


def parse_sampling_frequency_line(row: Row) -> int:
    return int(row[0])


class DevicesParser:
    """Parser for device names, coordinates and units lines of the CSV file."""
    dev_names: List[str]
    section_type: SectionType

    def add_section_type(self, section_type: SectionType):
        self.section_type = section_type

    def parse_device_names_line(self, row: Row) -> List[str]:
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
            'tried to parse device names without first calling add_section_type'
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
        device_names = self.parse_device_names_line(row)

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


class SectionHeaderParser:
    # TODO it should be this class's responsibility to store the frequencies
    # to keep track of # subframes per frame
    pass


class SectionMeasurementsParser(abc.ABC):
    pass


class ForcesEMGMeasurementsParser:
    pass


class TrajectoriesMeasurementsParser:
    pass


class ViconForceEMGParser:
    """Parses the force plate and EMG output of experiments."""
    def __init__(self, check_data: bool = True):
        self.check_data = check_data


def read_data(filename, check_data=True):
    def columns_are_empty(row: Row) -> bool:
        return not ''.join(row)

    def check_first_line(row: Row):
        is_valid = row[0] == 'Devices' and columns_are_empty(row[1:])
        message = 'this line should contain "Devices" in its first column and nothing else'
        return {'is_valid': is_valid, 'message': message}

    def check_second_line(row: Row):
        try:
            is_valid = int(row[0])
        except ValueError:
            is_valid = False

        is_valid = is_valid and columns_are_empty(row[1:])
        message = 'this line should contain an integer in its first column and nothing else'
        return {'is_valid': is_valid, 'message': message}

    def parse_second_line(row, check_data):
        frequency = int(row[0])

        if not check_data:
            return

    def raise_if_invalid(check_data: bool, is_valid: bool, message: str,
                         line_number: int, csv_filename: str) -> None:
        if check_data and not is_valid:
            raise ValueError(
                f'error parsing line {line_number} of file {csv_filename}: {message}'
            )

    with open(filename) as csvfile:
        data_reader = csv.reader(csvfile)

        first_line = next(data_reader)
        check_first_line(first_line, check_data)
