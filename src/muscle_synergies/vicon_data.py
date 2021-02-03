"""Load data from the csv outputted by Vicon Nexus."""

import abc
from collections import defaultdict
import csv
from dataclasses import dataclass
from enum import Enum
import re
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator, Generic, TypeVar, NewType, Union)

import pint

T = TypeVar('T')
X = TypeVar('X')
Y = TypeVar('Y')

# a row from the CSV file is simply a list of strings,
# corresponding to different values
Row = NewType('Row', List[str])


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


@dataclass
class ForcePlateMeasurement:
    FORCE = 1
    MOMENT = 2
    COP = 3


class DataCheck:
    # A data check is a dict representing the result of validating the data of a
    # single line of the CSV data. Its keys:
    #   'is_valid' maps to a bool that answers
    #
    #   'error_message' maps to a str containing an error message to be
    #   displayed in case there are problems with the data. The str is appended to
    #   the prefix "error parsing line {line_number} of file {filename}: "
    def __init__(self, is_valid: bool, error_message: Optional[str] = None):
        if not is_valid and error_message is None:
            raise ValueError(
                'an invalid data check should contain an error message')

        if is_valid:
            error_message = None

        self.is_valid = is_valid
        self.error_message = error_message

    @classmethod
    def valid_data(cls):
        return cls(is_valid=True, error_message=None)

    def combine(self, other: 'DataCheck') -> 'DataCheck':
        if not self.is_valid:
            return self

        return other

    @classmethod
    def combine_multiple(cls,
                         data_checks: Sequence['DataCheck']) -> 'DataCheck':
        if not data_checks:
            return cls.valid_data()

        head = data_checks[0]
        tail = data_checks[1:]

        if not head.is_valid:
            return head

        return cls.combine_multiple(tail)


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

    __call__ = validate

    def _raise_if_invalid(self, data_check_result: DataCheck):
        is_valid = data_check_result.is_valid
        error_message = data_check_result.error_message

        if not is_valid:
            raise ValueError(self._build_error_message(error_message))

    def _build_error_message(self, error_message: str) -> str:
        return f'error parsing line {self.current_line} of file {self.csv_filename}: {error_message}'


class _ReaderState(abc.ABC):
    @abc.abstractmethod
    def feed_row(self, row: Row, reader: 'Reader'):
        # TODO document:
        # this should call validator exactly once.
        # it should also change the state of the reader.
        pass

    def _preprocess_row(self, row: Row) -> Row:
        row = list(row)
        while row and row[-1]:
            row.pop()
        return Row(row)

    def _validate(self, validator: Validator, check_result: DataCheck):
        validator.validate(check_result)

    def _reader_validator(self, reader: 'Reader') -> Validator:
        return reader.get_validator()

    def _reader_data_builder(self, reader: 'Reader') -> 'DataBuilder':
        return reader.get_data_builder()

    def _reader_set_new_state(self, reader: 'Reader',
                              new_state: '_ReaderState'):
        reader.set_state(new_state)

    def _create_data_check(self,
                           is_valid: bool,
                           error_message: Optional[str] = None):
        return DataCheck(is_valid=is_valid, error_message=error_message)

    def _create_valid_data_check(self):
        return DataCheck.valid_data()


class _EasyReaderState(_ReaderState, Generic[T]):
    def feed_row(self, row: Row, reader: 'Reader'):
        row = self._preprocess_row(row)

        # check and validate
        check_result = self._check_row(row)
        validator = self._reader_validator(reader)
        self._validate(validator, check_result)

        # parse and build up the data
        parsed_row = self._parse_row(row)
        self._build_data(parsed_row, self._reader_data_builder(reader))

        # transition to next state
        new_state = self._new_state()
        self._reader_set_new_state(reader, new_state)

    @abc.abstractmethod
    def _check_row(self, row: Row) -> DataCheck:
        pass

    @abc.abstractmethod
    def _parse_row(self, row: Row) -> T:
        pass

    @abc.abstractmethod
    def _build_data(self, parsed_data: T, data_builder: 'DataBuilder'):
        pass

    @abc.abstractmethod
    def _new_state(self) -> '_ReaderState':
        pass


class SectionTypeState(_EasyReaderState):
    """The state of a reader that is expecting the section type line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for :py:class:ViconCSVLines.
    """
    def _check_row(self, row: Row) -> DataCheck:
        is_valid = row[0] in {'Devices', 'Trajectories'}
        message = (
            'this line should contain either "Devices" or "Trajectories"'
            ' in its first column')
        return self._create_data_check(is_valid, message)

    def _parse_row(self, row: Row) -> SectionType:
        section_type_str = row[0]

        if section_type_str == 'Devices':
            return SectionType.FORCES_EMG
        elif section_type_str == 'Trajectories':
            return SectionType.TRAJECTORIES

    def _build_data(self, parsed_data: SectionType,
                    data_builder: 'DataBuilder'):
        data_builder.add_section_type(parsed_data)

    def _new_state(self):
        return SamplingFrequencyState()


class SamplingFrequencyState(_EasyReaderState):
    """The state of a reader that is expecting the sampling frequency line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for :py:class:ViconCSVLines.
    """
    def _check_row(self, row: Row) -> DataCheck:
        try:
            is_valid = int(row[0])
        except ValueError:
            is_valid = False

        message = (
            'this line should contain an integer representing'
            f' sampling frequency in its first column and not {row[0]}.')

        return self._create_data_check(is_valid, message)

    def _parse_row(self, row: Row) -> int:
        try:
            return int(row[0])
        except ValueError:
            return None

    def _build_data(self, parsed_data: int, data_builder: 'DataBuilder'):
        data_builder.add_frequency(parsed_data)

    def _new_state(self) -> '_ReaderState':
        return DevicesState()


class Reader:
    """Reader for a single section of the CSV file outputted by Vicon Nexus.

    Initialize it
    """
    _state: _ReaderState
    _data_builder: 'DataBuilder'
    _validator: Validator

    def __init__(self, section_data_builder: 'DataBuilder',
                 validator: Validator):
        self.state = ViconCSVLines.SECTION_TYPE_LINE
        self.data_builder = section_data_builder
        self.validator = validator

    def set_state(self, new_state: _ReaderState):
        self._state = new_state

    def get_data_builder(self):
        return self._data_builder

    def get_validator(self):
        return self._validator

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
        if self.state == ViconCSVLines.SECTION_TYPE_LINE:
            return check_section_type_line
        elif self.state == ViconCSVLines.SAMPLING_FREQUENCY_LINE:
            return check_sampling_frequency_line
        elif self.state == ViconCSVLines.DEVICE_NAMES_LINE:
            return self.devices_reader.check_device_names_line

    def _get_read_function(self):
        if self.state == ViconCSVLines.SECTION_TYPE_LINE:
            return read_section_type_line
        elif self.state == ViconCSVLines.SAMPLING_FREQUENCY_LINE:
            return read_sampling_frequency_line
        elif self.state == ViconCSVLines.DEVICE_NAMES_LINE:
            return self.devices_reader.read_device_names_line

    def _get_build_function(self):
        if self.state == ViconCSVLines.SECTION_TYPE_LINE:
            return self.data_builder.add_section_type
        elif self.state == ViconCSVLines.SAMPLING_FREQUENCY_LINE:
            return self.data_builder.add_frequency
        elif self.state == ViconCSVLines.DEVICE_NAMES_LINE:
            # TODO Essa é a próxima linha
            return

    def _transition(self):
        if self.state == ViconCSVLines.SECTION_TYPE_LINE:
            self.state = ViconCSVLines.SAMPLING_FREQUENCY_LINE
        elif self.state == ViconCSVLines.SAMPLING_FREQUENCY_LINE:
            self.state = ViconCSVLines.DEVICE_NAMES_LINE
        elif self.state == ViconCSVLines.DEVICE_NAMES_LINE:
            return

    def _raise_if_ended(self):
        if self.state == ViconCSVLines.BLANK_LINE:
            raise EOFError(
                'tried to read another line from a section that has already been completely readd'
            )

    def _call_validator(self, data_check_result: DataCheck):
        self.validator.validate(data_check_result)


class DataBuilder:
    section_type: SectionType
    frequency: int

    force_plates_device_header: Optional[List['ForcePlate']]
    emg_device_header: Optional['DeviceHeader']
    trajectory_device_header: Optional[List['DeviceHeader']]

    def add_section_type(self, section_type: SectionType):
        self.section_type = section_type

    def add_frequency(self, frequency: int):
        self.frequency = frequency


class TimeSeriesDataBuilder:
    """Builds data of an individual time series.

    Columns in the CSV file correspond to measurements made over time (at
    least in parts of it, see :py:class:ViconCSVLines for more details on the
    structure of the CSV file).

    This class keeps track of the information for one such time series as that
    information is read line by line from the input.
    """
    coordinate_name: Optional[str]
    physical_unit: Optional[pint.Unit]
    data: List[float]

    def __init__(self):
        self.coordinate_name = None
        self.physical_unit = None
        self.data = []

    def add_coordinate(self, coord_name: str):
        """Adds the coordinate name."""
        self.coordinate_name = coord_name

    def add_unit(self, physical_unit: pint.Unit):
        """Adds the physical units."""
        self.physical_unit = physical_unit

    def add_data(self, data_entry: float):
        """Adds a data entry."""
        self.data.append(data_entry)


class DeviceHeaderDataBuilder:
    """The data builder of a single device header.

    Most device headers have exactly 3 columns, like the ones for trajectory
    markers. EMG device headers can have an arbitrary number of columns, each
    referring to a different muscle. What they all have in common is that they
    all have components.

    This class passes along data from each line following the device header
    line to each component. The exact pipeline is this:
    1. the caller finds the columns of the CSV file which refer to the device
       header asssociated with a :py:class:DeviceHeaderDataBuilder instance.
    2. the caller then passes exactly those columns to that instance using the
       appropriate method
       (e.g., :py:func:DeviceHeaderDataBuilder.add_coordinates)
    3. this class channels the data forward for different TimeSeriesDataBuilder
       objects, which are taken during initialization.

    Args:
        time_series_list: the list of :py:class:TimeSeriesDataBuilder objects
            to which the data has to be passed.
    """
    time_series_tuple: Tuple[TimeSeriesDataBuilder]

    def __init__(self, time_series_list: List[TimeSeriesDataBuilder]):
        self.time_series_tuple = tuple(time_series_list)

    def _call_method_on_each(self, parsed_data: List, method_name: str):
        """Calls a method on each time series with the data as argument.

        Args:
            parsed_data: the data, parsed from a line from the CSV input, to be
                passed along to the different :py:class:TimeSeriesDataBuilder
                instances.

            method_name: the name of the method to be called from each
                :py:class:TimeSeriesDataBuilder.

        Raises:
            ValueError: if the length of `parsed_data` doesn't match that of
               the `time_series_list` provided during initialization.
        """
        self._validate_data(parsed_data)

        for data_entry, time_series in zip(parsed_data,
                                           self.time_series_tuple):
            method = getattr(time_series, method_name)
            method(data_entry)

    def _validate_data(self, parsed_data: List):
        """Raises an exception if the data doesn't have the correct length.

        Args:
            parsed_data: the data the length of which is to be compared with
                that of the `time_series_list` provided during initialization.

        Raises:
            ValueError: if the length of `parsed_data` doesn't match that of
               the `time_series_list` provided during initialization.
        """
        if len(parsed_data) != len(self.time_series_tuple):
            raise ValueError(f'provided parsed_data argument with length'
                             f' {len(parsed_data)}'
                             f' but was expecting length'
                             f' {len(self.time_series_tuple)}')

    def add_coordinates(self, parsed_data: List[str]):
        """Add coordinates to each individual time series.

        Args:
            parsed_data: the strings from the CSV file describing the
                coordinate or muscle.

        Raises:
            ValueError: the data doesn't have the same length as the
                `time_series_list` provided during initialization.
        """
        self._call_method_on_each(parsed_data, 'add_coordinate')

    def add_units(self, parsed_data: List[pint.Unit]):
        """Add units to each individual time series.

        Args:
            parsed_data: the physical units parsed from the CSV file.

        Raises:
            ValueError: the data doesn't have the same length as the
                `time_series_list` provided during initialization.
        """
        self._call_method_on_each(parsed_data, 'add_unit')

    def add_data(self, parsed_data: List[float]):
        """Add a data entry to each individual time series.

        Args:
            parsed_data: data parsed from one line of the CSV file.

        Raises:
            ValueError: the data doesn't have the same length as the
                `time_series_list` provided during initialization.
        """
        self._call_method_on_each(parsed_data, 'add_data')


@dataclass
class DeviceHeader:
    """A device header.

    This class keeps track of 2 components referring to individual device
    headers (see :py:class:ViconCSVLines for an explanation of what is a device
    header):
    * :py:class:DeviceHeaderCols
    * :py:class:DeviceHeaderDataBuilder

    The first of those (:py:class:DeviceHeaderCols) keeps track of which
    columns from the CSV file refer to that device. The second
    (:py:class:DeviceHeaderDataBuilder) is accumulates data of different vector
    time series, each of which coming from an individual column of the CSV
    file.

    Args:
        device_cols: the DeviceHeaderCols object, which must refer to the
            same device header as `device_data_builder`.

        device_data_builder: the DeviceHeaderDataBuilder object, which must
            refer to the same device header as `device_cols`
    """
    device_cols: DeviceHeaderCols
    device_data_builder: DeviceHeaderDataBuilder

    @property
    def device_name(self):
        return self.device_cols.device_name

    @property
    def device_type(self):
        return self.device_cols.device_type


@dataclass
class ForcePlate:
    """The 3 DevicesHeader with the data of a single force plate.

    Since each force plate is represented in 3 different device headers,
    this class provides a standard way to unify them.

    Args:
        force: the device header with force measurements

        moment: the device header with moment measurements

        cop: the device header with cop measurements
    """
    force: DeviceHeader
    moment: DeviceHeader
    cop: DeviceHeader


class DataChanneler:
    """Channels rows of data from the CSV input to the its data builder.

    Args:
        devices: the list of Device objects to which to channel the data.
    """
    _DeviceSlice = Tuple[DeviceHeaderDataBuilder, slice]

    devices: Tuple[DeviceHeader]

    def __init__(self, devices: List[DeviceHeader]):
        self.devices = tuple(devices)

    def _device_builder(self, device: DeviceHeader) -> DeviceHeaderDataBuilder:
        """Gets data builder from a device."""
        return device.device_data_builder

    def _device_row_slice(self, device: DeviceHeader) -> slice:
        """Gets the slice object corresponding to a device."""
        device_cols = device.device_cols
        return device_cols.create_slice()

    def _iter_device_slice(self) -> Iterator[_DeviceSlice]:
        """Yields all pairs of data builder and row slice."""
        for device in self.devices:
            builder = self._device_builder(device)
            row_slice = self._device_row_slice(device)
            yield (builder, row_slice)

    def _call_method_of_each_device(self, parsed_row: List, method_name: str):
        """Calls method on each device with the parsed row as an argument.

        Args:
            parsed_row: the entire parsed row which is sliced and passed along
                to data builders.

            method_name: the name of the method which is to be called on each
                data builder.
        """
        for device, row_slice in self._iter_device_slice():
            data = parsed_row[row_slice]
            method = getattr(device, method_name)
            method(data)

    def add_coordinates(self, parsed_row: List[str]):
        """Adds coordinates to devices.

        Args:
            parsed_row: the coordinates line of the input.
        """
        self._call_method_of_each_device(parsed_row, 'add_coordinates')

    def add_units(self, parsed_row: List[pint.Unit]):
        """Adds physical units to devices.

        Args:
            parsed_row: the units line of the input, already parsed.
        """
        self._call_method_of_each_device(parsed_row, 'add_units')

    def add_data(self, parsed_row: List[float]):
        """Adds physical units to devices.

        Args:
            parsed_row: a data line of the input, already parsed.
        """
        self._call_method_of_each_device(parsed_row, 'add_data')


class ViconDataLoader:
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


class _TEMPORARY_STORAGE:
    """Temporary holder for code previously used for AllDevicesDataBuilder.

    This code will likely be useful in one of the Reader states, so it is kept
    here.

    Args:
        force_plates (optional): a list of :py:class:ForcePlateCols or None.

        emg_device (optional): a :py:class:DeviceHeaderCols representing the
            columns of an EMG device header or None.

        trajectory_marker_devices (optional): a list of
            :py:class:DeviceHeaderCols, each representing the columns of a data
            header which is a trajectory marker or None.

        device_header_data_builder_type (optional): the class used to represent
            individual device headers. By default, it is
            DeviceHeaderDataBuilder

        time_series_data_builder_type (optional): the class used to represent
            individual time series. By default, it is TimeSeriesDataBuilder.
    """
    class Obj:
        pass

    pt = Obj()
    pt.fixture = lambda f: f

    def __init__(self,
                 force_plate_cols: List[ForcePlateCols] = None,
                 emg_cols: Optional[DeviceHeaderCols] = None,
                 trajectory_marker_cols: List[DeviceHeaderCols] = None,
                 device_header_data_builder_type=DeviceHeaderDataBuilder,
                 time_series_data_builder_type=TimeSeriesDataBuilder):
        self._device_header_data_builder_type = device_header_data_builder_type
        self._time_series_data_builder_type = time_series_data_builder_type

        if force_plate_cols is None:
            self.force_plates = None
        else:
            self.force_plates = [
                self._create_force_plate_device(fp) for fp in force_plate_cols
            ]

        if emg_cols is None:
            self.emg = None
        else:
            self.emg = self._create_device(emg_cols)

        if trajectory_marker_cols is None:
            self.trajectory_markers = None
        else:
            self.trajectory_markers = [
                self._create_device(tm) for tm in trajectory_marker_cols
            ]

    def _create_force_plate_device(self,
                                   force_plates: ForcePlateCols) -> ForcePlate:
        """Creates a ForcePlates object.

        This is used during the initialization of
        :py:class:AllDevicesDataBuilder instances.

        Args:
            force_plates: the object describing columns of force plates passed
                for :py:class:AllDevicesDataBuilder.__init__
        """
        force = force_plates.force
        moment = force_plates.moment
        cop = force_plates.cop

        return self.ForcePlate(force=self._create_device(force),
                               moment=self._create_device(moment),
                               cop=self._create_device(cop))

    def _create_device(self, device_cols: DeviceHeaderCols) -> DeviceHeader:
        """Creates a Device object.

        This is used during the initialization of
        :py:class:AllDevicesDataBuilder instances.

        Args:
            device_cols: the device_cols member of the newly created Device
                object
        """
        return self.Device(
            device_cols=device_cols,
            device_data_builder=self._create_device_header_data_builder())

    def _create_device_header_data_builder(self):
        """Instantiates a new :py:class:DeviceHeaderDataBuilder

        This is used during the initialization of
        :py:class:AllDevicesDataBuilder instances.
        """
        return self._device_header_data_builder_type(
            time_series_data_builder_type=self._time_series_data_builder_type)

    @pt.fixture
    def mock_device_header_data_builder_type(self, mocker):
        singleton_device_header_data_builder_mock = mocker.create_autospec(
            vd.DeviceHeaderDataBuilder)

        data_builder_factory = mocker.Mock(
            spec=vd.DeviceHeaderDataBuilder.__init__,
            return_value=singleton_device_header_data_builder_mock)

        return data_builder_factory

    @pt.fixture
    def mock_time_series_data_builder_type(self, mocker):
        return 'fake time series data builder'

    @pt.fixture
    def force_plate_arg(self):
        return [
            vd.ForcePlateCols(
                force=vd.DeviceHeaderCols(
                    device_name='force plate 1 force',
                    device_type=vd.DeviceType.FORCE_PLATE,
                    first_col_index=2),
                moment=vd.DeviceHeaderCols(
                    device_name='force plate 1 moment',
                    device_type=vd.DeviceType.FORCE_PLATE,
                    first_col_index=5),
                cop=vd.DeviceHeaderCols(device_name='force plate 1 cop',
                                        device_type=vd.DeviceType.FORCE_PLATE,
                                        first_col_index=8),
            )
        ]

    @pt.fixture
    def trajectory_marker_header_cols(self):
        return [
            vd.DeviceHeaderCols(device_name='trajectory marker',
                                device_type=vd.DeviceType.TRAJECTORY_MARKER,
                                first_col_index=11)
        ]

    @pt.fixture
    def emg_header_cols(self):
        return vd.DeviceHeaderCols(device_name='emg',
                                   device_type=vd.DeviceType.EMG,
                                   first_col_index=14)

    @pt.fixture
    def all_devices_data_builder(self, mock_device_header_data_builder_type,
                                 mock_time_series_data_builder_type,
                                 force_plate_arg,
                                 trajectory_marker_header_cols,
                                 emg_header_cols):
        return vd.AllDevicesDataBuilder(
            force_plate_cols=force_plate_arg,
            emg_cols=emg_header_cols,
            trajectory_marker_cols=trajectory_marker_header_cols,
            device_header_data_builder_type=
            mock_device_header_data_builder_type,
            time_series_data_builder_type=mock_time_series_data_builder_type,
        )

    def test_initializes_data_header_creates_data_header_builders(
            self, all_devices_data_builder,
            mock_device_header_data_builder_type,
            mock_time_series_data_builder_type):
        mock_device_header_data_builder_type.assert_called_with(
            time_series_data_builder_type=mock_time_series_data_builder_type)

    def test_initializes_correct_number_of_data_header_builders(
            self, all_devices_data_builder,
            mock_device_header_data_builder_type):
        mock_object = mock_device_header_data_builder_type
        number_of_devices = 5
        assert mock_object.call_count == number_of_devices
