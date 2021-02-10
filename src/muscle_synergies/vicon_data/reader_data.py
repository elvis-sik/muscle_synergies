"""Data types used internally while parsing Vicon Nexus data."""

import abc
from collections import defaultdict
import csv
from dataclasses import dataclass
from enum import Enum
import re
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator, Generic, TypeVar, NewType, Union)

import pint
import pint_pandas

ureg = pint.UnitRegistry()
pint_pandas.PintType.ureg = ureg

T = TypeVar('T')
X = TypeVar('X')
Y = TypeVar('Y')

# a row from the CSV file is simply a list of strings,
# corresponding to different values
Row = NewType('Row', List[str])

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

    def section_type(self):
        if self in {DeviceType.EMG, DeviceType.FORCE_PLATE}:
            return SectionType.FORCES_EMG
        return SectionType.TRAJECTORIES


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


@dataclass(frozen=True, eq=True)
class ForcePlateDevices:
    """The 3 device headers with the data of a single force plate.

    Since each force plate is represented in 3 different device headers, this
    class provides a standard way to represent those.

    Args:
        name: the name of a force plate, that is, everything happening in its
            column in the devices line before the dash (excluding the space).
            For example, the header
            'Imported AMTI OR6 Series Force Plate #1 - Force' should have name
            'Imported AMTI OR6 Series Force Plate #1'.

        force: the device referring to the force time series.

        moment: the device referring to the moment time series.

        cop: the device referring to the cop time series.
    """
    name: str
    force: DeviceHeaderRepresentation
    moment: DeviceHeaderRepresentation
    cop: DeviceHeaderRepresentation

    def list_devices(self) -> List[DeviceHeaderRepresentation]:
        return [self.force, self.moment, self.cop]


@dataclass
class ViconNexusData:
    force_plates: Union[List[DeviceHeaderRepresentation],
                        ForcePlateRepresentation, 'DeviceMapping']
    emg: DeviceHeaderRepresentation
    trajectory_markers: List[DeviceHeaderRepresentation, 'DeviceMapping']

    def from_device_type(
            self, device_type: DeviceType
    ) -> Union[List[DeviceHeaderRepresentation], ForcePlateRepresentation,
               DeviceHeaderRepresentation, 'DeviceMapping']:
        if device_type is DeviceType.FORCE_PLATE:
            return self.force_plates
        if device_type is DeviceType.EMG:
            return self.emg
        if device_type is DeviceType.TRAJECTORY_MARKER:
            return self.trajectory_markers

        raise ValueError(f'device type {device_type} not understood')


@dataclass
class ColOfHeader:
    """The string describing a device and the column in which it occurs.

    This is used as an intermediate representation of the data being read in
    the device names line (see :py:class:ViconCSVLines). The structure of that
    line is complex, so the logic of its parsing is split into several classes.
    ColOfHeader is used for communication between them.

    Args:
        col_index: the index of the column in the CSV file in which the
            device header is described.

        header_str: the exact string occurring in that column.
    """
    col_index: int
    header_str: str


class DeviceHeaderCols:
    """Intermediate representation of the data read in the device names line.

    This class is used as a way of communicating data from the :py:class:Reader
    to the :py:class:DataBuilder. For more information on where the data that
    is held by this class, see the docs for :py:class:ViconCSVLines.

    Args:
        device_name: the name of the device, as read from the CSV file

        device_type: the type of the device

        first_col_index: the index in a Row in which the data for the device
            begins
    """
    _col_of_header: ColOfHeader
    device_type: DeviceType
    num_of_cols: Optional[int]

    def __init__(self, col_of_header: ColOfHeader, device_type: DeviceType):
        self._col_of_header = col_of_header
        self.device_type = device_type
        self.num_of_cols = None
        self._initialize_num_of_cols()

    @property
    def device_name(self):
        return self._col_of_header.header_str

    @property
    def first_col_index(self):
        return self._col_of_header.col_index

    def create_slice(self):
        """Creates a slice object corresponding to the device header columns.

        Raises:
            TypeError: if num_of_cols is None. This will happen for an EMG
                device header if :py:func:DeviceHeaderCols.add_num_cols isn't
                called explicitly.
        """
        if self.num_of_cols is None:
            raise TypeError('add_num_of_cols should be called before slice')

        return slice(self.first_col_index,
                     self.first_col_index + self.num_of_cols)

    def add_num_cols(self, num_of_cols: int):
        """Add number of columns.

        This should only be used for EMG devices, and only once.

        Raises:
            TypeError: if either the device isn't a EMG one or the method is
                called more than once.
        """
        if self.num_of_cols is not None:
            raise TypeError(
                'tried to set num_of_cols with the variable already set')

        if self.device_type is not DeviceType.EMG:
            raise TypeError(
                "tried to set num_of_cols for a device the type of which isn't EMG"
            )

        self.num_of_cols = num_of_cols

    def _initialize_num_of_cols(self):
        """Determines if possible the number of columns of the device"""
        if self.device_type is DeviceType.EMG:
            self.num_of_cols = None
        else:
            self.num_of_cols = 3


class _OnlyOnceMixin:
    def _raise_called_twice(self, what_was_added_twice: str):
        raise TypeError(
            f'attempted to add {what_was_added_twice} after it had ' +
            'been already added')


class _SectionDataBuilder(_OnlyOnceMixin, abc.ABC):
    finished: bool
    frequency: Optional[int]
    data_channeler: Optional['DataChanneler']

    def __init__(self):
        super().__init__()
        self._finished = False
        self.frequency = None
        self.data_channeler = None

    @property
    def finished(self) -> bool:
        return self._finished

    @abc.abstractproperty
    def section_type(self) -> SectionType:
        return

    @abc.abstractmethod
    def file_ended(self) -> ViconNexusData:
        pass

    @abc.abstractmethod
    def transition(self, data_builder: DataBuilder):
        self._finished = True

    def add_frequency(self, frequency: int):
        self._raise_if_finished('frequency')
        if self.frequency is not None:
            self._raise_called_twice('frequency')
        self.frequency = frequency

    def add_data_channeler(self, data_channeler: 'DataChanneler'):
        self._raise_if_finished('DataChanneler')
        if self.data_channeler is not None:
            self._raise_called_twice('DataChanneler')
        self.data_channeler = data_channeler

    def add_units(self, units: List[pint.Unit]):
        self._raise_if_finished('units')
        self.data_channeler.add_units(units)

    def add_measurements(self, data: List[float]):
        self._raise_if_finished('data')
        self.data_channeler.add_data(data)

    def _raise_if_finished(self, tried_to_add_what: str):
        if self.finished:
            raise TypeError(
                f'tried to add {tried_to_add_what} to a finished _SectionDataBuilder'
            )


class ForcesEMGDataBuilder(_SectionDataBuilder):
    section_type = SectionType.FORCES_EMG

    def file_ended(self) -> ViconNexusData:
        raise ValueError('file ended without a trajectory marker section.')

    def transition(self, data_builder: DataBuilder):
        super().transition(data_builder)
        data_builder.set_current_section(next_section_builder)


class TrajDataBuilder(_SectionDataBuilder):
    section_type = SectionType.TRAJECTORIES
    _frequencies_type = Frequencies

    # TODO
    # 1. finish these 3
    # 2. finish the devices line guys
    # 3. finish the routine that loads everything up from a CSV
    # 4. implement a simple integration test
    # 5. fix bugs one by one
    # 6. start abstracting unit tests
    # 7. write an example notebook

    def file_ended(self) -> ViconNexusData:
        self._build_frequencies(self._get_frequencies())

    def transition(self, data_builder: DataBuilder):
        super().transition(data_builder)

    def _get_frequencies(self):
        pass


class TimeSeriesDataBuilder(_OnlyOnceMixin):
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
        super().__init__()
        self.coordinate_name = None
        self.physical_unit = None
        self.data = []

    def add_coordinate(self, coord_name: str):
        """Adds the coordinate name."""
        if self.coordinate_name is not None:
            self._raise_called_twice('coordinate')
        self.coordinate_name = coord_name

    def add_unit(self, physical_unit: pint.Unit):
        """Adds the physical units."""
        if self.physical_unit is not None:
            self._raise_called_twice('physical unit')
        self.physical_unit = physical_unit

    def add_data(self, data_entry: float):
        """Adds a data entry."""
        self.data.append(data_entry)

    def get_coordinate_name(self) -> Optional[str]:
        return self.coordinate_name

    def get_physical_unit(self) -> Optional[pint.Unit]:
        return self.physical_unit

    def get_data(self) -> List[float]:
        """Gets the data added so far.

        The list returned is the same one used internally by this class, so
        mutating it will mutate it as well.
        """
        return self.data


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

    def get_time_series(self, ind: int) -> TimeSeriesDataBuilder:
        return self.time_series_tuple[ind]

    __getitem__ = get_time_series

    def __len__(self):
        return len(self.time_series_tuple)


@dataclass
class DeviceHeaderPair:
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


class DataChanneler:
    """Channels rows of data from the CSV input to the its data builder.

    Args:
        devices: the list of Device objects to which to channel the data.
    """

    devices: Tuple[DeviceHeaderPair]

    def __init__(self, devices: List[DeviceHeaderPair]):
        self.devices = tuple(devices)

    def _device_builder(self,
                        device: DeviceHeaderPair) -> DeviceHeaderDataBuilder:
        """Gets data builder from a device."""
        return device.device_data_builder

    def _device_row_slice(self, device: DeviceHeaderPair) -> slice:
        """Gets the slice object corresponding to a device."""
        device_cols = device.device_cols
        return device_cols.create_slice()

    def _iter_device_slice(
            self) -> Iterator[Tuple[DeviceHeaderDataBuilder, slice]]:
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


class FailableResult(Generic[T]):
    """Data class holding the result of a parsing process that can fail.

    FailableResult holds the data parsed from a process. If the process failed,
    it holds the :py:class:DataCheck with details on the failure instead.

    In case the process didn't fail, the data_check member of an instance will
    hold a valid :py:class:DataCheck.

    Args:
        parse_result: the data parsed.

        data_check: a check on the validity of the data.

    Raises:
        ValueError: if `data_check` is provided and its `is_valid` member is
            `False` but a `parse_result` was also provided. Also if a
            `data_check` with a `True` `is_valid` member is provided but no
            `parse_result` is provided. Also if none of the 2 arguments was
            provided.
    """

    parse_result: Optional[T]
    data_check: DataCheck

    def __init__(self,
                 *,
                 parse_result: Optional[T] = None,
                 data_check: Optional[DataCheck] = None):
        if parse_result is not None:
            self.data_check = self._process_data_check_result_provided(
                data_check)
            self.parse_result = parse_result
        else:
            self.parse_result = None
            self.data_check = self._process_data_check_without_result(
                data_check)

    def _process_data_check_result_provided(self,
                                            data_check: Optional[DataCheck]
                                            ) -> DataCheck:
        if data_check is None:
            return self._valid_data_check()

        if not data_check.is_valid:
            raise ValueError(
                'invalid data check and a parse result were provided')

        return data_check

    def _process_data_check_without_result(self,
                                           data_check: Optional[DataCheck]
                                           ) -> DataCheck:
        if data_check is None:
            # parse_result is assumed to be None
            return self._valid_data_check()

        if data_check.is_valid:
            raise ValueError('valid data check provided without parse result')

        return data_check

    @staticmethod
    def _valid_data_check():
        return DataCheck.valid_data()

    @classmethod
    def create_failed(cls, error_message: str) -> 'FailableResult':
        data_check = DataCheck(is_valid=False, error_message=error_message)
        return cls(data_check=data_check)

    @classmethod
    def create_successful(cls, parse_result: T) -> 'FailableResult[T]':
        return cls(parse_result=parse_result)

    @property
    def failed(self):
        return not self.data_check.is_valid

    def __eq__(self, other: 'FailableResult[T]') -> bool:
        return (self.failed == other.failed
                and self.parse_result == other.parse_result)


class _FailableMixin:
    _failable_result_class = FailableResult

    def _compute_on_failable(
            self,
            func: Union[Callable[[X], FailableResult[Y]], Callable[[X], Y]],
            arg: FailableResult[X],
            compose: bool = False) -> Union[FailableResult[Y], Y]:
        if self._fail_res_failed(arg):
            return arg

        res = func(arg)
        if not compose:
            return res

        return self._success(res)

    def _sequence_fail(self, failable_results: Sequence[FailableResult[T]]
                       ) -> FailableResult[List[T]]:
        if len(failable_results) == 0:
            return self._fail('called _sequence_fail on empty list')

        parsed_values = []

        for fail_res in failable_results:
            self._compute_on_failable(parsed_values.append,
                                      fail_res,
                                      compose=False)

        return self._success(parsed_values)

    def _fail(self, error_message: str) -> FailableResult[ColOfHeader]:
        return self._failable_result_class.create_failed(error_message)

    def _success(self, result: T) -> FailableResult[T]:
        return self._failable_result_class.create_successful(result)

    def _fail_res_failed(self, fail_res: FailableResult) -> bool:
        return fail_res.failed

    def _fail_res_parse_result(self, fail_res: FailableResult[T]) -> T:
        return fail_res.parse_result

    def _fail_res_data_check(self, fail_res: FailableResult) -> DataCheck:
        return DataCheck


class Frequencies:
    _freq_forces_emg_section: int
    _freq_trajectories_section: int
    num_frames: int

    def __init__(self, frequency_forces_emg_section: int,
                 frequency_trajectories_sequence: int, num_frames: int):
        self._freq_forces_emg_section = frequency_forces_emg_section
        self._freq_trajectories_section = frequency_trajectories_sequence
        self.num_frames = num_frames

    @property
    def num_subframes(self) -> int:
        num = self._freq_forces_emg_section / self._freq_trajectories_section
        assert num == int(num)
        return int(num)

    def index(self, device_type: DeviceType, frame: int, subframe: int) -> int:
        self._validate_frame_arg(frame)
        self._validate_subframe_arg(subframe)

        section_type = device_type.section_type()

        if section_type is SectionType.TRAJECTORIES:
            return frame - 1
        return self._index_forces_emg(frame, subframe)

    def frame_subframe(self, device_type: DeviceType,
                       index: int) -> Tuple[int, int]:
        section_type = device_type.section_type()

        if section_type is SectionType.TRAJECTORIES:
            self._validate_traj_index_arg(index)
            return index + 1, 0

        self._validate_forces_emg_index_arg(index)
        return self._forces_emg_frame_subframe(index)

    def frame_range(self) -> range:
        return range(1, self.num_frames + 1)

    def subframe_range(self) -> range:
        return range(self.num_subframes)

    def frequency_of(self, device_type: DeviceType) -> int:
        section_type = device_type.section_type()

        if section_type is SectionType.TRAJECTORIES:
            return self._freq_trajectories_section
        return self._freq_forces_emg_section

    def _forces_emg_frame_subframe(self, index: int) -> Tuple[int, int]:
        # + 1 at the end because Python is 0-indexed
        frame = (index // self.num_subframes) + 1
        subframe = index % self.num_subframes
        return frame, subframe

    def _index_forces_emg(self, frame: int, subframe: int) -> int:
        return (frame - 1) * self.num_subframes + subframe

    def _validate_frame_arg(self, frame: int):
        if frame not in self.frame_range():
            raise ValueError(
                f'last frame is {self.num_frames}, frame {frame} is out of bounds'
            )

    def _validate_subframe_arg(self, subframe: int):
        if subframe not in self.subframe_range():
            raise ValueError(
                f'subframe {subframe} out of range {self.subframe_range()}')

    def _validate_traj_index_arg(self, index: int):
        final_index = self.num_frames - 1

        if index not in range(final_index + 1):
            raise ValueError(
                f'final index for trajectory marker is {final_index}, '
                'index {index} is out of bounds')

    def _validate_forces_emg_index_arg(self, index: int):
        final_index = self.num_frames * self.num_subframes - 1
        if index not in range(final_index + 1):
            raise ValueError(
                f'final index for force plates and EMG data is {final_index}, '
                f'index {index} out of bounds')


class DeviceHeaderData:
    device_name: str
    device_type: DeviceType
    _frequencies: Frequencies
    dataframe: pd.DataFrame

    def __init__(
            self,
            device_name: str,
            device_type: DeviceType,
            frequencies: Frequencies,
            dataframe: pd.DataFrame,
    ):
        self.device_name = device_name
        self.device_type = device_type
        self._frequencies = frequencies
        self.dataframe = dataframe

    @property
    def sampling_frequency(self) -> int:
        return self.frequencies.frequency_of(self.device_type)

    def slice_frame_subframe(self,
                             *,
                             stop_frame: int,
                             stop_subframe: int,
                             start_frame: Optional[int] = None,
                             start_subframe: Optional[int] = None,
                             step: Optional[int] = None) -> slice:
        stop_index = self._frequencies_index(stop_frame, stop_subframe)
        if start_frame is None:
            return slice(stop_index)

        start_index = self._frequencies_index(start_frame, start_subframe)
        if step is None:
            return slice(start_index, stop_index)
        return slice(start_index, stop_index, step)

    def _frequencies_index(self, frame: int, subframe: int) -> int:
        return self._frequencies.index(self.device_type, frame, subframe)

    @classmethod
    def from_device_header_pair(cls, device_header_pair: DeviceHeaderPair,
                                frequencies: Frequencies) -> 'DeviceHeader':
        device_name = device_header_pair.device_name
        device_type = device_header_pair.device_type
        dataframe = cls._device_header_pair_dataframe(device_header_pair)
        return cls(device_name=device_name,
                   device_type=device_type,
                   frequencies=frequencies,
                   dataframe=dataframe)

    @classmethod
    def _device_header_pair_dataframe(cls, device_header_pair: DeviceHeaderPair
                                      ) -> pd.Dataframe:
        builder = device_header_pair.device_data_builder
        return cls._extract_dataframe(builder)

    @staticmethod
    def _extract_dataframe(device_header_builder: DeviceHeaderDataBuilder
                           ) -> pd.DataFrame:
        def create_pint_array(data, physical_unit):
            PintArray(data, dtype=physical_unit)

        data_dict = {}
        for time_series_builder in device_header_builder:
            coord_name = time_series_builder.get_coordinate_name()
            physical_unit = time_series_builder.get_physical_unit()
            data = time_series_builder.get_data()
            data_dict[coord_name] = create_pint_parray(data, physical_unit)

        return pd.DataFrame(data_dict)


class ForcePlateData(DeviceHeaderData):
    def __init__(
            self,
            device_name: str,
            frequencies: Frequencies,
            dataframe: pd.DataFrame,
    ):
        super().__init__(device_name=device_name,
                         device_type=DeviceType.FORCE_PLATE,
                         frequencies=frequencies,
                         dataframe=dataframe)

    @classmethod
    def from_force_plate(cls, force_plate: ForcePlateDevices,
                         frequencies: Frequencies):
        device_name = force_plate.name

        force_device = force_plate.force
        moment_device = force_plate.moment
        cop_device = force_plate.cop

        force_dataframe = cls._device_header_pair_dataframe(force_device)
        moment_dataframe = cls._device_header_pair_dataframe(moment_device)
        cop_dataframe = cls._device_header_pair_dataframe(cop_device)

        dataframe = cls._join_dataframes(force_dataframe, moment_dataframe,
                                         cop_dataframe)

        cls(device_name=device_name,
            frequencies=frequencies,
            dataframe=dataframe)

    @staticmethod
    def _join_dataframes(*args: Tuple[pd.DataFrame]) -> pd.DataFrame:
        assert args

        if len(args) == 1:
            return args[0]
        return args[0].join(args[1:])


class DeviceMapping(collections.abc.Mapping):
    device_list: List[Union[DeviceHeaderData, ForcePlateData]]
    devices_dict: Mapping[str, Union[DeviceHeaderData, ForcePlateData]]

    def __init__(
            self,
            device_list: List[Union[DeviceHeaderData, ForcePlateData]],
    ):
        self.device_list = list(device_list)
        self.devices_dict = self._build_devices_dict()

    def ith(self, i: int) -> Union[DeviceHeaderData, ForcePlateData]:
        return self.device_list[i]

    def _build_devices_dict(self):
        devices_dict = {}
        for device in device_list:
            device_name = device.device_name
            devices_dict[device_name] = device
        return devices_dict

    def __getitem__(self, ind: X) -> pd.DataFrame:
        return self._devices_dict.__getitem__(ind)

    def __len__(self) -> int:
        return len(self._devices_dict)

    def __iter__(self) -> Iterable[X]:
        yield from iter(self._devices_dict)
