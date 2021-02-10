"""Types that help the Reader build a representation of Vicon Nexus data."""

import abc
from collections import defaultdict
import csv
from dataclasses import dataclass
from enum import Enum
import re
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator, Generic, TypeVar, NewType, Union)

import pint

from definitions import (
    ureg,
    T,
    X,
    Y,
    Row,
    DeviceHeaderRepresentation,
    ForcePlateRepresentation,
    SectionType,
    ViconCSVLines,
    DeviceType,
    ForcePlateMeasurement,
)


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
    def file_ended(self, data_builder: DataBuilder) -> ViconNexusData:
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

    def file_ended(self, data_builder: DataBuilder) -> ViconNexusData:
        raise ValueError('file ended without a trajectory marker section.')

    def transition(self, data_builder: DataBuilder):
        super().transition(data_builder)
        data_builder.set_current_section(next_section_builder)


class TrajDataBuilder(_SectionDataBuilder):
    section_type = SectionType.TRAJECTORIES
    _frequencies_type = Frequencies

    # TODO
    # 1. somehow have a way for databuilder sections to know
    #    devices by their type and that forceplates must go together
    #    (I'm thinking specialized add_ methods)
    # 2. finish building data here
    # 3. finish the devices line guys
    # 4. finish the routine that loads everything up from a CSV
    # 5. implement a simple integration test
    # 6. fix bugs one by one
    # 7. start abstracting unit tests
    # 8. write an example notebook
    def transition(self, data_builder: DataBuilder):
        super().transition(data_builder)

    def file_ended(self, data_builder: DataBuilder) -> ViconNexusData:
        frequencies_obj = self._instantiate_frequencies_obj(
            forces_emg_freq=self._forces_emg_freq(data_builder),
            traj_freq=self.frequency,
            num_frames=self._get_num_frames(data_builder),
        )

    def _get_num_frames(self) -> int:
        pass

    def _instantiate_frequencies_obj(self, *, num_frames, forces_emg_freq,
                                     traj_freq) -> _frequencies_type:
        return Frequencies(forces_emg_freq, traj_freq, num_frames)

    def _forces_emg_freq(self, data_builder: DataBuilder) -> int:
        return self._get_forces_emg_builder(data_builder).frequency

    def _get_forces_emg_builder(self, data_builder: DataBuilder
                                ) -> ForcesEMGDataBuilder:
        return data_builder.get_section_builder(SectionType.FORCES_EMG)


class DataBuilder:
    _force_emg_builder: ForcesEMGDataBuilder
    _traj_builder: TrajDataBuilder
    _current_builder: _SectionDataBuilder

    def __init__(self, forces_emg_data_builder: SectionDataBuilder,
                 trajs_data_builder: SectionDataBuilder):
        self._force_emg_builder = forces_emg_data_builder
        self._traj_builder = trajs_data_builder

    @property
    def finished(self):
        force_emg_finished = self.get_section_builder(
            SectionType.FORCES_EMG).finished
        traj_finished = self.get_section_builder(
            SectionType.TRAJECTORIES).finished
        return force_emg_finished and traj_finished

    def get_section_builder(self, section_type: Optional[SectionType] = None
                            ) -> _SectionDataBuilder:
        if section_type is None:
            return self._current_builder
        if section_type is SectionType.FORCES_EMG:
            return self._force_emg_builder
        if section_type is SectionType.TRAJECTORIES:
            return self._traj_builder

    def set_current_section(self, section_type: SectionType):
        assert section_type in SectionType
        self._current_builder = self.get_section_builder(section_type)

    def file_ended(self) -> ViconNexusData:
        self.get_section_builder().file_ended(data_builder=self)

    def get_section_type(self) -> SectionType:
        return self.get_section_builder().section_type

    def transition(self):
        self.get_section_builder().transition()

    def add_frequency(self, frequency: int):
        self.get_section_builder().add_frequency(frequency)

    def add_data_channeler(self, data_channeler: 'DataChanneler'):
        self.get_section_builder().add_data_channeler(data_channeler)

    def add_units(self, units: List[pint.Unit]):
        self.get_section_builder().add_units(units)

    def add_measurements(self, data: List[float]):
        self.get_section_builder().add_data(data)


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
