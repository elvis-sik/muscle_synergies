"""Types that help the Reader build a representation of Vicon Nexus data."""

import abc
import collections.abc
from collections import defaultdict
import csv
from dataclasses import dataclass
from enum import Enum
import re
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator, Generic, TypeVar, NewType, Union,
                    Iterable)

import pandas as pd
import pint

from .definitions import (
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


# TODO leave this exclusively for TimeSeriesAggregator
class _OnlyOnceMixin:
    def _raise_called_twice(self, what_was_added_twice: str):
        raise TypeError(
            f'attempted to add {what_was_added_twice} after it had ' +
            'been already added')


class TimeSeriesAggregator(_OnlyOnceMixin):
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


class ComponentsAggregator:
    """The aggregator of a single device header.

    Most device headers have exactly 3 columns, like the ones for trajectory
    markers. EMG device headers can have an arbitrary number of columns, each
    referring to a different muscle. What they all have in common is that they
    all have components.

    This class passes along data from each line following the device header
    line to each component. The exact pipeline is this:
    1. the caller finds the columns of the CSV file which refer to the device
       header asssociated with a :py:class:DeviceHeaderAggregator instance.
    2. the caller then passes exactly those columns to that instance using the
       appropriate method
       (e.g., :py:func:DeviceHeaderAggregator.add_coordinates)
    3. this class channels the data forward for different TimeSeriesAggregator
       objects, which are taken during initialization.

    Args:
        time_series_list: the list of :py:class:TimeSeriesAggregator objects
            to which the data has to be passed.
    """
    time_series_tuple: Tuple[TimeSeriesAggregator]

    def __init__(self, time_series_list: List[TimeSeriesAggregator]):
        self.time_series_tuple = tuple(time_series_list)

    def _call_method_on_each(self, parsed_data: List, method_name: str):
        """Calls a method on each time series with the data as argument.

        Args:
            parsed_data: the data, parsed from a line from the CSV input, to be
                passed along to the different :py:class:TimeSeriesAggregator
                instances.

            method_name: the name of the method to be called from each
                :py:class:TimeSeriesAggregator.

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

    def get_time_series(self, ind: Union[int, slice]) -> TimeSeriesAggregator:
        return self.time_series_tuple[ind]

    __getitem__ = get_time_series

    def __len__(self):
        return len(self.time_series_tuple)


# TODO thinking this should actually be obsoleted
# all this logic goes into DevicesState
# DeviceAggregator gets first_col (mandatory), last_col (optional, def None)
# then has a add_num_cols method
# then ReaderState will have to know about number of cols of device types
# and it'll be responsible for identifying device types
# it'll only call:
# - Aggregator.add_force_plate(name, first_col, last_col)
#   plus the 2 equivalents for other types
# then CoordinatesState will call:
# - Aggregator.add_emg_num_cols(num_cols)
# - Aggregator.add_coordinates(coords)

# this also obsoletes ColOfHeader since the relevant information will be
# passed as method arguments

# finally, to get the name of a force plate only get its first device header
# split at '-' and get the first part


class DeviceCols:
    """Intermediate representation of the data read in the device names line.

    This class is used as a way of communicating data from the :py:class:Reader
    to the :py:class:Aggregator. For more information on where the data that
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
                device header if :py:func:DeviceCols.add_num_cols isn't
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
        elif self._device_type is DeviceType.FORCE_PLATE:
            self.num_cols = 9
        self.num_of_cols = 3


class DeviceAggregator:
    """A device header.

    This class keeps track of 2 components referring to individual device
    headers (see :py:class:ViconCSVLines for an explanation of what is a device
    header):
    * :py:class:DeviceCols
    * :py:class:DeviceHeaderAggregator

    The first of those (:py:class:DeviceCols) keeps track of which
    columns from the CSV file refer to that device. The second
    (:py:class:DeviceHeaderAggregator) is accumulates data of different vector
    time series, each of which coming from an individual column of the CSV
    file.

    Args:
        device_cols: the DeviceCols object, which must refer to the
            same device header as `device_aggregator`.

        device_aggregator: the DeviceHeaderAggregator object, which must
            refer to the same device header as `device_cols`
    """
    device_cols: DeviceCols
    components_aggregator: ComponentsAggregator

    def __init__(self, device_cols: DeviceCols,
                 components_aggregator: ComponentsAggregator):
        self.device_cols = device_cols
        self.components_aggregator = components_aggregator

    @classmethod
    def from_col_of_header():
        pass

    @property
    def device_name(self):
        return self.device_cols.device_name

    @property
    def device_type(self):

        return self.device_cols.device_type

    def add_coordinates(self, parsed_row: List[str]):
        """Adds coordinates to device.

        Args:
            parsed_row: the coordinates line of the input.
        """
        self._components_add_coordinates(self._get_my_cols(parsed_row))

    def add_units(self, parsed_row: List[pint.Unit]):
        """Adds physical units to device.

        Args:
            parsed_row: the units line of the input, already parsed.
        """
        self._components_add_units(self._get_my_cols(parsed_row))

    def add_data(self, parsed_row: List[float]):
        """Adds measurements to device.

        Args:
            parsed_row: a data line of the input, already parsed.
        """
        self._components_add_data(self._get_my_cols(parsed_row))

    def _get_my_cols(self, parsed_cols: List[Any]):
        return parsed_cols[self._create_slice()]

    def _create_slice(self):
        return self.device_cols.create_slice()

    def _components_add_coordinates(self, coords: List[str]):
        self.components_aggregator.add_coordinates(coords)

    def _components_add_units(self, units: List[pint.Unit]):
        self.components_aggregator.add_units(units)

    def _components_add_data(self, data: List[float]):
        self.components_aggregator.add_data(data)


class _SectionAggregator(_OnlyOnceMixin, abc.ABC):
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
    def file_ended(self, aggregator: 'Aggregator') -> ViconNexusData:
        pass

    @abc.abstractmethod
    def transition(self, aggregator: 'Aggregator'):
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
                f'tried to add {tried_to_add_what} to a finished _SectionAggregator'
            )


class ForcesEMGAggregator(_SectionAggregator):
    section_type = SectionType.FORCES_EMG
    emg_device: Optional['DeviceHeaderPair']
    force_plates: List['ForcePlateDevices']

    def __init__(self):
        super().__init__()
        self.emg_device = None
        self.force_plates = []

    def add_emg_device(self, emg_device: 'DeviceHeaderPair'):
        if self.emg_device is not None:
            self._raise_called_twice('EMG device')
        self.emg_device = emg_device

    def add_force_plates(self, force_plates: List['ForcePlateDevices']):
        if self.force_plates:
            self._raise_called_twice('list of force plates')
        self.force_plates.extend(force_plates)

    def file_ended(self, aggregator: 'Aggregator') -> ViconNexusData:
        raise ValueError('file ended without a trajectory marker section.')

    def transition(self, aggregator: 'Aggregator'):
        super().transition(aggregator)
        aggregator.set_current_section(next_section_aggregator)


class TrajAggregator(_SectionAggregator):
    section_type = SectionType.TRAJECTORIES
    trajectory_devices: List['DeviceHeaderPair']

    def __init__(self):
        super().__init__()
        self.trajectory_devices = []

    def add_trajectory_devices(self,
                               trajectory_devices: List['DeviceHeaderPair']):
        if self.trajectory_devices:
            self._raise_called_twice('list of trajectory markers')
        self.trajectory_devices.extend(trajectory_devices)

    def transition(self, aggregator: 'Aggregator'):
        super().transition(aggregator)

    def file_ended(self, aggregator: 'Aggregator') -> ViconNexusData:
        frequencies_obj = self._instantiate_frequencies_obj(
            forces_emg_freq=self._forces_emg_freq(aggregator),
            traj_freq=self.frequency,
            num_frames=self._get_num_frames(aggregator),
        )
        force_plates = self._build_force_plate_mapping(aggregator, frequencies)
        emg = self._build_emg_dev_data(aggregator, frequencies)
        trajectory_markers = self._build_trajectory_marker_mapping(
            aggregator, frequencies)

        return self._instantiate_vicon_nexus_data(
            force_plates=force_plates,
            emg=emg,
            trajectory_markers=trajectory_markers)

    def _build_force_plate_mapping(self, aggregator: 'Aggregator',
                                   frequencies: Frequencies
                                   ) -> DeviceMapping[ForcePlateData]:
        converted = []
        devices = self._force_plate_devices(aggregator)
        for device in devices:
            converted.append(
                self._instantiate_force_plate_data(device, frequencies))
        return self._instantiate_device_mapping(devices)

    def _build_emg_dev_data(self, aggregator: 'Aggregator',
                            frequencies: Frequencies
                            ) -> Optional[DeviceHeaderData]:
        emg_pair = self._emg_pair(aggregator)
        if emg_pair is None:
            return
        return self._instantiate_device_header_data(emg_pair, frequencies)

    def _build_trajectory_marker_mapping(self, aggregator: 'Aggregator',
                                         frequencies: Frequencies
                                         ) -> DeviceMapping[DeviceHeaderData]:
        converted = []
        devices = self.trajectory_devices
        for device in devices:
            converted.append(
                self._instantiate_device_header_data(device, frequencies))
        return self._instantiate_device_mapping(devices)

    def _get_num_frames(self) -> int:
        # TODO
        # This is a somewhat fragile solution which would fail if there is no
        # trajectory marker. There also is currently no check on the code
        # anywhere for consistency in the data: do all TimeSeriesAggregator
        # hold the same number of data entries? One consequence of it not being
        # the case would be the possibility that the number of frames here is
        # wrong.

        # I think the best way to do that would be to create a FrameCounter
        # object to be passed to both the Aggregators which then passes it
        # along to the DataChanneler and also has access to it. It could be
        # created when the DeviceCols are created, i.e., the DevicesLine.
        dev_pair = self.trajectory_devices[0]
        dev_aggregator = dev_pair.device_aggregator
        time_series_aggregator = dev_aggregator[0]
        data = time_series_aggregator.get_data()
        return len(data)

    def _forces_emg_freq(self, aggregator: 'Aggregator') -> int:
        return self._get_forces_emg_aggregator(aggregator).frequency

    def _force_plate_devices(self, aggregator: 'Aggregator'
                             ) -> List[ForcePlateDevices]:
        return self._get_forces_emg_aggregator(aggregator).force_plates

    def _emg_pair(self,
                  aggregator: 'Aggregator') -> Optional['DeviceHeaderPair']:
        return self._get_forces_emg_aggregator(aggregator).emg_device

    def _get_forces_emg_aggregator(self, aggregator: 'Aggregator'
                                   ) -> ForcesEMGAggregator:
        return aggregator.get_section_aggregator(SectionType.FORCES_EMG)

    def _instantiate_frequencies_obj(self, *, num_frames, forces_emg_freq,
                                     traj_freq) -> Frequencies:
        return Frequencies(forces_emg_freq, traj_freq, num_frames)

    def _instantiate_force_plate_data(self, force_plate_dev: ForcePlateDevices,
                                      frequencies: Frequencies
                                      ) -> ForcePlateData:
        return ForcePlateData.from_force_plate(force_plate_dev, frequencies)

    def _instantiate_device_header_data(self, dev_pair: 'DeviceHeaderPair',
                                        frequencies: Frequencies
                                        ) -> DeviceHeaderData:
        return DeviceHeaderData(dev_pair, frequencies)

    def _instantiate_device_mapping(
            self, device_list: List[Union[DeviceHeaderData, ForcePlateData]]
    ) -> DeviceMapping[Union[DeviceHeaderData, ForcePlateData]]:
        return DeviceMapping(device_list)

    def _instantiate_vicon_nexus_data(
            self, *, force_plates: DeviceMapping[ForcePlateDevices],
            emg: DeviceHeaderData,
            trajectory_markers: DeviceMapping) -> ViconNexusData:
        return ViconNexusData(force_plates=force_plates,
                              emg=emg,
                              trajectory_markers=trajectory_markers)


class Aggregator:
    _force_emg_aggregator: ForcesEMGAggregator
    _traj_aggregator: TrajAggregator
    _current_aggregator: _SectionAggregator

    def __init__(self, forces_emg_aggregator: _SectionAggregator,
                 trajs_aggregator: _SectionAggregator):
        self._force_emg_aggregator = forces_emg_aggregator
        self._traj_aggregator = trajs_aggregator
        self._current_aggregator = self._force_emg_aggregator

    @property
    def finished(self):
        force_emg_finished = self.get_section_aggregator(
            SectionType.FORCES_EMG).finished
        traj_finished = self.get_section_aggregator(
            SectionType.TRAJECTORIES).finished
        return force_emg_finished and traj_finished

    def get_section_aggregator(self,
                               section_type: Optional[SectionType] = None
                               ) -> _SectionAggregator:
        if section_type is None:
            return self._current_aggregator
        if section_type is SectionType.FORCES_EMG:
            return self._force_emg_aggregator
        if section_type is SectionType.TRAJECTORIES:
            return self._traj_aggregator

    def set_current_section(self, section_type: SectionType):
        assert section_type in SectionType
        self._current_aggregator = self.get_section_aggregator(section_type)

    def file_ended(self) -> ViconNexusData:
        self.get_section_aggregator().file_ended(aggregator=self)

    def get_section_type(self) -> SectionType:
        return self.get_section_aggregator().section_type

    def transition(self):
        self.get_section_aggregator().transition()

    def add_frequency(self, frequency: int):
        self.get_section_aggregator().add_frequency(frequency)

    def add_data_channeler(self, data_channeler: 'DataChanneler'):
        self.get_section_aggregator().add_data_channeler(data_channeler)

    def add_units(self, units: List[pint.Unit]):
        self.get_section_aggregator().add_units(units)

    def add_measurements(self, data: List[float]):
        self.get_section_aggregator().add_data(data)

    def add_emg_device(self, emg_device: 'DeviceHeaderPair'):
        self.get_section_aggregator().add_emg_device(emg_device)

    def add_force_plates(self, force_plates: List['ForcePlateDevices']):
        self.get_section_aggregator().add_force_plates(force_plates)

    def add_trajectory_devices(self,
                               trajectory_devices: List['DeviceHeaderPair']):
        self.get_section_aggregator().add_trajectory_devices(
            trajectory_devices)
