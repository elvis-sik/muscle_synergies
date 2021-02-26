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
    SamplingFreq,
)


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

    def _raise_called_twice(self, what_was_added_twice: str):
        raise TypeError(
            f'attempted to add {what_was_added_twice} after it had ' +
            'been already added')


class DeviceAggregator:
    """Aggregator for the data corresponding to a single device.

    Args:
        name: device name

        device_type: the type of the device

        first_col: the first column in the CSV file corresponding to the device

        last_col: the last column in the CSV file corresponding to the device.
            If `last_col` is None, assume all columns beginning from
            `first_col` belong to the device. In this case, the number of
            columns will be determined the first time data is fed to the device
            using one of the `add_` methods.

        device_aggregator: the DeviceHeaderAggregator object, which must
            refer to the same device header as `device_cols`
    """
    _TIME_SERIES_AGGREGATOR = TimeSeriesAggregator
    name: str
    device_type: DeviceType
    first_col: int
    last_col: Optional[int]
    time_series: Tuple[_TIME_SERIES_AGGREGATOR]

    def __init__(self,
                 name: str,
                 device_type: DeviceType,
                 first_col: int,
                 last_col: Optional[int] = None):
        self.name = name
        self.device_type = device_type
        self.first_col = first_col
        self.last_col = last_col
        if last_col is not None:
            self._initialize_time_series()

    def add_coordinates(self, parsed_row: List[str]):
        """Add coordinates to device.

        Args:
            parsed_row: the coordinates line of the input.
        """
        self._call_method_on_each(self._time_series_add_data,
                                  self._my_cols(parsed_row))

    def add_units(self, parsed_row: List[pint.Unit]):
        """Add physical units to device.

        Args:
            parsed_row: the units line of the input, already parsed.
        """
        self._call_method_on_each(self._time_series_add_data,
                                  self._my_cols(parsed_row))

    def add_data(self, parsed_row: List[float]):
        """Add measurements to device.

        Args:
            parsed_row: a data line of the input, already parsed.
        """
        self._call_method_on_each(self._time_series_add_data,
                                  self._my_cols(parsed_row))

    def _my_cols(self, parsed_cols: List[Any]) -> List[Any]:
        """Restrict parsed columns to the ones corresponding to device."""
        if self.time_series is None:
            assert self.last_col is None
            self.last_col = len(parsed_cols) - 1
            self._initialize_time_series()
        return parsed_cols[self._create_slice()]

    def _call_method_on_each(
            self, method: Callable[[_TIME_SERIES_AGGREGATOR, Any], None],
            parsed_data: List):
        """Calls a method on each time series with the data as argument.

        Args:
            parsed_data: the columns of the data referring to the device.

            method: the method to be called on each
                :py:class:TimeSeriesAggregator.
        """
        for data_entry, time_series in zip(parsed_data, self.time_series):
            method(time_series, data_entry)

    def _create_slice(self):
        """Create a slice object corresponding to the device columns."""
        return slice(self.first_col, self.last_col + 1)

    def _initialize_time_series(self):
        num_cols = last_col - first_col + 1
        self.time_series = tuple(self._create_time_series_aggregator()
                                 for _ in range(num_cols))

    def _create_time_series_aggregator(self) -> _TIME_SERIES_AGGREGATOR:
        return self._TIME_SERIES_AGGREGATOR()

    def _time_series_add_coordinate(self, time_series: _TIME_SERIES_AGGREGATOR,
                                    data_entry: str):
        time_series.add_coordinate(data_entry)

    def _time_series_add_units(self, time_series: _TIME_SERIES_AGGREGATOR,
                               data_entry: pint.Unit):
        time_series.add_unit(data_entry)

    def _time_series_add_data(self, time_series: _TIME_SERIES_AGGREGATOR,
                              data_entry: float):
        time_series.add_data(data_entry)

    def __getitem__(self, ind: int) -> _TIME_SERIES_AGGREGATOR:
        return self.time_series[ind]

    def __len__(self):
        return len(self.time_series)


class _SectionAggregator(_OnlyOnceMixin, abc.ABC):
    finished: bool
    frequency: Optional[int]
    devices: List[DeviceAggregator]

    def __init__(self):
        super().__init__()
        self._finished = False
        self.frequency = None
        self.devices = []

    @abc.abstractproperty
    def section_type(self) -> SectionType:
        return

    @abc.abstractmethod
    def transition(self, aggregator: 'Aggregator'):
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def add_device(self, name: str, device_type: DeviceType, first_col: int,
                   last_col: Optional[int]):
        self._raise_if_finished()
        self.devices.append(
            self._instantiate_device_aggregator(name, device_type, first_col,
                                                last_col))

    def add_frequency(self, frequency: int):
        self._raise_if_finished()
        self.frequency = frequency

    def add_coordinates(self, coords: List[str]):
        self._raise_if_finished()

        for device in self.devices:
            device.add_coordinates(units)

    def add_units(self, units: List[pint.Unit]):
        self._raise_if_finished()

        for device in self.devices:
            device.add_units(units)

    def add_data(self, data: List[float]):
        self._raise_if_finished()

        for device in self.devices:
            device.add_data(data)

    def _instantiate_device_aggregator(self, name: str,
                                       device_type: DeviceType, first_col: int,
                                       last_col: Optional[int]
                                       ) -> DeviceAggregator:
        return DeviceAggregator(name, device_type, first_col, last_col)

    def _raise_if_finished(self):
        if self.finished:
            raise TypeError(
                f'tried to add something to a finished _SectionAggregator')


class ForcesEMGAggregator(_SectionAggregator):
    section_type = SectionType.FORCES_EMG

    def transition(self, aggregator: 'Aggregator'):
        super().transition(aggregator)
        aggregator.set_current_section(SectionType.TRAJECTORIES)


class TrajAggregator(_SectionAggregator):
    section_type = SectionType.TRAJECTORIES

    def __init__(self):
        super().__init__()
        self._num_rows = 0

    def add_data(self, data: List[float]):
        self._num_rows += 1
        super().add_data(data)

    def get_num_rows(self) -> int:
        return self._num_rows

    def transition(self, aggregator: 'Aggregator'):
        super().transition(aggregator)
        aggregator.set_current_section(None)


class Aggregator:
    _force_emg_aggregator: ForcesEMGAggregator
    _traj_aggregator: TrajAggregator
    _current_aggregator: Optional[_SectionAggregator]

    def __init__(self,
                 forces_emg_agg: Optional[_SectionAggregator] = None,
                 trajs_agg: Optional[_SectionAggregator] = None):
        if forces_emg_agg is None:
            forces_emg_agg = ForcesEMGAggregator()
        if trajs_agg is None:
            trajs_agg = TrajAggregator()

        self._force_emg_aggregator = forces_emg_agg
        self._traj_aggregator = trajs_agg
        self._current_aggregator = self._force_emg_aggregator

    @property
    def finished(self):
        force_emg_finished = self._get_section_aggregator(
            SectionType.FORCES_EMG).finished
        traj_finished = self._get_section_aggregator(
            SectionType.TRAJECTORIES).finished
        return force_emg_finished and traj_finished

    def get_sampling_freq(self) -> SamplingFreq:
        forces_emg_agg = self._get_section_aggregator(SectionType.FORCES_EMG)
        traj_agg = self._get_section_aggregator(SectionType.TRAJECTORIES)

        freq_forces_emg = forces_emg_agg.frequency
        freq_traj = traj_agg.frequency
        num_frames = traj_agg.get_num_rows()
        return SamplingFreq(freq_forces_emg, freq_traj, num_frames)

    def get_devices(self) -> Sequence[DeviceAggregator]:
        forces_emg = self._get_section_aggregator(
            SectionType.FORCES_EMG).devices
        traj = self._get_section_aggregator(SectionType.TRAJECTORIES).devices
        return forces_emg + traj

    def _get_section_aggregator(self,
                                section_type: Optional[SectionType] = None
                                ) -> _SectionAggregator:
        if section_type is None:
            return
        if section_type is SectionType.FORCES_EMG:
            return self._force_emg_aggregator
        if section_type is SectionType.TRAJECTORIES:
            return self._traj_aggregator

    def set_current_section(self, section_type: Optional[SectionType]):
        assert (section_type in SectionType) or (section_type is None)
        self._current_aggregator = self._get_section_aggregator(section_type)

    def get_section_type(self) -> SectionType:
        return self._get_section_aggregator().section_type

    def transition(self):
        self._get_section_aggregator().transition()

    def add_frequency(self, frequency: int):
        self._get_section_aggregator().add_frequency(frequency)

    def add_units(self, units: List[pint.Unit]):
        self._get_section_aggregator().add_units(units)

    def add_data(self, data: List[float]):
        self._get_section_aggregator().add_data(data)

    def add_device(self, name: str, device_type: DeviceType, first_col: int,
                   last_col: Optional[int]):
        self._get_section_aggregator().add_device(name, device_type, first_col,
                                                  last_col)
