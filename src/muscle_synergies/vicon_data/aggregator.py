"""Types that help the Reader build a representation of Vicon Nexus data."""

import abc
from typing import Any, List, Optional, Sequence

from .definitions import DeviceType, SamplingFreq, SectionType


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

    name: str
    device_type: DeviceType
    first_col: int
    last_col: Optional[int]

    coords: Optional[List[str]]
    units: Optional[List[str]]
    data_rows: List[List[float]]

    _num_cols: Optional[int]

    def __init__(
        self,
        name: str,
        device_type: DeviceType,
        first_col: int,
        last_col: Optional[int] = None,
    ):
        self.name = name
        self.device_type = device_type
        self.first_col = first_col
        self.last_col = last_col
        self.coords = None
        self.units = None
        self._num_cols = None
        self.data_rows = []
        if self.last_col is not None:
            self._initialize_num_cols()

    def add_coordinates(self, parsed_row: List[str]):
        """Add coordinates to device.

        Args:
            parsed_row: the coordinates line of the input.
        """
        self.coords = self._my_cols(parsed_row)

    def add_units(self, parsed_row: List[str]):
        """Add physical units to device.

        Args:
            parsed_row: the units line of the input, already parsed.
        """
        self.units = self._my_cols(parsed_row)

    def add_data(self, parsed_row: List[float]):
        """Add measurements to device.

        Args:
            parsed_row: a data line of the input, already parsed.
        """
        self.data_rows.append(self._my_cols(parsed_row))

    def _my_cols(self, parsed_cols: List[Any]) -> List[Any]:
        """Restrict parsed columns to the ones corresponding to device."""
        if self._num_cols is None:
            assert self.last_col is None
            self.last_col = len(parsed_cols) - 1
            self._initialize_num_cols()
        return parsed_cols[self._create_slice()]

    def _create_slice(self):
        """Create a slice object corresponding to the device columns."""
        return slice(self.first_col, self.last_col + 1)

    def _initialize_num_cols(self):
        self._num_cols = self.last_col - self.first_col + 1


class _SectionAggregator(abc.ABC):
    frequency: Optional[int]
    devices: List[DeviceAggregator]

    def __init__(self):
        super().__init__()
        self._finished = False
        self.frequency = None
        self.devices = []

    @abc.abstractproperty
    def section_type(self) -> SectionType:
        pass

    @abc.abstractmethod
    def transition(self, aggregator: "Aggregator"):
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def add_device(
        self,
        name: str,
        device_type: DeviceType,
        first_col: int,
        last_col: Optional[int],
    ):
        self._raise_if_finished()
        self.devices.append(
            self._instantiate_device_aggregator(name, device_type, first_col, last_col)
        )

    def add_frequency(self, frequency: int):
        self._raise_if_finished()
        self.frequency = frequency

    def add_coordinates(self, coords: List[str]):
        self._raise_if_finished()

        for device in self.devices:
            device.add_coordinates(coords)

    def add_units(self, units: List[str]):
        self._raise_if_finished()

        for device in self.devices:
            device.add_units(units)

    def add_data(self, data: List[float]):
        self._raise_if_finished()

        for device in self.devices:
            device.add_data(data)

    def _instantiate_device_aggregator(
        self,
        name: str,
        device_type: DeviceType,
        first_col: int,
        last_col: Optional[int],
    ) -> DeviceAggregator:
        return DeviceAggregator(name, device_type, first_col, last_col)

    def _raise_if_finished(self):
        if self.finished:
            raise TypeError("tried to add something to a finished _SectionAggregator")


class ForcesEMGAggregator(_SectionAggregator):
    section_type = SectionType.FORCES_EMG

    def transition(self, aggregator: "Aggregator"):
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

    def transition(self, aggregator: "Aggregator"):
        super().transition(aggregator)
        aggregator.set_current_section(None)


class Aggregator:
    _force_emg_aggregator: ForcesEMGAggregator
    _traj_aggregator: TrajAggregator
    _current_aggregator: Optional[_SectionAggregator]

    def __init__(
        self,
        forces_emg_agg: Optional[ForcesEMGAggregator] = None,
        trajs_agg: Optional[TrajAggregator] = None,
    ):
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
            SectionType.FORCES_EMG
        ).finished
        traj_finished = self._get_section_aggregator(SectionType.TRAJECTORIES).finished
        return force_emg_finished and traj_finished

    def get_sampling_freq(self) -> SamplingFreq:
        forces_emg_agg = self._get_section_aggregator(SectionType.FORCES_EMG)
        traj_agg = self._get_section_aggregator(SectionType.TRAJECTORIES)

        freq_forces_emg = forces_emg_agg.frequency
        freq_traj = traj_agg.frequency
        num_frames = traj_agg.get_num_rows()
        return SamplingFreq(freq_forces_emg, freq_traj, num_frames)

    def get_devices(self) -> Sequence[DeviceAggregator]:
        forces_emg = self._get_section_aggregator(SectionType.FORCES_EMG).devices
        traj = self._get_section_aggregator(SectionType.TRAJECTORIES).devices
        return forces_emg + traj

    def _get_section_aggregator(
        self, section_type: Optional[SectionType] = None
    ) -> Optional[_SectionAggregator]:
        if section_type is SectionType.FORCES_EMG:
            return self._force_emg_aggregator
        if section_type is SectionType.TRAJECTORIES:
            return self._traj_aggregator
        return self._current_aggregator

    def set_current_section(self, section_type: Optional[SectionType]):
        assert (section_type in SectionType) or (section_type is None)
        if section_type is None:
            self._current_aggregator = None
        else:
            self._current_aggregator = self._get_section_aggregator(section_type)

    def get_section_type(self) -> SectionType:
        return self._get_section_aggregator().section_type

    def transition(self):
        self._get_section_aggregator().transition(aggregator=self)

    def add_frequency(self, frequency: int):
        self._get_section_aggregator().add_frequency(frequency)

    def add_coordinates(self, coordinates: List[str]):
        self._get_section_aggregator().add_coordinates(coordinates)

    def add_units(self, units: List[str]):
        self._get_section_aggregator().add_units(units)

    def add_data(self, data: List[float]):
        self._get_section_aggregator().add_data(data)

    def add_device(
        self,
        name: str,
        device_type: DeviceType,
        first_col: int,
        last_col: Optional[int],
    ):
        self._get_section_aggregator().add_device(
            name, device_type, first_col, last_col
        )
