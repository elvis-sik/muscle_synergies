"""Types that help the Reader build a representation of Vicon Nexus data.

The main class here is :py:class:`Aggregator`, which accepts messages relating
to the data being read and pass them forward to instances of other classes
(with names ending in `Aggregator`), which then become responsible for storing
those values.  The different classes are organized as a tree.

The CSV file has 2 sections (see
:py:class:`~muscle_synergies.vicon_data.definitions.SectionType`) the data in
which is similar and has to be processed similarly but there are some
differences between them. For this reason, there are 2 children of the
:py:class:`_SectionAggregator` ABC.

The data in the CSV file is classified according to the measuring device and
for that reason there is a :py:class:`DeviceAggregator`, which keeps track of
which columns belong to each device and stores that data as it is being read.

Refer to the documentation for the package
:py:mod:`muscle_synergies.vicon_data` for more on how `Aggregator`
fits together with the other classes used for reading the data from disk.
"""

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

    Attributes:
        coords: the coordinates (like `['Fx', 'Fy', 'Fz']`) of the device data.

        units: the physical units (like `['N', 'N', 'N']`) of the device data.

        data_rows: the actual time series of measurements corresponding to the
            device.
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
        """Restrict parsed columns to the ones corresponding to device.

        This uses the member `_num_cols` which is `None` if `last_col` is not
        passed during initialization.  If it in fact is `None` when `_my_cols`
        is called, `_num_cols` is initialized  by assuming that all the columns
        until the final one belong to the device.

        The reason for this is that unlike the other device types, which have a
        fixed number of columns, there can be a variable number of EMG columns.
        The EMG data is assumed to follow the one for force plates.
        """
        if self._num_cols is None:
            assert self.last_col is None
            self.last_col = len(parsed_cols) - 1
            self._initialize_num_cols()
        return parsed_cols[self._create_slice()]

    def _create_slice(self) -> slice:
        """Create a slice object corresponding to the device columns."""
        return slice(self.first_col, self.last_col + 1)

    def _initialize_num_cols(self):
        """Initalize the member storing the number of columns of the device."""
        self._num_cols = self.last_col - self.first_col + 1


class _SectionAggregator(abc.ABC):
    """Aggregator that stores data of a single section of the CSV file.

    Attributes:
        section_type: the type of the section whose data is stored in the
            `_SectionAggregator`.

        finished: whether the data in the section has finished being parsed.

        devices: stores the `DeviceAggregators` that occur in the section after
            they're first read.

        frequency: stores the sampling frequency in Hz after it is read.
    """

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
        """Transition Aggregator to its next section state."""
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
        """Add a new device that belongs to the section.

        The arguments are passed as they are to the initialization of
        :py:class:`DeviceAggregator`. The newly created instance is stored in the
        `self.devices` member.

        Raises:
            TypeError: if `self.finished` is True.
        """
        self._raise_if_finished()
        self.devices.append(
            self._instantiate_device_aggregator(name, device_type, first_col, last_col)
        )

    def add_frequency(self, frequency: int):
        """Add the sampling frequency for the data measurements in the section.

        Args:
            frequency: the sampling frequency.

        Raises:
            TypeError: if `self.finished` is True.
        """
        self._raise_if_finished()
        self.frequency = frequency

    def add_coordinates(self, coords: List[str]):
        """Add the coordinates of each data column in the section.

        Args:
            coords: the data from the coordinates line.

        Raises:
            TypeError: if `self.finished` is True.
        """
        self._raise_if_finished()

        for device in self.devices:
            device.add_coordinates(coords)

    def add_units(self, units: List[str]):
        """Add the units of each data column in the section.

        Args:
            units: the data from the units line.

        Raises:
            TypeError: if `self.finished` is True.
        """
        self._raise_if_finished()

        for device in self.devices:
            device.add_units(units)

    def add_data(self, data: List[float]):
        """Add a new line of measurements from the section.

        Args:
            data: the data contained in one of the data lines.

        Raises:
            TypeError: if `self.finished` is True.
        """
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
        """Create new DeviceAggregator instance.

        The arguments are passed directly to :py:class:`DeviceAggregator`.
        """
        return DeviceAggregator(name, device_type, first_col, last_col)

    def _raise_if_finished(self):
        """Raise TypeError if the section is over.

        Raises:
            TypeError: if `self.finished` is True.
        """
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
        """Get number of data rows fed.

        Each call to `add_data` counts as one data row.
        """
        return self._num_rows

    def transition(self, aggregator: "Aggregator"):
        super().transition(aggregator)
        aggregator.set_current_section(None)


class Aggregator:
    """Aggregate data as it is parsed line-by-line from the CSV file.

    Args:
        forces_emg_agg: if None, a new :py:class:`ForcesEMGAggregator` instance
            will be created

        trajs_agg: if None, a new :py:class:`TrajAggregator` instance will be
            created

    Attributes:
        finished: True if both sections have ended.
    """

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
    def finished(self) -> bool:
        force_emg_finished = self._get_section_aggregator(
            SectionType.FORCES_EMG
        ).finished
        traj_finished = self._get_section_aggregator(SectionType.TRAJECTORIES).finished
        return force_emg_finished and traj_finished

    def get_sampling_freq(self) -> SamplingFreq:
        """Get the sampling rate of each section and the number of frames.

        The number of frames is determined simply as the number of data lines
        in the trajectories section.
        """
        forces_emg_agg = self._get_section_aggregator(SectionType.FORCES_EMG)
        traj_agg = self._get_section_aggregator(SectionType.TRAJECTORIES)

        freq_forces_emg = forces_emg_agg.frequency
        freq_traj = traj_agg.frequency
        num_frames = traj_agg.get_num_rows()
        return SamplingFreq(freq_forces_emg, freq_traj, num_frames)

    def get_devices(self) -> Sequence[DeviceAggregator]:
        """Get all DeviceAggregator from both sections."""
        forces_emg = self._get_section_aggregator(SectionType.FORCES_EMG).devices
        traj = self._get_section_aggregator(SectionType.TRAJECTORIES).devices
        return forces_emg + traj

    def _get_section_aggregator(
        self, section_type: Optional[SectionType] = None
    ) -> Optional[_SectionAggregator]:
        """Get current section aggregator or the specified one.

        Args:
            section_type: if None, return the current section aggregator.
                Otherwise, return the section aggregator of the given section
                type.
        """
        if section_type is None:
            return self._current_aggregator
        if section_type is SectionType.FORCES_EMG:
            return self._force_emg_aggregator
        if section_type is SectionType.TRAJECTORIES:
            return self._traj_aggregator
        return self._current_aggregator

    def set_current_section(self, section_type: Optional[SectionType]):
        """Set the current section aggregator to have the specified type.

        Args:
            section_type: if None, set the current section aggregator to None,
                marking the process as finished. Otherwise, set the section
                aggregator of the given section type as the current one.
        """
        assert (section_type in SectionType) or (section_type is None)
        if section_type is None:
            self._current_aggregator = None
        else:
            self._current_aggregator = self._get_section_aggregator(section_type)

    def get_section_type(self) -> SectionType:
        """Get the type of the current section.

        This is done by checking which is the type of the current section
        aggregator.

        Raises:
            AttributeError: if the current section aggregator is None, as would
                be the case after the data for both sections has been
                completely fed to `Aggregator`.
        """
        return self._get_section_aggregator().section_type

    def transition(self):
        """Ask current section aggregator to transition to next section."""
        self._get_section_aggregator().transition(aggregator=self)

    def add_frequency(self, frequency: int):
        """Add sampling frequency.

        The arguments are passed as they are to
        :py:meth:`_SectionAggregator.add_frequency`.
        """
        self._get_section_aggregator().add_frequency(frequency)

    def add_coordinates(self, coordinates: List[str]):
        """Add the names of the coordinates of the data.

        The arguments are passed as they are to
        :py:meth:`_SectionAggregator.add_coordinates`.
        """
        self._get_section_aggregator().add_coordinates(coordinates)

    def add_units(self, units: List[str]):
        """Add the physical units of the different columns of data.

        The arguments are passed as they are to
        :py:meth:`_SectionAggregator.add_units`.
        """
        self._get_section_aggregator().add_units(units)

    def add_data(self, data: List[float]):
        """Add a new data line.

        The arguments are passed as they are to
        :py:meth:`_SectionAggregator.add_data`.
        """
        self._get_section_aggregator().add_data(data)

    def add_device(
        self,
        name: str,
        device_type: DeviceType,
        first_col: int,
        last_col: Optional[int],
    ):
        """Add a new measuring device.

        The arguments are passed as they are to
        :py:meth:`_SectionAggregator.add_device`.
        """
        self._get_section_aggregator().add_device(
            name, device_type, first_col, last_col
        )
