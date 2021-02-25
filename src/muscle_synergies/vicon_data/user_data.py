import abc
from dataclasses import dataclass
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator, Generic, TypeVar, NewType, Union,
                    Iterable)

from .aggregator import (
    Aggregator,
    ForcesEMGAggregator,
    TrajAggregator,
    DeviceAggregator,
)
from .definitions import (
    DeviceType,
    SectionType,
    SamplingFreq,
)


@dataclass
class ViconNexusData:
    force_plates: Tuple['DeviceData']
    emg: 'DeviceData'
    trajectory_markers: Tuple['DeviceData']


class Builder:
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def build(self, aggregator: Optional[Aggregator] = None) -> ViconNexusData:
        if aggregator is None:
            aggregator = self.aggregator

        frame_tracker = self._build_frame_tracker(aggregator)

        devices_by_type = defaultdict(list)
        for device_agg in self._devices(aggregator):
            device_data = self._build_device(device_agg)
            devices_by_type[device.device_type].append(device_data)

        return self._vicon_nexus_data(devices_by_type)

    def _build_device(self, device_agg: DeviceAggregator,
                      frame_tracker: '_SectionFrameTracker') -> 'DeviceData':
        device_name = device_agg.device_name
        device_type = device_agg.device_type
        dataframe = self._extract_dataframe(device_agg)
        return self._instantiate_device(device_name, device_type,
                                        frame_tracker, dataframe)

    def _build_frame_tracker(
            self, aggregator: Aggregator
    ) -> Tuple['ForcesEMGFrameTracker', 'TrajFrameTracker']:
        sampling_freq = self._aggregator_sampling_freq(aggregator)
        return (ForcesEMGFrameTracker(sampling_freq),
                TrajFrameTracker(sampling_freq))

    @staticmethod
    def _instantiate_device(device_name: str, device_type: DeviceType,
                            frame_tracker: '_SectionFrameTracker',
                            dataframe: pd.DataFrame) -> 'DeviceData':
        return DeviceData(device_name=device_name,
                          device_type=device_type,
                          frame_tracker=frame_tracker,
                          dataframe=dataframe)

    @staticmethod
    def _extract_dataframe(device_aggregator: DeviceAggregator,
                           ) -> pd.DataFrame:
        def create_pint_array(data, physical_unit):
            PintArray(data, dtype=physical_unit)

        data_dict = {}
        for time_series_aggregator in device_aggregator:
            coord_name = time_series_aggregator.get_coordinate_name()
            physical_unit = time_series_aggregator.get_physical_unit()
            data = time_series_aggregator.get_data()
            data_dict[coord_name] = create_pint_parray(data, physical_unit)

        return pd.DataFrame(data_dict)

    def _aggregator_sampling_freq(self,
                                  aggregator: Aggregator) -> 'SamplingFreq':
        return aggregator.get_sampling_freq()

    def _vicon_nexus_data(
            self, devices_by_type: Mapping[DeviceType, List['DeviceData']]
    ) -> ViconNexusData:

        return ViconNexusData(
            force_plates=DeviceType.FORCE_PLATE,
            emg=DeviceType.EMG,
            trajectory_markers=DeviceType.TRAJECTORY_MARKER,
        )

    def _devices(self, aggregator: Aggregator) -> Iterator[DeviceAggregator]:
        yield from aggregator.get_devices()


class FrameTracker:
    def __init__(self, forces_emg: 'ForcesEMGFrameTracker',
                 traj: 'TrajFrameTracker'):
        self._forces_emg = forces_emg
        self._traj = traj

    def get_num_frames():
        pass


class _SectionFrameTracker(abc.ABC):
    def __init__(self, sampling_freq=SamplingFreq):
        self._sampling_freq = sampling_freq

    @property
    def num_frames(self) -> int:
        return self._sampling_freq.num_frames

    @abc.abstractproperty
    def sampling_frequency(self) -> int:
        pass

    @abc.abstractmethod
    def index(self, frame: int, subframe: int) -> int:
        self._validate_frame_tracker_args(frame, subframe)

    @abc.abstractmethod
    def frame_tracker(self, index: int) -> Tuple[int, int]:
        self._validate_index_arg(index)

    @abc.abstractproperty
    def final_index(self) -> int:
        pass

    @property
    def num_subframes(self) -> int:
        num_subframes = self._freq_forces_emg / self._freq_traj
        assert num_subframes == int(num_subframes)
        return int(num_subframes)

    @property
    def _freq_forces_emg(self) -> int:
        return self._sampling_freq.freq_forces_emg

    @property
    def _freq_traj(self) -> int:
        return self._sampling_freq.freq_traj

    def _validate_index_arg(self, index: int):
        if index not in range(self.final_index + 1):
            raise ValueError(
                f'index {index} out of bounds (max is self.final_index)')

    def _validate_frame_tracker_args(self, frame: int, subframe: int):
        if frame not in range(1, self.num_frames + 1):
            raise ValueError(
                f'last frame is {self.num_frames}, frame {frame} is out of bounds'
            )
        if subframe not in range(self.num_subframes):
            raise ValueError(
                f'subframe {subframe} out of range {self.subframe_range()}')


class ForcesEMGFrameTracker(_SectionFrameTracker):
    @property
    def sampling_frequency(self) -> int:
        return self._freq_forces_emg

    def index(self, frame: int, subframe: int) -> int:
        super().index(frame, subframe)
        return (frame - 1) * self.num_subframes + subframe

    def frame_tracker(self, index: int) -> Tuple[int, int]:
        super().frame_tracker(index)
        frame = (index // self.num_subframes) + 1
        subframe = index % self.num_subframes
        return frame, subframe

    @property
    def final_index(self) -> int:
        return self.num_frames * self.num_subframes - 1


class TrajFrameTracker(_SectionFrameTracker):
    @property
    def sampling_frequency(self) -> int:
        return self._freq_traj

    def index(self, frame: int, subframe: int) -> int:
        super().index(frame, subframe)
        return frame - 1

    def frame_tracker(self, index: int) -> Tuple[int, int]:
        super().frame_tracker(index)
        return index + 1, 0

    @property
    def final_index(self) -> int:
        return self.num_frames - 1


class DeviceData:
    def __init__(
            self,
            device_name: str,
            device_type: DeviceType,
            frame_tracker: _SectionFrameTracker,
            dataframe: pd.DataFrame,
    ):
        self.device_name = device_name
        self.device_type = device_type
        self._frame_tracker = frame_tracker
        self.df = dataframe

    @property
    def sampling_frequency(self) -> int:
        return self._frame_tracker.sampling_frequency

    def _create_frame_slice(self,
                            *,
                            stop_frame: int,
                            stop_subframe: int,
                            start_frame: Optional[int] = None,
                            start_subframe: Optional[int] = None,
                            step: Optional[int] = None) -> slice:
        stop_index = self._frame_tracker_index(stop_frame, stop_subframe)
        if start_frame is None:
            return slice(stop_index)

        start_index = self._frame_tracker_index(start_frame, start_subframe)
        if step is None:
            return slice(start_index, stop_index)
        return slice(start_index, stop_index, step)

    def _frame_tracker_index(self, frame: int, subframe: int) -> int:
        return self._frame_tracker.index(self.device_type, frame, subframe)
