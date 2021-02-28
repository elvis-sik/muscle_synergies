import abc
from collections import defaultdict
from dataclasses import dataclass
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator, Generic, TypeVar, NewType, Union,
                    Iterable)

import pandas as pd

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
    force_plates: Sequence['DeviceData']
    emg: 'DeviceData'
    trajectory_markers: Sequence['DeviceData']


class Builder:
    def __init__(self, aggregator: Aggregator):
        self.aggregator = aggregator

    def build(self, aggregator: Optional[Aggregator] = None) -> ViconNexusData:
        if aggregator is None:
            aggregator = self.aggregator

        frame_tracker = self._build_frame_tracker(aggregator)

        devices_by_type = defaultdict(list)
        for device_agg in self._devices(aggregator):
            device_data = self._build_device(device_agg, frame_tracker)
            device_type = self._device_agg_type(device_agg)
            devices_by_type[device_type].append(device_data)

        return self._vicon_nexus_data(self._simplify_emg(devices_by_type))

    def _build_device(
            self, device_agg: DeviceAggregator,
            frame_tracker: Tuple['ForcesEMGFrameTracker', 'TrajFrameTracker']
    ) -> 'DeviceData':
        params_dict = self._params_device_data(device_agg, frame_tracker)
        return self._instantiate_device(**params_dict)

    def _params_device_data(
            self, device_agg: DeviceAggregator,
            frame_tracker: Tuple['ForcesEMGFrameTracker', 'TrajFrameTracker']
    ) -> Mapping[str, Union[str, DeviceType, '_SectionFrameTracker', pd.
                            DataFrame]]:
        return {
            'device_name': self._device_agg_name(device_agg),
            'device_type': self._device_agg_type(device_agg),
            'units': self._device_agg_units(device_agg),
            'frame_tracker':
            self._choose_frame_tracker(device_agg, *frame_tracker),
            'dataframe': self._extract_dataframe(device_agg)
        }

    def _build_frame_tracker(
            self, aggregator: Aggregator
    ) -> Tuple['ForcesEMGFrameTracker', 'TrajFrameTracker']:
        sampling_freq = self._aggregator_sampling_freq(aggregator)
        return (ForcesEMGFrameTracker(sampling_freq),
                TrajFrameTracker(sampling_freq))

    @staticmethod
    def _instantiate_device(device_name: str, device_type: DeviceType,
                            units: List[str],
                            frame_tracker: '_SectionFrameTracker',
                            dataframe: pd.DataFrame) -> 'DeviceData':
        return DeviceData(device_name=device_name,
                          device_type=device_type,
                          units=units,
                          frame_tracker=frame_tracker,
                          dataframe=dataframe)

    @classmethod
    def _extract_dataframe(cls, device_aggregator: DeviceAggregator
                           ) -> pd.DataFrame:
        data = cls._device_agg_data(device_aggregator)
        header = cls._device_agg_coords(device_aggregator)
        return pd.DataFrame(data, columns=header, dtype=float)

    def _simplify_emg(
            self, devices_by_type: Mapping[DeviceType, List['DeviceData']]
    ) -> Mapping[DeviceType, Union['DeviceData', List['DeviceData']]]:
        devices_by_type = dict(devices_by_type)
        emg: List['DeviceData'] = devices_by_type[DeviceType.EMG]
        if len(emg) != 1:
            raise ValueError(f'found {len(emg)} EMG devices - expected one')
        devices_by_type[DeviceType.EMG] = emg[0]
        return devices_by_type

    def _vicon_nexus_data(
            self,
            devices_by_type: Mapping[DeviceType,
                                     Union['DeviceData', List['DeviceData']]]
    ) -> ViconNexusData:

        return ViconNexusData(
            force_plates=devices_by_type[DeviceType.FORCE_PLATE],
            emg=devices_by_type[DeviceType.EMG],
            trajectory_markers=devices_by_type[DeviceType.TRAJECTORY_MARKER],
        )

    def _devices(self, aggregator: Aggregator) -> Iterator[DeviceAggregator]:
        yield from aggregator.get_devices()

    def _choose_frame_tracker(self, device_agg: DeviceAggregator,
                              forces_emg_tracker: 'ForcesEMGFrameTracker',
                              traj_tracker: 'TrajFrameTracker'
                              ) -> '_SectionFrameTracker':
        forces_emg = {DeviceType.FORCE_PLATE, DeviceType.EMG}
        if self._device_agg_type(device_agg) in forces_emg:
            return forces_emg_tracker
        return traj_tracker

    @staticmethod
    def _device_agg_name(device_aggregator: DeviceAggregator) -> str:
        return device_aggregator.name

    @staticmethod
    def _device_agg_type(device_aggregator: DeviceAggregator) -> DeviceType:
        return device_aggregator.device_type

    @staticmethod
    def _device_agg_units(device_aggregator: DeviceAggregator) -> List[str]:
        return device_aggregator.units

    @staticmethod
    def _device_agg_coords(device_aggregator: DeviceAggregator) -> List[str]:
        return device_aggregator.coords

    @staticmethod
    def _device_agg_data(device_aggregator: DeviceAggregator
                         ) -> List[List[float]]:
        return device_aggregator.data_rows

    @staticmethod
    def _aggregator_sampling_freq(aggregator: Aggregator) -> 'SamplingFreq':
        return aggregator.get_sampling_freq()


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
            raise ValueError(f'frame {frame} is out of bounds')
        if subframe not in range(self.num_subframes):
            raise ValueError(f'subframe {subframe} out of range')


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
            units: List[str],
            frame_tracker: _SectionFrameTracker,
            dataframe: pd.DataFrame,
    ):
        self.name = device_name
        self.dev_type = device_type
        self.units = tuple(units)
        self.df = dataframe
        self._frame_tracker = frame_tracker

    def __eq__(self, other) -> bool:
        return (self.name == other.name and self.dev_type == other.dev_type
                and self.units == other.units and self.df.equals(other.df))

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
        return self._frame_tracker.index(frame, subframe)
