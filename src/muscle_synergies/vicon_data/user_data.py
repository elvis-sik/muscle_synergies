"""Types that help building the final representation of the data.

From the point of view of the internal API, the main type in this module is
:py:class:`Builder`, which uses the data stored in an
:py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator` to build the
:py:class:`ViconNexusData`. That object, in turn, simply holds references to
:py:class:`DeviceData` instances corresponding to the different experimental
devices, organized by their type (see
:py:class:`~muscle_synergies.vicon_data.definitions.DeviceType`).

Refer to the documentation for the package
:py:mod:`muscle_synergies.vicon_data.__init__.py` for more on how
:py:class:`Builder` fits together with the other classes used for reading the
data from disk.
"""
import abc
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .aggregator import Aggregator, DeviceAggregator
from .definitions import DeviceType, SamplingFreq

FrameSubfr = Tuple[int, int]
"""Time given as frame and subframe."""


@dataclass
class ViconNexusData:
    """The data contained in a Vicon Nexus CSV file.

    The initialization arguments are stored as they are under the same names.

    Args:
        forcepl: a sequence of :py:class:`DeviceData` corresponding to the
            different force plate devices.

        emg: a single :py:class:`DeviceData` that includes all columns with EMG
            measurements.

        traj: a sequence of :py:class:`DeviceData` corresponding to the
            different trajectory devices.
    """

    forcepl: Sequence["DeviceData"]
    emg: "DeviceData"
    traj: Sequence["DeviceData"]

    def __repr__(self):
        return "ViconNexusData(forcepl=[...], emg=<DeviceData>, traj=[...])"

    def describe(self) -> str:
        """Represent ViconNexusData object as a Markdown list.

        This method is intended to help the user get a quick glance at what was
        loaded.  The returned value will be a multiline string similar to this:

            ViconNexusData:

            + emg: 8 columns
            + forcepl (2 devices): DeviceData("Force Plate 1"), DeviceData("Force Plate 2")
            + traj (14 devices): DeviceData("Traj 1"), ..., DeviceData("Traj 14")

        In the case of force plates and trajectory markers, if there are more
        than 2 devices, they are occluded as in the last line of the example.
        """
        emg_str = self._amount_str(self._num_cols(self.emg), "column")
        forcepl_len_str = self._amount_str(len(self.forcepl), "device")
        forcepl_members_str = self._stringify_list(self.forcepl)
        traj_len_str = self._amount_str(len(self.traj), "device")
        traj_members_str = self._stringify_list(self.traj)
        return f"""ViconNexusData:
+ emg: {emg_str}
+ forcepl ({forcepl_len_str}): {forcepl_members_str}
+ traj ({traj_len_str}): {traj_members_str}"""

    @staticmethod
    def _num_cols(dev: "DeviceData") -> int:
        """Get number of columns contained in :py:class:`DeviceData` object."""
        return len(dev.df.columns)

    @staticmethod
    def _amount_str(num: int, noun: str) -> str:
        """Add an "s" to a noun to make it plural if needed."""
        if num == 1:
            plural_s = ""
        else:
            plural_s = "s"
        return f"{num} {noun}{plural_s}"

    @staticmethod
    def _stringify_list(seq: Sequence) -> str:
        """Represent list as string occluding elements to make it short."""
        seq = list(seq)
        if len(seq) > 2:
            seq = [seq[0]] + ["..."] + [seq[-1]]
        return ", ".join(map(str, seq))


class Builder:
    """Build a ViconNexusData using the data stored in an Aggregator."""

    def __init__(self, aggregator: Optional[Aggregator] = None):
        self.aggregator = aggregator

    def build(self, aggregator: Optional[Aggregator] = None) -> ViconNexusData:
        """Build a ViconNexusData using the data stored in an Aggregator.

        Args:
            aggregator: if not provided, use the one given during
                initialization.

        Raises:
            ValueError if the number of EMG devices is not exactly 1.
        """
        if aggregator is None:
            aggregator = self.aggregator

        frame_tracker = self._build_frame_tracker(aggregator)

        devices_by_type = defaultdict(list)
        for device_agg in self._devices(aggregator):
            device_data = self._build_device(device_agg, frame_tracker)
            device_type = self._device_agg_type(device_agg)
            devices_by_type[device_type].append(device_data)

        # TODO fix a typing mess:
        # 1. make _vicon_nexus_data get 3 parameters corresponding to device
        #    type lists instead of a dict
        # 2. _simplify_emg now gets an emg_list and returns an emg_dev,
        #    checking if the list has too many entries
        # done.

        return self._vicon_nexus_data(self._simplify_emg(devices_by_type))

    def _build_device(
        self,
        device_agg: DeviceAggregator,
        frame_tracker: Tuple["ForcesEMGFrameTracker", "TrajFrameTracker"],
    ) -> "DeviceData":
        """Create new DeviceData from DeviceAggregator and frame trackers."""
        params_dict = self._params_device_data(device_agg, frame_tracker)
        return self._instantiate_device(**params_dict)

    def _params_device_data(
        self,
        device_agg: DeviceAggregator,
        frame_tracker: Tuple["ForcesEMGFrameTracker", "TrajFrameTracker"],
    ) -> Mapping[str, Union[str, DeviceType, "_SectionFrameTracker", pd.DataFrame]]:
        """Build a dict with the params to create a new DeviceData instance.

        This method sets up a dict corresponding to the keyword arguments
        required by :py:meth`~Builder._instantiate_device`.
        """
        return {
            "device_name": self._device_agg_name(device_agg),
            "device_type": self._device_agg_type(device_agg),
            "units": self._device_agg_units(device_agg),
            "frame_tracker": self._choose_frame_tracker(device_agg, *frame_tracker),
            "dataframe": self._extract_dataframe(device_agg),
        }

    def _build_frame_tracker(
        self, aggregator: Aggregator
    ) -> Tuple["ForcesEMGFrameTracker", "TrajFrameTracker"]:
        """Build frame trackers corresponding to Aggregator."""
        sampling_freq = self._aggregator_sampling_freq(aggregator)
        return (ForcesEMGFrameTracker(sampling_freq), TrajFrameTracker(sampling_freq))

    @staticmethod
    def _instantiate_device(
        device_name: str,
        device_type: DeviceType,
        units: List[str],
        frame_tracker: "_SectionFrameTracker",
        dataframe: pd.DataFrame,
    ) -> "DeviceData":
        """Instantiate new DeviceData object."""
        return DeviceData(
            device_name=device_name,
            device_type=device_type,
            units=units,
            frame_tracker=frame_tracker,
            dataframe=dataframe,
        )

    @classmethod
    def _extract_dataframe(cls, device_aggregator: DeviceAggregator) -> pd.DataFrame:
        """Create DataFrame with the data in the DeviceAggregator."""
        data = cls._device_agg_data(device_aggregator)
        header = cls._device_agg_coords(device_aggregator)
        return pd.DataFrame(data, columns=header, dtype=float)

    def _simplify_emg(
        self, devices_by_type: Mapping[DeviceType, List["DeviceData"]]
    ) -> Mapping[DeviceType, Union["DeviceData", List["DeviceData"]]]:
        """Replaces list of EMG devices with the single device in dict.

        Args:
            devices_by_type: a dict which lists all devices of each type.

        Returns:
            a copy of the dict with one change.
            `new_devices_by_type[DeviceType.EMG]` will not be a a list of
            devices but rather a single one as it is assumed that all EMG data
            is represented as being different coordinates of a single
            experimental device.

        Raises:
            ValueError if the number of EMG devices is not exactly 1.
        """
        new_devices_dict = dict(devices_by_type)
        emg_list = devices_by_type[DeviceType.EMG]
        if len(emg_list) != 1:
            raise ValueError(f"found {len(emg_list)} EMG devices - expected one")
        emg_dev = emg_list[0]
        new_devices_dict[DeviceType.EMG] = emg_dev
        return new_devices_dict

    @staticmethod
    def _vicon_nexus_data(
        devices_by_type: Mapping[DeviceType, Union["DeviceData", List["DeviceData"]]],
    ) -> ViconNexusData:
        """Instantiate new ViconNexusData object."""
        return ViconNexusData(
            forcepl=devices_by_type[DeviceType.FORCE_PLATE],
            emg=devices_by_type[DeviceType.EMG],
            traj=devices_by_type[DeviceType.TRAJECTORY_MARKER],
        )

    @staticmethod
    def _devices(aggregator: Aggregator) -> Iterator[DeviceAggregator]:
        """Yield all `DeviceAggregator`s stored in the Aggregator."""
        yield from aggregator.get_devices()

    def _choose_frame_tracker(
        self,
        device_agg: DeviceAggregator,
        forces_emg_tracker: "ForcesEMGFrameTracker",
        traj_tracker: "TrajFrameTracker",
    ) -> "_SectionFrameTracker":
        """Choose the correct frame tracker for device."""
        forces_emg = {DeviceType.FORCE_PLATE, DeviceType.EMG}
        if self._device_agg_type(device_agg) in forces_emg:
            return forces_emg_tracker
        return traj_tracker

    @staticmethod
    def _device_agg_name(device_aggregator: DeviceAggregator) -> str:
        """Get device name from DeviceAggregator."""
        return device_aggregator.name

    @staticmethod
    def _device_agg_type(device_aggregator: DeviceAggregator) -> DeviceType:
        """Get device type from DeviceAggregator."""
        return device_aggregator.device_type

    @staticmethod
    def _device_agg_units(device_aggregator: DeviceAggregator) -> List[str]:
        """Get device units from DeviceAggregator."""
        return device_aggregator.units

    @staticmethod
    def _device_agg_coords(device_aggregator: DeviceAggregator) -> List[str]:
        """Get device coordinates from DeviceAggregator."""
        return device_aggregator.coords

    @staticmethod
    def _device_agg_data(device_aggregator: DeviceAggregator) -> List[List[float]]:
        """Get the data rows stored in DeviceAggregator."""
        return device_aggregator.data_rows

    @staticmethod
    def _aggregator_sampling_freq(aggregator: Aggregator) -> "SamplingFreq":
        """Get the sampling frequencies stored in Aggregator."""
        return aggregator.get_sampling_freq()


class _SectionFrameTracker(abc.ABC):
    """Convert array indices to/from (frame, subframe) for a section.

    This class is abstract, subclasses implement the conversions, which differ
    between the 2 sections of the CSV file. The first data row will have index
    0 and correspond to frame 0 and subframe 0. The second data row will have
    index 1 but its frame and subframe will differ depending on the relative
    sampling rate of each section. See
    :py:class:`~muscle_synergies.vicon_data.definitions.SamplingFreq`.

    The 2 main methods of :py:class:`_SectionFrameTracker` are:

    + :py:meth:`~_SectionFrameTracker.index`: convert frame and subframe to the
      corresponding array index.
    + :py:meth:`~_SectionFrameTracker.frame_tracker`: convert an array index to
      the corresponding frame and subframe.
    """

    def __init__(self, sampling_freq=SamplingFreq):
        self._sampling_freq = sampling_freq

    @property
    def num_frames(self) -> int:
        """Total number of frames."""
        return self._sampling_freq.num_frames

    @abc.abstractproperty
    def sampling_frequency(self) -> int:
        """Sampling frequency in Hz with which the measurements were made."""
        pass

    @abc.abstractmethod
    def index(self, frame: int, subframe: int) -> int:
        """Array index associated with frame and subframe.

        Raises:
            ValueError if the arguments are outside of the allowed range.
                `frame` should be between 1 and
                :py:attr:`~_SectionFrameTracker.num_frames`.  `subframe` should
                be between 0 and
                :py:attr:`~_SectionFrameTracker.num_subframes`.
        """
        self._validate_frame_tracker_args(frame, subframe)

    @abc.abstractmethod
    def frame_tracker(self, index: int) -> Tuple[int, int]:
        """Frame and subframe associated with given array index.

        Raises:
            ValueError if the argument is outside of the allowed range (from 0
                to :py:attr:`~_SectionFrameTracker.final_index`).
        """
        self._validate_index_arg(index)

    @abc.abstractproperty
    def final_index(self) -> int:
        """The highest array index."""
        pass

    @property
    def num_subframes(self) -> int:
        """The total number of subframes."""
        return self._sampling_freq.num_subframes

    @property
    def _freq_forces_emg(self) -> int:
        """The sampling rate of the section with force plates and EMG."""
        return self._sampling_freq.freq_forces_emg

    @property
    def _freq_traj(self) -> int:
        """The sampling rate of the section with trajectories."""
        return self._sampling_freq.freq_traj

    def _validate_index_arg(self, index: int):
        """Raise exception if index is outside of allowed range."""
        if index not in range(self.final_index + 1):
            raise ValueError(f"index {index} out of bounds (max is self.final_index)")

    def _validate_frame_tracker_args(self, frame: int, subframe: int):
        """Raise exception if frame and subframe are not in allowed range."""
        if frame not in range(1, self.num_frames + 1):
            raise ValueError(f"frame {frame} is out of bounds")
        if subframe not in range(self.num_subframes):
            raise ValueError(f"subframe {subframe} out of range")

    def time_seq(self) -> pd.Series:
        """Create Series with times in seconds of all measurements."""
        return self._time_seq(self.sampling_frequency, self.final_index + 1)

    @staticmethod
    @lru_cache(maxsize=2)
    def _time_seq(sampling_frequency: int, num_measurements: int) -> pd.Series:
        """Memoized version of time_seq."""
        period = 1 / sampling_frequency
        return pd.Series(period * np.arange(1, num_measurements + 1, 1))


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
    """Data associated with a measurement device."""

    name: str
    """the name of the device, as it occurs on the CSV file. """
    dev_type: DeviceType
    """the data associated with the device."""
    units: Tuple[str]
    """physical units of each column in the :py:class:`~pandas.DataFrame`."""
    df: pd.DataFrame
    """the type of the device (can be a force plate, trajectory marker or EMG
    device).
    """

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

    @property
    def sampling_frequency(self) -> int:
        """Sampling rate with which measurements were made."""
        return self._frame_tracker.sampling_frequency

    def time_seq(self) -> pd.Series:
        """Compute the moment in seconds in which measurements were made.

        Returns:
            a :py:class:`pandas.Series` where each entry corresponds to
        """
        return self._frame_tracker.time_seq()

    def iloc(self, frame: int, subframe: int) -> pd.Series:
        """Index data rows by their frame and subframe.

        This method works similarly to :py:attr:`pandas.DataFrame.iloc`: its
        purpose is to help the user index the data referring to rows. Whereas
        the :py:class:`~pandas.DataFrame` version is used by directly indexing
        it (`datafr.iloc[0]` returns the first row), the :py:class:`DeviceData`
        version is a method.

        To get the i-th row of the :py:class:`~pandas.DataFrame`, use its own
        :py:attr:`~pandas.DataFrame.iloc`. This method should be used only when
        the goal is to get not the i-th row but the one corresponding to a
        given frame and subframe.

        Raises:
            KeyError: if the frame and subframe are out of bounds.
        """
        return self.df.iloc[self._convert_key(frame, subframe)]

    def frame_subfr(self, index: int) -> Tuple[ int, int]:
        """Find (frame, subframe) pair corresponding to index."""
        return self._frame_tracker.frame_tracker(index)

    def _key_slice_frame_subframe(
        self,
        stop: Tuple[int, int],
        start: Optional[Tuple[int, int]] = None,
        step: Optional[int] = None,
    ) -> slice:
        """Create slice with indexes corresponding to (frame, subframe) range.

        Raises:
            KeyError: if the frame and subframe are out-of-bounds.
        """
        stop_index = self._convert_key(*stop)
        if start is None:
            return slice(stop_index)

        start_index = self._convert_key(*start)
        if step is None:
            return slice(start_index, stop_index)
        return slice(start_index, stop_index, step)

    def _convert_key(self, frame: int, subframe: int) -> int:
        """Get index corresponding to given frame and subframe.

        Raises:
            KeyError: if the frame and subframe are out-of-bounds.
        """
        try:
            return self._frame_tracker_index(frame, subframe)
        except ValueError as err:
            raise KeyError from err

    def _frame_tracker_index(self, frame: int, subframe: int) -> int:
        """Call FrameTracker.index with arguments."""
        return self._frame_tracker.index(frame, subframe)

    def __eq__(self, other) -> bool:
        return (
            self.name == other.name
            and self.dev_type == other.dev_type
            and self.units == other.units
            and self.df.equals(other.df)
        )

    def __str__(self):
        return f'DeviceData("{self.name}")'

    def __repr__(self):
        return f"<{str(self)}>"
