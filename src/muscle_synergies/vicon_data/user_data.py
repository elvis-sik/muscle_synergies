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
from typing import (
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Callable,
    Any,
)

import numpy as np
import pandas as pd

from .aggregator import Aggregator, DeviceAggregator
from .definitions import DeviceType, SamplingFreq

FrameSubfr = Tuple[int, int]
"""Time given as frame and subframe."""


class ViconNexusData:
    """The data contained in a Vicon Nexus CSV file.

    The initialization arguments are stored as they are under the same names.
    They can also be accessed by indexing similar to
    `vicon_nexus_data["forcepl"]` or
    `vicon_nexus_data[DeviceType.FORCE_PLATE]`.  For all the supported string
    descriptions for device types, see
    :py:class:`~muscle_synergies.vicon_data.definitions.DeviceType`.

    Args:
        forcepl: a sequence of :py:class:`DeviceData` corresponding to the
            different force plate devices.

        emg: a single :py:class:`DeviceData` that includes all columns with EMG
            measurements.

        traj: a sequence of :py:class:`DeviceData` corresponding to the
            different trajectory devices.
    """

    def __init__(
        self,
        forcepl: Sequence["DeviceData"],
        emg: "DeviceData",
        traj: Sequence["DeviceData"],
    ):
        self.forcepl = forcepl
        self.emg = emg
        self.traj = traj

    def __getitem__(
        self, device_type: Union[DeviceType, str]
    ) -> Union["DeviceData", Sequence["DeviceData"]]:
        device_type = self._parse_device_type(device_type)
        if device_type is DeviceType.FORCE_PLATE:
            return self.forcepl
        elif device_type is DeviceType.EMG:
            return self.emg
        elif device_type is DeviceType.TRAJECTORY_MARKER:
            return self.traj
        raise KeyError(f"device type not understood: {device_type}")

    def get_cols(
        self,
        device_type: Union[str, DeviceType],
        time=None,
        device_inds: Optional[Sequence[int]] = None,
        cols=None,
    ) -> Union[
        pd.DataFrame, pd.Series, Tuple[pd.DataFrame, ...], Tuple[pd.Series, ...]
    ]:
        """Get the same data for many devices at once.

        This method is used to get (possibly a subset of) the rows and
        (possibly a subset of) the columns of the dataframes of (possibly a
        subset of) all of the devices of a given type.

        Args:
            device_type: if a `str`, a description of the device type similar
                to "emg". See :py:meth:`DeviceType.from_str` for the all the
                accepted values.

            time: if `None`, all of the rows of the each `~pandas.DataFrame` is
                returned. Otherwise, it is passed directly to
                :py:class:`DeviceData` via indexing as in `device_data[time]`.
                See the documentation for :py:class:`DeviceData` for a
                reference of what kinds of arguments are accepted.

            device_inds: a sequence of indices corresponding to which devices
                of the given type should be included.
                If `None`, data for all of the devices is returned.
                In case the device type is EMG, this is ignored.

            cols: if `None`, all columns of each :py:class:`~pandas.DataFrame`
                are returned. Otherwise, pass this argument to the dataframes
                as in `df[cols]`.

        Returns:
            In case the device type is EMG, either a single dataframe or series
            with the data. Otherwise, a tuple whose values are of one of those
            types, one value for each device in `device_ind`.
        """

        def iter_dev_data(
            vicon_nexus_data: ViconNexusData,
            device_type: Union[DeviceType, str],
            device_inds: Optional[Sequence[Int]],
        ) -> Iterator[DeviceData]:
            if device_inds is None:
                yield from vicon_nexus_data[device_type]

            for dev_data in vicon_nexus_data[device_type]:
                yield dev_data

        def get_cols_of_single_dev_data(
            device_data: DeviceData, time, cols
        ) -> Union[
            pd.DataFrame, pd.Series, Tuple[pd.DataFrame, ...], Tuple[pd.Series, ...]
        ]:
            if time is None:
                dataframe = device_data.df
            else:
                dataframe = device_data[time]
            return dataframe[cols]

        device_type = self._parse_device_type(device_type)

        if device_type is DeviceType.EMG:
            return get_cols_of_single_dev_data(self.emg, time, cols)

        data = []
        for dev_data in iter_dev_data(self, device_type, device_inds):
            data.append(get_cols_of_single_dev_data(dev_data, time, cols))
        return tuple(data)

    @staticmethod
    def _parse_device_type(device_type):
        try:
            return DeviceType.from_str(device_type)
        except ValueError:
            return device_type

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

    def to_index(
        self, frame: Union[int, FrameSubfr, slice], subframe: Optional[int]
    ) -> Union[int, slice]:
        """Array index associated with frame and subframe.

        Raises:
            `IndexError` if the arguments are outside of the allowed range.
                `frame` should be between 1 and
                :py:attr:`~_SectionFrameTracker.num_frames`.  `subframe` should
                be between 0 and
                :py:attr:`~_SectionFrameTracker.num_subframes`.
        """
        if subframe is None:
            if isinstance(frame, slice):
                slice_ = frame
                self._validate_slice(slice_, self._validate_framesubfr_args)
                return self._map_slice(slice_, self._to_index)
            frame, subframe = frame
            return self._to_index((frame, subframe))

        self._validate_framesubfr_args((frame, subframe))
        return self._to_index((frame, subframe))

    def to_framesubfr(self, index: Union[int, slice]) -> Union[FrameSubfr, slice]:
        """Frame and subframe associated with given array index.

        Raises:
            `IndexError` if the argument is outside of the allowed range (from 0
                to :py:attr:`~_SectionFrameTracker.final_index`).
        """
        if isinstance(index, slice):
            slice_ = index
            self._validate_slice(slice_, self._validate_index_arg)
            return self._map_slice(slice_, self._to_framesubfr)

        self._validate_index_arg(index)
        return self._to_framesubfr(index)

    @abc.abstractmethod
    def _to_index(self, framesubfr: FrameSubfr) -> int:
        """Convert (frame, subframe) pair to array index."""
        pass

    @abc.abstractmethod
    def _to_framesubfr(self, index: int) -> FrameSubfr:
        """Convert array index to (frame, subframe) pair."""
        pass

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
            raise IndexError(f"index {index} out of bounds (max is self.final_index)")

    def _validate_framesubfr_args(self, framesubfr: FrameSubfr):
        """Raise exception if frame and subframe are not in allowed range."""
        frame, subframe = framesubfr
        if frame not in range(1, self.num_frames + 1):
            raise IndexError(f"frame {frame} is out of bounds")
        if subframe not in range(self.num_subframes):
            raise IndexError(f"subframe {subframe} out of range")

    @staticmethod
    def _validate_slice(slice_, validate_func: Callable):
        validate_func(slice_.stop)

        for arg in {slice_.start, slice_.step}:
            if arg is not None:
                validate_func(arg)

    def time_seq(self) -> pd.Series:
        """Create Series with times in seconds of all measurements."""
        return self._time_seq(self.sampling_frequency, self.final_index + 1)

    @staticmethod
    @lru_cache(maxsize=2)
    def _time_seq(sampling_frequency: int, num_measurements: int) -> pd.Series:
        """Memoized version of time_seq."""
        period = 1 / sampling_frequency
        return pd.Series(period * np.arange(1, num_measurements + 1, 1))

    @staticmethod
    def _map_slice(slice_: slice, func: Callable) -> slice:
        """Apply func to each non-None member of a slice."""

        def apply_or_not(arg: Union[None, Any], func: Callable):
            """Apply function if value is not None, otherwise return None."""
            if arg is None:
                return None
            return func(arg)

        start = apply_or_not(slice_.start, func)
        step = apply_or_not(slice_.step, func)
        stop = apply_or_not(slice_.stop, func)
        return slice(start, stop, step)


class ForcesEMGFrameTracker(_SectionFrameTracker):
    @property
    def sampling_frequency(self) -> int:
        return self._freq_forces_emg

    def _to_index(self, framesubfr: FrameSubfr) -> int:
        frame, subframe = framesubfr
        return (frame - 1) * self.num_subframes + subframe

    def _to_framesubfr(self, index: int) -> FrameSubfr:
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

    def _to_index(self, framesubfr: FrameSubfr) -> int:
        frame, subframe = framesubfr
        return frame - 1

    def _to_framesubfr(self, index: int) -> FrameSubfr:
        return index + 1, 0

    @property
    def final_index(self) -> int:
        return self.num_frames - 1


class DeviceData:
    """Data associated with a measurement device.

    Slicing returns rows of the data frame at specific `(frame, subframe)`
    coordinates.  That may be helpful to get data for the same time period
    across different device types since they each have a different sampling
    rate. In case a range of `(frame, subframe)` is specified with a step, it
    should be an `int`.

    Examples:
        Access the data directly:

            >>> dev_data.df # doctest: +SKIP
            ... # returns entire DataFrame

        Get row corresponding to a specific frame and subframe:

            >>> dev_data[(frame, subfr)] # doctest: +SKIP
            ... # returns single row of data frame

        Get rows corresponding to range specified as frame and subframe:
            >>> dev_data[(start_fr, start_subf), (end_fr, end_subf), 3] # doctest: +SKIP
            ... # returns every 3 rows in range
    """

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

    def __getitem__(self, indices: Union["FrameSubfr", slice]) -> pd.DataFrame:
        if isinstance(indices, slice):
            return self.df.iloc[self.to_index(indices)]

        return self.df.iloc[self.to_index(*indices)]

    def to_framesubfr(self, index: Union[int, slice]) -> Union[FrameSubfr, slice]:
        """Find (frame, subframe) pair corresponding to index.

        Returns:
            If a single index is passed, a `(frame, subframe)` pair. If a
            `slice` of array indexes, a `slice` of such pairs is returned.
        """
        return self._frame_tracker.to_framesubfr(index)

    def to_index(
        self, frame: Union[int, slice, FrameSubfr], subframe: Optional[int] = None
    ) -> int:
        """Return array indices corresponding to (frame, subframe) pair.

        Returns:
            A single `int` containing the array index corresponding to the
            given `(frame, subframe)` pair. If a `slice` is passed containing a
            range of such pairs, another `slice` is returned, this time
            containing a range of array indexes.

        Raises:
            `ValueError` if the first argument (`frame`) is a `(frame,
            subframe)` and the optional second argument (`subframe`) is not
            `None`.
        """
        return self._frame_tracker.to_index(frame, subframe)

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
