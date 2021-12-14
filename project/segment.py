"""Segment EMG signal of lateral walk with band experiment."""

from collections import OrderedDict
from enum import Enum, auto
from itertools import cycle
from typing import List, Tuple, Union, Sequence, Mapping, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas

from muscle_synergies.vicon_data.user_data import (
    DeviceType,
    DeviceData,
    ViconNexusData,
    FrameSubfr,
)


class Phase(Enum):
    """The 4 phases of the movement.

    The terminology as used in this class is always centered on the right leg,
    which differs from the leg-agnostic version defined by Medeiros.

    Attributes:
        DAA: duplo apoio amplo. Both legs are on the ground on a wide stance.

        AS: apoio simples. Only the right leg is on the ground.

        DAE: duplo apoio estreito. Both legs are on the ground on a narrow
            stance.

        BL: balanço. Only the left leg is on the ground.
    """

    DAA = "DAA"
    AS = "AS"
    DAE = "DAE"
    BL = "BL"

    def from_str(phase: str) -> Optional["Phase"]:
        return {
            "DAA": Phase.DAA,
            "DAE": Phase.DAE,
            "AS": Phase.AS,
            "BL": Phase.BL,
        }[phase.upper()]


PhaseRef = Union[Phase, int]
"""Reference to a phase in a cycle.

Some methods expect to receive a reference to a specific phase in a specific
:py:class:`Cycle` in a specific :py:class:`Trecho`. An user could specify the
phase by telling the method either "the phase I want is DAA", in which case it
makes more sense for them to pass a :py:class:`Phase` to the method, or "the
phase I want is the 3-rd one in the cycle", in which case they should pass an
:py:class:`int`. The 0-indexing convention is followed, so to get the 3-rd phase one should
pass `2` to the method.
"""


class Trecho(Enum):
    """The 4 "trechos" (sections) of the measurements.

    These refer to 4 different times the subject walks laterally on the force
    plates during the experiments. The 1st and the 3rd are done in a
    right-to-left direction whereas the 2nd and 4th are left-to-right.

    In the plots of the ground reaction forces, these show up as 4 different
    regions in which there is an overlapping between the ground reaction of
    both force plates. They are then separated by a large period in which a
    force happens in only at most one of the force plates.

    Each of these starts on either a DAA or DAE phase and includes 8 phases (2
    full cycles, i.e., steps). There is a 9th phase which is either DAA or DAE
    in the measurement immediately after the last (8th) phase which does not
    belong to any section. The period with forces only in at most one force
    plate follows that 9th phase.
    """

    FIRST = auto()
    SECOND = auto()
    THIRD = auto()
    FOURTH = auto()


class Cycle(Enum):
    """Each of the 2 cycles occurring in a pass through the force plates.

    Each pass through the force plates (:py:class:`Trecho`) includes 2 whole
    steps. Each step (cycle) includes one of each of the 4 phases
    (:py:class:`Phase`).
    """

    FIRST = auto()
    SECOND = auto()


Segments = Mapping[Trecho, Mapping[Cycle, Mapping[Phase, slice]]]
"""Segments of the signal.

For each :py:class:`Phase` in each :py:class:`Cycle` in each
:py:class:`Trecho`, store a `slice`. The `slice` includes the (contiguous)
indices in the force plate data corresponding to that specific
:py:class:`Phase`, that is, it should be built similar to `slice((start_frame,
start_subframe), (final_frame, final_subframe))`.

The mapping from phases to slices should preserve order. That is, when
iterating over its keys, it should yield each of the four :py:class:`Phase`
members in the order in which they occurred in the cycle.
"""


def reactions(vicon_nexus_data: ViconNexusData) -> Tuple[pandas.Series, pandas.Series]:
    """Get (left, right) ground reactions of force plate."""
    left_fp, right_fp = vicon_nexus_data.forcepl
    return left_fp.df["Fz"], right_fp.df["Fz"]


class Segmenter:
    """Segment Vicon Nexus data.

    During its initialization, `Segmenter` parses the ground reaction data into
    "trechos", cycles and phases and then works essentially as a glorified
    dict. :py:meth:`~Segmenter.get_times_of` provides a way to get the times in
    which the different segments begin and end.

    Attributes:
        data (ViconNexusData): the data.

        segments (Segments).
    """

    def __init__(self, data: ViconNexusData):
        self.data = data
        self.segments = self._organize_transitions()

    def get_segments(
        self,
        device_type: str,
        trecho: Union[Union[Trecho, int], Sequence],
        cycle: Optional[Union[Cycle, int]] = None,
        phase: Optional[Union[PhaseRef, str]] = None,
        device_inds: Optional[Union[int, Sequence[int]]] = None,
        device_cols: Optional[Sequence[str]] = None,
    ) -> pandas.DataFrame:
        """TODO

        Args:
            device_type: one of `"emg"`, `"traj"` (or `"marker"`) or
                `"forcepl"` (or `"fp"` or `"force plate"`). Case is ignored.
            trecho: if an `int`, a number from 1 to 4.
            cycle: if an `int`, either 1 or 2.
            phase: if a `str`, one of `"DAA"`, `"DAE"`, `"AS"` or `"BL"`
                (case is ignored).

            device_inds: indices of the devices from which to retrieve the
                data. In the case of an EMG device,

        Raises:
            ValueError if one of the arguments does not have one of the values
                explicitly mentioned as valid or if both `trecho`and `phase`
                are present but `cycle`is not.
        """
        pass

    def _get_segment(
        self,
        device_type: str,
        trecho: Union[Trecho, int],
        cycle: Optional[Union[Cycle, int]] = None,
        phase: Optional[Union[PhaseRef, str]] = None,
        device_inds: Optional[Union[int, Sequence[int]]] = None,
        device_cols: Optional[Sequence[str]] = None,
    ) -> Union[pandas.DataFrame, pandas.Series, Tuple[pandas.DataFrame]]:
        device_type = self._parse_device_type(device_type)
        trecho, cycle, phase = self._parse_segment_args(trecho, cycle, phase)
        start, end = self.get_times_of(trecho, cycle, phase, return_slice=False)
        devices = self._get_devices(device_type, device_inds)
        for dev in devices:
            foo.append(dev.bar)

        # TODO devices will be either a single DeviceData (if device_inds is an int)
        # or a tuple of DeviceData (otherwise)
        # Gotta extract cols from them and return either a single Series, a single DataFrame
        # or several of those (in case device_inds is not an int)

        dev_segment_data = []
        for dev in devices:
            dev_segment_data.append(None)
        if 00:
            pass
        # TODO now index them

    @staticmethod
    def _parse_device_type(device_type: str) -> DeviceType:
        if device_type.upper() == "EMG":
            return DeviceType.EMG
        if device_type.upper() in {"FORCE PLATE", "FP", "FORCEPL"}:
            return DeviceType.FORCE_PLATE
        if device_type.upper() in {"TRAJ", "MARKER"}:
            return DeviceType.TRAJECTORY_MARKER
        raise ValueError(f"device type not understood: {device_type}")

    @classmethod
    def _parse_segment_args(
        cls,
        trecho: Union[Trecho, int],
        cycle: Optional[Union[Cycle, int]],
        phase: Optional[Union[PhaseRef, str]],
    ) -> Tuple[Trecho, Optional[Cycle], Optional[PhaseRef]]:
        return (
            cls._parse_trecho(trecho),
            cls._parse_cycle(cycle),
            cls._parse_phase(phase),
        )

    @staticmethod
    def _parse_trecho(trecho: Union[Trecho, int]) -> Trecho:
        if trecho in Trecho:
            return trecho
        trecho_ind = trecho - 1
        return tuple(Trecho)[trecho_ind]

    @staticmethod
    def _parse_cycle(
        cycle: Optional[Union[Cycle, int]] = None,
    ) -> Optional[Cycle]:
        if cycle is None:
            return cycle
        if cycle in Cycle:
            return cycle
        cycle_ind = cycle - 1
        return tuple(Cycle)[cycle_ind]

    @staticmethod
    def _parse_phase(
        phase: Optional[Union[PhaseRef, str]] = None,
    ) -> Optional[PhaseRef]:
        if phase in PhaseRef:
            return phase
        if phase is None:
            return phase
        return {
            "DAA": PhaseRef.DAA,
            "DAE": PhaseRef.DAE,
            "AS": PhaseRef.AS,
            "BL": PhaseRef.BL,
        }[phase.upper()]

    def _get_devices(
        self,
        device_type: DeviceType,
        device_inds: Optional[Union[int, Sequence[int]]] = None,
    ) -> Tuple[DeviceData]:
        if device_type is DeviceType.EMG:
            return self.data.emg
        elif device_type is DeviceType.FORCE_PLATE:
            devices = self.data.forcepl
        elif device_type is DeviceType.TRAJECTORY_MARKER:
            devices = self.data.traj
        else:
            raise ValueError(f"device type {device_type} not understood")
        try:
            return (devices[device_inds],)
        except IndexError:
            pass
        if device_inds is None:
            device_inds = range(len(devices))
        return tuple(devices[i] for i in device_inds)

    def get_times_of(
        self,
        trecho: Trecho,
        cycle: Optional[Cycle] = None,
        phase: Optional[PhaseRef] = None,
        return_slice: bool = True,
        device_type: DeviceType = DeviceType.FORCE_PLATE,
    ) -> Union[slice, Tuple["FrameSubfr", "FrameSubfr"]]:
        """Return times corresponding to segment of ground reactions signal.

        If `return_slice` is `True`, a `slice` object of the form
        `slice(first_index, last_index)` is returned, where the indices refer
        to the :py:class:`pandas.Series` containing the ground reaction forces.
        If `return_slice` is `False`, a tuple of the form `(framsubfr_start,
        framesubfr_end)` will be returned. That is, the time in which the given
        segment starts (or ends) will be given as a `FrameSubfr` tuple.

        If all of `trecho`, `cycle` and `phase` are given, the times will refer
        to the beginning and end of exactly that specific :py:class:`Phase`.
        If only `trecho` and `cycle` are given, the single returned `slice` (or
        the single returned tuple with two `FrameSubfr` objects) will contain
        the times for all 4 phases of that :py:class:`Cycle`. Similarly, the
        times will refer to a single trecho if only :py:class:`Trecho` is
        present.
        """
        if phase is not None:
            return self._get_times_of_phase(trecho, cycle, phase, return_slice)
        if cycle is not None:
            return self._get_times_of_cycle(trecho, cycle, return_slice)
        return self._get_times_of_trecho(trecho, return_slice, device_type)

    def _get_times_of_trecho(
        self, trecho: Trecho, return_slice: bool, device_type: DeviceType
    ) -> Union[slice, Tuple["FrameSubfr", "FrameSubfr"]]:
        first_cycle_slice = self._get_times_of_cycle(trecho, Cycle.FIRST, True)
        second_cycle_slice = self._get_times_of_cycle(trecho, Cycle.SECOND, True)
        trecho_slice = slice(first_cycle_slice.start, second_cycle_slice.stop)
        return self._proc_slice(trecho_slice, return_slice)

    def _get_times_of_cycle(
        self, trecho: Trecho, cycle: Cycle, return_slice: bool, device_type: DeviceType
    ) -> Union[slice, Tuple["FrameSubfr", "FrameSubfr"]]:
        first_phase = self.ith_phase_of_cycle(trecho, cycle, 0)
        last_phase = self.ith_phase_of_cycle(trecho, cycle, -1)
        first_phase_slice = self.segments[trecho][cycle][first_phase]
        last_phase_slice = self.segments[trecho][cycle][last_phase]
        cycle_slice = slice(first_phase_slice.start, last_phase_slice.stop)
        return self._proc_slice(cycle_slice, return_slice)

    def _get_times_of_phase(
        self,
        trecho: Trecho,
        cycle: Cycle,
        phase: PhaseRef,
        return_slice: bool,
        device_type: DeviceType,
    ) -> Union[slice, Tuple["FrameSubfr", "FrameSubfr"]]:
        if phase not in Phase:
            phase = self.ith_phase_of_cycle(trecho, cycle, phase)
        phase_slice = self.segments[trecho][cycle][phase]
        return self._proc_slice(phase_slice, return_slice)

    def _proc_slice(self, slic: slice, return_slice: bool, device_type: DeviceType):
        if not return_slice:
            return self._to_frame_subfr(slic.start, device_type), self._to_frame_subfr(
                slic.stop, device_type
            )
        return slic

    def ith_phase_of_cycle(self, trecho: Trecho, cycle: Cycle, i: int) -> Phase:
        all_phases = tuple(self.segments[trecho][cycle].keys())
        return all_phases[i]

    @property
    def left_forcepl(self):
        return self.data.forcepl[0]

    @property
    def right_forcepl(self):
        return self.data.forcepl[1]

    @property
    def left_reaction(self):
        return self.left_forcepl.df["Fz"]

    @property
    def right_reaction(self):
        return self.right_forcepl.df["Fz"]

    def _to_frame_subfr(self, ind) -> "FrameSubfr":
        return self.data.forcepl[0].frame_subfr(ind)

    def _organize_transitions(self) -> Segments:
        return _organize_transitions(self.data, self._find_all_transitions())

    def _find_all_transitions(self) -> Sequence[int]:
        return _transition_indices(*self._reactions())

    def _reactions(self):
        return reactions(self.data)



class SegmentPlotter:
    """Plot a rectangle indicating the different segments of the data.

    The main method is :py:meth:`~SegmentPlotter.plot_segment`.
    """

    segm: Segmenter

    def __init__(self, segmented_data: Segmenter):
        self.segm = segmented_data

    @property
    def left_forcepl(self):
        return self.segm.left_forcepl

    @property
    def right_forcepl(self):
        return self.segm.right_forcepl

    @property
    def left_reaction(self):
        return self.segm.left_reaction

    @property
    def right_reaction(self):
        return self.segm.right_reaction

    def plot_segment(
        self,
        box_legend: str,
        trecho: Trecho,
        cycle: Optional[Cycle] = None,
        phase: Optional[PhaseRef] = None,
        y_min=-800,
        y_max=0,
        show=True,
        show_entire=True,
        **kwargs,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """Plot a rectangle on top of ground reaction to indicate segment.

        Args:
            box_legend: the description of the rectangle that appears on the
                legend.

            y_min, y_max: the vertical dimension and position of the box.

            show: if `False`, return a tuple `(fig, ax)`. If `True`, the
                function does not return and the
                :py:func:`~matplotlib.pyplot.show` is called.

            kwargs: passed to :py:meth:`SegmentPlotter.plot_rectangle`.
        """
        begin_time, end_time = self._time_ind_of_segment(trecho, cycle, phase)

        bottom_left_corner = begin_time, y_min
        height = y_max - y_min
        width = end_time - begin_time

        fig, ax = self.plot_rectangle(
            bottom_left_corner,
            width,
            height,
            box_legend,
            forces_legend=["Left reaction", "Right reaction"],
            alpha=0.1,
            **kwargs,
        )
        if not show_entire:
            trecho_beginning, trecho_end = self._time_ind_of_segment(trecho, None, None)
            trecho_duration = trecho_end - trecho_beginning
            margin = trecho_duration * 0.3
            ax.set_xlim(trecho_beginning - margin, trecho_end + margin)
        if show:
            plt.show()
            return
        return fig, ax

    def _time_ind_of_segment(
        self,
        trecho: Optional[Trecho],
        cycle: Optional[Cycle],
        phase: Optional[PhaseRef],
    ) -> Tuple[float, float]:
        ind_slice = self.segm.get_times_of(trecho, cycle, phase, return_slice=True)
        ind_x_min = ind_slice.start
        ind_x_max = ind_slice.stop
        time_seq = self.left_forcepl.time_seq()
        return time_seq[ind_x_min], time_seq[ind_x_max]

    def plot_reactions(
        self,
        figsize=(13, 5),
        left_color="g",
        right_color="r",
        legend=["Left reaction", "Right reaction"],
        title="Force plates",
        xlabel="time (s)",
        ylabel="Force (N), z component",
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """Plot ground reactions.

        To display no legend, use `legend=None`.
        """
        fig, ax = plt.subplots()

        left_reaction_plot = ax.plot(
            self.left_forcepl.time_seq(), self.left_forcepl.df["Fz"], left_color
        )

        right_reaction_plot = ax.plot(
            self.right_forcepl.time_seq(), self.right_forcepl.df["Fz"], right_color
        )

        if legend is not None:
            fig.legend(legend)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.set_size_inches(*figsize)
        return plt.gcf(), plt.gca()

    def plot_rectangle(
        self,
        bottom_left_corner,
        width,
        height,
        box_legend,
        forces_legend=["Left reaction", "Right reaction"],
        alpha=0.1,
        **kwargs,
    ):
        """Plot a rectangle on given coordinates around reaction forces.

        Args:
            forces_legend: the description of the forces that appears on the
                legend.

            bottom_left_corner, width_height: position and size of the
                rectangle. `bottom_left_corner` should be a `(x, y)` sequence.
                :py:func:`~matplotlib.pyplot.show` is called.

            box_legend: the description of the rectangle that appears on the
                legend.

            alpha: transparency of the rectangle.

            kwargs: passed to :py:meth:`SegmentPlotter.plot_reactions`.
        """
        fig, ax = self.plot_reactions(legend=None, **kwargs)
        ax.add_patch(patches.Rectangle(bottom_left_corner, width, height, alpha=0.1))
        plt.legend(forces_legend + [box_legend])
        return plt.gcf(), plt.gca()


def _transition_indices(
    left_reaction: pandas.Series,
    right_reaction: pandas.Series,
    min_phase_size: int = 10,
    num_segments: int = 40,
) -> Sequence[int]:
    """Find indices where the number of force plates with z force changes.

    Args:
        left_reaction: ground reaction of the left force plate.

        right_reaction: ground reaction of the right force plate.

        min_phase_size: minimum number of adjacent measurements with a given
            property. The data has stretches in which, for example, the right
            force measurement remains negative (indicating that there is a
            reaction) and the left one oscillates a bit close to 0. For
            example, in adjacent subframes the left measurements could go like
            `0.0, `-24.7177, 0.0`. Since the detection of transitions happens
            by noticing when the number of force plates with nonzero
            measurements changes, this would mean that some phases of the
            movement would last a single subframe. This would make no sense,
            the subject is not moving that fast. So transitions are only
            considered if there are at least `min_phase_size` measurements
            displaying the relevant property. For example, only if there are
            `min_phase_size` measurements with `0.0` on the left force plate
            and a negative value on the right one.

        num_segments: number of segments to be looked for. If `0`, find as many
            segments as possible and return them.

    Returns:
        a sequence of indices. The first index will be 0, corresponding to the
        start of the signal in which a single force plate is measuring ground
        reaction. The next index will be the start of the first "trecho", when
        both force plates have a reaction.

    Raises:
        ValueError if a segment with the desired property (e.g., just a single
            leg active) could not be found. Generally this would mean that the
            parsing process got too close to the end of the measurements.
            Decreasing `min_phase_size` might help. If `num_segments` is equal
            to 0, this should never happen.

        IndexError if the end of the input (`left_reaction` and
            `right_reaction`) was reached before `num_segments` were
            identified. If `num_segments == 0`, this should never happen.
    """

    def has_num_of_active_legs(
        left_reaction, right_reaction, legs: int
    ) -> pandas.Series:
        """Identify whether given number of legs are active at each index."""
        if legs == 1:
            return np.logical_xor(left_reaction != 0, right_reaction != 0)
        elif legs == 2:
            return np.logical_and(left_reaction != 0, right_reaction != 0)

    def look_for(left_reaction, right_reaction, legs: int, min_phase_size) -> int:
        """Find first index with a given number of active legs."""
        correct_activation = has_num_of_active_legs(left_reaction, right_reaction, legs)
        indices_tupl = np.where(correct_activation)
        # indices_tupl is a singleton since there is a single dimension in the array
        indices = indices_tupl[0]
        for ind in indices:
            if correct_activation[ind : ind + min_phase_size].all():
                return ind
        raise ValueError(
            f"no phase found with {min_phase_size} adjacent measurements with {legs} leg(s) with a nonzero reaction"
        )

    starting_index = 0
    index_seq: List[int] = []
    num_legs_seq = cycle([1, 2])

    for num_legs in num_legs_seq:
        try:
            next_index = look_for(
                left_reaction, right_reaction, num_legs, min_phase_size
            )
        except (IndexError, ValueError):
            if num_segments == 0:
                return index_seq

        left_reaction = left_reaction[next_index:]
        right_reaction = right_reaction[next_index:]
        starting_index += next_index
        index_seq.append(starting_index)

        if len(index_seq) == num_segments:
            return index_seq


# NEXT TODO
# 1. DONE Make `organize_transitions` return slices of `FrameSubfr` instead of
#    indices somehow

# 2. Then `Segmenter` needs to learn to work with that. Advantage will be to
#    support different `DeviceData`.
#
#    Segmenter just returns slices now, it becomes simpler and cares much less
#    about its data. It does nota make sense anymore for it to have
#    `left_reaction`, for example.

# 3. Make `ViconNexusData` be able to fetch a bunch of data at once. I.e., what
#    I was trying to make `Segmenter` do.

# 4. Then I have to fix `SegmentPlotter`. It very much needs to concern itself
#    with force plates.

# 5. Finally, I test everything:
#    - Run the test suite
#    - Run the Jupyter Notebooks, including the interactive one
#    - Regenerate docs and see if the `FrameSubfr` thing worked. Try to use C-f
#      to see if the definition of `Segments` occur anywhere in the docs. If
#      they do, think about fixing that.
#    - Manually test it. Get the data for all trechos 1 and 3 BL then for all
#      trechos 2 and 4. Compute synergies.


def _organize_transitions(
    vicon_nexus_data: ViconNexusData,
    transitions: Sequence[int],
) -> Segments:
    """Organize transitions into segments.

    Args:
        vicon_nexus_data: this object holds the
            :py:class:`~muscle_synergies.vicon_data.user_data.DeviceData` of the
            force plates.

        transitions: see :py:func:`transition_indices`.
    """

    def frame_subfr(index: int) -> FrameSubfr:
        """Convert array index to (frame, subframe) time."""
        return vicon_nexus_data.forcepl[0].frame_subfr(index)

    def build_cycle_dict(
        cycle: Sequence[Phase],
        indices: Sequence[int],
    ) -> Mapping[Phase, slice]:
        """Build mapping a phase to a (frame, subframe) slice.

        Returns:
            an :py:class:`~collections.OrderedDict`. Its keys
            (:py:class:`Phase` members) are stored in the order in which the
            phases ocurred in the cycle.
        """
        slices = [
            slice(frame_subfr(indices[i]), frame_subfr(indices[i + 1] - 1))
            for i in range(len(indices) - 1)
        ]
        return OrderedDict(zip(cycle, slices))

    def phase_seq(phase_indices, trecho: Trecho) -> List[Phase]:
        """Return phases in the order they occur in cycles of trecho.

        Args:
            phase_indices: a sequence containing the 8 array indices marking
                the transitions between phases in a trecho.
            trecho: the trecho in which indices occur.

        Raises:
            :py:class:`ValueError` if the second phase is not either BL or AS.
        """

        def wrong_second_phase():
            raise ValueError("expected second phase in a cycle to be either BL or AS.")

        if trecho in {Trecho.FIRST, Trecho.THIRD}:
            if single_leg_phase_type(phase_indices[1]) is Phase.BL:
                return [Phase.DAA, Phase.BL, Phase.DAE, Phase.AS]
            elif single_leg_phase_type(phase_indices[1]) is Phase.AS:
                return [Phase.DAE, Phase.AS, Phase.DAA, Phase.BL]
            else:
                wrong_second_phase()
        elif trecho in {Trecho.SECOND, Trecho.FOURTH}:
            if single_leg_phase_type(phase_indices[1]) is Phase.BL:
                return [Phase.DAE, Phase.BL, Phase.DAA, Phase.AS]
            elif single_leg_phase_type(phase_indices[1]) is Phase.AS:
                return [Phase.DAA, Phase.AS, Phase.DAE, Phase.BL]
            else:
                wrong_second_phase()

    def single_leg_phase_type(ind: int) -> Phase:
        """Find the phase type of a single leg phase occurring at a moment.

        A "single leg phase" here refers to one of the phases in which a single
        leg is on the ground, that is, either :py:data:`Phase.BL` or
        :py:data:`Phase.AS`.

        Args:
            ind: index to be inspected to determine the phase. In particular,
                it should be an array index and not a `(frame, subframe)` pair.

        Raises:
            :py:class:`ValueError` if the phase seems to not be a single leg
            phase.
        """
        reaction_in_both = left_reaction[ind] != 0 and right_reaction[ind] != 0
        reaction_in_none = left_reaction[ind] == 0 and right_reaction[ind] == 0

        if reaction_in_both or reaction_in_none:
            raise ValueError(
                "expected index corresponding to a phase in which there is "
                + "ground reaction for exactly one leg."
            )

        if left_reaction[ind] != 0:
            return Phase.BL

        return Phase.AS

    def organize_cycles(
        phase_indices: Sequence[int], end_of_trecho: int, trecho: Trecho
    ) -> Mapping[Cycle, Mapping[Phase, slice]]:
        """Organize cycle transition indices into a mapping to form `Segments`.

        Args:
            phase_indices: the 8 adjacent array indices (not `(frame,
                subframe)` pairs) with the transitions occurring in a cycle.
            end_of_trecho: the first index not belong to the trecho.
            trecho: the trecho in which the cycles occur.

        Returns:
            the mapping from :py:class:`Cycle` that occur in the definition of
            :py:data:`Segments`.
        """
        phase_indices = list(phase_indices)
        cycle = phase_seq(phase_indices, trecho)

        return {
            Cycle.FIRST: build_cycle_dict(cycle, phase_indices[:5]),
            Cycle.SECOND: build_cycle_dict(cycle, phase_indices[4:] + [end_of_trecho]),
        }

    left_reaction, right_reaction = reactions(vicon_nexus_data)

    return {
        Trecho.FIRST: organize_cycles(transitions[1:9], transitions[9], Trecho.FIRST),
        Trecho.SECOND: organize_cycles(
            transitions[11:19], transitions[19], Trecho.SECOND
        ),
        Trecho.THIRD: organize_cycles(
            transitions[21:29], transitions[29], Trecho.THIRD
        ),
        Trecho.FOURTH: organize_cycles(
            transitions[31:39], transitions[39], Trecho.FOURTH
        ),
    }
