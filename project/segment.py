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

    @staticmethod
    def from_str(phase: str) -> "Phase":
        return {
            "DAA": Phase.DAA,
            "DAE": Phase.DAE,
            "AS": Phase.AS,
            "BL": Phase.BL,
        }[phase.upper()]


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

PhaseRef = Union[Phase, int, str]
"""Reference to a phase in a cycle.

Some methods expect to receive a reference to a specific phase in a specific
:py:class:`Cycle` in a specific :py:class:`Trecho`. An user could specify the
phase by telling the method either "the phase I want is DAA", in which case it
makes more sense for them to either pass a :py:class:`Phase` or a `"str"`
description of it (like `"BL"`) to the method, or "the phase I want is the 3-rd
one in the cycle", in which case they should pass an :py:class:`int`. The
1-indexing convention is followed, so to get the 3-rd phase one should pass `3`
to the method.
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

    Args:
        data (ViconNexusData): the data.
    """

    def __init__(self, data: ViconNexusData):
        self._segments = self._organize_transitions(data)

    def ith_phase(self, trecho: Union[Trecho, int], i: int) -> Phase:
        """Determine the i-th phase occurring in a given trecho.

        Args:
            trecho: needed because the phases in trechos 1 and 3 occur in a
                different order than the ones in trechos 2 and 4.
            i: the phase, should be a number between 1 and 4, with `i=1`
                referring to the first phase. 0-based indexing is *not* used.

        Raises:
            `IndexError` if `i` is not between 1 and 4.
        """
        if i not in range(1, 5):
            raise IndexError("i should be a number between 1 and 4")

        trecho = self._parse_trecho(trecho)
        i = ((i - 1) % 4)
        cycle = Cycle.FIRST
        all_phases = tuple(self._segments[trecho][cycle].keys())
        return all_phases[i]

    def get_times_of(
        self,
        trecho: Union[Trecho, int],
        cycle: Optional[Union[Cycle, int]] = None,
        phase: Optional[PhaseRef] = None,
    ) -> slice:
        """Return times corresponding to segment of ground reactions signal.

        If all of `trecho`, `cycle` and `phase` are given, the times will refer
        to the beginning and end of exactly that specific :py:class:`Phase`.
        If only `trecho` and `cycle` are given, the single returned `slice`
        will contain the times for all 4 phases of that :py:class:`Cycle`.
        Similarly, the times will refer to a single trecho if only
        :py:class:`Trecho` is present.

        Args:
            trecho: if an `int`, a number from 1 to 4.
            cycle: if an `int`, either 1 or 2.
            phase: if a `str`, one of `"DAA"`, `"DAE"`, `"AS"` or `"BL"` (case
                is ignored). If an `int`, should be a number from 1 to 4
                specifying the phase using its order in the cycle.

        Raises:
            `ValueError` if `phase` is not `None` but `cycle` is `None`.

        Returns:
            a `slice` object with the `(frame, subframe)` range of the segment.
            Its attributes can be used to get the `(frame, subframe)` instants
            in which the segment begins and ends like this: `slic.start` and
            `slic.stop`. Or they could be passed directly to
            :py:class:`~muscle_synergies.vicon_data.user_data.DeviceData`
            instances via indexing as in `dev_data[segmenter.get_times_of(1)]`.
        """
        trecho, cycle, phase = self._parse_segment_args(trecho, cycle, phase)
        if phase is not None:
            if cycle is None:
                raise ValueError("if a phase is given, a cycle should also be")
            return self._get_times_of_phase(trecho, cycle, phase)
        if cycle is not None:
            return self._get_times_of_cycle(trecho, cycle)
        return self._get_times_of_trecho(trecho)

    def _parse_segment_args(
        self,
        trecho: Union[Trecho, int],
        cycle: Optional[Union[Cycle, int]],
        phase_ref: Optional[PhaseRef],
    ) -> Tuple[Trecho, Optional[Cycle], Optional[Phase]]:
        trecho = self._parse_trecho(trecho)
        cycle = self._parse_cycle(cycle)
        phase = self._parse_phase(trecho, phase_ref)
        return trecho, cycle, phase

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

    def _parse_phase(
        self, trecho: Trecho, phase_ref: Optional[PhaseRef]
    ) -> Optional[Phase]:
        if phase_ref is None:
            return None
        if phase_ref in Phase:
            return phase_ref
        try:
            return Phase.from_str(phase_ref)
        except (KeyError, AttributeError):
            pass
        return self.ith_phase(trecho, phase_ref)

    def _get_times_of_trecho(self, trecho: Trecho) -> slice:
        first_cycle = self.get_times_of(trecho, Cycle.FIRST)
        second_cycle = self.get_times_of(trecho, Cycle.SECOND)
        return slice(first_cycle.start, second_cycle.stop)

    def _get_times_of_cycle(self, trecho: Trecho, cycle: Cycle) -> slice:
        first_phase = self.get_times_of(trecho, cycle, 1)
        last_phase = self.get_times_of(trecho, cycle, 4)
        return slice(first_phase.start, last_phase.stop)

    def _get_times_of_phase(
        self,
        trecho: Trecho,
        cycle: Cycle,
        phase: Phase,
    ) -> slice:
        return self._segments[trecho][cycle][phase]

    def _organize_transitions(self, data: ViconNexusData) -> Segments:
        return _organize_transitions(data, self._find_all_transitions(data))

    def _find_all_transitions(self, data: ViconNexusData) -> Sequence[int]:
        return _transition_indices(*reactions(data))


class SegmentPlotter:
    """Plot a rectangle indicating the different segments of the data.

    The most useful method is :py:meth:`~SegmentPlotter.plot_segment`.
    """

    segm: Segmenter

    def __init__(self, data: ViconNexusData, segmenter: Segmenter):
        self.data = data
        self.segm = segmenter

    @property
    def left_forcepl(self) -> DeviceData:
        return self.data.forcepl[0]

    @property
    def right_forcepl(self) -> DeviceData:
        return self.data.forcepl[1]

    @property
    def left_reaction(self) -> pandas.Series:
        return reactions(self.data)[0]

    @property
    def right_reaction(self) -> pandas.Series:
        return reactions(self.data)[1]

    def plot_segment(
        self,
        box_legend: str,
        trecho: Union[Trecho, int] = 0,
        cycle: Optional[Union[Cycle, int]] = None,
        phase: Optional[PhaseRef] = None,
        y_min=-800,
        y_max=0,
        show=True,
        show_entire=True,
        display_legend=True,
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
        begin_time, end_time = self._time_ind_of_segment(
            self.left_forcepl, trecho, cycle, phase
        )

        bottom_left_corner = begin_time, y_min
        height = y_max - y_min
        width = end_time - begin_time

        fig, ax = self.plot_reactions()

        self._add_rectangle(
            ax,
            box_legend,
            bottom_left_corner,
            width,
            height,
            alpha=0.1,
        )

        if not show_entire:
            trecho_beginning, trecho_end = self._time_ind_of_segment(
                self.left_forcepl, trecho, None, None
            )
            trecho_duration = trecho_end - trecho_beginning
            margin = trecho_duration * 0.3
            ax.set_xlim(trecho_beginning - margin, trecho_end + margin)

        ax.legend()

        if show:
            plt.show()
            return
        return fig, ax

    def _time_ind_of_segment(
        self,
        device: DeviceData,
        trecho: Union[Trecho, int] = 0,
        cycle: Optional[Union[Cycle, int]] = None,
        phase: Optional[PhaseRef] = None,
    ) -> Tuple[float, float]:
        framesubfr_slice = self.segm.get_times_of(trecho, cycle, phase)
        ind_slice = device.to_index(framesubfr_slice)
        ind_x_min = ind_slice.start
        ind_x_max = ind_slice.stop
        time_seq = self.left_forcepl.time_seq()
        return time_seq[ind_x_min], time_seq[ind_x_max]

    def plot_cols(
        self,
        # columns parameters
        device_type: Union[str, DeviceType] = "force plate",
        device_inds: Optional[Sequence[int]] = None,
        col: str = "Fz",
        # time parameters
        trecho: Union[Trecho, int] = 0,
        cycle: Optional[Union[Cycle, int]] = None,
        phase: Optional[PhaseRef] = None,
        # plot parameters
        box_legend: Optional[str] = None,
        y_min=-800,
        y_max=0,
        show=True,
        show_entire=True,
        **kwargs,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """Plot columns of data with rectangles on segments.

        By default, the ground reactions will be shown with a rectangle around
        the first trecho. This way, one can see the entire signal for the
        ground reactions and the part that refers to the first trecho, which
        could be useful to check that the signals have been segmented
        correctly. From that default plot, many things could be changed by
        fiddling with the arguments:

        + instead of highlighting the first trecho, any trecho, cycle or phase
          could be selected.

        + Instead of plotting the ground reactions measured by both force
          plates, a single column from any number of measurement devices of a
          single type could be plotted.

        + Graphical details of the plot could be changed, like the size and
          color of the rectangle box.

        The segment is specified with the parameters `trecho`, `cycle` and
        `phase`. The easiest way to specify those would be to use `int`'s to
        refer to the `trecho` and `cycle` (i.e., `trecho=1` and `cycle=2` for
        first trecho, second cycle) and either a `str` or an `int` for the
        phase. If `phase` is given as a `str`, it should be specified similar
        to `"DAA"`.  If an `int`, it should be between `1` and `4`, with `1`
        referring to the first phase occurring in the given cycle of the given
        trecho. For more on how they work and what is supported, refer to
        :py:meth:`Segmenter.get_times_of`.

        The data columns are specified with the parameters `device_type`,
        `device_inds` and `col`.  The first, `device_type` is one of `"force
        plate"`, `"traj"` and `"emg"`.  The second, `device_inds`, if given, is
        a sequence of indices referring to which devices, out of the list of
        all of them, should be picked. Finally, the third, `col`, is the label
        of the column of the data frames that will be plotted. Refer to
        :py:meth:`~muscle_synergies.vicon_data.user_data.ViconNexusData.get_cols`
        for further documentation on these parameters.

        Plot arguments:
            box_legend: the description of the rectangle (the one highlighting
                the phase) that appears on the legend.

            y_min, y_max: the vertical dimension and position of the box.

            show: if `False`, return a tuple `(fig, ax)`. If `True`, the
                function does not return and the
                :py:func:`~matplotlib.pyplot.show` is called.

            show_entire: if `False`, the x-axis will display the entire
                duration of the experiment, i.e., start at frame 1, subframe 0
                and end at the final ones. If `True`, the x-axis will be a
                smaller window around the highlighted segment.

            **kwargs:
        """
        pass

    def plot_reactions(
        self,
        figsize=(13, 5),
        left_color="g",
        right_color="r",
        labels=["Left reaction", "Right reaction"],
        title="Force plates",
        xlabel="time (s)",
        ylabel="Force (N), z component",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot ground reactions."""
        fig, ax = plt.subplots()

        ax.plot(
            self.left_forcepl.time_seq(),
            self.left_forcepl.df["Fz"],
            left_color,
            label=labels[0],
        )

        ax.plot(
            self.right_forcepl.time_seq(),
            self.right_forcepl.df["Fz"],
            right_color,
            label=labels[1],
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.set_size_inches(*figsize)
        return fig, ax

    def _add_rectangle(
        self,
        axes: plt.Axes,
        label: str,
        bottom_left_corner: Tuple[float, float],
        width: float,
        height: float,
        alpha: float = 0.1,
    ) -> plt.Axes:
        """Plot a rectangle on given coordinates around reaction forces.

        Args:
            forces_legend: the description of the forces that appears on the
                legend.

            bottom_left_corner, width_height: position and size of the
                rectangle. `bottom_left_corner` should be a `(x, y)` pair.
                :py:func:`~matplotlib.pyplot.show` is called.

            box_legend: the description of the rectangle that appears on the
                legend.

            alpha: transparency of the rectangle.

            kwargs: passed to :py:meth:`SegmentPlotter.plot_reactions`.
        """
        axes.add_patch(
            patches.Rectangle(bottom_left_corner, width, height, alpha=alpha, label=label)
        )
        return axes


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

    def to_framesubfr(index: int) -> FrameSubfr:
        """Convert array index to (frame, subframe) time."""
        return vicon_nexus_data.forcepl[0].to_framesubfr(index)

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
            slice(to_framesubfr(indices[i]), to_framesubfr(indices[i + 1] - 1))
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
