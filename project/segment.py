"""Segment EMG signal of lateral walk with band experiment."""

from collections import OrderedDict
from enum import Enum, auto
from itertools import cycle
from typing import List, Tuple, Union, Sequence, Mapping, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas

from muscle_synergies import ViconNexusData


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

    DAA = auto()
    AS = auto()
    DAE = auto()
    BL = auto()


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


FrameSubfr = Tuple[int, int]
"""Time given as frame and subframe."""

Segments = Mapping[Trecho, Mapping[Cycle, Mapping[Phase, slice]]]
"""Segments of the signal.

For each :py:class:`Phase` in each :py:class:`Cycle` in each
:py:class:`Trecho`, store a `slice`. The `slice` includes the (contiguous)
indices in the force plate data corresponding to that specific
:py:class:`Phase`, that is, it should be built similar to `slice(first_index,
final_index)`.

The indices are given as single `int` objects corresponding to
:py:class:`pandas.Series` indices. In particular, they are not given as
`(frame, subframe)` tuples.
"""


class Segmenter:
    """Segment Vicon Nexus data.

    Attributes:
        data (ViconNexusData): the data.

        segments (Segments).
    """

    def __init__(self, data: ViconNexusData):
        self.data = data
        self.segments = self._organize_transitions()

    def get_times_of(
        self,
        trecho: Trecho,
        cycle: Optional[Cycle] = None,
        phase: Optional[Phase] = None,
        return_slice: bool = False,
    ) -> Union[slice, Tuple[FrameSubfr, FrameSubfr]]:
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
        return self._get_times_of_trecho(trecho, return_slice)

    def _get_times_of_trecho(
        self, trecho: Trecho, return_slice: bool = False
    ) -> Union[slice, Tuple[FrameSubfr, FrameSubfr]]:
        first_cycle_slice = self._get_times_of_cycle(trecho, Cycle.FIRST, True)
        second_cycle_slice = self._get_times_of_cycle(trecho, Cycle.FIRST, True)
        trecho_slice = slice(first_cycle_slice.start, second_cycle_slice.stop)
        return self._proc_slice(trecho_slice, return_slice)

    def _get_times_of_cycle(
        self, trecho: Trecho, cycle: Cycle, return_slice: bool = False
    ) -> Union[slice, Tuple[FrameSubfr, FrameSubfr]]:
        first_phase = next(iter(self.segments[trecho][cycle].values()))
        last_phase = next(iter(reversed(self.segments[trecho][cycle].values())))
        cycle_slice = slice(first_phase.start, last_phase.stop)
        return self._proc_slice(cycle_slice, return_slice)

    def _get_times_of_phase(
        self, trecho: Trecho, cycle: Cycle, phase: Phase, return_slice: bool = False
    ) -> Union[slice, Tuple[FrameSubfr, FrameSubfr]]:
        phase_slice = self.segments[trecho][cycle][phase]
        return self._proc_slice(phase_slice, return_slice)

    def _proc_slice(self, slic: slice, return_slice: bool = False):
        if return_slice:
            return slic
        return self._to_frame_subfr(slic.start), self._to_frame_subfr(slic.stop)

    @property
    def left_forcepl(self):
        return self.data.forcepl[0]

    @property
    def right_forcepl(self):
        return self.data.forcepl[1]

    def _to_frame_subfr(self, ind) -> FrameSubfr:
        return self.data.forcepl[0].frame_subfr(ind)

    def _organize_transitions(self) -> Segments:
        return organize_transitions(
            *self._ground_reactions(), self._find_all_transitions()
        )

    def _find_all_transitions(self) -> Sequence[int]:
        return transition_indices(*self._ground_reactions())

    def _ground_reactions(self) -> Tuple[pandas.Series, pandas.Series]:
        """Return (left, right) ground reactions measured by force plates."""
        return self.left_forcepl.df["Fz"], self.right_forcepl.df["Fz"]


class SegmentPlotter:
    """ blabla """

    segm: Segmenter

    def __init__(self, segmented_data: Segmenter):
        self.segm = segmented_data

    @property
    def left_forcepl(self):
        return self.segm.left_forcepl

    @property
    def right_forcepl(self):
        return self.segm.right_forcepl

    def plot_segment(
        self,
        box_legend: str,
        trecho: Optional[Trecho],
        cycle: Optional[Cycle] = None,
        phase: Optional[Phase] = None,
        y_min=-800,
        y_max=0,
        **kwargs,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        begin_time, end_time = self._time_ind_of_segment(trecho, cycle, phase)

        bottom_left_corner = begin_time, y_min
        height = y_max - y_min
        width = end_time - begin_time

        self.plot_rectangle(
            bottom_left_corner,
            width,
            height,
            box_legend,
            forces_legend=["Left reaction", "Right reaction"],
            alpha=0.1,
            show=True,
            **kwargs,
        )

    def _time_ind_of_segment(
        self, trecho: Optional[Trecho], cycle: Optional[Cycle], phase: Optional[Phase]
    ) -> Tuple[float, float]:
        ind_slice = self.segm.get_times_of(trecho, cycle, phase, return_slice=True)
        ind_x_min = ind_slice.start
        ind_x_max = ind_slice.stop
        time_seq = self.left_forcepl.time_seq()
        return time_seq[ind_x_min], time_seq[ind_x_max]

    def plot_reactions(
        self,
        show=True,
        figsize=(13, 5),
        left_color="g",
        right_color="r",
        legend=["Left reaction", "Right reaction"],
        title="Force plates",
        xlabel="time (s)",
        ylabel="Force (N), z component",
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """ blabla """
        fig, ax = plt.subplots()

        left_reaction_plot = ax.plot(
            self.left_forcepl.time_seq(), self.left_forcepl.df["Fz"], left_color
        )

        right_reaction_plot = ax.plot(
            self.right_forcepl.time_seq(), self.right_forcepl.df["Fz"], right_color
        )

        fig.legend(legend)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.set_size_inches(*figsize)
        if show:
            plt.show()
            return None
        return plt.gcf(), plt.gca()

    def plot_rectangle(
        self,
        bottom_left_corner,
        width,
        height,
        box_legend,
        forces_legend=["Left reaction", "Right reaction"],
        alpha=0.1,
        show=True,
        **kwargs,
    ):
        """ blabla """
        fig, ax = self.plot_reactions(show=False, **kwargs)
        ax.add_patch(patches.Rectangle(bottom_left_corner, width, height, alpha=0.1))
        plt.legend(forces_legend + [box_legend])
        if show:
            plt.show()
            return None
        return plt.gcf(), plt.gca()


def transition_indices(
    left_reaction: pandas.Series, right_reaction: pandas.Series
) -> Sequence[int]:
    """Find indices where the number of force plates with z force changes.

    Returns:
        a sequence of indices. The first index will be 0, corresponding to the
        start of the signal in which a single force plate is measuring ground
        reaction. The next index will be the start of the first "trecho", when
        both force plates have a reaction.
    """

    def look_for_single_leg(left_reaction, right_reaction) -> int:
        indices_tupl = np.where(np.logical_xor(left_reaction != 0, right_reaction != 0))
        indices = indices_tupl[0]
        return indices[0]

    def look_for_duplo_apoio(left_reaction, right_reaction) -> int:
        indices_tupl = np.where(np.logical_and(left_reaction != 0, right_reaction != 0))
        indices = indices_tupl[0]
        return indices[0]

    starting_index = 0
    index_seq = []
    func_seq = cycle([look_for_single_leg, look_for_duplo_apoio])

    for parsing_func in func_seq:
        try:
            next_index = parsing_func(left_reaction, right_reaction)
        except IndexError:
            return index_seq

        left_reaction = left_reaction[next_index:]
        right_reaction = right_reaction[next_index:]
        starting_index += next_index
        index_seq.append(starting_index)


def organize_transitions(
    left_reaction: pandas.Series,
    right_reaction: pandas.Series,
    transitions: Sequence[int],
) -> Segments:
    def build_cycle_dict(
        cycle: Sequence[Phase],
        indices: Sequence[int],
    ) -> Mapping[Phase, slice]:
        slices = [
            slice(indices[i], indices[i + 1] - 1) for i in range(len(indices) - 1)
        ]
        return OrderedDict(zip(cycle, slices))

    def phase_seq_in_cycle_trechos_1_and_3(phase_indices) -> List[Phase]:
        if single_leg_phase_type(phase_indices[1]) is Phase.BL:
            return [Phase.DAA, Phase.BL, Phase.DAE, Phase.AS]
        elif single_leg_phase_type(phase_indices[1]) is Phase.AS:
            return [Phase.DAE, Phase.AS, Phase.DAA, Phase.BL]
        raise ValueError("expected second phase in a cycle to be either BL or AS.")

    def phase_seq_in_cycle_trechos_2_and_4(phase_indices) -> List[Phase]:
        if single_leg_phase_type(phase_indices[1]) is Phase.BL:
            return [Phase.DAE, Phase.BL, Phase.DAA, Phase.AS]
        elif single_leg_phase_type(phase_indices[1]) is Phase.AS:
            return [Phase.DAA, Phase.AS, Phase.DAE, Phase.BL]
        raise ValueError("expected second phase in a cycle to be either BL or AS.")

    def single_leg_phase_type(ind: int) -> Phase:
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
        phase_indices: Sequence[int], end_of_cycle: int, trecho: Trecho
    ) -> Mapping[Cycle, Mapping[Phase, slice]]:
        phase_indices = list(phase_indices)

        if trecho in {Trecho.FIRST, Trecho.THIRD}:
            cycle = phase_seq_in_cycle_trechos_1_and_3(phase_indices)
        else:
            cycle = phase_seq_in_cycle_trechos_2_and_4(phase_indices)

        return {
            Cycle.FIRST: build_cycle_dict(cycle, phase_indices[:5]),
            Cycle.SECOND: build_cycle_dict(
                cycle, phase_indices[4:] + [end_of_cycle + 1]
            ),
        }

    return {
        Trecho.FIRST: organize_cycles(
            transitions[1:9], transitions[9] - 1, Trecho.FIRST
        ),
        Trecho.SECOND: organize_cycles(
            transitions[11:19], transitions[19] - 1, Trecho.SECOND
        ),
        Trecho.THIRD: organize_cycles(
            transitions[21:29], transitions[29] - 1, Trecho.THIRD
        ),
        Trecho.FOURTH: organize_cycles(
            transitions[31:39], transitions[39] - 1, Trecho.FOURTH
        ),
    }
