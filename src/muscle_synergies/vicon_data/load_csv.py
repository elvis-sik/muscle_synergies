import csv
from typing import Iterator

from .definitions import Row
from .failures import Validator
from .aggregator import (
    Aggregator,
    ForcesEMGAggregator,
    TrajAggregator,
    ViconNexusData,
)
from .reader import (Reader, SectionTypeState)


def csv_row_stream(filename) -> Iterator[Row]:
    """Yields lines from a CSV file as a stream.

    Args:
        filename: the name of the CSV file which should be read. This argument
                  is passed to :py:func:open, so it can be a str among other
                  things.
    """
    with open(filename) as csvfile:
        data_reader = csv.reader(csvfile)
        yield from map(Row, data_reader)


def _initialize_aggregator() -> Aggregator:
    forces_emg_aggregator = ForcesEMGAggregator()
    traj_aggregator = TrajAggregator()
    return Aggregator(forces_emg_aggregator=forces_emg_aggregator,
                      trajs_aggregator=traj_aggregator)


def _initialize_validator(csv_filename: str,
                          should_raise: bool = True) -> Validator:
    return Validator(csv_filename=csv_filename, should_raise=should_raise)


def _initialize_reader_section_type_state() -> SectionTypeState:
    return SectionTypeState()


def _initialize_reader(initial_state: SectionTypeState, validator: Validator,
                       aggregator: Aggregator) -> Reader:
    return Reader(section_type_state=initial_state,
                  aggregator=aggregator,
                  validator=validator)


def create_reader(csv_filename: str, should_raise: bool = True):
    return _initialize_reader(
        initial_state=_initialize_reader_section_type_state(),
        validator=_initialize_validator(csv_filename=csv_filename,
                                        should_raise=should_raise),
        aggregator=_initialize_aggregator(),
    )


def load_vicon_file(csv_filename: str,
                    should_raise: bool = True) -> ViconNexusData:
    reader = create_reader(csv_filename=csv_filename,
                           should_raise=should_raise)
    for i, row in enumerate(csv_row_stream(csv_filename), start=1):
        try:
            reader.feed_row(row)
        except Exception as exception:
            raise RuntimeError(
                f'error parsing line {i} of file {csv_filename}: ' +
                str(e)) from exception
    return reader.file_ended()
