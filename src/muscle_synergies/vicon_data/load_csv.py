import csv
from typing import Iterator

from .failures import Validator
from .reader_data import (
    Row,
    DataBuilder,
    ForcesEMGDataBuilder,
    TrajDataBuilder,
    ViconNexusData,
)
from .reader import (Reader, SectionTypeState)


def csv_lines_stream(filename) -> Iterator[Row]:
    """Yields lines from a CSV file as a stream.

    Args:
        filename: the name of the CSV file which should be read. This argument
                  is passed to :py:func:open, so it can be a str among other
                  things.
    """
    with open(filename) as csvfile:
        data_reader = csv.reader(csvfile)
        yield from map(Row, data_reader)


def _initialize_data_builder() -> DataBuilder:
    forces_emg_builder = ForcesEMGDataBuilder()
    traj_builder = TrajDataBuilder()
    return DataBuilder(forces_emg_data_builder=forces_emg_builder,
                       trajs_data_builder=traj_builder)


def _initialize_validator(csv_filename: str,
                          should_raise: bool = True) -> Validator:
    return Validator(csv_filename=csv_filename, should_raise=should_raise)


def _initialize_reader_section_type_state() -> SectionTypeState:
    return SectionTypeState()


def _initialize_reader(initial_state: SectionTypeState, validator: Validator,
                       data_builder: DataBuilder) -> Reader:
    return Reader(section_type_state=initial_state,
                  data_builder=data_builder,
                  validator=validator)


def create_reader(csv_filename: str, should_raise: bool = True):
    return _initialize_reader(
        initial_state=_initialize_reader_section_type_state(),
        validator=_initialize_validator(csv_filename=csv_filename,
                                        should_raise=should_raise),
        data_builder=_initialize_data_builder(),
    )


def load_vicon_file(csv_filename: str,
                    should_raise: bool = True) -> ViconNexusData:
    reader = create_reader(csv_filename=csv_filename,
                           should_raise=should_raise)
    for row in csv_lines_stream(csv_filename):
        reader.feed_row(row)
    return reader.file_ended()
