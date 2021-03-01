import csv
from dataclasses import dataclass
from typing import Iterator

from .aggregator import Aggregator
from .definitions import Row
from .reader import Reader, SectionTypeState
from .user_data import Builder, ViconNexusData


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
    return Aggregator()


def _initialize_reader_section_type_state() -> SectionTypeState:
    return SectionTypeState()


def create_reader(initial_state=None, aggregator=None):
    if initial_state is None:
        initial_state = _initialize_reader_section_type_state()
    if aggregator is None:
        aggregator = (_initialize_aggregator(),)
    return Reader(section_type_state=initial_state, aggregator=aggregator)


def create_builder(aggregator=None):
    if aggregator is None:
        aggregator = _initialize_aggregator()
    return Builder(aggregator)


@dataclass
class _LoadingRun:
    reader: Reader
    builder: Builder


def create_loading_run() -> _LoadingRun:
    aggregator = _initialize_aggregator()
    reader = create_reader(aggregator=aggregator)
    builder = create_builder(aggregator=aggregator)
    return _LoadingRun(reader, builder)


def load_vicon_file(csv_filename: str) -> ViconNexusData:
    loading_run = create_loading_run()

    for i, row in enumerate(csv_row_stream(csv_filename), start=1):
        try:
            loading_run.reader.feed_row(row)
        except Exception as exception:
            raise RuntimeError(
                f"error parsing line {i} of file {csv_filename}: " + str(exception)
            ) from exception
    return loading_run.builder.build()
