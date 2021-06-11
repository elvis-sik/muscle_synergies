"""Functions that interact with the Vicon CSV file to load EMG data.

The main function in this module is :py:func:`load_vicon_file`, which uses
:py:func:`csv_row_stream` to get a stream of lines which it feeds to
:py:class:`~muscle_synergies.vicon_data.reader.Reader`. When the stream ends,
the function uses :py:class:`~muscle_synergies.vicon_data.user_data.Builder` to
build the final representation of the data which is delivered to the user.  The
other functions in this module are just used to initialize the different
objects using to parse the data.
"""
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
                  is passed to :py:func:`open`, so it can be a str among other
                  things.
    """
    with open(filename) as csvfile:
        data_reader = csv.reader(csvfile)
        yield from map(Row, data_reader)


def _initialize_aggregator() -> Aggregator:
    """Initializes an Aggregator instance."""
    return Aggregator()


def _initialize_reader_section_type_state() -> SectionTypeState:
    """Initializes a SectionTypeState instance."""
    return SectionTypeState()


def create_reader(initial_state=None, aggregator=None) -> Reader:
    """Initializes a new Reader.

    Args:
        initial_state: if provided, this is used as the
            :py:class:`~muscle_synergies.vicon_data.reader.Reader`'s state
            otherwise a fresh
            :py:class:`~muscle_synergies.vicon_data.definitions.SectionTypeState`
            instance is created.

        aggregator: if provided, this is used as the
            :py:class:`~muscle_synergies.vicon_data.reader.Reader`'s
            :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator`
            otherwise a fresh instance is created.
    """
    if initial_state is None:
        initial_state = _initialize_reader_section_type_state()
    if aggregator is None:
        aggregator = _initialize_aggregator()
    return Reader(section_type_state=initial_state, aggregator=aggregator)


def create_builder(aggregator=None) -> Builder:
    """Initializes a new Builder.

    Args:
        aggregator: if provided, this is used as the
            :py:class:`~muscle_synergies.vicon_data.user_data.Builder`'s
            :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator`
            otherwise a fresh instance is created.
    """
    if aggregator is None:
        aggregator = _initialize_aggregator()
    return Builder(aggregator)


@dataclass
class _LoadingRun:
    """The objects used to load the Vicon Nexus CSV file."""

    reader: Reader
    builder: Builder


def create_loading_run() -> _LoadingRun:
    """Create all objects needed to load the Vicon Nexus CSV file."""
    aggregator = _initialize_aggregator()
    reader = create_reader(aggregator=aggregator)
    builder = create_builder(aggregator=aggregator)
    return _LoadingRun(reader, builder)


def load_vicon_file(csv_filename: str) -> ViconNexusData:
    """Load data from Vicon Nexus CSV file."""
    loading_run = create_loading_run()

    for i, row in enumerate(csv_row_stream(csv_filename), start=1):
        try:
            loading_run.reader.feed_row(row)
        except Exception as exception:
            raise RuntimeError(
                f"error parsing line {i} of file {csv_filename}: " + str(exception)
            ) from exception
    return loading_run.builder.build()
