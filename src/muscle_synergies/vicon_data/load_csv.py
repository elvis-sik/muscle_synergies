from typing import Iterator

from .reader_data import Row


def csv_lines_stream(filename) -> Iterator[Row]:
    """Yields lines from a CSV file as a stream.

    Args:
        filename: the name of the CSV file which should be read. This argument
                  is passed to :py:func:open, so it can be a str among other
                  things.
    """
    with open(filename) as csvfile:
        data_reader = csv.reader(csvfile)
        yield from data_reader


def initialize_vicon_nexus_reader():
    pass


def load_vicon_file():
    pass
