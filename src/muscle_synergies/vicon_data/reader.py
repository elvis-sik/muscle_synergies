"""Types that parse the data in the Vicon CSV file line-by-line.

The main class here is :py:class:`Reader`, which must be fed lines through
:py:meth:`Reader.feed_row`. `Reader` keeps a pointer to a
:py:class:`_ReaderState` instance. All the other types in this module are
either subclasses of :py:class:`_ReaderState` that implement the functionality
of the :py:class:`Reader` as it goes through different states or classes that
aid those. The different states, in turn, refer to which line the
:py:class:`Reader` is expecting (see
:py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`).

Refer to the documentation for the package
:py:mod:`muscle_synergies.vicon_data.__init__.py` for more on how
:py:class:`Reader` fits together with the other classes used for reading the
data from disk.
"""
import abc
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Mapping, Optional, Tuple, Union

from muscle_synergies.vicon_data.aggregator import Aggregator
from muscle_synergies.vicon_data.definitions import (
    DeviceType,
    Row,
    SectionType,
    ViconCSVLines,
)


class Reader:
    """A line-by-line parser of the CSV file outputted by Vicon Nexus.

    Lines should be fed in order to :py:meth:`Reader.feed_row` and they will
    automatically be parsed and sent to the
    :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator`.

    The logic of parsing the lines, storing the parsed data and knowing which
    line is expected next is actually implemented by different
    :py:class:`_ReaderState` subclasses.

    Args:
        section_type_state: the `Reader`'s initial state.

        aggregator: the
            :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator` used
            to store the data as it is being read.
    """

    _state: "_ReaderState"
    _aggregator: Aggregator

    def __init__(self, section_type_state: "SectionTypeState", aggregator: Aggregator):
        self._state = section_type_state
        self._aggregator = aggregator

    def feed_row(self, row: Row):
        """Feed a data row to the Reader.

        Args:
            row: a row from the CSV file. The rows should be fed to the
                `Reader` sequentially.
        """
        self._state.feed_row(row, reader=self)

    def set_state(self, new_state: "_ReaderState"):
        """Set the Reader's state."""
        self._state = new_state

    def get_aggregator(self) -> Aggregator:
        """Get the Reader's Aggregator."""
        return self._aggregator

    def get_section_type(self) -> SectionType:
        """Get the type of the section from which the Reader expects rows."""
        return self._aggregator.get_section_type()


class _ReaderState(abc.ABC):
    """ABC for the internal state of the Reader.

    :py:class:`Reader` only calls a single method of its internal state, which
    is :py:meth:`_ReaderState.feed_row`, so all subclasses of
    :py:class:`_ReaderState` must implement that method.  The only other thing
    a minimal implementation should provide is :py:attr:`_ReaderState.line`.
    """

    @abc.abstractmethod
    def feed_row(self, row: Row, *, reader: "Reader"):
        """Parse a row, store the result and transition to the next state.

        Exactly what this method achieves depends on the line that is being
        parsed and thus will differ across the different subclasses of
        :py:class:`_ReaderState`. Roughly, the method is responsible for 3 main
        things:

        1. parsing the data.
        2. Sending it to
           :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator`.
        3. Updating the state of the :py:class:`Reader`.

        Args:
            row: the next row from the Vicon Nexus CSV file.

            reader: the `Reader` object which holds the
                :py:class:`_ReaderState` instance.
        """

    @abc.abstractproperty
    def line(self) -> ViconCSVLines:
        """The Vicon CSV line expected by the state.

        This is mainly useful to provide more helpful error messages when
        something goes wrong.
        """

    def _preprocess_row(self, row: Row) -> Row:
        """Removes trailing/leading whitespace from cols and empty cols.

        All columns in the row have their whitespace removed using
        :py:meth`str.strip`. Then columns are removed from the end until a
        nonempty one is found.

        The operation is *not* done in-place, a new
        :py:class:`~muscle_synergies.vicon_data.definitions.Row` is returned.
        """
        processed = list(entry.strip() for entry in row)

        while processed and not processed[-1]:
            processed.pop()
        return Row(processed)

    def _reader_aggregator(self, reader: Reader) -> Aggregator:
        """Get the Aggregator associated with the Reader."""
        return reader.get_aggregator()

    def _reader_set_state(self, reader: Reader, new_state: "_ReaderState"):
        """Set the state of the Reader to a new state."""
        reader.set_state(new_state)

    def _reader_section_type(self, reader: Reader) -> SectionType:
        """Get the section type associated with the Reader."""
        return reader.get_section_type()


class _UpdateStateMixin:
    """Mixin for a _ReaderState that updates the state of the Reader.

    Subclasses should implement :py:meth:`~_UpdateStateMixin._next_state_type`
    and then they get to just call :py:meth:`_UpdateStateMixin._update_state`
    and the :py:class:`Reader`'s state will be updated.
    """

    def _update_state(self, reader: Reader):
        """Update the state of the Reader.

        The new state will be the return value of
        :py:meth:`~_UpdateStateMixin._new_state`.
        """
        self._reader_set_state(reader, self._new_state(reader))

    def _new_state(self, reader: Reader) -> _ReaderState:
        """Create a fresh new state object."""
        st_type = self._next_state_type(reader)
        return st_type()

    @abc.abstractmethod
    def _next_state_type(self, reader: Reader) -> Callable[[Any], _ReaderState]:
        """Determine the type of the next state.

        :py:class:`Reader` is passed as an argument because sometimes it need
        be inspected to determine which state should be the next.

        Returns:
            a callable that can be used to instantiate the next
            :py:class:`_ReaderState`.
        """


class _HasSingleColMixin:
    """Mixin for a _ReaderState that expects to see data in a single col."""

    def _validate_row_has_single_col(self, row: Row):
        """Raise exception if the row contains data outside its first column.

        The row is expected to have been preprocessed pruning away all empty
        columns at its end.

        Raises:
            ValueError: if the preprocessed row has entries outside its first
                one.
        """
        if row[1:]:
            raise ValueError(
                f"row {self.line} should contain nothing outside " + "its first column"
            )


class _AggregateDataMixin:
    """Mixin for a _ReaderState that stores data in the Aggregator.

    Subclasses of :py:class:`_ReaderState` adopting this mixin should implement
    :py:meth:`~_AggregateDataMixin._get_data_aggregate_method`. Then they can
    simply call :py:meth:`~_AggregateDataMixin._aggregate_data` to send the
    message to the :py:class:`Reader`'s
    :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator`.
    """

    @abc.abstractmethod
    def _get_data_aggregate_method(
        self, aggregator: Aggregator
    ) -> Callable[[Any], None]:
        """Get the correct Aggregator method to store the data.

        Returns:
            a callable, corresponding to the method that should be called to
                send the message to the
                :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator`.
        """

    def _aggregate_data(self, data: Any, reader: Reader):
        """Aggregate the results of parsing the line.

        This gets the
        :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator`.
        associated with the :py:class:`Reader` and calls its appropriate method
        to send parsed data to it.
        """
        method = self._get_data_aggregate_method(self._reader_aggregator(reader))
        method(data)


class _FixedNumColsMixin:
    """Mixin for a _ReaderState that knows how many columns it expects.

    Subclasses should have a property :py:attr:`~_FixedNumColsMixin.num_cols`
    and then :py:meth:`_ReaderState._preprocess_row` will be supplanted by
    :py:meth:`_FixedNumColsMixin._preprocess_row`. That method will restrict
    the row to exactly the number of columns it should have.
    """

    def _preprocess_row(self, row: Row) -> Row:
        """Keep only the correct number of columns of the row."""
        return Row(row[0 : self.num_cols])

    @abc.abstractproperty
    def num_cols(self) -> int:
        """Number of columns expected by the _ReaderState."""


class SectionTypeState(_UpdateStateMixin, _HasSingleColMixin, _ReaderState):
    """The state of a reader that is expecting the section type line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for
    :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.
    """

    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.SECTION_TYPE_LINE

    def _next_state_type(self, reader: Reader):
        return SamplingFrequencyState

    def feed_row(self, row: Row, *, reader: Reader):
        """Parse the section type line.

        Raises:
            ValueError in these cases

                + the row contains data outside its first column.
                + The first column is not one of `"Devices"` or
                  `"Trajectories"`.
                + The parsed section type doesn't match the
                  :py:class:`Reader`'s.
        """
        row = self._preprocess_row(row)
        self._validate_row_has_single_col(row)
        section_type = self._parse_section_type(row)
        self._validate_section_type(section_type, reader)
        self._update_state(reader)

    def _validate_row_valid_values(self, row: Row):
        """Raise exception if the first column is not understood."""
        if row[0] not in {"Devices", "Trajectories"}:
            raise ValueError(
                'first row in a section should contain "Devices" or "Trajectories" in its first column'
            )

    def _parse_section_type(self, row: Row) -> SectionType:
        """Parse section type from row."""
        section_type_str = row[0]

        if section_type_str == "Devices":
            return SectionType.FORCES_EMG
        if section_type_str == "Trajectories":
            return SectionType.TRAJECTORIES
        raise ValueError(
            'first row in a section should contain "Devices" or "Trajectories" in its first column'
        )

    def _validate_section_type(self, parsed_type: SectionType, reader: Reader):
        """Raise exception if the parsed section type is wrong."""
        current_type = self._reader_section_type(reader)
        if current_type is not parsed_type:
            raise ValueError(
                f"row implies current section is {parsed_type} but expected {current_type}"
            )


class SamplingFrequencyState(
    _UpdateStateMixin, _AggregateDataMixin, _HasSingleColMixin, _ReaderState
):
    """The state of a reader that is expecting the sampling frequency line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for
    :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.
    """

    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.SAMPLING_FREQUENCY_LINE

    def _next_state_type(self, reader: Reader):
        if self._reader_section_type(reader) is SectionType.FORCES_EMG:
            return ForcesEMGDevicesState
        return TrajDevicesState

    def _get_data_aggregate_method(
        self, aggregator: Aggregator
    ) -> Callable[[int], None]:
        return aggregator.add_frequency

    def feed_row(self, row: Row, *, reader: Reader):
        """Parse the frequency line.

        The `row` is assumed to have a single entry, in its first element
        (`row[0]`), corresponding to the sampling rate with which the
        measurements were made. The sampling rate is assumed to be given as an
        integer in Hz.

        Raises:
            ValueError: if the row doesn't have exactly one column, its first,
                with data.
        """
        row = self._preprocess_row(row)
        self._validate_row_has_single_col(row)
        freq = self._parse_freq(row)
        self._aggregate_data(freq, reader)
        self._update_state(reader)

    @staticmethod
    def _parse_freq(row: Row):
        """Parse the sampling rate from the row containing it."""
        return int(row[0])


@dataclass
class ColOfHeader:
    """The string describing a device and the column in which it occurs.

    This is used as an intermediate representation of the data being read in
    the device names line (see
    :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`). The
    structure of that line is complex, so the logic of its parsing is split
    into several classes. :py:class:`ColOfHeader` is used for communication
    between them.

    Args:
        col_index: the index of the column in the CSV file in which the
            device header is described.

        header_str: the exact string occurring in that column.
    """

    col_index: int
    header_str: str


class DevicesHeaderFinder:
    """Find device names in the devices line.

    The devices line has 2 empty columns and then one device header every 3
    columns. :py:meth:`~DevicesHeaderFinder.find_headers` returns those headers
    together with the index in which they occur.

    Examples:
        >>> devices_line = ['', '', 'Device_1', '', '', 'Device_2', '', '']
        >>> finder = DevicesHeaderFinder()
        >>> finder(devices_line)
        [ColOfHeader(col_index=2, header_str='Device_1'), ColOfHeader(col_index=5, header_str='Device_2')]
    """

    def find_headers(self, row: Row) -> List[ColOfHeader]:
        """Find device names in the devices line.

        Raises:
            ValueError: if the row does not conform to the expected structure.
        """
        self._validate_row_values_in_correct_cols(row)
        return self._list_col_of_headers(row)

    __call__ = find_headers

    def _list_col_of_headers(self, row: Row) -> List[ColOfHeader]:
        """Build a ColOfHeader for each expected device name.

        This method creates a :py:class:`ColOfHeader` object for every 3
        positions after the first 2. Those are the positions in which device
        headers are expected to occur.
        """
        return [self._col_of_header(row[i], i) for i in range(2, len(row), 3)]

    def _col_of_header(self, name: str, col: int) -> ColOfHeader:
        """Create new ColOfHeader instance."""
        return ColOfHeader(col_index=col, header_str=name)

    def _validate_row_values_in_correct_cols(self, row: Row):
        """Raise exception if the row does not have the expected structure."""

        def error():
            raise ValueError(
                "this line should contain two blank columns "
                + "then one device name every 3 columns"
            )

        if row[0] or row[1]:
            error()

        for col_num, col_val in enumerate(row[2:], start=2):
            if self._col_should_contain_name(col_num):
                current_is_correct = col_val
            else:
                current_is_correct = not col_val

            if not current_is_correct:
                error()

    @staticmethod
    def _col_should_contain_name(col_num) -> bool:
        """Determine if column should contain a device name."""
        return (col_num - 2) % 3 == 0


class ForcePlateGrouper:
    """Group force plate devices.

    Force plates are represented as 3 headers, for example:

    1. `Imported AMTI OR6 Series Force Plate #1 - Force`
    2. `Imported AMTI OR6 Series Force Plate #1 - Moment`
    3. `Imported AMTI OR6 Series Force Plate #1 - CoP`

    It is assumed that each of these span 3 columns (referring to the 3 spatial
    coordinates) and occur contiguously precisely in that order, that is, for
    each force plate, "Force" comes first, then "Moment" and finally "CoP".

    :py:meth:`ForcePlateGrouper.group` unifies the representation of these
    various device headers referring to the same force plate (in the case under
    discussion, "Imported AMTI OR6 Series Force Plate #1").
    """

    def group(self, headers: List[ColOfHeader]) -> List[ColOfHeader]:
        """Group force plate devices.

        Args:
            headers: a list of device headers as they would be found in the
                devices line. It is assumed that every adjacent 3 of them refer
                to a unique force plate.

        Returns:
            only the first device header of each force plate, with its name
            simplified. For example, the 3 device headers (see
            :py:class:`ColOfHeader`) below would be renamed to "Imported AMTI
            OR6 Series Force Plate #1", keeping the column index of the first
            one.

            1. Imported AMTI OR6 Series Force Plate #1 - Force
            2. Imported AMTI OR6 Series Force Plate #1 - Moment
            3. Imported AMTI OR6 Series Force Plate #1 - CoP
        """
        grouped_headers = []
        for header in self._filter_individual_force_plates(headers):
            grouped_headers.append(self._rename_force_plate(header))
        return grouped_headers

    __call__ = group

    def _filter_individual_force_plates(
        self, headers: List[ColOfHeader]
    ) -> Iterator[ColOfHeader]:
        """Filter individual force plates from the list of device headers.

        Yields:
            the first and then every third device header.
        """
        for i, head in enumerate(headers):
            if i % 3 == 0:
                yield head

    def _rename_force_plate(self, header: ColOfHeader) -> ColOfHeader:
        """Rename force plate."""
        header_str = self._col_of_header_header_str(header)
        new_name = self._force_plate_name(header_str)
        first_col = self._col_of_header_first_col(header)
        return self._col_of_header(new_name, first_col)

    def _force_plate_name(self, header_str: str):
        """Find the name of the force plate from the text in the column header.

        For example, "Imported AMTI OR6 Series Force Plate #1 - Force" becomes
        "Imported AMTI OR6 Series Force Plate #1".
        """
        force_plate_name, _ = header_str.split("-")
        return force_plate_name[:-1]

    def _col_of_header_header_str(self, header: ColOfHeader) -> str:
        """Get the device header string of a ColOfHeader object."""
        return header.header_str

    def _col_of_header_first_col(self, header: ColOfHeader) -> int:
        """Get the device header colum index of a ColOfHeader object."""
        return header.col_index

    def _col_of_header(self, header_str: str, first_col: int) -> ColOfHeader:
        """Build new ColOfHeader."""
        return ColOfHeader(header_str=header_str, col_index=first_col)


class _DevicesState(_UpdateStateMixin, _ReaderState):
    """ABC for the state of a Reader that is expecting the devices line.

    Parsing of this line differs depending on the section of the file being
    parsed. The logic is implemented by subclasses of
    :py:class:`_DevicesState`. The main pieces of information found in this
    line are the names of the different devices and the columns belonging to
    each of them. A minimal implementation must provide
    :py:meth:`~_DevicesState.last_col` and
    :py:meth:`~_DevicesState._send_headers_to_aggregator`.

    For an explanation of what are the different lines of the CSV input, see
    the docs for
    :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.

    Args:
        finder: if `None` is provided, instantiate a new
            :py:class:`DevicesHeaderFinder`, which will be used to find the
            device headers.
    """

    finder: DevicesHeaderFinder

    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.DEVICE_NAMES_LINE

    def __init__(self, finder: Optional[DevicesHeaderFinder] = None):
        super().__init__()
        if finder is None:
            finder = self._instantiate_finder()
        self.finder = finder

    def feed_row(self, row: Row, *, reader: Reader):
        """Parse the devices line and add the devices to the Aggregator.

        Following this call, the :py:class:`Reader`'s state will always be
        transitioned to  :py:class:`CoordinatesState`.
        """
        row = self._preprocess_row(row)
        headers = self._find_headers(row)
        self._send_headers_to_aggregator(headers, reader)
        self._update_state(reader)

    def _add_device(self, reader: Reader, header: ColOfHeader, device_type: DeviceType):
        """Add new device to Aggregator."""
        self._aggregator_add_device(
            self._reader_aggregator(reader),
            **self._build_add_device_params_dict(header, device_type),
        )

    @staticmethod
    def _aggregator_add_device(
        aggregator: Aggregator,
        name: str,
        device_type: DeviceType,
        first_col: int,
        last_col: int,
    ):
        """Ask Aggregator to store new device."""
        aggregator.add_device(
            name=name, device_type=device_type, first_col=first_col, last_col=last_col
        )

    def _next_state_type(self, reader: Reader):
        return CoordinatesState

    def _build_add_device_params_dict(
        self, header: ColOfHeader, device_type: DeviceType
    ) -> Mapping:
        """Build parameter dict for _DevicesState._aggregator_add_device.

        This method accomplishes 2 things:

        1. get the parameter dict in the shape expected by
           :py:meth:`Aggregator.add_device`
        2. compute a previously unavailable piece of information expected by
           that method, the last column of the device. This uses
           :py:meth:`~_DevicesState._last_col`.
        """
        unpacked = self._unpack_col_of_header(header)
        unpacked["device_type"] = device_type
        first_col = unpacked["first_col"]
        unpacked["last_col"] = self._last_col(device_type, first_col)
        return unpacked

    def _unpack_col_of_header(
        self, header: ColOfHeader
    ) -> Mapping[str, Union[str, int]]:
        """Unpack values from ColOfHeader."""
        return {"first_col": header.col_index, "name": header.header_str}

    def _find_headers(self, row: Row):
        """Find device headers on the devices line."""
        return self.finder(row)

    def _instantiate_finder(self) -> DevicesHeaderFinder:
        """Instantiate a new DevicesHeaderFinder object."""
        return DevicesHeaderFinder()

    @abc.abstractmethod
    def _send_headers_to_aggregator(self, headers: List[ColOfHeader], reader: Reader):
        """Process a list of device headers and send them to the Aggregator.

        Args:
            headers: this is a list of all device headers in the line. If they
                must be processed (e.g., unifying different device headers that
                actually belong to a single force plate), concrete
                implementations should do it.
        """

    @abc.abstractmethod
    def _last_col(self, device_type: DeviceType, first_col: int) -> Optional[int]:
        """Determine the last column corresponding to the given device.

        This number follows from these assumptions:

        1. a force plate uses 9 contiguous columns across its 3 device headers.
        2. a trajectory marker uses 3.
        3. EMG measurements have 1+ columns and end at the end of the line.
        4. the CSV file can have blank columns at the end, so it's impossible
           to know how many EMG columns there are just by looking at the
           devices line.

        To help understand the last point, take the following 2 lines
        representing a simplified version of the devices line and the
        coordinates line, which comes after it:

            EMG - Voltage,,,,,
            VL,GM,,,,

        There are 3 empty columns at the end. Just by looking at the first
        line, one cannot know how many EMG measurements there are.
        """


class ForcesEMGDevicesState(_DevicesState):
    """The state of a Reader expecting the devices line of the first section.

    The main pieces of information found in this line are the names of the
    different devices and the columns belonging to each of them. For an
    explanation of what are the different lines of the CSV input, see the docs
    for :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.

    Args:
        finder: if `None` is provided, instantiate a new
            :py:class:`DevicesHeaderFinder`, which will be used to find the
            device headers.

        grouper: if `None` is provided, instantiate a new
            :py:class:`ForcePlateGrouper`, which will be used to unify the
            different headers referring to the same force plate.
    """

    grouper: ForcePlateGrouper

    def __init__(
        self,
        finder: Optional[DevicesHeaderFinder] = None,
        grouper: Optional[ForcePlateGrouper] = None,
    ):
        super().__init__(finder)
        if grouper is None:
            grouper = self._instantiate_grouper()
        self.grouper = grouper

    def _send_headers_to_aggregator(self, headers: List[ColOfHeader], reader: Reader):
        force_plates_headers, emg = self._separate_headers(headers)
        grouped_force_plates = self._group_force_plates(force_plates_headers)
        for header in grouped_force_plates:
            self._add_device(reader, header, DeviceType.FORCE_PLATE)
        self._add_device(reader, emg, DeviceType.EMG)

    def _separate_headers(
        self, headers: List[ColOfHeader]
    ) -> Tuple[List[ColOfHeader], ColOfHeader]:
        """Separate all the force plates from the EMG device."""
        force_plates_headers = headers[:-1]
        emg_header = headers[-1]
        return force_plates_headers, emg_header

    def _group_force_plates(self, headers: List[ColOfHeader]) -> List[ColOfHeader]:
        """Unify device headers relating to the same force plate."""
        return self.grouper.group(headers)

    def _instantiate_grouper(self) -> ForcePlateGrouper:
        """Create a new ForcePlateGrouper instance."""
        return ForcePlateGrouper()

    def _last_col(self, device_type: DeviceType, first_col: int) -> Optional[int]:
        assert device_type is not DeviceType.TRAJECTORY_MARKER

        if device_type is DeviceType.EMG:
            return self._last_col_of_emg()
        if device_type is DeviceType.FORCE_PLATE:
            return self._last_col_of_force_plate(first_col)
        return None

    def _last_col_of_force_plate(self, first_col: int) -> int:
        """Determine the last column of a force plate."""
        return first_col + 9 - 1

    def _last_col_of_emg(self) -> None:
        """Determine the last column of an EMG device."""
        return None


class TrajDevicesState(_DevicesState):
    """The state of a Reader expecting the devices line of the second section.

    The main pieces of information found in this line are the names of the
    different devices and the columns belonging to each of them. For an
    explanation of what are the different lines of the CSV input, see the docs
    for :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.

    Args:
        finder: if `None` is provided, instantiate a new
            :py:class:`DevicesHeaderFinder`, which will be used to find the
            device headers.
    """

    def _send_headers_to_aggregator(self, headers: List[ColOfHeader], reader: Reader):
        for header in headers:
            self._add_device(reader, header, DeviceType.TRAJECTORY_MARKER)

    def _last_col(self, _: DeviceType, first_col: int) -> int:
        return first_col + 3 - 1


class CoordinatesState(_AggregateDataMixin, _ReaderState):
    """The state of a reader that is expecting the coordinates line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for
    :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.
    """

    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.COORDINATES_LINE

    def feed_row(self, row: Row, *, reader: Reader):
        """Parse the coordinates line.

        The number of data columns in all subsequent lines of the section is
        determined by this method and passed along to the next
        :py:class:`_ReaderState` in the chain. All empty columns at the end of
        the row are assumed to not contain any data.
        """
        row = self._preprocess_row(row)
        num_cols = len(row)
        self._aggregate_data(row, reader)
        self._update_state(reader, num_cols)

    def _get_data_aggregate_method(
        self, aggregator: Aggregator
    ) -> Callable[[List[str]], None]:
        return aggregator.add_coordinates

    def _update_state(self, reader: Reader, num_cols: int):
        self._reader_set_state(reader, self._new_state(num_cols))

    def _new_state(self, num_cols: int) -> _ReaderState:
        return UnitsState(num_cols)


class UnitsState(_FixedNumColsMixin, _AggregateDataMixin, _ReaderState):
    """The state of a reader that is expecting the units line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for
    :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.

    Args:
        num_cols: see :py:attr:`_AggregateDataMixin.num_cols`.
    """

    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.UNITS_LINE

    def __init__(self, num_cols: int):
        super().__init__()
        self._num_cols = num_cols

    @property
    def num_cols(self) -> int:
        return self._num_cols

    def feed_row(self, row: Row, *, reader: Reader):
        """Parse the units line."""
        units = self._preprocess_row(row)
        self._aggregate_data(units, reader)
        self._update_state(reader)

    def _get_data_aggregate_method(
        self, aggregator: Aggregator
    ) -> Callable[[List[str]], None]:
        return aggregator.add_units

    def _update_state(self, reader: Reader):
        self._reader_set_state(reader, self._new_state())

    def _new_state(self) -> _ReaderState:
        return GettingMeasurementsState(num_cols=self.num_cols)


class GettingMeasurementsState(_ReaderState):
    """The state of a reader expecting data lines until the end of the section.

    An arbitrary number of data lines is fed to the :py:class:`Reader`.
    Following them, a blank line marks the end of the section in the file.
    :py:class:`GettingMeasurementsState` receives these lines from the
    :py:class:`Reader` and passes them forward to one of two parsers:

    + a :py:class:`DataState` instance
    + a :py:class:`BlankState` instance

    These parsers are actually full-fledged instances of
    :py:class:`_ReaderState` though they are not meant to ever be set as the
    :py:class:`Reader`'s state. They just implement the logic of the two
    possible lines that can be received by
    :py:class:`GettingMeasurementsState`.

    For an explanation of what are the different lines of the CSV input, see
    the docs for
    :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.

    Args:
        num_cols: see :py:attr:`_AggregateDataMixin.num_cols`.

        data_state: if `None` is provided, instantiate a new
            :py:class:`DataState`, which will be used to handle actual data
            lines.

        blank_state: if `None` is provided, instantiate a new
            :py:class:`BlankState`, which will be used to handle an empty line
            marking the end of a section in the CSV file.
    """

    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.DATA_LINE

    def __init__(
        self,
        data_state: Optional["DataState"] = None,
        blank_state: Optional["BlankState"] = None,
        num_cols: Optional[int] = None,
    ):
        if data_state is None:
            self.data_state = DataState(num_cols)
        if blank_state is None:
            self.blank_state = BlankState()

    def feed_row(self, row: Row, *, reader: Reader):
        """Feed the row forward to the appropriate parser.

        A row is considered blank if after striping whitespace, all its columns
        are empty strings. Empty rows are fed to
        :py:attr:`~GettingMeasurementsState.blank_state` and nonempty ones are
        fed to :py:attr:`~GettingMeasurementsState.data_state`.
        """
        if self._is_blank_line(row):
            self.blank_state.feed_row(row, reader=reader)
        else:
            self.data_state.feed_row(row, reader=reader)

    def _is_blank_line(self, row: Row) -> bool:
        """Determine if a row has no data."""
        return not bool(self._preprocess_row(row))


class DataState(_FixedNumColsMixin, _AggregateDataMixin, _ReaderState):
    """The state of a reader that is being fed a data line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for
    :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.

    Args:
        num_cols: see :py:attr:`_AggregateDataMixin.num_cols`.
    """

    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.DATA_LINE

    def __init__(self, num_cols):
        super().__init__()
        self._num_cols = num_cols

    @property
    def num_cols(self) -> int:
        return self._num_cols

    def feed_row(self, row: Row, *, reader: Reader):
        """Parse a new data line.

        Only the columns of index smaller than :py:attr:`~DataState.num_cols`
        are parsed. Those should either be an empty string, which is then
        converted to a value of `None` or a string containing a float, which is
        converted to it. The parsed data is then fed to the
        :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator`.
        """
        row = self._preprocess_row(row)
        floats = self._parse_row(row)
        self._aggregate_data(floats, reader)

    def _parse_row(self, row: Row) -> List[Optional[float]]:
        """Parse all columns of a data line."""
        return list(map(self._parse_entry, row))

    def _parse_entry(self, row_entry: str) -> Optional[float]:
        """Parse a single entry of a data line."""
        if not row_entry:
            return None
        return float(row_entry)

    def _get_data_aggregate_method(
        self, aggregator: Aggregator
    ) -> Callable[[List[Optional[float]]], None]:
        return aggregator.add_data


class BlankState(_UpdateStateMixin, _ReaderState):
    """The state of a reader that is being fed the blank line.

    This refers to the empty line at the end of a section.  For an explanation
    of what are the different lines of the CSV input, see the docs for
    :py:class:`~muscle_synergies.vicon_data.definitions.ViconCSVLines`.

    Args:
        num_cols: see :py:attr:`_AggregateDataMixin.num_cols`.
    """

    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.BLANK_LINE

    def feed_row(self, row: Row, *, reader: Reader):
        """Transition the parsing process to a new section.

        This notifies the
        :py:class:`~muscle_synergies.vicon_data.aggregator.Aggregator` that the
        section has changed and updates the state of the :py:class:`Reader`.
        The `row` itself is not used.
        """
        self._aggregator_transition(self._reader_aggregator(reader))
        self._update_state(reader)

    def _next_state_type(self, reader: Reader):
        return SectionTypeState

    def _aggregator_transition(self, aggregator: Aggregator):
        """Notify Aggregator that a new section will start."""
        aggregator.transition()
