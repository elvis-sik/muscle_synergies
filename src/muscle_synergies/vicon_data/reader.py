import abc
from collections import defaultdict
import csv
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator, TypeVar, NewType, Union, Iterable)

import pandas as pd
import pint
from pint_pandas import PintArray

from .definitions import (
    ureg,
    T,
    X,
    Y,
    Row,
    DeviceHeaderRepresentation,
    ForcePlateRepresentation,
    SectionType,
    ViconCSVLines,
    DeviceType,
    ForcePlateMeasurement,
)
from .aggregator import (
    ViconNexusData,
    ColOfHeader,
    DeviceHeaderPair,
    DeviceHeaderCols,
    DeviceHeaderAggregator,
    ForcePlateDevices,
    Aggregator,
    DataChanneler,
)


class Reader:
    """Reader for a single section of the CSV file outputted by Vicon Nexus.

    Initialize it
    """
    _state: '_ReaderState'
    _aggregator: Aggregator

    def __init__(self, section_type_state: 'SectionTypeState',
                 aggregator: Aggregator):
        self._state = section_type_state
        self._aggregator = aggregator

    def file_ended(self) -> ViconNexusData:
        self._state.file_ended(reader=self)

    def feed_row(self, row: Row):
        self._state.feed_row(row, reader=self)

    def set_state(self, new_state: '_ReaderState'):
        self._state = new_state

    def get_aggregator(self):
        return self._aggregator

    def get_section_type(self) -> SectionType:
        return self._aggregator.get_section_type()


class _ReaderState(abc.ABC):
    @abc.abstractmethod
    def feed_row(self, row: Row, *, reader: 'Reader'):
        pass

    @abc.abstractproperty
    def line(self) -> ViconCSVLines:
        pass

    def file_ended(self, reader: 'Reader') -> ViconNexusData:
        current_section_type = self._reader_section_type(reader)
        current_line = self.line
        data_check = self._create_data_check(
            False, "file doesn't seem to have the expected structure. " +
            "It was expected to have two sections with data " +
            "(one for force plate and EMG data, the other for trajectory markers) "
            +
            "with each section having 5 lines before a variable number of lines "
            + "containing measurements. " +
            f"Current section type is {current_section_type} and the line "
            f"currently expected is {current_line}.")
        validator = self._reader_validator(reader)
        self._validate(validator, data_check)

    def _preprocess_row(self, row: Row) -> Row:
        row = list(entry.strip() for entry in row)

        while row and row[-1]:
            row.pop()
        return Row(row)

    def _reader_aggregator(self, reader: 'Reader') -> Aggregator:
        return reader.get_aggregator()

    def _reader_set_state(self, reader: 'Reader', new_state: '_ReaderState'):
        reader.set_state(new_state)

    def _reader_section_type(self, reader: Reader) -> SectionType:
        return reader.get_section_type()

    def _reader_set_new_section_type(self, reader: Reader,
                                     new_section_type: SectionType):
        reader.set_section_type(new_section_type)


class _UpdateStateMixin:
    def _update_state(self, reader: Reader):
        self._reader_set_state(reader, self._new_state())

    def _new_state(self):
        st_type = self._next_state_type
        return st_type()

    @abc.abstractproperty
    def _next_state_type(self):
        pass


class _HasSingleColMixin:
    def _validate_row_has_single_col(self, row: Row):
        if row[1:]:
            raise ValueError(
                f'first row of a section should contain nothing outside its first column'
            )


class _AggregateDataMixin:
    @abc.abstractmethod
    def _get_data_aggregate_method(self, aggregator: Aggregator
                                   ) -> Callable[[Any], None]:
        pass

    def _aggregate_data(self, data: Any, reader: Reader):
        method = self._get_data_aggregate_method(
            self._reader_aggregator(reader))
        method(data)


class _EntryByEntryMixin(abc.ABC):
    def _parse_row(self, row: Row) -> List[T]:
        return [self._parse_entry(row_entry) for row_entry in row]

    @abc.abstractmethod
    def _parse_entry(row_entry: str) -> T:
        pass


class _PassUpFileEndedMixin:
    def file_ended(self, reader: Reader) -> ViconNexusData:
        aggregator = self._reader_aggregator(reader)
        return aggregator.file_ended()


class SectionTypeState(_UpdateStateMixin, _HasSingleEntryMixin, _ReaderState):
    """The state of a reader that is expecting the section type line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for :py:class:ViconCSVLines.
    """
    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.SECTION_TYPE_LINE

    @property
    def _next_state_type(self):
        return SamplingFrequencyState

    def feed_row(self, row: Row, reader: Reader):
        row = self._preprocess_row(row)
        self._validate_row_valid_values(row)
        self._validate_row_has_single_col(row)
        section_type = self._parse_section_type(row)
        self._validate(section_type)
        self._update_state(reader)

    def _validate_row_valid_values(self, row: Row):
        if row[0] not in {'Devices', 'Trajectories'}:
            raise ValueError(
                f'first row in a section should contain "Devices" or "Trajectories" in its first column'
            )

    def _parse_section_type(self, row: Row) -> SectionType:
        section_type_str = row[0]

        if section_type_str == 'Devices':
            return SectionType.FORCES_EMG
        elif section_type_str == 'Trajectories':
            return SectionType.TRAJECTORIES

    def _validate_section_type(self, parsed_type: SectionType, reader: Reader):
        current_type = self._reader_section_type(reader)
        if current_type is not parsed_type:
            raise ValueError(
                f'row implies current section is {parsed_type} but expected {current_type}'
            )


class SamplingFrequencyState(_AggregateDataMixin, _UpdateStateMixin,
                             _HasSingleColMixin, _ReaderState):
    """The state of a reader that is expecting the sampling frequency line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for :py:class:ViconCSVLines.
    """
    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.SAMPLING_FREQUENCY_LINE

    @property
    def _next_state_type(self):
        return DevicesState

    def _get_data_aggregate_method(self, aggregator: Aggregator
                                   ) -> Callable[[int], None]:
        return aggregator.add_frequency

    def feed_row(self, row: Row, reader: Reader):
        row = self._preprocess_row(row)
        self._validate_has_single_col(row)
        freq = self._parse_freq(row)
        self._aggregate_data(freq, reader)
        self._update_state(reader)

    @staticmethod
    def _parse_freq(row: Row):
        return int(row[0])


@dataclass
class ColOfHeader:
    """The string describing a device and the column in which it occurs.

    This is used as an intermediate representation of the data being read in
    the device names line (see :py:class:ViconCSVLines). The structure of that
    line is complex, so the logic of its parsing is split into several classes.
    ColOfHeader is used for communication between them.

    Args:
        col_index: the index of the column in the CSV file in which the
            device header is described.

        header_str: the exact string occurring in that column.
    """
    col_index: int
    header_str: str


class DevicesHeaderFinder:
    def find_headers(self, row: Row) -> List[ColOfHeader]:
        self._validate_row_values_in_correct_cols(row)
        return self._find_headers_unsafe(row)

    __call__ = find_headers

    def _find_headers_unsafe(self, row: Row) -> List[ColOfHeader]:
        return [self._col_of_header(row[i], i) for i in range(2, len(row), 3)]

    def _col_of_header(self, name, col) -> ColOfHeader:
        return ColOfHeader(col_index=col, header_str=name)

    def _validate_row_values_in_correct_cols(self, row: Row):
        def error():
            raise ValueError('this line should contain two blank columns ' +
                             'then one device name every 3 columns')

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
    def _col_should_contain_name(col_num):
        return (col_num - 2) % 3 == 0


class _DevicesState(_UpdateStateMixin, _ReaderState):
    finder: DevicesHeaderFinder

    def __init__(self, finder: Optional[DevicesHeaderFinder] = None):
        super().__init__()
        if finder is None:
            self._finder = self._instantiate_finder()
        self._finder = finder

    def feed_row(self, row: Row, reader: Reader):
        row = self._preprocess_row(row)
        headers = self._find_headers(row)
        self._send_headers_to_aggregator(headers, reader)
        self._update_state(reader)

    def _add_device(self, reader: Reader, header: ColOfHeader,
                    device_type: DeviceType,
                    last_col_func: Callable[[int], None]):
        self._aggregator_add_device(
            self._reader_aggregator(reader),
            **self._build_add_device_params_dict(header, device_type))

    def _aggregator_add_device(self, aggregator: Aggregator, name: str,
                               device_type: DeviceType, first_col: int,
                               last_col: int):
        aggregator.add_device(name=name,
                              device_type=device_type,
                              first_col=first_col,
                              last_col=last_col)

    @property
    def _next_state_type(self):
        return CoordinatesState

    def _build_add_device_params_dict(self, header: ColOfHeader,
                                      device_type: DeviceType):
        unpacked = self._unpack_col_of_header(header)
        unpacked['device_type'] = device_type
        first_col = unpacked['first_col']
        unpacked['last_col'] = self._last_col(device_type, first_col)
        return unpacked

    def _unpack_col_of_header(self, header: ColOfHeader
                              ) -> Mapping[str, Union[str, int]]:
        return {'first_col': header.col_index, 'name': header.header_str}

    def _find_headers(self, row: Row):
        return self.finder(row)

    def _instantiate_finder(self):
        return DevicesHeaderFinder()

    @abc.abstractmethod
    def _send_headers_to_aggregator(self, headers: List[ColOfHeader],
                                    reader: Reader):
        pass

    @abc.abstractmethod
    def _last_col(self, device_type: DeviceType, first_col: int) -> Any:
        pass


class ForcePlateGrouper:
    def group(self, headers: List[ColOfHeader]) -> List[ColOfHeader]:
        grouped_headers = []
        for header in self._filter_individual_force_plates(headers):
            grouped_headers.append(self._rename_force_plate(header))
        return grouped_headers

    def _filter_individual_force_plates(self, headers: List[ColOfHeader]
                                        ) -> Iterator[ColOfHeader]:
        for i, head in enumerate(headers):
            if i % 3 == 0:
                yield head

    def _rename_force_plate(self, header: ColOfHeader) -> ColOfHeader:
        header_str = self._header_str(header)
        new_name = self._get_force_plate_name(header_str)
        first_col = self._col_of_header_first_col(header)
        return self._col_of_header(new_name, first_col)

    def _get_force_plate_name(self, header_str: str):
        force_plate_name, measurement_name = header_str.split('-')
        return force_plate_name[:-1]

    def _col_of_header_header_str(self, header: ColOfHeader) -> str:
        return header.header_str

    def _col_of_header_first_col(self, header: ColOfHeader) -> int:
        return header.col_index

    def _col_of_header(self, header_str: str, first_col: int) -> ColOfHeader:
        return ColOfHeader(header_str, first_col)


class ForcesEMGDevicesState(_DevicesState):
    _grouper: ForcePlateGrouper

    def __init__(self, finder: Optional[DevicesHeaderFinder],
                 grouper: Optional[ForcePlateGrouper]):
        super().__init__(finder)
        if grouper is None:
            self._grouper = self._instantiate_grouper
        self._grouper = grouper

    def _send_headers_to_aggregator(self, headers: List[ColOfHeader],
                                    reader: Reader):
        force_plates_headers, emg = self._separate_headers(headers)
        grouped_force_plates = self._group_force_plates(force_plates_headers)
        self._aggregate_emg(header, reader)
        for header in grouped_force_plates:
            self._add_force_plate(header, reader)
        self._add_emg(emg, reader)

    def _add_emg(self, header: ColOfHeader, reader: Reader):
        self._add_device(reader, header, DeviceType.EMG, self._last_col_of_emg)

    def _add_force_plate(self, header: ColOfHeader, reader: Reader):

        self._add_device(reader, header, DeviceType.FORCE_PLATE,
                         self._last_col_of_force_plate)

    def _separate_headers(self, headers: List[ColOfHeader]
                          ) -> Tuple[List[ColOfHeader], ColOfHeader]:
        force_plates_headers = headers[:-1]
        emg_header = headers[-1]
        return force_plates_headers, emg_header

    def _group_force_plates(self,
                            headers: List[ColOfHeader]) -> List[ColOfHeader]:
        return self._grouper.group(headers)

    def _instantiate_grouper(self):
        return ForcePlateGrouper()

    def _last_col(self, device_type: DeviceType,
                  first_col: int) -> Optional[int]:
        assert device_type is not DeviceType.TRAJECTORY_MARKER

        if device_type is DeviceType.EMG:
            return self._last_col_of_emg(first_col)
        if device_type is DeviceType.FORCE_PLATE:
            return self._last_col_of_force_plate(first_col)

    def _last_col_of_force_plate(self, first_col: int) -> int:
        return first_col + 9

    def _last_col_of_emg(self, first_col: int) -> None:
        return None


class TrajDevicesState(_DevicesState):
    def _send_headers_to_aggregator(self, headers: List[ColOfHeader],
                                    reader: Reader):
        for header in headers:
            self._add_device(reader, header, DeviceType.TRAJECTORY_MARKER)
            name = header.header_str
            first_col = header.col_index
            last_col = self._last_col_of_traj(first_col)
            self._add_device(reader, name, DeviceType.TRAJECTORY_MARKER,
                             first_col, last_col)

        def _last_col(self, _: DeviceType, first_col: int) -> int:
            return first_col + 3


class DeviceColsCreator(FailableMixin):
    _cols_class = DeviceHeaderCols

    def create_cols(self, headers: List[ColOfHeader]
                    ) -> FailableResult[List[DeviceHeaderCols]]:
        cols = []
        for col_of_header in headers:
            new_col = self._failable_create_device_header_cols(col_of_header)
            cols.append(new_col)

        return self._sequence_fail(cols)

    __call__ = create_cols

    def _failable_create_device_header_cols(
            self,
            col_of_header: ColOfHeader) -> FailableResult[DeviceHeaderCols]:
        device_name = col_of_header.header_str
        inferred_device_type = self._infer_device_type(device_name)
        cols_creator_method = partial(self._create_device_header_cols,
                                      col_of_header)
        return self._compute_on_failable(cols_creator_method,
                                         inferred_device_type,
                                         compose=True)

    def _create_device_header_cols(self,
                                   col_of_header: ColOfHeader,
                                   device_type=DeviceType) -> DeviceHeaderCols:
        return self._cols_class(device_type=device_type,
                                col_of_header=col_of_header)

    def _infer_device_type(self,
                           device_name: str) -> FailableResult[DeviceType]:
        lowercase_device_name = device_name.lower()

        if "force plate" in lowercase_device_name:
            return self._success(DeviceType.FORCE_PLATE)
        if "emg" in lowercase_device_name:
            return self._success(DeviceType.EMG)
        if "angelica" in lowercase_device_name:
            return self._success(DeviceType.TRAJECTORY_MARKER)

        error_message = (
            f'device name {device_name} not understood. '
            'Expected one of the following strings to occur in it ignoring'
            ' case: "force plate", "emg" or "angelica"')

        return self._fail(error_message)


class DeviceCategorizer(FailableMixin):
    # TODO Refactor - using DeviceType.section_type should make this a bit
    #  simpler
    def categorize(self, dev_cols: List[DeviceHeaderCols]
                   ) -> FailableResult[ViconNexusData]:
        grouped_headers = self._group_headers(dev_cols)
        fail_res = self._fail_if_section_is_inconsistent(grouped_headers)
        self._compute_on_failable(self._aggregate_categorized_headers,
                                  fail_res,
                                  compose=True)

    __call__ = categorize

    def _group_headers(self, dev_cols: List[DeviceHeaderCols]
                       ) -> Mapping[DeviceType, DeviceHeaderCols]:
        grouped = defaultdict(list)

        for dev in dev_cols:
            grouped[dev.device_type].append(dev)

        return grouped

    def _fail_if_section_is_inconsistent(
            self, grouped_headers: Mapping[DeviceType, DeviceHeaderCols]
    ) -> FailableResult[Mapping[DeviceType, DeviceHeaderCols]]:
        if not len(grouped_headers[DeviceType.EMG]) <= 1:
            error_message = (
                'expected only a single device header of type EMG, '
                f'but got both {emg_cols.device_name} and {col.device_name}')
            return self._fail(error_message)

        is_forces_emg_section = (bool(grouped_headers[DeviceType.FORCE_PLATE])
                                 or bool(grouped_headers[DeviceType.EMG]))

        is_trajs_section = bool(grouped_headers[DeviceType.TRAJECTORY_MARKER])

        if is_forces_emg_section and is_trajs_section:
            error_message = (
                'each section of the file is expected to have either only EMG '
                'and force plate devices or only trajectory ones. The row '
                'given mixes types from the 2 sections.')
            return self._fail(error_message)

        return self._success(grouped_headers)

    def _aggregate_categorized_headers(
            self, grouped_headers: Mapping[DeviceType, DeviceHeaderCols]
    ) -> ViconNexusData:
        emg_list = grouped_headers[DeviceType.EMG]
        try:
            emg = emg_list[0]
        except AttributeError:
            emg = None

        force_plates = grouped_headers[DeviceType.FORCE_PLATE]
        trajectory_markers = grouped_headers[DeviceType.TRAJECTORY_MARKER]

        return ViconNexusData(force_plates, emg, trajectory_markers)

    def _exclusive_or(a, b):
        return bool(a) != bool(b)


class ForcePlateGrouper(FailableMixin):
    def group_force_plates(self, dev_cols: List[DeviceHeaderCols]
                           ) -> ForcePlateDevices:
        # TODO if the method called below is able to group force plates
        # by their names, it should also be able to understand their types
        # so the only thing remaining after it is run would be to
        # 1. check that the 3 exact necessary headers are there
        # 2. aggregate the grouped representation
        grouped_force_plate_headers = self._group_force_plates_headers(
            dev_cols)

        fail_res = self._grouped_force_plates(grouped_force_plate_headers)
        return self._compute_on_failable(self._aggregate_grouped, fail_res)

    __call__ = group_force_plates

    def _group_force_plate_headers(self, dev_cols: List[DeviceHeaderCols]
                                   ) -> Mapping[str, List[DeviceHeaderCols]]:
        for dev in dev_cols:
            force_plate_name = 0

    def _check_grouped_force_plates(
            self, grouped: Mapping[str, DeviceHeaderCols]
    ) -> FailableResult[Mapping[str, DeviceHeaderCols]]:
        pass

    def _aggregate_grouped(self, grouped: Mapping[str, DeviceHeaderCols]
                           ) -> ForcePlateDevices:
        pass

    def _group_force_plates(self, cols_list: List['DeviceHeaderCols']
                            ) -> '_GroupingResult':
        def empty_force_plate_dict():
            return {'force': None, 'cop': None, 'moment': None}

        def aggregate_dict_up():
            pass

        plates_by_name = defaultdict(empty_force_plate_dict)

        for col in cols_list:
            force_plate_header = col.device_name

            try:
                first_part, second_part = force_plate_header.split('-')
            except ValueError:
                data_check = {
                    'is_valid':
                    False,
                    'error_message':
                    'expected force plates to obey format "name - var"'
                    ' where var is one of "force", "moment" or "CoP".'
                }
                return _GroupingResult(data_check=data_check,
                                       force_plate_cols=[])

            force_plate_name = first_part[:-1]
            measured_data = second_part[1:]


class DevicesState(_ReaderState):
    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.DEVICE_NAMES_LINE

    # Responsibilities:
    # 1. understanding from the device headers what is their type
    # 2. initialize DeviceCols with that
    # 3. initialize DeviceSectionAggregators (trivial)
    # 4. create the DeviceHeaders (trivial)
    #    TODO figure out if there is a design pattern for that
    #    it honestly seems to me that abstracting all of that crap out of
    #    this class is good
    # 5. create DataChanneler
    # 6. TODO who keeps track of DataChanneler? I believe the States, have to
    #    check my notes
    # 7. pass along to the next state an EMG device if there is one
    #    TODO decide if I split in 2 the next state
    #    leaning towards yes
    #    (though if this will be added to data aggregator, it doesn't need to)
    def feed_row(self, row: Row, reader: 'Reader'):
        row = self._preprocess_row(row)

    def _parse_and_process(self, row: Row, reader: 'Reader'):
        parse_result = self._parse_row(row)
        self._validate(reader, parse_result.data_check)
        grouping_result = self._group_force_plates(
            parse_result.force_plate_device_cols)
        self._validate(reader, grouping_result.data_check)

    def _create_device_header(self, device_name: str,
                              first_col_index: int) -> 'DeviceHeader':
        device_header_cols = self._create_device_header_cols(
            device_name, first_col_index)
        device_header_aggregator = self._create_aggregator()

    def _create_aggregator(self):
        return DeviceHeaderAggregator()


class CoordinatesState(_StepByStepReaderState):
    emg_cols: Optional[DeviceHeaderCols]

    @dataclass
    class _RowCols:
        row: Row
        emg_num_cols: Optional[int]

    def __init__(self, emg_cols: Optional[DeviceHeaderCols]):
        super().__init__()
        self.emg_cols = emg_cols

    @property
    def line(self) -> ViconCSVLines:
        # TODO all of the line methods could be refactored as class vars
        # in concrete classes
        return ViconCSVLines.COORDINATES_LINE

    def _check_row(self, row: Row) -> DataCheck:
        if self.emg_cols is not None and len(row) < self._emg_first_col():
            return self._create_data_check(
                False, 'row ends before the first EMG column')
        return self._create_valid_data_check()

    def _parse_row(self, row: Row) -> _RowCols:
        if self.emg_cols is not None:
            num_cols = len(row) - self._emg_first_col()
        else:
            num_cols = None

        return self._RowCols(row=row, emg_num_cols=num_cols)

    def _aggregate_data(self, parsed_data: _RowCols, aggregator: Aggregator):
        self._emg_add_num_cols_if_needed(parsed_data.emg_num_cols)
        self._aggregator_add_coordinates(aggregator, parsed_data.row)

    def _new_state(self) -> 'UnitsState':
        return UnitsState(
            units_line_parser=self._instantiate_units_line_parser())

    def _emg_first_col(self) -> int:
        return self.emg_cols.first_col_index

    def _emg_add_num_cols_if_needed(self, num_cols: Optional[int]):
        if self.emg_cols is not None:
            self.emg_cols.add_num_cols(num_cols)

    def _aggregator_add_coordinates(self, aggregator: Aggregator,
                                    coords_line: List[str]):
        aggregator.add_coordinates(coords)

    def _instantiate_units_line_parser(self) -> 'UnitsLineParser':
        return UnitsLineParser()


class UnitsState(_UpdateStateMixin, _AggregateDataMixin, _EntryByEntryMixin,
                 _ReaderState):
    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.UNITS_LINE

    ureg: pint.UnitRegistry

    def __init__(self, unit_registry=ureg):
        super().__init__()
        self.ureg = unit_registry

    def feed_row(self, row: Row, reader: Reader):
        row = self._preprocess_row(row)
        units = self._parse_row(row)
        self._aggregate_data(units, reader)
        self._update_state(reader)

    def _parse_entry(self, entry: str) -> pint.Unit:
        return self.ureg(entry)

    def _get_data_aggregate_method(self, aggregator: Aggregator
                                   ) -> Callable[[List[unit]], None]:
        return aggregator.add_units

    def _next_state_type(self):
        return GettingMeasurementsState


class GettingMeasurementsState(_AggregateDataMixin, _EntryByEntryMixin,
                               _PassUpFileEndedMixin, _ReaderState):
    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.DATA_LINE

    def feed_row(self, row: Row, reader: Reader):
        row = self._preprocess_row(row)

        if self._is_blank_line(row):
            self._transition(reader)
        else:
            floats = self._parse_row(row)
            self._aggregate_data(floats, reader)

    @staticmethod
    def _is_blank_line(row: Row) -> bool:
        return bool(row)

    def _parse_entry(row_entry: str) -> float:
        return float(row_entry)

    def _get_data_aggregate_method(self, aggregator: Aggregator
                                   ) -> Callable[[List[float]], None]:
        return aggregator.add_measurements

    def _transition(self, reader: Reader):
        # TODO maybe transitioning should be abstracted into a single class
        current_section_type = self._reader_section_type()

        if current_section_type is SectionType.FORCES_EMG:
            self._set_new_section_state(reader)
        elif current_section_type is SectionType.TRAJECTORIES:
            self._set_blank_state(reader)
        else:
            raise TypeError(
                "current section type isn't a member of SectionType")

        self._reader_transition_section(reader)

    def _reader_transition_section(self, reader: Reader):
        aggregator = self._reader_aggregator(reader)
        aggregator.transition()

    def _set_new_section_state(self, reader: Reader):
        new_state = SectionTypeState()
        self._reader_set_state(reader, new_state)

    def _set_blank_state(self, reader: Reader):
        new_state = BlankLinesState()
        self._reader_set_state(reader, new_state)


class BlankLinesState(_PassUpFileEndedMixin, _ReaderState):
    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.BLANK_LINE

    def feed_row(self, row: Row, reader: Reader):
        row = self._preprocess_row(row)
        if row:
            raise ValueError(
                f'BlankLinesState expects to only be fed empty rows but got this: {row}'
            )
