import abc
from collections import defaultdict
import csv
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator, Generic, TypeVar, NewType, Union,
                    Iterable)

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
from .reader_data import (
    ViconNexusData,
    ColOfHeader,
    DeviceHeaderPair,
    DeviceHeaderCols,
    DeviceHeaderDataBuilder,
    ForcePlateDevices,
    DataBuilder,
    DataChanneler,
)


class Reader:
    """Reader for a single section of the CSV file outputted by Vicon Nexus.

    Initialize it
    """
    _state: '_ReaderState'
    _data_builder: DataBuilder

    def __init__(self, section_type_state: 'SectionTypeState',
                 data_builder: DataBuilder):
        self._state = section_type_state
        self._data_builder = data_builder

    def file_ended(self) -> ViconNexusData:
        self._state.file_ended(reader=self)

    def feed_row(self, row: Row):
        self._state.feed_row(row, reader=self)

    def set_state(self, new_state: '_ReaderState'):
        self._state = new_state

    def get_data_builder(self):
        return self._data_builder

    def get_section_type(self) -> SectionType:
        return self._data_builder.get_section_type()


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

    def _reader_data_builder(self, reader: 'Reader') -> DataBuilder:
        return reader.get_data_builder()

    def _reader_set_state(self, reader: 'Reader', new_state: '_ReaderState'):
        reader.set_state(new_state)

    def _reader_section_type(self, reader: Reader) -> SectionType:
        return reader.get_section_type()

    def _reader_set_new_section_type(self, reader: Reader,
                                     new_section_type: SectionType):
        reader.set_section_type(new_section_type)


class _UpdateStateMixin:
    def _update_state(self, reader: Reader):
        pass

    def _new_state(self):
        pass

    @abc.abstractproperty
    def _next_state_type(self):
        pass


class _HasSingleColMixin:
    def _validate_row_has_single_col(self, row: Row):
        if row[1:]:
            raise ValueError(
                f'first row of a section should contain nothing outside its first column'
            )


class _BuildDataMixin:
    @abc.abstractmethod
    def _get_data_build_method(self, data_builder: DataBuilder
                               ) -> Callable[[Any], None]:
        pass

    def _build_data(self, data: Any, reader: Reader):
        method = self._get_data_build_method(self._reader_data_builder(reader))
        method(data)


class _EntryByEntryMixin(Generic[T], abc.ABC):
    def _parse_row(self, row: Row) -> List[T]:
        return [self._parse_entry(row_entry) for row_entry in row]

    @abc.abstractmethod
    def _parse_entry(row_entry: str) -> T:
        pass


class _PassUpFileEndedMixin:
    def file_ended(self, reader: Reader) -> ViconNexusData:
        data_builder = self._reader_data_builder(reader)
        return data_builder.file_ended()


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


class SamplingFrequencyState(_BuildDataMixin, _UpdateStateMixin,
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

    def _get_data_build_method(self, data_builder: DataBuilder
                               ) -> Callable[[int], None]:
        return data_builder.add_frequency

    def feed_row(self, row: Row, reader: Reader):
        row = self._preprocess_row(row)
        self._validate_has_single_col(row)
        freq = self._parse_freq(row)
        self._build_data(freq, reader)
        self._update_state(reader)

    @staticmethod
    def _parse_freq(row: Row):
        return int(row[0])


class DevicesLineFinder(FailableMixin):
    def find_headers(self, row: Row) -> FailableResult[List[ColOfHeader]]:
        fail_res = self._check_row(row)
        return self._compute_on_failable(self._find_headers_unsafe,
                                         fail_res,
                                         compose=True)

    __call__ = find_headers

    def _find_headers_unsafe(self, row: Row) -> List[ColOfHeader]:
        return [self._col_of_header(row[i], i) for i in range(2, len(row), 3)]

    def _col_of_header(self, name, col) -> ColOfHeader:
        return ColOfHeader(col_index=col, header_str=name)

    def _check_row(self, row: Row) -> FailableResult[None]:
        error_message = ('this line should contain two blank columns '
                         'then one device name every 3 columns')

        if row[0] or row[1]:
            return self._fail(error_message)

        for col_num, col_val in enumerate(row[2:], start=2):
            if self._col_should_contain_name(col_num):
                current_is_correct = col_val
            else:
                current_is_correct = not col_val

            if not current_is_correct:
                return self._fail(error_message)

        return self._success(None)

    @staticmethod
    def _col_should_contain_name(col_num):
        return (col_num - 2) % 3 == 0


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
        self._compute_on_failable(self._build_categorized_headers,
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

    def _build_categorized_headers(
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
        # 2. build the grouped representation
        grouped_force_plate_headers = self._group_force_plates_headers(
            dev_cols)

        fail_res = self._grouped_force_plates(grouped_force_plate_headers)
        return self._compute_on_failable(self._build_grouped, fail_res)

    __call__ = group_force_plates

    def _group_force_plate_headers(self, dev_cols: List[DeviceHeaderCols]
                                   ) -> Mapping[str, List[DeviceHeaderCols]]:
        for dev in dev_cols:
            force_plate_name = 0

    def _check_grouped_force_plates(
            self, grouped: Mapping[str, DeviceHeaderCols]
    ) -> FailableResult[Mapping[str, DeviceHeaderCols]]:
        pass

    def _build_grouped(self, grouped: Mapping[str, DeviceHeaderCols]
                       ) -> ForcePlateDevices:
        pass

    def _group_force_plates(self, cols_list: List['DeviceHeaderCols']
                            ) -> '_GroupingResult':
        def empty_force_plate_dict():
            return {'force': None, 'cop': None, 'moment': None}

        def build_dict_up():
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
    # 3. initialize DeviceSectionDataBuilders (trivial)
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
    #    (though if this will be added to data builder, it doesn't need to)
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
        device_header_data_builder = self._create_data_builder()

    def _create_data_builder(self):
        return DeviceHeaderDataBuilder()


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

    def _build_data(self, parsed_data: _RowCols, data_builder: DataBuilder):
        self._emg_add_num_cols_if_needed(parsed_data.emg_num_cols)
        self._data_builder_add_coordinates(data_builder, parsed_data.row)

    def _new_state(self) -> 'UnitsState':
        return UnitsState(
            units_line_parser=self._instantiate_units_line_parser())

    def _emg_first_col(self) -> int:
        return self.emg_cols.first_col_index

    def _emg_add_num_cols_if_needed(self, num_cols: Optional[int]):
        if self.emg_cols is not None:
            self.emg_cols.add_num_cols(num_cols)

    def _data_builder_add_coordinates(self, data_builder: DataBuilder,
                                      coords_line: List[str]):
        data_builder.add_coordinates(coords)

    def _instantiate_units_line_parser(self) -> 'UnitsLineParser':
        return UnitsLineParser()


class UnitsState(_UpdateStateMixin, _BuildDataMixin, _EntryByEntryMixin,
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
        self._build_data(units, reader)
        self._update_state(reader)

    def _parse_entry(self, entry: str) -> pint.Unit:
        return self.ureg(entry)

    def _get_data_build_method(self, data_builder: DataBuilder
                               ) -> Callable[[List[unit]], None]:
        return data_builder.add_units

    def _next_state_type(self):
        return GettingMeasurementsState


class GettingMeasurementsState(_PassUpFileEndedMixin, _ReaderState):
    _data_line_parser: DataLineParser

    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.DATA_LINE

    def __init__(self, data_line_parser: DataLineParser):
        super().__init__()
        self._data_line_parser = data_line_parser

    def feed_row(self, row: Row, reader: Reader):
        row = self._preprocess_row(row)

        if self._is_blank_line(row):
            self._transition(reader)
        else:
            self._feed_forward(row, reader)

    @staticmethod
    def _is_blank_line(row: Row) -> bool:
        return bool(row)

    def _feed_forward(self, row: Row, reader: Reader):
        self._data_line_parser.feed_row(row, reader)

    def _transition(self, reader: Reader):
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
        data_builder = self._reader_data_builder(reader)
        data_builder.transition()

    def _set_new_section_state(self, reader: Reader):
        new_state = SectionTypeState()
        self._reader_set_state(reader, new_state)

    def _set_blank_state(self, reader: Reader):
        new_state = BlankLinesState()
        self._reader_set_state(reader, new_state)


class BlankLinesState(_PassUpFileEndedMixin, _StepByStepReaderState):
    @property
    def line(self) -> ViconCSVLines:
        return ViconCSVLines.BLANK_LINE

    def _check_row(self, row: Row) -> DataCheck:
        assert not bool(row)

    def _new_state(self) -> _ReaderState:
        return self

    def _do_nothing(self, *args, **kwargs) -> None:
        return

    _parse_row = _do_nothing
    _build_data = _do_nothing
