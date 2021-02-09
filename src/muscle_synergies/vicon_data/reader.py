import abc
import collections.abc
from collections import defaultdict
import csv
from dataclasses import dataclass
from enum import Enum
from functools import partial
import re
from typing import (List, Set, Dict, Tuple, Optional, Sequence, Callable, Any,
                    Mapping, Iterator, Generic, TypeVar, NewType, Union,
                    Iterable)

import pandas as pd
import pint
from pint_pandas import PintArray

from .reader_data import (
    CategorizedHeaders,
    ColOfHeader,
    SectionDataBuilder,
    DataCheck,
    DeviceHeaderPair,
    DeviceHeaderCols,
    DeviceHeaderDataBuilder,
    DeviceType,
    ForcePlateDevices,
    Row,
    SectionType,
    Union,
    Validator,
    ViconCSVLines,
    T,
    X,
    Y,
    DeviceHeaderRepresentation,
    Failable,
    FailableResult,
    ureg,
)



class DeviceHeader:
    def __init__(
            self,
            freqs,
    ):
        # TODO acho que assim:
        # 1. implemento a funcionalidade de ler frequências
        # 2. daí acho que está quase pronto aqui, falta terminar as 2 vias de
        #    inicialização
        pass

    @classmethod
    def from_device_header_pair(cls, device_header_pair: DeviceHeaderPair
                                ) -> 'DeviceHeader':
        builder = device_header_pair.device_data_builder
        device_name = device_header_pair.device_name

    @staticmethod
    def _extract_dataframe(device_header_builder: DeviceHeaderDataBuilder
                           ) -> pd.DataFrame:
        def create_pint_array(data, physical_unit):
            PintArray(data, dtype=physical_unit)

        data_dict = {}
        for time_series_builder in device_header_builder:
            coord_name = time_series_builder.get_coordinate_name()
            physical_unit = time_series_builder.get_physical_unit()
            data = time_series_builder.get_data()
            data_dict[coord_name] = create_pint_parray(data, physical_unit)

        return pd.DataFrame(data_dict)


class _ParsedDataRepresentation(collections.abc.Mapping):
    _data_frame_dict: Mapping[X, pd.DataFrame]

    def __init__(
            self,
            data_frame_dict: Mapping[X, pd.DataFrame],
    ):
        self._data_frame_dict = dict(data_frame_dict)

    # TODO Preciso terminar esse planejamento:
    # 1. implementar esse método
    # 2. o que vai ser devolvido pro user precisa ser o seguinte:
    #    - algo muito similar a um CategorizedHeaders
    #    - o membro force_plates pode literalmente ser ForcePlate
    #    - os 3 dev_headers de ForcePlate podem ser um
    #      _ParsedDataRepresentation (?)
    # 3. a classe _ParsedDataRepresentation precisa saber a respeito de
    #    frames e subframes.
    # 4. talvez ter um from_list_of_device_headers na ABC. Talvez nas
    #    filhas. Ou isso é responsabilidade do caller?
    # 5. acho que um merge_force_plates em algum lugar! Assim cada force_plate
    #    acaba sendo um único DataFrame. Vai precisar de uma classe especial
    #    MergedForcePlate pq tem membros relativos aos nomes de cada
    #    device_header. Esses nomes são úteis para debugging.
    # 6. daí faltam implementar:
    #    a. 2 métodos de DataBuilder
    #    b. mudar as classes do topo desse arquivo para o reader_data
    #    c. corrigir o Reader
    #    d. finalizar os 2 ReaderState pela metade
    #    e. começar os testes

    def __getitem__(self, ind: X) -> pd.DataFrame:
        return self._data_frame_dict.__getitem__(ind)

    def __len__(self) -> int:
        return len(self._data_frame_dict)

    def __iter__(self) -> Iterable[X]:
        yield from iter(self._data_frame_dict)


class TrajectoriesData(_ParsedDataRepresentation):
    pass


class ForcePlateData(_ParsedDataRepresentation):
    pass


class EMGData(_ParsedDataRepresentation):
    pass


class DataBuilder:
    _current_section_type: Optional[SectionType]

    _forces_emg_data_builder: SectionDataBuilder
    _trajs_data_builder: SectionDataBuilder

    def __init__(self,
                 forces_emg_data_builder: SectionDataBuilder,
                 trajs_data_builder: SectionDataBuilder,
                 initial_section_type=SectionType.FORCES_EMG):
        self._forces_emg_data_builder = forces_emg_data_builder
        self._trajs_data_builder = trajs_data_builder
        self._current_section_type = initial_section_type

    def get_section_type(self) -> SectionType:
        return self._current_section_type

    def get_section_data_builder(self) -> Optional[SectionDataBuilder]:
        if self.get_section_type() is SectionType.FORCES_EMG:
            return self._forces_emg_data_builder
        elif self.get_section_type() is SectionType.TRAJECTORIES:
            return self._trajs_data_builder
        return

    def transition(self):
        if self.get_section_type() is SectionType.FORCES_EMG:
            self._current_section_type = SectionType.TRAJECTORIES
        elif self.get_section_type() is SectionType.TRAJECTORIES:
            self._build_data()
        elif self.get_section_type() is None:
            raise TypeError(
                'tried to transition the state of a finished data builder')

    @property
    def finished(self):
        return self._current_section_type is None

    def get_built_data(self):
        pass

    def _build_data(self) -> 'DataRepresentation':
        pass


class Reader:
    """Reader for a single section of the CSV file outputted by Vicon Nexus.

    Initialize it
    """
    _state: '_ReaderState'
    _data_builder: SectionDataBuilder
    _validator: Validator
    _section_type: SectionType

    def __init__(self,
                 section_data_builder: SectionDataBuilder,
                 validator: Validator,
                 initial_section_type=SectionType.FORCES_EMG):
        self.state = ViconCSVLines.SECTION_TYPE_LINE
        self.data_builder = section_data_builder
        self.validator = validator
        self._section_type = initial_section_type

    def set_state(self, new_state: '_ReaderState'):
        self._state = new_state

    def get_data_builder(self):
        return self._data_builder

    def get_validator(self):
        return self._validator

    def get_section_type(self) -> SectionType:
        return self._section_type

    def set_section_type(self, new_section_type: SectionType):
        self._section_type = new_section_type

    def feed_row(self, row: Row):
        self._raise_if_ended()

        check_function = self._get_check_function()
        read_function = self._get_read_function()
        aggregation_function = self._get_build_function()

        self._call_validator(check_function(row))
        read_line = read_function(row)
        aggregation_function(read_line)

        self._transition()

    def _get_check_function(self):
        if self.state == ViconCSVLines.SECTION_TYPE_LINE:
            return check_section_type_line
        elif self.state == ViconCSVLines.SAMPLING_FREQUENCY_LINE:
            return check_sampling_frequency_line
        elif self.state == ViconCSVLines.DEVICE_NAMES_LINE:
            return self.devices_reader.check_device_names_line

    def _get_read_function(self):
        if self.state == ViconCSVLines.SECTION_TYPE_LINE:
            return read_section_type_line
        elif self.state == ViconCSVLines.SAMPLING_FREQUENCY_LINE:
            return read_sampling_frequency_line
        elif self.state == ViconCSVLines.DEVICE_NAMES_LINE:
            return self.devices_reader.read_device_names_line

    def _get_build_function(self):
        if self.state == ViconCSVLines.SECTION_TYPE_LINE:
            return self.data_builder.add_section_type
        elif self.state == ViconCSVLines.SAMPLING_FREQUENCY_LINE:
            return self.data_builder.add_frequency
        elif self.state == ViconCSVLines.DEVICE_NAMES_LINE:
            # TODO Essa é a próxima linha
            return

    def _transition(self):
        if self.state == ViconCSVLines.SECTION_TYPE_LINE:
            self.state = ViconCSVLines.SAMPLING_FREQUENCY_LINE
        elif self.state == ViconCSVLines.SAMPLING_FREQUENCY_LINE:
            self.state = ViconCSVLines.DEVICE_NAMES_LINE
        elif self.state == ViconCSVLines.DEVICE_NAMES_LINE:
            return

    def _raise_if_ended(self):
        if self.state == ViconCSVLines.BLANK_LINE:
            raise EOFError(
                'tried to read another line from a section that has already been completely readd'
            )

    def _call_validator(self, data_check_result: DataCheck):
        self.validator.validate(data_check_result)


class _ReaderState(abc.ABC):
    @abc.abstractmethod
    def feed_row(self, row: Row, reader: 'Reader'):
        # TODO document:
        # this should call validator exactly once.
        # it also has exactly 4 responsibilities apart from that one
        pass

    def _preprocess_row(self, row: Row) -> Row:
        row = list(entry.strip() for entry in row)

        while row and row[-1]:
            row.pop()
        return Row(row)

    def _validate(self, validator: Validator, check_result: DataCheck):
        validator.validate(check_result)

    def _reader_validator(self, reader: 'Reader') -> Validator:
        return reader.get_validator()

    def _reader_data_builder(self, reader: 'Reader') -> 'SectionDataBuilder':
        return reader.get_data_builder()

    def _reader_set_new_state(self, reader: 'Reader',
                              new_state: '_ReaderState'):
        reader.set_state(new_state)

    def _reader_section_type(self, reader: Reader) -> SectionType:
        return reader.get_section_type()

    def _reader_set_new_section_type(self, reader: Reader,
                                     new_section_type: SectionType):
        reader.set_section_type(new_section_type)

    def _create_data_check(self,
                           is_valid: bool,
                           error_message: Optional[str] = None):
        return DataCheck(is_valid=is_valid, error_message=error_message)

    def _create_valid_data_check(self):
        return DataCheck.valid_data()


class _StepByStepReaderState(_ReaderState, Generic[T]):
    """A reader step that fulfills each of its responsibilities in turn.

    Reader states have 4 responsibilities:
    1. validating the data
    2. parsing it
    3. building up a representation of the data being parsed
    4. transition the :py:class:Reader to its next state

    This abc is here to simplify the implementation of reader states which have
    these 4 responsibilities well-defined in happening in sucession,
    abstracting the boilerplate needed to make them achieved its child classes'
    goals.
    """
    def feed_row(self, row: Row, reader: 'Reader'):
        row = self._preprocess_row(row)

        # check and validate
        check_result = self._check_row(row)
        validator = self._reader_validator(reader)
        self._validate(validator, check_result)

        # parse and build up the data
        parsed_row = self._parse_row(row)
        self._build_data(parsed_row, self._reader_data_builder(reader))

        # transition to next state
        new_state = self._new_state()
        self._reader_set_new_state(reader, new_state)

    @abc.abstractmethod
    def _check_row(self, row: Row) -> DataCheck:
        pass

    @abc.abstractmethod
    def _parse_row(self, row: Row) -> T:
        pass

    @abc.abstractmethod
    def _build_data(self, parsed_data: T, data_builder: 'SectionDataBuilder'):
        pass

    @abc.abstractmethod
    def _new_state(self) -> '_ReaderState':
        pass


class SectionTypeState(_StepByStepReaderState):
    """The state of a reader that is expecting the section type line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for :py:class:ViconCSVLines.
    """
    def _check_row(self, row: Row) -> DataCheck:
        is_valid = row[0] in {'Devices', 'Trajectories'}
        message = (
            'this line should contain either "Devices" or "Trajectories"'
            ' in its first column')
        return self._create_data_check(is_valid, message)

    def _parse_row(self, row: Row) -> SectionType:
        section_type_str = row[0]

        if section_type_str == 'Devices':
            return SectionType.FORCES_EMG
        elif section_type_str == 'Trajectories':
            return SectionType.TRAJECTORIES

    def _build_data(self, parsed_data: SectionType,
                    data_builder: 'SectionDataBuilder'):
        data_builder.add_section_type(parsed_data)

    def _new_state(self):
        return SamplingFrequencyState()


class SamplingFrequencyState(_StepByStepReaderState):
    """The state of a reader that is expecting the sampling frequency line.

    For an explanation of what are the different lines of the CSV input, see
    the docs for :py:class:ViconCSVLines.
    """
    def _check_row(self, row: Row) -> DataCheck:
        try:
            is_valid = int(row[0])
        except ValueError:
            is_valid = False

        message = (
            'this line should contain an integer representing'
            f' sampling frequency in its first column and not {row[0]}.')

        return self._create_data_check(is_valid, message)

    def _parse_row(self, row: Row) -> int:
        try:
            return int(row[0])
        except ValueError:
            return None

    def _build_data(self, parsed_data: int,
                    data_builder: 'SectionDataBuilder'):
        data_builder.add_frequency(parsed_data)

    def _new_state(self) -> '_ReaderState':
        return DevicesState()


class DevicesLineFinder(Failable):
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


class DeviceColsCreator(Failable):
    def __init__(self,
                 *,
                 cols_class=DeviceHeaderCols,
                 failable_result_class=FailableResult):
        super().__init__(failable_result_class)
        self._cols_class = cols_class

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


class DeviceCategorizer(Failable):
    # TODO Refactor - using DeviceType.section_type should make this a bit
    #  simpler
    def categorize(self, dev_cols: List[DeviceHeaderCols]
                   ) -> FailableResult[CategorizedHeaders]:
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
    ) -> CategorizedHeaders:
        emg_list = grouped_headers[DeviceType.EMG]
        try:
            emg = emg_list[0]
        except AttributeError:
            emg = None

        force_plates = grouped_headers[DeviceType.FORCE_PLATE]
        trajectory_markers = grouped_headers[DeviceType.TRAJECTORY_MARKER]

        return CategorizedHeaders(force_plates, emg, trajectory_markers)

    def _exclusive_or(a, b):
        return bool(a) != bool(b)


class ForcePlateGrouper(Failable):
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
    def feed_row(self, row: Row, reader: 'Reader'):
        row = self._preprocess_row(row)

    def _parse_and_process(self, row: Row, reader: 'Reader'):
        parse_result = self._parse_row(row)
        self._validate(reader, parse_result.data_check)
        grouping_result = self._group_force_plates(
            parse_result.force_plate_device_cols)
        self._validate(reader, grouping_result.data_check)


class CoordinatesState(_StepByStepReaderState):
    def _create_device_header(self, device_name: str,
                              first_col_index: int) -> 'DeviceHeader':
        device_header_cols = self._create_device_header_cols(
            device_name, first_col_index)
        device_header_data_builder = self._create_data_builder()

    def _create_data_builder(self):
        return DeviceHeaderDataBuilder()


class _EntryByEntryParser(_ReaderState, Failable, Generic[T]):
    """A parser which parses each row entry independently."""
    def feed_row(self, row: Row, reader: Reader):
        # 1 and 2. parse and check
        row = self._preprocess_row(row)
        fail_res = self._parse_row(row)
        self._validate_fail_res(fail_res, reader)

        # 3. build data representation
        parse_result = self._fail_res_parse_result(fail_res)
        self._build_data(parse_result)

    def _parse_row(self, row: Row) -> FailableResult[List[T]]:
        frame_cols, data_cols = row[:2], row[2:]
        parsed_cols = [
            self._parse_entry(data_entry) for data_entry in data_cols
        ]
        return self._compute_on_failable(self._prepend_two_none_cols,
                                         parsed_cols,
                                         compose=True)

    def _prepend_two_none_cols(self, cols: List[T]) -> List[T]:
        return [None, None] + cols

    def _validate_fail_res(self, fail_res: FailableResult, reader: Reader):
        self._validate(self._reader_validator(reader),
                       self._fail_res_data_check(fail_res))

    def _build_data(self, parse_result: List[T], reader: Reader):
        data_builder = self._reader_data_builder(reader)
        build_method = self._get_build_method(data_builder)
        build_method(parse_result)

    @abc.abstractmethod
    def _parse_entry(self, entry: str) -> FailableResult[T]:
        pass

    @abc.abstractmethod
    def _get_build_method(self, data_builder: SectionDataBuilder
                          ) -> Callable[[T], Any]:
        pass


class UnitsLineParser(_EntryByEntryParser):
    def _parse_entry(self, entry: str) -> FailableResult[pint.Unit]:
        try:
            unit = ureg(entry)
        except pint.UndefinedUnitError:
            return self._fail(f'unit "{entry}" not understood')
        else:
            return self._success(unit)

    def _get_build_method(self, data_builder: SectionDataBuilder
                          ) -> Callable[[T], Any]:
        return data_builder.add_units


class UnitsState(_ReaderState):
    _units_line_parser: UnitsLineParser

    def __init__(self, units_line_parser: UnitsLineParser):
        self._units_line_parser

    def feed_row(self, row: Row, reader: Reader):
        row = self._preprocess_row(row)
        self._feed_forward(row, reader)
        new_state = self._new_state()
        self._reader_set_new_state(new_state)

    def _feed_forward(self, row: Row, reader: Reader):
        self._units_line_parser.feed_row(row, reader)

    def _new_state(self) -> _ReaderState:
        return GettingMeasurementsState(
            failable_result_class=self._failable_result_class)


class DataLineParser(_EntryByEntryParser):
    def _parse_entry(self, entry: str) -> FailableResult[float]:
        try:
            data = float(entry)
        except ValueError:
            return self._fail(
                f'real-valued measurement "{entry}" not understood')
        else:
            return self._success(data)

    def _get_build_method(self, data_builder: SectionDataBuilder
                          ) -> Callable[[T], Any]:
        return data_builder.add_measurements


class GettingMeasurementsState(_ReaderState):
    _data_line_parser: DataLineParser

    def __init__(self, data_line_parser: DataLineParser):
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
            self._new_section_state(reader)
        elif current_section_type is SectionType.TRAJECTORIES:
            self._blank_state(reader)
        else:
            raise TypeError(
                "current section type isn't a member of SectionType")

        self._reader_transition_section(reader)

    def _reader_transition_section(self, reader: Reader):
        reader.transition_section()

    def _new_section_state(self, reader: Reader):
        new_state = SectionTypeState()
        self._reader_set_new_state(reader, new_state)

    def _blank_state(self, reader: Reader):
        new_state = BlankLinesState()
        self._reader_set_new_state(reader, new_state)


class BlankLinesState(_StepByStepReaderState):
    def _check_row(self, row: Row) -> DataCheck:
        assert not bool(row)

    def _new_state(self) -> _ReaderState:
        return self

    def _do_nothing(self, *args, **kwargs) -> None:
        return

    _parse_row = _do_nothing
    _build_data = _do_nothing
