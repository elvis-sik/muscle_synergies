from functools import partial

from .vicon_data import *


class Reader:
    """Reader for a single section of the CSV file outputted by Vicon Nexus.

    Initialize it
    """
    _state: _ReaderState
    _data_builder: 'DataBuilder'
    _validator: Validator

    def __init__(self, section_data_builder: 'DataBuilder',
                 validator: Validator):
        self.state = ViconCSVLines.SECTION_TYPE_LINE
        self.data_builder = section_data_builder
        self.validator = validator

    def set_state(self, new_state: _ReaderState):
        self._state = new_state

    def get_data_builder(self):
        return self._data_builder

    def get_validator(self):
        return self._validator

    def feed_line(self, row: Row):
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
        # it should also change the state of the reader.
        pass

    def _preprocess_row(self, row: Row) -> Row:
        row = list(row)
        while row and row[-1]:
            row.pop()
        return Row(row)

    def _validate(self, validator: Validator, check_result: DataCheck):
        validator.validate(check_result)

    def _reader_validator(self, reader: 'Reader') -> Validator:
        return reader.get_validator()

    def _reader_data_builder(self, reader: 'Reader') -> 'DataBuilder':
        return reader.get_data_builder()

    def _reader_set_new_state(self, reader: 'Reader',
                              new_state: '_ReaderState'):
        reader.set_state(new_state)

    def _create_data_check(self,
                           is_valid: bool,
                           error_message: Optional[str] = None):
        return DataCheck(is_valid=is_valid, error_message=error_message)

    def _create_valid_data_check(self):
        return DataCheck.valid_data()


class _EasyReaderState(_ReaderState, Generic[T]):
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
    def _build_data(self, parsed_data: T, data_builder: 'DataBuilder'):
        pass

    @abc.abstractmethod
    def _new_state(self) -> '_ReaderState':
        pass


class SectionTypeState(_EasyReaderState):
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
                    data_builder: 'DataBuilder'):
        data_builder.add_section_type(parsed_data)

    def _new_state(self):
        return SamplingFrequencyState()


class SamplingFrequencyState(_EasyReaderState):
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

    def _build_data(self, parsed_data: int, data_builder: 'DataBuilder'):
        data_builder.add_frequency(parsed_data)

    def _new_state(self) -> '_ReaderState':
        return DevicesState()


class FailableResult(Generic[T]):
    """Data class holding the result of a parsing process that can fail.

    FailableResult holds the data parsed from a process. If the process failed,
    it holds the :py:class:DataCheck with details on the failure instead.

    In case the process didn't fail, the data_check member of an instance will
    hold a valid :py:class:DataCheck.

    Args:
        parse_result: the data parsed.

        data_check: a check on the validity of the data.

    Raises:
        ValueError: if `data_check` is provided and its `is_valid` member is
            `False` but a `parse_result` was also provided. Also if a
            `data_check` with a `True` `is_valid` member is provided but no
            `parse_result` is provided. Also if none of the 2 arguments was
            provided.
    """

    parse_result: Optional[T]
    data_check: DataCheck

    def __init__(self,
                 *,
                 parse_result: Optional[T] = None,
                 data_check: Optional[DataCheck] = None):
        if parse_result is not None:
            self.data_check = self._process_data_check_result_provided(
                data_check)
            self.parse_result = parse_result
        else:
            self.parse_result = None
            self.data_check = self._process_data_check_without_result(
                data_check)

    def _process_data_check_result_provided(self,
                                            data_check: Optional[DataCheck]
                                            ) -> DataCheck:
        if data_check is None:
            return self._valid_data_check()

        if not data_check.is_valid:
            raise ValueError(
                'invalid data check and a parse result were provided')

        return data_check

    def _process_data_check_without_result(self,
                                           data_check: Optional[DataCheck]
                                           ) -> DataCheck:
        if data_check is None:
            # parse_result is assumed to be None
            return self._valid_data_check()

        if data_check.is_valid:
            raise ValueError('valid data check provided without parse result')

        return data_check

    @staticmethod
    def _valid_data_check():
        return DataCheck.valid_data()

    @classmethod
    def create_failed(cls, error_message: str) -> 'FailableResult':
        data_check = DataCheck(is_valid=False, error_message=error_message)
        return cls(data_check=data_check)

    @classmethod
    def create_successful(cls, parse_result: T) -> 'FailableResult[T]':
        return cls(parse_result=parse_result)

    @property
    def failed(self):
        return not self.data_check.is_valid

    def __eq__(self, other: 'FailableResult[T]') -> bool:
        return (self.failed == other.failed
                and self.parse_result == other.parse_result)


class _DevicesLineStateStep:
    def __init__(self, failable_result_class=FailableResult):
        self._failable_result_class = failable_result_class

    def _compute_on_failable(
            self,
            func: Union[Callable[[X], FailableResult[Y]], Callable[[X], Y]],
            arg: FailableResult[X],
            compose: bool = False) -> Union[FailableResult[Y], Y]:
        if self._fail_res_failed(arg):
            return arg

        res = func(arg)
        if not compose:
            return res

        return self._success(res)

    def _sequence_fail(self, failable_results: Sequence[FailableResult[T]]
                       ) -> FailableResult[List[T]]:
        if len(failable_results) == 0:
            return self._fail('called _sequence_fail on empty list')

        parsed_values = []

        for fail_res in failable_results:
            self._compute_on_failable(parsed_values.append,
                                      fail_res,
                                      compose=False)

        return self._success(parsed_values)

    def _fail(self, error_message: str) -> FailableResult[ColOfHeader]:
        return self._failable_result_class.create_failed(error_message)

    def _success(self, result: T) -> FailableResult[T]:
        return self._failable_result_class.create_successful(result)

    def _fail_res_failed(self, fail_res: FailableResult) -> bool:
        return fail_res.failed

    def _fail_res_parse_result(self, fail_res: FailableResult[T]) -> T:
        return fail_res.parse_result


class DevicesLineFinder(_DevicesLineStateStep):
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


class DeviceColsCreator(_DevicesLineStateStep):
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


class DeviceCategorizer(_DevicesLineStateStep):
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


class ForcePlateGrouper(_DevicesLineStateStep):
    def group_force_plates(self,
                           dev_cols: List[DeviceHeaderCols]) -> ForcePlate:
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

    def _build_grouped(self,
                       grouped: Mapping[str, DeviceHeaderCols]) -> ForcePlate:
        pass

    def _group_force_plates(self, cols_list: List['DeviceHeaderCols']
                            ) -> _GroupingResult:
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
    # 3. initialize DeviceDataBuilders (trivial)
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
        pass

    def _parse_and_process(self, row: Row, reader: 'Reader'):
        parse_result = self._parse_row(row)
        self._validate(reader, parse_result.data_check)
        grouping_result = self._group_force_plates(
            parse_result.force_plate_device_cols)
        self._validate(reader, grouping_result.data_check)


class CoordinatesState(_EasyReaderState):
    def _create_device_header(self, device_name: str,
                              first_col_index: int) -> 'DeviceHeader':
        device_header_cols = self._create_device_header_cols(
            device_name, first_col_index)
        device_header_data_builder = self._create_data_builder()

    def _create_data_builder(self):
        return DeviceHeaderDataBuilder()
