from typing import Optional, Sequence, Generic, Union, Callable, List

from .definitions import T, X, Y


class DataCheck:
    # A data check is a dict representing the result of validating the data of a
    # single line of the CSV data. Its keys:
    #   'is_valid' maps to a bool that answers
    #
    #   'error_message' maps to a str containing an error message to be
    #   displayed in case there are problems with the data. The str is appended to
    #   the prefix "error parsing line {line_number} of file {filename}: "
    def __init__(self, is_valid: bool, error_message: Optional[str] = None):
        if not is_valid and error_message is None:
            raise ValueError(
                'an invalid data check should contain an error message')

        if is_valid:
            error_message = None

        self.is_valid = is_valid
        self.error_message = error_message

    @classmethod
    def valid_data(cls):
        return cls(is_valid=True, error_message=None)

    def combine(self, other: 'DataCheck') -> 'DataCheck':
        if not self.is_valid:
            return self

        return other

    @classmethod
    def combine_multiple(cls,
                         data_checks: Sequence['DataCheck']) -> 'DataCheck':
        if not data_checks:
            return cls.valid_data()

        head = data_checks[0]
        tail = data_checks[1:]

        if not head.is_valid:
            return head

        return cls.combine_multiple(tail)


class Validator:
    current_line: int
    csv_filename: str
    should_raise: bool = True

    def __init__(self, csv_filename: str, should_raise: bool = True):
        self.current_line = 1
        self.csv_filename = csv_filename
        self.should_raise = should_raise

    def validate(self, data_check_result: DataCheck):
        if self.should_raise:
            self._raise_if_invalid(data_check_result)

        self.current_line += 1

    __call__ = validate

    def _raise_if_invalid(self, data_check_result: DataCheck):
        is_valid = data_check_result.is_valid
        error_message = data_check_result.error_message

        if not is_valid:
            raise ValueError(self._build_error_message(error_message))

    def _build_error_message(self, error_message: str) -> str:
        return f'error parsing line {self.current_line} of file {self.csv_filename}: {error_message}'


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


class FailableMixin:
    _failable_result_class = FailableResult

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

    def _fail_res_data_check(self, fail_res: FailableResult) -> DataCheck:
        return DataCheck
