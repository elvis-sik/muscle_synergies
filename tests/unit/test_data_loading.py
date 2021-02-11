import random
from typing import List

import pytest as pt

import muscle_synergies.vicon_data as vd


def test_dev_type_section_type():
    assert vd.DeviceType.FORCE_PLATE.section_type(
    ) is vd.SectionType.FORCES_EMG
    assert vd.DeviceType.EMG.section_type() is vd.SectionType.FORCES_EMG
    assert (vd.DeviceType.TRAJECTORY_MARKER.section_type() is
            vd.SectionType.TRAJECTORIES)


@pt.fixture
def valid_check():
    return vd.DataCheck.valid_data()


@pt.fixture
def invalid_check():
    return vd.DataCheck(False, '')


class TestDataCheck:
    def test_cant_create_with_error(self):
        with pt.raises(ValueError):
            vd.DataCheck(is_valid=False, error_message=None)

    @pt.fixture
    def another_valid_check(self):
        return vd.DataCheck.valid_data()

    def test_combine_2_valids_gives_valid(self, valid_check,
                                          another_valid_check):
        combined = valid_check.combine(another_valid_check)
        assert combined.is_valid
        assert combined.error_message is None

    @pt.fixture
    def first_invalid(self):
        return vd.DataCheck(is_valid=False, error_message='first invalid')

    def test_combine_invalid_with_valid_gives_invalid(self, first_invalid,
                                                      valid_check):
        combined = first_invalid.combine(valid_check)
        assert not combined.is_valid
        assert combined.error_message == 'first invalid'

    def test_combine_valid_with_invalid_gives_invalid(self, first_invalid,
                                                      valid_check):
        combined = valid_check.combine(first_invalid)
        assert not combined.is_valid
        assert combined.error_message == 'first invalid'

    @pt.fixture
    def second_invalid(self):
        return vd.DataCheck(is_valid=False, error_message='second invalid')

    def test_combine_invalids_give_invalid(self, first_invalid,
                                           second_invalid):
        combined = first_invalid.combine(second_invalid)
        assert not combined.is_valid

    def test_combine_preserves_order(self, first_invalid, second_invalid):
        combined = first_invalid.combine(second_invalid)
        assert combined.error_message == 'first invalid'

    third_valid = another_valid_check

    def test_combine_multiple_valid(self, valid_check, another_valid_check,
                                    third_valid):
        checks = (valid_check, another_valid_check, third_valid)
        combined = vd.DataCheck.combine_multiple(checks)
        assert combined.is_valid

    def test_combine_multiple_invalid(self, valid_check, another_valid_check,
                                      first_invalid):
        checks = (valid_check, another_valid_check, first_invalid)
        combined = vd.DataCheck.combine_multiple(checks)
        assert not combined.is_valid

    def test_combine_multiple_order(self, second_invalid, first_invalid):
        checks = (second_invalid, first_invalid)
        combined = vd.DataCheck.combine_multiple(checks)
        assert combined.error_message == 'second invalid'

    def test_combine_multiple_empty(self):
        checks = []
        combined = vd.DataCheck.combine_multiple(checks)
        assert combined.is_valid


class TestFailableResult:
    def test_initialize_with_result_ok(self, valid_check):
        # dummy check
        assert vd.FailableResult(parse_result=3) is not None
        assert vd.FailableResult(parse_result=3,
                                 data_check=valid_check) is not None

    def test_initialize_with_result_and_invalid_check_raises(
            self, invalid_check):
        with pt.raises(ValueError):
            vd.FailableResult(parse_result=3, data_check=invalid_check)

    @pt.fixture
    def failed(self, invalid_check):
        return vd.FailableResult(data_check=invalid_check)

    def test_initialize_with_invalid_check_ok(self, failed):
        assert failed.parse_result is None

    def test_auto_adds_valid_check(self):
        result = vd.FailableResult(parse_result=3)
        assert not result.failed

    def test_no_data_check_no_result_ok(self):
        result = vd.FailableResult()
        assert not result.failed
        assert result.parse_result == None

    @pt.fixture
    def result_1(self):
        return vd.FailableResult(parse_result=1)

    @pt.fixture
    def result_2(self):
        return vd.FailableResult(parse_result=2)


class TestValidator:
    @pt.fixture
    def validator_that_should_raise(self):
        return vd.Validator(csv_filename='not important.csv',
                            should_raise=True)

    @pt.fixture
    def validator_that_doesnt_raise(self):
        return vd.Validator(csv_filename='not important.csv',
                            should_raise=False)

    def test_validator_starts_at_1(self, validator_that_should_raise):
        validator = validator_that_should_raise
        assert validator.current_line == 1

    def test_validator_increases_line_count(self, validator_that_should_raise,
                                            valid_check):
        validator = validator_that_should_raise
        data_check = valid_check
        validator.validate(data_check)
        assert validator.current_line == 2

    def test_validator_that_shouldnt_doesnt_raise(self,
                                                  validator_that_doesnt_raise,
                                                  invalid_check):
        validator = validator_that_doesnt_raise
        data_check = invalid_check
        assert validator(data_check) is None

    def test_validator_raises_with_invalid(self, validator_that_should_raise,
                                           invalid_check):
        validator = validator_that_should_raise
        data_check = invalid_check

        with pt.raises(ValueError):
            validator(data_check)

    def test_validator_doesnt_raise_with_valid(self,
                                               validator_that_should_raise,
                                               valid_check):
        validator = validator_that_should_raise
        data_check = valid_check
        assert validator(data_check) is None


class TestTimeSeriesDataBuilder:
    @pt.fixture
    def data_builder(self):
        return vd.TimeSeriesDataBuilder()

    def test_add_coordinate_name(self, data_builder):
        coord_name = 'foo'
        data_builder.add_coordinate(coord_name=coord_name)
        assert coord_name == data_builder.coordinate_name

    def test_add_unit(self, data_builder):
        unit = 'foo'
        data_builder.add_unit(physical_unit=unit)
        assert unit == data_builder.physical_unit

    def test_add_data(self, data_builder):
        first_entry = 1.
        data_builder.add_data(data_entry=first_entry)
        assert [first_entry] == data_builder.data

        second_entry = 2.
        data_builder.add_data(data_entry=second_entry)
        assert [first_entry, second_entry] == data_builder.data


class TestDeviceHeaderDataBuilder:
    @pt.fixture
    def mock_time_series(self, mocker):
        return mocker.Mock(autospec=vd.TimeSeriesDataBuilder)

    mock_another_time_series = mock_time_series

    @pt.fixture
    def data_builder(self, mock_time_series, mock_another_time_series):
        return vd.DeviceHeaderDataBuilder(
            time_series_list=[mock_time_series, mock_another_time_series])

    def test_add_coordinates(self, data_builder, mock_time_series,
                             mock_another_time_series):
        first_data = 'first'
        second_data = 'second'

        parsed_data = [first_data, second_data]
        data_builder.add_coordinates(parsed_data)

        mock_time_series.add_coordinate.assert_called_once_with(first_data)
        mock_another_time_series.add_coordinate.assert_called_once_with(
            second_data)

    def test_add_coordinates_wrong_number(self, data_builder):
        parsed_data = ['one', 'two', 'three']

        with pt.raises(ValueError):
            data_builder.add_coordinates(parsed_data)

    def test_add_units(self, data_builder, mock_time_series,
                       mock_another_time_series):
        first_data = 'first'
        second_data = 'second'

        parsed_data = [first_data, second_data]
        data_builder.add_units(parsed_data)

        mock_time_series.add_unit.assert_called_once_with(first_data)
        mock_another_time_series.add_unit.assert_called_once_with(second_data)

    def test_add_units_wrong_number(self, data_builder):
        parsed_data = ['one', 'two', 'three']

        with pt.raises(ValueError):
            data_builder.add_units(parsed_data)

    def test_add_data(self, data_builder, mock_time_series,
                      mock_another_time_series):
        first_data = 'first'
        second_data = 'second'

        parsed_data = [first_data, second_data]
        data_builder.add_data(parsed_data)

        mock_time_series.add_data.assert_called_once_with(first_data)
        mock_another_time_series.add_data.assert_called_once_with(second_data)

    def test_add_data_wrong_number(self, data_builder):
        parsed_data = ['one', 'two', 'three']

        with pt.raises(ValueError):
            data_builder.add_data(parsed_data)


class TestDeviceHeaderCols:
    @pt.fixture
    def emg_dev_cols(self):
        col_of_header = vd.ColOfHeader(3, 'emg')
        return vd.DeviceHeaderCols(device_type=vd.DeviceType.EMG,
                                   col_of_header=col_of_header)

    @pt.fixture
    def force_plate_dev_cols(self):
        col_of_header = vd.ColOfHeader(0, 'force plate')
        return vd.DeviceHeaderCols(device_type=vd.DeviceType.FORCE_PLATE,
                                   col_of_header=col_of_header)

    def test_initialization_non_emg(self, force_plate_dev_cols):
        assert force_plate_dev_cols.num_of_cols == 3

    def test_initialization_emg(self, emg_dev_cols):
        assert emg_dev_cols.num_of_cols is None

    def test_add_num_cols(self, emg_dev_cols):
        emg_dev_cols.add_num_cols(4)
        assert emg_dev_cols.num_of_cols == 4

    def test_add_num_cols_twice_raises(self, emg_dev_cols):
        emg_dev_cols.add_num_cols(4)

        with pt.raises(TypeError):
            emg_dev_cols.add_num_cols(4)

    def test_add_num_cols_non_emg_raises(self, force_plate_dev_cols):
        with pt.raises(TypeError):
            force_plate_dev_cols.add_num_cols(4)

    def test_create_slice_no_num_cols_raises(self, emg_dev_cols):
        with pt.raises(TypeError):
            emg_dev_cols.create_slice()

    def test_create_slice(self, force_plate_dev_cols):
        created_slice = force_plate_dev_cols.create_slice()
        expected_slice = slice(0, 3)
        assert created_slice == expected_slice


class TestDataChanneler:
    @pt.fixture
    def mock_data_builder(self, mocker):
        return mocker.Mock(autospec=vd.DeviceHeaderDataBuilder)

    mock_another_data_builder = mock_data_builder

    @pt.fixture
    def mock_cols_0_1(self, mocker):
        device_cols = mocker.Mock()
        device_cols.create_slice = mocker.Mock(return_value=slice(0, 2))
        return device_cols

    @pt.fixture
    def mock_cols_2_3(self, mocker):
        device_cols = mocker.Mock()
        device_cols.create_slice = mocker.Mock(return_value=slice(2, 4))
        return device_cols

    @pt.fixture
    def mock_device_0_1(self, mock_data_builder, mock_cols_0_1):
        return vd.DeviceHeader(device_cols=mock_cols_0_1,
                               device_data_builder=mock_data_builder)

    @pt.fixture
    def mock_device_2_3(self, mock_another_data_builder, mock_cols_2_3):
        return vd.DeviceHeader(device_cols=mock_cols_2_3,
                               device_data_builder=mock_another_data_builder)

    @pt.fixture
    def data_channeler(self, mock_device_0_1, mock_device_2_3):
        devices = [mock_device_0_1, mock_device_2_3]
        return vd.DataChanneler(devices)

    @pt.fixture
    def row(self):
        return ['first', 'second', 'third', 'fourth']

    def test_add_coordinates(self, data_channeler, row, mock_data_builder,
                             mock_another_data_builder):
        data_channeler.add_coordinates(row)
        mock_data_builder.add_coordinates.assert_called_once_with(
            ['first', 'second'])
        mock_another_data_builder.add_coordinates.assert_called_once_with(
            ['third', 'fourth'])

    def test_add_units(self, data_channeler, row, mock_data_builder,
                       mock_another_data_builder):
        data_channeler.add_units(row)
        mock_data_builder.add_units.assert_called_once_with(
            ['first', 'second'])
        mock_another_data_builder.add_units.assert_called_once_with(
            ['third', 'fourth'])

    def test_add_data(self, data_channeler, row, mock_data_builder,
                      mock_another_data_builder):
        data_channeler.add_data(row)
        mock_data_builder.add_data.assert_called_once_with(['first', 'second'])
        mock_another_data_builder.add_data.assert_called_once_with(
            ['third', 'fourth'])


class TestSectionDataBuilder:
    pass


class TestReader:
    # NEXT create tests for these behaviors:

    # 0. change name to Reader since it is not constrained for a single section
    #    anymore

    # 1. when it is fed a row, it simply calls .feed_row(row, itself) on its
    #    current state (which then becomes responsible for all of the following:
    #    1. checking and calling Validator
    #    2. passing parsed data to data builder (which it gets from the Reader)
    #    3. deciding whether the state should change and telling the Reader if it
    #       should
    #    4. the blank state will basically notify builder to change its own
    #       state
    #    5. when Builder's first state creates the second one, it passes itself
    #       as an argument.
    #    6. when the second "change state" message comes,
    pass


@pt.mark.skip
class TestSectionReader:
    @staticmethod
    def patch_section_reader_method(mocker, method_name):
        full_method_name = f'muscle_synergies.vicon_data.SectionReader.{method_name}'
        mocker.patch(full_method_name)

    @pt.fixture
    def patch_get_check_function(self, mocker):
        mocked_method = '_get_check_function'
        self.patch_section_reader_method(mocker, mocked_method)

    @pt.fixture
    def patch_get_read_function(self, mocker):
        mocked_method = '_get_read_function'
        self.patch_section_reader_method(mocker, mocked_method)

    @pt.fixture
    def patch_get_build_function(self, mocker):
        mocked_method = '_get_build_function'
        self.patch_section_reader_method(mocker, mocked_method)

    @pt.fixture
    def section_reader_with_patched_getters(self, mocker,
                                            patch_get_check_function,
                                            patch_get_read_function,
                                            patch_get_build_function):
        return vd.SectionReader(
            section_data_builder=mocker.Mock(name='builder'),
            validator=mocker.Mock(name='validator'))

    @pt.fixture
    def row_mock(self, mocker):
        return mocker.Mock(name='row')

    def test_feed_line_calls_getters(self, section_reader_with_patched_getters,
                                     row_mock):
        section_reader = section_reader_with_patched_getters
        section_reader.feed_line(row_mock)


class TestViconDataLoader:
    pass


class TestViconNexusCSVReader:
    # TODO this class is obsolete
    @pt.fixture
    def section_reader_mock(self, mocker):
        mock = mocker.Mock(name='SectionReader')

    another_section_reader_mock = section_reader_mock

    @pt.fixture
    def section_reader_that_raises_mock(self, mocker):
        times_called = 0

        def mock_side_effect():
            nonlocal times_called
            times_called += 1

            if times_called == 2:
                raise EOFError

        mock = mocker.Mock(name='SectionReader that raises',
                           side_effect=mock_side_effect)

    @pt.fixture
    def vicon_reader(self, section_reader_mock, another_section_reader_mock):
        return vd.ViconNexusCSVReader(
            forces_emg_reader=section_reader_mock,
            trajectories_reader=another_section_reader_mock)

    @pt.fixture
    def vicon_reader_that_faces_eof(self, section_reader_mock,
                                    section_reader_that_raises_mock):
        return vd.ViconNexusCSVReader(
            fores_emg_reader=section_reader_that_raises_mock,
            trajectories_reader=section_reader_mock)


class TestState:
    @pt.fixture
    def mock_validator(self, mocker):
        validator = mocker.Mock(autospec=vd.Validator)

        def record_call(data_check_result):
            nonlocal validator
            validator.mock_validate_call = data_check_result

        # The Validator mock uses a side effect
        # because otherwise it is difficult to check whether .validate
        # was called with the proper value for the 'is_valid' key.
        validator.validate.side_effect = record_call
        return validator

    @pt.fixture
    def mock_data_builder(self, mocker):
        return mocker.Mock(autospec=vd.DataBuilder)

    @pt.fixture
    def mock_reader(self, mocker, mock_validator, mock_data_builder):
        reader = mocker.Mock(name='reader')
        reader.get_validator = mocker.Mock(name='get_validator',
                                           return_value=mock_validator)
        reader.get_data_builder = mocker.Mock(name='get_data_builder',
                                              return_value=mock_data_builder)
        reader.set_state = mocker.Mock(name='set_state')
        return reader

    class TestSectionTypeState:
        @pt.fixture
        def patch_next_state_init(self, mocker, monkeypatch):
            mock_init = mocker.Mock(return_value=None)
            monkeypatch.setattr(vd.SamplingFrequencyState, '__init__',
                                mock_init)
            return mock_init

        @pt.fixture
        def state(self, patch_next_state_init):
            return vd.SectionTypeState()

        @pt.fixture(params=[('Devices', vd.SectionType.FORCES_EMG),
                            ('Trajectories', vd.SectionType.TRAJECTORIES)])
        def row_and_expected_output(self, request):
            return ([request.param[0], '', '', '', '', '', '',
                     ''], request.param[1])

        def test_accepts_valid(self, state, row_and_expected_output,
                               mock_reader, mock_validator):
            row = row_and_expected_output[0]
            state.feed_row(row, mock_reader)
            mock_validator.validate.assert_called_once()
            assert mock_validator.mock_validate_call.is_valid

        @pt.fixture
        def invalid_row(self):
            return ['Invalid', '', '', '', '', '']

        def test_doesnt_accept_invalid(self, state, invalid_row, mock_reader,
                                       mock_validator):
            state.feed_row(invalid_row, mock_reader)
            mock_validator.validate.assert_called_once()
            assert not mock_validator.mock_validate_call.is_valid

        def test_creates_new_state(self, state, row_and_expected_output,
                                   mock_reader, patch_next_state_init):
            row = row_and_expected_output[0]
            state.feed_row(row, mock_reader)
            patch_next_state_init.assert_called_once_with()

        def test_builds_data_line(self, state, row_and_expected_output,
                                  mock_reader, mock_data_builder):
            row, expected_output = row_and_expected_output
            state.feed_row(row, mock_reader)
            mock_data_builder.add_section_type.assert_called_once_with(
                expected_output)

    class TestSamplingFrequencyState:
        # TODO this whole class is literally a copy-paste of
        # TestSectionTypeState with just small bits changed.
        # pytest is powerful enough that this redundancy could likely be
        # avoided.
        @pt.fixture
        def patch_next_state_init(self, mocker, monkeypatch):
            mock_init = mocker.Mock(return_value=None)
            monkeypatch.setattr(vd.DevicesState, '__init__', mock_init)
            return mock_init

        @pt.fixture
        def state(self, patch_next_state_init):
            return vd.SamplingFrequencyState()

        @pt.fixture(params=[('1200', 1200), ('31', 31)])
        def row_and_expected_output(self, request):
            return ([request.param[0], '', '', '', '', '', '',
                     ''], request.param[1])

        def test_accepts_valid(self, state, row_and_expected_output,
                               mock_reader, mock_validator):
            row = row_and_expected_output[0]
            state.feed_row(row, mock_reader)
            mock_validator.validate.assert_called_once()
            assert mock_validator.mock_validate_call.is_valid

        @pt.fixture
        def invalid_row(self):
            return ['73.2', '', '', '', '', '']

        def test_doesnt_accept_invalid(self, state, invalid_row, mock_reader,
                                       mock_validator):
            state.feed_row(invalid_row, mock_reader)
            mock_validator.validate.assert_called_once()
            assert not mock_validator.mock_validate_call.is_valid

        def test_creates_new_state(self, state, row_and_expected_output,
                                   mock_reader, patch_next_state_init):
            row = row_and_expected_output[0]
            state.feed_row(row, mock_reader)
            patch_next_state_init.assert_called_once_with()

        def test_builds_data_line(self, state, row_and_expected_output,
                                  mock_reader, mock_data_builder):
            row, expected_output = row_and_expected_output
            state.feed_row(row, mock_reader)
            mock_data_builder.add_frequency.assert_called_once_with(
                expected_output)

    class TestDeviceColsCreator:
        @pt.fixture
        def force_plate_header_str(self):
            return 'Imported AMTI OR6 Series Force Plate #1 - Force'

        @pt.fixture
        def trajectory_header_str(self):
            return 'Angelica:HV'

        @pt.fixture
        def emg_header_str(self):
            return 'EMG2000 - Voltage'

        @pt.fixture
        def force_plate_col_of_header(self, force_plate_header_str):
            return vd.ColOfHeader(2, force_plate_header_str)

        @pt.fixture
        def force_plate_exp_type(self):
            return vd.DeviceType.FORCE_PLATE

        @pt.fixture
        def trajectory_col_of_header(self, trajectory_header_str):
            return vd.ColOfHeader(5, trajectory_header_str)

        @pt.fixture
        def trajectory_exp_type(self):
            return vd.DeviceType.TRAJECTORY_MARKER

        @pt.fixture
        def emg_col_of_header(self, emg_header_str):
            return vd.ColOfHeader(8, emg_header_str)

        @pt.fixture
        def emg_exp_type(self):
            return vd.DeviceType.EMG

        @pt.fixture
        def unknown_device(self):
            return vd.ColOfHeader(2, 'Trajectory Marker')

        @pt.fixture
        def creator(self):
            return vd.DeviceColsCreator()

        @pt.mark.parametrize(
            'col_of_header,exp_type',
            [
                # force plate case
                pt.lazy_fixture(
                    ['force_plate_col_of_header', 'force_plate_exp_type']),

                # emg case
                pt.lazy_fixture(['emg_col_of_header', 'emg_exp_type']),

                # trajectory marker case
                pt.lazy_fixture(
                    ['trajectory_col_of_header', 'trajectory_exp_type']),
            ])
        def test_create_force_plate_correct_type(self, creator, col_of_header,
                                                 exp_type):
            inp = [col_of_header]
            failable_result = creator.create_cols(inp)
            header_cols = failable_result.parse_result[0]
            assert header_cols.device_type is exp_type

        @pt.mark.parametrize(
            'col_of_header,exp_name',
            [
                # force plate case
                pt.lazy_fixture(
                    ['force_plate_col_of_header', 'force_plate_header_str']),

                # emg case
                pt.lazy_fixture(['emg_col_of_header', 'emg_header_str']),

                # trajectory marker case
                pt.lazy_fixture(
                    ['trajectory_col_of_header', 'trajectory_header_str']),
            ])
        def test_create_force_plate_correct_name(self, creator, col_of_header,
                                                 exp_name):
            inp = [col_of_header]
            failable_result = creator.create_cols(inp)
            header_cols = failable_result.parse_result[0]
            assert header_cols.device_name == exp_name

        def test_create_unknown(self, creator, unknown_device):
            inp = [unknown_device]
            failable_result = creator.create_cols(inp)
            assert failable_result.parse_result is None
            assert not failable_result.data_check.is_valid

    class TestDeviceHeaderFinder:
        # yapf: disable
        WRONG_ROWS = (
            [
                'WRONG', '',
                'first_value', '', '',
                'second_value', '', '',
                '', '', '', '',
            ], [
                '', '',
                '', '', '',
                'second_value', '', '',
                '', '', '', '',
            ])
        # yapf: enable

        @pt.fixture
        def finder(self):
            return vd.DevicesLineFinder()

        @pt.mark.parametrize('wrong_row', WRONG_ROWS)
        def test_raises_if_wrong(self, finder, wrong_row):
            fail_res = finder.find_headers(wrong_row)
            assert fail_res.failed

        @pt.fixture
        def correct_row(self):
            # yapf: disable
            return  [
                '', '',
                'first_value', '', '',
                'second_value',
            ]
            # yapf: enable

        @pt.fixture
        def expected_output(self):
            return [
                vd.ColOfHeader(2, 'first_value'),
                vd.ColOfHeader(5, 'second_value')
            ]

        def test_finds_valid_input_doesnt_fail(self, finder, correct_row):
            fail_res = finder.find_headers(correct_row)
            assert not fail_res.failed

        def test_finds_correctly(self, finder, correct_row, expected_output):
            fail_res = finder.find_headers(correct_row)
            parse_result = fail_res.parse_result
            assert parse_result == expected_output

    class TestColsCategorizer:
        @pt.fixture
        def categorizer(self):
            return vd.ColsCategorizer()

        def test_empty_cols_list_fails(self, categorizer):
            fail_res = categorizer([])
            assert fail_res.failed

        @pt.fixture
        def force_plate_header_cols(self):
            col_of_header = vd.ColOfHeader(0, 'force plate')
            return vd.DeviceHeaderCols(device_type=vd.DeviceType.FORCE_PLATE,
                                       col_of_header=col_of_header)

        @pt.fixture
        def traj_header_cols(self):
            col_of_header = vd.ColOfHeader(0, 'traj')
            return vd.DeviceHeaderCols(
                device_type=vd.DeviceType.TRAJECTORY_MARKER,
                col_of_header=col_of_header)

        @pt.fixture
        def emg_header_cols(self):
            col_of_header = vd.ColOfHeader(0, 'emg')
            return vd.DeviceHeaderCols(device_type=vd.DeviceType.EMG,
                                       col_of_header=col_of_header)

        each_singleton_header_cols = pt.lazy_fixture(
            ['force_plate_header_cols', 'traj_header_cols', 'emg_header_cols'])

        @pt.mark.parametrize('inp', each_singleton_header_cols)
        def test_categorizes_singleton_doesnt_fail(self, categorizer, inp):
            fail_res = categorizer(inp)
            assert not fail_res.failed

        @pt.mark.parametrize('inp', each_singleton_header_cols)
        def test_categorizes_singleton_correctly(self, categorizer, inp):
            dev_type = inp.device_type
            fail_res = categorizer(inp)
            categorized_cols = fail_res.parse_result
            list_of_exp_type = categorized_cols.from_device_type(dev_type)
            assert list_of_exp_type[0] == inp
            assert len(list_of_exp_type) == 1

        @pt.mark.parametrize('inp', each_singleton_header_cols)
        def test_doesnt_categorize_spuriously(self, categorizer, inp):
            fail_res = categorizer(inp)
            categorized_cols = fail_res.parse_result
            all_header_cols = categorized_cols.all_header_cols()
            assert len(all_header_cols) == 1

        def test_categorize_3_at_once(self, categorizer,
                                      force_plate_header_cols,
                                      traj_header_cols, emg_header_cols):
            inp = [force_plate_header_cols, traj_header_cols, emg_header_cols]
            exp = vd.CategorizedCols(force_plates=[force_plate_header_cols],
                                     emg=emg_header_cols,
                                     trajectory_markers=[traj_header_cols])
            out = categorizer(inp)
            assert out == exp

        def test_fails_if_nonsense_section(self):
            # TODO fails if section doesn't make sense
            raise NotImplementedError()

    class TestForcePlateGrouper:
        @pt.fixture
        def first_force_plate_name(self):
            return 'Imported AMTI OR6 Series Force Plate #1'

        @pt.fixture
        def second_force_plate_name(self):
            return 'Imported AMTI OR6 Series Force Plate #2'

        @pt.fixture
        def first_force_plate_str(self):
            return {
                'force': 'Imported AMTI OR6 Series Force Plate #1 - Force',
                'moment': 'Imported AMTI OR6 Series Force Plate #1 - Moment',
                'cop': 'Imported AMTI OR6 Series Force Plate #1 - CoP'
            }

        @pt.fixture
        def second_force_plate_str(self):
            return {
                'force': 'Imported AMTI OR6 Series Force Plate #2 - Force',
                'moment': 'Imported AMTI OR6 Series Force Plate #2 - Moment',
                'cop': 'Imported AMTI OR6 Series Force Plate #2 - CoP'
            }

        def create_device_headers(self, device_names: List[str]
                                  ) -> List[vd.DeviceHeaderCols]:
            all_dev_cols = []

            for dev_name in device_names:
                dev_col = vd.DeviceHeaderCols(vd.DeviceType.FORCE_PLATE,
                                              dev_name, 0)
                all_dev_cols.append(dev_col)

            return all_dev_cols

        @pt.fixture
        def inp_cols_first_force_plate(self, first_force_plate_str):
            return create_device_headers(first_force_plate_str)

        @pt.fixture
        def inp_cols_second_force_plate(self, second_force_plate_str):
            return create_device_headers(second_force_plate_str)

        @pt.fixture
        def shuffled_inp(self, inp_cols_first_force_plate,
                         inp_cols_second_force_plate):
            full_inp = inp_cols_first_force_plate + inp_cols_second_force_plate
            random.seed(0)
            full_inp.shuffle()
            return full_inp

        @pt.fixture
        def exp_first_force_plate_cols(self, first_force_plate_str):
            return vd.ForcePlateCols(**first_force_plate_str)

        @pt.fixture
        def exp_second_force_plate_cols(self, second_force_plate_str):
            return vd.ForcePlateCols(**second_force_plate_str)

        @pt.fixture
        def grouper(self):
            return vd.ForcePlateGrouper()

        def test_group_valid_input_doesnt_fail(self, shuffled_inp):
            fail_res = grouper.group(shuffled_inp)
            assert not fail_res.failed

        @pt.fixture
        def inp_with_deleted_col(self, shuffled_inp):
            inp = list(shuffled_inp)
            inp.pop()
            return inp

        def test_group_invalid_fails(self, grouper, inp_with_deleted_col):
            fail_res = grouper.group(inp_with_deleted_col)
            assert fail_res.failed

        def test_groups_correctly(self, grouper, shuffled_inp,
                                  exp_first_force_plate_cols,
                                  exp_second_force_plate_cols):
            fail_res = grouper.group(inp_with_deleted_col)
            out_force_plate_col_list = fail_res.parse_result
            expected = {
                exp_first_force_plate_cols, exp_second_force_plate_cols
            }
            output = set(out_force_plate_col_list)
            assert output == expected

        def test_empty_list_fails(self, grouper):
            fail_res = grouper.group([])
            assert fail_res.failed

    class TestDevicesState:
        def successful_fail_res(self, parse_res):
            fail_res = mocker.Mock()
            fail_res.failed = False
            fail_res.parse_result = parse_res
            return fail_res

        @pt.fixture
        def mock_succ_finder(self, mocker):
            return mocker.Mock(return_value=self.successful_fail_res('finder'))

        @pt.fixture
        def mock_succ_creator(self, mocker):
            creator

            return mocker.Mock(return_value=self.successful_fail_res())

        @pt.fixture
        def mock_succ_categorizer(self, mocker):
            return mocker.Mock(
                return_value=self.successful_fail_res('categorizer'))

        @pt.fixture
        def mock_succ_grouper(self, mocker):
            return mocker.Mock(
                return_value=self.successful_fail_res('grouper'))

        @pt.fixture
        def succ_state(self, mock_succ_finder, mock_succ_creator,
                       mock_succ_categorizer, mock_succ_grouper):
            return vd.DevicesState(finder=mock_succ_finder,
                                   creator=mock_succ_creator,
                                   categorizer=mock_succ_categorizer,
                                   grouper=mock_succ_grouper)

        @pt.fixture
        def mock_row(self, mocker):
            return mocker.Mock(name='row')

        def test_succ_state_passes_output_along_components(
                self, succ_state, mock_row, mock_succ_finder,
                mock_succ_creator, mock_succ_categorizer, mock_succ_grouper):
            succ_state.feed_row(mock_row)
            mock_succ_finder.assert_called_once_with(mock_row)
            # TODO should get member .force_plates of categorizer output
            # to pass along to grouper

        def test_succ_state_creates_next_state(self):
            pass
            # TODO should get member .emg at the very least and pass along

        def test_succ_state_calls_validator(self):
            pass
            # TODO

        # TODO unscessful states can likely be all done in a single
        # parametrized test

        # this state does not call any DataBuilder method
        # the next one (coordinates line) does after it creates
        # a DataChanneler
        # after that, it should be easy


class TestFrequencies:
    LAST_INDEX_FORCES_EMG = 124460 - 1
    LAST_INDEX_TRAJ = 6223 - 1
    LAST_FRAME = 6223
    LAST_SUBFRAME_FORCES_EMG = 19
    LAST_SUBFRAME_TRAJ = 0

    @pt.fixture
    def frequencies(self):
        return vd.Frequencies(2000, 100, 6223)

    def test_frame_of_ind_first_section(self, frequencies):
        dev_type = vd.DeviceType.FORCE_PLATE
        last_index = self.LAST_INDEX_FORCES_EMG
        frame, subframe = frequencies.frame_subframe(dev_type, last_index)
        assert frame == self.LAST_FRAME
        assert subframe == self.LAST_SUBFRAME_FORCES_EMG

    def test_frame_of_ind_second_section(self, frequencies):
        dev_type = vd.DeviceType.TRAJECTORY_MARKER
        last_index = self.LAST_INDEX_TRAJ
        frame, subframe = frequencies.frame_subframe(dev_type, last_index)
        assert frame == self.LAST_FRAME
        assert subframe == self.LAST_SUBFRAME_TRAJ

    def test_index_of_first_section(self, frequencies):
        dev_type = vd.DeviceType.EMG
        last_index = self.LAST_INDEX_FORCES_EMG
        frame = self.LAST_FRAME
        subframe = self.LAST_SUBFRAME_FORCES_EMG
        index = frequencies.index(dev_type, frame, subframe)
        assert index == last_index

    def test_index_of_second_section(self, frequencies):
        dev_type = vd.DeviceType.TRAJECTORY_MARKER
        last_index = self.LAST_INDEX_TRAJ
        frame = self.LAST_FRAME
        subframe = self.LAST_SUBFRAME_TRAJ
        index = frequencies.index(dev_type, frame, subframe)
        assert index == last_index
