import pytest as pt

import muscle_synergies.vicon_data as vd


class TestValidator:
    @pt.fixture
    def validator_that_should_raise(self):
        return vd.Validator(csv_filename='not important.csv',
                            should_raise=True)

    @pt.fixture
    def validator_that_doesnt_raise(self):
        return vd.Validator(csv_filename='not important.csv',
                            should_raise=False)

    @pt.fixture
    def valid_data_check(self):
        return {'is_valid': True, 'error_message': 'irrelevant'}

    @pt.fixture
    def invalid_data_check(self):
        return {'is_valid': False, 'error_message': 'the error message'}

    def test_validator_starts_at_1(self, validator_that_should_raise):
        validator = validator_that_should_raise
        assert validator.current_line == 1

    def test_validator_increases_line_count(self, validator_that_should_raise,
                                            valid_data_check):
        validator = validator_that_should_raise
        data_check = valid_data_check
        validator.validate(data_check)
        assert validator.current_line == 2

    def test_validator_that_shouldnt_doesnt_raise(self,
                                                  validator_that_doesnt_raise,
                                                  invalid_data_check):
        validator = validator_that_doesnt_raise
        data_check = invalid_data_check
        assert validator(data_check) is None

    def test_validator_raises_with_invalid(self, validator_that_should_raise,
                                           invalid_data_check):
        validator = validator_that_should_raise
        data_check = invalid_data_check

        with pt.raises(ValueError):
            validator(data_check)

    def test_validator_doesnt_raise_with_valid(self,
                                               validator_that_should_raise,
                                               valid_data_check):
        validator = validator_that_should_raise
        data_check = valid_data_check
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
        return vd.DeviceHeaderCols(device_type=vd.DeviceType.EMG,
                                   device_name='EMG',
                                   first_col_index=3)

    @pt.fixture
    def force_plate_dev_cols(self):
        return vd.DeviceHeaderCols(device_type=vd.DeviceType.FORCE_PLATE,
                                   device_name='Force Plate',
                                   first_col_index=0)

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
        raise NotImplementedError


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

    class TestSectionTypeLineState:
        @pt.fixture
        def patch_next_state_init(self, mocker):
            mock_init = mocker.Mock(name='next state init',
                                    autospec=vd.SamplingFrequencyLineState)
            mocker.patch.object(vd, 'SamplingFrequencyLineState', mock_init)
            return mock_init

        @pt.fixture
        def state(self, patch_next_state_init):
            return vd.SectionTypeLineState()

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
            assert mock_validator.mock_validate_call['is_valid']

        @pt.fixture
        def invalid_row(self):
            return ['Invalid', '', '', '', '', '']

        def test_doesnt_accept_invalid(self, state, invalid_row, mock_reader,
                                       mock_validator):
            state.feed_row(invalid_row, mock_reader)
            mock_validator.validate.assert_called_once()
            assert not mock_validator.mock_validate_call['is_valid']

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
