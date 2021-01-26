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

    def test_validator_starts_at_1(validator_that_should_raise):
        validator = validator_that_should_raise
        assert validator.current_line == 1

    def test_validator_increases_line_count(validator_that_should_raise,
                                            valid_data_check):
        validator = validator_that_should_raise
        data_check = valid_data_check
        validator.validate(data_check)
        assert validator.current_line == 2

    def test_validator_that_shouldnt_doesnt_raise(validator_that_doesnt_raise,
                                                  invalid_data_check):
        validator = validator_that_doesnt_raise
        data_check = invalid_data_check
        assert validator(data_check) is None

    def test_validator_raises_with_invalid(validator_that_should_raise,
                                           invalid_data_check):
        validator = validator_that_should_raise
        data_check = invalid_data_check

        with pt.raises(ValueError):
            validator(data_check)

    def test_validator_doesnt_raise_with_valid(validator_that_should_raise,
                                               valid_data_check):
        validator = validator_that_should_raise
        data_check = valid_data_check
        assert validator(data_check) is None


class TestTimeSeriesDataBuilder:
    pass


class TestDeviceHeaderDataBuilder:
    pass


class TestAllDevicesDataBuilder:
    """Class spec:

    1. initialization:

    Arguments:
    - force_plates (Mapping[str, List[DeviceHeaderCols]]): a dict like
      {'Force plate 1': [its, DeviceHeaderCols]}
    - emg_device (DeviceHeaderCols)
    - trajectory_marker_devices (List[DeviceHeaderCols])
    - device_header_data_builder_type (optional): the class used to represent
      individual device headers. By default, it is DeviceHeaderDataBuilder
    - time_series_data_builder_type (optional): the class used to represent
      individual time series. By default, it is TimeSeriesDataBuilder.

    Behavior:
    - initializes one DeviceHeaderDataBuilder per DeviceHeaderCols it is
      fed
    - the class blindly believes the data it is provided with makes sense
    """
    @pt.fixture
    def mock_device_header_data_builder_type(self, mocker):
        return mocker.create_autospec(vd.DeviceHeaderDataBuilder)

    @pt.fixture
    def mock_time_series_data_builder_type(self, mocker):
        return mocker.create_autospec(vd.TimeSeriesDataBuilder)

    @pt.fixture
    def force_plate_header_cols(self):
        return {
            'first force plate': [
                vd.DeviceHeaderCols(device_name='first force plate',
                                    device_type=vd.DeviceType.FORCE_PLATE,
                                    first_col_index=2)
            ]
        }

    @pt.fixture
    def trajectory_marker_header_cols(self):
        return vd.DeviceHeaderCols(device_name='trajectory marker',
                                   device_type=vd.DeviceType.TRAJECTORY_MARKER,
                                   first_col_index=5)

    @pt.fixture
    def emg_header_cols(self):
        return vd.DeviceHeaderCols(device_name='emg',
                                   device_type=vd.DeviceType.EMG,
                                   first_col_index=8)

    @pt.fixture
    def all_devices_data_builder(self, mock_device_header_data_builder_type,
                                 mock_time_series_data_builder_type,
                                 force_plate_header_cols,
                                 trajectory_marker_header_cols,
                                 emg_header_cols):
        return vd.AllDevicesDataBuilder(
            force_plates=force_plater_header_cols,
            emg_device=emg_header_cols,
            trajectory_marker_devices=trajectory_marker_header_cols,
            device_header_data_builder_type=
            mock_device_header_data_builder_type,
            time_series_data_builder_type=mock_time_series_data_builder_type,
        )

    def test_initializes_data_header_for_force_plate(
            self, all_devices_data_builder,
            mock_device_header_data_builder_type,
            mock_time_series_data_builder_type):
        device_name = 'first force plate'
        device_type = vd.DeviceType.FORCE_PLATE
        mock_time_series_data_builder_type
        mock_device_header_data_builder_type.__init__.assert_called_with(
            device_name=device_name,
            device_type=device_type,
            time_series_data_builder_type=mock_time_series_data_builder_type)

    def test_initializes_data_header_for_trajectory_marker(
            self, all_devices_data_builder,
            mock_device_header_data_builder_type,
            mock_time_series_data_builder_type):
        device_name = 'trajectory marker'
        device_type = vd.DeviceType.TRAJECTORY_MARKER
        mock_time_series_data_builder_type
        mock_device_header_data_builder_type.__init__.assert_called_with(
            device_name=device_name,
            device_type=device_type,
            time_series_data_builder_type=mock_time_series_data_builder_type)

    def test_initializes_data_header_for_force_plate(
            self, all_devices_data_builder,
            mock_device_header_data_builder_type,
            mock_time_series_data_builder_type):
        device_name = 'emg'
        device_type = vd.DeviceType.EMG
        mock_time_series_data_builder_type
        mock_device_header_data_builder_type.__init__.assert_called_with(
            device_name=device_name,
            device_type=device_type,
            time_series_data_builder_type=mock_time_series_data_builder_type)

    # 2. has a add_coordinates method:
    #
    #    Arguments:
    #    - parsed_row (Row): the coordinates line of the input with its empty
    #                        columns at the end stripped.
    #
    #    Behaviors:
    #    - passes downstream the columns for the DeviceHeaderDataBuilder
    #      instances that were created during AllDevicesDataBuilder
    #      initialization.
    #    - the method it calls is named add_coordinates as well
    #    - each device spans 3 columns, the first one being the one given in
    #      AllDevicesDataBuilder initialization
    #    - EMG devices are different, they have a variable number of columns
    #      which should corresponds to the number of columns from its first
    #      until the final one provided

    # 3. has a add_units method
    #
    #    Arguments:
    #    - physical_units_row (List[pint.Unit]): the units line of the csv file
    #                                            with its empty columns at the
    #                                            end stripped
    #
    #    Behaviors:
    #    - similar to add_coordinates

    # 4. has a add_data method
    #
    #    Arguments:
    #    - data_row (List[float]): a data line of the csv file with its empty
    #                              columns at the end stripped
    #
    #    Behaviors:
    #    - similar to add_coordinates


class TestSectionDataBuilder:
    pass


class TestSectionReader:
    # NEXT create tests for these behaviors:

    # 1. when it is fed a row, it simply calls .feed_row(row, itself) on its
    #    current state (which then becomes responsible for all of the following:
    #    + checking and calling Validator
    #    + passing parsed data to data builder (which it gets from the Reader)
    #    + deciding whether the state should change and telling the Reader if it
    #    should)

    # 2. ???? it should tell the builder to do something after everything ends?
    #    I don't know.
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


class TestViconNexusCSVReader:
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

    # TODO idea is for this guy to have its own state (unneeded for tests)
    # then call his forces_emg_reader in its first state
    # then transition to the second when it raises EOF
