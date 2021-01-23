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
    pass


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
