import pytest as pt

import muscle_synergies.vicon_data as vd


class TestSectionParser:
    @staticmethod
    def patch_section_parser_method(mocker, method_name):
        full_method_name = f'muscle_synergies.vicon_data.SectionParser.{method_name}'
        mocker.patch(full_method_name)

    @pt.fixture
    def patch_get_check_function(self, mocker):
        mocked_method = 'get_check_function'
        self.patch_section_parser_method(mocker, mocked_method)

    @pt.fixture
    def patch_get_parse_function(self, mocker):
        mocked_method = 'get_parse_function'
        self.patch_section_parser_method(mocker, mocked_method)

    @pt.fixture
    def patch_get_build_function(self, mocker):
        mocked_method = 'get_build_function'
        self.patch_section_parser_method(mocker, mocked_method)

    @pt.fixture
    def section_parser_with_patched_getters(self, mocker,
                                            patch_get_check_function,
                                            patch_get_parse_function,
                                            patch_get_build_function):
        return vd.SectionParser(builder=mocker.Mock(name='builder'),
                                validator=mocker.Mock(name='validator'))

    @pt.fixture
    def row_mock(self, mocker):
        return mocker.Mock(name='row')

    def test_feed_line_calls_getters(self, section_parser_with_patched_getters,
                                     row_mock):
        section_parser = section_parser_with_patched_getters
        row = row_mock
        section_parser.feed_line(cols_mock)
