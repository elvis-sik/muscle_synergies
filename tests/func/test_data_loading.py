import pathlib

import pytest as pt

import muscle_synergies.vicon_data as vd

this_file = pathlib.Path(__file__)
project_root = this_file.parent.parent.parent
sample_data_dir = project_root / 'sample_data'
abridged_csv = sample_data_dir / 'abridged_data.csv'


@pt.fixture(params=(abridged_csv, ), scope='module')
def vicon_nexus_data(request):
    csv_file = request.param
    return vd.load_vicon_file(csv_file)


def test_loads_data(vicon_nexus_data):
    assert vicon_nexus_data.emg is not None
    assert vicon_nexus_data.force_plates is not None
    assert vicon_nexus_data.trajectory_markers is not None
