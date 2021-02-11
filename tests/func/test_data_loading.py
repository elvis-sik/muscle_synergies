import pathlib

import pytest as pt

import muscle_synergies.vicon_data as vd

this_file = pathlib.Path(__file__)
project_root = this_file.parent.parent.parent
sample_data_dir = project_root / 'sample_data'
abridged_csv = sample_data_dir / 'abridged_data.csv'


@pt.mark.parametrize('csv_file', [abridged_csv])
def test_loads_data(csv_file):
    vicon_nexus_data = vd.load_vicon_file(csv_file)
    assert vicon_nexus_data.emg is not None
    assert vicon_nexus_data.force_plates is not None
    assert vicon_nexus_data.trajectory_markers is not None
