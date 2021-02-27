import pathlib

import pandas as pd
from pint_pandas import PintArray
import pytest as pt

import muscle_synergies.vicon_data as vd

this_file = pathlib.Path(__file__)
project_root = this_file.parent.parent.parent
sample_data_dir = project_root / 'sample_data'
abridged_csv = sample_data_dir / 'abridged_data.csv'


@pt.fixture(scope='module')
def abridged_data():
    return vd.load_vicon_file(abridged_csv)


@pt.fixture(scope='module')
def abridged_emg(module_mocker):
    device_name = 'EMG2000 - Voltage'
    device_type = vd.DeviceType.EMG
    frame_tracker = module_mocker.Mock()
    data = [
        [
            0.0037236, 0.00722359, 0.00344124, 0.00149971, -0.000798493,
            -0.00196037, -0.00602333, -0.00232391
        ],
        [
            0.00463913, 0.00478218, 0.00206795, 0.000889358, -3.56E-05,
            -0.00150261, -0.00373451, -0.0036972
        ],
        [
            0.00448654, 0.00142525, 0.000389481, -2.62E-05, -0.000798493,
            -0.00241814, 0.00191124, -0.00537567
        ],
        [
            0.00235031, -0.00147392, -0.00098381, -0.0021624, -0.000493317,
            -0.000587082, 0.00786217, -0.00644379
        ],
        [
            0.00204514, -0.00223686, -0.000220871, -0.0021624, -0.00156143,
            0.00200691, 0.0128976, -0.00522308
        ],
        [
            0.000519257, 5.20E-05, 0.00115242, -0.000789109, -0.00140884,
            0.00246468, 0.014576, -0.0012558
        ],
    ]
    pint_data = PintArray(data, dtype=vd.ureg.volts)
    coords = ('VL', 'RF', 'GMED', 'TFL', 'GMAXS', 'GMAXI', 'BF', 'ST')
    dataframe = pd.DataFrame(pint_data, columns=coords)
    return vd.DeviceData(device_name, device_type, frame_tracker, dataframe)


def test_loads_data(abridged_data):
    assert abridged_data.emg is not None
    assert abridged_data.force_plates is not None
    assert abridged_data.trajectory_markers is not None


def test_loads_emg_correctly(abridged_data, abridged_emg):
    assert abridged_data.emg == abridged_emg
