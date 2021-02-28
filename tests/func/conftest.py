import pathlib

import pandas as pd
import pytest as pt

import muscle_synergies.vicon_data as vd

this_file = pathlib.Path(__file__)
project_root = this_file.parent.parent.parent
sample_data_dir = project_root / 'sample_data'
abridged_csv = sample_data_dir / 'abridged_data.csv'


@pt.fixture(scope='module')
def abridged_data():
    return vd.load_vicon_file(abridged_csv)


@pt.fixture
def loaded_emg(abridged_data):
    return abridged_data.emg


@pt.fixture
def loaded_forcep1(abridged_data):
    return abridged_data.force_plates[0]


@pt.fixture
def loaded_forcep2(abridged_data):
    return abridged_data.force_plates[1]


@pt.fixture
def loaded_angelica_hv(abridged_data):
    return abridged_data.trajectory_markers[0]


@pt.fixture
def loaded_angelica_cme(abridged_data):
    return abridged_data.trajectory_markers[1]


@pt.fixture
def loaded_angelica_cle(abridged_data):
    return abridged_data.trajectory_markers[2]


@pt.fixture
def loaded_angelica_elastdp(abridged_data):
    return abridged_data.trajectory_markers[3]


@pt.fixture(scope='module')
def exp_emg(module_mocker):
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
    coords = ('VL', 'RF', 'GMED', 'TFL', 'GMAXS', 'GMAXI', 'BF', 'ST')
    units = ('V', 'V', 'V', 'V', 'V', 'V', 'V', 'V')
    dataframe = pd.DataFrame(data, columns=coords)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='module')
def exp_forcep1(module_mocker):
    device_name = 'Imported AMTI OR6 Series Force Plate #1'
    device_type = DeviceType.FORCE_PLATE
    frame_tracker = module_mocker.Mock()
    data = [[0, 0, 0, 0, 0, 0, 232, 254, 0], [0, 0, 0, 0, 0, 0, 232, 254, 0],
            [0, 0, 0, 0, 0, 0, 232, 254, 0], [0, 0, 0, 0, 0, 0, 232, 254, 0],
            [0, 0, 0, 0, 0, 0, 232, 254, 0], [0, 0, 0, 0, 0, 0, 232, 254, 0]]
    coords = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz']
    units = ['N', 'N', 'N', 'N.mm', 'N.mm', 'N.mm', 'mm', 'mm', 'mm']
    dataframe = pd.DataFrame(data, columns=coords)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='module')
def exp_forcep2(module_mocker):
    device_name = 'Imported AMTI OR6 Series Force Plate #2'
    device_type = DeviceType.FORCE_PLATE
    frame_tracker = module_mocker.Mock()
    data = [
        [0, 0, 0, 0, 0, 0, 232, 769, 0],
        [0, 0, 0, 0, 0, 0, 232, 769, 0],
        [0, 0, 0, 0, 0, 0, 232, 769, 0],
        [0, 0, 0, 0, 0, 0, 232, 769, 0],
        [0, 0, 0, 0, 0, 0, 232, 769, 0],
        [0, 0, 0, 0, 0, 0, 232, 769, 0],
    ]
    coords = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz']
    units = ['N', 'N', 'N', 'N.mm', 'N.mm', 'N.mm', 'mm', 'mm', 'mm']
    dataframe = pd.DataFrame(data, columns=coords)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='module')
def exp_angelica_hv(module_mocker):
    device_name = 'Angelica:HV'
    device_type = DeviceType.TRAJECTORY_MARKER
    frame_tracker = module_mocker.Mock()
    data = [
        [209331, 1219.74, 1780.67],
        [209475, 1219.82, 1780.88],
    ]
    coords = ('X', 'Y', 'Z')
    units = ('mm', 'mm', 'mm')
    dataframe = pd.DataFrame(data, columns=coords)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='module')
def exp_angelica_cme(module_mocker):
    device_name = 'Angelica:CM_E'
    device_type = DeviceType.TRAJECTORY_MARKER
    frame_tracker = module_mocker.Mock()
    data = [
        [None, None],
        [None, None],
    ]
    coords = ('X', 'Y', 'Z')
    units = ('mm', 'mm', 'mm')
    dataframe = pd.DataFrame(data, columns=coords)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='module')
def exp_angelica_cle(module_mocker):
    device_name = 'Angelica:CL_E'
    device_type = DeviceType.TRAJECTORY_MARKER
    frame_tracker = module_mocker.Mock()
    data = [
        [227725, 1091.81, 496721],
        [227702, 1091.8, 496729],
    ]
    coords = ('X', 'Y', 'Z')
    units = ('mm', 'mm', 'mm')
    dataframe = pd.DataFrame(data, columns=coords)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='module')
def exp_angelica_elastdp(module_mocker):
    device_name = 'Angelica:ELAST_DP'
    device_type = DeviceType.TRAJECTORY_MARKER
    frame_tracker = module_mocker.Mock()
    data = [
        [None, None],
        [None, None],
    ]
    coords = ('X', 'Y', 'Z')
    units = ('mm', 'mm', 'mm')
    dataframe = pd.DataFrame(data, columns=coords)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)
