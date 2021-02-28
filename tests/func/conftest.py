import pathlib

import pandas as pd
import pytest as pt
from pytest_cases import fixture as cases_fixture
from pytest_cases import parametrize

import muscle_synergies.vicon_data as vd

this_file = pathlib.Path(__file__)
project_root = this_file.parent.parent.parent
sample_data_dir = project_root / 'sample_data'
abridged_csv = sample_data_dir / 'abridged_data.csv'
full_data_csv = sample_data_dir / 'dynamic_trial.csv'


@pt.fixture(scope='package')
def full_data():
    return vd.load_vicon_file(full_data_csv)


@pt.fixture(scope='package')
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


@pt.fixture(scope='package')
def exp_emg(package_mocker):
    device_name = 'EMG2000 - Voltage'
    device_type = vd.DeviceType.EMG
    frame_tracker = package_mocker.Mock()
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


@pt.fixture(scope='package')
def exp_forcep1(package_mocker):
    device_name = 'Imported AMTI OR6 Series Force Plate #1'
    device_type = vd.DeviceType.FORCE_PLATE
    frame_tracker = package_mocker.Mock()
    data = [[0, 0, 0, 0, 0, 0, 232, 254, 0], [0, 0, 0, 0, 0, 0, 232, 254, 0],
            [0, 0, 0, 0, 0, 0, 232, 254, 0], [0, 0, 0, 0, 0, 0, 232, 254, 0],
            [0, 0, 0, 0, 0, 0, 232, 254, 0], [0, 0, 0, 0, 0, 0, 232, 254, 0]]
    coords = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz']
    units = ['N', 'N', 'N', 'N.mm', 'N.mm', 'N.mm', 'mm', 'mm', 'mm']
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_forcep2(package_mocker):
    device_name = 'Imported AMTI OR6 Series Force Plate #2'
    device_type = vd.DeviceType.FORCE_PLATE
    frame_tracker = package_mocker.Mock()
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
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_angelica_hv(package_mocker):
    device_name = 'Angelica:HV'
    device_type = vd.DeviceType.TRAJECTORY_MARKER
    frame_tracker = package_mocker.Mock()
    data = [
        [209.331, 1219.74, 1780.67],
        [209.475, 1219.82, 1780.88],
    ]
    coords = ('X', 'Y', 'Z')
    units = ('mm', 'mm', 'mm')
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_angelica_cme(package_mocker):
    device_name = 'Angelica:CM_E'
    device_type = vd.DeviceType.TRAJECTORY_MARKER
    frame_tracker = package_mocker.Mock()
    data = [
        [None, None, None],
        [None, None, None],
    ]
    coords = ('X', 'Y', 'Z')
    units = ('mm', 'mm', 'mm')
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_angelica_cle(package_mocker):
    device_name = 'Angelica:CL_E'
    device_type = vd.DeviceType.TRAJECTORY_MARKER
    frame_tracker = package_mocker.Mock()
    data = [
        [227.725, 1091.81, 496.721],
        [227.702, 1091.8, 496.729],
    ]
    coords = ('X', 'Y', 'Z')
    units = ('mm', 'mm', 'mm')
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_angelica_elastdp(package_mocker):
    device_name = 'Angelica:ELAST_DP'
    device_type = vd.DeviceType.TRAJECTORY_MARKER
    frame_tracker = package_mocker.Mock()
    data = [
        [None, None, None],
        [None, None, None],
    ]
    coords = ('X', 'Y', 'Z')
    units = ('mm', 'mm', 'mm')
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture
def frame_subframe_seq():
    return [
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]


@pt.fixture
def forces_emg_index_seq():
    return [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


@pt.fixture
def traj_index_seq():
    return [
        0,
        0,
        0,
        1,
        1,
        1,
    ]


@pt.fixture
def invalid_frame_subframe_seq():
    return [
        (-1, 0),
        (0, 3),
        (1, 3),
        (3, 0),
        (3, 2),
    ]


_forces_emg_loaded = [
    loaded_emg,
    loaded_forcep1,
    loaded_forcep2,
]
_traj_loaded = [
    loaded_angelica_hv,
    loaded_angelica_cme,
    loaded_angelica_cle,
    loaded_angelica_elastdp,
]
_all_loaded = _forces_emg_loaded + _traj_loaded


@cases_fixture
@parametrize('device_type', _forces_emg_loaded)
def forces_emg_loaded(device_type):
    return device_type


@cases_fixture
@parametrize('device_type', _traj_loaded)
def traj_loaded(device_type):
    return device_type


@cases_fixture
@parametrize('device_type', _all_loaded)
def all_loaded(device_type):
    return device_type


_forces_emg_exp = [
    exp_emg,
    exp_forcep1,
    exp_forcep2,
]
_traj_exp = [
    exp_angelica_hv,
    exp_angelica_cme,
    exp_angelica_cle,
    exp_angelica_elastdp,
]
_all_exp = _forces_emg_exp + _traj_exp


@cases_fixture
@parametrize('device_type', _forces_emg_exp)
def forces_emg_exp(device_type):
    return device_type


@cases_fixture
@parametrize('device_type', _traj_exp)
def traj_exp(device_type):
    return device_type


@cases_fixture
@parametrize('device_type', _all_exp)
def all_exp(device_type):
    return device_type


@cases_fixture
@parametrize('loaded, exp', zip(_all_loaded, _all_exp))
def all_loaded_exp(loaded, exp):
    return loaded, exp
