from itertools import chain
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
def forcep_cols():
    return ('Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz')


@pt.fixture(scope='package')
def emg_cols():
    return ('VL', 'RF', 'GMED', 'TFL', 'GMAXS', 'GMAXI', 'BF', 'ST')


@pt.fixture(scope='package')
def traj_cols():
    return ('X', 'Y', 'Z')


@pt.fixture(scope='package')
def forcep_units():
    return ['N', 'N', 'N', 'N.mm', 'N.mm', 'N.mm', 'mm', 'mm', 'mm']


@pt.fixture(scope='package')
def emg_units():
    return ('V', 'V', 'V', 'V', 'V', 'V', 'V', 'V')


@pt.fixture(scope='package')
def traj_units():
    return ('mm', 'mm', 'mm')


@pt.fixture(scope='package')
def exp_emg(package_mocker, emg_cols, emg_units):
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
    coords = emg_cols
    units = emg_units
    dataframe = pd.DataFrame(data, columns=coords)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_forcep1(package_mocker, forcep_cols, forcep_units):
    device_name = 'Imported AMTI OR6 Series Force Plate #1'
    device_type = vd.DeviceType.FORCE_PLATE
    frame_tracker = package_mocker.Mock()
    data = [[0, 0, 0, 0, 0, 0, 232, 254, 0], [0, 0, 0, 0, 0, 0, 232, 254, 0],
            [0, 0, 0, 0, 0, 0, 232, 254, 0], [0, 0, 0, 0, 0, 0, 232, 254, 0],
            [0, 0, 0, 0, 0, 0, 232, 254, 0], [0, 0, 0, 0, 0, 0, 232, 254, 0]]
    coords = forcep_cols
    units = forcep_units
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_forcep2(package_mocker, forcep_cols, forcep_units):
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
    coords = forcep_cols
    units = forcep_units
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_angelica_hv(package_mocker, traj_cols, traj_units):
    device_name = 'Angelica:HV'
    device_type = vd.DeviceType.TRAJECTORY_MARKER
    frame_tracker = package_mocker.Mock()
    data = [
        [209.331, 1219.74, 1780.67],
        [209.475, 1219.82, 1780.88],
    ]
    coords = traj_cols
    units = traj_units
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_angelica_cme(package_mocker, traj_cols, traj_units):
    device_name = 'Angelica:CM_E'
    device_type = vd.DeviceType.TRAJECTORY_MARKER
    frame_tracker = package_mocker.Mock()
    data = [
        [None, None, None],
        [None, None, None],
    ]
    coords = traj_cols
    units = traj_units
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_angelica_cle(package_mocker, traj_cols, traj_units):
    device_name = 'Angelica:CL_E'
    device_type = vd.DeviceType.TRAJECTORY_MARKER
    frame_tracker = package_mocker.Mock()
    data = [
        [227.725, 1091.81, 496.721],
        [227.702, 1091.8, 496.729],
    ]
    coords = traj_cols
    units = traj_units
    dataframe = pd.DataFrame(data, columns=coords, dtype=float)
    return vd.DeviceData(device_name, device_type, units, frame_tracker,
                         dataframe)


@pt.fixture(scope='package')
def exp_angelica_elastdp(package_mocker, traj_cols, traj_units):
    device_name = 'Angelica:ELAST_DP'
    device_type = vd.DeviceType.TRAJECTORY_MARKER
    frame_tracker = package_mocker.Mock()
    data = [
        [None, None, None],
        [None, None, None],
    ]
    coords = traj_cols
    units = traj_units
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


@pt.fixture
def full_data_forcep(full_data):
    return full_data.force_plates


@pt.fixture
def full_data_emg(full_data):
    return [full_data.emg]


@pt.fixture
def full_data_forces_emg(full_data_forcep, full_data_emg):
    return full_data_forcep + full_data_emg


@pt.fixture
def full_data_traj(full_data):
    return full_data.trajectory_markers


@pt.fixture
def full_data_forcep_names():
    return [
        'Imported AMTI OR6 Series Force Plate #1',
        'Imported AMTI OR6 Series Force Plate #2',
    ]


@pt.fixture
def full_data_emg_names():
    return [
        'EMG2000 - Voltage',
    ]


@pt.fixture
def full_data_traj_names():
    return [
        'Angelica:HV',
        'Angelica:AUXH_D',
        'Angelica:AUXH_E',
        'Angelica:SEL',
        'Angelica:C7',
        'Angelica:T8',
        'Angelica:IJ',
        'Angelica:PX',
        'Angelica:CIAS_D',
        'Angelica:CIAS_E',
        'Angelica:CIPS_D',
        'Angelica:CIPS_E',
        'Angelica:AUXP_D',
        'Angelica:AUXP_E',
        'Angelica:TROC_E',
        'Angelica:PFC_E',
        'Angelica:CM_E',
        'Angelica:CL_E',
        'Angelica:TROC_D',
        'Angelica:PFC_D',
        'Angelica:CM_D',
        'Angelica:CL_D',
        'Angelica:TT_E',
        'Angelica:FH_E',
        'Angelica:MM_E',
        'Angelica:ML_E',
        'Angelica:TT_D',
        'Angelica:FH_D',
        'Angelica:MM_D',
        'Angelica:ML_D',
        'Angelica:CAL_E',
        'Angelica:MT1_E',
        'Angelica:MT5_E',
        'Angelica:CAL_D',
        'Angelica:MT1_D',
        'Angelica:MT5_D',
        'Angelica:ELAST_DA',
        'Angelica:ELAST_EA',
        'Angelica:ELAST_EP',
        'Angelica:ELAST_DP',
    ]


@pt.fixture
def full_data_forcep_shape():
    return 124460, 9


@pt.fixture
def full_data_emg_shape():
    return 124460, 8


@pt.fixture
def full_data_traj_shape():
    return 6223, 3


@pt.fixture
def full_data_forces_emg_means():
    return [
        1.75390396659708, -582.415392400259, -224261.899956882,
        2930.43315779098, 305.616860233506, 6143.9331267351, 115103.72594715,
        127978.653400808, 0, 86.4862138638483, 376.811106649415,
        -210797.394100982, 3726.82722993296, 3621.2547001867,
        -7597.25182040396, 110247.387950198, 358138.123797518, 0, None, 53,
        None, 146, -107, None, None, None
    ]


@pt.fixture
def full_data_forces_emg_series(full_data_forcep, full_data_emg):
    all_cols = []
    for dev in chain(full_data_forcep, full_data_emg):
        df = dev.df
        for col in df:
            all_cols.append([df[col], dev])
    return all_cols


@pt.fixture
def full_data_traj_means():
    return [
        86254.6503759399, 441750.737376509, 1710.43548387097, 89083.3316115702,
        466789.050856982, 1583.93220338983, 74485.4062078273, 435400.986374696,
        1583.08333333333, 65471.2488408037, 439095.981927711, 1563.09259259259,
        289051.956723891, 432509.803571429, 1515.64516129032, 370345.374461207,
        430112.819397993, 1315.57142857143, 187244.840435015, 443328.544333683,
        1427.90909090909, 115628.150347913, 439105.443782383, 1219.98245614035,
        201159.587224906, 503231.118074234, 981138.481433394, 184587.652508961,
        324946.594459582, 982229.877121865, 369286.582885786, 450095.862452523,
        1031.83928571429, 364119.987001424, 390798.143796992, 1030.75806451613,
        264295.59037642, 513369.995182013, 1069.671875, 244492.636200717,
        300625.81838717, 1067.08235294118, 271428.795819936, 278052.540188595,
        926517.204347051, 124915.414349601, 320515.049823633, 648768.235934988,
        None, None, None, 163912.935679828, 286816.834319527, 484565.121990369,
        279104.393556426, 538425.058972199, 936325.080373161, 154375.836357104,
        501481.870258384, 797904.630994989, None, None, None, 157512.582706767,
        548077.878122427, 489614.968738835, 150756.891035963, 314112.733940972,
        416703.78888295, 209109.526605179, 267892.384918895, 442905.946633238,
        None, None, None, 293481.419079765, 259685.635437431, 112790.378028169,
        144356.303940975, 535653.256558086, 414677.173478494, 197890.552776299,
        547410.663575042, 448577.227288984, None, None, None, 284453.11212938,
        550929.683778778, 111320.290552585, 347953.29250133, 295133.104998954,
        65732.9650145773, 152825.430686887, 298736.884229904, 44328.8263157895,
        191521.435943061, 228325.867743344, 35072.1387387387, 348163.184271259,
        530699.574671773, 81448.086574655, 148923.439687333, 530565.738205618,
        50839.1935483871, 189936.626609442, 502073.090742996, 42207.4635036496
    ]


@pt.fixture
def full_data_traj_series(full_data_traj):
    all_cols = []
    for dev in full_data_traj:
        df = dev.df
        for col in df:
            all_cols.append([df[col], dev])
    return all_cols
