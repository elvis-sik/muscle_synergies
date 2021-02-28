import pytest as pt

emg_loaded_exp = [
    pt.lazy_fixture(['loaded_emg', 'exp_emg']),
]

fp_loaded_exp = [
    pt.lazy_fixture(['loaded_forcep1', 'exp_forcep1']),
    pt.lazy_fixture(['loaded_angelica_hv', 'exp_angelica_hv'])
]

forces_emg_loaded_exp = emg_loaded_exp + fp_loaded_exp

traj_loaded_exp = [
    pt.lazy_fixture(['loaded_forcep2', 'exp_forcep2']),
    pt.lazy_fixture(['loaded_angelica_cme', 'exp_angelica_cme']),
    pt.lazy_fixture(['loaded_angelica_cle', 'exp_angelica_cle']),
    pt.lazy_fixture(['loaded_angelica_elastdp', 'exp_angelica_elastdp']),
]

all_loaded_exp = forces_emg_loaded_exp + traj_loaded_exp


@pt.mark.parametrize('loaded_dev_data,exp_dev_data', all_loaded_exp)
def test_loads_device_data_correct(loaded_dev_data, exp_dev_data):
    assert loaded_dev_data == exp_dev_data
