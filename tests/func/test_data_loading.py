import pytest as pt


def test_loads_data(abridged_data):
    assert abridged_data.emg is not None
    assert abridged_data.force_plates is not None
    assert abridged_data.trajectory_markers is not None


@pt.mark.parametrize('loaded_dev_data,exp_dev_data', [
    pt.lazy_fixture(['loaded_emg', 'exp_emg']),
    pt.lazy_fixture(['loaded_forcep1', 'exp_forcep1']),
    pt.lazy_fixture(['loaded_forcep2', 'exp_forcep2']),
    pt.lazy_fixture(['loaded_angelica_hv', 'exp_angelica_hv']),
    pt.lazy_fixture(['loaded_angelica_cme', 'exp_angelica_cme']),
    pt.lazy_fixture(['loaded_angelica_cle', 'exp_angelica_cle']),
    pt.lazy_fixture(['loaded_angelica_elastdp', 'exp_angelica_elastdp']),
])
def test_loads_device_data_correctly(loaded_dev_data, exp_dev_data):
    assert loaded_dev_data == exp_dev_data
