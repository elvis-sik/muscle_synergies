import pytest as pt
from pytest_cases import fixture_ref, parametrize


class TestFullData:
    """Tests on a realistic dataset."""

    # shape of pandas arrays
    # number of devices
    # sampling frequency
    def test_laptop_doesnt_die(self, full_data):
        assert full_data.emg is not None


class TestAbridgedData:
    """Tests using a toy dataset."""
    def test_loads_correct_name(self, all_loaded_exp):
        loaded_dev_data, exp_dev_data = all_loaded_exp
        assert loaded_dev_data.name == exp_dev_data.name

    def test_loads_correct_dev_type(self, all_loaded_exp):
        loaded_dev_data, exp_dev_data = all_loaded_exp
        assert loaded_dev_data.dev_type == exp_dev_data.dev_type

    def test_loads_correct_units(self, all_loaded_exp):
        loaded_dev_data, exp_dev_data = all_loaded_exp
        assert loaded_dev_data.units == exp_dev_data.units

    def test_loads_correct_df(self, all_loaded_exp):
        loaded_dev_data, exp_dev_data = all_loaded_exp
        assert loaded_dev_data.df.equals(exp_dev_data.df)

    def test_traj_loads_correct_sampling_freq(self, traj_loaded):
        assert traj_loaded.sampling_frequency == 100

    def test_forces_emg_loads_correct_sampling_freq(self, forces_emg_loaded):
        assert forces_emg_loaded.sampling_frequency == 300

    @parametrize('device, index_seq', [
        (fixture_ref('forces_emg_loaded'),
         fixture_ref('forces_emg_index_seq')),
        (fixture_ref('traj_loaded'), fixture_ref('traj_index_seq')),
    ])
    def test_iloc(self, device, frame_subframe_seq, index_seq):
        for ((frame, subframe), ind) in zip(frame_subframe_seq, index_seq):
            loaded_row = device.iloc(frame, subframe)
            exp_row = device.df.iloc[ind]
            assert loaded_row.equals(exp_row)

    def test_specific_iloc(self, loaded_angelica_hv):
        loaded_row = loaded_angelica_hv.iloc(2, 2)
        loaded_row = list(loaded_row)
        expected_row = [209.475, 1219.82, 1780.88]
        assert loaded_row == expected_row

    def test_forces_emg_loads_correct_sampling_freq(
            self, all_loaded, invalid_frame_subframe_seq):
        device_data = all_loaded
        for (frame, subframe) in invalid_frame_subframe_seq:
            with pt.raises(KeyError):
                device_data.iloc(frame, subframe)
