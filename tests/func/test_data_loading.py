# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
import numpy as np
import pytest as pt
from pytest_cases import fixture_ref, parametrize


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

    @parametrize(
        "device, index_seq",
        [
            (fixture_ref("forces_emg_loaded"), fixture_ref("forces_emg_index_seq")),
            (fixture_ref("traj_loaded"), fixture_ref("traj_index_seq")),
        ],
    )
    def test_get_frame_subfr(self, device, frame_subframe_seq, index_seq):
        for ((frame, subframe), ind) in zip(frame_subframe_seq, index_seq):
            loaded_row = device[frame, subframe]
            exp_row = device.df.iloc[ind]
            assert loaded_row.equals(exp_row)

    def test_specific_get_frame_subfr(self, loaded_angelica_hv):
        loaded_row = loaded_angelica_hv[2, 2]
        loaded_row = list(loaded_row)
        expected_row = [209.475, 1219.82, 1780.88]
        assert loaded_row == expected_row

    def test_forces_emg_invalid_frame_subframe(
        self, all_loaded, invalid_frame_subframe_seq
    ):
        device_data = all_loaded
        for (frame, subframe) in invalid_frame_subframe_seq:
            with pt.raises(IndexError):
                # pylint: disable=pointless-statement
                device_data[frame, subframe]
                # pylint: enable=pointless-statement


class TestFullData:
    """Tests on a realistic dataset."""

    def test_correct_num_force_plates(self, full_data):
        assert len(full_data.forcepl) == 2

    def test_there_is_emg(self, full_data):
        assert full_data.emg is not None

    def test_correct_num_traj(self, full_data):
        assert len(full_data.traj) == 40

    @parametrize(
        "devices, exp_names",
        [
            (fixture_ref("full_data_forcep"), fixture_ref("full_data_forcep_names")),
            (fixture_ref("full_data_emg_list"), fixture_ref("full_data_emg_names")),
            (fixture_ref("full_data_traj"), fixture_ref("full_data_traj_names")),
        ],
    )
    def test_load_correct_names(self, devices, exp_names):
        for (dev, name) in zip(devices, exp_names):
            assert dev.name == name

    @parametrize(
        "devices, exp_cols",
        [
            (fixture_ref("full_data_forcep"), fixture_ref("forcep_cols")),
            (fixture_ref("full_data_emg_list"), fixture_ref("emg_cols")),
            (fixture_ref("full_data_traj"), fixture_ref("traj_cols")),
        ],
    )
    def test_correct_cols(self, devices, exp_cols):
        for dev in devices:
            coords = tuple(dev.df.columns)
            assert coords == tuple(exp_cols)

    @parametrize(
        "devices, exp_units",
        [
            (fixture_ref("full_data_forcep"), fixture_ref("forcep_units")),
            (fixture_ref("full_data_emg_list"), fixture_ref("emg_units")),
            (fixture_ref("full_data_traj"), fixture_ref("traj_units")),
        ],
    )
    def test_correct_units(self, devices, exp_units):
        for dev in devices:
            loaded_units = dev.units
            assert tuple(loaded_units) == tuple(exp_units)

    def test_traj_sampling_freq(self, full_data_traj):
        for dev in full_data_traj:
            assert dev.sampling_frequency == 100

    def test_forces_emg_sampling_freq(self, full_data_forces_emg):
        for dev in full_data_forces_emg:
            assert dev.sampling_frequency == 2000

    @parametrize(
        "devices, exp_shape",
        [
            (fixture_ref("full_data_forcep"), fixture_ref("full_data_forcep_shape")),
            (fixture_ref("full_data_emg_list"), fixture_ref("full_data_emg_shape")),
            (fixture_ref("full_data_traj"), fixture_ref("full_data_traj_shape")),
        ],
    )
    def test_traj_data_shape(self, devices, exp_shape):
        for dev in devices:
            assert dev.df.shape == exp_shape

    def test_col_average_traj(self, full_data_angelica_hv, angelica_hv_average):
        datafr = full_data_angelica_hv.df
        exp_x, exp_y, exp_z = angelica_hv_average
        mean_x = datafr["X"].mean()
        assert np.isclose(mean_x, exp_x)
        mean_y = datafr["Y"].mean()
        assert np.isclose(mean_y, exp_y)
        mean_z = datafr["Z"].mean()
        assert np.isclose(mean_z, exp_z)

    def test_col_average_forcepl_last_5000(self, full_data_forcepl_2, forcepl2_average):
        datafr = full_data_forcepl_2.df
        for (col, exp_average) in zip(datafr, forcepl2_average):
            last_5000 = datafr[col].iloc[-5000:]
            mean = last_5000.mean()
            assert np.isclose(mean, exp_average)
