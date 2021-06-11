"""Functions to analyze the EMG signal and compute muscle synergies.

All functions work with :py:class:`pandas.DataFrame`. Each column of the
:py:class:`~pandas.DataFrame` is assumed to be a different 1D signal.  For
example, each column could correspond to the electromyogram of a single muscle.

The functions in this file should be thought of as a thin layer on top of
standard functions in the Python scientific ecosystem. More complex use cases
will benefit from using those directly. Hopefully in those cases the
functions present here can be an useful starting point.
"""

from collections import OrderedDict
from dataclasses import dataclass
import functools
from typing import Tuple, Sequence, Union, Optional, Mapping, Any

import matplotlib.pyplot as plt

plt.style.use("bmh")
import numpy as np
import pandas
from scipy.fftpack import fft, fftfreq
import scipy.signal as signal
import scipy.interpolate as interpolate
import seaborn as sns
from sklearn.decomposition import NMF

_NUMPY_ARRAY_LIKE = Any
"""An object that can be cast to a NumPy array of a numeric type."""


def plot_signal(
    signal_df: pandas.DataFrame,
    *,
    title="Raw EMG signal",
    xlabel="time (s)",
    ylabel="Volts",
    columns=None,
    plot_dims=None,
    xticks_off=False,
    figsize=(18, 10),
    suptitle_fontsize=20,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a signal."""
    if isinstance(columns, str):
        pass
    if columns is None:
        columns = signal_df.columns
    if len(columns) > 1:
        return plot_columns(
            signal_df[columns],
            title=title,
            plot_dims=plot_dims,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks_off=xticks_off,
            figsize=figsize,
            suptitle_fontsize=suptitle_fontsize,
        )


def plot_columns(
    signal_df,
    *,
    title,
    plot_dims,
    xlabel="time (s)",
    ylabel="Volts",
    xticks_off=False,
    figsize=(18, 10),
    suptitle_fontsize=20,
):
    """Plot EMG DataFrame with 8 columns."""
    assert len(signal_df.columns) == np.prod(plot_dims)
    fig, axs = plt.subplots(plot_dims[0], plot_dims[1], figsize=figsize)
    if len(axs.shape) == 1:
        axs = np.expand_dims(axs, axis=1)
    for (ax, col) in zip(axs.flat, signal_df.columns):
        signal_df[col].plot(ax=ax)
        ax.set_title(col)
        if xticks_off:
            ax.set_xticks([], [])
        ax.set(xlabel=xlabel)
    fig.suptitle(title, fontsize=suptitle_fontsize)
    axs[0, 0].set_ylabel(ylabel)
    axs[1, 0].set_ylabel(ylabel)
    plt.show()


def synergy_heatmap(
    components: pandas.DataFrame, columns=None, synergy_names: Sequence[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot synergy heatmap.

    Args:
        components: a `(num_components, num_muscles)`
            :py:class:`~pandas.DataFrame` with one synergy component per row.

        columns: column labels referring to the muscles that will be displayed
            in the heatmap. For example, if the synergy components were
            obtained from a :py:class:`~pandas.DataFrame`, its `.columns`
            member could contain labels that might make the plot more
            informative.

        synergy_names: the names of each synergy component (each row of
            `components`) to be displayed in the heatmap. By default, `"synergy
            {i}"` will be used as a format string and `i` will start from 1.
    """
    fig, ax = plt.subplots()
    num_synergies = components.shape[0]
    if synergy_names is None:
        synergy_names = [f"synergy {i}" for i in range(1, num_synergies + 1)]
    synergies = pandas.DataFrame(components, index=synergy_names, columns=columns)
    sns.heatmap(synergies, annot=True, fmt=".2f", ax=ax)
    plt.title("Heatmap of muscle synergies")
    return fig, ax


def fft_spectrum(
    signal_df: pandas.DataFrame, sampling_frequency: int
) -> pandas.DataFrame:
    """Find the spectrum corresponding to positive frequencies.

    More complex use cases should use :py:mod:`scipy.fft` directly.

    Args:
        signal_df: a :py:class:`~pandas.DataFrame` with a different
            discrete-time signal in each of its columns.

        sampling_frequency: the sampling rate with which measurements were
            made.

    Returns:
        a new :py:class:`~pandas.DataFrame`. Each of its columns contain the
            spectrum of the corresponding column in `signal_df`. The
            frequencies are given as its :py:attr:`~pandas.DataFrame.index`.
            These frequencies are given in the same units as
            `sampling_frequency`.
    """
    signal_df = pandas.DataFrame(signal_df)
    num_samples = signal_df.shape[0]
    sample_spacing = 1.0 / sampling_frequency
    fft_freqs = fftfreq(num_samples, sample_spacing)
    positive_freq_ind = fft_freqs > 0
    fft_freqs = fft_freqs[positive_freq_ind]
    fft_arr = fft(signal_df, axis=0)
    fft_ampls_arr = np.abs(fft_arr[positive_freq_ind])
    return pandas.DataFrame(fft_ampls_arr, index=fft_freqs, columns=signal_df.columns)


def plot_fft(
    signal_df,
    sampling_frequency,
    plot_dims=(2, 4),
    title="EMG spectrum",
    xlabel="frequency (Hz)",
    ylabel="magnitude (V)",
    figsize=(18, 10),
):
    """Plot spectrum of signal"""
    spectrum_df = fft_spectrum(signal_df, sampling_frequency)
    plot_emg_signal(
        spectrum_df,
        plot_dims=plot_dims,
        title="EMG spectrum",
        xlabel="frequency (Hz)",
        ylabel="magnitude (V)",
        xticks_off=False,
        figsize=figsize,
    )


def _recreate_signal(
    signal_df: pandas.DataFrame,
    inplace: bool = False,
    with_array: Optional[_NUMPY_ARRAY_LIKE] = None,
) -> pandas.DataFrame:
    """Create new DataFrame if needed.

    This is mainly a helper for other functions that want to support both
    in-place and pure transformations.

    Args:
        signal_df: a :py:class:`~pandas.DataFrame` with a different
            discrete-time signal in each of its columns.

        inplace: if `True`, the original
            :py:class:`~pandas.DataFrame` will be returned. If
            `False`, a new one will be.

        with_array: if provided, should be an array-like object that will be
            used to replace the data of `signal_df` or of its copy (depending on
            the value of `inplace`).
    """
    if inplace is False:
        signal_df = pandas.DataFrame(signal_df, copy=True)
    if with_array is not None:
        signal_df[:] = with_array
    return signal_df


def zero_center(signal_df: pandas.DataFrame, inplace: bool = False) -> pandas.DataFrame:
    """Subtract the mean of each column from it.

    This is used to make the mean the signal corresponding to each muscle
    exactly 0. The EMG signal in theory has 0 mean but in practice the
    different muscles will commonly have different non-zero but very small
    means.

    Args:
        signal_df: a :py:class:`~pandas.DataFrame` with a different
            discrete-time signal in each of its columns.

        inplace: if `True`, the data in the original
            :py:class:`~pandas.DataFrame` will be modified directly. If
            `False`, the transformation will be applied to a copy of the data.
    """
    return _recreate_signal(signal_df, inplace) - signal_df.mean()


def linear_envelope(
    signal_df: pandas.DataFrame,
    critical_freqs: Union[float, Sequence[float]],
    sampling_frequency: int,
    order: int,
    filter_type: str = "butter",
    zero_lag: bool = True,
    cheby_param: Optional[float] = None,
    zero_center_: bool = True,
    inplace: bool = False,
) -> pandas.DataFrame:
    """Find the linear envelope of a signal.

    This function finds the linear envelope of the raw EMG signal by:
    1. (optionally) zero-centering each signal.
    2. Taking the `abs` of each value (full-wave rectification).
    3. Low-pass filtering the signal.

    Args:
        signal_df: a :py:class:`~pandas.DataFrame` with a different
            discrete-time signal in each of its columns.

        critical_freqs: passed along to :py:func:`digital_filter`.

        sampling_frequency: passed along to :py:func:`digital_filter`.

        order: passed along to :py:func:`digital_filter`.

        filter_type: passed along to :py:func:`digital_filter`.

        zero_lag: passed along to :py:func:`digital_filter`.

        cheby_param: passed along to :py:func:`digital_filter`.

        zero_center_: if `True`, zero-center the data before taking its
            absolute value.

        inplace: if `True`, the data in the original
            :py:class:`~pandas.DataFrame` will be modified directly. If
            `False`, the transformations will be applied to a copy of the data.
    """
    if zero_center_:
        signal_df = zero_center(signal_df, inplace=inplace)
    if inplace:
        signal_df[:] = signal_df.abs()
    else:
        signal_df = signal_df.abs()

    return digital_filter(
        signal_df=signal_df,
        critical_freqs=critical_freqs,
        sampling_frequency=sampling_frequency,
        order=order,
        filter_type=filter_type,
        band_type="lowpass",
        zero_lag=zero_lag,
        cheby_param=cheby_param,
        inplace=inplace,
    )


def digital_filter(
    signal_df: pandas.DataFrame,
    critical_freqs: Union[float, Sequence[float]],
    sampling_frequency: int,
    order: int,
    filter_type: str = "butter",
    band_type: str = "lowpass",
    zero_lag: bool = True,
    cheby_param: Optional[float] = None,
    inplace: bool = False,
) -> pandas.DataFrame:
    """Apply digital filter to signal.

    This function implements Butterworth filters as well as Chebyshev (both
    type I and type II) ones. The filters can be high-pass, low-pass,
    band-pass or band-stop and of any order.

    For more complex use cases, refer to :py:mod:`scipy.signal`.

    Args:
        signal_df: a :py:class:`~pandas.DataFrame` with a different
            discrete-time signal in each of its columns.

        order: the order of the filter.

        critical_freqs: the critical frequency. Either a sequence of 2 values
            (if `band_type` is either `"bandpass"` or `"bandstop"`) or a single
            value (if `band_type` is either `"highpass"` or `"lowpass"`).
            Assumed to have the same units as `sampling_frequency`.

        filter_type: one of `"butter"` (Butterworth filter), `"cheby1"`
            (Chebyshev type I filter) or `"cheby2"` (Chebyshev type II filter).

        band_type: one of `"lowpass"`, `"highpass"`, `"bandpass"` or
            `"bandstop"`.

        sampling_frequency: the sampling rate with which measurements were
            made.

        zero_lag: if `True`, apply the filter both forward and backward along
            the signal. Otherwise, apply it forward only.

        cheby_param: the maximum ripple allowed below unity gain in the
            passband if `filter_type` is `"cheby1"`.  If it is `"cheby2"`, the
            minimum attenuation required in the stop band.  In both cases, this
            should be given in decibels as a positive number.  If `filter_type`
            is `"butter"` this argument is ignored.

        inplace: if `True`, the data in the original
            :py:class:`~pandas.DataFrame` will be modified directly. If
            `False`, the transformation will be applied to a copy of the data.

    Returns:
        a new :py:class:`~pandas.DataFrame` holding the result of the filter
        application. This :py:class:`~pandas.DataFrame` will have the same
        columns and index as the original one. If `inplace` is True, the
        original :py:class:`~pandas.DataFrame` will be returned with its data
        modified.
    """

    def filter_coeffs(
        filter_type: str,
        order: int,
        sampling_frequency: int,
        critical_freqs: Union[Sequence[float], float],
        band_type: str = "lowpass",
        cheby_param: Optional[float] = None,
    ) -> np.ndarray:
        """Determine filter coefficients."""
        if filter_type == "butter":
            return signal.butter(
                order,
                critical_freqs,
                btype=band_type,
                output="sos",
                fs=sampling_frequency,
            )
        elif filter_type == "cheby1":
            coeff_func = signal.cheby1
        elif filter_type == "cheby2":
            coeff_func = signal.cheby2
        return coeff_func(
            order,
            cheby_param,
            critical_freqs,
            btype=band_type,
            output="sos",
            fs=sampling_frequency,
        )

    def apply_filter(
        signal_df: pandas.DataFrame, coeffs: np.ndarray, zero_lag: bool, inplace: bool
    ) -> pandas.DataFrame:
        """Apply digital filter to signal."""
        if zero_lag:
            filt_func = signal.sosfiltfilt
        else:
            filt_func = signal.sosfilt

        signal_arr = signal_df.to_numpy()
        filtered_arr = filt_func(coeffs, signal_arr, axis=0)
        return _recreate_signal(signal_df, inplace, filtered_arr)

    if filter_type not in {"butter", "cheby1", "cheby2"}:
        raise ValueError("filter type not understood.")

    coeffs = filter_coeffs(
        filter_type,
        order,
        sampling_frequency,
        critical_freqs,
        band_type,
        cheby_param,
    )

    return apply_filter(signal_df, coeffs, zero_lag, inplace)


def rms(
    signal_df: pandas.DataFrame,
    window_size: Union[int, float],
    inplace: bool = False,
    sampling_frequency: Optional[int] = None,
) -> pandas.DataFrame:
    """Find the RMS of a signal.

    The RMS of a signal is given by sliding a window across it and taking its
    RMS (square root of the mean of the squares). The i-th entry of the output
    will be given by the RMS of the i-th window. This function only supports a
    stride of exactly 1 and will always output a signal with the same shape as
    the original one.

    Args:
        signal_df: a :py:class:`~pandas.DataFrame` with a different
            discrete-time signal in each of its columns.

        window_size: if `sampling_frequency` is not provided, the window size
            is the number of adjacent measurements that will be used to obtain
            a single entry in the output. If `sampling_frequency` is provided,
            the window size should be given in units of time. The number of
            elements in each window will then be `window_size *
            sampling_frequency`. For example, if the given sampling rate is
            `10` Hz, the period will be 0.1 s. If the window size is `1` s,
            each window will contain 10 array entries.

        inplace: if `True`, the data in the original
            :py:class:`~pandas.DataFrame` will be modified directly. If
            `False`, a new array will be returned.

        sampling_frequency: the sampling rate with which measurements were
            made.

    Returns:
        a new signal with the same shape, column labels and index as the
        original one containing its RMS.
    """

    def single_channel_rms(signal_arr: np.ndarray, window_size: int) -> np.ndarray:
        """Find the RMS of a 1D digital signal.

        Args:
            signal_arr: a 1D array with successive measurements.

            window_size: the number of adjacent elements that should go into
                each window.
        """
        # inspiration
        # https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal
        square = signal_arr ** 2
        window_mean_factor = 1 / float(window_size)
        window = window_mean_factor * np.ones(window_size)
        return np.sqrt(np.convolve(square, window, "same"))

    def window_size_in_num_entries(
        window_size: Union[int, float], sampling_frequency: Optional[int]
    ) -> int:
        """Ensure window size is given in units of number of array elements."""
        if sampling_frequency is not None:
            return window_size * sampling_frequency
        return window_size

    window_size = window_size_in_num_entries(window_size, sampling_frequency)
    fixed_window_rms = functools.partial(single_channel_rms, window_size=window_size)
    rms_arr = np.apply_along_axis(fixed_window_rms, 0, signal_df)
    return _recreate_signal(signal_df, inplace, rms_arr)


def normalize(signal_df: pandas.DataFrame, inplace: bool = False) -> pandas.DataFrame:
    """Divide each column by its max absolute value.

    Args:
        signal_df: a :py:class:`~pandas.DataFrame` with a different
            discrete-time signal in each of its columns.

        inplace: if `True`, the data in the original
            :py:class:`~pandas.DataFrame` will be modified directly. If
            `False`, the transformation will be applied to a copy of the data.
    """
    signal_df = _recreate_signal(signal_df, inplace)
    return signal_df / abs(signal_df).max()


def subsample(
    signal_df: pandas.DataFrame,
    keep_every: Optional[int] = None,
) -> pandas.DataFrame:
    """Reduce number of measurements by keeping only subset of measurements.

    Args:
        signal_df: a :py:class:`~pandas.DataFrame` with a different
            discrete-time signal in each of its columns.

        keep_every: every i-th frame is kept, where i is given by this
            parameter.

    Returns:
        a new :py:class:`~pandas.DataFrame` containing only every `keep_every`
        row.

    See also:
        :py:func:`time_normalize`
    """
    return signal_df.iloc[0:keep_every:, ...]


def time_normalize(
    signal_df: pandas.DataFrame,
    reduce_to: int,
    kind: Optional[Union[int, str]] = "linear",
    fill_value="extrapolate",
) -> pandas.DataFrame:
    """Express signal in terms of normalized time.

    Args:
        signal_df: a :py:class:`~pandas.DataFrame` with a different
            discrete-time signal in each of its columns.

        reduce_to: the desired number of rows in the output signal.

        kind: determines the kind of interpolation to be performed. Passed
            directly to :py:func:`scipy.interpolate.interp1d`.

        fill_value: determines how values outside of the signal should be
            handled. This would only be relevant in case `reduce_to` is bigger
            than the number of rows of `signal_df`. Passed directly to
            :py:func:`scipy.interpolate.interp1d`.

    Returns:
        a new :py:class:`~pandas.DataFrame` with the time-normalized data. Its
        column labels are the same as the original one, its index is a linear
        range from 0 to 1 with `reduce_to` values.

    See also:
        :py:func:`subsample`
    """
    signal_len = signal_df.shape[0]
    percent_domain = np.linspace(0, 1, signal_len)
    interp_func = interpolate.interp1d(
        percent_domain, signal_df, copy=False, kind=kind, fill_value=fill_value
    )
    desired_domain = np.linspace(0, 1, reduce_to)
    return pandas.DataFrame(
        interp_func(signal_df), index=desired_domain, columns=signal_df.columns
    )


def vaf(
    original_df: pandas.DataFrame,
    transformed_signal: Optional[_NUMPY_ARRAY_LIKE] = None,
    components: Optional[_NUMPY_ARRAY_LIKE] = None,
    reconstructed_signal: Optional[_NUMPY_ARRAY_LIKE] = None,
) -> pandas.DataFrame:
    """Calculate VAF between reconstructed and original signal.

    The VAF ("variance accounted for") is given by:

    :math::`\text{VAF} = 1 - \frac{ \| (x - x_r) \|^2}{\| x \|^2}`

    Where the norm is the Frobenius norm, :math:`x` is the original signal and
    :math:`x_r` is the version of the signal reconstructed using the synergy
    components. In the case of this function, :math:`x_r` will be given by
    `transformed_df @ components`.

    Args:
        original_df: an array of shape `(num_measurements, num_muscles)`.  Must
            be a :py:class:`~pandas.DataFrame` because that's where the column
            labels for the output come form. This is the original version of
            the signal (the already processed EMG signal) which was used to
            find the synergies.

        transformed_signal: an array of shape `(num_measurements,
            num_synergies)`. This is the version of the signal expressed in
            terms of the synergy components. If provided, then `components`
            should also be and `reconstructed_signal` must not be present.

        components: a :py:class:`~pandas.DataFrame` of shape `(num_synergies,
            num_muscles)`. These are the synergy components. If provided, then
            `transformed_signal` should also be and `reconstructed_signal` must
            not be present.

        reconstructed_signal: the reconstructed signal :math:`x_r`. If
            provided, then none of `transformed_signal` and `components` should
            be present.

    Returns:
        a :py:class:`~pandas.DataFrame` with shape `(1, 1 + num_muscles)`. The
        first column contains the VAF with respect to the whole signal and the
        other columns contain the VAF for individual columns (muscles).
    """

    def sum_of_squares(
        arr: _NUMPY_ARRAY_LIKE, axis: Optional[int]
    ) -> Union[float, np.ndarray]:
        """Compute the sum of squares of an array."""
        return np.sum(arr ** 2, axis=axis)

    def vaf_along_axis(
        original_signal: _NUMPY_ARRAY_LIKE,
        error: _NUMPY_ARRAY_LIKE,
        axis: Optional[int],
    ) -> Union[float, np.ndarray]:
        """Find the VAF along a given axis."""
        return 1 - sum_of_squares(error, axis) / sum_of_squares(
            original_signal, axis=axis
        )

    if reconstructed_signal is None:
        reconstructed_signal = transformed_signal @ components
    error = original_df - reconstructed_signal
    vaf_all_signals = vaf_along_axis(original_df, error, axis=None)
    vaf_per_column = vaf_along_axis(original_df, error, axis=1)
    vaf_values = [vaf_all_signals] + list(vaf_per_column.reshape(-1))
    column_labels = ["All signals"] + original_df.columns
    return pandas.DataFrame(vaf_values, columns=column_labels)


@dataclass
class SynergyRunResult:
    """The result of a synergy run.

    :py:class:`SynergyRunResult` holds the result of matrix factorizations
    performed to find muscle synergies. When finding muscle synergies, one of
    the key parameters is the number of synergy components to look for. For
    that reason, an user may want to factor the matrix using, say, both 2 and 3
    synergy components and compare their VAFs to see how well they explain the
    variance in the original signal. So this class can either hold the results
    of a run with a single fixed number of synergy components, like 2, or the
    results of one with several possible number of components.

    Attributes:
        vaf_values: the VAF for all muscles as well as for each individual one.
            If several runs are made with different number of components in
            each, then each row will correspond to a different number of
            components. The number of components will be the index of the
            :py:class:`~pandas.DataFrame`.

        components: the synergy components, one per row. Each column correspond
            to a different muscle. If several runs are made each with a
            different number of components, then this will be a `dict` mapping
            from the `int` number of components to its corresponding
            :py:class:`~pandas.DataFrame`.

        model: the :py:class:`sklearn.decomposition.NMF` used to decompose the
            matrix. If several runs are made each with a different number of
            components, then this will be a `dict` mapping from the `int`
            number of components to its corresponding
            :py:class:`~pandas.DataFrame`.
    """

    vaf_values: pandas.DataFrame
    components: Union[pandas.DataFrame, Mapping[int, pandas.DataFrame]]
    model: Union[NMF, Mapping[int, NMF]]


def find_synergies(
    processed_emg_df: pandas.DataFrame,
    n_components: int,
    max_components: Optional[int] = None,
    *,
    max_iter: int = 100_000,
    tol: float = 1e-6,
    **sklearn_kwargs,
) -> SynergyRunResult:
    """Find spatial synergy components in processed EMG signal.

    Find synergy components that explain well the variance in the processed EMG
    signal (given by `processed_emg_df`). The method used is non-negative
    matrix factorization (specifically the implementation in
    :py:class:`sklearn.decomposition.NMF`). This method factorizes a
    non-negative matrix into non-negative factors. So the `processed_emg_df`
    argument may contain no negative entries. The non-negative factors are
    found through optimization and there is no guarantee that 2 different runs
    will yield the same synergy components unless the random seed is specified
    (though this doesn't seem to be a problem in practice).

    The implementation here follows the notation in
    :py:class:`sklearn.decomposition.NMF`, which differs from the one used in
    many papers (but not all) which use this method to find muscle synergies.
    :py:func:`find_synergies` expects one muscle per column. In particular:
    + the `processed_emg_signal` has shape `(num_measurements, num_muscles)`.
    + the `transformed_signal` has shape `(num_measurements, num_components)`.
    + the `synergy_components` have shape `(num_components, num_muscles)`.

    The original `processed_emg_signal` is approximated by the
    `reconstructed_signal`, which is obtained by the matrix product
    `transformed_signal @ synergy_components`. The synergy components are the
    rows of `synergy_components`.

    This function supports 2 use cases:
    + looking for a specific number of synergy components. In this case,
      `n_components` should contain that number and `max_components` should be
      `None`.
    + Looking for a range of synergy components. This may be useful when
      looking for the minimum number required to achieve a desired VAF. In this
      case, `n_components` should be the minimum number of components and
      `max_components` should be the maximum number.

    The other arguments to this function are passed along to
    :py:class:`sklearn.decomposition.NMF` as keyword arguments, including
    `max_iter` and `tol`. Except for those 2, the user should look refer
    directly to the documentation for :py:class:`sklearn.decomposition.NMF` to
    understand what arguments are available and what they do.  One
    :py:class:`sklearn.decomposition.NMF` is created for each solution. That
    is, there will be 1 if `max_components is None` and more otherwise.  Each
    of these :py:class:`~sklearn.decomposition.NMF` instances is returned to
    the user and can be inspected to investigate the convergence of the fit.
    The VAFs and the synergy components are also returned.

    Args:
        processed_emg_df: a `(num_measurements, num_muscles)` array with the
            signal from which to determine the synergy components.

        n_components: if `max_components` is provided, the minimum number of
            components to be solved for. Otherwise, the number of components to
            be solved for.

        max_components: if not `None`, the maximum number of components to be
            solved for. If this is `None`, a single solution containing exactly
            `n_components` will be searched.

        max_iter: the maximum number of iterations to be used in the
            optimization process.

        tol: the tolerance of the stopping condition of the optimization
            process. With the default :py:class:`sklearn.decomposition.NMF`
            settings, the objective function is :math:`\frac{1}{2} \| x - x_r
            \|^2`. The norm is the Frobenius norm, :math:`x` is the original
            signal (`processed_emg_signal`) and :math:`x_r` is the
            reconstructed signal.

        sklearn_kwargs: keyword arguments passed along to each created instance
            of :py:class:`sklearn.decomposition.NMF`.

    Raises:
        ValueError if the number of synergies fall outside their given range.
        `num_features >= max_components >= n_components >= 1`.  If
        `max_components is None`, then the requirement simplifies to
        `num_features >= n_components >= 1`. `num_features` in these equations
        is simply the number of muscles, determined as the number of columns of
        `processed_emg_df`.

    Returns:
        a :py:class:`SynergyRunResult`. In case `max_components` was not
        provided, its `model` member will hold the
        :py:class:`~sklearn.decomposition.NMF` model and its `components`
        member will contain a :py:class:`~pandas.DataFrame` with one synergy
        component per row. Its column labels will be the same as the ones of
        `processed_emg_df`.

        In case `max_components` is provided, both members will instead be
        `dicts` mapping from the number of components to the different
        :py:class:`~sklearn.decomposition.NMF` models and
        :py:class:`~pandas.DataFrame`s with synergy components.

        The `vaf_values` member will contain a :py:class:`~pandas.DataFrame` in
        both cases and the number of rows will correspond to the number of
        synergy runs. For example, if `n_components == 2` and `max_components`
        was not provided, it will have a single row. If `max_components == 3`,
        it will have 2 rows, the first one containing the VAFs of the
        decomposition with 2 synergy components and the second one the ones for
        3 components. The columns of this :py:class:`~pandas.DataFrame` are the
        ones returned by :py:func:`vaf`.
    """

    def validate_num_components(
        processed_emg_df: pandas.DataFrame,
        n_components: int,
        max_components: Optional[int],
    ):
        """Validate the number of synergy components."""
        if processed_emg_df.empty:
            raise ValueError("empty EMG DataFrame")

        num_features = len(processed_emg_df.columns)

        error_msg = "invalid number of components"
        if n_components < 1 or n_components > num_features:
            raise ValueError(error_msg)

        if max_components is not None:
            if max_components < n_components or max_components > num_features:
                raise ValueError(error_msg)

    def nnmf(
        matrix: _NUMPY_ARRAY_LIKE,
        n_components: int,
        **sklearn_kwargs,
    ) -> Tuple[np.ndarray, NMF]:
        """Factorize non-negative matrix into non-negative factors.

        Returns:
             a tuple `(reconstructed_signal, model)` with the version of the
             original signal at each instant expressed as a linear combination
             of the synergy components and the
             :py:class:`sklearn.decomposition.NMF` model used to look for the
             components.
        """
        model = NMF(n_components=n_components, **sklearn_kwargs)
        reconstructed_signal = model.fit_transform(matrix)
        return reconstructed_signal, model

    def single_synergy_run(
        processed_emg_df: pandas.DataFrame, n_components: int, **sklearn_kwargs
    ) -> SynergyRunResult:
        """Find synergies and determine their VAF."""
        reconstructed_signal, model = nnmf(
            processed_emg_df, n_components, **sklearn_kwargs
        )
        vaf_values = vaf(processed_emg_df, reconstructed_signal=reconstructed_signal)
        comps = pandas.DataFrame(
            model.components_, columns=processed_emg_df.columns, index=[n_components]
        )
        return SynergyRunResult(vaf_values, comps, model)

    def merge_run_results(
        run_results: Mapping[int, SynergyRunResult]
    ) -> SynergyRunResult:
        """Merge different SynergyRunResult objects into a single one."""
        vaf_values = pandas.concat([res.vaf_values for res in run_results.values()])
        vaf_values.set_index(run_results.keys())
        comps = {n_comp: res.components for (n_comp, res) in run_results.items()}
        models = {n_comp: res.model for (n_comp, res) in run_results.items()}
        return SynergyRunResult(vaf_values, comps, models)

    validate_num_components(processed_emg_df, n_components, max_components)

    if max_components is None:
        return single_synergy_run(
            processed_emg_df, n_components, max_iter=max_iter, tol=tol, **sklearn_kwargs
        )

    run_results = OrderedDict()

    for n in range(n_components, max_components + 1):
        run_results[n] = single_synergy_run(
            processed_emg_df, n, max_iter=max_iter, tol=tol, **sklearn_kwargs
        )

    return merge_run_results(run_results)
