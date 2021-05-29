"""Signal analysis functions.

Very hacky for the time being because I'm still figuring out how this analysis
works.
"""

from collections import defaultdict
import functools
import matplotlib.pyplot as plt

plt.style.use("bmh")
import numpy as np
import pandas
from scipy.fftpack import fft, fftfreq
import scipy.signal as signal
import seaborn as sns
from sklearn.decomposition import NMF


def plot_emg_signal(
    emg_df,
    title,
    plot_dims=(2, 4),
    xlabel="time (s)",
    ylabel="Volts",
    xticks_off=False,
    figsize=(18, 10),
):
    """Plot EMG DataFrame with 8 columns."""
    assert len(emg_df.columns) == np.prod(plot_dims)
    fig, axs = plt.subplots(plot_dims[0], plot_dims[1], figsize=figsize)
    if len(axs.shape) == 1:
        axs = np.expand_dims(axs, axis=1)
    for (ax, col) in zip(axs.flat, emg_df.columns):
        emg_df[col].plot(ax=ax)
        ax.set_title(col)
        if xticks_off:
            ax.set_xticks([], [])
        ax.set(xlabel=xlabel)
    fig.suptitle(title, fontsize=20)
    axs[0, 0].set_ylabel(ylabel)
    axs[1, 0].set_ylabel(ylabel)
    plt.show()


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


def plot_spectrum(
    signal_df,
    sampling_frequency,
    plot_dims=(2, 4),
    title="EMG spectrum",
    xlabel="frequency (Hz)",
    ylabel="magnitude (V)",
    figsize=(18, 10),
):
    """Plot spectrum of signal"""
    spectrum_df = positive_spectrum(signal_df, sampling_frequency)
    plot_emg_signal(
        spectrum_df,
        plot_dims=plot_dims,
        title="EMG spectrum",
        xlabel="frequency (Hz)",
        ylabel="magnitude (V)",
        xticks_off=False,
        figsize=figsize,
    )


def digital_filter(
    signal_df: pandas.DataFrame,
    critical_freqs: Union[float, Sequence[float]],
    sampling_frequency: float,
    order: int,
    filter_type: str = "butter",
    band_type: str = "lowpass",
    zero_lag: bool = True,
    cheby_param: Optional[float] = None,
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

    Returns:
        a new :py:class:`~pandas.DataFrame` holding the result of the filter
        application. This :py:class:`~pandas.DataFrame` will have the same
        columns and index as the original one.
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
        signal_df: pandas.DataFrame, coeffs: np.ndarray, zero_lag: bool
    ) -> pandas.DataFrame:
        """Apply digital filter to signal."""
        if zero_lag:
            filt_func = signal.sosfiltfilt
        else:
            filt_func = signal.sosfilt

        signal_arr = signal_df.to_numpy()
        filtered_arr = filt_func(coeffs, signal_arr, axis=0)
        return pandas.DataFrame(
            filtered_arr, columns=signal_df.columns, index=signal_df.index
        )

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

    return apply_filter(signal_df, coeffs, zero_lag)


def _single_channel_rms(signal: np.ndarray, window_size: int):
    """Find the RMS of a 1D digital signal.

    Inspired by this example: https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal
    """
    square = signal ** 2
    window_mean_factor = 1 / float(window_size)
    window = window_mean_factor * np.ones(window_size)
    return np.sqrt(np.convolve(square, window, 'same'))

def rms_envelope(emg_signal, window_size: int, inplace=False):
    """Find the RMS of a signal composed of multiple 1D channels."""
    emg_signal = pandas.DataFrame(emg_signal, copy=not inplace)
    fixed_window_rms = functools.partial(_single_channel_rms, window_size=window_size)
    emg_signal[:] = np.apply_along_axis(fixed_window_rms, 0, emg_signal)
    return emg_signal

def nnmf(matrix_df, num_components, *, max_iter=100_000, tol=1e-6):
    """Factor matrix into non-negative factors."""
    model = NMF(max_iter=100000, tol=1e-6, n_components=num_components)
    transformed_emg_signal = model.fit_transform(matrix_df)
    return transformed_emg_signal, model.components_, model.n_iter_


def vaf(original_df, transformed_df, components):
    """Calculate VAF between reconstructed and original signal."""
    error = original_df - transformed_df @ components
    return 1 - np.linalg.norm(error, ord=2) / np.linalg.norm(original_df, ord=2)


def find_synergies(
    processed_emg_df, min_components=2, max_components=None, max_iter=100_000, tol=1e-6
):
    """Find synergy components in processed EMG signal."""
    num_features = len(processed_emg_df.columns)

    assert num_features > 0
    if max_components is None:
        max_components = num_features
    assert max_components >= 1 and max_components <= num_features
    assert min_components <= max_components and min_components >= 1

    vaf_df = defaultdict(list)
    components_dict = {}

    for num_components in range(min_components, max_components + 1):
        transformed_df, components, n_iter = nnmf(
            processed_emg_df, num_components, max_iter=max_iter, tol=tol
        )
        synergies_vaf = vaf(processed_emg_df, transformed_df, components)
        vaf_df["num_components"].append(num_components)
        vaf_df["vaf"].append(synergies_vaf)
        vaf_df["n_iter"].append(n_iter)
        components_dict[num_components] = components

    vaf_df = pandas.DataFrame(vaf_df)
    vaf_df = vaf_df.set_index("num_components")
    return vaf_df, components_dict


def synergy_heatmap(components, columns):
    """Plot synergy heatmap."""
    num_synergies = components.shape[0]
    synergy_names = [f"synergy {i}" for i in range(1, num_synergies + 1)]
    synergies = pandas.DataFrame(components, index=synergy_names, columns=columns)
    sns.heatmap(synergies, annot=True, fmt=".2f")
    plt.title("Heatmap of muscle synergies")
    return plt.gca()
