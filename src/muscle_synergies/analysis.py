"""Signal analysis functions.

Very hacky for the time being because I'm still figuring out how this analysis
works.
"""

from collections import defaultdict
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


def positive_spectrum(signal_df, sampling_frequency):
    """Find the spectrum corresponding to frequencies."""
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


def butterworth_filter(
    data, critical_freqs, sampling_freq, order, filter_type="lowpass", zero_lag=True
):
    """Apply Butterworth filter to the data.

    Modified from this example
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """

    filter_coeffs = _filter_coeffs(order, sampling_freq, critical_freqs, filter_type)

    if zero_lag:
        filt_func = signal.sosfiltfilt
    else:
        filt_func = signal.sosfilt

    return filt_func(filter_coeffs, data, axis=0)


def _filter_coeffs(
    order: int, sampling_freq: int, critical_freqs, filter_type="lowpass"
):
    """Determine Butterworth filter coefficients."""
    return signal.butter(
        order, critical_freqs, btype=filter_type, output="sos", fs=sampling_freq
    )


def butterworth_filter(
    data, critical_freqs, sampling_freq, order, filter_type="lowpass", zero_lag=True
):
    """Apply Butterworth filter to the data."""
    filter_coeffs = _filter_coeffs(order, sampling_freq, critical_freqs, filter_type)

    if zero_lag:
        filt_func = signal.sosfiltfilt
    else:
        filt_func = signal.sosfilt

    return filt_func(filter_coeffs, data, axis=0)


def _filter_coeffs(
    order: int, sampling_freq: int, critical_freqs, filter_type="lowpass"
):
    """Determine Butterworth filter coefficients."""
    return signal.butter(
        order, critical_freqs, btype=filter_type, output="sos", fs=sampling_freq
    )


def nnmf(matrix_df, num_components, *, max_iter=100_000, tol=1e-6):
    model = NMF(max_iter=100000, tol=1e-6, n_components=num_components)
    transformed_emg_signal = model.fit_transform(matrix_df)
    return transformed_emg_signal, model.components_, model.n_iter_


def vaf(original_df, transformed_df, components):
    error = original_df - transformed_df @ components
    return 1 - np.linalg.norm(error, ord=2) / np.linalg.norm(original_df, ord=2)


def find_synergies(
    processed_emg_df, min_components=2, max_components=None, max_iter=100_000, tol=1e-6
):
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
    num_synergies = components.shape[0]
    synergy_names = [f'synergy {i}' for i in range(1, num_synergies + 1)]
    synergies = pandas.DataFrame(components, index=synergy_names, columns=columns)
    sns.heatmap(synergies, annot=True, fmt=".2f")
    plt.title("Heatmap of muscle synergies")
    return plt.gca()
