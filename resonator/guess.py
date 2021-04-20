"""
Functions for generating initial parameters used by model `guess` methods.
"""
from __future__ import absolute_import, division, print_function

import numpy as np


def smooth(data, fraction=0.05, flatten_edges=True):
    width = int(fraction * data.size)
    if width:
        gaussian = np.exp(-np.linspace(-4, 4, width) ** 2)
        gaussian /= np.sum(gaussian)
        smoothed = np.convolve(gaussian, data, mode='same')
        if flatten_edges:
            smoothed[:width] = smoothed[width]
            smoothed[-width:] = smoothed[-(width + 1)]
        return smoothed
    else:
        return data


def distances(data, pad_ends=True):
    """
    Return an array containing the sum of the nearest-neighbor distances of the given data.

    :param data: complex scattering parameter data.
    :param pad_ends: if True, duplicate the end values so that the returned array has the same size as the data.
    :return: array[float]
    """
    d = (np.sqrt((data.real[1:-1] - data.real[:-2]) ** 2 + (data.imag[1:-1] - data.imag[:-2]) ** 2)
         + np.sqrt((data.real[2:] - data.real[1:-1]) ** 2 + (data.imag[2:] - data.imag[1:-1]) ** 2))
    if pad_ends:
        d = np.concatenate((d[:1], d, d[-1:]))
    return d


def distances_per_frequency(frequency, data, pad_ends=True):
    """
    Return an array containing the sum of the nearest-neighbor distances of the given data divided by the sum of the
    nearest-neighbor frequency differences. This should give the same result as `distances` for data points that are
    equally spaced in frequency, but will give more relevant results for irregularly-spaced data.

    :param data: complex scattering parameter data.
    :param frequency: the frequencies at which the data was collected.
    :param pad_ends: if True, duplicate the end values so that the returned array has the same size as the data.
    :return: array[float]
    """
    d = distances(data=data, pad_ends=pad_ends)
    f = np.diff(frequency[:-1]) + np.diff(frequency[1:])
    if pad_ends:
        f = np.concatenate((f[:1], f, f[-1:]))
    return d / f


def smallest(values, fraction=0.1):
    return np.argsort(values)[:int(fraction * values.size)]


def largest(values, fraction=0.1):
    return np.argsort(values)[-int(fraction * values.size):]


def polyfit_phase_delay(frequency, data):
    # For a resonator in reflection, one of these will often follow the 2 pi phase wrap,
    # but the other will follow the actual electrical delay.
    poly_wrapped, res_wrapped, _, _, _ = np.polyfit(frequency, np.angle(data), 1, full=True)
    poly_unwrapped, res_unwrapped, _, _, _ = np.polyfit(frequency, np.unwrap(np.angle(data)), 1, full=True)
    if res_wrapped < res_unwrapped:
        phase_slope, phase_offset = poly_wrapped
    else:
        phase_slope, phase_offset = poly_unwrapped
    delay = phase_slope / (2 * np.pi)
    return phase_offset, delay


def polyfit_magnitude_slope_offset(frequency, data):
    slope, offset = np.polyfit(frequency, np.abs(data), 1)
    return slope, offset


def guess_smooth(frequency, data):
    smooth_data = smooth(data)
    resonance_frequency = np.median(frequency[largest(distances(smooth_data), fraction=0.1)])
    resonance_index = np.argmin(np.abs(frequency - resonance_frequency))
    linewidth = abs(frequency[np.argmin(smooth_data.imag)] - frequency[np.argmax(smooth_data.imag)])
    internal_plus_coupling = linewidth / resonance_frequency
    internal_over_coupling = 2 / (np.abs(smooth_data[resonance_index]) + 1) - 1
    coupling_loss = internal_plus_coupling / (1 + internal_over_coupling)
    internal_loss = internal_plus_coupling / (1 + 1 / internal_over_coupling)
    return resonance_frequency, coupling_loss, internal_loss
