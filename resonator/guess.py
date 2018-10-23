"""
Functions for generating initial parameters used by model `guess` methods.
"""
from __future__ import absolute_import, division, print_function

import numpy as np


def distances(data, pad_ends=True):
    """
    Return an array of the same size as the given data containing the sum of the nearest-neighbor distances.

    :param data: complex scattering parameter data.
    :param pad_ends: if True, duplicate the end values so that the returned array has the same size as the data.
    :return: array[float]
    """
    d = (np.sqrt((data.real[1:-1] - data.real[:-2]) ** 2 + (data.imag[1:-1] - data.imag[:-2]) ** 2)
         + np.sqrt((data.real[2:] - data.real[1:-1]) ** 2 + (data.imag[2:] - data.imag[1:-1]) ** 2))
    if pad_ends:
        d = np.concatenate((d[:1], d, d[-1:]))
    return d


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
