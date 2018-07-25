"""
This module contains models and fitters for resonators that are operated in the reflection configuration.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import lmfit

from . import background, fitter


class Reflection(lmfit.model.Model):
    """
    This class models a resonator operated in reflection.
    """
    reference_point = -1 + 0j

    def __init__(self, *args, **kwds):
        """
        Instantiate.

        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keyword arguments passed directly to lmfit.model.Model.__init__().
        """
        def func(frequency, resonance_frequency, internal_loss, coupling_loss):
            detuning = frequency / resonance_frequency - 1
            return -1 + (2 / (1 + (internal_loss + 2j * detuning) / coupling_loss))
        super(Reflection, self).__init__(func=func, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        params = self.make_params()
        width = frequency.size // 10
        gaussian = np.exp(-np.linspace(-4, 4, width) ** 2)
        gaussian /= np.sum(gaussian)
        smoothed = np.convolve(gaussian, data, mode='same')  # This array has the same number of points as the data
        # The edges are corrupted by zero-padding, so set them equal to non-corrupted points
        smoothed[:width] = smoothed[width]
        smoothed[-width:] = smoothed[-(width + 1)]
        resonance_index = np.argmax(smoothed.real)
        resonance_frequency_guess = frequency[resonance_index]
        params['resonance_frequency'].set(value=resonance_frequency_guess, min=frequency.min(), max=frequency.max())
        linewidth = frequency[np.argmin(smoothed.imag)] - frequency[np.argmax(smoothed.imag)]
        internal_plus_coupling = linewidth / resonance_frequency_guess
        internal_over_coupling = 2 / (np.abs(smoothed[resonance_index]) + 1) - 1
        params['internal_loss'].set(value=internal_plus_coupling / (1 + 1 / internal_over_coupling), min=1e-12, max=1)
        params['coupling_loss'].set(value=internal_plus_coupling / (1 + internal_over_coupling), min=1e-12, max=1)
        return params


class ReflectionFitter(fitter.ResonatorFitter):

    def __init__(self, frequency, data, background_model=None, errors=None, **kwds):
        if background_model is None:
            background_model = background.ComplexConstant()
        super(ReflectionFitter, self).__init__(frequency=frequency, data=data, foreground_model=Reflection(),
                                               background_model=background_model, errors=errors, **kwds)

    def invert(self, time_ordered_data):
        z = self.coupling_loss * (2 / (1 + time_ordered_data) - 1)
        detuning = z.imag / 2
        internal_loss = z.real
        return detuning, internal_loss




