"""
This module contains models and fitters for resonators that are operated in the shunt ("hanger") configuration.
"""
from __future__ import absolute_import, division, print_function

import lmfit
import numpy as np

from . import background, base


class Shunt(lmfit.model.Model):
    """
    This class models a resonator operated in the shunt ("hanger") configuration.
    """
    reference_point = 1 + 0j

    def __init__(self, *args, **kwds):
        """
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """
        def shunt(frequency, resonance_frequency, internal_loss, coupling_loss, asymmetry):
            detuning = frequency / resonance_frequency - 1
            return 1 - ((1 + 1j * asymmetry) /
                        (1 + (internal_loss + 2j * detuning) / coupling_loss))
        super(Shunt, self).__init__(func=shunt, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        # ToDo: use the lowest point of the smoothed data, being careful of edges
        resonance_frequency_guess = frequency[np.argmin(np.abs(data))]
        width = frequency.size // 10
        gaussian = np.exp(-np.linspace(-4, 4, width) ** 2)
        gaussian /= np.sum(gaussian)
        smoothed = np.convolve(gaussian, np.abs(data), mode='same')
        derivative = np.convolve(np.array([1, -1]), smoothed, mode='same')
        # ToDo: investigate how well this is actually working -- for clean data it should calculate the right linewidth
        # Exclude the edges, which are affected by zero padding.
        linewidth = (frequency[np.argmax(derivative[width:-width])] -
                     frequency[np.argmin(derivative[width:-width])])
        internal_plus_coupling = linewidth / resonance_frequency_guess
        internal_over_coupling = 1 / (1 / np.min(np.abs(data)) - 1)
        internal_loss_guess = internal_plus_coupling * internal_over_coupling / (1 + internal_over_coupling)
        coupling_loss_guess = internal_plus_coupling / (1 + internal_over_coupling)
        params = self.make_params(resonance_frequency=resonance_frequency_guess, internal_loss=internal_loss_guess,
                                  coupling_loss=coupling_loss_guess, asymmetry=0)
        params['resonance_frequency'].set(min=frequency.min(), max=frequency.max())
        params['internal_loss'].set(min=1e-12, max=1)
        params['coupling_loss'].set(min=1e-12, max=1)
        params['asymmetry'].set(min=-10, max=10)
        return params


class ShuntFitter(base.ResonatorFitter):

    def __init__(self, frequency, data, background_model=None, errors=None, **kwds):
        """
        Fit the given data to a composite model that is the product of a background response model and the Shunt model.

        :param frequency: an array of floats containing the frequencies at which the data was measured.
        :param data: an array of complex numbers containing the data.
        :param background_model: an instance (not the class) of a model representing the background response without the
          resonator; the default of background.ComplexConstant assumes that this is modeled well by a single complex
          constant at all frequencies.
        :param errors: an array of complex numbers containing the standard errors of the mean of the data points.
        :param kwds: keyword arguments passed directly to lmfit.model.Model.fit().
        """
        if background_model is None:
            background_model = background.ComplexConstant()
        super(ShuntFitter, self).__init__(frequency=frequency, data=data, foreground_model=Shunt(),
                                          background_model=background_model, errors=errors, **kwds)

    def invert(self, time_ordered_data):
        z = self.coupling_loss * ((1 + 1j * self.asymmetry) / (1 - time_ordered_data) - 1)
        detuning = z.imag / 2
        internal_loss = z.real
        return detuning, internal_loss

