"""
This module contains various models for resonators that are subclasses of the lmfit.model.Model class.


"""
from __future__ import absolute_import, division, print_function

import numpy as np
import lmfit

from . import fitter


class Shunt(lmfit.model.Model):
    """
    This class models a resonator operated in the shunt ("hanger") configuration.
    """

    def __init__(self, *args, **kwargs):
        def func(frequency, resonance_frequency, internal_loss, coupling_loss, asymmetry):
            detuning = frequency / resonance_frequency - 1
            return 1 - ((1 + 1j * asymmetry) /
                        (1 + (internal_loss + 2j * detuning) / coupling_loss))
        super(Shunt, self).__init__(func=func, *args, **kwargs)

    def guess(self, data=None, frequency=None, **kwargs):
        resonance_frequency_guess = frequency[np.argmin(np.abs(data))]  # guess that the resonance is the lowest point
        width = frequency.size // 10
        gaussian = np.exp(-np.linspace(-4, 4, width) ** 2)
        gaussian /= np.sum(gaussian)  # not necessary
        smoothed = np.convolve(gaussian, abs(data), mode='same')
        derivative = np.convolve(np.array([1, -1]), smoothed, mode='same')
        # Exclude the edges, which are affected by zero padding.
        linewidth = (frequency[np.argmax(derivative[width:-width])] -  # Removed factor of 1/2
                     frequency[np.argmin(derivative[width:-width])])
        internal_plus_coupling = linewidth / resonance_frequency_guess
        internal_over_coupling = 1 / (1 / np.min(np.abs(data)) - 1)
        internal_loss_guess = internal_plus_coupling * internal_over_coupling / (1 + internal_over_coupling)
        coupling_loss_guess = internal_plus_coupling / (1 + internal_over_coupling)
        params = self.make_params(resonance_frequency=resonance_frequency_guess, internal_loss=internal_loss_guess,
                                  coupling_loss=coupling_loss_guess, asymmetry=0)
        params['{}resonance_frequency'.format(self.prefix)].set(min=frequency.min(), max=frequency.max())
        params['{}internal_loss'.format(self.prefix)].set(min=1e-12, max=1)
        params['{}coupling_loss'.format(self.prefix)].set(min=1e-12, max=1)
        params['{}asymmetry'.format(self.prefix)].set(min=-10, max=10)
        return params


class ShuntFitter(fitter.ResonatorFitter):

    def __init__(self, frequency, data, errors=None, **kwargs):
        super(ShuntFitter, self).__init__(model=Shunt, frequency=frequency, data=data,
                                          errors=errors, **kwargs)

    def invert(self, time_ordered_data):
        z = self.coupling_loss * ((1 + 1j * self.asymmetry) / (1 - time_ordered_data) - 1)
        detuning = z.imag / 2
        internal_loss = z.real
        return detuning, internal_loss

    # ToDo: decide whether to keep these aliases
    @property
    def f_r(self):
        return self.resonance_frequency

    @property
    def Q_r(self):
        return 1 / (self.internal_loss + self.coupling_loss)

    @property
    def Q_i(self):
        return 1 / self.internal_loss

    @property
    def Q_c(self):
        return 1 / self.coupling_loss
