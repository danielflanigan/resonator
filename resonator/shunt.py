"""
This module contains models and fitters for resonators that are operated in the shunt ("hanger") configuration.
"""
from __future__ import absolute_import, division, print_function

import lmfit
import numpy as np

from . import background, base, nonlinear


# Models

class AbstractShunt(lmfit.model.Model):
    """
    This is an abstract class that models a resonator operated in the shunt-coupled configuration.
    """
    reference_point = 1 + 0j

    @staticmethod
    def guess_smooth(frequency, data):
        # ToDo: use the lowest point of the smoothed data, being careful of edges
        resonance_frequency = frequency[np.argmin(np.abs(data))]
        width = frequency.size // 10
        gaussian = np.exp(-np.linspace(-4, 4, width) ** 2)
        gaussian /= np.sum(gaussian)
        smoothed = np.convolve(gaussian, np.abs(data), mode='same')
        derivative = np.convolve(np.array([1, -1]), smoothed, mode='same')
        # ToDo: investigate how well this is actually working -- for clean data it should calculate the right linewidth
        # Exclude the edges, which are affected by zero padding.
        linewidth = (frequency[np.argmax(derivative[width:-width])] -
                     frequency[np.argmin(derivative[width:-width])])
        internal_plus_coupling = linewidth / resonance_frequency
        internal_over_coupling = 1 / (1 / np.min(np.abs(data)) - 1)
        coupling_loss = internal_plus_coupling / (1 + internal_over_coupling)
        internal_loss = internal_plus_coupling * internal_over_coupling / (1 + internal_over_coupling)
        return resonance_frequency, coupling_loss, internal_loss


class Shunt(AbstractShunt):
    """
    This class models a linear resonator operated in the shunt-coupled configuration.
    """

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
        resonance_frequency, coupling_loss, internal_loss = self.guess_smooth(frequency=frequency, data=data)
        params = self.make_params()
        params['resonance_frequency'].set(value=resonance_frequency, min=frequency.min(), max=frequency.max())
        params['coupling_loss'].set(value=coupling_loss, min=1e-12, max=1)
        params['internal_loss'].set(value=internal_loss, min=1e-12, max=1)
        params['asymmetry'].set(value=0, min=-10, max=10)
        return params


class KerrShunt(AbstractShunt, nonlinear.Kerr):
    """
    This class models a resonator operated in the shunt-coupled configuration with a Kerr-type nonlinearity.
    """

    # See nonlinear.Kerr.kerr_detuning()
    geometry_coefficient = 1 / 2

    def __init__(self, choose, *args, **kwds):
        """
        :param choose: a numpy ufunc; see nonlinear.Kerr.kerr_detuning().
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """

        def kerr_shunt(frequency, resonance_frequency, internal_loss, coupling_loss, asymmetry, normalized_input_rate):
            detuning = frequency / resonance_frequency - 1
            kerr_detuning = self.kerr_detuning(detuning=detuning, coupling_loss=coupling_loss,
                                               internal_loss=internal_loss, kerr_input=normalized_input_rate,
                                               geometry_coefficient=self.input_rate_coefficient, choose=choose)
            return 1 - ((1 + 1j * asymmetry) /
                        (1 + (internal_loss + 2j * (detuning - kerr_detuning)) / coupling_loss))

        super(KerrShunt, self).__init__(func=kerr_shunt, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        resonance_frequency, coupling_loss, internal_loss = self.guess_smooth(frequency=frequency, data=data)
        params = self.make_params()
        params['resonance_frequency'].set(value=resonance_frequency, min=frequency.min(), max=frequency.max())
        params['coupling_loss'].set(value=coupling_loss, min=1e-12, max=1)
        params['internal_loss'].set(value=internal_loss, min=1e-12, max=1)
        params['asymmetry'].set(value=0, min=-10, max=10)
        params['normalized_input_rate'].set(value=0)
        return params


# ResonatorFitters

class ShuntFitter(base.ResonatorFitter):
    """
    This class fits data from a linear shunt-coupled resonator.
    """

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

    def invert(self, scattering_data):
        z = self.coupling_loss * ((1 + 1j * self.asymmetry) / (1 - scattering_data) - 1)
        detuning = z.imag / 2
        internal_loss = z.real
        return detuning, internal_loss


class KerrShuntFitter(base.ResonatorFitter):
    """
    This class fits data from a shunt-coupled resonator with a Kerr-type nonlinearity.
    """

    def __init__(self, frequency, data, background_model=None, errors=None, choose=np.max, **kwds):
        if background_model is None:
            background_model = background.ComplexConstant()
        super(KerrShuntFitter, self).__init__(frequency=frequency, data=data,
                                              foreground_model=KerrShunt(choose=choose),
                                              background_model=background_model, errors=errors, **kwds)

    # ToDo: math
    def invert(self, scattering_data):
        pass
