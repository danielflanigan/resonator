"""
This module contains models and fitters for resonators that are operated in the reflection configuration.
"""
from __future__ import absolute_import, division, print_function

import lmfit
import numpy as np

from . import background, base, nonlinear


# Models

class AbstractReflection(lmfit.model.Model):
    """
    This is abstract class that models a resonator operated in reflection.
    """

    # This is the value of the scattering data far from resonance.
    reference_point = -1 + 0j

    @staticmethod
    def guess_smooth(frequency, data):
        width = frequency.size // 10
        gaussian = np.exp(-np.linspace(-4, 4, width) ** 2)
        gaussian /= np.sum(gaussian)
        smoothed = np.convolve(gaussian, data, mode='same')  # This array has the same number of points as the data
        # The edges are corrupted by zero-padding, so set them equal to non-corrupted points
        smoothed[:width] = smoothed[width]
        smoothed[-width:] = smoothed[-(width + 1)]
        resonance_index = np.argmax(smoothed.real)
        resonance_frequency = frequency[resonance_index]
        linewidth = frequency[np.argmin(smoothed.imag)] - frequency[np.argmax(smoothed.imag)]
        internal_plus_coupling = linewidth / resonance_frequency
        internal_over_coupling = 2 / (np.abs(smoothed[resonance_index]) + 1) - 1
        coupling_loss = internal_plus_coupling / (1 + internal_over_coupling)
        internal_loss = internal_plus_coupling / (1 + 1 / internal_over_coupling)
        return resonance_frequency, coupling_loss, internal_loss


# ToDo: rename to LinearReflection
class Reflection(AbstractReflection):
    """
    This class models a linear resonator operated in reflection.
    """

    def __init__(self, *args, **kwds):
        """
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """
        def reflection(frequency, resonance_frequency, internal_loss, coupling_loss):
            detuning = frequency / resonance_frequency - 1
            return -1 + (2 / (1 + (internal_loss + 2j * detuning) / coupling_loss))
        super(Reflection, self).__init__(func=reflection, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        resonance_frequency, coupling_loss, internal_loss = self.guess_smooth(frequency=frequency, data=data)
        params = self.make_params()
        params['resonance_frequency'].set(value=resonance_frequency, min=frequency.min(), max=frequency.max())
        params['coupling_loss'].set(value=coupling_loss, min=1e-12, max=1)
        params['internal_loss'].set(value=internal_loss, min=1e-12, max=1)
        return params


# ToDo: test with new photon number code
class ReflectionNonlinear(AbstractReflection):
    """
    This class models a resonator operated in reflection with a Kerr-type nonlinearity.
    """

    # See nonlinear.kerr_detuning()
    input_rate_coefficient = 1

    def __init__(self, choose, *args, **kwds):
        """
        :param choose: a numpy ufunc; see nonlinear.kerr_detuning().
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """
        def reflection_nonlinear(frequency, resonance_frequency, coupling_loss, internal_loss, normalized_input_rate):
            detuning = frequency / resonance_frequency - 1
            kerr_detuning = nonlinear.kerr_detuning(detuning=detuning, coupling_loss=coupling_loss,
                                                    internal_loss=internal_loss,
                                                    normalized_input_rate=normalized_input_rate,
                                                    input_rate_coefficient=self.input_rate_coefficient, choose=choose)
            return -1 + (2 / (1 + (internal_loss + 2j * (detuning - kerr_detuning)) / coupling_loss))
        super(ReflectionNonlinear, self).__init__(func=reflection_nonlinear, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        resonance_frequency, coupling_loss, internal_loss = self.guess_smooth(frequency=frequency, data=data)
        params = self.make_params()
        params['resonance_frequency'].set(value=resonance_frequency, min=frequency.min(), max=frequency.max())
        params['coupling_loss'].set(value=coupling_loss, min=1e-12, max=1)
        params['internal_loss'].set(value=internal_loss, min=1e-12, max=1)
        params['normalized_input_rate'].set(value=0)
        return params


# ResonatorFitters

#ToDo: rename to LinearReflectionFitter
class ReflectionFitter(base.ResonatorFitter):
    """
    This class fits data from a resonator operated in reflection.
    """

    def __init__(self, frequency, data, background_model=None, errors=None, **kwds):
        if background_model is None:
            background_model = background.ComplexConstant()
        super(ReflectionFitter, self).__init__(frequency=frequency, data=data, foreground_model=Reflection(),
                                               background_model=background_model, errors=errors, **kwds)

    def invert(self, scattering_data):
        z = self.coupling_loss * (2 / (1 + scattering_data) - 1)
        detuning = z.imag / 2
        internal_loss = z.real
        return detuning, internal_loss


# ToDo: rename to KnownXLinearReflectionFitter
class KnownReflectionFitter(ReflectionFitter):
    """
    This class fits data from a linear resonator operated in reflection.
    """

    def __init__(self, frequency, data, background_frequency, background_data, errors=None, **kwds):
        foreground_model = Reflection()
        # Compensate for the pi phase shift present in the reflected background data.
        background_model = background.Known(measurement_frequency=background_frequency,
                                            measurement_data=background_data / foreground_model.reference_point)
        super(ReflectionFitter, self).__init__(frequency=frequency, data=data, foreground_model=foreground_model,
                                               background_model=background_model, errors=errors, **kwds)


# ToDo: rename to NonlinearReflectionFitter
class ReflectionNonlinearFitter(base.ResonatorFitter):
    """
    This class fits data from a resonator operated in reflection with a Kerr-type nonlinearity.
    """
    # ToDo: add static methods to choose roots

    # ToDo: figure out how to handle knowledge of kerr or input power
    def __init__(self, frequency, data, background_model=None, errors=None, choose=np.min, **kwds):
        if background_model is None:
            background_model = background.ComplexConstant()
        super(ReflectionNonlinearFitter, self).__init__(frequency=frequency, data=data,
                                                        foreground_model=ReflectionNonlinear(choose=choose),
                                                        background_model=background_model, errors=errors, **kwds)

    # ToDo: math
    def invert(self, scattering_data):
        pass


