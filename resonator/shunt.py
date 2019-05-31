"""
This module contains models and fitters for resonators that are operated in the shunt ("hanger") configuration.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from . import background, base, guess, linear, kerr


class AbstractShunt(base.ResonatorModel):
    """
    This is an abstract class that models a resonator operated in the shunt-coupled configuration.
    """
    # This is the value of the scattering data far from resonance.
    reference_point = 1 + 0j

    # See kerr.kerr_detuning_shift for the meaning of this.
    io_coupling_coefficient = 1 / 2


# Linear models and fitters

class LinearShunt(AbstractShunt):
    """
    This class models a linear resonator operated in the shunt-coupled configuration.

    Here, linear means that the energy stored in the resonator is proportional to the power on the feedline. When the
    input power is sufficiently high, this relationship will start to become more complicated and the shape of the
    resonance will change. In this case, the resonance can still be fit with a Kerr model.
    """

    def __init__(self, *args, **kwds):
        """
        This class can be used directly, like any lmfit Model, but it is easier to use the LinearShuntFitter wrapper
        class that is defined in this module.

        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """

        def linear_shunt(frequency, resonance_frequency, coupling_loss, internal_loss, asymmetry):
            detuning = frequency / resonance_frequency - 1
            return 1 - ((1 + 1j * asymmetry) / (1 + (internal_loss + 2j * detuning) / coupling_loss))

        super(LinearShunt, self).__init__(func=linear_shunt, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        resonance_frequency, coupling_loss, internal_loss = guess.guess_smooth(frequency=frequency, data=data)
        params = self.make_params()
        params['resonance_frequency'].set(value=resonance_frequency, min=frequency.min(), max=frequency.max())
        params['coupling_loss'].set(value=coupling_loss, min=1e-12, max=1)
        params['internal_loss'].set(value=internal_loss, min=1e-12, max=1)
        params['asymmetry'].set(value=0, min=-10, max=10)
        return params


class LinearShuntFitter(linear.LinearResonatorFitter):
    """
    This class fits data from a linear shunt-coupled resonator.

    Here, linear means that the energy stored in the resonator is proportional to the power on the feedline. When the
    input power is sufficiently high, this relationship will start to become more complicated and the shape of the
    resonance will change. In this case, the resonance can still be fit with a Kerr model.
    """

    def __init__(self, frequency, data, background_model=None, errors=None, **kwds):
        """
        Fit the given data to a composite model that is the product of a background model and the LinearShunt model.

        :param frequency: an array of floats containing the frequencies at which the data was measured.
        :param data: an array of complex numbers containing the data.
        :param background_model: an instance (not the class) of a model representing the background response without the
          resonator; the default of `background.MagnitudePhase()` assumes that this is modeled well by a single complex
          constant at all frequencies.
        :param errors: an array of complex numbers containing the standard errors of the mean of the data points.
        :param kwds: keyword arguments passed directly to `lmfit.model.Model.fit()`.
        """
        if background_model is None:
            background_model = background.MagnitudePhase()
        super(LinearShuntFitter, self).__init__(frequency=frequency, data=data, foreground_model=LinearShunt(),
                                                background_model=background_model, errors=errors, **kwds)

    def invert(self, scattering_data):
        z = self.coupling_loss * ((1 + 1j * self.asymmetry) / (1 - scattering_data) - 1)
        detuning = z.imag / 2
        internal_loss = z.real
        return detuning, internal_loss


# Kerr models and fitters

class KerrShunt(AbstractShunt):
    """
    This class models a resonator operated in the shunt-coupled configuration with a Kerr-type nonlinearity.
    """

    def __init__(self, choose, *args, **kwds):
        """
        This class can be used directly, like any lmfit Model, but it is easier to use the KerrShuntFitter wrapper
        class that is defined in this module.

        :param choose: a numpy ufunc; see `kerr.kerr_detuning_shift`.
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """

        def kerr_shunt(frequency, resonance_frequency, internal_loss, coupling_loss, asymmetry, kerr_input):
            detuning = frequency / resonance_frequency - 1
            shift = kerr.kerr_detuning_shift(detuning=detuning, coupling_loss=coupling_loss,
                                             internal_loss=internal_loss, kerr_input=kerr_input,
                                             io_coupling_coefficient=self.io_coupling_coefficient, choose=choose)
            return 1 - ((1 + 1j * asymmetry) / (1 + (internal_loss + 2j * (detuning - shift)) / coupling_loss))

        super(KerrShunt, self).__init__(func=kerr_shunt, *args, **kwds)

    def guess(self, data=None, frequency=None, **kwds):
        resonance_frequency, coupling_loss, internal_loss = guess.guess_smooth(frequency=frequency, data=data)
        params = self.make_params()
        params['resonance_frequency'].set(value=resonance_frequency, min=frequency.min(), max=frequency.max())
        params['coupling_loss'].set(value=coupling_loss, min=1e-12, max=1)
        params['internal_loss'].set(value=internal_loss, min=1e-12, max=1)
        params['asymmetry'].set(value=0, min=-10, max=10)
        params['kerr_input'].set(value=0)
        return params

    @classmethod
    def absolute_kerr_input_at_bifurcation(cls, coupling_loss, internal_loss):
        return ((internal_loss + coupling_loss) ** 3
                / (3 ** (3 / 2) * cls.io_coupling_coefficient * coupling_loss))


class KerrShuntFitter(kerr.KerrFitter):
    """
    This class fits data from a shunt-coupled resonator with a Kerr-type nonlinearity.
    """

    def __init__(self, frequency, data, choose=np.max, background_model=None, errors=None, **fit_kwds):
        """
        Fit the given data to a composite model that is the product of a background model and the KerrShunt model.

        :param frequency: an array of floats containing the frequencies at which the data was measured.
        :param data: an array of complex numbers containing the data.
        :param choose: a numpy ufunc; see `kerr.kerr_detuning_shift`.
        :param background_model: an instance (not the class) of a model representing the background response without the
          resonator; the default of `background.MagnitudePhase()` assumes that this is modeled well by a single complex
          constant at all frequencies.
        :param errors: an array of complex numbers containing the standard errors of the mean of the data points.
        :param kwds: keyword arguments passed directly to lmfit.model.Model.fit().
        """
        if background_model is None:
            background_model = background.MagnitudePhase()
        super(KerrShuntFitter, self).__init__(frequency=frequency, data=data, choose=choose,
                                              foreground_model=KerrShunt(choose=choose),
                                              background_model=background_model, errors=errors, **fit_kwds)

    # ToDo: math
    def invert(self, scattering_data):
        pass
