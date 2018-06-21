from __future__ import absolute_import, division, print_function

import numpy as np
import lmfit


class One(lmfit.model.Model):

    def __init__(self, *args, **kwargs):
        def func(frequency):
            return np.ones(frequency.size, dtype='complex')
        super(One, self).__init__(func=func, *args, **kwargs)

    def guess(self, data, **kwargs):
        return self.make_params()


# ToDo: determine whether magnitude and phase is better than real and imaginary
class ComplexConstant(lmfit.model.Model):

    def __init__(self, *args, **kwargs):
        def func(frequency, gain, phase):
            return gain * np.exp(1j * phase) * np.ones(frequency.size)
        super(ComplexConstant, self).__init__(func=func, *args, **kwargs)

    def guess(self, data, **kwargs):
        params = self.make_params()
        params['gain'].value = np.mean(np.abs(data))
        params['gain'].min = 0
        params['phase'].value = np.mean(np.angle(data))
        return params


class ConstantGainConstantDelay(lmfit.model.Model):

    def __init__(self, *args, **kwargs):
        def func(frequency, frequency_reference, delay, phase, gain):
            return gain * np.exp(1j * (2 * np.pi * (frequency - frequency_reference) * delay + phase))
        super(ConstantGainConstantDelay, self).__init__(func=func, *args, **kwargs)

    def guess(self, data, frequency=None, **kwargs):
        frequency_reference = frequency.min()
        phase_slope, phase_reference = np.polyfit(frequency - frequency_reference, np.unwrap(np.angle(data)), 1)
        gain = np.abs(np.mean(data * np.exp(-1j * (phase_slope * (frequency - frequency_reference) + phase_reference))))
        params = self.make_params(frequency_reference=frequency_reference, delay=phase_slope / (2 * np.pi),
                                  phase=phase_reference, gain=gain)
        params['frequency_reference'].vary = False
        params['phase'].set(min=phase_reference - np.pi, max=phase_reference + np.pi)  # ToDo: copied; necessary?
        params.update(**kwargs)
        return params


class LinearGainConstantDelay(lmfit.model.Model):

    def __init__(self, *args, **kwargs):
        def func(frequency, frequency_reference, delay, phase, gain_slope, gain_reference):
            gain = gain_reference + gain_slope * (frequency - frequency_reference)
            return gain * np.exp(1j * (2 * np.pi * (frequency - frequency_reference) * delay + phase))
        super(LinearGainConstantDelay, self).__init__(func=func, *args, **kwargs)

    def guess(self, data, frequency=None, **kwargs):
        frequency_reference = frequency.min()
        phase_slope, phase_reference = np.polyfit(frequency - frequency_reference, np.unwrap(np.angle(data)), 1)
        gain_slope, gain_reference = np.polyfit(frequency - frequency_reference, np.abs(data), 1)
        params = self.make_params(frequency_reference=frequency_reference, delay=phase_slope / (2 * np.pi),
                                  phase=phase_reference, gain_slope=gain_slope, gain_reference=gain_reference)
        params['frequency_reference'].vary = False
        params['phase'].set(min=phase_reference - np.pi, max=phase_reference + np.pi)
        return params


# ToDo: use this to help fitting of transmission resonators
# ToDo: upgrade to a more sophisticated interpolation function if necessary
class KnownBackground(lmfit.model.Model):
    """
    This model has no free parameters and should be used when the background transmission has been measured.
    """

    def __init__(self, measurement_frequency, measurement_data, *args, **kwargs):
        def func(frequency):
            data_real = np.interp(frequency, measurement_frequency, measurement_data.real)
            data_imag = np.interp(frequency, measurement_frequency, measurement_data.imag)
            return data_real + 1j * data_imag
        super(KnownBackground, self).__init__(func=func, *args, **kwargs)

    def guess(self, data, **kwargs):
        return self.make_params()

