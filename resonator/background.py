"""
This module contains models for the background response of a system.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import lmfit


class One(lmfit.model.Model):
    """
    This class represents background response that is calibrated in both magnitude and phase. It has no parameters.
    """

    def __init__(self, *args, **kwargs):
        def func(frequency):
            return np.ones(frequency.size, dtype='complex')
        super(One, self).__init__(func=func, *args, **kwargs)

    def guess(self, data, **kwargs):
        return self.make_params()


class UnitNorm(lmfit.model.Model):
    """
    This class represents background response that is calibrated in magnitude but not in phase. Its single parameter is
    the phase (in radians), which is constant in frequency.

    It can be used when the system has been calibrated except for a constant phase offset.
    """

    def __init__(self, *args, **kwargs):
        def func(frequency, phase):
            return np.ones(frequency.size, dtype='complex') * np.exp(1j * phase)
        super(UnitNorm, self).__init__(func=func, *args, **kwargs)

    def guess(self, data, **kwargs):
        params = self.make_params()
        params['phase'].value = np.mean(np.angle(data))
        return params


class ComplexConstant(lmfit.model.Model):
    """
    This class represents background response that is constant in frequency. Its two parameters are the magnitude of the
    response and its phase (in radians).

    It can be used when the cable delay has been accurately calibrated but a constant phase offset remains, and the gain
    has not been calibrated.
    """

    def __init__(self, *args, **kwargs):
        def func(frequency, magnitude, phase):
            return magnitude * np.exp(1j * phase) * np.ones(frequency.size)
        super(ComplexConstant, self).__init__(func=func, *args, **kwargs)

    def guess(self, data, fraction=0.1):
        """
        This function should calculate very good inital values for configurations in which the transmission far from
        resonance is nonzero, such as the shunt and reflection configurations. It will underestimate the magnitude for
        the transmission configuration, especially when the internal loss is high. For transmission resonators, the
        magnitude guess may be improved by using a smaller median fraction so that it uses fewer points closer to the
        resonance.

        :param data: an array of complex data points.
        :param fraction: the fraction of points to use when estimating the background magnitude; it should be large
          enough that any spurious high-magnitude points do not bias the median; for transmission resonators it should
          be small enough that points far from the peak are not included.
        :return: lmfit.Parameters
        """
        params = self.make_params()
        # Use a fraction of the points with the largest magnitude to estimate the background magnitude.
        median_indices = np.argsort(np.abs(data))[-int(fraction * data.size):]
        median = np.median(data[median_indices].real) + 1j * np.median(data[median_indices].imag)
        params['magnitude'].set(value=np.abs(median), min=0)
        params['phase'].value = np.angle(median)
        return params


class ConstantGainConstantDelay(lmfit.model.Model):
    """
    This class represents background response that has constant magnitude and a fixed time delay and phase offset.

    This is a reasonable model to use for VNA data for which no calibration has been done. In order to fit the phase
    wrapping, the frequency range should be substantially larger than the resonator linewidth and the frequency spacing
    should be substantially less than the period of the phase wrapping.
    """

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
    """
    This class represents background response for which the magnitude varies linearly with frequency and there is a
    fixed time delay and phase offset. The parameters are the reference frequency, the time delay, the phase at the
    reference frequency, the slope of the magnitude with frequency, and the magnitude at the reference frequency. The
    reference frequency is a fixed parameter that is set equal to the minimum frequency; using zero as the reference
    frequency would be simpler but this fails in practice because the frequency range is too small.

    This is a reasonable model to use for VNA data for which no calibration has been done. In order to fit the phase
    wrapping and magnitude variation, the frequency range should be substantially larger than the resonator linewidth
    and the frequency spacing should be substantially less than the period of the phase wrapping.
    """

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


class KnownMagnitudeConstantDelay(lmfit.model.Model):
    """
    This model represents background response that has been calibrated in magnitude but has an uncalibrated time delay.
    """
    pass


class KnownBackground(lmfit.model.Model):
    """
    This model represents background response that has been measured, so it has no free parameters.
    """

    def __init__(self, measurement_frequency, measurement_data, *args, **kwargs):
        def func(frequency):
            data_real = np.interp(frequency, measurement_frequency, measurement_data.real)
            data_imag = np.interp(frequency, measurement_frequency, measurement_data.imag)
            return data_real + 1j * data_imag
        super(KnownBackground, self).__init__(func=func, *args, **kwargs)

    def guess(self, data, **kwargs):
        return self.make_params()

