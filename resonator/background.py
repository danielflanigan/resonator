"""
This module contains models for the background response of a system.

"""
from __future__ import absolute_import, division, print_function

import numpy as np

from . import base


class One(base.BackgroundModel):
    """
    This class represents background response that is calibrated in both magnitude and phase. It has no free parameters.
    """

    def __init__(self, *args, **kwds):
        def one(frequency):
            return np.ones_like(frequency, dtype='complex')

        super(One, self).__init__(func=one, *args, **kwds)


class Phase(base.BackgroundModel):
    """
    This class represents background response that is constant in frequency and calibrated in magnitude but not in
    phase. Its single parameter is the phase in radians.

    It can be used when the system has been calibrated except for a constant phase offset.
    """

    def __init__(self, *args, **kwds):
        def phase(frequency, phase):
            return np.ones_like(frequency) * np.exp(1j * phase)

        super(Phase, self).__init__(func=phase, *args, **kwds)

    def guess(self, data, fraction=0.1, **kwds):
        params = self.make_params()
        # Use a fraction of the points with the largest magnitude to estimate the background phase.
        median_indices = np.argsort(np.abs(data))[-int(fraction * data.size):]
        median = np.median(data[median_indices].real) + 1j * np.median(data[median_indices].imag)
        params['phase'].value = np.angle(median)
        return params


class Magnitude(base.BackgroundModel):
    """
    This class represents background response that is constant in frequency and calibrated in phase but not in
    magnitude. Its single parameter is the magnitude in scattering parameter units (i.e. V / V and not dB).

    It can be used when the system phase offset and delay have been perfectly calibrated.
    """

    def __init__(self, *args, **kwds):
        def magnitude(frequency, magnitude):
            return np.ones_like(frequency) * np.exp(1j * magnitude)

        super(Magnitude, self).__init__(func=magnitude, *args, **kwds)

    def guess(self, data, fraction=0.1, **kwds):
        params = self.make_params()
        # Use a fraction of the points with the largest magnitude to estimate the background magnitude.
        median_indices = np.argsort(np.abs(data))[-int(fraction * data.size):]
        median = np.median(data[median_indices].real) + 1j * np.median(data[median_indices].imag)
        params['magnitude'].value = np.abs(median)
        return params


class MagnitudePhase(base.BackgroundModel):
    """
    This class represents background response that is constant in frequency. Its parameters are the magnitude in
    scattering parameter units (i.e. V / V and not dB) and the phase in radians.

    This is a reasonable model to use for data acquired with a VNA when the electrical delay has been set exactly and
    the gain is nearly constant. In order to fit the phase wrapping, the frequency range should be substantially
    larger than the resonator linewidth and the frequency spacing should be substantially less than the period of the
    phase wrapping.

    Many of the ResonatorFitters use this as their default background model because it is both complex enough to handle
    typical cases and simple enough to avoid fit convergence issues.
    """

    def __init__(self, *args, **kwds):
        def magnitude_phase(frequency, magnitude, phase):
            return magnitude * np.exp(1j * phase) * np.ones_like(frequency)

        super(MagnitudePhase, self).__init__(func=magnitude_phase, *args, **kwds)

    def guess(self, data, fraction=0.1, **kwds):
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
        :param kwds: currently ignored.
        :return: lmfit.Parameters
        """
        params = self.make_params()
        # Use a fraction of the points with the largest magnitude to estimate the background magnitude.
        median_indices = np.argsort(np.abs(data))[-int(fraction * data.size):]
        median = np.median(data[median_indices].real) + 1j * np.median(data[median_indices].imag)
        params['magnitude'].set(value=np.abs(median), min=0)
        params['phase'].value = np.angle(median)
        return params


class MagnitudePhaseDelay(base.BackgroundModel):
    """
    This class represents background response that has constant magnitude and a fixed time delay and phase offset.

    The free parameters are the magnitude, the phase at the reference frequency, and an electrical time delay. The
    reference frequency is a fixed parameter that is set equal to the mean frequency; always using zero as the
    reference frequency would be simpler but this fails in practice because the frequency range is too small and too
    far from the origin.

    This is a reasonable model to use for data acquired with a VNA when the reference plane has not been set and the
    background is fairly flat. In order to fit the phase wrapping, the frequency range should be substantially larger
    than the resonator linewidth and the frequency spacing should be substantially less than the period of the phase
    wrapping.
    """

    def __init__(self, *args, **kwds):
        def magnitude_phase_delay(frequency, frequency_reference, magnitude, phase, delay):
            return magnitude * np.exp(1j * (2 * np.pi * (frequency - frequency_reference) * delay + phase))

        super(MagnitudePhaseDelay, self).__init__(func=magnitude_phase_delay, *args, **kwds)

    def guess(self, data, frequency=None, fraction=0.1, **kwds):
        params = self.make_params()
        frequency_reference = frequency.mean()
        params['frequency_reference'].set(value=frequency_reference, vary=False)
        phase_slope, phase_reference = np.polyfit(frequency - frequency_reference, np.unwrap(np.angle(data)), 1)
        params['phase'].set(value=phase_reference, min=phase_reference - np.pi, max=phase_reference + np.pi)
        delay = phase_slope / (2 * np.pi)
        params['delay'].set(value=delay)
        # Use a fraction of the points with the largest magnitude to estimate the background magnitude.
        unwrapped = data * np.exp(-1j * (2 * np.pi * (frequency - frequency_reference) * delay + phase_reference))
        median_indices = np.argsort(np.abs(unwrapped))[-int(fraction * unwrapped.size):]
        median = np.median(unwrapped[median_indices].real) + 1j * np.median(unwrapped[median_indices].imag)
        params['magnitude'].set(value=np.abs(median), min=0)
        return params


class MagnitudeSlopeOffsetPhaseDelay(base.BackgroundModel):
    """
    This class represents background response for which the magnitude varies linearly with frequency and there is a
    fixed time delay and phase offset.

    The free parameters are the magnitude at a fixed reference frequency, the slope of the magnitude with frequency,
    the phase at the reference frequency, and an electrical time delay. The reference frequency is a fixed parameter
    that is set equal to the mean frequency; always using zero as the reference frequency would be simpler but this
    fails in practice because the frequency range is too small and too far from the origin.

    This is a reasonable model to use for data acquired with a VNA when the reference plane has not been set and the
    background magnitude is not flat. In order to fit the phase wrapping, the frequency range should be substantially
    larger than the resonator linewidth and the frequency spacing should be substantially less than the period of the
    phase wrapping.
    """

    def __init__(self, *args, **kwds):
        def magnitude_slope_offset_phase_delay(frequency, frequency_reference, magnitude_slope, magnitude_offset, phase,
                                               delay):
            magnitude = magnitude_offset + magnitude_slope * (frequency - frequency_reference)
            return magnitude * np.exp(1j * (2 * np.pi * (frequency - frequency_reference) * delay + phase))

        super(MagnitudeSlopeOffsetPhaseDelay, self).__init__(func=magnitude_slope_offset_phase_delay, *args, **kwds)

    def guess(self, data, frequency=None, **kwds):
        params = self.make_params()
        frequency_reference = frequency.mean()
        params['frequency_reference'].set(value=frequency_reference, vary=False)
        phase_slope, phase_reference = np.polyfit(frequency - frequency_reference, np.unwrap(np.angle(data)), 1)
        params['phase'].set(value=phase_reference, min=phase_reference - np.pi, max=phase_reference + np.pi)
        params['delay'].set(value=phase_slope / (2 * np.pi))
        magnitude_slope, magnitude_offset = np.polyfit(frequency - frequency_reference, np.abs(data), 1)
        params['magnitude_slope'].set(value=magnitude_slope)
        params['magnitude_offset'].set(value=magnitude_offset)
        return params


class Known(base.BackgroundModel):
    """
    This model represents background response that has been measured, so it has no free parameters. It uses linear
    interpolation in the complex plane to return the background values between the measurement points.

    This background is most useful when it is somehow possible to measure the background without the resonator
    present, possibly by changing the temperature or applying a magnetic field. In the shunt case, this procedure
    should work well and no correction should be necessary. In the reflection case, even if the resonance frequency
    can be shifted by many linewidths, the pi phase shift due to the reflection will still be present,
    so the measured background data should be multiplied by -1 before being passed to this class. In the transmission
    case, the background could be measured either using switches to bypass the resonator or during a separate
    measurement.
    """

    def __init__(self, measurement_frequency, measurement_data, *args, **kwds):
        def known(frequency):
            data_real = np.interp(frequency, measurement_frequency, measurement_data.real)
            data_imag = np.interp(frequency, measurement_frequency, measurement_data.imag)
            return data_real + 1j * data_imag

        super(Known, self).__init__(func=known, *args, **kwds)
