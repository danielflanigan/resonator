"""
This module contains models and fitters for resonators that are operated in the reflection configuration.
"""
from __future__ import absolute_import, division, print_function

import lmfit
import numpy as np

from . import background, base


class Reflection(lmfit.model.Model):
    """
    This class models a resonator operated in reflection.
    """
    reference_point = -1 + 0j

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


class ReflectionFitter(base.ResonatorFitter):

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


class KnownReflectionFitter(ReflectionFitter):

    def __init__(self, frequency, data, background_frequency, background_data, errors=None, **kwds):
        foreground_model = Reflection()
        # Compensate for the pi phase shift present in the reflected background data.
        background_model = background.Known(measurement_frequency=background_frequency,
                                            measurement_data=background_data / foreground_model.reference_point)
        super(ReflectionFitter, self).__init__(frequency=frequency, data=data, foreground_model=foreground_model,
                                               background_model=background_model, errors=errors, **kwds)


class ReflectionNonlinear(lmfit.model.Model):
    """
    This class models a resonator operated in reflection.
    """
    reference_point = -1 + 0j

    def __init__(self, choose, *args, **kwds):
        """
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """
        self.choose = choose
        def reflection(frequency, resonance_frequency, internal_loss, coupling_loss, KXin):
            detuning = frequency / resonance_frequency - 1
            kerr_detuning = KX(detuning=detuning, coupling_loss=coupling_loss, internal_loss=internal_loss, KXin=KXin,
                               choose=self.choose)
            return -1 + (2 / (1 + (internal_loss + 2j * (detuning - kerr_detuning)) / coupling_loss))
        super(ReflectionNonlinear, self).__init__(func=reflection, *args, **kwds)

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
        params['KXin'].set(value=0)
        return params


class ReflectionNonlinearFitter(base.ResonatorFitter):

    def __init__(self, frequency, data, background_model=None, errors=None, choose=np.min, **kwds):
        if background_model is None:
            background_model = background.ComplexConstant()
        super(ReflectionNonlinearFitter, self).__init__(frequency=frequency, data=data,
                                                        foreground_model=ReflectionNonlinear(choose=choose),
                                                        background_model=background_model, errors=errors, **kwds)

    def invert(self, time_ordered_data):
        pass


def KX(detuning, coupling_loss, internal_loss, KXin, choose):
    """
    """
    is_scalar = False
    is_zero_size = False
    if np.isscalar(detuning):
        is_scalar = True
        detuning = np.array([detuning])
    elif not detuning.shape:
        is_zero_size = True
        detuning.shape = (0,)
    roots = np.zeros(detuning.size)
    b = -2.0 * detuning
    c = ((coupling_loss + internal_loss) / 2) ** 2 + detuning ** 2
    d = -KXin * coupling_loss
    delta0 = b ** 2 - 3 * c
    delta1 = 2 * b ** 3 - 9 * b * c + 27 * d

    delta = (4 * delta0 ** 3 - delta1 ** 2) / 27
    three_distinct_real = delta > 0
    multiple_real = delta == 0
    one_real = delta < 0

    cc_one_real = np.cbrt((delta1[one_real] + np.sqrt(delta1[one_real] ** 2 - 4 * delta0[one_real] ** 3)) / 2)
    if one_real.any():
        roots[one_real] = np.real(-(b[one_real] + cc_one_real + delta0[one_real] / cc_one_real) / 3)

    triple = multiple_real & (delta0 == 0)
    if triple.any():
        roots[triple] = -b[triple] / 3
    double_and_simple = multiple_real & (delta0 != 0)
    # This occurs right at bifurcation
    if double_and_simple.any():
        double_root = (9 * d - b[double_and_simple] * c[double_and_simple]) / (2 * delta0[double_and_simple])
        simple_root = (4 * b[double_and_simple] * c[double_and_simple] - 9 * d - b[double_and_simple] ** 3) / delta0[
            double_and_simple]
        roots[double_and_simple] = choose(np.vstack((double_root, simple_root)), axis=0)

    if three_distinct_real.any():
        cc_three_distinct_real = ((delta1[three_distinct_real] + np.sqrt(
            (delta1[three_distinct_real] ** 2 - 4 * delta0[three_distinct_real] ** 3).astype(np.complex))) / 2) ** (
                                             1 / 3)
        xi = (-1 + 1j * np.sqrt(3)) / 2
        x0 = np.real(-1 / 3 * (b[three_distinct_real] + cc_three_distinct_real + delta0[
            three_distinct_real] / cc_three_distinct_real))
        x1 = np.real(-1 / 3 * (b[three_distinct_real] + xi * cc_three_distinct_real + delta0[three_distinct_real] / (
                    xi * cc_three_distinct_real)))
        x2 = np.real(-1 / 3 * (
                    b[three_distinct_real] + xi ** 2 * cc_three_distinct_real + delta0[three_distinct_real] / (
                        xi ** 2 * cc_three_distinct_real)))
        roots[three_distinct_real] = choose(np.vstack((x0, x1, x2)), axis=0)

    if is_scalar or is_zero_size:
        roots = roots[0]
    return roots


def KXin_bifurcation(coupling_loss, internal_loss):
    return 3 ** (-3 / 2) * (internal_loss + coupling_loss) ** 3 / coupling_loss
