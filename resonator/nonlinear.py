"""
This module contains functions related to the nonlinearity of photon number with input power.
"""
from __future__ import absolute_import, division, print_function

import numpy as np


class Kerr(object):
    """A mix-in class that contains equations for modeling resonators with a Kerr-type nonlinearity."""

    # Subclasses must define this; see kerr_detuning() documentation.
    geometry_coefficient = None

    @classmethod
    def kerr_detuning(cls, detuning, coupling_loss, internal_loss, kerr_input, choose):
        """
        Return one real root of the cubic polynomial
        0 = a y^3 + b y^2 + c y + d
          = y^3 - 2 x y^2 + [(loss_i + loss_c)^2 / 4 + x^2] y - g \chi,
        where the variables have the following meanings:
          x = f / f_r - 1 = detuning -- the the dimensionless fractional frequency detuning,
          loss_c = coupling_loss -- the coupling loss (inverse coupling quality factor),
          loss_i = internal_loss -- the internal loss (inverse internal quality factor),
          g = geometry_coefficient -- a number of order 1 that depends on the coupling geometry, and
          \chi = kerr_input -- the dimensionless, rescaled input photon rate, discussed below.
        The rescaled input photon rate is
          \chi = K X_{in} / \omega_r^2,
        where the input photon rate X_{in} means the rate (in s^{-1}) at which photons arrive from the input port,
        typically numbered port 1; K is the Kerr coefficient; and, \omega_r = 2 \pi f_r the resonance angular frequency.
        The input power Pin = \hbar \omega Xin, where omega is the signal angular frequency. The independent variable is
          y = K n / omega_r,
        where n = <a^\dag a> is the average photon number in the resonator. This quantity y is the detuning shift
        (dimensionless) that is caused by the Kerr nonlinearity. Note that it can be either positive or negative.

        This cubic equation can be derived by considering a resonator with hamiltonian
          H = \hbar \omega_r a^\dag a + (\hbar K / 2) a^\dag a^\dag a a,
        where a (a^\dag) is the annihilation (creation) operator for photons in the resonator. The cubic equation is
        exactly equivalent to Equation 37 of B. Yurke and E. Buks Journal of Lightwave Technology 24. 5054 (2006)
        with the nonlinear loss term \gamma_3 = 0. Without a priori knowledge of either the Kerr coefficient or the
        input power, the only way to avoid degeneracy between the parameters is to use the Kerr detuning, instead of
        the photon number, as the independent variable.

        Because the

        :param detuning: a 1D array[float] or single float,
        :param coupling_loss: a single value of the inverse coupling quality factor.
        :param internal_loss: a single value of the inverse internal quality factor.
        :param kerr_input: a single value of the input photon rate KXin described above.
        :param choose: a function used to choose which root to return when the cubic has multiple roots; it is called as
          choose(np.vstack((array_of_roots_0, ... , array_of_roots_n)), axis=0)
          and recommended values are np.min or np.max, which respectively select either the minimum or maximum root.
        :return: array[float]
        """
        is_scalar = False
        is_zero_size = False
        if np.isscalar(detuning):
            is_scalar = True
            detuning = np.array([detuning])
        elif not detuning.shape:
            is_zero_size = True
            detuning.shape = (1,)
        roots = np.zeros(detuning.size)
        b = -2 * detuning
        c = ((coupling_loss + internal_loss) / 2) ** 2 + detuning ** 2
        d = -cls.geometry_coefficient * kerr_input
        delta0 = b ** 2 - 3 * c
        delta1 = 2 * b ** 3 - 9 * b * c + 27 * d
        delta = (4 * delta0 ** 3 - delta1 ** 2) / 27
        # These boolean arrays partition the result into three cases.
        three_distinct_real = delta > 0
        multiple_real = delta == 0
        one_real = delta < 0

        # Three distinct real roots
        if three_distinct_real.any():
            # Cast to complex so that sqrt() returns one of the a complex roots.
            sqrt_arg = (delta1[three_distinct_real] ** 2 - 4 * delta0[three_distinct_real] ** 3).astype(np.complex)
            cc_three_distinct_real = ((delta1[three_distinct_real] + np.sqrt(sqrt_arg)) / 2) ** (1 / 3)
            xi = (-1 + 1j * np.sqrt(3)) / 2
            x0 = np.real(-1 / 3 * (b[three_distinct_real] + cc_three_distinct_real
                                   + delta0[three_distinct_real] / cc_three_distinct_real))
            x1 = np.real(-1 / 3 * (b[three_distinct_real] + xi * cc_three_distinct_real
                                   + delta0[three_distinct_real] / (xi * cc_three_distinct_real)))
            x2 = np.real(-1 / 3 * (b[three_distinct_real] + xi ** 2 * cc_three_distinct_real
                                   + delta0[three_distinct_real] / (xi ** 2 * cc_three_distinct_real)))
            roots[three_distinct_real] = choose(np.vstack((x0, x1, x2)), axis=0)

        # Three real roots with a multiple root
        triple = multiple_real & (delta0 == 0)
        if triple.any():
            roots[triple] = -b[triple] / 3
        double_and_simple = multiple_real & (delta0 != 0)  # This occurs exactly at bifurcation
        if double_and_simple.any():
            double_root = (9 * d - b[double_and_simple] * c[double_and_simple]) / (2 * delta0[double_and_simple])
            simple_root = ((4 * b[double_and_simple] * c[double_and_simple] - 9 * d - b[double_and_simple] ** 3)
                           / delta0[double_and_simple])
            roots[double_and_simple] = choose(np.vstack((double_root, simple_root)), axis=0)

        # One real root
        if one_real.any():
            # One can choose either sign of the square root as long as the argument of the cube root is nonzero.
            square_root = np.sqrt(delta1[one_real] ** 2 - 4 * delta0[one_real] ** 3)
            plus = (delta1[one_real] + square_root) / 2
            minus = (delta1[one_real] - square_root) / 2
            cbrt_argument = np.where(plus != 0, plus, minus)
            cc_one_real = np.cbrt(cbrt_argument)
            roots[one_real] = np.real(-(b[one_real] + cc_one_real + delta0[one_real] / cc_one_real) / 3)

        if is_scalar or is_zero_size:
            return roots[0]
        else:
            return roots

    # ToDo: check math
    @classmethod
    def absolute_kerr_detuning_at_bifurcation(cls, coupling_loss, internal_loss):
        y = cls.geometry_coefficient * 2 * 3 ** (-3 / 2) * (internal_loss + coupling_loss) ** 3 / coupling_loss
        return y

    @classmethod
    def kerr_coefficient_given_input_rate(cls, input_rate, resonance_frequency, coupling_loss, kerr_input):
        kerr_coefficient = ((kerr_input * (2 * np.pi * resonance_frequency) ** 2 / coupling_loss)
                            / input_rate)
        return kerr_coefficient

    @classmethod
    def input_rate_given_kerr_coefficient(cls, kerr_coefficient, resonance_frequency, coupling_loss, kerr_input):
        input_rate = ((kerr_input * (2 * np.pi * resonance_frequency) ** 2 / coupling_loss)
                      / kerr_coefficient)
        return input_rate

    @staticmethod
    def photon_number(resonance_frequency, coupling_loss, kerr_detuning, kerr_input, input_rate):
        n = (coupling_loss * input_rate * kerr_detuning) / (2 * np.pi * resonance_frequency * kerr_input)
        return n


# ToDo: use these slow functions to check the fast function above.

def chosen_photon_number(detuning, coupling_loss, internal_loss, normalized_kerr, normalized_input, choose):
    roots = [choose(roots) for roots in photon_number_roots(
        detuning=detuning, coupling_loss=coupling_loss, internal_loss=internal_loss,
        normalized_kerr=normalized_kerr, normalized_input=normalized_input)]
    return np.array(roots)


def photon_number_roots(detuning, coupling_loss, internal_loss, normalized_kerr, normalized_input):
    return np.roots(photon_number_cubic(detuning=detuning, coupling_loss=coupling_loss, internal_loss=internal_loss,
                                        normalized_kerr=normalized_kerr, normalized_input=normalized_input))


def photon_number_cubic(detuning, coupling_loss, internal_loss, normalized_kerr, normalized_input):
    a = normalized_kerr ** 2
    b = -2 * normalized_kerr * detuning
    c = ((coupling_loss + internal_loss) / 2) ** 2 + detuning ** 2
    d = -normalized_input
    return np.array([a, b, c, d])


