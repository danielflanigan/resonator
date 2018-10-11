"""
This module contains functions related to the nonlinearity of photon number with input power.
"""
from __future__ import absolute_import, division, print_function

import numpy as np


# ToDo: split this into another function that returns all roots, for plotting and debugging
def kerr_detuning(detuning, coupling_loss, internal_loss, normalized_input_rate, input_rate_coefficient, choose):
    """
    Return one real root of the cubic polynomial
    0 = a y^3 + b y^2 + c y + d
      = y^3 - 2 x y^2 + [(loss_i + loss_c)^2 / 4 + x^2] y - r loss_c in
    where x = f / f_r - 1 is the the dimensionless fractional frequency detuning, loss_c is the inverse coupling quality
    factor, loss_i is the inverse internal quality factor, r is a dimensionless input rate coefficient that depends on
    the coupling configuration, and KXin is related to the input photon arrival rate X_in by
      KXin = K X_in / omega_r^2,
    with K the Kerr coefficient and omega_r = 2 \pi f_r the resonance angular frequency.

    :param detuning: a 1D array[float] or single float,
    :param coupling_loss: a single value of the inverse coupling quality factor.
    :param internal_loss: a single value of the inverse internal quality factor.
    :param normalized_input_rate: a single value of the input photon rate KXin described above.
    :param input_rate_coefficient: a single value of the coefficient of the quantity loss_c * KXin; see cubic above.
    :param choose: a function used to choose which root to return when the cubic has multiple roots; it is called as
      choose(np.vstack((array_of_roots_0, ... , array_of_roots_n)), axis=0)
      and recommended values are np.min or np.max, which respectively select either the minimum or maximum root.
    :return:
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
    d = -input_rate_coefficient * coupling_loss * normalized_input_rate  # a float, not an array
    delta0 = b ** 2 - 3 * c
    delta1 = 2 * b ** 3 - 9 * b * c + 27 * d
    delta = (4 * delta0 ** 3 - delta1 ** 2) / 27
    # These boolean arrays partition the result into three cases.
    three_distinct_real = delta > 0
    multiple_real = delta == 0
    one_real = delta < 0

    # Three distinct real roots
    if three_distinct_real.any():
        cc_three_distinct_real = ((delta1[three_distinct_real]
                                   + np.sqrt((delta1[three_distinct_real] ** 2 -
                                              4 * delta0[three_distinct_real] ** 3).astype(np.complex))) / 2) ** (1 / 3)
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
        cc_one_real = np.cbrt((delta1[one_real] + np.sqrt(delta1[one_real] ** 2 - 4 * delta0[one_real] ** 3)) / 2)
        roots[one_real] = np.real(-(b[one_real] + cc_one_real + delta0[one_real] / cc_one_real) / 3)

    if is_scalar or is_zero_size:
        return roots[0]
    else:
        return roots


def kerr_detuning_at_bifurcation(coupling_loss, internal_loss):
    return 3 ** (-3 / 2) * (internal_loss + coupling_loss) ** 3 / coupling_loss


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


