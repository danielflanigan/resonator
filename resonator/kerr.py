"""
Functions and classes related to resonators with a Kerr-type nonlinearity.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

# from scipy.constants import h
# Avoid adding dependence on scipy for just one constant
h = 6.62607004e-34

from . import base


def kerr_detuning_shift(detuning, coupling_loss, internal_loss, kerr_input, io_coupling_coefficient, choose):
    """
    Return one chosen real root of the cubic polynomial
    0 = a y^3 + b y^2 + c y + d
      = y^3 - 2 x y^2 + [(loss_i + loss_c)^2 / 4 + x^2] y - g loss_c \chi,
    where the variables have the following meanings:
      x = f / f_r - 1 = detuning -- the the dimensionless fractional frequency detuning,
      loss_c = coupling_loss -- the coupling loss (inverse coupling quality factor),
      loss_i = internal_loss -- the internal loss (inverse internal quality factor),
      g = io_coupling_coefficient -- a number of order 1 that depends on the coupling geometry, and
      \chi = kerr_input -- the dimensionless, rescaled input photon rate, discussed below.
    The rescaled input photon rate is
      \chi = K X_{in} / \omega_r^2,
    where the input photon rate X_{in} means the rate (in s^{-1}) at which photons arrive from the input port,
    typically numbered port 1; K is the Kerr coefficient; and, \omega_r = 2 \pi f_r the resonance angular frequency.
    The input power is
      P_{in} = \hbar \omega X_{in},
    where \omega is the signal angular frequency. The independent variable is
      y = K n / \omega_r,
    where n = <a^\dag a> is the average photon number in the resonator. This quantity y is the detuning shift
    (dimensionless) that is caused by the Kerr nonlinearity and it has the same sign as the Kerr coefficient.

    This cubic equation can be derived by considering a resonator with hamiltonian
      H = \hbar \omega_r a^\dag a + (\hbar K / 2) a^\dag a^\dag a a,
    where a (a^\dag) is the annihilation (creation) operator for photons in the resonator. The cubic equation is
    exactly equivalent to Equation 37 of B. Yurke and E. Buks Journal of Lightwave Technology 24, 5054 (2006)
    with the nonlinear loss term \gamma_3 = 0. Without a priori knowledge of either the Kerr coefficient or the
    input power, the only way to avoid degeneracy between the parameters is to use the Kerr detuning, instead of
    the photon number, as the independent variable.

    The `choose` function selects one real root when there are multiple real roots that correspond to multiple stable
    photon number states in the resonator. The recommended value of this function is `np.max` when fitting data taken
    with a VNA that sweeps the frequency in the positive direction, as is typically done. If the frequency is swept in
    the negative direction, the recommended value is `np.min`. The reason for this is as follows. When the Kerr
    coefficient is positive, the low-frequency value of the photon number is continuously connected to the branch in the
    bifurcation region with higher photon number. The resonator may be expected to stay in this branch through the
    bifurcation region. In this case, the Kerr detuning shift is also positive, and thus `np.max` selects this branch.
    When the Kerr coefficient is negative, the low-frequency value of the photon number is continuously connected to the
    branch with lower photon number. In this case, the Kerr detuning shift is negative, and thus `np.max` again selects
    this branch since it corresponds to a less negative value of the shift. For frequency sweeps in the negative
    direction, the above arguments are reversed and `np.min` is the recommended value. Note that the resonator is not
    guaranteed to stay in one of the bistable states, and it is recommended to collect time-ordered data acquired with
    a large bandwidth in order to detect jumps between the states.

    :param detuning: a 1D array[float] or single float; the fractional frequency detuning
    :param coupling_loss: a single float; the inverse coupling quality factor.
    :param internal_loss: a single float; the inverse internal quality factor.
    :param kerr_input: a 1D array[float] or single float; the rescaled input photon rate \chi described above.
    :param io_coupling_coefficient: a single float; the parameter g defined above.
    :param choose: a function used to choose which root to return when the cubic has multiple roots; it is called as
      choose(np.vstack((array_of_roots_0, ... , array_of_roots_n)), axis=0), with either two or three arrays; see above
      for recommended functions.
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
    d = -io_coupling_coefficient * coupling_loss * kerr_input
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


def absolute_kerr_input_at_bifurcation(coupling_loss, internal_loss, io_coupling_coefficient):
    return ((internal_loss + coupling_loss) ** 3
            / (3 ** (3 / 2) * io_coupling_coefficient * coupling_loss))


def kerr_given_input_rate(input_rate, resonance_frequency, kerr_input):
    return kerr_input * (2 * np.pi * resonance_frequency) ** 2 / input_rate


def input_rate_given_kerr(kerr_coefficient, resonance_frequency, kerr_input):
    return kerr_input * (2 * np.pi * resonance_frequency) ** 2 / kerr_coefficient


def photon_number(resonance_frequency, kerr_detuning_shift, kerr_input, input_rate):
    return (input_rate * kerr_detuning_shift) / (2 * np.pi * resonance_frequency * kerr_input)


# ToDo: add static methods to choose roots
# ToDo: add methods to calculate the kerr detuning and kerr frequency shift
class KerrFitter(base.ResonatorFitter):

    def __init__(self, frequency, data, choose, foreground_model=None, background_model=None, errors=None,
                 params=None, **fit_kwds):
        """
        :param choose: a numpy ufunc that chooses a number to return given multiple roots; see `kerr_detuning_shift`;
          this is baked into the model function when the model object is created, so if you want to compare the results
          of using another choose function, create a new fitter.
        """
        self._choose = choose  # Modifying this will lead to inconsistent results
        super(KerrFitter, self).__init__(frequency=frequency, data=data, foreground_model=foreground_model,
                                         background_model=background_model, errors=errors, params=params, **fit_kwds)

    def photon_number(self, input_frequency, input_rate, choose=None):
        if choose is None:
            choose = self._choose
        detuning = input_frequency / self.resonance_frequency - 1
        shift = kerr_detuning_shift(
            detuning=detuning, coupling_loss=self.coupling_loss, internal_loss=self.internal_loss,
            kerr_input=self.kerr_input, io_coupling_coefficient=self.foreground_model.io_coupling_coefficient,
            choose=choose)
        return photon_number(resonance_frequency=self.resonance_frequency, kerr_detuning_shift=shift,
                             kerr_input=self.kerr_input, input_rate=input_rate)

    def kerr_coefficient(self, input_rate):
        return kerr_given_input_rate(input_rate=input_rate, resonance_frequency=self.resonance_frequency,
                                     kerr_input=self.kerr_input)

    def kerr_coefficient_from_power(self, input_power_dBm):
        input_rate = 1e-3 * 10 ** (input_power_dBm / 10) / (h * self.resonance_frequency)
        return self.kerr_coefficient(input_rate=input_rate)

    def input_rate(self, kerr_coefficient):
        return input_rate_given_kerr(kerr_coefficient=kerr_coefficient, resonance_frequency=self.resonance_frequency,
                                     kerr_input=self.kerr_input)


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


# ToDo: verify that these always give the same result as np.max and np.min
# ToDo: are these used? parameter 'axis' value was not used
def maxabs(roots, axis=0):
    return roots[np.argmax(np.abs(roots), axis=axis), np.arange(roots.shape[1])]


def minabs(roots, axis=0):
    return roots[np.argmin(np.abs(roots), axis=axis), np.arange(roots.shape[1])]
