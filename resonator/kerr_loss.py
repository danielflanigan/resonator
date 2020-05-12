"""
Functions and classes related to resonators with a Kerr-type nonlinearity and a nonlinear loss.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from . import base


def photon_number(detuning, coupling_loss, internal_loss, nonlinear_loss, reduced_kerr, reduced_input_rate,
                  io_coupling_coefficient, choose):
    """
    Return one chosen real root of a cubic polynomial that arises when considering a resonator with hamiltonian
      H = \hbar \omega_r a^\dag a + (\hbar K / 2) a^\dag a^\dag a a,
    where \omega_r = 2 \pi f_r is the resonance angular frequency, K is the Kerr coefficient, and a (a^\dag) is the
    annihilation (creation) operator for photons in the resonator. The equation of motion for the annihilation operator
    leads to a cubic equation for the photon number in the resonator n = <a^\dag a>. The equation used here is
    equivalent to Equation 37 of B. Yurke and E. Buks Journal of Lightwave Technology 24, 5054 (2006):
      0 = a n^3 + b n^2 + c n + d
        = [k^2 + \loss_n^2 / 4]  n^3 - [2 x k + (loss_i + loss_c) loss_n / 4] n^2 + [(loss_i + loss_c)^2 / 4 + x^2] n
          - g loss_c x_in,
    where the variables have the following meanings:
      x = f / f_r - 1 = detuning -- the dimensionless fractional frequency detuning, f is the signal frequency and f_r
        is the resonance frequency;
      loss_c = coupling_loss -- the coupling loss (inverse coupling quality factor);
      loss_i = internal_loss -- the internal loss (inverse internal quality factor);
      loss_n = nonlinear_loss -- the nonlinear loss;
      k = K / \omega_r = reduced_kerr -- the Kerr coefficient K divided by the resonance angular frequency;
      x_in = X_in / \omega_r = reduced_input_rate -- the reduced input rate, where the input photon rate X_{in} means
    the rate (in s^{-1}) at which photons arrive from the input port, typically numbered port 1;
      g = io_coupling_coefficient -- a number of order 1 that depends on the coupling geometry.
    The input power is P_{in} = \hbar \omega X_{in}.

    # ToDo: clean up
    The `choose` function selects one real root when there are multiple real roots that correspond to multiple stable
    photon number states in the resonator. The recommended value of this function is `np.min` when fitting data taken
    with a VNA that sweeps the frequency in the positive direction, as is typically done. If the frequency is swept in
    the negative direction, the recommended value is `np.min`.

    Note that the resonator is not
    guaranteed to stay in one of the bistable states, and it is recommended to collect time-ordered data acquired with
    a large bandwidth in order to detect jumps between the states.

    :param detuning: a 1D array[float] or single float; the fractional frequency detuning
    :param coupling_loss: a single float; the inverse coupling quality factor.
    :param internal_loss: a single float; the inverse internal quality factor.
    :param nonlinear_loss: a
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
    roots = [one_photon_number(x, coupling_loss, internal_loss, nonlinear_loss, reduced_kerr,
                               reduced_input_rate, io_coupling_coefficient, choose)
             for x in detuning]
    if is_scalar or is_zero_size:
        return roots[0]
    else:
        return np.array(roots)


def one_photon_number(detuning, coupling_loss, internal_loss, nonlinear_loss, reduced_kerr, reduced_input_rate,
                      io_coupling_coefficient, choose):
    return choose(photon_number_roots(
        detuning=detuning, coupling_loss=coupling_loss, internal_loss=internal_loss, nonlinear_loss=nonlinear_loss,
        reduced_kerr=reduced_kerr, reduced_input_rate=reduced_input_rate,
        io_coupling_coefficient=io_coupling_coefficient))


def photon_number_roots(detuning, coupling_loss, internal_loss, nonlinear_loss, reduced_kerr, reduced_input_rate,
                        io_coupling_coefficient):
    return np.roots(photon_number_cubic(
        detuning=detuning, coupling_loss=coupling_loss, internal_loss=internal_loss, nonlinear_loss=nonlinear_loss,
        reduced_kerr=reduced_kerr, reduced_input_rate=reduced_input_rate,
        io_coupling_coefficient=io_coupling_coefficient))


def photon_number_cubic(detuning, coupling_loss, internal_loss, nonlinear_loss, reduced_kerr, reduced_input_rate,
                        io_coupling_coefficient):
    a = reduced_kerr ** 2 + nonlinear_loss ** 2 / 4
    b = -(2 * detuning * reduced_kerr + (coupling_loss + internal_loss) * nonlinear_loss / 4)
    c = ((coupling_loss + internal_loss) / 2) ** 2 + detuning ** 2
    d = -io_coupling_coefficient * coupling_loss * reduced_input_rate
    return np.array([a, b, c, d])


def choose_min(roots):
    return np.min(roots[roots.imag == 0].real)


def choose_max(roots):
    return np.max(roots[roots.imag == 0].real)


class KerrLossFitter(base.ResonatorFitter):

    def __init__(self, frequency, data, choose, foreground_model=None, background_model=None, errors=None, params=None,
                 **fit_kwds):
        """
        :param choose: a numpy ufunc that chooses a number to return given multiple roots; see `photon_number`; this
          is baked into the model function when the model object is created, so if you want to compare the results of
          using another choose function, create a new fitter.
        """
        self._choose = choose  # Modifying this will lead to inconsistent results
        super(KerrLossFitter, self).__init__(frequency=frequency, data=data, foreground_model=foreground_model,
                                             background_model=background_model, errors=errors, params=params,
                                             **fit_kwds)

    def photon_number(self, input_frequency, choose=None):
        if choose is None:
            choose = self._choose
        return photon_number(detuning=input_frequency / self.resonance_frequency - 1, coupling_loss=self.coupling_loss,
                             internal_loss=self.internal_loss, nonlinear_loss=self.nonlinear_loss,
                             reduced_kerr=self.reduced_kerr, reduced_input_rate=self.reduced_input_rate,
                             io_coupling_coefficient=self.foreground_model.io_coupling_coefficient, choose=choose)

    @property
    def kerr_coefficient(self):
        return 2 * np.pi * self.resonance_frequency * self.reduced_kerr

    # ToDo: this is assumed to be fixed at all frequencies...
    @property
    def input_rate(self):
        return 2 * np.pi * self.resonance_frequency * self.reduced_input_rate
