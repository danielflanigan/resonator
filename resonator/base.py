"""
This module contains the ResonatorFitter class and the MeasurementModelResonance object.
"""
from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np

from . import background


# A container for both measured and fit data that is used by the see.py plotting module.
# The data arrays may either all have the background divided out or not.
# The measurement_ arrays are the measurement frequency and corresponding response data;
# The model_ arrays are the model frequency and response data evaluated there;
# The resonance_ points are the resonance frequency (float) and the model (complex) evaluated there.
MeasurementModelResonance = namedtuple('MeasurementModelResonance',
                                       field_names=['measurement_frequency', 'measurement_data',
                                                    'model_frequency', 'model_data',
                                                    'resonance_frequency', 'resonance_data'])


class ResonatorFitter(object):
    """
    This class is a wrapper for composite models that represent the response of a resonator multiplied by the background
    response of the system. Its subclasses represent models for resonators used in specific configurations.
    """

    def __init__(self, frequency, data, foreground_model, background_model=None, errors=None, params=None, **fit_kwds):
        """
        Fit the given data using the given resonator model.

        :param frequency: an array of floats containing the frequencies at which the data was measured.
        :param data: an array of complex numbers containing the data, probably forward transmission S_{21} or forward
          reflection S_{11}.
        :param foreground_model: an instance (not the class) of a lmfit.model.Model subclass representing the resonator
          to be fit.
        :param background_model: an instance (not the class) of a lmfit.model.Model subclass representing the
          background, meaning everything other than the target resonator(s), including effects such
          as gain, cable delay, and other resonances; the default of background.One assumes that the data have been
          perfectly calibrated to the resonator plane, which is unlikely to be the case in practice.
        :param errors: Standard error of the mean for the real and imaginary parts of the data, used to assign weights
          in the least-squares fit; the default of None means to use equal errors and thus equal weights for each point;
          to exclude a point, set the errors to (1 + 1j) * np.inf for that point.
        :param params: a lmfit.Parameters object containing Parameters to use as initial values for the fit; these are
          passed to fit() and will overwrite Parameters with the same names obtained from guess().
        :param fit_kwds: keyword arguments passed directly to lmfit.model.Model.fit(), except for params, as explained
          above; see the lmfit documentation.
        """
        if not np.iscomplexobj(data):
            raise TypeError("Resonator data must be complex.")
        if background_model is None:
            background_model = background.One()
        if errors is not None and not np.iscomplexobj(errors):
            raise TypeError("Resonator errors must be complex.")
        self.frequency = frequency
        self.data = data
        self.errors = errors
        self.model = background_model * foreground_model  # lmfit.CompositeModel
        self.result = None  # This is updated immediately by the next line
        self.fit(params=params, **fit_kwds)

    def __getattr__(self, attr):
        if attr.endswith('_error'):
            name = attr[:-len('_error')]
            try:
                return self.result.params[name].stderr
            except KeyError:
                raise AttributeError("Couldn't find error for {} in self.result".format(name))
        else:
            try:
                return self.result.params[attr].value
            except KeyError:
                raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, attr))

    def __dir__(self):
        return sorted(set(dir(ResonatorFitter) +
                          list(self.__dict__.keys()) +
                          list(self.result.params.keys()) +
                          [name + '_error' for name in self.result.params.keys()]))

    def __str__(self):
        return "{}: {} * {}".format(self.__class__.__name__, self.background_model.__class__.__name__,
                                    self.foreground_model.__class__.__name__)

    @property
    def weights(self):
        """
        The weights, calculated from self.errors, that are used to weight the residuals.
        See https://github.com/numpy/numpy/issues/5261
        """
        if self.errors is None:
            return None
        else:
            return 1 / self.errors.real + 1j / self.errors.imag

    @property
    def background_model(self):
        """The lmfit.Model object representing the background."""
        return self.model.left

    @property
    def foreground_model(self):
        """The lmfit.Model object representing the foreground."""
        return self.model.right

    @property
    def background_model_values(self):
        """The background model evaluated at the measurement frequencies with the best-fit parameters."""
        return self.background_model.eval(params=self.result.params, frequency=self.frequency)

    @property
    def foreground_model_values(self):
        """The foreground model evaluated at the measurement frequencies with the best-fit parameters."""
        return self.foreground_model.eval(self.result.params, frequency=self.frequency)

    # ToDo: simplify by dividing data passed to background by reference_point
    def guess(self, frequency, data):
        """
        Use the frequency and data arrays to make a reasonable guess at the best-fit values, in order to provide a good
        initial condition for the fit.

        :param frequency: an array of floats that are the frequencies corresponding to the data.
        :param data: an array of complex data values.
        :return: a lmfit.Parameters object.
        """
        guess = self.background_model.guess(data=data, frequency=frequency,
                                            reference_point=self.foreground_model.reference_point)
        background_guess = self.background_model.eval(params=guess, frequency=frequency)
        guess.update(self.foreground_model.guess(data=data / background_guess, frequency=frequency))
        return guess

    def fit(self, params=None, **fit_kwds):
        """
        Fit the object's model to its data, overwriting the existing result.

        Example: improve the initial parameters
        improved_guess = r.guess(frequency=self.frequency, data=self.data)  # Start with the default guess
        # improve on the guess somehow
        r.result = r.fit(guess=improved_guess)

        :param params: a lmfit.Parameters object containing Parameters to use as initial values for the fit;
          self.guess(), which uses the guessing functions of first the background and then the foreground.
        :param fit_kwds: a dict of keywords passed directly to lmfit.model.Model.fit().
        :return: None
        """
        initial_params = self.guess(frequency=self.frequency, data=self.data)
        if params is not None:
            initial_params.update(params)
        self.result = self.model.fit(frequency=self.frequency, data=self.data, weights=self.weights,
                                     params=initial_params, **fit_kwds)

    def evaluate_model(self, frequency=None, params=None):
        """
        Evaluate the total model (background * foreground) at the given frequencies with the given parameters.

        :param frequency: float or array of floats; the default is to use the frequencies corresponding to the data.
        :param params: lmfit.Parameters object; the default is to use the current best-fit parameters.
        :return: array of complex.
        """
        if params is None:
            params = self.result.params
        if frequency is None:
            frequency = self.frequency
        return self.model.eval(frequency=frequency, params=params)

    def remove_background(self, frequency, data):
        """
        Normalize data to the foreground plane by dividing it by the background evaluated at the given frequencies
        using the current best fit params.

        The returned data should produce a circle somewhere in the complex plane. For nonlinear resonators, part of the
        circle may be missing.

        :param frequency: float or array of floats representing frequencies corresponding to the given data.
        :param data: complex or array of complex data to be normalized.
        """
        return data / self.background_model.eval(params=self.result.params, frequency=frequency)

    # ToDo: replace with individual methods
    def measurement_model_resonance(self, normalize=False, num_model_points=None):
        """
        Return a MeasurementModelResonance object (see above) containing three pairs of frequency and data values:
        - arrays containing the measurement frequencies and measured data;
        - arrays containing various frequencies within the span of the measurement frequencies and the model evaluated
          at these frequencies;
        - the model resonance frequency and the model evaluated at this frequency.

        :param normalize: If True, return all data values with the background model removed.
        :param num_model_points: The number of frequencies to use in evaluating the model between the minimum and
          maximum measurement frequencies; if None (default), evaluate the model at the measurement frequencies.
        :return: A MeasurementModelResonance namedtuple containing frequency and data arrays.
        """
        measurement_frequency = self.frequency.copy()
        measurement_data = self.data.copy()
        if num_model_points is None:
            model_frequency = self.frequency.copy()
        else:
            model_frequency = np.linspace(measurement_frequency.min(), measurement_frequency.max(), num_model_points)
        model_data = self.model.eval(params=self.result.params, frequency=model_frequency)
        resonance_data = self.model.eval(params=self.result.params, frequency=self.resonance_frequency)
        if normalize:
            measurement_data = self.remove_background(frequency=measurement_frequency, data=measurement_data)
            model_data = self.remove_background(frequency=model_frequency, data=model_data)
            resonance_data = self.remove_background(frequency=self.resonance_frequency, data=resonance_data)
        return MeasurementModelResonance(measurement_frequency, measurement_data,
                                         model_frequency, model_data,
                                         self.resonance_frequency, resonance_data)

    # ToDo: include background inversion.
    def invert(self, scattering_data):
        """
        Return the resonator detuning and internal_loss that correspond to the given data, obtained by inverting the
        resonator model.

        Many parameters of superconducting microresonators are constant under different measurement conditions. For
        example, the coupling loss is typically defined lithographically. However, the resonance frequency (and thus
        the detuning from the measurement frequency) and the internal dissipation can vary due to various physical
        effects, such as a changing magnetic field or a changing density of quasiparticles. When using a resonator as
        a transducer, these detuning and dissipation signals are the desired quantities.

        This calculation assumes that only the detuning and internal loss vary in time. It also currently assumes
        that the detuning excursions are sufficiently small that the background can be treated as constant,
        though this effect could be included with a more complicated calculation. In this case, when measuring at
        constant frequency (i.e. in continuous-wave mode with a VNA), each measured complex number in the scattering
        parameter complex plane (S21 or S11) corresponds to a point in the complex plane defined by
          z = internal_loss + 2j * detuning.
        This quantity appears in all of the resonator models used in this package. In order for this analytic inversion
        of the scattering data to be strictly accurate, the data bandwidth must be less than the single-sided
        resonator bandwidth given by
          f_ss = f_r * (coupling_loss + internal_loss).
        If this is not the case, a more complicated calculation involving the resonator transfer function may be
        required. See J. Zmuidzinas, Annu. Rev. Cond. Matter Phys. 3, 169 (2012), available at
        https://doi.org/10.1146/annurev-conmatphys-020911-125022

        The scattering data must be normalized to the foreground (resonator) plane. That is, for a shunt-coupled
        resonator the data should equal 1 + 0j far from resonance; for a reflection from a resonator the data should
        equal -1 + 0j far from resonance; and for the transmission configuration, the data should equal
        1 / (1 + internal_loss / coupling_loss) + 0j exactly at the resonance. Raw data taken in the same configuration
        as the data used can be analyzed using remove_background_and_invert().

        :param scattering_data: Normalized scattering data, typically time-ordered.
        :return: detuning, internal_loss; both array[float], calculated by inverting the the resonator model.
        """
        raise NotImplementedError("Subclasses should implement this using their parameters.")


    def remove_background_and_invert(self, raw_scattering_data, measurement_frequency):
        """
        Return the resonator detuning and internal_loss that correspond to the given data, obtained by inverting the
        resonator model. The given data array is normalized to the resonator plane by dividing it by the single complex
        background value at the given measurement frequency, and the resulting values are passed to invert().

        :param raw_scattering_data: Raw scattering data, typically time-ordered.
        :param measurement_frequency: the frequency at which the scattering data was measured.
        :return: detuning, internal_loss; see invert().
        """
        return self.invert(self.remove_background(frequency=measurement_frequency, data=raw_scattering_data))

    # ToDo: add photon number calculation

    # Aliases for common resonator properties

    @property
    def f_r(self):
        return self.resonance_frequency

    @property
    def f_r_error(self):
        return self.resonance_frequency_error

    @property
    def omega_r(self):
        return 2 * np.pi * self.resonance_frequency

    @property
    def omega_r_error(self):
        return 2 * np.pi * self.resonance_frequency_error

    @property
    def total_loss(self):
        """The total loss is the inverse of the total / loaded / resonator quality factor."""
        return self.internal_loss + self.coupling_loss

    @property
    def total_loss_error(self):
        """Assume that the errors of the internal loss and coupling loss are independent."""
        return (self.internal_loss_error ** 2 + self.coupling_loss_error ** 2) ** (1 / 2)

    @property
    def Q_i(self):
        return 1 / self.internal_loss

    @property
    def Q_i_error(self):
        return self.internal_loss_error / self.internal_loss ** 2

    @property
    def Q_c(self):
        return 1 / self.coupling_loss

    @property
    def Q_c_error(self):
        return self.coupling_loss_error / self.coupling_loss ** 2

    @property
    def Q_r(self):
        return 1 / (self.internal_loss + self.coupling_loss)

    @property
    def Q_r_error(self):
        return self.total_loss_error / self.total_loss ** 2

    @property
    def kappa_i(self):
        return self.omega_r * self.internal_loss

    @property
    def kappa_i_error(self):
        """Assume that the errors of the resonance frequency and internal loss are independent."""
        return self.kappa_i * ((self.resonance_frequency_error / self.resonance_frequency) ** 2 +
                               (self.internal_loss_error / self.internal_loss) ** 2) ** (1 / 2)

    @property
    def kappa_c(self):
        return self.omega_r * self.coupling_loss

    @property
    def kappa_c_error(self):
        """Assume that the errors of the resonance frequency and coupling loss are independent."""
        return self.kappa_c * ((self.resonance_frequency_error / self.resonance_frequency) ** 2 +
                               (self.coupling_loss_error / self.coupling_loss) ** 2) ** (1 / 2)

    @property
    def kappa_r(self):
        return self.omega_r * (self.internal_loss + self.coupling_loss)

    @property
    def kappa_r_error(self):
        """Assume that the errors of the resonance frequency, internal loss, and coupling loss are independent."""
        return self.kappa_r * ((self.resonance_frequency_error / self.resonance_frequency) ** 2 +
                               (self.total_loss_error / self.total_loss) ** 2) ** (1 / 2)
