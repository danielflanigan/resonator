"""
This module contains the ResonatorFitter class and the MeasurementModelResonance object.
"""

from __future__ import absolute_import, division, print_function

from collections import namedtuple

import numpy as np
import lmfit

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

    def __init__(self, frequency, data, foreground_model, background_model=None, errors=None, **fit_kwds):
        """
        Fit the given data using the given resonator model.

        Parameters
        ----------
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
        :param fit_kwds: keyword arguments passed directly to lmfit.model.Model.fit(); see the lmfit documentation.
        """
        if not np.iscomplexobj(data):
            raise TypeError("Resonator data must be complex.")
        if background_model is None:
            background_model = background.One()
        if errors is not None and not np.iscomplexobj(errors):
            raise TypeError("Resonator errors must be complex.")
        if errors is None:
            weights = None
        else:
            weights = 1 / errors.real + 1j / errors.imag
        self.frequency = frequency
        self.data = data
        self.errors = errors
        self.weights = weights
        self.model = background_model * foreground_model
        self.result = self.fit()

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
        return "ResonatorFitter: {} * {}".format(self.background_model.__class__.__name__,
                                                 self.foreground_model.__class__.__name__)

    @property
    def background_model(self):
        return self.model.left

    @property
    def foreground_model(self):
        return self.model.right

    @property
    def foreground_data(self):
        return self.foreground_model.eval(self.result.params, frequency=self.frequency)

    @property
    def background_data(self):
        return self.background_model.eval(params=self.result.params, frequency=self.frequency)

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

    def fit(self, guess=None, **fit_kwds):
        """
        Fit the object's model to its data without overwriting the existing result. This function is called when the
        object is created. Note that this function does not modify the object's state, and the result must be stored to
        used by other methods, as in the examples below.

        Example: improve the initial parameters
        improved_guess = r.guess(frequency=self.frequency, data=self.data)  # Start with the default guess
        # improve on the guess somehow
        r.result = r.fit(guess=improved_guess)

        :param guess: a lmfit.Parameters object containing reasonable initial values; the default is created by calling
          self.guess(), which uses the guessing functions of the background then the foreground.
        :param fit_kwds: a dict of keywords passed directly to lmfit.model.Model.fit().
        :return: None
        """
        if guess is None:
            guess = self.guess(frequency=self.frequency, data=self.data)
        return self.model.fit(frequency=self.frequency, data=self.data, weights=self.weights, params=guess, **fit_kwds)

    def evaluate_model(self, frequency=None, params=None):
        if params is None:
            params = self.result.params
        if frequency is None:
            frequency = self.frequency
        return self.model.eval(frequency=frequency, params=params)

    def remove_background(self, frequency, data):
        """
        Normalize data to the appropriate plane by dividing it by the background evaluated at the given frequencies
        using the current best fit params.

        frequency : float or array of floats
            Frequency (in same units as the model was instantiated) at which to remove the background.
        data : complex or array of complex
            Raw data to be normalized.
        """
        return data / self.background_model.eval(params=self.result.params, frequency=frequency)

    def measurement_model_resonance(self, normalize=False, num_model_points=None):
        """
        Return a MeasurementModelResonance object (see above) containing three pairs of frequency and data values:
        - arrays containing the measurement frequencies and measured data;
        - arrays containing various frequencies within the span of the measurement frequencies and the model evaluated
          at these frequencies;
        - the model resonance frequency and the model evaluated at this frequency.

        Parameters
        ----------
        normalize : bool, default False
            If True, return all data values with the background model removed.
        num_model_points : int or None (default)
            The number of data points to use in evaluating the model over the span of the measurement frequencies;
            if None, evaluate the model at the measured frequencies.

        Returns
        -------
        MeasurementModelResonance, a namedtuple defined in this module
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

    def invert(self, time_ordered_data):
        """
        Invert the resonator model and return the time-ordered resonator parameters x(t) and Q_i^{-1}(t) that correspond
        to the given time-ordered data. These data should be normalized appropriately, depending on the resonance
        coupling configuration. For the shunt configuration, the data should equal 1 + 0j far from resonance. For the
        reflection configuration, the data should equal -1 + 0j far from resonance. For the transmission configuration,
        the data should be normalized in a way to be determined...

        Parameters
        ----------
        time_ordered_data : ndarray (complex)
            The normalized time-ordered data.

        Returns
        -------
        ndarray (real)
            The time-ordered values of the fractional frequency detuning x = f_g / f_r - 1, where f_g is the generator
            (or readout) frequency.
        ndarray (real)
            The time-ordered values of the inverse internal quality factor loss_i = 1 / Q_i.
        """
        raise NotImplementedError("Subclasses should implement this using their parameters.")

    def remove_background_and_invert(self, time_ordered_data, measurement_frequency):
        return self.invert(self.remove_background(frequency=measurement_frequency, data=time_ordered_data))

    # Aliases for common resonator properties
    # ToDo: add power flow calculations
    # ToDo: calculate photon number from power flows

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
