"""
This module contains various models for resonators that are subclasses of the lmfit.model.Model class.


"""
from __future__ import absolute_import, division, print_function

import numpy as np
import lmfit

from . import fitter


class Reflection(lmfit.model.Model):
    """
    This class models a resonator operated in reflection.
    """

    def __init__(self, *args, **kwargs):
        """
        Instantiate.

        :param args: arguments passed directly to lmfit.model.Model.
        :param kwargs: keyword arguments passed directly to lmfit.model.Model.
        """
        def func(frequency, resonance_frequency, internal_loss, coupling_loss):
            detuning = frequency / resonance_frequency - 1
            return -1 + (2 / (1 + (internal_loss + 2j * detuning) / coupling_loss))
        super(Reflection, self).__init__(func=func, *args, **kwargs)

    # ToDo: add prefix support
    def guess(self, data, **kwargs):
        frequency = kwargs.pop('frequency')
        params = self.make_params()
        params['resonance_frequency'].value = kwargs.get('resonance_frequency', np.mean(frequency))
        params['resonance_frequency'].min = frequency.min()
        params['resonance_frequency'].max = frequency.max()
        params['internal_loss'].value = kwargs.get('internal_loss', 1e-6)
        params['internal_loss'].min = 0
        params['coupling_loss'].value = kwargs.get('coupling_loss', 1e-2)
        params['coupling_loss'].min = 0
        return params


