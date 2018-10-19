"""
Functions and classes related to linear resonators.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from . import base


def photon_number(frequency, resonance_frequency, coupling_loss, internal_loss, input_rate, io_coupling_coefficient):
    detuning = frequency / resonance_frequency - 1
    return (4 * io_coupling_coefficient * coupling_loss / (coupling_loss + internal_loss) ** 2
            * 1 / (1 + 4 * detuning ** 2 / (coupling_loss + internal_loss) ** 2)
            * input_rate / (2 * np.pi * resonance_frequency))


class LinearResonatorFitter(base.ResonatorFitter):

    def photon_number(self, input_frequency, input_rate):
        return photon_number(frequency=input_frequency, resonance_frequency=self.resonance_frequency,
                             coupling_loss=self.coupling_loss, internal_loss=self.internal_loss, input_rate=input_rate,
                             io_coupling_coefficient=self.foreground_model.io_coupling_coefficient)
