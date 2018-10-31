# Resonator
Fit and analyze scattering parameter data from resonators.

## Overview
The response of a microwave system will include effects such as loss, amplifier gain, phase shifts, and cable delay.
The package can usually fit data from resonators with reasonably high quality factors even when the system gain and phase have not been characterized.
It does this by assuming that the measured scattering parameter is the product of an ideal resonator model, which is what one would measure in an imaginary on-chip microwave measurement, and the transmission of everything else in the system.
The system response is called the *background*, and the resonator response is called the *foreground*, so the models are of the form
`model = background * foreground`.
The user can choose between background models with different degrees of complexity, depending on how many non-ideal effects are necessary to describe the data.

The modules `reflection.py`, `shunt.py`, and `transmission.py` contain classes to fit data from resonators in the following coupling configurations: shunt-coupled (signal transmitted past resonator), reflection (signal reflected from resonator), and transmission (signal transmitted through resonator).
The module `background.py` contains models for everything except for the resonator.
The module `see.py` contains functions to plot resonator data and fits using `matplotlib`.
The `examples` folder contains Jupyter notebooks with detailed examples of fitting.

The fitting is done using [lmfit](https://lmfit.github.io/lmfit-py/), a fitting package that is built on routines in `scipy.optimize` but allows for more control and flexibility.

## Install
Check out the repository from [GitHub](https://github.com/danielflanigan/resonator):
```bash
/directory/for/code$ git clone https://github.com/danielflanigan/resonator.git
```
I recommend installing the package in editable mode:
```bash
/directory/for/code$ pip install -e resonator
```
Instead of moving the code to the site-packages directory, this command creates a link there that tells Python where to find the code.

The package requirements are lmfit >= 0.9.3, numpy, and matplotlib for the `see.py` plotting functions.
The code should run in Python 2.7 or 3.6+.

## Quick start
Data from a resonator can be fit and analyzed in a few lines of code.
For example, to fit a resonator in the shunt-coupled (or "hanger") configuration, print data about the fit, then plot the data, fit, and response at resonance in the complex plane, do the following:
```python
from matplotlib import pyplot as plt
from resonator import shunt, see
frequency, s21 = get_the_resonator_data()
r = shunt.LinearShuntFitter(frequency=frequency, data=s21)
print(r.result.fit_report())
fig, ax = plt.subplots()
see.real_and_imaginary(resonator=r, axes=ax)
``` 
where `frequency` is a numpy array of frequencies corresponding to the complex `s21` data array.
The scattering parameter models are parameterized in terms of resonator **inverse quality factors**, which are called *losses* in the code.
Thus, the fit report above will show parameters called `internal_loss` and `coupling_loss`, which are the inverses of the corresponding quality factors. 
See below for discussion of this choice.
The fitter object `r` makes the best-fit parameters as well as quality factors, energy decay rates, and the standard errors of all of these available for attribute access.
Try `print(dir(r))` to see a list of the available attributes.
For example, `r.Q_i` is the internal quality factor, and `r.coupling_energy_decay_rate_error` is the standard error of the coupling energy decay rate. 

The example code above uses a default background model, called `MagnitudePhase`, which takes the background magnitude and phase to be independent of frequency.
To fit the same data with a more complex background model that will additionally fit for a time delay, do the following:
```python
from resonator import background
r = shunt.LinearShuntFitter(frequency=frequency, data=s21, background_model=background.MagnitudePhaseDelay())
```
(Note that the background model is an **instance**, not a class.)
All the fits are done in the complex plane, in which the noise from amplifiers should be isotropic. If the complex standard errors of the data points are available, these can be passed to the fitters.
Since the default assumes equal, isotropic errors at each point, this is important only when the errors differ significantly.  

## Internal calculations
Models for resonators are typically written either in terms of quality factors (e.g. Q_internal, Q_external) or in terms
of energy decay rates that are equal to the resonance angular frequency divided by a quality factor (e.g.
kappa_external = omega_r / Q_external)).

The resonator models in this package use inverse quality factors, which are called "losses" in the code.
These have the expected definition:
```
loss_x = 1 / Q_x = P_x / (omega_r * E),
```
where P_x is the power lost to channel x, omega_r is the resonance angular frequency, and E is the total energy stored in the resonator.
For calculations, these are more useful than quality factors because energy losses to independent channels simply add.
For example, if Q_i is the internal quality factor and Q_c is the coupling (or "external") quality factor, then
internal_loss = 1 / Q_i,
coupling_loss = 1 / Q_c,
and thus
total_loss = internal_loss + coupling_loss,
the inverse of the total (or "resonator" or "loaded") quality factor.
Inverse quality factors are preferred here over the energy decay rates because they are dimensionless.

In order to make this choice transparent to users, the ResonatorFitter class (and thus all of its subclasses) has
properties that calculate the quality factors and energy decay rates as well as their standard errors.
