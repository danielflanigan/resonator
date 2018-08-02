# Resonator
Fit and analyze data from resonators.

## Install
Check out the repository from GitHub:
```bash
/directory/for/code$ git clone https://github.com/danielflanigan/resonator.git
```
I recommend installing the package in editable mode:
```bash
/directory/for/code$ pip install -e resonator
```
Instead of moving the code to the site-packages directory, this command creates a link there that tells Python where to find the code.

The package requirements are matplotlib (only for the `see.py` plotting functions), numpy, and lmfit (at least 0.9.3).

## Quick start
Data from a resonator can be fit in a few lines of code. For example, to fit a resonator in the shunt-coupled (or "hanger") configuration, print data about the fit, then plot the data, fit, and response at resonance in the complex plane, do
```python
from matplotlib import pyplot as plt
from resonator import shunt, see
frequency, s21 = get_the_resonator_data()
r = shunt.ShuntFitter(frequency=frequency, data=s21)
print(r.result.fit_report())
fig, ax = plt.subplots()
see.real_and_imaginary(resonator=r, axes=ax)
``` 
where `frequency` is a numpy array of frequencies corresponding to the complex `s21` data array.

## Overview
The package is intended to quickly fit data from resonators even when the system gain and phase have not been characterized. The modules `reflection.py`, `shunt.py`, and `transmission.py` contain classes to fit data from resonators in the following coupling configurations: shunt-coupled (signal transmitted past resonator), reflection (signal reflected from resonator), and transmission (signal transmitted through resonator). The `background.py` module contains models for everything except for the resonator.

The models used in the fits are the product of the response of the resonator itself, called the foreground, and the background response of the system due to other effects such as loss and cable delay. This flexibility is enabled by the ability of `lmfit` to produce composite models. The example code above uses a default background model, called `ComplexConstant`, which takes the background magnitude and phase to be independent of frequency. To fit the same data with a more complex background model that will additionally fit for a time delay, do
```python
from resonator import background
r = shunt.ShuntFitter(frequency=frequency, data=s21, background_model=background.ConstantGainConstantDelay())
```
(Note that the background model is an **instance**, not a class.) All the fits are done in the complex plane, in which the noise from amplifiers should be isotropic. If the complex standard errors of the data points are available, these can be passed to the fitters. Since the default assumes equal, isotropic errors at each point, this is important only when the errors differ significantly.  

The quantities used directly in the models are inverse quality factors, which are called *losses* in the code. See the package `__init__.py` for discussion of this choice. Quality factors and standard errors are available as attributes of the fitters. For example, `r.Q_i` is the internal quality factor, and `r.coupling_loss_error` is the standard error of the coupling loss, or inverse coupling quality factor. Try `>>> dir(r)` to see a list of the available attributes.
