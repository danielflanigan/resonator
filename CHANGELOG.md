# Changelog

## [0.3.0] 2018-10-19
### Added
- This changelog.
- Code in `kerr.py` used for fitting resonators measured with sufficiently high power that a Kerr-type nonlinearity is important.
- Models `reflection.KerrReflection` and `shunt.KerrShunt` and corresponding fitters that use the Kerr model.
- Photon number calculations for both linear and Kerr nonlinear resonators in both shunt and reflection configurations.
- Example notebooks that demonstrate fitting with the Kerr model.
- Plotting function `see.triptych` that creates three plots.
- Background model `Magnitude` with only free magnitude.
- Method `base.initial_model_values` that evaluates the model with initial values, useful for debugging.
- Example notebooks `basic.ipynb` and `advanced.ipynb` that demonstrate basic fitting and some more advanced techniques.

### Changed
- The linear resonator model classes and fitters have been renamed:
- `Shunt` -> `LinearShunt` and `ShuntFitter` -> `LinearShuntFitter`
- `Reflection` -> `LinearReflection` and `ReflectionFitter` -> `LinearReflectionFitter`
- `SymmetricTransmission` -> `LinearSymmetricTransmission`
- Most of the backgrounds have been renamed by their free parameters:
- `UnitNorm` -> `Phase`
- `ComplexConstant` -> `MagnitudePhase`
- `ConstantMagnitudeConstantDelay` -> `MagnitudePhaseDelay`
- `LinearMagnitudeConstantDelay` -> `MagnitudeSlopeOffsetPhaseDelay`
- A `lmfit.Parameters` object passed to a fitter on creation now updates the initial parameters, giving the user full control over initial values, bounds, and which parameters are varied in the fit.
- `ResonatorFitter.fit` now overwrites the existing result instead of returning it.
- Plotting functions now give the axes reasonable labels by default.
- Energy loss rates ("kappas") have been renamed, since kappa is used to mean various things in papers.
- `background.MagnitudeSlopeOffsetPhaseDelay`: magnitude_reference renamed magnitude_offset
- `ResonatorFitter.evaluate_model` -> `model_values`

### Fixed
- `One`, `Phase`, and `MagnitudePhase` model functions accepted numpy scalars but not floats.