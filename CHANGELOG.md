# Changelog

## [unreleased]
### Added
- This changelog.
- New models reflection.ReflectionNonlinear and shunt.ShuntNonlinear and corresponding fitters that can fit data from
resonators measured with sufficiently high power that a Kerr-type nonlinearity is important.
- New example notebooks

### Changed
- Added support for 
- ResonatorFitter.fit() now overwrites the result instead of returning it.
- Plotting functions now give the axes reasonable labels by default.
- Energy loss rates ("kappas") have been renamed, since kappa is used to mean various things in papers.
- background.LinearMagnitudeConstantDelay: magnitude_reference renamed magnitude_offset

### Fixed
- One, UnitNorm, and ComplexConstant model functions accepted numpy scalars but not floats.

