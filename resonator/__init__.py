"""
Models for resonators are typically written either in terms of quality factors (e.g. Q_internal, Q_external) or in terms
of quantities (often denoted by \kappa) that represent the resonance angular frequency divided by a quality factor (e.g.
\kappa_external = \omega_r / Q_external)).

The resonator models in this package use inverse quality factors, which are called "losses" in the code. These are more
useful than quality factors because energy losses to independent channels simply add. For example, if Q_i is the
internal quality factor and Q_c is the coupling (or "external") quality factor, then
internal_loss = 1 / Q_i,
coupling_loss = 1 / Q_c,
and thus
total_loss = internal_loss + coupling_loss,
the inverse of the total (or "resonator" or "loaded") quality factor.

Inverse quality factors are also more useful for fitting than the "kappas" because they are independent of the resonance
frequency, and common physical effects may alter the resonance frequency without altering the energy flow out of the
resonator, or alter the dissipation without altering the resonance frequency.

In order to make this choice transparent to users, the ResonatorFitter class (and thus all of its subclasses) has
properties that calculate the quality factors and kappas as well as their standard errors.
"""