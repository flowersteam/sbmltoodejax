# Offline assignment-rule models

These synthetic SBML Level 3 Version 2 models are original test fixtures, not
BioModels downloads or reproductions of a biological study. All use a fixed
2-litre compartment, initial amounts A = 2 mol and B = 0 mol, and one irreversible
conversion A → B. They have no events, custom functions, or varying compartments.
The tests check libSBML consistency before simulation.

| File | Reaction rate (mol/s) | Rules | Exact A amount (mol) |
| --- | --- | --- | --- |
| `constant.xml` | k0 A | k0 = 1/s, constant | 2 exp(-t) |
| `time_dependent.xml` | k A | k = k0 + alpha t; k0 = 1/s, alpha = 1/s² | 2 exp(-t - t²/2) |
| `state_dependent.xml` | k A | k = k0 A; k0 = 1/(mol s) | 2/(1 + 2t) |
| `chained.xml` | k A | k = q; q = k0 + alpha t; dependent rule listed first | 2 exp(-t - t²/2) |
| `assigned_species.xml` | k0 U A | U amount = u0 + growth t; u0 = 1 mol, growth = 1 mol/s | 2 exp(-t - t²/2) |
| `concentration_species.xml` | k0 [U] [A] V | U concentration = u0 + growth t; u0 = 1 mol/l, growth = 1 mol/(l s) | 2 exp(-t - t²/2) |
| `rate_rule.xml` | k A | dk/dt = alpha = 1/s², k(0) = 1/s | 2 exp(-t - t²/2) |

In every case B = 2 - A. The time variable in these compact formulas is its
numeric value in seconds. In `assigned_species`, k0 = 1/(mol s); in
`concentration_species`, k0 = 1 l/(mol s). The latter uses concentration-valued
species symbols and initial concentrations A = 1 mol/l, B = 0, U = 1 mol/l.
The generated JAX vectors still store amounts. RoadRunner's bare species IDs
select amounts; COPASI's concentration outputs are multiplied by the fixed volume.
The non-unit volume makes an accidental amount/concentration comparison visible.
