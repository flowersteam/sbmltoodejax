# Design principles

`SBMLtoODEjax` automatically parses a given SBML file and convert it to a python file with several *variables* and *modules* (parametrized functions).

:::{admonition} New to the Systems Biology Markup Language (SBML)?
:class: seealso
SBML language is a standard format for representing mathematical models of biological systems. 
SBML files are written in XML format and contain information about the variables, parameters, and ODE-based equations that describe the behavior of the system 
(as well as additional metadata about the model).
For more information, you can have a look at the official [SBML website](https://sbml.org/).
:::

## Structure of the generated python file

### Variables 
| SBMLtoODEjax | Description                                                                                                                                                                                     |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| y                | vector of variables that represent the model's species amount (e.g. gene expression level) and that is evolving in time at a rate governed by the model's ODE-based reactions                   |
| w                | vector of non-constant parameters that intervene either in the model reactions or assignment rules (e.g. kinematic parameters) and whose state is evolving in time according to assigment rules |
| c                | vector of constant parameters that intervene either in the model reactions or assignment rules (e.g. kinematic parameters)                                                                      |
| t                | time serie of points for which to solve for y and w                                                                                                                                             |

### Modules
| SBMLtoODEjax   | Math                                                                   | Description                                                                                                                                                                                                 |
|-------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RateofSpeciesChange | $y(t), t, w(t) \mapsto \frac{dy(t)}{dt}$                               | system of ODE-governed equations that governs the rate of species changes $\frac{dy}{dt}$                                                                                                                   |
| AssignmentRule    | $w(t), y(t+\Delta t),  t+\Delta t \mapsto w(t+\Delta t)$               | system of equations that governs the temporal evolution of $w$                                                                                                                                              |
| ModelStep         | $y(t),  w(t), c, t, \mapsto y(t+\Delta t),  w(t+\Delta t), c, t+\Delta t$ | iteratively integrates `RateofSpeciesChange` using jax's [odeint](https://github.com/google/jax/blob/main/jax/experimental/ode.py) and calls `AssignmentRule` to update variables $y$ and $w$ in time |
| ModelRollout      | $y(0), w(0), c, t0, T \mapsto y[0..T], w[0..T]$                        | iteratively calls `ModelStep` for a given number of steps                                                                                                                                                   |
