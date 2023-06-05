# SBMLtoODEjax

## About
SBMLtoODEjax is a lightweight library that allows to automatically parse and convert SBML models into python models written end-to-end in [JAX](https://github.com/google/jax), 
a high-performance numerical computing library with automatic differentiation capabilities. 
SBMLtoODEjax is targeted at researchers that aim to incorporate SBML-specified ordinary differential equation (ODE) models into their python projects and machine learning pipelines, 
in order to perform efficient numerical simulation and optimization with only a few lines of code (by taking advantage of JAX’s core transformation features). 


SBMLtoODEjax extends [SBMLtoODEpy](https://github.com/AnabelSMRuggiero/sbmltoodepy), a python library developed in 2019
for converting SBML files into python files written in Numpy/Scipy. 
The chosen conventions for the generated variables and modules are slightly different from the standard SBML conventions (used in the SBMLtoODEpy library) 
with the aim here to accommodate for more flexible manipulations while preserving JAX-like functional programming style. 

*In short, SBMLtoODEjax facilitates the re-use of biological network models and their manipulation in python projects while tailoring them
to take advantage of JAX main features for efficient and parallel computations. For more details, check our [👀 Why use SBMLtoODEjax?](https://developmentalsystems.org/sbmltoodejax/why_use.html) 
and [🎨 Design Principles](https://developmentalsystems.org/sbmltoodejax/design_principles.html) pages.*

## Documentation
The documentation is available at https://developmentalsystems.org/sbmltoodejax/.
It provides details about SBMLtoODEjax’s
main design principles, advantages and limitations,
and the full API docs. 

<b>👉 The documentation includes various hands-on tutorials for:</b>
1. [Loading and simulating models from BioModels website](https://developmentalsystems.org/sbmltoodejax/tutorials/biomodels_curation.html)
2. [Running simulations in parallel for a batch of initial conditions](https://developmentalsystems.org/sbmltoodejax/tutorials/parallel_execution.html) 
3. [Using gradient descent to optimize model parameters](https://developmentalsystems.org/sbmltoodejax/tutorials/gradient_descent.html)


## Installation

The latest stable release of `SBMLtoODEjax` can be installed via `pip`:

```bash
pip install sbmltoodejax
```

Requires SBMLtoODEpy, JAX (cpu) and Equinox. 

## Quick Start
With only a few lines of python code, you can load and simulate existing SBML models.
Below is an example  code and output snapshot for reproducing simulation results of [biomodel #10](https://www.ebi.ac.uk/biomodels/BIOMD0000000010#Curation), from Kholodenko 2000’s paper. 
```python 
import matplotlib.pyplot as plt
from sbmltoodejax.utils import load_biomodel

# load and simulate model 
model, _, _, _ = load_biomodel(10)
n_secs = 150*60
n_steps = int(n_secs / model.deltaT)
ys, ws, ts = model(n_steps)

# plot time course simulation as in original paper
y_indexes = model.modelstepfunc.y_indexes
plt.figure(figsize=(6, 4))
plt.plot(ts/60, ys[y_indexes["MAPK"]], color="lawngreen", label="MAPK")
plt.plot(ts/60, ys[y_indexes["MAPK_PP"]], color="blue", label="MAPK-PP")
plt.xlim([0,150])
plt.ylim([0,300])
plt.xlabel("Reaction time (mins)")
plt.ylabel("Concentration")
plt.legend()
plt.show()
```
![biomodel #10 default simulation](./docs/source/_static/biomd_simulation.svg)
## Contributing
SBMLtoODEjax is in its early stage and any sort of contribution will be highly appreciated.
To learn how you can get involved, please read our [guide for contributing](https://developmentalsystems.org/sbmltoodejax/contributing.html).

## License

The SBMLtoODEjax project is licensed under the [MIT license](https://github.com/flowersteam/sbmltoodejax/blob/main/LICENSE).

## Acknowledgements
SBMLtoODEjax builds on:
* [SBMLtoODEpy](https://github.com/AnabelSMRuggiero/sbmltoodepy)'s parsing and conversion of SBML files, by Steve M. Ruggiero and Ashlee N. Ford
* [JAX](https://github.com/google/jax)'s composable transformations, by the Google team
* [Equinox](https://github.com/patrick-kidger/equinox)'s module abstraction, by Patrick Kidger
* [BasiCO](https://github.com/copasi/basico/blob/d058c10dd51f2c3e926efeaa29c6194f86bfdc90/basico/biomodels.py)'s access the BioModels REST api, by the COPASI team

Our documentation was also inspired by the [GPJax](https://docs.jaxgaussianprocesses.com/) documentation, by Thomas Pinder and team.
