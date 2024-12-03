# Welcome to SBMLtoODEjax!

SBMLtoODEjax is a lightweight library that allows to automatically parse and convert SBML models into python models written end-to-end in [JAX](https://github.com/google/jax), 
a high-performance numerical computing library with automatic differentiation capabilities. 
SBMLtoODEjax is targeted at researchers that aim to incorporate SBML-specified ordinary differential equation (ODE) models into their python projects and machine learning pipelines, 
in order to perform efficient numerical simulation and optimization with only a few lines of code (by taking advantage of JAX’s core transformation features). 
For an overview of SBMLtoODEjax’s advantages and limitations, please check the [👀 Why use SBMLtoODEjax?](./design_principles.md) section.

SBMLtoODEjax extends [SBMLtoODEpy](https://github.com/AnabelSMRuggiero/sbmltoodepy), a python library developed in 2019
for converting SBML files into python files written in Numpy/Scipy. 
The chosen conventions for the generated variables and modules are slightly different from the standard SBML conventions (used in the SBMLtoODEpy library) 
with the aim here to accommodate for more flexible manipulations while preserving JAX-like functional programming style. 
For more details on the structure/conventions of the generated files, please check the [🎨 Design Principles](./design_principles.md) section.

## Quick start

:::{admonition} Install
:class: note
SBMLtoODEjax can be installed via pip. See our [Installation](./installation.md) guide for further details.
```bash
pip install sbmltoodejax
```
:::

:::{admonition} Begin
:class: seealso
Go have a look at our [Numerical Simulation](./tutorials/biomodels_curation.ipynb) tutorial to learn how to load and simulate SBML models.
:::

## License

The SBMLtoODEjax project is licensed under the [MIT license](https://github.com/flowersteam/sbmltoodejax/blob/main/LICENSE).

## Acknowledgements
SBMLtoODEjax builds on:
* [SBMLtoODEpy](https://github.com/AnabelSMRuggiero/sbmltoodepy)'s parsing and conversion of SBML files, by Steve M. Ruggiero and Ashlee N. Ford
* [JAX](https://github.com/google/jax)'s composable transformations, by the Google team
* [Equinox](https://github.com/patrick-kidger/equinox)'s module abstraction, by Patrick Kidger
* [BasiCO](https://github.com/copasi/basico/blob/d058c10dd51f2c3e926efeaa29c6194f86bfdc90/basico/biomodels.py)'s access the BioModels REST api, by the COPASI team

Our documentation was also inspired by the [GPJax](https://docs.jaxgaussianprocesses.com/) documentation, by Thomas Pinder and team.



```{toctree}
:caption: SBMLtoODEjax
:hidden: 
:maxdepth: 2

🏡 Home <self>
```

```{toctree}
:caption: Getting Started
:hidden: 
:maxdepth: 2

🛠️ Installation <installation.md>
👀 Why SBMLtoODEjax? <why_use.md>
🎨 Design Principles <design_principles.md>
🤝 Contributing <contributing.md>
📎 Jax 101 [external] <jax101.md>
```

```{toctree}
:caption: 🎓 Tutorials
:hidden:
:maxdepth: 2

tutorials/biomodels_curation
tutorials/parallel_execution
tutorials/benchmark
tutorials/gradient_descent
```

```{toctree}
:caption: Public API - sbmltoodejax package
:hidden:
:maxdepth: 2

api/parse
api/modulegeneration
api/jaxfuncs
api/biomodels_api
api/utils
```