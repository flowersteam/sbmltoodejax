# Welcome to SBMLtoODEjax!

SBMLtoODEjax project builds upon the [SBMLtoODEpy](https://github.com/AnabelSMRuggiero/sbmltoodepy) project which allows to convert
Systems Biology Markup Language (SBML) models into python classes that can then easily be
simulated (and manipulated based on one's need) in python projects.

At the difference of SBMLtoODEpy, SBMLtoODEjax library is written end-to-end in [JAX](https://github.com/google/jax) 
and with different [design principles](./design_principles.md).

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
* [JAX](https://github.com/google/jax)'s main principles and computing performances, by the Google team
* [Equinox](https://github.com/patrick-kidger/equinox)'s module implementation, by Patrick Kidger
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
```