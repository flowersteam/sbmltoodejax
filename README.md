<p align="center">
<img width="700" height="300" src="https://raw.githubusercontent.com/flowersteam/sbmltoodejax/main/docs/source/_static/logo.svg" alt="SBMLtoODEjax's logo">
</p>

[![PyPI version](https://badge.fury.io/py/SBMLtoODEjax.svg)](https://badge.fury.io/py/SBMLtoODEjax)
[![Downloads](https://pepy.tech/badge/sbmltoodejax)](https://pepy.tech/project/sbmltoodejax)
[![arXiv](http://img.shields.io/badge/qbio.BM-arXiv%2307.08452-B31B1B.svg)](https://arxiv.org/abs/2307.08452)

# About

SBMLtoODEjax is a lightweight library that allows to automatically parse and convert SBML models into python models
written end-to-end in [JAX](https://github.com/google/jax),
a high-performance numerical computing library with automatic differentiation capabilities.
SBMLtoODEjax is targeted at researchers that aim to incorporate SBML-specified ordinary differential equation (ODE)
models into their python projects and machine learning pipelines,
in order to perform efficient numerical simulation and optimization with only a few lines of code (by taking advantage
of JAX’s core transformation features).

SBMLtoODEjax extends [SBMLtoODEpy](https://github.com/AnabelSMRuggiero/sbmltoodepy), a python library developed in 2019
for converting SBML files into python files written in Numpy/Scipy.
The chosen conventions for the generated variables and modules are slightly different from the standard SBML
conventions (used in the SBMLtoODEpy library)
with the aim here to accommodate for more flexible manipulations while preserving JAX-like functional programming style.

> *👉 In short, SBMLtoODEjax facilitates the re-use of biological network models and their manipulation in python projects
while tailoring them to take advantage of JAX main features for efficient and parallel computations.*

> 📖 The documentation, notebook tutorials and public APU are available at https://developmentalsystems.org/sbmltoodejax/.

# Installation

The latest stable release of `SBMLtoODEjax` can be installed via `pip`:

```bash
pip install sbmltoodejax
```

Requires SBMLtoODEpy, JAX (cpu) and Equinox.

# Why use SBMLtoODEjax?

## Simplicity and extensibility
SBMLtoODEjax retains the simplicity of the original [SBMLtoODEPy](https://github.com/AnabelSMRuggiero/sbmltoodepy) library to facilitate incorporation and refactoring of the
ODE models into one’s own python
projects. As shown below, with only a few lines of python code one can load and simulate existing SBML
files.

![Figure 1](https://raw.githubusercontent.com/flowersteam/sbmltoodejax/main/paper/fig1.png)
<font size="2" color="gray" > Example code (left) and output snapshot (right) reproducing [original simulation results](https://www.ebi.ac.uk/biomodels/BIOMD0000000010#Curation)
of Kholodenko 2000's paper hosted on BioModels website.
</font>

> **👉 Check our [Numerical Simulation](tutorials/biomodels_curation.ipynb) tutorial to reproduce results yourself and see more examples.**


## JAX-friendly
The generated python models are tailored to take advantage of JAX main features.
```python
class ModelRollout(eqx.Module):
    
    def __call__(self, n_steps, y0, w0, c, t0=0.0):

        @jit # use of jit transformation decorator
        def f(carry, x):
            y, w, c, t = carry
            return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
        
        # use of scan primitive to replace for loop (and reduce compilation time)
        (y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps)) 
        ys = jnp.moveaxis(ys, 0, -1)
        ws = jnp.moveaxis(ws, 0, -1)
        
        return ys, ws, ts
```
As shown above, model rollouts use `jit` transformation and `scan` primitive to reduce compilation and execution time of
the recursive ODE integration steps, which is particularly useful when running large numbers of steps (long reaction
times). Models also inherit from the [Equinox module abstraction](https://docs.kidger.site/equinox/api/module/module/) and are registered as PyTree
containers, which facilitates the application of JAX core transformations to any SBMLtoODEjax object.

## Efficiency simulation and optimization
The application of JAX core transformations, such as just-in-time
compilation (`jit`), automatic vectorization (`vmap`) and automatic differentiation (`grad`), to the generated models make it very easy (and
seamless) to efficiently run simulations in parallel. 

For instance, as shown below, with only a few lines of python
code one can vectorize calls to model rollout and perform batched computations efficiently, which is particularly useful when
considering large batch sizes.
![Figure 2](https://raw.githubusercontent.com/flowersteam/sbmltoodejax/main/paper/fig2.png)
 <font size="2" color="gray"> (left) Example code to vectorize calls to model rollout
(right) Results of a (rudimentary) benchmark comparing the average simulation time of models implemented with SBMLtoODEpy
versus SBMLtoODEjax (for different number of rollouts i.e. batch size).
</font>

> **👉 Check our [Benchmarking](https://developmentalsystems.org/sbmltoodejax/tutorials/benchmark.html) notebook for additional details on the benchmark results**.


Finally, as shown below, SBMLtoODEjax models can also be integrated within [Optax](https://github.com/deepmind/optax) pipelines,
a gradient processing and optimization library for [JAX](https://github.com/google/jax), allowing to optimize model parameters and/or
external interventions with stochastic gradient descent.

![Figure 3](https://raw.githubusercontent.com/flowersteam/sbmltoodejax/main/paper/fig3.png)
<font size="2" color="gray"> (left) Default simulation results of [biomodel #145](https://www.ebi.ac.uk/biomodels/BIOMD0000000145) which models ATP-induced intracellular calcium oscillations,
and target sine-wave pattern for Ca_Cyt concentration.
(middle) Training loss obtained when running the Optax optimization loop, with Adam optimizer, over the model kinematic parameters *c*.
(right) Simulation results obtained after optimization. 
</font>

> **👉 Check our [Gradient Descent](https://developmentalsystems.org/sbmltoodejax/tutorials/gradient_descent.html) tutorial to reproduce the result yourself and try more-advanced optimization usages.**

# All contributions are welcome!

SBMLtoODEjax is in its early stage and any sort of contribution will be highly appreciated.


## Suggested contributions
They are several use cases that are not handled by the current codebase including:
1) **Events**: SBML files with events (discrete occurrences that can trigger discontinuous changes in the model) are not handled
2) **Math Functions**: we handle a large portion, but not all, of functions possibly-used in SBML files (see `mathFuncs` in `sbmltoodejax.modulegeneration.GenerateModel`)
3) **Custom solvers**: To integrate the model's equation, we use jax experimental `odeint` solver but do not yet allow for other solvers.
4) **NaN/Negative values**: numerical simulation sometimes leads to NaN values (or negative values for the species amounts) which could either be due to wrong parsing or solver issues

This means that a large portion of the possible SBML files cannot yet be simulated, for instance as we detail on the below image, out of 1048
curated models that one can load from the BioModels website, only 232 can successfully be simulated (given the default initial conditions) in SBMLtoODEjax:
<img src="https://raw.githubusercontent.com/flowersteam/sbmltoodejax/main/docs/source/_static/error_cases.png" width="450px">

> 👉 Please consider contributing and check our [Contribution Guidelines](https://developmentalsystems.org/sbmltoodejax/contributing.html) to learn how to do so.

# License

The SBMLtoODEjax project is licensed under
the [MIT license](https://github.com/flowersteam/sbmltoodejax/blob/main/LICENSE).

# Acknowledgements

SBMLtoODEjax builds on:

* [SBMLtoODEpy](https://github.com/AnabelSMRuggiero/sbmltoodepy)'s parsing and conversion of SBML files, by Steve M.
  Ruggiero and Ashlee N. Ford
* [JAX](https://github.com/google/jax)'s composable transformations, by the Google team
* [Equinox](https://github.com/patrick-kidger/equinox)'s module abstraction, by Patrick Kidger
* [BasiCO](https://github.com/copasi/basico/blob/d058c10dd51f2c3e926efeaa29c6194f86bfdc90/basico/biomodels.py)'s access
  the BioModels REST api, by the COPASI team

Our documentation was also inspired by the [GPJax](https://docs.jaxgaussianprocesses.com/) documentation, by Thomas
Pinder and team.

# Citing SBMLtoODEjax

If you use SBMLtoODEjax in your research, please cite the [arXiv paper](https://arxiv.org/abs/2307.08452).

```
@misc{etcheverry2023sbmltoodejax,
      title={SBMLtoODEjax: efficient simulation and optimization of ODE SBML models in JAX}, 
      author={Mayalen Etcheverry and Michael Levin and Clément Moulin-Frier and Pierre-Yves Oudeyer},
      year={2023},
      eprint={2307.08452},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM}
}
```