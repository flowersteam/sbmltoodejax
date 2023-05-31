---
title: "SBMLtoODEjax: converting SBML models into ODE models in JAX"
tags:
  - Python
  - JAX
  - SBML
authors:
  - name: Mayalen Etcheverry^[corresponding author]
    affiliation: "1, 2"
  - name: Clément Moulin-Frier
    affiliation: "1"
  - name: Pierre-Yves Oudeyer
    affiliation: "1"
  - name: Michael Levin
    affiliation: "3"
affiliations:
  - name: INRIA, University of Bordeaux, Talence 33405, France
    index: 1
  - name: Poietis, Pessac 33600, France
    index: 2
  - name: Allen Discovery Center, Tufts University, Medford, MA, USA
    index: 3

date: 24 May 2023
bibliography: paper.bib
---

# SBMLtoODEjax: converting SBML models into ODE models in JAX

## Summary


SBMLtoODEjax is a lightweight library that allows to automatically parse and convert SBML models into python models written end-to-end in JAX, a high-performance numerical computing library with automatic differentiation capabilities  [@jax2018github]. SBMLtoODEjax is targeted at researchers that aim to incorporate SBML models into their python projects and ML pipelines, while taking advantage of JAX's *performant*, *batchable* and *differentiable* simulations capabilities. SBMLtoODEjax extends SBMLtoODEpy, a python library developed in 2019 for converting SBML files into python files written in Numpy/Scipy [@ruggieroSBMLtoODEpySoftwareProgram2019]. The chosen conventions for the generated variables and modules are slightly different than the standard SBML conventions (as used in the SBMLtoODEpy library) with the aim to accommodate for more flexible manipulations while preserving JAX-like functional programming style. Please refer to our [documentation](https://developmentalsystems.org/sbmltoodejax/index.html) for additional details on SBMLtoODEjax's advantages and limitations, main design principles, and example tutorials.


## Statement of Need
Developing methods to explore, predict and control the dynamic behavior of biological systems, from protein pathways to complex cellular processes, is an essential frontier of research for bioengineering and biomedicine [@kitanoSystemsBiologyBrief2002]. Thus, significant effort has gone in computational inference and mathematical modeling of biological systems [@dejongModelingSimulationGenetic2002;@delgadoComputationalMethodsGene2019]. This effort has resulted in the development of large collections of publicly-available models, typically stored and exchanged on online platforms (such as the BioModels Database [@glontBioModelsExpandingHorizons2018;@malik-sheriffBioModels15Years2020a]) using the Systems Biology Markup Language (SBML), a standard format for representing mathematical models of biological systems [@huckaSystemsBiologyMarkup2003;@SystemsBiologyMarkup]. 

Yet, despite the wealth of available models, scientists still lack an in-depth understanding of the range of possible behaviors that these models can exhibit under different initial data and environmental stimuli. 
Except for a subset of simple networks where system behavior and response to stimuli can be well understood analytically (or with exhaustive enumeration methods), onerous sampling of the parameter space and time-consuming numerical simulations are often needed which remains a major roadblock for progress in biological network analysis. 

On the other hand, recent progress in machine learning (ML) has led to the development of novel computational tools that leverage high-performance computation, parallel execution and differentiable programming and that promise to accelerate research across multiple areas of science, including biological network analysis [@muzioBiologicalNetworkAnalysis2021] and applications in drug discovery and molecular medicine [@camachoNextGenerationMachineLearning2018a;@alquraishiDifferentiableBiologyUsing2021]. However, to our knowledge, there is no software tool that allows seamless integration of existing mathematical models of cellular molecular pathways (SBML files constructed by biologists) with ML-supported pipelines and programming frameworks.

Indeed, whereas there exists many software tools for manipulation and numerical simulation of SBML models, they typically rely either on specialized simulation platforms limiting the flexibility for customization and scripting (such as COPASI [@hoopsCOPASICOmplexPAthway2006;@Bergmann_copasi_basico_Release_0_48_2023], Virtual Cell [@loewVirtualCellSoftware2001, @slepchenkoQuantitativeCellBiology2003a] and Cell Designer [@funahashiCellDesignerProcessDiagram2003; @funahashiCellDesignerVersatileModeling2008]) or provide scripting interfaces in Python or Matlab but rely on backend engines that do not support hardware acceleration or automatic differentiation (like Tellurium [@choiTelluriumExtensiblePythonbased2018;@TelluriumNotebooksEnvironment] and SBMLtoODEpy [@ruggieroSBMLtoODEpySoftwareProgram2019] python packages, or the Systems Biology Format Converter (SBFC) which generates MATLAB and OCTAVE code [@rodriguezSystemsBiologyFormat2016]).

SBMLtoODEjax seeks to bridge that gap by bringing SBML simulation to the JAX ecosystem: it retains the simplicity of the SBMLtoODEPy library to facilitate incorporation and refactoring of the generated ODE models into python projects while carefully tailoring them to make advantage of JAX main features. Indeed, the generated python classes inherit from the Equinox *module* abstraction [@kidger2021equinox] and are registered as PyTree containers, which facilitates the application of JAX core transformations such as just-in-time compilation (`jit`), automatic vectorization (`vmap`) and automatic differentiation (`grad`) to any SBMLtoODEjax object. 

The SBMLtoODEjax [documentation](https://developmentalsystems.org/sbmltoodejax/index.html) includes various tutorials for 1) reproducing simulation results of systems biology publications, 2) make use of `vmap` transformation for parallel simulation given a batch of initial conditions, 3) benchmark compute time for long-simulation rollouts and large batch sizes, and 4) integrate SBMLtoODEjax models within the Optax [@deepmind2020jax] pipeline for gradient-based optimization of the model kinematic parameters and/or stimuli-based interventions.  



## Software requirements and external usage
SBMLtoODEjax is developed under the MIT license and available on `PyPI` via `pip install sbmltoodejax`. It is written on top of SBMLtoODEpy [@ruggieroSBMLtoODEpySoftwareProgram2019], JAX (cpu) [@jax2018github] and Equinox [@kidger2021equinox], which are the main requirements. SBMLtoODEjax has been used in the [AutoDiscJax](https://github.com/flowersteam/autodiscjax) library and in one [research project](https://developmentalsystems.org/curious-exploration-of-grn-competencies/paper.html). 

## Acknowledgements
SBMLtoODEjax builds on SBMLtoODEpy's parsing and conversion of SBML files [@ruggieroSBMLtoODEpySoftwareProgram2019], JAX's composable transformations [@jax2018github], Equinox's module abstraction [@kidger2021equinox] and BasiCO’s access to the BioModels REST api [@Bergmann_copasi_basico_Release_0_48_2023]. 

This work has been funded by the biotechnology company Poietis and the French National Association of Research and Technology (ANRT). 

## References