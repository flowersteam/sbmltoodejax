---
title: "SBMLtoODEjax: efficient simulation and optimization of ODE SBML models in JAX"
tags:
  - SBML
  - Biological Network Analysis
  - Python
  - JAX
  - high performance computing
  - parallel computing
authors:
  - name: Mayalen Etcheverry^[corresponding author]
    affiliation: "1, 2"
  - name: Michael Levin
    affiliation: "3"
  - name: Clément Moulin-Frier
    affiliation: "1"
  - name: Pierre-Yves Oudeyer
    affiliation: "1"
affiliations:
  - name: INRIA, University of Bordeaux, Talence 33405, France
    index: 1
  - name: Poietis, Pessac 33600, France
    index: 2
  - name: Allen Discovery Center, Tufts University, Medford, MA, USA
    index: 3

date: 01 June 2023
bibliography: paper.bib
---

# SBMLtoODEjax: efficient simulation and optimization of ODE SBML models in JAX

## Summary

Developing methods to explore, predict and control the dynamic behavior of biological systems, from protein pathways to
complex cellular processes, is an essential frontier of research for bioengineering and
biomedicine [@kitanoSystemsBiologyBrief2002]. Thus, significant effort has gone in computational inference and
mathematical modeling of biological systems [@dejongModelingSimulationGenetic2002;@delgadoComputationalMethodsGene2019].
This effort has resulted in the development of large collections of publicly-available models, typically stored and
exchanged on online platforms (such as the BioModels
Database [@glontBioModelsExpandingHorizons2018; @malik-sheriffBioModels15Years2020a]) using the Systems Biology Markup
Language (SBML), a standard format for representing mathematical models of biological
systems [@huckaSystemsBiologyMarkup2003;@SystemsBiologyMarkup].

SBMLtoODEjax is a lightweight library that allows to automatically parse and convert SBML models into python models
written end-to-end in JAX, a high-performance numerical computing library with automatic differentiation
capabilities [@jax2018github]. SBMLtoODEjax is targeted at researchers that aim to incorporate SBML-specified ordinary
differential equation (ODE) models into their python projects and machine learning pipelines, in order to perform
efficient numerical
simulation and optimization with only a few lines of code. Taking advantage of JAX’s core transformation features, one
can easily boost the speed of ODE models time-course simulations and perform efficient search and optimization by
running simulations in parallel and/or using automatic differentiation to find derivatives. SBMLtoODEjax extends
SBMLtoODEpy, a python library developed in 2019 for converting SBML files into python files written in
Numpy/Scipy [@ruggieroSBMLtoODEpySoftwareProgram2019]. The chosen conventions for the generated variables and modules
are slightly different from the standard SBML conventions (used in the SBMLtoODEpy library) with the aim to
accommodate for more flexible manipulations while preserving JAX-like functional programming style.

## Statement of Need

Despite the wealth of available SBML models, scientists still lack an in-depth understanding of the range of possible
behaviors that these models can exhibit under different initial data and environmental stimuli, and lack effective ways
to search and optimize those behaviors via external interventions. Except for a subset of simple networks where system
behavior and response to stimuli can be well understood analytically (or with exhaustive enumeration methods), onerous
sampling of the parameter space and time-consuming numerical simulations are often needed which remains a major
roadblock for progress in biological network analysis.

On the other hand, recent progress in machine learning (ML) has led to the development of novel computational tools that
leverage high-performance computation, parallel execution and differentiable programming and that promise to accelerate
research across multiple areas of science, including biological network analysis [@muzioBiologicalNetworkAnalysis2021]
and applications in drug discovery and molecular
medicine [@camachoNextGenerationMachineLearning2018a;@alquraishiDifferentiableBiologyUsing2021]. However, to our
knowledge, there is no software tool that allows seamless integration of existing mathematical models of cellular
molecular pathways (SBML files constructed by biologists) with ML-supported pipelines and programming frameworks.
Whereas there exists many software tools for manipulation and numerical simulation of SBML models, they
typically rely either on specialized simulation platforms limiting the flexibility for customization and scripting (such
as COPASI [@hoopsCOPASICOmplexPAthway2006;@Bergmann_copasi_basico_Release_0_48_2023], Virtual
Cell [@loewVirtualCellSoftware2001;@slepchenkoQuantitativeCellBiology2003a] and Cell
Designer [funahashiCellDesignerProcessDiagram2003;@funahashiCellDesignerVersatileModeling2008]) or provide scripting
interfaces in Python or Matlab but rely on backend engines that do not support hardware acceleration or automatic
differentiation (like Tellurium [@choiTelluriumExtensiblePythonbased2018;@TelluriumNotebooksEnvironment] and
SBMLtoODEpy [@ruggieroSBMLtoODEpySoftwareProgram2019] python packages, or the Systems Biology Format Converter (SBFC)
which generates MATLAB and OCTAVE code [@rodriguezSystemsBiologyFormat2016]).

SBMLtoODEjax seeks to bridge that gap by bringing SBML simulation to
the [JAX ecosystem](https://github.com/n2cholas/awesome-jax), a thriving community of JAX libraries
that aim to accelerate research in machine learning and beyond,
with diverse applications spanning molecular
dynamics [@schoenholzJAXFrameworkDifferentiable2020], protein engineering [@maReimplementingUnirepJAX2020], quantum
physics [@carleoNetKetMachineLearning2019], cosmology [@campagneJAXCOSMOEndtoEndDifferentiable2023], ocean
modeling [@hafnerFastCheapTurbulent2021], photovoltaic research [@mannPVEndtoendDifferentiable2022], acoustic
simulations [@stanziolaJWaveOpensourceDifferentiable2023] and fluid
dynamics [@bezginJAXFluidsFullydifferentiableHighorder2023]. SBMLtoODEjax aims to integrate this ecosystem and provide
tools to accelerate research
in biological network analysis.

***Simplicity and extensibility***
SBMLtoODEjax retains the simplicity of the SBMLtoODEPy library to facilitate incorporation and refactoring of the
ODE models into one’s own python
projects. As shown in \autoref{fig:fig1}, with only a few lines of python code one can load and simulate existing SBML
files.

![Example code (left) and output snapshot (right) reproducing [original simulation results](https://www.ebi.ac.uk/biomodels/BIOMD0000000010#Curation)
of Kholodenko 2000's paper [@kholodenkoNegativeFeedbackUltrasensitivity2000] hosted on BioModels website. \label{fig:fig1}](fig1.png){width=80%}

***JAX-friendly*** The generated python models are tailored to take advantage of JAX main features.
Model rollouts use `jit` transformation and `scan` primitive to reduce compilation and execution time of
the recursive ODE integration steps, which is particularly useful when running large numbers of steps (long reaction
times).
Models also inherit from the Equinox module abstraction [22] and are registered as PyTree
containers, which facilitates the application of JAX core transformations to any SBMLtoODEjax object.

***Efficiency simulation and optimization*** The application of JAX core transformations, such as just-in-time
compilation (`jit`),
automatic vectorization (`vmap`) and automatic differentiation (`grad`), to the generated models make it very easy (and
seamless) to
efficiently run simulations in parallel. For instance, as shown in \autoref{fig:fig2}, with only a few lines of python
code one can
vectorize calls to model rollout and perform batched computations efficiently, which is particularly useful when
considering large batch sizes.

![(left) Example code to vectorize calls to model rollout
(right) Results of a (rudimentary) benchmark comparing the average simulation time of models implemented with SBMLtoODEpy
versus SBMLtoODEjax (for different number of rollouts i.e. batch size).
For additional details on the comparison, please refer to our [Benchmarking](https://developmentalsystems.org/sbmltoodejax/tutorials/benchmark.html) notebook.
\label{fig:fig2}](fig2.png){width=100%}

As shown in \autoref{fig:fig3}, SBMLtoODEjax models can also be integrated within Optax pipelines,
a gradient processing and optimization library for JAX [@deepmind2020jax], allowing to optimize model parameters and/or
external
interventions with stochastic gradient descent.

![(left) Default simulation results of [biomodel #145](https://www.ebi.ac.uk/biomodels/BIOMD0000000145) which models ATP-induced intracellular calcium oscillations,
and (arbitrary) target sine-wave pattern for Ca_Cyt concentration.
(middle) Training loss obtained when running the Optax optimization loop, with Adam optimizer, over the model kinematic parameters $c$.
(right) Simulation results obtained after optimization. The full example is available at our [Gradient Descent](https://developmentalsystems.org/sbmltoodejax/tutorials/gradient_descent.html) tutorial.
\label{fig:fig3}](fig3.png){width=100%}

Altogether, the parallel execution capabilities and the differentiability of the generated models opens
interesting possibilities to design and optimize intervention strategies.

***Current limitations*** SBMLtoODEjax is still in its early phase and does not yet handle all possible cases of SBML
files, but we welcome contributions and aim
to handle more cases in future releases. Similarly, SBMLtoODEjax only integrates one ODE solver for
now (`jax.experimental.odeint`), but could benefit from more [@stadterBenchmarkingNumericalIntegration2021a].
Finally, whereas SBMLtoODEjax is to our knowledge the first software tool enabling gradient backpropagation through the
SBML model rollout,
applying it in practice can be hard and other optimization methods, such as evolutionary strategies, might be more
adapted.

***Documentation*** Please refer to https://developmentalsystems.org/sbmltoodejax/ for additional details on
SBMLtoODEjax’s
main [design principles](https://developmentalsystems.org/sbmltoodejax/design_principles.html), [advantages and limitations](https://developmentalsystems.org/sbmltoodejax/why_use.html),
and API docs as well as for various hands-on tutorials for
[loading and simulating biomodels](https://developmentalsystems.org/sbmltoodejax/tutorials/biomodels_curation.html), [parallel execution](https://developmentalsystems.org/sbmltoodejax/tutorials/parallel_execution.html)
and [gradient descent](https://developmentalsystems.org/sbmltoodejax/tutorials/gradient_descent.html).

## Software requirements and external usage

SBMLtoODEjax is developed under the MIT license and available on `PyPI` via `pip install sbmltoodejax`. It is written on
top of SBMLtoODEpy [@ruggieroSBMLtoODEpySoftwareProgram2019], JAX (cpu) [@jax2018github] and
Equinox [@kidger2021equinox], which are the main requirements. SBMLtoODEjax has been used in
the [AutoDiscJax](https://github.com/flowersteam/autodiscjax) library and in
one [ongoing research project](https://developmentalsystems.org/curious-exploration-of-grn-competencies/paper.html).

## Acknowledgements

SBMLtoODEjax builds on SBMLtoODEpy's parsing and conversion of SBML files [@ruggieroSBMLtoODEpySoftwareProgram2019],
JAX's composable transformations [@jax2018github], Equinox's module abstraction [@kidger2021equinox] and BasiCO’s access
to the BioModels REST api [@Bergmann_copasi_basico_Release_0_48_2023].

This work has been funded by the biotechnology company Poietis and the French National Association of Research and
Technology (ANRT),
as well as by the French National Research Agency (ANR, project DeepCuriosity).
It also benefited from two mobility research scholarships given by the French Academy (Jean Walter Zellidja scholarship)
and the University of Bordeaux (UBGRS-Mob scholarship).
Finally, this work benefited from the use of the Jean Zay supercomputer associated with the Genci grant A0091011996.

## References