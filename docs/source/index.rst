sbmltoodejax
========================================

The ``sbmltoodejax`` project builds upon the ``sbmltoodepy`` project which allows to convert
Systems Biology Markup Language (SBML) models into python classes that can then easily be
simulated (and manipulated based on one's need) in python projects.

At the difference of the original ``sbmltoodepy`` project, ``sbmltoodejax`` makes use of jax framework
allowing to take advantage of ``jax`` mains features which are

* just-in-time compilation (**jit**)
* automatic vectorization (**vmap**)
* automatic differentiation (**grad**)


Table of Contents
------------
.. toctree::
   :maxdepth: 2

   install
   benchmarks
   tutorials

License
-------

The project is licensed under the MIT license.

Acknowledgements
-------
SBMLtoODEjax is a software package based on the SBMLtoODEpy_, jax_ and equinox_ packages.

.. _SBMLtoODEpy: https://github.com/AnabelSMRuggiero/sbmltoodepy
.. _jax: https://github.com/google/jax
.. _equinox: https://github.com/patrick-kidger/equinox