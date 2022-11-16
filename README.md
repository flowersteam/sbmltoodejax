# SBMLtoODEjax

The `sbmltoodejax` project builds upon the `sbmltoodepy` project which
allows to convert Systems Biology Markup Language (SBML) models into
python classes that can then easily be simulated (and manipulated based
on one\'s need) in python projects.

At the difference of the original `sbmltoodepy` project, `sbmltoodejax`
makes use of jax framework allowing to take advantage of `jax` mains
features which are

-   just-in-time compilation (**jit**)
-   automatic vectorization (**vmap**)
-   automatic differentiation (**grad**)


## License

The project is licensed under the MIT license.

## Acknowledgements 

SBMLtoODEjax is a software package based
on the [SBMLtoODEpy](https://github.com/AnabelSMRuggiero/sbmltoodepy),
[jax](https://github.com/google/jax) and
[equinox](https://github.com/patrick-kidger/equinox) packages.
