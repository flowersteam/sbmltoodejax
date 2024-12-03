# Design principles

Before diving into the specifications of SBMLtoODEjax, we provide some context on the [SBML](https://sbml.org/) core components, 
as well as on the core principles behind the [JAX](https://jax.readthedocs.io/en/latest/) library and the *module* abstraction of the [Equinox](https://docs.kidger.site/equinox/) library, 
all of which is useful to better understand SBMLtoODEjax main design principles.
Feel free to jump directly to [🔍 Structure of the generated python file](structure-of-the-generated-python-file) if you want to get into the specifics of SBMLtoODEjax.

## 💡 Context

### SBML core components
SBML (Systems Biology Markup Language) is a standard file format used in computational systems biology to represent models of biological processes. 

SBML files are written in XML format, such that it can be read and simulated by different software tools.
To decide whether you should use SBMLtoODEjax or not will depend on your needs and experience in programming, 
be sure to check our [Why use SBMLtoODEjax?](./why_use.md) section.

SBML models use the following core components to describe the structure and dynamics of biological systems:

* Compartments: compartments represent the physical (or conceptual) spaces where biological entities reside and are defined by their spatial characteristics, in particular their *size*.
* Species: species represent the individual entities in a biological system (e.g. molecules or proteins). Each species is associated with a compartment where it resides and participate in reactions. The initial amount/concentration of species is often provided in the SBML file.
* Reactions: Reactions describe the transformations or interactions between species. Each reaction consists of reactants, products, and modifiers (species that affect the reaction but are not consumed or produced).
* Parameters: Parameters are used to define numerical values in mathematical equations or to represent model constants. They can be associated with species, reactions, or other elements in the model. 
* Events: Events represent discrete occurrences that can trigger discontinuous changes in the model (e.g. addition or removal of species or changes in reaction rates). 
⚠️ Models with events are currently not handled in SBMLtoODEjax.
* Rules: Rules define mathematical relationships or constraints in the model, and can be of different types: 
  * Algebraic rules specify relationships between variables
  * Differential rules describe the rate of change of a variable
  * Assignment rules set the value of a variable based on a mathematical expression

:::{seealso}
There is much more to know about SBML, such as how it integrates annotations and metadata, how it handles units and quantities, and so on.
It can be quite involving to dig into their [specifications](https://sbml.org/documents/specifications/), but you will find all the necessary information in there.
:::

### JAX main principles

JAX is a recently-developed python library which provides a simple and powerful API for writing accelerated numerical code,
which is why we use it in SBMLtoODEjax. However, it is important to understand how JAX operates to use it properly.

Here are the main things to know about JAX for us:
1. **Numpy-inspired Syntax**: JAX API closely mirrors the Numpy API, allowing an easy entry into the library for users already familiar with Numpy.
2. **Functional programming style**: JAX encourages the use of pure functions, and contrary to Numpy JAX arrays are immutable.
Using pure functions can sometimes be cumbersome, but it brings several benefits and in particular the use of JAX transformations.
3. **Transformations**: JAX provides several operations which act on pure functions such as
just-in-time compilation (`jit`), automatic vectorization (`vmap`) and automatic differentiation (`grad`). 
3. **PyTree abstraction**: JAX introduces the use of *PyTrees* to represent any nested data structure as trees with leaves (arrays, scalars, or other simple Python data types)
and internal nodes (tuples, lists, dictionaries, or custom nodes). This abstraction is useful to create more complex data structures that are fully compatible with JAX transformations.
4. **Efficient execution**: JAX leverages XLA (Accelerated Linear Algebra), a domain-specific compiler that optimizes and compiles numerical 
computations for efficient execution on CPUs (also GPUs and TPUs but we don't use it in SBMLtoODEjax).
5. **Automatic differentiation**: JAX provides automatic differentiation capabilities allowing to easily compute gradients of complex functions and to perform gradient-descent optimization.

:::{tip}
In short, if your code respect the standard JAX practices where **everything is just PyTrees and transformations on PyTrees**, you should be able to make full advantage of jax main features.
:::

:::{seealso}
There is much more to know bout JAX and we recommend checking their documentation to know more about [how to make advantage of JAX function transformations](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) 
and [how to think in JAX](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html).
:::


### Equinox's *Module* abstraction

Equinox is a small library from the JAX ecosystem, which introduces the *Module* abstraction to represent parameterised functions (such as neural networks) 
following a *class-based* syntax (simple syntax similar to PyTorch modules) but that is registered as a *PyTree* (allows to use JAX transformation such as `jit`,`vmap` and `grad`).

We use this abstraction to represent parametrized functions in the python-generated files.
This allows to have parameters of the models (such as the stoichiometric Matrix and the parameters of the ODE solver) but also other modules to be specified as fields at the class-level,
instead of having to pass those parameters as inputs (together with all other variables) each time the functions are called.

Equinox also defines filters such as `filter_jit`, `filter_grad` and `filter_vmap` to customize with respect to which parameters/variables one wishes to operate JAX transformations.

(structure-of-the-generated-python-file)=
## 🔍 Structure of the generated python file
SBMLtoODEjax automatically parses a given SBML file and convert it to a python file written in JAX.
The generated files contains several *variables*,  *modules* and some additional *data* allowing to simulate the original SBML model, and to manipulate it based on one's needs.

### Variables 

Among the original SBML conventions, we found the following to be difficulty compatible with a functional programming paradigm:
*parameters* can represent both constants and variables that are manipulated by the model's rules, *species* mainly represent
variables that are reactants and/or products of the model's reactions but it happens that some of them are neither consumed nor produced, 
*compartments* are distinguished from parameters but in the end they are (another) parameter of the model.

Moreover, in most existing software tools, in order to modify the model's parameters and/or variable states, users often need to access the variables by their specific identifier/name.

Here, we use slightly different conventions for representing the model variables that we believe allow for more general mathematical formalism (and hence functional-style implementation)
as well as more flexible manipulations:

| SBMLtoODEjax | Description                                                                                                                                                                                     |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| y                | vector of variables that represent the model's species amount (e.g. gene expression level) and that is evolving in time at a rate governed by the model's ODE-based reactions                   |
| w                | vector of non-constant parameters that intervene either in the model reactions or assignment rules (e.g. kinematic parameters) and whose state is evolving in time according to assigment rules |
| c                | vector of constant parameters that intervene either in the model reactions or assignment rules (e.g. kinematic parameters)                                                                      |
| t                | time serie of points for which to solve for y and w                                                                                                                                             |

### Modules

The generated files contain 4 modules (functions) that operate on the above variables:

| SBMLtoODEjax   | Math                                                                   | Description                                                                                                                                                                                                 |
|-------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RateofSpeciesChange | $y(t), t, w(t) \mapsto \frac{dy(t)}{dt}$                               | system of ODE-governed equations that governs the rate of species changes $\frac{dy}{dt}$                                                                                                                   |
| AssignmentRule    | $w(t), y(t+\Delta t),  t+\Delta t \mapsto w(t+\Delta t)$               | system of equations that governs the temporal evolution of $w$                                                                                                                                              |
| ModelStep         | $y(t),  w(t), c, t, \mapsto y(t+\Delta t),  w(t+\Delta t), c, t+\Delta t$ | iteratively integrates `RateofSpeciesChange` using jax's [odeint](https://github.com/google/jax/blob/main/jax/experimental/ode.py) and calls `AssignmentRule` to update variables $y$ and $w$ in time |
| ModelRollout      | $y(0), w(0), c, t0, T \mapsto y[0..T], w[0..T]$                        | iteratively calls `ModelStep` for a given number of steps                                                                                                                                                   |


### Data

The generated files also contain additional data about the initial state of the (y,w,t) variables and about their identifiers/names in the original SBML file:

| SBMLtoODEjax | Description                                                                                            |
|--------------|--------------------------------------------------------------------------------------------------------|
| y0           | default initial state of $y$ (as provided in the SBML file)                                            |
| w0           | default initial state of $w$ (as provided in the SBML file)                                            |
| t0           | default initial state of $t$ (0.0)                                                                     |
| y_indexes    | mapping between the SBML species identifiers and their index in the $y$ vector                         |
| w_indexes    | mapping between the SBML identifiers and their index in the $w$ vector |
| c_indexes    | mapping between the SBML identifiers and their index in the $c$ vector         |

