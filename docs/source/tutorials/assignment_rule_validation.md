# Checking assignment rules against independent simulators

An SBML assignment rule is an equality that holds throughout a simulation. For
example, if a reaction A → B has rate `k * A` and `k = 1 + time`, then

$$
\frac{dA}{dt}=-(1+t)A,\qquad A(0)=2,
\qquad A(t)=2\exp(-t-t^2/2).
$$

Holding `k` fixed while integrating an output interval changes this differential
equation. With one interval from 0 to 1, that approximation produces
`A(1) = 2 exp(-1) = 0.735759`, while the correct value is `0.446260`.
Tightening the ODE solver tolerance cannot remove an error in the supplied
right-hand side. Reducing the output interval can hide it, but changes the
experiment's answer with a setting that should primarily control output and
intervention timing.

`ModelStep` evaluates `AssignmentRule` at every internal derivative evaluation,
using the solver's current state and time, and recomputes the returned `w` at the
endpoint. Its public arguments and return values are unchanged. Regenerate
previously exported Python models to obtain this behavior. Direct users of
`RateofSpeciesChange` still supply the instantaneous `w` themselves.

The [SBML rule specification](https://sbml.org/software/libsbml/5.18.0/docs/formatted/python-api/classlibsbml_1_1_assignment_rule.html)
distinguishes assignment rules from rate rules and disallows assignment-rule
cycles. An assignment is not a recurrence that reads its own previous value.
This correction does not add general algebraic-rule/DAE support, events, or
varying compartment sizes.

## An independent check

The repository includes seven small, offline SBML models with exact solutions:
constant and rate-rule controls; time-dependent and state-dependent assignments;
chained assignments listed out of dependency order; and assigned species with
amount or concentration semantics. The same XML files are imported separately
by libRoadRunner, COPASI, and SBMLtoODEjax.

The reference implementations use different numerical solvers: CVODE through
libRoadRunner and LSODA through COPASI/BasiCO. Each is checked against the exact
solution and against the other simulator before it is used as a JAX oracle.
Agreement between two programs alone is not assumed to establish correctness.

All comparisons use species **amounts**, a common time grid, and the same initial
conditions. The fixtures use a 2-litre compartment, making unit mistakes visible.
The COPASI adapter explicitly converts concentration to amount and rejects
unexpected units or compartment definitions. These adapters cover these fixtures;
they are not general-purpose SBML normalization code.

## Run the checks

From a source checkout with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install setuptools wheel -r validation/requirements.txt
python -m pip install --no-build-isolation -e .
python -m pytest test/test_jaxfuncs.py test/test_assignment_rules.py test/test_simulator_oracles.py -q
python -m validation.compare_simulators --output outputs/oracles
python -m validation.plot_comparison outputs/oracles
```

The two-step installation accommodates the existing `setup.py`, which imports
the package during its build and therefore requires its runtime dependencies
to be installed first. The oracle packages are optional and do not become
dependencies of the installed library. Without them, the oracle test module is
skipped; the analytic regression tests remain available.

The report command refuses to overwrite an existing output directory. It saves
CSV trajectories and a JSON report with package versions, solver settings, file
hashes, and every comparison. It exits unsuccessfully if any required comparison
fails. `previous_sample_and_hold` is an explicit implementation of the former
ModelStep algorithm for comparison, not a second installation of the old package.

Reference solvers use relative tolerance `1e-10` and absolute tolerance `1e-12`.
JAX uses `1e-6` and `1e-7`, since the existing generated rate arrays are float32.
Acceptance uses `abs(actual - reference) <= atol + rtol * abs(reference)`:
`rtol=2e-8, atol=2e-9` between references and exact solutions, and
`rtol=2e-5, atol=2e-6` for JAX. Every model is checked with output intervals of
1 second and 0.1 seconds. The regression tests also exercise JIT, batching,
parameter gradients, a nonzero start time, stale incoming `w`, custom generated
class names, and the existing rollout sampling convention.

![Independent simulator comparison](/_static/assignment_rules.png)

## Scope of the evidence

These tests check a specific semantic correction. They do not establish complete
SBML conformance, reproduce the GRN memory study, or validate every BioModels
model. Discontinuous input schedules still need explicit treatment of transition
times; this change is not an event solver. Unsupported features should be checked
before extending this comparison to other models.

The existing `test/test_modulegeneration.py` downloads BioModels at runtime and
is separate from this offline suite. For larger biological models, compare
baseline trajectories and individual observables before applying stimulation
protocols or constructing an RNM transition graph.
