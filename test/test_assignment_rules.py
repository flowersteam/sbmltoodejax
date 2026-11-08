"""Assignment rules are equalities at every solver stage, not sampled updates."""

import jax
import jax.numpy as jnp
import libsbml
import numpy as np
import pytest

from .assignment_rule_cases import CASES, FIXTURES, exact, generate, jax_trajectory


@pytest.mark.parametrize("case", CASES)
def test_fixtures_are_valid_sbml(case):
    document = libsbml.readSBML(str(FIXTURES / (case + ".xml")))
    document.checkConsistency()
    errors = [document.getError(i).getMessage() for i in range(document.getNumErrors())
              if document.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR]
    assert not errors


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("intervals", [1, 10])
def test_assignment_trajectories_match_exact_solution(tmp_path, case, intervals):
    module = generate(case, tmp_path)
    times = np.linspace(0, 1, intervals + 1)
    actual = jax_trajectory(module, times)
    np.testing.assert_allclose(actual, exact(case, times), rtol=2e-5, atol=2e-6)
    # Generated rate arrays are float32; allow accumulated roundoff across steps.
    np.testing.assert_allclose(actual.sum(axis=1), 2, rtol=0, atol=1e-5)


def test_assignment_uses_current_state_and_time_not_stale_w(tmp_path):
    module = generate("time_dependent", tmp_path)
    # Deliberately stale incoming cache, as after an external state intervention.
    step = module.ModelStep(atol=1e-7, rtol=1e-6)
    y, w, _, t = step(module.y0, module.w0 * 99, module.c, 1.0, 1.0)
    a = 2 * np.exp(-2.5)  # integral of (1+t) from 1 to 2
    np.testing.assert_allclose(y, [a, 2-a], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(w, [3], rtol=0, atol=1e-6)
    assert float(t) == 2


def test_rollout_preserves_output_convention_and_assignment_outputs(tmp_path):
    module = generate("time_dependent", tmp_path)
    rollout = module.ModelRollout(deltaT=0.25, atol=1e-7, rtol=1e-6)
    ys, ws, ts = rollout(5)
    np.testing.assert_allclose(ts, [0, 0.25, 0.5, 0.75, 1])
    indices = [module.y_indexes[x] for x in ("A", "B")]
    np.testing.assert_allclose(np.asarray(ys)[indices].T, exact("time_dependent", ts),
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(ws[0], 1 + ts)


@pytest.mark.parametrize("case,volume_factor", [("assigned_species", 1), ("concentration_species", 2)])
def test_assigned_species_output_is_an_amount(tmp_path, case, volume_factor):
    module = generate(case, tmp_path)
    step = module.ModelStep(atol=1e-7, rtol=1e-6)
    _, w, _, _ = step(module.y0, module.w0, module.c, module.t0, 1.0)
    np.testing.assert_allclose(w[module.w_indexes["U"]], 2 * volume_factor)


def test_assignment_solution_is_differentiable_and_batchable(tmp_path):
    module = generate("time_dependent", tmp_path)
    step = module.ModelStep(atol=1e-7, rtol=1e-6)
    index = module.c_indexes["alpha"]

    def endpoint(alpha):
        c = module.c.at[index].set(alpha)
        y, _, _, _ = step(module.y0, module.w0, c, module.t0, 1.0)
        return y[module.y_indexes["A"]]

    alphas = jnp.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(jax.jit(jax.vmap(endpoint))(alphas),
                               2 * np.exp(-1 - np.asarray(alphas)/2),
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(jax.grad(endpoint)(1.0), -np.exp(-1.5),
                               rtol=3e-5, atol=2e-6)


def test_custom_generated_class_names(tmp_path):
    module = generate("chained", tmp_path, RateofSpeciesChangeName="Rates",
                      AssignmentRuleName="Assignments", ModelStepName="Step",
                      ModelRolloutName="Rollout")
    step = module.Step(atol=1e-7, rtol=1e-6)
    y, _, _, _ = step(module.y0, module.w0, module.c, module.t0, 1.0)
    np.testing.assert_allclose(y, exact("chained", [1])[0], rtol=2e-5, atol=2e-6)
