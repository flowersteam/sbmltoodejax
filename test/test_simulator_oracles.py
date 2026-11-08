"""Optional integration checks: install validation/requirements.txt to enable."""

import numpy as np
import pytest

pytest.importorskip("roadrunner")
pytest.importorskip("basico")

from .assignment_rule_cases import CASES, FIXTURES, exact, generate, jax_trajectory
from validation.oracles import (JAX_ATOL, JAX_RTOL, REFERENCE_ATOL, REFERENCE_RTOL,
                                compare, copasi_amounts, roadrunner_amounts)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("intervals", [1, 10])
def test_independent_simulators_agree(tmp_path, case, intervals):
    times = np.linspace(0, 1, intervals + 1)
    path = FIXTURES / (case + ".xml")
    expected = exact(case, times)
    rr = roadrunner_amounts(path, times)
    copasi = copasi_amounts(path, times)
    jax = jax_trajectory(generate(case, tmp_path), times)
    for name, actual in [("RoadRunner", rr), ("COPASI", copasi)]:
        result = compare(actual, expected, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
        assert result["passed"], (name, result)
    result = compare(rr, copasi, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    assert result["passed"], result
    for name, oracle in [("exact", expected), ("RoadRunner", rr), ("COPASI", copasi)]:
        result = compare(jax, oracle, rtol=JAX_RTOL, atol=JAX_ATOL)
        assert result["passed"], (name, result)
