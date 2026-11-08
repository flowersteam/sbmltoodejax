"""Small, independently solvable SBML cases shared by tests and oracle checks."""

import importlib.util
from pathlib import Path

import numpy as np

from sbmltoodejax.modulegeneration import GenerateModel
from sbmltoodejax.parse import ParseSBMLFile


FIXTURES = Path(__file__).parent / "fixtures" / "assignment_rules"
CASES = ("constant", "time_dependent", "state_dependent", "chained",
         "assigned_species", "concentration_species", "rate_rule")


def generate(case, directory, **kwargs):
    """Generate from the same XML used by both independent simulators."""
    path = Path(directory) / (case + ".py")
    GenerateModel(ParseSBMLFile(str(FIXTURES / (case + ".xml"))), str(path), **kwargs)
    spec = importlib.util.spec_from_file_location("generated_" + case, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(case, times):
    """Species AMOUNTS in moles; cell volume is deliberately 2 litres."""
    times = np.asarray(times)
    if case == "constant":
        a = 2 * np.exp(-times)
    elif case == "state_dependent":
        a = 2 / (1 + 2 * times)
    else:
        a = 2 * np.exp(-times - times**2 / 2)
    return np.column_stack((a, 2 - a))


def jax_trajectory(module, times):
    """Use ModelStep to retain both the initial state and final endpoint."""
    step = module.ModelStep(atol=1e-7, rtol=1e-6)
    y, w, c, t = module.y0, module.w0, module.c, module.t0
    indices = [module.y_indexes[x] for x in ("A", "B")]
    rows = [np.asarray(y)[indices]]
    for end in times[1:]:
        y, w, c, t = step(y, w, c, t, float(end - t))
        rows.append(np.asarray(y)[indices])
    return np.asarray(rows)
