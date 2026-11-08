"""Export a reproducible COPASI/RoadRunner/JAX comparison on offline SBML cases.

Run from the checkout: python -m validation.compare_simulators --output outputs/oracles
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import tempfile

import numpy as np

from test.assignment_rule_cases import CASES, FIXTURES, exact, generate, jax_trajectory
from validation.oracles import (JAX_ATOL, JAX_RTOL, REFERENCE_ATOL, REFERENCE_RTOL,
                                compare, copasi_amounts, roadrunner_amounts)


def sample_and_hold(module, times):
    """Reproduce the previous ModelStep algorithm for illustration, without patching it."""
    from jax.experimental.ode import odeint
    import jax.numpy as jnp

    rate, assignment = module.RateofSpeciesChange(), module.AssignmentRule()
    y, w, c = module.y0, module.w0, module.c
    indices = [module.y_indexes[x] for x in ("A", "B")]
    rows = [np.asarray(y)[indices]]
    for start, end in zip(times[:-1], times[1:]):
        y = odeint(rate, y, jnp.array([start, end]), w, c, atol=1e-7, rtol=1e-6)[-1]
        w = assignment(y, w, c, end)
        rows.append(np.asarray(y)[indices])
    return np.asarray(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    packages = ("jax", "jaxlib", "equinox", "numpy", "sbmltoodepy", "python-libsbml",
                "libroadrunner", "copasi-basico", "python-copasi")
    generator = Path(__file__).resolve().parents[1] / "sbmltoodejax" / "modulegeneration.py"
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in packages},
        "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "reference_solver_settings": {"RoadRunner": "CVODE", "COPASI": "LSODA",
                                      "rtol": 1e-10, "atol": 1e-12},
        "jax_solver_settings": {"solver": "jax.experimental.ode.odeint", "rtol": 1e-6,
                                "atol": 1e-7, "generated_rate_dtype": "float32"},
        "comparison_tolerances": {"reference": {"rtol": REFERENCE_RTOL, "atol": REFERENCE_ATOL},
                                  "jax": {"rtol": JAX_RTOL, "atol": JAX_ATOL}},
        "quantities": "A and B amounts, mol; COPASI concentrations multiplied by 2-litre volume",
        "cases": [],
    }
    with tempfile.TemporaryDirectory() as temp:
        for case in CASES:
            module = generate(case, temp)
            path = FIXTURES / (case + ".xml")
            for intervals in (1, 10):
                times = np.linspace(0, 1, intervals + 1)
                traces = {"exact": exact(case, times), "roadrunner": roadrunner_amounts(path, times),
                          "copasi": copasi_amounts(path, times), "jax": jax_trajectory(module, times),
                          "previous_sample_and_hold": sample_and_hold(module, times)}
                checks = {}
                for left, right in (("roadrunner", "exact"), ("copasi", "exact"), ("roadrunner", "copasi"),
                                    ("jax", "exact"), ("jax", "roadrunner"), ("jax", "copasi")):
                    tol = (JAX_RTOL, JAX_ATOL) if left == "jax" else (REFERENCE_RTOL, REFERENCE_ATOL)
                    checks[f"{left}_vs_{right}"] = compare(traces[left], traces[right], rtol=tol[0], atol=tol[1])
                report["cases"].append({
                    "case": case, "intervals": intervals,
                    "sbml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "checks": checks,
                    "previous_sample_and_hold_vs_exact": compare(traces["previous_sample_and_hold"],
                                                               traces["exact"], rtol=JAX_RTOL, atol=JAX_ATOL),
                })
                with (args.output / f"{case}_{intervals}.csv").open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["time"] + [f"{name}_{species}" for name in traces for species in ("A", "B")])
                    for index, time in enumerate(times):
                        writer.writerow([time] + [float(v) for trace in traces.values() for v in trace[index]])
    passed = all(check["passed"] for case in report["cases"] for check in case["checks"].values())
    report["passed"] = passed
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(report['cases'])} cases: {'PASS' if passed else 'FAIL'}; {args.output / 'report.json'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
