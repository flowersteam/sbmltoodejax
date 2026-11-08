"""Run the checked-in assignment-rule fixtures in independent SBML engines.

The adapters deliberately cover these fixtures, not arbitrary SBML. Amounts are
compared, with explicit concentration-to-amount conversion for COPASI. Unsupported
units or a changed compartment definition fail rather than silently comparing
different quantities.
"""

import numpy as np


REFERENCE_RTOL = 2e-8
REFERENCE_ATOL = 2e-9
JAX_RTOL = 2e-5
JAX_ATOL = 2e-6


def roadrunner_amounts(path, times):
    import roadrunner

    rr = roadrunner.RoadRunner(str(path))
    rr.setIntegrator("cvode")
    rr.integrator.relative_tolerance = 1e-10
    rr.integrator.absolute_tolerance = 1e-12
    result = np.asarray(rr.simulate(times=list(times), selections=["time", "A", "B"]))
    np.testing.assert_allclose(result[:, 0], times, rtol=0, atol=1e-10)
    # Bare SBML IDs select AMOUNTS in RoadRunner; [A] would select concentration.
    return result[:, 1:]


def copasi_amounts(path, times):
    import basico
    import COPASI

    model = basico.load_model(str(path))
    if model is None:
        raise ValueError(f"COPASI could not load {path}")
    try:
        if (model.getModel().getQuantityUnit(), model.getModel().getVolumeUnit()) != ("mol", "l"):
            raise ValueError("These fixtures require COPASI units mol and l")
        compartments = basico.get_compartments(model=model)
        if len(compartments) != 1 or compartments.loc["cell", "type"] != "fixed":
            raise ValueError("These fixtures require one fixed compartment")
        volume = float(compartments.loc["cell", "size"])
        if volume != 2:
            raise ValueError("Expected the fixture's 2-litre compartment")
        result = basico.run_time_course(
            model=model, values=list(times), method="deterministic",
            use_initial_values=True, automatic=False, use_sbml_id=True,
            use_concentrations=True, r_tol=1e-10, a_tol=1e-12,
        )
        np.testing.assert_allclose(result.index, times, rtol=0, atol=1e-10)
        return result.loc[:, ["A", "B"]].to_numpy() * volume
    finally:
        COPASI.CRootContainer.removeDatamodel(model)


def compare(actual, expected, *, rtol, atol):
    """Tolerance-normalized error <= 1 means the entire trajectory passes."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite or incompatible simulation outputs")
    error = np.abs(actual - expected)
    scaled = error / (atol + rtol * np.abs(expected))
    return {"max_abs_error": float(error.max()), "max_scaled_error": float(scaled.max()),
            "passed": bool(np.all(scaled <= 1))}
