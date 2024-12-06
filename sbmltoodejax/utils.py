import importlib
from sbmltoodejax.biomodels_api import get_content_for_model
from sbmltoodejax.modulegeneration import GenerateModel
from sbmltoodejax.parse import ParseSBMLFile

def generate_biomodel(model_idx, model_fp="jax_model.py",
                      vary_constant_reactants=False, vary_boundary_reactants=False,
                      deltaT=0.1, atol=1e-6, rtol=1e-12, mxstep=5000000,
                      solver_type='diffrax', diffrax_solver='Tsit5'):
    """Calls the `sbmltoodejax.modulegeneration_3.GenerateModel` for a SBML model hosted on the BioModel website and indexed by the provided `model_idx`.

    Args:
        model_idx: either an integer, or a valid model id
        model_fp (str): filepath for the generated file
        deltaT (float, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 0.1.
        atol (float, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 1e-6.
        rtol (float, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 1e-12.
        mxstep (int, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 5000000.
        solver_type (str, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 'diffrax'.
        diffrax_solver (str, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 'Tsit5'.

    Returns:
        model_fp (str): the filepath containing the generated python file
    """
    model_xml_body = get_content_for_model(model_idx)
    model_data = ParseSBMLFile(model_xml_body)
    GenerateModel(model_data, model_fp,
                  vary_constant_reactants=vary_constant_reactants, vary_boundary_reactants=vary_boundary_reactants,
                  deltaT=deltaT, atol=atol, rtol=rtol, mxstep=mxstep,
                  solver_type=solver_type, diffrax_solver=diffrax_solver)

    return model_fp


def load_biomodel(model_idx, model_fp="jax_model.py",
                  vary_constant_reactants=False, vary_boundary_reactants=False,
                  deltaT=0.1, atol=1e-6, rtol=1e-12, mxstep=5000000,
                  solver_type='diffrax', diffrax_solver='Tsit5'):
    """Calls the generate_biomodel function for a SBML model hosted on the BioModel website and indexed by the provided `model_idx`,
    then loads and returns the generated `model` module and `y0`, `w0`, `c` variables.

    Args:
        model_idx: either an integer, or a valid model id
        model_fp (str): filepath for the generated file
        deltaT (float, optional): parameter passed to `generate_biomodel`. Default to 0.1.
        atol (float, optional): parameter passed to `generate_biomodel`. Default to 1e-6.
        rtol (float, optional): parameter passed to `generate_biomodel`. Default to 1e-12.
        mxstep (int, optional): parameter passed to `generate_biomodel`. Default to 5000000.
        solver_type (str, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 'diffrax'.
        diffrax_solver (str, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 'Tsit5'.

    Returns:
        tuple containing

        - model (ModelRollout): generated model rollout module
        - y0 (jax.numpy.Array): default initial state of y variable(as provided in the SBML file)
        - w0 (jax.numpy.Array): default initial state of w variable (as provided in the SBML file)
        - c (jax.numpy.Array): default values of constant kinematic parameters c (as provided in the SBML file)
    """
    model_fp = generate_biomodel(model_idx, model_fp=model_fp,
                                 vary_constant_reactants=vary_constant_reactants, vary_boundary_reactants=vary_boundary_reactants,
                                 deltaT=deltaT, atol=atol, rtol=rtol, mxstep=mxstep,
                                 solver_type=solver_type, diffrax_solver=diffrax_solver)
    spec = importlib.util.spec_from_file_location("JaxModelSpec", model_fp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls = getattr(module, "ModelRollout")
    model = model_cls()
    y0 = getattr(module, "y0")
    w0 = getattr(module, "w0")
    c = getattr(module, "c")

    return model, y0, w0, c


