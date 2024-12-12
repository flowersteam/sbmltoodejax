import importlib
from sbmltoodejax.biomodels_api import get_content_for_model
from sbmltoodejax.modulegeneration import GenerateModel
from sbmltoodejax.parse import ParseSBMLFile

def generate_biomodel(model_idx, model_fp="jax_model.py",
                      vary_constant_reactants=False, vary_boundary_reactants=False,
                      diffrax_solver='Tsit5', iterative_solve=True,
                      deltaT=0.1, atol=1e-6, rtol=1e-12, max_steps=5000000):
    """Calls the `sbmltoodejax.modulegeneration.GenerateModel` for a SBML model hosted on the BioModel website and indexed by the provided `model_idx`.

    Args:
        model_idx: either an integer, or a valid model id
        model_fp (str): filepath for the generated file
        vary_constant_reactants (bool, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to False.
        vary_boundary_reactants (bool, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to False.
        diffrax_solver (str, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 'Tsit5'.
        iterative_solve (bool, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to True.
        deltaT (float, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 0.1.
        atol (float, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 1e-6.
        rtol (float, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 1e-12.
        max_steps (int, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 5000000.        

    Returns:
        model_fp (str): the filepath containing the generated python file
    """
    model_xml_body = get_content_for_model(model_idx)
    model_data = ParseSBMLFile(model_xml_body)
    GenerateModel(model_data, model_fp,
                  vary_constant_reactants=vary_constant_reactants, vary_boundary_reactants=vary_boundary_reactants,
                  diffrax_solver=diffrax_solver, iterative_solve=iterative_solve,
                  deltaT=deltaT, atol=atol, rtol=rtol, max_steps=max_steps)

    return model_fp


def load_biomodel(model_idx, model_fp="jax_model.py",
                  vary_constant_reactants=False, vary_boundary_reactants=False,
                  diffrax_solver='Tsit5', iterative_solve=True,
                  deltaT=0.1, atol=1e-6, rtol=1e-12, max_steps=5000000):
    """Calls the generate_biomodel function for a SBML model hosted on the BioModel website and indexed by the provided `model_idx`,
    then loads and returns the generated `model` module and `y0`, `w0`, `c` variables.

    Args:
        model_idx: either an integer, or a valid model id
        model_fp (str): filepath for the generated file
        vary_constant_reactants (bool, optional): parameter passed to `generate_biomodel`. Default to False.
        vary_boundary_reactants (bool, optional): parameter passed to `generate_biomodel`. Default to False.
        diffrax_solver (str, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to 'Tsit5'.
        iterative_solve (bool, optional): parameter passed to `sbmltoodejax.modulegeneration.GenerateModel`. Default to True.
        deltaT (float, optional): parameter passed to `generate_biomodel`. Default to 0.1.
        atol (float, optional): parameter passed to `generate_biomodel`. Default to 1e-6.
        rtol (float, optional): parameter passed to `generate_biomodel`. Default to 1e-12.
        max_steps (int, optional): parameter passed to `generate_biomodel`. Default to 5000000.        

    Returns:
        tuple containing

        - model (ModelRollout): generated model rollout module
        - y0 (jax.numpy.Array): default initial state of y variable(as provided in the SBML file)
        - w0 (jax.numpy.Array): default initial state of w variable (as provided in the SBML file)
        - c (jax.numpy.Array): default values of constant kinematic parameters c (as provided in the SBML file)
    """
    model_fp = generate_biomodel(model_idx, model_fp=model_fp,
                                 vary_constant_reactants=vary_constant_reactants, vary_boundary_reactants=vary_boundary_reactants,
                                 diffrax_solver=diffrax_solver, iterative_solve=iterative_solve,
                                 deltaT=deltaT, atol=atol, rtol=rtol, max_steps=max_steps)
    spec = importlib.util.spec_from_file_location("JaxModelSpec", model_fp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls = getattr(module, "ModelRollout")
    model = model_cls()
    y0 = getattr(module, "y0")
    w0 = getattr(module, "w0")
    c = getattr(module, "c")

    return model, y0, w0, c


