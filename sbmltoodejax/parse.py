import libsbml
import os
from sbmltoodepy.parse import *

def ParseSBMLFile(file: str):
    """
    Wrapper of SBMLtoODEpy's ``ParseSBMLFile`` function.
    This function extract the SBML model’s elements using libSBML, and returns an instance ModelData
    which is intended to be then passed to the :func:`~sbmltoodejax.modulegeneration.GenerateModel`.

    See Also:
        This function buils upon SBMLtoODEpy ``ParseSBMLFile`` function, see their documentation at https://sbmltoodepy.readthedocs.io/en/latest/Parse.html#parse.ParseSBMLFile

    Note:
        Here, at the difference of the original SBMLtoODEpy's function, ``ParseSBMLFile``:

        - takes as input a string which can be a filepath but also directly a string containing the content of the file
          (allows to directly parse online-hosted SBML files, see Example)
        - raises an error if the SBML model contains Events (not handled yet neither in SBMLtoODEpy nor in SBMLtoODEjax)

    Args:
        file (str): can be either the filepath of the SBML model to be parsed or a string with the content of the file

    Returns:
        modelData (sbmltoodepy.dataclasses.ModelData): An sbmltoodepy object containing the model’s components and their properties

    Raises:
        ValueError: if there is an error during LibSBML reading of the file
        NotImplementedError: if the SBML model contains events


    Example:
        .. code-block:: python

            from sbmltoodejax.parse import ParseSBMLFile
            from urllib.request import urlopen

            model_idx = 647
            model_url = f"https://www.ebi.ac.uk/biomodels/model/download/BIOMD{model_idx:010d}.2?filename=BIOMD{model_idx:010d}_url.xml"
            with urlopen(model_url) as response:
                model_xml_body = response.read().decode("utf-8")
            model_data = ParseSBMLFile(model_xml_body)

    """

    if os.path.exists(file):
        libsbml.readSBML(file)
        doc = libsbml.readSBML(file)

    else:
        doc = libsbml.readSBMLFromString(file)

    # Raise an Error if SBML error
    if doc.getNumErrors() > 0:
        raise ValueError("LibSBML read error")
    
    # Raise an Error if the model contains events as they are not handled by SBMLtoODEpy
    model = doc.getModel()
    if model.getNumEvents() > 0:
        raise NotImplementedError("Events are not handled")

    modelData = sbmltoodepy.dataclasses.ModelData()
    for i in range(model.getNumParameters()):
        newParameter = ParseParameterAssignment(i, model.getParameter(i))
        modelData.parameters[newParameter.Id] = newParameter
    for i in range(model.getNumCompartments()):
        newCompartment = ParseCompartment(i, model.getCompartment(i))        
        modelData.compartments[newCompartment.Id] = newCompartment
    for i in range(model.getNumSpecies()):
        newSpecies = ParseSpecies(i, model.getSpecies(i))
        modelData.species[newSpecies.Id] = newSpecies 
    for i in range(model.getNumFunctionDefinitions()):
        newFunction = ParseFunction(i, model.getFunctionDefinition(i))
        modelData.functions[newFunction.Id] = newFunction
    for i in range(model.getNumRules()):
        newRule = ParseRule(i,model.getRule(i))
        if type(newRule) == sbmltoodepy.dataclasses.AssignmentRuleData:
            modelData.assignmentRules[newRule.Id] = newRule
        elif type(newRule) == sbmltoodepy.dataclasses.RateRuleData:
            modelData.rateRules[newRule.Id] = newRule
    for i in range(model.getNumReactions()):
        newReaction = ParseReaction(i, model.getReaction(i))     
        modelData.reactions[newReaction.Id] = newReaction
    for i in range(model.getNumInitialAssignments()):
        newAssignment = ParseInitialAssignment(i, model.getInitialAssignment(i))
        modelData.initialAssignments[newAssignment.Id] = newAssignment
    
    return modelData


