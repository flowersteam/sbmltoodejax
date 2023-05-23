import libsbml
import os
import sbmltoodepy
from tempfile import NamedTemporaryFile

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
        filePath = file
        libsbml.readSBML(filePath)
        doc = libsbml.readSBML(file)

    else:
        tmp_sbml_file = NamedTemporaryFile(suffix=".xml")
        with open(tmp_sbml_file.name, 'w') as f:
            f.write(file)
        doc = libsbml.readSBMLFromString(file)
        filePath = tmp_sbml_file.name

    # Raise an Error if SBML error
    if doc.getNumErrors() > 0:
        raise ValueError("LibSBML read error")
    
    # Raise an Error if the model contains events as they are not handled by SBMLtoODEpy
    model = doc.getModel()
    if model.getNumEvents() > 0:
        raise NotImplementedError("Events are not handled")

    modelData = sbmltoodepy.parse.ParseSBMLFile(filePath)
    
    return modelData


