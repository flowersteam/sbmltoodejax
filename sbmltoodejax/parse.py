import libsbml
import os
import sbmltoodepy
from tempfile import NamedTemporaryFile

def ParseSBMLFile(file: str):
    """
    Wrapper of sbmltoodepy's ParseSBMLFile.
    In addition to the original function:

     - file can be a filepath or a string with the content of the file (file.read().decode("utf-8"))
     - throws an error when the sbml file contains events as they are not handled at the moment
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
    
    # Raise an Error if the model contains events as they are not handled by SBMLtoODEpy
    model = doc.getModel()
    if model.getNumEvents() > 0:
        raise NotImplementedError("Events not Handled")

    modelData = sbmltoodepy.parse.ParseSBMLFile(filePath)
    
    return modelData


