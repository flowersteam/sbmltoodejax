import jax.numpy as jnp
from sbmltoodejax import jaxfuncs
import re
import sys

def GenerateModel(modelData, outputFilePath,
                  RateofSpeciesChangeName: str ='RateofSpeciesChange',
                  AssignmentRuleName: str='AssignmentRule',
                  ModelStepName: str='ModelStep',
                  ModelRolloutName: str='ModelRollout',
                  vary_constant_reactants: bool=False,
                  vary_boundary_reactants: bool=False,
                  deltaT: float =0.1,
                  atol: float=1e-6,
                  rtol: float = 1e-12,
                  mxstep: int = 5000000,
                  solver_type: str = 'odeint',
                  diffrax_solver: str = 'Tsit5'
                  ):
    """
    This function takes model data created by :func:`~sbmltoodejax.parse.ParseSBMLFile` and generates a python file containing
    variables and modules that implement the SBML model.

    Note:
        This function is adapted from ``sbmltoodepy.modulegeneration.GenerateModel`` function, however the generated python file is
        written in JAX and follows different conventions for the generated variables and modules, as detailed in :ref:`structure-of-the-generated-python-file`


    Args:
        modelData (sbmltoodepy.dataclasses.ModelData): An object containing all the model components and values.

        outputFilePath (str): The desired file path of the resulting python file.

        RateofSpeciesChangeName (str, optional): The name of the RateofSpeciesChange module defined in the resulting python file. Default to 'RateofSpeciesChange'.

        AssignmentRuleName (str): The name of the AssignmentRule module defined in the resulting python file. Default to 'AssignmentRule'.

        ModelStepName (str): The name of the ModelStep module defined in the resulting python file. Default to 'ModelStep'.

        ModelRolloutName (str): The name of the ModelRollout module defined in the resulting python file. Default to'ModelRollout'.

        deltaT (float): Time step size (in seconds). Default to 0.1.

        atol (float): Absolute local error tolerance for ``jax.experimental.odeint`` solver. Default to 1e-6.

        rtol (float): Relative local error tolerance for ``jax.experimental.odeint`` solver. Default to 1e-12.

        mxstep (int): Maximum number of steps to take for each timepoint for ``jax.experimental.odeint`` solver. Default to 5000000.


    """

    jnp.set_printoptions(threshold=sys.maxsize)

    outputFile = open(outputFilePath, "w")

    parameters = modelData.parameters
    compartments = modelData.compartments
    for k, v in compartments.items():
        if not v.isConstant:
            raise NotImplementedError("Varying compartment size is not handled")
    species = modelData.species
    reactions = modelData.reactions
    functions = modelData.functions

    assignmentRules = modelData.assignmentRules
    rateRules = modelData.rateRules
    initialAssignments = modelData.initialAssignments

    mathFuncs = {'abs': 'jnp.abs',
                 'max': 'jnp.max',
                 'min': 'jnp.min',
                 'pow': 'jnp.power',
                 'exp': 'jnp.exp',
                 'floor': 'jnp.floor',
                 'ceiling': 'jnp.ceil',
                 'ln': 'jnp.log',
                 'log': 'jnp.log10',
                 'factorial': 'jaxfuncs.factorial',
                 'sqrt': 'no.sqrt',

                 'eq': 'jnp.equal',
                 'neq': 'jnp.not_equal',
                 'gt': 'jnp.greater',
                 'lt': 'jnp.less',
                 'geq': 'jnp.greater_equal',
                 'leq': 'jnp.less_equal',

                 'and': 'and',
                 'or': 'or',
                 'xor': 'jnp.logical_xor',
                 'not': 'not',

                 'sin': 'jnp.sin',
                 'cos': 'jnp.cos',
                 'tan': 'jnp.tan',
                 'sec': 'jaxfuncs.sec',
                 'csc': 'jaxfuncs.csc',
                 'cot': 'jaxfuncs.cot',
                 'sinh': 'jnp.sinh',
                 'cosh': 'jnp.cosh',
                 'tanh': 'jnp.tanh',
                 'sech': 'jaxfuncs.sech',
                 'csch': 'jaxfuncs.csch',
                 'coth': 'jaxfuncs.coth',
                 'arcsin': 'jnp.arcsin',
                 'arccos': 'jnp.arccos',
                 'arctan': 'jnp.arctan',
                 'arcsinh': 'jnp.arcsinh',
                 'arccosh': 'jnp.arccosh',
                 'arctanh': 'jnp.arctanh',

                 'true': 'True',
                 'false': 'False',
                 'notanumber': 'jnp.nan',
                 'pi': 'jnp.pi',
                 'infinity': 'jnp.inf',
                 'exponentiale': 'jnp.e',
                 'piecewise': 'jaxfuncs.piecewise'
                 }

    # TODO: Add in user defined functions

    # ================================================================================================================================

    outputFile.write("import equinox as eqx\n")
    outputFile.write("from functools import partial\n")
    outputFile.write("from jax import jit, lax, vmap\n")
    outputFile.write("from jax.experimental.ode import odeint\n")
    outputFile.write("import jax.numpy as jnp\n")
    outputFile.write("from diffrax import ODETerm, Tsit5, Dopri5, Dopri8, Euler, Midpoint, Heun, Bosh3, Ralston\n")
    outputFile.write("from typing import Any\n\n")
    outputFile.write("from sbmltoodejax import jaxfuncs\n\n")


    # ================================================================================================================================
    t0 = 0.0

    y0 = []
    y_indexes = {}
    for reaction_name, reaction in reactions.items():
        reactants = [reactant for (reactantCoeff, reactant) in reaction.reactants]
        for reactant in reactants:
            if reactant not in y_indexes:
                if reactant in species:
                    if species[reactant].valueType == "Amount":
                        y_amount = species[reactant].value
                    elif species[reactant].valueType == "Concentration":
                        y_amount = species[reactant].value * compartments[species[reactant].compartment].size
                    else:
                        raise ValueError("Specie value is not of type amount nor concentration")
                else:
                    raise NotImplementedError("Reactant is not a specie")
                if not species[reactant].isConstant or vary_constant_reactants:
                    y0.append(y_amount)
                    y_indexes[reactant] = len(y0) - 1


    for rule_name, rule in rateRules.items():
        if rule.variable not in y_indexes:
            if rule.variable in species:
                if species[rule.variable].valueType == "Amount":
                    y_amount = species[rule.variable].value
                elif species[rule.variable].valueType == "Concentration":
                    y_amount = species[rule.variable].value * compartments[species[rule.variable].compartment].size
                else:
                    raise ValueError("Specie value is not of type amount nor concentration")
            elif rule.variable in parameters:
                y_amount = parameters[rule.variable].value
            elif rule.variable in compartments:
                raise NotImplementedError("Varying compartment size is not handled")
            y0.append(y_amount)
            y_indexes[rule.variable] = len(y0) - 1

    w0 = []
    w_indexes = {}
    for k, v in assignmentRules.items():
        if v.variable in species:
            if species[v.variable].valueType == "Amount":
                w_amount = species[v.variable].value
            elif species[v.variable].valueType == "Concentration":
                w_amount = species[v.variable].value * compartments[species[v.variable].compartment].size
            else:
                raise ValueError("Specie value is not of type amount nor concentration")
        elif v.variable in parameters:
            w_amount = parameters[v.variable].value
        elif v.variable in compartments:
            w_amount = compartments[v.variable].size
        w0.append(w_amount)
        w_indexes[v.variable] = len(w0) - 1


    # Add all parameters that are not in w_indexes or y_indexes
    c = []
    c_indexes = {}
    for k, v in species.items():
        if (k not in y_indexes) and (k not in w_indexes):
            if v.valueType == "Amount":
                c_amount = v.value
            elif v.valueType == "Concentration":
                c_amount = v.value * compartments[v.compartment].size
            else:
                raise ValueError("Specie value is not of type amount nor concentration")
            c.append(c_amount)
            c_indexes[k] = len(c) - 1
    for k, v in parameters.items():
        if (k not in y_indexes) and (k not in w_indexes):
            c.append(parameters[k].value)
            c_indexes[k] = len(c) - 1
    for k, v in compartments.items():
        if (k not in y_indexes) and (k not in w_indexes):
            c.append(compartments[k].size)
            c_indexes[k] = len(c) - 1

    # Add constant parameters that are defined in reactions or that for some reason are not written as "constant" in their definition
    for reaction_name, reaction in reactions.items():
        for param in reaction.rxnParameters:
            param_name = reaction_name + "_" + param[0]
            assert (param_name not in y_indexes) and (param_name not in w_indexes)
            if (param_name not in c_indexes):
                c.append(param[1])
                c_indexes[param_name] = len(c) - 1

    # ================================================================================================================================

    def ParseLHS(rawLHS):

        assert rawLHS in w_indexes
        returnLHS = f"w = w.at[{w_indexes[rawLHS]}].set("
        if rawLHS in species:
            if not species[rawLHS].hasOnlySubstanceUnits:
                returnLHS += f"{compartments[species[rawLHS].compartment].size} * "
        returnLHS += "("

        return returnLHS

    def ParseRHS(rawRateLaw, extended_param_names=[], reaction_name=None, yvar="y", wvar="w", cvar="c", tvar="t"):
        # The main purpose of this function is to turn math strings given by libSBML into code formated to properly call self.c, self.w and y

        rawRateLaw = rawRateLaw.replace("^", "**").replace("&&", "&").replace("||", "|")  # Replace not understood operators
        variables = []
        for match in re.finditer(r'\b[a-zA-Z_]\w*', rawRateLaw):  # look for variable names
            # ToDo: check for function calls
            variables.append([rawRateLaw[match.start():match.end()], match.span()])

        returnRHS = ''
        oldSpan = None
        if variables != []:
            for variable in variables:
                if oldSpan == None and variable[1][0] != 0:
                    returnRHS += rawRateLaw[0:variable[1][0]]
                elif oldSpan != None:
                    returnRHS += rawRateLaw[oldSpan[1]:variable[1][0]]
                oldSpan = variable[1]

                if variable[0] in extended_param_names and reaction_name is not None:
                    variable[0] = reaction_name + "_" + variable[0]

                if variable[0] in species and not species[variable[0]].hasOnlySubstanceUnits:
                    returnRHS += '('

                if variable[0] in c_indexes:
                    returnRHS += f'{cvar}[{c_indexes[variable[0]]}]'
                elif variable[0] in w_indexes:
                    returnRHS += f'{wvar}[{w_indexes[variable[0]]}]'
                elif variable[0] in y_indexes:
                    returnRHS += f'{yvar}[{y_indexes[variable[0]]}]'
                elif variable[0] in mathFuncs:
                    returnRHS += mathFuncs[variable[0]]
                elif variable[0] in functions:
                    raise NotImplementedError("Custom functions are not handled")
                elif variable[0] == "time":
                    returnRHS += f'{tvar}'
                elif variable[0] == "pi":
                    returnRHS += "jnp.pi"
                else:
                    raise (Exception('New case: unkown Reaction variable: ' + variable[0]))

                if variable[0] in species and not species[variable[0]].hasOnlySubstanceUnits:
                    returnRHS += f'/{compartments[species[variable[0]].compartment].size})'
            returnRHS += rawRateLaw[variable[1][1]:len(rawRateLaw)]

        else:
            returnRHS = rawRateLaw

        return returnRHS

    # ================================================================================================================================
    ruleDefinedVars = [rule.variable for rule in assignmentRules.values()]
    for key, assignment in initialAssignments.items():
        ruleDefinedVars.append(assignment.variable)

    for key, rule in assignmentRules.items():
        rule.dependents = []
        for match in re.finditer(r'\b[a-zA-Z_]\w*', rule.math):  # look for variable names
            rule.dependents.append(rule.math[match.start():match.end()])
        originalLen = len(rule.dependents)
        for i in range(originalLen):
            if rule.dependents[originalLen - i - 1] not in ruleDefinedVars:
                rule.dependents.pop(originalLen - i - 1)

    for key, assignment in initialAssignments.items():
        assignment.dependents = []
        for match in re.finditer(r'\b[a-zA-Z_]\w*', assignment.math):  # look for variable names
            assignment.dependents.append(assignment.math[match.start():match.end()])
        originalLen = len(assignment.dependents)
        for i in range(originalLen):
            if assignment.dependents[originalLen - i - 1] not in ruleDefinedVars:
                assignment.dependents.pop(originalLen - i - 1)

    while True:
        continueVar = False
        breakVar = True
        varDefinedThisLoop = None
        for key, rule in assignmentRules.items():
            if rule.dependents == []:
                var_amount = eval(ParseRHS(rule.math, yvar="y0", wvar="w0", cvar="c", tvar="t0"))
                if rule.variable in species and not species[rule.variable].hasOnlySubstanceUnits:
                    var_amount *= compartments[species[rule.variable].compartment].size
                if isinstance(var_amount, jnp.ndarray):
                    var_amount = var_amount.item()
                if rule.variable in y_indexes:
                    y0[y_indexes[rule.variable]] = var_amount
                elif rule.variable in w_indexes:
                    w0[w_indexes[rule.variable]] = var_amount
                else:
                    raise ValueError("Rule variable is not in y nor w")
                varDefinedThisLoop = rule.variable
                rule.dependents = None
                continueVar = True
                breakVar = False
                break
            elif not rule.dependents == None:
                breakVar = False

        if not continueVar:
            for key, assignment in initialAssignments.items():
                if assignment.dependents == []:
                    var_amount = eval(ParseRHS(assignment.math, yvar="y0", wvar="w0", cvar="c"))
                    if assignment.variable in species and not (
                    species[assignment.variable].hasOnlySubstanceUnits):
                        var_amount *= compartments[species[assignment.variable].compartment].size
                    if isinstance(var_amount, jnp.ndarray):
                        var_amount = var_amount.item()
                    if assignment.variable in y_indexes:
                        y0[y_indexes[assignment.variable]] = var_amount
                    elif assignment.variable in w_indexes:
                        w0[w_indexes[assignment.variable]] = var_amount
                    elif assignment.variable in c_indexes:
                        c[c_indexes[assignment.variable]] = var_amount
                    else:
                        raise ValueError("Assignment variable is not in y, w nor c")
                    varDefinedThisLoop = assignment.variable
                    assignment.dependents = None
                    continueVar = True
                    breakVar = False
                    break
                elif not assignment.dependents == None:
                    breakVar = False

        for rule in assignmentRules.values():
            if not rule.dependents == None:
                originalLen = len(rule.dependents)
                for i in range(originalLen):
                    if rule.dependents[originalLen - i - 1] == varDefinedThisLoop:
                        rule.dependents.pop(originalLen - i - 1)

        for assignment in initialAssignments.values():
            if not assignment.dependents == None:
                originalLen = len(assignment.dependents)
                for i in range(originalLen):
                    if assignment.dependents[originalLen - i - 1] == varDefinedThisLoop:
                        assignment.dependents.pop(originalLen - i - 1)

        if continueVar:
            continue
        elif breakVar:
            break
        else:
            raise Exception('Algebraic Loop in AssignmentRules')

    # ================================================================================================================================
    outputFile.write(f"t0 = {t0}\n\n")

    outputFile.write(f"y0 = jnp.array({y0})\n")
    outputFile.write(f"y_indexes = {y_indexes}\n\n")

    outputFile.write(f"w0 = jnp.array({w0})\n")
    outputFile.write(f"w_indexes = {w_indexes}\n\n")

    outputFile.write(f"c = jnp.array({c}) \n")
    outputFile.write(f"c_indexes = {c_indexes}\n\n")
    # ================================================================================================================================

    # Set up stoichCoeffMat, a matrix of stoichiometric coefficients for solving the reactions
    reactionCounter = 0
    reactionIndex = {}

    stoichCoeffMat = jnp.zeros([len(y_indexes), max(len(reactions), 1)])

    for rxnId in reactions:
        reactionIndex[rxnId] = reactionCounter
        reactionCounter += 1
        reaction = reactions[rxnId]
        for reactant in reaction.reactants:
            if (reactant[1] in y_indexes):
                if (not species[reactant[1]].isBoundarySpecies) or vary_boundary_reactants:
                    stoichCoeffMat = stoichCoeffMat.at[y_indexes[reactant[1]], reactionIndex[rxnId]].add(reactant[0])

    rateArray = ['0.0'] * len(y_indexes)
    for rule_name, rule in rateRules.items():
        rateArray[y_indexes[rule.variable]] = 'self.Rate' + rule.variable + '(y, w, c, t)'


    # Write
    outputFile.write("class " + RateofSpeciesChangeName + "(eqx.Module):\n")
    outputFile.write(f"\tstoichiometricMatrix = jnp.array({str(stoichCoeffMat.tolist())}, dtype=jnp.float32) \n\n")

    outputFile.write("\t@jit\n")
    outputFile.write("\tdef __call__(self, y, t, w, c):\n")



    outputFile.write('\t\trateRuleVector = jnp.array([' + ', '.join(var for var in rateArray) + '], dtype=jnp.float32)\n\n')
    outputFile.write('\t\treactionVelocities = self.calc_reaction_velocities(y, w, c, t)\n\n')

    outputFile.write('\t\trateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector\n\n')
    outputFile.write('\t\treturn rateOfSpeciesChange\n\n')

    outputFile.write(f'\n\tdef calc_reaction_velocities(self, y, w, c, t):\n')

    reactionElements = ''

    outputFile.write('\t\treactionVelocities = jnp.array([')
    if reactions:
        for reactionId in reactions:
            if reactionElements == '':
                reactionElements += ('self.' + str(reactionId) + '(y, w, c, t)')
            else:
                reactionElements += (', self.' + str(reactionId) + '(y, w, c, t)')
    else:
        reactionElements = '0'
    outputFile.write(reactionElements + '], dtype=jnp.float32)\n\n')

    outputFile.write('\t\treturn reactionVelocities\n\n')

    for reaction_name in reactions.keys():
        outputFile.write(f'\n\tdef {reaction_name}(self, y, w, c, t):\n')
        rxnParamNames = [param[0] for param in reactions[reaction_name].rxnParameters]
        rateLaw = ParseRHS(reactions[reaction_name].rateLaw, extended_param_names=rxnParamNames, reaction_name=reaction_name, yvar="y", wvar="w", cvar="c")
        outputFile.write('\t\treturn ' + rateLaw + '\n\n')

    for key, rateRule in rateRules.items():
        outputFile.write("\tdef Rate" + rateRule.variable + "(self, y, w, c, t):\n")
        rateLaw = ParseRHS(rateRule.math, yvar="y", wvar="w", cvar="c")
        outputFile.write('\t\treturn ' + rateLaw + '\n\n')

    # ================================================================================================================================

    outputFile.write("class " + AssignmentRuleName + "(eqx.Module):\n")

    outputFile.write("\t@jit\n")
    outputFile.write("\tdef __call__(self, y, w, c, t):\n")

    ruleDefinedVars = [rule.variable for rule in assignmentRules.values()]

    for key, rule in assignmentRules.items():
        rule.dependents = []
        for match in re.finditer(r'\b[a-zA-Z_]\w*', rule.math):  # look for variable names
            rule.dependents.append(rule.math[match.start():match.end()])
        originalLen = len(rule.dependents)
        for i in range(originalLen):
            if rule.dependents[originalLen - i - 1] not in ruleDefinedVars:
                rule.dependents.pop(originalLen - i - 1)

    while True:
        continueVar = False
        breakVar = True
        varDefinedThisLoop = None
        for key, rule in assignmentRules.items():
            if rule.dependents == []:
                ruleLHS = ParseLHS(rule.variable)
                ruleRHS = ParseRHS(rule.math, yvar="y", wvar="w", cvar="c")
                outputFile.write("\t\t" + ruleLHS + ruleRHS + '))\n\n')
                varDefinedThisLoop = rule.variable
                rule.dependents = None
                continueVar = True
                breakVar = False
                break
            elif not rule.dependents == None:
                breakVar = False

        for rule in assignmentRules.values():
            if not rule.dependents == None:
                originalLen = len(rule.dependents)
                for i in range(originalLen):
                    if rule.dependents[originalLen - i - 1] == varDefinedThisLoop:
                        rule.dependents.pop(originalLen - i - 1)

        if continueVar:
            continue
        elif breakVar:
            break
        else:
            raise Exception('Algebraic Loop in AssignmentRules')

    outputFile.write("\t\treturn w\n\n")

    # ================================================================================================================================

    outputFile.write("class " + ModelStepName + "(eqx.Module):\n")
    outputFile.write("\ty_indexes: dict = eqx.static_field()\n")
    outputFile.write("\tw_indexes: dict = eqx.static_field()\n")
    outputFile.write("\tc_indexes: dict = eqx.static_field()\n")
    outputFile.write(f"\tratefunc: {RateofSpeciesChangeName}\n")
    outputFile.write("\tatol: float = eqx.static_field()\n")
    outputFile.write("\trtol: float = eqx.static_field()\n")
    outputFile.write("\tmxstep: int = eqx.static_field()\n")
    outputFile.write(f"\tassignmentfunc: {AssignmentRuleName}\n")
    outputFile.write("\tsolver_type: str = eqx.static_field()\n")
    outputFile.write("\tsolver: Any = eqx.static_field()\n\n")

    outputFile.write(f"\tdef __init__(self, "
                     f"y_indexes={y_indexes}, "
                     f"w_indexes={w_indexes}, "
                     f"c_indexes={c_indexes}, "
                     f"atol={atol}, rtol={rtol}, mxstep={mxstep}, "
                     f"solver_type='{solver_type}', diffrax_solver='{diffrax_solver}'):\n\n")

    outputFile.write("\t\tself.y_indexes = y_indexes\n")
    outputFile.write("\t\tself.w_indexes = w_indexes\n")
    outputFile.write("\t\tself.c_indexes = c_indexes\n")
    outputFile.write(f"\t\tself.ratefunc = {RateofSpeciesChangeName}()\n")
    outputFile.write("\t\tself.rtol = rtol\n")
    outputFile.write("\t\tself.atol = atol\n")
    outputFile.write("\t\tself.mxstep = mxstep\n")
    outputFile.write(f"\t\tself.assignmentfunc = {AssignmentRuleName}()\n")
    outputFile.write("\t\tself.solver_type = solver_type\n")
    outputFile.write("\t\tif solver_type == 'odeint':\n")
    outputFile.write("\t\t\tself.solver = odeint\n")
    outputFile.write("\t\telif solver_type == 'diffrax':\n")
    outputFile.write("\t\t\tfrom diffrax import ODETerm, Tsit5, Dopri5, Dopri8, Euler, Midpoint, Heun, Bosh3, Ralston\n")
    outputFile.write("\t\t\tif diffrax_solver == 'Tsit5':\n")
    outputFile.write("\t\t\t\tself.solver = Tsit5()\n")
    outputFile.write("\t\t\telif diffrax_solver == 'Dopri5':\n")
    outputFile.write("\t\t\t\tself.solver = Dopri5()\n")
    outputFile.write("\t\t\telif diffrax_solver == 'Dopri8':\n")
    outputFile.write("\t\t\t\tself.solver = Dopri8()\n")
    outputFile.write("\t\t\telif diffrax_solver == 'Euler':\n")
    outputFile.write("\t\t\t\tself.solver = Euler()\n")
    outputFile.write("\t\t\telif diffrax_solver == 'Midpoint':\n")
    outputFile.write("\t\t\t\tself.solver = Midpoint()\n")
    outputFile.write("\t\t\telif diffrax_solver == 'Heun':\n")
    outputFile.write("\t\t\t\tself.solver = Heun()\n")
    outputFile.write("\t\t\telif diffrax_solver == 'Bosh3':\n")
    outputFile.write("\t\t\t\tself.solver = Bosh3()\n")
    outputFile.write("\t\t\telif diffrax_solver == 'Ralston':\n")
    outputFile.write("\t\t\t\tself.solver = Ralston()\n")
    outputFile.write("\t\t\telse:\n")
    outputFile.write("\t\t\t\traise ValueError(f'Unknown diffrax solver: {diffrax_solver}')\n")
    outputFile.write("\t\telse:\n")
    outputFile.write("\t\t\traise ValueError(f'Unknown solver type: {solver_type}')\n\n")

    outputFile.write("\t@jit\n")
    outputFile.write("\tdef __call__(self, y, w, c, t, deltaT):\n")
    outputFile.write("\t\tif self.solver_type == 'odeint':\n")
    outputFile.write("\t\t\ty_new = odeint(self.ratefunc, y, jnp.array([t, t + deltaT]), w, c, atol=self.atol, rtol=self.rtol, mxstep=self.mxstep)[-1]\n")
    outputFile.write("\t\telse:  # diffrax\n")
    outputFile.write("\t\t\tterm = ODETerm(lambda t, y, args: self.ratefunc(y, t, *args))\n")
    outputFile.write("\t\t\ttprev, tnext = t, t + deltaT\n")
    outputFile.write("\t\t\tstate = self.solver.init(term, tprev, tnext, y, (w, c))\n")
    outputFile.write("\t\t\ty_new, _, _, _, _ = self.solver.step(term, tprev, tnext, y, (w, c), state, made_jump=False)\n")
    outputFile.write("\t\tt_new = t + deltaT\n")
    outputFile.write("\t\tw_new = self.assignmentfunc(y_new, w, c, t_new)\n")
    outputFile.write("\t\treturn y_new, w_new, c, t_new\n\n")

    # ================================================================================================================================

    outputFile.write("class " + ModelRolloutName + "(eqx.Module):\n")
    outputFile.write("\tdeltaT: float = eqx.static_field()\n")
    outputFile.write(f"\tmodelstepfunc: {ModelStepName}\n\n")

    outputFile.write(f"\tdef __init__(self, deltaT={deltaT}, atol={atol}, rtol={rtol}, mxstep={mxstep}, solver_type='{solver_type}', diffrax_solver='{diffrax_solver}'):\n\n")
    outputFile.write("\t\tself.deltaT = deltaT\n")
    outputFile.write(f"\t\tself.modelstepfunc = {ModelStepName}(atol=atol, rtol=rtol, mxstep=mxstep, solver_type=solver_type, diffrax_solver=diffrax_solver)\n\n")

    outputFile.write("\t@partial(jit, static_argnames=(\"n_steps\",))\n")
    outputFile.write("\tdef __call__(self, n_steps, "
                     f"y0=jnp.array({y0}), "
                     f"w0=jnp.array({w0}), "
                     f"c=jnp.array({c}), "
                     f"t0={t0}"
                     f"):\n\n")

    outputFile.write("\t\t@jit\n")
    outputFile.write("\t\tdef f(carry, x):\n")
    outputFile.write("\t\t\ty, w, c, t = carry\n")
    outputFile.write("\t\t\treturn self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)\n")

    outputFile.write("\t\t(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))\n")

    outputFile.write("\t\tys = jnp.moveaxis(ys, 0, -1)\n")
    outputFile.write("\t\tws = jnp.moveaxis(ws, 0, -1)\n")

    outputFile.write("\t\treturn ys, ws, ts\n\n")

    # ================================================================================================================================
    outputFile.close()










