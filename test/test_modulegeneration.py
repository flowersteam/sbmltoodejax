from sbmltoodejax.utils import load_biomodel

def test_modulegeneration():
    """
    Check the initial assignments of models states and (constant) parameter values with respect to onrs specified in PDF files on BioModels website
    """

    #BIOMD 3
    model, y0, w0, c = load_biomodel(3)
    y_indexes = model.modelstepfunc.y_indexes
    c_indexes = model.modelstepfunc.c_indexes
    assert y0[y_indexes['C']] == 0.01
    assert y0[y_indexes['M']] == 0.01
    assert y0[y_indexes['X']] == 0.01
    assert c[c_indexes['VM1']] == 3.0
    assert c[c_indexes['VM3']] == 1.
    assert c[c_indexes['Kc']] == 0.5

    # BIOMD 4
    model, y0, w0, c = load_biomodel(4)
    y_indexes = model.modelstepfunc.y_indexes
    c_indexes = model.modelstepfunc.c_indexes

    assert y0[y_indexes['C']] == 0.01
    assert y0[y_indexes['MI']] == 0.99
    assert y0[y_indexes['M']] == 0.01
    assert y0[y_indexes['XI']] == 0.99
    assert y0[y_indexes['X']] == 0.01
    assert c[c_indexes['VM1']] == 3.0
    assert c[c_indexes['VM3']] == 1.
    assert c[c_indexes['Kc']] == 0.5


    # BIOMD 6
    model, y0, w0, c = load_biomodel(6)
    y_indexes = model.modelstepfunc.y_indexes
    c_indexes = model.modelstepfunc.c_indexes

    y0[y_indexes['EmptySet']] == 1.0
    assert y0[y_indexes['z']] == 0.0
    assert y0[y_indexes['u']] == 0.0
    assert y0[y_indexes['v']] == 0.0
    assert c[c_indexes['kappa']] == 0.015
    assert c[c_indexes['k6']] == 1.0
    assert c[c_indexes['k4']] == 180.0
    assert c[c_indexes['k4prime']] == 0.018


    # BIOMD 8
    model, y0, w0, c = load_biomodel(8)
    y_indexes = model.modelstepfunc.y_indexes
    c_indexes = model.modelstepfunc.c_indexes

    assert y0[y_indexes['C']] == 0.0
    assert y0[y_indexes['M']] == 0.0
    assert y0[y_indexes['X']] == 0.0
    assert y0[y_indexes['Y']] == 1.0
    assert y0[y_indexes['Z']] == 1.0
    assert c[c_indexes['K6']] == 0.3
    assert c[c_indexes['V1p']] == 0.75
    assert c[c_indexes['V3p']] == 0.3

    # BIOMD 10
    model, y0, w0, c = load_biomodel(10)
    y_indexes = model.modelstepfunc.y_indexes
    c_indexes = model.modelstepfunc.c_indexes

    assert y0[y_indexes['MKKK']] == 90.0
    assert y0[y_indexes['MKKK_P']] == 10.0
    assert y0[y_indexes['MKK']] == 280.0
    assert y0[y_indexes['MKK_P']] == 10.0
    assert y0[y_indexes['MKK_PP']] == 10.0
    assert y0[y_indexes['MAPK']] == 280.0
    assert y0[y_indexes['MAPK_P']] == 10.0
    assert y0[y_indexes['MAPK_PP']] == 10.0


