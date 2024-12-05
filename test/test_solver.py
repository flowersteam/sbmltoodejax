from sbmltoodejax.utils import load_biomodel
import jax.numpy as jnp

def test_solver():
    """
    Check that solvers (odeint and diffrax variants) provide valid results for the biomodels
    """
    
    for biomodel_id in [10, 37, 50, 52, 84, 167, 240]:

        # load model with odeint solver
        model, y0, w0, c = load_biomodel(biomodel_id, solver_type="odeint")
        
        # run model for 2 seconds
        n_secs = 2
        n_steps = int(n_secs / model.deltaT)
        ys, ws, _ = model(n_steps)

        for diffrax_solver in ['Tsit5', 'Dopri5', 'Dopri8', 'Midpoint', 'Heun', 'Bosh3', 'Ralston']: # we remove Euler as gives quite different results
            
            # load model with diffrax solver
            model, y0, w0, c = load_biomodel(biomodel_id, solver_type="diffrax", diffrax_solver=diffrax_solver)

            # run model
            ys_diffrax, ws_diffrax, _ = model(n_steps)

            # Assert that difference with odeint solver is small
            thresh = 0.05
            assert (jnp.abs(ys-ys_diffrax)<=thresh*(ys.max(1)-ys.min(1))[:, jnp.newaxis]).all()
            if ws.shape[0] > 0:
                assert (jnp.abs(ws-ws_diffrax)<=thresh*(ws.max(1)-ws.min(1))[:, jnp.newaxis]).all()


