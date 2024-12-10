import diffrax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax import vmap
from sbmltoodejax.utils import load_biomodel

def test_solver():
    """
    Check that solvers (odeint and diffrax variants) provide valid results for the biomodels
    """
    
    for biomodel_id in [10, 37, 50, 52, 84, 167, 240]:

        # load model
        model, y0, w0, c = load_biomodel(biomodel_id)
        
        # run model for 3 seconds  with odeint solver
        n_secs = 3
        deltaT = 0.1
        n_steps = int(n_secs / deltaT)
        ts = jnp.linspace(0, n_secs, n_steps)
        def ode_func(y, t, args):
            w, c = args
            w = model.assignmentfunc(y, w, c, t)
            dy_dt = model.ratefunc(y, t, w, c)
            return dy_dt
        ys = odeint(ode_func, y0, ts, (w0, c), rtol=1e-12, atol=1e-6, mxstep=5000000)
        ws = vmap(lambda t, y: model.assignmentfunc(y, w0, c, t))(ts, ys)
        ys = jnp.moveaxis(ys, 0, -1)
        ws = jnp.moveaxis(ws, 0, -1)

        for diffrax_solver in ['Tsit5', 'Dopri5', 'Dopri8', 'Midpoint', 'Heun', 'Bosh3', 'Ralston']: 
            
            # run model with diffrax solver
            model = model.__class__(solver=eval(f"diffrax.{diffrax_solver}()"))
            ys_diffrax, ws_diffrax, _ = model(n_secs, deltaT=deltaT)

            # Assert that difference with odeint solver is small
            thresh = 0.05
            eps = 1e-4
            assert (jnp.abs(ys-ys_diffrax)<=max(thresh*(ys.max()-ys.min()), eps)).all()
            if ws.shape[0] > 0:
                assert (jnp.abs(ws-ws_diffrax)<=max(thresh*(ws.max()-ws.min()), eps)).all()


