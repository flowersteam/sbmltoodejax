import importlib
import jax
import jax.numpy as jnp
import numpy as np
from sbmltoodepy.modulegeneration import GenerateModel
from sbmltoodejax.modulegeneration import GenerateModel as GenerateJaxModel
from sbmltoodejax.parse import ParseSBMLFile
from tempfile import NamedTemporaryFile
import time
from urllib.request import urlopen


def get_sbmltoodepy_model_variables(model, y_indexes, w_indexes, c_indexes):
    y = np.zeros(len(y_indexes))
    w = np.zeros(len(w_indexes))
    c = np.zeros(len(c_indexes))

    for k, v in model.s.items():
        if k in y_indexes:
            y[y_indexes[k]] = v.amount
        elif k in w_indexes:
            w[w_indexes[k]] = v.amount
        elif k in c_indexes:
            c[c_indexes[k]] = v.amount

    for k, v in model.p.items():
        if k in y_indexes:
            y[y_indexes[k]] = v.value
        elif k in w_indexes:
            w[w_indexes[k]] = v.value
        elif k in c_indexes:
            c[c_indexes[k]] = v.value

    for k, v in model.c.items():
        if k in y_indexes:
            y[y_indexes[k]] = v.size
        elif k in w_indexes:
            w[w_indexes[k]] = v.size
        elif k in c_indexes:
            c[c_indexes[k]] = v.size

    for k, v in model.r.items():
        for sub_k, sub_v in v.p.items():
            if f"{k}_{sub_k}" in w_indexes:
                w[w_indexes[f"{k}_{sub_k}"]] = sub_v.value
            elif f"{k}_{sub_k}" in c_indexes:
                c[c_indexes[f"{k}_{sub_k}"]] = sub_v.value

    return y, w, c

def test_modulegeneration():
    jax.config.update("jax_platform_name", "cpu")

    for model_idx in [3, 4, 6, 8, 10, 12, 14, 21]:

        model_url = f'https://www.ebi.ac.uk/biomodels/model/download/BIOMD{model_idx:010d}.2?filename=BIOMD{model_idx:010d}_url.xml'
        with urlopen(model_url) as response:
            model_data = ParseSBMLFile(response.read().decode("utf-8"))
        model_py_file = NamedTemporaryFile(suffix=".py")
        model_jax_file = NamedTemporaryFile(suffix=".py")
        # from collections import namedtuple
        # tmpfile = namedtuple("tmpfile", "name")
        # model_py_file = tmpfile("py_model.py")
        # model_jax_file = tmpfile("jax_model.py")
        model_name = "Model"

        deltaT = 0.01
        atol = 1e-6
        rtol = 1e-12
        mxstep = 5000
        n_secs = 200
        n_steps = int(n_secs/deltaT)

        # Load Jax-based Model
        model_rollout_name = 'ModelRollout'
        GenerateJaxModel(model_data, model_jax_file.name,
                         ModelRolloutName=model_rollout_name)

        module_name = "jaxmodelfuncs"
        spec = importlib.util.spec_from_file_location(
            module_name, model_jax_file.name
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modelrolloutfunc_cls = getattr(module, model_rollout_name)
        model_rollout = modelrolloutfunc_cls(deltaT=deltaT, atol=atol, rtol=rtol, mxstep=mxstep)

        c = getattr(module, "c")
        c_indexes = getattr(module, "c_indexes")
        y0 = getattr(module, "y0")
        y_indexes = getattr(module, "y_indexes")
        w0 = getattr(module, "w0")
        w_indexes = getattr(module, "w_indexes")
        t0 = getattr(module, "t0")

        # Simulate for n time steps
        jaxcstart = time.time()
        ys, ws, times = model_rollout(n_steps, y0, w0, c, t0)
        jaxcend = time.time()


        # Load original Numpy/Scipy-based Model
        GenerateModel(model_data, model_py_file.name, objectName=model_name)
        module_name = "modelclass"
        spec = importlib.util.spec_from_file_location(
            module_name, model_py_file.name
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_cls = getattr(module, model_name)
        model = model_cls()
        py_y0, py_w0, py_c0 = get_sbmltoodepy_model_variables(model, y_indexes, w_indexes, c_indexes)

        # Simulate for n time steps
        cstart = time.time()
        py_ys = np.zeros((len(y_indexes), n_steps))
        py_ys[:, 0] = py_y0
        py_ws = np.zeros((len(w_indexes), n_steps))
        py_ws[:, 0] = py_w0
        py_times = np.zeros(n_steps)
        py_times[0] = 0.0
        for i in range(1, n_steps):
            model.RunSimulation(
                deltaT, absoluteTolerance=atol, relativeTolerance=rtol
            )
            py_times[i] = model.time
            py_y, py_w, py_c = get_sbmltoodepy_model_variables(model, y_indexes, w_indexes, c_indexes)
            py_ys[..., i] = py_y
            py_ws[..., i] = py_w
        cend = time.time()


        # check that initial values are the same between pymodel and jaxmodel
        assert jnp.isclose(py_y0, y0).all()
        assert jnp.isclose(py_w0, w0).all()
        assert jnp.isclose(py_c0, c).all()

        # compare plot of intergrated values between pymodel and jaxmodel
        difference_ys = jnp.abs(py_ys - ys)
        difference_ws = jnp.abs(py_ws - ws)

        show_plot = True
        if show_plot:
            import matplotlib.pyplot as plt
            fig = plt.figure(constrained_layout=True, figsize=(20, 10))
            subfigs = fig.subfigures(nrows=2, ncols=1)

            subfigs[0].suptitle(f'Differences in y(t=-1): {difference_ys[:,-1].mean():.3f} (mean), {difference_ys[:,-1].max():.3f} (max)')
            axarr = subfigs[0].subplots(nrows=1, ncols=3)
            for k, k_idx in y_indexes.items():
                axarr[0].plot(times, py_ys[k_idx], label=k)
                axarr[0].set_title(f'sbmltoodepy: {cend-cstart:.2f} secs')
                axarr[1].plot(times, ys[k_idx], label=k)
                axarr[1].set_title(f'sbmltoodejax: {jaxcend - jaxcstart:.2f} secs')
            axarr[0].legend()
            axarr[1].legend()
            im = axarr[2].imshow(difference_ys, cmap="hot", vmin=0, interpolation='nearest', aspect='auto')
            plt.colorbar(im)
            axarr[2].set_yticks(ticks=range(len(y_indexes)), labels=list(y_indexes.keys()))

            if len(w_indexes) > 0:
                subfigs[1].suptitle(f'Differences in w(t=-1): {difference_ws[:, -1].mean():.3f} (mean), {difference_ws[:, -1].max():.3f} (max)')
                axarr = subfigs[1].subplots(nrows=1, ncols=3)
                for k, k_idx in w_indexes.items():
                    axarr[0].plot(times, py_ws[k_idx], label=k)
                    axarr[0].set_title(f'sbmltoodepy: {cend-cstart:.2f} secs')
                    axarr[1].plot(times, ws[k_idx], label=k)
                    axarr[1].set_title(f'sbmltoodejax: {jaxcend - jaxcstart:.2f} secs')
                axarr[0].legend()
                axarr[1].legend()
                im = axarr[2].imshow(difference_ws, cmap="hot", vmin=0, interpolation='nearest', aspect='auto')
                plt.colorbar(im)
                axarr[2].set_yticks(ticks=range(len(w_indexes)), labels=list(w_indexes.keys()))

            plt.suptitle(f"Model #{model_idx}")
            plt.legend()
            plt.show()

        # epsilon_ys = jnp.maximum(0.1 * py_ys.max(1), 0.1)
        # epsilon_ws = jnp.maximum(0.1 * py_ws.max(1), 0.1)
        # assert (difference_ys.max(1) < epsilon_ys).all(), print(model_idx, difference_ys.max(1), epsilon_ys)
        # assert (difference_ws.max(1) < epsilon_ws).all(), print(model_idx, difference_ws.max(1), epsilon_ws)


