import jax
jax.config.update("jax_platform_name", "cpu")

import matplotlib.pyplot as plt
import jax.numpy as jnp
from sbmltoodejax.utils_2 import load_biomodel

# List of solvers to try
solvers = ['odeint', 'Euler', 'Midpoint', 'Heun', 'Ralston', 'Tsit5', 'Dopri5', 'Dopri8']

# Function to run simulation for a given solver
def run_simulation(solver):
    try:
        print(f"Attempting to load biomodel with solver: {solver}")
        if solver == 'odeint':
            model, _, _, c = load_biomodel(240, solver_type='odeint')
        else:
            model, _, _, c = load_biomodel(240, solver_type='diffrax', diffrax_solver=solver)
        print(f"Successfully loaded biomodel with solver: {solver}")
        n_secs = 21 * 3600
        n_steps = int(n_secs / model.deltaT)
        ys, ws, ts = model(n_steps)
        return model, ys, ws, ts, c
    except Exception as e:
        print(f"Error with solver {solver}: {str(e)}")
        return None, None, None, None, None

# Run simulations for each solver
results = {}
w_indexes = None
for solver in solvers:
    model, ys, ws, ts, c = run_simulation(solver)
    if model is not None and ys is not None and ws is not None and ts is not None:
        results[solver] = (model, ys, ws, ts, c)
        if w_indexes is None:
            w_indexes = model.modelstepfunc.w_indexes

# Plot time course simulations
plt.figure(figsize=(12, 8))
for solver, (model, ys, ws, ts, _) in results.items():
    color = plt.cm.tab10(solvers.index(solver) / len(solvers))
    plt.plot(ts/3600, ws[w_indexes["DegU_Total"]], label=f"DegU {solver}", color=color)

plt.xlabel("Reaction time (hours)")
plt.ylabel("Concentration")
plt.xlim([0, 21])
plt.ylim([0, 600])
plt.legend()
plt.title("Comparison of Different Solvers for Biomodel 240")
plt.show()

# Calculate and plot differences from odeint integration
if 'odeint' in results:
    reference_model, reference_ys, reference_ws, reference_ts, _ = results['odeint']
    
    plt.figure(figsize=(12, 8))
    for solver, (model, ys, ws, ts, _) in results.items():
        if solver != 'odeint':
            diff = (ws[w_indexes["DegU_Total"]] - reference_ws[w_indexes["DegU_Total"]])**2
            plt.plot(ts/3600, diff, label=f"{solver}")
    
    plt.xlim([0, 21])
    plt.yscale('log')
    plt.xlabel("Reaction time (hours)")
    plt.ylabel("MSE from odeint (log scale)")
    plt.legend()
    plt.title("Differences from odeint Integration for Biomodel 240")
    plt.show()

    # Print final values
    print("Final values:")
    print(f"odeint: {reference_ws[w_indexes['DegU_Total']][-1]}")
    for solver, (model, ys, ws, _, _) in results.items():
        if solver != 'odeint':
            print(f"{solver:7}: {ws[w_indexes['DegU_Total']][-1]}")

    # Print MSE differences from odeint
    print("\nMSE differences from odeint:")
    for solver, (model, ys, ws, _, _) in results.items():
        if solver != 'odeint':
            MSE = jnp.mean((ws[w_indexes['DegU_Total']] - reference_ws[w_indexes['DegU_Total']])**2)
            print(f"{solver:7}: {MSE}")
else:
    print("odeint solver failed, cannot calculate differences.")
