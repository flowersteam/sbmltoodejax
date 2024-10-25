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
            model, _, _, _ = load_biomodel(37, solver_type='odeint')
        else:
            model, _, _, _ = load_biomodel(37, solver_type='diffrax', diffrax_solver=solver)
        print(f"Successfully loaded biomodel with solver: {solver}")
        n_secs = 25
        n_steps = int(n_secs / model.deltaT)
        ys, ws, ts = model(n_steps)
        return model, ys, ts
    except Exception as e:
        print(f"Error with solver {solver}: {str(e)}")
        return None, None, None

# Run simulations for each solver
results = {}
y_indexes = None
for solver in solvers:
    model, ys, ts = run_simulation(solver)
    if model is not None and ys is not None and ts is not None:
        results[solver] = (model, ys, ts)
        if y_indexes is None:
            y_indexes = model.modelstepfunc.y_indexes

# Plot time course simulations
plt.figure(figsize=(12, 8))
for solver, (model, ys, ts) in results.items():
    color = plt.cm.tab10(solvers.index(solver) / len(solvers))
    plt.plot(ts, ys[y_indexes["S"]], label=f"S {solver}", color=color)

plt.xlim([0, 25])
plt.ylim([0, 60])
plt.xlabel("Reaction time")
plt.ylabel("Concentration of S")
plt.legend()
plt.title("Comparison of Different Solvers for Biomodel 37")
plt.show()

# Calculate and plot differences from odeint integration
if 'odeint' in results:
    reference_model, reference_ys, reference_ts = results['odeint']
    
    plt.figure(figsize=(12, 8))
    for solver, (model, ys, ts) in results.items():
        if solver != 'odeint':
            diff = (ys[y_indexes["S"]] - reference_ys[y_indexes["S"]])**2
            plt.plot(ts, diff, label=f"{solver}")
    
    plt.xlim([0, 25])
    plt.yscale('log')
    plt.xlabel("Reaction time")
    plt.ylabel("MSE from odeint (log scale)")
    plt.legend()
    plt.title("Differences from odeint Integration for Biomodel 37")
    plt.show()

    # Print final values
    print("Final values:")
    print(f"odeint: {reference_ys[y_indexes['S']][-1]}")
    for solver, (model, ys, _) in results.items():
        if solver != 'odeint':
            print(f"{solver:7}: {ys[y_indexes['S']][-1]}")

    # Print MSE differences from odeint
    print("\nMSE differences from odeint:")
    for solver, (model, ys, _) in results.items():
        if solver != 'odeint':
            MSE = jnp.mean((ys[y_indexes['S']] - reference_ys[y_indexes['S']])**2)
            print(f"{solver:7}: {MSE}")
else:
    print("odeint solver failed, cannot calculate differences.")
