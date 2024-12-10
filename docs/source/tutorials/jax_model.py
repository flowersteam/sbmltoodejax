import diffrax
import equinox as eqx
from jax import jit, lax, vmap
import jax.numpy as jnp
from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.999999999999999, 0.0, 3.0, 7.0])
y_indexes = {'Timeract': 0, 'CellCact': 1, 'Effectoract': 2, 'HR': 3, 'NHEJ': 4}

w0 = jnp.array([10.0, 10.0, 10.0, 9.000000000000002])
w_indexes = {'Effectorina': 0, 'Damage': 1, 'Timerinact': 2, 'CellCina': 3}

c = jnp.array([10.0, 10.0, 10.0, 2.0, 10.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0]) 
c_indexes = {'CellCycletot': 0, 'Effectortot': 1, 'Timertot': 2, 'Kd2t': 3, 'Kti2t': 4, 'Kcc2ch': 5, 'Kt2cc': 6, 'Kcc2a': 7, 'Kd2ch': 8, 'Kch2cc': 9, 'Km1': 10, 'Km10': 11, 'nucleus': 12}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([self.RateTimeract(y, w, c, t), self.RateCellCact(y, w, c, t), self.RateEffectoract(y, w, c, t), self.RateHR(y, w, c, t), self.RateNHEJ(y, w, c, t)], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([0], dtype=jnp.float32)

		return reactionVelocities

	def RateTimeract(self, y, w, c, t):
		return c[3] * (w[1]/1.0) * (w[2]/1.0) / (c[10] + (w[2]/1.0)) - c[4] * (y[0]/1.0) / (c[10] + (y[0]/1.0))

	def RateCellCact(self, y, w, c, t):
		return (c[7] + (y[1]/1.0)) * (w[3]/1.0) / (c[11] + (w[3]/1.0)) - c[6] * (y[0]/1.0) * (y[1]/1.0) / (c[11] + (y[1]/1.0)) - c[9] * (y[1]/1.0) * (y[2]/1.0) / (c[11] + (y[1]/1.0))

	def RateEffectoract(self, y, w, c, t):
		return c[8] * (w[1]/1.0) * (w[0]/1.0) / (c[11] + (w[0]/1.0)) - c[5] * (y[1]/1.0) * (y[2]/1.0) / (c[11] + (y[2]/1.0))

	def RateHR(self, y, w, c, t):
		return -(y[3]/1.0) * 0.2

	def RateNHEJ(self, y, w, c, t):
		return -(y[4]/1.0) * 0.5

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(1.0 * ((c[1]/1.0) - (y[2]/1.0)))

		w = w.at[1].set(1.0 * ((y[3]/1.0) + (y[4]/1.0)))

		w = w.at[2].set(1.0 * ((c[2]/1.0) - (y[0]/1.0)))

		w = w.at[3].set(1.0 * ((c[0]/1.0) - (y[1]/1.0)))

		return w

class ModelRollout(eqx.Module):
	y_indexes: dict = eqx.static_field()
	w_indexes: dict = eqx.static_field()
	c_indexes: dict = eqx.static_field()
	ratefunc: RateofSpeciesChange
	assignmentfunc: AssignmentRule
	ode_term: diffrax.ODETerm
	solver: diffrax.AbstractERK = eqx.static_field()
	iterative_solve: bool = eqx.static_field()

	def __init__(self, y_indexes={'Timeract': 0, 'CellCact': 1, 'Effectoract': 2, 'HR': 3, 'NHEJ': 4}, w_indexes={'Effectorina': 0, 'Damage': 1, 'Timerinact': 2, 'CellCina': 3}, c_indexes={'CellCycletot': 0, 'Effectortot': 1, 'Timertot': 2, 'Kd2t': 3, 'Kti2t': 4, 'Kcc2ch': 5, 'Kt2cc': 6, 'Kcc2a': 7, 'Kd2ch': 8, 'Kch2cc': 9, 'Km1': 10, 'Km10': 11, 'nucleus': 12}, solver=diffrax.Tsit5(), iterative_solve=True):

		self.y_indexes = y_indexes
		self.w_indexes = w_indexes
		self.c_indexes = c_indexes

		self.ratefunc = RateofSpeciesChange()
		self.assignmentfunc = AssignmentRule()

		def ode_func(t, y, args):
			w, c = args
			# Update w using the assignment rule
			w = self.assignmentfunc(y, w, c, t)

			# Calculate the rate of change
			dy_dt = self.ratefunc(y, t, w, c)

			return dy_dt

		self.ode_term = diffrax.ODETerm(ode_func)
		self.solver = solver
		self.iterative_solve = iterative_solve

	@eqx.filter_jit
	def step(self, y, w, c, t, deltaT=0.1):
		t_new = t + deltaT
		state = self.solver.init(self.ode_term, t, t_new, y, (w, c))
		y_new, _, _, _, _ = self.solver.step(self.ode_term, t, t_new, y, (w, c), state, made_jump=False)
		w_new = self.assignmentfunc(y_new, w, c, t_new)
		return y_new, w_new, c, t_new

	@eqx.filter_jit
	def __call__(self, t1,y0=jnp.array([0.0, 0.999999999999999, 0.0, 3.0, 7.0]), w0=jnp.array([10.0, 10.0, 10.0, 9.000000000000002]), c=jnp.array([10.0, 10.0, 10.0, 2.0, 10.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0]), t0=0.0, deltaT=0.1, stepsize_controller=diffrax.PIDController(atol=1e-06, rtol=1e-12), max_steps=5000000):

		# Number of steps
		n_steps = int(t1 / deltaT)

		# Solve the ODE system
		if self.iterative_solve:
			def f(carry, x):
				y, w, c, t = carry
				return self.step(y, w, c, t, deltaT), (y, w, t)
			(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps)) 

		else:
			sol = diffrax.diffeqsolve(
				self.ode_term,
				self.solver,
				t0=t0,
				t1=t1,
				dt0=deltaT,
				y0=y0,
				args=(w0, c),
				saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_steps)),
				stepsize_controller=stepsize_controller,
				max_steps=max_steps
			)

			# Extract results and recompute ws
			ts = sol.ts
			ys = sol.ys
			ws = vmap(lambda t, y: self.assignmentfunc(y, w0, c, t))(ts, ys)
		ys = jnp.moveaxis(ys, 0, -1) #(n_species, n_steps)
		ws = jnp.moveaxis(ws, 0, -1) #(n_params, n_steps)
		return ys, ws, ts

