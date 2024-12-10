import diffrax
import equinox as eqx
from jax import jit, lax, vmap
import jax.numpy as jnp
from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
y_indexes = {'AprE': 0, 'DegUP': 1, 'Dim': 2, 'DegU': 3, 'mDegU': 4, 'mAprE': 5}

w0 = jnp.array([0.00400000005, 0.149999998125, 10.0])
w_indexes = {'kphos': 0, 'kdephos': 1, 'DegU_Total': 2}

c = jnp.array([0.048, 0.004, 0.4, 0.02, 7.0, 12.0, 7.0, 7.0, 7.0, 1.0, 0.025, 0.1, 0.0004, 0.0001, 0.01, 0.04, 0.04, 0.15, 0.004, 0.026666667, 1.0]) 
c_indexes = {'Imax': 0, 'Io': 1, 'Irmax': 2, 'Iro': 3, 'K': 4, 'Kdim': 5, 'Kr': 6, 'Kr1': 7, 'R': 8, 'V': 9, 'ka': 10, 'kd': 11, 'kdeg': 12, 'kdegA': 13, 'kdegm': 14, 'ksyn': 15, 'ksyn1': 16, 'p': 17, 'q': 18, 'ratio': 19, 'univ': 20}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -2.0, 2.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.AprEdeg(y, w, c, t), self.AprEsyn(y, w, c, t), self.DimerAss(y, w, c, t), self.DimerDis(y, w, c, t), self.degradation1(y, w, c, t), self.degradation2(y, w, c, t), self.degradation3(y, w, c, t), self.degradationmRNA(y, w, c, t), self.dephosphorylation(y, w, c, t), self.mRNAAprEdeg(y, w, c, t), self.mRNAAprEsyn(y, w, c, t), self.phosphorylation(y, w, c, t), self.synthesisDegU(y, w, c, t), self.synthesismRNA(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def AprEdeg(self, y, w, c, t):
		return c[12] * (y[0]/1.0)


	def AprEsyn(self, y, w, c, t):
		return c[15] * (y[5]/1.0) * c[20]


	def DimerAss(self, y, w, c, t):
		return c[10] * (y[1]/1.0)**2


	def DimerDis(self, y, w, c, t):
		return c[11] * (y[2]/1.0)


	def degradation1(self, y, w, c, t):
		return c[12] * (y[3]/1.0) * c[20]


	def degradation2(self, y, w, c, t):
		return c[12] * (y[1]/1.0) * c[20]


	def degradation3(self, y, w, c, t):
		return c[12] * (y[2]/1.0) * c[20]


	def degradationmRNA(self, y, w, c, t):
		return c[14] * (y[4]/1.0)


	def dephosphorylation(self, y, w, c, t):
		return w[1] * (y[1]/1.0)


	def mRNAAprEdeg(self, y, w, c, t):
		return c[14] * (y[5]/1.0)


	def mRNAAprEsyn(self, y, w, c, t):
		return (c[7] / (c[8] + c[7])) * (c[3] * ((y[2]/1.0) * c[20] / c[5] + 1) / (1 + (y[2]/1.0) * c[20] / c[5] + ((y[2]/1.0) * c[20])**2 / c[5]**2 + c[8] / c[6]) + c[2] * ((y[2]/1.0) * c[20])**2 / (c[5]**2 * (1 + (y[2]/1.0) * c[20] / c[5] + ((y[2]/1.0) * c[20])**2 / c[5]**2 + c[8] / c[6])))


	def phosphorylation(self, y, w, c, t):
		return w[0] * (y[3]/1.0)


	def synthesisDegU(self, y, w, c, t):
		return c[16] * (y[4]/1.0) * c[20]


	def synthesismRNA(self, y, w, c, t):
		return c[1] * c[4] / ((y[2]/1.0) * c[20] + c[4]) + c[0] * (y[2]/1.0) * c[20] / ((y[2]/1.0) * c[20] + c[4])

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set((c[19] * c[17]))

		w = w.at[1].set((c[18] / c[19]))

		w = w.at[2].set(((y[3]/1.0) + (y[1]/1.0) + 2 * (y[2]/1.0)))

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

	def __init__(self, y_indexes={'AprE': 0, 'DegUP': 1, 'Dim': 2, 'DegU': 3, 'mDegU': 4, 'mAprE': 5}, w_indexes={'kphos': 0, 'kdephos': 1, 'DegU_Total': 2}, c_indexes={'Imax': 0, 'Io': 1, 'Irmax': 2, 'Iro': 3, 'K': 4, 'Kdim': 5, 'Kr': 6, 'Kr1': 7, 'R': 8, 'V': 9, 'ka': 10, 'kd': 11, 'kdeg': 12, 'kdegA': 13, 'kdegm': 14, 'ksyn': 15, 'ksyn1': 16, 'p': 17, 'q': 18, 'ratio': 19, 'univ': 20}, solver=diffrax.Tsit5(), iterative_solve=True):

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
	def __call__(self, t1, deltaT=0.1, t0=0.0, y0=jnp.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]), w0=jnp.array([0.00400000005, 0.149999998125, 10.0]), c=jnp.array([0.048, 0.004, 0.4, 0.02, 7.0, 12.0, 7.0, 7.0, 7.0, 1.0, 0.025, 0.1, 0.0004, 0.0001, 0.01, 0.04, 0.04, 0.15, 0.004, 0.026666667, 1.0]), stepsize_controller=diffrax.PIDController(atol=1e-06, rtol=1e-12), max_steps=5000000):

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

