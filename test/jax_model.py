import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp
from diffrax import ODETerm, Tsit5, Dopri5, Dopri8, Euler, Midpoint, Heun, Bosh3, Ralston
from typing import Any

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

class ModelStep(eqx.Module):
	y_indexes: dict = eqx.static_field()
	w_indexes: dict = eqx.static_field()
	c_indexes: dict = eqx.static_field()
	ratefunc: RateofSpeciesChange
	atol: float = eqx.static_field()
	rtol: float = eqx.static_field()
	mxstep: int = eqx.static_field()
	assignmentfunc: AssignmentRule
	solver_type: str = eqx.static_field()
	solver: Any = eqx.static_field()

	def __init__(self, y_indexes={'AprE': 0, 'DegUP': 1, 'Dim': 2, 'DegU': 3, 'mDegU': 4, 'mAprE': 5}, w_indexes={'kphos': 0, 'kdephos': 1, 'DegU_Total': 2}, c_indexes={'Imax': 0, 'Io': 1, 'Irmax': 2, 'Iro': 3, 'K': 4, 'Kdim': 5, 'Kr': 6, 'Kr1': 7, 'R': 8, 'V': 9, 'ka': 10, 'kd': 11, 'kdeg': 12, 'kdegA': 13, 'kdegm': 14, 'ksyn': 15, 'ksyn1': 16, 'p': 17, 'q': 18, 'ratio': 19, 'univ': 20}, atol=1e-06, rtol=1e-12, mxstep=5000000, solver_type='diffrax', diffrax_solver='Ralston'):

		self.y_indexes = y_indexes
		self.w_indexes = w_indexes
		self.c_indexes = c_indexes
		self.ratefunc = RateofSpeciesChange()
		self.rtol = rtol
		self.atol = atol
		self.mxstep = mxstep
		self.assignmentfunc = AssignmentRule()
		self.solver_type = solver_type
		if solver_type == 'odeint':
			self.solver = odeint
		elif solver_type == 'diffrax':
			from diffrax import ODETerm, Tsit5, Dopri5, Dopri8, Euler, Midpoint, Heun, Bosh3, Ralston
			valid_solvers = {'Tsit5', 'Dopri5', 'Dopri8', 'Euler', 'Midpoint', 'Heun', 'Bosh3', 'Ralston'}
			if diffrax_solver not in valid_solvers:
				raise ValueError(f'Unknown diffrax solver: {diffrax_solver}')
			self.solver = eval(diffrax_solver)()
		else:
			raise ValueError(f'Unknown solver type: {solver_type}')

	@jit
	def __call__(self, y, w, c, t, deltaT):
		if self.solver_type == 'odeint':
			y_new = self.solver(self.ratefunc, y, jnp.array([t, t + deltaT]), w, c, atol=self.atol, rtol=self.rtol, mxstep=self.mxstep)[-1]
		else:  # diffrax
			term = ODETerm(lambda t, y, args: self.ratefunc(y, t, *args))
			tprev, tnext = t, t + deltaT
			state = self.solver.init(term, tprev, tnext, y, (w, c))
			y_new, _, _, _, _ = self.solver.step(term, tprev, tnext, y, (w, c), state, made_jump=False)
		t_new = t + deltaT
		w_new = self.assignmentfunc(y_new, w, c, t_new)
		return y_new, w_new, c, t_new

class ModelRollout(eqx.Module):
	deltaT: float = eqx.static_field()
	modelstepfunc: ModelStep

	def __init__(self, deltaT=0.1, atol=1e-06, rtol=1e-12, mxstep=5000000, solver_type='diffrax', diffrax_solver='Ralston'):

		self.deltaT = deltaT
		self.modelstepfunc = ModelStep(atol=atol, rtol=rtol, mxstep=mxstep, solver_type=solver_type, diffrax_solver=diffrax_solver)

	@partial(jit, static_argnames=("n_steps",))
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]), w0=jnp.array([0.00400000005, 0.149999998125, 10.0]), c=jnp.array([0.048, 0.004, 0.4, 0.02, 7.0, 12.0, 7.0, 7.0, 7.0, 1.0, 0.025, 0.1, 0.0004, 0.0001, 0.01, 0.04, 0.04, 0.15, 0.004, 0.026666667, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

