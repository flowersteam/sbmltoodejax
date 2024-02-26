import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
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

class ModelStep(eqx.Module):
	y_indexes: dict = eqx.static_field()
	w_indexes: dict = eqx.static_field()
	c_indexes: dict = eqx.static_field()
	ratefunc: RateofSpeciesChange
	atol: float = eqx.static_field()
	rtol: float = eqx.static_field()
	mxstep: int = eqx.static_field()
	assignmentfunc: AssignmentRule

	def __init__(self, y_indexes={'Timeract': 0, 'CellCact': 1, 'Effectoract': 2, 'HR': 3, 'NHEJ': 4}, w_indexes={'Effectorina': 0, 'Damage': 1, 'Timerinact': 2, 'CellCina': 3}, c_indexes={'CellCycletot': 0, 'Effectortot': 1, 'Timertot': 2, 'Kd2t': 3, 'Kti2t': 4, 'Kcc2ch': 5, 'Kt2cc': 6, 'Kcc2a': 7, 'Kd2ch': 8, 'Kch2cc': 9, 'Km1': 10, 'Km10': 11, 'nucleus': 12}, atol=1e-06, rtol=1e-12, mxstep=5000000):

		self.y_indexes = y_indexes
		self.w_indexes = w_indexes
		self.c_indexes = c_indexes

		self.ratefunc = RateofSpeciesChange()
		self.rtol = rtol
		self.atol = atol
		self.mxstep = mxstep
		self.assignmentfunc = AssignmentRule()

	@jit
	def __call__(self, y, w, c, t, deltaT):
		y_new = odeint(self.ratefunc, y, jnp.array([t, t + deltaT]), w, c, atol=self.atol, rtol=self.rtol, mxstep=self.mxstep)[-1]	
		t_new = t + deltaT	
		w_new = self.assignmentfunc(y_new, w, c, t_new)	
		return y_new, w_new, c, t_new	

class ModelRollout(eqx.Module):
	deltaT: float = eqx.static_field()
	modelstepfunc: ModelStep

	def __init__(self, deltaT=0.1, atol=1e-06, rtol=1e-12, mxstep=5000000):

		self.deltaT = deltaT
		self.modelstepfunc = ModelStep(atol=atol, rtol=rtol, mxstep=mxstep)

	@partial(jit, static_argnames=("n_steps",))
	def __call__(self, n_steps, y0=jnp.array([0.0, 0.999999999999999, 0.0, 3.0, 7.0]), w0=jnp.array([10.0, 10.0, 10.0, 9.000000000000002]), c=jnp.array([10.0, 10.0, 10.0, 2.0, 10.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

