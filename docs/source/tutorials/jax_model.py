import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
y_indexes = {'R': 0, 'Rin': 1, 'x1': 2, 'x1p': 3, 'x2': 4, 'x2p': 5, 'x3': 6, 'x3p': 7}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 1.0, 0.1, 0.01, 0.1, 1.0, 0.1, 0.3, 1.0, 1.0, 0.1, 0.3, 1.0, 1.0, 0.1, 0.3, 1.0, 0.0, 1.0]) 
c_indexes = {'compartment': 0, 'v1_Vm1': 1, 'v1_Km1': 2, 'v2_Vm2': 3, 'v2_Km2': 4, 'v3_k3': 5, 'v3_Km3': 6, 'v4_Vm4': 7, 'v4_Km4': 8, 'v5_k5': 9, 'v5_Km5': 10, 'v6_Vm6': 11, 'v6_Km6': 12, 'v7_k7': 13, 'v7_Km7': 14, 'v8_Vm8': 15, 'v8_Km8': 16, 'v8_Inh': 17, 'v8_Ki8': 18}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.v1(y, w, c, t), self.v2(y, w, c, t), self.v3(y, w, c, t), self.v4(y, w, c, t), self.v5(y, w, c, t), self.v6(y, w, c, t), self.v7(y, w, c, t), self.v8(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def v1(self, y, w, c, t):
		return c[1] * y[0] / (c[2] + y[0])


	def v2(self, y, w, c, t):
		return c[3] * y[1] / (c[4] + y[1])


	def v3(self, y, w, c, t):
		return c[5] * y[0] * y[2] / (c[6] + y[2])


	def v4(self, y, w, c, t):
		return c[7] * y[3] / (c[8] + y[3])


	def v5(self, y, w, c, t):
		return c[9] * y[3] * y[4] / (c[10] + y[4])


	def v6(self, y, w, c, t):
		return c[11] * y[5] / (c[12] + y[5])


	def v7(self, y, w, c, t):
		return c[13] * y[5] * y[6] / (c[14] + y[6])


	def v8(self, y, w, c, t):
		return c[15] * y[7] / c[16] / (1 + y[7] / c[16] + c[17] / c[18])

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
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

	def __init__(self, y_indexes={'R': 0, 'Rin': 1, 'x1': 2, 'x1p': 3, 'x2': 4, 'x2p': 5, 'x3': 6, 'x3p': 7}, w_indexes={}, c_indexes={'compartment': 0, 'v1_Vm1': 1, 'v1_Km1': 2, 'v2_Vm2': 3, 'v2_Km2': 4, 'v3_k3': 5, 'v3_Km3': 6, 'v4_Vm4': 7, 'v4_Km4': 8, 'v5_k5': 9, 'v5_Km5': 10, 'v6_Vm6': 11, 'v6_Km6': 12, 'v7_k7': 13, 'v7_Km7': 14, 'v8_Vm8': 15, 'v8_Km8': 16, 'v8_Inh': 17, 'v8_Ki8': 18}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.5, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]), w0=jnp.array([]), c=jnp.array([1.0, 1.0, 0.1, 0.01, 0.1, 1.0, 0.1, 0.3, 1.0, 1.0, 0.1, 0.3, 1.0, 1.0, 0.1, 0.3, 1.0, 0.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

