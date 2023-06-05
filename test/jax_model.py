import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([90.0, 10.0, 280.0, 10.0, 10.0, 280.0, 10.0, 10.0])
y_indexes = {'MKKK': 0, 'MKKK_P': 1, 'MKK': 2, 'MKK_P': 3, 'MKK_PP': 4, 'MAPK': 5, 'MAPK_P': 6, 'MAPK_PP': 7}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 2.5, 9.0, 1.0, 10.0, 0.25, 8.0, 0.025, 15.0, 0.025, 15.0, 0.75, 15.0, 0.75, 15.0, 0.025, 15.0, 0.025, 15.0, 0.5, 15.0, 0.5, 15.0]) 
c_indexes = {'uVol': 0, 'J0_V1': 1, 'J0_Ki': 2, 'J0_n': 3, 'J0_K1': 4, 'J1_V2': 5, 'J1_KK2': 6, 'J2_k3': 7, 'J2_KK3': 8, 'J3_k4': 9, 'J3_KK4': 10, 'J4_V5': 11, 'J4_KK5': 12, 'J5_V6': 13, 'J5_KK6': 14, 'J6_k7': 15, 'J6_KK7': 16, 'J7_k8': 17, 'J7_KK8': 18, 'J8_V9': 19, 'J8_KK9': 20, 'J9_V10': 21, 'J9_KK10': 22}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.J0(y, w, c, t), self.J1(y, w, c, t), self.J2(y, w, c, t), self.J3(y, w, c, t), self.J4(y, w, c, t), self.J5(y, w, c, t), self.J6(y, w, c, t), self.J7(y, w, c, t), self.J8(y, w, c, t), self.J9(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def J0(self, y, w, c, t):
		return c[0] * c[1] * (y[0]/1.0) / ((1 + ((y[7]/1.0) / c[2])**c[3]) * (c[4] + (y[0]/1.0)))


	def J1(self, y, w, c, t):
		return c[0] * c[5] * (y[1]/1.0) / (c[6] + (y[1]/1.0))


	def J2(self, y, w, c, t):
		return c[0] * c[7] * (y[1]/1.0) * (y[2]/1.0) / (c[8] + (y[2]/1.0))


	def J3(self, y, w, c, t):
		return c[0] * c[9] * (y[1]/1.0) * (y[3]/1.0) / (c[10] + (y[3]/1.0))


	def J4(self, y, w, c, t):
		return c[0] * c[11] * (y[4]/1.0) / (c[12] + (y[4]/1.0))


	def J5(self, y, w, c, t):
		return c[0] * c[13] * (y[3]/1.0) / (c[14] + (y[3]/1.0))


	def J6(self, y, w, c, t):
		return c[0] * c[15] * (y[4]/1.0) * (y[5]/1.0) / (c[16] + (y[5]/1.0))


	def J7(self, y, w, c, t):
		return c[0] * c[17] * (y[4]/1.0) * (y[6]/1.0) / (c[18] + (y[6]/1.0))


	def J8(self, y, w, c, t):
		return c[0] * c[19] * (y[7]/1.0) / (c[20] + (y[7]/1.0))


	def J9(self, y, w, c, t):
		return c[0] * c[21] * (y[6]/1.0) / (c[22] + (y[6]/1.0))

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

	def __init__(self, y_indexes={'MKKK': 0, 'MKKK_P': 1, 'MKK': 2, 'MKK_P': 3, 'MKK_PP': 4, 'MAPK': 5, 'MAPK_P': 6, 'MAPK_PP': 7}, w_indexes={}, c_indexes={'uVol': 0, 'J0_V1': 1, 'J0_Ki': 2, 'J0_n': 3, 'J0_K1': 4, 'J1_V2': 5, 'J1_KK2': 6, 'J2_k3': 7, 'J2_KK3': 8, 'J3_k4': 9, 'J3_KK4': 10, 'J4_V5': 11, 'J4_KK5': 12, 'J5_V6': 13, 'J5_KK6': 14, 'J6_k7': 15, 'J6_KK7': 16, 'J7_k8': 17, 'J7_KK8': 18, 'J8_V9': 19, 'J8_KK9': 20, 'J9_V10': 21, 'J9_KK10': 22}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([90.0, 10.0, 280.0, 10.0, 10.0, 280.0, 10.0, 10.0]), w0=jnp.array([]), c=jnp.array([1.0, 2.5, 9.0, 1.0, 10.0, 0.25, 8.0, 0.025, 15.0, 0.025, 15.0, 0.75, 15.0, 0.75, 15.0, 0.025, 15.0, 0.025, 15.0, 0.5, 15.0, 0.5, 15.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

