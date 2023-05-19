import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([0.01, 0.01, 0.01])
y_indexes = {'C': 0, 'M': 1, 'X': 2}

w0 = jnp.array([0.0588235294117647, 0.01])
w_indexes = {'V1': 0, 'V3': 1}

c = jnp.array([3.0, 1.0, 0.5, 1.0, 0.025, 0.01, 0.25, 0.02, 0.005, 1.5, 0.005, 0.005, 0.005, 0.5]) 
c_indexes = {'VM1': 0, 'VM3': 1, 'Kc': 2, 'cell': 3, 'reaction1_vi': 4, 'reaction2_kd': 5, 'reaction3_vd': 6, 'reaction3_Kd': 7, 'reaction4_K1': 8, 'reaction5_V2': 9, 'reaction5_K2': 10, 'reaction6_K3': 11, 'reaction7_K4': 12, 'reaction7_V4': 13}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.reaction1(y, w, c, t), self.reaction2(y, w, c, t), self.reaction3(y, w, c, t), self.reaction4(y, w, c, t), self.reaction5(y, w, c, t), self.reaction6(y, w, c, t), self.reaction7(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def reaction1(self, y, w, c, t):
		return c[3] * c[4]


	def reaction2(self, y, w, c, t):
		return (y[0]/1.0) * c[3] * c[5]


	def reaction3(self, y, w, c, t):
		return (y[0]/1.0) * c[3] * c[6] * (y[2]/1.0) * ((y[0]/1.0) + c[7])**-1


	def reaction4(self, y, w, c, t):
		return c[3] * (1 + -1 * (y[1]/1.0)) * w[0] * (c[8] + -1 * (y[1]/1.0) + 1)**-1


	def reaction5(self, y, w, c, t):
		return c[3] * (y[1]/1.0) * c[9] * (c[10] + (y[1]/1.0))**-1


	def reaction6(self, y, w, c, t):
		return c[3] * w[1] * (1 + -1 * (y[2]/1.0)) * (c[11] + -1 * (y[2]/1.0) + 1)**-1


	def reaction7(self, y, w, c, t):
		return c[3] * c[13] * (y[2]/1.0) * (c[12] + (y[2]/1.0))**-1

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(((y[0]/1.0) * c[0] * ((y[0]/1.0) + c[2])**-1))

		w = w.at[1].set(((y[1]/1.0) * c[1]))

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

	def __init__(self, y_indexes={'C': 0, 'M': 1, 'X': 2}, w_indexes={'V1': 0, 'V3': 1}, c_indexes={'VM1': 0, 'VM3': 1, 'Kc': 2, 'cell': 3, 'reaction1_vi': 4, 'reaction2_kd': 5, 'reaction3_vd': 6, 'reaction3_Kd': 7, 'reaction4_K1': 8, 'reaction5_V2': 9, 'reaction5_K2': 10, 'reaction6_K3': 11, 'reaction7_K4': 12, 'reaction7_V4': 13}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([0.01, 0.01, 0.01]), w0=jnp.array([0.0588235294117647, 0.01]), c=jnp.array([3.0, 1.0, 0.5, 1.0, 0.025, 0.01, 0.25, 0.02, 0.005, 1.5, 0.005, 0.005, 0.005, 0.5]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

