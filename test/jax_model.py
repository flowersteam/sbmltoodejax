import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp
from diffrax import ODETerm, Tsit5, Dopri5, Dopri8, Euler, Midpoint, Heun, Bosh3, Ralston
from typing import Any

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([10.0, 0.0, 6.0, 0.0, 200.0, 0.0, 0.0, 0.9, 0.0, 0.0, 30.0, 0.0])
y_indexes = {'Pfr': 0, 'Pr': 1, 'Xi': 2, 'Xa': 3, 'prepreS': 4, 'preS': 5, 'S': 6, 'Ya': 7, 'Gluc': 8, 'Yi': 9, 'V': 10, 'Pi': 11}

w0 = jnp.array([])
w_indexes = {}

c = jnp.array([1.0, 0.1, 0.0, 0.1, 0.8, 0.2, 1.0, 0.1, 30.0, 50.0, 1.0, 1.0, 0.1]) 
c_indexes = {'compartment': 0, 'Photoreceptor_activation_IfrSfrPfr': 1, 'Photoreceptor_inactivation_IrSrPr': 2, 'Transducer_activation_kia': 3, 'Transducer_inactivation_kai': 4, 'preS_formation_kx': 5, 'S_generation_ky': 6, 'Glucose_sensor_inactivation_kG': 7, 'S_formation_alpha1': 8, 'V_formation_alpha2': 9, 'S_degradation_kd_s': 10, 'V_degradation_kd_v': 11, 'Photoreceptor_decay_kd': 12}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.Photoreceptor_activation(y, w, c, t), self.Photoreceptor_inactivation(y, w, c, t), self.Transducer_activation(y, w, c, t), self.Transducer_inactivation(y, w, c, t), self.preS_formation(y, w, c, t), self.S_generation(y, w, c, t), self.Glucose_sensor_inactivation(y, w, c, t), self.S_formation(y, w, c, t), self.V_formation(y, w, c, t), self.S_degradation(y, w, c, t), self.V_degradation(y, w, c, t), self.Photoreceptor_decay(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def Photoreceptor_activation(self, y, w, c, t):
		return c[0] * (y[0]/1.0) * c[1]


	def Photoreceptor_inactivation(self, y, w, c, t):
		return c[2] * (y[1]/1.0) * c[0]


	def Transducer_activation(self, y, w, c, t):
		return (y[2]/1.0) * c[3] * (y[1]/1.0) * c[0]


	def Transducer_inactivation(self, y, w, c, t):
		return c[4] * (y[3]/1.0) * c[0]


	def preS_formation(self, y, w, c, t):
		return (y[4]/1.0) * c[5] * (y[3]/1.0) * c[0]


	def S_generation(self, y, w, c, t):
		return (y[5]/1.0) * c[6] * (y[7]/1.0) * c[0]


	def Glucose_sensor_inactivation(self, y, w, c, t):
		return c[7] * (y[7]/1.0) * (y[8]/1.0) * c[0]


	def S_formation(self, y, w, c, t):
		return c[0] * (c[8] / (1 + (y[10]/1.0)**3))


	def V_formation(self, y, w, c, t):
		return c[0] * (c[9] / (1 + (y[6]/1.0)**3))


	def S_degradation(self, y, w, c, t):
		return c[10] * (y[6]/1.0) * c[0]


	def V_degradation(self, y, w, c, t):
		return c[0] * (y[10]/1.0) * c[11]


	def Photoreceptor_decay(self, y, w, c, t):
		return c[0] * c[12] * (y[1]/1.0)

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
	solver_type: str = eqx.static_field()
	solver: Any = eqx.static_field()

	def __init__(self, y_indexes={'Pfr': 0, 'Pr': 1, 'Xi': 2, 'Xa': 3, 'prepreS': 4, 'preS': 5, 'S': 6, 'Ya': 7, 'Gluc': 8, 'Yi': 9, 'V': 10, 'Pi': 11}, w_indexes={}, c_indexes={'compartment': 0, 'Photoreceptor_activation_IfrSfrPfr': 1, 'Photoreceptor_inactivation_IrSrPr': 2, 'Transducer_activation_kia': 3, 'Transducer_inactivation_kai': 4, 'preS_formation_kx': 5, 'S_generation_ky': 6, 'Glucose_sensor_inactivation_kG': 7, 'S_formation_alpha1': 8, 'V_formation_alpha2': 9, 'S_degradation_kd_s': 10, 'V_degradation_kd_v': 11, 'Photoreceptor_decay_kd': 12}, atol=1e-06, rtol=1e-12, mxstep=5000000, solver_type='diffrax', diffrax_solver='Dopri8'):

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
			self.solver = Dopri8()
		else:
			raise ValueError(f'Unknown solver type: {solver_type}')

	@jit
	def __call__(self, y, w, c, t, deltaT):
		if self.solver_type == 'odeint':
			y_new = odeint(self.ratefunc, y, jnp.array([t, t + deltaT]), w, c, atol=self.atol, rtol=self.rtol, mxstep=self.mxstep)[-1]
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

	def __init__(self, deltaT=0.1, atol=1e-06, rtol=1e-12, mxstep=5000000, solver_type='diffrax', diffrax_solver='Dopri8'):

		self.deltaT = deltaT
		self.modelstepfunc = ModelStep(atol=atol, rtol=rtol, mxstep=mxstep, solver_type=solver_type, diffrax_solver=diffrax_solver)

	@partial(jit, static_argnames=("n_steps",))
	def __call__(self, n_steps, y0=jnp.array([10.0, 0.0, 6.0, 0.0, 200.0, 0.0, 0.0, 0.9, 0.0, 0.0, 30.0, 0.0]), w0=jnp.array([]), c=jnp.array([1.0, 0.1, 0.0, 0.1, 0.8, 0.2, 1.0, 0.1, 30.0, 50.0, 1.0, 1.0, 0.1]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

