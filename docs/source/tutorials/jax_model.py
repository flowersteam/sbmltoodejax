import equinox as eqx
from functools import partial
from jax import jit, lax, vmap
from jax.experimental.ode import odeint
import jax.numpy as jnp

from sbmltoodejax import jaxfuncs

t0 = 0.0

y0 = jnp.array([1.0, 9.0, 1.0, 1000.0, 200.0])
y_indexes = {'Galpha_GTP': 0, 'APLC': 1, 'IP3': 2, 'Ca_ER': 3, 'Ca_Cyt': 4}

w0 = jnp.array([1.0, 0.6923076923076923, 0.045454545454545456, 2.5599934464167773e-06, 0.009900990099009901, 6.399590426212723e-05, 0.16666666666666666, 0.09090909090909091, 0.9995783029034626, 1.0])
w_indexes = {'DG': 0, 'Raplc': 1, 'Rpkc': 2, 'Rgalpha_gtp': 3, 'Rdg': 4, 'Rip3': 5, 'Rcyt1': 6, 'Rcyt2': 7, 'Rer': 8, 'PLC': 9}

c = jnp.array([4.0, 10.0, 200.0, 4.0, 25.0, 2.0, 25.0, 1000.0, 2000.0, 3.0, 75.0, 10.0, 0.1, 3.4, 4.0, 4.5, 1.2, 0.12, 14.0, 2.0, 10500.0, 600.0, 3000.0, 260.0, 1.0, 1.0]) 
c_indexes = {'Kp': 0, 'Kd': 1, 'Kr': 2, 'n': 3, 'Kg': 4, 'm': 5, 'Ks': 6, 'Kc1': 7, 'Kc2': 8, 'w': 9, 'Ker': 10, 'Cplc_total': 11, 'k0': 12, 'k1': 13, 'k2': 14, 'k3': 15, 'k4': 16, 'k5': 17, 'k6': 18, 'k7': 19, 'k8': 20, 'k9': 21, 'k10': 22, 'k11': 23, 'Cytosol': 24, 'ER': 25}

class RateofSpeciesChange(eqx.Module):
	stoichiometricMatrix = jnp.array([[1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0010000000474974513, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.009999999776482582, -0.05000000074505806, 0.05000000074505806]], dtype=jnp.float32) 

	@jit
	def __call__(self, y, t, w, c):
		rateRuleVector = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

		reactionVelocities = self.calc_reaction_velocities(y, w, c, t)

		rateOfSpeciesChange = self.stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange


	def calc_reaction_velocities(self, y, w, c, t):
		reactionVelocities = jnp.array([self.R1(y, w, c, t), self.R2(y, w, c, t), self.R3(y, w, c, t), self.R4(y, w, c, t), self.R5(y, w, c, t), self.R6(y, w, c, t), self.R7(y, w, c, t), self.R8(y, w, c, t), self.R9(y, w, c, t), self.R10(y, w, c, t), self.R11(y, w, c, t)], dtype=jnp.float32)

		return reactionVelocities


	def R1(self, y, w, c, t):
		return c[24] * c[12]


	def R2(self, y, w, c, t):
		return c[24] * c[13] * (y[0]/1.0)


	def R3(self, y, w, c, t):
		return c[24] * c[14] * w[1] * (y[0]/1.0)


	def R4(self, y, w, c, t):
		return c[24] * c[15] * w[2] * (y[0]/1.0)


	def R5(self, y, w, c, t):
		return c[24] * c[16] * w[3] * w[4] * (w[9]/1.0)


	def R6(self, y, w, c, t):
		return c[24] * c[17] * (y[1]/1.0)


	def R7(self, y, w, c, t):
		return c[24] * c[18] * (y[1]/1.0)


	def R8(self, y, w, c, t):
		return c[24] * c[19] * (y[2]/1.0)


	def R9(self, y, w, c, t):
		return c[25] * (c[20] * w[5] * w[8] - c[21] * w[6])


	def R10(self, y, w, c, t):
		return c[24] * c[22] * w[7]


	def R11(self, y, w, c, t):
		return c[24] * c[23]

class AssignmentRule(eqx.Module):
	@jit
	def __call__(self, y, w, c, t):
		w = w.at[0].set(1.0 * ((y[2]/1.0)))

		w = w.at[1].set(((y[1]/1.0) / (c[0] + (y[1]/1.0))))

		w = w.at[2].set((((w[0]/1.0) / (c[1] + (w[0]/1.0))) * (y[4]/1.0) / (c[2] + (y[4]/1.0))))

		w = w.at[3].set(((y[0]/1.0)**c[3] / (c[4]**c[3] + (y[0]/1.0)**c[3])))

		w = w.at[4].set(((w[0]/1.0)**c[5] / (c[1]**c[5] + (w[0]/1.0)**c[5])))

		w = w.at[5].set(((y[2]/1.0)**3 / (c[6]**3 + (y[2]/1.0)**3)))

		w = w.at[6].set(((y[4]/1.0) / (c[7] + (y[4]/1.0))))

		w = w.at[7].set(((y[4]/1.0) / (c[8] + (y[4]/1.0))))

		w = w.at[8].set(((y[3]/1.0)**c[9] / (c[10]**c[9] + (y[3]/1.0)**c[9])))

		w = w.at[9].set(1.0 * (c[11] - (y[1]/1.0)))

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

	def __init__(self, y_indexes={'Galpha_GTP': 0, 'APLC': 1, 'IP3': 2, 'Ca_ER': 3, 'Ca_Cyt': 4}, w_indexes={'DG': 0, 'Raplc': 1, 'Rpkc': 2, 'Rgalpha_gtp': 3, 'Rdg': 4, 'Rip3': 5, 'Rcyt1': 6, 'Rcyt2': 7, 'Rer': 8, 'PLC': 9}, c_indexes={'Kp': 0, 'Kd': 1, 'Kr': 2, 'n': 3, 'Kg': 4, 'm': 5, 'Ks': 6, 'Kc1': 7, 'Kc2': 8, 'w': 9, 'Ker': 10, 'Cplc_total': 11, 'k0': 12, 'k1': 13, 'k2': 14, 'k3': 15, 'k4': 16, 'k5': 17, 'k6': 18, 'k7': 19, 'k8': 20, 'k9': 21, 'k10': 22, 'k11': 23, 'Cytosol': 24, 'ER': 25}, atol=1e-06, rtol=1e-12, mxstep=5000000):

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
	def __call__(self, n_steps, y0=jnp.array([1.0, 9.0, 1.0, 1000.0, 200.0]), w0=jnp.array([1.0, 0.6923076923076923, 0.045454545454545456, 2.5599934464167773e-06, 0.009900990099009901, 6.399590426212723e-05, 0.16666666666666666, 0.09090909090909091, 0.9995783029034626, 1.0]), c=jnp.array([4.0, 10.0, 200.0, 4.0, 25.0, 2.0, 25.0, 1000.0, 2000.0, 3.0, 75.0, 10.0, 0.1, 3.4, 4.0, 4.5, 1.2, 0.12, 14.0, 2.0, 10500.0, 600.0, 3000.0, 260.0, 1.0, 1.0]), t0=0.0):

		@jit
		def f(carry, x):
			y, w, c, t = carry
			return self.modelstepfunc(y, w, c, t, self.deltaT), (y, w, t)
		(y, w, c, t), (ys, ws, ts) = lax.scan(f, (y0, w0, c, t0), jnp.arange(n_steps))
		ys = jnp.moveaxis(ys, 0, -1)
		ws = jnp.moveaxis(ws, 0, -1)
		return ys, ws, ts

