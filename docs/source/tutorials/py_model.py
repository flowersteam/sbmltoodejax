import sbmltoodepy.modelclasses
from scipy.integrate import odeint
import numpy as np
import operator
import math

class SBMLmodel(sbmltoodepy.modelclasses.Model):

	def __init__(self):

		self.p = {} #Dictionary of model parameters

		self.c = {} #Dictionary of compartments
		self.c['uVol'] = sbmltoodepy.modelclasses.Compartment(1.0, 3, True, metadata = sbmltoodepy.modelclasses.SBMLMetadata(""))

		self.s = {} #Dictionary of chemical species
		self.s['MKKK'] = sbmltoodepy.modelclasses.Species(90.0, 'Concentration', self.c['uVol'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("Mos"))
		self.s['MKKK_P'] = sbmltoodepy.modelclasses.Species(10.0, 'Concentration', self.c['uVol'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("Mos-P"))
		self.s['MKK'] = sbmltoodepy.modelclasses.Species(280.0, 'Concentration', self.c['uVol'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("Mek1"))
		self.s['MKK_P'] = sbmltoodepy.modelclasses.Species(10.0, 'Concentration', self.c['uVol'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("Mek1-P"))
		self.s['MKK_PP'] = sbmltoodepy.modelclasses.Species(10.0, 'Concentration', self.c['uVol'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("Mek1-PP"))
		self.s['MAPK'] = sbmltoodepy.modelclasses.Species(280.0, 'Concentration', self.c['uVol'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("Erk2"))
		self.s['MAPK_P'] = sbmltoodepy.modelclasses.Species(10.0, 'Concentration', self.c['uVol'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("Erk2-P"))
		self.s['MAPK_PP'] = sbmltoodepy.modelclasses.Species(10.0, 'Concentration', self.c['uVol'], False, constant = False, metadata = sbmltoodepy.modelclasses.SBMLMetadata("Erk2-PP"))

		self.r = {} #Dictionary of reactions
		self.r['J0'] = J0(self)
		self.r['J1'] = J1(self)
		self.r['J2'] = J2(self)
		self.r['J3'] = J3(self)
		self.r['J4'] = J4(self)
		self.r['J5'] = J5(self)
		self.r['J6'] = J6(self)
		self.r['J7'] = J7(self)
		self.r['J8'] = J8(self)
		self.r['J9'] = J9(self)

		self.f = {} #Dictionary of function definitions
		self.time = 0

		self.AssignmentRules()



	def AssignmentRules(self):

		return

	def _SolveReactions(self, y, t):

		self.time = t
		self.s['MKKK'].amount, self.s['MKKK_P'].amount, self.s['MKK'].amount, self.s['MKK_P'].amount, self.s['MKK_PP'].amount, self.s['MAPK'].amount, self.s['MAPK_P'].amount, self.s['MAPK_PP'].amount = y
		self.AssignmentRules()

		rateRuleVector = np.array([ 0, 0, 0, 0, 0, 0, 0, 0], dtype = np.float64)

		stoichiometricMatrix = np.array([[-1,1,0,0,0,0,0,0,0,0.],[ 1,-1,0,0,0,0,0,0,0,0.],[ 0,0,-1,0,0,1,0,0,0,0.],[ 0,0,1,-1,1,-1,0,0,0,0.],[ 0,0,0,1,-1,0,0,0,0,0.],[ 0,0,0,0,0,0,-1,0,0,1.],[ 0,0,0,0,0,0,1,-1,1,-1.],[ 0,0,0,0,0,0,0,1,-1,0.]], dtype = np.float64)

		reactionVelocities = np.array([self.r['J0'](), self.r['J1'](), self.r['J2'](), self.r['J3'](), self.r['J4'](), self.r['J5'](), self.r['J6'](), self.r['J7'](), self.r['J8'](), self.r['J9']()], dtype = np.float64)

		rateOfSpeciesChange = stoichiometricMatrix @ reactionVelocities + rateRuleVector

		return rateOfSpeciesChange

	def RunSimulation(self, deltaT, absoluteTolerance = 1e-12, relativeTolerance = 1e-6):

		finalTime = self.time + deltaT
		y0 = np.array([self.s['MKKK'].amount, self.s['MKKK_P'].amount, self.s['MKK'].amount, self.s['MKK_P'].amount, self.s['MKK_PP'].amount, self.s['MAPK'].amount, self.s['MAPK_P'].amount, self.s['MAPK_PP'].amount], dtype = np.float64)
		self.s['MKKK'].amount, self.s['MKKK_P'].amount, self.s['MKK'].amount, self.s['MKK_P'].amount, self.s['MKK_PP'].amount, self.s['MAPK'].amount, self.s['MAPK_P'].amount, self.s['MAPK_PP'].amount = odeint(self._SolveReactions, y0, [self.time, finalTime], atol = absoluteTolerance, rtol = relativeTolerance, mxstep=5000000)[-1]
		self.time = finalTime
		self.AssignmentRules()

class J0:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("MAPKKK activation")
		self.p['V1'] = sbmltoodepy.modelclasses.Parameter(2.5, 'V1')
		self.p['Ki'] = sbmltoodepy.modelclasses.Parameter(9.0, 'Ki')
		self.p['n'] = sbmltoodepy.modelclasses.Parameter(1.0, 'n')
		self.p['K1'] = sbmltoodepy.modelclasses.Parameter(10.0, 'K1')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['V1'].value * self.parent.s['MKKK'].concentration / ((1 + (self.parent.s['MAPK_PP'].concentration / self.p['Ki'].value)**self.p['n'].value) * (self.p['K1'].value + self.parent.s['MKKK'].concentration))

class J1:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("MAPKKK inactivation")
		self.p['V2'] = sbmltoodepy.modelclasses.Parameter(0.25, 'V2')
		self.p['KK2'] = sbmltoodepy.modelclasses.Parameter(8.0, 'KK2')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['V2'].value * self.parent.s['MKKK_P'].concentration / (self.p['KK2'].value + self.parent.s['MKKK_P'].concentration)

class J2:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("phosphorylation of MAPKK")
		self.p['k3'] = sbmltoodepy.modelclasses.Parameter(0.025, 'k3')
		self.p['KK3'] = sbmltoodepy.modelclasses.Parameter(15.0, 'KK3')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['k3'].value * self.parent.s['MKKK_P'].concentration * self.parent.s['MKK'].concentration / (self.p['KK3'].value + self.parent.s['MKK'].concentration)

class J3:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("phosphorylation of MAPKK-P")
		self.p['k4'] = sbmltoodepy.modelclasses.Parameter(0.025, 'k4')
		self.p['KK4'] = sbmltoodepy.modelclasses.Parameter(15.0, 'KK4')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['k4'].value * self.parent.s['MKKK_P'].concentration * self.parent.s['MKK_P'].concentration / (self.p['KK4'].value + self.parent.s['MKK_P'].concentration)

class J4:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("dephosphorylation of MAPKK-PP")
		self.p['V5'] = sbmltoodepy.modelclasses.Parameter(0.75, 'V5')
		self.p['KK5'] = sbmltoodepy.modelclasses.Parameter(15.0, 'KK5')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['V5'].value * self.parent.s['MKK_PP'].concentration / (self.p['KK5'].value + self.parent.s['MKK_PP'].concentration)

class J5:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("dephosphorylation of MAPKK-P")
		self.p['V6'] = sbmltoodepy.modelclasses.Parameter(0.75, 'V6')
		self.p['KK6'] = sbmltoodepy.modelclasses.Parameter(15.0, 'KK6')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['V6'].value * self.parent.s['MKK_P'].concentration / (self.p['KK6'].value + self.parent.s['MKK_P'].concentration)

class J6:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("phosphorylation of MAPK")
		self.p['k7'] = sbmltoodepy.modelclasses.Parameter(0.025, 'k7')
		self.p['KK7'] = sbmltoodepy.modelclasses.Parameter(15.0, 'KK7')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['k7'].value * self.parent.s['MKK_PP'].concentration * self.parent.s['MAPK'].concentration / (self.p['KK7'].value + self.parent.s['MAPK'].concentration)

class J7:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("phosphorylation of MAPK-P")
		self.p['k8'] = sbmltoodepy.modelclasses.Parameter(0.025, 'k8')
		self.p['KK8'] = sbmltoodepy.modelclasses.Parameter(15.0, 'KK8')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['k8'].value * self.parent.s['MKK_PP'].concentration * self.parent.s['MAPK_P'].concentration / (self.p['KK8'].value + self.parent.s['MAPK_P'].concentration)

class J8:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("dephosphorylation of MAPK-PP")
		self.p['V9'] = sbmltoodepy.modelclasses.Parameter(0.5, 'V9')
		self.p['KK9'] = sbmltoodepy.modelclasses.Parameter(15.0, 'KK9')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['V9'].value * self.parent.s['MAPK_PP'].concentration / (self.p['KK9'].value + self.parent.s['MAPK_PP'].concentration)

class J9:

	def __init__(self, parent, metadata = None):

		self.parent = parent
		self.p = {}
		if metadata:
			self.metadata = metadata
		else:
			self.metadata = sbmltoodepy.modelclasses.SBMLMetadata("dephosphorylation of MAPK-P")
		self.p['V10'] = sbmltoodepy.modelclasses.Parameter(0.5, 'V10')
		self.p['KK10'] = sbmltoodepy.modelclasses.Parameter(15.0, 'KK10')

	def __call__(self):
		return self.parent.c['uVol'].size * self.p['V10'].value * self.parent.s['MAPK_P'].concentration / (self.p['KK10'].value + self.parent.s['MAPK_P'].concentration)

