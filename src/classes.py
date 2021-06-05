import numpy as np
from math import pow

class World:
	"""
	This class should represent the world where balls will evolve.
	I don't know if it will be usefull.
	"""
	nbr_Maizes = 0
	nbr_steps = 0	# Ne compte pas la 1er etape
	step = 0
	tfinal = 0
	notinitialized = True

	@classmethod
	def init(cls, plan=None):
		"""
		Initialise le monde
		"""
		if World.notinitialized:
			if type(plan) == Plan:
				World.plans = [plan]
			else:
				World.plans = []
			World.notinitialized = False

	@classmethod
	def setTime(cls, h=None, nh=None, tf=None):
		"""
		Set times constants
		"""
		a = tf!=None
		b = nh!=None
		c = h!=None
		if World.tfinal==0 and ((a and (b or c)) or (not(a) and b and c)):
			if not a:	#if tf==None
				World.nbr_steps = nh
				World.step = h
				World.tfinal = int(h*nh)
			elif not b:	#if nh==None
				World.tfinal = tf
				World.step = h
				World.nbr_steps = int(tf/h)
			else:	#if h==None
				World.tfinal = tf
				World.nbr_steps = nh
				World.step = int(th/nh)

	@classmethod	
	def create_Maizes(cls, nbr):
		World.nbr_Maizes = nbr
		World.maizes = [Maize(np.array([0,0,0]), np.array([0,0,0])) for i in range(nbr)]
	@classmethod
	def create_plan(cls, equaCart):
		World.plans.append(Plan(equaCart))

class Plan:
	"""
	Représente un plan
	"""
	def __init__(self, equaCart):
		self.equa_cart = equaCart
		self.normal = np.array([equaCart[0], equaCart[1], equaCart[2]])
	def __str__(self):
		return str(self.equa_cart[0]) + "x + " + str(self.equa_cart[1]) + "y + " + str(self.equa_cart[2]) + "z + " + str(self.equa_cart[3]) + " = 0"
	def __repr__(self):
		return "Plan([" + str(self.equa_cart[0]) + ", " + str(self.equa_cart[1]) + ", " + str(self.equa_cart[2]) + ", " + str(self.equa_cart[3]) + "])"
	
	def getNormal(self):
		"""
		Retourne un vecteur normal au plan
		"""
		return self.normal

class Maize:
	"""
	This class represent a ball of maize
	"""
	ro = 1180	# kg/m3
	coef_resti_contact = 0.5
	young = 340
	fm = 0.86
	fa = 0.54
	poisson = 0.3

	def __init__(self, pos, vits, R=0.005):
		"""
		initialize a maize
		pos		array	position of the ball at t=0
		vits	array	velocity of the ball at t=0
		"""
		self.positions = np.zeros((World.nbr_steps, 3))
		self.velocities = np.zeros((World.nbr_steps, 3))
		self.accels = np.zeros((World.nbr_steps, 3))
		self.positions[0] = pos
		self.velocities[0] = vits
		self.R = 0.005
		self.vol = (4/3)*np.pi*pow(R, 3)
		self.masse = self.vol*self.ro
		Maize.Estar = Maize.young/(2-2*pow(Maize.poisson, 2))
	
	def setInit(self, pos, vits):
		self.positions[0] = pos
		self.velocities[0] = vits

World.init(Plan(np.array([0,1,0,0])))
World.setTime(1, tf=5)
World.create_Maizes(1)

maize = World.maizes[0]
maize.setInit(np.array([0, 1, 0]), np.array([0,0,0]))

