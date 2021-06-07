import numpy as np
from numpy import cos, sin
from math import pow, sqrt

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
	gravity = 9.81
	t0 = 0

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
				World.step = int(tf/nh)

	@classmethod	
	def create_Maizes(cls, nbr):
		World.nbr_Maizes = nbr
		World.maizes = [Maize(np.array([0,0,0]), np.array([0,0,0])) for i in range(nbr)]
	@classmethod
	def create_plan(cls, equaCart):
		World.plans.append(Plan(equaCart))

	@classmethod
	def distPlan(cls, posi, iplan):
		"""
		Evalue la distance d'une bille au plan (pour l'instant il n'y en a qu'un et c'est un vrai plan)
		"""
		N = (np.dot(cls.plans[iplan].equa_cart, np.array([posi[0], posi[1], posi[2], 1])))
		D = sqrt(np.dot(cls.plans[iplan].normal, cls.plans[iplan].normal))
		return N/D
	
	@classmethod
	def process(cls):
		# Calc accels
		for i in range(World.nbr_steps-1):
			for maize in World.maizes:
				posi = maize.getPosi(i)			# recupère la position de la bille à l'instant i
				dplan = World.distPlan(posi, 0)		# calcule sa distance au plan
				if dplan < maize.R:				# Si elle est dans le plan
					# print(dplan)
					delta = maize.R - dplan			# calcul delta
					Kstar = 4*Maize.Estar*sqrt(maize.R*delta)/3
					accel = (Kstar*delta/maize.masse)*np.array([cos(World.plans[0].theta)*sin(World.plans[0].psi), cos(World.plans[0].theta)*cos(World.plans[0].psi), sin(World.plans[0].theta)]) - np.array([0, World.gravity, 0])
				else:
					accel = np.array([0, -World.gravity, 0])
				velocity = maize.velocities[i] + World.step*accel
				position = maize.positions[i] + World.step*velocity
				maize.setAccel(accel, i)
				maize.setVel(velocity, i+1)
				maize.setPosi(position, i+1)
				# print(maize.getAccel(i))
				# print(maize.getVel(i))
				# print(maize.getPosi(i))
				# input()

class Plan:
	"""
	Représente un plan
	!!! Il faudra changer tout ca !!!
	"""
	def __init__(self, equaCart):
		self.equa_cart = equaCart
		self.normal = np.array([equaCart[0], equaCart[1], equaCart[2]])
		self.psi = 0		# Faudrat les calculer, ou inverser le truc
		self.theta = 0		# Faudrat les calculer, ou inverser le truc
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
	young = 340e6
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
	
	def getPosi(self, i):
		return self.positions[i]
	def getVel(self, i):
		return self.velocities[i]
	def getAccel(self, i):
		return self.accels[i]
	
	def setPosi(self, posi, i):
		self.positions[i] = posi
	def setVel(self, vel, i):
		self.velocities[i] = vel
	def setAccel(self, accel, i):
		"""
		Set la i-eme ligne d'accels
		"""
		self.accels[i] = accel

World.init(Plan(np.array([0,1,0,0])))
World.setTime(h=0.000001, tf=2)
World.create_Maizes(1)

maize = World.maizes[0]
maize.setInit(np.array([0, 0.2, 0]), np.array([0.1,0,0]))
World.process()

print(World.step)
print(World.nbr_steps)
print(World.tfinal)

print()
print(maize.R)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

X = maize.positions[:, 0]
Y = maize.positions[:, 1]

fig = plt.figure()

plt.plot([0, 1], [0, 0], "-k")
plt.axis("equal")

bille = plt.Circle((X[0], Y[0]), radius=maize.R)

ani_h = 50

nbr_frames = int(World.tfinal*ani_h*1000)
new_h = int(ani_h*0.001/World.step)

print()
print("nbr_frames", nbr_frames)
print("World.nbr_steps", World.nbr_steps)
print("new_h", new_h)

from time import time
World.t0 = 0

def init_a():
	plt.gca().add_patch(bille)

def animate(i):
	j = new_h*i
	if j<np.size(X,0):
		bille.center = (X[j], Y[j])
	t = time()
	print(t - World.t0)
	World.t0 = t
	return bille

hms = World.step*1000
# if World.step<0.001:
ani = FuncAnimation(fig, animate, init_func=init_a, interval=ani_h, frames=nbr_frames)
# else:
	# ani = FuncAnimation(fig, animate, init_func=init_a, interval=World.step*1000, frames=World.nbr_steps)

t0 = time()
plt.show()

# print("\n")
# print(maize.accels)