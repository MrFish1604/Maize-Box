import numpy as np
from numpy import cos, sin, array
from math import pow, sqrt

def distPlan(cls, posi, plan):
		"""
		Evalue la distance d'une bille au plan (pour l'instant il n'y en a qu'un et c'est un vrai plan)
		"""
		N = (np.dot(plan.equa, array([posi[0], posi[1], posi[2], 1])))
		D = sqrt(np.dot(plan.normal, plan.normal))
		return N/D

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
	def init(cls):
		"""
		Initialise le monde
		"""
		if World.notinitialized:
			World.plans = []
			World.notinitialized = False
	
	def addPlan(plan):
		"""
		Ajoute un plan
		"""
		World.plans.append(plan)
	def addPlans(plans):
		"""
		Ajoute des plans
		"""
		for plan in plans:
			World.plans.append(plan)

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
		World.maizes = [Maize(array([0,0,0]), array([0,0,0])) for i in range(nbr)]
	@classmethod
	def create_plan(cls, equaCart):
		World.plans.append(Plan(equaCart))

	@classmethod
	def distPlan(cls, posi, iplan):
		"""
		Evalue la distance d'une bille au plan (pour l'instant il n'y en a qu'un et c'est un vrai plan)
		"""
		N = (np.dot(cls.plans[iplan].equa, array([posi[0], posi[1], posi[2], 1])))
		D = sqrt(np.dot(cls.plans[iplan].normal, cls.plans[iplan].normal))
		return N/D
	
	@classmethod
	def process(cls):
		# Calc accels
		next_percent = 10
		for i in range(World.nbr_steps-1):
			percent = int(100*i/World.nbr_steps)
			if percent==next_percent:
				print(str(percent)+"%")
				next_percent+=10
			for maize in World.maizes:
				posi = maize.getPosi(i)			# recupère la position de la bille à l'instant i
				dplan = World.distPlan(posi, 0)		# calcule sa distance au plan
				if dplan < maize.R:				# Si elle est dans le plan
					# print(dplan)
					delta = maize.R - dplan			# calcul delta
					Kstar = 4*Maize.Estar*sqrt(maize.R*delta)/3
					accel = (Kstar*delta/maize.masse)*array([cos(World.plans[0].theta)*sin(World.plans[0].psi), cos(World.plans[0].theta)*cos(World.plans[0].psi), sin(World.plans[0].theta)]) - array([0, World.gravity, 0])
				else:
					accel = array([0, -World.gravity, 0])
				velocity = maize.velocities[i] + World.step*accel
				position = maize.positions[i] + World.step*velocity
				maize.setAccel(accel, i)
				maize.setVel(velocity, i+1)
				maize.setPosi(position, i+1)
				# print(maize.getAccel(i))
				# print(maize.getVel(i))
				# print(maize.getPosi(i))
				# input()
		print("100%")

class Plan:
	"""
	Représente un plan
	!!! Il faudra changer tout ca !!!
	"""
	n_id = 1
	def __init__(self, psi, theta, point):
		self.ID = Plan.n_id
		Plan.n_id+=1
		self.psi = psi
		self.theta = theta
		self.point = point
		self.calc_equa()
	
	def set_rota(self, psi, theta):
		"""
		Modifie les angles et calcul les équations
		"""
		self.theta = theta
		self.point = point
		self.calc_equa()
	
	def calc_equa(self):
		"""
		Calcul les equations du plan
		"""
		self.normal = array([-cos(self.theta)*sin(self.psi), cos(self.theta)*cos(self.psi), sin(self.theta)])
		self.d = np.dot(-self.normal, self.point)
		self.equa = array([self.normal[0], self.normal[1], self.normal[2], self.d])
		
	def show2D(self, size, style="-k"):
		x1 = self.point[0] - size*cos(self.psi)
		y1 = self.point[1] - size*sin(self.psi)
		x2 = self.point[0] + size*cos(self.psi)
		y2 = self.point[1] + size*sin(self.psi)
		plt.plot([x1, x2], [y1, y2], style)
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
		self.sum_F = np.ones((World.nbr_steps, 3))*np.array([0, -World.gravity*self.masse, 0])
		self.positions[0] = pos
		self.velocities[0] = vits
		self.R = 0.005
		self.Rstar = self.R/2
		self.vol = (4/3)*np.pi*pow(R, 3)
		self.masse = self.vol*self.ro
		Maize.Estar = Maize.young/(2-2*pow(Maize.poisson, 2))
	
	def PFD(self, i):
		"""
		Cherche les forces qui agissent sur la bille
		"""
		posi = self.positions[i]
		# Recherche les plans	
		for plan in World.plans:
			dist = distPlan(posi, plan)
			if dist<self.R:
				delta = dist-self.R
				self.sum_F[i] += 4*self.Estar*delta*sqrt(self.R*delta)*plan.normal
		# Amortissement
		# Recherche les contact avec d'autres bille
		# Frottements
		self.accels[i] = self.sum_F[i]/self.masse

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

World.init()
plan = Plan(0, 0, array([0,0,0]))
World.addPlan(plan)
World.setTime(h=0.0001, tf=3)

World.create_Maizes(1)
World.maizes[0].setInit(array([-0.3, 0.2, 0]), array([0.2,0,0]))
World.process()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

X = World.maizes[0].positions[:, 0]
Y = World.maizes[0].positions[:, 1]

fig = plt.figure()
plt.axis("equal")
plan.show2D(0.5)

circle = plt.Circle((X[0], Y[0]), radius=World.maizes[0].R)

ani_h=50
nbr_frames = int(World.tfinal*ani_h*1000)
new_h = int(ani_h*0.001/World.step)

def init_ani():
	fig.gca().add_patch(circle)

def animate(i):
	j = new_h*i
	if j<np.size(X,0):
		circle.center = (X[j], Y[j])

ani = FuncAnimation(fig, animate, init_func=init_ani, interval=ani_h, frames=nbr_frames)

input("Press enter to display...")
plt.show()