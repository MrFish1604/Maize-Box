import numpy as np
from numpy import cos, sin, array, pi, log
from numpy.linalg import norm
from math import pow, sqrt
from time import time

PIs2 = pi/2
PIs4 = pi/4
PIs6 = pi/6

def distPlan(posi, equa):
		"""
		Evalue la distance d'une bille au plan
		"""
		normal = array([equa[0], equa[1], equa[2]])
		N = abs(np.dot(equa, array([posi[0], posi[1], posi[2], 1])))
		D = sqrt(np.dot(normal, normal))
		return N/D

class World:
	"""
	This class should represent the world where balls will evolve.
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
	
	@classmethod
	def addPlan(cls, plan):
		"""
		Ajoute un plan
		"""
		World.plans.append(plan)
	@classmethod
	def addPlans(cls, plans):
		"""
		Ajoute des plans
		"""
		for plan in plans:
			World.plans.append(plan)

	@classmethod
	def addBox(cls, box):
		World.box = box
	
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
		n_p = 10
		for i in range(World.nbr_steps-1):
			World.box.move2D(i)
			World.box.save(i)
			percent = int(100*i/World.nbr_steps)
			if percent==n_p:
				print(str(percent)+"%")
				n_p+=10
			for maize in World.maizes:
				maize.PFD(i)
				maize.calcVel(i+1)
				maize.calcPosi(i+1)

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
		self.calc_normal()
		self.calc_equa()
		self.carX = [0,0]
		self.carY = [0,0]
		# Pour sauvegarder l'état du plan
		self.states = np.zeros((World.nbr_steps, 4))
	
	def set_rota(self, psi, theta):
		"""
		Modifie les angles et calcul les équations
		"""
		self.psi = psi
		self.theta = theta
		self.calc_normal()
		self.calc_equa()
	
	def save(self, i):
		self.states[i] = self.equa
	
	def calc_equa(self):
		"""
		Calcul les equations du plan
		"""
		self.d = np.dot(-self.normal, self.point)
		self.equa = array([self.normal[0], self.normal[1], self.normal[2], self.d])
	def calc_normal(self):
		self.normal = array([-cos(self.theta)*sin(self.psi), cos(self.theta)*cos(self.psi), sin(self.theta)])
		self.normal = self.normal/norm(self.normal)
	
	def calc_car(self, size):
		"""
		Calcule des points appartenants au plan permettant de l'afficher
		"""
		self.carX[0] = self.point[0] - size*cos(self.psi)
		self.carY[0] = self.point[1] - size*sin(self.psi)
		self.carX[1] = self.point[0] + size*cos(self.psi)
		self.carY[1] = self.point[1] + size*sin(self.psi)

	def show2D(self, size, style="-k"):
		self.calc_car(size)
		self.repr, = plt.plot(self.carX, self.carY, style)
		return self.repr
	
	def update2D(self, i, size):
		pass

class Box:
	"""
	Représente une boite
	"""
	def __init__(self, dim, ampli=PIs4, pulse=2.5133):	#	pi/4=0.7854		2pi/2.5 = 2.5133
		self.dim = dim
		self.demiH = self.dim[1]/2
		self.demiL = self.dim[0]/2
		self.ampli = ampli
		self.pulse = pulse
		self.bot = Plan(0, 0, np.array([0,0,0]))
		self.top = Plan(pi, 0, np.array([0, dim[1], 0]))
		self.left = Plan(-PIs2, 0, np.array([-self.demiL, self.demiH, 0]))
		self.right = Plan(PIs2, 0, np.array([self.demiL, self.demiH, 0]))
		self.index = [self.bot, self.top, self.left, self.right]
		self.move2D(0)
	
	def move2D(self, i):
		# Move bottom
		self.bot.psi = self.ampli*sin(self.pulse*World.step*i)
		self.bot.calc_normal()
		self.bot.calc_equa()
		# Move left
		self.left.psi = (self.ampli)*sin(self.pulse*World.step*i) - PIs2
		self.left.calc_normal()
		self.left.point = -self.demiL*self.left.normal+ self.demiH*self.bot.normal
		self.left.calc_equa()
		# Move right
		self.right.psi = (self.ampli)*sin(self.pulse*World.step*i) + PIs2
		self.right.calc_normal()
		self.right.point = self.demiL*self.left.normal + self.demiH*self.bot.normal
		self.right.calc_equa()
		# Move top
		self.top.psi = (self.ampli)*sin(self.pulse*World.step*i)
		self.top.calc_normal()
		self.top.point = self.dim[1]*self.bot.normal
		self.top.calc_equa()
		# Save box state
	
	def evalDelta(self, maize, i):
		"""
		Retourne une liste des interpénétration de la bille avec chaque plan de la boite dans l'orde,
		[bottom, top, left, right]
		"""
		deltas = array([0]*4, dtype=float)
		for j in range(4):
			dist = distPlan(maize.positions[i], self.index[j].states[i])
			if dist<maize.R:
				deltas[j] = maize.R - dist
		return deltas
	
	def save(self, i):
		self.bot.save(i)
		self.top.save(i)
		self.left.save(i)
		self.right.save(i)
	
	def calc_car(self):
		self.bot.calc_car(self.demiL)
		self.left.calc_car(self.demiH)
		self.right.calc_car(self.demiH)
		self.top.calc_car(self.demiL)
	
	def show2D(self):
		self.r_bot = self.bot.show2D(self.demiL)
		self.r_top = self.top.show2D(self.demiL)
		self.r_left = self.left.show2D(self.demiH)
		self.r_right = self.right.show2D(self.demiH)
	
	def update2D(self, i):
		self.move2D(i)
		self.calc_car()
		self.r_bot.set_data(self.bot.carX, self.bot.carY)
		self.r_left.set_data(self.left.carX, self.left.carY)
		self.r_right.set_data(self.right.carX, self.right.carY)
		self.r_top.set_data(self.top.carX, self.top.carY)
	
	def reset(self):
		self.move(0)


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
	Estar = young/(2-2*pow(poisson, 2))
	lne = log(coef_resti_contact)
	coef_amorti_contact = lne/sqrt(pow(pi,2) + pow(lne,2))

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
		self.Rstar = self.R/2
		self.vol = (4/3)*pi*pow(R, 3)
		self.masse = self.vol*self.ro
		self.sum_F = np.ones((World.nbr_steps, 3))*np.array([0, -World.gravity*self.masse, 0])	# Soumets la bille à son poids à chaque pas
		self.delta0 = 0
	
	def PFD(self, i):
		"""
		Cherche les forces qui agissent sur la bille
		"""
		posi = self.positions[i]
		# Recherche les plans de la boite en contact avec la bille
		deltas = World.box.evalDelta(self, i)
		deltaPoints = array([0,0,0,0])
		if i==0:
			deltasPoint = deltas/World.step
		else:
			deltaPoints = (deltas - World.box.evalDelta(self, i-1))/World.step
		# if deltas.any()!=0:
		# 	print(deltas)
		for j in range(4):
			Kstar = 4*Maize.Estar*sqrt(self.R*deltas[j])/3
			# Répulsion de Hertz et amortissement
			self.sum_F[i] += (Kstar*deltas[j] - 2*Maize.coef_amorti_contact*deltaPoints[j]*sqrt(Kstar*self.masse))*World.box.index[j].normal
		# Recherche les contact avec d'autres bille
		# Frottements
		self.accels[i] = self.sum_F[i]/self.masse

	def calcVel(self, i):
		"""
		Intègre l'accélération pour obtenir la i-ème vitesse
		"""
		self.velocities[i] = self.velocities[i-1] + World.step*self.accels[i-1]
	def calcPosi(self, i):
		"""
		Intègre la vitesse pour obtenir la i-ème position
		"""
		self.positions[i] = self.positions[i-1] + World.step*self.velocities[i]

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

# World.init()
# plan = Plan(0, 0, array([0,0,0]))
# World.addPlan(plan)
World.setTime(h=0.00001, tf=5)
box = Box((1, 0.3))
World.addBox(box)
World.create_Maizes(1)
World.maizes[0].setInit(array([0, 0.15, 0]), array([0,0,0]))
# World.maizes[1].setInit(array([-0.2, 0.15, 0]), array([0,0,0]))
# World.maizes[1].setInit(array([0.2, 0.15, 0]), array([0.3,0,0]))
t0 = time()
World.process()
print(time()-t0)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

X = World.maizes[0].positions[:, 0]
Y = World.maizes[0].positions[:, 1]

fig = plt.figure()
plt.axis("equal")
box.show2D()

circle = plt.Circle((X[0], Y[0]), radius=World.maizes[0].R)

ani_h=50
nbr_frames = int(World.tfinal*ani_h*1000)
new_h = int(ani_h*0.001/World.step)

def init_ani():
	fig.gca().add_patch(circle)

def animate(i):
	j = (new_h*i)%np.size(X,0)
	if j<np.size(X,0):
		circle.center = (X[j], Y[j])
		box.update2D(j)

ani = FuncAnimation(fig, animate, init_func=init_ani, interval=ani_h, frames=nbr_frames)
plt.gca().set_xlim(-0.7, 0.7)
plt.gca().set_ylim(-0.6, 0.8)
input("Press enter to display...")
plt.show()