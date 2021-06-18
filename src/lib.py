import numpy as np
from numpy import cos, sin, array, pi, log, sqrt
from numpy.linalg import norm
from math import pow
from time import time
import os
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d

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

def distPoints(point1, point2):
	"""
	Retourne la distance entre deux points de l'espace
	"""
	return sqrt(((point1 - point2)**2).sum())

def strListToFloat(L):
	"""
	Useless for now
	"""
	return [float(l) for l in L]

def makeUnit(vect):
	"""
	Retourne un vect de même sens et direction que vect mais de norme 1
	"""
	return vect/norm(vect)

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
	def show2D_maizes(cls, fig):
		cls.maizes_repr = [plt.Circle((World.maizes[i].positions[0,0], World.maizes[i].positions[0,1]), radius=World.maizes[i].R) for i in range(cls.nbr_Maizes)]
		ca = fig.gca()	# Get Current Axe
		for i in range(cls.nbr_Maizes):
			# cls.maizes_repr[i] = plt.Circle((World.maizes[i].positions[0,0], World.maizes[i].positions[0,1]), radius=World.maizes[i].R)
			ca.add_patch(cls.maizes_repr[i])
	@classmethod
	def show3D_maizes(cls, ca):
		cls.maizes_repr = [plt.plot(World.maizes[i].positions[0,0], World.maizes[i].positions[0,1], "or")[0] for i in range(cls.nbr_Maizes)]
	
	@classmethod
	def update2D_maizes(cls, i):
		for j in range(cls.nbr_Maizes):
			cls.maizes_repr[j].center = (cls.maizes[j].positions[i,0], cls.maizes[j].positions[i,1])
	
	@classmethod
	def update3D_maizes(cls, i):
		for j in range(cls.nbr_Maizes):
			cls.maizes_repr[j].set_data(cls.maizes[j].positions[i,0], cls.maizes[j].positions[i,1])
			cls.maizes_repr[j].set_3d_properties(cls.maizes[j].positions[i,2])
	

	@classmethod
	def init_save(cls, path, name):
		"""
		Initialise la méthode de sauvegarde de la simulation
		Doit être utilisé après la création des maizes
		"""
		World.save_path = os.path.abspath(path) + "/" + name
		if os.path.isfile(World.save_path):
			print("Specify a directory")
		elif not os.path.isdir(World.save_path):
			os.makedirs(World.save_path)
		with open(World.save_path + "/World.conf", 'w') as file:
			file.write("maizes=" + str(World.nbr_Maizes) + ";\n")
			file.write("tfinal=" + str(World.tfinal) + ";\n")
			file.write("step=" + str(World.step) + ";")
			# Rajouter la box
		World.save_inited = True
	
	@classmethod
	def load_World(cls, path):
		if os.path.isdir(path):
			World.save_path = os.path.abspath(path)
			content = ""
			with open(path+"/World.conf") as file:
				content = file.read()
			content = content.replace('\n', '')
			lines = content.split(';')
			conf = dict()
			for line in lines:
				if len(line)>0:
					buff = line.split('=')
					conf[buff[0]]=float(buff[1])
			World.setTime(h=conf["step"], tf=conf["tfinal"])
			World.create_Maizes(int(conf["maizes"]))
			return True
		else:
			return False

	
	@classmethod
	def load_save(cls, ani_h):
		if World.nbr_Maizes==0:
			return False
		else:
			for i in range(0, World.nbr_steps-1, ani_h):
				content = ""
				with open(World.save_path + "/save." + str(i) + ".tsv", "r") as file:
					content = file.read()
				lines = content.split('\n')[1:]
				for j in range(World.nbr_Maizes):
					values = lines[j].split('\t')
					maize = World.maizes[j]
					# print(values[1:4])
					# print(values[4:7])
					# print(values[7:])
					maize.accels[i] = array([float(val) for val in values[1:4]])
					maize.velocities[i] = array([float(val) for val in values[4:7]])
					maize.positions[i] = array([float(val) for val in values[7:]])
		return True
	
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
		if not World.save_inited:
			print("Save is not initialized")
			return False
		with open(World.save_path + "/progress.log", "w") as progressFile:
			progressFile.write("Started\n")
		path = World.save_path + "/save."
		n_p = 1
		for i in range(World.nbr_steps-1):
			file = open(path+str(i)+".tsv", "w")
			file.write("Temps(s)\tax\tay\taz\tvx\tvy\tvz\tx\ty\tz (UI)")
			World.box.move2D(i)
			World.box.save(i)
			percent = int(100*i/World.nbr_steps)
			if percent>=n_p:
				toprint = str(percent)+"%\t"+str(i)+"\n"
				with open(World.save_path + "/progress.log", "a") as progressFile:
					progressFile.write(toprint)
				print(toprint)
				n_p+=1
			for maize in World.maizes:
				maize.PFD(i)
				maize.calcVel(i+1)
				maize.calcPosi(i+1)
				file.write('\n')
				file.write("maize")
				for v in [maize.accels[i], maize.velocities[i], maize.positions[i]]:
						for j in range(3):
							file.write('\t')
							file.write(str(v[j]))
			file.close()

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
	
	def evalDeltaBox(self, maize, i):
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
		self.move2D(0)

def evalDeltaMaize(maize1, maize2, i):
	sumR = maize1.R + maize2.R
	dist = distPoints(maize1.positions[i], maize2.positions[i])
	if dist < sumR:
		return sumR - dist
	else:
		return 0

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
	n_id = 0

	def __init__(self, pos, vits, R=0.005):
		"""
		initialize a maize
		pos		array	position of the ball at t=0
		vits	array	velocity of the ball at t=0
		"""
		self.ID = Maize.n_id
		Maize.n_id+=1
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
		deltas = World.box.evalDeltaBox(self, i)
		deltaPoints = array([0,0,0,0])
		if i==0:
			deltasPoint = deltas/World.step
		else:
			deltaPoints = (deltas - World.box.evalDeltaBox(self, i-1))/World.step
		for j in range(4):
			Kstar = 4*Maize.Estar*sqrt(self.R*deltas[j])/3
			# Répulsion de Hertz et amortissement
			self.sum_F[i] += (Kstar*deltas[j] - 2*Maize.coef_amorti_contact*deltaPoints[j]*sqrt(Kstar*self.masse))*World.box.index[j].normal
		# Recherche les contact avec d'autres bille
		for j in range(self.ID+1, World.nbr_Maizes):
			maize = World.maizes[j]
			delta = evalDeltaMaize(self, maize, i)
			deltaPoint = 0 if i==0 else (delta - evalDeltaMaize(self, maize, i-1))/World.step
			if delta!=0:
				normal = makeUnit(maize.positions[i] - self.positions[i])
				Kstar = 4*Maize.Estar*sqrt(self.R*delta/2)/3
				F = (2*Maize.coef_amorti_contact*deltaPoint*sqrt(Kstar*self.masse/2) - Kstar*delta)*normal
				self.sum_F[i] += F
				maize.sum_F[i] -= F
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

if __name__=="__main__":
	# World.init()
	# plan = Plan(0, 0, array([0,0,0]))
	# World.addPlan(plan)
	World.setTime(h=1e-5, tf=5)
	box = Box((1, 0.3))
	World.addBox(box)
	World.create_Maizes(150)
	# World.maizes[0].setInit(array([0, 0.2, 0]), array([0,0,0]))
	d = World.maizes[0].R + 0.01	# distance entre les centres des billes
	c = 0
	l = 0.25
	for i in range(World.nbr_Maizes):
		if c==24:
			l-=d
		c = i%25
		World.maizes[i].setInit(array([c*d-0.3, l, 0]), array([0,0,0]))
	t0 = time()
	World.init_save("../simulations", "150m_5s_1e-5")
	box.move2D(0)
	World.process()
	tf = time() - t0
	with open(World.save_path + "/progress.log", "a") as pfile:
		pfile.write(str(round(tf,3)))
	# print(tf)

	# import matplotlib.pyplot as plt
	# from matplotlib.animation import FuncAnimation

	# X = World.maizes[0].positions[:, 0]
	# Y = World.maizes[0].positions[:, 1]
	# X1 = World.maizes[1].positions[:, 0]
	# Y1 = World.maizes[1].positions[:, 1]

	# fig = plt.figure()
	# plt.axis("equal")
	# box.show2D()

	# circle = plt.Circle((X[0], Y[0]), radius=World.maizes[0].R)
	# circle1 = plt.Circle((X1[0], Y1[0]), radius=World.maizes[1].R)

	# ani_h=50
	# nbr_frames = int(World.tfinal*ani_h*1000)
	# new_h = int(ani_h*0.001/World.step)

	# def init_ani():
	# 	fig.gca().add_patch(circle)
	# 	fig.gca().add_patch(circle1)

	# def animate(i):
	# 	j = (new_h*i)%np.size(X,0)
	# 	if j<np.size(X,0):
	# 		circle.center = (X[j], Y[j])
	# 		circle1.center = (X1[j], Y1[j])
	# 		box.update2D(j)

	# ani = FuncAnimation(fig, animate, init_func=init_ani, interval=ani_h, frames=nbr_frames)
	# plt.gca().set_xlim(-0.7, 0.7)
	# plt.gca().set_ylim(-0.6, 0.8)
	# input("Press enter to display...")
	# plt.show()