import numpy as np
from numpy import cos, sin
from math import pow, sqrt
import os

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
	save_path = None
	save_inited = False

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
	def process1(cls):
		# Calc accels
		with open(World.save_path + "/save.tsv", 'w') as file:
			maize = World.maizes[0]
			file.write("Temps(s)\tax\tay\taz\tvx\tvy\tvz\tx\ty\tz")
			pcs = 10
			for i in range(World.nbr_steps-1):
				file.write('\n')
				percent = int(100*i/World.nbr_steps)
				if percent==pcs:
					print(str(percent)+"%")
					pcs+=10
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
				# Save vectors
				file.write(str(i*World.step))
				for v in [accel, velocity, position]:
					for j in range(2):
						file.write('\t')
						file.write(str(v[j]))
		print("100%")
	@classmethod
	def process2(cls):
		pcs = 10
		for i in range(World.nbr_steps-1):
			with open(World.save_path + "/save." + str(i) + ".tsv", 'w') as file:
				percent = int(100*i/World.nbr_steps)
				if percent==pcs:
					print(str(percent)+"%")
					pcs+=10
				file.write("Temps(s)\tax\tay\taz\tvx\tvy\tvz\tx\ty\tz")
				for maize in World.maizes:
					file.write('\n')
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
					file.write(str(maize.id))
					file.write('\t')
					for v in [accel, velocity, position]:
						for j in range(2):
							file.write('\t')
							file.write(str(v[j]))
		print("100%")
	@classmethod
	def processNS(cls):
		pcs = 10
		for i in range(World.nbr_steps-1):
			percent = int(100*i/World.nbr_steps)
			if percent==pcs:
				print(str(percent)+"%")
				pcs+=10
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
		print("100%")
	@classmethod
	def process(cls):
		if World.save_inited:
			if World.nbr_Maizes==1:
				World.process1()
			else:
				World.process2()
		else:
			print("The world will not be saved")
			World.processNS()

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
	next_id = 0

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
		self.id = Maize.next_id
		Maize.next_id+=1
	
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
World.setTime(h=0.001, tf=5)
World.create_Maizes(2)

maize = World.maizes[0]
maize.setInit(np.array([0, 0.2, 0]), np.array([0.1,0,0]))
World.maizes[1].setInit(np.array([0, 0.3, 0]), np.array([0.2,0,0]))

World.init_save("simus", "test0")
World.process()