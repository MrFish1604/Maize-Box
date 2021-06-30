import numpy as np
from numpy import array
from numpy.linalg import norm
from math import pow, cos, sin, pi, log, sqrt
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
		normal = equa[:-1]
		N = abs(np.dot(equa, array([posi[0], posi[1], posi[2], 1])))
		D = sqrt(np.dot(normal, normal))
		return N/D

def distPoints(point1, point2):
	"""
	Retourne la distance entre deux points de l'espace
	"""
	return sqrt(((point1 - point2)**2).sum())

def makeUnit(vect):
	"""
	Retourne un vecteur de même sens et direction que vect mais de norme 1
	"""
	return vect/norm(vect)

def evalDeltaMaize(maize1, maize2, i):
	"""
	Evalue l'interpénétration de maize1 et maize2 à l'instant i
	"""
	sumR = maize1.R + maize2.R
	dist = distPoints(maize1.positions[i], maize2.positions[i])
	return sumR-dist if dist<sumR else 0

class World:
	"""
	Représente le monde dans lequel évolue les billes
	"""
	nbr_Maizes = 0	# Nombre de bille
	nbr_steps = 0	# Nombre d'étapes
	step = 0		# Pas
	tfinal = 0		# Temps de simulations
	gravity = 9.81/2	# Gravité du monde (ca peut être drole de la changer)
	save_inited = False	# Vraie si la sauvegarde à été paramétré
	
	@classmethod
	def show2D_maizes(cls, fig):
		"""
		Prépare l'affichage en 2D des billes
		"""
		cls.maizes_repr = [plt.Circle((World.maizes[i].positions[0,0], World.maizes[i].positions[0,1]), radius=World.maizes[i].R, color=(i/50, i/100, 0)) for i in range(cls.nbr_Maizes)]	# Crée des cercle pour toutes les billes
		ca = fig.gca()	# Get Current Axe
		for i in range(cls.nbr_Maizes):
			ca.add_patch(cls.maizes_repr[i])	# Ajoute les cercle au graphique
	@classmethod
	def show3D_maizes(cls, ca):
		"""
		Prépare l'affichage en 3D des billes
		Il y a encore des problèmes
		"""
		cls.maizes_repr = [plt.plot(World.maizes[i].positions[0,0], World.maizes[i].positions[0,1], "or")[0] for i in range(cls.nbr_Maizes)]
	
	@classmethod
	def update2D_maizes(cls, i):
		"""
		Met à jour l'affichage 2D des billes
		"""
		for j in range(cls.nbr_Maizes):
			cls.maizes_repr[j].center = (cls.maizes[j].positions[i,0], cls.maizes[j].positions[i,1])
	
	@classmethod
	def update3D_maizes(cls, i):
		"""
		Met à jour l'affichage 3D des billes
		Il y a encore des problèmes
		"""
		for j in range(cls.nbr_Maizes):
			cls.maizes_repr[j].set_data(cls.maizes[j].positions[i,0], cls.maizes[j].positions[i,1])
			cls.maizes_repr[j].set_3d_properties(cls.maizes[j].positions[i,2])
	

	@classmethod
	def init_save(cls, path, name):
		"""
		Initialise la sauvegarde de la simulation
		Doit être utilisé après la création des billes
		
		path	le chemin vers le répertoire au sera placé la simulation
		name	nom de la simulation
		"""
		World.save_path = os.path.abspath(path) + "/" + name
		if os.path.isfile(World.save_path):	# path doit être un dossier
			print("Specify a directory")
		elif not os.path.isdir(World.save_path):	# si le dossier n'existe pas, le créé
			os.makedirs(World.save_path)
		with open(World.save_path + "/World.conf", 'w') as file:	# Crée un fichier de configuration pour plus de praticité
			file.write("maizes=" + str(World.nbr_Maizes) + ";\n")
			file.write("tfinal=" + str(World.tfinal) + ";\n")
			file.write("step=" + str(World.step) + ";")
			# Rajouter la box
		World.save_inited = True
	
	@classmethod
	def load_World(cls, path, name):
		"""
		Charge les paramètres d'un monde depuis une sauvegarde
		
		path	le chemin vers le répertoire de la sauvegarde
		name	nom de la simulation

		Retourne True si le monde a chargé correctement
		"""
		if os.path.isdir(path):	# Si le dossier existe
			World.save_path = os.path.abspath(path) + "/" + name
			content = ""
			with open(World.save_path+"/World.conf") as file:	# Lie le contenu
				content = file.read()
			content = content.replace('\n', '')
			lines = content.split(';')
			# Récupère les paramètres
			conf = dict()
			for line in lines:
				if len(line)>0:
					buff = line.split('=')
					conf[buff[0]]=float(buff[1])
			print(conf["step"])
			World.setTime(h=conf["step"], tf=conf["tfinal"])
			World.create_Maizes(int(conf["maizes"]))
			return True
		else:
			return False

	
	@classmethod
	def load_save(cls, ani_h):
		"""
		Charge la sauvegarde du monde

		ani_h	le pas de l'animation
		
		Retourne True si la sauvegarde a chargé correctement
		"""
		if World.nbr_Maizes==0:
			return False
		else:
			for i in range(0, World.nbr_steps-1, ani_h):	# Parcours les fichiers avec le pas de l'animation
				content = ""
				with open(World.save_path + "/save." + str(i) + ".tsv", "r") as file:
					content = file.read()
				lines = content.split('\n')[1:]
				for j in range(World.nbr_Maizes):
					values = lines[j].split('\t')
					maize = World.maizes[j]
					maize.accels[i] = array([float(val) for val in values[1:4]])
					maize.vitesses[i] = array([float(val) for val in values[4:7]])
					maize.positions[i] = array([float(val) for val in values[7:]])
			return True

	@classmethod
	def addBox(cls, box):
		"""
		Ajoute une boite dans le monde
		"""
		World.box = box
	
	@classmethod
	def setTime(cls, h=None, nh=None, tf=None):
		"""
		Définie les constantes de temps
		"""
		a = tf!=None
		b = nh!=None
		c = h!=None
		# Si le temps n'a pas déja été définie et si suffisament de constantes ont été données
		if World.tfinal==0 and ((a and (b or c)) or (not(a) and b and c)):
			# Calcule la constante manquante
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
		"""
		Crée nbr de bille à l'origine et à vitesse initial nulle.
		"""
		World.nbr_Maizes = nbr
		World.maizes = [Maize(array([0,0,0]), array([0,0,0]), id=i) for i in range(nbr)]
	
	@classmethod
	def process(cls):
		"""
		Calcule la position de toutes les billes à chaque instant
		
		Retourne True si il n'y a pas de problème
		"""
		# Calc accels
		if not World.save_inited:
			print("Save is not initialized")
			return False
		path = World.save_path + "/save."
		n_p = 10
		for i in range(World.nbr_steps-1):	# Pour chaque instant
			file = open(path+str(i)+".tsv", "w")
			file.write("Temps(s)\tax\tay\taz\tvx\tvy\tvz\tx\ty\tz (UI)")
			World.box.move2D(i)	# Bouge la boite
			World.box.save(i)	# Sauvegarde l'état de la boite
			# Affiche l'avancement
			percent = int(100*i/World.nbr_steps)
			if percent==n_p:
				print(str(percent)+"%")
				n_p+=10
			# Parcours les billes du monde
			for maize in World.maizes:
				maize.PFD(i)	# Calcule la sommes des forces s'exerçant sur la bille
				maize.calcVel(i+1) # Calcule la (i+1)-ème vitesse
				maize.calcPosi(i+1)	# Calcule la i-ème vitesse
				# Ecris tous dans un fichier
				file.write('\n')
				file.write("maize")
				for v in [maize.accels[i], maize.vitesses[i], maize.positions[i]]:
						for j in range(3):
							file.write('\t')
							file.write(str(v[j]))
			file.close()
		return True

class Plan:
	"""
	Représente un plan de l'espace
	"""
	n_id = 1
	def __init__(self, psi, theta, point):
		"""
		Initialise l'objet
		
		psi, theta	les angles de rotations du plan
		point	un point par lequel passe le plan
		"""
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
		"""
		Sauvegarde l'état du plan à l'instant i
		"""
		self.states[i] = self.equa
	
	def calc_equa(self):
		"""
		Calcul l'équation cartésienne du plan
		"""
		self.d = np.dot(-self.normal, self.point)
		self.equa = array([self.normal[0], self.normal[1], self.normal[2], self.d])
	
	def calc_normal(self):
		"""
		Calcule un vecteur unitaire normal au plan
		"""
		self.normal = array([-cos(self.theta)*sin(self.psi), cos(self.theta)*cos(self.psi), sin(self.theta)])
		self.normal = makeUnit(self.normal)
	
	def calc_car(self, size):
		"""
		Calcule des points appartenants au plan permettant de l'afficher
		"""
		self.carX[0] = self.point[0] - size*cos(self.psi)
		self.carY[0] = self.point[1] - size*sin(self.psi)
		self.carX[1] = self.point[0] + size*cos(self.psi)
		self.carY[1] = self.point[1] + size*sin(self.psi)

	def show2D(self, size, style="-k"):
		"""
		Prépare l'affichage 2D du plan
		"""
		self.calc_car(size)
		self.repr, = plt.plot(self.carX, self.carY, style)
		return self.repr

class Box:
	"""
	Représente une boite
	"""
	def __init__(self, dim, ampli=PIs4, pulse=2.5133):	#	pulse = 2pi/2.5 = 2.5133
		"""
		Initialise l'objet
		"""
		self.dim = dim	# Dimension de la boite
		self.demiH = self.dim[1]/2
		self.demiL = self.dim[0]/2
		self.ampli = ampli	# amplitude du mouvement de rotation
		self.pulse = pulse	# pulsation de la rotation
		self.bot = Plan(0, 0, np.array([0,0,0]))								# Plan de la boite
		self.top = Plan(pi, 0, np.array([0, dim[1], 0]))						# Plan de la boite
		self.left = Plan(-PIs2, 0, np.array([-self.demiL, self.demiH, 0]))		# Plan de la boite
		self.right = Plan(PIs2, 0, np.array([self.demiL, self.demiH, 0]))		# Plan de la boite
		self.index = [self.bot, self.top, self.left, self.right]	# Pour parcourir les plans
		self.reset()	# Place la boite à sa position initiale
	
	def move2D(self, i):
		"""
		Fait une rotation de la boite selon l'axe z
		"""
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
	
	def evalDeltaBox(self, maize, i):
		"""
		Retourne une liste des interpénétrations de la bille avec chaque plan de la boite dans l'orde,
		[bottom, top, left, right]
		"""
		deltas = array([0]*4, dtype=float)
		for j in range(4):
			dist = distPlan(maize.positions[i], self.index[j].states[i])
			if dist<maize.R:
				deltas[j] = maize.R - dist
		return deltas
	
	def save(self, i):
		"""
		Sauvegarde l'état de la boite
		"""
		self.bot.save(i)
		self.top.save(i)
		self.left.save(i)
		self.right.save(i)
	
	def calc_car(self):
		"""
		Calcul les equations caractéristique des plan de la boite
		"""
		self.bot.calc_car(self.demiL)
		self.left.calc_car(self.demiH)
		self.right.calc_car(self.demiH)
		self.top.calc_car(self.demiL)
	
	def show2D(self):
		"""
		Prépare l'affichage 2D de la boite
		"""
		self.r_bot = self.bot.show2D(self.demiL)
		self.r_top = self.top.show2D(self.demiL)
		self.r_left = self.left.show2D(self.demiH)
		self.r_right = self.right.show2D(self.demiH)
	
	def update2D(self, i):
		"""
		Met à jour l'affiche de la boite à l'instant i
		"""
		self.move2D(i)
		self.calc_car()
		self.r_bot.set_data(self.bot.carX, self.bot.carY)
		self.r_left.set_data(self.left.carX, self.left.carY)
		self.r_right.set_data(self.right.carX, self.right.carY)
		self.r_top.set_data(self.top.carX, self.top.carY)
	
	def reset(self):
		"""
		Place la boite à sa position initiale
		"""
		self.move2D(0)

class Maize:
	"""
	Représente une bille de maïsse
	"""
	ro = 1180	# kg/m3					# Constante du maïsse
	coef_resti_contact = 0.5			# Constante du maïsse
	young = 340e6						# Constante du maïsse
	fm = 0.86							# Constante du maïsse
	fa = 0.54							# Constante du maïsse
	poisson = 0.3						# Constante du maïsse
	Estar = young/(2-2*pow(poisson, 2))	# Constante du maïsse
	lne = log(coef_resti_contact)		# Constante du maïsse
	coef_amorti_contact = lne/sqrt(pow(pi,2) + pow(lne,2))	# Constante du maïsse
	n_id = 0

	def __init__(self, pos, vits, R=0.005, id=None):
		"""
		Initialise une bille
		pos		array	position de la bille à t=0
		vits	array	vitesse de la bille à t=0
		"""
		if id==None:
			self.ID = Maize.n_id
			Maize.n_id+=1
		else:
			self.ID = id
		self.positions = np.zeros((World.nbr_steps, 3))
		self.vitesses = np.zeros((World.nbr_steps, 3))
		self.accels = np.zeros((World.nbr_steps, 3))
		self.positions[0] = pos
		self.vitesses[0] = vits
		self.R = R
		self.Rstar = R/2
		self.vol = (4/3)*pi*pow(R, 3)
		self.masse = self.vol*self.ro
		self.sum_F = np.ones((World.nbr_steps, 3))*np.array([0, -World.gravity*self.masse, 0])	# Soumets la bille à son poids à chaque pas
	
	def PFD(self, i):
		"""
		Cherche les forces qui agissent sur la bille, fait leur somme et calcule l'accélération à l'instant i
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
			if deltas[j]!=0:
				Kstar = 4*Maize.Estar*sqrt(self.R*deltas[j])/3
				# Répulsion de Hertz et amortissement
				self.sum_F[i] += (Kstar*deltas[j] - 2*Maize.coef_amorti_contact*deltaPoints[j]*sqrt(Kstar*self.masse))*World.box.index[j].normal
		# Recherche les contact avec d'autres bille
		for j in range(self.ID+1, World.nbr_Maizes):
			maize = World.maizes[j]
			delta = evalDeltaMaize(self, maize, i)
			if delta!=0:
				deltaPoint = 0 if i==0 else (delta - evalDeltaMaize(self, maize, i-1))/World.step
				normal = makeUnit(maize.positions[i] - self.positions[i])
				Kstar = 4*Maize.Estar*sqrt(self.R*delta/2)/3
				F = (2*Maize.coef_amorti_contact*deltaPoint*sqrt(Kstar*self.masse/2) - Kstar*delta)*normal
				self.sum_F[i] += F
				maize.sum_F[i] -= F
		self.accels[i] = self.sum_F[i]/self.masse

	def calcVel(self, i):
		"""
		Intègre l'accélération pour obtenir la i-ème vitesse

		/!\ la i-ème accélération est à l'indice i-1
		"""
		self.vitesses[i] = self.vitesses[i-1] + World.step*self.accels[i-1]
	def calcPosi(self, i):
		"""
		Intègre la vitesse pour obtenir la i-ème position
		"""
		self.positions[i] = self.positions[i-1] + World.step*self.vitesses[i]

	def setInit(self, pos, vits):
		"""
		Définie les conditions initiales de la bille
		"""
		self.positions[0] = pos
		self.vitesses[0] = vits
	
	def getPosi(self, i):
		"""
		Retourne la i-ème position
		"""
		return self.positions[i]
	def getVel(self, i):
		"""
		Retourne la i-ème vitesse
		"""
		return self.vitesses[i]
	def getAccel(self, i):
		"""
		Retourne la i-ème accélération
		"""
		return self.accels[i]
	
	def setPosi(self, posi, i):
		"""
		Définie la i-ème position
		"""
		self.positions[i] = posi
	def setVel(self, vel, i):
		"""
		Définie la i-ème vitesse
		"""
		self.vitesses[i] = vel
	def setAccel(self, accel, i):
		"""
		Définie la i-ème accélération
		"""
		self.accels[i] = accel