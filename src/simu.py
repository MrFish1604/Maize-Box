import numpy as np

class World:
	"""
	This class should represent the world where balls will evolve.
	I don't know if it will be usefull.
	"""
	nbr_Maize = 0
	nbr_steps = 0	# Ne compte pas la 1er etape
	step = 0
	tfinal = 0

	def setTime(tf=None, nh=None, h=None):
		"""
		Set times constants
		"""
		a = tf!=None
		b = nh!=None
		c = h!=None
		if World.tfinal==0 and ((a and (b or c)) or (not(a) and b and c)):
			if not a:	#Si tf==None


		
	def create_Maizes(nbr):
		World.nbr_Maize = nbr
		World.maizes = [Maizes() for i in range(9)]
	

class Maize:
	"""
	This class represent a ball of maize
	"""
	def __init__(self, pos, vits):
		"""
		initialize a maize
		pos		array	position of the ball at t=0
		vits	array	velocity of the ball at t=0
		"""
		self.positions = np.zeros((World.nbr_steps, 3))
		self.velocities = np.zeros((World.nbr_steps, 3))
		self.positions[0] = pos
		self.velocities[0] = vits

#m2 = Maize(np.array([0,1,0]), np.array([0,0,9]))