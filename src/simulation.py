from lib import World, Maize, Box
from numpy import array
from time import time

SAVE_PATH = "./"	# Répertoire de la simulation
SAVE_NAME = "super_simulation"	# Nom de la simulation

World.setTime(h=1e-5, tf=5) # Initialise les paramètre de temps
box = Box((1, 0.3)) # Crée une boite de 1m * 0.3m
World.addBox(box)   # Ajoute la boite au monde
World.create_Maizes(1) # Crée 50 bille de même taille dans le monde

# Initialise la position et la vitesse initial de toutes les billes
d = World.maizes[0].R + 0.01	# distance entre les centres des billes
c=0
l=box.dim[1] - 0.05
for i in range(World.nbr_Maizes):
		if c==24:
			l-=d
		c = i%25
		World.maizes[i].setInit(array([c*d-0.3, l, 0]), array([0,0,0]))

World.init_save(SAVE_PATH, SAVE_NAME)   # Initialise les paramètres de sauvegarde
box.reset() # Place la boite à sa position initiale

t0 = time() # Sauvegarde le temps avant le début des calcules
print("Start process")
World.process() # Fait touts les calcules
t = time() - t0
h = t//3600
m = t//60 - 60*h
s = t%60
print("{}h{}m{}s".format(h,m,s))	# Affiche le temps de calcules en s