from lib import World, Maize, Box
from numpy import array
from time import time

SAVE_PATH = "/media/matt/128Go_CABILLOT/simulation"
SAVE_NAME = "2m_1e-5_5s"

World.setTime(h=1e-5, tf=5) # Initialise les paramètre de temps
box = Box((1, 0.3)) # Crée une boite de 1m * 0.3m
World.addBox(box)   # Ajoute la boite au monde
World.create_Maizes(2) # Crée 50 bille de même taille dans le monde

# Initialise la position et la vitesse initial de toutes les billes
for i in range(World.nbr_Maizes):
    World.maizes[i].setInit(array([i/80, 0.2, 0]), array([0,0,0]))

World.init_save(SAVE_PATH, SAVE_NAME)   # Initialise les paramètres de sauvegarde
box.reset() # Place la boite à sa position initiale

t0 = time() # Sauvegarde le temps avant le début des calules
print("Start to process")
World.process() # Fait tout les calcules
t = time() - t0
h = t//3600
m = t//60 - 60*h
s = t%60
print("{}h{}m{}s".format(h,m,s))	# Affiche le temps de calcules en s