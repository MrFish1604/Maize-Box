from lib import World, Box
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from numpy import size
from time import time
from os.path import isdir
from math import pow

SIMU_NAME = "1m_2grav_1e-5"
SAVE_PATH = "/media/matt/128Go_CABILLOT/simulations"
VIDEO_PATH = "../video/" + SIMU_NAME + ".mp4"

t0 = time()
if World.load_World(SAVE_PATH, SIMU_NAME):
	ani_h = 0.05	# Interval de rafraichissement de l'animation
	nbr_frames = int(World.tfinal/ani_h)	# Nombre de frames
	fps = int(nbr_frames/World.tfinal)
	# ani_h=0.05
	new_h = int(ani_h*World.nbr_steps/World.tfinal)	# Calcule le nombre de pas à sauté

	if World.load_save(new_h):	# Charge le monde à partir des fichiers
		box = Box((1, 0.3))	# Crée une boite
		World.addBox(box)	# Ajoute la boite au monde

		fig = plt.figure()	# Crée une figure
		plt.axis("equal")

		frames = [new_h*i for i in range(nbr_frames)]	# Crée la liste des indices de temps utilisé pour l'affichage

		def init():
			"""
			Initialise l'animation
			"""
			World.show2D_maizes(fig)
			box.reset()	# Met la boite à la position 0
			box.show2D()	# Prépare l'affichage de la boite

		def animate(i):
			"""
			Appelé à chaque rafraichissement de l'image
			"""
			World.update2D_maizes(i)	# Met à jour la position des billes
			box.update2D(i)				# Met à jour la position de la boite

		ani = FuncAnimation(fig, animate, init_func=init, interval=ani_h*1000, frames=frames, repeat=True)	# Crée l'animation

		plt.gca().set_xlim(-0.7, 0.7)	# Règle les limite de l'axe x
		plt.gca().set_ylim(-0.6, 0.8)	# Règle les limite de l'axe y

		input("Press enter to display...")
		plt.show()	# Affiche l'animation en temps réel
else:
	print("Can't load the world")