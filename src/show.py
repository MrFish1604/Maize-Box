from lib import World, Box
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from numpy import size
from time import time
from os.path import isdir
from math import pow

SIMU_NAME = "super_simulation"
SAVE_PATH = "./"
VIDEO_PATH = "./" + SIMU_NAME + ".mp4"

t0 = time()
if World.load_World(SAVE_PATH, SIMU_NAME):
	fps = 60	# frames per seconds
	nbr_frames = int(fps*World.tfinal)	# Nombre de frames
	ani_h = World.tfinal/nbr_frames	# Interval de rafraichissement de l'animation
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

		# Sauvegarde la vidéo
		print("Saving video...")
		writervideo = FFMpegWriter(fps=fps)
		ani.save(VIDEO_PATH, writer=writervideo)

		input("Press enter to display...")
		plt.show()	# N'est pas en temps réel car l'ordinateur n'est pas capable d'executer animate (plus toute la blackbox qu'est matplotlib) en moins de ani_h secondes
		#			La vidéo sera cependant en temps réel
		#			Pour avoir un affiche en temps réel, il faut fixé ani_h a 0.05s
else:
	print("Can't load the world")