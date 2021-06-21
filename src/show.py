from lib import World, Box
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from numpy import size
from time import time
from os.path import isdir
from math import pow

# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

SIMU_NAME = "50m_5s_1e-5"
SAVE_PATH = "/media/matt/128Go_CABILLOT/" + SIMU_NAME
VIDEO_PATH = "../video/" + SIMU_NAME + ".mp4"

print(SAVE_PATH, isdir(SAVE_PATH))

t0 = time()
print(World.load_World(SAVE_PATH))

fps = 60
nbr_frames = int(fps*World.tfinal)	# s
ani_h = World.tfinal/nbr_frames
new_h = int(ani_h*World.nbr_steps/World.tfinal)		# Calcule le nombre de pas sauté

World.load_save(new_h)	# Charge le monde à partir des fichiers
print(time() - t0)

print("ani_h=",ani_h)

box = Box((1, 0.3))	# Crée une boite
World.addBox(box)	# Ajoute la boite au monde

fig = plt.figure()
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
	# j = i%World.nbr_steps
	World.update2D_maizes(i)	# Met à jour la position des billes
	box.update2D(i)				# Met à jour la position de la boite

ani = FuncAnimation(fig, animate, init_func=init, interval=ani_h*1000, frames=frames, repeat=True)	# Crée l'animation

plt.gca().set_xlim(-0.7, 0.7)	# Règle les limite de l'axe x
plt.gca().set_ylim(-0.6, 0.8)	# Règle les limite de l'axe y

# Save the video
writervideo = FFMpegWriter(fps=fps)
ani.save(VIDEO_PATH, writer=writervideo)

input("Press enter to display...")
plt.show()