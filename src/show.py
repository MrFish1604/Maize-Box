from classes import World, Box
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import size
from time import time

SAVE_PATH = "../simulations/test2"

t0 = time()
print(World.load_save(SAVE_PATH))
print(time() - t0)

box = Box((1, 0.3))
World.addBox(box)

fig = plt.figure()
plt.axis("equal")

box.reset()
box.show2D()
# World.show2D_maizes(fig)

ani_h=50
nbr_frames = int(World.tfinal*ani_h*1000)
new_h = int(ani_h*0.001/World.step)

maize = World.maizes[0]
print(maize.positions)

circle = plt.Circle((maize.positions[0,0], maize.positions[0,1]), radius=maize.R)

def init_ani():
	fig.gca().add_patch(circle)

def animate(i):
	j = (new_h*i)%World.nbr_steps
	circle.center = (maize.positions[j, 0], maize.positions[j, 1])
	box.update2D(j)

ani = FuncAnimation(fig, animate, init_func=init_ani, interval=ani_h, frames=nbr_frames)
plt.gca().set_xlim(-0.7, 0.7)
plt.gca().set_ylim(-0.6, 0.8)
input("Press enter to display...")
plt.show()