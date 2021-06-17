from lib import World, Box
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from numpy import size
from time import time

# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

SIMU_NAME = "10m_BnM_1e-5"
SAVE_PATH = "../simulations/" + SIMU_NAME
VIDEO_PATH = "../video/" + SIMU_NAME + ".mp4"

t0 = time()
World.load_World(SAVE_PATH)

ani_h=50
nbr_frames = int(2*World.tfinal*ani_h*1000)
new_h = int(ani_h*0.001/(2*World.step))

World.load_save(new_h)
print(time() - t0)

box = Box((1, 0.3))
World.addBox(box)

fig = plt.figure()
plt.axis("equal")

box.reset()
box.show2D()
World.show2D_maizes(fig)

def animate(i):
	j = (new_h*i)%World.nbr_steps
	World.update2D_maizes(j)
	box.update2D(j)

ani = FuncAnimation(fig, animate, interval=ani_h, frames=nbr_frames, repeat=False)

# Save the video
# writervideo = FFMpegWriter(fps=60)
# writervideo = PillowWriter(fps=30) 
# ani.save(VIDEO_PATH, writer=writervideo)

plt.gca().set_xlim(-0.7, 0.7)
plt.gca().set_ylim(-0.6, 0.8)
input("Press enter to display...")
plt.show()