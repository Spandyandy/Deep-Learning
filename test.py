import matplotlib.pyplot as plt
from numpy import *
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = arange(0, 2*pi, 0.01)
line, = ax.plot(x, sin(x))


def animate(i):
    line.set_ydata(sin(x + i/10.0))  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, arange(1, 200), init_func=init,
                              interval=25, blit=True)
plt.show()