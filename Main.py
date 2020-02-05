'''
Created on 2020/02/05

@author: user
'''
import FDTD
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma import masked_array
import time
import Pulser
import PhysicalProperty as prop

if __name__ == '__main__':
    freq = 5.0e6
    numOfPulsers = 16
    pulserPitch = 0.25e-3
    pulserAngle = 36.6
    velocity = 3230

    pls = Pulser.Pulser(freq, velocity, numOfPulsers, pulserPitch, pulserAngle)
    pls.calcForcalLow(0)
    nx = 960
    ny = 480
    f = FDTD.eFDTD(nx, ny, pls)


    x = f.px
    x = np.append(x, x + (pls.height / pls.dx) * np.tan(np.deg2rad(pulserAngle)))
    x = np.append(x, x[1] + 400)
    y = f.py
    y = np.append(y, y + (pls.height / pls.dx))
    y = np.append(y, y[1] + 400)
    
    
    xTicks = np.linspace(0, nx, 7)
    xTicksLab = np.floor(xTicks * pls.dx * 1e3)
    yTicks = np.linspace(0, ny, 7)
    yTicksLab = np.floor(yTicks * pls.dx * 1e3)

    fig = plt.figure()

    def plot(data):
        plt.cla()
        for i in np.arange(5):
            res = masked_array(f.Vx**2 + f.Vy**2, f.model!=i)
            plt.imshow(res, cmap= prop.cmap[i], vmin=0.0, vmax=1.0e-6,  interpolation = "none")
        plt.plot(x, y, "--", color = "w")
        plt.title("time:{0:.1f}".format((f.n_point * pls.dt * (10 ** 9))) + "[ns]")
        plt.xticks(xTicks, xTicksLab)
        plt.yticks(yTicks, yTicksLab)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.grid(which='major',color='gray',linestyle='-')

        start = time.time()
        for i in range(10):
            f.updateV()
            f.updateS()
            f.updateSource()
            f.n_point += 1

        elapsed_time = time.time() - start
        print ("{0} elapsed_time:{1:.3f}".format(f.n_point, elapsed_time) + "[sec]")

    ani = animation.FuncAnimation(fig, plot, frames = 400)
    #plt.show()
    ani.save('test10+A00-3.gif', writer='pillow', fps = 30)