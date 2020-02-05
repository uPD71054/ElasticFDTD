'''
Created on 2020/01/12

@author: user
'''
import numpy as np
import PhysicalProperty as prop
import matplotlib.pyplot as plt
from dask.array.ufunc import deg2rad

class Pulser(object):
    '''
    classdocs
    '''

    def __init__(self, frequency, velocity, elements, pitch, angle = 0.0):
        '''
        Constructor
        '''
        # pulser frequency [Hz]
        self.frequency = frequency
        # velocity [m/s]
        self.velocity = velocity
        # number of elements
        self.elements = elements
        # element pitch
        self.pitch = pitch
        # angle of melements(default:0 deg)
        self.angle = angle
        # amplitude of pulse
        self.amp = 5.0e-3
        # PML length at pixel
        self.pmlLength = 20
        # shoe height [m]
        self.height = 4.0e-3
        
        
        # stability conditions
        # dx:spatical resolution [m]
        waveLength =np.min(prop.vs[np.nonzero(prop.vs)]) / self.frequency
        self.dx = min([waveLength / 10, 0.025e-3])
        # dt:time resolution:dt [s]
        self.dt = self.dx / (np.sqrt(2) * np.max(prop.vp[np.nonzero(prop.vp)])) / 2
        self.dt = min([self.dt, 1.0e-9])
        
        print("spatial resolution : %f [mm]" % (self.dx * (10**3)))
        print("time resolution : %f [ns]" % (self.dt * (10 ** 9)))
        
        
        
        self.__calcElementOffset()
        
        self.__calcPulserOffset()
        
        self.__genShoe()
        # calculate pulse
        self.calcSinPulse()
        
        # initialize forcal low(delay time)
        self.delay = np.zeros(self.elements, dtype='int32')
    
    
    def __genShoe(self):
        self.pSize = [np.max(self.iy) - np.min(self.iy), np.max(self.ix) - np.min(self.ix)]
        
        w = self.pSize[1] + 4 * self.pitch / self.dx
        h = w * np.tan(deg2rad(self.angle))
        H = self.height / self.dx + h / 2
        W = w + H * np.tan(np.deg2rad(self.angle))
        
        w = w.astype('int32')
        h = h.astype('int32')
        W = W.astype('int32')
        H = H.astype('int32')
        
        self.center = np.array([h / 2, w / 2], dtype='int32')
        self.shoe = np.ones((H, W), dtype = "int32") * 3
        
        for i in np.arange(h):
            self.shoe[i, :(w + i / np.tan(np.deg2rad(-self.angle))).astype('int32')] = 4
    
    
    def __calcElementOffset(self):
        i = np.arange(-np.round(self.pitch / (2 * self.dx)), np.round(self.pitch / (2 * self.dx)))
        # elements offset from elements center
        self.ex = i * np.cos(np.deg2rad(self.angle))
        self.ex = np.array(np.round(self.ex), dtype='int32')
        self.ey = i * (- np.sin(np.deg2rad(self.angle)))
        self.ey = np.array(np.round(self.ey), dtype='int32')
    
    
    def __calcPulserOffset(self):
        i = (np.arange(self.elements) - self.elements / 2 + 1/2) * self.pitch
        # pulser offset from pulser center
        self.ix = i * np.cos(np.deg2rad(self.angle))
        self.ix = np.array(np.round(self.ix / self.dx), dtype='int32')
        self.iy = i * (- np.sin(np.deg2rad(self.angle)))
        self.iy = np.array(np.round(self.iy / self.dx), dtype='int32')
    
    
    def calcSinPulse(self):
        length = np.floor((0.5/self.frequency)/self.dt)
        t = np.arange(0, length) * self.dt
        p = np.sin((2.0 * np.pi * self.frequency * t ))
        
        self.pulse = np.zeros(1024 * 32)
        self.pulse[:p.size] = p
    
    
    def calcForcalLow(self, steer):
        self.delay = np.arange(self.elements) * self.pitch * np.tan(np.deg2rad(steer)) / self.velocity
        self.delay -= np.min(self.delay)
        self.delay = np.array(np.floor(self.delay / self.dt), dtype='int32')
        
