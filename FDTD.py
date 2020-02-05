'''
Created on 2019/12/22

@author: user
'''

import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import Pulser
import PhysicalProperty as prop

class eFDTD(object):
    '''
    classdocs
    '''
    def __init__(self, nx, ny, pls):
        '''
        Constructor
        '''
        # Calculate Parameter
        # spatial resolution of each axis
        self.nx = nx
        self.ny = ny
        # pulser
        self.pls = pls
        # PML length at pixel
        self.pmlLength = 20
        # reflection coefficient
        self.rcoef = 0.0001
        # time step
        self.n_point = 0;

        self.sx = 30
        self.sy = 30
        self.px = self.sx + self.pls.center[1]
        self.py = self.sy + self.pls.center[0]

                # Initialize Model
        self.model = np.ones((self.ny, self.nx)) * 4
        self.model[self.sy:self.sy + pls.shoe.shape[0] , self.sx:self.sx + pls.shoe.shape[1]] = pls.shoe
        self.model[self.sy+pls.shoe.shape[0]:self.sy+pls.shoe.shape[0]+160,:] = 0
        self.model[self.sy+pls.shoe.shape[0]+0:self.sy+pls.shoe.shape[0]+80,340:345] = 4
        #self.model[240:320,475:485] = 4



        # Lame parameters (normalized with density)
        # vp = sqrt((λ+2μ)/ρ) [m/s]
        # vs = sqrt(μ/ρ) [m/s]
        lam1st = prop.density * (prop.vp**2 - 2 * prop.vs**2)
        lam2nd = prop.density * prop.vs**2

        # Lame parameter and density at x:+0, y:+0
        self.lam = np.zeros((4, self.ny, self.nx), dtype = 'float')
        self.mu = np.zeros((4, self.ny, self.nx), dtype = 'float')
        self.rho = np.zeros((4, self.ny, self.nx), dtype = 'float')
        for i in np.arange(5):
            self.lam[0,:,:] += np.where(self.model == i, lam1st[i], 0)
            self.mu[0,:,:] += np.where(self.model == i, lam2nd[i], 0)
            self.rho[0,:,:] += np.where(self.model == i, prop.density[i], 0)

        self.rho[1,:,:] = self.rho[0,:,:]
        self.rho[2,:,:] = self.rho[0,:,:]
        self.rho[3,:,:] = self.rho[0,:,:]
        # Lame parameter at x:+1/2, y:+0
        self.lam[1,:,:-1] = (self.lam[0,:,:-1] + self.lam[0,:,1:]) / 2
        self.mu[1,:,:-1] = (self.mu[0,:,:-1] + self.mu[0,:,1:]) / 2
        self.rho[1,:,:-1] = (self.rho[0,:,:-1] + self.rho[0,:,1:]) / 2
        # Lame parameter at x:+0, y:+1/2
        self.lam[2,:-1,:] = (self.lam[0,:-1,:] + self.lam[0,1:,:]) / 2
        self.mu[2,:-1,:] = (self.mu[0,:-1,:] + self.mu[0,1:,:]) / 2
        self.rho[2,:-1,:] = (self.rho[0,:-1,:] + self.rho[0,1:,:]) / 2
        # Lame parameter at x:+1/2, y:+1/2
        self.lam[3,:-1,:-1] = (self.lam[0,1:,1:] + self.lam[0,1:,:-1] + self.lam[0,:-1,1:] + self.lam[0,:-1,:-1]) / 4
        self.mu[3,:-1,:-1] = (self.mu[0,1:,1:] + self.mu[0,1:,:-1] + self.mu[0,:-1,1:] + self.mu[0,:-1,:-1]) / 4
        self.rho[3,:-1,:-1] = (self.rho[0,1:,1:] + self.rho[0,1:,:-1] + self.rho[0,:-1,1:] + self.rho[0,:-1,:-1]) / 4

        # (λ+2μ)/dx
        self.lambdaPlus2muOverDelta = (self.lam + 2 * self.mu) / self.pls.dx
        # λ/dx
        self.lambdaOverDelta = self.lam / self.pls.dx
        # μ/dx
        self.muOverDelta = self.mu / self.pls.dx
        # 1/ρdx
        self.oneOverRhoDelta = 1 / (self.rho * self.pls.dx)

        # PML damping parameter
        # d(x) = d0 * (x/δ)~2
        # d0 = (3vp/2δ) * log(1/Rcoef)
        #    δ:Length of PML layer [mm]
        #    Rcoef(<<1):theoretical refrection coeffient
        d0 = 3.0 * prop.vs[3] * np.log(1.0/self.rcoef) / (2.0 * self.pmlLength * self.pls.dx)

        dh = np.zeros((self.ny, self.nx), dtype = 'float')
        dv = np.zeros((self.ny, self.nx), dtype = 'float')

        dh[:,:self.pmlLength] = (d0 * (np.arange(self.pmlLength, 0, -1) / self.pmlLength)**2) /2
        dh[:,-self.pmlLength:] = (d0 * (np.arange(1, self.pmlLength + 1, 1) / self.pmlLength)**2) / 2
        dv[:self.pmlLength,:] = (d0 * (np.arange(self.pmlLength, 0, -1).reshape([-1, 1]) / self.pmlLength)**2)/2
        dv[-self.pmlLength:,:] = (d0 * (np.arange(1, self.pmlLength + 1, 1).reshape([-1, 1]) / self.pmlLength)**2) / 2

        self.dhm = (1 / self.pls.dt) - dh
        self.dhp = (1 / self.pls.dt) + dh
        self.dvm = (1 / self.pls.dt) - dv
        self.dvp = (1 / self.pls.dt) + dv



        # Initialize Field Values
        # Vx = Vx1+Vx2:x component of velocity [kg・/m^2s]
        self.Vx = np.zeros((self.ny, self.nx))
        self.Vx1 = np.zeros((self.ny, self.nx))
        self.Vx2 = np.zeros((self.ny, self.nx))
        # Vy = Vy1+Vy2:y component of velocity [kg・/m^2s]
        self.Vy = np.zeros((self.ny, self.nx))
        self.Vy1 = np.zeros((self.ny, self.nx))
        self.Vy2 = np.zeros((self.ny, self.nx))
        # σx = σx1 + σx2:x component of normal stress [N/m^2]
        self.Sx = np.zeros((self.ny, self.nx))
        self.Sx1 = np.zeros((self.ny, self.nx))
        self.Sx2 = np.zeros((self.ny, self.nx))
        # σy = σy1 + σy2:y component of normal stress [N/m^2]
        self.Sy = np.zeros((self.ny, self.nx))
        self.Sy1 = np.zeros((self.ny, self.nx))
        self.Sy2 = np.zeros((self.ny, self.nx))
        # τ = τ1 + τ2:shear stress [N/m^2]
        self.T = np.zeros((self.ny, self.nx))
        self.T1 = np.zeros((self.ny, self.nx))
        self.T2 = np.zeros((self.ny, self.nx))





    def updateS(self):
        # update Sx
        self.Sx1[1:,:-1] = (self.Sx1[1:,:-1] * self.dhm[1:,:-1] + self.lambdaPlus2muOverDelta[0,1:,:-1] * np.diff(self.Vx[1:,:], axis = 1)) / self.dhp[1:,:-1]
        self.Sx2[1:,:-1] = (self.Sx2[1:,:-1] * self.dvm[1:,:-1] + self.lambdaOverDelta[0,1:,:-1] * np.diff(self.Vy[:,:-1], axis = 0)) / self.dvp[1:,:-1]
        self.Sx = self.Sx1 + self.Sx2
        # update Sy
        self.Sy1[1:,:-1] = (self.Sy1[1:,:-1] * self.dhm[1:,:-1] + self.lambdaOverDelta[0,1:,:-1] * np.diff(self.Vx[1:,:], axis = 1)) / self.dhp[1:,:-1]
        self.Sy2[1:,:-1] = (self.Sy2[1:,:-1] * self.dvm[1:,:-1] + self.lambdaPlus2muOverDelta[0,1:,:-1] * np.diff(self.Vy[:,:-1], axis = 0)) / self.dvp[1:,:-1]
        self.Sy = self.Sy1 + self.Sy2
        # update T
        self.T1[:-1,1:] = (self.T1[:-1,1:] * self.dhm[:-1,1:] + self.muOverDelta[0,:-1,1:] * np.diff(self.Vy[:-1,:], axis = 1)) / self.dhp[:-1,1:]
        self.T2[:-1,1:] = (self.T2[:-1,1:] * self.dvm[:-1,1:] + self.muOverDelta[0,:-1,1:] * np.diff(self.Vx[:,1:], axis = 0)) / self.dvp[:-1,1:]
        self.T = self.T1 + self.T2


    def updateV(self):
        # update Vx
        self.Vx1[1:,1:] = (self.Vx1[1:,1:] * self.dhm[1:,1:] + self.oneOverRhoDelta[0,1:,1:] * np.diff(self.Sx[1:,:], axis = 1)) / self.dhp[1:,1:]
        self.Vx2[1:,1:] = (self.Vx2[1:,1:] * self.dvm[1:,1:] + self.oneOverRhoDelta[0,1:,1:] * np.diff(self.T[:,1:], axis = 0)) / self.dvp[1:,1:]
        self.Vx = self.Vx1 + self.Vx2
        # update Vy
        self.Vy1[:-1,:-1] = (self.Vy1[:-1,:-1] * self.dhm[:-1,:-1] + self.oneOverRhoDelta[0,:-1,:-1] * np.diff(self.T[:-1,:], axis = 1)) / self.dhp[:-1,:-1]
        self.Vy2[:-1,:-1] = (self.Vy2[:-1,:-1] * self.dvm[:-1,:-1] + self.oneOverRhoDelta[0,:-1,:-1] * np.diff(self.Sy[:,:-1], axis = 0)) / self.dvp[:-1,:-1]
        self.Vy = self.Vy1 + self.Vy2


    def updateSource(self):
        if (self.n_point < self.pls.pulse.size):
            for i in np.arange(self.pls.elements):
                x = self.px + self.pls.ix[i]
                y = self.py + self.pls.iy[i]
                self.Vx1[y + self.pls.ey, x + self.pls.ex] += np.sin(np.deg2rad(self.pls.angle)) * self.pls.amp * self.pls.pulse[self.n_point - self.pls.delay[i]]
                self.Vy2[y + self.pls.ey, x + self.pls.ex] += np.cos(np.deg2rad(self.pls.angle)) * self.pls.amp * self.pls.pulse[self.n_point - self.pls.delay[i]]


    def updateReciever(self):
        if (self.n_point < self.t.size):
            self.reciever1[self.n_point] = np.abs(self.Sx[240,340] + self.Sy[240,340] + self.T[240,340])
            self.reciever2[self.n_point] = np.abs(self.Sx[240,440] + self.Sy[240,440] + self.T[240,440])