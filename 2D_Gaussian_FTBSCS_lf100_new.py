#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:56:37 2022

@author: admin
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
from matplotlib.ticker import NullFormatter, StrMethodFormatter, LogLocator

#instability details
e = 0.3
me = 0.5
epsilon = 0.5
alpha = 1e-3
Tp = 5e-6 #in MeV
#ne = 1e+20 * (2e-11)**3 #in MeV^3 ****** check units **********
ne = 1e+16 
nb = alpha * ne
wp = 8980 * np.sqrt(ne) * 6.6e-22 #in MeV
#wp = np.sqrt(ne * e**2/(me * epsilon)) 
J = 10 
delta = 1e-1 #typical value
W0 = (ne * (2e-11)**3) * Tp  #background


upsilon = delta * wp * W0 / (nb * (2e-11)**3)


dc = np.pi * (e**2) * W0 * J / wp

print('plasma frequency=',wp)
print('drift=',upsilon)
print('diffusion=',dc)

# grid setup
xmin = 0
xmax = 1

ymin = -0.5
ymax = 0.5

xdim = xmax - xmin

ydim = ymax - ymin
# intervals in x-, y- directions
dx = dy = 0.01
# Diffusion coefficient for a growth rate of 10^-3; explicit time-depends makes the solution unstable
D = dc #as long as D and L are comparable the D term counters numerical anti-diffusion and the solution is stable
L= upsilon
Lperp = upsilon*1e-50

Noparticle, Allparticles = 0.01, 0.1 #bounded by probability distribution

#nx, ny = int(xdim/dx), int(ydim/dy)
nxmin, nxmax, nymin, nymax = int(xmin/dx), int(xmax/dx), int(ymin/dy), int(ymax/dy)
nx = nxmax - nxmin
ny = nymax - nymin

dx2, dy2 = dx*dx, dy*dy

#dt = 1/((L * (dx + dy))/(dx * dy)  +  (2 * D * (dx2 + dy2))/(dx2 * dy2))
#dt = min(dx**2/(2*D), dy**2/(2*D)) / (2*np.abs(L/dx) + 2*np.abs(Lperp/dy) + 1)
dt = 1/(((L * dx + Lperp * dy))/(dx * dy)  +  (2 * D * (dx2 + dy2))/(dx2 * dy2))
print('dt=',dt)

u0 = Noparticle * np.ones((nx, ny))
u = u0.copy()

# Initial conditions - gaussian centered at (cx,cy)

#s0, cx, cy = 0.01, 0.25, 0
s0, cx, cy = 0.01, 0.5, 0.


s02 = s0**2
for i in range(nxmin, nxmax):
    for j in range(nymin, nymax):
        cent = 50/100 # this should be the center pixel divided by 100
        u0[i,j] = (1/(2*np.pi)) * (1/(s02)) * np.exp(-(1/s02)*(i*dx-cent)**2 - (1/s02)*(j*dy-cent)**2)


def do_timestep(u0, u):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + dt * ( L * (u0[1:-1,1:-1] - u0[:-2,1:-1])/dx + Lperp * (u0[1:-1,1:-1] - u0[1:-1,:-2])/dy ) + D * dt * (
          (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
          + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )
    

    u0 = u.copy()

    #sdx = np.std(u.copy(), axis = 0)
    #sdy = np.std(u.copy(), axis =1)
    return u0, u #, sdx, sdy

#

# Number of timesteps
nsteps = 10001
# Output 4 figures at these timesteps
mfig = [1, 50, 100, 1000]
fignum = 0
fig = plt.figure()
if L**2 <= L + 2*D <=1: #stability in FTBSCS
    for m in range(nsteps):
        u0, u = do_timestep(u0, u)
        if m in mfig:
            fignum += 1
            print(m, fignum)
            ax1 = fig.add_subplot(140 + fignum)
            imageraw = u.copy().T
            image = imageraw/(np.linalg.norm(imageraw))
            #image = np.log10(imageraw/(np.linalg.norm(imageraw)))
            #meanx = image[49].sum(0)/len(image[49])
            #print(meanx)
            #im = ax1.imshow(image[:][50:99], cmap=plt.get_cmap('hot'), )
            im = ax1.imshow(image, cmap=plt.get_cmap('hot'), vmin=Noparticle, vmax=Allparticles, extent = [0,100,-50,50])
            #im = ax1.imshow(image, cmap=plt.get_cmap('hot'), norm=LogNorm(vmin=1e-3, vmax=Allparticles), extent = [0,100,-50,50])
            #norm=matplotlib.colors.LogNorm(vmin=a.min(), vmax=a.max())
            #peakloc, _ = find_peaks(image, height=0)
            #print(peakloc)
    #        ax.set_axis_off()
            #ax1.scatter((cx)/dx, (cy-0.5)/dy, marker = 'x', color = 'green')
            #ax1.scatter((cx)/dx, (cy-0.5)/dy, marker = 'x', color = 'green')
            #ax1.invert_yaxis()
            ax1.set_xlabel('$p_{||}$ (MeV)')
            ax1.set_ylabel('$p_\perp$ (MeV)')
            #ax1.set_title('{:.1f} $/\omega_p$'.format((wp/(2*delta*wp))*np.log(2*m*dt*delta*wp*1e+10))) 
            ax1.set_title('{:.1f} $/\omega_p$'.format((1/(delta))*np.log(2*delta*wp*m*dt*1e+4))) 
            #ax1.set_xlim(xmin/dx,xmax/dx)
            #ax1.set_ylim(ymin/dy,ymax/dy)
            ax1.figure.savefig('fevol-lf100-updated.pdf')
    fig.subplots_adjust(right=0.85)
    #cbar_ax1 = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax1 = fig.add_axes([ax1.get_position().x1+0.025,ax1.get_position().y0,0.015,ax1.get_position().y0/1.75],yscale = 'log')
    cbar_ax1.set_xlabel('MeV$^{-2}$', labelpad=20)
    cbar_ax1.yaxis.set_major_locator(LogLocator(base=10))
    fig.colorbar(im, cax=cbar_ax1, fraction=0.046, pad=0.04, shrink=0.5)

    plt.show()
else:
    print(L)