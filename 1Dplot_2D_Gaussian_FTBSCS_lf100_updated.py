#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:56:37 2022

@author: admin
"""


import scipy
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.signal import find_peaks
from scipy.signal import find_peaks_cwt
from scipy.signal import peak_widths
from scipy.signal import peak_prominences
from scipy.interpolate import CubicSpline

#instability details
e = 0.3
me = 0.5
epsilon = 0.5
alpha = 1e-4
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
#print('dt=',dt)

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
mfig = [1500, 2000, 2500, 3000]
fignum = 0
fig = plt.figure()
if L**2 <= L + 2*D <=1: #stability in FTBSCS
    with open("mustdfinal.txt", "a+") as file:
        for m in range(nsteps):
            u0, u = do_timestep(u0, u)
            if m in mfig:
                fignum += 1
                print(m, fignum)
                ax1 = fig.add_subplot(140 + fignum)
                imageraw = u.copy().T
                image = imageraw/(np.linalg.norm(imageraw))
                #meanx = image[49].sum(0)/len(image[49])
                #print(meanx)
                #im = ax1.imshow(image[:][50:99], cmap=plt.get_cmap('hot'), vmin=Noparticle,vmax=Allparticles)    
                im = ax1.plot(image[50])
                ax1.get_window_extent([0,100])
                tdat = (1/(delta))*np.log(2*delta*wp*m*dt*1e+4)
                #print(tdat)
                #peakloc, heights_peak_0 = find_peaks(image[50], height=0)
                #peakheight = heights_peak_0['peak_heights']
                # Find the indices of non-zero elements in the image array
                #image_height, image_width = im.shape
                peaks, properties = find_peaks(image[50], height=0)
                x_coordinates = np.arange(len(image[50]))
                y_coordinates = image[50][peaks]
                mux = x_coordinates[peaks]
                print(mux)
                #spline = CubicSpline(x_coordinates, y_coordinates)
                #derivative_zeros = spline.derivative().roots()
                #print(derivative_zeros)
                #peakloc = properties["peak_position"]
                #x_coordinate = np.arange(image_width)
                #y_coordinate = image[50][peakloc]
                #print(x_coordinate, y_coordinate)
                #prominences = peak_prominences(image[50], peakloc)[0]
                #highest_prominence_index = np.argmax(prominences)
                #print(highest_prominence_index)
                #print(peakheight)
                #prominences = peak_prominences(image[50], peakloc)[0]
                #print(prominences)
                peakwidth = peak_widths(image[50],rel_height = 0.5, peaks = peaks)
                stdx = peakwidth[0]/2
                #print(stdx) 
                #prominences = peak_prominences(image[50], peakloc)[0]
                #print(prominences)
                #peakid = np.argmax(prominences)
                #print(peakid)
                stddat = np.squeeze(stdx).tolist() #squeeze to remove square brackets, tolist to convert array to list
                #mudat = np.squeeze(image[50][peakloc]).tolist()
                mudat = np.squeeze(mux).tolist()

                #if isinstance(peakdat,int):
                    #print('Peakdatttt: ' + str(isinstance(peakdat, int)))
                #    mudat = peakdat
                #    stddat = stdxdat
                #else:
                #    #print('blabla')
                #    mudat = peakdat[peakid]
                #    stddat = stdxdat[peakid]
                #peakgl = peakdat(np.where(prominences == max(prominences)))
                data = [m, tdat, stddat, mudat]
                #print('blablerfa')
                print(data)
                #file.write('\n')
                file.write(' '.join(str(x) for x in data)+'\n')
                ax1.set_xlabel('$p_{\mid \mid}$')
                ax1.set_ylabel('$f(\mathbf{p})$')
                    #ax1.set_xlim(0,100)
                ax1.set_ylim(0,0.5)
                ax1.set_title('{:.1f} $/\omega_p$'.format((1/(delta))*np.log(2*delta*wp*m*dt*1e+4)))
                ax1.minorticks_on()
                    #ax.set_xlim(0,50)
                    #ax.set_ylim(0,50)
                ax1.figure.savefig('fpx1D-lf100.pdf')
            fig.subplots_adjust(right=0.85)
else:
    print(L)