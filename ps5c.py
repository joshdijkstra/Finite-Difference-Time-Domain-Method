# -*- coding: utf-8 -*-
"""
Created on Mar 7 2020

@author: shugh
"""

# fdtd2d_V1p0.py

"""
2d FDTD Simulation for TM: Ez,Dz,Hy,Hx,
Simple dielectric available, simple update animation
Units for E -> E*sqrt(exp0/mu0), D = eps*E
No PML (perfectly matched layers)
Dipole Source (injected at a single space point)
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20})
import math
import scipy.constants as constants
import timeit
import numba
import seaborn as sns
import matplotlib.ticker as ticker

"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.0001

"Quick and dirty graph to save"
filename = "fdtd_2d_example1.pdf" # save at final time slice (example)

"Basic Geometry and Dielectric Parameters"
printgraph_flag = 0 # print graph to pdf (1)
livegraph_flag = 1 # update graphs on screen every cycle (1)
Xmax = 1610  # no of FDTD cells in x
Ymax = 1610 # no of FDTD cells in x
nsteps = 200 # total number of FDTD time steps
cycle = 80 # for graph updates
Ex = np.zeros((Xmax),float) # E array
Hy = np.zeros((Xmax),float) # H array
EzSnapshots = np.zeros([4,Xmax,Ymax],float)
frames = []

"2d Arrays"
Ez = np.zeros([Xmax,Ymax],float);
Hx = np.zeros([Xmax,Ymax],float); Hy = np.zeros([Xmax,Ymax],float)
Dz = np.zeros([Xmax,Ymax],float); ga=np.ones([Xmax,Ymax],float)
cb = np.zeros([Xmax,Ymax],float) # for spatially varying dielectric constant
EzMonTime1=[]; PulseMonTime=[] # two time-depenet field monitors

c = constants.c # speed of light in vacuum
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
fs = constants.femto # 1.e-15 - useful for pulses
tera = constants.tera # 1.e12 - used for optical frequencues

# dipole source position, atr center just now
isource = int(Ymax/2)
jsource = int(Xmax/2)

"Pulse parameters and points per wavelength"
spread=2.* fs/dt # 2 fs for this example
t0=spread*6
freq_in = 2*math.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 2*math.pi*c/freq_in # near 1.5 microns
eps2 = 9 # dielectric box (so 1 is just free space)
ppw = int(lam/ddx/np.sqrt(eps2)) #  rounded down
print('points per wavelength:',ppw, '(should be > 15)')

# dielectric box coordinates
X1=isource+10; X2=X1+40
Y1=jsource+10; Y2=Y1+40
for j in range (0,Ymax):
    for i in range (0,Xmax):
        if i>X1 and i<X2+1 and j>Y1 and j<Y2+1:
            ga[i,j] = 1./eps2

# an array for x,y spatial points (with first and last points)
xs = np.arange(0,Xmax)
ys = np.arange(0,Ymax)


def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 19})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return

def snapshots():
    # Creates 2x2 snapshots of the animation
    niceFigure()
    fig, axs = plt.subplots(2,2)#sharey = True, sharex = True)

    img1 = axs[0,0].contourf(EzSnapshots[0,:,:])
    axs[0,0].set(title="$frame:"+str(frames[0])+"$",ylabel="$y$",xlabel="$x$")
    axs[0,0].vlines(X1,Y1,Y2,colors='r')
    axs[0,0].vlines(X2,Y1,Y2,colors='r')
    axs[0,0].hlines(Y1,X1,X2,colors='r')
    axs[0,0].hlines(Y2,X1,X2,colors='r')

    img2 = axs[0,1].contourf(EzSnapshots[1,:,:])
    axs[0,1].set(title="$frame:"+str(frames[1])+"$",ylabel="$y$",xlabel="$x$")
    axs[0,1].vlines(X1,Y1,Y2,colors='r')
    axs[0,1].vlines(X2,Y1,Y2,colors='r')
    axs[0,1].hlines(Y1,X1,X2,colors='r')
    axs[0,1].hlines(Y2,X1,X2,colors='r')

    img3 = axs[1,0].contourf(EzSnapshots[2,:,:])
    axs[1,0].set(title="$frame:"+str(frames[2])+"$",ylabel="$y$",xlabel="$x$")
    axs[1,0].vlines(X1,Y1,Y2,colors='r')
    axs[1,0].vlines(X2,Y1,Y2,colors='r')
    axs[1,0].hlines(Y1,X1,X2,colors='r')
    axs[1,0].hlines(Y2,X1,X2,colors='r')

    img4 = axs[1,1].contourf(EzSnapshots[3,:,:])
    axs[1,1].set(title="$frame:"+str(frames[3])+"$",ylabel="$y$",xlabel="$x$")
    axs[1,1].vlines(X1,Y1,Y2,colors='r')
    axs[1,1].vlines(X2,Y1,Y2,colors='r')
    axs[1,1].hlines(Y1,X1,X2,colors='r')
    axs[1,1].hlines(Y2,X1,X2,colors='r')


    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${}\times10^{{{}}}$'.format(a, b)

    font_size = 12
    nbinss = 8
    # Format the colour bars
    cbar1=plt.colorbar(img1, ax=axs[0,0],format=ticker.FuncFormatter(fmt))
    cbar1.set_label('$Ez$')
    cbar1.ax.tick_params(labelsize=font_size)
    cbar2=plt.colorbar(img2, ax=axs[0,1],format=ticker.FuncFormatter(fmt))
    cbar2.set_label('$Ez$')
    cbar2.ax.tick_params(labelsize=font_size)
    cbar3=plt.colorbar(img3, ax=axs[1,0],format=ticker.FuncFormatter(fmt))
    cbar3.set_label('$Ez$')
    cbar3.ax.tick_params(labelsize=font_size)
    cbar4=plt.colorbar(img4, ax=axs[1,1],format=ticker.FuncFormatter(fmt))
    cbar4.set_label('$Ez$')
    cbar4.ax.tick_params(labelsize=font_size)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1.2, hspace=0.70)
    plt.savefig('./Graphs/C-HeatMaps.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

#@numba.jit(nopython=True)
def updateDE(Dz,Hy,Hx,Ez,pulse):
    # Updates D and E fields (not sliced)
    for y in range (1,Ymax-1):
        for x in range (1,Xmax-1):
            Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1])
            Ez[x,y] =  ga[x,y]*(Dz[x,y])
    Dz[isource,jsource] =  Dz[isource,jsource] + pulse # soft source in simulation center
    Ez[isource,jsource] =  ga[isource,jsource]*(Dz[isource,jsource])
    return Dz , Ez

#@numba.jit(nopython=True)
def updateH(Hx,Hy,Ez):
    # Updates the H field (not sliced)
    for x in range (0,Xmax-1):
        for y in range (0,Ymax-1):
            Hx[x,y] = Hx[x,y] + 0.5*(Ez[x,y]-Ez[x,y+1])
            Hy[x,y] = Hy[x,y] + 0.5*(Ez[x+1,y]-Ez[x,y])
    return Hx , Hy

" Main FDTD loop iterated over nsteps"
def FDTD_loop(nsteps,cycle,graph_on):
    # loop over all time steps
    for i in range (0,nsteps):
        t=i-1 # iterate pulse
        pulse = np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))
# calculate Dz (Hy is diff sign to before with Dz term from curl eqs)
        updateDE(Dz,Hy,Hx,Ez,pulse)
# save one point in time just to see the transient
        EzMonTime1.append(Ez[isource,jsource])
        PulseMonTime.append(pulse)
# update H (could also do slicing - but let's make it clear just now)
        updateH(Hx,Hy,Ez)


        if (i == 0 or i == 500 or i == 1000 or i == 1500):
            EzSnapshots[int(i/500),:,:] = Ez
            frames.append(i)

# update graph every cycle
        if graph_on:
            if (i % cycle == 0 and livegraph_flag == 1): # simple animation
                graph(t)
# could also just have graph stuff in here

#@numba.jit(nopython=True)
def updateslicer(Dz,Hy,Hx,Ez,pulse):
    Dz[1:-1,:] -=  0.5*(Hy[:-2,:])
    Dz[:,1:-1] += 0.5*(Hx[:,:-2])
    Dz[:,:] += 0.5*(Hy[:,:]-Hx[:,:])
    Ez[:,:] =  ga[:,:]*Dz[:,:]
    Dz[isource,jsource] =  Dz[isource,jsource] + pulse
    Ez[isource,jsource] =  ga[isource,jsource]*(Dz[isource,jsource])
    #EzMonTime1.append(Ez[isource,jsource])
    #PulseMonTime.append(pulse)
    Hx[:,1:-1] -= 0.5 * Ez[:,2:]
    Hx[:,:] += 0.5 * Ez[:,:]
    Hy[1:-1,:] += 0.5 * Ez[2:,:]
    Hy[:,:] -= 0.5*Ez[:,:]
    return Dz,Hy,Hx,Ez

def FDTD_loop_sliced(nsteps,cycle,graph_on):
    # FDTD loop for sliced vector approach.
    for i in range (0,nsteps):
        t=i-1 # iterate pulse
        pulse = np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))
        updateslicer(Dz,Hy,Hx,Ez,pulse)
        if graph_on:
            if (i % cycle == 0 and livegraph_flag == 1): # simple animation
                graph(t)

def graph(t):

# main graph is E(z,y, time snapshops), and a small graph of E(t) as center
    plt.clf() # close each time for new update graph/colormap
    ax = fig.add_axes([.25, .25, .6, .6])
    #ax2 = fig.add_axes([.015, .8, .15, .15])

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
    img = ax.contourf(Ez)
    cbar=plt.colorbar(img, ax=ax)
    cbar.set_label('$Ez$ (arb. units)')

# add labels to axes
    ax.set_xlabel('Grid Cells ($x$)')
    ax.set_ylabel('Grid Cells ($y$)')

# dielectric box - comment if not using of course (if eps2=1)
    ax.vlines(X1,Y1,Y2,colors='r')
    ax.vlines(X2,Y1,Y2,colors='r')
    ax.hlines(Y1,X1,X2,colors='r')
    ax.hlines(Y2,X1,X2,colors='r')

# add title with current simulation time step
    ax.set_title("frame time {}".format(t))


# Small graph to see time development as a single point
    PulseNorm = np.asarray(PulseMonTime)*0.2;
    #ax2.plot(PulseNorm,'r',linewidth=1.6)
    #ax2.plot(EzMonTime1,'b',linewidth=1.6)
    #ax2.set_yticklabels([])
    #ax2.set_xticklabels([])
#    ax2.set_title('$E_{in}(t)$')

    plt.pause(time_pause) # pause sensible value to watch what is happening

#%%
# set figure for graphics output
if livegraph_flag==1:
    fig = plt.figure(figsize=(8,6))


graph_on = False # Toggles animation.
"Main FDTD: time steps = nsteps, cycle for very simple animation"
start = timeit.default_timer()
FDTD_loop(nsteps,cycle,graph_on)
#snapshots()
stop = timeit.default_timer()
print ("Time for total simulation Base/Numba", stop - start)

#Reset Variables
Ez = np.zeros([Xmax,Ymax],float);
Hx = np.zeros([Xmax,Ymax],float); Hy = np.zeros([Xmax,Ymax],float)
Dz = np.zeros([Xmax,Ymax],float);
EzMonTime1=[]; PulseMonTime=[] # two time-depenet field monitors

start = timeit.default_timer()
FDTD_loop_sliced(nsteps,cycle,graph_on)
stop = timeit.default_timer()
print ("Time for total simulation Sliced/ Sliced + Numba: ", stop - start)
print("To view simulation time for Numba  uncomment the numba.jit above the function updateslicer()")
"Save Last Slice - adjust as you like"
if printgraph_flag == 1:
    plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')
