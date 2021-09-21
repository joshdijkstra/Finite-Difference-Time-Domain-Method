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

from mpi4py import MPI
import numpy as np
import time as tm
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20})
import math
import scipy.constants as constants
import timeit
import numba
import sys

def timestamp() :
    print ( "Local time : ", tm.ctime( tm.time() ) )
    return

comm=MPI.COMM_WORLD
id=comm.Get_rank()
p=comm.Get_size()
print("RANK: "+ str(p))

if (id==0) :
    timestamp()
    print( "" )
    print( "FDTD_MPI_2D:" )
    print( "  Python/MPI version" )

if (id==0) :
    wtime = MPI.Wtime()


"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.01

"Quick and dirty graph to save"
filename = "fdtd_2d_example1.pdf" # save at final time slice (example)

"Basic Geometry and Dielectric Parameters"
printgraph_flag = 0 # print graph to pdf (1)
livegraph_flag = 1 # update graphs on screen every cycle (1)
Xmax = 160  # no of FDTD cells in x
Ymax = 160 # no of FDTD cells in x
nsteps = 2000 # total number of FDTD time steps
cycle = 80 # for graph updates
Ex = np.zeros((Xmax),float) # E array
Hy = np.zeros((Xmax),float) # H array

"2d Arrays"
n = int(Xmax/p)
print(n)

if id==0:
    EzMaster = np.zeros([Xmax,Ymax],float)

Ez = np.zeros([(n+2),Ymax],float);
Hx = np.zeros([(n+2),Ymax],float); Hy = np.zeros([(n+2),Ymax],float)
Dz = np.zeros([(n+2),Ymax],float); ga=np.ones([(n),Ymax],float)
cb = np.zeros([n,Ymax],float) # for spatially varying dielectric constant
EzMonTime1=[]; PulseMonTime=[] # two time-depenet field monitors

c = constants.c # speed of light in vacuum
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
fs = constants.femto # 1.e-15 - useful for pulses
tera = constants.tera # 1.e12 - used for optical frequencues

# dipole source position, atr center just now
isource = int((Ymax/2)+2)
jsource = int((Xmax/2)+2)

"Pulse parameters and points per wavelength"
spread=2.* fs/dt # 2 fs for this example
t0=spread*6
freq_in = 2*math.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 2*math.pi*c/freq_in # near 1.5 microns
eps2 = 3 # dielectric box (so 1 is just free space)
ppw = int(lam/ddx/eps2) #  rounded down
print('points per wavelength:',ppw, '(should be > 15)')

# dielectric box coordinates
X1=isource+10; X2=X1+40
Y1=jsource+10; Y2=Y1+40

for j in range (0,Ymax):
    for i in range (0,n):
        if i+((Xmax*id)/p)>X1 and i+((Xmax*id)/p)<X2+1 and j>Y1 and j<Y2+1:
            ga[i,j] = 1./eps2

# an array for x,y spatial points (with first and last points)
xs = np.arange(0,Xmax)
ys = np.arange(0,Ymax)

#@numba.jit(nopython=True)
def updateDE(Dz,Hy,Hx,Ez,pulse):
    # isource jsource conditions
    if id == 0:
        for y in range (1,Ymax-1):
            for x in range (1,n):
                x2 = x + (Xmax*id)/p
                Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1])
                Ez[x,y] =  ga[x,y]*(Dz[x,y])
        if isource >= (Xmax*id/p) and isource <= (n)*(id+1):
            isourcemod = isource % (n)
            Dz[isourcemod,jsource] =  Dz[isourcemod,jsource] + pulse # soft source in simulation center
            Ez[isourcemod,jsource] =  ga[isourcemod,jsource]*(Dz[isourcemod,jsource])
        return Dz , Ez
    if id == p:
        for y in range (1,Ymax-1):
            for x in range(1,n -1):
                x2 = x + (Xmax*id)/p
                Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1])
                Ez[x,y] =  ga[x,y]*(Dz[x,y])
        if isource >= (Xmax*id/p) and isource <= (n)*(id+1):
            isourcemod = isource % (n)
            Dz[isourcemod,jsource] =  Dz[isourcemod,jsource] + pulse # soft source in simulation center
            Ez[isourcemod,jsource] =  ga[isourcemod,jsource]*(Dz[isourcemod,jsource])
        return Dz , Ez
    else:
        for y in range (1,Ymax-1):
            for x in range(1,n):
                x2 = x + (Xmax*id)/p
                Dz[x,y] =  Dz[x,y] + 0.5*(Hy[x,y]-Hy[x-1,y]-Hx[x,y]+Hx[x,y-1])
                Ez[x,y] =  ga[x,y]*(Dz[x,y])
        if isource >= (Xmax*id/p) and isource <= (n)*(id+1):
            isourcemod = isource % (n)
            Dz[isourcemod,jsource] =  Dz[isourcemod,jsource] + pulse # soft source in simulation center
            Ez[isourcemod,jsource] =  ga[isourcemod,jsource]*(Dz[isourcemod,jsource])
        return Dz , Ez

#@numba.jit(nopython=True)
def updateH(Hx,Hy,Ez):
    if id == 0:
        for x in range (1,n):
            for y in range (0,Ymax-1):
                Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])
                Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y])
        return Hx , Hy
    if id == p:
        for x in range (0,n-1):
            for y in range (0,Ymax-1):
                Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])
                Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y])
        return Hx , Hy
    else:
        for x in range (1,n):
            for y in range (0,Ymax-1):
                Hx[x,y] = Hx[x,y]+ 0.5*(Ez[x,y]-Ez[x,y+1])
                Hy[x,y] = Hy[x,y]+ 0.5*(Ez[x+1,y]-Ez[x,y])
        return Hx , Hy
" Main FDTD loop iterated over nsteps"
def FDTD_loop(nsteps,cycle,id):
    # loop over all time steps
    print("Running Loop")
    for i in range (0,nsteps):
        up  = int(id-1)
        down = int(id+1)
        if p >= 2:
            if ( id != 0 and id != p-1) : # All processors apart from id = 0
                """For Ez"""
                data = Ez[1][:]
                comm.send(data, dest=up, tag=1) # send data up
                rec = comm.recv(source=up, tag=2) # Receive from below
                Ez[0][:] = rec
                data = Ez[-2][:]
                comm.send(data,dest=down, tag=2) # send data up
                rec = comm.recv(source=down, tag=1) # Receive from below
                Ez[-1][:] = rec

                """For Dz"""
                data = Dz[1][:]
                comm.send(data, dest=up, tag=3) # send data up
                rec = comm.recv(source=up, tag=4) # Receive from below
                Dz[0][:] = rec
                data = Dz[-2][:]
                comm.send(data,dest=down, tag=4) # send data up
                rec = comm.recv(source=down, tag=3) # Receive from below
                Dz[-1][:] = rec

                """For Hy"""
                data = Hy[1][:]
                comm.send(data, dest=up, tag=5) # send data up
                rec = comm.recv(source=up, tag=6) # Receive from below
                Hy[0][:] = rec
                data = Hy[-2][:]
                comm.send(data,dest=down, tag=6) # send data up
                rec = comm.recv(source=down, tag=5) # Receive from below
                Hy[-1][:] = rec

                """For Hx"""
                data = Hx[1][:]
                comm.send(data, dest=up, tag=7) # send data up
                rec = comm.recv(source=up, tag=8) # Receive from below
                Hx[0][:] = rec
                data = Hx[-2][:]
                comm.send(data,dest=down, tag=8) # send data up
                rec = comm.recv(source=down, tag=7) # Receive from below
                Hx[-1][:] = rec

            if ( id == 0 ) : # All processors apart from last
                """For Ez"""
                data = Ez[-3][:]
                comm.send(data, dest=down, tag=2) # Send data down
                rec = comm.recv(source=up, tag=1) # Receive from below
                Ez[-2][:] = rec

                """For Dz"""
                data = Dz[-3][:]
                comm.send(data, dest=down, tag=4) # Send data down
                rec = comm.recv(source=up, tag=3) # Receive from below
                Dz[-2][:] = rec

                """For Hx"""
                data = Hx[-3][:]
                comm.send(data, dest=down, tag=6) # Send data down
                rec = comm.recv(source=up, tag=5) # Receive from below
                Hx[-2][:] = rec

                """For Hy"""
                data = Hy[-3][:]
                comm.send(data, dest=down, tag=8) # Send data down
                rec = comm.recv(source=up, tag=7) # Receive from below
                Hy[-2][:] = rec

            if ( id == p-1 ) : # All processors apart from id == 0
                """For Ez"""
                data = Ez[1][:]
                comm.send(data, dest=up, tag=1) # send data up
                rec = comm.recv(source=up, tag=2) # Receive from below
                Ez[0][:] = rec

                """For Dz"""
                data = Dz[1][:]
                comm.send(data, dest=up, tag=3) # send data up
                rec = comm.recv(source=up, tag=4) # Receive from below
                Dz[0][:] = rec

                """For Hy"""
                data = Hy[1][:]
                comm.send(data, dest=up, tag=5) # send data up
                rec = comm.recv(source=up, tag=6) # Receive from below
                Hy[0][:] = rec

                """For Hx"""
                data = Hx[1][:]
                comm.send(data, dest=up, tag=7) # send data up
                rec = comm.recv(source=up, tag=8) # Receive from below
                Hx[0][:] = rec


        t=i-1 # iterate pulse
        pulse = np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))
        updateDE(Dz,Hy,Hx,Ez,pulse)
        """
        EzMonTime1.append(Ez[isource,jsource])
        PulseMonTime.append(pulse)
        """
        updateH(Hx,Hy,Ez)

        EzMaster = comm.gather(Ez, root = 0)


        if (id == 0):
            if (i % cycle == 0 and livegraph_flag == 1): # simple animation
                graph(t)
# could also just have graph stuff in here


def graph(t):
# main graph is E(z,y, time snapshops), and a small graph of E(t) as center
    plt.clf() # close each time for new update graph/colormap
    ax = fig.add_axes([.25, .25, .6, .6])
    #ax2 = fig.add_axes([.015, .8, .15, .15])

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
    img = ax.contourf(EzMaster)
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
    """
    PulseNorm = np.asarray(PulseMonTime)*0.2;
    ax2.plot(PulseNorm,'r',linewidth=1.6)
    ax2.plot(EzMonTime1,'b',linewidth=1.6)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    """
#    ax2.set_title('$E_{in}(t)$')

    plt.pause(time_pause) # pause sensible value to watch what is happening

#%%
# set figure for graphics output
if livegraph_flag==1:
    fig = plt.figure(figsize=(8,6))

"Main FDTD: time steps = nsteps, cycle for very simple animation"
start = timeit.default_timer()
FDTD_loop(nsteps,cycle,id)
stop = timeit.default_timer()
print ("Time for total simulation", stop - start)

"Save Last Slice - adjust as you like"
if printgraph_flag == 1:
    plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')
