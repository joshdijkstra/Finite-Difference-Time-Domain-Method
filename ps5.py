# Josh Dykstra Assignment 5
# -*- coding: utf-8 -*-
"""
Created on Feb 20 2010

@author: shugh
"""

# fdtd_V1p0.py

"""
1d FDTD Simulation for Ex nd Hy, simple dielectric, simple animation
Units for E -> E*sqrt(exp0/mu0), so E and H are comparible (and same for a plane wave)
No absorbing BC (ABC) - so will bounce off walls
Source (injected at a single space point) goes in both directions
"""

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20})
#from matplotlib import animation
import math
import cmath
import scipy.constants as constants

"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.001

"Quick and durty graph to save"
filename = "fdtd_1d.pdf" # save at final time slice (example)

"Basic Geometry and Dielectric Parameters"
graph_flag = 0 # 0 will not graphics save
Xmax = 801  # no of FDTD cells in x
nsteps = 7000 # number of FDTD tiem steps
cycle = 100 # for graph updates
Ex = np.zeros((Xmax),float)# E array
Exs = np.zeros((Xmax,nsteps+1),float)
Hy = np.zeros((Xmax),float) # H array
e = np.ones((Xmax),float)
R = np.zeros((nsteps+1),float)
T = np.zeros((nsteps+1),float)
#ns =  np.zeros((Xmax),float)
#r1 =  np.zeros((Xmax),float)
#r2 =  np.zeros((Xmax),float)


ExSnapshots = []
c = constants.c # speed of light in vacuum
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
tlist = np.arange(0,nsteps+1,1)
L = 50

fs = constants.femto # 1.e-15 - useful for pulses
tera = constants.tera # 1.e12 - used for optical frequencues

# source positions
isource = 100

"Pulse parameters and points per wavelength"
spread=2.* fs/dt # 2 fs for this example
X1=int(Xmax/2) # center position
t0=spread*6
freq_in = 2*math.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 2*math.pi*c/freq_in # near 1.5 microns
ppw = int(lam/ddx) # will round down
print('points per wavelength',ppw, 'should be > 15')

# an array for spatial points (without first and last points)
xs = np.arange(1,Xmax-1)

# initial time
t=0

#dielectric
start = int((Xmax-1)/2)
end = start + L
for j in range (start,end):
    e[j] = 9

# Reflection Transmission Positions
Rpos = 5
Tpos = Xmax - 5

def niceFigure(useLatex=True):
    # Plot function that creates latex style graphs
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

def snapshots(ExSnapshots,xs):
    niceFigure()
    # Function that creates the 2x2 snapshots of the animation
    frames = [300,400,500,700]
    sf = 50
    xs = xs /sf
    fig, axs = plt.subplots(2,2)#sharey = True, sharex = True)
    axs[0,0].set(ylim=(-0.7,0.7),xlim=(0, (Xmax-1)/sf),title="$frame:"+str(frames[0])+"$",ylabel="$E_x (V/m)$",xlabel="$z (\mu m)$")
    axs[0,0].plot(xs,Exs[1:Xmax-1,frames[0]])
    axs[0,1].plot(xs,Exs[1:Xmax-1,frames[1]])
    axs[0,1].set(ylim=(-0.7,0.7),xlim=(0, (Xmax-1)/sf),title="$frame:"+str(frames[1])+"$",ylabel="$E_x (V/m)$",xlabel="$z (\mu m)$")
    axs[1,0].plot(xs,Exs[1:Xmax-1,frames[2]])
    axs[1,0].set(ylim=(-0.7,0.7),xlim=(0, (Xmax-1)/sf),title="$frame:"+str(frames[2])+"$",ylabel="$E_x (V/m)$",xlabel="$z (\mu m)$")
    axs[1,1].plot(xs,Exs[1:Xmax-1,frames[3]])
    axs[1,1].set(ylim=(-0.7,0.7),xlim=(0, (Xmax-1)/sf),title="$frame:"+str(frames[3])+"$",ylabel="$E_x (V/m)$",xlabel="$z (\mu m)$")

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.70)
    plt.savefig('./Graphs/A-SnapshotsScatter.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def pulsefun(t):
    # Injected pulse function
     pulse = -np.exp(-0.5*(t-t0)**2/spread**2)\
     *(np.cos(t*w_scale))
     return pulse
#Boundary conditions
" Main FDTD loop iterated iter_t times"
def FDTD_loop(nsteps,cycle):
    # loop over all time steps
    for i in range (0,nsteps+1): # time loop, from 0 to nsteps
       t=i-1 # iterative time dep pulse as source
       pulse = pulsefun(t)
       pulsedt = pulsefun(t+0.5)

# update E
       for x in range (1,Xmax-2):
           #Ex[x] = Ex[x] + (c*dt/ddx)*(Hy[x-1]-Hy[x])      # without dielectric
           Ex[x] = Ex[x] + (dt*c/(ddx*e[x]))*(Hy[x-1]-Hy[x])  # with dielectric
           Exs[x,i] = Ex[x]

           if i > 2:
               # ABC conditions
               Ex[0] = Exs[1,i-2]
               Ex[Xmax-2] = Exs[Xmax-3,i-2]

       #Ex[isource] = Ex[isource] - pulse*0.5

    #Part (b) - Forward wave only
       Ex[isource] = Ex[isource] - pulsedt*0.5      # Scatter field approx
       Hy[isource-1] = Hy[isource-1] - 0.5 * pulse


       if (i == 0 or i == 500 or i == 1000 or i == 2000):
           # Creates snapshots of the animation at frames 0, 500, 1000, 2000
           ExSnapshots.append(Ex[1:Xmax-1])

       for x in range (0,Xmax-1):
           Hy[x] = Hy[x] + (dt*c/(ddx))*(Ex[x]-Ex[x+1]) #

# update graph every cycle

       if (i % cycle == 0): # simple animation
           im.set_ydata(Ex[1:Xmax-1])
           ax.set_title("frame time {}".format(i))
           plt.pause(time_pause) # sinsible pause to watch animation

       # Record transmission and reflection amplitudes
       Exs[:,i] = Ex
       R[i] = Ex[Rpos]
       T[i] = Ex[Tpos]

# initialize graph, fix scalign for thsi first example
def init1():
    plt.ylim((-1, 1))
    plt.xlim((0, Xmax-1))
    plt.axvline(x=start,color='r') # Vert line separator
    plt.axvline(x=end,color='r')
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    #plt.show()

#%%
"Define first (only in this simple example) graph for updating Ex at varios times"
lw=2 # line thickness for graphs
fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([.18, .18, .7, .7])
[im] = ax.plot(xs,Ex[1:Xmax-1],linewidth=lw)
init1() # initialize, then we will just update the y data and title frame

"Main FDTD: time steps = nsteps, cycle for very simple animation"
FDTD_loop(nsteps,cycle)
#snapshots(ExSnapshots,xs)

"Save Last Slice"
if graph_flag == 1:
    plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')
plt.clf()

pulses = []
for x in range(nsteps+1):
    pulses.append(pulsefun(tlist[x]))

def reftran():
    # Plots the graph of the three waveforms: incident, reflected, transmitted
    niceFigure()
    plt.plot(tlist,pulses,label="$E_{in}$",color="b") # incident wave
    plt.plot(tlist,R,label="$E_r$",color="orange") # Reflected wave
    plt.plot(tlist,T,label="$E_t$",color="r") # Transmitted
    plt.legend(loc=1)
    plt.xlabel("$Time (\Delta t)$")
    plt.ylabel("$E_x$")
    plt.xlim(0,4000)
    #plt.title("Reflected and Transmitted against Time")
    plt.savefig('./Graphs/A-ReflectTransmit.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

def fouriertrans():
    # Plots the fourier transform
    ftEt = np.fft.fftshift(np.abs(np.fft.fft(T)))
    ftEr = np.fft.fftshift(np.abs(np.fft.fft(R)))
    ftEin = np.fft.fftshift(np.abs(np.fft.fft(pulses)))

    Tf = np.abs(ftEt/ftEin)**2
    Rf = np.abs(ftEr/ftEin)**2

    w = np.fft.fftshift(np.fft.fftfreq(nsteps+1,dt))/tera
    niceFigure()

    plt.plot(w,Rf,label="$R$")
    plt.plot(w,Tf,label="$T$")
    plt.plot(w,Rf+Tf,label="$Sum$")
    plt.legend(loc=1)
    plt.xlabel("$\omega/2\pi (THz)$")
    plt.ylabel("$R,T$")
    #plt.title("Fourier Transforms")
    plt.xlim(100,300)
    plt.ylim(0,1.1)
    plt.savefig('./Graphs/A-FourierTransform.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()
    return w , Rf, Tf

def analytical(w):
    # Creates the analytical solution values
    L = 1E-6 # Length of dielectric
    k_0 = (w*tera*np.pi*2)/c
    n = 3
    rw =  np.zeros((nsteps+1),float)
    tw =  np.zeros((nsteps+1),float)
    #n = math.sqrt(e[i])
    r1 = (1-n)/(1+n)
    r2 = (n-1)/(n+1)
    # From equations in the notes
    for i in range(len(w)):
        rw[i] = np.abs((r1+r2*cmath.exp(2*cmath.sqrt(-1)*k_0[i]*L*n))
            /(1+r1*r2*cmath.exp(2*cmath.sqrt(-1)*k_0[i]*L*n)))**2
        tw[i] = np.abs(((1+r1)*(1+r2)*cmath.exp(cmath.sqrt(-1)*k_0[i]*L*n))
            /((1+r1*r2*cmath.exp(2*cmath.sqrt(-1)*k_0[i]*L*n))))**2
    return w , rw , tw

reftran()
w , Rf , Tf = fouriertrans()
wan , Ra , Ta = analytical(w)
#w = w / tera
def fftan():
    # Plots the fourier transforms and the analytical solutions together
    niceFigure()
    plt.plot(w,Rf,label="$R$",color="r")
    plt.plot(w,Tf, label = "$T$", color="b")
    plt.plot(w,Rf+Tf,label="$Sum$")
    plt.plot(wan,Ra, label = "$R_{an}$",linestyle="--", color = "r")
    plt.plot(wan,Ta, label = "$T_{an}$",linestyle="--", color = "b")
    plt.xlim(150,250)
    plt.xlabel("$\omega/2\pi (THz)$")
    plt.ylabel("$R,T$")
    plt.legend(loc=1)
    plt.ylim(0,1.1)
    plt.savefig('./Graphs/A-AnalyticalSol.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()
fftan()
