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

"Basic Geometry and Dielectric Parameters"
graph_flag = 0 # 0 will not graphics save
Xmax = 801  # no of FDTD cells in x
nsteps = 10000 # number of FDTD tiem steps
cycle = 100 # for graph updates


c = constants.c # speed of light in vacuum
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
tlist = np.arange(0,nsteps+1,1)
wp = 1.26E15
alph = 1.4E14




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

def pulsefun(t):
     pulse = -np.exp(-0.5*(t-t0)**2/spread**2)\
     *(np.cos(t*w_scale))
     return pulse
#Boundary conditions
" Main FDTD loop iterated iter_t times"
def FDTD_loop(nsteps,cycle,L,Lorentz=False):

    # loop over all time steps
    for i in range (0,nsteps+1): # time loop, from 0 to nsteps
       t=i-1 # iterative time dep pulse as source
       pulse = pulsefun(t)
       pulsedt = pulsefun(t+0.5)

# update E
       for x in range (1,Xmax-1):
           Dx[x] = Dx[x] + 0.5 * (Hy[x-1]-Hy[x]) # Added displacement field
       Dx[isource] = Dx[isource] - 0.5 * pulsedt
       if Lorentz:
           # If a lorentzian dispersion model
           f0 = 0.05
           alph = 4 * math.pi * 1E12
           w0 = math.pi * 2 * 200E12
           beta = math.sqrt(w0**2-alph**2)
           Sn[:,i] = 2 * math.exp(-alph*dt) * math.cos(beta*dt) * Sn[:,i-1] - math.exp(-2*alph*dt)* Sn[:,i-2] + dt*w0**2*(f0)*(1/beta)*math.exp(-alph*dt)* math.sin(beta*dt) * Exs[:,i-1]
       else:
           # Drude dispersion model
           alph = 1.4E14
           wp = 1.26E15
           Sn[:,i] = (1+math.exp(-alph*dt))*Sn[:,i-1] - math.exp(-alph*dt)*Sn[:,i-2] + (dt*wp**2)/(alph)*(1- math.exp(-alph*dt)) * Exs[:,i-1]

       for x in range(1,Xmax-1):
           Ex[x] =  Dx[x]
           if (x >= start and x <= start + L):
               Ex[x] = Dx[x] - Sn[x,i]
           if i > 2:
               Ex[0] = Exs[1,i-2]
               Ex[Xmax-2] = Exs[Xmax-3,i-2]
       Hy[isource-1] = Hy[isource-1] - 0.5 * pulse

       for x in range (0,Xmax-1):
           Hy[x] = Hy[x] + (dt*c/(ddx))*(Ex[x]-Ex[x+1])

# update graph every cycle

       if (i % cycle == 0): # simple animation
           im.set_ydata(Ex[1:Xmax-1])
           ax.set_title("frame time {}".format(i))
           plt.pause(time_pause) # sinsible pause to watch animation

    # Transmission and reflection recording
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

"Main FDTD: time steps = nsteps, cycle for very simple animation"
Ls = [5,25,50,100]
Lan =  [100E-9, 500E-9, 1E-6, 2E-6]
LsTitle = [0.1,0.5,1,2]
niceFigure()

for j in range(len(Ls)):
    # Make true for Lorentzian dispersion model, Drude if False
    Lorentz = False

    plottitle = "$"+str(LsTitle[j]) + "\mu$"
    title = "B-" +str(Ls[j])
    # Reset all the fields each run
    Ex = np.zeros((Xmax),float)# E array
    Dx = np.zeros((Xmax),float)
    Sn = np.zeros((Xmax,nsteps+1),float)
    Exs = np.zeros((Xmax,nsteps+1),float)
    Hy = np.zeros((Xmax),float) # H array
    e = np.ones((Xmax),float)
    R = np.zeros((nsteps+1),float)
    T = np.zeros((nsteps+1),float)
    # Reflection Transmission Positions
    Rpos = 5
    Tpos = Xmax - 5
    start = int((Xmax-1)/2)
    end = start + Ls[j]

    "Define first (only in this simple example) graph for updating Ex at varios times"
    lw=2 # line thickness for graphs
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    [im] = ax.plot(xs,Ex[1:Xmax-1],linewidth=lw)
    init1() # initialize, then we will just update the y data and title frame

    FDTD_loop(nsteps,cycle,Ls[j],Lorentz)

    pulses = []
    for x in range(nsteps+1):
        pulses.append(pulsefun(tlist[x]))

    def reftran():
        plt.clf()
        # Plots the graph of the three waveforms: incident, reflected, transmitted
        plt.plot(tlist,pulses,label="$E_{in}$",color="b") # incident wave
        plt.plot(tlist,R,label="$E_r$",color="orange") # Reflected wave
        plt.plot(tlist,T,label="$E_t$",color="r") # Transmitted
        plt.xlim(0,4000)
        plt.legend()
        plt.xlabel("$Time (\Delta t)$")
        plt.ylabel("$E_x$")
        plt.title(plottitle)
        if Lorentz:
            plt.savefig('./Graphs/'+title+'ReflectTransmit.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
        else:
            plt.savefig('./Graphs/'+title+'ReflectTransmitDrude.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
        plt.show()

    def fouriertrans():
        # Plots the fourier transform
        ftEt = np.fft.fftshift(np.abs(np.fft.fft(T)))
        ftEr = np.fft.fftshift(np.abs(np.fft.fft(R)))
        ftEin = np.fft.fftshift(np.abs(np.fft.fft(pulses)))

        Tf = np.abs(ftEt/ftEin)**2
        Rf = np.abs(ftEr/ftEin)**2

        w = np.fft.fftshift(np.fft.fftfreq(nsteps+1,dt))/tera

        plt.plot(w,Rf,label="$R$",color="r")
        plt.plot(w,Tf,label="$T$",color="b")
        plt.plot(w,Rf+Tf,label="$Sum$",color="black")
        plt.legend(loc=1)
        plt.xlabel("$\omega/2\pi (THz)$")
        plt.ylabel("$R,T$")
        plt.xlim(100,300)
        plt.ylim(0,1.1)
        plt.title(plottitle)
        if Lorentz:
            plt.savefig('./Graphs/'+title+'FourierTransform.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
        else:
            plt.savefig('./Graphs/'+title+'FourierTransformDrude.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
        plt.show()
        return w , Rf, Tf

    def analytical(w,L):
        # Creates the analytical solution values
        alph = 1.4E14
        wp = 1.26E15
        w= 2*np.pi*(np.fft.fftshift(np.fft.fftfreq(nsteps+1,dt)))
        k_0 = (w*tera*np.pi*2)/c
        n = np.zeros((nsteps+1),complex)
        r1 = np.zeros((nsteps+1),complex)
        r2 = np.zeros((nsteps+1),complex)
        k0 = np.zeros((nsteps+1),complex)
        e = np.zeros((nsteps+1),complex)
        e2 = np.zeros((nsteps+1),complex)
        r = np.zeros((nsteps+1),complex)
        t = np.zeros((nsteps+1),complex)

        for x in range(len(w)):
            # From equations in the notes
            n[x] = cmath.sqrt(1-(wp**2)/(w[x]**2+1j*w[x]*alph))
            r1[x] = (1-n[x])/(1+n[x])
            r2[x] = (n[x]-1)/(n[x]+1)
            k0[x] = w[x]/c
            e[x] = np.exp(2*1j*k0[x]*L*n[x])
            e2[x] = np.exp(1j*k0[x]*L*n[x])
            r[x] = (r1[x] + r2[x]*e[x])/(1+r1[x]*r2[x]*e[x])
            t[x] = ((1+r1[x])*(1+r2[x])*e2[x])/(1+r1[x]*r2[x]*e[x])
        Ra = np.abs(r)**2
        Ta = np.abs(t)**2
        return w , Ra , Ta

    def analyticalLorentz(w,L):
        # Finds the analytical solution to the Lorentz dispersion
        w= 2*np.pi*(np.fft.fftshift(np.fft.fftfreq(nsteps+1,dt)))
        f0 = 0.05
        alpha = 4 * math.pi * 1E12
        w0 = math.pi * 2 * 200E12
        n = np.zeros((nsteps+1),complex)
        r1 = np.zeros((nsteps+1),complex)
        r2 = np.zeros((nsteps+1),complex)
        k0 = np.zeros((nsteps+1),complex)
        e = np.zeros((nsteps+1),complex)
        e2 = np.zeros((nsteps+1),complex)
        r = np.zeros((nsteps+1),complex)
        t = np.zeros((nsteps+1),complex)
        for x in range(len(w)):
            n[x]= cmath.sqrt(1. + f0*w0**2/(w0**2 - w[x]**2 - 1j*2*w[x]*alpha))
            r1[x] = (1-n[x])/(1+n[x])
            r2[x] = (n[x]-1)/(n[x]+1)
            k0[x] = w[x]/c
            e[x] = np.exp(2*1j*k0[x]*L*n[x])
            e2[x] = np.exp(1j*k0[x]*L*n[x])
            r[x] = (r1[x] + r2[x]*e[x])/(1+r1[x]*r2[x]*e[x])
            t[x] = ((1+r1[x])*(1+r2[x])*e2[x])/(1+r1[x]*r2[x]*e[x])
        Ra = np.abs(r)**2
        Ta = np.abs(t)**2
        return w, Ra, Ta

    reftran()
    w , Rf , Tf = fouriertrans()

    if Lorentz:
        wan , Ra , Ta = analyticalLorentz(w,Lan[j])
    else:
        wan , Ra , Ta = analytical(w,Lan[j])

    def fftan():
        # Plots fourier transform and the analytical solution together.
        freq= (np.fft.fftshift(np.fft.fftfreq(nsteps+1,dt)))/tera
        plt.plot(freq,Rf,label="$R$",color="r")
        plt.plot(freq,Tf, label = "$T$", color="b")
        plt.plot(freq,Ra, label = "$R_{an}$",linestyle="--", color = "r")
        plt.plot(freq,Ta, label = "$T_{an}$",linestyle="--", color = "b")
        plt.xlim(150,250)
        plt.xlabel("$\omega/2\pi (THz)$")
        plt.ylabel("$R,T$")
        plt.legend(loc=1)
        plt.title(plottitle)
        plt.ylim(0,1.1)
        if Lorentz:
            plt.savefig('./Graphs/'+title+'AnalyticalSol.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
        else:
            plt.savefig('./Graphs/'+title+'AnalyticalSolDrude.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
        plt.show()
    fftan()
