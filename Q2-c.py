# -*- coding: utf-8 -*-

#fdtd_V1p0.py

"""
1d FDTD Simulation for Ex nd Hy, simple dielectric, simple animation
Units for E -> E*sqrt(exp0/mu0), so E and H are comparible (and same for a plane wave)
No absorbing BC (ABC) - so will bounce off walls
Source (injected at a single space point) goes in both directions
"""

import numpy as np
import cmath
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20})
from matplotlib import animation
import math
import scipy.constants as constants

"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.005

"Quick and durty graph to save"
filename = "fdtd_1d.pdf" # save at final time slice (example)

"Basic Geometry and Dielectric Parameters"
graph_flag = 0 # 0 will not graphics save
Xmax = 801  # no of FDTD cells in x
nsteps = 8000 # number of FDTD tiem steps
cycle = 100 # for graph updates


Ex = np.zeros((Xmax),float) # E array
Hy = np.zeros((Xmax),float) # H array
Dx = np. zeros (( Xmax ),float )  # D array

Exn = np.zeros((Xmax,nsteps+1),float)
Sn = np.zeros((Xmax,nsteps+1),float)
Dn = np.zeros((Xmax,nsteps+1),float)

c = constants.c # speed of light in vacuum
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
fs = constants.femto # 1.e-15 - useful for pulses
tera = constants.tera # 1.e12 - used for optical frequencues
alpha = 4*np.pi*tera
wp = 1.25e15
w0 = 2*np.pi*200*tera
f0 = 0.05
beta = np.sqrt(w0**2-alpha**2)

L = [100E-9, 500E-9, 1E-6, 2E-6] #thickness
LL = [100E-9/ddx, 500E-9/ddx, 1E-6/ddx, 2E-6/ddx]

# source positions
isource = 200

"Pulse parameters and points per wavelength"
spread=2.* fs/dt # 2 fs for this example
X1=int(Xmax/2) # center position
X0=[int(X1 - 100E-9/ddx), int(X1 - 500E-9/ddx), int(X1 - 1E-6/ddx), int(X1 - 2E-6/ddx)]
t0=spread*6
freq_in = 2*math.pi*200*tera # incident (angular) frequency
w_scale = freq_in*dt
lam = 2*math.pi*c/freq_in # near 1.5 microns
ppw = int(lam/ddx) # will round down
print('points per wavelength',ppw, 'should be > 15')


print(X0)
# an array for spatial points (without first and last points)
xs = np.arange(1,Xmax-1)

time = np.zeros((nsteps+1),float)

Er = np.zeros((nsteps+1),float)
Et = np.zeros((nsteps+1),float)

R = np.zeros((nsteps+1),float)
T = np.zeros((nsteps+1),float)
R_an = np.zeros((nsteps+1),complex)
T_an = np.zeros((nsteps+1),complex)

# initial time
t=0

def pulse(t):
    return (-np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*w_scale)))

" Main FDTD loop iterated iter_t times"
def FDTD_loop(nsteps,cycle,start):
    # loop over all time steps

    for i in range (0,nsteps+1): # time loop, from 0 to nsteps
       t=i-1 # iterative time dep pulse as source
       time[i] = t

# update E
       for x in range (1,Xmax-1):
           Dx[x] = Dx[x] + 0.5*(Hy[x-1]-Hy[x])

       Dx[isource] = Dx[isource] - pulse(t+0.5)*0.5
       Dn[:,i] = Dx

       for x in range (1, Xmax -1):
           if x<start or x>X1:
               Sn[x,i] = 0
           else:
               Sn[x,i] = (2*np.exp(-alpha*dt)*np.cos(beta*dt))*Sn[x,i-1] - np.exp(-2*alpha*dt)*Sn[x,i-2] + dt*f0*w0**2/beta*(np.exp(-alpha*dt)*np.sin(beta*dt))*Exn[x,i-1]
           Ex[x] = Dn[x,i] - Sn[x,i]

       Exn[:,i] = Ex

       if i > 2 : #ABC
           Ex[0] = Exn[1, i-2]
           Ex[Xmax-1] = Exn[Xmax-2, i-2]

       Et[i] = Ex[Xmax-5]
       Er[i] = Ex[5]

# update H
       for x in range (0,Xmax-1):
           Hy[x] = Hy[x] + 0.5*(Ex[x]-Ex[x+1])

       Hy[isource-1] = Hy[isource-1] - pulse(t)*0.5 #TOTAL-FIELD-SCATTER.FIELD

# update graph every cycle
       if (i % cycle == 0): # simple animation
           im.set_ydata(Ex[1:Xmax-1])
           ax.set_title("frame time {}".format(i))
           #if  i ==100 or i==200 or i== 800 or i==1200 or i==2000 or i==3500:
            #   plt.savefig('./Q2_a_' + str(i)+ '.pdf', format='pdf', dpi=1200,bbox_inches = 'tight')
           #plt.show()
           plt.pause(time_pause) # sinsible pause to watch animation


# initialize graph, fix scalign for thsi first example
def init1():
    plt.ylim((-1.,1.)) #(-1E-5, 1E-5))
    plt.xlim((0, Xmax-1)) #((0,100))
    plt.axvline(x=X1,color='r') # Vert line separator
    plt.axvline(x=X0[2],color='r') # Vert line separator
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    #plt.show()

def analitical(w,L):
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
    R = np.abs(r)**2
    T = np.abs(t)**2
    return R, T

def fourier(Er,Et,Ein):
    Er = np.abs(np.fft.fftshift(np.fft.fft(Er)))
    Et = np.abs(np.fft.fftshift(np.fft.fft(Et)))
    Ein = np.abs(np.fft.fftshift(np.fft.fft(Ein)))
    R = Er**2 / Ein**2
    T = Et**2 / Ein**2
    return R, T

#%%
"Define first (only in this simple example) graph for updating Ex at varios times"
lw=2 # line thickness for graphs
fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([.18, .18, .7, .7])
[im] = ax.plot(xs,Ex[1:Xmax-1],linewidth=lw)
init1() # initialize, then we will just update the y data and title frame

"Main FDTD: time steps = nsteps, cycle for very simple animation"
for x in range(len(L)):
    FDTD_loop(nsteps,cycle,X0[x])

    "Save Last Slice"
    if graph_flag == 1:
        plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.clf()

    incident=[]
    for i in range(0,len(time)):
        incident.append(pulse(time[i]))

    plt.plot(time,incident, label='$E_{in}$')
    plt.plot(time,Er, label='$E_{r}$')
    plt.plot(time,Et, label='$E_{t}$')
    plt.xlabel('Time ($\Delta$t)')
    plt.ylabel('$E_{x}$')
    plt.legend()
    plt.savefig("Q2_c"+str(X0[x])+"_E.pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()

    freq= (np.fft.fftshift(np.fft.fftfreq(nsteps+1,dt)))/tera

    R, T = fourier(Er,Et,incident)
    plt.plot(freq,R, 'r', label='R')
    plt.plot(freq,T, 'b', label='T')
    plt.plot(freq,R+T, 'y', label='Sum')
    plt.xlim((100,300))
    plt.ylim((0, 1.2))
    plt.xlabel('$w/2\pi (GHz)$')
    plt.ylabel(' ')
    plt.legend()
    #plt.savefig("Q2_c"+str(X0[x])+"_RT.pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
    #plt.show()

    w= 2*np.pi*(np.fft.fftshift(np.fft.fftfreq(nsteps+1,dt)))

    R, T = analitical(w,L[x])
    plt.plot(freq,R, '--r', label='$R_{an}$')
    plt.plot(freq,T, '--b', label='$T_{an}$')
    plt.plot(freq,R+T, '--y', label = '$Sum_{an}$')
    plt.xlim((100,300))
    plt.ylim((0, 1.2))
    plt.xlabel('$w/2\pi (THz)$')
    plt.ylabel(' ')
    plt.legend()
    plt.savefig("Q2_c_both"+str(X0[x])+".pdf", format='pdf', dpi=1200,bbox_inches = 'tight')
    plt.show()
