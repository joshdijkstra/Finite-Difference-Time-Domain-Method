import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 20})
#from matplotlib import animation
import math
import scipy.constants as constants

"For animation updates - will slow down the loop to see Ex frames better"
time_pause = 0.0005

"Quick and durty graph to save"
filename = "fdtd_1d.pdf" # save at final time slice (example)

"Basic Geometry and Dielectric Parameters"
graph_flag = 0 # 0 will not graphics save
Xmax = 801  # no of FDTD cells in x
nsteps = 3500 # number of FDTD time steps
cycle = 100 # for graph updates
Ex = np.zeros((Xmax),float) # E array
Hy = np.zeros((Xmax),float) # H array
c = constants.c # speed of light in vacuum
ddx = 20.e-9 #  FDTD grid size in space, in SI Units
dt = ddx/(2.*c) # FDTD time step
fs = constants.femto # 1.e-15 - useful for pulses
tera = constants.tera # 1.e12 - used for optical frequencues

#r,t,in fields
Er = np.zeros((nsteps+1),float) # E array
Et = np.zeros((nsteps+1),float) # E array
Ein = np.zeros((nsteps+1),float) # E array

# source positions
isource = 200

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

#Thin Film
L = 1e-6
n = 3


EMinPrev = 0
EMaxPrev = 0
def pulse(t):
    return -np.exp(-0.5 * (t - t0) ** 2 / spread ** 2) * (np.cos(t * w_scale))

def cb(L,n,x):
    #coefficient based on index of refraction within film range
    if X1 <= x <= (X1 + L /ddx):    #
        ep = n**2
    else:
        ep = 1
    return 1 / (2*ep)

def analyticSoln(w,L):
    r1 = (1-n)/(1+n)
    r2 = (n-1)/(n+1)
    k0 = w/c

    e = np.exp(2*1j*k0*L*n)
    r = (r1 + r2*e)/(1+r1*r2*e)
    t = ((1+r1)*(1+r2)*e)/(1+r1*r2*e)
    R = np.abs(r)**2
    T = np.abs(t)**2
    return R, T

" Main FDTD loop iterated iter_t times"
def FDTD_loop(nsteps,cycle):
    snaps = [900,1500]
    #Boundary conditions
    BC1 = np.zeros(nsteps+1)
    BC2 = np.zeros((nsteps+1))
    # loop over all time steps
    for i in range (0,nsteps+1): # time loop, from 0 to nsteps
       t=i-1 # iterative time dep pulse as source

# update E
       for x in range (1,Xmax-1):
           # Ex[x] = Ex[x] + 0.5*(Hy[x-1]-Hy[x])  #Ex parts a,b
           Ex[x] = Ex[x] + cb(L,n,x) * (Hy[x - 1] - Hy[x])  #Ex part c,d

       # Ex[isource] = Ex[isource] - pulse*0.5
       Ex[isource] = Ex[isource] - pulse(t) * 0.5
       BC1[i] = Ex[1]       #Eqn 7
       BC2[i] = Ex[-3]      #Eqn 8
       if i > 1:                #BCs 0 for n = 0,1
           Ex[0] = BC1[i-2]
           Ex[-2] = BC2[i-2]

# update H
       for x in range (0,Xmax-1):
           Hy[x] = Hy[x] + 0.5*(Ex[x]-Ex[x+1])

       Hy[isource-1] = Hy[isource-1] - pulse(t-0.5) * 0.5

       Er[i] = Ex[199]         #reflected field
       Et[i] = Ex[500]         #transmitted field
       if i <= 600:
           Ein[i] = Ex[200]    #incident field

# update graph every cycle
       if (i % cycle == 0): # simple animation
           im.set_ydata(Ex[1:Xmax-1])
           ax.set_title("frame time {}".format(i))
           # plt.show()
           # if i in snaps:
               # plt.ylim((-0.00001,0.00001))
               # plt.xlim((20,100))
               # plt.savefig("frame time {}".format(i)+".png", format='png', dpi=1200, bbox_inches='tight')
               # plt.xlim((0,Xmax-1))
               # plt.ylim((-0.7, 0.7))
           plt.pause(time_pause) # sinsible pause to watch animation


# initialize graph, fix scalign for thsi first example
def init1():
    plt.ylim((-0.7, 0.7))
    plt.xlim((0, Xmax-1))
    # plt.axvline(x=X1,color='r') # Vert line separator
    plt.axvspan(400,450,color = 'r', alpha = 0.7)
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    # plt.show()

#%%
"Define first (only in this simple example) graph for updating Ex at varios times"
lw=2 # line thickness for graphs
fig = plt.figure(figsize=(12,10))
ax = fig.add_axes([.18, .18, .7, .7])
[im] = ax.plot(xs,Ex[1:Xmax-1],linewidth=lw)
init1() # initialize, then we will just update the y data and title frame

"Main FDTD: time steps = nsteps, cycle for very simple animation"
FDTD_loop(nsteps,cycle)

## Separated Field plots
#
# plt.figure(figsize=(12,12))
# plt.rcParams['font.size']=22
# plt.plot(np.arange(0,nsteps+1,1), Er, color = 'r', label = 'Reflected')
# plt.plot(np.arange(0,nsteps+1,1), Et, color = 'b', label = 'Transmitted')
# plt.plot(np.arange(0,nsteps+1,1), Ein, color = 'k', label = 'Input')
# plt.xlabel("$t$ (s)")
# plt.ylabel("$E(t)$")
# plt.title("Partitioned Fields")
# plt.legend()
# # plt.savefig("PartitionedFields.pdf", format='pdf', dpi=1200, bbox_inches='tight')
# plt.show()


#Partitioned Fields
EtOmega = np.fft.fftshift(np.abs(np.fft.fft(Et)))
ErOmega = np.fft.fftshift(np.abs(np.fft.fft(Er)))
EinOmega = np.fft.fftshift(np.abs(np.fft.fft(Ein)))

T = np.abs(EtOmega/EinOmega)**2
R = np.abs(ErOmega/EinOmega)**2



freq = np.fft.fftshift(np.fft.fftfreq(nsteps+1,dt))/1e12

plt.figure(figsize=(12,12))
plt.plot(freq,ErOmega, color = 'r', label = 'Reflected')
plt.plot(freq,EinOmega, color = 'k', label = 'Incident')
plt.plot(freq,EtOmega, color = 'b', label = 'Transmitted')
plt.ylim(0,80)
plt.xlim(0,600)
plt.xlabel("$\omega$) GHz")
plt.ylabel("E$(\omega)$")
plt.title("Fourier Transforms")
plt.legend()
# plt.savefig("1c.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12,12))
plt.plot(freq,T, color = 'r', label = 'Transmission')
plt.plot(freq,R, color = 'k', label = 'Reflection')
plt.plot(freq,R+T, color = 'violet', label = 'T+R')
plt.ylim(0,1.2)
plt.xlim(0,300)
plt.xlabel("$\omega$ GHz")
# plt.ylabel("Magnitude")
plt.title("Transmission and Reflection Coefficients")
plt.legend()
# plt.savefig("1c.pdf", format='pdf', dpi=1200, bbox_inches='tight')
plt.show()


## D

# eps = 9
# n = np.sqrt(eps)
# r1 = (1-n) / (1+n)
# r2 = (n-1) / (n+1)
# k0 = freq / c
#
# def r(omega):
#     return (r1 + r2*np.exp(2*1j*k0))



"Save Last Slice"
if graph_flag == 1:
    plt.savefig(filename, format='pdf', dpi=1200,bbox_inches = 'tight')
