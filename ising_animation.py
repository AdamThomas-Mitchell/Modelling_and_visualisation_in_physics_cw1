import matplotlib
matplotlib.use('TKAgg')

import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

### FUNCTIONS ###
def init_spin(l):
    #function to initialise spins randomly in NxN numpy array
    spin = np.random.rand(l,l)
    spin[spin<0.5] = -1
    spin[spin>=0.5] = 1

    return spin

def glauber(l,kT):
    #function for Glauber dynamics

    #select spin randomly
    itrial=np.random.randint(0,l)
    jtrial=np.random.randint(0,l)
    spin_new=-spin[itrial,jtrial]

    #compute delta E
    nn = (spin[(itrial+1)%l ,jtrial] + spin[(itrial-1)%l ,jtrial] + spin[itrial,(jtrial+1)%l] + spin[itrial,(jtrial-1)%l])
    delta_E = -2.0*spin_new*nn

    #perform metropolis test
    if delta_E < 0:
        spin[itrial,jtrial] = spin_new
    elif np.random.rand() < math.exp(-delta_E/kT):
        spin[itrial,jtrial] = spin_new

def kawasaki(l,kT):
    #function for Kawaski dynamics

    #select first spin randomly
    itrial1=np.random.randint(0,l)
    jtrial1=np.random.randint(0,l)
    spin1=spin[itrial1,jtrial1]

    #select second spin randomly
    itrial2=np.random.randint(0,l)
    jtrial2=np.random.randint(0,l)
    spin2=spin[itrial2,jtrial2]

    #only swap if spins different
    if spin1 != spin2:

        #swap spins
        spin1, spin2 = spin2, spin1

        #calculate delta_E
        nn1 = (spin[(itrial1+1)%l ,jtrial1] + spin[(itrial1-1)%l ,jtrial1] + spin[itrial1,(jtrial1+1)%l] + spin[itrial1,(jtrial1-1)%l])
        delta1 = -2.0*spin1*nn1

        nn2 = (spin[(itrial2+1)%l ,jtrial2] + spin[(itrial2-1)%l ,jtrial2] + spin[itrial2,(jtrial2+1)%l] + spin[itrial2,(jtrial2-1)%l])
        delta2 = -2.0*spin2*nn2

        delta_E = delta1 + delta2

        #perform metropolis test
        if delta_E < 0:
            spin[itrial1,jtrial1], spin[itrial2,jtrial2] = spin1, spin2
        elif np.random.rand() < math.exp(-delta_E/kT):
            spin[itrial1,jtrial1], spin[itrial2,jtrial2] = spin1, spin2

### INPUT ###
if(len(sys.argv) != 3):
    print("Usage python ising_animation.py N T")
    sys.exit()
l=int(sys.argv[1])
kT=float(sys.argv[2])
nstep=25000000

#Ask user for method
choice = input("Type (1) for Glauber dynamics or (2) for Kawaski dynamics.     ")
method=None
if choice == "1":
    method=1
elif choice == "2":
    method=2

### METHOD ###
#initialise spins randomly in NxN numpy array
spin = init_spin(l)
fig = plt.figure()
im=plt.imshow(spin, animated=True)

#main loop
for n in range(nstep):
    if method==1:
        glauber(l,kT)

    elif method==2:
        kawasaki(l,kT)

    #occasionally plot and update measurements every 10 sweeps
    if(n%25000==0):
        #update measurements
        #dump output
        f=open('spins.dat','w')
        for i in range(l):
            for j in range(l):
                f.write('%d %d %lf\n'%(i,j,spin[i,j]))
        f.close()
        #show animation
        plt.cla()
        im=plt.imshow(spin, animated=True)
        plt.draw()
        plt.pause(0.0001)
