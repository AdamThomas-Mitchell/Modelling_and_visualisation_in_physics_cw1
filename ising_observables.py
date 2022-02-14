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
    #function to initialise spins randomly in lxl numpy array
    spin = np.random.rand(l,l)
    spin[spin<0.5] = int(-1)
    spin[spin>=0.5] = int(1)

    return spin

def glauber(l,kT):
    #function for Glauber dynamics

    #select spin randomly
    itrial=np.random.randint(0,l)
    jtrial=np.random.randint(0,l)
    spin_new=-spin[itrial,jtrial]

    #compute delta E
    nn = spin[(itrial+1)%l ,jtrial] + spin[(itrial-1)%l ,jtrial] + spin[itrial,(jtrial+1)%l] + spin[itrial,(jtrial-1)%l]        #nearest neighbour
    delta_E = -2.0*spin_new*nn

    #perform metropolis test
    if delta_E < 0:
        spin[itrial,jtrial] = spin_new
    elif np.random.rand() < math.exp(-delta_E/kT):
        spin[itrial,jtrial] = spin_new

def kawasaki(l,kT):
    #function for Kawasaki dynamics

    #select first spin randomly
    itrial1=np.random.randint(0,l)
    jtrial1=np.random.randint(0,l)
    spin1=spin[itrial1,jtrial1]

    #select second spin randomly
    itrial2=np.random.randint(0,l)
    jtrial2=np.random.randint(0,l)
    spin2=spin[itrial2,jtrial2]

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

def magnetisation(state):
    #function to calculate extensive magnetisation of system
    mag = np.sum(state)
    return mag

def total_energy(spin,l):
    #function to calculate total energy of spin system
    energy=0.0
    for i in range(l):
        for j in range(l):
            nn = spin[(i+1)%l ,j] + spin[(i-1)%l ,j] + spin[i,(j+1)%l] + spin[i,(j-1)%l]
            energy += -nn * spin[i,j]

    return energy/2.0

### INPUT ###
#temperature range, number of steps and system size
l=50
nstep=25000000
temp=np.arange(1.0,3.1,0.1)

#specify dynamics
choice = input("Type (1) for Glauber dynamics or (2) for Kawaski dynamics.     ")
method=None
if choice == "1":
    method=1
elif choice == "2":
    method=2

### GLAUBER ###
if method==1:
    #initialise spin state in ground state
    spin = np.ones((l,l))

    #main loop
    #open file to write data
    f=open('observ_glauber.dat','w')
    for i in range(len(temp)):

        #define temperature
        kT=temp[i]

        #initialise arrays for magnetisation and energy values
        mag_array = np.empty(0)
        abs_mag_array = np.empty(0)
        energy_array = np.empty(0)

        for n in range(nstep):
            glauber(l,kT)

            #only record data after first 200 sweeps so system can equilibrate
            if n>500000:

            #record data every 10 sweeps
                if n%25000==0:
                    mag = magnetisation(spin)
                    mag_array = np.append(mag_array, mag)
                    abs_mag_array = np.append(abs_mag_array, abs(mag))
                    energy = total_energy(spin,l)
                    energy_array = np.append(energy_array, energy)

        #observables measurements
        mag_mean = np.mean(mag_array)
        abs_mag_mean = np.mean(abs_mag_array)
        energy_mean = np.mean(energy_array)
        chi=(np.mean(mag_array**2.0) - np.mean(mag_array)**2.0)/((l**2.0) * kT)
        heatcap=(np.mean(energy_array**2.0) - np.mean(energy_array)**2.0)/((l**2.0) * kT**2.0)

        ###observables error###
        #susceptibility error
        sus_samples = np.zeros(101)
        for k in range(101):
            mag_resample = np.zeros_like(mag_array)
            for j in range(len(mag_array)):
                mag_resample[j] += random.choice(mag_array)
            sus_new = (np.mean(mag_resample**2.0) - np.mean(mag_resample)**2.0)/((l**2.0) * kT)
            sus_samples[k] += sus_new
        sus_error = math.sqrt( np.mean(sus_samples**2.0) - np.mean(sus_samples)**2.0 )

        #heat capacity error#
        C_errorsamples = np.zeros(101)
        for k in range(101):
            E_resample = np.zeros_like(energy_array)
            for j in range(len(energy_array)):
                E_resample[j] += random.choice(energy_array)
            C = (np.mean(E_resample**2.0) - np.mean(E_resample)**2.0)/((l**2.0) * kT**2.0)
            C_errorsamples[k] += C
        heatcap_error = math.sqrt( np.mean(C_errorsamples**2.0) - np.mean(C_errorsamples)**2.0 )

        #dump output
        f.write('%lf %lf %lf %lf %lf %lf %lf\n'%(kT, abs_mag_mean, chi, energy_mean, heatcap, sus_error, heatcap_error))
    f.close()

### KAWASAKI ###
if method==2:
    #initialise spin state
    spin=init_spin(l)

    #main loop
    #open file to write data
    f=open('obs_kawasaki.dat','w')
    for i in range(len(temp)):

        #define temperature
        kT=temp[i]

        #initialise arrays for magnetisation and energy values
        mag_array = np.empty(0)
        abs_mag_array = np.empty(0)
        energy_array = np.empty(0)

        for n in range(nstep):
            kawasaki(l,kT)

            #only record data after first 200 sweeps so system can equilibrate
            if n>500000:

            #record data every 10 sweeps
                if n%25000==0:
                    mag = magnetisation(spin)
                    mag_array = np.append(mag_array, mag)
                    abs_mag_array = np.append(abs_mag_array, abs(mag))
                    energy = total_energy(spin,l)
                    energy_array = np.append(energy_array, energy)

        abs_mag_mean = np.mean(abs_mag_array)
        energy_mean = np.mean(energy_array)
        chi=(np.mean(mag_array**2.0) - np.mean(mag_array)**2.0)/((l**2.0) * kT)
        heatcap=(np.mean(energy_array**2.0) - np.mean(energy_array)**2.0)/((l**2.0) * kT**2.0)

        ###observables error###
        #susceptibility error
        sus_samples = np.zeros(101)
        for k in range(101):
            mag_resample = np.zeros_like(mag_array)
            for j in range(len(mag_array)):
                mag_resample[j] += random.choice(mag_array)
            sus_new = (np.mean(mag_resample**2.0) - np.mean(mag_resample)**2.0)/((l**2.0) * kT)
            sus_samples[k] += sus_new
        sus_error = math.sqrt( np.mean(sus_samples**2.0) - np.mean(sus_samples)**2.0 )

        #heat capacity error#
        C_errorsamples = np.zeros(101)
        for k in range(101):
            E_resample = np.zeros_like(energy_array)
            for j in range(len(energy_array)):
                E_resample[j] += random.choice(energy_array)
            C = (np.mean(E_resample**2.0) - np.mean(E_resample)**2.0)/((l**2.0) * kT**2.0)
            C_errorsamples[k] += C
        heatcap_error = math.sqrt( np.mean(C_errorsamples**2.0) - np.mean(C_errorsamples)**2.0 )

        #dump output
        f.write('%lf %lf %lf %lf %lf %lf %lf\n'%(kT, abs_mag_mean, chi, energy_mean, heatcap, sus_error, heatcap_error))
    f.close()
