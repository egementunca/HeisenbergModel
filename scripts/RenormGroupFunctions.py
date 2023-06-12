"""
This file contains essential functions required for Renormalization Group
trajectory initialization and tracking.

It can be used for Classical Heisenberg Model on Hierarchical Lattice with Hamiltonian
including only nearest neighbour interactions and performs (in theory) exact scaling.

Both ,pure and random bond, calculations are executed on other scripts.
"""

import numpy as np
from scipy.special import spherical_jn
import numba

import scipy.integrate as integrate

#import sys
#sys.path.append('../')
#Clebsch Gordan coefficients tracked up to (l1,l2,l)==(50,50,50)
clebsch_gordan = np.load('../data/cleb.npy')

#Creates Legendre Fourier coefficients with given J to
#express our exponantiated hamiltonian in series form.
def lfc_initialize(J, l_prec):
	x = np.arange(l_prec)
	x = np.real((2*x+1)*(1j**(x))*spherical_jn(x,-1j*J))
	return x
	#return x/np.amax(np.abs(x))

#Bond Move Process of Renormalization Group
#Takes 2 LFC groups and returns "bond-moved" LFC group
@numba.jit()
def bond_move(lfc1, lfc2, l_prec):
	lfc_bond_moved = np.zeros(l_prec)
	for l in range(l_prec):
		val = 0
		for l1 in range(l_prec):
			for l2 in range(l_prec):
				x = lfc1[l1]*lfc2[l2]*clebsch_gordan[l1,l2,l]
				val += x
		lfc_bond_moved[l] += val
	y = np.amax(np.abs(lfc_bond_moved))
	return lfc_bond_moved/y

#Decimation Process of Renormalization Group
#Takes 2 LFC groups and returns decimated LFC group
@numba.jit()
def decimate(lfc1, lfc2):
	l_prec = len(lfc1)
	lfc_decimated = np.arange(l_prec, dtype=np.float64)
	lfc_decimated = (lfc1*lfc2)/(2*lfc_decimated+1)
	x = np.amax(np.abs(lfc_decimated))
	return lfc_decimated/x
	#return lfc_decimated

from scipy.special import eval_legendre

def helper(lfc, gamma):
    val = 0
    for l in range(len(lfc)):
        val += lfc[l]*eval_legendre(l,np.cos(gamma))
    return val

def decimateVacancy(lfc1, lfc2, lfc3, J, delta):

		l_prec = len(lfc1)
		odd_nums = 2*np.arange(l_prec)+1

		lfc_decimated = (lfc1*lfc2*lfc3)/(odd_nums)**2

		lfc_combined = np.exp(-2*delta)*lfc_decimated
		
		lfc_1 = np.zeros(l_prec)
		lfc_1[0] += 1

		lfc_2, lfc_3 = np.zeros(l_prec), np.zeros(l_prec)
		lfc_2[0] += np.exp(-delta)*(lfc2[0]/(np.sqrt(4*np.pi)))
		lfc_3[0] += np.exp(-delta)*(lfc3[0]/(np.sqrt(4*np.pi)))

		lfc = lfc_combined+lfc_1+lfc_2+lfc_3

		x_axis = np.linspace(0,2*np.pi,1000)
		y1 = helper(lfc_1,x_axis)
		y2 = helper(lfc_2,x_axis)
		y3 = helper(lfc_3,x_axis)
		y4 = helper(lfc_combined,x_axis)

		probs = []

		probs.append(integrate.simpson(y1,x_axis))
		probs.append(integrate.simpson(y2,x_axis))
		probs.append(integrate.simpson(y3,x_axis))
		probs.append(integrate.simpson(y4,x_axis))
		
		probs = np.array(probs)
		probsSum = sum(probs)
		probs = probs/probsSum
		
		print(probs)
		print(sum(probs))

		xi = np.random.random()

		if xi < probs[0]:
			return lfc_1/np.amax(np.abs(lfc_1))
		if xi >= probs[0] and xi<(probs[0]+probs[1]):
			return lfc_2/np.amax(np.abs(lfc_2))
		if xi >= (probs[0]+probs[1]) and xi<(probs[0]+probs[1]+probs[2]):
			return lfc_3/np.amax(np.abs(lfc_3))
		if xi >= (probs[0]+probs[1]+probs[2]):
			return lfc_combined/np.amax(np.abs(lfc_combined))

"""
@numba.jit()
def decimateVacancy(lfc1, lfc2, lfc3, J, delta):

		l_prec = len(lfc1)
		odd_nums = 2*np.arange(l_prec)+1

		lfc_decimated = (lfc1*lfc2*lfc3)/(odd_nums)**2

		lfc_combined = np.exp(-2*delta)*lfc_decimated
		
		lfc_1 = np.zeros(l_prec)
		lfc_1[0] += 1

		lfc_2, lfc_3 = np.zeros(l_prec), np.zeros(l_prec)
		lfc_2[0] += np.exp(-delta)*(lfc2[0]/np.sqrt(4*np.pi))
		lfc_3[0] += np.exp(-delta)*(lfc3[0]/np.sqrt(4*np.pi))

		lfc = (lfc_combined+lfc_1+lfc_2+lfc_3)
		return lfc
		x = np.amax(np.abs(lfc))
	
		return lfc/x"""

