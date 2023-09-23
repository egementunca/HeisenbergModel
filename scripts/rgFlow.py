"""
Classical Heisenberg Model on Hierarchical Lattice with Quenched Randomness
The code creates sample pools of LFC groups that represents Ferromagnetic and Antiferromagnetic
distributions (+J and -J respectively) and performs Renormalization Group on the pool

Vacancy Model Solution performs same calculations
"""
import RenormGroupFunctions as rg
import numpy as np
import random

def probFunc(delta):
	if delta>=200:
		return 0
	if delta<=-200:
		return 1
	return np.exp(-2*delta)/(1+2*np.exp(-delta)+np.exp(-2*delta))

#Creates a pool of LFC groups with given J, size, and
#probability (p) of consisting symmetrical antiferromagnetic LFC's
#vacancy rate (q) is the ratio of the pool with no bonds
def startPool(J, p, q, size, l_prec):

	pool = np.zeros((size, l_prec))

	ferro_lfc = rg.lfc_initialize(J, l_prec)
	antiferro_lfc = rg.lfc_initialize(-J, l_prec)

	lfc0 = np.zeros(l_prec)
	lfc0[0] += 1

	q_size = int(round(q*size))
	pool[:q_size, ::] = lfc0

	remainingSize = int(size-q_size)
	ferroSize = int(np.around((1-p)*remainingSize))

	pool[q_size:q_size+ferroSize, ::] = ferro_lfc
	pool[q_size+ferroSize:, ::] = antiferro_lfc

	return pool

#Bond Move for pool: Creates a pool n times  bond moved lfc's
#(n = scaling factor^(dimension-1))
def poolBM(pool, n):

	size, l_prec = len(pool), len(pool[0])
	pools = []
	pools.append(pool)

	for i in range(n-1):
		bm_step = []
		for j in range(size):
			lfc1, lfc2 = pools[0][random.randint(0,size-1)], pools[i][random.randint(0,size-1)]
			lfc_bm = rg.bond_move(lfc1,lfc2,l_prec)
			bm_step.append(lfc_bm)
		pools.append(bm_step)
	return pools[-1]

#Decimation for pool:
def poolDEC(pool, dim):

	size, l_prec = len(pool), len(pool[0])
	pools = []
	pools.append(pool)

	for i in range(dim-1):
		dec_step = []
		for j in range(size):
			ix1, ix2 = random.randint(0,size-1), random.randint(0,size-1)
			lfc1, lfc2 = pools[0][ix1], pools[i][ix2]
			lfc_dec = rg.decimate(lfc1,lfc2)
			dec_step.append(lfc_dec)
		pools.append(dec_step)

	return pools[-1]

#first decimation step for BEG model hamiltonain with Heisenberg interactions with b=3
def vacancyStep(pool, J, g, dim):

	size, l_prec = len(pool), len(pool[0])
	vacancy_decimation = []
	for i in range(size):
		ix1, ix2, ix3 = random.randint(0,size-1), random.randint(0,size-1), random.randint(0,size-1)
		lfc1, lfc2, lfc3 = pool[ix1], pool[ix2], pool[ix3]

		delta = g*J
		lfc_dec = rg.decimateVacancy(lfc1,lfc2,lfc3,J,delta)
		vacancy_decimation.append(lfc_dec)

	return vacancy_decimation

#nth (n!=1) decimation step for BEG model hamiltonain with Heisenberg interactions
#bonds either exist or does not depending on the function of delta 
def poolDECVacancy(pool, dim, delta):
	
	size, l_prec = len(pool), len(pool[0])
	pools = []
	pools.append(pool)

	lfc_vacancy = np.zeros(l_prec)
	lfc_vacancy[0] += 1

	r_threshold = probFunc(delta)

	for i in range(dim-1):
		dec_step = []
		for j in range(size):
			r1, r2 = np.random.random(), np.random.random()
			ix1, ix2 = random.randint(0,size-1), random.randint(0,size-1)
			if r1 <= r_threshold:
				lfc1 = pools[0][ix1]
			else:
				lfc1 = lfc_vacancy
			if i==0:
				if r2 <= r_threshold:
					lfc2 = pools[i][ix2]
				else:
					lfc2 = lfc_vacancy
			else:
				lfc2 = pools[i][ix2]
			lfc_dec = rg.decimate(lfc1,lfc2)
			dec_step.append(lfc_dec)
		pools.append(dec_step)

	return pools[-1]

#nth (n!=1) bond move step for BEG model hamiltonain with Heisenberg interactions
#bonds either exist or does not depending on the function of delta
def poolBMVacancy(pool, n, delta):
	
	size, l_prec = len(pool), len(pool[0])
	pools = []
	pools.append(pool)

	lfc_vacancy = np.zeros(l_prec)
	lfc_vacancy[0] += 1

	r_threshold = probFunc(delta)

	for i in range(n-1):
		bm_step = []
		for j in range(size):

			r1, r2 = np.random.random(), np.random.random()
			if r1 <= r_threshold:
				lfc1 = pools[0][random.randint(0,size-1)]
			else:
				lfc1 = lfc_vacancy
			if i==0:
				if r2 <= r_threshold:
					lfc2 = pools[i][random.randint(0,size-1)]
				else:
					lfc2 = lfc_vacancy
			else:
				lfc2 = pools[i][random.randint(0,size-1)]

			lfc_bm = rg.bond_move(lfc1,lfc2,l_prec)
			bm_step.append(lfc_bm)
		pools.append(bm_step)
	return pools[-1]

#Renormalization Group with given Bond Moving and Decimaiton numbers
def rgTransform(pool, dim, n):

	random.seed(17)
	pool_transformed = poolDEC(pool, dim)
	random.seed(34)
	pool_transformed = poolBM(pool_transformed, n)

	return pool_transformed

#Main function to track RG flows
def rgTrajectory(J, p, q, n, dim, pool_size, l_prec, rg_step):

	LFC_flow = []

	pool = startPool(J, p, q, pool_size, l_prec)

	LFC_flow.append(pool)

	for i in range(rg_step):
		rg_pool = rgTransform(pool, dim, n)
		LFC_flow.append(rg_pool)
		pool = rg_pool

	return np.array(LFC_flow)