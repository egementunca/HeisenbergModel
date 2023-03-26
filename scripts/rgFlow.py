"""
Classical Heisenberg Model on Hierarchical Lattice with Quenched Randomness
The code creates sample pools of LFC groups that represents Ferromagnetic and Antiferromagnetic
distributions (+J and -J respectively) and performs Renormalization Group on the pool
"""
import RenormGroupFunctions as rg
import numpy as np
import random

#Creates a pool of LFC groups with given J, size, and
#probability (p) of consisting symmetrical antiferromagnetic LFC's
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

	np.random.shuffle(pool)

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

#Bond Move for pool: Creates a pool n times  bond moved lfc's
#(n = scaling factor^(dimension-1))
def poolBM_old(pool, n):

	size, l_prec = len(pool), len(pool[0])
	pools = []
	pools.append(pool)
	n = bin(n)[:1:-1]
	indices = [i for i, x in enumerate(n) if x == "1"]

#Generate necessary bond move pools based on powers of 2 to reach n
#It can be understood as the binary represantation above
	for i in range(len(n)-1):
		bm_step = []
		for j in range(size):
			lfc1, lfc2 = pools[-1][random.randint(0,size-1)], pools[-1][random.randint(0,size-1)]
			lfc_bm = rg.bond_move(lfc1,lfc2,l_prec)
			bm_step.append(lfc_bm)
		pools.append(bm_step)
	pools.append(pools[indices[0]])

#To reach number n of bond moves use pools created before (if needed)
	for i in range(len(indices)-1):
		bm_step = []
		for j in range(size):
			lfc1, lfc2 = pools[-1][random.randint(0,size-1)], pools[indices[i+1]][random.randint(0,size-1)]
			lfc_bm = rg.bond_move(lfc1,lfc2,l_prec)
			bm_step.append(lfc_bm)
		pools.append(bm_step)

#Last pool is the one that bond moved n times
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

#Renormalization Group with given Bond Moving and Decimaiton numbers
def rgTransform(pool, dim, n):

	random.seed(17)
	pool_transformed = poolDEC(pool, dim)
	random.seed(34)
	pool_transformed = poolBM(pool_transformed, n)

	return pool_transformed

def rgTransformVacancy(pool, J, g, dim, n):

	random.seed(17)
	pool_transformed = vacancyStep(pool, J, g, dim)
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

#Main function to track RG flows
def rgTrajectoryVacancy(J, g, p, q, n, dim, pool_size, l_prec, rg_step):

	LFC_flow = []

	pool = startPool(J, p, q, pool_size, l_prec)
	
	LFC_flow.append(pool)

	for i in range(rg_step):

		if i == 0:
			rg_pool = rgTransformVacancy(pool, J, g, dim, n)
			LFC_flow.append(rg_pool)
			pool = rg_pool
		
		else:
			rg_pool = rgTransform(pool, dim, n)
			LFC_flow.append(rg_pool)
			pool = rg_pool

	return np.array(LFC_flow)


