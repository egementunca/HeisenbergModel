"""
Classical Heisenberg Model on Hierarchical Lattice with Quenched Randomness
The code creates sample pools of LFC groups that represents Ferromagnetic and Antiferromagnetic
distributions (+J and -J respectively) and performs Renormalization Group on the pool

Vacancy Model Solution performs same calculations
"""
import RenormGroupFunctionsVacancy as rg
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

#first decimation step for BEG model hamiltonain with Heisenberg interactions with b=3
def vacancyStep(pool, J, g, dim):

	size, l_prec = len(pool), len(pool[0])
	vacancy_decimation = []
	delta_pool = []

	for i in range(size):
		ix1, ix2, ix3 = random.randint(0,size-1), random.randint(0,size-1), random.randint(0,size-1)
		lfc1, lfc2, lfc3 = pool[ix1], pool[ix2], pool[ix3]

		delta = g*J
		lfc_dec, delta_ = rg.decimateVacancy(lfc1,lfc2,lfc3,delta)
		vacancy_decimation.append(lfc_dec)
		delta_pool.append(3*delta+(delta_/dim**(dim)))

	return vacancy_decimation, delta_pool


#nth (n!=1) decimation step for BEG model hamiltonain with Heisenberg interactions
#bonds either exist or does not depending on the function of delta 
def poolDECVacancy(pool, delta_pool, dim, rg_step):
	
	size, l_prec = len(pool), len(pool[0])
	pools = []
	delta_pools = []
	pools.append(pool)
	delta_pools.append(delta_pool)

	lfc_vacancy = np.zeros(l_prec)
	lfc_vacancy[0] += 1

	for i in range(dim-1):
		dec_step = []
		delta_step = []

		for j in range(size):
			idx1, idx2 = random.randint(0,size-1), random.randint(0,size-1)
			
			lfc1, delta1 = pools[0][idx1], delta_pools[0][idx1]
			lfc2, delta2 = pools[i][idx2], delta_pools[i][idx2]

			r_threshold1 = probFunc(delta1)
			r_threshold2 = probFunc(delta2)

			r1 = np.random.random()
			if r1 <= r_threshold1:
				pass
			else:
				lfc1 = lfc_vacancy

			if i==0:
				r2 = np.random.random()
				if r2 <= r_threshold2:
					pass
				else:
					lfc2 = lfc_vacancy

			lfc_dec, delta_ = rg.decimate(lfc1,lfc2)
			dec_step.append(lfc_dec)
			delta_step.append(delta1+delta2+(delta_/dim**(dim*rg_step)))
		pools.append(dec_step)
		delta_pools.append(delta_step)

	return pools[-1], delta_pools[-1]

#nth (n!=1) bond move step for BEG model hamiltonain with Heisenberg interactions
#bonds either exist or does not depending on the function of delta
def poolBMVacancy(pool, delta_pool, n):
	
	size, l_prec = len(pool), len(pool[0])
	pools = []
	delta_pools = []

	pools.append(pool)
	delta_pools.append(delta_pool)

	lfc_vacancy = np.zeros(l_prec)
	lfc_vacancy[0] += 1

	for i in range(n-1):
		bm_step = []
		delta_step = []

		for j in range(size):
			idx1, idx2 = random.randint(0,size-1), random.randint(0,size-1)
			
			lfc1, delta1 = pools[0][idx1], delta_pools[0][idx1]
			lfc2, delta2 = pools[i][idx2], delta_pools[i][idx2]

			r_threshold1 = probFunc(delta1)
			r_threshold2 = probFunc(delta2)

			r1 = np.random.random()
			if r1 <= r_threshold1:
				pass
			else:
				lfc1 = lfc_vacancy

			if i==0:
				r2 = np.random.random()
				if r2 <= r_threshold2:
					pass
				else:
					lfc2 = lfc_vacancy

			lfc_bm = rg.bond_move(lfc1,lfc2,l_prec)
			bm_step.append(lfc_bm)
			delta_step.append(delta1+delta2)
		pools.append(bm_step)
		delta_pools.append(delta_step)

	return pools[-1], delta_pools[-1]

#First step for BEG model with Heisenberg interactions
def rgTransformVacancy1(pool, J, g, dim, n):
	
	delta = J*g
	random.seed(21)
	pool_transformed, delta_pool = vacancyStep(pool, J, g, dim)
	random.seed(34)
	pool_transformed, delta_pool = poolBMVacancy(pool_transformed, delta_pool, n)

	return (pool_transformed, delta_pool)

#nth step for BEG model with Heisenberg interactions
def rgTransformVacancy2(pool, delta_pool, J, g, dim, n, rg_step):
	
	delta = J*g
	random.seed(21)
	pool_transformed, delta_pool_rg = poolDECVacancy(pool, delta_pool, dim, rg_step)
	random.seed(34)
	pool_transformed, delta_pool_rg = poolBMVacancy(pool_transformed, delta_pool_rg, n)

	return (pool_transformed, delta_pool_rg)

#Main function to track RG flows in BEG case
def rgTrajectoryVacancy(J, g, p, q, n, dim, pool_size, l_prec, rg_step):

	LFC_flow = []

	pool = startPool(J, p, q, pool_size, l_prec)
	LFC_flow.append(pool)
	
	for i in range(rg_step):

		if i == 0:
			rg_pool, rg_delta = rgTransformVacancy1(pool, J, g, dim, n)
			LFC_flow.append(rg_pool)
			pool = rg_pool
		
		else:
			rg_pool, rg_delta = rgTransformVacancy2(pool, rg_delta, J, g, dim, n, i+1)
			LFC_flow.append(rg_pool)
			pool = rg_pool


	return np.array(LFC_flow)

#HELPER FUNCTIONS TO ANALYZE FLOWS

#mix 2 near trajectories at a point
def mixPool(pool1, pool2):
	
	new_pool = np.zeros(shape=pool1.shape)
	new_pool[len(pool1)//2:,::] = pool1[len(pool1)//2:,::]
	new_pool[:len(pool1)//2,::] = pool2[:len(pool1)//2,::]

	return new_pool

#Continue RG trajectory between of give two pools at a step
def rgTrajectoryContinue(pool1, pool2, g, p, q, n, dim, pool_size, l_prec, rg_step):

	LFC_flow = []

	pool = mixPool(pool1, pool2)
	
	LFC_flow.append(pool)

	for i in range(rg_step):

		rg_pool = rgTransform(pool, dim, n)
		LFC_flow.append(rg_pool)
		pool = rg_pool

	return np.array(LFC_flow)


