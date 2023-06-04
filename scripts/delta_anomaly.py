import numpy as np
import rgFlow as flow

def checkPhase(J, p, n, q, dim, pool_size, l_prec, rg_step, iter_num):
	data = []

	deltaDisorder = .58
	deltaOrder = .6
	
	for i in range(iter_num):
		deltaNew = (deltaOrder + deltaDisorder)/2
		f = flow.rgTrajectoryVacancy(J=J, g=deltaNew, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)
		if abs(f[-1,0,0]) != 1:
			deltaOrder = deltaNew
			print('delta:{} is {}'.format(deltaNew, 'ordered'))
			data.append([deltaNew, 'ordered'])	
		else:
			deltaDisorder = deltaNew
			print('delta{} is {}'.format(deltaNew, 'disordered'))
			data.append([deltaNew, 'disordered'])

	return data

def checkPhase2(g, p, n, q, dim, pool_size, l_prec, rg_step, iter_num):
	data = []

	tDisorder = .23
	tOrder = .08

	for i in range(iter_num):
		tNew = (tOrder + tDisorder)/2
		f = flow.rgTrajectoryVacancy(J=1/tNew, g=g, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)
		if abs(f[-1,0,0]) != 1:
			tOrder = tNew
			print('t{} is {}'.format(tNew, 'ordered'))
			data.append([tNew, 'ordered'])	
		else:
			tDisorder = tNew
			print('t{} is {}'.format(tNew, 'disordered'))
			data.append([tNew, 'disordered'])

	return data

#data = checkPhase(J=1/0.14, p=0.5, q=0, n=9, dim=3, pool_size=30000, l_prec=21, rg_step=25, iter_num=30)
#np.save('data.npy', data)

data2 = checkPhase2(g=0.595312417950481, p=0.5, q=0, n=9, dim=3, pool_size=30000, l_prec=21, rg_step=25, iter_num=30)
np.save('data2.npy', data2)
##anomaly delta: 0.5982879872433842