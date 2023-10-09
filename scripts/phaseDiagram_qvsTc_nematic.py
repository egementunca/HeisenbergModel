import numpy as np
import rgFlow as flow

def checkPhase(p, n, q, dim, pool_size, l_prec, rg_step, iter_num):
	data = []

	jFerro = 5
	jOrder = 50

	for i in range(iter_num):
		jNew = (jNematic + jFerro)/2
		f = flow.rgTrajectory(J=jNew, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)
		avg_lfc = np.mean(abs(f[-1]), axis=0)
		if (sum(avg_lfc) < 15.37):
			jNematic = jNew
			print('j{} is {}'.format(jNew, 'nematic'))
			data.append([jNew, 'nematic'])	
		else:
			jFerro = jNew
			print('j{} is {}'.format(jNew, 'ferro'))
			data.append([jNew, 'ferro'])

	return data

def phase_diagram_qvsTc(p, n, dim, pool_size, l_prec, rg_step, iter_num):
	q_vals = np.linspace(0, .64, 65)
	j_vals = np.zeros(len(q_vals))
	
	for i,q in enumerate(q_vals):
		j_crit = checkPhase(p=p, n=n, q=q, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step, iter_num=iter_num)[-1][0]
		j_vals[i] = j_crit
	np.save('../data/qvsTc_p{}_nematic.npy'.format(p), (q_vals, j_vals))
	return 0

phase_diagram_qvsTc(p=.1, n=9, dim=3, pool_size=30000, l_prec=21, rg_step=30, iter_num=30)