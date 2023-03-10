import numpy as np
import rgFlow as flow

def checkPhase(p, n, q, dim, pool_size, l_prec, rg_step, iter_num):
	data = []

	jDisorder = 1/0.08
	jOrder = 50

	for i in range(iter_num):
		jNew = (jOrder + jDisorder)/2
		f = flow.rgTrajectory(J=jNew, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)
		if abs(f[-1,0,0]) != 1:
			jOrder = jNew
			data.append([jNew, 'ordered'])	
		else:
			jDisorder = jNew
			data.append([jNew, 'disordered'])

	return data

def phase_diagram_qvsTc(p, n, dim, pool_size, l_prec, rg_step, iter_num):
	q_vals = np.linspace(0, .64, 65)
	j_vals = np.zeros(len(q_vals))
	
	for i,q in enumerate(q_vals):
		j_crit = checkPhase(p=p, n=n, q=q, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step, iter_num=iter_num)[-1][0]
		j_vals[i] = j_crit
	np.save('qvsTc_p{}.npy'.format(p), (q_vals, j_vals))
	return 0

phase_diagram_qvsTc(p=.5, n=9, dim=3, pool_size=50000, l_prec=21, rg_step=30, iter_num=30)