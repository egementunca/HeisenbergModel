import numpy as np
import rgFlow as flow

def checkPhase(g, p, n, q, dim, pool_size, l_prec, rg_step, iter_num):
	data = []

	jDisorder = 1
	jOrder = 50

	for i in range(iter_num):
		jNew = (jOrder + jDisorder)/2
		f = flow.rgTrajectoryVacancy(J=jNew, g=g, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)
		if abs(f[-1,0,0]) != 1:
			jOrder = jNew
			print(f'p:{p} at j:{jNew} is ordered')
			data.append([jNew, 'ordered'])	
		else:
			jDisorder = jNew
			print(f'p:{p} at j:{jNew} is disordered')
			data.append([jNew, 'disordered'])

	return data

def phase_diagram_pvsTc(g, q, n, dim, pool_size, l_prec, rg_step, iter_num):
	p_vals = np.linspace(0, .5, 101)
	j_vals = np.zeros(len(p_vals))
	
	for i,p in enumerate(p_vals):
		j_crit = checkPhase(g=g, p=p, n=n, q=q, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step, iter_num=iter_num)[-1][0]
		j_vals[i] = j_crit
	np.save('orderline_delta_neg2.npy', (p_vals, j_vals))
	return 0

phase_diagram_pvsTc(g=-2, q=0, n=9, dim=3, pool_size=5000, l_prec=21, rg_step=25, iter_num=25)