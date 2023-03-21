import numpy as np
import rgFlow as flow

def checkPhase(p, n, q, g, dim, pool_size, l_prec, rg_step, iter_num, Thermal):
	data = []

	jDisorder = 1
	jOrder = 50

	for i in range(iter_num):
		jNew = (jOrder + jDisorder)/2
		f = flow.rgTrajectory(J=jNew, p=p, q=q, g=g, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step, Thermal=Thermal)
		if abs(f[-1,0,0]) != 1:
			jOrder = jNew
			print('j{} is {}'.format(jNew, 'ordered'))
			data.append([jNew, 'ordered'])	
		else:
			jDisorder = jNew
			print('j{} is {}'.format(jNew, 'disordered'))
			data.append([jNew, 'disordered'])

	return data

def phase_diagram_deltavsTc(p, q, n, dim, pool_size, l_prec, rg_step, iter_num, Thermal):
	g_vals = np.linspace(0, .1, 20)
	j_vals = np.zeros(len(g_vals))
	
	for i,g in enumerate(g_vals):
		j_crit = checkPhase(p=p, n=n, q=q, g=g, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step, iter_num=iter_num, Thermal=Thermal)[-1][0]
		j_vals[i] = j_crit
	np.save('deltavsTc_p{}.npy'.format(p), (g_vals, j_vals))
	return 0

phase_diagram_deltavsTc(p=0, q=0, n=9, dim=3, pool_size=30000, l_prec=21, rg_step=25, iter_num=25, Thermal=True)