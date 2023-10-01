import numpy as np
import rgFlowVacancy as flow

def normalize_pool(pool):
    size, l_prec = pool.shape
    norm_vals = np.amax(np.abs(pool), axis=1)
    for i in range(size):
        pool[i,::] /= norm_vals[i]
    return pool

def checkPhase(g, p, n, q, dim, pool_size, l_prec, rg_step, iter_num):
	data = []

	jFerro = 10
	jNematic = 50

	for i in range(iter_num):
		jNew = (jNematic + jFerro)/2
		f = flow.rgTrajectoryVacancy(J=jNew, g=g, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)
		check_pool = normalize_pool(fOrder[-1])
		avg_lfc = np.mean(abs(check_pool), axis=0)
		if (sum(avg_lfc) < 15.37):
			jNematic = jNew
			print('j{} is {}'.format(jNew, 'nematic'))
			data.append([jNew, 'nematic'])	
		else:
			jFerro = jNew
			print('j{} is {}'.format(jNew, 'ferro'))
			data.append([jNew, 'ferro'])

	return data

def phase_diagram_deltavsTc(p, q, n, dim, pool_size, l_prec, rg_step, iter_num):
	g_vals = np.linspace(-5, 0, 101)
	j_vals = np.zeros(len(g_vals))
	
	for i,g in enumerate(g_vals):
		j_crit = checkPhase(g=g, p=p, n=n, q=q, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step, iter_num=iter_num)[-1][0]
		j_vals[i] = j_crit
	np.save('deltavsTc_p{}_nematic.npy'.format(p), (g_vals, j_vals))
	return 0

phase_diagram_deltavsTc(p=0.1, q=0, n=9, dim=3, pool_size=30000, l_prec=21, rg_step=30, iter_num=30)