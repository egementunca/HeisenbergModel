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
		jNew = (jFerro + jNematic)/2
		f = flow.rgTrajectoryVacancy(J=jNew, g=g, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)
		check_pool = normalize_pool(f[-1])
		avg_lfc = np.mean(abs(check_pool),axis=0)
		if (sum(avg_lfc) < 15.37):
			jNematic = jNew
			data.append([jNew, 'nematic'])
		else:
			jFerro = jNew
			data.append([jNew, 'ferromagnetic'])

	return data

def phase_diagram_pvsTc_nematic(g, q, n, dim, pool_size, l_prec, rg_step, iter_num):
	p_vals = np.linspace(0.04, 0.105, 14)
	j_vals = np.zeros(len(p_vals))
	
	for i,p in enumerate(p_vals):
		j_crit = checkPhase(g=g, p=p, n=n, q=q, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step, iter_num=iter_num)[-1][0]
		j_vals[i] = j_crit
	np.save('nematic_border_delta.npy', (p_vals, j_vals))
	return 0

phase_diagram_pvsTc(g=-2, q=0, n=9, dim=3, pool_size=30000, l_prec=21, rg_step=30, iter_num=30)