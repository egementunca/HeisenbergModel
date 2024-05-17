import numpy as np
import rgFlowVacancy as flow

def normalize_pool(pool):
    size, l_prec = pool.shape
    norm_vals = np.amax(np.abs(pool), axis=1)
    for i in range(size):
        pool[i,::] /= norm_vals[i]
    return pool

def checkPhase(J, p, n, q, dim, pool_size, l_prec, rg_step, iter_num):
    data = []
    g_left = 0
    g_right = 0.5

    for i in range(iter_num):
        g_new = (g_left + g_right) / 2
        f, g = flow.rgTrajectoryVacancy(J=J, g=g_new, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)
        g_check = np.mean(g[-1])
        if g_check < 0:
            g_left = g_new
            print(rf"g:{g_new} goes $\to -\infty$")
            data.append([g_new, "minus"])
        else:
            g_right = g_new
            print(rf"g:{g_new} goes $\to +\infty$")
            data.append([g_new, "plus"])
    
    return data

def phase_diagram_deltaVertical(p, q, n, dim, pool_size, l_prec, rg_step, iter_num):
    j_vals = np.linspace(50, 1, 99)
    g_vals = np.zeros(len(j_vals)) 
    
    for i, J in enumerate(j_vals):
        g_crit = checkPhase(J=J, p=p, n=n, q=q, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step, iter_num=iter_num)[-1][0]
        g_vals[i] = g_crit
    np.save("./deltaVerticalAF_p{}.npy".format(p), (j_vals, g_vals))
    return 0

phase_diagram_deltaVertical(p=0.5, q=0, n=9, dim=3, pool_size=30000, l_prec=21, rg_step=30, iter_num=30)