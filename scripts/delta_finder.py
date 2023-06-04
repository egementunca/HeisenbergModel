import numpy as np
import RenormGroupFunctions as rg
import rgFlow as flow

def checkPhase3(J, p, n, q, dim, pool_size, l_prec, rg_step, iter_num):
	data = []

	gDisorder = .4
	gOrder = .8

	fOrder = flow.rgTrajectoryVacancy(J=J, g=gOrder, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)

	if sum(abs(fOrder[-1,0]))==1:
		data.append([gOrder, 'disorder'])
		return data

	for i in range(iter_num):
		gNew = (gOrder + gDisorder)/2
		f = flow.rgTrajectoryVacancy(J=J, g=gNew, p=p, q=q, n=n, dim=dim, pool_size=pool_size, l_prec=l_prec, rg_step=rg_step)
		avg_lfc = np.sum(abs(f[-1]),axis=0)/pool_size
		print(sum(avg_lfc))
		print(sum(avg_lfc)>1.1)

		if sum(abs(f[-1,0]))==1:
			gDisorder = gNew
			data.append([gNew, 'disorder'])
			print(f'{gNew} is disorder')
		else:
			gOrder = gNew
			data.append([gNew, 'order'])
			print(f'{gNew} is order')

	return data

data = checkPhase3(J=1/.14, p=.5, n=9, q=0, dim=3, pool_size=30000, l_prec=21, rg_step=25, iter_num=25)
np.save('data_deltacrit.npy', data)