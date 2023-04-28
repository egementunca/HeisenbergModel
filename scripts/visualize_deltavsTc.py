import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 1.5

plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8

fig, axs = plt.subplots(1,1,figsize=(16,9),dpi=120,)
for e in ['x', 'y']:
    axs.tick_params(axis=e,which='major',size=8,width=1,direction='in',top='on',right='on')

axs.set_ylabel(r'Temperature 1/J', fontsize='large')
axs.set_xlabel(r'$\frac{\Delta}{J}$', fontsize='large')

data = np.load('./data/deltavsTc_p0.0.npy')
data2 = np.load('./data/deltavsTc_p0.5.npy')

axs.plot(data[0], 1/data[1], c='mediumblue', linewidth=2, label='p=0')
axs.plot(data2[0], 1/data2[1], c='red', linewidth=2, label='p=0.5')

data3 = np.load('./data/deltavsTc_p0.5_anomaly.npy')
axs.scatter(data3[0], 1/data3[1], c='red', linewidth=2, label='p=0.5')

plt.xlim(-5,2)
plt.ylim(0.02,1)

plt.legend(loc='upper right')
plt.tight_layout()

#plt.savefig('./figures/deltavsTc.png')
#plt.savefig('./figures/deltavsTc.eps', format='eps')
plt.show()


