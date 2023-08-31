uname -a
python3 --version
python3 -m pip install --upgrade pmcx numpy matplotlib jdata
python3

import pmcx
import numpy as np

# print gpus
pmcx.gpuinfo()
gpus=pmcx.gpuinfo()
gpus[0]
gpus[1]

# method 1: define a simple simulation using a dict
cfg={}
cfg['nphoton']=1e5
cfg['vol']=np.ones([60,60,60], dtype='uint8')
cfg['tstart']=0
cfg['tend']=5e-9
cfg['tstep']=5e-9
cfg['srcpos']=[30,30,0]
cfg['srcdir']=[0,0,1]
cfg['prop']=[[0,0,1,1],[0.005,1,0.01,1.37]]

# run the simulation
res=pmcx.run(cfg)

# analyzing the result
res.keys()

res['flux'].shape

# plotting the data
import matplotlib.pyplot as plt
plt.imshow(np.log10(res['flux'][30,:,:]))
plt.colorbar()
plt.show()

# method 2: run a simulation with a two-liner

cfg = {'nphoton': 1e5, 'vol':np.ones([60,60,60],dtype='uint8'), 'tstart':0, 'tend':5e-9, 'tstep':5e-9, 'srcpos': [30,30,0], 'srcdir':[0,0,1], 'prop':[[0,0,1,1],[0.005,1,0.01,1.37]]}
res=pmcx.run(cfg)

# method 3: run a simulation with a one-liner
res = pmcx.run(nphoton=1000000, vol=np.ones([60, 60, 60], dtype='uint8'), tstart=0, tend=5e-9, tstep=5e-9, srcpos=[30,30,0], srcdir=[0,0,1], prop=np.array([[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]]))