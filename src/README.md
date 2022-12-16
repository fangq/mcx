![](http://mcx.space/img/mcx18_banner.png)

# PMCX - Python bindings for Monte Carlo eXtreme photon transport simulator

- Copyright: (C) Matin Raayai Ardakani (2022) <raayaiardakani.m@northeastern.edu> and Qianqian Fang (2019-2022) <q.fang at neu.edu>
- License: GNU Public License V3 or later
- Version: 0.0.6
- URL: https://github.com/fangq/mcx

[![Build Status](https://travis-ci.com/fangq/mcx.svg?branch=master)](https://travis-ci.com/fangq/mcx)

This module provides a Python binding to 

Monte Carlo eXtreme (MCX) is a fast photon transport simulation software for 3D 
heterogeneous turbid media. By taking advantage of the massively parallel 
threads and extremely low memory latency in a modern graphics processing unit 
(GPU), MCX is capable of performing Monte Carlo (MC) photon simulations at a 
blazing speed, typically hundreds to a thousand times faster than a fully 
optimized CPU-based MC implementation.

## How to install

* PIP: run `pip install pmcx` see https://pypi.org/project/pmcx/
* Other binaries: download from http://mcx.space


Dependencies:
* **numpy**: PIP: run `pip install numpy` or `sudo apt-get install python3-numpy`
* (optional) **jdata**: PIP: run `pip install jdata` or `sudo apt-get install python3-jdata`, see https://pypi.org/project/jdata/, only needed to read/write JNIfTI output files
* (optional) **bjdata**: PIP: run `pip install bjdata` or `sudo apt-get install python3-bjdata`, see https://pypi.org/project/bjdata/, only needed to read/write BJData/UBJSON files
* (optional) **matplotlib**: PIP: run `pip install matplotlib`, often needed when one wants to plot the results

Replacing `pip` by `pip3` if you are using Python 3.x. If either `pip` or `pip3` 
does not exist on your system, please run
```
    sudo apt-get install python3-pip
```
Please note that in some OS releases (such as Ubuntu 20.04), python2.x and python-pip 
are no longer supported.

One can also install this module from the source code. To do this, you first
check out a copy of the latest code from Github by
```
    git clone https://github.com/fangq/mcx.git
    cd mcx/pmcx
```
then install the module to your local user folder by
```
    python3 setup.py install --user
```
or, if you prefer, install to the system folder for all users by
```
    sudo python3 setup.py install
```
Please replace `python` by `python3` if you want to install it for Python 3.x instead of 2.x.

Instead of installing the module, you can also import the jdata module directly from 
your local copy by cd the root folder of the unzipped pyjdata package, and run
```
   import pmcx
```

## How to use

The PMCX module is easy to use. You can use the `gpuinfo()` function to first verify
if you have NVIDIA/CUDA compatible GPUs installed; if there are NVIDIA GPUs detected,
you can then call the `run()` function to launch a photon simulation.

A simulation can be defined conveniently in two approaches - a one-liner and a two-liner.

For the one-liner, one simply pass on each MCX simulation setting as positional
argument. The supported setting names are compatible to nearly all the input fields
for the MATLAB version of MCX - [MCXLAB](https://github.com/fangq/mcx/blob/master/mcxlab/mcxlab.m))

```
import pmcx
import numpy as np
import matplotlib.pyplot as plt

res=pmcx.mcx(nphoton=1000000,vol=np.ones([60,60,60],dtype='uint8'), tstart=0, tend=5e-9, tstep=5e-9, srcpos=[30,30,0], srcdir=[0,0,1], prop=np.array([[0,0,1,1],[0.005,1,0.01,1.37]]));
res['flux'].shape

plt.imshow(np.log10(res['flux'][30,:, :]))
plt.show()
```

Alternatively, one can also define a Python dict object containing each setting
as a key, and pass on the dict object to `pmcx.run()`

```
cfg={'nphoton':1000000, 'vol':np.ones([60,60,60],dtype='uint8'), 'tstart':0, 'tend':5e-9, 'tend':5e-9, 'srcpos':[30,30,0], 'srcdir':[0,0,1], 'prop':[[0,0,1,1],[0.005,1,0.01,1.37]]};
res=pmcx.run(cfg);
```


## Utility


## Test
