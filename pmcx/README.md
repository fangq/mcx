![](http://mcx.space/img/mcx18_banner.png)

# PMCX - Python bindings for Monte Carlo eXtreme photon transport simulator

- Copyright: (C) Matin Raayai Ardakani (2022-2023) <raayaiardakani.m at northeastern.edu> 
and Qianqian Fang (2019-2023) <q.fang at neu.edu>
- License: GNU Public License V3 or later
- Version: 0.0.13
- URL: https://pypi.org/project/pmcx/
- Github: https://github.com/fangq/mcx

[![Build Status](https://travis-ci.com/fangq/mcx.svg?branch=master)](https://travis-ci.com/fangq/mcx)

This module provides a Python binding for Monte Carlo eXtreme (MCX).
For other binaries, including the standalone executable and the MATLAB bindings, see [our website](http://mcx.space).

Monte Carlo eXtreme (MCX) is a fast photon transport simulation software for 3D 
heterogeneous turbid media. By taking advantage of the massively parallel 
threads and extremely low memory latency in a modern graphics processing unit 
(GPU), MCX is capable of performing Monte Carlo (MC) photon simulations at a 
blazing speed, typically hundreds to a thousand times faster than a fully 
optimized CPU-based MC implementation.

## How to Install

* PIP: ```pip install pmcx``` see https://pypi.org/project/pmcx/

## Runtime Dependencies
* **NVIDIA GPU Driver**: A CUDA-capable NVIDIA GPU and driver is required to run MCX. An up-to-date driver is recommended.
The binary wheel distributed over pip runs on NVIDIA drivers with CUDA 10.1 support on Windows, CUDA 9.2 support on Linux, and
CUDA 10.2 support on macOS, respectively. For more details on driver versions and their CUDA support, see the 
[CUDA Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). 
To download the latest driver for your system, see the 
[NVIDIA Driver Download Page](https://www.nvidia.com/download/index.aspx).
**You shouldn't need to have CUDA toolkit installed**. MCX is built with the static CUDA runtime library. 
* **Python**: Python 3.6 and newer is required. **Python 2 is not supported**.
* **numpy**: Used to pass/receive volumetric information to/from pmcx. To install, use either conda or pip 
package managers: `pip install numpy` or `conda install numpy`
* (optional) **jdata**: Only needed to read/write JNIfTI output files. To install, use pip: `pip install jdata` 
on all operating systems; For Debian-based Linux distributions, you can also install to the system interpreter 
using apt-get: `sudo apt-get install python3-jdata`. See https://pypi.org/project/jdata/ for more details. 
* (optional) **bjdata**: Only needed to read/write BJData/UBJSON files. To install, run `pip install bjdata` 
on all operating systems; For Debian-based Linux distributions, you can also install to the system interpreter 
using apt-get: `sudo apt-get install python3-bjdata`. See https://pypi.org/project/bjdata/ for more details. 
* (optional) **matplotlib**: For plotting the results. To install, run either `pip install matplotlib` or
`conda install matplotlib`

## Build Instructions

### Build Dependencies
* **Operating System**: Windows and Linux are fully supported; For building MCX on macOS, OSX 10.13 (High Sierra) and 
older are highly recommended since 10.13 was the last version of macOS with NVIDIA CUDA support, and matching the CUDA 
compiler version with the C/C++ compiler shipped with Xcode is easier. Newer macOS versions can be used for building MCX, 
but need to have System Integrity Protection disabled prior to installing the CUDA toolkit due to the NVIDIA installer copying
its payload under the ```/Developer``` directory under root.
* **NVIDIA CUDA Toolkit**: CUDA 7.5 or newer is required. On macOS, 10.2 is the last available CUDA version.
For details on how to install CUDA, see the [CUDA Download Page](https://developer.nvidia.com/cuda-downloads). 
The NVIDIA GPU driver of the target system must support the selected CUDA toolkit.
* **Python Interpreter**: Python 3.6 or above. The ```pip``` Python package manager and the ```wheel``` package (available
  via ```pip```) are not required but recommended.
* **C/C++ Compiler**: CUDA Toolkit supports only the following compilers:
  * GNU GCC for Linux-based distributions.
  * Microsoft Visual Studio C/C++ Compiler for Windows.
  * Apple Clang for macOS, available via Xcode. The last Xcode version supported by CUDA 10.2 is 10.3. If using an OSX 
  version higher than 10.15 it can be downloaded and installed from [Apple's Developer Website](https://developer.apple.com/download/) 
  with an Apple ID. After installation, select the proper Xcode version from the commandline, and set the ```SDKROOT```
  environment variable:
    ```zsh
    sudo xcode-select -s /Applications/Xcode_10.3.app/Contents/Developer/
    export SDKROOT=/Applications/Xcode_10.3.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    ```
  
  Refer to each OS's online documentations for more in-depth information on how to install these compilers.
  Note that the version of the C/C++ compiler used must be supported by the CUDA toolkit version. If not, compilation
  will fail with an error notifying you of this problem. See the [CUDA Installation Guides](https://developer.nvidia.com/cuda-toolkit-archive)
  for more details.
* **OpenMP**: The installed C/C++ Compiler should have support for [OpenMP](https://www.openmp.org/). 
  GCC and Microsoft Visual Studio compiler support OpenMP out of the box. Apple Clang, however, requires manual 
  installation of OpenMP libraries for Apple Clang. The easiest way to do this is via the [Brew](https://brew.sh/) package
  manager, preferably after selecting the correct Xcode version:
  ```zsh
    brew install libomp
    brew link --force libomp
  ```

* **CMake**: CMake version 3.15 and later is required. Refer to the [CMake website](https://cmake.org/download/) for more information on how to download.
  CMake is also widely available on package managers across all operating systems.
  Additionally, on Windows, make sure **Visual Studio's C++ CMake tools for Windows** is also installed by selecting its option
  during installation.
* **Zlib Compression Development Headers**: On Linux, this is generally available via the built-in package manager. For 
  example, on Debian-based distributions like Ubuntu it is available via ```apt``` under the name ```zlib1g-dev```. On
  macOS, brew provides it under the name ```zlib```. No packaged versions of Zlib are available for windows, therefore it must be
  downloaded manually and added to the CMake environment variables in your working Powershell session:
  ```powershell
    curl.exe --retry 3 -kL https://cytranet.dl.sourceforge.net/project/gnuwin32/zlib/1.2.3/zlib-1.2.3-lib.zip --output zlib.zip
    Expand-Archive .\zlib.zip -DestinationPath zlib\
    $env:CMAKE_INCLUDE_PATH=$PWD\zlib\include
    $env:CMAKE_LIBRARY_PATH=$PWD\zlib\lib
  ```

### Build Steps
1. Ensure that ```cmake```, ```nvcc``` (NVIDIA CUDA Compiler) and the C/C++ compiler are all located over your ```PATH```.
This can be queried via ```echo $env:PATH``` on Windows or ```echo $PATH``` on Linux. If not, locate them and add their folder to the ```PATH```.

2. Clone the repository and switch to the ```pmcx/``` folder:
    ```bash
        git clone --recursive https://github.com/fangq/mcx.git
        cd mcx/pmcx
    ```

3. Either run ```python setup.py install``` or ```pip install .``` to directly install, or run ```pip wheel .``` to only
build the Python wheel without installing it.


## How to use

The PMCX module is easy to use. You can use the `pmcx.gpuinfo()` function to first verify
if you have NVIDIA/CUDA compatible GPUs installed; if there are NVIDIA GPUs detected,
you can then call the `run()` function to launch a photon simulation.

A simulation can be defined conveniently in two approaches - a one-liner and a two-liner:

* For the one-liner, one simply pass on each MCX simulation setting as positional
argument. The supported setting names are compatible to nearly all the input fields
for the MATLAB version of MCX - [MCXLAB](https://github.com/fangq/mcx/blob/master/mcxlab/mcxlab.m))

```python3
import pmcx
import numpy as np
import matplotlib.pyplot as plt

res = pmcx.run(nphoton=1000000, vol=np.ones([60, 60, 60], dtype='uint8'), tstart=0, tend=5e-9, 
               tstep=5e-9, srcpos=[30,30,0], srcdir=[0,0,1], prop=np.array([[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]]))
res['flux'].shape

plt.imshow(np.log10(res['flux'][30,:, :]))
plt.show()
```

* Alternatively, one can also define a Python dict object containing each setting
as a key, and pass on the dict object to `pmcx.run()`

```python3
import pmcx
import numpy as np
cfg = {'nphoton': 1000000, 'vol':np.ones([60,60,60],dtype='uint8'), 'tstart':0, 'tend':5e-9, 'tstep':5e-9,
       'srcpos': [30,30,0], 'srcdir':[0,0,1], 'prop':[[0,0,1,1],[0.005,1,0.01,1.37]]}
res = pmcx.run(cfg)
```
