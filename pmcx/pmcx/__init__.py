"""PMCX - Python bindings for Monte Carlo eXtreme photon transport simulator

Example usage:

# To list available GPUs
import pmcx
pmcx.gpuinfo()

# To run a simulation
res = pmcx.run(nphoton=1000000, vol=np.ones([60, 60, 60], dtype='uint8'),
               tstart=0, tend=5e-9, tstep=5e-9, srcpos=[30,30,0],
               srcdir=[0,0,1], prop=np.array([[0, 0, 1, 1], [0.005, 1, 0.01, 1.37]]))
"""

try:
    from _pmcx import gpuinfo, run
except ImportError:  # pragma: no cover
    print('the pmcx binary extension (_pmcx) is not compiled! please compile first')
    raise


__version__ = '0.0.10'

__all__ = ('gpuinfo', 'run')

