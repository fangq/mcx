import numpy as np

def detweight(detp, prop):
    """
    Recalculate the detected photon weight using partial path data and 
    optical properties (for perturbation Monte Carlo or detector readings)

    author: Qianqian Fang (q.fang <at> neu.edu)
    transated to python

    input:
        detp: the 2nd output from mcxlab. detp must be a dict
        prop: optical property list, as defined in the cfg['prop'] field of mcxlab's input
        unitinmm: voxel edge-length in mm, should use cfg['unitinmm'] used to generate detp; 
                  if ignored, assume to be 1 (mm)

    output:
        detw: re-caculated detected photon weight based on the partial path data and optical property table

    License: GPLv3, see http://mcx.space/ for details
    """

    if 'prop' in detp:
        prop = detp['prop']

    medianum = prop.shape[0]
    if medianum <= 1:
        raise ValueError('empty property list')

    if unitinmm is None:
        if 'unitinmm' in detp:
            unitinmm = detp['unitinmm']
        else:
            unitinmm = 1

    if isinstance(detp, dict):
        if 'w0' not in detp:
            detw = np.ones(detp['ppath'].shape[0])
        else:
            detw = detp['w0']

        for i in range(medianum - 1):
            detw *= np.exp(-prop[i + 1, 0] * detp['ppath'][:, i] * unitinmm)
    else:
        raise ValueError('the first input must be a dict with a key named "ppath"')

    return detw
