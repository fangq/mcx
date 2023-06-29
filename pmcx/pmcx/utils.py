import numpy as np


def detweight(detp, prop):
    """
    Recalculate the detected photon weight using partial path data and 
    optical properties (for perturbation Monte Carlo or detector readings)

    author: Qianqian Fang (q.fang <at> neu.edu)
    translated to python: Kuznetsov Ilya

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


def cwdref(detp, cfg):
    """
    Compute CW diffuse reflectance from MC detected photon profiles.

    author: Shijie Yan (yan.shiji <at> northeastern.edu)
    translated to python: Kuznetsov Ilya

    input:
        detp: profiles of detected photons
        cfg:  a dictionary. Each element of cfg defines
              the parameters associated with a MC simulation.

    output:
        dref: CW diffuse reflectance at detectors

    this file is part of Monte Carlo eXtreme (MCX)
    License: GPLv3, see http://mcx.sf.net for details
    see Yao2018
    """

    unitinmm = 1
    if 'unitinmm' in cfg:
        unitinmm = cfg['unitinmm']

    det_weight = detweight(detp, cfg['prop'])
    detnum = len(np.unique(detp['detid']))
    detweightsum = np.zeros(detnum)

    for i in range(len(detp['detid'])):
        index = int(detp['detid'][i]) - 1
        detweightsum[index] += det_weight[i]

    area = np.pi * (cfg['detpos'][:, 3] * unitinmm) ** 2
    dref = detweightsum / area / cfg['nphoton']

    return dref