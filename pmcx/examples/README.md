# pmcx examples

Python ports of the MATLAB/Octave demo scripts in `mcxlab/examples`. Each
`demo_*.py` mirrors the `.m` script of the same name: the same configurations,
the same printed summaries and roughly the same figures.

All simulations go through `pmcx.mcxlab(cfg)`, which applies the same
pre- and post-processing as `mcxlab.m`. For example, it converts the
`outputtype` units, unpacks the detected-photon data and the trajectories,
and normalizes replay Jacobians. `pmcx.run(cfg)` calls only the binary
module.

## Running

```
cd pmcx/examples
python3 demo_mcxlab_basic.py
```

A few demos load the sample data shipped in `mcxlab/examples`, by relative path
(`../../mcxlab/examples/*.mat`), so run them from inside this folder of an mcx
source tree: `demo_colin27_atlas`, `demo_digimouse_sfdi`, `demo_fullhead_atlas`
and `demo_svmc_brain19_5`.

## Dependencies

- all demos: `pmcx`, `numpy`, `matplotlib`
- `scipy`: `.mat` loading, convolution and special functions
- `scikit-image`: the marching-cubes step of `pmcx.svmc()` (the `demo_svmc_*` demos)
- `iso2mesh` (`pip install iso2mesh`): `pmcx.preview()`, and the meshes of the Redbird comparisons
- `redbirdpy` (`pip install redbirdpy`): the FEM diffusion reference in `demo_validate_mcx_*`

## Helper modules

- `sphdiffusion.py`: analytical diffusion solution for a sphere in an infinite
  medium, ported from the MMC sphere-diffusion toolbox. Used by
  `demo_sphere_cube_subpixel`.
- `rbfemhelper.py`: the slab mesh, nodal Jacobians and cut-plane interpolation
  shared by the Redbird validation demos.

## Not ported

The GPU benchmarking scripts `bench_reseedlimit.m`, `mcx_gpu_benchmarks.m` and
`mcx_gpu_contest.m` are not ported.

## Validation

Every `pmcx.mcxlab` call in every port was checked against the matching `mcxlab`
call in Octave. Both runs used the same configuration with fewer photons. The
comparison covered the absorbed fraction, the output sums (flux, diffuse
reflectance, Jacobians) and the number of detected photons, all within
Monte Carlo noise. Derived results, such as `pmcx.svmc()`, the
sphere-diffusion solution and the Redbird Jacobians, match their MATLAB
counterparts: `pmcx.svmc()` exactly, the sphere-diffusion solution to about
1e-11, and the Redbird FEM solutions and Jacobians (redbirdpy vs redbird-m) to
about 0.1%.
