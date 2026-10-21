# MCX tests

## Feature tests: `testmcx.sh`

```
cd test
./testmcx.sh [extra mcx options]
```

Runs about 50 checks against `../bin/mcx`: command line flags, built-in
benchmarks, JSON input/output, absorbed-fraction values of the cube60
benchmarks, boundary conditions, detectors, replay, trajectory output, and
memory checks (valgrind, compute-sanitizer) when those tools are available.
It needs a CUDA (or ROCm) GPU. The last check runs `testtraj.sh` when python3
with numpy is available.

## Step-by-step trajectory tests: `testtraj.sh`

```
cd test
./testtraj.sh [-u] [extra mcx options]
```

Each box-shaped case under `traj_cases/` runs with one GPU thread and a fixed
seed (`-A 0 -t 1 -T 1 -E 12345 -D M`), so the saved photon trajectory is
reproducible. `trajtest.py` then checks every recorded step against physical
invariants:

- every step is inside the domain box
- each photon starts at the source position
- the photon weight never increases
- in a homogeneous, index-matched medium, the weight lost between two
  scattering events equals `exp(-mua*L)` for the segment length `L`
- each photon ends on the domain boundary or with near-zero weight

The trajectory is also compared step by step with `golden/<case>.json`. Use
`-u` to regenerate the golden files after an intended change to the photon
transport.

The golden files are solver-neutral: one list of `[x,y,z,w]` steps per photon,
sorted by photon id. The same cases are used by MCX-CL and MMC on the same box
domain. To compare mcx with golden data from another solver, set `REF` to its
golden folder:

```
REF=../../mcxcl/test/golden REFTOL=1e-3 ./testtraj.sh
```

`trajtest.py` needs only python3 and numpy. It reads the `*_traj.jdt` files
written with `-F jnii`:

```
python3 trajtest.py session_traj.jdt --case traj_cases/homogeneous.json \
        [--ref golden/homogeneous.json] [--save my.json --solver mcx]
```

## Host unit tests: `make unittest`

```
cd src
make unittest
```

Builds `unit/test_raytrace.cu` with nvcc. It calls the ray-tracing helpers of
`mcx_core.cu` on the CPU (voxel intersection `hitgrid`, Fresnel `reflectcoeff`,
Snell `transmit`, scattering `rotatevector`, the xorshift128+ RNG) and compares
them with analytic results. Only the CUDA toolkit is needed; no GPU is required.
