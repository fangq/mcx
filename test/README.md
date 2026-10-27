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

Each box-shaped case under `traj_cases/` runs with a fixed seed and one photon
per GPU thread (`-A 0 -t <photons> -T 1 -D M`), so every photon draws from its
own reproducible random number stream and the saved trajectory is
deterministic. MCX-CL derives the per-thread seeds in the same way.
`trajtest.py` then checks every recorded step against physical
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
REF=../../mcxcl/test/golden REFTOL=5e-2 MAXFORK=0.1 ./testtraj.sh
```

Single-precision round-off differs slightly between solvers and devices and
accumulates along long paths, so the comparison is done per photon:

- a photon with the same number of steps as the reference must match every
  step within `REFTOL` (`--tol`/`--wtol` of `trajtest.py`, 5e-2 mm by default)
- a photon reaching a reflect/escape decision with a ~1e-7 difference can take
  the other branch, after which the two random walks are unrelated; such
  forked photons are tolerated if they match the reference up to the fork and
  make up at most `MAXFORK` (`--max-fork`, 10% by default) of all photons.
  Each fork is printed with the distance of its last matched step to the
  nearest domain face.

The comparison with a solver's own golden files stays strict (`--tol 1e-4
--max-fork 0` for mcx).

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

## Mutation fuzzing: `fuzz/dfuzz.py`

```
cd test
python3 fuzz/dfuzz.py --bin ../bin/mcx -n 300                  # one binary: crashes, hangs, sanitizer errors
python3 fuzz/dfuzz.py --bin ../bin/mcx --ref ref/mcx -n 300    # differential: also compare with a reference build
```

Each case takes a random input from `builtin/`, mutates it field by field (type swaps,
extreme numbers, wrong array shapes, deleted keys) and in ways that reach the output
accumulation (output types, pattern and RF sources, time gates), adds random command-line
flags, and runs it with a few thousand photons. A case is reported when the binary crashes,
hangs or prints a sanitizer error. With `--ref`, it is also reported when the two builds exit
differently or, in deterministic runs (`-A 0` with a fixed thread count, so both simulate the
same photons), print different absorbed fractions or output totals (`--tol`, 1e-3 relative).
Reported cases are saved under `fuzzout/caseNNNN/` (input, arguments, logs); `--seed` makes
a campaign reproducible. Build the binaries with `-fsanitize=address,undefined` to catch
memory errors (see the header of `dfuzz.py`); it needs python3 with numpy.
