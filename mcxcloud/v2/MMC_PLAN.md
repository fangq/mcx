# Plan: MMC (mesh-based MC) support in MCX Cloud v2

Goal: extend the shared JSON schema + frontend so users can submit **MMC** simulations
(tetrahedral-mesh domains) alongside MCX (voxel domains), supporting a *safe subset* of
MMC's JSON input — not every MMC feature. Sources: `mmc/src/mmc_utils.c` (`mcx_loadjson`,
line ~1566) vs `mcx/src/mcx_utils.c`, compared construct-by-construct.

## 1. How the two formats relate

MMC reads the same five top-level sections; the *domain* is the only structural fork:

| Section | MCX | MMC | Verdict |
|---|---|---|---|
| `Shapes` | array of shape objects, or JData voxel volume | object with `MeshNode`/`MeshElem`/`InitElem` (section may also be named `Mesh`; `Shapes` is accepted as an alias, mmc_utils.c:1572) | disjoint by construction — a `oneOf` branch cleanly discriminates them, and the backend can detect the engine from the doc shape |
| `Session` | superset | subset + `RayTracer`, `BasisOrder`, `Checkpoints` | large shared core |
| `Forward` | `T0/T1/Dt` | same + `N0`, `Omega` | shared core |
| `Domain` | `Dim`, `MediaFormat`, `VolumeFile`, `OriginType`, `MieScatter`, … | only `Media`, `Step`, `LengthUnit` | `Media` identical (object rows or `[mua,mus,g,n]` rows) |
| `Optode` | `Source` + `Detector` | same, smaller key set | shared core |

Critically, **MMC has the same multi-source format** (`Pos/Dir/Param1/Param2` as arrays of
vectors, equal lengths, `_NaN_` sentinels in `Dir[3]`, `Source.ID` dispatch) — the v2
schema's multi-src support carries over unchanged.

Both parsers ignore unknown keys, so a merged schema is safe as long as the backend
routes each doc to the right engine.

## 2. Safe subset — include

**`Shapes` (mesh branch, new `oneOf` alternative "Tetrahedral mesh (MMC)")**
- `MeshNode` — JData `single` [N×3] (zlib/base64) or plain array of `[x,y,z]`
- `MeshElem` — JData `int32` [N×5] (4 node indices + region tag) or plain array; the
  parser also accepts N×11 second-order elements, but N×5 is enough for phase 1
- `InitElem` — integer ≥ 1, **required** (mmc errors without it; it cannot be derived
  cheaply server-side)
- heavy arrays flow through the existing blob normalization untouched (`normalize()`
  extracts any `_ArrayZipData_`/`_ArrayData_` node; the frontend `decodeJDataArray`
  already maps `int32`)

**`Session`** (shared, already in schema): `ID`, `Photons`, `RNGSeed`, `DoMismatch`,
`DoSaveVolume`, `DoNormalize`, `DoPartialPath`, `DoSpecular`, `DoDCS`, `DoSaveExit`,
`DoSaveSeed`, `DebugFlag` (integer form), `OutputType` restricted to the common letters
(see §4). Plus mmc-only scalars: `RayTracer`, `BasisOrder` (§2b).

**`Forward`**: `T0`, `T1`, `Dt` (identical). Plus mmc-only scalars: `N0`, `Omega` (§2b).

**`Domain`**: `Media` (identical), `LengthUnit` (identical), `Step` (mcx wants 3 elements,
mmc reads only the first — `[s,s,s]` satisfies both).

**`Optode.Source`**: `Pos`, `Dir`, `Param1`, `Param2` (incl. multi-src), `Type`
(restricted to MMC's 14 types for mesh docs: pencil…slit — MMC lacks `pattern3d`,
`pencilarray`, `hyperboloid`, `ring`), `ID`, `Pattern` (both JData and `{Nx,Ny,Nz,Data}`
forms parse identically).

**`Optode.Detector`**: array of `{Pos, R}` (identical).

## 2b. Exact schema additions (field by field)

**`Shapes` — third `oneOf` branch "Tetrahedral mesh (MMC)"** — object, required
`[MeshNode, MeshElem, InitElem]`:

| Field | Type | Notes |
|---|---|---|
| `MeshNode` | oneOf: JData object or plain array | JData form: `_ArrayType_` enum `["single"]`, `_ArraySize_` `[N,3]`, `_ArrayZipType_` `zlib\|gzip`, `_ArrayZipSize_`, `_ArrayZipData_`; plain form: array of `[x,y,z]` number triplets |
| `MeshElem` | oneOf: JData object or plain array | JData form: `_ArrayType_` enum `["int32","uint32"]` (parser matches `int32` via `strstr`), `_ArraySize_` `[N,5]` (or `[N,11]` second-order); plain form: array of 5-integer rows `[n1,n2,n3,n4,tag]`, 1-based node indices, `tag` ≥ 1 indexes `Domain.Media` |
| `InitElem` | integer ≠ 0, **required** | 1-based index of the tetrahedron enclosing the source; `-1` (schema default) triggers mmc's automatic enclosing-element search (`mesh_initelem`, mmc_mesh.c:1325) |

**`Session`** — two new optional scalars (mmc-only; harmless to mcx, which ignores them):

| Field | Type | Default | Notes |
|---|---|---|---|
| `RayTracer` | string enum `["g","p","h","b","s"]` | `"g"` | ray-tracing method: g = dual-grid (DMMC, voxel jnii output — renders in the existing preview), p = Plücker, h = Havel, b = Badouel, s = branchless Badouel; non-`g` methods emit per-node output (download-only until the mesh renderer lands, §4) |
| `BasisOrder` | integer enum `[0,1]` | `1` | fluence basis: 0 = piecewise-constant (per-element), 1 = piecewise-linear (per-node) |

**`Forward`** — two new optional scalars (mmc-only):

| Field | Type | Default | Notes |
|---|---|---|---|
| `N0` | number ≥ 0 | `1` | refractive index of the medium *outside* the mesh |
| `Omega` | number ≥ 0 | `0` | RF modulation **angular** frequency in rad/s (0 = CW); note mcx expresses the same physics as `Optode.Source.Frequency` in Hz — the two keys coexist without conflict, each engine reads its own |

**Nothing else changes**: `Domain.Media`/`Step`/`LengthUnit`, all of `Optode.Source`
(incl. multi-src) and `Optode.Detector`, and `Forward.T0/T1/Dt` are already in the schema
and parse identically in mmc.

## 3. Exclude (and why)

| Construct | Reason |
|---|---|
| `Mesh.MeshID` | references external `node_*.dat`/`elem_*.dat` files — meaningless in the cloud; embedded `MeshNode/MeshElem` is the only cloud-safe path |
| `Mesh.MeshROI` | implicit-MMC (iMMC) edge/node/face ROIs — advanced, small user base, defer |
| `Session.OutputFormat` beyond the shared enum (`ascii`, `bin`) | the worker enforces `-F jnii` (CLI overrides JSON in mmc) so the output pipeline/preview stay uniform |
| `Session.Checkpoints` | progressive checkpointing — irrelevant for short cloud jobs |
| `Optode.Detector.Dir/Param1/Param2` (mmc adjoint/wide-field) | niche; **also buggy upstream**: mmc_utils.c:2194/2212 parse `Optode.Detector.Param1/Param2` from the *Source* object (`FIND_JSON_OBJ(..., src)`), and `Detector.Dir` (:2350) is looked up on the detector *array*, which cJSON cannot resolve — worth fixing in mmc before exposing |
| any detector item shape other than exactly `{Pos, R}` | **upstream segfault** (found in production 2026-07-19): mmc_utils.c:2171 gates the per-item `Pos` lookup on `cJSON_GetArraySize(det) == 2`; a third member (e.g. `Dir`) leaves `pos` pointing at the Detector *array* and `pos->child->next->valuedouble` NULL-derefs with a single detector — mmc dies before flushing its banner (empty log). `checkLimits` rejects such docs for mesh jobs; the editor no longer emits a `Dir` column (removed from the Detector schema — table-format arrays render every schema property unconditionally, which is how the crash was triggered) |
| mcx-only keys in mesh docs (`MieScatter`, `IQUV`, `WaveLength`, `SrcNum`, `Weight`, `Frequency`, `AngleInverseCDF`, `BCFlags`, `SaveDataMask`, `MaxDetPhoton`, `MinEnergy`, `DoSaveRef`, `DoAutoThread`, `Dim`, `MediaFormat`, `VolumeFile`, `OriginType`, …) | silently ignored by mmc — the backend should *warn/reject* them for mesh docs so users are not misled |

## 4. Cross-engine gotchas (validation must know these)

- **`Photons`** is parsed with `valueint` in mmc → capped at 2^31−1. The cloud preview cap
  (5e8) is already below this; keep it.
- **`OutputType`**: shared letters `x f e j l p` (+ replay-only `r s a d u v w q`); mmc has
  no `m t b`. Mesh docs should restrict the enum accordingly.
- **`DebugFlag`** letters differ entirely (mcx `RMPT` vs mmc `SCBWDIOXATRPEM`); accept the
  integer form for mesh docs, or per-engine patterns.
- **`LengthUnit`** exists under both `Mesh` and `Domain` in mmc, and `Domain` wins (parsed
  later). Standardize on `Domain.LengthUnit`.
- **mmc jnii output shape** depends on the ray tracer and basis (mmc_utils.c:941,
  mmc_mesh.c:1728): `RayTracer g` (dual-grid DMMC, forces `BasisOrder 0`) writes a voxel
  grid `[nx,ny,nz,nt,srcnum]` with `_ArrayOrder_:"c"` — the existing voxel preview + 4D
  frame selector render it unchanged. Mesh-mode tracers write per-NODE values
  (`BasisOrder 1`, `nn` per gate) or per-ELEMENT values (`BasisOrder 0`, `ne` per gate),
  frames laid out `[slot][gate][vox]` with vox fastest. On GPU (OpenCL/CUDA) workers,
  non-`g` tracers silently promote to `s` (mmc_utils.c:3542). ⚠ upstream: for
  element-basis output `mcx_savedata` writes `Dim[0]=nodenum` although the data holds
  `ne` values per gate (mmc_utils.c:945) — the frontend therefore detects the basis from
  the decoded data length, not the header.

## 5. Implementation phases

**Phase A — schema + validation (small) — DONE**
1. ✅ "Tetrahedral mesh (MMC)" branch in the `Shapes` `oneOf` (MeshNode/MeshElem
   JData-or-array, required `InitElem`, default −1 = auto-detect); `Session.RayTracer`/
   `BasisOrder`, `Forward.N0`/`Omega`; `ascii`/`bin` added to the `OutputFormat` enum
   (rejected for mcx docs in `checkLimits`).
2. ✅ Backend `schema.js`: exported `detectEngine()` (`Shapes`/`Mesh` object with
   `MeshNode` → `mmc`); relaxed `Domain.required` to `[Media]` (mesh docs carry no
   `Dim`/`OriginType`); `checkLimits` now engine-aware: mesh caps (≤ 300k nodes /
   1.5M tets), mmc source-type + OutputType subsets, cross-engine physics keys rejected
   both ways (mesh: IQUV/WaveLength/MieScatter, Source.Frequency, AngleInverseCDF,
   InverseCDF, BCFlags; voxel: Omega, N0≠1, ascii/bin). Verified against
   `mmc/examples/sfdi2layer/t1.json` and `skinvessel/skinvessel_1.json` (validate,
   engine=mmc) with all mcx examples unregressed.

**Phase B — backend job routing + worker (moderate) — DONE**
3. ✅ `jobs.engine` column (migration 003, default `mcx`); `POST /jobs` stores
   `detectEngine(doc)`; scheduler reads it and passes it to `createMcxService`, which maps
   engine → image (`WORKER_IMAGE` / `WORKER_IMAGE_MMC`, default `fangqq/mmc:v2025.10`)
   and exports `ENGINE=` into the container. Redbird later = one map entry + env.
4. ✅ Worker script branch: `mmc -f input.json -s output -F jnii` — `-F jnii` enforced so
   the output pipeline stays uniform; **no `-M`/`-c` flags**, so the JSON
   `Session.RayTracer` (default `g`) and mmc's default GPU backend (opencl) apply;
   `CUDA_VISIBLE_DEVICES` constrains NVIDIA OpenCL the same way. Detected photons: mmc
   emits `output_detp.jdb` (binary BJData) — added to the upload glob.
   Verified locally: mesh/voxel submissions store `engine` = `mmc`/`mcx`; swarm
   end-to-end pending a deploy with the new code (the currently running API predates it).

**Phase C — frontend (moderate) — DONE**
5. ✅ Editor: comes free from the schema (`oneOf` switcher), same as Shapes-vs-volume.
6. ✅ Preview: new dependency-free `js/mesh.js` ports iso2mesh's `volface` (exterior
   surface = faces referenced by exactly one tet, outward winding) and `qmeshcut`
   (axis-aligned planar cuts; tri for 3 cut edges, quad cycle `[0,1,3,2]` for 4),
   node-tested against a 6-tet cube (12 boundary tris, exact cut areas on all axes)
   and the real skinvessel/sfdi2layer meshes (130k tets: surface 0.8 s once, 61 ms per
   cut). `preview.js` renders the tag-colored exterior surface and drives cross-sections
   from the existing X/Y/Z crop sliders: the surface is clipped by `clippingPlanes` and
   qmeshcut patches at each active crop plane reveal the interior regions. Output
   rendering: nothing to do (DMMC grid, §4).
7. ✅ Seeded `MMC_BUILTIN:skinvessel` (1.1k nodes/6.4k tets, 4 regions, disk src) and
   `MMC_BUILTIN:sfdi2layer` (21.7k nodes/130k tets, fourier SFDI src, InitElem −1
   auto-search) into the library (`db/seeds/mmc-*.json`); MeshNode/MeshElem JData blobs
   round-trip through the content-addressed store. (colin27_lzma cannot be seeded as-is:
   `lzma` is outside the schema's zlib/gzip `_ArrayZipType_` — re-export with zlib.)

**Mesh-valued output rendering (was Phase D) — DONE**
- The preview now renders per-node and per-element mmc outputs on the input mesh:
  `drawPreview` detects a mesh-valued output (input doc has `Shapes.MeshNode`, output
  lacks the `'c'` order tag, decoded length per frame equals `nn` → node basis or `ne`
  → element basis), log-scales each frame, and colors the volface surface + qmeshcut
  cross-sections through the active colormap (per-vertex interpolated values for node
  basis — qmeshcut's `cutvalue`; flat per-patch values for element basis). The 4D frame
  spinner steps gates/sources (recolor-only, no geometry rebuild) and the colormap
  picker re-luts live. Verified: qmeshcut reproduces linear fields exactly at cut
  points; basis detection is immune to the upstream `Dim[0]` bug.

**Phase D — later / out of scope now**
- `MeshROI` (iMMC), adjoint detectors, replay.

## 5b. Production shakeout findings (2026-07-19/20)

- **Per-node mmc-OpenCL matrix** (fangqq/mmc:v2025.10, same sfdi control input):
  kylin (GTX 980 Ti/1080, driver 470.239) ✅; erlang (GTX 1080 ×2, 470.182) ⚠ works but
  intermittently fails `clGetPlatformIDs` with `CL_PLATFORM_NOT_FOUND` (-1001) on cold
  start; neza (RTX 2080 + GTX 1050, 530.30) ❌ `clBuildProgram` fails on BOTH GPUs
  (`-c cuda` reports the same error from mmc_cl_host.c — the CUDA flag does not bypass
  the OpenCL host in this build); mobi unverified (image pull > 150 s). Mitigations:
  `WORKER_NODE_CONSTRAINT_MMC` + `node.labels.mmc==1` (kylin labeled), image pre-pull.
- **`mmc/examples` docs are not all CLI/GPU-ready** (both were withdrawn from the
  public library and replaced by the generated `MMC_BUILTIN:twolayerslab`):
  - `sfdi2layer/t1.json`: wide-field (fourier) sources need source-cap elements tagged
    **−1** (`mesh_srcdetelem`, mmc_mesh.c:397) but the embedded mesh only has tag-0
    void elements spanning z=9–33.7 → `srcelemlen=0`, photons launch dead, 0 % absorbed.
  - `skinvessel/skinvessel_1.json`: GPU kernel **hangs indefinitely** (even 1e4 photons
    exceed 40 s on the working kylin node) — disk source *outside* the mesh
    (z=−0.005) + tag-0 void elements apparently traps the OpenCL void-entry search.
  - schema note: MeshElem plain-array items now allow ≥ −2 so −1/−2 wide-field
    source/detector tags can be expressed once supported.

## 6. Open questions

1. Worker image: publish `fangqq/mmc` with CUDA/OpenCL support, or fold the mmc binary
   into the existing `fangqq/mcx` image (one image, one service template)?
2. Should mesh jobs share the single job queue/limits or get their own rate limits /
   photon cap (GPU mmc is still slower per photon than mcx)?
3. Verify (during Phase B) whether mmc's `srcnum>1` pattern-sharing grid output interleaves
   the pattern dim below x like mcx does — affects the preview frame extractor's gather
   path for mmc outputs.
