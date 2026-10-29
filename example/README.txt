= README for examples =

In this folder, you will find a number of examples to validate, test, or
demonstrate the functionalities of MCX. Each one is run by the run_*.sh (Linux
and macOS) or run_*.bat (Windows) script in its folder, which calls the mcx
binary in ../../bin; folders with a README.txt explain the procedures and how
to interpret the results.

* quicktest      - a homogeneous cube to check that mcx runs (also lists the GPUs)
* validation     - a semi-infinite medium compared with the diffusion solution
* benchmark      - the standard benchmarks: a homogeneous cube, and with a sphere
* sphbox         - a sphere inside a box, compared with an analytical solution
* srctest        - one input per source type (pencil, cone, planar, fourier, slit ...)
* multipattern   - several patterned sources simulated together ("photon sharing")
* multisrc       - several sources in one simulation (arrays of Pos/Dir in Optode.Source)
* masktest       - a volume built from a mask file
* shapetest      - volumes built from JSON shape constructs (spheres, boxes, layers)
* muamus         - continuously varying mua/mus (and g/n) per voxel
* svmcsphshells  - spherical shells with split-voxel MC (SVMC)
* polarized      - polarized light in a homogeneous slab, with cyclic boundaries
* rfreplay       - RF (frequency-domain) Jacobians by replaying detected photons
* skinvessel     - the skin-vessel benchmark of mcxyz
* cylfinger      - a cylindrical finger model (cylfinger665.json, cylfinger884.json)
* colin27        - the Colin27 adult brain atlas
* digimouse      - the Digimouse mouse atlas
* usc19-5        - the USC 19.5-year-old brain atlas
* rngcycle       - a test of the period of the random number generators
* speedtest      - the speed benchmark of the original MCX paper
* mcx_demo.mcxp  - a project file for MCX Studio

More examples, as MATLAB/Octave scripts, are in ../mcxlab/examples, and as
Python scripts in ../pmcx/examples.
