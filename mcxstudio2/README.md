MCX Studio 2
============

A graphical front end for [MCX](https://mcx.space), MCX-CL and MMC: set up a
Monte Carlo photon transport simulation, run it, and look at what came out.

It is a rewrite, beside the original `mcxstudio/` rather than in place of it.
The two do not share code and can be installed together.


Why a rewrite
-------------

The first Studio grew a field at a time for fifteen years, and it showed:

* **Four serialisers.** The same setting was written out once for the
  command line, once for the JSON, once for the project file, and once for
  the preview. Adding a setting meant four edits, and mcx grew faster than
  the GUI, so the GUI fell behind.
* **A format of its own.** A `.mcxp` project held the fields of the window
  rather than the simulation. Nothing else could read it.
* **Hard-coded colours.** Legible on the grey desktops of 2009 and not since.
* **GLScene.** A large dependency, pinned to an old Lazarus, for a picture of
  a box with some spheres in it.

So the rewrite is organised around one idea: **a simulation is a `.json`
file, and nothing else.** There is no project format. The document is the
file mcx reads, kept as JSON in memory, and every control on the window is
one row of one table that says where in that file it lives.

```pascal
(Ctl:'edPhotons'; Path:'Session.Photons'; Kind:mkInt; Level:mlWizard;
 Backends:[]; Domains:[]; Min:1; Max:0; Choices:''; EnableIf:''),
```

Adding a setting is adding a row and dropping a control on the form with the
matching name. Loading, saving, enabling, hiding, the wizard/expert split and
the self-test all follow from the table.


Building
--------

Needs Lazarus 2.2 or newer and FPC 3.2. On Debian or Ubuntu:

    sudo apt-get install lazarus
    make

`make` writes `bin/mcxstudio`. It uses a project-local Lazarus config
(`.lazpcp`) so it neither touches nor needs your own IDE configuration, and
works for an unprivileged user and in CI.

    make            # release build
    make debug      # with debug info and range checks
    make test       # both layers of tests, below
    make install PREFIX=$HOME/.local

The Lazarus directory is found from whichever `lazbuild` is on PATH, so the
LCL always matches the tool about to use it. On a machine with more than one
version, name the one you want:

    make LAZBUILD=/usr/lib/lazarus/2.2.0/lazbuild

and on a layout this does not know about, name the directory outright:

    make LAZDIR=/opt/lazarus       # or export LAZARUSDIR

If nothing is found, nothing is passed and lazbuild uses its own config,
which is what a normally installed Lazarus wants anyway.

On macOS the widgetset is Cocoa, and

    make app

builds `bin/MCXStudio.app`: one universal binary (arm64 and x86_64, each
compiled on its own and joined with `lipo`; FPC's macOS installer carries the
compilers of both) in a bundle with its icon and an ad-hoc signature, which
arm64 Macs require. `make CPU=x86_64` builds a single slice for the other
processor.


Tests
-----

Two layers, because they need different things:

    make test-doc    # links mcxdoc alone: no LCL, no widgetset, no display
    make test-bind   # drives the real form; needs a widgetset, not a desktop

`test-doc` is 221 checks on the document: that a file opened and written
straight back is byte-identical, that a float prints as short as it can
without changing value, that a path like `Domain.Media[2].mua` resolves and
is created on demand, that BJData arrays decode.

`test-bind` opens every `.json` under `../example` -- 46 real files -- drives
each one through every binding, writes it back and compares. It runs headless
under `xvfb-run`.

    make check-paths

reports keys mcx's own parser reads that no control on the form reaches. Not
enforced: mcx grows keys faster than a GUI grows controls, and a red build
for that would only teach people to ignore it.


What is where
-------------

| unit | what it is |
| --- | --- |
| `mcxdoc.pas` | the document, and **the binding table** -- start here |
| `mcxmain.pas` | the window: navigator, cards, docking, running |
| `mcxrun.pas` | finding a backend, asking it about devices, running it |
| `mcxjd.pas` | JData and BJData: reading `.jnii`, `.bnii`, `.jdat` |
| `mcxmcxp.pas` | opening a first-Studio project (`.mcxp`) as mcx JSON |
| `mcxdomain.pas` | a volume or mesh from a file or pasted JSON, as the JData mcx/mmc read |
| `mcxdatadlg.pas` | the Edit Volume / Edit Mesh dialog |
| `mcxmesh.pas` | tetrahedral meshes, and the surface of one |
| `mcxgl.pas` | OpenGL 3.3 core: matrices, shaders, meshes, volumes |
| `mcxview.pas` | the 3-D view: shapes, mesh, source, fluence, photon paths |
| `mcxsoft.pas` | the same view drawn on the processor, when there is no OpenGL 3.3 |
| `mcxglprobe.pas` | asks GLX for a 3.3 core context before the view does |
| `mcxdisp.pas` | the display controls under the picture |
| `mcxhelp.pas` | what each setting does, for the hint and for F1 |
| `mcxtheme.pas` | the three colours a theme is |
| `mcxdpi.pas` | display scaling, which gtk2 will not do for us |
| `mcxicons.pas` | the icon set, rasterised from `../icons/svg` |


Using it
--------

Open a `.json`, or start from one of mcx's built-in benchmarks. The sections
down the left are the order a simulation is described in; **F1** over any
setting says what it does and which mcx option it is. The Command tab shows
the line that will be run and the JSON tab shows the file that will be
written, so nothing the GUI does is hidden.

A second argument opens a result beside the input:

    mcxstudio benchmark1.json benchmark1.jnii
    mcxstudio benchmark1.json benchmark1_traj.jdat

`--run` opens a simulation and runs it straight away, as the Run button
would; `--processor` picks what to run it on (`nvidia`, `amd`, `intel` or
`cpu`), which together with the domain decides the solver -- mcx, mcx-hip,
mcxcl, or mmc and its `-c`. With `--quit` as well, the window closes when the
run ends, the simulator's output goes to stdout and its exit code becomes the
program's, so a script or a CI job can use the GUI's own run path:

    mcxstudio --run benchmark1.json                       # run, and keep the window
    mcxstudio --run --processor intel benchmark1.json     # the same input on mcxcl
    xvfb-run -a mcxstudio --run --quit --processor cpu onecube.json   # mmc -c sse, headless

A run writes its output beside its input.  An input with nowhere writable to
put it -- a new document, or an example installed under the program folder --
is saved, with the volume or mesh files it names, to
`MCXOutput/<simulator>sessions/<session>/`, as the first Studio did.  That
folder is beside the program, or under the home folder with `--user`, which
is how the Windows installer's shortcuts start it.

The first Studio's options still work: `-u`/`--user` as above,
`-p`/`--project f.mcxp` (or just naming the `.mcxp`) opens a session of an
old project -- converted as `utils/mcxp2json.py` would, the session picked
from a list when there are several -- and `-l`/`--lang` and
`-r`/`--registry` are accepted and ignored: this Studio is not translated,
and the installer owns the file associations.

A simulator is looked for where the first Studio looked, in its order --
`MCXSuite/<simulator>/bin` beside the program, beside the program,
`~/MCXStudio`, `<simulator>/bin` beside the program -- then in the source
trees a developer builds in, then on `PATH`.

`--snapshot out.png` writes the 3-D view of the input -- and of the result,
if one is named after it -- to a PNG and exits, which is a thumbnail for a
gallery or a check in CI that the view draws:

    xvfb-run -a mcxstudio --snapshot qtest.png qtest.json qtest.jnii


Drawing without OpenGL
----------------------

The 3-D view draws with OpenGL 3.3 when it can. When it cannot -- a remote X
display, a virtual machine, a driver older than 3.3 -- it draws the same
picture on the processor (`mcxsoft.pas`): the triangles and lines the GL
path would upload, rasterised with a depth buffer, and the volume ray-cast
ray for ray as the GLSL shader casts it, split across every processor and
marched at half resolution with a third of the steps while it is being
turned. The method is the one the figure widget of mimamo uses.

On X11 the choice is made before any window exists, by asking GLX for a 3.3
core context the way the GL control will (`mcxglprobe.pas`), so a display
without one never sees an error. `--software`, or `MCXSTUDIO_SOFTWARE=1` in
the environment, picks the software renderer outright, for a display where
even asking misbehaves. Help > About says which renderer is drawing.

Coming from the first Studio, convert your projects:

    ../utils/mcxp2json.py mcx_demo.mcxp -o sessions/

One `.json` per session. What does not convert is what was never part of the
simulation -- thread counts, which GPU, the remote-execution settings -- and
it says so rather than dropping them silently.


Licence
-------

GPL v3 or later, as mcx itself is.
