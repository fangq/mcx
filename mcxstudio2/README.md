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

`make` writes `bin/mcxstudio2`. It uses a project-local Lazarus config
(`.lazpcp`) so it neither touches nor needs your own IDE configuration, and
works for an unprivileged user and in CI.

    make            # release build
    make debug      # with debug info and range checks
    make test       # both layers of tests, below
    make install PREFIX=$HOME/.local


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
| `mcxmesh.pas` | tetrahedral meshes, and the surface of one |
| `mcxgl.pas` | OpenGL 3.3 core: matrices, shaders, meshes, volumes |
| `mcxview.pas` | the 3-D view: shapes, mesh, source, fluence, photon paths |
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

    mcxstudio2 benchmark1.json benchmark1.jnii
    mcxstudio2 benchmark1.json benchmark1_traj.jdat

Coming from the first Studio, convert your projects:

    ../utils/mcxp2json.py mcx_demo.mcxp -o sessions/

One `.json` per session. What does not convert is what was never part of the
simulation -- thread counts, which GPU, the remote-execution settings -- and
it says so rather than dropping them silently.


Licence
-------

GPL v3 or later, as mcx itself is.
