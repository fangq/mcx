=============================================================
          Getting started with MCX Studio (v2026)
=============================================================

MCX Studio is installed. Start it from the Start menu, from the
desktop shortcut if you asked for one, or by double-clicking an
MCX input file (.json, through "Open with") or a project from
the first MCX Studio (.mcxp).

-------------------------------------------------------------
Your first simulation, in five steps
-------------------------------------------------------------

1. Click New, or Benchmarks on the toolbar to start from one of
   the standard examples that come with MCX.
2. Under Simulator, pick your hardware (NVIDIA, AMD or Intel GPU,
   or the CPU), how the domain is described (a voxel grid,
   primitive shapes, or a tetrahedral or surface mesh) and the
   kind of medium. MCX Studio picks the simulator that suits that
   choice and says which one, under "Runs as". In Expert mode,
   Compute > Simulator runs another one instead (MCX-CL on an
   NVIDIA card, say, or MMC with a given backend), and Extra
   options adds your own command-line options.
3. Describe the domain:
     - Shapes: add a grid, spheres, boxes, cylinders, slabs or
       layers. A new shape takes the next free medium label, and
       the Optical properties table gains a row for it, copied
       from the last one -- edit mua, mus, g and n there.
     - Volume data or Mesh: load a volume file or mesh, or paste
       one as JSON (the edit dialog shows the expected layout).
4. Set the light source and detectors, the photon count and the
   time gates. Next and Back walk through the settings in order;
   the Wizard/Expert button switches between the key settings
   and every setting. F1 explains the setting you are on.
5. Click Run. The progress and the simulator's messages appear
   in the Log pane, and the result opens in the 3-D preview.

-------------------------------------------------------------
Looking at results
-------------------------------------------------------------

The 3-D preview shows the domain before a run and the fluence
after it. Drag to rotate, scroll to zoom, and click a shape to
select it in the Shapes list. The Display bar under the view sets:

  Show          the simulation output, or the input domain it was
                computed in -- switch back and forth at any time
  Colour map, Rendering (maximum intensity or translucent),
  Log scale, Threshold, Opacity, and the X/Y/Z range sliders
  that cut into the volume

Photon paths (saved with "-D M") are drawn too, with a range of
photon numbers to pick out single paths. Open result... shows any
.jnii/.bnii file, and Save image... writes the view to a PNG.

The JSON tab is the input file exactly as the simulator reads it,
and the Command tab the command line that runs it -- both can be
copied, to run the same simulation from a terminal or a script.

-------------------------------------------------------------
Arranging the window
-------------------------------------------------------------

The Sections, Views and Log panes can be dragged by their headers
and docked along any edge, or floated. Reset layout puts them
back. The Theme button switches between the system colours and
the light, dark and ocean themes.

-------------------------------------------------------------
What else was installed
-------------------------------------------------------------

  MCXSuite\<simulator>\bin
        mcx, mcxcl, mmc, vkmmc and umcx, for use from a terminal
        or script. If you kept "Add to PATH variable", they can
        be run from any new command prompt. Windows has an mmc of
        its own (the Management Console), which comes first: run
        MMC by its full path, MCXSuite\mmc\bin\mmc, or with photon.
  MATLAB
        the MATLAB/Octave toolboxes mcxlab, mcxlabcl, mmclab,
        umcxlab, iso2mesh, redbird and zmat. If you kept "Add the
        toolboxes to the MATLAB and Octave paths", every MATLAB and
        Octave started from now on finds them, with no addpath needed:
        they are listed in the MATLABPATH and OCTAVE_PATH
        environment variables (yours, or every user's for an
        install for all users), and taken out again on uninstall.
        The Octave toolboxes need Octave 10 or later.
  photon
        a single command-line launcher for mcx, mcxcl and mmc
        (needs Perl); "photon --info" explains it.
  Python
        redbirdpy. The simulators' Python modules are installed
        separately with:  pip install pmcx pmcxcl pmmc

New sessions, and runs of the examples, are written to the
MCXOutput folder in your user folder (%USERPROFILE%\MCXOutput).

-------------------------------------------------------------
Hardware and drivers
-------------------------------------------------------------

  MCX     needs an NVIDIA GPU and a current NVIDIA driver.
  MCX-CL  needs an OpenCL driver: the GPU vendor's driver, or an
          OpenCL runtime for the CPU.
  vkmmc   needs a Vulkan driver with ray query: NVIDIA RTX,
          AMD Radeon RX 6000 or later, or Intel Arc.
  MMC and uMCX run on the CPU without extra drivers.

Devices on the toolbar lists what each simulator can see on this
computer -- the quickest check when a run will not start.

-------------------------------------------------------------
Help and news
-------------------------------------------------------------

  Website and documentation  https://mcx.space
  Wiki                       https://mcx.space/wiki/
  User forum                 https://groups.google.com/g/mcx-users
  Bug reports                https://github.com/fangq/mcx/issues

The Start menu folder has links to the website, wiki and forum.
This page is saved as GETTING_STARTED.txt in the install folder.
