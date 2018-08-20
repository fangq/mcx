---------------------------------------------------------------------
  MCX Studio - A graphical user interface for MCX, MMC and MCX-CL
---------------------------------------------------------------------

Author:  Qianqian Fang <q.fang at neu.edu>
License: GNU General Public License version 3 (GPLv3)
Version: 0.9 (v2018)
Website: http://mcx.space

---------------------------------------------------------------------

Table of Content:

I.    Introduction
II.   Installation
III.  Using MCX Studio GUI
IV.   Remote Execution
V.    Acknowledgement
VI.   Reference

---------------------------------------------------------------------

I.  Introduction

MCXStudio is a light-weight graphical user interface (GUI) for MCX, MMC and MCX-CL. 
It gives user an intuitive way to browse and set parameters for different
simulations. It also provides a way to save, edit and reopen a set of
pre-defined simulations and share among different users.

MCXStudio is written in Object Pascal language using the open-source
Lazarus rapid application development (RAD) environment and LCL (Lazarus
component library). The OpenGL rendering of the GUI is provided by the
GLScene Library.

MCXStudio is cross-platform and can be run on Windows, Linux and Mac OS.
The GUI itself does not contain any Monte Carlo (MC) modeling codes. Instead,
it serves as a shell within which one can launch MC simulations by calling
externally compiled mcx, mmc and mcxcl binaries. 

MCXStudio contains many features that are designed to ease the pre-processing,
simulation and post-processing of a biomedical optics researcher. It is compatible
with mcx (NVIDIA GPU), mmc (CPU) and mcxcl (cross-vendor CPUs/GPUs). It has built-in
domain designer, source designer with convenient media/detector settings. It also
supports visualization of simulation output data (mc2, nii, hdr/img) by calling
MATLAB or Octave installed in your system. One can also launch remote MCX/MMC 
simulations using the Remote Execution panel.

---------------------------------------------------------------------

II.   Installation

MCXStudio binary is portable and can be directly executed without installation.
However, in order for MCXStudio to find mcx/mmc/mcxcl binaries, you should use
one of the following methods

# place mcx/mmc/mcxcl executables side-by-side with mcxstudio executable
# change system environment variable and add the paths to mcx/mmc/mcxcl to the PATH variable
# create a MCXSuite/ subfolder in the same directory storing mcxstudio, and \
extract mcx/mmc/mcxcl binary packags inside MCXSuite/ folder, so that one has \
the following folder structure

<pre>MCXStudio/
├── MCXSuite/
│   ├── mcx/bin/mcx
│   ├── mcxcl/bin/mcxcl
│   └── mmc/bin/mmc
├── mcxstudio
├── mcx
├── mmc
├── mcxcl
└── Output/
</pre>

the mcx/mmc/mcxcl files on the same level as mcxstudio can be a link or a dummy file
with that name (create using touch command in linux or mac) indicating that the module 
has been installed inside MCXSuite/. If you download the MCXStudio package, the
above folder structure already exist, and you can simply double click on the 
mcxstudio executable to start the program.

In order to use MATLAB or Octave for plotting the output in MCXStudio, you must
install either of the software in your system, and add the matlab/octave command
in the PATH environment variable, so that typing "matlab" or "octave" in the command
window starts the program directly. If you type these commands and get file not found
error, that means your OS can not find those progams in the search path.

Instructions on setting MATLAB can be found at

http://mcx.space/wiki/index.cgi?Workshop/MCX18Preparation/MethodA

For windows users, the first time to start MCXStudio, you may see a popup window
asking you that your system registry needs to be modified in order for MCX/MCXCL
to run more than 5 seconds. You must click "yes" and then reboot the computer in 
order for this setting to be effective. 

If you click yes and receive a permission error, you will have to quit MCXStudio, 
and then right-click on the executable and select "Run as administrator" instead.

Alternatively, one should open file browser, navigate into mcxcl/setup/win64 folder,
and right-click on the "apply_timeout_registry_fix.bat" file and select 
"Run as Administrator".

'''You must reboot your computer for this change to be effective!'''


---------------------------------------------------------------------------
III. Using MCX Studio GUI


Once launched, MCX Studio first automatically check if mcx
binary is in the search path, if so, the "GPU" button in the 
toolbar will be enabled. If you do not have NVIDIA GPUs and intend
to use mcxcl for your simulations, you may click on the "New" button
in the toolbar, and select the 3rd option (NVIDIA/AMD/Inte CPUs/GPUs MCX-CL).
After click on OK, and if mcxcl binary presents in your folders, 
the GPU button will also become active.

It is recommended to click on this button once, and see if you can see 
a list of GPUs and their parameters printed in the floating output window. 
If you are able to see this information, your system is ready
to run MCX (or mcxcl) simulations. If you get error messages or not able
to see any usable GPU, please check the drivers and the binary files.

If your system has been properly configured, you can now add new simulations 
by clicking the "New" button. MCX Studio will ask you to give a session
ID string for this new simulation. Then you are allowed to adjust the parameters
based on your needs. Once you finish the adjustment, you should click the 
"Verify" button to see if there are missing settings. If everything looks
fine, the "Run" button will be activated. Click on it once will start your
simulation. If you want to abort the current simulation, you can click
the "Stop" button.

You can create multiple tasks with MCX Studio by hitting the "New"
button again. The information for all session configurations can
be saved as a project file (with .mcxp extension) by clicking the
"Save" button. You can load a previously saved project file back
to MCX Studio by clicking the "Load" button.

In both the unzipped binary packags of mcx and mcxcl, under the folder named
"examples", one can find a pre-created project file (*_examples.mcxp) 
for MCXStudio. Loading these files using the "Load" button allows you to
see these preconfigured simulation.

On Windows, once you start MCXStudio once, the .mcxp suffix becomes 
associated with mcxstudio executable. To load a .mcxp file, you can simply
double click on the file, mcxstudio will be started automatically.

---------------------------------------------------------------------------
IV. Remote Execution

MCXStudio not only supports running mmc/mmc/mcxcl simulations on your local
computer, but also running those remotely on a more powerful server/cluster.
Additional information and video tutorial regarding remote execution can be 
found at

http://mcx.space/wiki/index.cgi?Workshop/MCX18Preparation/MethodB

To use the remote execution feature, one must download and install the 
needed MC simulator (mcx/mmc/mcxcl) in the remote server first, and make sure
those executables can be found by your PATH environment variable. On the client
side, no need to change any software structure. Only the remote copy of mcx/mmc
or mcxcl will be used. They are not restricted by the processors present on your
local computer.

The remote execution is performed using the "ssh" command. One must enable
the ssh sever on the remote server in order to run commands remotely, and install
ssh/scp client on the local computer (your laptop, for example).

On the server side (where mcx/mmc/mcxcl simulations are executed), if it is a 
Linux computer, you can install ssh server by using 

 sudo apt-get install openssh-server
 
or

 sudo dnf install openssh-server
 
If your server is a Mac machine, you should enable SSH access by clicking on
System Preference\Sharing\Remote Login and add your username.

If your remote server is a Windows machine, you can install and enable 
the builtin ssh sever using the instructions below 

https://winscp.net/eng/docs/guide_windows_openssh_server

On the client-side (where mcxstudio is executed, like your laptop), you should
have ssh/scp client software ready. If you use Linux or MacOS, ssh and scp commands are 
typically included by default. If you use Windows, you may download pscp.exe
and plink.exe from the following link

https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html

and place the two executables in the same folder as mcxstudio.exe (or add to system PATH).

=== Key-based SSH authentication ===

Running mcx/mmc/mcxcl simulations remotely requires one to login a remote server 
using ssh. Typically, one must type password to login. This is the default process
when using MCXStudio. However, if one wants to avoid typing password every time making
a connection, one can setup ssh key-based authentication by following the steps in the
below link 

http://www.linuxproblem.org/art_9.html

=== Clusters with shared file system ===

If your client computer and server computer have a shared file path (i.e., the folder
storing mcxstudio can be both accessed from both computers with the same path,
typical for an NFS-based Linux cluster), then, you should check the "Shared file system"
in the Remote Execution section of MCXStudio window.

If the simulation is launched on a shared file system, your simulation can involve
local files (.bin, meshes etc), and all outputs are stored under MCXStudio/Output/ folder.
You may directly call the menus under the Plot button in MCXStudio to plot the volume
data.

=== Remote servers without shared file system ===

If your client and sever do not have shared file folders, you should uncheck the "Shared file
system". This way, MCXStudio will send all needed setting info via ssh command and create
the output in the ~/Output directory of the server.

In order for a user to plot the output data, one must click on the Plot button in MCXStudio
toolboar, and select one of the Download ... menus to retrieve the file back to your client
drive in order to visualize the results. You must have scp commands available for the file 
transfer to work.

---------------------------------------------------------------------------
V.    Acknowledgement

The icon set was created by Qianqian Fang, with a style inspired by 
the "Uniform" icon-set 
URL: https://github.com/0rAX0/uniform-icon-theme

The JSON shape editor uses an function "ShpwJSONData"  
adapted from the "jsonviewer" example in Lazarus.
URL: http://wiki.freepascal.org/fcl-json#From_JsonViewer

---------------------------------------------------------------------------
VI.   Reference

[1] Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon \
Migration in 3D Turbid Media Accelerated by Graphics Processing Units,"
Optics Express, vol. 17, issue 22, pp. 20178-20190 (2009).

[2] Leiming Yu, Fanny Nina-Paravecino, David Kaeli, Qianqian Fang, \
"Scalable and massively parallel Monte Carlo photon transport simulations \
for heterogeneous computing platforms," J. Biomed. Opt. 23(1), 010504 (2018).

If you use MCX in your research, the author of this software would like
you to cite the above papers in your related publications.

