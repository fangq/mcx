unit mcxhelp;

{ mcxstudio2 - what each setting means.

  One table, keyed by the control's name, exactly as mcxdoc's binding table
  is.  The same string serves as the hover hint and as what F1 shows, because
  two texts for one setting is two texts to keep true, and the one that is
  wrong is always the one nobody reads.

  Each entry names mcx's own command-line option for the setting.  That is
  not decoration: the Command tab shows the line that will be run, the manual
  and every paper's methods section are written in terms of those letters, and
  a GUI that hides them leaves you unable to carry what you learned here
  anywhere else. }

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

{ The help for one control, or '' when there is none. }
function McxHelpFor(const ACtl: string): string;

{ The help for one control wrapped for a message box, with the control's own
  caption as a title. }
function McxHelpText(const ACtl, ACaption: string): string;

{ The same text folded to AColumns, for a hover hint.  The LCL's hint window
  is as wide as the widest line it is given, so an unfolded paragraph becomes
  one line running off the edge of the screen. }
function McxWrap(const AText: string; AColumns: Integer = 68): string;

{ What Param1 and Param2 hold for one source type.  Empty when that type
  takes no parameters, which is the signal to hide the row entirely.

  mcx packs up to four numbers into each, and what they mean is decided
  entirely by the type -- "Param1" on its own is a box you have to read the
  manual to fill in.  The meanings are read off mcx_core.cu's launch
  switch, not off the old GUI, because two of the types are newer than it. }
procedure McxSrcParams(const AType: string; out ALabel1, ALabel2: string);

{ What F1 shows when nothing in particular is focused. }
function McxHelpOverview: string;

implementation

type
  TMcxHelpEntry = record
    Ctl : string;
    Text: string;
  end;

const
  Helps: array[0..42] of TMcxHelpEntry = (
    (Ctl:'rgBackend'; Text:'Which simulator runs. MCX needs an NVIDIA card; ' +
      'MCX-CL runs the same simulation on any OpenCL device, including the ' +
      'CPU; MMC works on a tetrahedral mesh instead of voxels. The choice ' +
      'decides which settings below apply.'),
    (Ctl:'rgDomainKind'; Text:'Where the geometry comes from: a voxel volume ' +
      'in a separate file, shapes described in this file (-P/--shapes), or a ' +
      'tetrahedral mesh for MMC.'),
    (Ctl:'cbMediaFormat'; Text:'How each voxel of the volume is stored ' +
      '(-K/--mediabyte). "byte" is one label per voxel indexing the medium ' +
      'table; the float forms carry optical properties per voxel instead and ' +
      'make the table unnecessary.'),

    (Ctl:'edT0'; Text:'Start of the time window, in seconds. Photons ' +
      'arriving before this are not recorded.'),
    (Ctl:'edT1'; Text:'End of the time window, in seconds. 5e-9 covers the ' +
      'whole decay for most tissue geometries.'),
    (Ctl:'edDt'; Text:'Width of one time gate, in seconds. The output has ' +
      '(T1-T0)/Dt gates; making Dt equal to T1-T0 gives a single ' +
      'continuous-wave image and is much smaller to store.'),

    (Ctl:'edSessionID'; Text:'Names the run and every file it writes ' +
      '(-s/--session): <id>.jnii for the fluence, <id>_detp.jdat for ' +
      'detected photons, <id>_traj.jdat for photon paths.'),
    (Ctl:'edPhotons'; Text:'How many photons to launch (-n/--photon). ' +
      'Noise falls as the square root, so ten times the photons is about ' +
      'three times smoother, and ten times the run.'),
    (Ctl:'rgOutFormat'; Text:'The file the fluence is written as ' +
      '(-F/--outputformat). jnii is JSON with the array encoded in it and is ' +
      'readable anywhere; bnii is the same thing in binary and is smaller ' +
      'and faster; mc2 is the old headerless float dump.'),
    (Ctl:'edSeed'; Text:'Random seed (-E/--seed). A fixed positive number ' +
      'makes the run repeat exactly, which is what you want when comparing ' +
      'two settings; -1 takes the seed from the clock. A file name here ' +
      'replays the seeds saved by a previous run.'),
    (Ctl:'cbOutType'; Text:'What is accumulated in the output ' +
      '(-O/--outputtype). Fluence rate is the usual one. The Jacobians are ' +
      'for reconstruction and are computed by replaying a saved run rather ' +
      'than from a fresh simulation.'),

    (Ctl:'ckMismatch'; Text:'Reflect and refract at boundaries between media ' +
      'of different refractive index (-b/--reflect). Off means every ' +
      'boundary is treated as matched, which is faster and wrong wherever ' +
      'n actually differs -- tissue against air, most obviously.'),
    (Ctl:'ckNormalize'; Text:'Divide the output by the number of photons ' +
      'launched (-U/--normalize), so the result does not change when you ' +
      'ask for more photons. Turn it off only if you want raw counts.'),
    (Ctl:'ckSaveVolume'; Text:'Write the fluence volume (-S/--save2pt). Off ' +
      'is for runs that only want detected photons, and saves writing a ' +
      'large array nobody reads.'),
    (Ctl:'ckSaveDetp'; Text:'Record every photon that reaches a detector ' +
      '(-d/--savedet), with its path length in each medium. This is what ' +
      'makes replay, time-resolved fitting and Jacobians possible.'),
    (Ctl:'ckSaveRef'; Text:'Also record diffuse reflectance, as negative ' +
      'values in the boundary voxels just outside the domain ' +
      '(-X/--saveref).'),
    (Ctl:'ckSaveExit'; Text:'Add each detected photon''s exit position and ' +
      'direction to the detected-photon file (-x/--saveexit).'),
    (Ctl:'ckSaveSeed'; Text:'Save the random seed of every detected photon ' +
      '(-q/--saveseed) so those exact photons can be replayed later to ' +
      'compute a Jacobian.'),
    (Ctl:'ckSpecular'; Text:'Account for the specular reflection at the ' +
      'first surface (-V/--specular). Off launches the full weight into the ' +
      'tissue as if nothing bounced off it.'),
    (Ctl:'ckDCS'; Text:'Record momentum transfer for each detected photon ' +
      '(-m/--momentum), which is what diffuse correlation spectroscopy ' +
      'needs.'),

    (Ctl:'edDim'; Text:'The volume''s size in voxels, x y z. With shapes ' +
      'this is the grid they are rasterised onto; with a volume file it must ' +
      'match the file.'),
    (Ctl:'edUnit'; Text:'The edge length of one voxel in millimetres ' +
      '(-u/--unitinmm). Everything else -- mua, mus, source position, ' +
      'detector radius -- is in these units, so changing it rescales the ' +
      'whole problem.'),
    (Ctl:'ckOriginType'; Text:'Whether the first voxel''s corner is at ' +
      '[0,0,0] or at [1,1,1] (-z/--srcfrom0). It shifts the source and ' +
      'detectors by half a voxel; set it to match whatever produced your ' +
      'coordinates.'),
    (Ctl:'edVolumeFile'; Text:'The voxel volume to load. A binary dump in ' +
      'the format chosen above, or a .jnii/.nii holding the array.'),

    (Ctl:'cbSrcType'; Text:'The shape of the light source. Each type reads ' +
      'Param1 and Param2 differently -- the preview draws what the current ' +
      'numbers actually mean, which is the quickest way to check them.'),
    (Ctl:'edSrcPos'; Text:'Where the source sits, x y z, in voxel units. It ' +
      'may be outside the volume: the photon is then launched towards it ' +
      'along the direction below.'),
    (Ctl:'edSrcDir'; Text:'Which way the source points, as a vector -- it ' +
      'does not have to be normalised. A fourth number, where a source type ' +
      'takes one, is the focal length: positive converges, negative ' +
      'diverges, zero is collimated.'),
    (Ctl:'edSrcParam1'; Text:'The first four numbers the source type takes. ' +
      'What they mean depends on the type: a radius for a disk, the two ' +
      'edge vectors for a planar source, the cone half-angle for a cone.'),
    (Ctl:'edSrcParam2'; Text:'The second four numbers the source type takes, ' +
      'where it needs more than four.'),
    (Ctl:'edSrcFreq'; Text:'Modulation frequency in Hz, for frequency-domain ' +
      'runs. Zero is continuous wave.'),
    (Ctl:'edSrcNum'; Text:'How many patterns a pattern source carries. More ' +
      'than one simulates them all in a single pass, sharing the photons ' +
      'between them.'),
    (Ctl:'edSrcWavelen'; Text:'Wavelength in nm. Recorded in the output and ' +
      'used by the polarised (Stokes) modes.'),

    (Ctl:'ckAutoThread'; Text:'Let mcx choose how many threads and how big a ' +
      'block to use (-A/--autopilot). It asks the driver what the device ' +
      'is; it is right nearly always and the two boxes below are for the ' +
      'times it is not.'),
    (Ctl:'edThread'; Text:'Number of threads (-t/--thread). Each runs one ' +
      'photon at a time, so this is how many are in flight at once.'),
    (Ctl:'edBlock'; Text:'Threads per block (-T/--blocksize). A tuning knob ' +
      'for the GPU''s occupancy; 64 suits most cards.'),
    (Ctl:'edWorkload'; Text:'How to split the photons between several ' +
      'devices (-W/--workload), one number each, in proportion. Only ' +
      'meaningful when more than one device is selected.'),

    (Ctl:'edBC'; Text:'What happens at each of the six faces ' +
      '(-B/--bc): six letters in the order -x +x -y +y -z +z, then six more ' +
      'saying which faces detect. _ absorbs, r reflects, c is cyclic, m is a ' +
      'mirror.'),
    (Ctl:'cgDebug'; Text:'Extra output from the run (-D/--debug). "Record ' +
      'photon trajectories" writes <id>_traj.jdt, which this window draws ' +
      'when the run finishes. A progress bar is always asked for, whatever ' +
      'is ticked here, because it is how the window knows how far along a ' +
      'run is.'),
    (Ctl:'edMaxJump'; Text:'How many photon positions to keep when ' +
      'trajectories are recorded (--maxjumpdebug). mcx''s own default is ' +
      'ten million, which is a 200 MB file that takes longer to write than ' +
      'the simulation took to run. Half a million is about ten megabytes ' +
      'and already more paths than can be told apart on screen.'),
    (Ctl:'cgSaveMask'; Text:'Which fields each detected photon carries ' +
      '(-w/--savedetflag). Every one you add multiplies the size of the ' +
      'detected-photon file by roughly the number of media.'),
    (Ctl:'edMaxDetp'; Text:'How many detected photons to keep ' +
      '(-H/--maxdetphoton). The buffer is allocated up front, so a large ' +
      'number costs memory whether or not it fills.'),
    (Ctl:'edMinEnergy'; Text:'The weight at which a photon enters the ' +
      'roulette (-e/--minenergy). Lower follows photons further into the ' +
      'tail and costs time; 0 disables the roulette entirely.'),
    (Ctl:'edRootPath'; Text:'The folder the input is read from and the ' +
      'output written to (--root). Blank means the folder this file is in.')
  );

function McxWrap(const AText: string; AColumns: Integer): string;
var
  Words: TStringList;
  i, Used: Integer;
begin
  Result := '';
  Used := 0;
  Words := TStringList.Create;
  try
    Words.Delimiter := ' ';
    Words.StrictDelimiter := True;
    Words.DelimitedText := StringReplace(AText, LineEnding, ' ',
      [rfReplaceAll]);
    for i := 0 to Words.Count - 1 do
    begin
      if Words[i] = '' then Continue;
      if (Used > 0) and (Used + 1 + Length(Words[i]) > AColumns) then
      begin
        Result := Result + LineEnding;
        Used := 0;
      end
      else if Used > 0 then
      begin
        Result := Result + ' ';
        Inc(Used);
      end;
      Result := Result + Words[i];
      Inc(Used, Length(Words[i]));
    end;
  finally
    Words.Free;
  end;
end;

function McxHelpFor(const ACtl: string): string;
var
  i: Integer;
begin
  for i := 0 to High(Helps) do
    if SameText(Helps[i].Ctl, ACtl) then Exit(Helps[i].Text);
  Result := '';
end;

function McxHelpText(const ACtl, ACaption: string): string;
begin
  Result := McxHelpFor(ACtl);
  if Result = '' then Exit(McxHelpOverview);
  if ACaption <> '' then
    Result := ACaption + LineEnding + LineEnding + Result;
end;

procedure McxSrcParams(const AType: string; out ALabel1, ALabel2: string);
begin
  ALabel1 := '';
  ALabel2 := '';
  { pencil, isotropic and arcsine take none and fall through to ''. }
  if AType = 'cone' then
    ALabel1 := 'Half angle (rad):'
  else if AType = 'gaussian' then
    ALabel1 := 'Waist radius, wavelength:'
  else if AType = 'zgaussian' then
    ALabel1 := 'Angular variance (rad):'
  else if AType = 'disk' then
    ALabel1 := 'Radius, inner radius:'
  else if AType = 'ring' then
    ALabel1 := 'Outer, inner radius, start, end angle:'
  else if AType = 'hyperboloid' then
    ALabel1 := 'Waist radius, focal distance, Rayleigh range:'
  else if (AType = 'line') or (AType = 'slit') then
    ALabel1 := 'Far end (x, y, z):'
  else if AType = 'planar' then
  begin
    ALabel1 := 'First edge vector Vx:';
    ALabel2 := 'Second edge vector Vy:';
  end
  else if AType = 'pattern' then
  begin
    ALabel1 := 'Edge vector Vx, then Nx:';
    ALabel2 := 'Edge vector Vy, then Ny:';
  end
  else if AType = 'pattern3d' then
    ALabel1 := 'Pattern size (Nx, Ny, Nz):'
  else if AType = 'fourier' then
  begin
    ALabel1 := 'Edge vector Vx, then kx:';
    ALabel2 := 'Edge vector Vy, then ky:';
  end
  else if AType = 'fourierx' then
  begin
    ALabel1 := 'Edge vector Vx, then |Vy|:';
    ALabel2 := 'kx, ky, phase shift, depth:';
  end
  else if AType = 'fourierx2d' then
  begin
    ALabel1 := 'Edge vector Vx, then |Vy|:';
    ALabel2 := 'kx, ky, x phase, y phase:';
  end
  else if AType = 'pencilarray' then
  begin
    ALabel1 := 'Edge vector Vx, then count:';
    ALabel2 := 'Edge vector Vy, then count:';
  end;
end;

function McxHelpOverview: string;
begin
  Result :=
    'MCX Studio' + LineEnding + LineEnding +
    'The sections down the left are the order a simulation is described in: ' +
    'which simulator, what the tissue is, where the light goes in and where ' +
    'it is measured, what to write out, and what to run it on.' + LineEnding +
    LineEnding +
    'Press F1 over any setting for what it does and which mcx option it is. ' +
    'The Command tab shows the line that will be run; the JSON tab shows the ' +
    'file that will be written.';
end;

end.
