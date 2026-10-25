{ mcxstudio2 - a graphical front end for MCX, MMC and MCX-CL.

  The order of the calls in Main is load-bearing; each one says why. }
program mcxstudio2;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  Interfaces, Forms, SysUtils, Graphics,
  mcxdpi, mcxicons, mcxmain, mcxview, mcxrun;

{$R *.res}

var
  Failures: Integer;
  { The command line, once the options are taken out of it: the files, in
    order, and what --run, --quit and --processor asked for. }
  Files: array of string;
  DoRun, DoQuit: Boolean;
  Processor, Snapshot: string;

{ mcxstudio [--run [--quit]] [--processor nvidia|amd|intel|cpu] [--software]
            [input.json [result]]

  The two files are positional, as they always were; the options may come
  anywhere.  An unknown option is an error rather than a file name, so a typo
  does not quietly open a blank window. }
procedure ParseCommandLine;
var
  i: Integer;
  A: string;
begin
  SetLength(Files, 0);
  { An environment variable as well, for a desktop entry or a remote session
    where the command line is not the user's to change. }
  if GetEnvironmentVariable('MCXSTUDIO_SOFTWARE') = '1' then McxForceSoftware := True;
  DoRun := False;
  DoQuit := False;
  Processor := '';
  Snapshot := '';
  i := 1;
  while i <= ParamCount do
  begin
    A := ParamStr(i);
    if A = '--run' then DoRun := True
    else if A = '--quit' then DoQuit := True
    else if A = '--software' then McxForceSoftware := True
    else if A = '--snapshot' then
    begin
      Inc(i);
      Snapshot := ParamStr(i);
    end
    else if A = '--processor' then
    begin
      Inc(i);
      Processor := LowerCase(ParamStr(i));
    end
    else if (A = '--help') or (A = '-h') then
    begin
      WriteLn('usage: mcxstudio [--run [--quit]] [--processor nvidia|amd|intel|cpu] [--software] [input.json [result]]');
      WriteLn('  --run        run the simulation once the window is up, as the Run button does');
      WriteLn('  --quit       with --run: exit when it ends, with its exit code, printing its output');
      WriteLn('  --processor  what to run it on; with the domain, picks mcx, mcxcl, mcx-hip or mmc');
      WriteLn('  --snapshot f.png  write the 3-D view of the input (and result) to a PNG, and exit');
      WriteLn('  -p, --project f.mcxp  open a session of a first-Studio project (also: just name it)');
      WriteLn('  -u, --user   keep sessions that have no writable folder of their own under');
      WriteLn('               ~/MCXOutput rather than beside the program');
      WriteLn('  --software   draw the 3-D view on the processor instead of OpenGL (also MCXSTUDIO_SOFTWARE=1);');
      WriteLn('               used anyway when OpenGL 3.3 is not available');
      WriteLn('  --self-test [dir], --dump-icons [file.png]: the test and build helpers');
      Halt(0);
    end
    { The first Studio's flags.  --user keeps sessions under the user's home
      folder (see McxWorkFolder), which is what the installer's shortcuts
      ask for.  --lang is taken and ignored -- this Studio is not translated
      -- because shortcuts and file associations an earlier installer made
      still pass it after an upgrade puts this executable in their place. }
    else if (A = '--user') or (A = '-u') then McxUseUserFolder := True
    else if Copy(A, 1, 7) = '--lang=' then
    else if (A = '--lang') or (A = '-l') then Inc(i)
    { -p/--project opens a first-Studio project, as it did there; a .mcxp
      named on its own does the same. }
    else if (A = '--project') or (A = '-p') then
    begin
      Inc(i);
      SetLength(Files, Length(Files) + 1);
      Files[High(Files)] := ParamStr(i);
    end
    else if Copy(A, 1, 10) = '--project=' then
    begin
      SetLength(Files, Length(Files) + 1);
      Files[High(Files)] := Copy(A, 11, MaxInt);
    end
    { -r/--registry re-made the .mcxp association from inside the first
      Studio; the installer owns associations now. }
    else if (A = '--registry') or (A = '-r') then
    else if (Copy(A, 1, 1) = '-') and (Length(A) > 1) then
    begin
      WriteLn(StdErr, 'mcxstudio: unknown option ', A, ' (see --help)');
      Halt(2);
    end
    else
    begin
      SetLength(Files, Length(Files) + 1);
      Files[High(Files)] := A;
    end;
    Inc(i);
  end;
  if DoQuit and not DoRun then
  begin
    WriteLn(StdErr, 'mcxstudio: --quit only applies with --run');
    Halt(2);
  end;
  if DoRun and ((Length(Files) = 0) or not FileExists(Files[0])) then
  begin
    WriteLn(StdErr, 'mcxstudio: --run needs an input file that exists');
    Halt(2);
  end;
end;

{ No ghost window at startup.

  When the main form's handle is made, LCL's gtk2 backend maps a small empty
  dummy window in the middle of the screen, waits for the window manager to
  decorate it, and reads back the frame size (gtk2wsforms.pp, Create-
  DummyWidgetFrame) -- a flash of a blank window before ours appears.  It
  skips that when the program was started with --disableaccurateframe, and
  asks the process's own arguments, so the option is added to them here,
  after ParseCommandLine has read what the user actually typed.  Without the
  measurement LCL estimates the frame, which only matters for placing a
  window to the pixel, and this one is centred. }
procedure SkipFrameProbe;
{$IFDEF LINUX}
const
  Opt: PChar = '--disableaccurateframe';
var
  NewArgv: PPChar;
  i: Integer;
begin
  GetMem(NewArgv, (argc + 2) * SizeOf(PChar));
  for i := 0 to argc - 1 do NewArgv[i] := argv[i];
  NewArgv[argc] := Opt;
  NewArgv[argc + 1] := nil;
  argv := NewArgv;
  Inc(argc);
end;
{$ELSE}
begin
end;
{$ENDIF}

{ A headless run has nobody to press OK.

  Anything the LCL does not handle itself ends in TApplication.HandleException,
  which puts up a modal and waits for a click (application.inc:1546) -- so a
  fault in a batch run does not fail it, it stops it, and there is nothing on
  stdout to say why, because the dialog was the whole of the report.  That is
  a CI job hanging until its timeout rather than turning red in a minute.

  HandleException calls this instead of the dialog when it is assigned, so the
  same fault becomes a line on stderr and an exit code.  RunSelfTest already
  catches what a file can raise; what this adds is everything before the first
  file, while the form itself is still being built -- an .lfm carrying a
  property this Lazarus does not have is exactly that. }
type
  TMcxHeadless = class
    class procedure Report(Sender: TObject; E: Exception);
  end;

class procedure TMcxHeadless.Report(Sender: TObject; E: Exception);
begin
  { No OpenGL is not a reason to stop: the view draws in software instead. }
  if McxViewTakesException(E) then Exit;
  WriteLn(StdErr, E.ClassName, ': ', E.Message);
  Flush(StdErr);
  Halt(2);
end;

{ The name GetAppConfigDir files the settings under. }
function McxAppName: string;
begin
  Result := 'mcxstudio2';
end;

{ Renders the icon sheet and exits.  A GUI still has to be initialised for it,
  because the glyphs are drawn on an LCL canvas. }
function DumpIcons: Boolean;
var
  Target: string;
begin
  Result := (ParamCount >= 1) and (ParamStr(1) = '--dump-icons');
  if not Result then Exit;
  if ParamCount >= 2 then Target := ParamStr(2) else Target := 'icons.png';
  Application.OnException := @TMcxHeadless.Report;
  Application.Initialize;
  if McxSaveIconSheet(Target, 32, clBlack) then
    WriteLn('wrote ', Target)
  else
    WriteLn('failed to write ', Target);
end;

begin
  { The program is installed as mcxstudio, but its settings -- the saved
    layout and the theme -- stay where earlier builds kept them, and apart
    from the first Studio's, which shipped under the same executable name. }
  OnGetApplicationName := @McxAppName;
  RequireDerivedFormResource := True;
  Application.Title := 'MCX Studio';

  {$IFDEF LINUX}
  { gtk2 caps Application.Scaled at the Xft DPI and reverts a manual bump on
    Show, so it cannot honour the desktop's integer window-scaling factor.
    Leave LCL auto-scaling off and scale the forms ourselves below.  See
    mcxdpi, ported from the sibling led and GotBox projects. }
  Application.Scaled := False;
  {$ELSE}
  { Windows and macOS report a true per-monitor DPI; LCL is right there. }
  Application.Scaled := True;
  {$ENDIF}

  if DumpIcons then Exit;

  { --self-test builds the form, drives every binding over a corpus of real
    input files and exits with the failure count.  It needs a widgetset, so it
    is not as portable as the mcxdoc tests, but it is the only way to prove the
    binding table and the form agree. }
  if (ParamCount >= 1) and (ParamStr(1) = '--self-test') then
  begin
    Application.OnException := @TMcxHeadless.Report;
    Application.Initialize;
    Application.CreateForm(TfmMain, fmMain);
    if ParamCount >= 2 then Failures := fmMain.RunSelfTest(ParamStr(2))
    else Failures := fmMain.RunSelfTest('../example');
    WriteLn(Format('%d failure(s)', [Failures]));
    if Failures > 0 then Halt(1);
    Halt(0);
  end;

  ParseCommandLine;
  SkipFrameProbe;

  { A run nobody watches has nobody to press OK either: a fault becomes a line
    on stderr and an exit code, as in the self-test. }
  if DoQuit then Application.OnException := @TMcxHeadless.Report;

  Application.Initialize;

  { mcx's own logo, so a window in the task bar is recognisably this program
    and not a generic form.  After Initialize, because it needs a widgetset to
    hold the bitmap. }
  McxApplyWindowIcon;

  { Before the first widget of all: a widget is measured as it is built, and
    one measured with the theme's own font keeps that size for the rest of the
    session, however the style changes afterwards. }
  McxInstallChromeStyle;

  { Before the first form, so that every form there will ever be is scaled as
    it is shown -- including the message boxes and the unhandled-exception
    dialog the LCL builds where we cannot reach them. }
  McxInstallFormScaler;

  Application.CreateForm(TfmMain, fmMain);

  { A file named on the command line opens instead of the blank cube -- so
    that a simulation can be handed over by a file manager, a shell, or a
    desktop entry's %f. }
  if DoRun then fmMain.RunLater(DoQuit);
  if Snapshot <> '' then fmMain.SnapshotLater(Snapshot);
  if (Length(Files) >= 1) and FileExists(Files[0]) then
    fmMain.OpenDocument(Files[0]);

  { After the file, which sets the tiles from what it describes. }
  if (Processor <> '') and not fmMain.SetProcessor(Processor) then
  begin
    WriteLn(StdErr, 'mcxstudio: unknown processor ', Processor,
      ' (nvidia, amd, intel or cpu)');
    Halt(2);
  end;

  { A second file is a result to show -- a .jnii, .bnii or a _traj.jdat --
    so that a simulation and what it produced can be opened together. }
  if (Length(Files) >= 2) and FileExists(Files[1]) then
    fmMain.ShowResultLater(Files[1]);

  { Reaches the forms that already exist; the scaler above handles the rest. }
  McxApplyAdaptiveScale;

  Application.Run;
end.
