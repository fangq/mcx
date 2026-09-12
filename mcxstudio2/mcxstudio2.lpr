{ mcxstudio2 - a graphical front end for MCX, MMC and MCX-CL.

  The order of the calls in Main is load-bearing; each one says why. }
program mcxstudio2;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  Interfaces, Forms, SysUtils, Graphics,
  mcxdpi, mcxicons, mcxmain;

{$R *.res}

{ Renders the icon sheet and exits.  A GUI still has to be initialised for it,
  because the glyphs are drawn on an LCL canvas. }
function DumpIcons: Boolean;
var
  Target: string;
begin
  Result := (ParamCount >= 1) and (ParamStr(1) = '--dump-icons');
  if not Result then Exit;
  if ParamCount >= 2 then Target := ParamStr(2) else Target := 'icons.png';
  Application.Initialize;
  if McxSaveIconSheet(Target, 32, clBlack) then
    WriteLn('wrote ', Target)
  else
    WriteLn('failed to write ', Target);
end;

begin
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

  Application.Initialize;

  { Before the first widget of all: a widget is measured as it is built, and
    one measured with the theme's own font keeps that size for the rest of the
    session, however the style changes afterwards. }
  McxInstallChromeStyle;

  { Before the first form, so that every form there will ever be is scaled as
    it is shown -- including the message boxes and the unhandled-exception
    dialog the LCL builds where we cannot reach them. }
  McxInstallFormScaler;

  Application.CreateForm(TfmMain, fmMain);

  { Reaches the forms that already exist; the scaler above handles the rest. }
  McxApplyAdaptiveScale;

  Application.Run;
end.
