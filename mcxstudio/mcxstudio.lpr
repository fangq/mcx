program mcxstudio;

//{$mode objfpc}{$H+}
{$mode delphi}

{$IFDEF Darwin}
  {$IFDEF LCLcocoa}
    {$DEFINE NO_GLSCENE}
  {$ENDIF}
{$ENDIF}

uses
  {$IFDEF UNIX}
  cthreads,
  {$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms, mcxgui, lazcontrols, anchordockpkg, DefaultTranslator, mcxabout
  {$IFNDEF NO_GLSCENE}, mcxshape, mcxview{$ENDIF}, mcxnewsession, mcxsource, mcxloadfile, mcxconfig,
  mcxstoprun, runexec{$IFDEF USE_SYNAPSE}, runssh{$ENDIF} {$IFDEF WINDOWS}, sendkeys{$ENDIF};

{$R *.res}

begin
  Application.Scaled:=True;
  RequireDerivedFormResource:=True;
  Application.Title:='MCX Studio';
  Application.Initialize;
  Application.CreateForm(TfmMCX, fmMCX);
  Application.Run;
end.

