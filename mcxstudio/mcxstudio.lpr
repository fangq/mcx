program mcxstudio;

//{$mode objfpc}{$H+}
{$mode delphi}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms, mcxgui, lazcontrols, anchordockpkg, mcxabout,
  mcxshape, mcxnewsession, mcxsource, mcxview, mcxloadfile, mcxconfig,
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

