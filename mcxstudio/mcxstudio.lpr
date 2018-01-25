program mcxstudio;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms, GLSceneLCL_RunTime
  { you can add units after this }, mcxgui, lazcontrols, tachartlazaruspkg,
  anchordockpkg, mcxabout, mcxshape, mcxnewsession, mcxsource, mcxoutput{$IFDEF WINDOWS}, sendkeys{$ENDIF};

{$R *.res}

begin
  RequireDerivedFormResource:=True;
  Application.Title:='MCX Studio';
  Application.Initialize;
  Application.CreateForm(TfmMCX, fmMCX);
  Application.Run;
end.

