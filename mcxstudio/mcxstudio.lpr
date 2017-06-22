program mcxstudio;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Interfaces, // this includes the LCL widgetset
  Forms
  { you can add units after this }, mcxgui, lazcontrols, tachartlazaruspkg,
  mcxabout;

{$IFDEF WINDOWS}{$R mcxstudio.rc}{$ENDIF}

{$R *.res}

begin
  Application.Title:='MCX Studio';
  Application.Initialize;
  Application.CreateForm(TfmMCX, fmMCX);
  Application.Run;
end.

