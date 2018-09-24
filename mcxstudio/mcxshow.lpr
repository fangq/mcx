program mcxshow;

{$MODE Delphi}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Forms, GLScene_RunTime, Interfaces,
  mcxrender;

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TfmDomain, fmDomain);
  Application.Run;
end.
