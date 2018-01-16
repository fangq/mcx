program mcxshow;

{$MODE Delphi}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Forms, Interfaces,
  mcxrender, GLSceneLCL_RunTime {Form1};

{$R *.res}

begin
  Application.Initialize;
  Application.CreateForm(TfmDomain, fmDomain);
  Application.Run;
end.
