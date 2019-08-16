unit runexec;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

procedure Exec(filename: string; ExecDone: TNotifyEvent = nil; TerminateProcess: boolean = true);

procedure StopExecing;

implementation

uses
  process;

var
  StopExec: boolean = false;
  AppClosing: boolean = false;

type
  TExecThread = class(TThread)
  private
    procedure Done;
  protected
    procedure Execute; override;
  public
    FOnExecDone: TNotifyEvent;
    Command: string;
    Param: TStringList;
    FTerminateProcess: boolean;
    constructor Create(const cmd: string; ExecDone: TNotifyEvent = nil;
       TerminateProcess: boolean = true);
  end;

{ TExecThread }

constructor TExecThread.Create(const cmd: string;  ExecDone: TNotifyEvent;
   TerminateProcess: boolean);
begin
  FOnExecDone := ExecDone;
  FTerminateProcess := TerminateProcess;
  Command := cmd;
  inherited create(false);
  FreeOnTerminate := true;
end;

procedure TExecThread.Done;
begin
  if assigned(FOnExecDone) then
    FOnExecDone(self);
end;

procedure TExecThread.Execute;
var
  ExecProc: TProcess;
begin
  ExecProc := TProcess.create(nil);
  try
    ExecProc.Executable := Command;
    ExecProc.Parameters:=Param;
    ExecProc.Options := [poNoConsole];
    ExecProc.execute;
    while ExecProc.Running do begin
      if StopExec or AppClosing then begin
        if StopExec or FTerminateProcess then
          ExecProc.terminate(1);
        exit;
      end
      else
        sleep(1);
    end;
  finally
    ExecProc.free;
    if assigned(FOnExecDone) and not AppClosing then
      synchronize(@Done);
  end;
end;

procedure Exec(filename: string; ExecDone: TNotifyEvent; TerminateProcess: boolean);
begin
  StopExec := false;
  TExecThread.create(filename, ExecDone, TerminateProcess);
end;

procedure StopExecing;
begin
  StopExec := true;
end;

finalization
  AppClosing := true;
end.

