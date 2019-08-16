unit runssh;

interface

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, SynEdit, sshclient;

var
  StopSSH: boolean = false;
  AppClosing: boolean = false;

type
  TSSHThread = class(TThread)
  private
    procedure Done;
  protected
    procedure Execute; override;
    procedure AddLog;
  public
    FOnSSHDone: TNotifyEvent;
    Host: string;
    Port: string;
    UserName: string;
    Password: string;
    Command: string;
    Param: TStringList;
    OutputMemo: TSynEdit;
    Buf: string;
    constructor Create(const hostname, sshport, userid, pass, cmd: string; SSHDone: TNotifyEvent = nil;
       CreateSuspended: boolean = true);
  end;

{ TSSHThread }

implementation

constructor TSSHThread.Create(const hostname, sshport, userid, pass, cmd: string;  SSHDone: TNotifyEvent;
   CreateSuspended: boolean);
begin
  FOnSSHDone := SSHDone;
  Host:=hostname;
  Port:=sshport;
  UserName:=userid;
  Password:=pass;
  Command := cmd;
  OutputMemo:=nil;
  Buf:='';

  inherited Create(CreateSuspended);
  FreeOnTerminate := true;
end;

procedure TSSHThread.Done;
begin
  if assigned(FOnSSHDone) then
    FOnSSHDone(self);
end;

procedure TSSHThread.AddLog;
begin
    if(OutputMemo<>nil) and (Buf<>'') then
        OutputMemo.Lines.Add(Buf);
end;

procedure TSSHThread.Execute;
var
    lSSh: TSSHClient;
begin
  try
    lSSh := TSSHClient.Create(host,port, username, password);
    if lSSh.LogIn then
    begin
      Buf:='SSH Connected!.';
      Synchronize(@AddLog);
      (* Get welcome message *)

      while lSSh.HasBuffer do
      begin
        Buf:=lSSh.ReadBuffer;
        Synchronize(@AddLog);
      end;

      (* Send command *)
      lSSh.SendCommand(Command);
      (* Receive results *)

      while lSSh.HasBuffer do
      begin
        Buf:=lSSh.ReadBuffer;
        Synchronize(@AddLog);
      end;
      Buf:=lSSh.ReadBuffer;
      Synchronize(@AddLog);
      lSSh.LogOut;
      Buf:='SSH Logged out.';
      Synchronize(@AddLog);
    end
    else begin
      Buf:='SSH Can''t connect.';
      Synchronize(@AddLog);
    end;
    lSSh.Free;
  finally
    if assigned(FOnSSHDone) and not AppClosing then
      Synchronize(@Done);
  end;
end;

end.



