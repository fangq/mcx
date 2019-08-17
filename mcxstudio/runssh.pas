unit runssh;

interface

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, ComCtrls, RegExpr, SynEdit, strutils, sshclient;

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
    isshowprogress: boolean;
    Param: TStringList;
    OutputMemo: TSynEdit;
    sbInfo: TStatusBar;
    ProgressBar: TProgressBar;
    RegEngine:TRegExpr;
    Buf: string;
    constructor Create(const hostname, sshport, userid, pass, cmd: string; isprogress: boolean; SSHDone: TNotifyEvent = nil;
       CreateSuspended: boolean = true);
  end;

{ TSSHThread }

implementation

constructor TSSHThread.Create(const hostname, sshport, userid, pass, cmd: string; isprogress: boolean; SSHDone: TNotifyEvent;
   CreateSuspended: boolean);
begin
  FOnSSHDone := SSHDone;
  Host:=hostname;
  Port:=sshport;
  UserName:=userid;
  Password:=pass;
  Command := cmd;
  OutputMemo:=nil;
  ProgressBar:=nil;
  isshowprogress:=isprogress;
  sbInfo:=nil;
  Buf:='';

  RegEngine:=TRegExpr.Create('%[0-9 ]{4}\]');

  inherited Create(CreateSuspended);
  FreeOnTerminate := true;
end;

procedure TSSHThread.Done;
begin
  if assigned(FOnSSHDone) then
    FOnSSHDone(self);
  RegEngine.Free;
end;

procedure TSSHThread.AddLog;
var
    revbuf, percent: string;
    total: integer;
    sl: TStringList;
begin
    if(OutputMemo<>nil) and (Buf<>'') then begin
        Buf:=StringReplace(Buf,#8, '',[rfReplaceAll]);
        Buf:=ReplaceRegExpr(#27'\[(\d+;)*\d+m',Buf,'',false);
        sl:=TStringList.Create;
        sl.StrictDelimiter:=true;
        sl.Delimiter:='|';
        sl.DelimitedText:=Buf;
        OutputMemo.Lines.AddStrings(sl);
        sl.Free;
        OutputMemo.SelStart := length(OutputMemo.Text);
        OutputMemo.LeftChar:=0;
        if isshowprogress and (sbInfo<>nil) and (ProgressBar<>nil) then begin
               revbuf:=ReverseString(Buf);
               if RegEngine.Exec(revbuf) then begin
                     percent:=ReverseString(RegEngine.Match[0]);
                     if(sscanf(percent,']%d\%', [@total])=1) then begin
                        sbInfo.Panels[1].Text:=Format('%d%%',[total]);
                        sbInfo.Tag:=total;
                        ProgressBar.Position:=total;
                        sbInfo.Repaint;
                     end;
               end;
        end;
    end;
end;

procedure TSSHThread.Execute;
var
    lSSh: TSSHClient;
begin
  try
    lSSh := TSSHClient.Create(host,port, username, password);
    if lSSh.LogIn then
    begin
      Buf:='SSH: Connected!.';
      Synchronize(@AddLog);

      (* Send command *)
      lSSh.SendCommand(Command);
      (* Receive results *)

      while lSSh.HasBuffer do
      begin
          Buf:=lSSh.ReadBuffer;
          Synchronize(@AddLog);
          Sleep(100);
      end;
      Buf:=lSSh.ReadBuffer;
      Synchronize(@AddLog);
      lSSh.LogOut;
      Buf:='SSH: Logged out.';
      Synchronize(@AddLog);
    end
    else begin
        Buf:='SSH: Fail to connect.';
        Synchronize(@AddLog);
    end;
    lSSh.Free;
  finally
    if assigned(FOnSSHDone) and not AppClosing then
      Synchronize(@Done);
  end;
end;

end.



