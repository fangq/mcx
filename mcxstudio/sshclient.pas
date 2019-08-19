unit sshclient;

interface

uses
  tlntsend, ssl_openssl, ssl_openssl_lib, ssl_libssh2,SysUtils, strutils, Classes;

type
  TSSHClient = class
  private
    FTelnetSend: TTelnetSend;
    lastpos: integer;
    FullLog: string;
  public
    constructor Create(AHost, APort, AUser, APass: string);
    destructor Destroy; override;
    procedure SendCommand(ACommand: string);
    procedure LogOut;
    function ReceiveData: string;
    function HasBuffer: Boolean;
    function ReadBuffer: string;
    function LogIn: Boolean;
    function GetFullLog: string;
  end;

implementation

{ TSSHClient }

constructor TSSHClient.Create(AHost, APort, AUser, APass: string);
begin
  FTelnetSend := TTelnetSend.Create;
  FTelnetSend.TargetHost := AHost;
  FTelnetSend.TargetPort := APort;
  FTelnetSend.UserName := AUser;
  FTelnetSend.Password := APass;
  lastpos:=0;
  FullLog:='';
end;

destructor TSSHClient.Destroy;
begin
  FTelnetSend.Free;
  inherited;
end;

function TSSHClient.LogIn: Boolean;
begin
  Result := FTelnetSend.SSHLogin;
end;

function TSSHClient.GetFullLog: string;
begin
   Result:=FullLog;
end;

procedure TSSHClient.LogOut;
begin
  FTelnetSend.Logout;
end;

function TSSHClient.HasBuffer: Boolean;
begin
  Result:= FTelnetSend.Sock.CanRead(2000) or (FTelnetSend.Sock.WaitingData>0);
end;

function TSSHClient.ReadBuffer: String;
var
    lPos: Integer;
    slog, sl: TStringList;
    i: integer;
    newbuf: string;
begin
  Result:='';
  lPos := Length(FTelnetSend.SessionLog);
  FTelnetSend.Sock.RecvPacket(1000);
  if(Length(FTelnetSend.SessionLog)>lPos) then begin
      newbuf:=Copy(FTelnetSend.SessionLog, lPos+1, Length(FTelnetSend.SessionLog)-lPos);
      newbuf:=StringReplace(newbuf,#13, '',[rfReplaceAll]);
      FullLog:=FullLog + newbuf;
      slog:=TStringList.Create;
      slog.StrictDelimiter:=true;
      slog.Delimiter:=#10;
      slog.DelimitedText:=FullLog;
      if(slog.Count-1>lastpos) then begin
          sl:=TStringList.Create;
          sl.StrictDelimiter:=true;
          sl.Delimiter:='|';
          for i:=lastpos to slog.Count-1 do begin
             sl.Add(slog[i]);
          end;
          Result:=sl.DelimitedText;
          lastpos:=slog.Count-1;
          sl.Free;
      end;
      slog.Free;
  end;
end;

function TSSHClient.ReceiveData: string;
begin
  Result := '';
  while HasBuffer do
  begin
    Result := Result + ReadBuffer;
  end;
end;

procedure TSSHClient.SendCommand(ACommand: string);
begin
  FTelnetSend.Send(ACommand + #13);
end;

end.
