unit sshclient;

interface

uses
  tlntsend, ssl_openssl, ssl_openssl_lib, ssl_libssh2;

type
  TSSHClient = class
  private
    FTelnetSend: TTelnetSend;
  public
    constructor Create(AHost, APort, AUser, APass: string);
    destructor Destroy; override;
    procedure SendCommand(ACommand: string);
    procedure LogOut;
    function ReceiveData: string;
    function HasBuffer: Boolean;
    function ReadBuffer: string;
    function LogIn: Boolean;
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

procedure TSSHClient.LogOut;
begin
  FTelnetSend.Logout;
end;

function TSSHClient.HasBuffer: Boolean;
begin
  Result:= FTelnetSend.Sock.CanRead(1000) or (FTelnetSend.Sock.WaitingData>0);
end;

function TSSHClient.ReadBuffer: String;
var
    lPos: Integer;
begin
  lPos := Length(FTelnetSend.SessionLog)+1;
  FTelnetSend.Sock.RecvPacket(1000);
  Result := Copy(FTelnetSend.SessionLog, lPos, Length(FTelnetSend.SessionLog));
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
