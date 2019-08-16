{
  ssl_libssh2.pas version 0.2

  SSH2 support (draft) plugin for Synapse Library (http://www.ararat.cz/synapse) by LibSSH2 (http://libssh2.org)
  Requires: libssh2 pascal interface - http://www.lazarus.freepascal.org/index.php/topic,15935.msg86465.html#msg86465 and
  libssh2.dll with OpenSSL.

  (С) Alexey Suhinin http://x-alexey.narod.ru
}

{$IFDEF FPC}
  {$MODE DELPHI}
{$ENDIF}
{$H+}

unit ssl_libssh2;

interface

uses
  SysUtils,
  blcksock, synsock,
  libssh2;

type
  {:@abstract(class implementing CryptLib SSL/SSH plugin.)
   Instance of this class will be created for each @link(TTCPBlockSocket).
   You not need to create instance of this class, all is done by Synapse itself!}
  TSSLLibSSH2 = class(TCustomSSL)
  protected
    FSession: PLIBSSH2_SESSION;
    FChannel: PLIBSSH2_CHANNEL;
    function SSHCheck(Value: integer): Boolean;
    function DeInit: Boolean;
  public
    {:See @inherited}
    constructor Create(const Value: TTCPBlockSocket); override;
    destructor Destroy; override;
    function Connect: boolean; override;
    function LibName: String; override;
    function Shutdown: boolean; override;
    {:See @inherited}
    function BiShutdown: boolean; override;
    {:See @inherited}
    function SendBuffer(Buffer: TMemory; Len: Integer): Integer; override;
    {:See @inherited}
    function RecvBuffer(Buffer: TMemory; Len: Integer): Integer; override;
    {:See @inherited}
    function WaitingData: Integer; override;
    {:See @inherited}
    function GetSSLVersion: string; override;
  published
  end;

implementation

{==============================================================================}
function TSSLLibSSH2.SSHCheck(Value: integer): Boolean;
var
  PLastError: PAnsiChar;
  ErrMsgLen: Integer;
begin
  Result := true;
  FLastError := 0;
  FLastErrorDesc := '';
  if Value<0 then
  begin
    FLastError := libssh2_session_last_error(FSession, PLastError, ErrMsglen, 0);
    FLastErrorDesc := PLastError;
    Result := false;
  end;
end;


function TSSLLibSSH2.DeInit: Boolean;
begin
  if Assigned(FChannel) then
  begin
    libssh2_channel_free(FChannel);
    FChannel := nil;
  end;
  if Assigned(FSession) then
  begin
    libssh2_session_disconnect(FSession,'Goodbye');
    libssh2_session_free(FSession);
    FSession := nil;
  end;
  FSSLEnabled := False;
  Result := true;
end;

constructor TSSLLibSSH2.Create(const Value: TTCPBlockSocket);
begin
  inherited Create(Value);
  FSession := nil;
  FChannel := nil;
end;

destructor TSSLLibSSH2.Destroy;
begin
  DeInit;
  inherited Destroy;
end;

function TSSLLibSSH2.Connect: boolean;
begin
  Result := False;
  if SSLEnabled then DeInit;
  if (FSocket.Socket <> INVALID_SOCKET) and (FSocket.SSL.SSLType = LT_SSHv2) then
    begin
      FSession := libssh2_session_init();
      if not Assigned(FSession) then
      begin
        FLastError := -999;
        FLastErrorDesc := 'Cannot initialize SSH session';
        exit;
      end;
      if not SSHCheck(libssh2_session_startup(FSession, FSocket.Socket)) then
        exit;
      if (FSocket.SSL.PrivateKeyFile<>'') then
      begin
        if (not SSHCheck(libssh2_userauth_publickey_fromfile(FSession, PChar(FSocket.SSL.Username), nil, PChar(FSocket.SSL.PrivateKeyFile), PChar(FSocket.SSL.KeyPassword)))) then
          exit;
      end
      else
      if (FSocket.SSL.Username<>'') and (FSocket.SSL.Password<>'') then
      begin
        if (not SSHCheck(libssh2_userauth_password(FSession, PChar(FSocket.SSL.Username), PChar(FSocket.SSL.Password)))) then
          exit;
      end;
      FChannel := libssh2_channel_open_session(FSession);
      if not assigned(FChannel) then
      begin
        SSHCheck(-1); // get error
        if FLastError = 0 then
        begin
          FLastError := -999; // unknown error
          FLastErrorDesc := 'Cannot open session';
        end;
        exit;
      end;
      if not SSHCheck(libssh2_channel_request_pty(FChannel, 'vanilla')) then
        exit;
      if not SSHCheck(libssh2_channel_shell(FChannel)) then
        exit;
      FSSLEnabled := True;
      Result := True;
    end;
end;

function TSSLLibSSH2.LibName: String;
begin
  Result := 'ssl_libssh2';
end;

function TSSLLibSSH2.Shutdown: boolean;
begin
  Result := DeInit;
end;


function TSSLLibSSH2.BiShutdown: boolean;
begin
  Result := DeInit;
end;

function TSSLLibSSH2.SendBuffer(Buffer: TMemory; Len: Integer): Integer;
begin
  Result:=libssh2_channel_write(FChannel, PChar(Buffer), Len);
  SSHCheck(Result);
end;

function TSSLLibSSH2.RecvBuffer(Buffer: TMemory; Len: Integer): Integer;
begin
  result:=libssh2_channel_read(FChannel, PChar(Buffer), Len);
  SSHCheck(Result);
end;

function TSSLLibSSH2.WaitingData: Integer;
begin
  if libssh2_poll_channel_read(FChannel, Result) <> 1 then Result := 0;
end;

function TSSLLibSSH2.GetSSLVersion: string;
begin
  Result:=libssh2_version(0);
end;

initialization
  if libssh2_init(0)=0 then
				SSLImplementation := TSSLLibSSH2;

finalization
  libssh2_exit;

end.
