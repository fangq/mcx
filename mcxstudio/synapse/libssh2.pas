unit libssh2;

// Ludo Brands ported to freepascal

// **zm ** translated to pascal

interface
{$ifdef fpc}
  {$mode delphi}
  uses
    ctypes;
  {$IFDEF WINDOWS}
  const
    libssh2_name = 'libssh2.dll';
  {$ENDIF}
  {$IFDEF LINUX}
  const
    libssh2_name = 'libssh2.so';
  {$ENDIF}
  type
    Uint=cuint;
    ULong=culong;
    Short=cshort;
    PUCHAR=pcuchar;
{$else}
  uses
  {$IFDEF WIN32}
    Windows;
  {$ELSE}
    Wintypes, WinProcs;
  {$ENDIF}
  const
    libssh2_name = 'libssh2.dll';
{$ENDIF}

{+// Copyright (c) 2004-2009, Sara Golemon <sarag@libssh2.org> }
{-* Copyright (c) 2009 by Daniel Stenberg }
{-* Copyright (c) 2010 Simon Josefsson <simon@josefsson.org>}
{-* All rights reserved. }
{-* }
{-* Redistribution and use in source and binary forms, }
{-* with or without modification, are permitted provided }
{-* that the following conditions are met: }
{-* }
{-* Redistributions of source code must retain the above }
{-* copyright notice, this list of conditions and the }
{-* following disclaimer. }
{-* }
{-* Redistributions in binary form must reproduce the above }
{-* copyright notice, this list of conditions and the following }
{-* disclaimer in the documentation and/or other materials }
{-* provided with the distribution. }
{-* }
{-* Neither the name of the copyright holder nor the names }
{-* of any other contributors may be used to endorse or }
{-* promote products derived from this software without }
{-* specific prior written permission. }
{-* }
{-* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND }
{-* CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, }
{-* INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES }
{-* OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE }
{-* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR }
{-* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, }
{-* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, }
{-* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR }
{-* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS }
{-* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, }
{-* WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING }
{-* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE }
{-* USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY }
{-* OF SUCH DAMAGE. }
{= }


type
  libssh2_uint64_t = UInt64;
type
  libssh2_int64_t = Int64;
type
  uint32_t = UInt;
type
  ssize_t = Integer;
type
   time_t = ULong;

{+// We use underscore instead of dash when appending CVS in dev versions just }
{-to make the BANNER define (used by src/session.c) be a valid SSH }
{-banner. Release versions have no appended strings and may of course not }
{=have dashes either. }
const
  _LIBSSH2_VERSION = '1.2.6';

{+// The numeric version number is also available "in parts" by using these }
{=defines: }
const
  LIBSSH2_VERSION_MAJOR = 1;
const
  LIBSSH2_VERSION_MINOR = 2;
const
  LIBSSH2_VERSION_PATCH = 6;

const
  SHA_DIGEST_LENGTH = 20;
  MD5_DIGEST_LENGTH = 16;


{+// This is the numeric version of the libssh2 version number, meant for easier }
{-parsing and comparions by programs. The LIBSSH2_VERSION_NUM define will }
{-always follow this syntax: }

{-0xXXYYZZ }

{-Where XX, YY and ZZ are the main version, release and patch numbers in }
{-hexadecimal (using 8 bits each). All three numbers are always represented }
{-using two digits. 1.2 would appear as "0x010200" while version 9.11.7 }
{-appears as "0x090b07". }

{-This 6-digit (24 bits) hexadecimal number does not show pre-release number, }
{-and it is always a greater number in a more recent release. It makes }
{-comparisons with greater than and less than work. }
{= }
const
  LIBSSH2_VERSION_NUM = $010206;

{+// }
{-* This is the date and time when the full source package was created. The }
{-* timestamp is not stored in CVS, as the timestamp is properly set in the }
{-* tarballs by the maketgz script. }
{-* }
{-* The format of the date should follow this template: }
{-* }
{-* "Mon Feb 12 11:35:33 UTC 2007" }
{= }
const
  LIBSSH2_TIMESTAMP = 'Thu Jun 10 08:19:51 UTC 2010';

{+// Part of every banner, user specified or not*/ }
const
  LIBSSH2_SSH_BANNER = 'SSH-2.0-libssh2_'  + _LIBSSH2_VERSION;

{+// We*could* add a comment here if we so chose*/ }
const
  LIBSSH2_SSH_DEFAULT_BANNER = LIBSSH2_SSH_BANNER;
const
  LIBSSH2_SSH_DEFAULT_BANNER_WITH_CRLF = LIBSSH2_SSH_DEFAULT_BANNER + '#13#10';

{+// Default generate and safe prime sizes for diffie-hellman-group-exchange-sha1*/ }
const
  LIBSSH2_DH_GEX_MINGROUP = 1024;
const
  LIBSSH2_DH_GEX_OPTGROUP = 1536;
const
  LIBSSH2_DH_GEX_MAXGROUP = 2048;

{+// Defaults for pty requests*/ }
const
  LIBSSH2_TERM_WIDTH = 80;
const
  LIBSSH2_TERM_HEIGHT = 24;
const
  LIBSSH2_TERM_WIDTH_PX = 0;
const
  LIBSSH2_TERM_HEIGHT_PX = 0;

{+// 1/4 second*/ }
const
  LIBSSH2_SOCKET_POLL_UDELAY = 250000;
{+// 0.25* 120 == 30 seconds*/ }
const
  LIBSSH2_SOCKET_POLL_MAXLOOPS = 120;

{+// Maximum size to allow a payload to compress to, plays it safe by falling }
{=short of spec limits }
const
  LIBSSH2_PACKET_MAXCOMP = 32000;

{+// Maximum size to allow a payload to deccompress to, plays it safe by }
{=allowing more than spec requires }
const
  LIBSSH2_PACKET_MAXDECOMP = 40000;

{+// Maximum size for an inbound compressed payload, plays it safe by }
{=overshooting spec limits }
const
  LIBSSH2_PACKET_MAXPAYLOAD = 40000;

{+// Malloc callbacks*/ }
// ovo je vec definisano u ssh2_priv alloc, realloc, free

type
  _LIBSSH2_SESSION = record
  end;
 _LIBSSH2_CHANNEL  = record
  end;
 _LIBSSH2_LISTENER = record
  end;
 _LIBSSH2_KNOWNHOSTS = record
  end;
 _LIBSSH2_AGENT = record
  end;

type
  LIBSSH2_SESSION = _LIBSSH2_SESSION;
  LIBSSH2_CHANNEL = _LIBSSH2_CHANNEL;
  LIBSSH2_LISTENER = _LIBSSH2_LISTENER;
  LIBSSH2_KNOWNHOSTS = _LIBSSH2_KNOWNHOSTS;
  LIBSSH2_AGENT = _LIBSSH2_AGENT;
  PLIBSSH2_SESSION = ^LIBSSH2_SESSION;
  PLIBSSH2_CHANNEL = ^LIBSSH2_CHANNEL;
  PLIBSSH2_LISTENER = ^LIBSSH2_LISTENER;
  PLIBSSH2_KNOWNHOSTS = ^LIBSSH2_KNOWNHOSTS;
  PLIBSSH2_AGENT = ^LIBSSH2_AGENT;

 SIZE_T = UINT;

type
  _LIBSSH2_USERAUTH_KBDINT_PROMPT = record
    text: PAnsiChar;
    length: UInt;
    echo: Byte;
  end {_LIBSSH2_USERAUTH_KBDINT_PROMPT};
  LIBSSH2_USERAUTH_KBDINT_PROMPT = _LIBSSH2_USERAUTH_KBDINT_PROMPT;
  PLIBSSH2_USERAUTH_KBDINT_PROMPT = ^LIBSSH2_USERAUTH_KBDINT_PROMPT;

type
  _LIBSSH2_USERAUTH_KBDINT_RESPONSE = record
    text: PAnsiChar;
    length: UInt;
  end {_LIBSSH2_USERAUTH_KBDINT_RESPONSE};
  LIBSSH2_USERAUTH_KBDINT_RESPONSE = _LIBSSH2_USERAUTH_KBDINT_RESPONSE;

{/* 'publickey' authentication callback */}
type 
 LIBSSH2_USERAUTH_PUBLICKEY_SIGN_FUNC = function(
  session: PLIBSSH2_SESSION; var sig: PByte; sig_len: size_t;
           const data: PByte; data_len: size_t; abstract: Pointer): Integer; cdecl;

{+// 'keyboard-interactive' authentication callback*/ }
type
  LIBSSH2_USERAUTH_KBDINT_RESPONSE_FUNC = procedure (const name: PAnsiChar;
                name_len: Integer;
                const instruction: PAnsiChar;
                instruction_len: Integer;
                num_prompts: Integer;
                const prompts: PLIBSSH2_USERAUTH_KBDINT_PROMPT;
                var responses: LIBSSH2_USERAUTH_KBDINT_RESPONSE;
                abstract: Pointer); cdecl;
{+// Callbacks for special SSH packets*/ }
type
  LIBSSH2_IGNORE_FUNC = procedure (session: PLIBSSH2_SESSION;
               const message: PAnsiChar;
               message_len: Integer;
               abstract: Pointer); cdecl  ;
type
  LIBSSH2_DEBUG_FUNC = procedure (session: PLIBSSH2_SESSION;
               always_display: Integer; 
               const message: PAnsiChar; 
               message_len: Integer;
               const language: PAnsiChar;
               language_len: Integer;
               abstract: Pointer); cdecl  ;
type
  LIBSSH2_DISCONNECT_FUNC = procedure(session: PLIBSSH2_SESSION;
               reason: Integer;
               const message: PAnsiChar;
               message_len: Integer;
               const language: PAnsiChar;
               language_len: Integer;
               abstract: Pointer); cdecl  ;
type
  LIBSSH2_PASSWD_CHANGEREQ_FUNC =  procedure(session: PLIBSSH2_SESSION;
               var newpw: PAnsiChar;
               var newpw_len: Integer;
               abstract: Pointer); cdecl  ;
type
  LIBSSH2_MACERROR_FUNC = function (session: PLIBSSH2_SESSION;
              const packet: PAnsiChar; 
              packet_len: Integer; 
              abstract: Pointer): Integer; cdecl  ;
type
  LIBSSH2_X11_OPEN_FUNC = procedure (session: PLIBSSH2_SESSION;
               channel: PLIBSSH2_CHANNEL;
               const shost: PAnsiChar;
               sport: Integer;
               abstract: Pointer); cdecl  ;
type
  LIBSSH2_CHANNEL_CLOSE_FUNC = procedure (session: PLIBSSH2_SESSION;
               var session_abstract: Pointer;
               channel: PLIBSSH2_CHANNEL;
               var channel_abstract: Pointer); cdecl  ;

{+// libssh2_session_callback_set() constants*/ }
const
  LIBSSH2_CALLBACK_IGNORE = 0;
const
  LIBSSH2_CALLBACK_DEBUG = 1;
const
  LIBSSH2_CALLBACK_DISCONNECT = 2;
const
  LIBSSH2_CALLBACK_MACERROR = 3;
const
  LIBSSH2_CALLBACK_X11 = 4;

{+// libssh2_session_method_pref() constants*/ }
const
  LIBSSH2_METHOD_KEX = 0;
const
  LIBSSH2_METHOD_HOSTKEY = 1;
const
  LIBSSH2_METHOD_CRYPT_CS = 2;
const
  LIBSSH2_METHOD_CRYPT_SC = 3;
const
  LIBSSH2_METHOD_MAC_CS = 4;
const
  LIBSSH2_METHOD_MAC_SC = 5;
const
  LIBSSH2_METHOD_COMP_CS = 6;
const
  LIBSSH2_METHOD_COMP_SC = 7;
const
  LIBSSH2_METHOD_LANG_CS = 8;
const
  LIBSSH2_METHOD_LANG_SC = 9;

{+// session.flags bits*/ }
const
  LIBSSH2_FLAG_SIGPIPE = $00000001;

type
  PLIBSSH2_POLLFD = ^_LIBSSH2_POLLFD;
  _LIBSSH2_POLLFD = record
    _type: Byte;
{= LIBSSH2_POLLFD_* below }
    socket: Integer;
{= File descriptors -- examined with system select() call }
    channel: PLIBSSH2_CHANNEL;
{= Examined by checking internal state }
    listener: PLIBSSH2_LISTENER;
{- Read polls only -- are inbound }
{=connections waiting to be accepted? }
  end {fd};
  LIBSSH2_POLLFD = _LIBSSH2_POLLFD;

{= Requested Events }
{= Returned Events }

{+// Poll FD Descriptor Types*/ }
const
  LIBSSH2_POLLFD_SOCKET = 1;
const
  LIBSSH2_POLLFD_CHANNEL = 2;
const
  LIBSSH2_POLLFD_LISTENER = 3;

{+// Note: Win32 Doesn't actually have a poll() implementation, so some of these }
{=values are faked with select() data }
{+// Poll FD events/revents -- Match sys/poll.h where possible*/ }
const
  LIBSSH2_POLLFD_POLLIN = $0001; {/* Data available to be read or}
const
  LIBSSH2_POLLFD_POLLPRI = $0002; {/* Priority data available to
                                                  be read -- Socket only */}
const
  LIBSSH2_POLLFD_POLLEXT = $0002; {/* Extended data available to
                                                  be read -- Channel only */}
const
  LIBSSH2_POLLFD_POLLOUT = $0004; {/* Can may be written --
                                                  Socket/Channel */}
const
  LIBSSH2_POLLFD_POLLERR = $0008; {/* Error Condition -- Socket*/}
const
  LIBSSH2_POLLFD_POLLHUP = $0010; {/* HangUp/EOF -- Socket*/}
const
  LIBSSH2_POLLFD_SESSION_CLOSED = $0010; {/* Session Disconnect*/}
const
  LIBSSH2_POLLFD_POLLNVAL = $0020; {/* Invalid request -- Socket
                                                  Only */}
const
  LIBSSH2_POLLFD_POLLEX = $0040; {/* Exception Condition --
                                                  Socket/Win32 */}
const
  LIBSSH2_POLLFD_CHANNEL_CLOSED =  $0080; {/* Channel Disconnect */}
const
  LIBSSH2_POLLFD_LISTENER_CLOSED = $0080; {/* Listener Disconnect*/}

const
  HAVE_LIBSSH2_SESSION_BLOCK_DIRECTION = 1;
	
{+// Block Direction Types*/ }
const
  LIBSSH2_SESSION_BLOCK_INBOUND = $0001;
const
  LIBSSH2_SESSION_BLOCK_OUTBOUND = $0002;

{+// Hash Types*/ }
const
  LIBSSH2_HOSTKEY_HASH_MD5 = 1;
const
  LIBSSH2_HOSTKEY_HASH_SHA1 = 2;

{+// Hostkey Types */ }
const
  LIBSSH2_HOSTKEY_TYPE_UNKNOWN = 0;
const
  LIBSSH2_HOSTKEY_TYPE_RSA = 1;
const
  LIBSSH2_HOSTKEY_TYPE_DSS = 2;

{+// Disconnect Codes (defined by SSH protocol)*/ }
const
  SSH_DISCONNECT_HOST_NOT_ALLOWED_TO_CONNECT = 1;
const
  SSH_DISCONNECT_PROTOCOL_ERROR = 2;
const
  SSH_DISCONNECT_KEY_EXCHANGE_FAILED = 3;
const
  SSH_DISCONNECT_RESERVED = 4;
const
  SSH_DISCONNECT_MAC_ERROR = 5;
const
  SSH_DISCONNECT_COMPRESSION_ERROR = 6;
const
  SSH_DISCONNECT_SERVICE_NOT_AVAILABLE = 7;
const
  SSH_DISCONNECT_PROTOCOL_VERSION_NOT_SUPPORTED = 8;
const
  SSH_DISCONNECT_HOST_KEY_NOT_VERIFIABLE = 9;
const
  SSH_DISCONNECT_CONNECTION_LOST = 10;
const
  SSH_DISCONNECT_BY_APPLICATION = 11;
const
  SSH_DISCONNECT_TOO_MANY_CONNECTIONS = 12;
const
  SSH_DISCONNECT_AUTH_CANCELLED_BY_USER = 13;
const
  SSH_DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE = 14;
const
  SSH_DISCONNECT_ILLEGAL_USER_NAME = 15;

{+// Error Codes (defined by libssh2)*/ }
const
  LIBSSH2_ERROR_NONE = 0;
const
  LIBSSH2_ERROR_SOCKET_NONE = -1;
const
  LIBSSH2_ERROR_BANNER_NONE = -2;
const
  LIBSSH2_ERROR_BANNER_SEND = -3;
const
  LIBSSH2_ERROR_INVALID_MAC = -4;
const
  LIBSSH2_ERROR_KEX_FAILURE = -5;
const
  LIBSSH2_ERROR_ALLOC = -6;
const
  LIBSSH2_ERROR_SOCKET_SEND = -7;
const
  LIBSSH2_ERROR_KEY_EXCHANGE_FAILURE = -8;
const
  LIBSSH2_ERROR_TIMEOUT = -9;
const
  LIBSSH2_ERROR_HOSTKEY_INIT = -10;
const
  LIBSSH2_ERROR_HOSTKEY_SIGN = -11;
const
  LIBSSH2_ERROR_DECRYPT = -12;
const
  LIBSSH2_ERROR_SOCKET_DISCONNECT = -13;
const
  LIBSSH2_ERROR_PROTO = -14;
const
  LIBSSH2_ERROR_PASSWORD_EXPIRED = -15;
const
  LIBSSH2_ERROR_FILE = -16;
const
  LIBSSH2_ERROR_METHOD_NONE = -17;
const
  LIBSSH2_ERROR_AUTHENTICATION_FAILED = -18;
const
  LIBSSH2_ERROR_PUBLICKEY_UNRECOGNIZED = LIBSSH2_ERROR_AUTHENTICATION_FAILED;
const
  LIBSSH2_ERROR_PUBLICKEY_UNVERIFIED = -19;
const
  LIBSSH2_ERROR_CHANNEL_OUTOFORDER = -20;
const
  LIBSSH2_ERROR_CHANNEL_FAILURE = -21;
const
  LIBSSH2_ERROR_CHANNEL_REQUEST_DENIED = -22;
const
  LIBSSH2_ERROR_CHANNEL_UNKNOWN = -23;
const
  LIBSSH2_ERROR_CHANNEL_WINDOW_EXCEEDED = -24;
const
  LIBSSH2_ERROR_CHANNEL_PACKET_EXCEEDED = -25;
const
  LIBSSH2_ERROR_CHANNEL_CLOSED = -26;
const
  LIBSSH2_ERROR_CHANNEL_EOF_SENT = -27;
const
  LIBSSH2_ERROR_SCP_PROTOCOL = -28;
const
  LIBSSH2_ERROR_ZLIB = -29;
const
  LIBSSH2_ERROR_SOCKET_TIMEOUT = -30;
const
  LIBSSH2_ERROR_SFTP_PROTOCOL = -31;
const
  LIBSSH2_ERROR_REQUEST_DENIED = -32;
const
  LIBSSH2_ERROR_METHOD_NOT_SUPPORTED = -33;
const
  LIBSSH2_ERROR_INVAL = -34;
const
  LIBSSH2_ERROR_INVALID_POLL_TYPE = -35;
const
  LIBSSH2_ERROR_PUBLICKEY_PROTOCOL = -36;
const
  LIBSSH2_ERROR_EAGAIN = -37;
const
  LIBSSH2_ERROR_BUFFER_TOO_SMALL = -38;
const
  LIBSSH2_ERROR_BAD_USE = -39;
const
  LIBSSH2_ERROR_COMPRESS = -40;
const
  LIBSSH2_ERROR_OUT_OF_BOUNDARY = -41;
const
  LIBSSH2_ERROR_AGENT_PROTOCOL = -42;

{+// Global API*/}
const
  LIBSSH2_INIT_NO_CRYPTO = $0001;
{/*
 * libssh2_init()
 *
 * Initialize the libssh2 functions.  This typically initialize the
 * crypto library.  It uses a global state, and is not thread safe --
 * you must make sure this function is not called concurrently.
 *
 * Flags can be:
 * 0:                              Normal initialize
 * LIBSSH2_INIT_NO_CRYPTO:         Do not initialize the crypto library (ie.
 *                                 OPENSSL_add_cipher_algoritms() for OpenSSL
 *
 * Returns 0 if succeeded, or a negative value for error.
 */}
function libssh2_init(flags: Integer): Integer; cdecl;

{/*
 * libssh2_exit()
 *
 * Exit the libssh2 functions and free's all memory used internal.
 */}
procedure libssh2_exit; cdecl;

type
// abstract je void**, tako da pazite!!!!
LIBSSH2_ALLOC_FUNC = function(count: UINT; abstract: Pointer): Pointer; cdecl;
LIBSSH2_REALLOC_FUNC = function(ptr: Pointer; count: UINT; abstract: Pointer): Pointer; cdecl;
LIBSSH2_FREE_FUNC = procedure(ptr: Pointer; abstract: Pointer); cdecl;

{+// Session API*/ }

function libssh2_session_init_ex(my_alloc: LIBSSH2_ALLOC_FUNC;
                                 my_free: LIBSSH2_FREE_FUNC;
                                 my_realloc: LIBSSH2_REALLOC_FUNC;
                                 abstract: Pointer): PLIBSSH2_SESSION; cdecl;
                                  
function libssh2_session_init: PLIBSSH2_SESSION; inline;

function libssh2_session_abstract(session: PLIBSSH2_SESSION): Pointer; cdecl;

function libssh2_session_callback_set(session: PLIBSSH2_SESSION;
                                      cbtype: Integer;
                                      callback: Pointer): Pointer; cdecl;

function libssh2_banner_set(session: PLIBSSH2_SESSION;
                            const banner: PAnsiChar): Integer; cdecl;


function libssh2_session_startup(session: PLIBSSH2_SESSION;
                                 sock: Integer): Integer; cdecl;

function libssh2_session_disconnect_ex(session: PLIBSSH2_SESSION;
                                       reason: Integer; 
                                       const description: PAnsiChar; 
                                       const lang: PAnsiChar): Integer; cdecl  ;

function libssh2_session_disconnect(session: PLIBSSH2_SESSION; const description: PAnsiChar): Integer; inline;

function libssh2_session_free(session: PLIBSSH2_SESSION): Integer; cdecl  ;


function libssh2_hostkey_hash(session: PLIBSSH2_SESSION;
                              hash_type: Integer): PAnsiChar; cdecl  ;

function libssh2_session_hostkey(session: PLIBSSH2_SESSION;
                                                var len: size_t;
                                                var _type: Integer): PAnsiChar; cdecl;


function libssh2_session_method_pref(session: PLIBSSH2_SESSION;
                                     method_type: Integer; 
                                     const prefs: PAnsiChar): Integer; cdecl  ; 

function libssh2_session_methods(session: PLIBSSH2_SESSION;
                                 method_type: Integer): PAnsiChar; cdecl  ; 

function libssh2_session_last_error(session: PLIBSSH2_SESSION;
                                    var errmsg: PAnsiChar; 
                                    var errmsg_len: Integer; 
                                    want_buf: Integer): Integer; cdecl; 

function libssh2_session_last_errno(session: PLIBSSH2_SESSION): Integer; cdecl  ;

function libssh2_session_block_directions(session: PLIBSSH2_SESSION): Integer; cdecl  ;


function libssh2_session_flag(session: PLIBSSH2_SESSION;
                              flag: Integer; 
                              value: Integer): Integer; cdecl  ; 

{+// Userauth API*/ }

function libssh2_userauth_list(session: PLIBSSH2_SESSION;
                               const username: PAnsiChar; 
                               username_len: UINT): PAnsiChar; cdecl  ; 

function libssh2_userauth_authenticated(session: PLIBSSH2_SESSION): Integer; cdecl  ;


function libssh2_userauth_password_ex(session: PLIBSSH2_SESSION;
                                      const username: PAnsiChar; 
                                      username_len: Uint; 
                                      const password: PAnsiChar; 
                                      password_len: Uint;
                                      passwd_change_cb: LIBSSH2_PASSWD_CHANGEREQ_FUNC): Integer; cdecl  ;

function libssh2_userauth_password(session: PLIBSSH2_SESSION; const username: PAnsiChar; const password: PAnsiChar): Integer; inline;

function libssh2_userauth_publickey_fromfile_ex(session: PLIBSSH2_SESSION;
                                                const username: PAnsiChar; 
                                                username_len: Uint; 
                                                const publickey: PAnsiChar; 
                                                const privatekey: PAnsiChar;
                                                const passphrase: PAnsiChar): Integer; cdecl  ;

function libssh2_userauth_publickey_fromfile(session: PLIBSSH2_SESSION; const username: PAnsiChar;
    const publickey: PAnsiChar; const privatekey: PAnsiChar; const passphrase: PAnsiChar): Integer; inline;

function libssh2_userauth_hostbased_fromfile_ex(session: PLIBSSH2_SESSION;
                                                const username: PAnsiChar;
                                                username_len: Uint; 
                                                const publickey: PAnsiChar;
                                                const privatekey: PAnsiChar;
                                                const passphrase: PAnsiChar;
                                                const hostname: PAnsiChar;
                                                hostname_len: UInt;
                                                local_username: PAnsiChar;
                                                local_username_len: UInt): Integer; cdecl  ;

function libssh2_userauth_hostbased_fromfile(session: PLIBSSH2_SESSION; const username: PAnsiChar; const publickey: PAnsiChar;
    const privatekey: PAnsiChar; const passphrase: PAnsiChar; const hostname: PAnsiChar): Integer; inline;
    
{+// }
{-* response_callback is provided with filled by library prompts array, }
{-* but client must allocate and fill individual responses. Responses }
{-* array is already allocated. Responses data will be freed by libssh2 }
{-* after callback return, but before subsequent callback invokation. }
{= }

function libssh2_userauth_keyboard_interactive_ex(session: PLIBSSH2_SESSION;
                                                  const username: PAnsiChar;
                                                  username_len: UInt;
                                                  response_callback: LIBSSH2_USERAUTH_KBDINT_RESPONSE_FUNC): Integer; cdecl  ;

function libssh2_userauth_keyboard_interactive(session: PLIBSSH2_SESSION; const username: PAnsiChar;  response_callback: LIBSSH2_USERAUTH_KBDINT_RESPONSE_FUNC): Integer; inline;

function libssh2_poll(var fds: LIBSSH2_POLLFD; 
                      nfds: UInt;
                      timeout: LongInt): Integer; cdecl  ; 

{+// Channel API*/ }
const
  LIBSSH2_CHANNEL_WINDOW_DEFAULT = 65536;
const
  LIBSSH2_CHANNEL_PACKET_DEFAULT = 32768;
const
  LIBSSH2_CHANNEL_MINADJUST = 1024;

{+// Extended Data Handling*/ }
const
  LIBSSH2_CHANNEL_EXTENDED_DATA_NORMAL = 0;
const
  LIBSSH2_CHANNEL_EXTENDED_DATA_IGNORE = 1;
const
  LIBSSH2_CHANNEL_EXTENDED_DATA_MERGE = 2;

const
  SSH_EXTENDED_DATA_STDERR = 1;

{+// Returned by any function that would block during a read/write opperation*/ }
const
  LIBSSH2CHANNEL_EAGAIN = LIBSSH2_ERROR_EAGAIN;

function libssh2_channel_open_ex(session: PLIBSSH2_SESSION;
                                 const channel_type: PAnsiChar; 
                                 channel_type_len: Uint;
                                 window_size: Uint;
                                 packet_size: Uint; 
                                 const message: PAnsiChar; 
                                 message_len: Uint): PLIBSSH2_CHANNEL; cdecl  ; 

function libssh2_channel_open_session(session: PLIBSSH2_SESSION): PLIBSSH2_CHANNEL; inline;

function libssh2_channel_direct_tcpip_ex(session: PLIBSSH2_SESSION;
                                         const host: PAnsiChar;
                                         port: Integer; 
                                         const shost: PAnsiChar; 
                                         sport: Integer): PLIBSSH2_CHANNEL; cdecl  ;

function libssh2_channel_direct_tcpip(session: PLIBSSH2_SESSION; const host: PAnsiChar; port: Integer): PLIBSSH2_CHANNEL; inline;

function libssh2_channel_forward_listen_ex(session: PLIBSSH2_SESSION;
                                           const host: PAnsiChar;
                                           port: Integer;
                                           var bound_port: Integer;
                                           queue_maxsize: Integer): PLIBSSH2_LISTENER cdecl  ;

function libssh2_channel_forward_listen(session: PLIBSSH2_SESSION; port: Integer): PLIBSSH2_LISTENER; inline;

function libssh2_channel_forward_cancel(listener: PLIBSSH2_LISTENER): Integer; cdecl  ;


function libssh2_channel_forward_accept(listener: PLIBSSH2_LISTENER): PLIBSSH2_CHANNEL; cdecl  ;


function libssh2_channel_setenv_ex(channel: PLIBSSH2_CHANNEL;
                                   const varname: PAnsiChar;
                                   varname_len: Uint;
                                   const value: PAnsiChar;
                                   value_len: UInt): Integer; cdecl  ; 

function libssh2_channel_setenv(channel: PLIBSSH2_CHANNEL; const varname: PAnsiChar; const value: PAnsiChar): Integer; inline;

function libssh2_channel_request_pty_ex(channel: PLIBSSH2_CHANNEL;
                                        const term: PAnsiChar;
                                        term_len: Uint;
                                        const modes: PAnsiChar;
                                        modes_len: Uint;
                                        width: Integer;
                                        height: Integer;
                                        width_px: Integer; 
                                        height_px: Integer): Integer; cdecl  ;

function libssh2_channel_request_pty(channel: PLIBSSH2_CHANNEL; const term: PAnsiChar): Integer; inline;

function libssh2_channel_request_pty_size_ex(channel: PLIBSSH2_CHANNEL;
                                             width: Integer;
                                             height: Integer;
                                             width_px: Integer; 
                                             height_px: Integer): Integer; cdecl  ;

function libssh2_channel_request_pty_size(channel: PLIBSSH2_CHANNEL; width: Integer; height: Integer): Integer; inline;

function libssh2_channel_x11_req_ex(channel: PLIBSSH2_CHANNEL;
                                    single_connection: Integer; 
                                    const auth_proto: PAnsiChar; 
                                    const auth_cookie: PAnsiChar; 
                                    screen_number: Integer): Integer; cdecl  ;

function libssh2_channel_x11_req(channel: PLIBSSH2_CHANNEL; screen_number: Integer): Integer; inline;

function libssh2_channel_process_startup(channel: PLIBSSH2_CHANNEL;
                                         const request: PAnsiChar; 
                                         request_len: UInt;
                                         const message: PAnsiChar;
                                         message_len: UInt): Integer; cdecl  ;
                                         
function libssh2_channel_shell(channel: PLIBSSH2_CHANNEL): Integer; inline;

function libssh2_channel_exec(channel: PLIBSSH2_CHANNEL; const command: PAnsiChar): Integer; inline;

function libssh2_channel_subsystem(channel: PLIBSSH2_CHANNEL; const subsystem: PAnsiChar): Integer; inline;

function libssh2_channel_read_ex(channel: PLIBSSH2_CHANNEL;
                                 stream_id: Integer; 
                                 buf: PAnsiChar;
                                 buflen: SIZE_T): Integer; cdecl  ;

function libssh2_channel_read(channel: PLIBSSH2_CHANNEL; buf: PAnsiChar; buflen: SIZE_T): Integer; inline;

function libssh2_channel_read_stderr(channel: PLIBSSH2_CHANNEL; buf: PAnsiChar; buflen: SIZE_T): Integer; inline;

function libssh2_poll_channel_read(channel: PLIBSSH2_CHANNEL;
                                   extended: Integer): Integer; cdecl  ; 


function libssh2_channel_window_read_ex(channel: PLIBSSH2_CHANNEL;
                                        var read_avail: LongInt; 
                                        var window_size_initial: LongInt): ULong; cdecl  ;

function libssh2_channel_window_read(channel: PLIBSSH2_CHANNEL): ULong; inline;

{+// libssh2_channel_receive_window_adjust is DEPRECATED, do not use!*/ }

function libssh2_channel_receive_window_adjust(channel: PLIBSSH2_CHANNEL;
                                               adjustment: LongInt; 
                                               force: Byte): LongInt; cdecl  ; 


function libssh2_channel_receive_window_adjust2(channel: PLIBSSH2_CHANNEL;
                                                adjustment: LongInt; 
                                                force: Byte; 
                                                var storewindow: ULong): Integer; cdecl  ; 


function libssh2_channel_write_ex(channel: PLIBSSH2_CHANNEL;
                                  stream_id: Integer; 
                                  const buf: PAnsiChar;
                                  buflen: ULong): Integer; cdecl  ;

function libssh2_channel_write(channel: PLIBSSH2_CHANNEL; const buf: PAnsiChar; buflen: ULong): Integer; inline;

function libssh2_channel_write_stderr(channel: PLIBSSH2_CHANNEL; const buf: PAnsiChar; buflen: ULong): Integer; inline;


function libssh2_channel_window_write_ex(channel: PLIBSSH2_CHANNEL;
                                         var window_size_initial: LongInt): ULong; cdecl  ;

function libssh2_channel_window_write(channel: PLIBSSH2_CHANNEL): ULong; inline;

procedure libssh2_session_set_blocking(session: PLIBSSH2_SESSION;
                                      blocking: Integer); cdecl  ;

function libssh2_session_get_blocking(session: PLIBSSH2_SESSION): Integer; cdecl  ;


procedure libssh2_channel_set_blocking(channel: PLIBSSH2_CHANNEL;
                                      blocking: Integer); cdecl  ;

{+// libssh2_channel_handle_extended_data is DEPRECATED, do not use!*/ }

procedure libssh2_channel_handle_extended_data(channel: PLIBSSH2_CHANNEL;
                                              ignore_mode: Integer); cdecl  ;

function libssh2_channel_handle_extended_data2(channel: PLIBSSH2_CHANNEL;
                                               ignore_mode: Integer): Integer; cdecl  ;

{+// libssh2_channel_ignore_extended_data() is defined below for BC with version }
{-* 0.1 }
{-* }
{-* Future uses should use libssh2_channel_handle_extended_data() directly if }
{-* LIBSSH2_CHANNEL_EXTENDED_DATA_MERGE is passed, extended data will be read }
{-* (FIFO) from the standard data channel }
{= }
{+// DEPRECATED*/ }
procedure libssh2_channel_ignore_extended_data(channel: PLIBSSH2_CHANNEL; ignore: Integer); inline;

const
  LIBSSH2_CHANNEL_FLUSH_EXTENDED_DATA = -1;
const
  LIBSSH2_CHANNEL_FLUSH_ALL = -2;

function libssh2_channel_flush_ex(channel: PLIBSSH2_CHANNEL;
                                  streamid: Integer): Integer; cdecl  ;

function libssh2_channel_flush(channel: PLIBSSH2_CHANNEL): Integer; inline;

function libssh2_channel_flush_stderr(channel: PLIBSSH2_CHANNEL): Integer; inline;

function libssh2_channel_get_exit_status(channel: PLIBSSH2_CHANNEL): Integer; cdecl  ;

function libssh2_channel_send_eof(channel: PLIBSSH2_CHANNEL): Integer; cdecl  ;

function libssh2_channel_eof(channel: PLIBSSH2_CHANNEL): Integer; cdecl  ;

function libssh2_channel_wait_eof(channel: PLIBSSH2_CHANNEL): Integer; cdecl  ;

function libssh2_channel_close(channel: PLIBSSH2_CHANNEL): Integer; cdecl  ;

function libssh2_channel_wait_closed(channel: PLIBSSH2_CHANNEL): Integer; cdecl  ;

function libssh2_channel_free(channel: PLIBSSH2_CHANNEL): Integer; cdecl  ;


type
 Pstruct_stat = ^struct_stat;
 struct_stat = record
   st_dev: UINT;
   st_ino: Word;
   st_mode: Word;
   st_nlink: Short;
   st_uid: Short;
   st_gid: Short;
   st_rdev: UINT;
   st_size: LongInt;
   st_atime: Int64;
   st_mtime: Int64;
   st_ctime: Int64;
 end;

function libssh2_scp_recv(session: PLIBSSH2_SESSION;
                          const path: PAnsiChar; 
                          var sb: struct_stat): PLIBSSH2_CHANNEL; cdecl  ;

function libssh2_scp_send_ex(session: PLIBSSH2_SESSION;
                             const path: PAnsiChar;
                             mode: Integer;
                             size: SIZE_T;
                             mtime: LongInt;
                             atime: LongInt): PLIBSSH2_CHANNEL; cdecl  ;

function libssh2_scp_send64(session: PLIBSSH2_SESSION; const path: PAnsiChar; mode: Integer;
                   size: Int64; mtime: time_t; atime: time_t): PLIBSSH2_CHANNEL; cdecl ;

function libssh2_scp_send(session: PLIBSSH2_SESSION; const path: PAnsiChar; mode: Integer; size: SIZE_T): PLIBSSH2_CHANNEL; inline;

function libssh2_base64_decode(session: PLIBSSH2_SESSION;
                               var dest: PAnsiChar;
                               var dest_len: Uint;
                               const src: PAnsiChar;
                               src_len: Uint): Integer; cdecl; 

function libssh2_version(req_version_num: Integer): PAnsiChar; cdecl  ; 

const
  HAVE_LIBSSH2_KNOWNHOST_API = $010101; {/* since 1.1.1 */}
const
  HAVE_LIBSSH2_VERSION_API = $010100; {/* libssh2_version since 1.1 */}

type
PLIBSSH2_KNOWNHOST = ^LIBSSH2_KNOWNHOST;
LIBSSH2_KNOWNHOST = record
    magic: UInt;  {/* magic stored by the library */}
    node: Pointer; {/* handle to the internal representation of this host */}
    name: PAnsiChar; {/* this is NULL if no plain text host name exists */}
    key: PAnsiChar;  {/* key in base64/printable format */}
    typemask: Integer;
end;

{/*
 * libssh2_knownhost_init
 *
 * Init a collection of known hosts. Returns the pointer to a collection.
 *
 */}
function libssh2_knownhost_init(session: PLIBSSH2_SESSION): PLIBSSH2_KNOWNHOSTS; cdecl;

{/*
 * libssh2_knownhost_add
 *
 * Add a host and its associated key to the collection of known hosts.
 *
 * The 'type' argument specifies on what format the given host is:
 *
 * plain  - ascii "hostname.domain.tld"
 * sha1   - SHA1(<salt> <host>) base64-encoded!
 * custom - another hash
 *
 * If 'sha1' is selected as type, the salt must be provided to the salt
 * argument. This too base64 encoded.
 *
 * The SHA-1 hash is what OpenSSH can be told to use in known_hosts files.  If
 * a custom type is used, salt is ignored and you must provide the host
 * pre-hashed when checking for it in the libssh2_knownhost_check() function.
 *
 */}

{/* host format (2 bits) */}
const
  LIBSSH2_KNOWNHOST_TYPE_MASK = $ffff;
const
  LIBSSH2_KNOWNHOST_TYPE_PLAIN = 1;
const
  LIBSSH2_KNOWNHOST_TYPE_SHA1 = 2; {/* always base64 encoded */}
const
  LIBSSH2_KNOWNHOST_TYPE_CUSTOM = 3;

{/* key format (2 bits) */}
const LIBSSH2_KNOWNHOST_KEYENC_MASK = (3 shl 16);
const LIBSSH2_KNOWNHOST_KEYENC_RAW = (1 shl 16);
const LIBSSH2_KNOWNHOST_KEYENC_BASE64 = (2 shl 16);

{/* type of key (2 bits) */}
const LIBSSH2_KNOWNHOST_KEY_MASK = (3 shl 18);
const LIBSSH2_KNOWNHOST_KEY_SHIFT = 18;
const LIBSSH2_KNOWNHOST_KEY_RSA1 = (1 shl 18);
const LIBSSH2_KNOWNHOST_KEY_SSHRSA = (2 shl 18);
const LIBSSH2_KNOWNHOST_KEY_SSHDSS = (3 shl 18);

function libssh2_knownhost_add(hosts: PLIBSSH2_KNOWNHOSTS;
                      host,
                      salt,
                      key: PAnsiChar; keylen: size_t; typemask: Integer;
                      var store: PLIBSSH2_KNOWNHOST): Integer; cdecl ;

{/*
 * libssh2_knownhost_addc
 *
 * Add a host and its associated key to the collection of known hosts.
 *
 * Takes a comment argument that may be NULL.  A NULL comment indicates
 * there is no comment and the entry will end directly after the key
 * when written out to a file.  An empty string "" comment will indicate an
 * empty comment which will cause a single space to be written after the key.
 *
 * The 'type' argument specifies on what format the given host and keys are:
 *
 * plain  - ascii "hostname.domain.tld"
 * sha1   - SHA1(<salt> <host>) base64-encoded!
 * custom - another hash
 *
 * If 'sha1' is selected as type, the salt must be provided to the salt
 * argument. This too base64 encoded.
 *
 * The SHA-1 hash is what OpenSSH can be told to use in known_hosts files.  If
 * a custom type is used, salt is ignored and you must provide the host
 * pre-hashed when checking for it in the libssh2_knownhost_check() function.
 *
 * The keylen parameter may be omitted (zero) if the key is provided as a
 * NULL-terminated base64-encoded string.
 */}

function libssh2_knownhost_addc(hosts: PLIBSSH2_KNOWNHOSTS;
                       host,
                       salt,
                       key: PAnsiChar;
                       keylen: size_t;
                       comment: PAnsiChar;
                       commentlen: size_t; typemask: Integer;
                       var store: PLIBSSH2_KNOWNHOST): Integer; cdecl ;

{/*
 * libssh2_knownhost_check
 *
 * Check a host and its associated key against the collection of known hosts.
 *
 * The type is the type/format of the given host name.
 *
 * plain  - ascii "hostname.domain.tld"
 * custom - prehashed base64 encoded. Note that this cannot use any salts.
 *
 *
 * 'knownhost' may be set to NULL if you don't care about that info.
 *
 * Returns:
 *
 * LIBSSH2_KNOWNHOST_CHECK_* values, see below
 *
 */}

const
  LIBSSH2_KNOWNHOST_CHECK_MATCH     = 0;
const
  LIBSSH2_KNOWNHOST_CHECK_MISMATCH  = 1;
const
  LIBSSH2_KNOWNHOST_CHECK_NOTFOUND  = 2;
const
  LIBSSH2_KNOWNHOST_CHECK_FAILURE   = 3;

function libssh2_knownhost_check(hosts: PLIBSSH2_KNOWNHOSTS;
                        host, key: PAnsiChar; keylen: size_t;
                        typemask: Integer;
                        var knownhost: PLIBSSH2_KNOWNHOST): Integer; cdecl;

{/* this function is identital to the above one, but also takes a port
   argument that allows libssh2 to do a better check */}
function libssh2_knownhost_checkp(hosts: PLIBSSH2_KNOWNHOSTS;
                         const host: PAnsiChar; port: Integer;
                         const key: PAnsiChar; keylen: size_t;
                         typemask: Integer;
                         var knownhost: PLIBSSH2_KNOWNHOST): Integer; cdecl ;

{/*
 * libssh2_knownhost_del
 *
 * Remove a host from the collection of known hosts. The 'entry' struct is
 * retrieved by a call to libssh2_knownhost_check().
 *
 */}
function libssh2_knownhost_del(hosts: PLIBSSH2_KNOWNHOSTS;
                               entry: PLIBSSH2_KNOWNHOST): Integer; cdecl;

{/*
 * libssh2_knownhost_free
 *
 * Free an entire collection of known hosts.
 *
 */}
procedure libssh2_knownhost_free(hosts: PLIBSSH2_KNOWNHOSTS); cdecl;

{/*
 * libssh2_knownhost_readline()
 *
 * Pass in a line of a file of 'type'. It makes libssh2 read this line.
 *
 * LIBSSH2_KNOWNHOST_FILE_OPENSSH is the only supported type.
 *
 */}
function libssh2_knownhost_readline(hosts: PLIBSSH2_KNOWNHOSTS;
                           const line: PAnsiChar; len: size_t; _type: Integer): Integer; cdecl;


{/*
 * libssh2_knownhost_readfile
 *
 * Add hosts+key pairs from a given file.
 *
 * Returns a negative value for error or number of successfully added hosts.
 *
 * This implementation currently only knows one 'type' (openssh), all others
 * are reserved for future use.
 */}

const
  LIBSSH2_KNOWNHOST_FILE_OPENSSH = 1;

function libssh2_knownhost_readfile(hosts: PLIBSSH2_KNOWNHOSTS;
                           const filename: PAnsiChar; _type: Integer): Integer; cdecl;

{/*
 * libssh2_knownhost_writeline()
 *
 * Ask libssh2 to convert a known host to an output line for storage.
 *
 * Note that this function returns LIBSSH2_ERROR_BUFFER_TOO_SMALL if the given
 * output buffer is too small to hold the desired output.
 *
 * This implementation currently only knows one 'type' (openssh), all others
 * are reserved for future use.
 *
 */}
function libssh2_knownhost_writeline(hosts: PLIBSSH2_KNOWNHOSTS;
                            known: PLIBSSH2_KNOWNHOST;
                            buffer: PAnsiChar; buflen: size_t;
                            var outlen: size_t; {/* the amount of written data */}
                            _type: Integer): Integer; cdecl;

{/*
 * libssh2_knownhost_writefile
 *
 * Write hosts+key pairs to a given file.
 *
 * This implementation currently only knows one 'type' (openssh), all others
 * are reserved for future use.
 */}

function libssh2_knownhost_writefile(hosts: PLIBSSH2_KNOWNHOSTS;
                            const filename: PAnsiChar; _type: Integer): Integer; cdecl;

{/*
 * libssh2_knownhost_get()
 *
 * Traverse the internal list of known hosts. Pass NULL to 'prev' to get
 * the first one. Or pass a poiner to the previously returned one to get the
 * next.
 *
 * Returns:
 * 0 if a fine host was stored in 'store'
 * 1 if end of hosts
 * [negative] on errors
 */}
function libssh2_knownhost_get(hosts: PLIBSSH2_KNOWNHOSTS;
                      var store: PLIBSSH2_KNOWNHOST;
                      prev: PLIBSSH2_KNOWNHOST): Integer; cdecl;

const
 HAVE_LIBSSH2_AGENT_API = $010202; {/* since 1.2.2 */}


type
 libssh2_agent_publickey = record
    magic: UInt;         {/* magic stored by the library */}
    node: Pointer;	    {/* handle to the internal representation of key */}
    blob: PUCHAR;       {/* public key blob */}
    blob_len: SIZE_T;               {/* length of the public key blob */}
    comment: PAnsiChar;                 {/* comment in printable format */}
  end;
  PLIBSSH2_AGENT_PUBLICKEY = ^libssh2_agent_publickey;

{/*
 * libssh2_agent_init
 *
 * Init an ssh-agent handle. Returns the pointer to the handle.
 *
 */}
function libssh2_agent_init(session: PLIBSSH2_SESSION): PLIBSSH2_AGENT; cdecl;

{/*
 * libssh2_agent_connect()
 *
 * Connect to an ssh-agent.
 *
 * Returns 0 if succeeded, or a negative value for error.
 */}
function libssh2_agent_connect(agent: PLIBSSH2_AGENT): Integer; cdecl;

{/*
 * libssh2_agent_list_identities()
 *
 * Request an ssh-agent to list identities.
 *
 * Returns 0 if succeeded, or a negative value for error.
 */}
function libssh2_agent_list_identities(agent: PLIBSSH2_AGENT): Integer; cdecl;

{/*
 * libssh2_agent_get_identity()
 *
 * Traverse the internal list of public keys. Pass NULL to 'prev' to get
 * the first one. Or pass a poiner to the previously returned one to get the
 * next.
 *
 * Returns:
 * 0 if a fine public key was stored in 'store'
 * 1 if end of public keys
 * [negative] on errors
 */}
function libssh2_agent_get_identity(agent: PLIBSSH2_AGENT;
			    var store: PLIBSSH2_AGENT_PUBLICKEY;
			    prev: PLIBSSH2_AGENT_PUBLICKEY): Integer; cdecl;

{/*
 * libssh2_agent_userauth()
 *
 * Do publickey user authentication with the help of ssh-agent.
 *
 * Returns 0 if succeeded, or a negative value for error.
 */}
function libssh2_agent_userauth(agent: PLIBSSH2_AGENT;
		       const username: PAnsiChar;
           identity: PLIBSSH2_AGENT_PUBLICKEY): Integer; cdecl;

{/*
 * libssh2_agent_disconnect()
 *
 * Close a connection to an ssh-agent.
 *
 * Returns 0 if succeeded, or a negative value for error.
 */}
function libssh2_agent_disconnect(agent: PLIBSSH2_AGENT): Integer; cdecl;

{/*
 * libssh2_agent_free()
 *
 * Free an ssh-agent handle.  This function also frees the internal
 * collection of public keys.
 */}
procedure libssh2_agent_free(agent: PLIBSSH2_AGENT); cdecl;


{/*
 * libssh2_keepalive_config()
 *
 * Set how often keepalive messages should be sent.  WANT_REPLY
 * indicates whether the keepalive messages should request a response
 * from the server.  INTERVAL is number of seconds that can pass
 * without any I/O, use 0 (the default) to disable keepalives.  To
 * avoid some busy-loop corner-cases, if you specify an interval of 1
 * it will be treated as 2.
 *
 * Note that non-blocking applications are responsible for sending the
 * keepalive messages using libssh2_keepalive_send().
 */}
procedure libssh2_keepalive_config(session: PLIBSSH2_SESSION;
                                           want_reply: Integer;
                                           interval: Cardinal); cdecl;

{/*
 * libssh2_keepalive_send()
 *
 * Send a keepalive message if needed.  SECONDS_TO_NEXT indicates how
 * many seconds you can sleep after this call before you need to call
 * it again.  Returns 0 on success, or LIBSSH2_ERROR_SOCKET_SEND on
 * I/O errors.
 */}
function libssh2_keepalive_send(session: PLIBSSH2_SESSION;
                                var seconds_to_next: Integer): Integer; cdecl;
                      

{+// NOTE NOTE NOTE }
{-libssh2_trace() has no function in builds that aren't built with debug }
{-enabled }
{= }

function libssh2_trace(session: PLIBSSH2_SESSION;
                       bitmask: Integer): Integer; cdecl;
const
  LIBSSH2_TRACE_TRANS = (1 shl 1);
const
  LIBSSH2_TRACE_KEX = (1 shl 2);
const
  LIBSSH2_TRACE_AUTH = (1 shl 3);
const
  LIBSSH2_TRACE_CONN = (1 shl 4);
const
  LIBSSH2_TRACE_SCP = (1 shl 5);
const
  LIBSSH2_TRACE_SFTP = (1shl 6);
const
  LIBSSH2_TRACE_ERROR = (1 shl 7);
const
  LIBSSH2_TRACE_PUBLICKEY = (1 shl 8);
const
  LIBSSH2_TRACE_SOCKET = (1 shl 9);

type
  LIBSSH2_TRACE_HANDLER_FUNC = procedure(session: PLIBSSH2_SESSION; P: Pointer;
    const C: PAnsiChar; S: size_t); cdecl;

function libssh2_trace_sethandler(session: PLIBSSH2_SESSION;
                                         context: Pointer;
                                         callback: LIBSSH2_TRACE_HANDLER_FUNC): Integer; cdecl ;


implementation

function libssh2_init; external libssh2_name;
procedure libssh2_exit; external libssh2_name;
function libssh2_session_init_ex; external libssh2_name;
function libssh2_session_abstract; external libssh2_name;
function libssh2_session_callback_set; external libssh2_name;
function libssh2_banner_set; external libssh2_name;
function libssh2_session_startup; external libssh2_name;
function libssh2_session_disconnect_ex; external libssh2_name;
function libssh2_session_free; external libssh2_name;
function libssh2_hostkey_hash; external libssh2_name;
function libssh2_session_hostkey; external libssh2_name;
function libssh2_session_method_pref; external libssh2_name;
function libssh2_session_methods; external libssh2_name;
function libssh2_session_last_error; external libssh2_name;
function libssh2_session_last_errno; external libssh2_name;
function libssh2_session_block_directions; external libssh2_name;
function libssh2_session_flag; external libssh2_name;
function libssh2_userauth_list; external libssh2_name;
function libssh2_userauth_authenticated; external libssh2_name;
function libssh2_userauth_password_ex; external libssh2_name;
function libssh2_userauth_publickey_fromfile_ex; external libssh2_name;
function libssh2_userauth_hostbased_fromfile_ex; external libssh2_name;
function libssh2_userauth_keyboard_interactive_ex; external libssh2_name;
function libssh2_poll; external libssh2_name;
function libssh2_channel_open_ex; external libssh2_name;
function libssh2_channel_direct_tcpip_ex; external libssh2_name;
function libssh2_channel_forward_listen_ex; external libssh2_name;
function libssh2_channel_forward_cancel; external libssh2_name;
function libssh2_channel_forward_accept; external libssh2_name;
function libssh2_channel_setenv_ex; external libssh2_name;
function libssh2_channel_request_pty_ex; external libssh2_name;
function libssh2_channel_request_pty_size_ex; external libssh2_name;
function libssh2_channel_x11_req_ex; external libssh2_name;
function libssh2_channel_process_startup; external libssh2_name;
function libssh2_channel_read_ex; external libssh2_name;
function libssh2_poll_channel_read; external libssh2_name;
function libssh2_channel_window_read_ex; external libssh2_name;
function libssh2_channel_receive_window_adjust; external libssh2_name;
function libssh2_channel_receive_window_adjust2; external libssh2_name;
function libssh2_channel_write_ex; external libssh2_name;
function libssh2_channel_window_write_ex; external libssh2_name;
procedure libssh2_session_set_blocking; external libssh2_name;
function libssh2_session_get_blocking; external libssh2_name;
procedure libssh2_channel_set_blocking; external libssh2_name;
procedure libssh2_channel_handle_extended_data; external libssh2_name;
function libssh2_channel_handle_extended_data2; external libssh2_name;
function libssh2_channel_flush_ex; external libssh2_name;
function libssh2_channel_get_exit_status; external libssh2_name;
function libssh2_channel_send_eof; external libssh2_name;
function libssh2_channel_eof; external libssh2_name;
function libssh2_channel_wait_eof; external libssh2_name;
function libssh2_channel_close; external libssh2_name;
function libssh2_channel_wait_closed; external libssh2_name;
function libssh2_channel_free; external libssh2_name;
function libssh2_scp_recv; external libssh2_name;
function libssh2_scp_send_ex; external libssh2_name;
function libssh2_scp_send64; external libssh2_name;
function libssh2_base64_decode; external libssh2_name;
function libssh2_version; external libssh2_name;
function libssh2_trace; external libssh2_name;
function libssh2_knownhost_init; external libssh2_name;
function libssh2_knownhost_add; external libssh2_name;
function libssh2_knownhost_addc; external libssh2_name;
function libssh2_knownhost_check; external libssh2_name;
function libssh2_knownhost_checkp; external libssh2_name;
function libssh2_knownhost_del; external libssh2_name;
procedure libssh2_knownhost_free; external libssh2_name;
function libssh2_knownhost_readline; external libssh2_name;
function libssh2_knownhost_readfile; external libssh2_name;
function libssh2_knownhost_writeline; external libssh2_name;
function libssh2_knownhost_writefile; external libssh2_name;
function libssh2_knownhost_get; external libssh2_name;
function libssh2_agent_init; external libssh2_name;
function libssh2_agent_connect; external libssh2_name;
function libssh2_agent_list_identities; external libssh2_name;
function libssh2_agent_get_identity; external libssh2_name;
function libssh2_agent_userauth; external libssh2_name;
function libssh2_agent_disconnect; external libssh2_name;
procedure libssh2_agent_free; external libssh2_name;
procedure libssh2_keepalive_config; external libssh2_name;
function libssh2_keepalive_send; external libssh2_name;
//function libssh2_trace; external libssh2_name;
function libssh2_trace_sethandler; external libssh2_name;

function libssh2_session_init: PLIBSSH2_SESSION;
var
 P1: LIBSSH2_ALLOC_FUNC;
 P2: LIBSSH2_REALLOC_FUNC;
 P3: LIBSSH2_FREE_FUNC;
 P4: Pointer;
begin
  P1 := nil; P2 := nil; P3 := nil; P4 := nil;
  Result := libssh2_session_init_ex(P1, P3, P2, P4);
end;

function libssh2_session_disconnect(session: PLIBSSH2_SESSION; const description: PAnsiChar): Integer;
begin
  Result := libssh2_session_disconnect_ex(session, SSH_DISCONNECT_BY_APPLICATION, description, '');
end;

function libssh2_userauth_password(session: PLIBSSH2_SESSION; const username: PAnsiChar; const password: PAnsiChar): Integer;
var
 P: LIBSSH2_PASSWD_CHANGEREQ_FUNC;
begin
 P := nil;
 Result := libssh2_userauth_password_ex(session, username, Length(username), password, Length(password), P)
end;

function libssh2_userauth_publickey_fromfile(session: PLIBSSH2_SESSION; const username: PAnsiChar;
    const publickey: PAnsiChar; const privatekey: PAnsiChar; const passphrase: PAnsiChar): Integer;
begin
  Result := libssh2_userauth_publickey_fromfile_ex(session, username, Length(username), publickey, privatekey, passphrase);
end;

function libssh2_userauth_hostbased_fromfile(session: PLIBSSH2_SESSION; const username: PAnsiChar; const publickey: PAnsiChar;
    const privatekey: PAnsiChar; const passphrase: PAnsiChar; const hostname: PAnsiChar): Integer;
begin
  Result := libssh2_userauth_hostbased_fromfile_ex(session, username, Length(username), publickey, privatekey, passphrase, hostname, Length(hostname), username, Length(username));  
end;

function libssh2_userauth_keyboard_interactive(session: PLIBSSH2_SESSION; const username: PAnsiChar;  response_callback: LIBSSH2_USERAUTH_KBDINT_RESPONSE_FUNC): Integer;
begin
  Result := libssh2_userauth_keyboard_interactive_ex(session, username, Length(username), response_callback);
end;

function libssh2_channel_open_session(session: PLIBSSH2_SESSION): PLIBSSH2_CHANNEL;
begin
  Result := libssh2_channel_open_ex(session, 'session', Length('session') , LIBSSH2_CHANNEL_WINDOW_DEFAULT, LIBSSH2_CHANNEL_PACKET_DEFAULT, nil, 0);
end;

function libssh2_channel_direct_tcpip(session: PLIBSSH2_SESSION; const host: PAnsiChar; port: Integer): PLIBSSH2_CHANNEL;
begin
  Result := libssh2_channel_direct_tcpip_ex(session, host, port, '127.0.0.1', 22);
end;

function libssh2_channel_forward_listen(session: PLIBSSH2_SESSION; port: Integer): PLIBSSH2_LISTENER;
var
 I: Integer;
begin
  I := 0;
  Result := libssh2_channel_forward_listen_ex(session, nil, port, I, 16);
end;

function libssh2_channel_setenv(channel: PLIBSSH2_CHANNEL; const varname: PAnsiChar; const value: PAnsiChar): Integer;
begin
  Result := libssh2_channel_setenv_ex(channel, varname, Length(varname), value, Length(value));
end;

function libssh2_channel_request_pty(channel: PLIBSSH2_CHANNEL; const term: PAnsiChar): Integer;
begin
  Result := libssh2_channel_request_pty_ex(channel, term, Length(term), nil, 0, LIBSSH2_TERM_WIDTH, LIBSSH2_TERM_HEIGHT, LIBSSH2_TERM_WIDTH_PX, LIBSSH2_TERM_HEIGHT_PX);
end;

function libssh2_channel_request_pty_size(channel: PLIBSSH2_CHANNEL; width: Integer; height: Integer): Integer;
begin
  Result := libssh2_channel_request_pty_size_ex(channel, width, height, 0, 0);
end;

function libssh2_channel_x11_req(channel: PLIBSSH2_CHANNEL; screen_number: Integer): Integer;
begin
  Result := libssh2_channel_x11_req_ex(channel, 0, nil, nil, screen_number);
end;

function libssh2_channel_shell(channel: PLIBSSH2_CHANNEL): Integer;
begin
  Result := libssh2_channel_process_startup(channel, 'shell', Length('shell'), nil, 0);
end;

function libssh2_channel_exec(channel: PLIBSSH2_CHANNEL; const command: PAnsiChar): Integer;
begin
  Result := libssh2_channel_process_startup(channel, 'exec', Length('exec'), command, Length(command));
end;

function libssh2_channel_subsystem(channel: PLIBSSH2_CHANNEL; const subsystem: PAnsiChar): Integer;
begin
  Result := libssh2_channel_process_startup(channel, 'subsystem', Length('subsystem'), subsystem, Length(subsystem));
end;

function libssh2_channel_read(channel: PLIBSSH2_CHANNEL; buf: PAnsiChar; buflen: SIZE_T): Integer;
begin
  Result := libssh2_channel_read_ex(channel, 0, buf, buflen);
end;

function libssh2_channel_read_stderr(channel: PLIBSSH2_CHANNEL; buf: PAnsiChar; buflen: SIZE_T): Integer;
begin
  Result := libssh2_channel_read_ex(channel, SSH_EXTENDED_DATA_STDERR, buf, buflen);
end;

function libssh2_channel_window_read(channel: PLIBSSH2_CHANNEL): ULong;
var
  I: Integer;
begin
  I := 0;
  Result := libssh2_channel_window_read_ex(channel, I, I);
end;

function libssh2_channel_write(channel: PLIBSSH2_CHANNEL; const buf: PAnsiChar; buflen: ULong): Integer;
begin
  Result := libssh2_channel_write_ex(channel, 0, buf, buflen);
end;

function libssh2_channel_write_stderr(channel: PLIBSSH2_CHANNEL; const buf: PAnsiChar; buflen: ULong): Integer;
begin
  Result := libssh2_channel_write_ex(channel, SSH_EXTENDED_DATA_STDERR, buf, buflen);
end;

function libssh2_channel_window_write(channel: PLIBSSH2_CHANNEL): ULong;
var
 I: Integer;
begin
  I := 0;
  Result := libssh2_channel_window_write_ex(channel, I);
end;

procedure libssh2_channel_ignore_extended_data(channel: PLIBSSH2_CHANNEL; ignore: Integer);
var
 I: Integer;
begin
  if ignore <> 0 then
    I := LIBSSH2_CHANNEL_EXTENDED_DATA_IGNORE
  else
    I := LIBSSH2_CHANNEL_EXTENDED_DATA_NORMAL;
  libssh2_channel_handle_extended_data(channel, I);
end;

function libssh2_channel_flush(channel: PLIBSSH2_CHANNEL): Integer;
begin
  Result := libssh2_channel_flush_ex(channel, 0);
end;

function libssh2_channel_flush_stderr(channel: PLIBSSH2_CHANNEL): Integer;
begin
  Result := libssh2_channel_flush_ex(channel, SSH_EXTENDED_DATA_STDERR);
end;

function libssh2_scp_send(session: PLIBSSH2_SESSION; const path: PAnsiChar; mode: Integer; size: SIZE_T): PLIBSSH2_CHANNEL; inline;
begin
  Result := libssh2_scp_send_ex(session, path, mode, size, 0, 0);
end;

end.
