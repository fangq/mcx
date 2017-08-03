{****************************************************}
{              SendKeys Unit for Delphi 32           }
{    Copyright (c) 1999 by Borut Batagelj (Slovenia) }
{                       Aleksey Kuznetsov (Ukraine)  }
{            Home Page: www.utilmind.com             }
{            E-Mail: info@utilmind.com               }
{****************************************************}

unit SendKeys;

interface

uses
    Windows, SysUtils;

const
    SK_BKSP = #8;
    SK_TAB = #9;
    SK_ENTER = #13;
    SK_ESC = #27;
    SK_ADD = #107;
    SK_SUB = #109;
    SK_F1 = #228;
    SK_F2 = #229;
    SK_F3 = #230;
    SK_F4 = #231;
    SK_F5 = #232;
    SK_F6 = #233;
    SK_F7 = #234;
    SK_F8 = #235;
    SK_F9 = #236;
    SK_F10 = #237;
    SK_F11 = #238;
    SK_F12 = #239;
    SK_HOME = #240;
    SK_END = #241;
    SK_UP = #242;
    SK_DOWN = #243;
    SK_LEFT = #244;
    SK_RIGHT = #245;
    SK_PGUP = #246;
    SK_PGDN = #247;
    SK_INS = #248;
    SK_DEL = #249;
    SK_SHIFT_DN = #250;
    SK_SHIFT_UP = #251;
    SK_CTRL_DN = #252;
    SK_CTRL_UP = #253;
    SK_ALT_DN = #254;
    SK_ALT_UP = #255;

procedure SendKeyString(s: String);
procedure SendKeysToTitle(WindowTitle: String; Text: String);
procedure SendKeysToHandle(WindowHandle: hWnd; Text: String);
procedure MakeWindowActive(wHandle: hWnd);
function GetHandleFromWindowTitle(TitleText: String): hWnd;

implementation

procedure SimulateKeyDown(Key : byte);
begin
  keybd_event(Key, 0, 0, 0);
end;

procedure SimulateKeyUp(Key : byte);
begin
  keybd_event(Key, 0, KEYEVENTF_KEYUP, 0);
end;

procedure SimulateKeystroke(Key : byte;
                            extra : DWORD);
begin
  keybd_event(Key,
              extra,
              0,
              0);
  keybd_event(Key,
              extra,
              KEYEVENTF_KEYUP,
              0);
end;

procedure SendKeyString(s : string);
var
  i : integer;
  flag : bool;
  w : word;
begin
 {Get the state of the caps lock key}
  flag := not GetKeyState(VK_CAPITAL) and 1 = 0;
 {If the caps lock key is on then turn it off}
  if flag then
    SimulateKeystroke(VK_CAPITAL, 0);
  for i := 1 to Length(s) do begin
    w := VkKeyScan(s[i]);
   {If there is not an error in the key translation}
    if ((HiByte(w) <> $FF) and
        (LoByte(w) <> $FF)) then begin
     {If the key requires the shift key down - hold it down}
      if HiByte(w) and 1 = 1 then
        SimulateKeyDown(VK_SHIFT);
     {Send the VK_KEY}
      SimulateKeystroke(LoByte(w), 0);
     {If the key required the shift key down - release it}
      if HiByte(w) and 1 = 1 then
        SimulateKeyUp(VK_SHIFT);
    end;
  end;
 {if the caps lock key was on at start, turn it back on}
  if flag then
    SimulateKeystroke(VK_CAPITAL, 0);
end;


procedure MakeWindowActive(wHandle: hWnd);
begin
    if IsIconic(wHandle) then
        ShowWindow(wHandle, SW_RESTORE)
    else
        BringWindowToTop(wHandle);
end;

function GetHandleFromWindowTitle(TitleText: String): hWnd;
var
    StrBuf: Array[0..$FF] of Char;
begin
    Result := FindWindow(PChar(0), StrPCopy(StrBuf, TitleText));
end;

procedure SendKeysToTitle(WindowTitle: String; Text: String);
var
    Window: hWnd;
begin
    Window := GetHandleFromWindowTitle(WindowTitle);
    MakeWindowActive(Window);
    SendKeyString(Text);
end;

procedure SendKeysToHandle(WindowHandle: hWnd; Text: String);
begin
    MakeWindowActive(WindowHandle);
    SendKeyString(Text);
end;

end.
