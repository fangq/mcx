//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Cross platform support functions and types for GLScene.

   Ultimately, *no* cross-platform or cross-version defines should be present
   in the core GLScene units, and have all moved here instead.

 History :  
       22/04/14 - PW -  Ceased support of GLS_DELPHI_5 and GLS_COMPILER_5 or DOWN
       10/11/12 - PW - Added CPP compatibility: restored $NODEFINE to remove
                          redeclarations of RGB, GLPoint, GLRect and some other types
       30/06/11 - DaStr - Added CharToWideChar()
       19/06/11 - Yar - Added IsDirectoryWriteable
       15/04/11 - AsmRu - Added GetPlatformInfo, GetPlatformVersion
       19/03/11 - Yar - Added procedure FixPathDelimiter, RelativePath
       04/11/10 - DaStr - Added functions missing in Delphi5 and Delphi6:
                             TAssertErrorProc, GetValueFromStringsIndex and some types
       31/10/10 - Yar - Added GLSTime
       18/10/10 - Yar - Added functions FloatToHalf, HalfToFloat (Thanks to Fantom)
       04/09/10 - Yar - Added IsDesignTime variable, SetExeDirectory
       15/06/10 - Yar - Replace Shell to fpSystem
       04/03/10 - DanB - Added CharInSet, for Delphi versions < 2009
       07/01/10 - DaStr - Bugfixed GetDeviceCapabilities() for Unix
                             (thanks Predator)
       17/12/09 - DaStr - Moved screen utility functions to GLScreen.pas
                             (thanks Predator)
       11/11/09 - DaStr - Added GLS_FONT_CHARS_COUNT constant
       07/11/09 - DaStr - Improved FPC compatibility (BugtrackerID = 2893580)
                             (thanks Predator)
       24/08/09 - DaStr - Added IncludeTrailingPathDelimiter for Delphi 5
       03/06/09 - DanB - Re-added Sleep procedure, for Delphi 5
       07/05/09 - DanB - Added FindUnitName (to provide functionality of TObject.UnitName,
                            on prior versions of Delphi)
       24/03/09 - DanB - Moved Dialog utility functions to GLUtils.pas, new ShowHTMLUrl procedure
       19/03/09 - DanB - Removed some Kylix IFDEFs, and other changes mostly affecting D5/FPC
       29/05/08 - DaStr - Added StrToFloatDef(), TryStrToFloat()
       10/04/08 - DaStr - Added TGLComponent (BugTracker ID = 1938988)
       07/04/08 - DaStr - Added IsInfinite, IsNan
       18/11/07 - DaStr - Added ptrInt and PtrUInt types (BugtrackerID = 1833830)
                              (thanks Dje and Burkhard Carstens)
       06/06/07 - DaStr - Added WORD type
                             Got rid of GLTexture.pas dependancy
                             Moved GetRValue, GetGValue, GetBValue, InitWinColors
                               to GLColor.pas (BugtrackerID = 1732211)
       02/04/07 - DaStr - Added MakeSubComponent
                             Fixed some IFDEFs to separate FPC from Kylix
       25/03/07 - DaStr - Replaced some UNIX IFDEFs with KYLIX
                             Added IdentToColor, ColorToIdent, ColorToString,
                                   AnsiStartsText, IsSubComponent
                             Added TPoint, PPoint, TRect, PRect, TPicture, TGraphic,
                                   TBitmap, TTextLayout, TMouseButton, TMouseEvent,
                                   TKeyEvent, TKeyPressEvent
                             Added IInterface, S_OK, E_NOINTERFACE,
                                   glKey_PRIOR, glKey_NEXT, glKey_CONTROL
       24/03/07 - DaStr - Added TPenStyle, TPenMode, TBrushStyle, more color constants,
                             Added "Application" function
       17/03/07 - DaStr - Dropped Kylix support in favor of FPC (BugTracekrID=1681585)
       02/08/04 - LR, YHC - BCB corrections: Added namespace 'Graphics' to TGLBitmap
                               use $NODEFINE to remove declarations of duplicate variables
       08/07/04 - LR - Added clBlack
       03/07/04 - LR - Added constant for Keyboard (glKey_TAB, ...)
                          Added function GLOKMessageBox to avoid the uses of Forms
                          Added other abstraction calls
                          Added procedure ShowHTMLUrl for unit Info.pas
                          Added GLShowCursor, GLSetCursorPos, GLGetCursorPos,
                          GLGetScreenWidth, GLGetScreenHeight for GLNavigation
                          Added GLGetTickCount for GLFPSMovement
       28/06/04 - LR - Added TGLTextLayout, GLLoadBitmapFromInstance
                          Added GetDeviceCapabilities to replace the old function
       30/05/03 - EG - Added RDTSC and RDTSC-based precision timing for non-WIN32
       22/01/02 - EG - Added OpenPictureDialog, ApplicationTerminated
       07/01/02 - EG - Added QuestionDialog and SavePictureDialog,
                          Added PrecisionTimer funcs
       06/12/01 - EG - Added several abstraction calls
       31/08/01 - EG - Creation
  
}
unit GLCrossPlatform;

interface

{$INCLUDE GLScene.inc}

uses
{$IFDEF MSWINDOWS}
  Windows,
{$ENDIF}
{$IFDEF UNIX}
  Unix, BaseUnix,
{$ENDIF}
{$IFDEF GLS_X11_SUPPORT}
  xlib,
{$ENDIF}
  Types,
   Classes, SysUtils, StrUtils, Graphics,  Controls,
   Forms,  Dialogs,LCLVersion,  LCLType,  LazFileUtils, LazUTF8;



type


  // Several aliases to shield us from the need of ifdef'ing between
  // the "almost cross-platform" units like Graphics/QGraphics etc.
  // Gives a little "alien" look to names, but that's the only way around :(

  // DaStr: Actually, there is a way around, see TPenStyle for example.

  TGLPoint = TPoint;

  PGLPoint = ^TGLPoint;
  TGLRect = TRect;
  PGLRect = ^TGLRect;
  TDelphiColor = TColor;

  TGLPicture = TPicture;
  TGLGraphic = TGraphic;
  TGLBitmap = TBitmap;
  TGraphicClass = class of TGraphic;

  TGLTextLayout = (tlTop, tlCenter, tlBottom); // idem TTextLayout;

  TGLMouseButton = (mbLeft, mbRight, mbMiddle); // idem TMouseButton;
  TGLMouseEvent = procedure(Sender: TObject; Button: TGLMouseButton;
    Shift: TShiftState; X, Y: Integer) of object;
  TGLMouseMoveEvent = TMouseMoveEvent;
  TGLKeyEvent = TKeyEvent;
  TGLKeyPressEvent = TKeyPressEvent;

  TPlatformInfo = record
    Major: DWORD;
    Minor: DWORD;
    Revision: DWORD;
    Version: string;
    PlatformId   :DWORD;
    ID: string;
    CodeName: string;
    Description: string;
    ProductBuildVersion: string;
  end;

  TPlatformVersion =
    (
      pvUnknown,
      pvWin95,
      pvWin98,
      pvWinME,
      pvWinNT3,
      pvWinNT4,
      pvWin2000,
      pvWinXP,
      pvWin2003,
      pvWinVista,
      pvWinSeven,
      pvWin2008,
      pvWinNew,

      pvLinuxArc,
      pvLinuxDebian,
      pvLinuxopenSUSE,
      pvLinuxFedora,
      pvLinuxGentoo,
      pvLinuxMandriva,
      pvLinuxRedHat,
      pvLinuxTurboLinux,
      pvLinuxUbuntu,
      pvLinuxXandros,
      pvLinuxOracle,
      pvAppleMacOSX
    );

  EGLOSError = EOSError;



  TGLComponent = class(TComponent);

  DWORD = System.DWORD;
  TPoint = Types.TPoint;
  PPoint = ^TPoint;
  TRect = Types.TRect;
  PRect = ^TRect;


  TProjectTargetNameFunc = function(): string;

  THalfFloat = type Word;
  PHalfFloat = ^THalfFloat;

const
{$IFDEF MSWINDOWS}
  glpf8Bit = pf8bit;
  glpf24bit = pf24bit;
  glpf32Bit = pf32bit;
  glpfDevice = pfDevice;
{$ENDIF}
{$IFDEF UNIX}
  glpf8Bit = pf8bit;
  glpf24bit = pf32bit;
  glpf32Bit = pf32bit;
  glpfDevice = pf32bit;
{$ENDIF}

  // standard keyboard
  glKey_TAB = VK_TAB;
  glKey_SPACE = VK_SPACE;
  glKey_RETURN = VK_RETURN;
  glKey_DELETE = VK_DELETE;
  glKey_LEFT = VK_LEFT;
  glKey_RIGHT = VK_RIGHT;
  glKey_HOME = VK_HOME;
  glKey_END = VK_END;
  glKey_CANCEL = VK_CANCEL;
  glKey_UP = VK_UP;
  glKey_DOWN = VK_DOWN;
  glKey_PRIOR = VK_PRIOR;
  glKey_NEXT = VK_NEXT;
  glKey_CONTROL = VK_CONTROL;

  // TPenStyle.
  //FPC doesn't support TPenStyle "psInsideFrame", so provide an alternative
  psInsideFrame = psSolid;


  // Several define from unit Consts
const

  glsAllFilter: string = 'All';
  GLS_FONT_CHARS_COUNT = 256;


var
  IsDesignTime: Boolean = False;
  vProjectTargetName: TProjectTargetNameFunc;

function GLPoint(const x, y: Integer): TGLPoint;
{ Builds a TColor from Red Green Blue components. }
function RGB(const r, g, b: Byte): TColor; {$NODEFINE RGB}
function GLRect(const aLeft, aTop, aRight, aBottom: Integer): TGLRect;{$NODEFINE GLRect}
{ Increases or decreases the width and height of the specified rectangle.
   Adds dx units to the left and right ends of the rectangle and dy units to
   the top and bottom. }
procedure InflateGLRect(var aRect: TGLRect; dx, dy: Integer);
procedure IntersectGLRect(var aRect: TGLRect; const rect2: TGLRect);
function PtInRect(const Rect: TGLRect; const P: TPoint): Boolean;

procedure RaiseLastOSError;

{ Number of pixels per logical inch along the screen width for the device.
   Under Win32 awaits a HDC and returns its LOGPIXELSX. }
function GetDeviceLogicalPixelsX(device: HDC): Integer;
{ Number of bits per pixel for the current desktop resolution. }
function GetCurrentColorDepth: Integer;
{ Returns the number of color bits associated to the given pixel format. }
function PixelFormatToColorBits(aPixelFormat: TPixelFormat): Integer;

{ Returns the bitmap's scanline for the specified row. }
function BitmapScanLine(aBitmap: TGLBitmap; aRow: Integer): Pointer;

{ Replace path delimiter to delimiter of the current platform. }
procedure FixPathDelimiter(var S: string);
{ Remove if possible part of path witch leads to project executable. }
function RelativePath(const S: string): string;
{ Returns the current value of the highest-resolution counter.
   If the platform has none, should return a value derived from the highest
   precision time reference available, avoiding, if possible, timers that
   allocate specific system resources. }
procedure QueryPerformanceCounter(var val: Int64);
{ Returns the frequency of the counter used by QueryPerformanceCounter.
   Return value is in ticks per second (Hz), returns False if no precision
   counter is available. }
function QueryPerformanceFrequency(var val: Int64): Boolean;

{ Starts a precision timer.
   Returned value should just be considered as 'handle', even if it ain't so.
   Default platform implementation is to use QueryPerformanceCounter and
   QueryPerformanceFrequency, if higher precision references are available,
   they should be used. The timer will and must be stopped/terminated/released
   with StopPrecisionTimer. }
function StartPrecisionTimer: Int64;
{ Computes time elapsed since timer start.
   Return time lap in seconds. }
function PrecisionTimerLap(const precisionTimer: Int64): Double;
{ Computes time elapsed since timer start and stop timer.
   Return time lap in seconds. }
function StopPrecisionTimer(const precisionTimer: Int64): Double;

{ Returns time in milisecond from application start.}
function GLSTime: Double;

{ Returns the number of CPU cycles since startup.
   Use the similarly named CPU instruction. }
function RDTSC: Int64;

function GLOKMessageBox(const Text, Caption: string): Integer;
procedure GLLoadBitmapFromInstance(Instance: LongInt; ABitmap: TBitmap; AName: string);
procedure ShowHTMLUrl(Url: string);
function GLGetTickCount: int64;
procedure SetExeDirectory;
function GetDecimalSeparator: Char;
procedure SetDecimalSeparator(AValue: Char);

// StrUtils.pas
function AnsiStartsText(const ASubText, AText: string): Boolean;

// Classes.pas
function IsSubComponent(const AComponent: TComponent): Boolean;
procedure MakeSubComponent(const AComponent: TComponent; const Value: Boolean);

function FindUnitName(anObject: TObject): string; overload;
function FindUnitName(aClass: TClass): string; overload;

{$IFNDEF GLS_COMPILER_2009_UP}
function CharInSet(C: AnsiChar; const CharSet: TSysCharSet): Boolean; overload;
function CharInSet(C: WideChar; const CharSet: TSysCharSet): Boolean; overload;
{$ENDIF}

function FloatToHalf(Float: Single): THalfFloat;
function HalfToFloat(Half: THalfFloat): Single;

function GetValueFromStringsIndex(const AStrings: TStrings; const AIndex: Integer): string;

function GetPlatformInfo: TPlatformInfo;
function GetPlatformVersion : TPlatformVersion;
function GetPlatformVersionStr : string;

{ Determine if the directory is writable. }
function IsDirectoryWriteable(const AName: string): Boolean;

function CharToWideChar(const AChar: AnsiChar): WideChar;

implementation

uses
{$IFDEF MSWINDOWS}ShellApi{$ENDIF}
{$IFDEF Darwin}XMLRead,DOM,{$ENDIF}
{$IFDEF UNIX}  LCLProc  {$ENDIF} ;


var
  vInvPerformanceCounterFrequency: Double;
  vInvPerformanceCounterFrequencyReady: Boolean = False;
  vLastProjectTargetName: string;

function IsSubComponent(const AComponent: TComponent): Boolean;
begin
  Result := (csSubComponent in AComponent.ComponentStyle);
end;

procedure MakeSubComponent(const AComponent: TComponent; const Value: Boolean);
begin
  AComponent.SetSubComponent(Value);
end;

function AnsiStartsText(const ASubText, AText: string): Boolean;
begin
  Result := AnsiStartsText(ASubText, AText);
end;

function GLOKMessageBox(const Text, Caption: string): Integer;
begin
  Result := Application.MessageBox(PChar(Text), PChar(Caption), MB_OK);
end;

procedure GLLoadBitmapFromInstance(Instance: LongInt; ABitmap: TBitmap; AName: string);
begin
{$IFDEF MSWINDOWS}
  ABitmap.Handle := LoadBitmap(Instance, PChar(AName));
{$ENDIF}
{$IFDEF UNIX}
  ABitmap.LoadFromResourceName(Instance, PChar(AName));
{$ENDIF}
end;

function GLGetTickCount: int64;
begin
{$IFDEF MSWINDOWS}
  result := GetTickCount64;
{$ENDIF}
{$IFDEF UNIX}
  QueryPerformanceCounter(result);
{$ENDIF}
end;

procedure ShowHTMLUrl(Url: string);
begin
{$IFDEF MSWINDOWS}
  ShellExecute(0, 'open', PChar(Url), nil, nil, SW_SHOW);
{$ENDIF}
{$IFDEF UNIX}
  fpSystem(PChar('env xdg-open ' + Url));
{$ENDIF}
end;

// GLPoint
//

function GLPoint(const x, y: Integer): TGLPoint;
begin
  Result.X := x;
  Result.Y := y;
end;

// RGB
//

function RGB(const r, g, b: Byte): TColor;
begin
  Result := r or (g shl 8) or (b shl 16);
end;

// GLRect
//

function GLRect(const aLeft, aTop, aRight, aBottom: Integer): TGLRect;
begin
  Result.Left := aLeft;
  Result.Top := aTop;
  Result.Right := aRight;
  Result.Bottom := aBottom;
end;

// InflateRect
//

procedure InflateGLRect(var aRect: TGLRect; dx, dy: Integer);
begin
  aRect.Left := aRect.Left - dx;
  aRect.Right := aRect.Right + dx;
  if aRect.Right < aRect.Left then
    aRect.Right := aRect.Left;
  aRect.Top := aRect.Top - dy;
  aRect.Bottom := aRect.Bottom + dy;
  if aRect.Bottom < aRect.Top then
    aRect.Bottom := aRect.Top;
end;

// IntersectGLRect
//

procedure IntersectGLRect(var aRect: TGLRect; const rect2: TGLRect);
var
  a: Integer;
begin
  if (aRect.Left > rect2.Right) or (aRect.Right < rect2.Left)
    or (aRect.Top > rect2.Bottom) or (aRect.Bottom < rect2.Top) then
  begin
    // no intersection
    a := 0;
    aRect.Left := a;
    aRect.Right := a;
    aRect.Top := a;
    aRect.Bottom := a;
  end
  else
  begin
    if aRect.Left < rect2.Left then
      aRect.Left := rect2.Left;
    if aRect.Right > rect2.Right then
      aRect.Right := rect2.Right;
    if aRect.Top < rect2.Top then
      aRect.Top := rect2.Top;
    if aRect.Bottom > rect2.Bottom then
      aRect.Bottom := rect2.Bottom;
  end;
end;

// RaiseLastOSError
//

procedure RaiseLastOSError;
var
  e: EGLOSError;
begin

  e := EGLOSError.Create('OS Error : ' + SysErrorMessage(GetLastOSError));
  raise e;
end;

type
  TDeviceCapabilities = record
    Xdpi, Ydpi: integer; // Number of pixels per logical inch.
    Depth: integer; // The bit depth.
    NumColors: integer; // Number of entries in the device's color table.
  end;

function GetDeviceCapabilities: TDeviceCapabilities;
{$IFDEF MSWINDOWS}
var
  Device: HDC;
begin
  Device := GetDC(0);
  try
    result.Xdpi := GetDeviceCaps(Device, LOGPIXELSX);
    result.Ydpi := GetDeviceCaps(Device, LOGPIXELSY);
    result.Depth := GetDeviceCaps(Device, BITSPIXEL);
    result.NumColors := GetDeviceCaps(Device, NUMCOLORS);
  finally
    ReleaseDC(0, Device);
  end;
end;
{$ELSE}
{$IFDEF GLS_X11_SUPPORT}
var
  dpy: PDisplay;
begin
  dpy := XOpenDisplay(nil);
  Result.Depth := DefaultDepth(dpy, DefaultScreen(dpy));
  XCloseDisplay(dpy);

  Result.Xdpi := 96;
  Result.Ydpi := 96;
  Result.NumColors := 1;
end;
{$ELSE}
begin
  {$MESSAGE Warn 'Needs to be implemented'}
end;
{$ENDIF}

{$ENDIF}

// GetDeviceLogicalPixelsX
//

function GetDeviceLogicalPixelsX(device: HDC): Integer;
begin
  result := GetDeviceCapabilities().Xdpi;
end;

// GetCurrentColorDepth
//

function GetCurrentColorDepth: Integer;
begin
  result := GetDeviceCapabilities().Depth;
end;

// PixelFormatToColorBits
//

function PixelFormatToColorBits(aPixelFormat: TPixelFormat): Integer;
begin
  case aPixelFormat of
    pfCustom{$IFDEF WIN32}, pfDevice{$ENDIF}: // use current color depth
      Result := GetCurrentColorDepth;
    pf1bit: Result := 1;
{$IFDEF WIN32}
    pf4bit: Result := 4;
    pf15bit: Result := 15;
{$ENDIF}
    pf8bit: Result := 8;
    pf16bit: Result := 16;
    pf32bit: Result := 32;
  else
    Result := 24;
  end;
end;

// BitmapScanLine
//

function BitmapScanLine(aBitmap: TGLBitmap; aRow: Integer): Pointer;
begin
  Assert(False, 'BitmapScanLine unsupported');
  Result := nil;
end;

procedure FixPathDelimiter(var S: string);
var
  I: Integer;
begin
  for I := Length(S) downto 1 do
    if (S[I] = '/') or (S[I] = '\') then
      S[I] := PathDelim;
end;

function RelativePath(const S: string): string;
var
  path: UTF8String;

begin
  Result := S;
  if IsDesignTime then
  begin
    if Assigned(vProjectTargetName) then
    begin
      path :=  vProjectTargetName();
      if Length(path) = 0 then
        path := vLastProjectTargetName
      else
        vLastProjectTargetName := path;
      path := IncludeTrailingPathDelimiter(ExtractFilePath(path));
    end
    else
      exit;
  end
  else
  begin
    path := ExtractFilePath(ParamStr(0));
    path := IncludeTrailingPathDelimiter(path);
  end;
  if Pos(path, S) = 1 then
    Delete(Result, 1, Length(path));
end;

// QueryPerformanceCounter
//
{$IFDEF UNIX}
var
  vProgStartSecond: int64;

procedure Init_vProgStartSecond;
var
  tz: timeval;
begin
  fpgettimeofday(@tz, nil);
  vProgStartSecond := tz.tv_sec;
end;
{$ENDIF}

procedure QueryPerformanceCounter(var val: Int64);
{$IFDEF MSWINDOWS}
begin
  Windows.QueryPerformanceCounter(val);
end;
{$ENDIF}
{$IFDEF UNIX}
var
  tz: timeval;
begin
  fpgettimeofday(@tz, nil);
  val := tz.tv_sec - vProgStartSecond;
  val := val * 1000000;
  val := val + tz.tv_usec;
end;
{$ENDIF}

// QueryPerformanceFrequency
//

function QueryPerformanceFrequency(var val: Int64): Boolean;
{$IFDEF MSWINDOWS}
begin
  Result := Boolean(Windows.QueryPerformanceFrequency(val));
end;
{$ENDIF}
{$IFDEF UNIX}
begin
  val := 1000000;
  Result := True;
end;
{$ENDIF}

// StartPrecisionTimer
//

function StartPrecisionTimer: Int64;
begin
  QueryPerformanceCounter(Result);
end;

// PrecisionTimeLap
//

function PrecisionTimerLap(const precisionTimer: Int64): Double;
begin
  // we can do this, because we don't really stop anything
  Result := StopPrecisionTimer(precisionTimer);
end;

// StopPrecisionTimer
//

function StopPrecisionTimer(const precisionTimer: Int64): Double;
var
  cur, freq: Int64;
begin
  QueryPerformanceCounter(cur);
  if not vInvPerformanceCounterFrequencyReady then
  begin
    QueryPerformanceFrequency(freq);
    vInvPerformanceCounterFrequency := 1.0 / freq;
    vInvPerformanceCounterFrequencyReady := True;
  end;
  Result := (cur - precisionTimer) * vInvPerformanceCounterFrequency;
end;

var
  vGLSStartTime : TDateTime;
{$IFDEF MSWINDOWS}
  vLastTime: TDateTime;
  vDeltaMilliSecond: TDateTime;
{$ENDIF}

function GLSTime: Double;
{$IFDEF MSWINDOWS}
var
  SystemTime: TSystemTime;
begin
  GetLocalTime(SystemTime);
  with SystemTime do
    Result := (wHour * (MinsPerHour * SecsPerMin * MSecsPerSec) +
             wMinute * (SecsPerMin * MSecsPerSec) +
             wSecond * MSecsPerSec +
             wMilliSeconds) - vGLSStartTime;
  // Hack to fix time precession
  if Result - vLastTime = 0 then
  begin
    Result := Result + vDeltaMilliSecond;
    vDeltaMilliSecond := vDeltaMilliSecond + 0.1;
  end
  else begin
    vLastTime := Result;
    vDeltaMilliSecond := 0.1;
  end;
end;
{$ENDIF}

{$IFDEF UNIX}
var
  tz: timeval;
begin
  fpgettimeofday(@tz, nil);
  Result := tz.tv_sec - vGLSStartTime;
  Result := Result * 1000000;
  Result := Result + tz.tv_usec;
// Delphi for Linux variant (for future ;)
//var
//  T: TTime_T;
//  TV: TTimeVal;
//  UT: TUnixTime;
//begin
//  gettimeofday(TV, nil);
//  T := TV.tv_sec;
//  localtime_r(@T, UT);
//  with UT do
//    Result := (tm_hour * (MinsPerHour * SecsPerMin * MSecsPerSec) +
//             tm_min * (SecsPerMin * MSecsPerSec) +
//             tm_sec * MSecsPerSec +
//             tv_usec div 1000) - vGLSStartTime;
end;
{$ENDIF}

// RDTSC
//
function RDTSC: Int64;

begin
  raise exception.create('Using GLCrossPlatform.RDTSC is a bad idea!');
  Result := 0;
end;


type
  PClassData = ^TClassData;
  TClassData = record
    ClassType: TClass;
    ParentInfo: Pointer;
    PropCount: SmallInt;
    UnitName: ShortString;
  end;


function FindUnitName(anObject: TObject): string;

var
  LClassInfo: Pointer;
begin
  Result := '';
  if anObject = nil then
    Exit;

  LClassInfo := anObject.ClassInfo;
  if LClassInfo <> nil then
   Result := string(PClassData(Integer(LClassInfo) + 2 + PByte(Integer(LClassInfo) + 1)^).UnitName);
end;


function FindUnitName(aClass: TClass): string;

var
  LClassInfo: Pointer;
begin
  Result := '';
  if aClass = nil then
    Exit;

  LClassInfo := aClass.ClassInfo;
  if LClassInfo <> nil then
    Result := string(PClassData(Integer(LClassInfo) + 2 + PByte(Integer(LClassInfo) + 1)^).UnitName);
end;


function CharInSet(C: AnsiChar; const CharSet: TSysCharSet): Boolean;
begin
  Result := C in CharSet;
end;

function CharInSet(C: WideChar; const CharSet: TSysCharSet): Boolean;
begin
  Result := (C < #$0100) and (AnsiChar(C) in CharSet);
end;

procedure SetExeDirectory;
var
  path: UTF8String;

begin
  if IsDesignTime then
  begin
    if Assigned(vProjectTargetName) then
    begin
      path :=  vProjectTargetName();
      if Length(path) = 0 then
        path := vLastProjectTargetName
      else
        vLastProjectTargetName := path;
      path := IncludeTrailingPathDelimiter(ExtractFilePath(path));
      SetCurrentDirUTF8(path);
    end;
  end
  else
  begin

    path := ExtractFilePath(ParamStr(0));
    path := IncludeTrailingPathDelimiter(path);
    SetCurrentDirUTF8(path);
  end;
end;

function GetDecimalSeparator: Char;
begin
  Result := DefaultFormatSettings.DecimalSeparator;
end;

procedure SetDecimalSeparator(AValue: Char);
begin
 DefaultFormatSettings.DecimalSeparator := AValue;
end;

function HalfToFloat(Half: THalfFloat): Single;
var
  Dst, Sign, Mantissa: LongWord;
  Exp: LongInt;
begin
  // extract sign, exponent, and mantissa from half number
  Sign := Half shr 15;
  Exp := (Half and $7C00) shr 10;
  Mantissa := Half and 1023;

  if (Exp > 0) and (Exp < 31) then
  begin
    // common normalized number
    Exp := Exp + (127 - 15);
    Mantissa := Mantissa shl 13;
    Dst := (Sign shl 31) or (LongWord(Exp) shl 23) or Mantissa;
    // Result := Power(-1, Sign) * Power(2, Exp - 15) * (1 + Mantissa / 1024);
  end
  else if (Exp = 0) and (Mantissa = 0) then
  begin
    // zero - preserve sign
    Dst := Sign shl 31;
  end
  else if (Exp = 0) and (Mantissa <> 0) then
  begin
    // denormalized number - renormalize it
    while (Mantissa and $00000400) = 0 do
    begin
      Mantissa := Mantissa shl 1;
      Dec(Exp);
    end;
    Inc(Exp);
    Mantissa := Mantissa and not $00000400;
    // now assemble normalized number
    Exp := Exp + (127 - 15);
    Mantissa := Mantissa shl 13;
    Dst := (Sign shl 31) or (LongWord(Exp) shl 23) or Mantissa;
    // Result := Power(-1, Sign) * Power(2, -14) * (Mantissa / 1024);
  end
  else if (Exp = 31) and (Mantissa = 0) then
  begin
    // +/- infinity
    Dst := (Sign shl 31) or $7F800000;
  end
  else //if (Exp = 31) and (Mantisa <> 0) then
  begin
    // not a number - preserve sign and mantissa
    Dst := (Sign shl 31) or $7F800000 or (Mantissa shl 13);
  end;

  // reinterpret LongWord as Single
  Result := PSingle(@Dst)^;
end;

function FloatToHalf(Float: Single): THalfFloat;
var
  Src: LongWord;
  Sign, Exp, Mantissa: LongInt;
begin
  Src := PLongWord(@Float)^;
  // extract sign, exponent, and mantissa from Single number
  Sign := Src shr 31;
  Exp := LongInt((Src and $7F800000) shr 23) - 127 + 15;
  Mantissa := Src and $007FFFFF;

  if (Exp > 0) and (Exp < 30) then
  begin
    // simple case - round the significand and combine it with the sign and exponent
    Result := (Sign shl 15) or (Exp shl 10) or ((Mantissa + $00001000) shr 13);
  end
  else if Src = 0 then
  begin
    // input float is zero - return zero
    Result := 0;
  end
  else
  begin
    // difficult case - lengthy conversion
    if Exp <= 0 then
    begin
      if Exp < -10 then
      begin
        // input float's value is less than HalfMin, return zero
        Result := 0;
      end
      else
      begin
        // Float is a normalized Single whose magnitude is less than HalfNormMin.
        // We convert it to denormalized half.
        Mantissa := (Mantissa or $00800000) shr (1 - Exp);
        // round to nearest
        if (Mantissa and $00001000) > 0 then
          Mantissa := Mantissa + $00002000;
        // assemble Sign and Mantissa (Exp is zero to get denotmalized number)
        Result := (Sign shl 15) or (Mantissa shr 13);
      end;
    end
    else if Exp = 255 - 127 + 15 then
    begin
      if Mantissa = 0 then
      begin
        // input float is infinity, create infinity half with original sign
        Result := (Sign shl 15) or $7C00;
      end
      else
      begin
        // input float is NaN, create half NaN with original sign and mantissa
        Result := (Sign shl 15) or $7C00 or (Mantissa shr 13);
      end;
    end
    else
    begin
      // Exp is > 0 so input float is normalized Single

      // round to nearest
      if (Mantissa and $00001000) > 0 then
      begin
        Mantissa := Mantissa + $00002000;
        if (Mantissa and $00800000) > 0 then
        begin
          Mantissa := 0;
          Exp := Exp + 1;
        end;
      end;

      if Exp > 30 then
      begin
        // exponent overflow - return infinity half
        Result := (Sign shl 15) or $7C00;
      end
      else
        // assemble normalized half
        Result := (Sign shl 15) or (Exp shl 10) or (Mantissa shr 13);
    end;
  end;
end;

function GetValueFromStringsIndex(const AStrings: TStrings; const AIndex: Integer): string;
begin
  Result := AStrings.ValueFromIndex[AIndex];
end;

function GetPlatformInfo: TPlatformInfo;
var
  {$IFDEF MSWINDOWS}
  OSVersionInfo : TOSVersionInfo;
  {$ENDIF}
  {$IFDEF UNIX}
    {$IFNDEF DARWIN}
  ReleseList: TStringList;
    {$ENDIF}
  str: String;
    {$IFDEF DARWIN}
  Documento: TXMLDocument;
  Child: TDOMNode;
  i:integer;
    {$ENDIF}
  {$ENDIF}
begin
  {$IFDEF MSWINDOWS}
  With Result, OSVersionInfo do
  begin
    dwOSVersionInfoSize := sizeof(TOSVersionInfo);

    if not GetVersionEx(OSVersionInfo) then Exit;

    Minor := DwMinorVersion;
    Major := DwMajorVersion;
    Revision := 0;
    PlatformId := dwPlatformId;
    Version :=  InttoStr(DwMajorVersion)+'.'+InttoStr(DwMinorVersion);
  end;
  {$ENDIF}
  {$IFDEF UNIX}
  {$IFNDEF DARWIN}
  ReleseList := TStringList.Create;

  with Result,ReleseList do
  begin
    if FileExists('/etc/lsb-release')  then
      LoadFromFile('/etc/lsb-release')
    else Exit;

    ID := Values['DISTRIB_ID'];
    Version := Values['DISTRIB_RELEASE'];
    CodeName := Values['DISTRIB_CODENAME'];
    Description := Values['DISTRIB_DESCRIPTION'];
    Destroy;
  end;
  {$ENDIF}
  {$IFDEF Darwin}
  if FileExists('System/Library/CoreServices/ServerVersion.plist')  then
    ReadXMLFile(Documento, 'System/Library/CoreServices/ServerVersion.plist')
  else Exit;
  Child := Documento.DocumentElement.FirstChild;

  if Assigned(Child) then
  begin
    with Child.ChildNodes do
    try
      for i := 0 to (Count - 1) do
      begin
        if Item[i].FirstChild.NodeValue='ProductBuildVersion' then
          Result.ProductBuildVersion:=Item[i].NextSibling.FirstChild.NodeValue;
        if Item[i].FirstChild.NodeValue='ProductName' then
          Result.ID:=Item[i].NextSibling.FirstChild.NodeValue;
        if Item[i].FirstChild.NodeValue='ProductVersion' then
          Result.Version:=Item[i].NextSibling.FirstChild.NodeValue;
      end;
    finally
      Free;
    end;
  end;
  {$ENDIF}
  //Major.Minor.Revision
  str:=Result.Version;
  if str='' then Exit;
  Result.Major:=StrtoInt( Utf8Copy(str, 1, Utf8Pos('.',str)-1) );
  Utf8Delete(str, 1, Utf8Pos('.', str) );

  //10.04
  if Utf8Pos('.', str) = 0 then
  begin
    Result.Minor:=StrtoInt( Utf8Copy(str, 1, Utf8Length(str)) );
    Result.Revision:=0;
  end else
  //10.6.5
    begin
       Result.Minor:=StrtoInt( Utf8Copy(str, 1, Utf8Pos('.',str)-1) );
       Utf8Delete(str, 1, Utf8Pos('.', str) );
       Result.Revision:=StrtoInt( Utf8Copy(str, 1, Utf8Length(str)) );
    end;
  {$ENDIF}
end;


function GetPlatformVersion : TPlatformVersion;
{$IFDEF Unix}
var
  i: integer;
const
VersStr : array[TPlatformVersion] of string = (
  '',  '',  '',  '',  '',  '',
  '',  '',  '',  '',  '',  '',
  '',
  'Arc',
  'Debian',
  'openSUSE',
  'Fedora',
  'Gentoo',
  'Mandriva',
  'RedHat',
  'TurboLinux',
  'Ubuntu',  // протестировано
  'Xandros',
  'Oracle',
  'Mac OS X' // теоретич. работает
  );
{$ENDIF}
begin
  Result := pvUnknown;                      // Неизвестная версия ОС
  {$IFDEF MSWINDOWS}
  with GetPlatformInfo do
  begin
        if Version='' then Exit;
        case Major of
          0: Result := pvUnknown;
          1..2: Result := pvUnknown;
          3:  Result := pvWinNT3;              // Windows NT 3
          4:  case Minor of
                0: if PlatformId = VER_PLATFORM_WIN32_NT
                   then Result := pvWinNT4     // Windows NT 4
                   else Result := pvWin95;     // Windows 95
                10: Result := pvWin98;         // Windows 98
                90: Result := pvWinME;         // Windows ME
              end;
          5:  case Minor of
                0: Result := pvWin2000;         // Windows 2000
                1: Result := pvWinXP;          // Windows XP
                2: Result := pvWin2003;        // Windows 2003
              end;
          6:  case Minor of
                0: Result := pvWinVista;         // Windows Vista
                1: Result := pvWinSeven;          // Windows Seven
                2: Result := pvWin2008;        // Windows 2008   //возможно
                3..4: Result := pvWinNew;
              end;
          7:  Result := pvWinNew;
        end;
   end;
  {$ENDIF}
  {$IFDEF Unix}
  with GetPlatformInfo do
  begin
    if Version='' then Exit; //функция не смогла считать информацию
    For i:= 13 to Length(VersStr)-1 do
     if ID=VersStr[TPlatformVersion(i)] then
       Result := TPlatformVersion(i);
  end;
  {$ENDIF}
end;

function GetPlatformVersionStr : string;
const
  VersStr : array[TPlatformVersion] of string = (
    'Unknown',
    'Windows 95',
    'Windows 98',
    'Windows ME',
    'Windows NT 3',
    'Windows NT 4',
    'Windows 2000',
    'Windows XP',
    'Windows 2003',
    'Windows Vista',
    'Windows Seven',
    'Windows 2008',
    'Windows New',

    'Linux Arc',
    'Linux Debian',
    'Linux openSUSE',
    'Linux Fedora',
    'Linux Gentoo',
    'Linux Mandriva',
    'Linux RedHat',
    'Linux TurboLinux',
    'Linux Ubuntu',
    'Linux Xandros',
    'Linux Oracle',
    'Apple MacOSX');
begin
  Result := VersStr[GetPlatformVersion];
end;

function IsDirectoryWriteable(const AName: string): Boolean;
var
  LFileName: String;
{$IFDEF MSWINDOWS}
  LHandle: THandle;
{$ENDIF}
begin
  LFileName := IncludeTrailingPathDelimiter(AName) + 'chk.tmp';
{$IFDEF MSWINDOWS}
  LHandle := CreateFile(PChar(LFileName), GENERIC_READ or GENERIC_WRITE, 0, nil,
    CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY or FILE_FLAG_DELETE_ON_CLOSE, 0);
  Result := LHandle <> INVALID_HANDLE_VALUE;
  if Result then
    CloseHandle(LHandle);
{$ELSE}
  Result := fpAccess(PChar(LFileName), W_OK) <> 0;
{$ENDIF}
end;


function CharToWideChar(const AChar: AnsiChar): WideChar;
{$IFDEF MSWINDOWS}
var
  lResult: PWideChar;
begin
  GetMem(lResult, 2);
  MultiByteToWideChar(CP_ACP, 0, @AChar, 1, lResult, 2);
  Result := lResult^;
  FreeMem(lResult, 2);
{$ELSE}
var
  S: string;
  lResult: WideString;
begin
  S := AnsiToUtf8(AChar);
  lResult := Utf8ToUtf16(S);
  Result := lResult[1];
{$ENDIF}
end;

function PtInRect(const Rect: TGLRect; const P: TPoint): Boolean;
begin
  Result := (P.X >= Rect.Left) and (P.X < Rect.Right) and (P.Y >= Rect.Top)
    and (P.Y < Rect.Bottom);
end;

initialization
  vGLSStartTime := GLSTime;

{$IFDEF UNIX}
  Init_vProgStartSecond;
{$ENDIF}

end.
