//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Win32 specific Context.

    History :  
       11/09/11 - Yar - Added layers support (not tested because need Quadro or FireFX VGA)
       18/07/11 - Yar - Added ability of creating OpenGL ES 2.0 profile context
       03/12/10 - Yar - Fixed window tracking (thanks to Gabriel Corneanu)
       04/11/10 - DaStr - Restored Delphi5 and Delphi6 compatibility   
       23/08/10 - Yar - Replaced OpenGL1x to OpenGLTokens. Improved context creation.
       18/06/10 - Yar - Changed context sharing method for similarity to GLX
       06/06/10 - Yar - Moved forward context creation to DoCreateContext
                           make outputDevice HWND type
       19/05/10 - Yar - Added choice between hardware and software acceleration
       06/05/10 - Yar - Added vLastVendor clearing when multithreading is enabled
       06/04/10 - Yar - Added DoGetHandles to TGLWin32Context (thanks Rustam Asmandiarov aka Predator)
       28/03/10 - Yar - Added 3.3 forward context creation and eliminate memory leaks when multithreading
       06/03/10 - Yar - Added forward context creation in TGLWin32Context.DoActivate
       20/02/10 - DanB - Allow double-buffered memory viewers, if you want single
                            buffered, or no swapping, then change buffer options instead.
                            Some changes from Cardinal to the appropriate HDC /HGLRC type.
       15/01/10 - DaStr - Bugfixed TGLWin32Context.ChooseWGLFormat()
                             (BugtrackerID = 2933081) (thanks YarUndeoaker)
       08/01/10 - DaStr - Added more AntiAliasing modes (thanks YarUndeoaker)
       13/12/09 - DaStr - Modified for multithread support (thanks Controller)
       30/08/09 - DanB - vIgnoreContextActivationFailures renamed to
                            vContextActivationFailureOccurred + check removed.
       06/11/07 - mrqzzz - Ignore ContextActivation failure
                   if GLContext.vIgnoreContextActivationFailures=true
       15/02/07 - DaStr - Integer -> Cardinal because $R- was removed in GLScene.pas
       11/09/06 - NC - Added support for Multiple-Render-Target
       03/10/04 - NC - Added float texture support
       03/07/02 - EG - ChooseWGLFormat Kyro fix (Patrick Chevalley)
       13/03/02 - EG - aaDefault now prefers non-AA when possible
       03/03/02 - EG - Fixed aaNone mode (AA specifically off)
       01/03/02 - EG - Fixed CurrentPixelFormatIsHardwareAccelerated
       22/02/02 - EG - Unified ChooseWGLFormat for visual & non-visual
       21/02/02 - EG - AntiAliasing support *experimental* (Chris N. Strahm)
       05/02/02 - EG - Fixed UnTrackWindow
       03/02/02 - EG - Added experimental Hook-based window tracking
       29/01/02 - EG - Improved recovery for ICDs without pbuffer  support
       21/01/02 - EG - More graceful recovery for ICDs without pbuffer support
       07/01/02 - EG - DoCreateMemoryContext now retrieved topDC when needed
       15/12/01 - EG - Added support for AlphaBits
       30/11/01 - EG - Hardware acceleration support now detected
       20/11/01 - EG - New temp HWnd code for memory contexts (improved compat.)
       04/09/01 - EG - Added ChangeIAttrib, support for 16bits depth buffer
       25/08/01 - EG - Added pbuffer support and CreateMemoryContext interface
       24/08/01 - EG - Fixed PropagateSharedContext
       12/08/01 - EG - Handles management completed
       22/07/01 - EG - Creation (glcontext.omm)
    
}
unit GLWin32Context;

interface

{$I GLScene.inc}

{$IFNDEF MSWINDOWS}{$MESSAGE Error 'Unit is Windows specific'}{$ENDIF}

uses
  Windows,
  Messages,
  SysUtils,
  Classes,
  Forms,

   
  OpenGLTokens,
  OpenGLAdapter,
  GLContext,
  GLCrossPlatform,
  GLStrings,
  GLState,
 GLSLog,
  GLVectorGeometry;


const
  WGL_SWAP_MAIN_PLANE = $00000001;
  WGL_SWAP_OVERLAY1 = $00000002;
  WGL_SWAP_OVERLAY2 = $00000004;
  WGL_SWAP_OVERLAY3 = $00000008;
  WGL_SWAP_OVERLAY4 = $00000010;
  WGL_SWAP_OVERLAY5 = $00000020;
  WGL_SWAP_OVERLAY6 = $00000040;
  WGL_SWAP_OVERLAY7 = $00000080;
  WGL_SWAP_OVERLAY8 = $00000100;
  WGL_SWAP_OVERLAY9 = $00000200;
  WGL_SWAP_OVERLAY10 = $00000400;
  WGL_SWAP_OVERLAY11 = $00000800;
  WGL_SWAP_OVERLAY12 = $00001000;
  WGL_SWAP_OVERLAY13 = $00002000;
  WGL_SWAP_OVERLAY14 = $00004000;
  WGL_SWAP_OVERLAY15 = $00008000;
  WGL_SWAP_UNDERLAY1 = $00010000;
  WGL_SWAP_UNDERLAY2 = $00020000;
  WGL_SWAP_UNDERLAY3 = $00040000;
  WGL_SWAP_UNDERLAY4 = $00080000;
  WGL_SWAP_UNDERLAY5 = $00100000;
  WGL_SWAP_UNDERLAY6 = $00200000;
  WGL_SWAP_UNDERLAY7 = $00400000;
  WGL_SWAP_UNDERLAY8 = $00800000;
  WGL_SWAP_UNDERLAY9 = $01000000;
  WGL_SWAP_UNDERLAY10 = $02000000;
  WGL_SWAP_UNDERLAY11 = $04000000;
  WGL_SWAP_UNDERLAY12 = $08000000;
  WGL_SWAP_UNDERLAY13 = $10000000;
  WGL_SWAP_UNDERLAY14 = $20000000;
  WGL_SWAP_UNDERLAY15 = $40000000;


type

  // TGLWin32Context
  //
  { A context driver for standard Windows OpenGL (via MS OpenGL). }
  TGLWin32Context = class(TGLContext)
  protected
     
    FDC: HDC;
    FRC: HGLRC;
    FShareContext: TGLWin32Context;
    FHPBUFFER: Integer;
    FiAttribs: packed array of Integer;
    FfAttribs: packed array of Single;
    FLegacyContextsOnly: Boolean;
    FSwapBufferSupported: Boolean;

    procedure SpawnLegacyContext(aDC: HDC); // used for WGL_pixel_format soup
    procedure CreateOldContext(aDC: HDC); dynamic;
    procedure CreateNewContext(aDC: HDC); dynamic;

    procedure ClearIAttribs;
    procedure AddIAttrib(attrib, value: Integer);
    procedure ChangeIAttrib(attrib, newValue: Integer);
    procedure DropIAttrib(attrib: Integer);
    procedure ClearFAttribs;
    procedure AddFAttrib(attrib, value: Single);

    procedure DestructionEarlyWarning(sender: TObject);

    procedure ChooseWGLFormat(DC: HDC; nMaxFormats: Cardinal; piFormats:
      PInteger; var nNumFormats: Integer; BufferCount: integer = 1);
    procedure DoCreateContext(ADeviceHandle: HDC); override;
    procedure DoCreateMemoryContext(outputDevice: HWND; width, height:
      Integer; BufferCount: integer); override;
    function DoShareLists(aContext: TGLContext): Boolean; override;
    procedure DoDestroyContext; override;
    procedure DoActivate; override;
    procedure DoDeactivate; override;
    { DoGetHandles must be implemented in child classes,
       and return the display + window }

    procedure DoGetHandles(outputDevice: HWND; out XWin: HDC); virtual; abstract;

  public
     
    constructor Create; override;
    destructor Destroy; override;

    function IsValid: Boolean; override;
    procedure SwapBuffers; override;

    function RenderOutputDevice: Pointer; override;

    property DC: HDC read FDC;
    property RC: HGLRC read FRC;
  end;


resourcestring
  strForwardContextFailed = 'Can not create forward compatible context: #%X, %s';
  strBackwardContextFailed = 'Can not create backward compatible context: #%X, %s';
  strFailHWRC = 'Unable to create rendering context with hardware acceleration - down to software';
  strTmpRC_Created = 'Temporary rendering context created';
  strDriverNotSupportFRC = 'Driver not support creating of forward context';
  strDriverNotSupportOESRC = 'Driver not support creating of OpenGL ES 2.0 context';
  strDriverNotSupportDebugRC = 'Driver not support creating of debug context';
  strOESvsForwardRC = 'OpenGL ES 2.0 context incompatible with Forward context - flag ignored';
  strFRC_created = 'Forward core context seccussfuly created';
  strOESRC_created = 'OpenGL ES 2.0 context seccussfuly created';
  strPBufferRC_created = 'Backward compatible core PBuffer context successfully created';

function CreateTempWnd: HWND;

var
  { This boolean controls a hook-based tracking of top-level forms destruction,
    with the purpose of being able to properly release OpenGL contexts before
    they are (improperly) released by some drivers upon top-level form
    destruction. }
  vUseWindowTrackingHook: Boolean = True;

  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------


var
  vTrackingCount: Integer;
  vTrackedHwnd: array of HWND;
  vTrackedEvents: array of TNotifyEvent;
  vTrackingHook: HHOOK;

  // TrackHookProc
  //

function TrackHookProc(nCode: Integer; wParam: wParam; lParam: LPARAM): Integer;
  stdcall;
var
  i: Integer;
  p: PCWPStruct;
begin
  if nCode = HC_ACTION then
  begin
    p := PCWPStruct(lParam);
    //   if (p.message=WM_DESTROY) or (p.message=WM_CLOSE) then begin // destroy & close variant
    if p.message = WM_DESTROY then
    begin
      // special care must be taken by this loop, items may go away unexpectedly
      i := vTrackingCount - 1;
      while i >= 0 do
      begin
        if IsChild(p.hwnd, vTrackedHwnd[i]) then
        begin
          // got one, send notification
          vTrackedEvents[i](nil);
        end;
        Dec(i);
        while i >= vTrackingCount do
          Dec(i);
      end;
    end;
    CallNextHookEx(vTrackingHook, nCode, wParam, lParam);
    Result := 0;
  end
  else
    Result := CallNextHookEx(vTrackingHook, nCode, wParam, lParam);
end;

// TrackWindow
//

procedure TrackWindow(h: HWND; notifyEvent: TNotifyEvent);
begin
  if not IsWindow(h) then
    Exit;
  if vTrackingCount = 0 then
    vTrackingHook := SetWindowsHookEx(WH_CALLWNDPROC, @TrackHookProc, 0,
      GetCurrentThreadID);
  Inc(vTrackingCount);
  SetLength(vTrackedHwnd, vTrackingCount);
  vTrackedHwnd[vTrackingCount - 1] := h;
  SetLength(vTrackedEvents, vTrackingCount);
  vTrackedEvents[vTrackingCount - 1] := notifyEvent;
end;

// UnTrackWindows
//

procedure UnTrackWindow(h: HWND);
var
  i, k: Integer;
begin
  if not IsWindow(h) then
    Exit;
  if vTrackingCount = 0 then
    Exit;
  k := 0;
  for i := 0 to MinInteger(vTrackingCount, Length(vTrackedHwnd)) - 1 do
  begin
    if vTrackedHwnd[i] <> h then
    begin
      if(k <> i) then
      begin
        vTrackedHwnd[k] := vTrackedHwnd[i];
        vTrackedEvents[k] := vTrackedEvents[i];
      end;
      Inc(k);
    end
  end;
  if(k >= vTrackingCount) then exit;
  Dec(vTrackingCount);
  SetLength(vTrackedHwnd, vTrackingCount);
  SetLength(vTrackedEvents, vTrackingCount);
  if vTrackingCount = 0 then
    UnhookWindowsHookEx(vTrackingHook);
end;

var
  vUtilWindowClass: TWndClass = (
    style: 0;
    lpfnWndProc: @DefWindowProc;
    cbClsExtra: 0;
    cbWndExtra: 0;
    hInstance: 0;
    hIcon: 0;
    hCursor: 0;
    hbrBackground: 0;
    lpszMenuName: nil;
    lpszClassName: 'GLSUtilWindow');

  // CreateTempWnd
  //

function CreateTempWnd: HWND;
var
  classRegistered: Boolean;
  tempClass: TWndClass;
begin
  vUtilWindowClass.hInstance := HInstance;
  classRegistered := GetClassInfo(HInstance, vUtilWindowClass.lpszClassName,
    tempClass);
  if not classRegistered then
    Windows.RegisterClass(vUtilWindowClass);
  Result := CreateWindowEx(WS_EX_TOOLWINDOW, vUtilWindowClass.lpszClassName,
    '', WS_POPUP, 0, 0, 0, 0, 0, 0, HInstance, nil);
end;

// ------------------
// ------------------ TGLWin32Context ------------------
// ------------------

  // Create
  //

constructor TGLWin32Context.Create;
begin
  inherited Create;
  ClearIAttribs;
  ClearFAttribs;
end;

// Destroy
//

destructor TGLWin32Context.Destroy;
begin
  inherited Destroy;
end;

// SetupPalette
//

function SetupPalette(DC: HDC; PFD: TPixelFormatDescriptor): HPalette;
var
  nColors, I: Integer;
  LogPalette: TMaxLogPalette;
  RedMask, GreenMask, BlueMask: Byte;
begin
  nColors := 1 shl Pfd.cColorBits;
  LogPalette.palVersion := $300;
  LogPalette.palNumEntries := nColors;
  RedMask := (1 shl Pfd.cRedBits) - 1;
  GreenMask := (1 shl Pfd.cGreenBits) - 1;
  BlueMask := (1 shl Pfd.cBlueBits) - 1;
  with LogPalette, PFD do
    for I := 0 to nColors - 1 do
    begin
      palPalEntry[I].peRed := (((I shr cRedShift) and RedMask) * 255) div
        RedMask;
      palPalEntry[I].peGreen := (((I shr cGreenShift) and GreenMask) * 255) div
        GreenMask;
      palPalEntry[I].peBlue := (((I shr cBlueShift) and BlueMask) * 255) div
        BlueMask;
      palPalEntry[I].peFlags := 0;
    end;

  Result := CreatePalette(PLogPalette(@LogPalette)^);
  if Result <> 0 then
  begin
    SelectPalette(DC, Result, False);
    RealizePalette(DC);
  end
  else
    RaiseLastOSError;
end;

// ClearIAttribs
//

procedure TGLWin32Context.ClearIAttribs;
begin
  SetLength(FiAttribs, 1);
  FiAttribs[0] := 0;
end;

// AddIAttrib
//

procedure TGLWin32Context.AddIAttrib(attrib, value: Integer);
var
  n: Integer;
begin
  n := Length(FiAttribs);
  SetLength(FiAttribs, n + 2);
  FiAttribs[n - 1] := attrib;
  FiAttribs[n] := value;
  FiAttribs[n + 1] := 0;
end;

// ChangeIAttrib
//

procedure TGLWin32Context.ChangeIAttrib(attrib, newValue: Integer);
var
  i: Integer;
begin
  i := 0;
  while i < Length(FiAttribs) do
  begin
    if FiAttribs[i] = attrib then
    begin
      FiAttribs[i + 1] := newValue;
      Exit;
    end;
    Inc(i, 2);
  end;
  AddIAttrib(attrib, newValue);
end;

// DropIAttrib
//

procedure TGLWin32Context.DropIAttrib(attrib: Integer);
var
  i: Integer;
begin
  i := 0;
  while i < Length(FiAttribs) do
  begin
    if FiAttribs[i] = attrib then
    begin
      Inc(i, 2);
      while i < Length(FiAttribs) do
      begin
        FiAttribs[i - 2] := FiAttribs[i];
        Inc(i);
      end;
      SetLength(FiAttribs, Length(FiAttribs) - 2);
      Exit;
    end;
    Inc(i, 2);
  end;
end;

// ClearFAttribs
//

procedure TGLWin32Context.ClearFAttribs;
begin
  SetLength(FfAttribs, 1);
  FfAttribs[0] := 0;
end;

// AddFAttrib
//

procedure TGLWin32Context.AddFAttrib(attrib, value: Single);
var
  n: Integer;
begin
  n := Length(FfAttribs);
  SetLength(FfAttribs, n + 2);
  FfAttribs[n - 1] := attrib;
  FfAttribs[n] := value;
  FfAttribs[n + 1] := 0;
end;

// DestructionEarlyWarning
//

procedure TGLWin32Context.DestructionEarlyWarning(sender: TObject);
begin
  if IsValid then
    DestroyContext;
end;

// ChooseWGLFormat
//
procedure TGLWin32Context.ChooseWGLFormat(DC: HDC; nMaxFormats: Cardinal; piFormats:
  PInteger; var nNumFormats: Integer; BufferCount: integer);
const
  cAAToSamples: array[aaNone..csa16xHQ] of Integer =
    (1, 2, 2, 4, 4, 6, 8, 16, 8, 8, 16, 16);
  cCSAAToSamples: array[csa8x..csa16xHQ] of Integer = (4, 8, 4, 8);

  procedure ChoosePixelFormat;
  begin
    if not FGL.WChoosePixelFormatARB(DC, @FiAttribs[0], @FfAttribs[0],
      32, PGLint(piFormats), @nNumFormats) then
      nNumFormats := 0;
  end;

var
  float: boolean;
  aa: TGLAntiAliasing;
begin
  // request hardware acceleration
  case FAcceleration of
    chaUnknown: AddIAttrib(WGL_ACCELERATION_ARB, WGL_GENERIC_ACCELERATION_ARB);
    chaHardware: AddIAttrib(WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB);
    chaSoftware: AddIAttrib(WGL_ACCELERATION_ARB, WGL_NO_ACCELERATION_ARB);
  end;

  float := (ColorBits = 64) or (ColorBits = 128); // float_type

  if float then
  begin // float_type
    if GL.W_ATI_pixel_format_float then
    begin // NV40 uses ATI_float, with linear filtering
      AddIAttrib(WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_FLOAT_ATI);
    end
    else
    begin
      AddIAttrib(WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB);
      AddIAttrib(WGL_FLOAT_COMPONENTS_NV, GL_TRUE);
    end;
  end;

  if BufferCount > 1 then
    // 1 front buffer + (BufferCount-1) aux buffers
    AddIAttrib(WGL_AUX_BUFFERS_ARB, BufferCount - 1);

  AddIAttrib(WGL_COLOR_BITS_ARB, ColorBits);
  if AlphaBits > 0 then
    AddIAttrib(WGL_ALPHA_BITS_ARB, AlphaBits);
  AddIAttrib(WGL_DEPTH_BITS_ARB, DepthBits);
  if StencilBits > 0 then
    AddIAttrib(WGL_STENCIL_BITS_ARB, StencilBits);
  if AccumBits > 0 then
    AddIAttrib(WGL_ACCUM_BITS_ARB, AccumBits);
  if AuxBuffers > 0 then
    AddIAttrib(WGL_AUX_BUFFERS_ARB, AuxBuffers);
  if (AntiAliasing <> aaDefault) and GL.W_ARB_multisample then
  begin
    if AntiAliasing = aaNone then
      AddIAttrib(WGL_SAMPLE_BUFFERS_ARB, GL_FALSE)
    else
    begin
      AddIAttrib(WGL_SAMPLE_BUFFERS_ARB, GL_TRUE);
      AddIAttrib(WGL_SAMPLES_ARB, cAAToSamples[AntiAliasing]);
      if (AntiAliasing >= csa8x) and (AntiAliasing <= csa16xHQ) then
        AddIAttrib(WGL_COLOR_SAMPLES_NV, cCSAAToSamples[AntiAliasing]);
    end;

  end;

  ClearFAttribs;
  ChoosePixelFormat;
  if (nNumFormats = 0) and (DepthBits >= 32) then
  begin
    // couldn't find 32+ bits depth buffer, 24 bits one available?
    ChangeIAttrib(WGL_DEPTH_BITS_ARB, 24);
    ChoosePixelFormat;
  end;
  if (nNumFormats = 0) and (DepthBits >= 24) then
  begin
    // couldn't find 24+ bits depth buffer, 16 bits one available?
    ChangeIAttrib(WGL_DEPTH_BITS_ARB, 16);
    ChoosePixelFormat;
  end;
  if (nNumFormats = 0) and (ColorBits >= 24) then
  begin
    // couldn't find 24+ bits color buffer, 16 bits one available?
    ChangeIAttrib(WGL_COLOR_BITS_ARB, 16);
    ChoosePixelFormat;
  end;
  if (nNumFormats = 0) and (AntiAliasing <> aaDefault) then
  begin
    // Restore DepthBits
    ChangeIAttrib(WGL_DEPTH_BITS_ARB, DepthBits);
    if (AntiAliasing >= csa8x) and (AntiAliasing <= csa16xHQ) then
    begin
      DropIAttrib(WGL_COLOR_SAMPLES_NV);
      case AntiAliasing of
        csa8x, csa8xHQ: AntiAliasing := aa8x;
        csa16x, csa16xHQ: AntiAliasing := aa16x;
      end;
      ChangeIAttrib(WGL_SAMPLES_ARB, cAAToSamples[AntiAliasing]);
    end;
    ChoosePixelFormat;

    if nNumFormats = 0 then
    begin
      aa := AntiAliasing;
      repeat
        Dec(aa);
        if aa = aaNone then
        begin
          // couldn't find AA buffer, try without
          DropIAttrib(WGL_SAMPLE_BUFFERS_ARB);
          DropIAttrib(WGL_SAMPLES_ARB);
          ChoosePixelFormat;
          break;
        end;
        ChangeIAttrib(WGL_SAMPLES_ARB, cAAToSamples[aa]);
        ChoosePixelFormat;
      until nNumFormats <> 0;
      AntiAliasing := aa;
    end;
  end;
  // Check DepthBits again
  if (nNumFormats = 0) and (DepthBits >= 32) then
  begin
    // couldn't find 32+ bits depth buffer, 24 bits one available?
    ChangeIAttrib(WGL_DEPTH_BITS_ARB, 24);
    ChoosePixelFormat;
  end;
  if (nNumFormats = 0) and (DepthBits >= 24) then
  begin
    // couldn't find 24+ bits depth buffer, 16 bits one available?
    ChangeIAttrib(WGL_DEPTH_BITS_ARB, 16);
    ChoosePixelFormat;
  end;
  if (nNumFormats = 0) and (ColorBits >= 24) then
  begin
    // couldn't find 24+ bits color buffer, 16 bits one available?
    ChangeIAttrib(WGL_COLOR_BITS_ARB, 16);
    ChoosePixelFormat;
  end;
  if nNumFormats = 0 then
  begin
    // ok, last attempt: no AA, restored depth and color,
    // relaxed hardware-acceleration request
    ChangeIAttrib(WGL_COLOR_BITS_ARB, ColorBits);
    ChangeIAttrib(WGL_DEPTH_BITS_ARB, DepthBits);
    DropIAttrib(WGL_ACCELERATION_ARB);
    ChoosePixelFormat;
  end;
end;

procedure TGLWin32Context.CreateOldContext(aDC: HDC);
begin
  if not FLegacyContextsOnly then
  begin
    case Layer of
      clUnderlay2: FRC := wglCreateLayerContext(aDC, -2);
      clUnderlay1: FRC := wglCreateLayerContext(aDC, -1);
      clMainPlane: FRC := wglCreateContext(aDC);
      clOverlay1: FRC := wglCreateLayerContext(aDC, 1);
      clOverlay2: FRC := wglCreateLayerContext(aDC, 2);
    end;
  end
  else
    FRC := wglCreateContext(aDC);

  if FRC = 0 then
    RaiseLastOSError;
  FDC := aDC;

  if not wglMakeCurrent(FDC, FRC) then
    raise EGLContext.Create(Format(strContextActivationFailed,
      [GetLastError, SysErrorMessage(GetLastError)]));

  if not FLegacyContextsOnly then
  begin
    if Assigned(FShareContext) and (FShareContext.RC <> 0) then
    begin
      if not wglShareLists(FShareContext.RC, FRC) then
      {$IFDEF GLS_LOGGING}
        GLSLogger.LogWarning(strFailedToShare)
      {$ENDIF}
      else
      begin
        FSharedContexts.Add(FShareContext);
        PropagateSharedContext;
      end;
    end;
    FGL.DebugMode := False;
    FGL.Initialize;
    MakeGLCurrent;
    // If we are using AntiAliasing, adjust filtering hints
    if AntiAliasing in [aa2xHQ, aa4xHQ, csa8xHQ, csa16xHQ] then
      // Hint for nVidia HQ modes (Quincunx etc.)
      GLStates.MultisampleFilterHint := hintNicest
    else
      GLStates.MultisampleFilterHint := hintDontCare;

    if rcoDebug in Options then
      GLSLogger.LogWarning(strDriverNotSupportDebugRC);
    if rcoOGL_ES in Options then
      GLSLogger.LogWarning(strDriverNotSupportOESRC);
    if GLStates.ForwardContext then
      GLSLogger.LogWarning(strDriverNotSupportFRC);
    GLStates.ForwardContext := False;
  end
  else
    GLSLogger.LogInfo(strTmpRC_Created);
end;

procedure TGLWin32Context.CreateNewContext(aDC: HDC);
var
  bSuccess, bOES: Boolean;
begin
  bSuccess := False;
  bOES := False;

  try
    ClearIAttribs;
    // Initialize forward context
    if GLStates.ForwardContext then
    begin
      if FGL.VERSION_4_2 then
      begin
        AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 4);
        AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 2);
      end
      else if FGL.VERSION_4_1 then
      begin
        AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 4);
        AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 1);
      end
      else if FGL.VERSION_4_0 then
      begin
        AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 4);
        AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 0);
      end
      else if FGL.VERSION_3_3 then
      begin
        AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 3);
        AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 3);
      end
      else if FGL.VERSION_3_2 then
      begin
        AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 3);
        AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 2);
      end
      else if FGL.VERSION_3_1 then
      begin
        AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 3);
        AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 1);
      end
      else if FGL.VERSION_3_0 then
      begin
        AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 3);
        AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 0);
      end
      else
        Abort;
      AddIAttrib(WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB);
      if rcoOGL_ES in Options then
        GLSLogger.LogWarning(strOESvsForwardRC);
    end
    else if rcoOGL_ES in Options then
    begin
      if FGL.W_EXT_create_context_es2_profile then
      begin
        AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 2);
        AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 0);
        AddIAttrib(WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_ES2_PROFILE_BIT_EXT);
        bOES := True;
      end
      else
        GLSLogger.LogError(strDriverNotSupportOESRC);
    end;

    if rcoDebug in Options then
    begin
      AddIAttrib(WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_DEBUG_BIT_ARB);
      FGL.DebugMode := True;
    end;

    case Layer of
      clUnderlay2: AddIAttrib(WGL_CONTEXT_LAYER_PLANE_ARB, -2);
      clUnderlay1: AddIAttrib(WGL_CONTEXT_LAYER_PLANE_ARB, -1);
      clOverlay1: AddIAttrib(WGL_CONTEXT_LAYER_PLANE_ARB, 1);
      clOverlay2: AddIAttrib(WGL_CONTEXT_LAYER_PLANE_ARB, 2);
    end;

    FRC := 0;
    if Assigned(FShareContext) then
    begin
      FRC := FGL.WCreateContextAttribsARB(aDC, FShareContext.RC, @FiAttribs[0]);
      if FRC <> 0 then
      begin
        FSharedContexts.Add(FShareContext);
        PropagateSharedContext;
      end
      else
        GLSLogger.LogWarning(strFailedToShare)
    end;

    if FRC = 0 then
    begin
      FRC := FGL.WCreateContextAttribsARB(aDC, 0, @FiAttribs[0]);
      if FRC = 0 then
      begin
        if GLStates.ForwardContext then
          GLSLogger.LogErrorFmt(strForwardContextFailed,
            [GetLastError, SysErrorMessage(GetLastError)])
        else
          GLSLogger.LogErrorFmt(strBackwardContextFailed,
            [GetLastError, SysErrorMessage(GetLastError)]);
        Abort;
      end;
    end;

    FDC := aDC;

    if not wglMakeCurrent(FDC, FRC) then
    begin
      GLSLogger.LogErrorFmt(strContextActivationFailed,
        [GetLastError, SysErrorMessage(GetLastError)]);
      Abort;
    end;

    FGL.Initialize;
    MakeGLCurrent;
    // If we are using AntiAliasing, adjust filtering hints
    if AntiAliasing in [aa2xHQ, aa4xHQ, csa8xHQ, csa16xHQ] then
      // Hint for nVidia HQ modes (Quincunx etc.)
      GLStates.MultisampleFilterHint := hintNicest
    else
      GLStates.MultisampleFilterHint := hintDontCare;

    if GLStates.ForwardContext then
      GLSLogger.LogInfo(strFRC_created);
    if bOES then
      GLSLogger.LogInfo(strOESRC_created);
    bSuccess := True;
  finally
    GLStates.ForwardContext := GLStates.ForwardContext and bSuccess;
    PipelineTransformation.LoadMatricesEnabled := not GLStates.ForwardContext;
  end;
end;

// DoCreateContext
//

procedure TGLWin32Context.DoCreateContext(ADeviceHandle: HDC);
const
  cMemoryDCs = [OBJ_MEMDC, OBJ_METADC, OBJ_ENHMETADC];
  cBoolToInt: array[False..True] of Integer = (GL_FALSE, GL_TRUE);
  cLayerToSet: array[TGLContextLayer] of Byte = (32, 16, 0, 1, 2);
var
  pfDescriptor: TPixelFormatDescriptor;
  pixelFormat, nbFormats, softwarePixelFormat: Integer;
  aType: DWORD;
  iFormats: array[0..31] of Integer;
  tempWnd: HWND;
  tempDC: HDC;
  localDC: HDC;
  localRC: HGLRC;
  sharedRC: TGLWin32Context;

  function CurrentPixelFormatIsHardwareAccelerated: Boolean;
  var
    localPFD: TPixelFormatDescriptor;
  begin
    Result := False;
    if pixelFormat = 0 then
      Exit;
    with localPFD do
    begin
      nSize := SizeOf(localPFD);
      nVersion := 1;
    end;
    DescribePixelFormat(ADeviceHandle, pixelFormat, SizeOf(localPFD), localPFD);
    Result := ((localPFD.dwFlags and PFD_GENERIC_FORMAT) = 0);
  end;

var
  i, iAttrib, iValue: Integer;
begin

  DoGetHandles(HWND(ADeviceHandle), ADeviceHandle);

  if vUseWindowTrackingHook and not FLegacyContextsOnly then
    TrackWindow(WindowFromDC(ADeviceHandle), DestructionEarlyWarning);

  // Just in case it didn't happen already.
  if not InitOpenGL then
    RaiseLastOSError;

  // Prepare PFD
  FillChar(pfDescriptor, SizeOf(pfDescriptor), 0);
  with PFDescriptor do
  begin
    nSize := SizeOf(PFDescriptor);
    nVersion := 1;
    dwFlags := PFD_SUPPORT_OPENGL;
    aType := GetObjectType(ADeviceHandle);
    if aType = 0 then
      RaiseLastOSError;
    if aType in cMemoryDCs then
      dwFlags := dwFlags or PFD_DRAW_TO_BITMAP
    else
      dwFlags := dwFlags or PFD_DRAW_TO_WINDOW;
    if rcoDoubleBuffered in Options then
      dwFlags := dwFlags or PFD_DOUBLEBUFFER;
    if rcoStereo in Options then
      dwFlags := dwFlags or PFD_STEREO;
    iPixelType := PFD_TYPE_RGBA;
    cColorBits := ColorBits;
    cDepthBits := DepthBits;
    cStencilBits := StencilBits;
    cAccumBits := AccumBits;
    cAlphaBits := AlphaBits;
    cAuxBuffers := AuxBuffers;
    case Layer of
      clUnderlay2, clUnderlay1: iLayerType := Byte(PFD_UNDERLAY_PLANE);
      clMainPlane: iLayerType := PFD_MAIN_PLANE;
      clOverlay1, clOverlay2: iLayerType := PFD_OVERLAY_PLANE;
    end;
    bReserved := cLayerToSet[Layer];
    if Layer <> clMainPlane then
      dwFlags := dwFlags or PFD_SWAP_LAYER_BUFFERS;

  end;
  pixelFormat := 0;

  // WGL_ARB_pixel_format is used if available
  //
  if not (IsMesaGL or FLegacyContextsOnly or (aType in cMemoryDCs)) then
  begin
    // the WGL mechanism is a little awkward: we first create a dummy context
    // on the TOP-level DC (ie. screen), to retrieve our pixelformat, create
    // our stuff, etc.
    tempWnd := CreateTempWnd;
    tempDC := GetDC(tempWnd);
    localDC := 0;
    localRC := 0;
    try
      SpawnLegacyContext(tempDC);
      try
        DoActivate;
        try
          FGL.ClearError;
          if FGL.W_ARB_pixel_format then
          begin
            // New pixel format selection via wglChoosePixelFormatARB
            ClearIAttribs;
            AddIAttrib(WGL_DRAW_TO_WINDOW_ARB, GL_TRUE);
            AddIAttrib(WGL_STEREO_ARB, cBoolToInt[rcoStereo in Options]);
            AddIAttrib(WGL_DOUBLE_BUFFER_ARB, cBoolToInt[rcoDoubleBuffered in
              Options]);

            ChooseWGLFormat(ADeviceHandle, 32, @iFormats, nbFormats);
            if nbFormats > 0 then
            begin
              if FGL.W_ARB_multisample and (AntiAliasing in [aaNone, aaDefault]) then
              begin
                // Pick first non AntiAliased for aaDefault and aaNone modes
                iAttrib := WGL_SAMPLE_BUFFERS_ARB;
                for i := 0 to nbFormats - 1 do
                begin
                  pixelFormat := iFormats[i];
                  iValue := GL_FALSE;
                  FGL.WGetPixelFormatAttribivARB(ADeviceHandle, pixelFormat, 0, 1,
                    @iAttrib, @iValue);
                  if iValue = GL_FALSE then
                    Break;
                end;
              end
              else
                pixelFormat := iFormats[0];
              if GetPixelFormat(ADeviceHandle) <> pixelFormat then
              begin
                if not SetPixelFormat(ADeviceHandle, pixelFormat, @PFDescriptor) then
                  RaiseLastOSError;
              end;
            end;
          end;
        finally
          DoDeactivate;
        end;
      finally
        sharedRC := FShareContext;
        DoDestroyContext;
        FShareContext := sharedRC;
        GLSLogger.LogInfo('Temporary rendering context destroyed');
      end;
    finally
      ReleaseDC(0, tempDC);
      DestroyWindow(tempWnd);
      FDC := localDC;
      FRC := localRC;
    end;
  end;

  if pixelFormat = 0 then
  begin
    // Legacy pixel format selection
    pixelFormat := ChoosePixelFormat(ADeviceHandle, @PFDescriptor);
    if (not (aType in cMemoryDCs)) and (not
      CurrentPixelFormatIsHardwareAccelerated) then
    begin
      softwarePixelFormat := pixelFormat;
      pixelFormat := 0;
    end
    else
      softwarePixelFormat := 0;
    if pixelFormat = 0 then
    begin
      // Failed on default params, try with 16 bits depth buffer
      PFDescriptor.cDepthBits := 16;
      pixelFormat := ChoosePixelFormat(ADeviceHandle, @PFDescriptor);
      if not CurrentPixelFormatIsHardwareAccelerated then
        pixelFormat := 0;
      if pixelFormat = 0 then
      begin
        // Failed, try with 16 bits color buffer
        PFDescriptor.cColorBits := 16;
        pixelFormat := ChoosePixelFormat(ADeviceHandle, @PFDescriptor);
      end;
      if not CurrentPixelFormatIsHardwareAccelerated then
      begin
        // Fallback to original, should be supported by software
        pixelFormat := softwarePixelFormat;
      end;
      if pixelFormat = 0 then
        RaiseLastOSError;
    end;
  end;

  if GetPixelFormat(ADeviceHandle) <> pixelFormat then
  begin
    if not SetPixelFormat(ADeviceHandle, pixelFormat, @PFDescriptor) then
      RaiseLastOSError;
  end;

  // Check the properties we just set.
  DescribePixelFormat(ADeviceHandle, pixelFormat, SizeOf(PFDescriptor), PFDescriptor);
  with PFDescriptor do
  begin
    if (dwFlags and PFD_NEED_PALETTE) <> 0 then
      SetupPalette(ADeviceHandle, PFDescriptor);
    FSwapBufferSupported := (dwFlags and PFD_SWAP_LAYER_BUFFERS) <> 0;
    if bReserved = 0 then
      FLayer := clMainPlane;
  end;

  if not FLegacyContextsOnly then
  begin
    if ((pfDescriptor.dwFlags and PFD_GENERIC_FORMAT) > 0)
      and (FAcceleration = chaHardware) then
    begin
      FAcceleration := chaSoftware;
      GLSLogger.LogWarning(strFailHWRC);
    end;
  end;

  if not FLegacyContextsOnly
    and FGL.W_ARB_create_context
    and (FAcceleration = chaHardware) then
    CreateNewContext(ADeviceHandle)
  else
    CreateOldContext(ADeviceHandle);

  if not FLegacyContextsOnly then
  begin
    // Share identifiers with other context if it deffined
    if (ServiceContext <> nil) and (Self <> ServiceContext) then
    begin
      if wglShareLists(TGLWin32Context(ServiceContext).FRC, FRC) then
      begin
        FSharedContexts.Add(ServiceContext);
        PropagateSharedContext;
      end
      else
        GLSLogger.LogWarning('DoCreateContext - Failed to share contexts with resource context');
    end;
  end;
end;

// SpawnLegacyContext
//

procedure TGLWin32Context.SpawnLegacyContext(aDC: HDC);
begin
  try
    FLegacyContextsOnly := True;
    try
      DoCreateContext(aDC);
    finally
      FLegacyContextsOnly := False;
    end;
  except
    on E: Exception do
    begin
      raise Exception.Create(strUnableToCreateLegacyContext + #13#10
        + E.ClassName + ': ' + E.Message);
    end;
  end;
end;

// DoCreateMemoryContext
//

procedure TGLWin32Context.DoCreateMemoryContext(outputDevice: HWND; width,
  height: Integer; BufferCount: integer);
var
  nbFormats: Integer;
  iFormats: array[0..31] of Integer;
  iPBufferAttribs: array[0..0] of Integer;
  localHPBuffer: Integer;
  localRC: HGLRC;
  localDC, tempDC: HDC;
  tempWnd: HWND;
  shareRC: TGLWin32Context;
  pfDescriptor: TPixelFormatDescriptor;
  bOES: Boolean;
begin
  localHPBuffer := 0;
  localDC := 0;
  localRC := 0;
  bOES := False;
  // the WGL mechanism is a little awkward: we first create a dummy context
  // on the TOP-level DC (ie. screen), to retrieve our pixelformat, create
  // our stuff, etc.
  tempWnd := CreateTempWnd;
  tempDC := GetDC(tempWnd);
  try
    SpawnLegacyContext(tempDC);
    try
      DoActivate;
      try
        FGL.ClearError;
        if FGL.W_ARB_pixel_format and FGL.W_ARB_pbuffer then
        begin
          ClearIAttribs;
          AddIAttrib(WGL_DRAW_TO_PBUFFER_ARB, 1);
          ChooseWGLFormat(tempDC, 32, @iFormats, nbFormats, BufferCount);
          if nbFormats = 0 then
            raise
              EPBuffer.Create('Format not supported for pbuffer operation.');
          iPBufferAttribs[0] := 0;

          localHPBuffer := FGL.WCreatePbufferARB(tempDC, iFormats[0], width,
            height,
            @iPBufferAttribs[0]);
          if localHPBuffer = 0 then
            raise EPBuffer.Create('Unabled to create pbuffer.');
          try
            localDC := FGL.WGetPbufferDCARB(localHPBuffer);
            if localDC = 0 then
              raise EPBuffer.Create('Unabled to create pbuffer''s DC.');
            try
              if FGL.W_ARB_create_context then
              begin
                // Modern creation style
                ClearIAttribs;
                // Initialize forward context
                if GLStates.ForwardContext then
                begin
                  if FGL.VERSION_4_2 then
                  begin
                    AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 4);
                    AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 2);
                  end
                  else if FGL.VERSION_4_1 then
                  begin
                    AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 4);
                    AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 1);
                  end
                  else if FGL.VERSION_4_0 then
                  begin
                    AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 4);
                    AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 0);
                  end
                  else if FGL.VERSION_3_3 then
                  begin
                    AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 3);
                    AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 3);
                  end
                  else if FGL.VERSION_3_2 then
                  begin
                    AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 3);
                    AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 2);
                  end
                  else if FGL.VERSION_3_1 then
                  begin
                    AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 3);
                    AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 1);
                  end
                  else if FGL.VERSION_3_0 then
                  begin
                    AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 3);
                    AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 0);
                  end
                  else
                    Abort;
                  AddIAttrib(WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB);
                  if rcoOGL_ES in Options then
                    GLSLogger.LogWarning(strOESvsForwardRC);
                end
                else if rcoOGL_ES in Options then
                begin
                  if FGL.W_EXT_create_context_es2_profile then
                  begin
                    AddIAttrib(WGL_CONTEXT_MAJOR_VERSION_ARB, 2);
                    AddIAttrib(WGL_CONTEXT_MINOR_VERSION_ARB, 0);
                    AddIAttrib(WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_ES2_PROFILE_BIT_EXT);
                  end
                  else
                    GLSLogger.LogError(strDriverNotSupportOESRC);
                end;

                if rcoDebug in Options then
                begin
                  AddIAttrib(WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_DEBUG_BIT_ARB);
                  FGL.DebugMode := True;
                end;

                case Layer of
                  clUnderlay2: AddIAttrib(WGL_CONTEXT_LAYER_PLANE_ARB, -2);
                  clUnderlay1: AddIAttrib(WGL_CONTEXT_LAYER_PLANE_ARB, -1);
                  clOverlay1: AddIAttrib(WGL_CONTEXT_LAYER_PLANE_ARB, 1);
                  clOverlay2: AddIAttrib(WGL_CONTEXT_LAYER_PLANE_ARB, 2);
                end;

                localRC := FGL.WCreateContextAttribsARB(localDC, 0, @FiAttribs[0]);
                if localRC = 0 then
               {$IFDEF GLS_LOGGING}
                begin
                  if GLStates.ForwardContext then
                    GLSLogger.LogErrorFmt(strForwardContextFailed,
                      [GetLastError, SysErrorMessage(GetLastError)])
                  else
                    GLSLogger.LogErrorFmt(strBackwardContextFailed,
                      [GetLastError, SysErrorMessage(GetLastError)]);
                  Abort;
                end;
               {$ELSE}
                  raise Exception.Create('Unabled to create pbuffer''s RC.');
               {$ENDIF}
              end
              else
              begin
                // Old creation style
                localRC := wglCreateContext(localDC);
                if localRC = 0 then
                begin
                  GLSLogger.LogErrorFmt(strBackwardContextFailed,
                    [GetLastError, SysErrorMessage(GetLastError)]);
                  Abort;
                end;
              end;

            except
              FGL.WReleasePBufferDCARB(localHPBuffer, localDC);
              raise;
            end;
          except
            FGL.WDestroyPBufferARB(localHPBuffer);
            raise;
          end;
        end
        else
          raise EPBuffer.Create('WGL_ARB_pbuffer support required.');
        FGL.CheckError;
      finally
        DoDeactivate;
      end;
    finally
      shareRC := FShareContext;
      DoDestroyContext;
      FShareContext := shareRC;
    end;
  finally
    ReleaseDC(0, tempDC);
    DestroyWindow(tempWnd);
    FHPBUFFER := localHPBuffer;
    FDC := localDC;
    FRC := localRC;
  end;

  DescribePixelFormat(FDC, GetPixelFormat(FDC), SizeOf(PFDescriptor), PFDescriptor);
  if ((PFDescriptor.dwFlags and PFD_GENERIC_FORMAT) > 0)
    and (FAcceleration = chaHardware) then
  begin
    FAcceleration := chaSoftware;
    GLSLogger.LogWarning(strFailHWRC);
  end;

  Activate;
  FGL.Initialize;
  // If we are using AntiAliasing, adjust filtering hints
  if AntiAliasing in [aa2xHQ, aa4xHQ, csa8xHQ, csa16xHQ] then
    GLStates.MultisampleFilterHint := hintNicest
  else if AntiAliasing in [aa2x, aa4x, csa8x, csa16x] then
    GLStates.MultisampleFilterHint := hintFastest
  else GLStates.MultisampleFilterHint := hintDontCare;

  // Specific which color buffers are to be drawn into
  if BufferCount > 1 then
    FGL.DrawBuffers(BufferCount, @MRT_BUFFERS);

  if (ServiceContext <> nil) and (Self <> ServiceContext) then
  begin
    if wglShareLists(TGLWin32Context(ServiceContext).FRC, FRC) then
    begin
      FSharedContexts.Add(ServiceContext);
      PropagateSharedContext;
    end
    else
      GLSLogger.LogWarning('DoCreateContext - Failed to share contexts with resource context');
  end;

  if Assigned(FShareContext) and (FShareContext.RC <> 0) then
  begin
    if not wglShareLists(FShareContext.RC, FRC) then
      GLSLogger.LogWarning(strFailedToShare)
    else
    begin
      FSharedContexts.Add(FShareContext);
      PropagateSharedContext;
    end;
  end;

  Deactivate;

  if GLStates.ForwardContext then
    GLSLogger.LogInfo('PBuffer ' + strFRC_created);
  if bOES then
    GLSLogger.LogInfo('PBuffer ' + strOESRC_created);
  if not (GLStates.ForwardContext or bOES) then
    GLSLogger.LogInfo(strPBufferRC_created);
end;

// DoShareLists
//

function TGLWin32Context.DoShareLists(aContext: TGLContext): Boolean;
begin
  if aContext is TGLWin32Context then
  begin
    FShareContext := TGLWin32Context(aContext);
    if FShareContext.RC <> 0 then
      Result := wglShareLists(FShareContext.RC, RC)
    else
      Result := False;
  end
  else
    raise Exception.Create(strIncompatibleContexts);
end;

// DoDestroyContext
//

procedure TGLWin32Context.DoDestroyContext;
begin
  if vUseWindowTrackingHook then
    UnTrackWindow(WindowFromDC(FDC));

  if FHPBUFFER <> 0 then
  begin
    FGL.WReleasePbufferDCARB(FHPBuffer, FDC);
    FGL.WDestroyPbufferARB(FHPBUFFER);
    FHPBUFFER := 0;
  end;

  if FRC <> 0 then
    if not wglDeleteContext(FRC) then
      GLSLogger.LogErrorFmt(strDeleteContextFailed,
        [GetLastError, SysErrorMessage(GetLastError)]);

  FRC := 0;
  FDC := 0;
  FShareContext := nil;
end;

// DoActivate
//

procedure TGLWin32Context.DoActivate;
begin
  if not wglMakeCurrent(FDC, FRC) then
  begin
    GLSLogger.LogErrorFmt(strContextActivationFailed,
      [GetLastError, SysErrorMessage(GetLastError)]);
    Abort;
  end;

  if not FGL.IsInitialized then
    FGL.Initialize(CurrentGLContext = nil);
end;

// Deactivate
//

procedure TGLWin32Context.DoDeactivate;
begin
  if not wglMakeCurrent(0, 0) then
  begin
    GLSLogger.LogErrorFmt(strContextDeactivationFailed,
      [GetLastError, SysErrorMessage(GetLastError)]);
    Abort;
  end;
end;

// IsValid
//

function TGLWin32Context.IsValid: Boolean;
begin
  Result := (FRC <> 0);
end;

// SwapBuffers
//

procedure TGLWin32Context.SwapBuffers;
begin
  if (FDC <> 0) and (rcoDoubleBuffered in Options) then
    if FSwapBufferSupported then
    begin
      case Layer of
        clUnderlay2: wglSwapLayerBuffers(FDC, WGL_SWAP_UNDERLAY2);
        clUnderlay1: wglSwapLayerBuffers(FDC, WGL_SWAP_UNDERLAY1);
        clMainPlane: Windows.SwapBuffers(FDC);
        clOverlay1: wglSwapLayerBuffers(FDC, WGL_SWAP_OVERLAY1);
        clOverlay2: wglSwapLayerBuffers(FDC, WGL_SWAP_OVERLAY2);
      end;
    end
    else
      Windows.SwapBuffers(FDC);
end;

// RenderOutputDevice
//

function TGLWin32Context.RenderOutputDevice: Pointer;
begin
  Result := Pointer(FDC);
end;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
initialization
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------


end.
