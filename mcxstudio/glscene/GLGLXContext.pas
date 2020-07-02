//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   GLX specific Context.

    History :  
       29/08/10 - Yar - Rewrite DoCreateContext, added CSAA antialiasing
       18/06/10 - Yar - Improved memory context and context sharing
       11/06/10 - Yar - Fixed uses section after lazarus-0.9.29.26033 release
       06/06/10 - Yar - Fixes for Linux x64. DoActivate method now check contexts difference
       21/04/10 - Yar - Added support for GLX versions lower than 1.3
                           (by Rustam Asmandiarov aka Predator)
       06/04/10 - Yar - Update to GLX 1.3-1.4, added PBuffer, forward context creation
                           (by Rustam Asmandiarov aka Predator)
       07/11/09 - DaStr - Improved FPC compatibility (BugtrackerID = 2893580)
                             (thanks Predator)
       10/06/09 - DanB - Added to main GLScene CVS repository (from GLScene-Lazarus).
       14/01/05 - CU - Creation
    
}
unit GLGLXContext;

interface

{$I GLScene.inc}
{$IFDEF SUPPORT_GLX}
uses
  Classes, SysUtils, LCLType,
  GLCrossPlatform, GLContext, OpenGLTokens, OpenGLAdapter,
  x, xlib, xutil;

type
  TGLXFBConfigArray = array[0..MaxInt div (SizeOf(GLXFBConfig) * 2)] of GLXFBConfig;
  PGLXFBConfigArray = ^TGLXFBConfigArray;

  // TGLGLXContext
  //
  { A context driver for GLX. }
  TGLGLXContext = class(TGLContext)
  private
     
    FDisplay: PDisplay;
    FCurScreen: Integer;
    FDC: GLXDrawable;
    FRC: GLXContext;
    FShareContext: TGLGLXContext;
    FHPBUFFER: GLXPBuffer;
    FCurXWindow: HWND;
    FiAttribs: packed array of Integer;
    FFBConfigs: PGLXFBConfigArray;
    FNewTypeContext: boolean;
    procedure ChooseGLXFormat;
    function CreateTempWnd: TWindow;
    procedure DestroyTmpWnd(AWin: TWindow);
    procedure CreateOldContext;
    procedure CreateNewContext;
    procedure Validate;
    function _glXMakeCurrent(dpy: PDisplay; draw: GLXDrawable; ctx: GLXContext):boolean;
  protected
     
    procedure ClearIAttribs;
    procedure FreeIAttribs;
    procedure AddIAttrib(attrib, value: Integer);
    procedure ChangeIAttrib(attrib, newValue: Integer);
    procedure DropIAttrib(attrib: Integer);

    procedure DestructionEarlyWarning(sender: TObject);

    { DoGetHandles must be implemented in child classes,
       and return the display + window }
    procedure DoGetHandles(outputDevice: HWND; out XWin: HWND); virtual;
      abstract;
    procedure GetHandles(AWindowHandle: HWND);
    procedure DoCreateContext(ADeviceHandle: HDC); override;
    procedure DoCreateMemoryContext(ADeviceHandle: HWND; width, height:
      Integer; BufferCount: integer = 1); override;
    function DoShareLists(aContext: TGLContext): Boolean; override;
    procedure DoDestroyContext; override;
    procedure DoActivate; override;
    procedure DoDeactivate; override;

    property DC: GLXDrawable read FDC;
    property RenderingContext: GLXContext read FRC;
    property CurXWindow: HWND read FCurXWindow;
  public
     
    constructor Create; override;
    destructor Destroy; override;

    function IsValid: Boolean; override;
    procedure SwapBuffers; override;

    function RenderOutputDevice: Pointer; override;
  end;
  {$ENDIF}
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// -----------------------------------------------------------------
{$IFDEF SUPPORT_GLX}
uses
  GLState, GLSLog;

resourcestring
  cForwardContextFailed = 'Can not create OpenGL 3.x Forward Context';

  // ------------------
  // ------------------ TGLGLXContext ------------------
  // ------------------

procedure TGLGLXContext.ClearIAttribs;
begin
  SetLength(FiAttribs, 1);
  FiAttribs[0] := 0;
end;

procedure TGLGLXContext.FreeIAttribs;
begin
  SetLength(FiAttribs, 0);
end;

procedure TGLGLXContext.AddIAttrib(attrib, value: Integer);
var
  n: Integer;
begin
  n := Length(FiAttribs);
  SetLength(FiAttribs, n + 2);
  FiAttribs[n - 1] := attrib;
  FiAttribs[n] := value;
  FiAttribs[n + 1] := 0;
end;

procedure TGLGLXContext.ChangeIAttrib(attrib, newValue: Integer);
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

procedure TGLGLXContext.DropIAttrib(attrib: Integer);
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

// Create Temp Window And GLContext 1.1
//

function TGLGLXContext.CreateTempWnd: TWindow;
const
  Attribute: array[0..8] of Integer = (
    GLX_RGBA, GL_TRUE,
    GLX_RED_SIZE, 1,
    GLX_GREEN_SIZE, 1,
    GLX_BLUE_SIZE, 1,
    0);
var
  vi: PXvisualInfo;
begin
  // Lets create temporary window with glcontext
  Result := XCreateSimpleWindow(FDisplay, XRootWindow(FDisplay, FCurScreen),
    0, 0, 1, 1, 0, // need to define some realties dimensions,
    // otherwise the context will not work
    XBlackPixel(FDisplay, FCurScreen),
    XWhitePixel(FDisplay, FCurScreen));
  // XMapWindow(FDisplay, win); // For the test, to see micro window
  XFlush(FDisplay); // Makes XServer execute commands
  vi := glXChooseVisual(FDisplay, FCurScreen, Attribute);
  if vi <> nil then
    FRC := glXCreateContext(FDisplay, vi, nil, true);
  if FRC <> nil then
    glXMakeCurrent(FDisplay, Result, FRC);
  if vi <> nil then
    Xfree(vi);
end;

//Free Window and GLContext
//

procedure TGLGLXContext.DestroyTmpWnd(AWin: TWindow);
begin
  if FDisplay = nil then
    Exit;

  if FRC <> nil then
  begin
    glXMakeCurrent(FDisplay, 0, nil);
    glXDestroyContext(FDisplay, FRC);
    FRC := nil;
  end;

  if @AWin <> nil then
  begin
    XDestroyWindow(FDisplay, AWin);
    XFlush(FDisplay);
  end;
end;

procedure TGLGLXContext.CreateOldContext;
var
  vi: PXvisualInfo;
  shareRC: GLXContext;

begin
  ClearIAttribs;
  if FNewTypeContext then
  begin
    AddIAttrib(GLX_X_RENDERABLE, GL_True);
    AddIAttrib(GLX_RENDER_TYPE, GLX_RGBA_BIT);
    AddIAttrib(GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT);
  end;
  ChooseGLXFormat;

  if (ServiceContext <> nil) and (Self <> ServiceContext) then
    shareRC := TGLGLXContext(ServiceContext).FRC
  else if Assigned(FShareContext) then
    shareRC := FShareContext.FRC
  else
    shareRC := nil;

  try
    if FNewTypeContext then
    begin
      FRC := GL.XCreateNewContext(FDisplay, FFBConfigs[0], GLX_RGBA_TYPE, shareRC, true);
      if Assigned(shareRC) then
      begin
        if Assigned(FRC) then
        begin
          FSharedContexts.Add(FShareContext);
          PropagateSharedContext;
        end
        else
        begin
          GLSLogger.LogWarning(glsFailedToShare);
          FRC := GL.XCreateNewContext(FDisplay, FFBConfigs[0], GLX_RGBA_TYPE, nil, true);
        end;
      end;
    end
    else
    begin
      vi := glXChooseVisual(FDisplay, FCurScreen, @FiAttribs[0]);
      if vi = nil then
        raise EGLContext.Create('Failed to accept attributes');
      GLSLogger.Log('GLGLXContext: DoCreateContext->GLXFormat it is choosed');
      FRC := glXCreateContext(FDisplay, vi, shareRC, true);
      if Assigned(shareRC) then
      begin
        if Assigned(FRC) then
        begin
          if Assigned(FShareContext) then
          begin
            FSharedContexts.Add(FShareContext);
            PropagateSharedContext;
          end;
        end
        else
        begin
          GLSLogger.LogWarning(glsFailedToShare);
          FRC := glXCreateContext(FDisplay, vi, nil, true);
        end;
      end;
      XFree(vi);
    end;

    if not GL.X_ARB_create_context then
    begin
      // Down flags to backcompat
      GLStates.ForwardContext := False;
      PipelineTransformation.LoadMatricesEnabled := True;
      FGL.DebugMode := False;
    end;

    Activate;
    FGL.Initialize;
    // If we are using AntiAliasing, adjust filtering hints
    if AntiAliasing in [aa2xHQ, aa4xHQ, csa8xHQ, csa16xHQ] then
      GLStates.MultisampleFilterHint := hintNicest
    else if AntiAliasing in [aa2x, aa4x, csa8x, csa16x] then
      GLStates.MultisampleFilterHint := hintFastest
    else GLStates.MultisampleFilterHint := hintDontCare;

    GLSLogger.LogInfo('Backward compatible core context successfully created');
  finally
    if Active then
      Deactivate;
    XFree(FFBConfigs);
  end;
end;

procedure TGLGLXContext.CreateNewContext;
var
  bSuccess: Boolean;
  shareRC: GLXContext;
  fnelements: GLInt;
  vFBConfigs: PGLXFBConfigArray;
begin
  XSync(FDisplay, False);
  bSuccess := False;
  vFBConfigs := GL.XChooseFBConfig(FDisplay, FCurScreen, @FiAttribs[0], @fnelements);

  if (ServiceContext <> nil) and (Self <> ServiceContext) then
    shareRC := TGLGLXContext(ServiceContext).FRC
  else if Assigned(FShareContext) then
    shareRC := FShareContext.FRC
  else
    shareRC := nil;

  try
    ClearIAttribs;
    // Initialize forward context
    if GLStates.ForwardContext then
    begin
      if FGL.VERSION_4_1 then
      begin
        AddIAttrib(GLX_CONTEXT_MAJOR_VERSION_ARB, 4);
        AddIAttrib(GLX_CONTEXT_MINOR_VERSION_ARB, 1);
      end
      else if FGL.VERSION_4_0 then
      begin
        AddIAttrib(GLX_CONTEXT_MAJOR_VERSION_ARB, 4);
        AddIAttrib(GLX_CONTEXT_MINOR_VERSION_ARB, 0);
      end
      else if FGL.VERSION_3_3 then
      begin
        AddIAttrib(GLX_CONTEXT_MAJOR_VERSION_ARB, 3);
        AddIAttrib(GLX_CONTEXT_MINOR_VERSION_ARB, 3);
      end
      else if FGL.VERSION_3_2 then
      begin
        AddIAttrib(GLX_CONTEXT_MAJOR_VERSION_ARB, 3);
        AddIAttrib(GLX_CONTEXT_MINOR_VERSION_ARB, 2);
      end
      else if FGL.VERSION_3_1 then
      begin
        AddIAttrib(GLX_CONTEXT_MAJOR_VERSION_ARB, 3);
        AddIAttrib(GLX_CONTEXT_MINOR_VERSION_ARB, 1);
      end
      else if FGL.VERSION_3_0 then
      begin
        AddIAttrib(GLX_CONTEXT_MAJOR_VERSION_ARB, 3);
        AddIAttrib(GLX_CONTEXT_MINOR_VERSION_ARB, 0);
      end
      else
        Abort;
      AddIAttrib(GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB);
    end;

{$IFDEF GLS_LOGGING}
//    if rcoDebug in Options then
    begin
      AddIAttrib(GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_DEBUG_BIT_ARB);
      FGL.DebugMode := True;
    end;
{$ENDIF}

    FRC := nil;
    if Assigned(shareRC) then
    begin
      FRC := FGL.XCreateContextAttribsARB(FDisplay, vFBConfigs[0], shareRC, True, @FiAttribs[0]);
      if Assigned(FRC) then
      begin
        if Assigned(FShareContext) then
        begin
          FSharedContexts.Add(FShareContext);
          PropagateSharedContext;
        end;
      end
      else
        GLSLogger.LogWarning(glsFailedToShare);
    end;

    if not Assigned(FRC) then
    begin
      FRC := FGL.XCreateContextAttribsARB(FDisplay, vFBConfigs[0], nil, True, @FiAttribs[0]);
      if not Assigned(FRC) then
      begin
        GLSLogger.LogError(cForwardContextFailed);
        Abort;
      end;
    end;

    Activate;
    FGL.Initialize;
    // If we are using AntiAliasing, adjust filtering hints
    if AntiAliasing in [aa2xHQ, aa4xHQ, csa8xHQ, csa16xHQ] then
      GLStates.MultisampleFilterHint := hintNicest
    else if AntiAliasing in [aa2x, aa4x, csa8x, csa16x] then
      GLStates.MultisampleFilterHint := hintFastest
    else GLStates.MultisampleFilterHint := hintDontCare;

    if GLStates.ForwardContext then
      GLSLogger.LogInfo('Forward core context successfully created')
    else
      GLSLogger.LogInfo('Backward compatible core context successfully created');
    bSuccess := True;
  finally
    if Active then
      Deactivate;
    GLStates.ForwardContext := GLStates.ForwardContext and bSuccess;
    PipelineTransformation.LoadMatricesEnabled := not GLStates.ForwardContext;
    if Assigned(vFBConfigs) then
      XFree(vFBConfigs);
  end;
  XSync(FDisplay, False);
end;

procedure TGLGLXContext.Validate;
begin
  if FRC = nil then
    raise EGLContext.Create('Failed to create rendering context');
  if PtrUInt(FRC) = GLX_BAD_CONTEXT then
    raise EGLContext.Create('Bad context');
  GLSLogger.Log('GLGLXContext: RenderingContext it is created');
end;

function TGLGLXContext._glXMakeCurrent(dpy: PDisplay; draw: GLXDrawable;
  ctx: GLXContext): boolean;
begin
  if FNewTypeContext then
    Result := GL.XMakeContextCurrent(dpy, draw, draw, ctx)
  else
    Result := glXMakeCurrent(dpy, draw, ctx);
end;

// ChooseGLXFormat
//

procedure TGLGLXContext.ChooseGLXFormat;
var
  vFBConfigs: PGLXFBConfigArray;
  fnelements: Integer;

  function GetFixedAttribute(Attrib: TGLInt; Param: integer): Integer;
  var
    I, Res, OverRes: integer;
  begin
    { Appointment of a function to look for equal or approximate values
       of attributes from the list glx.
      If you just ask all the attributes
      that the user can put it out of ignorance
      Access Violation could appear as the list will be empty. }
    Result := -1;
    OverRes := -1;
    for i := 0 to fnelements - 1 do
    begin
      GL.XGetFBConfigAttrib(FDisplay, vFBConfigs[i], Attrib, @Res);
      if (Res > 0) and (Res <= Param) then
        Result := res;
      if (Res > param) and (OverRes < Res) then
        OverRes := res;
    end;
    if (Result = -1) and (i = fnelements - 1) then
      Result := OverRes;
  end;

  function ChooseFBConfig: Boolean;
  begin
    if Assigned(vFBConfigs) then
      XFree(vFBConfigs);
    vFBConfigs := GL.XChooseFBConfig(FDisplay, FCurScreen, @FiAttribs[0],
      @fnelements);
    Result := Assigned(vFBConfigs);
  end;

const
  cAAToSamples: array[aaDefault..csa16xHQ] of Integer =
    (0, 0, 2, 2, 4, 4, 6, 8, 16, 8, 8, 16, 16);
  cCSAAToSamples: array[csa8x..csa16xHQ] of Integer = (4, 8, 4, 8);

begin
  // Temporarily create a list of available attributes
  vFBConfigs := nil;
  if not ChooseFBConfig then
    raise EGLContext.Create('Failed to accept attributes');

  try
    ColorBits := GetFixedAttribute(GLX_BUFFER_SIZE, ColorBits);
    AddIAttrib(GLX_BUFFER_SIZE, ColorBits);

    if AlphaBits > 0 then
    begin
      AlphaBits := GetFixedAttribute(GLX_ALPHA_SIZE, AlphaBits);
      AddIAttrib(GLX_ALPHA_SIZE, AlphaBits);
    end
    else
      AddIAttrib(GLX_ALPHA_SIZE, 0);

    DepthBits := GetFixedAttribute(GLX_DEPTH_SIZE, DepthBits);
    AddIAttrib(GLX_DEPTH_SIZE, DepthBits);

    if AuxBuffers > 0 then
    begin
      // Even if it is 0 anyway will select something from the list FFBConfigs!
      AuxBuffers := GetFixedAttribute(GLX_AUX_BUFFERS, AuxBuffers);
      AddIAttrib(GLX_AUX_BUFFERS, AuxBuffers);
    end;

    if rcoDoubleBuffered in Options then
    begin
      AddIAttrib(GLX_DOUBLEBUFFER, GL_TRUE);
    end;

    //Stereo not support. See glxinfo
    if rcoStereo in Options then
    begin
      AddIAttrib(GLX_STEREO, GL_TRUE);
    end;

    if StencilBits > 0 then
    begin
      StencilBits := GetFixedAttribute(GLX_STENCIL_SIZE, StencilBits);
      AddIAttrib(GLX_STENCIL_SIZE, StencilBits);
    end;

    if AccumBits>0 then
        AccumBits:=GetFixedAttribute(GLX_ACCUM_RED_SIZE, AccumBits div 4)+
              GetFixedAttribute(GLX_ACCUM_GREEN_SIZE, AccumBits div 4)+
              GetFixedAttribute(GLX_ACCUM_BLUE_SIZE, AccumBits div 4)+
              GetFixedAttribute(GLX_ACCUM_ALPHA_SIZE, AccumBits div 4) ;

    if AccumBits > 0 then
    begin
      AddIAttrib(GLX_ACCUM_RED_SIZE, GetFixedAttribute(GLX_ACCUM_RED_SIZE,
        AccumBits div 4));
      AddIAttrib(GLX_ACCUM_GREEN_SIZE, GetFixedAttribute(GLX_ACCUM_GREEN_SIZE,
        AccumBits div 4));
      AddIAttrib(GLX_ACCUM_BLUE_SIZE, GetFixedAttribute(GLX_ACCUM_BLUE_SIZE,
        AccumBits div 4));
      AddIAttrib(GLX_ACCUM_ALPHA_SIZE, GetFixedAttribute(GLX_ACCUM_ALPHA_SIZE,
        AccumBits div 4));
    end;

    if GL.X_ARB_multisample then
      if AntiAliasing <> aaDefault then
      begin
        if AntiAliasing <> aaNone then
        begin
          AddIAttrib(GLX_SAMPLE_BUFFERS_ARB, GL_TRUE);
          AddIAttrib(GLX_SAMPLES_ARB, GetFixedAttribute(GLX_SAMPLES_ARB,
            cAAToSamples[AntiAliasing]));
          if GL.X_NV_multisample_coverage
            and (AntiAliasing >= csa8x)
            and (AntiAliasing <= csa16xHQ) then
            AddIAttrib(GLX_COLOR_SAMPLES_NV, GetFixedAttribute(GLX_COLOR_SAMPLES_NV,
            cCSAAToSamples[AntiAliasing]));
        end
        else
          AddIAttrib(GLX_SAMPLE_BUFFERS_ARB, GL_FALSE);
      end;
  finally
    if Assigned(vFBConfigs) then
      XFree(vFBConfigs);
  end;

  FFBConfigs := GL.XChooseFBConfig(FDisplay, FCurScreen, @FiAttribs[0], @fnelements);
  if FFBConfigs = nil then
    raise EGLContext.Create('Failed to accept attributes');
end;

procedure TGLGLXContext.DestructionEarlyWarning(sender: TObject);
begin
  DestroyContext;
end;

procedure TGLGLXContext.GetHandles(AWindowHandle: HWND);
begin
  DoGetHandles(AWindowHandle, FCurXWindow);
end;

// DoCreateContext
//

procedure TGLGLXContext.DoCreateContext(ADeviceHandle: HDC);
var
  tempWnd: TWindow;
begin
  // Just in case it didn't happen already.
  if not InitOpenGL then
    RaiseLastOSError;

  FDisplay := XOpenDisplay(nil);

  if FDisplay = nil then
    raise EGLContext.Create('Failed connect to XServer');

  GLSLogger.Log('GLGLXContext: DoCreateContext->Were connected to XServer');

  FCurScreen := XDefaultScreen(FDisplay);

  tempWnd := CreateTempWnd;
  GLSLogger.Log('GLGLXContext: DoCreateContext->Is created a temporary context');
  FGL.Initialize(True);
  FNewTypeContext := GL.X_VERSION_1_2 or GL.X_VERSION_1_3 or GL.X_VERSION_1_4;

  DestroyTmpWnd(tempWnd);
  GLSLogger.LogInfo('Temporary rendering context destroyed');

  GetHandles(HWND(ADeviceHandle));
  FDC := CurXWindow; //FDC - TWindow

  FAcceleration := chaHardware;

  if GL.X_ARB_create_context then
    CreateNewContext
  else
    CreateOldContext;
  Validate;

  if (ServiceContext <> nil) and (Self <> ServiceContext) then
  begin
    FSharedContexts.Add(ServiceContext);
    PropagateSharedContext;
  end;
end;

// DoCreateMemoryContext
//

procedure TGLGLXContext.DoCreateMemoryContext(ADeviceHandle: HWND; width,
  height: Integer; BufferCount: integer);
var
  TempW, TempH: Integer;
  tempWnd: TWindow;
  shareRC: GLXContext;
begin
  // Just in case it didn't happen already.
  if not InitOpenGL then
    RaiseLastOSError;

  FDisplay := XOpenDisplay(nil);

  if FDisplay = nil then
  begin
    raise EGLContext.Create('Failed connect to XServer');
    Exit;
  end;

  GLSLogger.Log('GLGLXContext: DoCreateContext->Were connected to XServer');
  FCurScreen := XDefaultScreen(FDisplay);

  tempWnd := CreateTempWnd;
  GLSLogger.Log('GLGLXContext: DoCreateContext->Is created a temporary context');
  FGL.Initialize(True);
  FNewTypeContext := GL.X_VERSION_1_3 or GL.X_VERSION_1_4;
  DestroyTmpWnd(tempWnd);
  GLSLogger.LogInfo('Temporary rendering context destroyed');

  if (ServiceContext <> nil) and (Self <> ServiceContext) then
    shareRC := TGLGLXContext(ServiceContext).FRC
  else if Assigned(FShareContext) then
    shareRC := FShareContext.FRC
  else
    shareRC := nil;

  if FNewTypeContext then
  begin
    ClearIAttribs;
    AddIAttrib(GLX_X_RENDERABLE, GL_True);
    AddIAttrib(GLX_RENDER_TYPE, GLX_RGBA_BIT);
    AddIAttrib(GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT);
    ChooseGLXFormat;
    try
      FGL.XGetFBConfigAttrib(FDisplay, FFBConfigs[0], GLX_MAX_PBUFFER_HEIGHT, @TempW);
      FGL.XGetFBConfigAttrib(FDisplay, FFBConfigs[0], GLX_MAX_PBUFFER_WIDTH, @TempH);
      if Width <= TempW then
        TempW := Width;
      if Height <= TempH then
        TempH := Height;
      ClearIAttribs;
      AddIAttrib(GLX_PBUFFER_WIDTH, TempW);
      AddIAttrib(GLX_PBUFFER_HEIGHT, TempH);
      AddIAttrib(GLX_LARGEST_BUFFER, 0);
      FHPBUFFER := FGL.XCreatePbuffer(FDisplay, FFBConfigs[0], @FiAttribs[0]);
      if FHPBUFFER = 0 then
        raise EPBuffer.Create('Unabled to create pbuffer.');
      FDC := FHPBUFFER;
      GLSLogger.Log('GLGLXContext: DoCreateContext->PBuffer is Created');

      FAcceleration := chaHardware;

      FRC := FGL.XCreateNewContext(FDisplay, FFBConfigs[0], GLX_RGBA_TYPE, shareRC, true);
      if Assigned(shareRC) then
      begin
        if Assigned(FRC) then
        begin
          FSharedContexts.Add(FShareContext);
          PropagateSharedContext;
        end
        else
        begin
          GLSLogger.LogWarning(glsFailedToShare);
          FRC := FGL.XCreateNewContext(FDisplay, FFBConfigs[0], GLX_RGBA_TYPE, nil, true);
        end;
      end;
      Validate;

      if not FGL.X_ARB_create_context then
      begin
        // Down flags to backcompat
        GLStates.ForwardContext := False;
        PipelineTransformation.LoadMatricesEnabled := True;
        FGL.DebugMode := False;
      end;

      Activate;
      FGL.Initialize;
      // If we are using AntiAliasing, adjust filtering hints
      if AntiAliasing in [aa2xHQ, aa4xHQ, csa8xHQ, csa16xHQ] then
        GLStates.MultisampleFilterHint := hintNicest
      else if AntiAliasing in [aa2x, aa4x, csa8x, csa16x] then
        GLStates.MultisampleFilterHint := hintFastest
      else GLStates.MultisampleFilterHint := hintDontCare;

      if BufferCount > 1 then
        FGL.DrawBuffers(BufferCount, @MRT_BUFFERS);

      if (ServiceContext <> nil) and (Self <> ServiceContext) then
      begin
        FSharedContexts.Add(ServiceContext);
        PropagateSharedContext;
      end;

      GLSLogger.LogInfo('Backward compatible core PBuffer context successfully created');
    finally
      if Active then
        Deactivate;
      XFree(FFBConfigs);
    end;
  end
  else
    raise
        EGLContext.Create('For PBuffer Context required GLX above 1.2');
end;

// DoShareLists
//

function TGLGLXContext.DoShareLists(aContext: TGLContext): Boolean;
var
  otherRC: GLXContext;
begin
  Result := False;
  if aContext is TGLGLXContext then
  begin
    otherRC := TGLGLXContext(aContext).RenderingContext;
    if RenderingContext <> nil then
    begin
      if (RenderingContext <> otherRC) then
      begin
        DestroyContext;
        FShareContext:= TGLGLXContext(aContext);
        Result := True;
      end;
    end
    else
    begin
      FShareContext := TGLGLXContext(aContext);
      Result := False;
    end;
  end
  else
    raise Exception.Create(cIncompatibleContexts);
end;

procedure TGLGLXContext.DoDestroyContext;
begin
  GLSLogger.Log('GLGLXContext: DoDestroyContext');
  if not Assigned(FDisplay) then
    raise EGLContext.Create('Lost connection XServer');
  if (glXGetCurrentContext() = FRC) and
    (not _glXMakeCurrent(FDisplay, 0, nil)) then
    raise EGLContext.Create(cContextDeactivationFailed);
  glXDestroyContext(FDisplay, FRC);
  GLSLogger.Log('GLGLXContext: DoDestroyContext->RenderingContext it is Destroyd');
  if FHPBUFFER <> 0 then
  begin
    GL.XDestroyPbuffer(FDisplay, FHPBUFFER);
    FHPBUFFER := 0;
    GLSLogger.Log('GLGLXContext: DoDestroyContext->RenderingContext it is Destroyd');
  end;
  FRC := nil;
  FDC := 0;
  FShareContext := nil;
  XCloseDisplay(FDisplay);
  FCurScreen := 0;
end;

// DoActivate
//

procedure TGLGLXContext.DoActivate;
begin
  if not _glXMakeCurrent(FDisplay, FDC, FRC) then
    raise EGLContext.Create(cContextActivationFailed);
  XSync(FDisplay, False);

  if not FGL.IsInitialized then
    FGL.Initialize;
end;

// Deactivate
//

procedure TGLGLXContext.DoDeactivate;
begin
  if not _glXMakeCurrent(FDisplay, 0, nil) then
    raise EGLContext.Create(cContextDeactivationFailed);
end;

constructor TGLGLXContext.Create;
begin
  inherited Create;
  ClearIAttribs;
end;

destructor TGLGLXContext.Destroy;
begin
  inherited Destroy;
end;

// IsValid
//

function TGLGLXContext.IsValid: Boolean;
begin
  Result := (FRC <> nil);
end;

// SwapBuffers
//

procedure TGLGLXContext.SwapBuffers;
begin
  if (FDC <> 0) and (rcoDoubleBuffered in Options) then
    glXSwapBuffers(FDisplay, FDC);
end;

function TGLGLXContext.RenderOutputDevice: Pointer;
begin
  Result := nil;
end;
{$ENDIF}
end.
