//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Carbon specific Context.

    History :  
       19/02/11 - PREDATOR - Added Share Context, MemoryViewerContext. 
                                Updated Chose Pixel Format
       16/02/11 - PREDATOR - Added support for Mac OS X. Tested on Mac OS X 10.6.5.
       10/06/09 - DanB - Added to main GLScene CVS repository (from GLScene-Lazarus).
       14/11/08 - Creation
    
}
unit GLCarbonContext;

{$I GLScene.inc}

interface

uses
  MacOSAll,
  Classes, SysUtils,LCLType,  GLCrossPlatform, GLContext, LCLProc, Forms, Controls,
  OpenGLAdapter, OpenGLTokens, CarbonDef, CarbonCanvas, CarbonProc, CarbonPrivate;

type
   // TGLCarbonContext
   //
   { A context driver for standard XOpenGL. }
   TGLCarbonContext = class (TGLContext)
      private
          
         FRC: TAGLContext;
         FShareContext: TGLCarbonContext;
         FHPBUFFER: PAGLPBuffer;
         FDC: TAGLDrawable;
         FBounds: TRect;
         FViewer, FForm: TControl;
         FIAttribs : packed array of Integer;
         FPixelFmt: TAGLPixelFormat;
         FDisp: GDHandle;
         FWindow: WindowRef;

         procedure ChooseAGLFormat;
         function GetFormBounds: TRect;
         procedure BoundsChanged;
         function CreateWindow: WindowRef;
         procedure DestroyWindow(AWin: WindowRef);
         procedure CreateOldContext;
         procedure CreateNewContext;
         procedure Validate;
      protected
          
         procedure ClearIAttribs;
         procedure AddIAttrib(attrib, value : Integer);
         procedure ChangeIAttrib(attrib, newValue : Integer);
         procedure DropIAttrib(attrib : Integer);

         procedure DestructionEarlyWarning(sender: TObject);

         { DoGetHandles must be implemented in child classes,
            and return the display + window }
         procedure DoGetHandles(outputDevice: HWND; out XWin: HWND); virtual;
           abstract;
         procedure GetHandles(outputDevice: HWND);
         procedure DoCreateContext(ADeviceHandle: HDC); override;
         procedure DoCreateMemoryContext(outputDevice : HWND;width, height : Integer; BufferCount : integer); override;
         function  DoShareLists(aContext : TGLContext): Boolean;  override;
         procedure DoDestroyContext; override;
         procedure DoActivate; override;
         procedure DoDeactivate; override;
         //property DC: HWND read FDC;
         property RenderingContext: TAGLContext read FRC;
      public
          
         constructor Create; override;
         destructor Destroy; override;

         function IsValid : Boolean; override;
         procedure SwapBuffers; override;

         function RenderOutputDevice : Pointer; override;
   end;
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// -----------------------------------------------------------------
uses
  GLState, GLSLog;


resourcestring
   cIncompatibleContexts =       'Incompatible contexts';
   cDeleteContextFailed =        'Delete context failed';
   cContextActivationFailed =    'Context activation failed: %X, %s';
   cContextDeactivationFailed =  'Context deactivation failed';
   cUnableToCreateLegacyContext= 'Unable to create legacy context';

{ TGLCarbonContext }

procedure TGLCarbonContext.ChooseAGLFormat;
var
  vPixelFmt: TAGLPixelFormat;

  function GetFixedAttribute(Attrib: TGLInt; Param: integer): Integer;
  var
    I, Res, OverRes: integer;
  begin
    { Appointment of a function to look for equal or approximate values
       of attributes from the list AGL.
      If you just ask all the attributes
      that the user can put it out of ignorance
      Access Violation could appear as the list will be empty. }
   { Result := -1;
    OverRes := -1;
    for i := 0 to 10 do    //10 - fnelements
    begin }

      GL.ADescribePixelFormat(vPixelFmt, Attrib, @Res);
     { if (Res > 0) and (Res <= Param) then}
        Result := res;
    {  if (Res > param) and (OverRes < Res) then
        OverRes := res;
    end;
    if (Result = -1) and (i = 10) then
      Result := OverRes;  }

  //GLSLogger.Log('GetFixedAttribute:'+'Attrib'+inttostr(Attrib)+' Param'+inttostr(Param)+' Res'+inttostr(Result));

  end;

  function ChoosePixelFormat: Boolean;
  begin
    if Assigned(vPixelFmt) then
      GL.ADestroyPixelFormat(vPixelFmt);
    vPixelFmt := GL.aChoosePixelFormat(@FDisp, 1, @FIAttribs[0]);
    Result := Assigned(vPixelFmt);
  end;

const
  cAAToSamples: array[aaDefault..csa16xHQ] of Integer =
    (0, 0, 2, 2, 4, 4, 6, 8, 16, 8, 8, 16, 16);
  cCSAAToSamples: array[csa8x..csa16xHQ] of Integer = (4, 8, 4, 8);

begin
  // Temporarily create a list of available attributes
  vPixelFmt := nil;
  if not ChoosePixelFormat then
    raise EGLContext.Create('vFailed to accept attributes');

  try
    ColorBits := GetFixedAttribute(AGL_BUFFER_SIZE, ColorBits);
    AddIAttrib(AGL_BUFFER_SIZE, ColorBits);

    if AlphaBits > 0 then
    begin
      AlphaBits := GetFixedAttribute(AGL_ALPHA_SIZE, AlphaBits);
      AddIAttrib(AGL_ALPHA_SIZE, AlphaBits);
    end
    else
      AddIAttrib(AGL_ALPHA_SIZE, 0);

    DepthBits := GetFixedAttribute(AGL_DEPTH_SIZE, DepthBits);
    AddIAttrib(AGL_DEPTH_SIZE, DepthBits);

    if AuxBuffers > 0 then
    begin
      // Even if it is 0 anyway will select something from the list FFBConfigs!
      AuxBuffers := GetFixedAttribute(AGL_AUX_BUFFERS, AuxBuffers);
      AddIAttrib(AGL_AUX_BUFFERS, AuxBuffers);
    end;

    if rcoDoubleBuffered in Options then
    begin
      AddIAttrib(AGL_DOUBLEBUFFER, GL_TRUE);
    end;

    //Stereo not support.
    if rcoStereo in Options then
    begin
      AddIAttrib(AGL_STEREO, GL_TRUE);
    end;

    if StencilBits > 0 then
    begin
      StencilBits := GetFixedAttribute(AGL_STENCIL_SIZE, StencilBits);
      AddIAttrib(AGL_STENCIL_SIZE, StencilBits);
    end;

    if AccumBits>0 then
        AccumBits:=GetFixedAttribute(AGL_ACCUM_RED_SIZE, AccumBits div 4)+
              GetFixedAttribute(AGL_ACCUM_GREEN_SIZE, AccumBits div 4)+
              GetFixedAttribute(AGL_ACCUM_BLUE_SIZE, AccumBits div 4)+
              GetFixedAttribute(AGL_ACCUM_ALPHA_SIZE, AccumBits div 4) ;

    if AccumBits > 0 then
    begin
      AddIAttrib(AGL_ACCUM_RED_SIZE, GetFixedAttribute(AGL_ACCUM_RED_SIZE,
        AccumBits div 4));
      AddIAttrib(AGL_ACCUM_GREEN_SIZE, GetFixedAttribute(AGL_ACCUM_GREEN_SIZE,
        AccumBits div 4));
      AddIAttrib(AGL_ACCUM_BLUE_SIZE, GetFixedAttribute(AGL_ACCUM_BLUE_SIZE,
        AccumBits div 4));
      AddIAttrib(AGL_ACCUM_ALPHA_SIZE, GetFixedAttribute(AGL_ACCUM_ALPHA_SIZE,
        AccumBits div 4));
    end;

    if GL.ARB_multisample then
      if AntiAliasing <> aaDefault then
      begin
        if AntiAliasing <> aaNone then
        begin
          AddIAttrib(AGL_SAMPLE_BUFFERS_ARB, GL_TRUE);
          AddIAttrib(AGL_SAMPLES_ARB, GetFixedAttribute(AGL_SAMPLES_ARB,
            cAAToSamples[AntiAliasing]));
         { if GL.X_NV_multisample_coverage
            and (AntiAliasing >= csa8x)
            and (AntiAliasing <= csa16xHQ) then
            AddIAttrib(AGL_COLOR_SAMPLES_NV, GetFixedAttribute(AGL_COLOR_SAMPLES_NV,
            cCSAAToSamples[AntiAliasing]));  }
        end
        else
          AddIAttrib(GL_SAMPLE_BUFFERS_ARB, GL_FALSE);
      end;

  finally
    if Assigned(vPixelFmt) then
      FGL.ADestroyPixelFormat(vPixelFmt);
  end;
  FPixelFmt := GL.aChoosePixelFormat(@FDisp, 1, @FIAttribs[0]);
  if FPixelFmt = nil then
    raise EGLContext.Create('Failed to accept attributes');
end;

function TGLCarbonContext.GetFormBounds: TRect;
begin
  Result.TopLeft := FForm.ScreenToClient(FViewer.ControlToScreen(Point(0, 0)));
  Result.Right := Result.Left + FViewer.Width;
  Result.Bottom := Result.Top + FViewer.Height;
end;

procedure TGLCarbonContext.BoundsChanged;
var
  Bounds: Array [0..3] of GLint;
begin
  Bounds[0] := FBounds.Left;
  Bounds[1] := FForm.Height - FBounds.Bottom;
  Bounds[2] := FBounds.Right - FBounds.Left;
  Bounds[3] := FBounds.Bottom - FBounds.Top;

  FGL.aSetInteger(FRC, AGL_BUFFER_RECT, @Bounds[0]);
  FGL.aEnable(FRC, AGL_BUFFER_RECT);

  {$MESSAGE Warn 'Removing child controls from clip region needs to be implemented'}
(*BoundsRGN := NewRgn;
  RectRgn(BoundsRGN, GetCarbonRect(TCarbonControlContext(DC).Owner.LCLObject.BoundsRect));

  aglSetInteger(FContext, AGL_CLIP_REGION, PGLInt(BoundsRGN));
  aglEnable(FContext, AGL_CLIP_REGION);*)

  FGL.aUpdateContext(FRC);
end;

procedure TGLCarbonContext.ClearIAttribs;
begin
  SetLength(FIAttribs, 1);
  FiAttribs[0]:=0;
end;

procedure TGLCarbonContext.AddIAttrib(attrib, value: Integer);
var
  N: Integer;
begin
  N := Length(FIAttribs);
  SetLength(FIAttribs, N+2);
  FiAttribs[N-1]:=attrib;
  FiAttribs[N]:=value;
  FiAttribs[N+1]:=0;
end;

procedure TGLCarbonContext.ChangeIAttrib(attrib, newValue: Integer);
var
  i : Integer;
begin
  i:=0;
  while i<Length(FiAttribs) do begin
    if FiAttribs[i]=attrib then begin
      FiAttribs[i+1]:=newValue;
      Exit;
    end;
    Inc(i, 2);
  end;
  AddIAttrib(attrib, newValue);
end;

procedure TGLCarbonContext.DropIAttrib(attrib: Integer);
var
  i: Integer;
begin
  i:=0;
  while i<Length(FiAttribs) do begin
    if FiAttribs[i]=attrib then begin
      Inc(i, 2);
      while i<Length(FiAttribs) do begin
        FiAttribs[i-2]:=FiAttribs[i];
        Inc(i);
      end;
      SetLength(FiAttribs, Length(FiAttribs)-2);
      Exit;
    end;
    Inc(i, 2);
  end;
end;

procedure TGLCarbonContext.DestructionEarlyWarning(sender: TObject);
begin
   DestroyContext;
end;

procedure TGLCarbonContext.GetHandles(outputDevice: HWND);
begin
  //DoGetHandles(outputDevice, FCurXWindow);
end;

function TGLCarbonContext.CreateWindow: WindowRef;
var
     windowAttrs : WindowAttributes;
     bounds      : MacOSAll.Rect;
     format            :   TAGLPixelFormat ;         //* OpenGL pixel format */
const
     Attributes: array[0..8] of GLint = (  //* OpenGL attributes */
                          AGL_RGBA, GL_TRUE,
                          AGL_GREEN_SIZE, 1,
                          AGL_DOUBLEBUFFER, GL_TRUE,
                          AGL_DEPTH_SIZE, 16,
                          AGL_NONE      );
begin
  // Lets create temporary window with glcontext
  windowAttrs := kWindowCloseBoxAttribute or kWindowCollapseBoxAttribute or kWindowStandardHandlerAttribute;

   MacOSAll.SetRect(bounds, 0, 0, 1, 1);
   MacOSAll.OffsetRect(bounds, 1, 1);

   CreateNewWindow(kDocumentWindowClass, windowAttrs, bounds, Result);
   ShowWindow(Result);
   //ShowHide(window,false);  //показать спрятать окошко
   format      := aglChoosePixelFormat(nil, 0, attributes);
   if  format<>nil then
     FRC := aglCreateContext(format, nil);
   if  format<>nil then
     aglDestroyPixelFormat(format);
   if (FRC<>nil) then
     begin
      aglSetDrawable(FRC, GetWindowPort(Result));
      aglSetCurrentContext(FRC);
     end;
end;

//Free Window and GLContext
//

procedure TGLCarbonContext.DestroyWindow(AWin: WindowRef);
begin

  if FRC <> nil then
  begin
    aglSetCurrentContext(nil);
    aglSetDrawable(FRC, nil);
    aglDestroyContext(FRC);
    FRC := nil;
  end;

  if @AWin <> nil then
  begin
    HideWindow(AWin);
    ReleaseWindow(AWin);
  end;
end;

procedure TGLCarbonContext.CreateOldContext;
var
  shareRC: TAGLContext;
begin
  ClearIAttribs;

  AddIAttrib(AGL_WINDOW, GL_TRUE);
  AddIAttrib(AGL_RGBA, GL_TRUE);

  ChooseAGLFormat;

  if (ServiceContext <> nil) and (Self <> ServiceContext) then
    shareRC := TGLCarbonContext(ServiceContext).FRC
  else if Assigned(FShareContext) then
    shareRC := FShareContext.FRC
  else
    shareRC := nil;

  try
      FRC := FGL.aCreateContext(FPixelFmt, shareRC);
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
          FRC := FGL.aCreateContext(FPixelFmt, nil);
        end;
      end;
    FGL.aSetDrawable(FRC, GetWindowPort(FWindow));

    FBounds := GetFormBounds;
    BoundsChanged;
    {$IFDEF GLS_LOGGING}
    GLSLogger.LogInfo('GLCarbonContext: BoundsChanged');
    {$ENDIF}

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
    GL.aDestroyPixelFormat(FPixelFmt);
  end;
end;

procedure TGLCarbonContext.CreateNewContext;
begin
  // OpenGL 3 not yet implemented
end;

procedure TGLCarbonContext.Validate;
begin
  if FRC = nil then
    raise EGLContext.Create('Failed to create rendering context');
  if PtrUInt(FRC) = AGL_BAD_CONTEXT then
    raise EGLContext.Create('Bad context');
  if PtrUInt(FRC) = AGL_BAD_PIXELFMT then
    raise EGLContext.Create('BAD PIXELForMaT');

  {$IFDEF GLS_LOGGING}
  GLSLogger.Log('GLAGLContext: RenderingContext it is created');
  {$ENDIF}
end;

procedure TGLCarbonContext.DoCreateContext(ADeviceHandle: HDC);
var
  DC: TCarbonDeviceContext absolute ADeviceHandle;
begin
  // Just in case it didn't happen already.
  if not InitOpenGL then
    RaiseLastOSError;

  FWindow := CreateWindow;
  GLSLogger.Log('GLCarbonContext: Is created a temporary context');

  FGL.Initialize(True);

  DestroyWindow(FWindow);
  FWindow := nil;
  GLSLogger.LogInfo('GLCarbonContext: Temporary rendering context destroyed');

  if not (CheckDC(ADeviceHandle, 'DoCreateContext') or (DC is TCarbonControlContext)) then
  begin
    raise EGLContext.Create('Creating context failed: invalid device context!');
    GLSLogger.LogInfo('GLCarbonContext:Creating context failed: invalid device context!');
  end;

  FViewer := TCarbonControlContext(DC).Owner.LCLObject;
  FForm := FViewer.GetTopParent;
  if not (FForm is TCustomForm) then
  begin
    raise EGLContext.Create('Creating context failed: control not on the form!');
    GLSLogger.LogInfo('GLCarbonContext: Creating context failed: control not on the form!');
  end;

  FWindow := TCarbonWindow((FForm as TWinControl).Handle).Window;
  FDC := FWindow;
  // create the AGL context
  FDisp := GetMainDevice();

  FAcceleration := chaHardware;
  GLSLogger.LogInfo('GLCarbonContext: Control Handle Accepted');

  CreateOldContext;
  Validate;

  if (ServiceContext <> nil) and (Self <> ServiceContext) then
  begin
    FSharedContexts.Add(ServiceContext);
    PropagateSharedContext;
  end;

end;

type
  TVP = array[0..1] of TGLint;

procedure TGLCarbonContext.DoCreateMemoryContext(outputDevice: HWND; width,
  height: Integer; BufferCount: integer);
var
  TempW, TempH  : Integer;
  shareRC       : TAGLContext;
  vs            : Integer;
  vp            : TVP ;
  target        : GLenum;
begin
  // Just in case it didn't happen already.
  if not InitOpenGL then
    RaiseLastOSError;

  FWindow := CreateWindow;
  GLSLogger.Log('GLCarbonContext: Is created a temporary context');

  FGL.Initialize(True);

  DestroyWindow(FWindow);
  FWindow := nil;
  GLSLogger.LogInfo('GLCarbonContext: Temporary rendering context destroyed');

  FDisp := GetMainDevice();

  ClearIAttribs;

  AddIAttrib(AGL_PBUFFER, GL_TRUE);
  AddIAttrib(AGL_RGBA, GL_TRUE);
  AddIAttrib(AGL_ACCELERATED, GL_TRUE);
  //  AddIAttrib(AGL_NO_RECOVERY, GL_TRUE);


  ChooseAGLFormat;

  if Assigned(FShareContext) then
    shareRC := FShareContext.FRC
  else
    shareRC := nil;

  try
    TempW := Width;
    TempH := Height;

    if Width <= 32 then
      TempW := 32;
    if Height <= 32 then
      TempH := 32;

    if Width >= 16384 then
      TempW := 16384;
    if Height >= 16384 then
      TempH := 16384;

    if TempH <> TempW then
      target := GL_TEXTURE_2D
    else  target := GL_TEXTURE_RECTANGLE_EXT;

    if not FGL.ACreatePBuffer(TempW, TempH, target, GL_RGBA,0, @FHPBUFFER) then
    if FHPBUFFER = nil then
      raise EPBuffer.Create('Unabled to create pbuffer.');
    FDC := FHPBUFFER;
    GLSLogger.Log('GLCarbonContext: PBuffer is Created');

    FAcceleration := chaHardware;

    FRC := FGL.aCreateContext(FPixelFmt, shareRC);
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
        FRC := FGL.aCreateContext(FPixelFmt, nil);
      end;
    end;

    vs := FGL.AGetVirtualScreen(FRC);
    if not FGL.ASetPBuffer(FRC, FHPBUFFER, 0, 0, vs) then
      raise EPBuffer.Create('pbuffer dont set');

    Validate;

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
    GL.aDestroyPixelFormat(FPixelFmt);
  end;
end;

function  TGLCarbonContext.DoShareLists(aContext: TGLContext): Boolean;
var
  otherRC: TAGLContext;
begin
  Result := False;
  if aContext is TGLCarbonContext then
  begin
    otherRC := TGLCarbonContext(aContext).RenderingContext;
    if RenderingContext <> nil then
    begin
      if (RenderingContext <> otherRC) then
      begin
        DestroyContext;
        FShareContext:= TGLCarbonContext(aContext);
        Result := True;
      end;
    end
    else
    begin
      FShareContext := TGLCarbonContext(aContext);
      Result := False;
    end;
  end
  else
    raise Exception.Create(cIncompatibleContexts);
end;

procedure TGLCarbonContext.DoDestroyContext;
begin
  if (FGL.aGetCurrentContext = FRC) and
     (not FGL.aSetCurrentContext(nil)) then
    raise EGLContext.Create('Failed to deselect rendering context');
  FGL.aDestroyContext(FRC);
  GLSLogger.Log('GLAGLContext: RenderingContext it is Destroyd');

  if FHPBUFFER <> nil then
  begin
    GL.ADestroyPBuffer(FHPBUFFER);
    FHPBUFFER := nil;
    GLSLogger.Log('GLAGLContext: PBUFFER it is Destroyd');
  end;

  FRC := nil;
  FDC := nil;
  FShareContext := nil;
end;

procedure TGLCarbonContext.DoActivate;
var
  B: TRect;
begin
  if FHPBUFFER = nil then
    begin
    B := GetFormBounds;
    if (B.Left <> FBounds.Left) or (B.Top <> FBounds.Top) or
      (B.Right <> FBounds.Right) or (B.Bottom <> FBounds.Bottom) then
    begin
      FBounds := B;
      BoundsChanged;
    end;
  end;

  if (not FGL.aSetCurrentContext(FRC)) then
    raise EGLContext.Create(cContextActivationFailed);

  if not FGL.IsInitialized then
     FGL.Initialize;
end;

procedure TGLCarbonContext.DoDeactivate;
begin
  if (not FGL.aSetCurrentContext(nil)) then
    raise EGLContext.Create(cContextDeactivationFailed);
end;

constructor TGLCarbonContext.Create;
begin
  inherited Create;
  ClearIAttribs;

end;

destructor TGLCarbonContext.Destroy;
begin
  inherited Destroy;
end;

function TGLCarbonContext.IsValid: Boolean;
begin
  Result := (FRC <> nil);
end;

procedure TGLCarbonContext.SwapBuffers;
begin
  if (FRC <> nil) and (rcoDoubleBuffered in Options) then
    FGL.aSwapBuffers(FRC);
end;

function TGLCarbonContext.RenderOutputDevice: Pointer;
begin
  Result := nil;
end;

end.
