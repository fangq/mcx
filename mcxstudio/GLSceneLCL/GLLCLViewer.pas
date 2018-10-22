//
// This unit is part of the GLScene Project, http://glscene.org
//
{
  A FPC specific Scene viewer.

   History :  
       02/06/10 - Yar - Fixes for Linux x64
       28/04/10 - Yar - Fixed conditions for windows platform
                           Added Render method
                           (by Rustam Asmandiarov aka Predator)
       02/04/10 - Yar - Bugfix bad graphics under Windows
                           (by Rustam Asmandiarov aka Predator)
       22/12/09 - DaStr - Published TabStop, TabOrder, OnEnter, OnExit
                              properties (thanks Yury Plashenkov)
       07/11/09 - DaStr - Improved FPC compatibility (BugtrackerID = 2893580)
                             (thanks Predator)
       13/07/09 - DanB - added the FieldOfView property + reduced OpenGL dependencies
       10/04/08 - DaStr - Bugfixed TGLSceneViewer.Notification()
                              (thanks z80maniac) (Bugtracker ID = 1936108)
       12/09/07 - DaStr - Removed old IFDEFs. Moved SetupVSync()
                              to GLViewer.pas (Bugtracker ID = 1786279)
       04/06/04 -  EG   - Created from GLWin32Viewer
   
}
unit GLLCLViewer;

interface

{$I GLScene.inc}

uses
  {$IFDEF MSWINDOWS} Windows,{$ENDIF}
  Messages, 
  Graphics, 
  Forms, 
  Classes, 
  Controls, 
  Menus, 
  LMessages, 
  LCLType,
  GLScene, 
  GLContext;

type
   { Component where the GLScene objects get rendered.
      This component delimits the area where OpenGL renders the scene,
      it represents the 3D scene viewed from a camera (specified in the
      camera property). This component can also render to a file or to a bitmap.
      It is primarily a windowed component, but it can handle full-screen
      operations : simply make this component fit the whole screen (use a
      borderless form).
      This viewer also allows to define rendering options such a fog, face culling,
      depth testing, etc. and can take care of framerate calculation. }
  TGLSceneViewer = class(TWinControl)
  private
    FBuffer: TGLSceneBuffer;
    FVSync: TVSyncMode;
    FOwnDC: HWND;
    FOnMouseEnter, FOnMouseLeave: TNotifyEvent;
    FMouseInControl: boolean;
    FLastScreenPos: TPoint;

    procedure LMEraseBkgnd(var Message: TLMEraseBkgnd); message LM_ERASEBKGND;
    procedure LMPaint(var Message: TLMPaint); message LM_PAINT;
    procedure LMSize(var Message: TLMSize); message LM_SIZE;
    procedure LMDestroy(var Message: TLMDestroy); message LM_DESTROY;

    procedure CMMouseEnter(var msg: TMessage); message CM_MOUSEENTER;
    procedure CMMouseLeave(var msg: TMessage); message CM_MOUSELEAVE;
    function GetFieldOfView: single;
    procedure SetFieldOfView(const Value: single);
    function GetIsRenderingContextAvailable: boolean;
  protected
    procedure SetBeforeRender(const val: TNotifyEvent);
    function GetBeforeRender: TNotifyEvent;
    procedure SetPostRender(const val: TNotifyEvent);
    function GetPostRender: TNotifyEvent;
    procedure SetAfterRender(const val: TNotifyEvent);
    function GetAfterRender: TNotifyEvent;
    procedure SetCamera(const val: TGLCamera);
    function GetCamera: TGLCamera;
    procedure SetBuffer(const val: TGLSceneBuffer);

    {$IFDEF MSWINDOWS}
    procedure CreateParams(var Params: TCreateParams); override;
    {$ENDIF}
    procedure CreateWnd; override;
    procedure DestroyWnd; override;
    procedure Loaded; override;
    procedure DoBeforeRender(Sender: TObject); dynamic;
    procedure DoBufferChange(Sender: TObject); virtual;
    procedure DoBufferStructuralChange(Sender: TObject); dynamic;

  public
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure Render;
    procedure Notification(AComponent: TComponent; Operation: TOperation); override;
    { Makes TWinControl's RecreateWnd public.
            This procedure allows to work around limitations in some OpenGL
            drivers (like MS Software OpenGL) that are not able to share lists
            between RCs that already have display lists. }

    property IsRenderingContextAvailable: boolean
      read GetIsRenderingContextAvailable;
    function LastFrameTime: single;
    function FramesPerSecond: single;
    function FramesPerSecondText(decimals: integer = 1): string;
    procedure ResetPerformanceMonitor;
    function CreateSnapShotBitmap: TBitmap;
    property RenderDC: HWND read FOwnDC;
    property MouseInControl: boolean read FMouseInControl;
  published
     
    { Camera from which the scene is rendered. }
    property Camera: TGLCamera read GetCamera write SetCamera;

         { Specifies if the refresh should be synchronized with the VSync signal.
            If the underlying OpenGL ICD does not support the WGL_EXT_swap_control
            extension, this property is ignored.  }
    property VSync: TVSyncMode read FVSync write FVSync default vsmNoSync;

         { Triggered before the scene's objects get rendered.
            You may use this event to execute your own OpenGL rendering. }
    property BeforeRender: TNotifyEvent read GetBeforeRender write SetBeforeRender;
         { Triggered just after all the scene's objects have been rendered.
            The OpenGL context is still active in this event, and you may use it
            to execute your own OpenGL rendering. }
    property PostRender: TNotifyEvent read GetPostRender write SetPostRender;
         { Called after rendering.
            You cannot issue OpenGL calls in this event, if you want to do your own
            OpenGL stuff, use the PostRender event. }
    property AfterRender: TNotifyEvent read GetAfterRender write SetAfterRender;

    { Access to buffer properties. }
    property Buffer: TGLSceneBuffer read FBuffer write SetBuffer;

         { Returns or sets the field of view for the viewer, in degrees.
         This value depends on the camera and the width and height of the scene.
         The value isn't persisted, if the width/height or camera.focallength is
         changed, FieldOfView is changed also. }
    property FieldOfView: single read GetFieldOfView write SetFieldOfView;
    property OnMouseLeave: TNotifyEvent read FOnMouseLeave write FOnMouseLeave;
    property OnMouseEnter: TNotifyEvent read FOnMouseEnter write FOnMouseEnter;
    property Align;
    property Anchors;
    property DragCursor;
    property DragMode;
    property Enabled;
    property HelpContext;
    property Hint;
    property PopupMenu;
    property Visible;
    property OnClick;
    property OnDblClick;
    property OnDragDrop;
    property OnDragOver;
    property OnStartDrag;
    property OnEndDrag;
    property OnMouseDown;
    property OnMouseMove;
    property OnMouseUp;
    property OnKeyDown;
    property OnKeyUp;
    property OnMouseWheel;
    property OnMouseWheelDown;
    property OnMouseWheelUp;
    property OnContextPopup;
    property TabStop;
    property TabOrder;
    property OnEnter;
    property OnExit;
  end;

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

uses SysUtils, LCLIntf, GLViewer
       {$if DEFINED(LCLWIN32) or DEFINED(LCLWIN64)}
         {$ifndef CONTEXT_INCLUDED}
  , GLWidgetContext
         {$define CONTEXT_INCLUDED}
         {$endif}
       {$endif}

       {$if DEFINED(LCLGTK) or DEFINED(LCLGTK2)}
         {$ifndef CONTEXT_INCLUDED}
  , GLWidgetContext
         {$define CONTEXT_INCLUDED}
         {$endif}
       {$endif}
       {$ifdef LCLCARBON}
         {$ifndef CONTEXT_INCLUDED}
  , GLWidgetContext
         {$define CONTEXT_INCLUDED}
         {$endif}
       {$endif}

       {$ifdef LCLQT}
         {$error unimplemented QT context}
       {$endif}       ;
// ------------------
// ------------------ TGLSceneViewer ------------------
// ------------------

constructor TGLSceneViewer.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  ControlStyle := [csClickEvents, csDoubleClicks, csOpaque, csCaptureMouse];
  if csDesigning in ComponentState then
    ControlStyle := ControlStyle + [csFramed];
  Width := 100;
  Height := 100;
  FVSync := vsmNoSync;
  FBuffer := TGLSceneBuffer.Create(Self);
  FBuffer.ViewerBeforeRender := DoBeforeRender;
  FBuffer.OnChange := DoBufferChange;
  FBuffer.OnStructuralChange := DoBufferStructuralChange;
end;

destructor TGLSceneViewer.Destroy;
begin
  FBuffer.Free;
  inherited Destroy;
end;

procedure TGLSceneViewer.Notification(AComponent: TComponent; Operation: TOperation);
begin
  if (Operation = opRemove) and (FBuffer <> nil) then
  begin
    if (AComponent = FBuffer.Camera) then
      FBuffer.Camera := nil;
  end;
  inherited;
end;

procedure TGLSceneViewer.SetBeforeRender(const val: TNotifyEvent);
begin
  FBuffer.BeforeRender := val;
end;

function TGLSceneViewer.GetBeforeRender: TNotifyEvent;
begin
  Result := FBuffer.BeforeRender;
end;

procedure TGLSceneViewer.SetPostRender(const val: TNotifyEvent);
begin
  FBuffer.PostRender := val;
end;

function TGLSceneViewer.GetPostRender: TNotifyEvent;
begin
  Result := FBuffer.PostRender;
end;

procedure TGLSceneViewer.SetAfterRender(const val: TNotifyEvent);
begin
  FBuffer.AfterRender := val;
end;

function TGLSceneViewer.GetAfterRender: TNotifyEvent;
begin
  Result := FBuffer.AfterRender;
end;

procedure TGLSceneViewer.SetCamera(const val: TGLCamera);
begin
  FBuffer.Camera := val;
end;

function TGLSceneViewer.GetCamera: TGLCamera;
begin
  Result := FBuffer.Camera;
end;

procedure TGLSceneViewer.SetBuffer(const val: TGLSceneBuffer);
begin
  FBuffer.Assign(val);
end;

{$IFDEF MSWINDOWS}

procedure TGLSceneViewer.CreateParams(var Params: TCreateParams);
begin
  inherited CreateParams(Params);
  with Params do
  begin
    Style := Style or WS_CLIPCHILDREN or WS_CLIPSIBLINGS;
    WindowClass.Style := WindowClass.Style or CS_OWNDC;
  end;
end;

{$ENDIF}

procedure TGLSceneViewer.CreateWnd;
begin
  inherited CreateWnd;
  // initialize and activate the OpenGL rendering context
  // need to do this only once per window creation as we have a private DC
  FBuffer.Resize(0, 0, Self.Width, Self.Height);
  FOwnDC := GetDC(Handle);
  FBuffer.CreateRC(FOwnDC, False);
end;

procedure TGLSceneViewer.DestroyWnd;
begin
  FBuffer.DestroyRC;
  if FOwnDC <> 0 then
  begin
    ReleaseDC(Handle, FOwnDC);
    FOwnDC := 0;
  end;
  inherited;
end;

procedure TGLSceneViewer.LMEraseBkgnd(var Message: TLMEraseBkgnd);
begin
  if IsRenderingContextAvailable then
    Message.Result := 1
  else
    inherited;
end;

procedure TGLSceneViewer.LMSize(var Message: TLMSize);
begin
  inherited;
  FBuffer.Resize(0, 0, Message.Width, Message.Height);
end;

procedure TGLSceneViewer.LMPaint(var Message: TLMPaint);
var
  PS: LCLType.TPaintStruct;
  p: TPoint;
begin
  p := ClientToScreen(Point(0, 0));
  if (FLastScreenPos.X <> p.X) or (FLastScreenPos.Y <> p.Y) then
  begin
    // Workaround for MS OpenGL "black borders" bug
    if FBuffer.RCInstantiated then
      PostMessage(Handle, WM_SIZE, SIZE_RESTORED,
        Width + (Height shl 16));
    FLastScreenPos := p;
  end;
  BeginPaint(Handle, PS);
  try
    if IsRenderingContextAvailable and (Width > 0) and (Height > 0) then
      FBuffer.Render;
  finally
    EndPaint(Handle, PS);
    Message.Result := 0;
  end;
end;

procedure TGLSceneViewer.LMDestroy(var Message: TLMDestroy);
begin
  FBuffer.DestroyRC;
  if FOwnDC <> 0 then
  begin
    ReleaseDC(Handle, FOwnDC);
    FOwnDC := 0;
  end;
  inherited;
end;

procedure TGLSceneViewer.CMMouseEnter(var msg: TMessage);
begin
  inherited;
  FMouseInControl := True;
  if Assigned(FOnMouseEnter) then
    FOnMouseEnter(Self);
end;

procedure TGLSceneViewer.CMMouseLeave(var msg: TMessage);
begin
  inherited;
  FMouseInControl := False;
  if Assigned(FOnMouseLeave) then
    FOnMouseLeave(Self);
end;

procedure TGLSceneViewer.Loaded;
begin
  inherited Loaded;
  // initiate window creation
  {$ifndef LCLGTK2}
  HandleNeeded;
  {$endif}
end;

procedure TGLSceneViewer.DoBeforeRender(Sender: TObject);
begin
  SetupVSync(VSync);
end;

procedure TGLSceneViewer.DoBufferChange(Sender: TObject);
begin
  if (not Buffer.Rendering) and (not Buffer.Freezed) then
    Invalidate;
end;

procedure TGLSceneViewer.DoBufferStructuralChange(Sender: TObject);
begin
  DestroyWnd;
  CreateWnd;
end;

procedure TGLSceneViewer.Render;
begin
  Buffer.Render;
end;

function TGLSceneViewer.LastFrameTime: single;
begin
  Result := FBuffer.LastFrameTime;
end;

function TGLSceneViewer.FramesPerSecond: single;
begin
  Result := FBuffer.FramesPerSecond;
end;

function TGLSceneViewer.FramesPerSecondText(decimals: integer = 1): string;
begin
  Result := Format('%.*f FPS', [decimals, FBuffer.FramesPerSecond]);
end;

procedure TGLSceneViewer.ResetPerformanceMonitor;
begin
  FBuffer.ResetPerformanceMonitor;
end;

function TGLSceneViewer.CreateSnapShotBitmap: TBitmap;
begin
{$IFDEF MSWINDOWS}
  Result := TBitmap.Create;
  Result.PixelFormat := pf24bit;
  Result.Width := Width;
  Result.Height := Height;

  BitBlt(Result.Canvas.Handle, 0, 0, Width, Height,
    RenderDC, 0, 0, SRCCOPY);
{$ELSE}
  Result := nil;
{$ENDIF}
end;

function TGLSceneViewer.GetFieldOfView: single;
begin
  if not Assigned(Camera) then
    Result := 0

  else if Width < Height then
    Result := Camera.GetFieldOfView(Width)

  else
    Result := Camera.GetFieldOfView(Height);
end;

function TGLSceneViewer.GetIsRenderingContextAvailable: boolean;
begin
  Result := FBuffer.RCInstantiated and FBuffer.RenderingContext.IsValid;
end;

procedure TGLSceneViewer.SetFieldOfView(const Value: single);
begin
  if Assigned(Camera) then
  begin
    if Width < Height then
      Camera.SetFieldOfView(Value, Width)

    else
      Camera.SetFieldOfView(Value, Height);
  end;
end;


// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
initialization
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  RegisterClass(TGLSceneViewer);
{$IF DEFINED(LCLwin32) or DEFINED(LCLwin64)}
  GLRegisterWSComponent(TGLSceneViewer);
{$ENDIF}

end.

