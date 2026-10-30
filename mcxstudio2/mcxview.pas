{ mcxstudio2 - the 3-D preview.

  What the window shows of a simulation before it is run: the domain box, a
  floor grid, the axes, every shape in the Shapes list and where the source
  sits.  It reads the document directly rather than the form, so what is drawn
  is what mcx would be given, whether the value was typed or came from a file.

  A class attached to a panel rather than a form of its own.  The plan called
  for a second form, from when the preview was to be a separate window; it is
  a page of the notebook now, and a form inside a page is a form whose Align,
  focus and close button all have to be argued with.

  Drawing lives in mcxgl; this file knows what a simulation looks like, and
  nothing in mcxgl knows what one is. }
unit mcxview;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, Controls, ExtCtrls, Graphics, Forms, fpjson, jsonparser,
  FPImage, FPWritePNG, IntfGraphics, GraphType, LCLType,
  LazFileUtils, OpenGLContext, GL, mcxdoc, mcxgl, mcxjd, mcxmesh, mcxsoft,
  mcxglprobe;

var
  { --software: draw on the processor from the start, never asking for an
    OpenGL context -- for a display where asking is what goes wrong. }
  McxForceSoftware: Boolean = False;

{ True if E is the OpenGL control failing to get a context, which the view
  has now answered by drawing in software.  For a handler that would
  otherwise stop the program over it -- the headless reporter of --quit. }
function McxViewTakesException(E: Exception): Boolean;

type
  { Somewhere to put a line that has nowhere else to go -- the GL version at
    startup, a shader that would not compile.  The view has no log of its
    own and should not grow one. }
  TMcxViewLog = procedure(Sender: TObject; const AText: string) of object;

  { What a click landed on: which command in the Shapes array, what it is
    called there, and the medium it assigns. }
  TMcxPickInfo = record
    Index: Integer;
    Verb: string;
    Tag: Integer;
  end;

  TMcxPickEvent = procedure(Sender: TObject; const AInfo: TMcxPickInfo) of object;

  TMcxView = class
  private
    FHost: TWinControl;
    FDoc: TMcxDoc;
    FGL: TOpenGLControl;
    { The software renderer, when there is no OpenGL 3.3 to draw with: the
      pane it paints, the raster it paints from, and the volume it marches,
      which the GL path keeps on the card and this one keeps here. }
    FSoftPane: TCustomControl;
    FSoft: Boolean;
    FSoftWhy: string;
    FRaster: TMcxSoftRaster;
    FSoftVol: TMcxSoftVolume;
    FVolNx, FVolNy, FVolNz: Integer;
    { The domain volume on the card, if the document carries one -- a JData
      array in Shapes, or a VolumeFile the view can read -- and what it was
      read from, so an edit elsewhere does not decode it again. }
    FDomainKey: string;
    FDomainShown: Boolean;
    FInDomain: Boolean;
    { The last result put on the card, kept so that the display bar can
      switch between it and the domain it was computed in without reading
      the file again.  A reference, not a copy: nothing writes into an array
      once it has been read. }
    FResult: TMcxArray;
    FHasResult: Boolean;
    FShowInput: Boolean;
    { Set while the switch puts a volume back, so that the camera stays where
      it was rather than being aimed again. }
    FSwitching: Boolean;
    { A VolumeFile that is itself a .json of Shapes -- how the quick-test and
      skin-vessel examples build their domains -- read once per change. }
    FShapeFileKey: string;
    FShapeFile: TJSONData;
    FVolumeChanged: Boolean;
    FShader: TMcxShader;
    FVolShader: TMcxShader;
    FSolidShader: TMcxShader;
    FLines: TMcxLines;
    FMesh: TMcxMesh;
    { The same geometry again, coloured by identifier instead of by medium.
      A second mesh rather than a second colour channel: it is built once per
      edit and read once per click, and one buffer per purpose is easier to
      be sure of than one buffer meaning two things. }
    FPick: TMcxMesh;
    FTarget: TMcxTarget;
    FPicks: array of TMcxPickInfo;
    FSelected: Integer;
    { The mmc mesh, and what it was loaded from.  Cached on the pair, because
      pulling the surface out of a head mesh is the better part of a second
      and the scene is rebuilt on every keystroke. }
    FMeshFaces: TMcxFaces;
    FMeshKey: string;
    FMeshLo, FMeshHi: TMcxNode;
    { Photon paths, in their own buffer: they are loaded separately from the
      scene and would otherwise be rebuilt with it on every keystroke. }
    { The three axis letters, in their own batch: they are rebuilt every
      frame to face the camera, and the rest of the wireframe is not. }
    FAxisText: TMcxLines;
    FAxisLo, FAxisHi: TMcxVec3;
    FAxisStep: array[0..2] of Single;
    FTraj: TMcxLines;
    FTrajCount: Integer;
    { The photon each segment belongs to, in buffer order -- which is photon
      order, so this is non-decreasing and a range of photons is a contiguous
      run that can be found by bisection. }
    FTrajIds: array of Integer;
    FTrajFirst, FTrajLast: Integer;
    FIdLo, FIdHi: Integer;
    { How big the scene is, so the source glyph can be sized against it. }
    FSceneSpan: Single;
    { The corners of whatever the scene was built around -- zero to Dim for a
      grid, the mesh's own extent for a mesh.  Kept so that a fit has
      something to fit to without working it out again. }
    FSceneLo, FSceneHi: TMcxVec3;
    FVolume: TMcxVolume;
    FCube: TMcxCube;
    { What the volume is drawn as and through.  All uniforms: changing any of
      them is a repaint, not an upload. }
    FStyle: Integer;
    FMap: Integer;
    FFloor: Single;
    FOpacity: Single;
    FSteps: Single;
    FLogScale: Boolean;
    FClipLo, FClipHi: TMcxVec3;
    { The window on both scales, worked out when the volume is loaded, so the
      Log scale box can be ticked either way afterwards.  A volume with
      nothing above zero, or of labels, has no log window. }
    FVolLow, FVolHigh: Single;
    FLogLow, FLogHigh: Single;
    FCanLog: Boolean;
    FCamera: TMcxCamera;
    { Latches, so a driver that cannot give us a core profile is reported
      once rather than on every repaint. }
    FReady: Boolean;
    FFailed: Boolean;
    { Filled on the first paint; the About box may ask before there is one. }
    FDescription: string;
    { What the picture is cleared to.  A theme colour, so the view is not the
      one light rectangle in a dark window or the reverse. }
    FBack: TMcxVec3;
    FDragging: Boolean;
    { Set for one frame after a zoom, the way FDragging is set for the length
      of a rotate: both mean "another one of these is coming". }
    FCoarse: Boolean;
    FDidDrag: Boolean;
    FDragX, FDragY: Integer;
    FSceneKey: string;
    FOnLog: TMcxViewLog;
    FOnPick: TMcxPickEvent;
    procedure Say(const AText: string);
    procedure Build;
    { Moves to the software renderer, saying why, once. }
    procedure UseSoftware(const AWhy: string);
    { The pane that is showing, whichever renderer draws it. }
    function  Pane: TControl;
    procedure Invalidate;
    function  PaneWidth: Integer;
    function  PaneHeight: Integer;
    procedure RenderSoft(AWidth, AHeight: Integer; ADraft: Boolean);
    procedure SoftPaint(Sender: TObject);
    procedure AppException(Sender: TObject; E: Exception);
    { A current GL context with the 3.3 programs built, or False having moved
      to software -- including when asking for the context raises, which is
      how the widget set reports that there is none to be had. }
    function  TryGL: Boolean;
    function  Start: Boolean;
    procedure GLPaint(Sender: TObject);
    procedure GLMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure GLMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
    procedure GLMouseUp(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure GLMouseWheel(Sender: TObject; Shift: TShiftState;
      WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
    procedure AddAxes(const ALo, AHi: TMcxVec3);
    procedure BuildAxisLabels;
    procedure AddShapes;
    procedure AddDomainVolume;
    function  ShapeFileShapes: TJSONData;
    procedure AddMesh;
    procedure AddSource;
    procedure RenderScene(AWidth, AHeight: Integer; AFlat: Boolean);
    function  PickAt(AX, AY: Integer): Integer;
    function  GetHasVolume: Boolean;
    procedure SetShowInput(AValue: Boolean);
    { The Log scale box, where the volume has a log window to draw it in. }
    function  UseLog: Boolean;
    procedure DrawVolume(const AMVP: TMcxMat4);
    procedure DrawTrajectory;
    function  FirstSegmentOf(AId: Integer): Integer;
  public
    { AHost is a panel the designer placed; the GL control is created into it.
      See Build for why it is not placed there directly. }
    constructor Create(AHost: TWinControl);
    destructor Destroy; override;
    { Draws ADoc.  Cheap enough to call on every edit: it is a few hundred
      line segments in one buffer. }
    procedure Rebuild;
    { Aims the camera so that what is on the card fills the pane. }
    procedure FitView;
    { Highlights one Shapes command, by its index in the array. }
    procedure SelectShape(AIndex: Integer);
    property SelectedShape: Integer read FSelected;
    { Puts a result on the card.  AArray is whatever mcxjd read out of a
      .jnii or .bnii; anything past the third dimension -- time gates, or
      the several outputs of one run -- is dropped to the first slice, which
      is what the old viewer showed too. }
    { ALabels is a domain rather than a result: tissue labels or optical
      properties, drawn on a linear scale from zero, so that a volume of one
      label throughout is a solid block rather than nothing at all. }
    function ShowVolume(const AArray: TMcxArray;
      ALabels: Boolean = False): Boolean;
    procedure ClearVolume;
    { A run's output is on the card, or is kept off it by ShowInput: a volume
      or a set of photon paths. }
    function  HasOutput: Boolean;
    { The input domain -- the shapes, or the label volume -- in place of the
      output, which is kept to switch back to.  A new result turns it off. }
    property ShowInput: Boolean read FShowInput write SetShowInput;
    { Drops the kept output, for a document that is not the one it came from. }
    procedure ForgetOutput;
    { 0 for maximum intensity, 1 for accumulation. }
    { Everything below is a uniform: setting one is a repaint, not an upload,
      which is why the display controls can be dragged live. }
    property Style: Integer read FStyle write FStyle;
    property LogScale: Boolean read FLogScale write FLogScale;
    { 0 jet, 1 hot, 2 viridis, 3 cool, 4 grey -- see the shader's ramp(). }
    property Colormap: Integer read FMap write FMap;
    { Fraction of the displayed range below which a voxel is not drawn. }
    property Threshold: Single read FFloor write FFloor;
    property Opacity: Single read FOpacity write FOpacity;
    { The visible slab, each component 0..1 across that axis of the volume. }
    property ClipLo: TMcxVec3 read FClipLo write FClipLo;
    property ClipHi: TMcxVec3 read FClipHi write FClipHi;
    { Asks for a repaint without rebuilding the scene. }
    procedure Redraw;
    { What the driver calls itself, once there has been a context to ask. }
    property Description: string read FDescription;
    { The clear colour, 0..1 per channel. }
    property Background: TMcxVec3 read FBack write FBack;
    property HasVolume: Boolean read GetHasVolume;
    { False until the GL context has started, on the first paint, unless
      the view draws in software; a volume cannot be shown before then. }
    function Ready: Boolean;
    { Set when Rebuild put a domain volume up or took one down; the owner
      clears it once its own controls have followed. }
    property VolumeChanged: Boolean read FVolumeChanged write FVolumeChanged;
    property Document: TMcxDoc read FDoc write FDoc;
    { Renders at any size into an offscreen target and writes a PNG. }
    { Loads the photon paths mcx writes with -D M, as <session>_traj.jdt. }
    function ShowTrajectory(const AFileName: string): Boolean;
    procedure ClearTrajectory;
    { Which photons to draw, by the identifier mcx gave them.  Drawing one
      path is what makes a path legible: half a million segments on top of
      each other is a cloud, and a cloud has no direction in it. }
    procedure SetPhotonRange(ALo, AHi: Integer);
    { The identifiers actually present, so the controls can be scaled to the
      file rather than to a guess. }
    property PhotonFirst: Integer read FTrajFirst;
    property PhotonLast: Integer read FTrajLast;
    property PhotonCount: Integer read FTrajCount;
    function SaveImage(const AFileName: string; AWidth, AHeight: Integer): Boolean;
    property OnLog: TMcxViewLog read FOnLog write FOnLog;
    property OnPick: TMcxPickEvent read FOnPick write FOnPick;
  end;

implementation

var
  GView: TMcxView = nil;

{ The widget set raises this from inside the GL control's handle creation --
  gtk2's "gdk_gl_context_share_new_usefpglx context creation failed", the
  others' own words for the same thing -- in the middle of the event loop,
  where nothing of ours is on the stack to catch it. }
function IsContextFailure(E: Exception): Boolean;
var
  M: string;
begin
  M := LowerCase(E.Message);
  Result := (Pos('context creation failed', M) > 0) or
            (Pos('opengl', M) > 0) or (Pos('glx', M) > 0) or
            (Pos('wglcreatecontext', M) > 0) or (Pos('pixel format', M) > 0) or
            (Pos('nsopengl', M) > 0);
end;

function McxViewTakesException(E: Exception): Boolean;
begin
  Result := (GView <> nil) and (GView.FGL <> nil) and IsContextFailure(E);
  if Result then GView.UseSoftware('the OpenGL context could not be created: ' + E.Message);
end;

type
  { The pane the software renderer paints: a plain double-buffered control
    whose Paint is the view's.  Declared here so that the view, in the same
    unit, can hand it the mouse events the GL control had. }
  TMcxSoftPane = class(TCustomControl)
  protected
    procedure Paint; override;
  public
    OnPaintPane: TNotifyEvent;
  end;

procedure TMcxSoftPane.Paint;
begin
  if Assigned(OnPaintPane) then OnPaintPane(Self) else inherited Paint;
end;

constructor TMcxView.Create(AHost: TWinControl);
begin
  FHost := AHost;
  FDescription := '(no context yet)';
  FBack := McxVec3(0.16, 0.16, 0.17);
  FStyle := 0;
  FMap := 0;
  FFloor := 0;
  FOpacity := 0.25;
  FSteps := 192;
  FLogScale := True;
  FClipLo := McxVec3(0, 0, 0);
  FClipHi := McxVec3(1, 1, 1);
  Build;
end;

destructor TMcxView.Destroy;
begin
  if GView = Self then GView := nil;
  Application.RemoveOnExceptionHandler(@AppException);
  FRaster.Free;
  FShapeFile.Free;
  FVolume.Free;
  FCube.Free;
  FMesh.Free;
  FPick.Free;
  FTraj.Free;
  FAxisText.Free;
  FTarget.Free;
  FSolidShader.Free;
  FVolShader.Free;
  FShader.Free;
  FLines.Free;
  FCamera.Free;
  inherited Destroy;
end;

{ ------------------------------------------------------------ software --- }

procedure TMcxView.UseSoftware(const AWhy: string);
var
  P: TMcxSoftPane;
begin
  if FSoft then Exit;
  FSoft := True;
  FSoftWhy := AWhy;
  FDescription := Format('software renderer on %d processor(s) (%s)',
    [McxSoftThreads, AWhy]);
  Say('drawing in software: ' + AWhy);

  { The GL control stays, hidden and deaf: this may be running inside its
    own paint handler, which is no place to free it. }
  if FGL <> nil then
  begin
    FGL.OnPaint := nil;
    FGL.OnMouseDown := nil;
    FGL.OnMouseMove := nil;
    FGL.OnMouseUp := nil;
    FGL.OnMouseWheel := nil;
    FGL.Visible := False;
  end;

  P := TMcxSoftPane.Create(FHost);
  P.Parent := FHost;
  P.Align := alClient;
  P.DoubleBuffered := True;
  P.OnPaintPane := @SoftPaint;
  P.OnMouseDown := @GLMouseDown;
  P.OnMouseMove := @GLMouseMove;
  P.OnMouseUp := @GLMouseUp;
  P.OnMouseWheel := @GLMouseWheel;
  FSoftPane := P;

  { What Start does for the GL path once it has a context. }
  if FDoc <> nil then Rebuild;
  Invalidate;
end;

function TMcxView.Pane: TControl;
begin
  if FSoft then Result := FSoftPane else Result := FGL;
end;

procedure TMcxView.Invalidate;
begin
  if Pane <> nil then Pane.Invalidate;
end;

function TMcxView.PaneWidth: Integer;
begin
  if Pane <> nil then Result := Max(1, Pane.Width) else Result := 1;
end;

function TMcxView.PaneHeight: Integer;
begin
  if Pane <> nil then Result := Max(1, Pane.Height) else Result := 1;
end;

{ The software renderer's RenderScene: the same projection, the same three
  passes in the same order, into the raster rather than into a framebuffer. }
procedure TMcxView.RenderSoft(AWidth, AHeight: Integer; ADraft: Boolean);
var
  MVP: TMcxMat4;
  First, Last: Integer;
begin
  if FRaster = nil then FRaster := TMcxSoftRaster.Create;
  if AHeight < 1 then AHeight := 1;
  if AWidth < 1 then AWidth := 1;
  MVP := McxMat4Mul(
    McxMat4Perspective(45, AWidth / AHeight, FCamera.Distance * 0.01,
      FCamera.Distance * 10),
    FCamera.View);
  FRaster.Start(AWidth, AHeight, FBack, MVP);

  { far faces, then near ones, as the two culled draws of RenderScene }
  FRaster.Solids(FMesh.Raw, FMesh.Count,
    McxVec3Norm(McxVec3Sub(FCamera.Eye, FCamera.Target)), 1);
  FRaster.Solids(FMesh.Raw, FMesh.Count,
    McxVec3Norm(McxVec3Sub(FCamera.Eye, FCamera.Target)), -1);

  FRaster.Lines(FLines.Raw, 0, FLines.Count);

  if (FTrajCount > 0) and not FShowInput then
  begin
    if (FIdLo <= FTrajFirst) and (FIdHi >= FTrajLast) then
      FRaster.Lines(FTraj.Raw, 0, FTraj.Count)
    else
    begin
      First := FirstSegmentOf(FIdLo);
      Last := FirstSegmentOf(FIdHi + 1);
      FRaster.Lines(FTraj.Raw, First * 2, (Last - First) * 2);
    end;
  end;

  BuildAxisLabels;
  FRaster.Lines(FAxisText.Raw, 0, FAxisText.Count);

  if GetHasVolume then
  begin
    FSoftVol.LogScale := UseLog;
    if UseLog then
    begin
      FSoftVol.Low := FLogLow;
      FSoftVol.High := FLogHigh;
    end
    else
    begin
      FSoftVol.Low := FVolLow;
      FSoftVol.High := FVolHigh;
    end;
    FSoftVol.Style := FStyle;
    FSoftVol.Map := FMap;
    FSoftVol.Floor := FFloor;
    FSoftVol.Opacity := FOpacity;
    FSoftVol.Steps := FSteps;
    FSoftVol.ClipLo := FClipLo;
    FSoftVol.ClipHi := FClipHi;
    FRaster.Volume(FSoftVol, FCamera.Eye, ADraft);
  end;
end;

procedure TMcxView.SoftPaint(Sender: TObject);
var
  Bmp: TBitmap;
  Img: TLazIntfImage;
  W, H, x, y: Integer;
  C: TFPColor;
  r8, g8, b8: Byte;
begin
  W := Max(1, FSoftPane.ClientWidth);
  H := Max(1, FSoftPane.ClientHeight);
  { Fewer rays and fewer steps while the picture moves, as the GL path. }
  RenderSoft(W, H, FDragging or FCoarse);

  Bmp := TBitmap.Create;
  try
    Bmp.PixelFormat := pf24bit;
    Bmp.SetSize(W, H);
    Img := Bmp.CreateIntfImage;
    try
      C.Alpha := alphaOpaque;
      for y := 0 to H - 1 do
        for x := 0 to W - 1 do
        begin
          FRaster.Pixel(x, y, r8, g8, b8);
          C.Red := r8 * 257;
          C.Green := g8 * 257;
          C.Blue := b8 * 257;
          Img.Colors[x, y] := C;
        end;
      Bmp.LoadFromIntfImage(Img);
    finally
      Img.Free;
    end;
    FSoftPane.Canvas.Draw(0, 0, Bmp);
  finally
    Bmp.Free;
  end;

  if FCoarse then
  begin
    FCoarse := False;
    Invalidate;
  end;
end;

function TMcxView.TryGL: Boolean;
begin
  Result := False;
  if FSoft then Exit;
  { No handle yet is not no OpenGL: the control has not been shown, and the
    GL path has always waited for its first paint rather than given up.  One
    that is showing but has no window yet -- its page was only just brought
    up, which on Windows creates the window later -- is given it now. }
  if (FGL <> nil) and not FGL.HandleAllocated and FGL.IsVisible then
    FGL.HandleNeeded;
  if (FGL = nil) or not FGL.HandleAllocated then Exit;
  try
    if not FGL.MakeCurrent then
    begin
      UseSoftware('no OpenGL context could be made current');
      Exit;
    end;
    if not Start then
    begin
      UseSoftware('OpenGL 3.3 is not available: ' + McxGLDescribe);
      Exit;
    end;
    Result := True;
  except
    on E: Exception do
      UseSoftware('the OpenGL context could not be created: ' + E.Message);
  end;
end;

{ Every unhandled exception comes here once a handler is registered, since
  the LCL then shows none itself: an OpenGL failure moves the view to
  software, and anything else is shown exactly as the LCL would have. }
procedure TMcxView.AppException(Sender: TObject; E: Exception);
begin
  if McxViewTakesException(E) then Exit;
  { The headless reporter of --quit has already been told, and stopped. }
  if Assigned(Application.OnException) then Exit;
  Application.ShowException(E);
end;

procedure TMcxView.Say(const AText: string);
begin
  if Assigned(FOnLog) then FOnLog(Self, AText);
end;


{ The OpenGL control is created here rather than placed in the designer.

  Either works -- lazbuild compiles LazOpenGLContext from the project's
  RequiredPackages whether or not the IDE has it installed -- but a
  designer-placed TOpenGLControl makes mcxmain.lfm unopenable for anyone who
  has not installed that package into their Lazarus, and the .lfm staying
  editable by anyone is the point of building the form in the designer at all.
  So the designer holds an ordinary TPanel and the control goes into it here,
  which is what MRIcroGL does for the same reason. }
procedure TMcxView.Build;
var
  Why: string;
begin
  FCamera := TMcxCamera.Create;
  FLines := TMcxLines.Create;
  FMesh := TMcxMesh.Create;
  FPick := TMcxMesh.Create;
  FTraj := TMcxLines.Create;
  FAxisText := TMcxLines.Create;
  FSelected := -1;

  if McxForceSoftware then
  begin
    UseSoftware('asked for with --software');
    Exit;
  end;

  { Asked outside any widget first: where there is no 3.3 context to be had,
    finding out inside the GL control is a modal error from the middle of
    the widget set's own code. }
  if not McxProbeGL(Why) then
  begin
    UseSoftware('OpenGL 3.3 is not available: ' + Why);
    Exit;
  end;

  { Owned by the host panel, so the control goes when the form does and
    this class does not have to be a TComponent to own it. }
  try
    FGL := TOpenGLControl.Create(FHost);
  except
    on E: Exception do
    begin
      FGL := nil;
      UseSoftware('no OpenGL control: ' + E.Message);
      Exit;
    end;
  end;
  GView := Self;
  Application.AddOnExceptionHandler(@AppException, True);
  FGL.Parent := FHost;
  FGL.Align := alClient;
  FGL.DepthBits := 24;
  FGL.MultiSampling := 4;
  { Ask for 3.3 core.  On Cocoa this is honoured strictly; on GLX it is a
    request the driver may answer with more, which is why what arrived is
    logged rather than assumed. }
  FGL.OpenGLMajorVersion := 3;
  FGL.OpenGLMinorVersion := 3;
  FGL.OnPaint := @GLPaint;
  FGL.OnMouseDown := @GLMouseDown;
  FGL.OnMouseMove := @GLMouseMove;
  FGL.OnMouseUp := @GLMouseUp;
  FGL.OnMouseWheel := @GLMouseWheel;
end;

{ First paint: the context only exists once the control has been realised, so
  everything that needs one waits until here. }
function TMcxView.Start: Boolean;
begin
  Result := FReady;
  if FSoft then Exit(False);
  if FReady or FFailed then Exit;

  if not McxGLLoad then
  begin
    FFailed := True;
    Say('OpenGL 3.3 is not available: ' + McxGLDescribe);
    Exit(False);
  end;

  FShader := TMcxShader.Create;
  if not FShader.Build(McxLineVertexShader, McxLineFragmentShader) then
  begin
    FFailed := True;
    Say('shader: ' + FShader.Error);
    FreeAndNil(FShader);
    Exit(False);
  end;

  FSolidShader := TMcxShader.Create;
  if not FSolidShader.Build(McxSolidVertexShader, McxSolidFragmentShader) then
  begin
    Say('solid shader: ' + FSolidShader.Error);
    FreeAndNil(FSolidShader);
  end;

  { A second program for the raycaster.  If it will not build the wireframe
    still works, which is worth more than refusing to draw anything. }
  FVolShader := TMcxShader.Create;
  if not FVolShader.Build(McxVolumeVertexShader, McxVolumeFragmentShader) then
  begin
    Say('volume shader: ' + FVolShader.Error);
    FreeAndNil(FVolShader);
  end;

  FDescription := McxGLDescribe;
  Say('OpenGL: ' + FDescription);
  FReady := True;
  Result := True;
  Rebuild;
end;

function TMcxView.Ready: Boolean;
begin
  Result := FReady or FSoft;
end;

function TMcxView.GetHasVolume: Boolean;
begin
  if FSoft then Result := Length(FSoftVol.Data) > 0
  else Result := (FVolume <> nil) and FVolume.Loaded;
end;

function TMcxView.UseLog: Boolean;
begin
  Result := FLogScale and FCanLog;
end;

procedure TMcxView.ClearVolume;
begin
  FreeAndNil(FVolume);
  SetLength(FSoftVol.Data, 0);
  FVolNx := 0; FVolNy := 0; FVolNz := 0;
  Invalidate;
end;

function TMcxView.ShowVolume(const AArray: TMcxArray;
  ALabels: Boolean): Boolean;
var
  nx, ny, nz: Integer;
  i, n, Spoiled: Int64;
  Buf: array of Single;
  V, Lo, Hi, Pos: Double;
  Any: Boolean;
begin
  Result := False;
  { A result opened over the domain replaces it, and is not the domain's to
    take down again when the document changes. }
  if not FInDomain then FDomainShown := False;
  { A context that cannot be had now will not be had at the next paint
    either, so the result is drawn in software rather than not at all. }
  if not TryGL and not FSoft then Exit;

  nx := 1; ny := 1; nz := 1;
  if Length(AArray.Dims) > 0 then nx := AArray.Dims[0];
  if Length(AArray.Dims) > 1 then ny := AArray.Dims[1];
  if Length(AArray.Dims) > 2 then nz := AArray.Dims[2];
  if (nx < 1) or (ny < 1) or (nz < 1) then Exit;

  n := Int64(nx) * ny * nz;
  if n > McxArrayCount(AArray) then Exit;

  { Converted to float once, here, whatever it was stored as: the texture is
    GL_R32F and the shader should not have to know that a fluence map is
    single and a label volume is uint8. }
  SetLength(Buf, n);
  Lo := 0;
  Hi := 0;
  Pos := 0;
  Any := False;
  Spoiled := 0;
  for i := 0 to n - 1 do
  begin
    V := McxArrayValue(AArray, i);
    Buf[i] := V;
    { A voxel that is not a number is not a value to scale against, and it
      cannot be compared either: NaN is unordered, so "V < Lo" is not false,
      it is an invalid operation, and FPC leaves that unmasked on x86_64.
      Opening a result that diverged used to raise EInvalidOp here. }
    if IsNan(V) or IsInfinite(V) then
    begin
      Inc(Spoiled);
      Continue;
    end;
    if not Any then
    begin
      Lo := V;
      Hi := V;
      Any := True;
    end
    else
    begin
      if V < Lo then Lo := V;
      if V > Hi then Hi := V;
    end;
    { the smallest value above zero, where the log window starts }
    if (V > 0) and ((Pos = 0) or (V < Pos)) then Pos := V;
  end;

  { Nothing to draw and nothing to scale it by.  Worth naming rather than
    showing an empty box: a result that is NaN throughout is a simulation that
    diverged -- an optical property that is not a number, a refractive index
    of zero, a time gate of no width -- and the file is the only place that
    shows.  mcx's own bin/example_session.jnii is one. }
  if not Any then
  begin
    Say(Format('every one of the %d values in this result is not a number -- '
      + 'the run that wrote it did not converge', [n]));
    Exit;
  end;

  { The ones that are not numbers are held down to the floor of the window,
    and to zero if that is higher, which is where both scales drop them: an
    unknown voxel should read as empty rather than as the brightest thing in
    the picture, which is what a NaN interpolated through GL_LINEAR does to
    the eight texels around it.  Zero alone would do for a fluence map and not
    for a Jacobian, whose minimum is negative -- so it is the floor itself,
    found above, and a second pass over the floats rather than a guess during
    the first. }
  if Spoiled > 0 then
  begin
    for i := 0 to n - 1 do
      if IsNan(Buf[i]) or IsInfinite(Buf[i]) then Buf[i] := Min(Lo, 0);
    Say(Format('%d of %d values are not numbers, and are drawn as empty',
      [Spoiled, n]));
  end;

  { Fluence spans many decades, so the log window runs from the smallest
    value above zero to the largest, and the shader takes the log too.  The
    whole of it rather than a fixed number of decades: the far tail of a
    fluence map is what shows the box it is in, and the Threshold slider is
    there to take it off again.  Labels are only ever linear, from zero, so a
    volume of one label throughout is a solid block. }
  if ALabels and (Lo > 0) then Lo := 0;
  FCanLog := (not ALabels) and (Pos > 0);
  if FCanLog then
  begin
    FLogHigh := Ln(Hi);
    FLogLow := Ln(Pos);
    { a flat volume, drawn at the top of its window rather than not at all }
    if FLogHigh <= FLogLow then FLogLow := FLogHigh - 1;
  end;
  FVolLow := Lo;
  FVolHigh := Hi;
  if FVolHigh <= FVolLow then FVolHigh := FVolLow + 1;

  FVolNx := nx; FVolNy := ny; FVolNz := nz;
  if FSoft then
  begin
    { Kept here and marched on the processor at every paint. }
    FSoftVol.Data := Buf;
    FSoftVol.Nx := nx; FSoftVol.Ny := ny; FSoftVol.Nz := nz;
    Result := True;
  end
  else
  begin
    if FVolume = nil then FVolume := TMcxVolume.Create;
    Result := FVolume.Upload(@Buf[0], nx, ny, nz, FVolLow, FVolHigh);
    if not Result then
    begin
      Say(Format('the card would not take a %dx%dx%d volume', [nx, ny, nz]));
      FreeAndNil(FVolume);
      FVolNx := 0; FVolNy := 0; FVolNz := 0;
    end;
  end;

  { The camera is aimed at the result rather than left wherever the document
    put it.  A result is drawn in its own voxels -- the shader marches a unit
    cube scaled by nx, ny, nz -- and those need not be Domain.Dim at all: a
    181-cube head opened over a blank 60-cube document was drawn a third off
    the side of the pane, and nothing had asked the camera to look at it.

    This is also the only place a fit belongs.  Rebuild re-aims only when the
    domain changes size, deliberately, so that typing a photon count does not
    throw the view away -- and opening a result is exactly the event that
    should. }
  if Result then
  begin
    FSceneKey := Format('volume %d %d %d', [nx, ny, nz]);
    if not FSwitching then FitView;
    if not (FInDomain or FSwitching) then
    begin
      FResult := AArray;
      FHasResult := True;
      FShowInput := False;
    end;
  end;

  Invalidate;
end;

function TMcxView.HasOutput: Boolean;
begin
  Result := FHasResult or (FTrajCount > 0);
end;

procedure TMcxView.ForgetOutput;
begin
  FResult.Dims := nil;
  FResult.Data := nil;
  FHasResult := False;
  FShowInput := False;
end;

procedure TMcxView.SetShowInput(AValue: Boolean);
begin
  if AValue = FShowInput then Exit;
  FShowInput := AValue;
  FSwitching := True;
  try
    if AValue then
    begin
      { The output off the card, and the domain's own volume back on it if
        the document has one: a key that matches nothing is what has
        AddDomainVolume read it again. }
      if not FDomainShown then ClearVolume;
      FDomainKey := #0;
      AddDomainVolume;
    end
    else if FHasResult then
      ShowVolume(FResult);
  finally
    FSwitching := False;
  end;
  Invalidate;
end;

{ Puts whatever is on the card inside the pane: the volume if there is one,
  otherwise the domain the scene was built around. }
procedure TMcxView.FitView;
var
  Aspect, Pad: Single;
  Lo, Hi: TMcxVec3;
begin
  Aspect := 1;
  if PaneHeight > 0 then Aspect := PaneWidth / PaneHeight;

  Lo := FSceneLo;
  Hi := FSceneHi;

  { A result is drawn in its own voxels and the axes around the domain's, and
    the two need not agree -- a 181-cube head over a blank 60-cube document
    is the case that showed it -- so what has to fit is both. }
  if GetHasVolume then
  begin
    if 0 < Lo.x then Lo.x := 0;
    if 0 < Lo.y then Lo.y := 0;
    if 0 < Lo.z then Lo.z := 0;
    if FVolNx > Hi.x then Hi.x := FVolNx;
    if FVolNy > Hi.y then Hi.y := FVolNy;
    if FVolNz > Hi.z then Hi.z := FVolNz;
  end;

  { Room for the ruler.  The numbers hang three ticks outside the low corner
    and the letters a little past the high one, and a tick is 0.025 of the
    span -- so fitting the box alone puts the graduations off the edge of the
    pane, which is a fit that loses the thing you fitted it to read. }
  Pad := Max(Hi.x - Lo.x, Max(Hi.y - Lo.y, Hi.z - Lo.z)) * 0.1;
  Lo := McxVec3(Lo.x - Pad, Lo.y - Pad, Lo.z - Pad);
  Hi := McxVec3(Hi.x + Pad, Hi.y + Pad, Hi.z + Pad);

  FCamera.FrameBox(Lo, Hi, 45, Aspect);
  Invalidate;
end;

{ Draws the volume inside the domain box.

  Back faces only, so the ray has a fragment to start from even when the
  camera is inside; depth writes off, because a translucent thing does not
  occlude what is drawn after it. }
procedure TMcxView.DrawVolume(const AMVP: TMcxMat4);
var
  Eye, Centre, Scale: TMcxVec3;
begin
  if not GetHasVolume then Exit;
  if FVolShader = nil then Exit;

  Scale := McxVec3(FVolume.Nx, FVolume.Ny, FVolume.Nz);
  Eye := FCamera.Eye;
  { The eye in volume coordinates: the shader marches through a unit cube. }
  Centre := McxVec3(Eye.x / Scale.x, Eye.y / Scale.y, Eye.z / Scale.z);

  if FCube = nil then FCube := TMcxCube.Create;

  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glDepthMask(GL_FALSE);

  FVolShader.Use;
  FVolShader.SetMat4('uMVP', AMVP);
  FVolShader.SetVec3('uScale', Scale);
  FVolShader.SetVec3('uEye', Centre);
  FVolShader.SetVec3('uMinSlice', FClipLo);
  FVolShader.SetVec3('uMaxSlice', FClipHi);
  FVolShader.SetFloat('uOpacity', FOpacity);
  FVolShader.SetFloat('uFloor', FFloor);
  FVolShader.SetInt('uMap', FMap);
  { Fewer samples along each ray while the view is being moved.

    The raycast is the whole cost of a frame: on an Intel iGPU a 181-cube
    head at 192 steps takes 25-55 ms, which is a picture that lags behind
    the mouse.  A third of the steps is a third of the work, and the
    difference is invisible on a moving image -- the opacity correction
    already keeps the brightness the same at any step count, so what changes
    is the fine structure, and that is exactly what cannot be read while the
    thing is turning.  The still frame that follows is at full quality. }
  if FDragging or FCoarse then
    FVolShader.SetFloat('uSteps', Max(48, FSteps / 3))
  else
    FVolShader.SetFloat('uSteps', FSteps);
  if UseLog then FVolShader.SetVec2('uClim', FLogLow, FLogHigh)
  else FVolShader.SetVec2('uClim', FVolLow, FVolHigh);
  FVolShader.SetInt('uStyle', FStyle);
  FVolShader.SetInt('uLog', Ord(UseLog));
  FVolShader.SetInt('uVolume', 0);

  FVolume.Bind(0);
  FCube.Draw;

  glDepthMask(GL_TRUE);
  glDisable(GL_CULL_FACE);
end;

{ Draws the scene at a given size, into whatever target is bound.  Shared by
  the on-screen paint and by the offscreen one that saves a picture. }
{ --------------------------------------------------------- trajectories --- }

procedure TMcxView.ClearTrajectory;
begin
  FTraj.Clear;
  FTrajCount := 0;
  SetLength(FTrajIds, 0);
  FTrajFirst := 0;
  FTrajLast := 0;
  Invalidate;
end;

{ The ramp a photon's weight is drawn on: dark red where almost nothing is
  left, through orange, to white where it still carries its launch weight.
  Light-on-dark in the same family as the fluence map's hot end, so paths and
  fluence can be read in one picture. }
function TrajColour(t: Double): TMcxVec3;
begin
  if t < 0 then t := 0;
  if t > 1 then t := 1;
  Result := McxVec3(0.45 + 0.55 * t,
                    0.08 + 0.82 * t * t,
                    0.05 + 0.80 * t * t * t * t);
end;

function TMcxView.ShowTrajectory(const AFileName: string): Boolean;
var
  Ids, Pts, Ws: TMcxArray;
  Got: TMcxArrayList;
  Order: TMcxOrder;
  i, n, a, b, Ends, Cut: Integer;
  HaveW: Boolean;
  w, wLo, wHi, t, t2: Double;
  P0, P1, C, C2, Lo, Hi: TMcxVec3;
  Edge: Single;

  { Within a couple of percent of any face of the domain. }
  function AtEdge(const P: TMcxVec3): Boolean;
  begin
    Result := (P.x - Lo.x < Edge) or (Hi.x - P.x < Edge) or
              (P.y - Lo.y < Edge) or (Hi.y - P.y < Edge) or
              (P.z - Lo.z < Edge) or (Hi.z - P.z < Edge);
  end;

begin
  Result := False;
  { All three out of one parse.  Asked for separately they cost three reads
    of the whole file, which for a text .jdt is the whole cost of loading it. }
  if not McxLoadArrays(AFileName, ['MCXData.Trajectory.photonid',
                                   'MCXData.Trajectory.p',
                                   'MCXData.Trajectory.w0'], Got) then Exit;
  Ids := Got[0];
  Pts := Got[1];
  Ws := Got[2];
  HaveW := McxArrayCount(Ws) > 0;
  if (McxArrayCount(Ids) = 0) or (McxArrayCount(Pts) = 0) then Exit;

  n := McxArrayCount(Ids);
  if (n < 2) or (McxArrayCount(Pts) < Int64(n) * 3) then Exit;

  { The domain's own edges, for telling a photon that left from one that was
    cut off.  A mesh knows its bounds; a voxel grid is zero to Dim. }
  if Length(FMeshFaces) > 0 then
  begin
    Lo := McxVec3(FMeshLo.x, FMeshLo.y, FMeshLo.z);
    Hi := McxVec3(FMeshHi.x, FMeshHi.y, FMeshHi.z);
  end
  else
  begin
    Lo := McxVec3(0, 0, 0);
    Hi := McxVec3(FDoc.AsInt('Domain.Dim[0]', 60), FDoc.AsInt('Domain.Dim[1]', 60),
                  FDoc.AsInt('Domain.Dim[2]', 60));
  end;
  Edge := 0.02 * Max(Hi.x - Lo.x, Max(Hi.y - Lo.y, Hi.z - Lo.z));

  { Sorted by photon, because mcx's threads append to one buffer through an
    atomic counter: in this file the identifier changes on nearly every row.
    Drawn in file order it is a spray of lines between unrelated points. }
  Order := McxSortTrajectory(Ids);

  { The weight itself, on a linear scale -- not its logarithm.

    A photon's weight decays exponentially along its own path, so a linear
    ramp over the raw weight is what shows that decay as a decay; taking the
    log first straightens it out and throws away the very thing the colour is
    there to say.  This is what utils/mcxplotphotons.m does: it hands the
    weight column straight to patch() with 'edgecolor','interp'. }
  wLo := 0;
  wHi := 1;
  if HaveW and (McxArrayCount(Ws) >= n) then
  begin
    wLo := 1e30;
    wHi := -1e30;
    for i := 0 to n - 1 do
    begin
      w := McxArrayValue(Ws, i);
      if w < wLo then wLo := w;
      if w > wHi then wHi := w;
    end;
    if wHi <= wLo then HaveW := False;
  end;

  FTraj.Clear;
  FTrajCount := 0;
  Ends := 0;
  Cut := 0;
  for i := 0 to n - 2 do
  begin
    a := Order[i];
    b := Order[i + 1];
    { A pair only makes a segment when both ends are the same photon.  Line
      segments rather than strips: no restart index to get wrong, at the
      price of writing each interior point twice. }
    if Round(McxArrayValue(Ids, a)) <> Round(McxArrayValue(Ids, b)) then Continue;

    P0 := McxVec3(McxArrayValue(Pts, Int64(a) * 3),
                  McxArrayValue(Pts, Int64(a) * 3 + 1),
                  McxArrayValue(Pts, Int64(a) * 3 + 2));
    P1 := McxVec3(McxArrayValue(Pts, Int64(b) * 3),
                  McxArrayValue(Pts, Int64(b) * 3 + 1),
                  McxArrayValue(Pts, Int64(b) * 3 + 2));

    { A colour at each end rather than one for the segment, so the shader
      interpolates along it: the weight changes between two scattering
      sites, and a flat segment draws that change as a step. }
    t := 1;
    t2 := 1;
    if HaveW then
    begin
      t := (McxArrayValue(Ws, a) - wLo) / (wHi - wLo);
      t2 := (McxArrayValue(Ws, b) - wLo) / (wHi - wLo);
    end;
    C := TrajColour(t);
    C2 := TrajColour(t2);
    FTraj.Add2(P0, P1, C, C2);

    { The far end of this segment is the photon's last event when the next
      event belongs to someone else.  Counted here rather than in a second
      pass, which would mean keeping a weight per segment. }
    if (i + 2 > n - 1) or
       (Round(McxArrayValue(Ids, Order[i + 2])) <> Round(McxArrayValue(Ids, b))) then
    begin
      Inc(Ends);
      { A path ends for one of two reasons: the photon left the domain, or
        the roulette killed it once its weight got small.  Anything else --
        stopping in open tissue still carrying weight -- is not an ending but
        a recording that ran out of room.

        Both tests are needed.  Weight alone is not enough: in a weakly
        absorbing mesh a photon leaves the boundary with 96% of its launch
        weight, and calling that truncated is how this first cried wolf. }
      if (not AtEdge(P1)) and (HaveW and (t2 > 0.02)) then Inc(Cut);
    end;
    if FTrajCount > High(FTrajIds) then
      SetLength(FTrajIds, (FTrajCount + 1) * 2);
    FTrajIds[FTrajCount] := Round(McxArrayValue(Ids, a));
    Inc(FTrajCount);
  end;
  SetLength(FTrajIds, FTrajCount);

  FTrajFirst := 0;
  FTrajLast := 0;
  if FTrajCount > 0 then
  begin
    FTrajFirst := FTrajIds[0];
    FTrajLast := FTrajIds[FTrajCount - 1];
  end;
  { Everything, until something narrows it. }
  FIdLo := FTrajFirst;
  FIdHi := FTrajLast;

  Say(Format('%s: %d events, %d path segments, photons %d to %d',
    [ExtractFileName(AFileName), n, FTrajCount, FTrajFirst + 1,
     FTrajLast + 1]));

  { A photon's path ends when it leaves the domain or the roulette kills it,
    and either way its last position is at a face or its last weight is
    tiny.  A path that stops in open tissue still carrying most of its weight
    did not end -- the recording did, because mcx's jump buffer filled.

    Worth saying out loud: the picture looks like a plausible cloud either
    way, and the only sign is that nothing reaches the far side. }
  if (Ends > 0) and (Cut * 3 > Ends) then
    Say(Format('  %d of %d paths stop with most of their weight still on ' +
      'them: the jump buffer filled before they finished.  Raise "Positions ' +
      'to keep", or run fewer photons.', [Cut, Ends]));

  Result := FTrajCount > 0;
  { New paths are output to look at, whatever the switch was left on. }
  if Result then ShowInput := False;
  Invalidate;
end;

{ The first segment belonging to photon AId or later.  Bisection, because the
  identifiers are in order: the alternative is a scan of half a million
  segments on every frame. }
function TMcxView.FirstSegmentOf(AId: Integer): Integer;
var
  Lo, Hi, Mid: Integer;
begin
  Lo := 0;
  Hi := FTrajCount;
  while Lo < Hi do
  begin
    Mid := (Lo + Hi) div 2;
    if FTrajIds[Mid] < AId then Lo := Mid + 1 else Hi := Mid;
  end;
  Result := Lo;
end;

{ Only the photons asked for.  One glDrawArrays over a span of the buffer
  rather than a rebuild of it, which is what makes dragging the range bar
  feel like moving a slider instead of reloading a file. }
procedure TMcxView.DrawTrajectory;
var
  First, Last: Integer;
begin
  if (FTrajCount = 0) or FShowInput then Exit;
  if (FIdLo <= FTrajFirst) and (FIdHi >= FTrajLast) then
  begin
    FTraj.Draw;
    Exit;
  end;
  First := FirstSegmentOf(FIdLo);
  Last := FirstSegmentOf(FIdHi + 1);
  { Two vertices a segment. }
  FTraj.DrawRange(First * 2, (Last - First) * 2);
end;

procedure TMcxView.SetPhotonRange(ALo, AHi: Integer);
begin
  FIdLo := ALo;
  FIdHi := AHi;
  Invalidate;
end;

{ ------------------------------------------------------------ picking ----- }

{ Which shape is under a point, or -1.

  Colour-identifier picking: the shapes are drawn again into an offscreen
  target with each one a flat colour that is its number, and the pixel under
  the cursor is read back.  The answer is exact whatever the geometry is, it
  costs one small render, and there is no limit on how many shapes there can
  be.

  The old renderer walked its own scene graph instead: it stopped at
  sixty-four hits, sorted them with a bubble sort, and could not pick a mesh,
  a disk, a cone or a line at all. }
function TMcxView.PickAt(AX, AY: Integer): Integer;
var
  r, g, b: Byte;
  Id: Integer;
begin
  Result := -1;
  if FPick.Count = 0 then Exit;
  if FSoft then
  begin
    if FRaster = nil then FRaster := TMcxSoftRaster.Create;
    FRaster.Start(PaneWidth, PaneHeight, McxVec3(0, 0, 0), McxMat4Mul(
      McxMat4Perspective(45, Max(1, PaneWidth) / Max(1, PaneHeight),
        FCamera.Distance * 0.01, FCamera.Distance * 10), FCamera.View));
    FRaster.PickSolids(FPick.Raw, FPick.Count);
    Id := FRaster.PickId(AX, AY);
    if (Id > 0) and (Id <= Length(FPicks)) then Result := Id - 1;
    { The raster now holds identifiers; the next paint draws it again. }
    Invalidate;
    Exit;
  end;
  if not TryGL then
  begin
    if FSoft then Result := PickAt(AX, AY);
    Exit;
  end;

  if FTarget = nil then FTarget := TMcxTarget.Create;
  { Full size rather than a one-pixel view: the projection has to be the one
    on screen or the pixel read back is not the pixel clicked. }
  if not FTarget.Bind(FGL.Width, FGL.Height) then
  begin
    FTarget.Unbind;
    Exit;
  end;
  try
    RenderScene(FGL.Width, FGL.Height, True);
    { GL counts rows from the bottom and the mouse from the top. }
    FTarget.ReadPixel(AX, FGL.Height - 1 - AY, r, g, b);
  finally
    FTarget.Unbind;
  end;

  Id := r + (g shl 8) + (b shl 16);
  if (Id > 0) and (Id <= Length(FPicks)) then Result := Id - 1;
end;

function TMcxView.SaveImage(const AFileName: string;
  AWidth, AHeight: Integer): Boolean;
var
  Raw: TBytes;
  Img: TFPMemoryImage;
  W: TFPWriterPNG;
  x, y, i: Integer;
  C: TFPColor;
  r8, g8, b8: Byte;
begin
  Result := False;
  if AWidth < 1 then AWidth := PaneWidth;
  if AHeight < 1 then AHeight := PaneHeight;
  if FSoft then
  begin
    RenderSoft(AWidth, AHeight, False);
    Img := TFPMemoryImage.Create(AWidth, AHeight);
    W := TFPWriterPNG.Create;
    try
      C.Alpha := alphaOpaque;
      for y := 0 to AHeight - 1 do
        for x := 0 to AWidth - 1 do
        begin
          FRaster.Pixel(x, y, r8, g8, b8);
          C.Red := r8 * 257;
          C.Green := g8 * 257;
          C.Blue := b8 * 257;
          Img.Colors[x, y] := C;
        end;
      Img.SaveToFile(AFileName, W);
      Result := True;
    except
      on E: Exception do Say('could not write ' + AFileName + ': ' + E.Message);
    end;
    Img.Free;
    W.Free;
    Invalidate;
    Exit;
  end;
  if not TryGL then
  begin
    if FSoft then Result := SaveImage(AFileName, AWidth, AHeight);
    Exit;
  end;

  if FTarget = nil then FTarget := TMcxTarget.Create;
  if not FTarget.Bind(AWidth, AHeight) then
  begin
    FTarget.Unbind;
    Say('the card would not give a ' + IntToStr(AWidth) + 'x' +
        IntToStr(AHeight) + ' offscreen target');
    Exit;
  end;
  try
    RenderScene(AWidth, AHeight, False);
    Raw := FTarget.ReadAll;
  finally
    FTarget.Unbind;
  end;
  if Length(Raw) = 0 then Exit;

  Img := TFPMemoryImage.Create(AWidth, AHeight);
  W := TFPWriterPNG.Create;
  try
    C.Alpha := alphaOpaque;
    for y := 0 to AHeight - 1 do
      for x := 0 to AWidth - 1 do
      begin
        { Bottom row first out of GL, top row first into the image. }
        i := ((AHeight - 1 - y) * AWidth + x) * 3;
        C.Red := Raw[i] * 257;
        C.Green := Raw[i + 1] * 257;
        C.Blue := Raw[i + 2] * 257;
        Img.Colors[x, y] := C;
      end;
    Img.SaveToFile(AFileName, W);
    Result := True;
  except
    on E: Exception do Say('could not write ' + AFileName + ': ' + E.Message);
  end;
  Img.Free;
  W.Free;
  { Back to the window, which the offscreen pass left unbound. }
  Invalidate;
end;

procedure TMcxView.RenderScene(AWidth, AHeight: Integer; AFlat: Boolean);
var
  MVP: TMcxMat4;
begin
  if AHeight < 1 then AHeight := 1;
  glViewport(0, 0, AWidth, AHeight);

  if AFlat then glClearColor(0, 0, 0, 1)
  else glClearColor(FBack.x, FBack.y, FBack.z, 1);
  glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  { Near and far follow the camera distance rather than being fixed, so a
    domain of sixty voxels and one of six hundred both keep their depth
    precision. }
  MVP := McxMat4Mul(
    McxMat4Perspective(45, AWidth / AHeight, FCamera.Distance * 0.01,
      FCamera.Distance * 10),
    FCamera.View);

  if AFlat then
  begin
    { The picking pass: identifiers, opaque, nothing else in the picture. }
    if FSolidShader = nil then Exit;
    glDisable(GL_BLEND);
    FSolidShader.Use;
    FSolidShader.SetMat4('uMVP', MVP);
    FSolidShader.SetInt('uFlat', 1);
    FPick.Draw;
    Exit;
  end;

  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  { Solids first, translucent, and drawn twice: the far side of everything,
    then the near side.

    Blending depends on the order the triangles arrive in, and they arrive in
    whatever order they were built; sorting them per frame is what a renderer
    does, but culling one way and then the other gets the same answer for any
    convex shape -- which every shape mcx has is -- for one extra draw call
    and no sorting.  Depth writes stay off so that a nearer shape does not
    hide the one it encloses. }
  if FSolidShader <> nil then
  begin
    FSolidShader.Use;
    FSolidShader.SetMat4('uMVP', MVP);
    FSolidShader.SetInt('uFlat', 0);
    FSolidShader.SetVec3('uLight', McxVec3Norm(McxVec3Sub(FCamera.Eye,
      FCamera.Target)));
    glDepthMask(GL_FALSE);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    FMesh.Draw;
    glCullFace(GL_BACK);
    FMesh.Draw;
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_TRUE);
  end;

  FShader.Use;
  FShader.SetMat4('uMVP', MVP);
  FLines.Draw;

  { Paths after the translucent solids and with depth writes on, so they read
    as being inside the domain rather than painted over it.  Line width stays
    at one: anything wider is not guaranteed in a core profile. }
  DrawTrajectory;

  { The letters and the tick numbers before the volume, and depth-tested.

    They used to be drawn last with the test off, on the grounds that they
    annotate the picture rather than sit in it -- which was true when the
    only thing that could hide one was the back edge of the box.  With a
    fluence cloud in the box it stopped being true: every number on the far
    side floated in front of the data at every angle, so the ruler read as
    being nearer than the thing it measures.

    Drawing them here fixes both halves at once, because the volume neither
    tests nor writes depth against what came before it: a number on the near
    side writes depth and the cloud does not paint over it, and one on the
    far side is painted over and dimmed by exactly the amount of cloud in
    front of it. }
  FShader.Use;
  FShader.SetMat4('uMVP', MVP);
  BuildAxisLabels;
  FAxisText.Draw;

  { The volume last: it is translucent, so it has to go over the wireframe
    rather than under it. }
  DrawVolume(MVP);
end;

procedure TMcxView.GLPaint(Sender: TObject);
begin
  if FSoft then Exit;
  { No context, or not a 3.3 one: the same picture, drawn on the processor. }
  if not FGL.MakeCurrent then
  begin
    UseSoftware('no OpenGL context could be made current');
    Exit;
  end;
  if not Start then
  begin
    UseSoftware('OpenGL 3.3 is not available: ' + McxGLDescribe);
    Exit;
  end;
  RenderScene(FGL.Width, FGL.Height, False);
  FGL.SwapBuffers;
  { A zoom has no end event to hang the full-quality frame off, so the coarse
    frame asks for it itself. }
  if FCoarse then
  begin
    FCoarse := False;
    Invalidate;
  end;
end;

procedure TMcxView.GLMouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  FDragging := Button = mbLeft;
  FDragX := X;
  FDragY := Y;
end;

procedure TMcxView.GLMouseMove(Sender: TObject; Shift: TShiftState;
  X, Y: Integer);
begin
  if not FDragging then Exit;
  { A quarter of a degree a pixel, which makes a full turn about the width of
    the pane whatever size it is. }
  FDidDrag := True;
  FCamera.Orbit((FDragX - X) * 0.008, (Y - FDragY) * 0.008);
  FDragX := X;
  FDragY := Y;
  Invalidate;
end;

procedure TMcxView.GLMouseUp(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
var
  Id: Integer;
begin
  if FDragging then
  begin
    FDragging := False;
    { The full-quality frame, now that the picture has stopped. }
    Invalidate;
  end;
  { A click, not the end of a drag: the camera moved is not a selection. }
  if (Button <> mbLeft) or FDidDrag then
  begin
    FDidDrag := False;
    Exit;
  end;

  Id := PickAt(X, Y);
  { What is selected is the command, not the solid that was clicked.  One
    command can draw several -- a ZLayers is a slab per row -- and
    highlighting only the one under the cursor said the others belonged to
    something else. }
  if (Id >= 0) and (Id < Length(FPicks)) then
  begin
    FSelected := FPicks[Id].Index;
    Rebuild;
    if Assigned(FOnPick) then FOnPick(Self, FPicks[Id]);
  end
  else
  begin
    FSelected := -1;
    Rebuild;
  end;
end;

{ Selects a command from outside -- the list in the Shapes editor -- by the
  place it has in the array, which is what a pick reports back. }
procedure TMcxView.SelectShape(AIndex: Integer);
begin
  if FSelected = AIndex then Exit;
  FSelected := AIndex;
  Rebuild;
end;

procedure TMcxView.GLMouseWheel(Sender: TObject; Shift: TShiftState;
  WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
begin
  FCamera.Zoom(WheelDelta / 120);
  FCoarse := True;
  Invalidate;
  Handled := True;
end;

{ Draws the Shapes list.

  Shapes is a sequence of commands rather than a set of objects: each one
  paints into the grid over whatever came before, and an Origin command moves
  the frame for everything after it.  So this walks the array in order and
  keeps the running origin, which is also why the drawing has to be in the
  same order -- two boxes at the same place are the second one.

  Colour comes from Tag, the medium a shape assigns, so two regions of the
  same material look the same.  UpperSpace is deliberately absent: a
  half-space has no outline to draw, and a big translucent plane across the
  domain would hide what it is meant to explain. }
{ A rounded spacing for graduations across a span.

  Aims for five to ten divisions and picks the largest round number that
  stays inside that: 1, 2 or 5 times a power of ten.  A 60-unit domain gets
  10, a 181-unit head gets 20, a 10-unit cube gets 2 -- all numbers a person
  reads without having to work them out, which is the whole point of putting
  them on an axis. }
function McxNiceStep(ASpan: Single): Single;
const
  Nice: array[0..2] of Single = (1, 2, 5);
var
  k, i: Integer;
  Mag, Step, n: Double;
begin
  Result := 1;
  if ASpan <= 0 then Exit;
  Result := 0;
  for k := -4 to 8 do
  begin
    Mag := Power(10, k);
    for i := 0 to 2 do
    begin
      Step := Nice[i] * Mag;
      n := ASpan / Step;
      { Largest step that still leaves at least five divisions. }
      if (n >= 5) and (n <= 10.5) and (Step > Result) then Result := Step;
    end;
  end;
  { Nothing fitted -- a span of an odd size.  Eight divisions of whatever it
    takes is better than no graduations at all. }
  if Result = 0 then Result := ASpan / 8;
end;

{ The three axes, each graduated and labelled.

  The axes run the whole length of the domain rather than a quarter of it:
  they are a ruler, and a ruler that stops short of what is being measured
  cannot be read against it.  The tick marks are where the numbers go, and
  the numbers are what say how big the thing on screen actually is.

  The lines and ticks go into the scene, which is rebuilt when the document
  changes.  The letters and numbers go in a batch of their own, rebuilt every
  frame to face the camera: fixed in world space a digit is a vertical line
  from half the angles you would look from. }
procedure TMcxView.AddAxes(const ALo, AHi: TMcxVec3);
var
  Red, Green, Blue: TMcxVec3;
  Tick: Single;

  procedure Graduate(AAxis: Integer; const AColour: TMcxVec3);
  var
    Lo, Hi, Step, t: Single;
    A, B: TMcxVec3;
    i: Integer;
  begin
    case AAxis of
      0: begin Lo := ALo.x; Hi := AHi.x; end;
      1: begin Lo := ALo.y; Hi := AHi.y; end;
    else begin Lo := ALo.z; Hi := AHi.z; end;
    end;
    Step := McxNiceStep(Hi - Lo);
    FAxisStep[AAxis] := Step;

    A := ALo;
    B := ALo;
    case AAxis of
      0: B.x := Hi;
      1: B.y := Hi;
    else B.z := Hi;
    end;
    FLines.Add(A, B, AColour);

    { A tick at every round multiple inside the span, and one tick, not two.

      There were two -- one into each of the directions the axis is not -- so
      that whichever way the scene was turned, one of them faced the camera.
      But only one of the pair ever had a number on the end of it, so the
      other was a stroke pointing at nothing, and on three axes at once that
      is a thicket around the origin corner.  What is left is the one that
      points at its own number, which is the direction Numbers offsets in:
      -y for x, and -x for both y and z. }
    i := Trunc(Lo / Step);
    while i * Step <= Hi + Step * 0.001 do
    begin
      t := i * Step;
      if t >= Lo - Step * 0.001 then
      begin
        A := ALo;
        B := ALo;
        case AAxis of
          0: begin A.x := t; B.x := t; B.y := ALo.y - Tick; end;
          1: begin A.y := t; B.y := t; B.x := ALo.x - Tick; end;
        else begin A.z := t; B.z := t; B.x := ALo.x - Tick; end;
        end;
        FLines.Add(A, B, AColour);
      end;
      Inc(i);
    end;
  end;

begin
  FAxisLo := ALo;
  FAxisHi := AHi;
  Red := McxVec3(0.90, 0.30, 0.25);
  Green := McxVec3(0.35, 0.75, 0.35);
  Blue := McxVec3(0.35, 0.55, 0.95);
  Tick := Max(AHi.x - ALo.x, Max(AHi.y - ALo.y, AHi.z - ALo.z)) * 0.025;
  Graduate(0, Red);
  Graduate(1, Green);
  Graduate(2, Blue);
end;

{ The letters and the numbers, turned to face the camera.

  Stroked rather than typeset: a core profile has no text of any kind, and a
  dozen glyphs do not justify a font atlas and a second shader.  The digits
  are the seven segments of a calculator display, which is the shortest
  description of a digit there is and reads cleanly at any size. }
procedure TMcxView.BuildAxisLabels;
const
  { u, v pairs in a unit box, two per stroke. }
  GlyphX: array[0..7] of Single = (0, 0, 1, 1,  0, 1, 1, 0);
  GlyphY: array[0..11] of Single =
    (0, 1, 0.5, 0.5,  1, 1, 0.5, 0.5,  0.5, 0.5, 0.5, 0);
  GlyphZ: array[0..11] of Single = (0, 1, 1, 1,  1, 1, 0, 0,  0, 0, 1, 0);
  { a b c d e f g, in the usual order. }
  { Narrower than they are tall -- 0.55 by 1 -- so a two-digit number is a
    number and not a box. }
  Seg: array[0..6, 0..3] of Single = (
    (0, 1, 0.55, 1), (0.55, 1, 0.55, 0.5), (0.55, 0.5, 0.55, 0),
    (0, 0, 0.55, 0), (0, 0.5, 0, 0), (0, 1, 0, 0.5), (0, 0.5, 0.55, 0.5));
  { Which segments each digit lights. }
  Digits: array[0..9] of array[0..6] of Boolean = (
    (True,  True,  True,  True,  True,  True,  False),
    (False, True,  True,  False, False, False, False),
    (True,  True,  False, True,  True,  False, True),
    (True,  True,  True,  True,  False, False, True),
    (False, True,  True,  False, False, True,  True),
    (True,  False, True,  True,  False, True,  True),
    (True,  False, True,  True,  True,  True,  True),
    (True,  True,  True,  False, False, False, False),
    (True,  True,  True,  True,  True,  True,  True),
    (True,  True,  True,  True,  False, True,  True));
var
  R, U: TMcxVec3;
  Big, Small, Tick: Single;
  Red, Green, Blue: TMcxVec3;

  { One stroke of a glyph, in the plane the camera faces. }
  procedure Stroke(const ACorner: TMcxVec3; AScale: Single;
    u0, v0, u1, v1: Single; const AColour: TMcxVec3);
  begin
    FAxisText.Add(
      McxVec3(ACorner.x + (R.x * u0 + U.x * v0) * AScale,
              ACorner.y + (R.y * u0 + U.y * v0) * AScale,
              ACorner.z + (R.z * u0 + U.z * v0) * AScale),
      McxVec3(ACorner.x + (R.x * u1 + U.x * v1) * AScale,
              ACorner.y + (R.y * u1 + U.y * v1) * AScale,
              ACorner.z + (R.z * u1 + U.z * v1) * AScale),
      AColour);
  end;

  procedure Glyph(const ACentre: TMcxVec3; AScale: Single;
    const AStrokes: array of Single; const AColour: TMcxVec3);
  var
    k: Integer;
    Corner: TMcxVec3;
  begin
    Corner := McxVec3(ACentre.x - (R.x + U.x) * AScale * 0.5,
                      ACentre.y - (R.y + U.y) * AScale * 0.5,
                      ACentre.z - (R.z + U.z) * AScale * 0.5);
    k := 0;
    while k + 3 <= High(AStrokes) do
    begin
      Stroke(Corner, AScale, AStrokes[k], AStrokes[k + 1],
             AStrokes[k + 2], AStrokes[k + 3], AColour);
      Inc(k, 4);
    end;
  end;

  { A number, centred on APos.  Each digit is 0.6 of the cell wide, so the
    string is laid out on that pitch. }
  procedure Number(const APos: TMcxVec3; AValue: Double; AScale: Single;
    const AColour: TMcxVec3);
  var
    S: string;
    i, k, d: Integer;
    Corner: TMcxVec3;
    Pitch, Left: Single;
  begin
    if Abs(AValue) < 1e-9 then AValue := 0;
    if Abs(AValue - Round(AValue)) < 1e-6 then S := IntToStr(Round(AValue))
    else S := FormatFloat('0.##', AValue);
    Pitch := AScale * 0.72;
    Left := -Pitch * (Length(S) - 1) * 0.5 - AScale * 0.275;
    for i := 1 to Length(S) do
    begin
      Corner := McxVec3(
        APos.x + R.x * (Left + (i - 1) * Pitch) - U.x * AScale * 0.5,
        APos.y + R.y * (Left + (i - 1) * Pitch) - U.y * AScale * 0.5,
        APos.z + R.z * (Left + (i - 1) * Pitch) - U.z * AScale * 0.5);
      if S[i] = '-' then
        Stroke(Corner, AScale, Seg[6][0], Seg[6][1], Seg[6][2], Seg[6][3], AColour)
      else if S[i] = '.' then
        Stroke(Corner, AScale, 0.2, 0, 0.35, 0, AColour)
      else if S[i] in ['0'..'9'] then
      begin
        d := Ord(S[i]) - Ord('0');
        for k := 0 to 6 do
          if Digits[d][k] then
            Stroke(Corner, AScale, Seg[k][0], Seg[k][1], Seg[k][2], Seg[k][3],
                   AColour);
      end;
    end;
  end;

  { The graduations of one axis. }
  procedure Numbers(AAxis: Integer; const AColour: TMcxVec3);
  var
    Lo, Hi, Step, t: Single;
    P: TMcxVec3;
    i: Integer;
  begin
    case AAxis of
      0: begin Lo := FAxisLo.x; Hi := FAxisHi.x; end;
      1: begin Lo := FAxisLo.y; Hi := FAxisHi.y; end;
    else begin Lo := FAxisLo.z; Hi := FAxisHi.z; end;
    end;
    Step := FAxisStep[AAxis];
    if Step <= 0 then Exit;
    i := Trunc(Lo / Step);
    while i * Step <= Hi + Step * 0.001 do
    begin
      t := i * Step;
      { The far end carries the axis letter, so its number would sit on top
        of it; and zero is the corner all three share. }
      if (t >= Lo - Step * 0.001) and (t < Hi - Step * 0.5) and (i <> 0) then
      begin
        P := FAxisLo;
        case AAxis of
          0: begin P.x := t; P.y := P.y - Tick * 3.0; end;
          1: begin P.y := t; P.x := P.x - Tick * 3.0; end;
        else begin P.z := t; P.x := P.x - Tick * 3.0; end;
        end;
        Number(P, t, Small, AColour);
      end;
      Inc(i);
    end;
  end;

begin
  FAxisText.Clear;
  if (FAxisHi.x <= FAxisLo.x) and (FAxisHi.y <= FAxisLo.y) then Exit;
  R := FCamera.ScreenRight;
  U := FCamera.ScreenUp;
  Red := McxVec3(0.90, 0.30, 0.25);
  Green := McxVec3(0.35, 0.75, 0.35);
  Blue := McxVec3(0.35, 0.55, 0.95);

  Tick := Max(FAxisHi.x - FAxisLo.x,
              Max(FAxisHi.y - FAxisLo.y, FAxisHi.z - FAxisLo.z)) * 0.025;
  Big := Tick * 2.6;
  Small := Tick * 2.0;

  Glyph(McxVec3(FAxisHi.x + Big, FAxisLo.y, FAxisLo.z), Big, GlyphX, Red);
  Glyph(McxVec3(FAxisLo.x, FAxisHi.y + Big, FAxisLo.z), Big, GlyphY, Green);
  Glyph(McxVec3(FAxisLo.x, FAxisLo.y, FAxisHi.z + Big), Big, GlyphZ, Blue);

  Numbers(0, Red);
  Numbers(1, Green);
  Numbers(2, Blue);
end;

{ A label volume as the 3-D view wants it: one float per voxel, x fastest.

  JData writes arrays row-major unless _ArrayOrder_ says "c", and mcx reads
  a Shapes array that way (mcx_utils.c sets isrowmajor for it), so the last
  index is the fastest one.  A fourth dimension is the per-voxel values of
  the continuous formats -- mua, mus, g, n for asgn_float -- and only the
  first is drawn.  It comes either last ([60,40,20,4], interleaved) or first
  ([4,60,40,20]); whichever of the two leaves the other three matching Dim
  is the one meant, and failing that the short one is the component. }
function DomainGrid(const A: TMcxArray; ARowMajor: Boolean; ANx, ANy,
  ANz: Integer; out AOut: TMcxArray): Boolean;
var
  nd, nx, ny, nz, nc, x, y, z: Integer;
  Lead: Boolean;
  Src, Dst: Int64;
  P: PSingle;
begin
  Result := False;
  nd := Length(A.Dims);
  if nd < 3 then Exit;
  Lead := False;
  nc := 1;
  if nd >= 4 then
  begin
    if (A.Dims[1] = ANx) and (A.Dims[2] = ANy) and (A.Dims[3] = ANz) then
      Lead := True
    else if not ((A.Dims[0] = ANx) and (A.Dims[1] = ANy) and (A.Dims[2] = ANz)) then
      Lead := A.Dims[0] < A.Dims[nd - 1];
    if Lead then nc := A.Dims[0] else nc := A.Dims[3];
  end;
  if Lead then
  begin
    nx := A.Dims[1]; ny := A.Dims[2]; nz := A.Dims[3];
  end
  else
  begin
    nx := A.Dims[0]; ny := A.Dims[1]; nz := A.Dims[2];
  end;
  if (nx < 1) or (ny < 1) or (nz < 1) or (nc < 1) then Exit;
  if Int64(nx) * ny * nz * nc > McxArrayCount(A) then Exit;

  AOut.Kind := akSingle;
  SetLength(AOut.Dims, 3);
  AOut.Dims[0] := nx; AOut.Dims[1] := ny; AOut.Dims[2] := nz;
  SetLength(AOut.Data, Int64(nx) * ny * nz * SizeOf(Single));
  P := PSingle(@AOut.Data[0]);
  Dst := 0;
  for z := 0 to nz - 1 do
    for y := 0 to ny - 1 do
      for x := 0 to nx - 1 do
      begin
        if ARowMajor then
        begin
          Src := (Int64(x) * ny + y) * nz + z;
          if not Lead then Src := Src * nc;
        end
        else
        begin
          Src := x + Int64(nx) * (y + Int64(ny) * z);
          if Lead then Src := Src * nc;
        end;
        P[Dst] := McxArrayValue(A, Src);
        Inc(Dst);
      end;
  Result := True;
end;

{ A raw VolumeFile: Dim voxels of one label each, in the width MediaFormat
  names.  The continuous formats are not labels and are left to the run. }
function RawVolume(const AFileName: string; AWidth, ANx, ANy, ANz: Integer;
  out AOut: TMcxArray): Boolean;
var
  F: TFileStream;
  n: Int64;
begin
  Result := False;
  n := Int64(ANx) * ANy * ANz;
  if n <= 0 then Exit;
  try
    F := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyNone);
    try
      if F.Size < n * AWidth then Exit;
      case AWidth of
        1: AOut.Kind := akUInt8;
        2: AOut.Kind := akUInt16;
      else
        AOut.Kind := akUInt32;
      end;
      SetLength(AOut.Dims, 3);
      AOut.Dims[0] := ANx; AOut.Dims[1] := ANy; AOut.Dims[2] := ANz;
      SetLength(AOut.Data, n * AWidth);
      F.ReadBuffer(AOut.Data[0], n * AWidth);
      Result := True;
    finally
      F.Free;
    end;
  except
    Result := False;
  end;
end;

{ The Shapes array of a VolumeFile that is a .json, or nil.  mcx reads such
  a file as a shapes description rather than a volume (mcx_loadvolume), so
  the card draws it the way it draws shapes written into the input. }
function TMcxView.ShapeFileShapes: TJSONData;
var
  Fn, Key: string;
  Age: LongInt;
  L: TStringList;
begin
  Result := nil;
  Fn := FDoc.AsStr('Domain.VolumeFile', '');
  if LowerCase(ExtractFileExt(Fn)) <> '.json' then Exit;
  if (FDoc.FileName <> '') and not FilenameIsAbsolute(Fn) then
    Fn := ExtractFilePath(FDoc.FileName) + Fn;
  Age := FileAge(Fn);
  if Age = -1 then Exit;
  Key := Format('%s %d', [Fn, Age]);
  if Key <> FShapeFileKey then
  begin
    FShapeFileKey := Key;
    FreeAndNil(FShapeFile);
    L := TStringList.Create;
    try
      try
        L.LoadFromFile(Fn);
        FShapeFile := GetJSON(L.Text);
      except
        FreeAndNil(FShapeFile);
      end;
    finally
      L.Free;
    end;
  end;
  if FShapeFile is TJSONObject then
    Result := TJSONObject(FShapeFile).Find('Shapes');
end;

{ The domain itself, when the document carries it as data rather than as
  shapes to draw: mcxlab and mcx --dumpjson write the volume into Shapes as
  a JData array, and a VolumeFile names one on disk.  Without this the card
  showed an empty box for every such input -- which is most of the demos.

  Drawn as a volume, the same way a result is, and only put up when the
  data itself changed: Rebuild runs on every edit, and a result opened
  since should not be thrown away because a photon count was typed. }
procedure TMcxView.AddDomainVolume;
var
  Sh, O: TJSONData;
  Key, Fn, Fmt, Ext: string;
  A, G: TMcxArray;
  Got, RowMajor: Boolean;
  nx, ny, nz, W: Integer;
  Age: LongInt;
begin
  if FDoc = nil then Exit;
  nx := FDoc.AsInt('Domain.Dim[0]', 0);
  ny := FDoc.AsInt('Domain.Dim[1]', 0);
  nz := FDoc.AsInt('Domain.Dim[2]', 0);
  Key := '';
  Fn := '';
  Sh := FDoc.Find('Shapes');
  if (Sh is TJSONObject) and (TJSONObject(Sh).Find('_ArrayType_') <> nil) then
  begin
    O := TJSONObject(Sh).Find('_ArrayZipData_');
    if O = nil then O := TJSONObject(Sh).Find('_ArrayData_');
    if O <> nil then
      Key := Format('jdata %d %s', [Length(O.AsJSON),
        Copy(O.AsJSON, 1, 64)]);
  end
  else
  begin
    Fn := FDoc.AsStr('Domain.VolumeFile', '');
    if Fn <> '' then
    begin
      if (FDoc.FileName <> '') and not FilenameIsAbsolute(Fn) then
        Fn := ExtractFilePath(FDoc.FileName) + Fn;
      Age := FileAge(Fn);
      if Age <> -1 then
        Key := Format('file %s %d %s %d %d %d', [Fn, Age,
          FDoc.AsStr('Domain.MediaFormat', 'byte'), nx, ny, nz])
      else
        Fn := '';
    end;
  end;

  { A result on the card stays there until the switch asks for the domain,
    which reads it afresh then. }
  if FHasResult and not FShowInput then Exit;
  if Key = FDomainKey then Exit;
  FDomainKey := Key;

  if Key = '' then
  begin
    { The domain this document had is gone; a result is left alone. }
    if FDomainShown then
    begin
      ClearVolume;
      FDomainShown := False;
      FVolumeChanged := True;
    end;
    Exit;
  end;

  Got := False;
  if Fn = '' then
  begin
    O := TJSONObject(Sh).Find('_ArrayOrder_');
    RowMajor := (O = nil) or (LowerCase(Copy(O.AsString, 1, 1)) <> 'c');
    Got := McxDecodeJData(TJSONObject(Sh), A) and
      DomainGrid(A, RowMajor, nx, ny, nz, G);
  end
  else
  begin
    Ext := LowerCase(ExtractFileExt(Fn));
    if (Ext = '.bin') or (Ext = '.vol') or (Ext = '.raw') then
    begin
      Fmt := LowerCase(FDoc.AsStr('Domain.MediaFormat', 'byte'));
      W := 0;
      if Fmt = 'byte' then W := 1
      else if Fmt = 'short' then W := 2
      else if Fmt = 'integer' then W := 4;
      if W > 0 then Got := RawVolume(Fn, W, nx, ny, nz, G);
    end
    else
      { A NIfTI or JNIfTI volume is read the way a result is, and is
        already x-fastest. }
      Got := McxLoadArray(Fn, '', A) and DomainGrid(A, False, nx, ny, nz, G);
  end;

  FInDomain := True;
  try
    if Got then Got := ShowVolume(G, True);
  finally
    FInDomain := False;
  end;
  if Got then
  begin
    FDomainShown := True;
    FVolumeChanged := True;
  end
  else if FDomainShown then
  begin
    ClearVolume;
    FDomainShown := False;
    FVolumeChanged := True;
  end;
end;

procedure TMcxView.AddShapes;
const
  { Translucent, because a domain is nested: a sphere inside a box inside a
    grid, and an opaque outer shape hides everything the simulation is
    actually about. }
  ShapeAlpha = 0.55;
  { The one that has been clicked is drawn nearly solid, which is the whole
    of the feedback: no outline to keep in step, no second object to free. }
  SelectedAlpha = 0.92;
  { One colour per medium, wrapping.  Distinct at a glance rather than a
    gradient: the tags are names, not amounts. }
  TagColours: array[0..7] of array[0..2] of Single = (
    (0.60, 0.62, 0.66), (0.35, 0.70, 0.95), (0.95, 0.55, 0.30),
    (0.45, 0.85, 0.45), (0.90, 0.45, 0.75), (0.95, 0.85, 0.35),
    (0.55, 0.50, 0.90), (0.40, 0.85, 0.80));
var
  Shapes: TJSONData;
  i: Integer;
  Origin: TMcxVec3;

  function Colour(ATag: Integer): TMcxVec3;
  begin
    if ATag < 0 then ATag := 0;
    ATag := ATag mod Length(TagColours);
    Result := McxVec3(TagColours[ATag][0], TagColours[ATag][1],
      TagColours[ATag][2]);
  end;

  { The identifier of the shape being built, as a colour.  One past the
    index, so that nothing is black and black can mean nothing. }
  function IdColour: TMcxVec3;
  var
    Id: Integer;
  begin
    Id := Length(FPicks);
    Result := McxVec3((Id and $FF) / 255, ((Id shr 8) and $FF) / 255,
      ((Id shr 16) and $FF) / 255);
  end;

  function Alpha: Single;
  begin
    if i = FSelected then Result := SelectedAlpha
    else Result := ShapeAlpha;
  end;

  { Notes what is about to be drawn, so a pick can say what it hit. }
  procedure Note(const AVerb: string; ATag: Integer);
  var
    n: Integer;
  begin
    n := Length(FPicks);
    SetLength(FPicks, n + 1);
    FPicks[n].Index := i;
    FPicks[n].Verb := AVerb;
    FPicks[n].Tag := ATag;
  end;

  { A triplet from a shape's field, offset by the running origin. }
  function Triplet(AObj: TJSONData; const AName: string;
    AOffset: Boolean = True): TMcxVec3;
  var
    A: TJSONData;
  begin
    Result := McxVec3(0, 0, 0);
    if AObj = nil then Exit;
    A := AObj.FindPath(AName);
    if (A = nil) or (A.JSONType <> jtArray) or (A.Count < 3) then Exit;
    Result := McxVec3(A.Items[0].AsFloat, A.Items[1].AsFloat,
      A.Items[2].AsFloat);
    if AOffset then
      Result := McxVec3(Result.x + Origin.x, Result.y + Origin.y,
        Result.z + Origin.z);
  end;

  function Number(AObj: TJSONData; const AName: string; ADef: Single): Single;
  var
    V: TJSONData;
  begin
    Result := ADef;
    if AObj = nil then Exit;
    V := AObj.FindPath(AName);
    if (V <> nil) and (V.JSONType = jtNumber) then Result := V.AsFloat;
  end;

  function Tag(AObj: TJSONData): Integer;
  begin
    Result := Round(Number(AObj, 'Tag', 1));
  end;

  { A box, filled and outlined: the fill says what it is and the outline
    keeps the edges readable where two shapes of similar colour meet. }
  procedure Solid(const AVerb: string; const ALo, AHi: TMcxVec3; ATag: Integer);
  begin
    Note(AVerb, ATag);
    FMesh.AddBox(ALo, AHi, Colour(ATag), Alpha);
    FPick.AddBox(ALo, AHi, IdColour, 1);
    FLines.AddBox(ALo, AHi, Colour(ATag));
  end;

  { A slab or layer is a pair or triple of bounds along one axis; it is drawn
    as the slice of the domain it claims. }
  procedure Slab(AAxis: Integer; ALo, AHi: Single; const AColour: TMcxVec3;
    ATagForSlab: Integer);
  var
    Lo, Hi: TMcxVec3;
    dx, dy, dz: Single;
  begin
    dx := FDoc.AsInt('Domain.Dim[0]', 60);
    dy := FDoc.AsInt('Domain.Dim[1]', 60);
    dz := FDoc.AsInt('Domain.Dim[2]', 60);
    case AAxis of
      0: begin Lo := McxVec3(ALo, 0, 0); Hi := McxVec3(AHi, dy, dz); end;
      1: begin Lo := McxVec3(0, ALo, 0); Hi := McxVec3(dx, AHi, dz); end;
    else
      begin Lo := McxVec3(0, 0, ALo); Hi := McxVec3(dx, dy, AHi); end;
    end;
    Note('Slab', ATagForSlab);
    FMesh.AddBox(Lo, Hi, AColour, Alpha);
    FPick.AddBox(Lo, Hi, IdColour, 1);
    FLines.AddBox(Lo, Hi, AColour);
  end;

  procedure Bands(AObj: TJSONData; AAxis: Integer; AWithTag: Boolean;
    ADefTag: Integer);
  var
    j: Integer;
    Row: TJSONData;
    T: Integer;
  begin
    if (AObj = nil) or (AObj.JSONType <> jtArray) then Exit;
    for j := 0 to AObj.Count - 1 do
    begin
      Row := AObj.Items[j];
      if (Row = nil) or (Row.JSONType <> jtArray) or (Row.Count < 2) then Continue;
      T := ADefTag;
      if AWithTag and (Row.Count >= 3) then T := Round(Row.Items[2].AsFloat);
      Slab(AAxis, Row.Items[0].AsFloat + Origin.x, Row.Items[1].AsFloat + Origin.x,
        Colour(T), T);
    end;
  end;

var
  Cmd, Obj, Bound: TJSONData;
  Verb: string;
  O, Size, C0, C1: TMcxVec3;
begin
  Shapes := FDoc.Find('Shapes');
  if (Shapes = nil) or (Shapes.JSONType <> jtArray) then Shapes := ShapeFileShapes;
  if (Shapes = nil) or (Shapes.JSONType <> jtArray) then Exit;
  Origin := McxVec3(0, 0, 0);

  for i := 0 to Shapes.Count - 1 do
  begin
    Cmd := Shapes.Items[i];
    if (Cmd = nil) or (Cmd.JSONType <> jtObject) or (Cmd.Count < 1) then Continue;
    Verb := TJSONObject(Cmd).Names[0];
    Obj := TJSONObject(Cmd).Items[0];

    if SameText(Verb, 'Origin') then
    begin
      Origin := Triplet(Cmd, 'Origin', False);
      Continue;
    end;
    if SameText(Verb, 'Name') then Continue;

    if SameText(Verb, 'Grid') or SameText(Verb, 'Subgrid') or
       SameText(Verb, 'SubGrid') then
    begin
      O := Triplet(Obj, 'O');
      Size := Triplet(Obj, 'Size', False);
      { A Grid has no O: it replaces the whole background. }
      if SameText(Verb, 'Grid') then O := Origin;
      Solid(Verb, O, McxVec3(O.x + Size.x, O.y + Size.y, O.z + Size.z),
        Tag(Obj));
    end
    else if SameText(Verb, 'Box') then
    begin
      O := Triplet(Obj, 'O');
      Size := Triplet(Obj, 'Size', False);
      Solid(Verb, O, McxVec3(O.x + Size.x, O.y + Size.y, O.z + Size.z),
        Tag(Obj));
    end
    else if SameText(Verb, 'Sphere') then
    begin
      { The documentation calls the centre C0 and the schema calls it O, and
        real files use both. }
      O := Triplet(Obj, 'O');
      if (Obj <> nil) and (Obj.FindPath('O') = nil) then O := Triplet(Obj, 'C0');
      Note(Verb, Tag(Obj));
      FMesh.AddSphere(O, Number(Obj, 'R', 1), Colour(Tag(Obj)), Alpha);
      FPick.AddSphere(O, Number(Obj, 'R', 1), IdColour, 1);
    end
    else if SameText(Verb, 'Cylinder') then
    begin
      C0 := Triplet(Obj, 'C0');
      C1 := Triplet(Obj, 'C1');
      Note(Verb, Tag(Obj));
      FMesh.AddCylinder(C0, C1, Number(Obj, 'R', 1), Colour(Tag(Obj)), Alpha);
      FPick.AddCylinder(C0, C1, Number(Obj, 'R', 1), IdColour, 1);
    end
    else if SameText(Verb, 'XLayers') then Bands(Obj, 0, True, 1)
    else if SameText(Verb, 'YLayers') then Bands(Obj, 1, True, 1)
    else if SameText(Verb, 'ZLayers') then Bands(Obj, 2, True, 1)
    else if SameText(Verb, 'XSlabs') or SameText(Verb, 'YSlabs') or
            SameText(Verb, 'ZSlabs') then
    begin
      Bound := nil;
      if Obj <> nil then Bound := Obj.FindPath('Bound');
      case UpCase(Verb[1]) of
        'X': Bands(Bound, 0, False, Tag(Obj));
        'Y': Bands(Bound, 1, False, Tag(Obj));
      else
        Bands(Bound, 2, False, Tag(Obj));
      end;
    end;
  end;
end;

{ Draws the mmc mesh, if this is an mmc simulation.

  mmc does not have a grid: the domain is a mesh of tetrahedra that lives
  beside the input file, named by Mesh.MeshID.  What a preview shows is the
  outside of it -- a hundred thousand tetrahedra drawn as tetrahedra is a
  solid block of edges -- so mcxmesh pulls out the boundary and this draws
  that, coloured by the medium each face belongs to.

  Loaded at most once per mesh: the surface of a head mesh takes most of a
  second to find, and Rebuild runs on every edit. }
procedure TMcxView.AddMesh;
const
  ShapeAlpha = 0.55;
  TagColours: array[0..7] of array[0..2] of Single = (
    (0.60, 0.62, 0.66), (0.35, 0.70, 0.95), (0.95, 0.55, 0.30),
    (0.45, 0.85, 0.45), (0.90, 0.45, 0.75), (0.95, 0.85, 0.35),
    (0.55, 0.50, 0.90), (0.40, 0.85, 0.80));
var
  Id, Dir, Key: string;
  Mesh: TMcxTetMesh;
  i, t: Integer;
  C: TMcxVec3;
  Embedded, Got: Boolean;
  Nodes, Elems: TMcxArray;
begin
  { Two ways a mesh arrives, and both are in use: named by Mesh.MeshID as a
    pair of .dat tables beside the input file, or embedded in the input under
    Shapes as two JData arrays.  onecube is the first, colin27 the second. }
  Embedded := (FDoc.Find('Shapes.MeshNode') <> nil) and
              (FDoc.Find('Shapes.MeshElem') <> nil);
  Id := FDoc.AsStr('Mesh.MeshID', '');
  Dir := ExtractFilePath(FDoc.FileName);
  if (not Embedded) and ((Id = '') or (Dir = '')) then
  begin
    SetLength(FMeshFaces, 0);
    FMeshKey := '';
    Exit;
  end;

  if Embedded then Key := FDoc.FileName + '|embedded'
  else Key := Dir + '|' + Id;
  if Key <> FMeshKey then
  begin
    FMeshKey := Key;
    SetLength(FMeshFaces, 0);
    Mesh := TMcxTetMesh.Create;
    try
      if Embedded then
      begin
        Got := McxDecodeJData(TJSONObject(FDoc.Find('Shapes.MeshNode')), Nodes)
           and McxDecodeJData(TJSONObject(FDoc.Find('Shapes.MeshElem')), Elems)
           and Mesh.LoadFromArrays(Nodes, Elems);
        Id := 'in the input file';
      end
      else
        Got := Mesh.LoadFromDir(Dir, Id);

      if Got then
      begin
        FMeshFaces := Mesh.Surface;
        Mesh.Bounds(FMeshLo, FMeshHi);
        Say(Format('mesh %s: %d nodes, %d elements, %d surface triangles',
          [Id, Mesh.NodeCount, Mesh.ElemCount, Length(FMeshFaces)]));
      end
      else
        Say('mesh ' + Id + ': ' + Mesh.Error);
    finally
      Mesh.Free;
    end;
  end;

  for i := 0 to High(FMeshFaces) do
  begin
    t := FMeshFaces[i].Tag;
    if t < 0 then t := 0;
    t := t mod Length(TagColours);
    C := McxVec3(TagColours[t][0], TagColours[t][1], TagColours[t][2]);
    FMesh.AddTri(McxVec3(FMeshFaces[i].A.x, FMeshFaces[i].A.y, FMeshFaces[i].A.z),
                 McxVec3(FMeshFaces[i].B.x, FMeshFaces[i].B.y, FMeshFaces[i].B.z),
                 McxVec3(FMeshFaces[i].C.x, FMeshFaces[i].C.y, FMeshFaces[i].C.z),
                 C, ShapeAlpha);
  end;
end;

{ The source, drawn as what it actually is.

  Every source gets an arrow showing where it points, because direction is
  the thing a position alone does not say.  On top of that each type gets the
  shape it emits from, taken from mcx's own meaning for Param1 and Param2
  (mcx_utils.c, the srctype table): a cone opens to its half-angle, a disk is
  a disk of its radius, a planar source is the quad its two vectors span.

  The old GUI drew twelve of the eighteen types and mcxcloud drew ten;
  neither drew pencilarray, hyperboloid or ring.  This draws what it can and
  falls back to the arrow alone for the rest, which is honest -- an arrow is
  true of every source. }
procedure TMcxView.AddSource;
const
  Amber: array[0..2] of Single = (1.0, 0.80, 0.25);
  SrcAlpha = 0.75;
var
  Pos, Dir, P1, P2, C: TMcxVec3;
  Kind: string;
  Scale, R: Single;
  i, j, n: Integer;

  function Vec(const APath: string; ADefZ: Single): TMcxVec3;
  var
    A: TJSONData;
  begin
    Result := McxVec3(0, 0, ADefZ);
    A := FDoc.Find(APath);
    if (A = nil) or (A.JSONType <> jtArray) or (A.Count < 3) then Exit;
    if not (A.Items[0].JSONType in [jtNumber, jtString]) then Exit;
    Result := McxVec3(A.Items[0].AsFloat, A.Items[1].AsFloat,
                      A.Items[2].AsFloat);
  end;

  function Param(const APath: string; AIndex: Integer): Single;
  var
    A: TJSONData;
  begin
    Result := 0;
    A := FDoc.Find(APath);
    if (A = nil) or (A.JSONType <> jtArray) or (A.Count <= AIndex) then Exit;
    if A.Items[AIndex].JSONType = jtNumber then
      Result := A.Items[AIndex].AsFloat;
  end;

begin
  Pos := Vec('Optode.Source.Pos', 0);
  Dir := Vec('Optode.Source.Dir', 1);
  C := McxVec3(Amber[0], Amber[1], Amber[2]);
  Kind := LowerCase(FDoc.AsStr('Optode.Source.Type', 'pencil'));

  { Sized against the domain, so the glyph is legible on a sixty-voxel cube
    and on a six-hundred-voxel one. }
  Scale := FSceneSpan * 0.18;
  if Scale <= 0 then Scale := 5;

  FMesh.AddArrow(Pos, Dir, Scale, C, SrcAlpha);

  P1 := McxVec3(Param('Optode.Source.Param1', 0),
                Param('Optode.Source.Param1', 1),
                Param('Optode.Source.Param1', 2));
  P2 := McxVec3(Param('Optode.Source.Param2', 0),
                Param('Optode.Source.Param2', 1),
                Param('Optode.Source.Param2', 2));

  if Kind = 'isotropic' then
    { No direction to speak of: a ball, and the arrow above is the only hint
      of the axis the file happens to name. }
    FMesh.AddSphere(Pos, Scale * 0.12, C, SrcAlpha)
  else if Kind = 'cone' then
  begin
    { Param1[0] is the half-angle in radians. }
    R := Scale * Tan(Param('Optode.Source.Param1', 0));
    if R <= 0 then R := Scale * 0.2;
    FMesh.AddCone(Pos, McxVec3(Pos.x + Dir.x * Scale, Pos.y + Dir.y * Scale,
      Pos.z + Dir.z * Scale), R, C, SrcAlpha * 0.5);
  end
  else if (Kind = 'disk') or (Kind = 'gaussian') or (Kind = 'zgaussian') then
  begin
    R := Param('Optode.Source.Param1', 0);
    if R <= 0 then R := Scale * 0.15;
    FMesh.AddDisk(Pos, Dir, R, C, SrcAlpha);
  end
  else if Kind = 'ring' then
  begin
    { Two radii: the outer as a disk, the inner drawn in the background
      colour would need a second pass, so the inner edge is a ring of line. }
    R := Param('Optode.Source.Param1', 0);
    if R <= 0 then R := Scale * 0.15;
    FMesh.AddDisk(Pos, Dir, R, C, SrcAlpha * 0.6);
    FLines.AddCircle(Pos, McxVec3Norm(McxVec3Cross(Dir, McxVec3(1, 0, 0))),
      McxVec3Norm(McxVec3Cross(Dir, McxVec3(0, 1, 0))),
      Param('Optode.Source.Param1', 1), C);
  end
  else if (Kind = 'planar') or (Kind = 'pattern') or (Kind = 'pattern3d') or
          (Kind = 'fourier') or (Kind = 'fourierx') or (Kind = 'fourierx2d') then
    { Param1 and Param2 are the two edge vectors from Pos. }
    FMesh.AddQuad(Pos,
      McxVec3(Pos.x + P1.x, Pos.y + P1.y, Pos.z + P1.z),
      McxVec3(Pos.x + P1.x + P2.x, Pos.y + P1.y + P2.y, Pos.z + P1.z + P2.z),
      McxVec3(Pos.x + P2.x, Pos.y + P2.y, Pos.z + P2.z), C, SrcAlpha * 0.7)
  else if (Kind = 'slit') or (Kind = 'line') then
    { A segment from Pos along Param1, drawn with enough body to see. }
    FMesh.AddCylinder(Pos,
      McxVec3(Pos.x + P1.x, Pos.y + P1.y, Pos.z + P1.z),
      Scale * 0.03, C, SrcAlpha)
  else if Kind = 'pencilarray' then
  begin
    { Param1 and Param2 are the two edge vectors, and Param1[3], Param2[3]
      how many pencils along each. }
    n := Round(Param('Optode.Source.Param1', 3));
    if n < 1 then n := 1;
    j := Round(Param('Optode.Source.Param2', 3));
    if j < 1 then j := 1;
    for i := 0 to n - 1 do
      for j := 0 to Round(Param('Optode.Source.Param2', 3)) - 1 do
        FMesh.AddSphere(McxVec3(
          Pos.x + P1.x * i / n + P2.x * j / n,
          Pos.y + P1.y * i / n + P2.y * j / n,
          Pos.z + P1.z * i / n + P2.z * j / n), Scale * 0.03, C, SrcAlpha);
  end;
end;

{ Builds the wireframe from the document: the domain box, a floor grid, the
  three axes and where the source sits.

  Rebuilt whole on every change, which is cheap because it is a few hundred
  line segments in one buffer -- and is the opposite of the old renderer,
  which created a scene-graph object per axis label and rebuilt some seven
  hundred of them on every repaint. }
procedure TMcxView.Redraw;
begin
  Invalidate;
end;

procedure TMcxView.Rebuild;
var
  dx, dy, dz, Step, Axis, t: Single;
  Grey, Faint: TMcxVec3;
  Key: string;
  i: Integer;
  P: TJSONData;

  function Dim(AIndex: Integer): Single;
  begin
    Result := FDoc.AsInt('Domain.Dim[' + IntToStr(AIndex) + ']', 60);
    if Result < 1 then Result := 1;
  end;

begin
  if FLines = nil then Exit;
  dx := Dim(0);
  dy := Dim(1);
  dz := Dim(2);

  Grey := McxVec3(0.75, 0.78, 0.82);
  Faint := McxVec3(0.30, 0.32, 0.35);

  FLines.Clear;
  FMesh.Clear;
  FPick.Clear;
  SetLength(FPicks, 0);
  FLines.AddBox(McxVec3(0, 0, 0), McxVec3(dx, dy, dz), Grey);

  { A grid on the z = 0 face, on the same rounded spacing the axes are
    graduated with -- so a line on the floor is a line you can read a number
    off, rather than a tenth of whatever the domain happens to be. }
  Step := McxNiceStep(dx);
  i := 1;
  while i * Step < dx do
  begin
    t := i * Step;
    FLines.Add(McxVec3(t, 0, 0), McxVec3(t, dy, 0), Faint);
    Inc(i);
  end;
  Step := McxNiceStep(dy);
  i := 1;
  while i * Step < dy do
  begin
    t := i * Step;
    FLines.Add(McxVec3(0, t, 0), McxVec3(dx, t, 0), Faint);
    Inc(i);
  end;

  { Axes at the origin corner, in the usual three colours, each labelled. }
  Axis := dx;
  if dy > Axis then Axis := dy;
  if dz > Axis then Axis := dz;
  FSceneSpan := Axis;
  FSceneLo := McxVec3(0, 0, 0);
  FSceneHi := McxVec3(dx, dy, dz);
  AddAxes(FSceneLo, FSceneHi);

  AddShapes;
  AddDomainVolume;
  AddMesh;

  { A mesh document has no grid, so the box and the camera come from where
    the mesh actually is rather than from a Dim that is not there. }
  if Length(FMeshFaces) > 0 then
  begin
    FLines.Clear;
    FLines.AddBox(McxVec3(FMeshLo.x, FMeshLo.y, FMeshLo.z),
                  McxVec3(FMeshHi.x, FMeshHi.y, FMeshHi.z), Grey);
    dx := FMeshHi.x - FMeshLo.x;
    dy := FMeshHi.y - FMeshLo.y;
    dz := FMeshHi.z - FMeshLo.z;
    { A mesh has no grid to hang them on, and it needs them more: its
      coordinates are whatever the mesh file says rather than a voxel count
      starting at zero. }
    Axis := dx;
    if dy > Axis then Axis := dy;
    if dz > Axis then Axis := dz;
    FSceneSpan := Axis;
    FSceneLo := McxVec3(FMeshLo.x, FMeshLo.y, FMeshLo.z);
    FSceneHi := McxVec3(FMeshHi.x, FMeshHi.y, FMeshHi.z);
    AddAxes(FSceneLo, FSceneHi);
    Key := Format('mesh %g %g %g', [dx, dy, dz]);
    if Key <> FSceneKey then
    begin
      FSceneKey := Key;
      FitView;
    end;
    Invalidate;
    Exit;
  end;

  AddSource;

  { Only re-aim the camera when the domain itself changed size.  Doing it on
    every edit would throw the view away every time a photon count was
    typed. }
  Key := Format('%g %g %g', [Dim(0), Dim(1), Dim(2)]);
  if Key <> FSceneKey then
  begin
    FSceneKey := Key;
    FitView;
  end;

  Invalidate;
end;


end.
