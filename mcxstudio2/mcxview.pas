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
  Classes, SysUtils, Controls, ExtCtrls, Graphics, fpjson,
  FPImage, FPWritePNG,
  OpenGLContext, GL, mcxdoc, mcxgl, mcxjd, mcxmesh;

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
    FVolume: TMcxVolume;
    FCube: TMcxCube;
    { What the volume is drawn as and through.  All uniforms: changing any of
      them is a repaint, not an upload. }
    FStyle: Integer;
    FOpacity: Single;
    FSteps: Single;
    FLogScale: Boolean;
    FClipLo, FClipHi: TMcxVec3;
    FVolLow, FVolHigh: Single;
    FCamera: TMcxCamera;
    { Latches, so a driver that cannot give us a core profile is reported
      once rather than on every repaint. }
    FReady: Boolean;
    FFailed: Boolean;
    FDragging: Boolean;
    FDidDrag: Boolean;
    FDragX, FDragY: Integer;
    FSceneKey: string;
    FOnLog: TMcxViewLog;
    FOnPick: TMcxPickEvent;
    procedure Say(const AText: string);
    procedure Build;
    function  Start: Boolean;
    procedure GLPaint(Sender: TObject);
    procedure GLMouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure GLMouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
    procedure GLMouseUp(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure GLMouseWheel(Sender: TObject; Shift: TShiftState;
      WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
    procedure AddShapes;
    procedure AddMesh;
    procedure RenderScene(AWidth, AHeight: Integer; AFlat: Boolean);
    function  PickAt(AX, AY: Integer): Integer;
    function  GetHasVolume: Boolean;
    procedure DrawVolume(const AMVP: TMcxMat4);
  public
    { AHost is a panel the designer placed; the GL control is created into it.
      See Build for why it is not placed there directly. }
    constructor Create(AHost: TWinControl);
    destructor Destroy; override;
    { Draws ADoc.  Cheap enough to call on every edit: it is a few hundred
      line segments in one buffer. }
    procedure Rebuild;
    { Puts a result on the card.  AArray is whatever mcxjd read out of a
      .jnii or .bnii; anything past the third dimension -- time gates, or
      the several outputs of one run -- is dropped to the first slice, which
      is what the old viewer showed too. }
    function ShowVolume(const AArray: TMcxArray): Boolean;
    procedure ClearVolume;
    { 0 for maximum intensity, 1 for accumulation. }
    property Style: Integer read FStyle write FStyle;
    property LogScale: Boolean read FLogScale write FLogScale;
    property HasVolume: Boolean read GetHasVolume;
    property Document: TMcxDoc read FDoc write FDoc;
    { Renders at any size into an offscreen target and writes a PNG. }
    function SaveImage(const AFileName: string; AWidth, AHeight: Integer): Boolean;
    property OnLog: TMcxViewLog read FOnLog write FOnLog;
    property OnPick: TMcxPickEvent read FOnPick write FOnPick;
  end;

implementation

constructor TMcxView.Create(AHost: TWinControl);
begin
  FHost := AHost;
  FStyle := 0;
  FOpacity := 0.25;
  FSteps := 192;
  FLogScale := True;
  FClipLo := McxVec3(0, 0, 0);
  FClipHi := McxVec3(1, 1, 1);
  Build;
end;

destructor TMcxView.Destroy;
begin
  FVolume.Free;
  FCube.Free;
  FMesh.Free;
  FPick.Free;
  FTarget.Free;
  FSolidShader.Free;
  FVolShader.Free;
  FShader.Free;
  FLines.Free;
  FCamera.Free;
  inherited Destroy;
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
begin
  FCamera := TMcxCamera.Create;
  FLines := TMcxLines.Create;
  FMesh := TMcxMesh.Create;
  FPick := TMcxMesh.Create;
  FSelected := -1;

  { Owned by the host panel, so the control goes when the form does and
    this class does not have to be a TComponent to own it. }
  FGL := TOpenGLControl.Create(FHost);
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

  Say('OpenGL: ' + McxGLDescribe);
  FReady := True;
  Result := True;
  Rebuild;
end;

function TMcxView.GetHasVolume: Boolean;
begin
  Result := (FVolume <> nil) and FVolume.Loaded;
end;

procedure TMcxView.ClearVolume;
begin
  FreeAndNil(FVolume);
  if FGL <> nil then FGL.Invalidate;
end;

function TMcxView.ShowVolume(const AArray: TMcxArray): Boolean;
var
  nx, ny, nz: Integer;
  i, n: Int64;
  Buf: array of Single;
  V, Lo, Hi: Double;
begin
  Result := False;
  if (FGL = nil) or not FGL.MakeCurrent then Exit;
  if not Start then Exit;

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
  for i := 0 to n - 1 do
  begin
    V := McxArrayValue(AArray, i);
    Buf[i] := V;
    if i = 0 then
    begin
      Lo := V;
      Hi := V;
    end
    else
    begin
      if V < Lo then Lo := V;
      if V > Hi then Hi := V;
    end;
  end;

  { Fluence spans many decades, so the window is set on the log of it and
    the shader takes the log too.  A flat volume falls back to its own
    range, which at least shows something. }
  if FLogScale and (Hi > 0) then
  begin
    FVolHigh := Ln(Hi);
    if Lo > 0 then FVolLow := Ln(Lo) else FVolLow := FVolHigh - 12;
    { Twelve decades of e is about five of ten, which is the span mcxcloud
      shows by default and about where a fluence map stops being noise. }
    if FVolHigh - FVolLow > 12 then FVolLow := FVolHigh - 12;
  end
  else
  begin
    FLogScale := False;
    FVolLow := Lo;
    FVolHigh := Hi;
    if FVolHigh <= FVolLow then FVolHigh := FVolLow + 1;
  end;

  if FVolume = nil then FVolume := TMcxVolume.Create;
  Result := FVolume.Upload(@Buf[0], nx, ny, nz, FVolLow, FVolHigh);
  if not Result then
  begin
    Say(Format('the card would not take a %dx%dx%d volume', [nx, ny, nz]));
    FreeAndNil(FVolume);
  end;
  if FGL <> nil then FGL.Invalidate;
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
  FVolShader.SetFloat('uSteps', FSteps);
  FVolShader.SetVec2('uClim', FVolLow, FVolHigh);
  FVolShader.SetInt('uStyle', FStyle);
  FVolShader.SetInt('uLog', Ord(FLogScale));
  FVolShader.SetInt('uVolume', 0);

  FVolume.Bind(0);
  FCube.Draw;

  glDepthMask(GL_TRUE);
  glDisable(GL_CULL_FACE);
end;

{ Draws the scene at a given size, into whatever target is bound.  Shared by
  the on-screen paint and by the offscreen one that saves a picture. }
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
  if (FGL = nil) or not FGL.MakeCurrent then Exit;
  if not Start then Exit;
  if FPick.Count = 0 then Exit;

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
begin
  Result := False;
  if (FGL = nil) or not FGL.MakeCurrent then Exit;
  if not Start then Exit;
  if AWidth < 1 then AWidth := FGL.Width;
  if AHeight < 1 then AHeight := FGL.Height;

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
  FGL.Invalidate;
end;

procedure TMcxView.RenderScene(AWidth, AHeight: Integer; AFlat: Boolean);
var
  MVP: TMcxMat4;
begin
  if AHeight < 1 then AHeight := 1;
  glViewport(0, 0, AWidth, AHeight);

  if AFlat then glClearColor(0, 0, 0, 1) else glClearColor(0.16, 0.16, 0.17, 1);
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

  { The volume last: it is translucent, so it has to go over the wireframe
    rather than under it. }
  DrawVolume(MVP);
end;

procedure TMcxView.GLPaint(Sender: TObject);
begin
  if not FGL.MakeCurrent then Exit;
  if not Start then
  begin
    { Nothing can be drawn, but the buffer still has to be cleared and shown
      or the control keeps whatever was behind it. }
    glClearColor(0.16, 0.16, 0.17, 1);
    glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT);
    FGL.SwapBuffers;
    Exit;
  end;
  RenderScene(FGL.Width, FGL.Height, False);
  FGL.SwapBuffers;
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
  FGL.Invalidate;
end;

procedure TMcxView.GLMouseUp(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
var
  Id: Integer;
begin
  FDragging := False;
  { A click, not the end of a drag: the camera moved is not a selection. }
  if (Button <> mbLeft) or FDidDrag then
  begin
    FDidDrag := False;
    Exit;
  end;

  Id := PickAt(X, Y);
  FSelected := Id;
  Rebuild;
  if Assigned(FOnPick) and (Id >= 0) and (Id < Length(FPicks)) then
    FOnPick(Self, FPicks[Id]);
end;

procedure TMcxView.GLMouseWheel(Sender: TObject; Shift: TShiftState;
  WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
begin
  FCamera.Zoom(WheelDelta / 120);
  FGL.Invalidate;
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
    if Length(FPicks) - 1 = FSelected then Result := SelectedAlpha
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

{ Builds the wireframe from the document: the domain box, a floor grid, the
  three axes and where the source sits.

  Rebuilt whole on every change, which is cheap because it is a few hundred
  line segments in one buffer -- and is the opposite of the old renderer,
  which created a scene-graph object per axis label and rebuilt some seven
  hundred of them on every repaint. }
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

  { A grid on the z = 0 face, ten lines each way whatever the size, so it
    reads as a floor rather than as a solid block of lines on a big domain. }
  Step := dx / 10;
  for i := 1 to 9 do
  begin
    t := i * Step;
    FLines.Add(McxVec3(t, 0, 0), McxVec3(t, dy, 0), Faint);
  end;
  Step := dy / 10;
  for i := 1 to 9 do
  begin
    t := i * Step;
    FLines.Add(McxVec3(0, t, 0), McxVec3(dx, t, 0), Faint);
  end;

  { Axes at the origin corner, in the usual three colours. }
  Axis := dx;
  if dy > Axis then Axis := dy;
  if dz > Axis then Axis := dz;
  Axis := Axis * 0.25;
  FLines.Add(McxVec3(0, 0, 0), McxVec3(Axis, 0, 0), McxVec3(0.90, 0.30, 0.25));
  FLines.Add(McxVec3(0, 0, 0), McxVec3(0, Axis, 0), McxVec3(0.35, 0.75, 0.35));
  FLines.Add(McxVec3(0, 0, 0), McxVec3(0, 0, Axis), McxVec3(0.35, 0.55, 0.95));

  AddShapes;
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
    Key := Format('mesh %g %g %g', [dx, dy, dz]);
    if Key <> FSceneKey then
    begin
      FSceneKey := Key;
      FCamera.Frame(McxVec3((FMeshLo.x + FMeshHi.x) / 2,
                            (FMeshLo.y + FMeshHi.y) / 2,
                            (FMeshLo.z + FMeshHi.z) / 2),
        Sqrt(dx * dx + dy * dy + dz * dz) / 2);
    end;
    if FGL <> nil then FGL.Invalidate;
    Exit;
  end;

  { The source, as a cross at its position.  Drawn from the document rather
    than from the form, so it is right whether the value was typed or came
    out of a file. }
  P := FDoc.Find('Optode.Source.Pos');
  if (P <> nil) and (P.JSONType = jtArray) and (P.Count >= 3) then
  begin
    t := Axis * 0.15;
    dx := P.Items[0].AsFloat;
    dy := P.Items[1].AsFloat;
    dz := P.Items[2].AsFloat;
    FLines.Add(McxVec3(dx - t, dy, dz), McxVec3(dx + t, dy, dz),
      McxVec3(1.0, 0.85, 0.25));
    FLines.Add(McxVec3(dx, dy - t, dz), McxVec3(dx, dy + t, dz),
      McxVec3(1.0, 0.85, 0.25));
    FLines.Add(McxVec3(dx, dy, dz - t), McxVec3(dx, dy, dz + t),
      McxVec3(1.0, 0.85, 0.25));
  end;

  { Only re-aim the camera when the domain itself changed size.  Doing it on
    every edit would throw the view away every time a photon count was
    typed. }
  Key := Format('%g %g %g', [Dim(0), Dim(1), Dim(2)]);
  if Key <> FSceneKey then
  begin
    FSceneKey := Key;
    FCamera.Frame(McxVec3(Dim(0) / 2, Dim(1) / 2, Dim(2) / 2),
      Sqrt(Dim(0) * Dim(0) + Dim(1) * Dim(1) + Dim(2) * Dim(2)) / 2);
  end;

  if FGL <> nil then FGL.Invalidate;
end;


end.
