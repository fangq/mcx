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
  OpenGLContext, GL, mcxdoc, mcxgl, mcxjd;

type
  { Somewhere to put a line that has nowhere else to go -- the GL version at
    startup, a shader that would not compile.  The view has no log of its
    own and should not grow one. }
  TMcxViewLog = procedure(Sender: TObject; const AText: string) of object;

  TMcxView = class
  private
    FHost: TWinControl;
    FDoc: TMcxDoc;
    FGL: TOpenGLControl;
    FShader: TMcxShader;
    FVolShader: TMcxShader;
    FLines: TMcxLines;
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
    FDragX, FDragY: Integer;
    FSceneKey: string;
    FOnLog: TMcxViewLog;
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
    property OnLog: TMcxViewLog read FOnLog write FOnLog;
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

procedure TMcxView.GLPaint(Sender: TObject);
var
  MVP: TMcxMat4;
  W, H: Integer;
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

  W := FGL.Width;
  H := FGL.Height;
  if H < 1 then H := 1;
  glViewport(0, 0, W, H);

  glClearColor(0.16, 0.16, 0.17, 1);
  glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  { Near and far follow the camera distance rather than being fixed, so a
    domain of sixty voxels and one of six hundred both keep their depth
    precision. }
  MVP := McxMat4Mul(
    McxMat4Perspective(45, W / H, FCamera.Distance * 0.01,
      FCamera.Distance * 10),
    FCamera.View);

  FShader.Use;
  FShader.SetMat4('uMVP', MVP);
  FLines.Draw;

  { The volume last: it is translucent, so it has to go over the wireframe
    rather than under it. }
  DrawVolume(MVP);

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
  FCamera.Orbit((FDragX - X) * 0.008, (Y - FDragY) * 0.008);
  FDragX := X;
  FDragY := Y;
  FGL.Invalidate;
end;

procedure TMcxView.GLMouseUp(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  FDragging := False;
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

  { A slab or layer is a pair or triple of bounds along one axis; it is drawn
    as the two faces it cuts the domain with. }
  procedure Slab(AAxis: Integer; ALo, AHi: Single; const AColour: TMcxVec3);
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
        Colour(T));
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
      FLines.AddBox(O, McxVec3(O.x + Size.x, O.y + Size.y, O.z + Size.z),
        Colour(Tag(Obj)));
    end
    else if SameText(Verb, 'Box') then
    begin
      O := Triplet(Obj, 'O');
      Size := Triplet(Obj, 'Size', False);
      FLines.AddBox(O, McxVec3(O.x + Size.x, O.y + Size.y, O.z + Size.z),
        Colour(Tag(Obj)));
    end
    else if SameText(Verb, 'Sphere') then
    begin
      { The documentation calls the centre C0 and the schema calls it O, and
        real files use both. }
      O := Triplet(Obj, 'O');
      if (Obj <> nil) and (Obj.FindPath('O') = nil) then O := Triplet(Obj, 'C0');
      FLines.AddSphere(O, Number(Obj, 'R', 1), Colour(Tag(Obj)));
    end
    else if SameText(Verb, 'Cylinder') then
    begin
      C0 := Triplet(Obj, 'C0');
      C1 := Triplet(Obj, 'C1');
      FLines.AddCylinder(C0, C1, Number(Obj, 'R', 1), Colour(Tag(Obj)));
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
