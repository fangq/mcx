{ mcxstudio2 - the 3-D view without OpenGL.

  The view draws through OpenGL 3.3 when it can.  When it cannot -- a remote
  X display, a virtual machine, a driver older than 3.3, or --software on the
  command line -- this draws the same picture on the processor: the same
  triangles and line segments mcxview builds into TMcxMesh and TMcxLines,
  read straight out of their buffers, and the volume ray-cast the GLSL
  shader in mcxgl does, ray for ray.

  The method is mimamo's (Mmm.SceneView, the figure widget of the mimamo
  project): one raster with a depth buffer, faces filled by edge functions,
  lines stepped a pixel at a time, and a volume marched front to back through
  a trilinear field.  What is drawn and in what order follows
  TMcxView.RenderScene exactly, so that switching renderers changes the speed
  and not the picture:

    1. the solids, translucent and two-sided: far faces then near faces,
       blended, neither tested nor written against depth -- what GL does with
       depth writes off and a freshly cleared buffer;
    2. the wireframe, the photon paths and the axis labels, depth-tested and
       depth-written;
    3. the volume, blended over everything, maximum intensity or accumulated.

  The volume is the cost of a frame, so its rows are split across every
  processor, and while the picture moves it is marched at every other pixel
  with a third of the steps -- the same trade the GL path makes. }
unit mcxsoft;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, mcxgl;

type
  { What the volume is and how it is drawn: the same values TMcxView hands
    the volume shader as uniforms. }
  TMcxSoftVolume = record
    Data: array of Single;      // x fastest, as mcx writes it
    Nx, Ny, Nz: Integer;
    Low, High: Single;          // the window, in log units under LogScale
    LogScale: Boolean;
    Style: Integer;             // 0 maximum intensity, 1 accumulate
    Map: Integer;               // 0 jet, 1 hot, 2 viridis, 3 cool, 4 grey
    Floor, Opacity, Steps: Single;
    ClipLo, ClipHi: TMcxVec3;   // 0..1 across each axis of the volume
  end;

  { A colour raster with a depth buffer, drawn into by the three passes. }
  TMcxSoftRaster = class
  private
    FW, FH: Integer;
    FCol: array of Single;      // r, g, b per pixel, 0..1
    FZ: array of Single;        // depth in normalised device units, -1..1
    FIds: array of LongWord;    // the picking pass: an identifier per pixel
    FMVP, FInv: TMcxMat4;
    { the volume pass, kept for the threads that share it }
    FVol: ^TMcxSoftVolume;
    FEye: TMcxVec3;
    FStride: Integer;
    FSteps: Single;
    FVX0, FVY0, FVX1, FVY1: Integer;
    procedure Plot(X, Y: Integer; const C: TMcxVec3; A: Single);
    procedure Triangle(const P: array of TMcxVec3; const C: TMcxVec3; A: Single;
      AId: LongWord; ADepth: Boolean);
    procedure MarchRows(AFrom, ATo: Integer);
  public
    { Starts a frame: the size, what it is cleared to, and the projection. }
    procedure Start(AW, AH: Integer; const ABack: TMcxVec3; const AMVP: TMcxMat4);
    { TMcxMesh's buffer (10 floats a vertex: position, normal, rgba), blended
      with the light of the solid shader.  ACull 1 drops the faces turned to
      the camera, -1 those turned away, 0 none. }
    procedure Solids(AData: PSingle; ACount: Integer; const ALight: TMcxVec3;
      ACull: Integer);
    { The picking pass: every triangle opaque, depth-tested, its colour the
      identifier the GL path packs into r + g*256 + b*65536. }
    procedure PickSolids(AData: PSingle; ACount: Integer);
    { TMcxLines' buffer (6 floats a vertex: position, rgb), vertices AFirst to
      AFirst+ACount-1, two to a segment, depth-tested and written. }
    procedure Lines(AData: PSingle; AFirst, ACount: Integer);
    { Ray-casts the volume over the frame.  ADraft: every other pixel, at a
      third of the steps, for a picture that is moving. }
    procedure Volume(var AVol: TMcxSoftVolume; const AEye: TMcxVec3; ADraft: Boolean);
    { The pixel at X, Y (top row first) as r, g, b bytes. }
    procedure Pixel(X, Y: Integer; out AR, AG, AB: Byte);
    function  PickId(X, Y: Integer): LongWord;
    property Width: Integer read FW;
    property Height: Integer read FH;
  end;

{ The processors the volume is split across. }
function McxSoftThreads: Integer;
{ The inverse of a 4x4 column-major matrix; identity if it has none. }
function McxMat4Inverse(const M: TMcxMat4): TMcxMat4;

implementation

var
  GThreads: Integer = 0;

function McxSoftThreads: Integer;
{$IFDEF LINUX}
var
  F: TStringList;
  i: Integer;
{$ENDIF}
begin
  if GThreads = 0 then
  begin
    GThreads := TThread.ProcessorCount;
    {$IFDEF LINUX}
    { FPC 3.2 answers 1 here on Linux whatever the machine has, so the
      processors the kernel lists are counted instead. }
    if GThreads <= 1 then
    begin
      F := TStringList.Create;
      try
        try
          F.LoadFromFile('/proc/cpuinfo');
          GThreads := 0;
          for i := 0 to F.Count - 1 do
            if Copy(F[i], 1, 9) = 'processor' then Inc(GThreads);
        except
          GThreads := 1;
        end;
      finally
        F.Free;
      end;
    end;
    {$ENDIF}
    if GThreads < 1 then GThreads := 1;
    if GThreads > 64 then GThreads := 64;
  end;
  Result := GThreads;
end;

function McxMat4Inverse(const M: TMcxMat4): TMcxMat4;
var
  A: array[0..3, 0..7] of Double;
  r, c, k, p: Integer;
  t, f: Double;
begin
  for r := 0 to 3 do
    for c := 0 to 3 do
    begin
      A[r, c] := M[c * 4 + r];
      if r = c then A[r, c + 4] := 1 else A[r, c + 4] := 0;
    end;
  for c := 0 to 3 do
  begin
    p := c;
    for r := c + 1 to 3 do
      if Abs(A[r, c]) > Abs(A[p, c]) then p := r;
    if Abs(A[p, c]) < 1e-30 then Exit(McxMat4Identity);
    if p <> c then
      for k := 0 to 7 do
      begin
        t := A[c, k]; A[c, k] := A[p, k]; A[p, k] := t;
      end;
    f := 1 / A[c, c];
    for k := 0 to 7 do A[c, k] := A[c, k] * f;
    for r := 0 to 3 do
      if r <> c then
      begin
        f := A[r, c];
        if f <> 0 then
          for k := 0 to 7 do A[r, k] := A[r, k] - f * A[c, k];
      end;
  end;
  for r := 0 to 3 do
    for c := 0 to 3 do
      Result[c * 4 + r] := A[r, c + 4];
end;

type
  TClipVert = record
    x, y, z, w: Single;
  end;

function Transform(const M: TMcxMat4; x, y, z: Single): TClipVert; inline;
begin
  Result.x := M[0] * x + M[4] * y + M[8] * z + M[12];
  Result.y := M[1] * x + M[5] * y + M[9] * z + M[13];
  Result.z := M[2] * x + M[6] * y + M[10] * z + M[14];
  Result.w := M[3] * x + M[7] * y + M[11] * z + M[15];
end;

{ A point between two clip-space vertices, at the near plane z = -w. }
function NearCut(const A, B: TClipVert): TClipVert;
var
  da, db, t: Single;
begin
  da := A.z + A.w;
  db := B.z + B.w;
  t := da / (da - db);
  Result.x := A.x + t * (B.x - A.x);
  Result.y := A.y + t * (B.y - A.y);
  Result.z := A.z + t * (B.z - A.z);
  Result.w := A.w + t * (B.w - A.w);
end;

{ ------------------------------------------------------------- raster --- }

procedure TMcxSoftRaster.Start(AW, AH: Integer; const ABack: TMcxVec3;
  const AMVP: TMcxMat4);
var
  i: Integer;
begin
  if AW < 1 then AW := 1;
  if AH < 1 then AH := 1;
  FW := AW;
  FH := AH;
  SetLength(FCol, FW * FH * 3);
  SetLength(FZ, FW * FH);
  SetLength(FIds, 0);
  for i := 0 to FW * FH - 1 do
  begin
    FCol[3 * i] := ABack.x;
    FCol[3 * i + 1] := ABack.y;
    FCol[3 * i + 2] := ABack.z;
    FZ[i] := 1;
  end;
  FMVP := AMVP;
  FInv := McxMat4Inverse(AMVP);
end;

procedure TMcxSoftRaster.Plot(X, Y: Integer; const C: TMcxVec3; A: Single);
var
  k: Integer;
begin
  k := 3 * (Y * FW + X);
  if A >= 1 then
  begin
    FCol[k] := C.x;
    FCol[k + 1] := C.y;
    FCol[k + 2] := C.z;
  end
  else
  begin
    FCol[k] := A * C.x + (1 - A) * FCol[k];
    FCol[k + 1] := A * C.y + (1 - A) * FCol[k + 1];
    FCol[k + 2] := A * C.z + (1 - A) * FCol[k + 2];
  end;
end;

{ One triangle in screen space: x, y in pixels with y down, z the depth.
  Filled at pixel centres by edge functions, so two triangles sharing an edge
  leave neither a crack nor a seam drawn twice -- which on a translucent face
  would show as a line of double density. }
procedure TMcxSoftRaster.Triangle(const P: array of TMcxVec3; const C: TMcxVec3;
  A: Single; AId: LongWord; ADepth: Boolean);
var
  X0, Y0, X1, Y1, X, Y, i: Integer;
  Area, W0, W1, W2, Z, PX, PY: Single;
begin
  Area := (P[1].x - P[0].x) * (P[2].y - P[0].y) - (P[2].x - P[0].x) * (P[1].y - P[0].y);
  if Abs(Area) < 1e-12 then Exit;
  X0 := Max(0, Floor(Min(P[0].x, Min(P[1].x, P[2].x))));
  X1 := Min(FW - 1, Ceil(Max(P[0].x, Max(P[1].x, P[2].x))));
  Y0 := Max(0, Floor(Min(P[0].y, Min(P[1].y, P[2].y))));
  Y1 := Min(FH - 1, Ceil(Max(P[0].y, Max(P[1].y, P[2].y))));
  for Y := Y0 to Y1 do
  begin
    PY := Y + 0.5;
    for X := X0 to X1 do
    begin
      PX := X + 0.5;
      W0 := ((P[2].x - P[1].x) * (PY - P[1].y) - (PX - P[1].x) * (P[2].y - P[1].y)) / Area;
      W1 := ((P[0].x - P[2].x) * (PY - P[2].y) - (PX - P[2].x) * (P[0].y - P[2].y)) / Area;
      W2 := 1 - W0 - W1;
      if (W0 < 0) or (W1 < 0) or (W2 < 0) then Continue;
      i := Y * FW + X;
      if ADepth then
      begin
        Z := W0 * P[0].z + W1 * P[1].z + W2 * P[2].z;
        if Z >= FZ[i] then Continue;
        FZ[i] := Z;
        FIds[i] := AId;
      end
      else
        Plot(X, Y, C, A);
    end;
  end;
end;

{ Clips a clip-space triangle to the near plane and hands what is left, as
  one or two triangles in screen space, to AEmit. }
type
  TScreenTri = array[0..2] of TMcxVec3;
  TScreenTris = array of TScreenTri;

function ToScreen(const V: TClipVert; AW, AH: Integer): TMcxVec3; inline;
begin
  Result.x := (V.x / V.w * 0.5 + 0.5) * AW;
  Result.y := (0.5 - V.y / V.w * 0.5) * AH;
  Result.z := V.z / V.w;
end;

function ClipTriangle(const A, B, C: TClipVert; AW, AH: Integer;
  out ATris: TScreenTris): Integer;
var
  Inp, Outp: array[0..3] of TClipVert;
  n, i, j: Integer;
  Cur, Nxt: TClipVert;
  CurIn, NxtIn: Boolean;
begin
  Inp[0] := A; Inp[1] := B; Inp[2] := C;
  n := 0;
  for i := 0 to 2 do
  begin
    Cur := Inp[i];
    Nxt := Inp[(i + 1) mod 3];
    CurIn := Cur.z + Cur.w > 0;
    NxtIn := Nxt.z + Nxt.w > 0;
    if CurIn then
    begin
      Outp[n] := Cur;
      Inc(n);
    end;
    if CurIn <> NxtIn then
    begin
      Outp[n] := NearCut(Cur, Nxt);
      Inc(n);
    end;
  end;
  Result := 0;
  if n < 3 then
  begin
    SetLength(ATris, 0);
    Exit;
  end;
  SetLength(ATris, n - 2);
  for j := 0 to n - 3 do
  begin
    ATris[j][0] := ToScreen(Outp[0], AW, AH);
    ATris[j][1] := ToScreen(Outp[j + 1], AW, AH);
    ATris[j][2] := ToScreen(Outp[j + 2], AW, AH);
  end;
  Result := n - 2;
end;

procedure TMcxSoftRaster.Solids(AData: PSingle; ACount: Integer;
  const ALight: TMcxVec3; ACull: Integer);
var
  t, k, m: Integer;
  V: array[0..2] of TClipVert;
  Tris: TScreenTris;
  P: PSingle;
  N, C: TMcxVec3;
  Area, d, Shade, Alpha: Single;
begin
  if (AData = nil) or (ACount < 3) then Exit;
  for t := 0 to ACount div 3 - 1 do
  begin
    P := AData + t * 30;
    for k := 0 to 2 do
      V[k] := Transform(FMVP, P[10 * k], P[10 * k + 1], P[10 * k + 2]);
    m := ClipTriangle(V[0], V[1], V[2], FW, FH, Tris);
    if m = 0 then Continue;
    { Which way it faces, from its winding on screen; y runs down here, so a
      face counter-clockwise in GL's window has a negative area. }
    Area := (Tris[0][1].x - Tris[0][0].x) * (Tris[0][2].y - Tris[0][0].y) -
            (Tris[0][2].x - Tris[0][0].x) * (Tris[0][1].y - Tris[0][0].y);
    if (ACull = 1) and (Area < 0) then Continue;    // facing the camera
    if (ACull = -1) and (Area > 0) then Continue;   // facing away
    { The solid shader: two-sided, ambient plus diffuse from the eye. }
    N := McxVec3(P[3], P[4], P[5]);
    d := Sqrt(N.x * N.x + N.y * N.y + N.z * N.z);
    if d > 0 then
      d := Abs((N.x * ALight.x + N.y * ALight.y + N.z * ALight.z) / d);
    Shade := 0.38 + 0.62 * d;
    C := McxVec3(P[6] * Shade, P[7] * Shade, P[8] * Shade);
    Alpha := P[9];
    for k := 0 to m - 1 do
      Triangle(Tris[k], C, Alpha, 0, False);
  end;
end;

procedure TMcxSoftRaster.PickSolids(AData: PSingle; ACount: Integer);
var
  t, k, m, i: Integer;
  V: array[0..2] of TClipVert;
  Tris: TScreenTris;
  P: PSingle;
  Id: LongWord;
begin
  SetLength(FIds, FW * FH);
  for i := 0 to FW * FH - 1 do
  begin
    FIds[i] := 0;
    FZ[i] := 1;
  end;
  if (AData = nil) or (ACount < 3) then Exit;
  for t := 0 to ACount div 3 - 1 do
  begin
    P := AData + t * 30;
    Id := LongWord(Round(P[6] * 255)) or (LongWord(Round(P[7] * 255)) shl 8) or
          (LongWord(Round(P[8] * 255)) shl 16);
    for k := 0 to 2 do
      V[k] := Transform(FMVP, P[10 * k], P[10 * k + 1], P[10 * k + 2]);
    m := ClipTriangle(V[0], V[1], V[2], FW, FH, Tris);
    for k := 0 to m - 1 do
      Triangle(Tris[k], McxVec3(0, 0, 0), 1, Id, True);
  end;
end;

procedure TMcxSoftRaster.Lines(AData: PSingle; AFirst, ACount: Integer);
var
  s, n, i, X, Y, k: Integer;
  A, B: TClipVert;
  P: PSingle;
  SA, SB: TMcxVec3;
  CA, CB, C: TMcxVec3;
  dx, dy, t, Z: Single;
  ta, tb: Single;
begin
  if (AData = nil) or (ACount < 2) then Exit;
  for s := 0 to ACount div 2 - 1 do
  begin
    P := AData + (AFirst + 2 * s) * 6;
    A := Transform(FMVP, P[0], P[1], P[2]);
    B := Transform(FMVP, P[6], P[7], P[8]);
    CA := McxVec3(P[3], P[4], P[5]);
    CB := McxVec3(P[9], P[10], P[11]);
    { The part in front of the near plane. }
    ta := A.z + A.w;
    tb := B.z + B.w;
    if (ta <= 0) and (tb <= 0) then Continue;
    if ta <= 0 then
    begin
      t := ta / (ta - tb);
      A := NearCut(A, B);
      CA := McxVec3(CA.x + t * (CB.x - CA.x), CA.y + t * (CB.y - CA.y), CA.z + t * (CB.z - CA.z));
    end
    else if tb <= 0 then
    begin
      t := tb / (tb - ta);
      B := NearCut(B, A);
      CB := McxVec3(CB.x + t * (CA.x - CB.x), CB.y + t * (CA.y - CB.y), CB.z + t * (CA.z - CB.z));
    end;
    SA := ToScreen(A, FW, FH);
    SB := ToScreen(B, FW, FH);
    dx := SB.x - SA.x;
    dy := SB.y - SA.y;
    n := Ceil(Max(Abs(dx), Abs(dy)));
    if n < 1 then n := 1;
    { A segment of thousands of pixels off the side of the pane is stepped
      only across the pane. }
    if n > 4 * (FW + FH) then n := 4 * (FW + FH);
    for i := 0 to n do
    begin
      t := i / n;
      X := Floor(SA.x + t * dx);
      Y := Floor(SA.y + t * dy);
      if (X < 0) or (Y < 0) or (X >= FW) or (Y >= FH) then Continue;
      Z := SA.z + t * (SB.z - SA.z);
      k := Y * FW + X;
      if Z >= FZ[k] then Continue;
      FZ[k] := Z;
      C := McxVec3(CA.x + t * (CB.x - CA.x), CA.y + t * (CB.y - CA.y), CA.z + t * (CB.z - CA.z));
      Plot(X, Y, C, 1);
    end;
  end;
end;

{ ------------------------------------------------------------- volume --- }

function Ramp(AMap: Integer; t: Single): TMcxVec3;

  function Clamp01(v: Single): Single; inline;
  begin
    if v < 0 then Result := 0 else if v > 1 then Result := 1 else Result := v;
  end;

begin
  t := Clamp01(t);
  case AMap of
    1: Result := McxVec3(Clamp01(t / 0.375), Clamp01((t - 0.375) / 0.375), Clamp01((t - 0.75) / 0.25));
    2: Result := McxVec3(
         Clamp01(0.2777 + t * (0.1051 + t * (-0.3309 + t * (-4.6342 + t * (6.2283 + t * (4.7764 + t * -5.4355)))))),
         Clamp01(0.0054 + t * (1.4046 + t * (0.2148 + t * (-5.7991 + t * (14.1799 + t * (-13.7451 + t * 4.6459)))))),
         Clamp01(0.3341 + t * (1.3846 + t * (0.0951 + t * (-19.3324 + t * (56.6906 + t * (-65.3530 + t * 26.3124)))))));
    3: Result := McxVec3(t, 1 - t, 1);
    4: Result := McxVec3(t, t, t);
  else
    Result := McxVec3(Clamp01(1.5 - Abs(4 * t - 3)), Clamp01(1.5 - Abs(4 * t - 2)),
      Clamp01(1.5 - Abs(4 * t - 1)));
  end;
end;

type
  TMcxBand = class(TThread)
  private
    FRaster: TMcxSoftRaster;
    FFrom, FTo: Integer;
  protected
    procedure Execute; override;
  public
    constructor Create(ARaster: TMcxSoftRaster; AFrom, ATo: Integer);
  end;

constructor TMcxBand.Create(ARaster: TMcxSoftRaster; AFrom, ATo: Integer);
begin
  FRaster := ARaster;
  FFrom := AFrom;
  FTo := ATo;
  FreeOnTerminate := False;
  inherited Create(False);
end;

procedure TMcxBand.Execute;
begin
  FRaster.MarchRows(FFrom, FTo);
end;

{ The rays of rows AFrom..ATo (in units of FStride), as the volume shader
  casts them: from the eye through the middle of the pixel, clipped to the
  box the volume fills, FSteps samples across it from a jittered start,
  sampled trilinearly with the edge clamped, and either the largest value
  kept or the colours accumulated front to back with the opacity corrected
  for the step.  Written into the raster blended as GL blends the shader's
  output: alpha over what is there. }
procedure TMcxSoftRaster.MarchRows(AFrom, ATo: Integer);
var
  V: ^TMcxSoftVolume;
  Nx, Ny, Nz, NxNy, Row, Col, X, Y, i, a, b: Integer;
  Lim: array[0..2] of Single;
  P0, P1, Dir, Org: array[0..2] of Double;
  W0, W1: TClipVert;
  NDX, NDY, tn, tf, t0, t1, Step, t, Jit, Best, s, Alpha, Aa, Inv: Double;
  Acc, Rc: TMcxVec3;
  Corr: Double;
  Miss: Boolean;
  LoC, HiC: array[0..2] of Double;
  Zd, Dd, Tz, Q, ZIn, TFull: Double;

  { the texture's value at a place in voxel units, as GL_LINEAR with
    GL_CLAMP_TO_EDGE: centres at half integers, the edge held }
  function Tex(QX, QY, QZ: Double): Double;
  var
    FX, FY, FZ, AX, AY, AZ: Double;
    I0, J0, K0, I1, J1, K1: Integer;
    D: PSingle;

    function At(I, J, K: Integer): Double; inline;
    begin
      Result := D[I + Nx * J + NxNy * K];
    end;

  begin
    D := @V^.Data[0];
    FX := EnsureRange(QX - 0.5, 0, Nx - 1);
    FY := EnsureRange(QY - 0.5, 0, Ny - 1);
    FZ := EnsureRange(QZ - 0.5, 0, Nz - 1);
    I0 := Trunc(FX); J0 := Trunc(FY); K0 := Trunc(FZ);
    I1 := Min(Nx - 1, I0 + 1); J1 := Min(Ny - 1, J0 + 1); K1 := Min(Nz - 1, K0 + 1);
    AX := FX - I0; AY := FY - J0; AZ := FZ - K0;
    Result :=
      ((At(I0, J0, K0) * (1 - AX) + At(I1, J0, K0) * AX) * (1 - AY) +
       (At(I0, J1, K0) * (1 - AX) + At(I1, J1, K0) * AX) * AY) * (1 - AZ) +
      ((At(I0, J0, K1) * (1 - AX) + At(I1, J0, K1) * AX) * (1 - AY) +
       (At(I0, J1, K1) * (1 - AX) + At(I1, J1, K1) * AX) * AY) * AZ;
  end;

  { the shader's sample1: outside the slab nothing, else the windowed value,
    dropped below the threshold }
  function Sample(QX, QY, QZ: Double): Double;
  begin
    if (QX < LoC[0]) or (QY < LoC[1]) or (QZ < LoC[2]) or
       (QX > HiC[0]) or (QY > HiC[1]) or (QZ > HiC[2]) then Exit(0);
    Result := Tex(QX, QY, QZ);
    if V^.LogScale then Result := Ln(Max(Result, 1e-12));
    Result := EnsureRange((Result - V^.Low) / Max(V^.High - V^.Low, 1e-12), 0, 1);
    if Result < V^.Floor then Result := 0;
  end;

  procedure Put(const C: TMcxVec3; A: Double);
  var
    k, dx, dy: Integer;
  begin
    for dy := 0 to FStride - 1 do
    begin
      if Y + dy >= FH then Break;
      for dx := 0 to FStride - 1 do
      begin
        if X + dx >= FW then Break;
        { A line or face nearer than where the ray enters the volume stays
          in front of it -- the near edges of the box, most of all. }
        if FZ[(Y + dy) * FW + X + dx] < ZIn then Continue;
        k := 3 * ((Y + dy) * FW + X + dx);
        FCol[k] := A * C.x + (1 - A) * FCol[k];
        FCol[k + 1] := A * C.y + (1 - A) * FCol[k + 1];
        FCol[k + 2] := A * C.z + (1 - A) * FCol[k + 2];
      end;
    end;
  end;

begin
  V := FVol;
  Nx := V^.Nx; Ny := V^.Ny; Nz := V^.Nz;
  NxNy := Nx * Ny;
  Lim[0] := Nx; Lim[1] := Ny; Lim[2] := Nz;
  LoC[0] := V^.ClipLo.x * Nx; LoC[1] := V^.ClipLo.y * Ny; LoC[2] := V^.ClipLo.z * Nz;
  HiC[0] := V^.ClipHi.x * Nx; HiC[1] := V^.ClipHi.y * Ny; HiC[2] := V^.ClipHi.z * Nz;
  Org[0] := FEye.x; Org[1] := FEye.y; Org[2] := FEye.z;
  Corr := 512 / FSteps;
  for Row := AFrom to ATo do
  begin
    Y := FVY0 + Row * FStride;
    if Y > FVY1 then Break;
    for Col := 0 to (FVX1 - FVX0) div FStride do
    begin
      X := FVX0 + Col * FStride;
      { the ray through the middle of this square, from the eye }
      NDX := (X + 0.5 * FStride) / FW * 2 - 1;
      NDY := 1 - (Y + 0.5 * FStride) / FH * 2;
      W0 := Transform(FInv, NDX, NDY, -1);
      W1 := Transform(FInv, NDX, NDY, 1);
      if (W0.w = 0) or (W1.w = 0) then Continue;
      P0[0] := W0.x / W0.w; P0[1] := W0.y / W0.w; P0[2] := W0.z / W0.w;
      P1[0] := W1.x / W1.w; P1[1] := W1.y / W1.w; P1[2] := W1.z / W1.w;
      for i := 0 to 2 do Dir[i] := P1[i] - P0[i];
      { the box the volume fills, by slabs; the eye is the start }
      tn := 0;
      tf := 1e30;
      Miss := False;
      for i := 0 to 2 do
      begin
        if Abs(Dir[i]) < 1e-12 then
        begin
          if (Org[i] < 0) or (Org[i] > Lim[i]) then Miss := True;
          Continue;
        end;
        Inv := 1 / Dir[i];
        t0 := (0 - Org[i]) * Inv;
        t1 := (Lim[i] - Org[i]) * Inv;
        if t0 > t1 then
        begin
          t := t0; t0 := t1; t1 := t;
        end;
        if t0 > tn then tn := t0;
        if t1 < tf then tf := t1;
      end;
      if Miss or (tf <= tn) then Continue;

      { The volume is composited with the depth the lines and faces left,
        as the GL path's depth test does, rather than painted over all of
        them.  A ray stops at the nearest thing drawn on its pixel, and
        nothing of the volume shows where that is in front of the box.
        Painted last and on top, the near edges vanished behind the volume,
        and with nothing else to say which way round the box is the eye
        read it inside out: dragging seemed to turn the volume against the
        box. }
      TFull := tf;
      Zd := FZ[Y * FW + X];
      if Zd < 1 then
      begin
        W0 := Transform(FInv, NDX, NDY, Zd);
        if W0.w <> 0 then
        begin
          Dd := 0;
          Tz := 0;
          for i := 0 to 2 do
          begin
            case i of
              0: Q := W0.x / W0.w;
              1: Q := W0.y / W0.w;
            else
              Q := W0.z / W0.w;
            end;
            Tz := Tz + (Q - Org[i]) * Dir[i];
            Dd := Dd + Dir[i] * Dir[i];
          end;
          if Dd > 0 then
          begin
            Tz := Tz / Dd;
            if Tz < tf then tf := Tz;
          end;
        end;
        if tf <= tn then Continue;
      end;
      W1 := Transform(FMVP, Org[0] + Dir[0] * tn, Org[1] + Dir[1] * tn,
        Org[2] + Dir[2] * tn);
      if W1.w > 1e-12 then ZIn := W1.z / W1.w else ZIn := -1;

      { Steps sized on the whole chord, so that a ray cut short by a line
        is sampled -- and its opacity corrected -- like its neighbours. }
      Step := (TFull - tn) / FSteps;
      { the shader's jitter, from the pixel, so the sample planes do not show }
      Jit := Frac(Sin((X + 0.5) * 12.9898 + (FH - Y - 0.5) * 78.233) * 43758.5453);
      if Jit < 0 then Jit := Jit + 1;
      t := tn + Step * Jit;
      if V^.Style = 0 then
      begin
        Best := 0;
        a := 0;
        while (t <= tf) and (a < 512) do
        begin
          s := Sample(Org[0] + Dir[0] * t, Org[1] + Dir[1] * t, Org[2] + Dir[2] * t);
          if s > Best then Best := s;
          t := t + Step;
          Inc(a);
        end;
        if Best > 0 then Put(Ramp(V^.Map, Best), 1);
      end
      else
      begin
        Acc := McxVec3(0, 0, 0);
        Alpha := 0;
        b := 0;
        while (t <= tf) and (Alpha <= 0.95) and (b < 512) do
        begin
          s := Sample(Org[0] + Dir[0] * t, Org[1] + Dir[1] * t, Org[2] + Dir[2] * t);
          if s > 0 then
          begin
            Aa := 1 - Power(1 - Min(s * V^.Opacity, 1), Corr);
            Rc := Ramp(V^.Map, s);
            Acc.x := Acc.x + (1 - Alpha) * Aa * Rc.x;
            Acc.y := Acc.y + (1 - Alpha) * Aa * Rc.y;
            Acc.z := Acc.z + (1 - Alpha) * Aa * Rc.z;
            Alpha := Alpha + (1 - Alpha) * Aa;
          end;
          t := t + Step;
          Inc(b);
        end;
        if Alpha > 0.001 then Put(Acc, Alpha);
      end;
    end;
  end;
end;

procedure TMcxSoftRaster.Volume(var AVol: TMcxSoftVolume; const AEye: TMcxVec3;
  ADraft: Boolean);
var
  Rows, n, i, Per: Integer;
  Bands: array of TMcxBand;
  C: TClipVert;
  sx, sy: Single;
  x, y, z: Integer;
  Behind: Boolean;
begin
  if (AVol.Nx < 1) or (AVol.Ny < 1) or (AVol.Nz < 1) or
     (Length(AVol.Data) < Int64(AVol.Nx) * AVol.Ny * AVol.Nz) then Exit;
  FVol := @AVol;
  FEye := AEye;
  if ADraft then
  begin
    FStride := 2;
    FSteps := Max(48, AVol.Steps / 3);
  end
  else
  begin
    FStride := 1;
    FSteps := Max(1, AVol.Steps);
  end;

  { Only the rectangle the box covers on screen; the whole pane if a corner
    is behind the eye, where a projection says nothing useful. }
  FVX0 := FW; FVY0 := FH; FVX1 := -1; FVY1 := -1;
  Behind := False;
  for i := 0 to 7 do
  begin
    x := (i and 1) * AVol.Nx;
    y := ((i shr 1) and 1) * AVol.Ny;
    z := ((i shr 2) and 1) * AVol.Nz;
    C := Transform(FMVP, x, y, z);
    if C.w <= 1e-6 then
    begin
      Behind := True;
      Break;
    end;
    sx := (C.x / C.w * 0.5 + 0.5) * FW;
    sy := (0.5 - C.y / C.w * 0.5) * FH;
    FVX0 := Min(FVX0, Floor(sx));
    FVX1 := Max(FVX1, Ceil(sx));
    FVY0 := Min(FVY0, Floor(sy));
    FVY1 := Max(FVY1, Ceil(sy));
  end;
  if Behind then
  begin
    FVX0 := 0; FVY0 := 0; FVX1 := FW - 1; FVY1 := FH - 1;
  end;
  FVX0 := Max(0, FVX0); FVY0 := Max(0, FVY0);
  FVX1 := Min(FW - 1, FVX1); FVY1 := Min(FH - 1, FVY1);
  if (FVX1 < FVX0) or (FVY1 < FVY0) then Exit;

  Rows := (FVY1 - FVY0) div FStride + 1;
  n := Min(McxSoftThreads, Rows);
  if n <= 1 then
  begin
    MarchRows(0, Rows - 1);
    Exit;
  end;
  { Bands of whole rows: each thread writes only its own rows of the raster,
    so they need nothing from each other. }
  Per := (Rows + n - 1) div n;
  SetLength(Bands, n);
  for i := 0 to n - 1 do
    Bands[i] := TMcxBand.Create(Self, i * Per, Min(Rows - 1, (i + 1) * Per - 1));
  for i := 0 to n - 1 do
  begin
    Bands[i].WaitFor;
    Bands[i].Free;
  end;
end;

procedure TMcxSoftRaster.Pixel(X, Y: Integer; out AR, AG, AB: Byte);
var
  k: Integer;
begin
  k := 3 * (Y * FW + X);
  AR := EnsureRange(Round(FCol[k] * 255), 0, 255);
  AG := EnsureRange(Round(FCol[k + 1] * 255), 0, 255);
  AB := EnsureRange(Round(FCol[k + 2] * 255), 0, 255);
end;

function TMcxSoftRaster.PickId(X, Y: Integer): LongWord;
begin
  Result := 0;
  if (X < 0) or (Y < 0) or (X >= FW) or (Y >= FH) or (Length(FIds) <> FW * FH) then Exit;
  Result := FIds[Y * FW + X];
end;

end.
