{ mcxstudio2 - the OpenGL plumbing: matrices, shaders, buffers, a camera.

  Nothing in here knows what a simulation is.  It is the layer mcxview draws
  the domain with, kept separate so that the geometry code never has to think
  about GL state and this file never has to think about JSON.

  Core profile, version 330, everywhere.  Not a preference: LCL's Cocoa
  context grants a strict 3.2+ core profile (glcocoanscontext.pas:258-263), in
  which glBegin, the matrix stack and gl_FragColor do not exist.  The GLX path
  never asks for a profile at all (glgtk3glxcontext.pas:679-697), so Linux
  will happily hand back a compatibility context and let sloppy code work
  here and fail on a Mac.  Writing core-only is the only way to find out on
  this machine.

  The version actually granted is logged at startup for the same reason. }
unit mcxgl;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, GL, GLext;

type
  TMcxVec3 = record
    x, y, z: Single;
  end;

  { Column-major, which is what glUniformMatrix4fv wants with transpose
    false -- so these can be handed straight to GL with no copy. }
  TMcxMat4 = array[0..15] of Single;

function McxVec3(x, y, z: Single): TMcxVec3;
function McxVec3Sub(const A, B: TMcxVec3): TMcxVec3;
function McxVec3Cross(const A, B: TMcxVec3): TMcxVec3;
function McxVec3Norm(const A: TMcxVec3): TMcxVec3;

function McxMat4Identity: TMcxMat4;
function McxMat4Mul(const A, B: TMcxMat4): TMcxMat4;
function McxMat4Perspective(AFovYDeg, AAspect, ANear, AFar: Single): TMcxMat4;
function McxMat4LookAt(const AEye, ACentre, AUp: TMcxVec3): TMcxMat4;
function McxMat4Translate(x, y, z: Single): TMcxMat4;
function McxMat4Scale(x, y, z: Single): TMcxMat4;

type
  { An orbit camera: a point to look at, and a direction and distance to look
    from.  A turntable rather than a free camera, because a domain is a box
    that people want to walk around, not fly through. }
  TMcxCamera = class
  private
    FTarget: TMcxVec3;
    FDistance: Single;
    FAzimuth: Single;
    FElevation: Single;
  public
    constructor Create;
    procedure Frame(const ACentre: TMcxVec3; ARadius: Single);
    procedure Orbit(ADx, ADy: Single);
    procedure Zoom(ASteps: Single);
    function View: TMcxMat4;
    function Eye: TMcxVec3;
    property Target: TMcxVec3 read FTarget write FTarget;
    property Distance: Single read FDistance write FDistance;
  end;

  { A compiled program, with the compiler's own complaint kept when it will
    not build.  A shader that fails silently is the worst kind: the window
    stays black and there is nothing to read. }
  TMcxShader = class
  private
    FProgram: GLuint;
    FError: string;
  public
    destructor Destroy; override;
    function Build(const AVertex, AFragment: string): Boolean;
    procedure Use;
    procedure SetMat4(const AName: string; const AValue: TMcxMat4);
    procedure SetVec3(const AName: string; const AValue: TMcxVec3);
    procedure SetFloat(const AName: string; AValue: Single);
    procedure SetVec2(const AName: string; A, B: Single);
    procedure SetInt(const AName: string; AValue: Integer);
    property Handle: GLuint read FProgram;
    property Error: string read FError;
  end;

  { A batch of coloured line segments in one buffer.

    Lines are all the domain preview needs -- a box, a grid, three axes, the
    outline of a shape -- and one buffer for all of them means one draw call
    however many there are.  The old renderer built a scene-graph object per
    label and rebuilt the lot on every repaint. }
  TMcxLines = class
  private
    FData: array of Single;
    FCount: Integer;
    FVAO, FVBO: GLuint;
    FDirty: Boolean;
  public
    destructor Destroy; override;
    procedure Clear;
    procedure Add(const A, B, AColour: TMcxVec3);
    procedure AddBox(const AMin, AMax, AColour: TMcxVec3);
    procedure AddCircle(const ACentre, AU, AV: TMcxVec3; ARadius: Single;
      const AColour: TMcxVec3);
    procedure AddSphere(const ACentre: TMcxVec3; ARadius: Single;
      const AColour: TMcxVec3);
    procedure AddCylinder(const AC0, AC1: TMcxVec3; ARadius: Single;
      const AColour: TMcxVec3);
    procedure Draw;
    property Count: Integer read FCount;
  end;

type
  { A scalar volume on the card, as one GL_R32F 3-D texture.

    The CPU touches the volume once, at upload.  Everything after -- the
    window on the values, the colour map, the clip planes, whether it is
    drawn as a maximum or as a surface -- is a uniform.  The old viewer
    re-uploaded on every tick of a cut-plane slider, after running a transfer
    function over the whole array in Pascal. }
  TMcxVolume = class
  private
    FTex: GLuint;
    FNx, FNy, FNz: Integer;
    FLow, FHigh: Single;
    FSmooth: Boolean;
  public
    destructor Destroy; override;
    { ADepth is x fastest, as mcx writes it.  ALow and AHigh are the range to
      map onto the colour scale. }
    function Upload(AData: PSingle; ANx, ANy, ANz: Integer;
      ALow, AHigh: Single): Boolean;
    procedure Bind(AUnit: Integer);
    property Loaded: Boolean read FSmooth write FSmooth;
    property Nx: Integer read FNx;
    property Ny: Integer read FNy;
    property Nz: Integer read FNz;
    property Low: Single read FLow;
    property High: Single read FHigh;
    property Handle: GLuint read FTex;
  end;

  { The unit cube the raycaster marches through: twelve triangles, back faces
    only, so that the fragment exists even when the camera is inside. }
  TMcxCube = class
  private
    FVAO, FVBO: GLuint;
  public
    destructor Destroy; override;
    procedure Draw;
  end;

const
  { The pair that draws TMcxLines.  Kept beside it, because the attribute
    locations here and the glVertexAttribPointer calls there have to agree and
    there is no third place that would make that obvious. }
  McxLineVertexShader =
    '#version 330 core'#10 +
    'layout(location = 0) in vec3 aPos;'#10 +
    'layout(location = 1) in vec3 aColour;'#10 +
    'uniform mat4 uMVP;'#10 +
    'out vec3 vColour;'#10 +
    'void main()'#10 +
    '{'#10 +
    '    vColour = aColour;'#10 +
    '    gl_Position = uMVP * vec4(aPos, 1.0);'#10 +
    '}'#10;

  McxLineFragmentShader =
    '#version 330 core'#10 +
    'in vec3 vColour;'#10 +
    'out vec4 oColour;'#10 +
    'void main()'#10 +
    '{'#10 +
    '    oColour = vec4(vColour, 1.0);'#10 +
    '}'#10;

  { Single-pass volume ray casting.

    The back faces are drawn and the entry point is worked out analytically,
    by intersecting the ray with the box.  The usual alternative renders the
    front faces into an FBO first and reads the entry point back from it;
    this needs no second pass and no attachment, and it keeps working when
    the camera is inside the volume, where the front faces are behind it.

    Borrowed from MRIcroGL's shader, which is BSD-2: the ray-start jitter
    that breaks up wood-grain banding, the opacity correction that keeps the
    apparent density the same when the step count changes, and stopping once
    the ray is opaque. }
  McxVolumeVertexShader =
    '#version 330 core'#10 +
    'layout(location = 0) in vec3 aPos;'#10 +
    'uniform mat4 uMVP;'#10 +
    'uniform vec3 uScale;'#10 +
    'out vec3 vPos;'#10 +
    'void main()'#10 +
    '{'#10 +
    '    vPos = aPos;'#10 +
    '    gl_Position = uMVP * vec4(aPos * uScale, 1.0);'#10 +
    '}'#10;

  McxVolumeFragmentShader =
    '#version 330 core'#10 +
    'in vec3 vPos;'#10 +
    'out vec4 oColour;'#10 +
    'uniform sampler3D uVolume;'#10 +
    'uniform vec3 uEye;'#10 +          { camera in volume coordinates }
    'uniform vec2 uClim;'#10 +
    'uniform vec3 uMinSlice;'#10 +
    'uniform vec3 uMaxSlice;'#10 +
    'uniform int  uStyle;'#10 +        { 0 maximum intensity, 1 accumulate }
    'uniform float uOpacity;'#10 +
    'uniform float uSteps;'#10 +
    'uniform int  uLog;'#10 +
    ''#10 +
    { A ramp with enough hue change to read small differences: blue through
      cyan and yellow to red, which is what mcxcloud shows. }
    'vec3 ramp(float t)'#10 +
    '{'#10 +
    '    t = clamp(t, 0.0, 1.0);'#10 +
    '    return clamp(vec3(1.5 - abs(4.0 * t - 3.0),'#10 +
    '                      1.5 - abs(4.0 * t - 2.0),'#10 +
    '                      1.5 - abs(4.0 * t - 1.0)), 0.0, 1.0);'#10 +
    '}'#10 +
    ''#10 +
    'float sample1(vec3 p)'#10 +
    '{'#10 +
    '    if (any(lessThan(p, uMinSlice)) || any(greaterThan(p, uMaxSlice)))'#10 +
    '        return 0.0;'#10 +
    '    float v = texture(uVolume, p).r;'#10 +
    '    if (uLog != 0) v = log(max(v, 1e-12));'#10 +
    '    return clamp((v - uClim.x) / max(uClim.y - uClim.x, 1e-12), 0.0, 1.0);'#10 +
    '}'#10 +
    ''#10 +
    'void main()'#10 +
    '{'#10 +
    '    vec3 dir = normalize(vPos - uEye);'#10 +
    { Slab intersection with the unit cube, so the ray starts at the face it
      actually enters through rather than at the camera. }
    '    vec3 inv = 1.0 / dir;'#10 +
    '    vec3 t0 = (vec3(0.0) - uEye) * inv;'#10 +
    '    vec3 t1 = (vec3(1.0) - uEye) * inv;'#10 +
    '    vec3 lo = min(t0, t1);'#10 +
    '    vec3 hi = max(t0, t1);'#10 +
    '    float tnear = max(max(lo.x, lo.y), lo.z);'#10 +
    '    float tfar  = min(min(hi.x, hi.y), hi.z);'#10 +
    '    tnear = max(tnear, 0.0);'#10 +
    '    if (tfar <= tnear) discard;'#10 +
    ''#10 +
    '    float step = (tfar - tnear) / uSteps;'#10 +
    { Start each ray a random fraction of a step in.  Without it the sample
      planes line up across the image and show as wood grain. }
    '    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233)))'#10 +
    '                         * 43758.5453);'#10 +
    '    float t = tnear + step * jitter;'#10 +
    ''#10 +
    '    if (uStyle == 0) {'#10 +
    '        float best = 0.0;'#10 +
    '        for (int i = 0; i < 512; i++) {'#10 +
    '            if (t > tfar) break;'#10 +
    '            best = max(best, sample1(uEye + dir * t));'#10 +
    '            t += step;'#10 +
    '        }'#10 +
    '        if (best <= 0.0) discard;'#10 +
    '        oColour = vec4(ramp(best), 1.0);'#10 +
    '        return;'#10 +
    '    }'#10 +
    ''#10 +
    '    vec3 acc = vec3(0.0);'#10 +
    '    float alpha = 0.0;'#10 +
    '    for (int i = 0; i < 512; i++) {'#10 +
    '        if (t > tfar || alpha > 0.95) break;'#10 +
    '        float s = sample1(uEye + dir * t);'#10 +
    { Opacity correction: without it, asking for more steps makes the same
      volume look denser. }
    '        float a = 1.0 - pow(1.0 - s * uOpacity, 512.0 / uSteps);'#10 +
    '        acc += (1.0 - alpha) * a * ramp(s);'#10 +
    '        alpha += (1.0 - alpha) * a;'#10 +
    '        t += step;'#10 +
    '    }'#10 +
    '    if (alpha <= 0.001) discard;'#10 +
    '    oColour = vec4(acc, alpha);'#10 +
    '}'#10;

{ Loads the entry points for the context that is current now.  Must be called
  with a context bound; returns False when the driver granted something older
  than 3.3, which is worth saying out loud rather than crashing later on a nil
  function pointer. }
function McxGLLoad: Boolean;

{ What the driver says it is, for the log. }
function McxGLDescribe: string;

implementation

const
  { FPC's OpenGL headers stop short of these two, so they are spelled out
    here.  Both are fixed by the specification and have been since 3.0. }
  GL_RED  = $1903;
  GL_R32F = $822E;


function McxVec3(x, y, z: Single): TMcxVec3;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function McxVec3Sub(const A, B: TMcxVec3): TMcxVec3;
begin
  Result := McxVec3(A.x - B.x, A.y - B.y, A.z - B.z);
end;

function McxVec3Cross(const A, B: TMcxVec3): TMcxVec3;
begin
  Result := McxVec3(A.y * B.z - A.z * B.y,
                    A.z * B.x - A.x * B.z,
                    A.x * B.y - A.y * B.x);
end;

function McxVec3Norm(const A: TMcxVec3): TMcxVec3;
var
  L: Single;
begin
  L := Sqrt(A.x * A.x + A.y * A.y + A.z * A.z);
  if L < 1e-12 then Exit(McxVec3(0, 0, 0));
  Result := McxVec3(A.x / L, A.y / L, A.z / L);
end;

function McxMat4Identity: TMcxMat4;
var
  i: Integer;
begin
  for i := 0 to 15 do Result[i] := 0;
  Result[0] := 1; Result[5] := 1; Result[10] := 1; Result[15] := 1;
end;

{ Result := A * B, with column-major storage: element (row, col) is at
  [col * 4 + row]. }
function McxMat4Mul(const A, B: TMcxMat4): TMcxMat4;
var
  r, c, k: Integer;
  S: Single;
begin
  for c := 0 to 3 do
    for r := 0 to 3 do
    begin
      S := 0;
      for k := 0 to 3 do S := S + A[k * 4 + r] * B[c * 4 + k];
      Result[c * 4 + r] := S;
    end;
end;

function McxMat4Perspective(AFovYDeg, AAspect, ANear, AFar: Single): TMcxMat4;
var
  f: Single;
  i: Integer;
begin
  for i := 0 to 15 do Result[i] := 0;
  f := 1 / Tan(AFovYDeg * Pi / 360);
  if AAspect <= 0 then AAspect := 1;
  Result[0] := f / AAspect;
  Result[5] := f;
  Result[10] := (AFar + ANear) / (ANear - AFar);
  Result[11] := -1;
  Result[14] := (2 * AFar * ANear) / (ANear - AFar);
end;

function McxMat4LookAt(const AEye, ACentre, AUp: TMcxVec3): TMcxMat4;
var
  f, s, u: TMcxVec3;
begin
  f := McxVec3Norm(McxVec3Sub(ACentre, AEye));
  s := McxVec3Norm(McxVec3Cross(f, AUp));
  u := McxVec3Cross(s, f);

  Result := McxMat4Identity;
  Result[0] := s.x;  Result[4] := s.y;  Result[8] := s.z;
  Result[1] := u.x;  Result[5] := u.y;  Result[9] := u.z;
  Result[2] := -f.x; Result[6] := -f.y; Result[10] := -f.z;
  Result[12] := -(s.x * AEye.x + s.y * AEye.y + s.z * AEye.z);
  Result[13] := -(u.x * AEye.x + u.y * AEye.y + u.z * AEye.z);
  Result[14] :=  (f.x * AEye.x + f.y * AEye.y + f.z * AEye.z);
end;

function McxMat4Translate(x, y, z: Single): TMcxMat4;
begin
  Result := McxMat4Identity;
  Result[12] := x;
  Result[13] := y;
  Result[14] := z;
end;

function McxMat4Scale(x, y, z: Single): TMcxMat4;
begin
  Result := McxMat4Identity;
  Result[0] := x;
  Result[5] := y;
  Result[10] := z;
end;

function McxGLLoad: Boolean;
begin
  Result := Load_GL_version_3_3_CORE();
end;

function McxGLDescribe: string;

  function Ask(AName: GLenum): string;
  var
    P: PChar;
  begin
    P := PChar(glGetString(AName));
    if P = nil then Result := '?' else Result := P;
  end;

begin
  Result := Ask(GL_RENDERER) + ' -- OpenGL ' + Ask(GL_VERSION) +
    ', GLSL ' + Ask(GL_SHADING_LANGUAGE_VERSION);
end;

{ TMcxVolume }

destructor TMcxVolume.Destroy;
begin
  if FTex <> 0 then glDeleteTextures(1, @FTex);
  inherited Destroy;
end;

function TMcxVolume.Upload(AData: PSingle; ANx, ANy, ANz: Integer;
  ALow, AHigh: Single): Boolean;
var
  Err: GLenum;
begin
  Result := False;
  if (AData = nil) or (ANx < 1) or (ANy < 1) or (ANz < 1) then Exit;
  FNx := ANx;
  FNy := ANy;
  FNz := ANz;
  FLow := ALow;
  FHigh := AHigh;

  if FTex = 0 then glGenTextures(1, @FTex);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, FTex);

  { Clamp to edge: a ray that steps a hair outside should see the face, not
    wrap round to the other side of the volume. }
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  { Rows are not padded: a volume whose x is not a multiple of four would
    otherwise be read with a gap at the end of every row. }
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  while glGetError() <> GL_NO_ERROR do ;          { clear anything stale }
  glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, ANx, ANy, ANz, 0, GL_RED,
    GL_FLOAT, AData);
  Err := glGetError();
  if Err <> GL_NO_ERROR then Exit;

  FSmooth := True;
  Result := True;
end;

procedure TMcxVolume.Bind(AUnit: Integer);
begin
  glActiveTexture(GL_TEXTURE0 + AUnit);
  glBindTexture(GL_TEXTURE_3D, FTex);
end;

{ TMcxCube }

destructor TMcxCube.Destroy;
begin
  if FVBO <> 0 then glDeleteBuffers(1, @FVBO);
  if FVAO <> 0 then glDeleteVertexArrays(1, @FVAO);
  inherited Destroy;
end;

procedure TMcxCube.Draw;
const
  { Twelve triangles over the unit cube, wound so the outside is
    counter-clockwise. }
  Verts: array[0..107] of Single = (
    0,0,0, 1,0,0, 1,1,0,  0,0,0, 1,1,0, 0,1,0,
    0,0,1, 1,1,1, 1,0,1,  0,0,1, 0,1,1, 1,1,1,
    0,0,0, 0,1,1, 0,0,1,  0,0,0, 0,1,0, 0,1,1,
    1,0,0, 1,0,1, 1,1,1,  1,0,0, 1,1,1, 1,1,0,
    0,0,0, 0,0,1, 1,0,1,  0,0,0, 1,0,1, 1,0,0,
    0,1,0, 1,1,1, 0,1,1,  0,1,0, 1,1,0, 1,1,1);
begin
  if FVAO = 0 then
  begin
    glGenVertexArrays(1, @FVAO);
    glGenBuffers(1, @FVBO);
    glBindVertexArray(FVAO);
    glBindBuffer(GL_ARRAY_BUFFER, FVBO);
    glBufferData(GL_ARRAY_BUFFER, SizeOf(Verts), @Verts[0], GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * SizeOf(Single), nil);
    glEnableVertexAttribArray(0);
  end;
  glBindVertexArray(FVAO);
  glDrawArrays(GL_TRIANGLES, 0, 36);
  glBindVertexArray(0);
end;

{ TMcxCamera }

constructor TMcxCamera.Create;
begin
  FTarget := McxVec3(0, 0, 0);
  FDistance := 3;
  FAzimuth := -0.9;
  FElevation := 0.5;
end;

{ Puts the whole of a sphere of ARadius about ACentre in view, with a little
  room around it. }
procedure TMcxCamera.Frame(const ACentre: TMcxVec3; ARadius: Single);
begin
  FTarget := ACentre;
  if ARadius <= 0 then ARadius := 1;
  FDistance := ARadius * 2.8;
end;

procedure TMcxCamera.Orbit(ADx, ADy: Single);
const
  { Just short of straight up: at exactly the pole the up vector and the view
    direction are parallel and the cross product that builds the view matrix
    collapses, which shows as the picture flipping over. }
  Limit = 1.5533;
begin
  FAzimuth := FAzimuth + ADx;
  FElevation := FElevation + ADy;
  if FElevation > Limit then FElevation := Limit;
  if FElevation < -Limit then FElevation := -Limit;
end;

procedure TMcxCamera.Zoom(ASteps: Single);
begin
  { Multiplicative, so that a notch of the wheel covers the same proportion of
    the distance whether the camera is near or far. }
  FDistance := FDistance * Exp(-ASteps * 0.15);
  if FDistance < 0.05 then FDistance := 0.05;
  if FDistance > 1e5 then FDistance := 1e5;
end;

function TMcxCamera.Eye: TMcxVec3;
begin
  Result.x := FTarget.x + FDistance * Cos(FElevation) * Cos(FAzimuth);
  Result.y := FTarget.y + FDistance * Cos(FElevation) * Sin(FAzimuth);
  Result.z := FTarget.z + FDistance * Sin(FElevation);
end;

function TMcxCamera.View: TMcxMat4;
begin
  { Z is up, because that is the axis mcx's domains are described along. }
  Result := McxMat4LookAt(Eye, FTarget, McxVec3(0, 0, 1));
end;

{ TMcxShader }

destructor TMcxShader.Destroy;
begin
  if FProgram <> 0 then glDeleteProgram(FProgram);
  inherited Destroy;
end;

function CompileStage(AKind: GLenum; const ASource: string;
  out AError: string): GLuint;
var
  Src: PChar;
  Status, Len: GLint;
  Log: string;
begin
  AError := '';
  Result := glCreateShader(AKind);
  Src := PChar(ASource);
  glShaderSource(Result, 1, @Src, nil);
  glCompileShader(Result);

  Status := 0;
  glGetShaderiv(Result, GL_COMPILE_STATUS, @Status);
  if Status <> 0 then Exit;

  Len := 0;
  glGetShaderiv(Result, GL_INFO_LOG_LENGTH, @Len);
  SetLength(Log, Len);
  if Len > 0 then glGetShaderInfoLog(Result, Len, @Len, @Log[1]);
  AError := Trim(Log);
  glDeleteShader(Result);
  Result := 0;
end;

function TMcxShader.Build(const AVertex, AFragment: string): Boolean;
var
  V, F: GLuint;
  Status, Len: GLint;
  Log: string;
begin
  FError := '';
  V := CompileStage(GL_VERTEX_SHADER, AVertex, FError);
  if V = 0 then
  begin
    FError := 'vertex shader: ' + FError;
    Exit(False);
  end;
  F := CompileStage(GL_FRAGMENT_SHADER, AFragment, FError);
  if F = 0 then
  begin
    glDeleteShader(V);
    FError := 'fragment shader: ' + FError;
    Exit(False);
  end;

  FProgram := glCreateProgram();
  glAttachShader(FProgram, V);
  glAttachShader(FProgram, F);
  glLinkProgram(FProgram);
  { Attached shaders live until the program is gone; flagging them now means
    they go with it and nothing has to remember them. }
  glDeleteShader(V);
  glDeleteShader(F);

  Status := 0;
  glGetProgramiv(FProgram, GL_LINK_STATUS, @Status);
  Result := Status <> 0;
  if Result then Exit;

  Len := 0;
  glGetProgramiv(FProgram, GL_INFO_LOG_LENGTH, @Len);
  SetLength(Log, Len);
  if Len > 0 then glGetProgramInfoLog(FProgram, Len, @Len, @Log[1]);
  FError := 'link: ' + Trim(Log);
  glDeleteProgram(FProgram);
  FProgram := 0;
end;

procedure TMcxShader.Use;
begin
  glUseProgram(FProgram);
end;

procedure TMcxShader.SetMat4(const AName: string; const AValue: TMcxMat4);
begin
  glUniformMatrix4fv(glGetUniformLocation(FProgram, PChar(AName)), 1,
    GL_FALSE, @AValue[0]);
end;

procedure TMcxShader.SetVec3(const AName: string; const AValue: TMcxVec3);
begin
  glUniform3f(glGetUniformLocation(FProgram, PChar(AName)),
    AValue.x, AValue.y, AValue.z);
end;

procedure TMcxShader.SetFloat(const AName: string; AValue: Single);
begin
  glUniform1f(glGetUniformLocation(FProgram, PChar(AName)), AValue);
end;

procedure TMcxShader.SetVec2(const AName: string; A, B: Single);
begin
  glUniform2f(glGetUniformLocation(FProgram, PChar(AName)), A, B);
end;

procedure TMcxShader.SetInt(const AName: string; AValue: Integer);
begin
  glUniform1i(glGetUniformLocation(FProgram, PChar(AName)), AValue);
end;

{ TMcxLines }

destructor TMcxLines.Destroy;
begin
  if FVBO <> 0 then glDeleteBuffers(1, @FVBO);
  if FVAO <> 0 then glDeleteVertexArrays(1, @FVAO);
  inherited Destroy;
end;

procedure TMcxLines.Clear;
begin
  SetLength(FData, 0);
  FCount := 0;
  FDirty := True;
end;

procedure TMcxLines.Add(const A, B, AColour: TMcxVec3);

  procedure Put(const P: TMcxVec3);
  var
    n: Integer;
  begin
    n := Length(FData);
    SetLength(FData, n + 6);
    FData[n] := P.x;       FData[n + 1] := P.y;       FData[n + 2] := P.z;
    FData[n + 3] := AColour.x; FData[n + 4] := AColour.y; FData[n + 5] := AColour.z;
  end;

begin
  Put(A);
  Put(B);
  Inc(FCount, 2);
  FDirty := True;
end;

procedure TMcxLines.AddBox(const AMin, AMax, AColour: TMcxVec3);

  procedure Edge(x1, y1, z1, x2, y2, z2: Single);
  begin
    Add(McxVec3(x1, y1, z1), McxVec3(x2, y2, z2), AColour);
  end;

begin
  { Four along x, four along y, four along z. }
  Edge(AMin.x, AMin.y, AMin.z, AMax.x, AMin.y, AMin.z);
  Edge(AMin.x, AMax.y, AMin.z, AMax.x, AMax.y, AMin.z);
  Edge(AMin.x, AMin.y, AMax.z, AMax.x, AMin.y, AMax.z);
  Edge(AMin.x, AMax.y, AMax.z, AMax.x, AMax.y, AMax.z);

  Edge(AMin.x, AMin.y, AMin.z, AMin.x, AMax.y, AMin.z);
  Edge(AMax.x, AMin.y, AMin.z, AMax.x, AMax.y, AMin.z);
  Edge(AMin.x, AMin.y, AMax.z, AMin.x, AMax.y, AMax.z);
  Edge(AMax.x, AMin.y, AMax.z, AMax.x, AMax.y, AMax.z);

  Edge(AMin.x, AMin.y, AMin.z, AMin.x, AMin.y, AMax.z);
  Edge(AMax.x, AMin.y, AMin.z, AMax.x, AMin.y, AMax.z);
  Edge(AMin.x, AMax.y, AMin.z, AMin.x, AMax.y, AMax.z);
  Edge(AMax.x, AMax.y, AMin.z, AMax.x, AMax.y, AMax.z);
end;

{ A circle in the plane spanned by AU and AV, which are assumed unit and
  perpendicular. }
procedure TMcxLines.AddCircle(const ACentre, AU, AV: TMcxVec3;
  ARadius: Single; const AColour: TMcxVec3);
const
  Segments = 48;
var
  i: Integer;
  a, b: Single;
  P, Q: TMcxVec3;
begin
  for i := 0 to Segments - 1 do
  begin
    a := 2 * Pi * i / Segments;
    b := 2 * Pi * (i + 1) / Segments;
    P := McxVec3(ACentre.x + ARadius * (Cos(a) * AU.x + Sin(a) * AV.x),
                 ACentre.y + ARadius * (Cos(a) * AU.y + Sin(a) * AV.y),
                 ACentre.z + ARadius * (Cos(a) * AU.z + Sin(a) * AV.z));
    Q := McxVec3(ACentre.x + ARadius * (Cos(b) * AU.x + Sin(b) * AV.x),
                 ACentre.y + ARadius * (Cos(b) * AU.y + Sin(b) * AV.y),
                 ACentre.z + ARadius * (Cos(b) * AU.z + Sin(b) * AV.z));
    Add(P, Q, AColour);
  end;
end;

{ Three great circles.  A wireframe sphere rather than a tessellated one:
  what a preview has to answer is where it is and how big, and three rings
  say that with 144 segments instead of a few thousand triangles. }
procedure TMcxLines.AddSphere(const ACentre: TMcxVec3; ARadius: Single;
  const AColour: TMcxVec3);
begin
  AddCircle(ACentre, McxVec3(1, 0, 0), McxVec3(0, 1, 0), ARadius, AColour);
  AddCircle(ACentre, McxVec3(1, 0, 0), McxVec3(0, 0, 1), ARadius, AColour);
  AddCircle(ACentre, McxVec3(0, 1, 0), McxVec3(0, 0, 1), ARadius, AColour);
end;

procedure TMcxLines.AddCylinder(const AC0, AC1: TMcxVec3; ARadius: Single;
  const AColour: TMcxVec3);
var
  Axis, U, V, Ref: TMcxVec3;
  i: Integer;
  a: Single;
  P, Q: TMcxVec3;
begin
  Axis := McxVec3Norm(McxVec3Sub(AC1, AC0));
  if (Axis.x = 0) and (Axis.y = 0) and (Axis.z = 0) then Exit;

  { Any vector not parallel to the axis will do to start the basis off; x
    unless the axis is x, in which case z. }
  if Abs(Axis.x) > 0.9 then Ref := McxVec3(0, 0, 1) else Ref := McxVec3(1, 0, 0);
  U := McxVec3Norm(McxVec3Cross(Axis, Ref));
  V := McxVec3Cross(Axis, U);

  AddCircle(AC0, U, V, ARadius, AColour);
  AddCircle(AC1, U, V, ARadius, AColour);

  { Four rules along the side, so the shape reads as a tube rather than as
    two loose rings. }
  for i := 0 to 3 do
  begin
    a := Pi * i / 2;
    P := McxVec3(AC0.x + ARadius * (Cos(a) * U.x + Sin(a) * V.x),
                 AC0.y + ARadius * (Cos(a) * U.y + Sin(a) * V.y),
                 AC0.z + ARadius * (Cos(a) * U.z + Sin(a) * V.z));
    Q := McxVec3(AC1.x + ARadius * (Cos(a) * U.x + Sin(a) * V.x),
                 AC1.y + ARadius * (Cos(a) * U.y + Sin(a) * V.y),
                 AC1.z + ARadius * (Cos(a) * U.z + Sin(a) * V.z));
    Add(P, Q, AColour);
  end;
end;

procedure TMcxLines.Draw;
begin
  if FCount = 0 then Exit;
  if FVAO = 0 then
  begin
    glGenVertexArrays(1, @FVAO);
    glGenBuffers(1, @FVBO);
  end;

  glBindVertexArray(FVAO);
  if FDirty then
  begin
    glBindBuffer(GL_ARRAY_BUFFER, FVBO);
    glBufferData(GL_ARRAY_BUFFER, Length(FData) * SizeOf(Single), @FData[0],
      GL_STATIC_DRAW);
    { Position at 0, colour at 1: the locations the shader declares, so no
      glBindAttribLocation and no lookup by name. }
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * SizeOf(Single), nil);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * SizeOf(Single),
      Pointer(3 * SizeOf(Single)));
    glEnableVertexAttribArray(1);
    FDirty := False;
  end;
  glDrawArrays(GL_LINES, 0, FCount);
  glBindVertexArray(0);
end;

end.
