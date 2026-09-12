{ mcxstudio2 - the simulation document.

  The document is the parsed mcx input file and nothing else.  Every setting is
  read and written through a dotted path -- 'Session.Photons',
  'Optode.Source.Pos[0]', 'Shapes[3].Sphere.R' -- straight into the fpjson tree
  that was loaded from disk.  Saving is FormatJSON.  There is no second
  representation to keep in step, which is the whole point: the old GUI had
  four non-symmetric serialisers and adding one option meant editing all of
  them.

  Because the tree is the file, anything we do not understand survives a round
  trip untouched: the Help blocks the examples carry, keys a newer mcx has
  grown, and the mmc-only Mesh section.

  This unit deliberately does not use the LCL, so the round-trip tests run
  headless.  The binding table below is plain data for the same reason; the
  binder that walks it and touches controls lives in mcxmain. }
unit mcxdoc;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpjson, jsonparser, jsonscanner;

type
  { Which editor a setting gets, and how its text is turned back into JSON. }
  TMcxKind = (mkBool, mkInt, mkFloat, mkText, mkChoice, mkFlags,
              mkVec, mkTable, mkFile, mkCustom);

  { The wizard shows a strict subset of the expert form, so a field carries the
    lowest mode it appears in rather than a set. }
  TMcxLevel = (mlWizard, mlExpert);

  TMcxBackend  = (mbMCX, mbMCXCL, mbMMC, mbHIP);
  TMcxBackends = set of TMcxBackend;

  TMcxDomain   = (mdVoxel, mdShapes, mdMesh);
  TMcxDomains  = set of TMcxDomain;

  { One row of the binding table: a design-time control name on the left, a
    path into the document on the right.  Adding a setting is one row here plus
    one control in the designer.

    Ctl is the component Name, which the form designer already guarantees is
    unique -- unlike the old GUI, which hid the key in the first word of Hint
    and so could not translate a hint without breaking persistence.

    Backends and Domains empty mean "applies to all".  EnableIf names another
    path and the value it must hold, e.g. 'Session.DoAutoThread=0'. }
  TMcxBind = record
    Ctl      : string;
    Path     : string;
    Kind     : TMcxKind;
    Level    : TMcxLevel;
    Backends : TMcxBackends;
    Domains  : TMcxDomains;
    Min, Max : Double;
    Choices  : string;
    EnableIf : string;
  end;

const
  { Enumerations, kept next to the table that uses them.  tx3 is deliberately
    absent from OutputFormat: it is a bespoke GL texture dump that only the old
    viewer read, and mcxstudio2 renders from jnii and bnii instead.  jnii leads
    because it is what mcx itself defaults to. }
  ChoiceBackend     = 'mcx,mcxcl,mmc,mcx-hip';
  ChoiceDomainKind  = 'voxel,shapes,mesh';
  ChoiceOutFormat   = 'jnii,bnii,nii,mc2,hdr';

  { A flag set is a string of letters -- "DP", "RM" -- or the equivalent
    bitmask.  Each entry is the letter, a colon, and what it means, so the
    check group can be captioned without a second table. }
  FlagsDebug   = 'R:RNG,M:Photon trajectory,P:Progress bar,T:Trajectory only';
  FlagsSaveData= 'D:Detector ID,S:Scattering counts,P:Partial path lengths,' +
                 'M:Momentum transfer,X:Exit position,V:Exit direction,' +
                 'W:Initial weight';
  ChoiceOutType     = 'x,f,e,j,p,m,r';
  ChoiceMediaFormat = 'byte,short,integer,asgn_float,svmc,mixlabel,labelplus,' +
                      'muamus_float,muamus_half,asgn_byte,muamus_short';
  ChoiceSrcType     = 'pencil,isotropic,cone,gaussian,planar,pattern,pattern3d,' +
                      'fourier,arcsine,disk,fourierx,fourierx2d,zgaussian,line,' +
                      'slit,pencilarray,hyperboloid,ring';

  { The binding table.  One row per setting: the control the designer placed on
    the left, the path it writes on the right.  This is the only place a
    setting is declared -- there is no argv builder, no INI writer and no
    JSON-to-widget reader to keep in step with it.

    The paths were not transcribed by hand.  mcx tags every key it reads with
    its full dotted path as the second argument of FIND_JSON_KEY/FIND_JSON_OBJ,
    so the authoritative list comes straight out of the parser:

      grep -oE 'FIND_JSON_(KEY|OBJ)\("[A-Za-z0-9_]+", *"[A-Za-z0-9_.]+"' \
        src/mcx_utils.c | sed 's/.*, *"//;s/"$//' | sort -u

    Running that in CI and diffing it against this table turns the silent drift
    that left the old GUI years behind the engine into a build failure.

    Which is how Frequency, SrcNum and WaveLength come to be here: mcx has read
    them for years and the old GUI never grew a field for any of them, because
    doing so meant four coordinated edits and a form-designer session.

    A path beginning @run is not part of the simulation at all -- it is a
    runtime choice such as which device to use -- and is routed to a second
    document that is saved with the preferences rather than with the file. }
  Binds: array[0..41] of TMcxBind = (
    { -- Types ------------------------------------------------------------- }
    (Ctl:'rgBackend';    Path:'@run.backend';     Kind:mkChoice; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:ChoiceBackend;    EnableIf:''),
    (Ctl:'rgDomainKind'; Path:'@run.domainkind';  Kind:mkChoice; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:ChoiceDomainKind; EnableIf:''),
    (Ctl:'cbMediaFormat';Path:'Domain.MediaFormat';Kind:mkChoice; Level:mlExpert;
     Backends:[]; Domains:[mdVoxel]; Min:0; Max:0; Choices:ChoiceMediaFormat; EnableIf:''),

    { -- Forward ----------------------------------------------------------- }
    (Ctl:'edT0'; Path:'Forward.T0'; Kind:mkFloat; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:1; Choices:''; EnableIf:''),
    (Ctl:'edT1'; Path:'Forward.T1'; Kind:mkFloat; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:1; Choices:''; EnableIf:''),
    (Ctl:'edDt'; Path:'Forward.Dt'; Kind:mkFloat; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:1; Choices:''; EnableIf:''),

    { -- Session ----------------------------------------------------------- }
    (Ctl:'edSessionID'; Path:'Session.ID'; Kind:mkText; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'edPhotons'; Path:'Session.Photons'; Kind:mkFloat; Level:mlWizard;
     Backends:[]; Domains:[]; Min:1; Max:9.2e18; Choices:''; EnableIf:''),
    (Ctl:'rgOutFormat'; Path:'Session.OutputFormat'; Kind:mkChoice; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:ChoiceOutFormat; EnableIf:''),
    (Ctl:'edSeed'; Path:'Session.RNGSeed'; Kind:mkFloat; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'cbOutType'; Path:'Session.OutputType'; Kind:mkChoice; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:ChoiceOutType; EnableIf:''),
    (Ctl:'ckMismatch'; Path:'Session.DoMismatch'; Kind:mkBool; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'ckNormalize'; Path:'Session.DoNormalize'; Kind:mkBool; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'ckSaveVolume'; Path:'Session.DoSaveVolume'; Kind:mkBool; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'ckSaveDetp'; Path:'Session.DoPartialPath'; Kind:mkBool; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'ckSaveRef'; Path:'Session.DoSaveRef'; Kind:mkBool; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'ckSaveExit'; Path:'Session.DoSaveExit'; Kind:mkBool; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:'Session.DoPartialPath=1'),
    (Ctl:'ckSaveSeed'; Path:'Session.DoSaveSeed'; Kind:mkBool; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:'Session.DoPartialPath=1'),
    (Ctl:'ckSpecular'; Path:'Session.DoSpecular'; Kind:mkBool; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'ckDCS'; Path:'Session.DoDCS'; Kind:mkBool; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),

    { -- Properties -------------------------------------------------------- }
    (Ctl:'edDim'; Path:'Domain.Dim'; Kind:mkVec; Level:mlWizard;
     Backends:[]; Domains:[mdVoxel,mdShapes]; Min:3; Max:3; Choices:''; EnableIf:''),
    (Ctl:'edUnit'; Path:'Domain.LengthUnit'; Kind:mkFloat; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'ckOriginType'; Path:'Domain.OriginType'; Kind:mkBool; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'edVolumeFile'; Path:'Domain.VolumeFile'; Kind:mkFile; Level:mlExpert;
     Backends:[]; Domains:[mdVoxel]; Min:0; Max:0; Choices:''; EnableIf:''),

    { -- Optode ------------------------------------------------------------ }
    (Ctl:'cbSrcType'; Path:'Optode.Source.Type'; Kind:mkChoice; Level:mlWizard;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:ChoiceSrcType; EnableIf:''),
    (Ctl:'edSrcPos'; Path:'Optode.Source.Pos'; Kind:mkVec; Level:mlWizard;
     Backends:[]; Domains:[]; Min:3; Max:3; Choices:''; EnableIf:''),
    (Ctl:'edSrcDir'; Path:'Optode.Source.Dir'; Kind:mkVec; Level:mlWizard;
     Backends:[]; Domains:[]; Min:3; Max:4; Choices:''; EnableIf:''),
    (Ctl:'edSrcParam1'; Path:'Optode.Source.Param1'; Kind:mkVec; Level:mlExpert;
     Backends:[]; Domains:[]; Min:4; Max:4; Choices:''; EnableIf:''),
    (Ctl:'edSrcParam2'; Path:'Optode.Source.Param2'; Kind:mkVec; Level:mlExpert;
     Backends:[]; Domains:[]; Min:4; Max:4; Choices:''; EnableIf:''),
    (Ctl:'edSrcFreq'; Path:'Optode.Source.Frequency'; Kind:mkFloat; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'edSrcNum'; Path:'Optode.Source.SrcNum'; Kind:mkInt; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'edSrcWavelen'; Path:'Optode.Source.WaveLength'; Kind:mkFloat; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),

    { -- Compute ----------------------------------------------------------- }
    (Ctl:'ckAutoThread'; Path:'Session.DoAutoThread'; Kind:mkBool; Level:mlWizard;
     Backends:[mbMCX,mbMCXCL,mbHIP]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'edThread'; Path:'@run.nthread'; Kind:mkInt; Level:mlExpert;
     Backends:[mbMCX,mbMCXCL,mbHIP]; Domains:[]; Min:1; Max:0; Choices:'';
     EnableIf:'Session.DoAutoThread=0'),
    (Ctl:'edBlock'; Path:'@run.nblock'; Kind:mkInt; Level:mlExpert;
     Backends:[mbMCX,mbMCXCL,mbHIP]; Domains:[]; Min:1; Max:0; Choices:'';
     EnableIf:'Session.DoAutoThread=0'),
    (Ctl:'edWorkload'; Path:'@run.workload'; Kind:mkText; Level:mlExpert;
     Backends:[mbMCX,mbMCXCL,mbHIP]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),

    { -- Advanced ---------------------------------------------------------- }
    (Ctl:'edBC'; Path:'Session.BCFlags'; Kind:mkText; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'cgDebug'; Path:'Session.DebugFlag'; Kind:mkFlags; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:FlagsDebug; EnableIf:''),
    (Ctl:'cgSaveMask'; Path:'Session.SaveDataMask'; Kind:mkFlags; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:FlagsSaveData;
     EnableIf:'Session.DoPartialPath=1'),
    (Ctl:'edMaxDetp'; Path:'Session.MaxDetPhoton'; Kind:mkFloat; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'edMinEnergy'; Path:'Session.MinEnergy'; Kind:mkFloat; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:''),
    (Ctl:'edRootPath'; Path:'Session.RootPath'; Kind:mkText; Level:mlExpert;
     Backends:[]; Domains:[]; Min:0; Max:0; Choices:''; EnableIf:'')
  );

type
  { Raised only for programming errors -- a malformed path literal.  Bad input
    files come back as a False from LoadFromFile with LastError set. }
  EMcxPath = class(Exception);

  TMcxDoc = class
  private
    FRoot: TJSONObject;
    FFileName: string;
    FModified: Boolean;
    FLastError: string;
    function  Walk(const APath: string; ACreate: Boolean;
      ALeafType: TJSONtype; out AParent: TJSONData;
      out AKey: string; out AIndex: Integer): TJSONData;
  public
    constructor Create;
    destructor Destroy; override;

    procedure Clear;
    function  LoadFromString(const AText: string): Boolean;
    function  LoadFromFile(const AFileName: string): Boolean;
    function  SaveToFile(const AFileName: string): Boolean;
    function  ToJSON(APretty: Boolean = True): string;

    { Nil when the path does not exist.  Never creates. }
    function  Find(const APath: string): TJSONData;
    function  Exists(const APath: string): Boolean;

    { Creates every missing container along the way, choosing an object or an
      array from whether the next step is a name or an index, and creates the
      leaf as ALeafType.  Returns the leaf. }
    function  Ensure(const APath: string; ALeafType: TJSONtype): TJSONData;

    procedure Delete(const APath: string);

    function  AsStr  (const APath: string; const ADef: string = ''): string;
    function  AsNum  (const APath: string; ADef: Double = 0): Double;
    function  AsInt  (const APath: string; ADef: Int64 = 0): Int64;
    function  AsBool (const APath: string; ADef: Boolean = False): Boolean;

    { The setters preserve the type already in the file wherever the new value
      fits it.  mcx accepts "DoMismatch": 1 and "DoMismatch": true alike, and
      rewriting one as the other on every save would churn every example file
      the first time it was opened. }
    procedure SetStr (const APath, AValue: string);
    procedure SetNum (const APath: string; AValue: Double);
    procedure SetInt (const APath: string; AValue: Int64);
    procedure SetBool(const APath: string; AValue: Boolean);

    property Root: TJSONObject read FRoot;
    property FileName: string read FFileName write FFileName;
    property Modified: Boolean read FModified write FModified;
    property LastError: string read FLastError;
  end;

{ Compares two trees the way mcx would see them rather than byte for byte:
  object key order is ignored, 1 and true are the same value, and numbers match
  within a relative tolerance so that 5e-09 and 0.000000005 agree.  Returns
  True when equal; otherwise ADiff names the first path that differs. }
function McxJSONSame(A, B: TJSONData; out ADiff: string;
  ATol: Double = 1e-12): Boolean;

implementation

const
  { Enough to hold any exact integer a double can carry, so a whole-numbered
    float is written back as an integer rather than 1.0000000000000000E+000. }
  MaxExactInt = Int64(1) shl 53;

{ ---------------------------------------------------------------- paths ---- }

type
  TMcxStep = record
    Name  : string;    { '' when this is an index step }
    Index : Integer;   { -1 when this is a name step   }
  end;
  TMcxSteps = array of TMcxStep;

{ 'Optode.Source.Pos[0]' -> Optode / Source / Pos / [0].  A leading '@' is kept
  as part of the first name so that the runtime-only '@run.*' paths, which live
  in a second document, parse with the same code. }
function ParsePath(const APath: string): TMcxSteps;
var
  i, n: Integer;
  Cur: string;

  procedure PushName;
  begin
    if Cur = '' then Exit;
    SetLength(Result, Length(Result) + 1);
    Result[High(Result)].Name := Cur;
    Result[High(Result)].Index := -1;
    Cur := '';
  end;

begin
  Result := nil;
  Cur := '';
  i := 1;
  while i <= Length(APath) do
  begin
    case APath[i] of
      '.':
        begin
          PushName;
          Inc(i);
        end;
      '[':
        begin
          PushName;
          Inc(i);
          n := 0;
          if (i > Length(APath)) or not (APath[i] in ['0'..'9']) then
            raise EMcxPath.CreateFmt('malformed index in path "%s"', [APath]);
          while (i <= Length(APath)) and (APath[i] in ['0'..'9']) do
          begin
            n := n * 10 + (Ord(APath[i]) - Ord('0'));
            Inc(i);
          end;
          if (i > Length(APath)) or (APath[i] <> ']') then
            raise EMcxPath.CreateFmt('unterminated index in path "%s"', [APath]);
          Inc(i);
          SetLength(Result, Length(Result) + 1);
          Result[High(Result)].Name := '';
          Result[High(Result)].Index := n;
        end;
    else
      Cur := Cur + APath[i];
      Inc(i);
    end;
  end;
  PushName;
  if Length(Result) = 0 then
    raise EMcxPath.CreateFmt('empty path "%s"', [APath]);
end;

function NewOfType(AType: TJSONtype): TJSONData;
begin
  case AType of
    jtObject:  Result := TJSONObject.Create;
    jtArray:   Result := TJSONArray.Create;
    jtString:  Result := TJSONString.Create('');
    jtNumber:  Result := TJSONIntegerNumber.Create(0);
    jtBoolean: Result := TJSONBoolean.Create(False);
  else
    Result := TJSONNull.Create;
  end;
end;

{ ------------------------------------------------------------- TMcxDoc ---- }

constructor TMcxDoc.Create;
begin
  inherited Create;
  FRoot := TJSONObject.Create;
end;

destructor TMcxDoc.Destroy;
begin
  FreeAndNil(FRoot);
  inherited Destroy;
end;

procedure TMcxDoc.Clear;
begin
  FreeAndNil(FRoot);
  FRoot := TJSONObject.Create;
  FFileName := '';
  FModified := False;
  FLastError := '';
end;

function TMcxDoc.LoadFromString(const AText: string): Boolean;
var
  P: TJSONParser;
  D: TJSONData;
begin
  Result := False;
  FLastError := '';
  { joStrict is deliberately absent.  Thirteen of mcx's own example files wrap
    a Help string across several lines, and a raw newline inside a string is
    exactly what strict JSON forbids -- with joStrict the parser rejects
    qtest.json, colin27.json, digimouse.json and the whole srcbenchmark set.
    Comments and trailing commas turn up in hand-written inputs for the same
    reason.  Refusing to open mcx's own examples would be far worse than
    accepting them.

    Such a file is written back out with the newline escaped, so saving it once
    makes it strictly valid.  The value is unchanged either way. }
  P := TJSONParser.Create(AText, [joUTF8, joComments, joIgnoreTrailingComma]);
  try
    try
      D := P.Parse;
    except
      on E: Exception do
      begin
        FLastError := E.Message;
        Exit;
      end;
    end;
    if not (D is TJSONObject) then
    begin
      FLastError := 'the document root is not a JSON object';
      D.Free;
      Exit;
    end;
    FreeAndNil(FRoot);
    FRoot := TJSONObject(D);
    FModified := False;
    Result := True;
  finally
    P.Free;
  end;
end;

function TMcxDoc.LoadFromFile(const AFileName: string): Boolean;
var
  S: TStringList;
begin
  Result := False;
  FLastError := '';
  S := TStringList.Create;
  try
    try
      S.LoadFromFile(AFileName);
    except
      on E: Exception do
      begin
        FLastError := E.Message;
        Exit;
      end;
    end;
    Result := LoadFromString(S.Text);
    if Result then FFileName := AFileName;
  finally
    S.Free;
  end;
end;

function TMcxDoc.ToJSON(APretty: Boolean): string;
begin
  if APretty then
    Result := FRoot.FormatJSON([], 2)
  else
    Result := FRoot.AsJSON;
end;

function TMcxDoc.SaveToFile(const AFileName: string): Boolean;
var
  S: TStringList;
begin
  Result := False;
  FLastError := '';
  S := TStringList.Create;
  try
    S.Text := ToJSON(True);
    try
      S.SaveToFile(AFileName);
    except
      on E: Exception do
      begin
        FLastError := E.Message;
        Exit;
      end;
    end;
    FFileName := AFileName;
    FModified := False;
    Result := True;
  finally
    S.Free;
  end;
end;

{ The one walker behind Find, Ensure and the setters.  AParent, AKey and
  AIndex describe where the leaf sits, so a setter can replace it in place
  rather than mutating a node whose type is wrong. }
function TMcxDoc.Walk(const APath: string; ACreate: Boolean;
  ALeafType: TJSONtype; out AParent: TJSONData; out AKey: string;
  out AIndex: Integer): TJSONData;
var
  Steps: TMcxSteps;
  Cur, Nxt: TJSONData;
  i, j: Integer;
  WantType: TJSONtype;
begin
  Result := nil;
  AParent := nil;
  AKey := '';
  AIndex := -1;

  Steps := ParsePath(APath);
  Cur := FRoot;

  for i := 0 to High(Steps) do
  begin
    { What the container at this level has to be to hold this step. }
    if Steps[i].Name <> '' then
      WantType := jtObject
    else
      WantType := jtArray;

    if Cur.JSONType <> WantType then
    begin
      if not ACreate then Exit(nil);
      { A wrong-typed container on the way in is replaced, because the caller
        has told us what shape the path has and the file disagrees. }
      if AParent = nil then Exit(nil);
      if AParent.JSONType = jtObject then
      begin
        Nxt := NewOfType(WantType);
        TJSONObject(AParent).Elements[AKey] := Nxt;
        Cur := Nxt;
      end
      else
      begin
        Nxt := NewOfType(WantType);
        TJSONArray(AParent).Items[AIndex] := Nxt;
        Cur := Nxt;
      end;
    end;

    AParent := Cur;
    AKey := Steps[i].Name;
    AIndex := Steps[i].Index;

    if Steps[i].Name <> '' then
      Nxt := TJSONObject(Cur).Find(Steps[i].Name)
    else
    begin
      if Steps[i].Index < TJSONArray(Cur).Count then
        Nxt := TJSONArray(Cur).Items[Steps[i].Index]
      else
        Nxt := nil;
    end;

    if Nxt = nil then
    begin
      if not ACreate then Exit(nil);
      { The leaf gets the requested type; everything above it is whatever the
        next step needs to live in. }
      if i = High(Steps) then
        WantType := ALeafType
      else if Steps[i + 1].Name <> '' then
        WantType := jtObject
      else
        WantType := jtArray;

      Nxt := NewOfType(WantType);
      if Steps[i].Name <> '' then
        TJSONObject(Cur).Add(Steps[i].Name, Nxt)
      else
      begin
        { Arrays are dense, so a gap is padded with nulls rather than left
          for the caller to trip over. }
        for j := TJSONArray(Cur).Count to Steps[i].Index - 1 do
          TJSONArray(Cur).Add(TJSONNull.Create);
        TJSONArray(Cur).Add(Nxt);
      end;
    end;

    Cur := Nxt;
  end;

  Result := Cur;
end;

function TMcxDoc.Find(const APath: string): TJSONData;
var
  P: TJSONData;
  K: string;
  I: Integer;
begin
  Result := Walk(APath, False, jtUnknown, P, K, I);
end;

function TMcxDoc.Exists(const APath: string): Boolean;
begin
  Result := Find(APath) <> nil;
end;

function TMcxDoc.Ensure(const APath: string; ALeafType: TJSONtype): TJSONData;
var
  P: TJSONData;
  K: string;
  I: Integer;
begin
  Result := Walk(APath, True, ALeafType, P, K, I);
  FModified := True;
end;

procedure TMcxDoc.Delete(const APath: string);
var
  P: TJSONData;
  K: string;
  I: Integer;
begin
  if Walk(APath, False, jtUnknown, P, K, I) = nil then Exit;
  if P = nil then Exit;
  if (K <> '') and (P.JSONType = jtObject) then
    TJSONObject(P).Delete(TJSONObject(P).IndexOfName(K))
  else if (I >= 0) and (P.JSONType = jtArray) then
    TJSONArray(P).Delete(I);
  FModified := True;
end;

function TMcxDoc.AsStr(const APath: string; const ADef: string): string;
var
  D: TJSONData;
begin
  D := Find(APath);
  if (D = nil) or (D.JSONType in [jtNull, jtObject, jtArray]) then
    Result := ADef
  else
    Result := D.AsString;
end;

function TMcxDoc.AsNum(const APath: string; ADef: Double): Double;
var
  D: TJSONData;
begin
  D := Find(APath);
  if (D = nil) or not (D.JSONType in [jtNumber, jtBoolean, jtString]) then
    Exit(ADef);
  try
    Result := D.AsFloat;
  except
    Result := ADef;
  end;
end;

function TMcxDoc.AsInt(const APath: string; ADef: Int64): Int64;
var
  D: TJSONData;
begin
  D := Find(APath);
  if (D = nil) or not (D.JSONType in [jtNumber, jtBoolean, jtString]) then
    Exit(ADef);
  try
    Result := D.AsInt64;
  except
    Result := ADef;
  end;
end;

function TMcxDoc.AsBool(const APath: string; ADef: Boolean): Boolean;
var
  D: TJSONData;
begin
  D := Find(APath);
  if D = nil then Exit(ADef);
  case D.JSONType of
    jtBoolean: Result := D.AsBoolean;
    jtNumber:  Result := D.AsFloat <> 0;
    jtString:  Result := (D.AsString <> '') and (D.AsString <> '0');
  else
    Result := ADef;
  end;
end;

{ Puts ANew where the path points, replacing the node rather than mutating it,
  so a type change is possible when one is genuinely wanted. }
procedure Replace(AParent: TJSONData; const AKey: string; AIndex: Integer;
  ANew: TJSONData);
begin
  if (AKey <> '') and (AParent.JSONType = jtObject) then
    TJSONObject(AParent).Elements[AKey] := ANew
  else if (AIndex >= 0) and (AParent.JSONType = jtArray) then
    TJSONArray(AParent).Items[AIndex] := ANew
  else
    ANew.Free;
end;

procedure TMcxDoc.SetStr(const APath, AValue: string);
var
  P, D: TJSONData;
  K: string;
  I: Integer;
  V: Double;
  Fs: TFormatSettings;
begin
  D := Walk(APath, True, jtString, P, K, I);
  if (D <> nil) and (D.JSONType = jtString) and (D.AsString = AValue) then Exit;

  { Several settings are spelled either way by different files -- DebugFlag is
    "RM" in one example and 2 in another -- so a numeric key whose text still
    reads as the same number keeps its type.  Without this, showing a file in
    the form and saving it again would rewrite 2 as "2". }
  if (D <> nil) and (D.JSONType = jtNumber) then
  begin
    Fs := DefaultFormatSettings;
    Fs.DecimalSeparator := '.';
    if TryStrToFloat(Trim(AValue), V, Fs) and (V = D.AsFloat) then Exit;
  end;

  Replace(P, K, I, TJSONString.Create(AValue));
  FModified := True;
end;

procedure TMcxDoc.SetNum(const APath: string; AValue: Double);
var
  P, D: TJSONData;
  K: string;
  I: Integer;
  Whole: Boolean;
begin
  D := Walk(APath, True, jtNumber, P, K, I);
  Whole := (Frac(AValue) = 0) and (Abs(AValue) < MaxExactInt);

  if D <> nil then
    case D.JSONType of
      { The file says this key is a flag; 0 and 1 keep it one. }
      jtBoolean:
        if (AValue = 0) or (AValue = 1) then
        begin
          if D.AsBoolean = (AValue <> 0) then Exit;
          Replace(P, K, I, TJSONBoolean.Create(AValue <> 0));
          FModified := True;
          Exit;
        end;
      jtNumber:
        begin
          if D.AsFloat = AValue then Exit;
          { An integer key stays an integer unless the new value needs a
            fraction, so photon counts and dimensions do not sprout .0 }
          if (TJSONNumber(D).NumberType in [ntInteger, ntInt64]) and Whole then
          begin
            Replace(P, K, I, TJSONInt64Number.Create(Round(AValue)));
            FModified := True;
            Exit;
          end;
        end;
    end;

  if Whole and ((D = nil) or (D.JSONType <> jtNumber)) then
    Replace(P, K, I, TJSONInt64Number.Create(Round(AValue)))
  else
    Replace(P, K, I, TJSONFloatNumber.Create(AValue));
  FModified := True;
end;

procedure TMcxDoc.SetInt(const APath: string; AValue: Int64);
begin
  SetNum(APath, AValue);
end;

procedure TMcxDoc.SetBool(const APath: string; AValue: Boolean);
var
  P, D: TJSONData;
  K: string;
  I: Integer;
begin
  D := Walk(APath, True, jtBoolean, P, K, I);
  if D <> nil then
    case D.JSONType of
      jtBoolean:
        begin
          if D.AsBoolean = AValue then Exit;
          Replace(P, K, I, TJSONBoolean.Create(AValue));
          FModified := True;
          Exit;
        end;
      { Written as 0/1 in the file, so keep it that way. }
      jtNumber:
        begin
          if (D.AsFloat <> 0) = AValue then Exit;
          Replace(P, K, I, TJSONIntegerNumber.Create(Ord(AValue)));
          FModified := True;
          Exit;
        end;
    end;
  Replace(P, K, I, TJSONBoolean.Create(AValue));
  FModified := True;
end;

{ ------------------------------------------------------------ compare ----- }

function NumbersClose(A, B: Double; ATol: Double): Boolean;
var
  S: Double;
begin
  if A = B then Exit(True);
  S := Abs(A);
  if Abs(B) > S then S := Abs(B);
  if S = 0 then Exit(True);
  Result := Abs(A - B) / S <= ATol;
end;

function Boolish(A: TJSONData; out V: Boolean): Boolean;
begin
  Result := True;
  case A.JSONType of
    jtBoolean: V := A.AsBoolean;
    jtNumber:
      begin
        Result := (A.AsFloat = 0) or (A.AsFloat = 1);
        V := A.AsFloat <> 0;
      end;
  else
    Result := False;
    V := False;
  end;
end;

function SameNode(A, B: TJSONData; const APath: string; out ADiff: string;
  ATol: Double): Boolean; forward;

function SameObject(A, B: TJSONObject; const APath: string; out ADiff: string;
  ATol: Double): Boolean;
var
  i, j: Integer;
  N: string;
begin
  if A.Count <> B.Count then
  begin
    ADiff := Format('%s: %d keys versus %d', [APath, A.Count, B.Count]);
    Exit(False);
  end;
  for i := 0 to A.Count - 1 do
  begin
    N := A.Names[i];
    j := B.IndexOfName(N);
    if j < 0 then
    begin
      ADiff := APath + '.' + N + ': missing on the right';
      Exit(False);
    end;
    if not SameNode(A.Items[i], B.Items[j], APath + '.' + N, ADiff, ATol) then
      Exit(False);
  end;
  Result := True;
end;

function SameArray(A, B: TJSONArray; const APath: string; out ADiff: string;
  ATol: Double): Boolean;
var
  i: Integer;
begin
  if A.Count <> B.Count then
  begin
    ADiff := Format('%s: %d items versus %d', [APath, A.Count, B.Count]);
    Exit(False);
  end;
  for i := 0 to A.Count - 1 do
    if not SameNode(A.Items[i], B.Items[i],
      Format('%s[%d]', [APath, i]), ADiff, ATol) then Exit(False);
  Result := True;
end;

function SameNode(A, B: TJSONData; const APath: string; out ADiff: string;
  ATol: Double): Boolean;
var
  BA, BB: Boolean;
begin
  ADiff := '';
  if (A = nil) or (B = nil) then
  begin
    Result := A = B;
    if not Result then ADiff := APath + ': one side is absent';
    Exit;
  end;

  { 1 and true are the same setting to mcx, so they are the same here. }
  if (A.JSONType <> B.JSONType) and Boolish(A, BA) and Boolish(B, BB) then
  begin
    Result := BA = BB;
    if not Result then ADiff := APath + ': flag differs';
    Exit;
  end;

  if (A.JSONType = jtNumber) and (B.JSONType = jtNumber) then
  begin
    Result := NumbersClose(A.AsFloat, B.AsFloat, ATol);
    if not Result then
      ADiff := Format('%s: %s versus %s', [APath, A.AsString, B.AsString]);
    Exit;
  end;

  if A.JSONType <> B.JSONType then
  begin
    ADiff := Format('%s: type %d versus %d',
      [APath, Ord(A.JSONType), Ord(B.JSONType)]);
    Exit(False);
  end;

  case A.JSONType of
    jtObject: Result := SameObject(TJSONObject(A), TJSONObject(B), APath, ADiff, ATol);
    jtArray:  Result := SameArray(TJSONArray(A), TJSONArray(B), APath, ADiff, ATol);
    jtNull:   Result := True;
  else
    Result := A.AsString = B.AsString;
    if not Result then
      ADiff := Format('%s: "%s" versus "%s"', [APath, A.AsString, B.AsString]);
  end;
end;

function McxJSONSame(A, B: TJSONData; out ADiff: string; ATol: Double): Boolean;
begin
  Result := SameNode(A, B, '', ADiff, ATol);
end;

end.
