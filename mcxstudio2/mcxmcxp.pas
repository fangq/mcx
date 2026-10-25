{ mcxstudio2 - reading the first Studio's projects (.mcxp).

  A .mcxp is an INI file with one section per session, holding what the old
  GUI's fields contained rather than what mcx reads.  This Studio has no
  format of its own -- a simulation is a .json and nothing else -- so a
  session is turned into the mcx input it stood for when it is opened, the
  same conversion utils/mcxp2json.py makes from the command line, and kept
  in step with it.

  What does not survive is what was never part of the simulation: thread
  and block counts, which GPU, the workload split, the remote-execution
  settings.  Those are how a run is launched, not what it computes; they
  are reported so nothing disappears silently.

  No LCL here, so the conversion can be tested headless. }
unit mcxmcxp;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpjson, jsonparser;

{ The session names in AFileName, in file order.  False if it cannot be read
  or holds none. }
function McxProjectSessions(const AFileName: string; ANames: TStrings): Boolean;

{ One session as mcx JSON text, or '' when there is no such session.  What
  had no place in the JSON is added to AWarn, one line each, if given. }
function McxProjectSession(const AFileName, ASession: string;
  AWarn: TStrings = nil): string;

implementation

const
  { .mcxp key -> JSON path, as mcxp2json.py's SCALARS. }
  Scalars: array[0..13] of array[0..1] of string = (
    ('PhotonNum',    'Session.Photons'),
    ('Seed',         'Session.RNGSeed'),
    ('DoReflect',    'Session.DoMismatch'),
    ('DoSave',       'Session.DoSaveVolume'),
    ('DoNormalize',  'Session.DoNormalize'),
    ('Autopilot',    'Session.DoAutoThread'),
    ('SaveDetector', 'Session.DoPartialPath'),
    ('DetectedNum',  'Session.MaxDetPhoton'),
    ('DoSaveRef',    'Session.DoSaveRef'),
    ('DoSaveExit',   'Session.DoSaveExit'),
    ('DoSaveSeed',   'Session.DoSaveSeed'),
    ('DoSpecular',   'Session.DoSpecular'),
    ('UnitInMM',     'Domain.LengthUnit'),
    ('DoSrcFrom0',   'Domain.OriginType'));

  { How a run is launched, not what it computes. }
  NotSimulation =
    ',threadnum,threadblock,gpuid,workload,respinnum,arrayorder,bubblesize,' +
    'gatenum,mcprogram,remotecmd,doremote,dosharedfs,moreparam,useatomic,' +
    'basicorder,debugphoton,doskipvoid,inputfile,';

  { The three debug switches, a character each. }
  DebugLetters = 'RMP';

{ The file as section -> list of key=value lines.  Read by hand rather than
  with TIniFile, which strips the quotes the MCXConfig rows depend on. }
function ReadProject(const AFileName: string; ASections: TStringList): Boolean;
var
  L: TStringList;
  i: Integer;
  S: string;
  Cur: TStringList;
begin
  Result := False;
  Cur := nil;
  L := TStringList.Create;
  try
    try
      L.LoadFromFile(AFileName);
    except
      Exit;
    end;
    for i := 0 to L.Count - 1 do
    begin
      S := Trim(L[i]);
      if (S = '') or (S[1] in [';', '#']) then Continue;
      if (S[1] = '[') and (S[Length(S)] = ']') then
      begin
        Cur := TStringList.Create;
        ASections.AddObject(Copy(S, 2, Length(S) - 2), Cur);
      end
      else if (Cur <> nil) and (Pos('=', S) > 1) then
        Cur.Add(S);
    end;
    Result := ASections.Count > 0;
  finally
    L.Free;
  end;
end;

procedure FreeSections(ASections: TStringList);
var
  i: Integer;
begin
  for i := 0 to ASections.Count - 1 do ASections.Objects[i].Free;
  ASections.Free;
end;

function McxProjectSessions(const AFileName: string; ANames: TStrings): Boolean;
var
  Secs: TStringList;
  i: Integer;
begin
  ANames.Clear;
  Secs := TStringList.Create;
  try
    Result := ReadProject(AFileName, Secs);
    for i := 0 to Secs.Count - 1 do ANames.Add(Secs[i]);
  finally
    FreeSections(Secs);
  end;
end;

{ A number if the text reads as one, otherwise the text.  A whole number
  comes back as an integer -- a photon count of 1e8 is counted -- and
  everything else keeps its value. }
function AsNumber(const AText: string): TJSONData;
var
  I: Int64;
  V: Double;
  Fs: TFormatSettings;
begin
  Fs := DefaultFormatSettings;
  Fs.DecimalSeparator := '.';
  if TryStrToInt64(Trim(AText), I) then Exit(TJSONInt64Number.Create(I));
  if TryStrToFloat(Trim(AText), V, Fs) then
  begin
    if (Frac(V) = 0) and (Abs(V) < 1e15) then
      Exit(TJSONInt64Number.Create(Round(V)));
    Exit(TJSONFloatNumber.Create(V));
  end;
  Result := TJSONString.Create(AText);
end;

{ Writes AValue at a dotted path, building the objects on the way. }
procedure Put(ADoc: TJSONObject; const APath: string; AValue: TJSONData);
var
  Parts: TStringList;
  Node: TJSONObject;
  i, k: Integer;
begin
  Parts := TStringList.Create;
  try
    Parts.Delimiter := '.';
    Parts.StrictDelimiter := True;
    Parts.DelimitedText := APath;
    Node := ADoc;
    for i := 0 to Parts.Count - 2 do
    begin
      k := Node.IndexOfName(Parts[i]);
      if (k < 0) or not (Node.Items[k] is TJSONObject) then
      begin
        if k >= 0 then Node.Delete(k);
        Node.Add(Parts[i], TJSONObject.Create);
        k := Node.IndexOfName(Parts[i]);
      end;
      Node := TJSONObject(Node.Items[k]);
    end;
    k := Node.IndexOfName(Parts[Parts.Count - 1]);
    if k >= 0 then Node.Delete(k);
    Node.Add(Parts[Parts.Count - 1], AValue);
  finally
    Parts.Free;
  end;
end;

{ One comma-separated row with Pascal's quoting: a field in double quotes,
  a doubled quote inside it standing for one. }
procedure SplitCsv(const ARow: string; AFields: TStrings);
var
  i: Integer;
  Cur: string;
  InQ: Boolean;
begin
  AFields.Clear;
  Cur := '';
  InQ := False;
  i := 1;
  while i <= Length(ARow) do
  begin
    if InQ then
    begin
      if ARow[i] = '"' then
      begin
        if (i < Length(ARow)) and (ARow[i + 1] = '"') then
        begin
          Cur := Cur + '"';
          Inc(i);
        end
        else
          InQ := False;
      end
      else
        Cur := Cur + ARow[i];
    end
    else if ARow[i] = '"' then
      InQ := True
    else if ARow[i] = ',' then
    begin
      AFields.Add(Cur);
      Cur := '';
    end
    else
      Cur := Cur + ARow[i];
    Inc(i);
  end;
  AFields.Add(Cur);
end;

{ The old GUI's '|' separated rows, blank ones dropped. }
procedure SplitRows(const AText: string; ARows: TStrings);
var
  L: TStringList;
  i: Integer;
begin
  ARows.Clear;
  L := TStringList.Create;
  try
    L.Delimiter := '|';
    L.StrictDelimiter := True;
    L.DelimitedText := AText;
    for i := 0 to L.Count - 1 do
      if Trim(L[i]) <> '' then ARows.Add(L[i]);
  finally
    L.Free;
  end;
end;

procedure ConvertTable(ADoc: TJSONObject; const AText: string; AMedia: Boolean);
var
  Rows, F: TStringList;
  Arr: TJSONArray;
  Obj: TJSONObject;
  Pos3: TJSONArray;
  r, c: Integer;

  function Field(AIndex: Integer): TJSONData;
  begin
    if (AIndex < F.Count) and (Trim(F[AIndex]) <> '') then
      Result := AsNumber(Trim(F[AIndex]))
    else
      Result := TJSONIntegerNumber.Create(0);
  end;

begin
  Rows := TStringList.Create;
  F := TStringList.Create;
  try
    SplitRows(AText, Rows);
    Arr := TJSONArray.Create;
    for r := 0 to Rows.Count - 1 do
    begin
      SplitCsv(Rows[r], F);
      { The grid wrote a leading empty cell for the row number. }
      if (F.Count > 4) and (Trim(F[0]) = '') then F.Delete(0);
      Obj := TJSONObject.Create;
      if AMedia then
      begin
        Obj.Add('mua', Field(0));
        Obj.Add('mus', Field(1));
        Obj.Add('g', Field(2));
        Obj.Add('n', Field(3));
      end
      else
      begin
        Pos3 := TJSONArray.Create;
        for c := 0 to 2 do Pos3.Add(Field(c));
        Obj.Add('Pos', Pos3);
        Obj.Add('R', Field(3));
      end;
      Arr.Add(Obj);
    end;
    if AMedia then Put(ADoc, 'Domain.Media', Arr)
    else Put(ADoc, 'Optode.Detector', Arr);
  finally
    F.Free;
    Rows.Free;
  end;
end;

{ MCXConfig: the rows of the old property grid, Section, Key, Value. }
procedure ConvertConfig(ADoc: TJSONObject; const AText: string; AWarn: TStrings);
var
  Rows, F, G: TStringList;
  r: Integer;
  Section, Key, Value: string;
  V: TJSONData;
begin
  Rows := TStringList.Create;
  F := TStringList.Create;
  G := TStringList.Create;
  try
    SplitRows(AText, Rows);
    for r := 0 to Rows.Count - 1 do
    begin
      SplitCsv(Rows[r], F);
      { A row whose value held commas was quoted whole, so it reads as one
        field the first time and has to be read again. }
      if F.Count = 1 then
      begin
        SplitCsv(F[0], G);
        F.Assign(G);
      end;
      if F.Count < 3 then Continue;
      Section := Trim(F[0]);
      Key := Trim(F[1]);
      Value := Trim(F[2]);
      if (Section = '') or (Key = '') or (Value = '') then Continue;
      { The volume designer wrote a placeholder rather than a path. }
      if Copy(Value, 1, 4) = 'See ' then Continue;
      if Value[1] = '[' then
      begin
        try
          V := GetJSON(Value);
        except
          if AWarn <> nil then
            AWarn.Add(Format('could not read the array in %s.%s: %s',
              [Section, Key, Value]));
          Continue;
        end;
      end
      else
        V := AsNumber(Value);
      Put(ADoc, Section + '.' + Key, V);
    end;
  finally
    G.Free;
    F.Free;
    Rows.Free;
  end;
end;

function McxProjectSession(const AFileName, ASession: string;
  AWarn: TStrings): string;
var
  Secs, Items: TStringList;
  Doc: TJSONObject;
  i, j, k: Integer;
  Key, Value, Letters: string;
  Done: Boolean;
  Shapes, D: TJSONData;
begin
  Result := '';
  Secs := TStringList.Create;
  Doc := TJSONObject.Create;
  try
    if not ReadProject(AFileName, Secs) then Exit;
    k := Secs.IndexOf(ASession);
    if k < 0 then Exit;
    Items := TStringList(Secs.Objects[k]);
    Put(Doc, 'Session.ID', TJSONString.Create(ASession));

    for i := 0 to Items.Count - 1 do
    begin
      Key := Trim(Items.Names[i]);
      Value := Trim(Copy(Items[i], Pos('=', Items[i]) + 1, MaxInt));
      if Value = '' then Continue;
      if Pos(',' + LowerCase(Key) + ',', NotSimulation) > 0 then Continue;

      Done := False;
      for j := 0 to High(Scalars) do
        if SameText(Key, Scalars[j][0]) then
        begin
          Put(Doc, Scalars[j][1], AsNumber(Value));
          Done := True;
          Break;
        end;
      if Done then Continue;

      if SameText(Key, 'MediaCfg') then ConvertTable(Doc, Value, True)
      else if SameText(Key, 'DetectorCfg') then ConvertTable(Doc, Value, False)
      else if SameText(Key, 'MCXConfig') then ConvertConfig(Doc, Value, AWarn)
      else if SameText(Key, 'ShapeCfg') then
      begin
        try
          Shapes := GetJSON(Value);
        except
          if AWarn <> nil then AWarn.Add('could not read ShapeCfg');
          Continue;
        end;
        if (Shapes is TJSONObject) and (TJSONObject(Shapes).Find('Shapes') <> nil) then
        begin
          D := TJSONObject(Shapes).Extract('Shapes');
          Shapes.Free;
          Shapes := D;
        end;
        Put(Doc, 'Shapes', Shapes);
      end
      else if SameText(Key, 'OutputFormat') then
        Put(Doc, 'Session.OutputFormat',
          TJSONString.Create(LowerCase(Trim(Copy(Value + ' ', 1, Pos(' ', Value + ' ') - 1)))))
      else if SameText(Key, 'OutputType') then
        { Saved as the combo box's text, 'X - fluence rate in ...': the
          letter mcx reads is the first character. }
        Put(Doc, 'Session.OutputType', TJSONString.Create(LowerCase(Value[1])))
      else if SameText(Key, 'DebugFlags') then
      begin
        Letters := '';
        for j := 1 to 3 do
          if (j <= Length(Value)) and (Value[j] = '1') then
            Letters := Letters + DebugLetters[j];
        if Letters <> '' then Put(Doc, 'Session.DebugFlag', TJSONString.Create(Letters));
      end
      else if SameText(Key, 'DoSaveMask') then
      begin
        if StrToIntDef(Value, 0) <> 0 then
          Put(Doc, 'Session.SaveDataMask', TJSONString.Create('DSPMXVW'));
      end
      else if SameText(Key, 'DoReplay') or SameText(Key, 'ReplayDet') then
      begin
        if StrToIntDef(Value, 0) <> 0 then
          Put(Doc, 'Session.ReplayDet',
            AsNumber(Items.Values['ReplayDet']));
      end
      else if AWarn <> nil then
        AWarn.Add(Format('%s: no home for %s=%s', [ASession, Key, Value]));
    end;

    { OriginType is mcx's srcfrom0, 1 or 2 rather than a flag: 1 puts the
      first voxel's corner at the origin. }
    D := Doc.FindPath('Domain.OriginType');
    if D <> nil then
    begin
      if D.AsFloat <> 0 then j := 1 else j := 2;
      Put(Doc, 'Domain.OriginType', TJSONIntegerNumber.Create(j));
    end;
    Result := Doc.FormatJSON;
  finally
    Doc.Free;
    FreeSections(Secs);
  end;
end;

end.
