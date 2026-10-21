{ mcxstudio2 - tetrahedral meshes, for mmc.

  mmc does not simulate in a grid; it simulates in a mesh of tetrahedra, and
  the mesh lives beside the input file as a pair of text tables that
  Mesh.MeshID names: node_<id>.dat and elem_<id>.dat.

  What a preview has to show is the outside of that mesh, not the inside: a
  hundred thousand tetrahedra drawn as tetrahedra is a solid block of edges.
  So this finds the surface -- every triangular face that belongs to exactly
  one element, which is the definition of being on the boundary -- and hands
  back those triangles.

  No LCL and no GL in here, so it can be tested headless. }
unit mcxmesh;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, mcxjd;

type
  TMcxNode = record
    x, y, z: Single;
  end;

  { One surface triangle, with the medium of the element it came off. }
  TMcxFace = record
    A, B, C: TMcxNode;
    Tag: Integer;
  end;
  TMcxFaces = array of TMcxFace;

  TMcxTetMesh = class
  private
    FNodes: array of TMcxNode;
    FTags: array of Integer;
    FElems: array of array[0..3] of Integer;
    FError: string;
    function LoadNodes(const AFileName: string): Boolean;
    function LoadElems(const AFileName: string): Boolean;
  public
    { Reads node_<id>.dat and elem_<id>.dat from ADir.  That is the layout
      mmc itself expects (Mesh.MeshID names the pair), so a mesh that mmc can
      run is a mesh this can show. }
    function LoadFromDir(const ADir, AMeshID: string): Boolean;
    { The other way a mesh arrives: inside the input file itself, as two
      JData arrays under Shapes -- MeshNode of [n,3] coordinates and MeshElem
      of [m,4] or [m,5] node numbers with a medium.  colin27 is written this
      way and onecube is written the other, so both have to work. }
    function LoadFromArrays(const ANodes, AElems: TMcxArray): Boolean;
    function Surface: TMcxFaces;
    procedure Bounds(out ALo, AHi: TMcxNode);
    function NodeCount: Integer;
    function ElemCount: Integer;
    property Error: string read FError;
  end;

implementation

function McxNode(x, y, z: Single): TMcxNode;
begin
  Result.x := x;
  Result.y := y;
  Result.z := z;
end;

function TMcxTetMesh.NodeCount: Integer;
begin
  Result := Length(FNodes);
end;

function TMcxTetMesh.ElemCount: Integer;
begin
  Result := Length(FElems);
end;

{ Both tables begin with a two-number header and then one row per entry, the
  first column being the index.  Whitespace-separated throughout, which is
  what mmc's own reader assumes. }
function SplitRow(const ALine: string; AOut: TStringList): Integer;
begin
  AOut.Clear;
  AOut.Delimiter := ' ';
  AOut.StrictDelimiter := False;      { runs of spaces and tabs count as one }
  AOut.DelimitedText := StringReplace(Trim(ALine), #9, ' ', [rfReplaceAll]);
  Result := AOut.Count;
end;

function TMcxTetMesh.LoadNodes(const AFileName: string): Boolean;
var
  Lines, Cols: TStringList;
  i, n: Integer;
  Fs: TFormatSettings;
begin
  Result := False;
  Fs := DefaultFormatSettings;
  Fs.DecimalSeparator := '.';
  Lines := TStringList.Create;
  Cols := TStringList.Create;
  try
    try
      Lines.LoadFromFile(AFileName);
    except
      on E: Exception do
      begin
        FError := E.Message;
        Exit;
      end;
    end;
    if Lines.Count < 2 then Exit;

    SetLength(FNodes, Lines.Count);          { an upper bound; trimmed below }
    n := 0;
    for i := 1 to Lines.Count - 1 do
    begin
      if SplitRow(Lines[i], Cols) < 4 then Continue;
      FNodes[n] := McxNode(StrToFloatDef(Cols[1], 0, Fs),
                           StrToFloatDef(Cols[2], 0, Fs),
                           StrToFloatDef(Cols[3], 0, Fs));
      Inc(n);
    end;
    SetLength(FNodes, n);
    Result := n > 0;
  finally
    Cols.Free;
    Lines.Free;
  end;
end;

function TMcxTetMesh.LoadElems(const AFileName: string): Boolean;
var
  Lines, Cols: TStringList;
  i, j, n: Integer;
begin
  Result := False;
  Lines := TStringList.Create;
  Cols := TStringList.Create;
  try
    try
      Lines.LoadFromFile(AFileName);
    except
      on E: Exception do
      begin
        FError := E.Message;
        Exit;
      end;
    end;
    if Lines.Count < 2 then Exit;

    SetLength(FElems, Lines.Count);
    SetLength(FTags, Lines.Count);
    n := 0;
    for i := 1 to Lines.Count - 1 do
    begin
      if SplitRow(Lines[i], Cols) < 5 then Continue;
      { Column zero is the element's own index; the four after it are node
        numbers, one-based, and a fifth if present is the medium. }
      for j := 0 to 3 do
        FElems[n][j] := StrToIntDef(Cols[j + 1], 0) - 1;
      if Cols.Count >= 6 then FTags[n] := StrToIntDef(Cols[5], 1)
      else FTags[n] := 1;
      Inc(n);
    end;
    SetLength(FElems, n);
    SetLength(FTags, n);
    Result := n > 0;
  finally
    Cols.Free;
    Lines.Free;
  end;
end;

function TMcxTetMesh.LoadFromDir(const ADir, AMeshID: string): Boolean;
var
  D: string;
begin
  FError := '';
  SetLength(FNodes, 0);
  SetLength(FElems, 0);
  D := IncludeTrailingPathDelimiter(ADir);
  Result := LoadNodes(D + 'node_' + AMeshID + '.dat') and
            LoadElems(D + 'elem_' + AMeshID + '.dat');
  if not Result and (FError = '') then
    FError := 'node_' + AMeshID + '.dat and elem_' + AMeshID +
      '.dat did not read as a mesh';
end;

function TMcxTetMesh.LoadFromArrays(const ANodes, AElems: TMcxArray): Boolean;
var
  i, j, nn, ne, cols: Integer;
begin
  Result := False;
  FError := '';
  SetLength(FNodes, 0);
  SetLength(FElems, 0);
  SetLength(FTags, 0);

  if (Length(ANodes.Dims) < 2) or (ANodes.Dims[1] < 3) then
  begin
    FError := 'MeshNode is not a table of coordinates';
    Exit;
  end;
  if (Length(AElems.Dims) < 2) or (AElems.Dims[1] < 4) then
  begin
    FError := 'MeshElem is not a table of node numbers';
    Exit;
  end;

  nn := ANodes.Dims[0];
  SetLength(FNodes, nn);
  { Row-major: n rows of three, which is how mcx and mmc write every table
    of this shape. }
  for i := 0 to nn - 1 do
    FNodes[i] := McxNode(McxArrayValue(ANodes, Int64(i) * ANodes.Dims[1]),
                         McxArrayValue(ANodes, Int64(i) * ANodes.Dims[1] + 1),
                         McxArrayValue(ANodes, Int64(i) * ANodes.Dims[1] + 2));

  ne := AElems.Dims[0];
  cols := AElems.Dims[1];
  SetLength(FElems, ne);
  SetLength(FTags, ne);
  for i := 0 to ne - 1 do
  begin
    for j := 0 to 3 do
      { One-based, as in the .dat tables. }
      FElems[i][j] := Round(McxArrayValue(AElems, Int64(i) * cols + j)) - 1;
    if cols >= 5 then
      FTags[i] := Round(McxArrayValue(AElems, Int64(i) * cols + 4))
    else
      FTags[i] := 1;
  end;
  Result := (nn > 0) and (ne > 0);
end;

procedure TMcxTetMesh.Bounds(out ALo, AHi: TMcxNode);
var
  i: Integer;
begin
  ALo := McxNode(0, 0, 0);
  AHi := McxNode(1, 1, 1);
  if Length(FNodes) = 0 then Exit;
  ALo := FNodes[0];
  AHi := FNodes[0];
  for i := 1 to High(FNodes) do
  begin
    if FNodes[i].x < ALo.x then ALo.x := FNodes[i].x;
    if FNodes[i].y < ALo.y then ALo.y := FNodes[i].y;
    if FNodes[i].z < ALo.z then ALo.z := FNodes[i].z;
    if FNodes[i].x > AHi.x then AHi.x := FNodes[i].x;
    if FNodes[i].y > AHi.y then AHi.y := FNodes[i].y;
    if FNodes[i].z > AHi.z then AHi.z := FNodes[i].z;
  end;
end;

type
  { One face of one tetrahedron: the three node numbers in ascending order,
    packed so that the same face from two elements compares equal. }
  TFaceKey = record
    Key: QWord;
    Elem: Integer;
    A, B, C: Integer;
  end;
  TFaceKeys = array of TFaceKey;

procedure SortFaces(var A: TFaceKeys; L, R: Integer);
var
  i, j: Integer;
  P: QWord;
  T: TFaceKey;
begin
  { Plain quicksort.  A colin27 mesh is about four hundred thousand faces,
    which is a fraction of a second here and would be several seconds with
    anything quadratic -- which is what the old renderer used for its much
    smaller pick list. }
  i := L;
  j := R;
  P := A[(L + R) div 2].Key;
  repeat
    while A[i].Key < P do Inc(i);
    while A[j].Key > P do Dec(j);
    if i <= j then
    begin
      T := A[i]; A[i] := A[j]; A[j] := T;
      Inc(i);
      Dec(j);
    end;
  until i > j;
  if L < j then SortFaces(A, L, j);
  if i < R then SortFaces(A, i, R);
end;

function TMcxTetMesh.Surface: TMcxFaces;
const
  { The four faces of a tetrahedron, as node positions within the element. }
  Faces: array[0..3, 0..2] of Integer =
    ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3));
var
  Keys: TFaceKeys;
  i, f, n, Run, Out_: Integer;
  v: array[0..2] of Integer;
  t: Integer;
begin
  Result := nil;
  if (Length(FElems) = 0) or (Length(FNodes) = 0) then Exit;

  SetLength(Keys, Length(FElems) * 4);
  n := 0;
  for i := 0 to High(FElems) do
    for f := 0 to 3 do
    begin
      v[0] := FElems[i][Faces[f][0]];
      v[1] := FElems[i][Faces[f][1]];
      v[2] := FElems[i][Faces[f][2]];
      if (v[0] < 0) or (v[1] < 0) or (v[2] < 0) then Continue;
      if (v[0] >= Length(FNodes)) or (v[1] >= Length(FNodes)) or
         (v[2] >= Length(FNodes)) then Continue;

      { Sorted, so that the same face seen from two elements packs to the
        same number.  Three at a time is a sort worth writing out. }
      if v[0] > v[1] then begin t := v[0]; v[0] := v[1]; v[1] := t; end;
      if v[1] > v[2] then begin t := v[1]; v[1] := v[2]; v[2] := t; end;
      if v[0] > v[1] then begin t := v[0]; v[0] := v[1]; v[1] := t; end;

      { Twenty-one bits each: two million nodes, which is far more than any
        mesh mmc runs, and it all fits in one comparable number. }
      Keys[n].Key := (QWord(v[0]) shl 42) or (QWord(v[1]) shl 21) or QWord(v[2]);
      Keys[n].Elem := i;
      Keys[n].A := v[0];
      Keys[n].B := v[1];
      Keys[n].C := v[2];
      Inc(n);
    end;
  SetLength(Keys, n);
  if n = 0 then Exit;

  SortFaces(Keys, 0, n - 1);

  { A face shared by two elements is interior; one that appears once is on
    the boundary.  After sorting, that is a run of length one. }
  SetLength(Result, n);
  Out_ := 0;
  i := 0;
  while i < n do
  begin
    Run := 1;
    while (i + Run < n) and (Keys[i + Run].Key = Keys[i].Key) do Inc(Run);
    if Run = 1 then
    begin
      Result[Out_].A := FNodes[Keys[i].A];
      Result[Out_].B := FNodes[Keys[i].B];
      Result[Out_].C := FNodes[Keys[i].C];
      Result[Out_].Tag := FTags[Keys[i].Elem];
      Inc(Out_);
    end;
    Inc(i, Run);
  end;
  SetLength(Result, Out_);
end;

end.
