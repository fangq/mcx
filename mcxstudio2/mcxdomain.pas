{ mcxstudio2 - the domain as data: a voxel volume, or a tetrahedral mesh.

  Shapes are edited command by command in mcxshapes.  A volume or a mesh is
  thousands of numbers, which nobody types; what a person does with one is
  bring it in from a file, or copy it from one input into another.  So this
  unit turns what can be brought in -- a volume file, a pasted array, mmc's
  node_/elem_ files, a JMesh file -- into the JSON mcx and mmc read from an
  input, and back:

    a volume  -> Shapes as a JData array, row-major, zlib-compressed
                 (mcx_utils.c mcx_jdatadecode, which reads Shapes as
                 row-major whatever it says);
    a mesh    -> MeshNode as JData single [Nn,3] and MeshElem as JData
                 int32 [Ne,5], four nodes and a label (mmc_utils.c, which
                 insists on exactly those types for annotated arrays).

  No LCL here, so all of it runs in the headless tests. }
unit mcxdomain;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpjson, jsonparser, base64, zstream, mcxdoc, mcxjd;

{ A JData object holding AArray, row-major as AArray already is, compressed
  with zlib.  The caller owns the object. }
function McxEncodeJData(const AArray: TMcxArray): TJSONObject;

{ The same numbers in row-major order, from an array stored column-major
  (first index fastest) -- what NIfTI, mcx's own .bin and anything written
  with _ArrayOrder_ "c" hold. }
function McxRowMajor(const AArray: TMcxArray): TMcxArray;

{ A volume for Shapes from pasted JSON: a JData object, kept as it is once
  it decodes, or a nested list of numbers ([[[..]]], how most tools print a
  3-D array), packed into the narrowest type that holds every value.  False
  with AError set when it is neither. }
function McxVolumeFromJSON(AData: TJSONData; out AObj: TJSONObject;
  out AError: string): Boolean;

{ A volume for Shapes read from a file: .json/.jnii/.bnii/.nii, or a raw
  .bin/.raw/.vol of ANx*ANy*ANz labels AWidth bytes each.  Kept in the order
  the file says, converted to the row-major mcx reads Shapes in. }
function McxVolumeFromFile(const AFileName: string; AWidth, ANx, ANy,
  ANz: Integer; out AObj: TJSONObject; out AError: string): Boolean;

{ The three grid dimensions of a volume array, and the MediaFormat its type
  implies ('' when the type says nothing -- a float label volume). }
procedure McxVolumeShape(AObj: TJSONObject; out ANx, ANy, ANz: Integer;
  out AFormat: string);

{ '60 x 60 x 60 uint8', for the card. }
function McxVolumeSummary(AObj: TJSONObject): string;

{ A mesh from pasted JSON: an object with MeshNode and MeshElem -- or the
  JMesh names MeshVertex3 and MeshTet4 -- each a JData array or a list of
  rows.  The caller owns ANode and AElem. }
function McxMeshFromJSON(AData: TJSONData; out ANode, AElem: TJSONData;
  out AError: string): Boolean;

{ A mesh from a file: mmc's node_<id>.dat with elem_<id>.dat beside it
  (either may be the one picked), or a .jmsh/.json holding the arrays.

  AMedia is the optical properties that came with it, as a Domain.Media
  list, or nil: mmc reads prop_<id>.dat when an input has no Media, and a
  mesh taken away from its MeshID takes that file's media with it, or the
  run fails looking for a prop file under no name at all. }
function McxMeshFromFile(const AFileName: string; out ANode, AElem,
  AMedia: TJSONData; out AError: string): Boolean;

{ prop_<id>.dat as a Domain.Media list -- medium 0, the background, first,
  as mmc puts it -- or nil when there is no such file. }
function McxReadProp(const AFileName: string): TJSONArray;

{ Node and element counts of a mesh array in either form, -1 if unknown. }
function McxMeshRows(AData: TJSONData): Integer;

implementation

type
  TMcxDoubles = array of Double;

{ ------------------------------------------------------------- encoding -- }

function McxEncodeJData(const AArray: TMcxArray): TJSONObject;
var
  Dst: TStringStream;
  Z: TCompressionStream;
  Size: TJSONArray;
  i: Integer;
  Zipped: string;
begin
  Dst := TStringStream.Create('');
  try
    Z := TCompressionStream.Create(cldefault, Dst);
    try
      if Length(AArray.Data) > 0 then Z.WriteBuffer(AArray.Data[0], Length(AArray.Data));
    finally
      Z.Free;
    end;
    Zipped := EncodeStringBase64(Dst.DataString);
  finally
    Dst.Free;
  end;

  Result := TJSONObject.Create;
  Result.Add('_ArrayType_', McxArrayKindName(AArray.Kind));
  Size := TJSONArray.Create;
  for i := 0 to High(AArray.Dims) do Size.Add(AArray.Dims[i]);
  Result.Add('_ArraySize_', Size);
  Result.Add('_ArrayZipType_', 'zlib');
  Result.Add('_ArrayZipSize_', TJSONArray.Create([1, McxArrayCount(AArray)]));
  Result.Add('_ArrayZipData_', Zipped);
end;

function McxRowMajor(const AArray: TMcxArray): TMcxArray;
var
  nd, k, Es: Integer;
  n, i, Src, Stride: Int64;
  Idx: array of Integer;
  ColStride: array of Int64;
begin
  Result.Kind := AArray.Kind;
  Result.Dims := Copy(AArray.Dims);
  SetLength(Result.Data, Length(AArray.Data));
  nd := Length(AArray.Dims);
  Es := McxElemSize(AArray.Kind);
  n := McxArrayCount(AArray);
  if (nd < 2) or (n = 0) then
  begin
    Result.Data := Copy(AArray.Data);
    Exit;
  end;
  SetLength(ColStride, nd);
  Stride := 1;
  for k := 0 to nd - 1 do
  begin
    ColStride[k] := Stride;
    Stride := Stride * AArray.Dims[k];
  end;
  { Walk the output in row-major order -- last index fastest -- and fetch
    each element from where the column-major input keeps it. }
  SetLength(Idx, nd);
  for k := 0 to nd - 1 do Idx[k] := 0;
  for i := 0 to n - 1 do
  begin
    Src := 0;
    for k := 0 to nd - 1 do Src := Src + Idx[k] * ColStride[k];
    Move(AArray.Data[Src * Es], Result.Data[i * Es], Es);
    k := nd - 1;
    while k >= 0 do
    begin
      Inc(Idx[k]);
      if Idx[k] < AArray.Dims[k] then Break;
      Idx[k] := 0;
      Dec(k);
    end;
  end;
end;

{ --------------------------------------------------------------- volumes -- }

{ A nested list of numbers, row-major by construction: the dimensions from
  following first elements down, the values by walking it in order. }
function NestedToArray(AData: TJSONArray; out AArray: TMcxArray;
  out AError: string): Boolean;
var
  Dims: array of Integer;
  Vals: array of Double;
  n: Int64;
  D: TJSONData;
  i: Integer;
  Lo, Hi: Double;
  AllInt: Boolean;

  function Walk(AD: TJSONData; ALevel: Integer): Boolean;
  var
    j: Integer;
  begin
    Result := False;
    if ALevel = Length(Dims) then
    begin
      if AD.JSONType <> jtNumber then
      begin
        AError := 'the array holds something that is not a number';
        Exit;
      end;
      if n >= Length(Vals) then SetLength(Vals, Length(Vals) * 2 + 1024);
      Vals[n] := AD.AsFloat;
      Inc(n);
      Exit(True);
    end;
    if (AD.JSONType <> jtArray) or (AD.Count <> Dims[ALevel]) then
    begin
      AError := 'the rows of the array are not all the same length';
      Exit;
    end;
    for j := 0 to AD.Count - 1 do
      if not Walk(AD.Items[j], ALevel + 1) then Exit;
    Result := True;
  end;

begin
  Result := False;
  AError := '';
  D := AData;
  SetLength(Dims, 0);
  while D.JSONType = jtArray do
  begin
    if D.Count = 0 then
    begin
      AError := 'the array is empty';
      Exit;
    end;
    SetLength(Dims, Length(Dims) + 1);
    Dims[High(Dims)] := D.Count;
    D := D.Items[0];
  end;
  n := 0;
  SetLength(Vals, 0);
  if not Walk(AData, 0) then Exit;

  AllInt := True;
  Lo := 0;
  Hi := 0;
  for i := 0 to n - 1 do
  begin
    if Frac(Vals[i]) <> 0 then AllInt := False;
    if (i = 0) or (Vals[i] < Lo) then Lo := Vals[i];
    if (i = 0) or (Vals[i] > Hi) then Hi := Vals[i];
  end;
  if AllInt and (Lo >= 0) and (Hi <= 255) then AArray.Kind := akUInt8
  else if AllInt and (Lo >= 0) and (Hi <= 65535) then AArray.Kind := akUInt16
  else if AllInt and (Lo >= -2147483648.0) and (Hi <= 2147483647.0) then AArray.Kind := akInt32
  else AArray.Kind := akSingle;
  AArray.Dims := Copy(Dims);
  SetLength(AArray.Data, n * McxElemSize(AArray.Kind));
  for i := 0 to n - 1 do
    case AArray.Kind of
      akUInt8:  AArray.Data[i] := Round(Vals[i]);
      akUInt16: PWord(@AArray.Data[i * 2])^ := Round(Vals[i]);
      akInt32:  PLongInt(@AArray.Data[i * 4])^ := Round(Vals[i]);
    else
      PSingle(@AArray.Data[i * 4])^ := Vals[i];
    end;
  Result := True;
end;

function McxVolumeFromJSON(AData: TJSONData; out AObj: TJSONObject;
  out AError: string): Boolean;
var
  A: TMcxArray;
begin
  Result := False;
  AObj := nil;
  AError := '';
  if AData = nil then
  begin
    AError := 'there is nothing to read';
    Exit;
  end;
  if (AData is TJSONObject) and (TJSONObject(AData).Find('_ArrayType_') <> nil) then
  begin
    if not McxDecodeJData(TJSONObject(AData), A) or
       (McxArrayCount(A) * McxElemSize(A.Kind) > Length(A.Data)) then
    begin
      AError := 'the JData array does not decode, or holds fewer values ' +
        'than its _ArraySize_ says';
      Exit;
    end;
    if Length(A.Dims) < 3 then
    begin
      AError := 'a volume needs three dimensions';
      Exit;
    end;
    { Column-major is turned round, because mcx reads Shapes row-major
      whatever the annotation says. }
    if (TJSONObject(AData).Find('_ArrayOrder_') <> nil) and
       (LowerCase(Copy(TJSONObject(AData).Find('_ArrayOrder_').AsString, 1, 1)) = 'c') then
      AObj := McxEncodeJData(McxRowMajor(A))
    else
      AObj := TJSONObject(AData.Clone);
    Exit(True);
  end;
  if AData is TJSONArray then
  begin
    if not NestedToArray(TJSONArray(AData), A, AError) then Exit;
    if Length(A.Dims) < 3 then
    begin
      AError := 'a volume needs three dimensions, as [[[...]]]';
      Exit;
    end;
    AObj := McxEncodeJData(A);
    Exit(True);
  end;
  AError := 'expected a JData array {"_ArrayType_": ...} or a nested ' +
    'list of numbers [[[...]]]';
end;

{ Whether a JData file stores its array column-major: mcx writes
  "_ArrayOrder_":"c" on every volume it saves; JData's default, and what
  MATLAB's and Python's writers use, is row-major with no annotation. }
function FileSaysColumnMajor(const AFileName: string): Boolean;
var
  F: TFileStream;
  S: string;
  p: Integer;
begin
  Result := False;
  try
    F := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyNone);
    try
      SetLength(S, F.Size);
      if F.Size > 0 then F.ReadBuffer(S[1], F.Size);
    finally
      F.Free;
    end;
  except
    Exit;
  end;
  p := Pos('_ArrayOrder_', S);
  if p = 0 then Exit;
  { text: "_ArrayOrder_":"c"; binary: a short string marker then 'c' }
  S := Copy(S, p + 12, 12);
  Result := (Pos('"c"', S) > 0) or (Pos('c', S) > 0) and (Pos('r', S) = 0);
end;

function McxVolumeFromFile(const AFileName: string; AWidth, ANx, ANy,
  ANz: Integer; out AObj: TJSONObject; out AError: string): Boolean;
var
  Ext: string;
  A: TMcxArray;
  F: TFileStream;
  n: Int64;
begin
  Result := False;
  AObj := nil;
  AError := '';
  Ext := LowerCase(ExtractFileExt(AFileName));
  if (Ext = '.bin') or (Ext = '.raw') or (Ext = '.vol') then
  begin
    { mcx's raw volume: Dim voxels, x fastest, one label each. }
    n := Int64(ANx) * ANy * ANz;
    if n <= 0 then
    begin
      AError := 'a raw volume has no header: set Dim in Domain first';
      Exit;
    end;
    case AWidth of
      1: A.Kind := akUInt8;
      2: A.Kind := akUInt16;
      4: A.Kind := akUInt32;
    else
      AError := 'a raw volume is read as byte, short or integer labels; ' +
        'set the volume format to one of them';
      Exit;
    end;
    try
      F := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyNone);
      try
        if F.Size < n * AWidth then
        begin
          AError := Format('%s holds %d bytes; Dim %dx%dx%d of %d-byte ' +
            'labels needs %d', [ExtractFileName(AFileName), F.Size, ANx,
            ANy, ANz, AWidth, n * AWidth]);
          Exit;
        end;
        SetLength(A.Dims, 3);
        A.Dims[0] := ANx; A.Dims[1] := ANy; A.Dims[2] := ANz;
        SetLength(A.Data, n * AWidth);
        F.ReadBuffer(A.Data[0], n * AWidth);
      finally
        F.Free;
      end;
    except
      on E: Exception do
      begin
        AError := E.Message;
        Exit;
      end;
    end;
    AObj := McxEncodeJData(McxRowMajor(A));
    Exit(True);
  end;

  if not McxLoadArray(AFileName, '', A) then
  begin
    AError := 'no array could be read out of ' + ExtractFileName(AFileName);
    Exit;
  end;
  if Length(A.Dims) < 3 then
  begin
    AError := ExtractFileName(AFileName) + ' does not hold a 3-D volume';
    Exit;
  end;
  if (Ext = '.nii') or FileSaysColumnMajor(AFileName) then A := McxRowMajor(A);
  AObj := McxEncodeJData(A);
  Result := True;
end;

procedure McxVolumeShape(AObj: TJSONObject; out ANx, ANy, ANz: Integer;
  out AFormat: string);
var
  S, T: TJSONData;
  D: array of Integer;
  i, Off: Integer;
  K: string;
begin
  ANx := 0; ANy := 0; ANz := 0;
  AFormat := '';
  if AObj = nil then Exit;
  S := AObj.Find('_ArraySize_');
  if not (S is TJSONArray) or (S.Count < 3) then Exit;
  SetLength(D, S.Count);
  for i := 0 to S.Count - 1 do D[i] := S.Items[i].AsInteger;
  { A fourth dimension is the values per voxel; it is the short one, and it
    comes last ([60,40,20,4]) or first ([4,60,40,20]). }
  Off := 0;
  if (Length(D) >= 4) and (D[0] <= 8) and (D[High(D)] > 8) then Off := 1;
  ANx := D[Off]; ANy := D[Off + 1]; ANz := D[Off + 2];
  T := AObj.Find('_ArrayType_');
  if T = nil then Exit;
  K := LowerCase(T.AsString);
  if Length(D) = 3 then
  begin
    if (K = 'uint8') or (K = 'int8') then AFormat := 'byte'
    else if (K = 'uint16') or (K = 'int16') then AFormat := 'short'
    else if (K = 'uint32') or (K = 'int32') then AFormat := 'integer';
  end
  else if (K = 'single') or (K = 'float32') then
  begin
    if D[High(D) * (1 - Off)] = 4 then AFormat := 'asgn_float'
    else if D[High(D) * (1 - Off)] = 2 then AFormat := 'muamus_float';
  end;
end;

function McxVolumeSummary(AObj: TJSONObject): string;
var
  S, T: TJSONData;
  i: Integer;
begin
  Result := '';
  if AObj = nil then Exit;
  S := AObj.Find('_ArraySize_');
  if S is TJSONArray then
    for i := 0 to S.Count - 1 do
    begin
      if i > 0 then Result := Result + ' x ';
      Result := Result + S.Items[i].AsString;
    end;
  T := AObj.Find('_ArrayType_');
  if T <> nil then Result := Result + ' ' + T.AsString;
end;

{ ----------------------------------------------------------------- meshes -- }

function McxMeshRows(AData: TJSONData): Integer;
var
  Sz: TJSONData;
begin
  Result := -1;
  if AData = nil then Exit;
  if AData.JSONType = jtArray then Exit(AData.Count);
  if AData.JSONType <> jtObject then Exit;
  Sz := TJSONObject(AData).Find('_ArraySize_');
  if (Sz is TJSONArray) and (Sz.Count > 0) then Result := Sz.Items[0].AsInteger;
end;

{ Rows of numbers from either form, as doubles in row order. }
function ReadRows(AData: TJSONData; out AVals: TMcxDoubles; out ARows,
  ACols: Integer; out AError: string): Boolean;
var
  A: TMcxArray;
  i, j: Integer;
  R: TJSONData;
begin
  Result := False;
  AError := '';
  ARows := 0;
  ACols := 0;
  SetLength(AVals, 0);
  if (AData is TJSONObject) and (TJSONObject(AData).Find('_ArrayType_') <> nil) then
  begin
    if not McxDecodeJData(TJSONObject(AData), A) or (Length(A.Dims) <> 2) then
    begin
      AError := 'a mesh array has to be a 2-D JData array';
      Exit;
    end;
    { mmc writes its own with _ArrayOrder_ "c"; read either. }
    if (TJSONObject(AData).Find('_ArrayOrder_') <> nil) and
       (LowerCase(Copy(TJSONObject(AData).Find('_ArrayOrder_').AsString, 1, 1)) = 'c') then
      A := McxRowMajor(A);
    ARows := A.Dims[0];
    ACols := A.Dims[1];
    SetLength(AVals, Int64(ARows) * ACols);
    for i := 0 to High(AVals) do AVals[i] := McxArrayValue(A, i);
    Exit(True);
  end;
  if not (AData is TJSONArray) or (AData.Count = 0) or
     (AData.Items[0].JSONType <> jtArray) then
  begin
    AError := 'a mesh array has to be a list of rows or a JData array';
    Exit;
  end;
  ARows := AData.Count;
  ACols := AData.Items[0].Count;
  SetLength(AVals, Int64(ARows) * ACols);
  for i := 0 to ARows - 1 do
  begin
    R := AData.Items[i];
    if (R.JSONType <> jtArray) or (R.Count <> ACols) then
    begin
      AError := 'the rows of a mesh array are not all the same length';
      Exit;
    end;
    for j := 0 to ACols - 1 do AVals[i * ACols + j] := R.Items[j].AsFloat;
  end;
  Result := True;
end;

{ MeshNode as mmc takes an annotated one: single, [Nn,3]. }
function EncodeNodes(const AVals: TMcxDoubles; ARows: Integer): TJSONObject;
var
  A: TMcxArray;
  i: Integer;
begin
  A.Kind := akSingle;
  SetLength(A.Dims, 2);
  A.Dims[0] := ARows; A.Dims[1] := 3;
  SetLength(A.Data, ARows * 3 * 4);
  for i := 0 to ARows * 3 - 1 do PSingle(@A.Data[i * 4])^ := AVals[i];
  Result := McxEncodeJData(A);
end;

{ MeshElem as mmc takes an annotated one: int32, [Ne,5] -- four nodes,
  counted from 1, and the region label.  Rows without a label get 1. }
function EncodeElems(const AVals: TMcxDoubles; ARows, ACols: Integer): TJSONObject;
var
  A: TMcxArray;
  i, j: Integer;
  V: Integer;
begin
  A.Kind := akInt32;
  SetLength(A.Dims, 2);
  A.Dims[0] := ARows; A.Dims[1] := 5;
  SetLength(A.Data, ARows * 5 * 4);
  for i := 0 to ARows - 1 do
    for j := 0 to 4 do
    begin
      if j < 4 then V := Round(AVals[i * ACols + j])
      else if ACols >= 5 then V := Round(AVals[i * ACols + 4])
      else V := 1;
      PLongInt(@A.Data[(i * 5 + j) * 4])^ := V;
    end;
  Result := McxEncodeJData(A);
end;

function BuildMesh(ANodeSrc, AElemSrc: TJSONData; out ANode, AElem: TJSONData;
  out AError: string): Boolean;
var
  NV, EV: TMcxDoubles;
  Nr, Nc, Er, Ec: Integer;
begin
  Result := False;
  ANode := nil;
  AElem := nil;
  if (ANodeSrc = nil) or (AElemSrc = nil) then
  begin
    AError := 'a mesh needs both MeshNode (or MeshVertex3) and MeshElem ' +
      '(or MeshTet4)';
    Exit;
  end;
  if not ReadRows(ANodeSrc, NV, Nr, Nc, AError) then
  begin
    AError := 'MeshNode: ' + AError;
    Exit;
  end;
  if not ReadRows(AElemSrc, EV, Er, Ec, AError) then
  begin
    AError := 'MeshElem: ' + AError;
    Exit;
  end;
  if Nc < 3 then
  begin
    AError := 'MeshNode rows need x, y and z';
    Exit;
  end;
  if Ec < 4 then
  begin
    AError := 'MeshElem rows need four node indices';
    Exit;
  end;
  if Nc > 3 then
  begin
    { a leading index column, as in node_*.dat, is dropped }
    AError := 'MeshNode rows have more than three numbers';
    Exit;
  end;
  ANode := EncodeNodes(NV, Nr);
  AElem := EncodeElems(EV, Er, Ec);
  Result := True;
end;

function McxMeshFromJSON(AData: TJSONData; out ANode, AElem: TJSONData;
  out AError: string): Boolean;
var
  O: TJSONObject;
  N, E: TJSONData;
begin
  Result := False;
  ANode := nil;
  AElem := nil;
  AError := '';
  if not (AData is TJSONObject) then
  begin
    AError := 'expected an object with MeshNode and MeshElem';
    Exit;
  end;
  O := TJSONObject(AData);
  { An input file, pasted whole, carries its mesh under Mesh or Shapes. }
  if (O.Find('MeshNode') = nil) and (O.Find('MeshVertex3') = nil) then
  begin
    if O.Find('Mesh') is TJSONObject then O := TJSONObject(O.Find('Mesh'))
    else if O.Find('Shapes') is TJSONObject then O := TJSONObject(O.Find('Shapes'));
  end;
  N := O.Find('MeshNode');
  if N = nil then N := O.Find('MeshVertex3');
  E := O.Find('MeshElem');
  if E = nil then E := O.Find('MeshTet4');
  Result := BuildMesh(N, E, ANode, AElem, AError);
end;

{ One of mmc's .dat mesh files: a "<a> <count>" header, then a row per
  node (index x y z) or element (index n1..n4 label). }
function ReadDat(const AFileName: string; ANodes: Boolean; out AVals: TMcxDoubles;
  out ARows: Integer; out AError: string): Boolean;
var
  L, W: TStringList;
  i, j, c, Want: Integer;
  Fs: TFormatSettings;
  V: Double;
begin
  Result := False;
  AError := '';
  ARows := 0;
  SetLength(AVals, 0);
  Fs := DefaultFormatSettings;
  Fs.DecimalSeparator := '.';
  if ANodes then Want := 3 else Want := 5;
  L := TStringList.Create;
  W := TStringList.Create;
  try
    try
      L.LoadFromFile(AFileName);
    except
      on E: Exception do
      begin
        AError := E.Message;
        Exit;
      end;
    end;
    W.Delimiter := ' ';
    W.StrictDelimiter := False;
    SetLength(AVals, Int64(L.Count) * Want);
    for i := 1 to L.Count - 1 do
    begin
      W.DelimitedText := StringReplace(Trim(L[i]), #9, ' ', [rfReplaceAll]);
      if W.Count < Want + 1 then Continue;
      c := 0;
      for j := 1 to Want do
      begin
        if not TryStrToFloat(W[j], V, Fs) then Break;
        AVals[ARows * Want + j - 1] := V;
        Inc(c);
      end;
      if c = Want then Inc(ARows);
    end;
    SetLength(AVals, Int64(ARows) * Want);
    if ARows = 0 then
      AError := 'no rows could be read from ' + ExtractFileName(AFileName)
    else
      Result := True;
  finally
    W.Free;
    L.Free;
  end;
end;

function McxReadProp(const AFileName: string): TJSONArray;
var
  L, W: TStringList;
  i: Integer;
  Fs: TFormatSettings;
  V: array[0..3] of Double;
  O: TJSONObject;
  Ok: Boolean;
  k: Integer;
begin
  Result := nil;
  if not FileExists(AFileName) then Exit;
  Fs := DefaultFormatSettings;
  Fs.DecimalSeparator := '.';
  L := TStringList.Create;
  W := TStringList.Create;
  try
    L.LoadFromFile(AFileName);
    W.Delimiter := ' ';
    W.StrictDelimiter := False;
    Result := TJSONArray.Create;
    O := TJSONObject.Create;
    O.Add('mua', 0); O.Add('mus', 0); O.Add('g', 1); O.Add('n', 1);
    Result.Add(O);
    for i := 1 to L.Count - 1 do
    begin
      W.DelimitedText := StringReplace(Trim(L[i]), #9, ' ', [rfReplaceAll]);
      if W.Count < 5 then Continue;
      Ok := True;
      for k := 0 to 3 do
        if not TryStrToFloat(W[k + 1], V[k], Fs) then Ok := False;
      if not Ok then Continue;
      O := TJSONObject.Create;
      O.Add('mua', V[0]); O.Add('mus', V[1]); O.Add('g', V[2]); O.Add('n', V[3]);
      Result.Add(O);
    end;
    if Result.Count < 2 then FreeAndNil(Result);
  finally
    W.Free;
    L.Free;
  end;
end;

function McxMeshFromFile(const AFileName: string; out ANode, AElem,
  AMedia: TJSONData; out AError: string): Boolean;
var
  Dir, Base, Id, Ext: string;
  NV, EV: TMcxDoubles;
  Nr, Er: Integer;
  L: TStringList;
  D: TJSONData;
begin
  Result := False;
  ANode := nil;
  AElem := nil;
  AMedia := nil;
  AError := '';
  Dir := ExtractFilePath(AFileName);
  Base := ExtractFileName(AFileName);
  Ext := LowerCase(ExtractFileExt(Base));
  if Ext = '.dat' then
  begin
    { node_<id>.dat or elem_<id>.dat: either names the pair. }
    if Pos('node_', LowerCase(Base)) = 1 then Id := Copy(Base, 6, Length(Base) - 9)
    else if Pos('elem_', LowerCase(Base)) = 1 then Id := Copy(Base, 6, Length(Base) - 9)
    else
    begin
      AError := 'pick mmc''s node_<id>.dat or elem_<id>.dat';
      Exit;
    end;
    if not ReadDat(Dir + 'node_' + Id + '.dat', True, NV, Nr, AError) then Exit;
    if not ReadDat(Dir + 'elem_' + Id + '.dat', False, EV, Er, AError) then Exit;
    ANode := EncodeNodes(NV, Nr);
    AElem := EncodeElems(EV, Er, 5);
    AMedia := McxReadProp(Dir + 'prop_' + Id + '.dat');
    Exit(True);
  end;
  L := TStringList.Create;
  try
    try
      L.LoadFromFile(AFileName);
      D := GetJSON(L.Text);
    except
      on E: Exception do
      begin
        AError := 'could not read ' + Base + ': ' + E.Message;
        Exit;
      end;
    end;
  finally
    L.Free;
  end;
  try
    Result := McxMeshFromJSON(D, ANode, AElem, AError);
  finally
    D.Free;
  end;
end;

end.
