{ mcxstudio2 - reading arrays out of JData and BJData files.

  mcx writes its results as JNIfTI (.jnii, text JSON) or BNIfTI (.bnii/.jdb,
  binary BJData).  Both carry the same annotations around an array -- what
  type it is, what shape, how it was compressed -- so one decoder serves both
  and only the container differs.

  Nothing here uses the LCL, so all of it is tested headless.

  Two things about BJData that are easy to get wrong, and were checked against
  the producer this repository vendors, src/ubj/ubjw.c:

  It is little-endian.  UBJSON is big-endian and the copy of the spec in
  Temp/bjdata is a stale 2020 draft that says so, but mcx writes BJData draft
  2 and self-documents as such (mcx_utils.c:669); ubjw.c:326-339 memcpy's
  natively when isbjdata, which is the default at ubjw.c:68.  So on x86 and
  ARM a plain Move is right and byte-swapping would be the bug.

  An optimized container gives its type with $ and then must give its count
  with #, and carries no end marker.  A type that is a marker on its own --
  T, F, Z -- has no payload bytes at all, so a thousand of them occupy no
  space and reading a thousand bytes for them walks off into the next value. }
unit mcxjd;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, base64, zstream, fpjson, jsonparser;

type
  TMcxArrayKind = (akNone, akUInt8, akInt8, akUInt16, akInt16, akUInt32,
                   akInt32, akUInt64, akInt64, akSingle, akDouble);

  { A numeric array as it was stored: the bytes, what they are, and the shape
    they were written in.  Kept as bytes rather than as an array of Double so
    that a volume costs what it costs and not eight times that. }
  TMcxArray = record
    Kind: TMcxArrayKind;
    Dims: array of Integer;
    Data: TBytes;
  end;

function McxArrayKindOf(const AName: string): TMcxArrayKind;
function McxArrayKindName(AKind: TMcxArrayKind): string;
function McxElemSize(AKind: TMcxArrayKind): Integer;
function McxArrayCount(const AArray: TMcxArray): Int64;
{ One element, as a double whatever it was stored as.  The raycaster wants
  floats and the media table wants integers; neither should care. }
function McxArrayValue(const AArray: TMcxArray; AIndex: Int64): Double;
{ The smallest and largest values present, for scaling a colour map. }
procedure McxArrayRange(const AArray: TMcxArray; out ALow, AHigh: Double);

{ Decodes the JData annotations on an object: _ArrayType_, _ArraySize_ and
  either _ArrayData_ or a base64 _ArrayZipData_ to be inflated first.

  Text JSON is a two-stage decode -- base64, then the codec -- which is what
  mcx_utils.c:4766-4776 does in the other direction. }
function McxDecodeJData(AObj: TJSONObject; out AArray: TMcxArray): Boolean;

type
  { A BJData file: the structure as a JSON tree, with the bulk arrays kept
    out of it.

    A typed array of any size becomes an object carrying the same JData
    annotations a text file would use, plus _ArrayIndex_ into Blobs.  Small
    ones are expanded into ordinary JSON numbers instead, because a shape
    like _ArraySize_ is written the same way as a volume and has to be
    readable as a list.  Exploding the volume too would turn a 190x496x104
    result into 9.8 million TJSONNumbers. }
  TMcxBJData = class
  private
    FStream: TStream;
    FRoot: TJSONData;
    FBlobs: array of TMcxArray;
    FError: string;
    function  ReadByteAt: Byte;
    function  PeekByte: Byte;
    function  ReadRaw(ACount: Integer): TBytes;
    function  ReadIntOf(AMarker: Byte): Int64;
    function  ReadString: string;
    function  ReadCount(out ADims: array of Integer; out ADimCount: Integer): Int64;
    function  ReadValue: TJSONData;
    function  ReadArray: TJSONData;
    function  ReadObject: TJSONData;
    function  MakeBlob(AKind: TMcxArrayKind; const ADims: array of Integer;
      ADimCount: Integer; ACount: Int64): TJSONData;
  public
    destructor Destroy; override;
    function LoadFromStream(AStream: TStream): Boolean;
    function LoadFromFile(const AFileName: string): Boolean;
    { The array at a path, whether it arrived as a blob or as annotations. }
    function GetArray(const APath: string; out AArray: TMcxArray): Boolean;
    { The same, given the object itself -- for a caller that went looking. }
    function GetArrayOf(AObj: TJSONObject; out AArray: TMcxArray): Boolean;
    property Root: TJSONData read FRoot;
    property Error: string read FError;
  end;

{ Opens a .jnii, .bnii or .jdb and hands back the array at APath -- or the
  first one it can find, when APath is empty. }
function McxLoadArray(const AFileName, APath: string;
  out AArray: TMcxArray): Boolean;

implementation

const
  { Above this many elements a typed array is kept as bytes rather than
    turned into JSON numbers.  A shape or a small header field stays
    readable; a volume does not become nine million objects. }
  BlobThreshold = 64;

  ElemSizes: array[TMcxArrayKind] of Integer =
    (0, 1, 1, 2, 2, 4, 4, 8, 8, 4, 8);

  KindNames: array[TMcxArrayKind] of string =
    ('', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
     'uint64', 'int64', 'single', 'double');

function McxArrayKindOf(const AName: string): TMcxArrayKind;
var
  K: TMcxArrayKind;
  N: string;
begin
  N := LowerCase(Trim(AName));
  { JData spells the two float types single and double; NIfTI and numpy call
    them float32 and float64, and both turn up in files people hand us. }
  if N = 'float32' then N := 'single';
  if N = 'float64' then N := 'double';
  if N = 'char' then N := 'uint8';
  for K := Low(TMcxArrayKind) to High(TMcxArrayKind) do
    if KindNames[K] = N then Exit(K);
  Result := akNone;
end;

function McxArrayKindName(AKind: TMcxArrayKind): string;
begin
  Result := KindNames[AKind];
end;

function McxElemSize(AKind: TMcxArrayKind): Integer;
begin
  Result := ElemSizes[AKind];
end;

function McxArrayCount(const AArray: TMcxArray): Int64;
begin
  if McxElemSize(AArray.Kind) = 0 then Exit(0);
  Result := Length(AArray.Data) div McxElemSize(AArray.Kind);
end;

function McxArrayValue(const AArray: TMcxArray; AIndex: Int64): Double;
var
  P: Pointer;
begin
  Result := 0;
  if (AIndex < 0) or (AIndex >= McxArrayCount(AArray)) then Exit;
  P := @AArray.Data[AIndex * McxElemSize(AArray.Kind)];
  case AArray.Kind of
    akUInt8:  Result := PByte(P)^;
    akInt8:   Result := PShortInt(P)^;
    akUInt16: Result := PWord(P)^;
    akInt16:  Result := PSmallInt(P)^;
    akUInt32: Result := PLongWord(P)^;
    akInt32:  Result := PLongInt(P)^;
    akUInt64: Result := PQWord(P)^;
    akInt64:  Result := PInt64(P)^;
    akSingle: Result := PSingle(P)^;
    akDouble: Result := PDouble(P)^;
  end;
end;

procedure McxArrayRange(const AArray: TMcxArray; out ALow, AHigh: Double);
var
  i, n: Int64;
  V: Double;
begin
  ALow := 0;
  AHigh := 0;
  n := McxArrayCount(AArray);
  if n = 0 then Exit;
  ALow := McxArrayValue(AArray, 0);
  AHigh := ALow;
  for i := 1 to n - 1 do
  begin
    V := McxArrayValue(AArray, i);
    if V < ALow then ALow := V;
    if V > AHigh then AHigh := V;
  end;
end;

{ ------------------------------------------------------------ JData ------- }

function Inflate(const AData: TBytes): TBytes;
var
  Src: TMemoryStream;
  Dst: TMemoryStream;
  Z: TDecompressionStream;
  Buf: array[0..65535] of Byte;
  n: Integer;
begin
  Result := nil;
  if Length(AData) = 0 then Exit;
  Src := TMemoryStream.Create;
  Dst := TMemoryStream.Create;
  Z := nil;
  try
    Src.Write(AData[0], Length(AData));
    Src.Position := 0;
    Z := TDecompressionStream.Create(Src);
    repeat
      n := Z.Read(Buf, SizeOf(Buf));
      if n > 0 then Dst.Write(Buf, n);
    until n <= 0;
    SetLength(Result, Dst.Size);
    if Dst.Size > 0 then Move(Dst.Memory^, Result[0], Dst.Size);
  except
    { A blob that will not inflate is a corrupt file, not a reason to take
      the program down. }
    Result := nil;
  end;
  Z.Free;
  Dst.Free;
  Src.Free;
end;

function McxDecodeJData(AObj: TJSONObject; out AArray: TMcxArray): Boolean;
var
  T, S, Z, D: TJSONData;
  i: Integer;
  Raw: TBytes;
  Text: string;
begin
  Result := False;
  AArray.Kind := akNone;
  SetLength(AArray.Dims, 0);
  SetLength(AArray.Data, 0);
  if AObj = nil then Exit;

  T := AObj.Find('_ArrayType_');
  if T = nil then Exit;
  AArray.Kind := McxArrayKindOf(T.AsString);
  if AArray.Kind = akNone then Exit;

  S := AObj.Find('_ArraySize_');
  if (S <> nil) and (S.JSONType = jtArray) then
  begin
    SetLength(AArray.Dims, S.Count);
    for i := 0 to S.Count - 1 do AArray.Dims[i] := S.Items[i].AsInteger;
  end
  else if (S <> nil) and (S.JSONType = jtNumber) then
  begin
    SetLength(AArray.Dims, 1);
    AArray.Dims[0] := S.AsInteger;
  end;

  Z := AObj.Find('_ArrayZipData_');
  D := AObj.Find('_ArrayData_');

  if Z <> nil then
  begin
    { base64 first, then the codec.  _ArrayZipType_ is always zlib across
      mcx's own output; anything else is left for whoever meets it. }
    { Copied rather than cast: a string and a dynamic byte array are not the
      same thing in memory, and TBytes(s) on a string is an access violation
      waiting for the first file big enough to notice. }
    Text := DecodeStringBase64(Z.AsString);
    SetLength(Raw, Length(Text));
    if Length(Text) > 0 then Move(Text[1], Raw[0], Length(Text));
    AArray.Data := Inflate(Raw);
    Result := Length(AArray.Data) > 0;
    Exit;
  end;

  if (D <> nil) and (D.JSONType = jtArray) then
  begin
    { A plain _ArrayData_ is a list of numbers, small by definition or it
      would have been compressed. }
    SetLength(AArray.Data, D.Count * McxElemSize(AArray.Kind));
    for i := 0 to D.Count - 1 do
      case AArray.Kind of
        akUInt8:  AArray.Data[i] := Byte(D.Items[i].AsInt64);
        akInt8:   PShortInt(@AArray.Data[i])^ := D.Items[i].AsInt64;
        akUInt16: PWord(@AArray.Data[i * 2])^ := D.Items[i].AsInt64;
        akInt16:  PSmallInt(@AArray.Data[i * 2])^ := D.Items[i].AsInt64;
        akUInt32: PLongWord(@AArray.Data[i * 4])^ := D.Items[i].AsInt64;
        akInt32:  PLongInt(@AArray.Data[i * 4])^ := D.Items[i].AsInt64;
        akUInt64: PQWord(@AArray.Data[i * 8])^ := D.Items[i].AsInt64;
        akInt64:  PInt64(@AArray.Data[i * 8])^ := D.Items[i].AsInt64;
        akSingle: PSingle(@AArray.Data[i * 4])^ := D.Items[i].AsFloat;
        akDouble: PDouble(@AArray.Data[i * 8])^ := D.Items[i].AsFloat;
      end;
    Result := True;
  end;
end;

{ ------------------------------------------------------------ BJData ------ }

destructor TMcxBJData.Destroy;
begin
  FRoot.Free;
  inherited Destroy;
end;

function TMcxBJData.ReadByteAt: Byte;
begin
  if FStream.Read(Result, 1) <> 1 then
    raise Exception.Create('unexpected end of file');
end;

function TMcxBJData.PeekByte: Byte;
begin
  Result := ReadByteAt;
  FStream.Position := FStream.Position - 1;
end;

function TMcxBJData.ReadRaw(ACount: Integer): TBytes;
begin
  SetLength(Result, ACount);
  if ACount = 0 then Exit;
  if FStream.Read(Result[0], ACount) <> ACount then
    raise Exception.Create('unexpected end of file');
end;

{ Little-endian throughout: a plain Move is the decode.  See the unit
  header for why, and for why the opposite looks plausible. }
function TMcxBJData.ReadIntOf(AMarker: Byte): Int64;
var
  B: TBytes;
begin
  case Chr(AMarker) of
    'i': begin B := ReadRaw(1); Result := PShortInt(@B[0])^; end;
    'U': begin B := ReadRaw(1); Result := B[0]; end;
    'I': begin B := ReadRaw(2); Result := PSmallInt(@B[0])^; end;
    'u': begin B := ReadRaw(2); Result := PWord(@B[0])^; end;
    'l': begin B := ReadRaw(4); Result := PLongInt(@B[0])^; end;
    'm': begin B := ReadRaw(4); Result := PLongWord(@B[0])^; end;
    'L': begin B := ReadRaw(8); Result := PInt64(@B[0])^; end;
    'M': begin B := ReadRaw(8); Result := Int64(PQWord(@B[0])^); end;
  else
    raise Exception.CreateFmt('expected an integer marker, got %s',
      [Chr(AMarker)]);
  end;
end;

function TMcxBJData.ReadString: string;
var
  Len: Int64;
  B: TBytes;
begin
  Len := ReadIntOf(ReadByteAt);
  B := ReadRaw(Len);
  SetLength(Result, Len);
  if Len > 0 then Move(B[0], Result[1], Len);
end;

{ After '#': either one integer, or '[' and a list of dimensions.

  mcx itself only ever writes the one-integer form -- every bulk array it
  emits is a flat 1-D container -- but jsonlab and pyjdata write the
  dimension vector, and files from those turn up here too. }
function TMcxBJData.ReadCount(out ADims: array of Integer;
  out ADimCount: Integer): Int64;
var
  M, T: Byte;
  n, i: Int64;
begin
  ADimCount := 0;
  M := ReadByteAt;
  if Chr(M) <> '[' then Exit(ReadIntOf(M));

  { A dimension vector, itself possibly optimized. }
  M := ReadByteAt;
  if Chr(M) = '$' then
  begin
    T := ReadByteAt;
    if Chr(ReadByteAt) <> '#' then
      raise Exception.Create('a typed dimension vector must give its count');
    n := ReadIntOf(ReadByteAt);
    Result := 1;
    for i := 0 to n - 1 do
    begin
      if ADimCount <= High(ADims) then
      begin
        ADims[ADimCount] := ReadIntOf(T);
        Result := Result * ADims[ADimCount];
        Inc(ADimCount);
      end
      else
        ReadIntOf(T);
    end;
    Exit;
  end;

  { A plain list, ending with ']'. }
  Result := 1;
  while Chr(M) <> ']' do
  begin
    if ADimCount <= High(ADims) then
    begin
      ADims[ADimCount] := ReadIntOf(M);
      Result := Result * ADims[ADimCount];
      Inc(ADimCount);
    end
    else
      ReadIntOf(M);
    M := ReadByteAt;
  end;
end;

function TMcxBJData.MakeBlob(AKind: TMcxArrayKind;
  const ADims: array of Integer; ADimCount: Integer; ACount: Int64): TJSONData;
var
  Obj: TJSONObject;
  Sizes: TJSONArray;
  i, n: Integer;
begin
  n := Length(FBlobs);
  SetLength(FBlobs, n + 1);
  FBlobs[n].Kind := AKind;
  SetLength(FBlobs[n].Dims, ADimCount);
  for i := 0 to ADimCount - 1 do FBlobs[n].Dims[i] := ADims[i];
  FBlobs[n].Data := ReadRaw(ACount * McxElemSize(AKind));

  { The placeholder speaks the same annotations a text file would, so that
    anything reading the tree sees one shape of array and not two. }
  Obj := TJSONObject.Create;
  Obj.Add('_ArrayType_', McxArrayKindName(AKind));
  Sizes := TJSONArray.Create;
  if ADimCount = 0 then
    Sizes.Add(ACount)
  else
    for i := 0 to ADimCount - 1 do Sizes.Add(ADims[i]);
  Obj.Add('_ArraySize_', Sizes);
  Obj.Add('_ArrayIndex_', n);
  Result := Obj;
end;

function TMcxBJData.ReadArray: TJSONData;
var
  M, T: Byte;
  Dims: array[0..15] of Integer;
  DimCount: Integer;
  n, i: Int64;
  Kind: TMcxArrayKind;
  Arr: TJSONArray;
  B: TBytes;
begin
  M := PeekByte;

  if Chr(M) = '$' then
  begin
    ReadByteAt;
    T := ReadByteAt;
    if Chr(ReadByteAt) <> '#' then
      raise Exception.Create('a typed container must give its count');
    n := ReadCount(Dims, DimCount);

    { A marker-only type carries no payload at all: N of them occupy no
      bytes, and reading N bytes for them would walk into the next value. }
    if Chr(T) in ['T', 'F', 'Z'] then
    begin
      Arr := TJSONArray.Create;
      for i := 0 to n - 1 do
        case Chr(T) of
          'T': Arr.Add(True);
          'F': Arr.Add(False);
        else
          Arr.Add(TJSONNull.Create);
        end;
      Exit(Arr);
    end;

    Kind := akNone;
    case Chr(T) of
      'i': Kind := akInt8;   'U': Kind := akUInt8;
      'I': Kind := akInt16;  'u': Kind := akUInt16;
      'l': Kind := akInt32;  'm': Kind := akUInt32;
      'L': Kind := akInt64;  'M': Kind := akUInt64;
      'd': Kind := akSingle; 'D': Kind := akDouble;
    end;
    if Kind = akNone then
      raise Exception.CreateFmt('unsupported container type %s', [Chr(T)]);

    if n > BlobThreshold then Exit(MakeBlob(Kind, Dims, DimCount, n));

    { Small enough to be read as numbers, which is what a shape or a header
      field wants to be. }
    Arr := TJSONArray.Create;
    for i := 0 to n - 1 do
      case Kind of
        akSingle: begin B := ReadRaw(4); Arr.Add(PSingle(@B[0])^); end;
        akDouble: begin B := ReadRaw(8); Arr.Add(PDouble(@B[0])^); end;
      else
        Arr.Add(ReadIntOf(T));
      end;
    Exit(Arr);
  end;

  if Chr(M) = '#' then
  begin
    ReadByteAt;
    n := ReadCount(Dims, DimCount);
    Arr := TJSONArray.Create;
    for i := 0 to n - 1 do Arr.Add(ReadValue);
    Exit(Arr);
  end;

  Arr := TJSONArray.Create;
  while Chr(PeekByte) <> ']' do Arr.Add(ReadValue);
  ReadByteAt;
  Result := Arr;
end;

function TMcxBJData.ReadObject: TJSONData;
var
  M: Byte;
  Obj: TJSONObject;
  Key: string;
  Count: Int64;
  Dims: array[0..15] of Integer;
  DimCount: Integer;
  i: Int64;
begin
  Obj := TJSONObject.Create;
  M := PeekByte;

  if Chr(M) = '$' then
    raise Exception.Create('a typed object is not something mcx writes');

  if Chr(M) = '#' then
  begin
    ReadByteAt;
    Count := ReadCount(Dims, DimCount);
    for i := 0 to Count - 1 do
    begin
      Key := ReadString;
      Obj.Add(Key, ReadValue);
    end;
    Exit(Obj);
  end;

  { Unsized, ending with a close-object marker, which is what mcx writes.  A
    key carries no S marker: it is an integer marker, a length and the
    text. }
  while Chr(PeekByte) <> '}' do
  begin
    Key := ReadString;
    Obj.Add(Key, ReadValue);
  end;
  ReadByteAt;
  Result := Obj;
end;

function TMcxBJData.ReadValue: TJSONData;
var
  M: Byte;
  B: TBytes;
begin
  M := ReadByteAt;
  { A no-op can appear anywhere and means nothing. }
  while Chr(M) = 'N' do M := ReadByteAt;

  case Chr(M) of
    'Z': Result := TJSONNull.Create;
    'T': Result := TJSONBoolean.Create(True);
    'F': Result := TJSONBoolean.Create(False);
    'i', 'U', 'I', 'u', 'l', 'm', 'L', 'M':
      Result := TJSONInt64Number.Create(ReadIntOf(M));
    'd': begin B := ReadRaw(4); Result := TJSONFloatNumber.Create(PSingle(@B[0])^); end;
    'D': begin B := ReadRaw(8); Result := TJSONFloatNumber.Create(PDouble(@B[0])^); end;
    'C': begin B := ReadRaw(1); Result := TJSONString.Create(Chr(B[0])); end;
    'S': Result := TJSONString.Create(ReadString);
    { High precision is an arbitrary-length decimal; kept as text, which is
      the only lossless thing to do with it. }
    'H': Result := TJSONString.Create(ReadString);
    '[': Result := ReadArray;
    '{': Result := ReadObject;
  else
    raise Exception.CreateFmt('unknown marker %s at %d',
      [Chr(M), FStream.Position - 1]);
  end;
end;

function TMcxBJData.LoadFromStream(AStream: TStream): Boolean;
begin
  Result := False;
  FError := '';
  FreeAndNil(FRoot);
  SetLength(FBlobs, 0);
  FStream := AStream;
  try
    FRoot := ReadValue;
    Result := FRoot <> nil;
  except
    on E: Exception do
    begin
      FError := E.Message;
      FreeAndNil(FRoot);
    end;
  end;
end;

function TMcxBJData.LoadFromFile(const AFileName: string): Boolean;
var
  F: TFileStream;
begin
  Result := False;
  try
    F := TFileStream.Create(AFileName, fmOpenRead or fmShareDenyNone);
  except
    on E: Exception do
    begin
      FError := E.Message;
      Exit;
    end;
  end;
  try
    Result := LoadFromStream(F);
  finally
    F.Free;
  end;
end;

function TMcxBJData.GetArrayOf(AObj: TJSONObject; out AArray: TMcxArray): Boolean;
var
  Idx: TJSONData;
begin
  Result := False;
  if AObj = nil then Exit;
  Idx := AObj.Find('_ArrayIndex_');
  if (Idx <> nil) and (Idx.AsInteger >= 0) and (Idx.AsInteger <= High(FBlobs)) then
  begin
    AArray := FBlobs[Idx.AsInteger];
    Exit(True);
  end;
  Result := McxDecodeJData(AObj, AArray);
end;

function TMcxBJData.GetArray(const APath: string; out AArray: TMcxArray): Boolean;
var
  D: TJSONData;
begin
  Result := False;
  if FRoot = nil then Exit;
  D := FRoot.FindPath(APath);
  if (D = nil) or (D.JSONType <> jtObject) then Exit;
  Result := GetArrayOf(TJSONObject(D), AArray);
end;

{ ------------------------------------------------------------ loading ----- }

{ Walks a tree for the first object that carries JData annotations.  Used
  when the caller does not know, or care, what the file calls its array --
  .jnii says NIFTIData, .jdb says whatever mcx was asked to name it. }
function FindFirstArray(AData: TJSONData; out AObj: TJSONObject): Boolean;
var
  i: Integer;
begin
  Result := False;
  AObj := nil;
  if AData = nil then Exit;
  if AData.JSONType = jtObject then
  begin
    if TJSONObject(AData).Find('_ArrayType_') <> nil then
    begin
      AObj := TJSONObject(AData);
      Exit(True);
    end;
    for i := 0 to AData.Count - 1 do
      if FindFirstArray(TJSONObject(AData).Items[i], AObj) then Exit(True);
  end
  else if AData.JSONType = jtArray then
    for i := 0 to AData.Count - 1 do
      if FindFirstArray(TJSONArray(AData).Items[i], AObj) then Exit(True);
end;

function McxLoadArray(const AFileName, APath: string;
  out AArray: TMcxArray): Boolean;
var
  Ext: string;
  BJ: TMcxBJData;
  Text: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
begin
  Result := False;
  Ext := LowerCase(ExtractFileExt(AFileName));

  if (Ext = '.bnii') or (Ext = '.jdb') or (Ext = '.bjd') then
  begin
    BJ := TMcxBJData.Create;
    try
      if not BJ.LoadFromFile(AFileName) then Exit;
      if APath <> '' then Exit(BJ.GetArray(APath, AArray));
      if not FindFirstArray(BJ.Root, Obj) then Exit;
      Result := BJ.GetArrayOf(Obj, AArray);
    finally
      BJ.Free;
    end;
    Exit;
  end;

  { Text JSON: .jnii, .json, anything else. }
  Text := TStringList.Create;
  Root := nil;
  try
    try
      Text.LoadFromFile(AFileName);
      Root := GetJSON(Text.Text);
    except
      Exit(False);
    end;
    if APath <> '' then
    begin
      Obj := nil;
      if Root.FindPath(APath) is TJSONObject then
        Obj := TJSONObject(Root.FindPath(APath));
    end
    else if not FindFirstArray(Root, Obj) then
      Obj := nil;
    if Obj = nil then Exit;
    Result := McxDecodeJData(Obj, AArray);
  finally
    Root.Free;
    Text.Free;
  end;
end;

end.
