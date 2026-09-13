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
  Classes, SysUtils, Math, base64, zstream, fpjson, jsonparser;

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

  TMcxArrayList = array of TMcxArray;

function McxArrayKindOf(const AName: string): TMcxArrayKind;
function McxArrayKindName(AKind: TMcxArrayKind): string;
function McxElemSize(AKind: TMcxArrayKind): Integer;
function McxArrayCount(const AArray: TMcxArray): Int64;
{ One element, as a double whatever it was stored as.  The raycaster wants
  floats and the media table wants integers; neither should care. }
function McxArrayValue(const AArray: TMcxArray; AIndex: Int64): Double;
{ The smallest and largest values present, for scaling a colour map.  False
  when nothing in the array is a finite number, which is a whole result that
  cannot be scaled rather than a range of zero. }
function McxArrayRange(const AArray: TMcxArray; out ALow, AHigh: Double): Boolean;

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
    { The bytes behind a placeholder, when the node is one. }
    function  BlobOf(AData: TJSONData; out AArray: TMcxArray): Boolean;
    { The same bytes, whether the reader kept them or expanded them. }
    function  BytesOf(AData: TJSONData; out ABytes: TBytes): Boolean;
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

type
  TMcxOrder = array of Integer;

{ An ordering of trajectory events that puts each photon's path together and
  keeps its events in the order they were written.

  mcx's threads append to one buffer through an atomic counter, so a photon's
  events are scattered through the file: in a real run of two hundred photons
  the identifier changes nearly nine hundred times.  Drawn in file order that
  is a spray of lines between unrelated points.

  The tie on the original position is what makes it a path rather than a set
  of points: sorted by identifier alone, a photon's events could come back in
  any order and the line would zigzag through them. }
function McxSortTrajectory(const AIds: TMcxArray): TMcxOrder;

{ Opens a .jnii, .bnii or .jdb and hands back the array at APath -- or the
  first one it can find, when APath is empty. }
function McxLoadArray(const AFileName, APath: string;
  out AArray: TMcxArray): Boolean;

{ Several arrays out of one file, read once.

  The trajectory reader wants three -- the photon ids, the positions and the
  weights -- and asking for them one at a time parses the whole file three
  times.  For a text .jdt that is the whole cost of loading it.

  A path that is not there gets an empty array rather than failing the lot:
  w0 is optional, and a file without it is still a set of paths. }
function McxLoadArrays(const AFileName: string; const APaths: array of string;
  out AArrays: TMcxArrayList): Boolean;

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

{ Non-finite values are stepped over rather than compared.

  A run that diverges writes a volume of NaN -- mcx's own bin/example_session
  .jnii is 216000 of them and nothing else -- and NaN is unordered, so "V <
  ALow" is not false, it is an invalid operation.  FPC leaves that unmasked on
  x86_64, so the comparison raises EInvalidOp and reading the file becomes a
  crash rather than a result with nothing in it.

  Skipping them is also the right answer for a volume that is only partly
  spoiled: the colour map is scaled by the values that are numbers, and the
  ones that are not are no more a minimum than an empty voxel is. }
function McxArrayRange(const AArray: TMcxArray; out ALow, AHigh: Double): Boolean;
var
  i, n: Int64;
  V: Double;
begin
  Result := False;
  ALow := 0;
  AHigh := 0;
  n := McxArrayCount(AArray);
  for i := 0 to n - 1 do
  begin
    V := McxArrayValue(AArray, i);
    if IsNan(V) or IsInfinite(V) then Continue;
    if not Result then
    begin
      ALow := V;
      AHigh := V;
      Result := True;
    end
    else
    begin
      if V < ALow then ALow := V;
      if V > AHigh then AHigh := V;
    end;
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

{ What an annotated array says it is -- the element type and the shape --
  without touching the payload.  Split out because a binary file carries the
  same two annotations as a text one and differs only in what the payload is
  made of: base64 in a string there, bytes already read here. }
function DecodeShape(AObj: TJSONObject; out AArray: TMcxArray): Boolean;
var
  T, S: TJSONData;
  i: Integer;
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
  Result := True;
end;

function McxDecodeJData(AObj: TJSONObject; out AArray: TMcxArray): Boolean;
var
  Z, D: TJSONData;
  i: Integer;
  Raw: TBytes;
  Text: string;
begin
  Result := False;
  if not DecodeShape(AObj, AArray) then Exit;

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

    { A container of strings, which is not a bulk array of anything: its
      elements are their own lengths.  The type marker is not repeated on
      each one -- the container already gave it -- so an element is what
      follows an S rather than an S, exactly as an object key is.

      This is not a corner of the format.  Every .bnii and .jdb mcx writes
      opens with _DataInfo_, and _DataInfo_.Parser lists three of these
      (mcx_utils.c:673) before the file says anything else, so refusing them
      refused every binary result mcx has ever produced.  C is one byte and
      H is a decimal written as text; both arrive the same way. }
    if Chr(T) in ['S', 'H', 'C'] then
    begin
      Arr := TJSONArray.Create;
      for i := 0 to n - 1 do
        if Chr(T) = 'C' then
        begin
          B := ReadRaw(1);
          Arr.Add(Chr(B[0]));
        end
        else
          Arr.Add(ReadString);
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

function TMcxBJData.BlobOf(AData: TJSONData; out AArray: TMcxArray): Boolean;
var
  Idx: TJSONData;
begin
  Result := False;
  if not (AData is TJSONObject) then Exit;
  Idx := TJSONObject(AData).Find('_ArrayIndex_');
  if Idx = nil then Exit;
  if (Idx.AsInteger < 0) or (Idx.AsInteger > High(FBlobs)) then Exit;
  AArray := FBlobs[Idx.AsInteger];
  Result := True;
end;

{ A compressed payload is a run of bytes whichever way it came back: kept as
  a blob when it is long, and expanded into numbers when it is shorter than
  the threshold -- which a small array in a binary file is, and which is the
  one shape of this that no real result file happens to have. }
function TMcxBJData.BytesOf(AData: TJSONData; out ABytes: TBytes): Boolean;
var
  Blob: TMcxArray;
  i: Integer;
begin
  Result := False;
  SetLength(ABytes, 0);
  if BlobOf(AData, Blob) then
  begin
    ABytes := Blob.Data;
    Exit(True);
  end;
  if AData is TJSONArray then
  begin
    SetLength(ABytes, AData.Count);
    for i := 0 to AData.Count - 1 do ABytes[i] := Byte(AData.Items[i].AsInt64);
    Exit(True);
  end;
end;

function TMcxBJData.GetArrayOf(AObj: TJSONObject; out AArray: TMcxArray): Boolean;
var
  Z, D: TJSONData;
  Payload: TMcxArray;
  Raw: TBytes;
begin
  Result := False;
  if AObj = nil then Exit;

  { The object is a bulk array we lifted out of the tree on the way past. }
  if BlobOf(AObj, AArray) then Exit(True);

  { Otherwise it is an annotated array, and the payload is one.  A text file
    holds that payload as base64 inside a string, which McxDecodeJData has;
    a binary one has it as bytes already, so the base64 stage does not exist
    and reading it as a string is what raised "cannot convert data from
    object value" on every compressed .bnii mcx writes.

    The shape comes from the annotations around it either way: the blob was
    read as the uint8 the container said it was, and what it holds is a
    volume of singles. }
  if not DecodeShape(AObj, AArray) then Exit;

  Z := AObj.Find('_ArrayZipData_');
  if (Z <> nil) and BytesOf(Z, Raw) then
  begin
    AArray.Data := Inflate(Raw);
    Exit(Length(AArray.Data) > 0);
  end;

  D := AObj.Find('_ArrayData_');
  if (D <> nil) and BlobOf(D, Payload) then
  begin
    AArray.Data := Payload.Data;
    Exit(Length(AArray.Data) > 0);
  end;

  { Neither payload is a blob: a small array that was expanded into numbers,
    or a text-shaped object that found its way in here. }
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

{ ---------------------------------------------------------- trajectory ---- }

procedure SortOrder(var AOrder: TMcxOrder; const AIds: TMcxArray;
  L, R: Integer);
var
  i, j, T, PivId, PivIdx: Integer;

  { True when the event at index A comes before the one at index B. }
  function Less(A, B, BId: Integer): Boolean;
  var
    AId: Integer;
  begin
    AId := Round(McxArrayValue(AIds, A));
    if AId <> BId then Exit(AId < BId);
    Result := A < B;
  end;

begin
  i := L;
  j := R;
  PivIdx := AOrder[(L + R) div 2];
  PivId := Round(McxArrayValue(AIds, PivIdx));
  repeat
    while Less(AOrder[i], PivIdx, PivId) do Inc(i);
    while Less(PivIdx, AOrder[j], Round(McxArrayValue(AIds, AOrder[j]))) do Dec(j);
    if i <= j then
    begin
      T := AOrder[i]; AOrder[i] := AOrder[j]; AOrder[j] := T;
      Inc(i);
      Dec(j);
    end;
  until i > j;
  if L < j then SortOrder(AOrder, AIds, L, j);
  if i < R then SortOrder(AOrder, AIds, i, R);
end;

function McxSortTrajectory(const AIds: TMcxArray): TMcxOrder;
var
  i, n: Integer;
begin
  n := McxArrayCount(AIds);
  SetLength(Result, n);
  for i := 0 to n - 1 do Result[i] := i;
  if n > 1 then SortOrder(Result, AIds, 0, n - 1);
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

function McxLoadArrays(const AFileName: string; const APaths: array of string;
  out AArrays: TMcxArrayList): Boolean;
var
  Ext: string;
  BJ: TMcxBJData;
  Text: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  i: Integer;
begin
  Result := False;
  SetLength(AArrays, Length(APaths));
  Ext := LowerCase(ExtractFileExt(AFileName));

  if (Ext = '.bnii') or (Ext = '.jdb') or (Ext = '.bjd') then
  begin
    BJ := TMcxBJData.Create;
    try
      if not BJ.LoadFromFile(AFileName) then Exit;
      for i := 0 to High(APaths) do
        if BJ.GetArray(APaths[i], AArrays[i]) then Result := True;
    finally
      BJ.Free;
    end;
    Exit;
  end;

  Text := TStringList.Create;
  Root := nil;
  try
    try
      Text.LoadFromFile(AFileName);
      Root := GetJSON(Text.Text);
    except
      Exit(False);
    end;
    for i := 0 to High(APaths) do
      if Root.FindPath(APaths[i]) is TJSONObject then
      begin
        Obj := TJSONObject(Root.FindPath(APaths[i]));
        if McxDecodeJData(Obj, AArrays[i]) then Result := True;
      end;
  finally
    Root.Free;
    Text.Free;
  end;
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
