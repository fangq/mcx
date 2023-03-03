{#####################################################################################}
{##                                                                                 ##}
{## Texture_3D_Unit                                                                 ##}
{##                                                                                 ##}
{## A small sample class for 3D textures                                            ##}
{##                                                                                 ##}
{## History                                                                         ##}
{##   13.12.2003 : Jürgen Abel  : First version                                     ##}
{##                                                                                 ##}
{#####################################################################################}


Unit texture_3d;

{$mode objfpc}{$H+}

interface


uses
  Classes,
  SysUtils,
  Graphics,
  Controls,
  Forms,
  Dialogs,
  ExtCtrls,
  ComCtrls,
  StdCtrls,
  OpenGLTokens,
  GLScene,
  GLObjects,
  GLLCLViewer,
  GLTexture,
  GLContext,
  GLVectorGeometry,
  GLCadencer,
  fpjson,
  jsonparser,
  zstream,
  base64;


Type
  TTexture_3D = Class (TObject)
  protected
    DataFormat : Longword;
    XDim : Integer;
    YDim : Integer;
    ZDim : Integer;
    TDim : Integer;
    TexelByte : Integer;
    M_Data : String;
    JDataType: TStringList;
    function Data_Type_To_Texel_Byte_Size (F_Data_Type : Integer) : Integer;
    procedure Set_Data_Type (F_Value : Longword);
    procedure Set_X_Size (F_Value : Integer);
    procedure Set_Y_Size (F_Value : Integer);
    procedure Set_Z_Size (F_Value : Integer);
    procedure Set_Data (Const F_Value : String);
  public
    constructor Create;
    destructor Destroy; override;
    procedure Save_To_File (const F_FileName : String);
    procedure Load_From_File (const F_FileName : String);
    procedure Load_From_JNIFTI_File (const F_FileName : String);
    procedure Load_From_File_Skip_Header(const F_FileName : string; nx, ny, nz:integer; nt: integer=1; skipbyte: integer=0; datatype: LongWord=GL_RGBA32F);
    procedure Load_From_File_Log_Float (Const F_FileName : String; skipbyte: integer=0; datatype: LongWord=GL_INVALID_VALUE);
    procedure Load_Texture_From_Stream (datastream: TStream; nx, ny, nz:integer; nt: integer=1; datatype: LongWord=GL_RGBA32F; colmajor: Boolean=true);
    property Data_Type : Longword read DataFormat write Set_Data_Type;
    property X_Size : Integer read XDim write Set_X_Size;
    property Y_Size : Integer read YDim write Set_Y_Size;
    property Z_Size : Integer read ZDim write Set_Z_Size;
    property Texel_Byte_Size : Integer read TexelByte;
    property Data : string read M_Data write Set_Data;
  end; { TTexture_3D }


//============================================================
implementation
//============================================================

{-------------------------------------------------------------------------------------}
{ Initializes dynamical data                                                          }
{-------------------------------------------------------------------------------------}
constructor TTexture_3D.Create;
begin
  inherited Create;

  { Initialize variables }
  DataFormat := GL_RGBA;
  XDim := 0;
  YDim := 0;
  ZDim := 0;
  TDim := 1;
  TexelByte := Data_Type_To_Texel_Byte_Size (DataFormat);
  SetLength (M_Data, XDim * YDim * ZDim * TDim * TexelByte);
  JDataType:= TStringList.Create;
  JDataType.AddObject('int8', TObject(GL_LUMINANCE));
  JDataType.AddObject('uint8', TObject(GL_LUMINANCE));
  JDataType.AddObject('int16', TObject(GL_RGBA16I));
  JDataType.AddObject('uint16', TObject(GL_RGBA16I));
  JDataType.AddObject('int32', TObject(GL_RGBA32I));
  JDataType.AddObject('uint32', TObject(GL_RGBA32I));
  JDataType.AddObject('single', TObject(GL_RGBA32F));
  JDataType.AddObject('double', TObject(GL_DOUBLE_EXT));
end;


{-------------------------------------------------------------------------------------}
{ Release dynamic data                                                                }
{-------------------------------------------------------------------------------------}
destructor TTexture_3D.Destroy;
begin
  { Free data }
  SetLength (M_Data, 0);
  JDataType.Free;
  Inherited Destroy;
end;

{-------------------------------------------------------------------------------------}
{ Calculates texel size in bytes                                                      }
{-------------------------------------------------------------------------------------}
function TTexture_3D.Data_Type_To_Texel_Byte_Size (F_Data_Type : Integer) : Integer;
begin
  case F_Data_Type Of
    GL_COLOR_INDEX : Result := 1;
    GL_STENCIL_INDEX : Result := 1;
    GL_DEPTH_COMPONENT : Result := 1;
    GL_RED : Result := 1;
    GL_GREEN : Result := 1;
    GL_BLUE : Result := 1;
    GL_ALPHA : Result := 1;
    GL_RGB : Result := 3;
    GL_RGBA : Result := 4;
    GL_RGB8I: Result:=1;
    GL_RGB8UI: Result:=1;
    GL_RGBA16I: Result:=2;
    GL_RGB16UI: Result:=2;
    GL_RGB32I: Result:= 4;
    GL_RGB32UI: Result:= 4;
    GL_LUMINANCE : Result := 1;
    GL_LUMINANCE_ALPHA : Result := 2;
    GL_RGBA32F: Result:= 4;
    GL_DOUBLE_EXT: Result:=8;
    else
      Result := 4;
  end; { Case }
end;


{-------------------------------------------------------------------------------------}
{ Sets data type                                                                      }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Set_Data_Type (F_Value : Longword);
begin
  DataFormat := F_Value;

  TexelByte := Data_Type_To_Texel_Byte_Size (DataFormat);
  SetLength (M_Data, XDim * YDim * ZDim  * TDim * TexelByte);
end;


{-------------------------------------------------------------------------------------}
{ Sets X size                                                                         }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Set_X_Size (F_Value : Integer);
begin
  XDim := F_Value;

  SetLength (M_Data, XDim * YDim * ZDim * TDim * TexelByte);
end;


{-------------------------------------------------------------------------------------}
{ Sets Y size                                                                         }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Set_Y_Size (F_Value : Integer);
begin
  YDim := F_Value;

  SetLength (M_Data, XDim * YDim * ZDim * TDim * TexelByte);
end;


{-------------------------------------------------------------------------------------}
{ Sets Z size                                                                         }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Set_Z_Size (F_Value : Integer);
begin
  ZDim := F_Value;

  SetLength (M_Data, XDim * YDim * ZDim * TDim * TexelByte);
end;


{-------------------------------------------------------------------------------------}
{ Sets data                                                                           }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Set_Data (Const F_Value : String);
begin
  SetLength (M_Data, Length (F_Value));
  M_Data := F_Value;
end;


{-------------------------------------------------------------------------------------}
{ Save data to file                                                                   }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Save_To_File (Const F_FileName : String);
var
  File_Stream : TFileStream;
begin
  File_Stream := TFileStream.Create (F_FileName, fmCreate or fmShareDenyWrite);
  try
    File_Stream.WriteBuffer (DataFormat, SizeOf (Longword));
    File_Stream.WriteBuffer (XDim, SizeOf (Integer));
    File_Stream.WriteBuffer (YDim, SizeOf (Integer));
    File_Stream.WriteBuffer (ZDim, SizeOf (Integer));
    File_Stream.WriteBuffer (PChar (M_Data)^, Length (M_Data));
  finally
    File_Stream.Free;
  end;
end;

function Base64ToStream(const ABase64: String; var AStream: TMemoryStream): Boolean;
var
  Str: String;
begin
  Result := False;
  if Length(Trim(ABase64)) = 0 then
    Exit;
  try
    Str := DecodeStringBase64(ABase64);
    AStream.Write(Pointer(Str)^, Length(Str) div SizeOf(Char));
    AStream.Position := 0;
    Result := AStream.Size > 0;
  except
    on E: Exception do
      ShowMessage(E.Message);
  end;
end;

{-------------------------------------------------------------------------------------}
{ Load data from file                                                                 }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Load_From_JNIFTI_File (Const F_FileName : String);
var
  jniiparser : TJSONParser;
  jniifile : TFileStream;
  jnii: TJSONData;
  niidata: TJSONObject;
  bufin, bufout: TMemoryStream;
  unzip: TDecompressionStream;
  idx: integer;

begin { TTexture_3D.Load_From_File }
  Screen.Cursor := crHourGlass;
  bufout:=TMemoryStream.Create;
  try
    jniifile:=TFileStream.Create(F_FileName,fmOpenRead);
    try
      jniiparser:=TJSONParser.Create(jniifile);
      try
        jnii:=jniiparser.Parse;
      finally
        jniiparser.Free;
      end;
    finally
      jniifile.Free;
    end;

    niidata:= TJSONObject(jnii.FindPath('NIFTIData'));
    if(niidata <> nil) and (niidata.FindPath('_ArrayZipData_') <> nil) then begin
        bufin:=TMemoryStream.Create;
        try
            Base64ToStream(niidata.FindPath('_ArrayZipData_').AsString, bufin);
            unzip:= TDecompressionStream.Create(bufin);
            bufout.CopyFrom(unzip, 0);
        finally
            unzip.Free;
            bufin.Free;
        end;
    end;
    XDim:=niidata.Arrays['_ArraySize_'].Items[0].AsInteger;
    YDim:=niidata.Arrays['_ArraySize_'].Items[1].AsInteger;
    ZDim:=niidata.Arrays['_ArraySize_'].Items[2].AsInteger;
    TDim:=1;
    if(niidata.Arrays['_ArraySize_'].Count>3) then
         TDim:=niidata.Arrays['_ArraySize_'].Items[3].AsInteger;
    idx := JDataType.IndexOf(niidata.FindPath('_ArrayType_').AsString);
    if idx >= 0 then
         DataFormat := PtrUInt(JDataType.Objects[idx]);

    bufout.Position:=0;
    Load_Texture_From_Stream(bufout, XDim, YDim, ZDim, TDim, DataFormat, false);
  finally
    bufout.Free;
    jnii.Free;
  end;
  Screen.Cursor := crDefault;
end;

{-------------------------------------------------------------------------------------}
{ Load 4D data from file                                                                 }
{-------------------------------------------------------------------------------------}

procedure TTexture_3D.Load_From_File_Skip_Header (const F_FileName : string; nx, ny, nz:integer; nt: integer=1; skipbyte: integer=0; datatype: LongWord=GL_RGBA32F);
begin
     XDim:=nx;
     YDim:=ny;
     ZDim:=nz;
     TDim:=nt;
     DataFormat:=datatype;
     Load_From_File_Log_Float (F_FileName, skipbyte, datatype);
end;

procedure TTexture_3D.Load_Texture_From_Stream (datastream: TStream; nx, ny, nz:integer; nt: integer=1; datatype: LongWord=GL_RGBA32F; colmajor: Boolean=true);
var
  mybuf: array of single;
  val, low, hi: single;
  pdata: pointer;
  ddata: ^double;
  sourcebyte, i, iout, ix, iy, iz, it, nxyz, nxy, nyzt, nzt: integer;
begin
    sourcebyte := Data_Type_To_Texel_Byte_Size (datatype);
    if(sourcebyte = 8) then
        SetLength (mybuf, nx * ny * nz * nt * 2)
    else
        SetLength (mybuf, nx * ny * nz * nt);
    datastream.ReadBuffer (PChar (mybuf)^, nx * ny * nz * nt * sourcebyte);

    DataFormat:=GL_LUMINANCE;
    TexelByte := Data_Type_To_Texel_Byte_Size (DataFormat);
    SetLength (M_Data, nx * ny * nz * nt * TexelByte);

    pdata:=@mybuf[0];
    low:=1e10;
    hi:=-1e10;
    nxyz:=nx*ny*nz;
    nxy:=nx*ny;
    nyzt:=ny*nz*nt;
    nzt:=nz*nt;

    if(sourcebyte = 8) then begin
       ddata:=@mybuf[0];
       for i:=0 to (nx * ny * nz * nt) - 1 do
       begin
             if(ddata[i]<=0) then begin
               val:=0./0.;
             end else begin
               ddata[i]:=log10(ddata[i]);
               val:=ddata[i];
             end;
             if(val<low) then low:=val;
             if(val>hi)  then hi:=val;
       end;
       hi:=1.0/(hi-low)*255;
       for ix:=0 to nx-1 do begin
             for iy:=0 to ny-1 do begin
                   for iz:=0 to nz-1 do begin
                         for it:=0 to nt-1 do begin
                               iout:=(it*nxyz + iz*nxy + iy*nx + ix)+1;
                               i:=(integer(colmajor)*(iout-1) + (1-integer(colmajor))*(ix*nyzt + iy*nzt + iz*nt + it)); ;
                               val:=ddata[i];
                               if(val<low) or (val=0) or (val<>val) then begin
                                  M_Data[iout]:=#0;
                               end else begin
                                  M_Data[iout]:=chr(Round((val-low)*hi));
                               end;
                         end;
                   end;
             end;
       end;
    end else begin
        for i:=0 to (nx * ny * nz * nt) - 1 do
        begin
              case datatype of
                  GL_RGBA32F:
                    if(mybuf[i]<=0) then begin
                      val:=0./0.;
                    end else begin
                      mybuf[i]:=log10(mybuf[i]);
                      val:=mybuf[i];
                    end;
                  GL_RGBA16I: val:=PShortInt(Pointer(nativeuint(pdata) + i*sourcebyte))^;
                  else val:=PByte(Pointer(nativeuint(pdata) + i*sourcebyte))^;
              end;
              if(val<low) then low:=val;
              if(val>hi)  then hi:=val;
        end;
        hi:=1.0/(hi-low)*255;
        for ix:=0 to nx-1 do begin
              for iy:=0 to ny-1 do begin
                    for iz:=0 to nz-1 do begin
                          for it:=0 to nt-1 do begin
                                iout:=(it*nxyz + iz*nxy + iy*nx + ix)+1;
                                i:=(integer(colmajor)*(iout-1) + (1-integer(colmajor))*(ix*nyzt + iy*nzt + iz*nt + it));
                                case datatype of
                                    GL_RGBA32F: val:=mybuf[i];
                                    GL_RGBA16I: val:=PShortInt(Pointer(nativeuint(pdata) + (i)*sourcebyte))^;
                                    else val:=PByte(Pointer(nativeuint(pdata) + (i)*sourcebyte))^;
                                end;
                                if(val<low) or (val=0) or (val<>val) then begin
                                   M_Data[iout]:=#0;
                                end else begin
                                   M_Data[iout]:=chr(Round((val-low)*hi));
                                end;
                          end;
                    end;
              end;
        end;
    end;
    setLength(mybuf, 0);
end;

{-------------------------------------------------------------------------------------}
{ Load data from file                                                                 }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Load_From_File_Log_Float (Const F_FileName : String; skipbyte: integer=0; datatype: LongWord=GL_INVALID_VALUE);
var
  File_Stream : TFileStream;
  val, low, hi: single;
  i,nx,ny,nz,nt,sourcebyte: integer;
  slen: Int16;
  dlen: int64;
begin { TTexture_3D.Load_From_File_Log_Float }
  Screen.Cursor := crHourGlass;
  File_Stream := TFileStream.Create (F_FileName, fmOpenRead or fmShareDenyWrite);
  try
    if(datatype=GL_INVALID_VALUE) then begin  // for tx3 file
        File_Stream.ReadBuffer (datatype, SizeOf (Longword));
        File_Stream.ReadBuffer (nx, SizeOf (Integer));
        File_Stream.ReadBuffer (ny, SizeOf (Integer));
        File_Stream.ReadBuffer (nz, SizeOf (Integer));
        nt:=1;
        XDim:=nx;
        YDim:=ny;
        ZDim:=nz;
        TDim:=nt;
    end else if(skipbyte=352) then begin
        File_Stream.ReadBuffer (skipbyte, SizeOf (Integer));
        if(skipbyte=348) then begin
            File_Stream.Seek(42,soBeginning);
            File_Stream.ReadBuffer (slen, SizeOf (Int16));
            nx:=slen;
            File_Stream.ReadBuffer (slen, SizeOf (Int16));
            ny:=slen;
            File_Stream.ReadBuffer (slen, SizeOf (Int16));
            nz:=slen;
            File_Stream.ReadBuffer (slen, SizeOf (Int16));
            nt:=slen;
        end else begin
            File_Stream.Seek(24,soBeginning);
            File_Stream.ReadBuffer (dlen, SizeOf (Int64));
            nx:=dlen;
            File_Stream.ReadBuffer (dlen, SizeOf (Int64));
            ny:=dlen;
            File_Stream.ReadBuffer (dlen, SizeOf (Int64));
            nz:=dlen;
            File_Stream.ReadBuffer (dlen, SizeOf (Int64));
            nt:=dlen;
        end;
        skipbyte:=skipbyte+4;
        nt:=1;
        XDim:=nx;
        YDim:=ny;
        ZDim:=nz;
        TDim:=nt;
    end else begin
        datatype:= DataFormat;
        nx:= XDim;
        ny:= YDim;
        nz:= ZDim;
        nt:= TDim;
    end;
    if(skipbyte>0) then File_Stream.Seek(skipbyte,soBeginning);
    Load_Texture_From_Stream(File_Stream, nx, ny, nz, nt, datatype);
  finally
    File_Stream.Free;
  end;
  Screen.Cursor := crDefault;
end;

{-------------------------------------------------------------------------------------}
{ Load data from file                                                                 }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Load_From_File (Const F_FileName : String);
var
  File_Stream : TFileStream;

begin { TTexture_3D.Load_From_File }
  Screen.Cursor := crHourGlass;
  File_Stream := TFileStream.Create (F_FileName, fmOpenRead or fmShareDenyWrite);
  try
    File_Stream.ReadBuffer (DataFormat, SizeOf (Longword));
    File_Stream.ReadBuffer (XDim, SizeOf (Integer));
    File_Stream.ReadBuffer (YDim, SizeOf (Integer));
    File_Stream.ReadBuffer (ZDim, SizeOf (Integer));
    TexelByte := Data_Type_To_Texel_Byte_Size (DataFormat);
    SetLength (M_Data, XDim * YDim * ZDim * TexelByte);
    File_Stream.ReadBuffer (PChar (M_Data)^, Length (M_Data));
  finally
    File_Stream.Free;
  end;
  Screen.Cursor := crDefault;
end;

end.

