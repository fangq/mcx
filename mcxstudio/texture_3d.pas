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

{$mode delphi}{$H+}

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
  GLCadencer;


Type
  TTexture_3D = Class (TObject)
  protected
    M_Data_Type : Longword;
    M_X_Size : Integer;
    M_Y_Size : Integer;
    M_Z_Size : Integer;
    M_Texel_Byte_Size : Integer;
    M_Data : String;
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
    procedure Load_From_File_Log_Float (Const F_FileName : String; datatype: LongWord=GL_INVALID_VALUE);
    procedure Load_From_File_No_Header (const F_FileName : string; nx, ny, nz: integer;  datatype: LongWord=GL_RGBA32F);
    property Data_Type : Longword read M_Data_Type write Set_Data_Type;
    property X_Size : Integer read M_X_Size write Set_X_Size;
    property Y_Size : Integer read M_Y_Size write Set_Y_Size;
    property Z_Size : Integer read M_Z_Size write Set_Z_Size;
    property Texel_Byte_Size : Integer read M_Texel_Byte_Size;
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
  M_Data_Type := GL_RGBA;
  M_X_Size := 0;
  M_Y_Size := 0;
  M_Z_Size := 0;
  M_Texel_Byte_Size := Data_Type_To_Texel_Byte_Size (M_Data_Type);
  SetLength (M_Data, M_X_Size * M_Y_Size * M_Z_Size * M_Texel_Byte_Size);
end;


{-------------------------------------------------------------------------------------}
{ Release dynamic data                                                                }
{-------------------------------------------------------------------------------------}
destructor TTexture_3D.Destroy;
begin
  { Free data }
  SetLength (M_Data, 0);
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
    GL_LUMINANCE : Result := 1;
    GL_LUMINANCE_ALPHA : Result := 2;
    else
      Result := 4;
  end; { Case }
end;


{-------------------------------------------------------------------------------------}
{ Sets data type                                                                      }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Set_Data_Type (F_Value : Longword);
begin
  M_Data_Type := F_Value;

  M_Texel_Byte_Size := Data_Type_To_Texel_Byte_Size (M_Data_Type);
  SetLength (M_Data, M_X_Size * M_Y_Size * M_Z_Size * M_Texel_Byte_Size);
end;


{-------------------------------------------------------------------------------------}
{ Sets X size                                                                         }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Set_X_Size (F_Value : Integer);
begin
  M_X_Size := F_Value;

  SetLength (M_Data, M_X_Size * M_Y_Size * M_Z_Size * M_Texel_Byte_Size);
end;


{-------------------------------------------------------------------------------------}
{ Sets Y size                                                                         }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Set_Y_Size (F_Value : Integer);
begin
  M_Y_Size := F_Value;

  SetLength (M_Data, M_X_Size * M_Y_Size * M_Z_Size * M_Texel_Byte_Size);
end;


{-------------------------------------------------------------------------------------}
{ Sets Z size                                                                         }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Set_Z_Size (F_Value : Integer);
begin
  M_Z_Size := F_Value;

  SetLength (M_Data, M_X_Size * M_Y_Size * M_Z_Size * M_Texel_Byte_Size);
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
    File_Stream.WriteBuffer (M_Data_Type, SizeOf (Longword));
    File_Stream.WriteBuffer (M_X_Size, SizeOf (Integer));
    File_Stream.WriteBuffer (M_Y_Size, SizeOf (Integer));
    File_Stream.WriteBuffer (M_Z_Size, SizeOf (Integer));
    File_Stream.WriteBuffer (PChar (M_Data)^, Length (M_Data));
  finally
    File_Stream.Free;
  end;
end;


{-------------------------------------------------------------------------------------}
{ Load data from file                                                                 }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Load_From_File_No_Header (const F_FileName : string; nx, ny, nz: integer; datatype: LongWord=GL_RGBA32F);
begin
     M_X_Size:=nx;
     M_X_Size:=ny;
     M_X_Size:=nz;
     Load_From_File_Log_Float (F_FileName, datatype);
end;

{-------------------------------------------------------------------------------------}
{ Load data from file                                                                 }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.Load_From_File_Log_Float (Const F_FileName : String; datatype: LongWord=GL_INVALID_VALUE);
var
  File_Stream : TFileStream;
  mybuf: array of single;
  val, low, hi: single;
  i,nx,ny,nz: integer;
begin { TTexture_3D.Load_From_File_Log_Float }
  Screen.Cursor := crHourGlass;
  File_Stream := TFileStream.Create (F_FileName, fmOpenRead or fmShareDenyWrite);
  try
    if(datatype=GL_INVALID_VALUE) then begin
        File_Stream.ReadBuffer (datatype, SizeOf (Longword));
        File_Stream.ReadBuffer (nx, SizeOf (Integer));
        File_Stream.ReadBuffer (ny, SizeOf (Integer));
        File_Stream.ReadBuffer (nz, SizeOf (Integer));
    end else begin
        datatype:= M_Data_Type;
        nx:= M_X_Size;
        ny:= M_Y_Size;
        nz:= M_Z_Size;
    end;
    M_Texel_Byte_Size := Data_Type_To_Texel_Byte_Size (datatype);
    SetLength (mybuf, nx * ny * nz);
    File_Stream.ReadBuffer (PChar (mybuf)^, Length (mybuf)*M_Texel_Byte_Size);

    M_X_Size:=nx;
    M_Y_Size:=ny;
    M_Z_Size:=nz;
    M_Data_Type:=GL_LUMINANCE;
    M_Texel_Byte_Size := Data_Type_To_Texel_Byte_Size (M_Data_Type);
    SetLength (M_Data, nx * ny * nz * M_Texel_Byte_Size);

    low:=1e10;
    hi:=-1e10;
    for i:=0 to nx * ny * nz-1 do
    begin
          if(mybuf[i]<=0) then begin
              val:=0./0.;
          end else begin
              mybuf[i]:=log10(mybuf[i]);
              val:=mybuf[i];
          end;
          if(val<low) then low:=val;
          if(val>hi)  then hi:=val;
    end;
    hi:=1.0/(hi-low)*255;
    for i:=0 to nx * ny * nz-1 do
    begin
          val:=mybuf[i];
          M_Data[i]:=chr(Round((val-low)*hi));
    end;
    setLength(mybuf, 0);
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
    File_Stream.ReadBuffer (M_Data_Type, SizeOf (Longword));
    File_Stream.ReadBuffer (M_X_Size, SizeOf (Integer));
    File_Stream.ReadBuffer (M_Y_Size, SizeOf (Integer));
    File_Stream.ReadBuffer (M_Z_Size, SizeOf (Integer));
    M_Texel_Byte_Size := Data_Type_To_Texel_Byte_Size (M_Data_Type);
    SetLength (M_Data, M_X_Size * M_Y_Size * M_Z_Size * M_Texel_Byte_Size);
    File_Stream.ReadBuffer (PChar (M_Data)^, Length (M_Data));
  finally
    File_Stream.Free;
  end;
  Screen.Cursor := crDefault;
end;

end.

