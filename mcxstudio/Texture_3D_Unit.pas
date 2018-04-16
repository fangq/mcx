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


Unit Texture_3D_Unit;


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
    procedure LoadFromFile(const F_FileName : String; nx,ny,nz: integer; datatype Longword=GL_R8);
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

{-------------------------------------------------------------------------------------}
{ Load data from file                                                                 }
{-------------------------------------------------------------------------------------}
procedure TTexture_3D.LoadFromFile(const F_FileName : String; nx,ny,nz: integer; datatype Longword=GL_R8);
var
  File_Stream : TFileStream;

begin { TTexture_3D.LoadFromFile }
  Screen.Cursor := crHourGlass;
  File_Stream := TFileStream.Create (F_FileName, fmOpenRead or fmShareDenyWrite);
  try
    M_Data_Type:=datatype;
    M_X_Size:=nx;
    M_Y_Size:=ny;
    M_Z_Size:=nz;
    M_Texel_Byte_Size := Data_Type_To_Texel_Byte_Size (M_Data_Type);
    SetLength (M_Data, M_X_Size * M_Y_Size * M_Z_Size * M_Texel_Byte_Size);
    File_Stream.ReadBuffer (PChar (M_Data)^, Length (M_Data));
  finally
    File_Stream.Free;
  end;
  Screen.Cursor := crDefault;
end;

end.

