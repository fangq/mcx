
unit mcxview;

{$mode objfpc}{$H+}

interface


uses
  Classes, SysUtils,
  LCLIntf, LCLType,
  Graphics,
  Controls,
  Forms,
  Dialogs,
  ExtCtrls,
  ComCtrls,
  StdCtrls, ExtDlgs, ActnList, Spin,
  OpenGLTokens,
  GLVectorTypes,
  GLScene,
  GLObjects,
  GLLCLViewer,
  GLTexture,
  GLContext,
  GLVectorGeometry,
  GLCadencer,
  GLCoordinates,
  GLCrossPlatform,
  GLRenderContextInfo,
  GLGraphics, GLWindowsFont, GLBitmapFont, GLGraph,
  texture_3d,
  mcxloadfile,
  Types;

type

  { TfmViewer }

  TfmViewer = class(TForm)
    btBackground: TColorButton;
    mcxplotResetCamera: TAction;
    ColorStep: TTrackBar;
    DCCoordsZ: TGLDummyCube;
    DCCoordsY: TGLDummyCube;
    DCCoordsX: TGLDummyCube;
    GLWinBmpFont: TGLWindowsBitmapFont;
    grDir: TRadioGroup;
    Label4: TLabel;
    mcxplotExit: TAction;
    mcxplotUseColor: TAction;
    mcxplotShowBBX: TAction;
    mcxplotRefresh: TAction;
    mcxplotSaveScreen: TAction;
    mcxplotOpen: TAction;
    acMCXPlot: TActionList;
    dlOpenFile: TOpenDialog;
    dlSaveScreen: TSavePictureDialog;
    GLScene: TGLScene;
    GLCamera: TGLCamera;
    GLDirectOpenGL: TGLDirectOpenGL;
    GLCadencer: TGLCadencer;
    ImageList3: TImageList;
    Panel10: TPanel;
    Timer: TTimer;
    Panel1: TPanel;
    Panel2: TPanel;
    glCanvas: TGLSceneViewer;
    ToolButton4: TToolButton;
    XYGrid: TGLXYZGrid;
    YZGrid: TGLXYZGrid;
    XZGrid: TGLXYZGrid;
    GLLightSource: TGLLightSource;
    Frame: TGLLines;
    glDomain: TGLCube;
    Panel3: TPanel;
    Panel4: TPanel;
    Panel5: TPanel;
    Panel6: TPanel;
    Cutting_Plane_Pos_TB: TTrackBar;
    tbProj: TTrackBar;
    Label1: TLabel;
    Label2: TLabel;
    Panel8: TPanel;
    Panel9: TPanel;
    Label3: TLabel;
    Alpha_Threshold_TB: TTrackBar;
    ToolBar1: TToolBar;
    btRefresh: TToolButton;
    ToolButton1: TToolButton;
    ToolButton10: TToolButton;
    ToolButton2: TToolButton;
    btOpaque: TToolButton;
    btRGB: TToolButton;
    btShowBBX: TToolButton;
    ToolButton3: TToolButton;
    ToolButton9: TToolButton;

    procedure btBackgroundColorChanged(Sender: TObject);
    procedure btOpaqueClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure grDirSelectionChanged(Sender: TObject);
    procedure mcxplotExitExecute(Sender: TObject);
    procedure mcxplotOpenExecute(Sender: TObject);
    Procedure Formshow(Sender : Tobject);
    procedure glCanvasMouseWheel(Sender: TObject; Shift: TShiftState;
      WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
    Procedure LoadTexture(filename: string; nx:integer=0; ny:integer=0; nz: integer=0; nt: integer=1; skipbyte: integer=0; datatype: LongWord=GL_INVALID_VALUE);
    procedure mcxplotRefreshExecute(Sender: TObject);
    procedure mcxplotResetCameraExecute(Sender: TObject);
    procedure mcxplotSaveScreenExecute(Sender: TObject);
    procedure mcxplotShowBBXExecute(Sender: TObject);
    procedure mcxplotUseColorExecute(Sender: TObject);
    procedure GLDirectOpenGLRender(Sender: TObject; var rci: TGLRenderContextInfo);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure ResetTexture;
    procedure glCanvasMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: integer);
    procedure glCanvasMouseMove(Sender: TObject; Shift: TShiftState; X, Y: integer);
    procedure GLCadencerProgress(Sender: TObject; const deltaTime, newTime: double);
    procedure TimerTimer(Sender: TObject);
    procedure Cutting_Plane_Pos_TBChange(Sender: TObject);
    procedure cbShowBBXClick(Sender: TObject);
    procedure Pseudocolor_CBClick(Sender: TObject);
    procedure Opaque_Hull_CBClick(Sender: TObject);
    procedure Alpha_Threshold_TBChange(Sender: TObject);
    procedure DrawAxis(Sender : TObject);
  protected
    procedure Calculate_Transfer_Function;
  public
    M_mx: integer;
    M_my: integer;
    M_3D_Texture: TGLTextureHandle;
    M_Refresh: boolean;
    M_Input_Texture_3D: TTexture_3D;
    M_Output_Texture_3D: TTexture_3D;
    M_CLUT: array [0..255] of integer;
  end;

var
  fmViewer: TfmViewer;
  AxisStep :  TGLFloat =  10;


implementation


{$R *.lfm}


const
  DIAGONAL_LENGTH = 1.732;
  AxisMini :  TGLFloat =  0;


var
  clip0: array [0..3] of double = (-1.0, 0.0, 0.0, 1 / 2.0);
  clip1: array [0..3] of double = (1.0, 0.0, 0.0, 1 / 2.0);
  clip2: array [0..3] of double = (0.0, -1.0, 0.0, 1 / 2.0);
  clip3: array [0..3] of double = (0.0, 1.0, 0.0, 1 / 2.0);
  clip4: array [0..3] of double = (0.0, 0.0, -1.0, 1 / 2.0);
  clip5: array [0..3] of double = (0.0, 0.0, 1.0, 1 / 2.0);
  //Border_Colors: array [0..3] of integer = (0, 0, 0, 0);



procedure TfmViewer.FormCreate(Sender: TObject);
begin
  M_3D_Texture:=nil;
  M_Input_Texture_3D:=nil;
  M_Output_Texture_3D:=nil;
  if(Application.HasOption('f','file')) then begin
        LoadTexture(Application.GetOptionValue('f', 'file'));
  end;
end;

procedure TfmViewer.FormDestroy(Sender: TObject);
begin
   ResetTexture;
end;

procedure TfmViewer.ResetTexture;
begin
  if(M_3D_Texture <> nil) then FreeAndNil(M_3D_Texture);
  if(M_Input_Texture_3D <> nil) then  FreeAndNil(M_Input_Texture_3D);
  if(M_Output_Texture_3D <> nil) then  FreeAndNil(M_Output_Texture_3D);
end;

procedure TfmViewer.glCanvasMouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: integer);
begin
  M_mx := x;
  M_my := y;
end;

procedure TfmViewer.glCanvasMouseMove(Sender: TObject; Shift: TShiftState;
  X, Y: integer);
begin
  if Shift = [ssLeft] then
  begin { then }
    GLCamera.MoveAroundTarget(M_my - y, M_mx - x);
  end; { then }

  if Shift = [ssRight] then
  begin { then }
    GLCamera.Position.AddScaledVector((M_my - y) * 0.01,
      GLCamera.AbsoluteVectorToTarget);
  end; { then }

  M_mx := x;
  M_my := y;
end;

procedure TfmViewer.GLCadencerProgress(Sender: TObject;
  const deltaTime, newTime: double);
begin
  glCanvas.Invalidate;
end;

procedure TfmViewer.TimerTimer(Sender: TObject);
begin
  Caption := 'MCX Studio Volume Renderer ' + glCanvas.FramesPerSecondText;
  glCanvas.ResetPerformanceMonitor;
end;

procedure TfmViewer.Calculate_Transfer_function;
var
  X: integer;
  Y: integer;
  Z: integer;
  Index: integer;
  Value, cid: integer;
  Alpha: integer;
  minx,miny,minz: integer;

begin
  minx:=0;
  miny:=0;
  minz:=0;
  if(grDir.ItemIndex=0) then begin
      minx:=Cutting_Plane_Pos_TB.Position;
  end else if(grDir.ItemIndex=1) then begin
      miny:=Cutting_Plane_Pos_TB.Position;
  end else begin
      minz:=Cutting_Plane_Pos_TB.Position;
  end;
  { Set texture values }
  for Z := 0 to M_Output_Texture_3D.Z_Size - 1 do
  begin { For }
    for Y := 0 to M_Output_Texture_3D.Y_Size - 1 do
    begin { For }
      for X := 0 to M_Output_Texture_3D.X_Size - 1 do
      begin { For }
        Index := (Z * M_Output_Texture_3D.Y_Size * M_Output_Texture_3D.X_Size) + (Y * M_Output_Texture_3D.X_Size) + X;
        Value := PByte(PChar(M_Input_Texture_3D.Data) + Index)^;

        if (Value < Alpha_Threshold_TB.Position) or (X = 0) or
          ((Y>miny) and (X>minx) and (Z>minz)) or
          (X = M_Output_Texture_3D.X_Size - 1) or (Y = 0) or
          (Y = M_Output_Texture_3D.Y_Size - 1) or (Z = 0) or
          (Z = M_Output_Texture_3D.Z_Size - 1) then
        begin { then }
          Alpha := 0;
        end else begin { else }
          if btOpaque.Down = True then
          begin { then }
            Alpha := 255;
          end { then }
          else
          begin { else }
            Alpha := Value;
          end; { else }
        end; { else }

        if btRGB.Down = True then
        begin { then }
          cid:=Value shr (8-ColorStep.Position);
          PLongWord((PChar(M_Output_Texture_3D.Data)) + (Index * 4))^ := M_Clut[cid*(1 shl (8-ColorStep.Position))];
        end else begin { else }
          PLongWord((PChar(M_Output_Texture_3D.Data)) + (Index * 4))^ := Value + (Value shl 8) + (Value shl 16);
        end; { else }

        PByte((PChar(M_Output_Texture_3D.Data)) + (Index * 4) + 3)^ := Alpha;
      end; { For }
    end; { For }
  end; { For }
end;

procedure TfmViewer.GLDirectOpenGLRender(Sender: TObject; var rci: TGLRenderContextInfo);

var
  i: integer;
  step: single;
  z: single;
  mat: TMatrix;
  v: TVector;
  vx, vy, vz: TAffineVector;

begin
  if M_3D_Texture.Handle = 0 then
  begin
    //Assert (GL_EXT_texture3D, 'Graphic card does not support 3D textures.');

    M_3D_Texture.AllocateHandle;

    gl.BindTexture(GL_TEXTURE_3D, M_3D_Texture.Handle);
    gl.TexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    gl.TexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    gl.TexParameterf(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    gl.TexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl.TexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.TexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); //GL_REPLACE);

    gl.TexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, M_Output_Texture_3D.X_Size,
      M_Output_Texture_3D.Y_Size, M_Output_Texture_3D.Z_Size, 0,
      M_Output_Texture_3D.Data_Type, GL_UNSIGNED_BYTE, PChar(M_Output_Texture_3D.Data));
  end;

  if M_Refresh = True then
  begin { then }
    Screen.Cursor := crHourGlass;
    M_Refresh := False;
    Calculate_Transfer_function;
    gl.TexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, M_Output_Texture_3D.X_Size,
                     M_Output_Texture_3D.Y_Size, M_Output_Texture_3D.Z_Size,
                     M_Output_Texture_3D.Data_Type, GL_UNSIGNED_BYTE, PChar(M_Output_Texture_3D.Data));
    Screen.Cursor := crDefault;
  end; { then }

  gl.PushAttrib(GL_ENABLE_BIT);
  gl.PushMatrix;

  gl.Enable(GL_BLEND);
  gl.BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  //gl.Disable (GL_CULL_FACE);
  //gl.Disable (GL_LIGHTING);

  gl.TexGenf(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
  gl.TexGenf(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
  gl.TexGenf(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
  SetVector(v, XVector, 0.5);
  gl.TexGenfv(GL_S, GL_OBJECT_PLANE, @v);
  SetVector(v, YVector, 0.5);
  gl.TexGenfv(GL_T, GL_OBJECT_PLANE, @v);
  SetVector(v, ZVector, 0.5);
  gl.TexGenfv(GL_R, GL_OBJECT_PLANE, @v);
  gl.Enable(GL_TEXTURE_GEN_S);
  gl.Enable(GL_TEXTURE_GEN_T);
  gl.Enable(GL_TEXTURE_GEN_R);

  gl.ClipPlane(GL_CLIP_PLANE0, @clip0);
  gl.ClipPlane(GL_CLIP_PLANE1, @clip1);
  gl.ClipPlane(GL_CLIP_PLANE2, @clip2);
  gl.ClipPlane(GL_CLIP_PLANE3, @clip3);
  gl.ClipPlane(GL_CLIP_PLANE4, @clip4);
  gl.ClipPlane(GL_CLIP_PLANE5, @clip5);

  gl.Enable(GL_CLIP_PLANE0);
  gl.Enable(GL_CLIP_PLANE1);
  gl.Enable(GL_CLIP_PLANE2);
  gl.Enable(GL_CLIP_PLANE3);
  gl.Enable(GL_CLIP_PLANE4);
  gl.Enable(GL_CLIP_PLANE5);

  gl.BindTexture(GL_TEXTURE_3D, M_3D_Texture.Handle);
  gl.Enable(GL_TEXTURE_3D);

  gl.GetFloatv(GL_MODELVIEW_MATRIX, @mat);
  vx.X := mat.X.X;
  vy.X := mat.X.Y;
  vz.X := mat.X.Z;
  vx.Y := mat.Y.X;
  vy.Y := mat.Y.Y;
  vz.Y := mat.Y.Z;
  vx.Z := mat.Z.X;
  vy.Z := mat.Z.Y;
  vz.Z := mat.Z.Z;
  ScaleVector(vx, DIAGONAL_LENGTH * 0.5 / VectorLength(vx));
  ScaleVector(vy, DIAGONAL_LENGTH * 0.5 / VectorLength(vy));
  ScaleVector(vz, DIAGONAL_LENGTH * 0.5 / VectorLength(vz));

  step := DIAGONAL_LENGTH / tbProj.Position;
  z := -DIAGONAL_LENGTH / 2;
  gl.begin_(GL_QUADS);
  for i := 0 to tbProj.Position - 1 do
  begin
    gl.Color4f(1.0, 1.0, 1.0, 1.0);

    gl.Normal3f(-GLCamera.AbsoluteVectorToTarget.X,
      -GLCamera.AbsoluteVectorToTarget.Y, -GLCamera.AbsoluteVectorToTarget.Z);

    gl.Vertex3f( vx.X+vy.X+vz.X*z, vx.Y+vy.Y+vz.Y*z, vx.Z+vy.Z+vz.Z*z);
    gl.Vertex3f(-vx.X+vy.X+vz.X*z,-vx.Y+vy.Y+vz.Y*z,-vx.Z+vy.Z+vz.Z*z);
    gl.Vertex3f(-vx.X-vy.X+vz.X*z,-vx.Y-vy.Y+vz.Y*z,-vx.Z-vy.Z+vz.Z*z);
    gl.Vertex3f( vx.X-vy.X+vz.X*z, vx.Y-vy.Y+vz.Y*z, vx.Z-vy.Z+vz.Z*z);
    z := z + step;
  end;
  gl.End_;

  gl.PopMatrix;
  gl.PopAttrib;
end;

Procedure TfmViewer.FormShow(Sender : Tobject);
function HSLtoRGB(H, S, L: single): longword;
const
  OneOverThree = 1 / 3;
var
  M1, M2: single;
  R, G, B: byte;

  function HueToColor(Hue: single): byte;
  var
    V: double;
  begin
    Hue := Hue - Floor(Hue);
    if 6 * Hue < 1 then
      V := M1 + (M2 - M1) * Hue * 6
    else
    if 2 * Hue < 1 then
      V := M2
    else
    if 3 * Hue < 2 then
      V := M1 + (M2 - M1) * (2 / 3 - Hue) * 6
    else
      V := M1;
    Result := Round(255 * V);
  end;

begin
  if S = 0 then
  begin
    R := Round(255 * L);
    G := R;
    B := R;
  end
  else
  begin
    if L <= 0.5 then
      M2 := L * (1 + S)
    else
      M2 := L + S - L * S;
    M1 := 2 * L - M2;
    R := HueToColor(H + OneOverThree);
    G := HueToColor(H);
    B := HueToColor(H - OneOverThree);
  end;
  Result := R + (G shl 8) + (B shl 16);
end;

const
COLOR_CONSTANT_LUMINANCE_HSL_H_OFFSET = 1.5 / 3;
COLOR_CONSTANT_LUMINANCE_HSL_H_FACTOR = 0.85;
COLOR_CONSTANT_LUMINANCE_HSL_S = 0.6;
COLOR_CONSTANT_LUMINANCE_HSL_L = 0.6;
var
  N_begin: integer;
  N_End: integer;
  Left_Value: integer;
  Right_Value: integer;
  I: integer;
  Value: longword;
  H_0: double;
  H_1: double;

begin
  if(M_Output_Texture_3D=nil) then begin
      M_Output_Texture_3D := TTexture_3D.Create;
      M_Output_Texture_3D.Data_Type := GL_RGBA;
      M_Output_Texture_3D.X_Size := 0;
      M_Output_Texture_3D.Y_Size := 0;
      M_Output_Texture_3D.Z_Size := 0;
  end;

  { Calculate Color Lookup Table }
  N_begin := 0;
  N_End := 255;
  Left_Value := 40;
  Right_Value := 255;

  for I := N_begin to Left_Value - 1 do
  begin { For }
    H_1 := COLOR_CONSTANT_LUMINANCE_HSL_H_OFFSET;

    Value := HSLToRGB(H_1, COLOR_CONSTANT_LUMINANCE_HSL_S,
      COLOR_CONSTANT_LUMINANCE_HSL_L);

    M_CLUT[I] := Value;
  end; { For }
  for I := Left_Value to Right_Value do
  begin { For }
    Value := I;

    H_0 := (Int(Value) - Int(Left_Value)) / Int(256);
    H_1 := COLOR_CONSTANT_LUMINANCE_HSL_H_OFFSET -
      (H_0 * COLOR_CONSTANT_LUMINANCE_HSL_H_FACTOR);

    Value := HSLToRGB(H_1, COLOR_CONSTANT_LUMINANCE_HSL_S,
      COLOR_CONSTANT_LUMINANCE_HSL_L);

    M_CLUT[I] := Value;
  end; { For }
  for I := Right_Value + 1 to N_End do
  begin { For }
    H_1 := COLOR_CONSTANT_LUMINANCE_HSL_H_OFFSET -
      (COLOR_CONSTANT_LUMINANCE_HSL_H_FACTOR);

    Value := HSLToRGB(H_1, COLOR_CONSTANT_LUMINANCE_HSL_S,
      COLOR_CONSTANT_LUMINANCE_HSL_L);

    M_CLUT[I] := Value;
  end; { For }
End;

procedure TfmViewer.glCanvasMouseWheel(Sender: TObject; Shift: TShiftState;
  WheelDelta: Integer; MousePos: TPoint; var Handled: Boolean);
begin
  glCamera.AdjustDistanceToTarget(Power(1.1, WheelDelta/1200.0));
end;

Procedure TfmViewer.LoadTexture(filename: string; nx:integer=0; ny:integer=0; nz: integer=0; nt: integer=1; skipbyte: integer=0; datatype: LongWord=GL_INVALID_VALUE);
var
  gridstep: double;
begin
  ResetTexture;
  Screen.Cursor := crHourGlass;
  M_3D_Texture := TGLTextureHandle.Create;

  M_Input_Texture_3D := TTexture_3D.Create;
  if(ExtractFileExt(filename) = '.jnii') then begin
      M_Input_Texture_3D.Load_From_JNIFTI_File(filename);
  end else begin
      if(nx=0) then
         M_Input_Texture_3D.Load_From_File_Log_Float(filename,skipbyte,datatype)
      else
         M_Input_Texture_3D.Load_From_File_Skip_Header(filename,nx,ny,nz,nt,skipbyte,datatype);
  end;

  M_Output_Texture_3D := TTexture_3D.Create;
  M_Output_Texture_3D.Data_Type := GL_RGBA;
  M_Output_Texture_3D.X_Size := M_Input_Texture_3D.X_Size;
  M_Output_Texture_3D.Y_Size := M_Input_Texture_3D.Y_Size;
  M_Output_Texture_3D.Z_Size := M_Input_Texture_3D.Z_Size;

  M_Refresh := True;
  Cutting_Plane_Pos_TB.Max:=M_Output_Texture_3D.X_Size;
  Cutting_Plane_Pos_TB.Position := M_Output_Texture_3D.X_Size div 2;
  Alpha_Threshold_TB.Position := 40;
  tbProj.Position := M_Output_Texture_3D.X_Size;

  Frame.Scale.X:=M_Output_Texture_3D.X_Size;
  Frame.Scale.Y:=M_Output_Texture_3D.Y_Size;
  Frame.Scale.Z:=M_Output_Texture_3D.Z_Size;

  DCCoordsX.Scale.X:=M_Output_Texture_3D.X_Size;
  DCCoordsX.Scale.Y:=M_Output_Texture_3D.Y_Size;
  DCCoordsX.Scale.Z:=M_Output_Texture_3D.Z_Size;
  DCCoordsY.Scale.X:=M_Output_Texture_3D.X_Size;
  DCCoordsY.Scale.Y:=M_Output_Texture_3D.Y_Size;
  DCCoordsY.Scale.Z:=M_Output_Texture_3D.Z_Size;
  DCCoordsZ.Scale.X:=M_Output_Texture_3D.X_Size;
  DCCoordsZ.Scale.Y:=M_Output_Texture_3D.Y_Size;
  DCCoordsZ.Scale.Z:=M_Output_Texture_3D.Z_Size;

  XYGrid.Scale.X:=M_Output_Texture_3D.X_Size;
  XYGrid.Scale.Y:=M_Output_Texture_3D.Y_Size;
  XYGrid.Scale.Z:=M_Output_Texture_3D.Z_Size;
  YZGrid.Scale.X:=M_Output_Texture_3D.X_Size;
  YZGrid.Scale.Y:=M_Output_Texture_3D.Y_Size;
  YZGrid.Scale.Z:=M_Output_Texture_3D.Z_Size;
  XZGrid.Scale.X:=M_Output_Texture_3D.X_Size;
  XZGrid.Scale.Y:=M_Output_Texture_3D.Y_Size;
  XZGrid.Scale.Z:=M_Output_Texture_3D.Z_Size;

  gridstep:=10.0/M_Output_Texture_3D.X_Size;
  XYGrid.XSamplingScale.Step:=gridstep;
  YZGrid.XSamplingScale.Step:=gridstep;
  XZGrid.XSamplingScale.Step:=gridstep;
  gridstep:=10.0/M_Output_Texture_3D.Y_Size;
  XYGrid.YSamplingScale.Step:=gridstep;
  YZGrid.YSamplingScale.Step:=gridstep;
  XZGrid.YSamplingScale.Step:=gridstep;
  gridstep:=10.0/M_Output_Texture_3D.Z_Size;
  XYGrid.ZSamplingScale.Step:=gridstep;
  YZGrid.ZSamplingScale.Step:=gridstep;
  XZGrid.ZSamplingScale.Step:=gridstep;

  GLDirectOpenGL.Scale.X:=M_Output_Texture_3D.X_Size;
  GLDirectOpenGL.Scale.Y:=M_Output_Texture_3D.Y_Size;
  GLDirectOpenGL.Scale.Z:=M_Output_Texture_3D.Z_Size;

  DCCoordsX.Position.X:=-M_Output_Texture_3D.X_Size/2;
  DCCoordsX.Position.Y:=-M_Output_Texture_3D.Y_Size/2;
  DCCoordsX.Position.Z:=-M_Output_Texture_3D.Z_Size/2;
  DCCoordsY.Position.X:=-M_Output_Texture_3D.X_Size/2;
  DCCoordsY.Position.Y:=-M_Output_Texture_3D.Y_Size/2;
  DCCoordsY.Position.Z:=-M_Output_Texture_3D.Z_Size/2;
  DCCoordsZ.Position.X:=-M_Output_Texture_3D.X_Size/2;
  DCCoordsZ.Position.Y:=-M_Output_Texture_3D.Y_Size/2;
  DCCoordsZ.Position.Z:=-M_Output_Texture_3D.Z_Size/2;

  XYGrid.Position.X:=-M_Output_Texture_3D.X_Size/2;
  XYGrid.Position.Y:=-M_Output_Texture_3D.Y_Size/2;
  XYGrid.Position.Z:=-M_Output_Texture_3D.Z_Size/2;
  YZGrid.Position.X:=-M_Output_Texture_3D.X_Size/2;
  YZGrid.Position.Y:=-M_Output_Texture_3D.Y_Size/2;
  YZGrid.Position.Z:=-M_Output_Texture_3D.Z_Size/2;
  XZGrid.Position.X:=-M_Output_Texture_3D.X_Size/2;
  XZGrid.Position.Y:=-M_Output_Texture_3D.Y_Size/2;
  XZGrid.Position.Z:=-M_Output_Texture_3D.Z_Size/2;

  GLCamera.DepthOfView:=2.0*sqrt(Frame.Scale.X*Frame.Scale.X+Frame.Scale.Y*Frame.Scale.Y+Frame.Scale.Z*Frame.Scale.Z);
  GLCamera.Position.X:=M_Output_Texture_3D.X_Size;
  GLCamera.Position.Y:=M_Output_Texture_3D.Y_Size*0.7;
  GLCamera.Position.Z:=M_Output_Texture_3D.Z_Size;

  DrawAxis(nil);

  GLDirectOpenGL.OnRender:=@GLDirectOpenGLRender;
  Screen.Cursor := crDefault;
end;

procedure TfmViewer.mcxplotRefreshExecute(Sender: TObject);
begin
  M_Refresh := True;
  btRefresh.Enabled := false;
end;

procedure TfmViewer.mcxplotResetCameraExecute(Sender: TObject);
begin
   GLCamera.DepthOfView:=2.0*sqrt(Frame.Scale.X*Frame.Scale.X+Frame.Scale.Y*Frame.Scale.Y+Frame.Scale.Z*Frame.Scale.Z);
   GLCamera.Position.X:=Frame.Scale.X;
   GLCamera.Position.Y:=Frame.Scale.Y*0.7;
   GLCamera.Position.Z:=Frame.Scale.Z;
end;

procedure TfmViewer.mcxplotSaveScreenExecute(Sender: TObject);
var
   bm : TBitmap;
   bmp32 : TGLBitmap32;
begin
     bmp32:=glCanvas.Buffer.CreateSnapShot;
     try
        bm:=bmp32.Create32BitsBitmap;
        try
           dlSaveScreen.DefaultExt := GraphicExtension(TBitmap);
           dlSaveScreen.Filter := GraphicFilter(TBitmap);
           if dlSaveScreen.Execute then
              bm.SaveToFile(dlSaveScreen.FileName);
        finally
           bm.Free;
        end;
     finally
        bmp32.Free;
     end;
end;

procedure TfmViewer.mcxplotShowBBXExecute(Sender: TObject);
begin
  Frame.Visible:=btShowBBX.Down;
  M_Refresh := True;
end;

procedure TfmViewer.mcxplotUseColorExecute(Sender: TObject);
begin
  M_Refresh := True;
end;

procedure TfmViewer.mcxplotOpenExecute(Sender: TObject);
const
   dataformat: array [0..3] of LongWord = (GL_RGBA32F,GL_RGB32I,GL_RGBA16I,GL_RGBA8);
var
  fm: TfmDataFile;
  nx,ny,nz,nt: integer;
  skipsize: integer;
  format: LongWord;
  fext: string;
begin
  if(dlOpenFile.Execute) then
  begin
    fext:=ExtractFileExt(dlOpenFile.FileName);
    if(fext <> '.tx3') and (fext <> '.jnii') then begin
        fm:=TfmDataFile.Create(Self);
        fm.edDataFile.FileName:=dlOpenFile.FileName;
        if(fext='.mc2') then begin
            fm.edHeaderSize.Value:=0;
        end else if(fext='.nii') then begin
            fm.edHeaderSize.Value:=352;
        end else if(fext='.img') then begin
            fm.edHeaderSize.Value:=0;
        end;
        fm.ShowModal;
        if(fm.ModalResult<>mrOk) then begin
            fm.Free;
            exit;
        end;
        nx:=fm.edNx.Value;
        ny:=fm.edNy.Value;
        nz:=fm.edNz.Value;
        nt:=fm.edNt.Value;
        skipsize:=fm.edHeaderSize.Value;
        format:=dataformat[fm.edDataFormat.ItemIndex];
        fm.Free;
        LoadTexture(dlOpenFile.FileName, nx,ny,nz,nt,skipsize,format);
    end else if(fext='.jnii') then begin
        LoadTexture(dlOpenFile.FileName);
    end else begin
        LoadTexture(dlOpenFile.FileName);
    end;
  end;
end;

procedure TfmViewer.mcxplotExitExecute(Sender: TObject);
begin
  GLDirectOpenGL.Visible:=false;
  Close;
end;

procedure TfmViewer.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  CloseAction := caFree;
end;

procedure TfmViewer.grDirSelectionChanged(Sender: TObject);
begin
  if(grDir.ItemIndex=0) then begin
      Cutting_Plane_Pos_TB.Max:=M_Output_Texture_3D.X_Size;
  end else if(grDir.ItemIndex=1) then begin
      Cutting_Plane_Pos_TB.Max:=M_Output_Texture_3D.Y_Size;
  end else begin
      Cutting_Plane_Pos_TB.Max:=M_Output_Texture_3D.Z_Size;
  end;
  M_Refresh := True;
end;

procedure TfmViewer.btOpaqueClick(Sender: TObject);
begin
  btRefresh.Enabled := True;
  M_Refresh := True;
end;

procedure TfmViewer.btBackgroundColorChanged(Sender: TObject);
begin
  glCanvas.Buffer.BackgroundColor:=btBackground.ButtonColor;
end;

procedure TfmViewer.Cutting_Plane_Pos_TBChange(Sender: TObject);
begin
  btRefresh.Enabled := True;
  M_Refresh := True;
end;

procedure TfmViewer.Alpha_Threshold_TBChange(Sender: TObject);
begin
  btRefresh.Enabled := True;
  //M_Refresh := True;
end;

procedure TfmViewer.cbShowBBXClick(Sender: TObject);
begin
  Frame.Visible := btShowBBX.Down;
end;

procedure TfmViewer.Pseudocolor_CBClick(Sender: TObject);
begin
  M_Refresh := True;
end;

procedure TfmViewer.Opaque_Hull_CBClick(Sender: TObject);
begin
  M_Refresh := True;
end;


Procedure TfmViewer.DrawAxis(Sender : TObject);
Var
  ScaleFactor : TGLFloat;
  CurrentXCoord: TGLFloat;

  CurrentYCoord: TGLFloat;
  CurrentZCoord: TGLFloat;
  CurrentFlatText: TGLFlatText;
Begin
  DCCoordsX.DeleteChildren;
  DCCoordsY.DeleteChildren;
  DCCoordsZ.DeleteChildren;
  ScaleFactor := 0.0025;
  { Draw X }
  CurrentXCoord := AxisMini;
  CurrentYCoord := 0;
  CurrentZCoord := 0;
  AxisStep:= 10.0/M_Output_Texture_3D.X_Size;
  while CurrentXCoord <= 1.0 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsX);
    with DCCoordsX do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(0, -1, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlBottom; { locate at z maximum }
        //Layout := tlTop; { or tlBottom, tlCenter }
        ModulateColor.AsWinColor := clRed;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentXCoord*M_Output_Texture_3D.X_Size));
      end;
    end;
    CurrentXCoord := CurrentXCoord + AxisStep;
  end;
  CurrentXCoord := AxisMini;
  while CurrentXCoord <= 1.0 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsX);
    with DCCoordsX do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(0, 1, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlBottom; { locate at z maximum }
        // Layout := tlTop; { or tlBottom, tlCenter }
        ModulateColor.AsWinColor := clRed;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentXCoord*M_Output_Texture_3D.X_Size));
      end;
    end;
    CurrentXCoord := CurrentXCoord + AxisStep;
  end;
  { Draw Y }
  CurrentXCoord := 0;
  CurrentYCoord := AxisMini;
  CurrentZCoord := 0;
  AxisStep:= 10.0/M_Output_Texture_3D.Y_Size;
  while CurrentYCoord <= 1.0 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsY);
    with DCCoordsY do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(1, 0, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlBottom; { locate at z maximum }
        // Layout := tlTop; { or tlBottom, tlCenter }
        ModulateColor.AsWinColor := clLime;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentYCoord*M_Output_Texture_3D.Y_Size));
      end;
    end;
    CurrentYCoord := CurrentYCoord + AxisStep;
  end;
  CurrentYCoord := AxisMini;
  while CurrentYCoord <= 1 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsY);
    with DCCoordsY do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(-1, 0, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlBottom; { locate at z maximum }
        // Layout := tlTop; { or tlBottom, tlCenter }
        ModulateColor.AsWinColor := clLime;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentYCoord*M_Output_Texture_3D.Y_Size));
      end;
    end;
    CurrentYCoord := CurrentYCoord + AxisStep;
  end;
  { Draw Z }
  CurrentXCoord := 0;
  CurrentYCoord := 0;
  CurrentZCoord := AxisMini;
  AxisStep:= 10.0/M_Output_Texture_3D.Z_Size;
  while CurrentZCoord <= 1 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsZ);
    with DCCoordsZ do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(0, -1, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlCenter;
        ModulateColor.AsWinColor := clBlue;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentZCoord*M_Output_Texture_3D.Z_Size));
      end;
    end;
    CurrentZCoord := CurrentZCoord + AxisStep;
  end;
  CurrentZCoord := AxisMini;
  while CurrentZCoord <= 1 do
  begin
    TGLFlatText.CreateAsChild(DCCoordsZ);
    with DCCoordsZ do
    begin
      CurrentFlatText := TGLFlatText(Children[Count -1]);
      with CurrentFlatText do
      begin
        BitmapFont := GLWinBmpFont;
        Direction.AsVector := VectorMake(0, 1, 0);
        Up.AsVector := VectorMake(0, 0, 1);
        Layout := tlCenter;
        ModulateColor.AsWinColor := clBlue;
        Position.AsVector := VectorMake(CurrentXCoord, CurrentYCoord, CurrentZCoord);
        Scale.AsVector := VectorMake(ScaleFactor, ScaleFactor, 0);
        Text := FloatToStr(Round(CurrentZCoord*M_Output_Texture_3D.Z_Size));
      end;
    end;
    CurrentZCoord := CurrentZCoord + AxisStep;
  end;
end;

end.
