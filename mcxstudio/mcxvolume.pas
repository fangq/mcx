unit mcxvolume;


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
  StdCtrls,
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
  Texture_3D_Unit;

type

  { TfmVolume }

  TfmVolume = class(TForm)
    btRefresh : Tbutton;
    GLScene: TGLScene;
    GLCamera: TGLCamera;
    GLDirectOpenGL: TGLDirectOpenGL;
    GLCadencer: TGLCadencer;
    Timer: TTimer;
    Panel1: TPanel;
    Panel2: TPanel;
    glVolume: TGLSceneViewer;
    GLLightSource: TGLLightSource;
    Frame: TGLLines;
    GLCube1: TGLCube;
    Panel3: TPanel;
    Panel4: TPanel;
    Panel5: TPanel;
    Panel6: TPanel;
    tbSlice: TTrackBar;
    tbProjection: TTrackBar;
    Panel7: TPanel;
    Label1: TLabel;
    Label2: TLabel;
    Cutting_Plane_Pos_L: TLabel;
    Projection_N_L: TLabel;
    ckShowBox: TCheckBox;
    ckUseColor: TCheckBox;
    ckOpaque: TCheckBox;
    Panel8: TPanel;
    Panel9: TPanel;
    Label3: TLabel;
    Alpha_Threshold_L: TLabel;
    tbAlpha: TTrackBar;

    Procedure btRefreshClick(Sender : Tobject);
    Procedure Formshow(Sender : Tobject);
    procedure GLDirectOpenGLRender(Sender: TObject; var rci: TGLRenderContextInfo);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure glVolumeMouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: integer);
    procedure glVolumeMouseMove(Sender: TObject; Shift: TShiftState; X, Y: integer);
    procedure GLCadencerProgress(Sender: TObject; const deltaTime, newTime: double);
    procedure TimerTimer(Sender: TObject);
    procedure tbSliceChange(Sender: TObject);
    procedure tbProjectionChange(Sender: TObject);
    procedure ckShowBoxClick(Sender: TObject);
    procedure ckUseColorClick(Sender: TObject);
    procedure ckOpaqueClick(Sender: TObject);
    procedure tbAlphaChange(Sender: TObject);
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
    InputFile: string;
  end;


var
  fmVolume: TfmVolume;


implementation


{$R *.lfm}


const
  TEXTURE_INPUT_FILENAME = 'head.tx3';
  DIAGONAL_LENGTH = 1.732;


var
  clip0: array [0..3] of double = (-1.0, 0.0, 0.0, 1 / 2.0);
  clip1: array [0..3] of double = (1.0, 0.0, 0.0, 1 / 2.0);
  clip2: array [0..3] of double = (0.0, -1.0, 0.0, 1 / 2.0);
  clip3: array [0..3] of double = (0.0, 1.0, 0.0, 1 / 2.0);
  clip4: array [0..3] of double = (0.0, 0.0, -1.0, 1 / 2.0);
  clip5: array [0..3] of double = (0.0, 0.0, 1.0, 1 / 2.0);
  //Border_Colors: array [0..3] of integer = (0, 0, 0, 0);



procedure TfmVolume.FormCreate(Sender: TObject);
begin
end;

procedure TfmVolume.FormDestroy(Sender: TObject);
begin
  M_3D_Texture.Free;
  M_Input_Texture_3D.Free;
  M_Output_Texture_3D.Free;
end;

procedure TfmVolume.glVolumeMouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: integer);
begin
  M_mx := x;
  M_my := y;
end;

procedure TfmVolume.glVolumeMouseMove(Sender: TObject; Shift: TShiftState;
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

procedure TfmVolume.GLCadencerProgress(Sender: TObject;
  const deltaTime, newTime: double);
begin
  glVolume.Invalidate;
end;

procedure TfmVolume.TimerTimer(Sender: TObject);
begin
  Caption := 'MCX Volume Renderer ' + glVolume.FramesPerSecondText;
  glVolume.ResetPerformanceMonitor;
end;

procedure TfmVolume.Calculate_Transfer_function;
var
  X: integer;
  Y: integer;
  Z: integer;
  Index: integer;
  Value: integer;
  Alpha: integer;

begin
  { Set texture values }
  for Z := 0 to M_Output_Texture_3D.Z_Size - 1 do
  begin { For }
    for Y := 0 to M_Output_Texture_3D.Y_Size - 1 do
    begin { For }
      for X := 0 to M_Output_Texture_3D.X_Size - 1 do
      begin { For }
        Index := (Z * M_Output_Texture_3D.Y_Size * M_Output_Texture_3D.X_Size) + (Y * M_Output_Texture_3D.X_Size) + X;
        Value := PByte(PChar(M_Input_Texture_3D.Data) + Index)^ * 2;

        if (Value < tbAlpha.Position) or
          (Y > tbSlice.Position) or (X = 0) or
          (X = M_Output_Texture_3D.X_Size - 1) or (Y = 0) or
          (Y = M_Output_Texture_3D.Y_Size - 1) or (Z = 0) or
          (Z = M_Output_Texture_3D.Z_Size - 1) then
        begin { then }
          Alpha := 0;
        end { then }
        else
        begin { else }
          if ckOpaque.Checked = True then
          begin { then }
            Alpha := 255;
          end { then }
          else
          begin { else }
            Alpha := Value;
          end; { else }
        end; { else }

        if ckUseColor.Checked = True then
        begin { then }
          PLongWord((PChar(M_Output_Texture_3D.Data)) + (Index * 4))^ := M_Clut[Value];
        end { then }
        else
        begin { else }
          PLongWord((PChar(M_Output_Texture_3D.Data)) + (Index * 4))^ := Value + (Value shl 8) + (Value shl 16);
        end; { else }

        PByte((PChar(M_Output_Texture_3D.Data)) + (Index * 4) + 3)^ := Alpha;
      end; { For }
    end; { For }
  end; { For }
end;

procedure TfmVolume.GLDirectOpenGLRender(Sender: TObject; var rci: TGLRenderContextInfo);

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
  //  glDisable (GL_CULL_FACE);
  //  glDisable (GL_LIGHTING);

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

  step := DIAGONAL_LENGTH / tbProjection.Position;
  z := -DIAGONAL_LENGTH / 2;
  gl.begin_(GL_QUADS);
  for i := 0 to tbProjection.Position - 1 do
  begin
    gl.Color4f(1.0, 1.0, 1.0, 1.0);

    gl.Normal3f(-GLCamera.AbsoluteVectorToTarget.X,
      -GLCamera.AbsoluteVectorToTarget.Y, -GLCamera.AbsoluteVectorToTarget.Z);

    gl.Vertex3f(vx.X + vy.X + vz.X * z, vx.Y + vy.Y + vz.Y * z,
      vx.Z + vy.Z + vz.Z * z);
    gl.Vertex3f(-vx.X + vy.X + vz.X * z, -vx.Y + vy.Y + vz.Y * z,
      -vx.Z + vy.Z + vz.Z * z);
    gl.Vertex3f(-vx.X - vy.X + vz.X * z, -vx.Y - vy.Y + vz.Y * z,
      -vx.Z - vy.Z + vz.Z * z);
    gl.Vertex3f(vx.X - vy.X + vz.X * z, vx.Y - vy.Y + vz.Y * z,
      vx.Z - vy.Z + vz.Z * z);
    z := z + step;
  end;
  gl.End_;

  gl.PopMatrix;
  gl.PopAttrib;
end;

Procedure TfmVolume.Formshow(Sender : Tobject);

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
  COLOR_CONSTANT_LUMINANCE_HSL_H_OFFSET = 2 / 3;
  COLOR_CONSTANT_LUMINANCE_HSL_H_FACTOR = 0.85;
  COLOR_CONSTANT_LUMINANCE_HSL_S = 0.9;
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

  if(not FileExist(InputFile)) then return;

  Screen.Cursor := crHourGlass;
  M_3D_Texture := TGLTextureHandle.Create;

  M_Input_Texture_3D := TTexture_3D.Create;
  M_Input_Texture_3D.Load_From_File(InputFile);

  M_Output_Texture_3D := TTexture_3D.Create;
  M_Output_Texture_3D.Data_Type := GL_RGBA;
  M_Output_Texture_3D.X_Size := M_Input_Texture_3D.X_Size;
  M_Output_Texture_3D.Y_Size := M_Input_Texture_3D.Y_Size;
  M_Output_Texture_3D.Z_Size := M_Input_Texture_3D.Z_Size;

  M_Refresh := True;
  tbSlice.Position := M_Output_Texture_3D.Y_Size div 2;
  tbAlpha.Position := 60;
  tbProjection.Position := M_Output_Texture_3D.Y_Size;

  { Calculate Color Lookup Table }
  N_begin := 0;
  N_End := 255;
  Left_Value := 60;
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

  Screen.Cursor := crDefault;
End;

Procedure TfmVolume.btRefreshClick(Sender : Tobject);
Begin
 M_Refresh := True;
 btRefresh.Enabled := false;
End;



procedure TfmVolume.tbSliceChange(Sender: TObject);
begin
  Cutting_Plane_Pos_L.Caption := IntToStr(tbSlice.Position);
  btRefresh.Enabled := True;
  M_Refresh := True;
end;

procedure TfmVolume.tbAlphaChange(Sender: TObject);
begin
  Alpha_Threshold_L.Caption := IntToStr(tbAlpha.Position);
  btRefresh.Enabled := True;
  //M_Refresh := True;
end;

procedure TfmVolume.tbProjectionChange(Sender: TObject);
begin
  Projection_N_L.Caption := IntToStr(tbProjection.Position);
end;

procedure TfmVolume.ckShowBoxClick(Sender: TObject);
begin
  Frame.Visible := ckShowBox.Checked;
end;

procedure TfmVolume.ckUseColorClick(Sender: TObject);
begin
  M_Refresh := True;
end;

procedure TfmVolume.ckOpaqueClick(Sender: TObject);
begin
  M_Refresh := True;
end;

end.
