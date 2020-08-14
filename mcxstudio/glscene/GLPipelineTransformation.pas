//
// This unit is part of the GLScene Project, http://glscene.org
//
{
    History :  
     13/12/13 - PW - Added GLSLog unit
     11/05/11 - Yar - Ranamed TGLPipelineTransformation to TGLTransformation
     16/11/10 - Yar - Added NormalModelMatrix
     23/08/10 - Yar - Creation
  
}

unit GLPipelineTransformation;

interface

{$I GLScene.inc}

uses
  OpenGLTokens,
  OpenGLAdapter,
  GLVectorGeometry,
  GLVectorTypes,
  GLSLog;

const
  MAX_MATRIX_STACK_DEPTH = 128;

type

  TGLPipelineTransformationState =
  (
    trsModelViewChanged,
    trsInvModelViewChanged,
    trsInvModelChanged,
    trsNormalModelChanged,
    trsViewProjChanged,
    trsFrustum
  );

  TGLPipelineTransformationStates = set of TGLPipelineTransformationState;

const
  cAllStatesChanged = [trsModelViewChanged, trsInvModelViewChanged, trsInvModelChanged, trsViewProjChanged, trsNormalModelChanged, trsFrustum];

type

  PTransformationRec = ^TTransformationRec;
  TTransformationRec = record
    FStates: TGLPipelineTransformationStates;
    FModelMatrix: TMatrix;
    FViewMatrix: TMatrix;
    FProjectionMatrix: TMatrix;
    FInvModelMatrix: TMatrix;
    FNormalModelMatrix: TAffineMatrix;
    FModelViewMatrix: TMatrix;
    FInvModelViewMatrix: TMatrix;
    FViewProjectionMatrix: TMatrix;
    FFrustum: TFrustum;
  end;

type

  TOnMatricesPush = procedure() of object;

  // TGLTransformation
  //
  TGLTransformation = class(TObject)
  private
    FStackPos: Integer;
    FStack: array of TTransformationRec;
    FLoadMatricesEnabled: Boolean;
    FOnPush: TOnMatricesPush;
    function GetModelMatrix: TMatrix;
    function GetViewMatrix: TMatrix;
    function GetProjectionMatrix: TMatrix;
    function GetModelViewMatrix: TMatrix;
    function GetInvModelViewMatrix: TMatrix;
    function GetInvModelMatrix: TMatrix;
    function GetNormalModelMatrix: TAffineMatrix;
    function GetViewProjectionMatrix: TMatrix;
    function GetFrustum: TFrustum;

    procedure SetModelMatrix(const AMatrix: TMatrix);
    procedure SetViewMatrix(const AMatrix: TMatrix);
    procedure SetProjectionMatrix(const AMatrix: TMatrix);
  protected
    procedure LoadModelViewMatrix; {$IFDEF GLS_INLINE} inline; {$ENDIF}
    procedure LoadProjectionMatrix; {$IFDEF GLS_INLINE} inline; {$ENDIF}
    procedure DoMatrcesLoaded; {$IFDEF GLS_INLINE} inline; {$ENDIF}
    property OnPush: TOnMatricesPush read FOnPush write FOnPush;
  public
    constructor Create;

    procedure IdentityAll;
    procedure Push(AValue: PTransformationRec = nil);
    procedure Pop;
    procedure ReplaceFromStack;
    function StackTop: TTransformationRec;

    property ModelMatrix: TMatrix read GetModelMatrix write SetModelMatrix;
    property ViewMatrix: TMatrix read GetViewMatrix write SetViewMatrix;
    property ProjectionMatrix: TMatrix read GetProjectionMatrix write SetProjectionMatrix;

    property InvModelMatrix: TMatrix read GetInvModelMatrix;
    property ModelViewMatrix: TMatrix read GetModelViewMatrix;
    property NormalModelMatrix: TAffineMatrix read GetNormalModelMatrix;
    property InvModelViewMatrix: TMatrix read GetInvModelViewMatrix;
    property ViewProjectionMatrix: TMatrix read GetViewProjectionMatrix;
    property Frustum: TFrustum read GetFrustum;

    property LoadMatricesEnabled: Boolean read FLoadMatricesEnabled write FLoadMatricesEnabled;
  end;

// Prevent Lazaruses issue with checksumm chenging!
type
  TGLCall = function(): TGLExtensionsAndEntryPoints;
var
  vLocalGL: TGLCall;

implementation

constructor TGLTransformation.Create;
begin
  FStackPos := 0;
  SetLength(FStack, 1);
  IdentityAll;
end;

procedure TGLTransformation.IdentityAll;
begin
  with FStack[FStackPos] do
  begin
    FModelMatrix := IdentityHmgMatrix;
    FViewMatrix := IdentityHmgMatrix;
    FProjectionMatrix := IdentityHmgMatrix;
    FStates := cAllStatesChanged;
  end;
  if LoadMatricesEnabled then
  begin
    LoadModelViewMatrix;
    LoadProjectionMatrix;
  end;
end;

procedure TGLTransformation.Push(AValue: PTransformationRec);
var
  prevPos: Integer;
begin
  if FStackPos > MAX_MATRIX_STACK_DEPTH then
  begin
    GLSLogger.LogWarningFmt('Transformation stack overflow, more then %d values',
      [MAX_MATRIX_STACK_DEPTH]);
  end;
  prevPos := FStackPos;
  Inc(FStackPos);
  if High(FStack) < FStackPos then
    SetLength(FStack, FStackPos+1);

  if Assigned(AValue) then
  begin
    FStack[FStackPos] := AValue^;
    if LoadMatricesEnabled then
    begin
      LoadModelViewMatrix;
      LoadProjectionMatrix;
    end;
    DoMatrcesLoaded;
  end
  else
    FStack[FStackPos] := FStack[prevPos];
end;

procedure TGLTransformation.Pop;
begin
  if FStackPos = 0 then
  begin
    GLSLogger.LogError('Transformation stack underflow');
    exit;
  end;

  Dec(FStackPos);
  if LoadMatricesEnabled then
  begin
    LoadModelViewMatrix;
    LoadProjectionMatrix;
  end;
end;

procedure TGLTransformation.ReplaceFromStack;
var
  prevPos: Integer;
begin
  if FStackPos = 0 then
  begin
    GLSLogger.LogError('Transformation stack underflow');
    exit;
  end;
  prevPos := FStackPos - 1;
  FStack[FStackPos].FModelMatrix := FStack[prevPos].FModelMatrix;
  FStack[FStackPos].FViewMatrix:= FStack[prevPos].FViewMatrix;
  FStack[FStackPos].FProjectionMatrix:= FStack[prevPos].FProjectionMatrix;
  FStack[FStackPos].FStates := FStack[prevPos].FStates;
  if LoadMatricesEnabled then
  begin
    LoadModelViewMatrix;
    LoadProjectionMatrix;
  end;
end;

procedure TGLTransformation.LoadModelViewMatrix;
var
  M: TMatrix;
begin
  M := GetModelViewMatrix;
  vLocalGL.LoadMatrixf(PGLFloat(@M));
end;

procedure TGLTransformation.LoadProjectionMatrix;
begin
  with vLocalGL do
  begin
    MatrixMode(GL_PROJECTION);
    LoadMatrixf(PGLFloat(@FStack[FStackPos].FProjectionMatrix));
    MatrixMode(GL_MODELVIEW);
  end;
end;

function TGLTransformation.GetModelMatrix: TMatrix;
begin
  Result := FStack[FStackPos].FModelMatrix;
end;

function TGLTransformation.GetViewMatrix: TMatrix;
begin
  Result := FStack[FStackPos].FViewMatrix;
end;

function TGLTransformation.GetProjectionMatrix: TMatrix;
begin
  Result := FStack[FStackPos].FProjectionMatrix;
end;

procedure TGLTransformation.SetModelMatrix(const AMatrix: TMatrix);
begin
  FStack[FStackPos].FModelMatrix := AMatrix;
  FStack[FStackPos].FStates := FStack[FStackPos].FStates +
    [trsModelViewChanged, trsInvModelViewChanged, trsInvModelChanged, trsNormalModelChanged];
  if LoadMatricesEnabled then
    LoadModelViewMatrix;
end;

procedure TGLTransformation.SetViewMatrix(const AMatrix: TMatrix);
begin
  FStack[FStackPos].FViewMatrix:= AMatrix;
  FStack[FStackPos].FStates := FStack[FStackPos].FStates +
    [trsModelViewChanged, trsInvModelViewChanged, trsViewProjChanged, trsFrustum];
  if LoadMatricesEnabled then
    LoadModelViewMatrix;
end;

function TGLTransformation.StackTop: TTransformationRec;
begin
  Result := FStack[FStackPos];
end;

procedure TGLTransformation.SetProjectionMatrix(const AMatrix: TMatrix);
begin
  FStack[FStackPos].FProjectionMatrix := AMatrix;
  FStack[FStackPos].FStates := FStack[FStackPos].FStates +
    [trsViewProjChanged, trsFrustum];
  if LoadMatricesEnabled then
    LoadProjectionMatrix;
end;

function TGLTransformation.GetModelViewMatrix: TMatrix;
begin
  if trsModelViewChanged in FStack[FStackPos].FStates then
  begin
    FStack[FStackPos].FModelViewMatrix :=
      MatrixMultiply(FStack[FStackPos].FModelMatrix, FStack[FStackPos].FViewMatrix);
    Exclude(FStack[FStackPos].FStates, trsModelViewChanged);
  end;
  Result := FStack[FStackPos].FModelViewMatrix;
end;

function TGLTransformation.GetInvModelViewMatrix: TMatrix;
begin
  if trsInvModelViewChanged in FStack[FStackPos].FStates then
  begin
    FStack[FStackPos].FInvModelViewMatrix := GetModelViewMatrix;
    InvertMatrix(FStack[FStackPos].FInvModelViewMatrix);
    Exclude(FStack[FStackPos].FStates, trsInvModelViewChanged);
  end;
  Result := FStack[FStackPos].FInvModelViewMatrix;
end;

function TGLTransformation.GetInvModelMatrix: TMatrix;
begin
  if trsInvModelChanged in FStack[FStackPos].FStates then
  begin
    FStack[FStackPos].FInvModelMatrix := MatrixInvert(FStack[FStackPos].FModelMatrix);
    Exclude(FStack[FStackPos].FStates, trsInvModelChanged);
  end;
  Result := FStack[FStackPos].FInvModelMatrix;
end;

function TGLTransformation.GetNormalModelMatrix: TAffineMatrix;
var
  M: TMatrix;
begin
  if trsNormalModelChanged in FStack[FStackPos].FStates then
  begin
    M := FStack[FStackPos].FModelMatrix;
    NormalizeMatrix(M);
    SetMatrix(FStack[FStackPos].FNormalModelMatrix, M);
    Exclude(FStack[FStackPos].FStates, trsNormalModelChanged);
  end;
  Result := FStack[FStackPos].FNormalModelMatrix;
end;

function TGLTransformation.GetViewProjectionMatrix: TMatrix;
begin
  if trsViewProjChanged in FStack[FStackPos].FStates then
  begin
    FStack[FStackPos].FViewProjectionMatrix :=
      MatrixMultiply(FStack[FStackPos].FViewMatrix, FStack[FStackPos].FProjectionMatrix);
    Exclude(FStack[FStackPos].FStates, trsViewProjChanged);
  end;
  Result := FStack[FStackPos].FViewProjectionMatrix;
end;

procedure TGLTransformation.DoMatrcesLoaded;
begin
  if Assigned(FOnPush) then
    FOnPush();
end;

function TGLTransformation.GetFrustum: TFrustum;
begin
  if trsFrustum in FStack[FStackPos].FStates then
  begin
    FStack[FStackPos].FFrustum := ExtractFrustumFromModelViewProjection(GetViewProjectionMatrix);
    Exclude(FStack[FStackPos].FStates, trsFrustum);
  end;
  Result := FStack[FStackPos].FFrustum;
end;

end.
