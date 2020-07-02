//
// This unit is part of the GLScene Project, http://glscene.org
//
{
  Calculations and manipulations on Bounding Boxes.

   History :  
   10/12/14 - PW - Renamed GeometryBB unit to GLGeometryBB
   20/11/12 - PW - CPP compatibility: back changed type THmgBoundingBox to record
                 Changed THmgBoundingBox = array [0..7] of TVector
                 to THmgBoundingBox = record BBox : array [0..7] of TVector;
   05/06/12 - Maverick - Added PlaneAABBIntersection routine
   02/07/11 - DaStr - Removed TAABB.Revision
   20/04/08 - DaStr - Added a NullBoundingBox constant and
                 BoundingBoxesAreEqual() function (thanks Pascal)
   19/09/07 - DaStr - Added OffsetBB(Point) procedures
   31/08/07 - LC - Replaced TriangleIntersectAABB with a working (and faster) version
   23/08/07 - LC - Added RayCastAABBIntersect
   24/03/07 - DaStr - Added explicit pointer dereferencing
                 (thanks Burkhard Carstens) (Bugtracker ID = 1678644)
   22/06/03 - MF - Added TBSphere for bounding spheres and classes to
                 determine whether one aabb/bsphere contains another aabb/bsphere
   21/06/03 - MF - Added IntersectAABBsAbsolute
   08/05/03 - DanB - Added Plane/Triangle-AABB collisions (Matheus Degiovani)
   07/02/03 - EG - Added IntersectAABBsAbsoluteXY (Dan Bartlett)
   22/01/03 - EG - IntersectAABBs moved in (Bernd Klaiber)
   04/09/03 - EG - New AABB functions
   17/08/01 - EG - Removed "math" dependency
   09/07/01 - EG - Added AABB types and functions
   31/03/01 - EG - Original Unit by Jacques Tur
   

}
unit GLGeometryBB;

interface

{$I GLScene.inc}

uses
  GLVectorGeometry, GLVectorLists;

type
  { : Structure for storing Bounding Boxes }
  PHmgBoundingBox = ^THmgBoundingBox;

  THmgBoundingBox = record
    BBox: array [0 .. 7] of TVector;
  end;

  { : Structure for storing Axis Aligned Bounding Boxes }
  TAABB = record
    Min, Max: TAffineVector;
  end;

  PAABB = ^TAABB;

  // TBSphere
  //
  { : Structure for storing BoundingSpheres. Similar to TAABB }
  TBSphere = record
    { : Center of Bounding Sphere }
    Center: TAffineVector;
    { : Radius of Bounding Sphere }
    Radius: Single;
  end;

  // TClipRect
  //
  TClipRect = record
    Left, Top: Single;
    Right, Bottom: Single;
  end;

  { : Result type for space intersection tests, like AABBContainsAABB or
    BSphereContainsAABB }
  TSpaceContains = (ScNoOverlap, ScContainsFully, ScContainsPartially);
  { : Structure for storing the corners of an AABB, used with ExtractAABBCorners }
  TAABBCorners = array [0 .. 7] of TAffineVector;

  const
   NullBoundingBox: THmgBoundingBox =
   (BBox:((X: 0; Y: 0; Z: 0; W: 1),
          (X: 0; Y: 0; Z: 0; W: 1),
          (X: 0; Y: 0; Z: 0; W: 1),
          (X: 0; Y: 0; Z: 0; W: 1),
          (X: 0; Y: 0; Z: 0; W: 1),
          (X: 0; Y: 0; Z: 0; W: 1),
          (X: 0; Y: 0; Z: 0; W: 1),
          (X: 0; Y: 0; Z: 0; W: 1)));

  // ------------------------------------------------------------------------------
  // Bounding Box functions
  // ------------------------------------------------------------------------------

function BoundingBoxesAreEqual(const ABoundingBox1, ABoundingBox2
  : THmgBoundingBox): Boolean; overload;
function BoundingBoxesAreEqual(const ABoundingBox1, ABoundingBox2
  : PHmgBoundingBox): Boolean; overload;

{ : Adds a BB into another BB.
  The original BB (c1) is extended if necessary to contain c2. }
function AddBB(var C1: THmgBoundingBox; const C2: THmgBoundingBox)
  : THmgBoundingBox;
procedure AddAABB(var Aabb: TAABB; const Aabb1: TAABB);

procedure SetBB(var C: THmgBoundingBox; const V: TVector);
procedure SetAABB(var Bb: TAABB; const V: TVector);

procedure BBTransform(var C: THmgBoundingBox; const M: TMatrix);
procedure AABBTransform(var Bb: TAABB; const M: TMatrix);
procedure AABBScale(var Bb: TAABB; const V: TAffineVector);

function BBMinX(const C: THmgBoundingBox): Single;
function BBMaxX(const C: THmgBoundingBox): Single;
function BBMinY(const C: THmgBoundingBox): Single;
function BBMaxY(const C: THmgBoundingBox): Single;
function BBMinZ(const C: THmgBoundingBox): Single;
function BBMaxZ(const C: THmgBoundingBox): Single;

{ : Resize the AABB if necessary to include p. }
procedure AABBInclude(var Bb: TAABB; const P: TAffineVector);
{ : Make an AABB that is formed by sweeping a sphere (or AABB) from Start to Dest }
procedure AABBFromSweep(var SweepAABB: TAABB; const Start, Dest: TVector;
  const Radius: Single);
{ : Returns the intersection AABB of two AABBs.
  If the AABBs don't intersect, will return a degenerated AABB (plane, line or point). }
function AABBIntersection(const Aabb1, Aabb2: TAABB): TAABB;

{ : Extract AABB information from a BB. }
function BBToAABB(const ABB: THmgBoundingBox): TAABB;
{ : Converts an AABB to its canonical BB. }
function AABBToBB(const AnAABB: TAABB): THmgBoundingBox; overload;
{ : Transforms an AABB to a BB. }
function AABBToBB(const AnAABB: TAABB; const M: TMatrix)
  : THmgBoundingBox; overload;

{ : Adds delta to min and max of the AABB. }
procedure OffsetAABB(var Aabb: TAABB; const Delta: TAffineVector); overload;
procedure OffsetAABB(var Aabb: TAABB; const Delta: TVector); overload;

{ : Adds delta to min and max of the BB. }
procedure OffsetBB(var Bb: THmgBoundingBox;
  const Delta: TAffineVector); overload;
procedure OffsetBB(var Bb: THmgBoundingBox; const Delta: TVector); overload;
{ : The same as above but uses AddPoint() instead of AddVector(). }
procedure OffsetBBPoint(var Bb: THmgBoundingBox; const Delta: TVector);
  overload;

{ : Determines if two AxisAlignedBoundingBoxes intersect.
  The matrices are the ones that convert one point to the other's AABB system }
function IntersectAABBs(const Aabb1, Aabb2: TAABB; const M1To2, M2To1: TMatrix)
  : Boolean; overload;
{ : Checks whether two Bounding boxes aligned with the world axes collide in the XY plane. }
function IntersectAABBsAbsoluteXY(const Aabb1, Aabb2: TAABB): Boolean;
{ : Checks whether two Bounding boxes aligned with the world axes collide in the XZ plane. }
function IntersectAABBsAbsoluteXZ(const Aabb1, Aabb2: TAABB): Boolean;
{ : Checks whether two Bounding boxes aligned with the world axes collide. }
function IntersectAABBsAbsolute(const Aabb1, Aabb2: TAABB): Boolean;
{ : Checks whether one Bounding box aligned with the world axes fits within
  another Bounding box. }
function AABBFitsInAABBAbsolute(const Aabb1, Aabb2: TAABB): Boolean;

{ : Checks if a point "p" is inside an AABB }
function PointInAABB(const P: TAffineVector; const Aabb: TAABB)
  : Boolean; overload;
function PointInAABB(const P: TVector; const Aabb: TAABB): Boolean; overload;

{ : Checks if a plane (given by the normal+d) intersects the AABB }
function PlaneIntersectAABB(Normal: TAffineVector; D: Single;
  Aabb: TAABB): Boolean;
{ Compute the intersection between a plane and the AABB}
function PlaneAABBIntersection(const plane : THmgPlane; const AABB : TAABB) : TAffineVectorList;
{ : Checks if a triangle (given by vertices v1, v2 and v3) intersects an AABB }
function TriangleIntersectAABB(const Aabb: TAABB;
  const V1, V2, V3: TAffineVector): Boolean;

{ : Extract the corners from an AABB }
procedure ExtractAABBCorners(const AABB: TAABB; var AABBCorners: TAABBCorners);

{ : Convert an AABB to a BSphere }
procedure AABBToBSphere(const AABB: TAABB; var BSphere: TBSphere);
{ : Convert a BSphere to an AABB }
procedure BSphereToAABB(const BSphere: TBSphere; var AABB: TAABB); overload;
function BSphereToAABB(const Center: TAffineVector; Radius: Single)
  : TAABB; overload;
function BSphereToAABB(const Center: TVector; Radius: Single): TAABB; overload;

{ : Determines to which extent one AABB contains another AABB }
function AABBContainsAABB(const MainAABB, TestAABB: TAABB): TSpaceContains;
{ : Determines to which extent a BSphere contains an AABB }
function BSphereContainsAABB(const MainBSphere: TBSphere; const TestAABB: TAABB)
  : TSpaceContains;
{ : Determines to which extent one BSphere contains another BSphere }
function BSphereContainsBSphere(const MainBSphere, TestBSphere: TBSphere)
  : TSpaceContains;
{ : Determines to which extent an AABB contains a BSpher }
function AABBContainsBSphere(const MainAABB: TAABB; const TestBSphere: TBSphere)
  : TSpaceContains;
{ : Determines to which extent a plane contains a BSphere }
function PlaneContainsBSphere(const Location, Normal: TAffineVector;
  const TestBSphere: TBSphere): TSpaceContains;
{ : Determines to which extent a frustum contains a BSphere }
function FrustumContainsBSphere(const Frustum: TFrustum;
  const TestBSphere: TBSphere): TSpaceContains;
{ : Determines to which extent a frustum contains an AABB }
function FrustumContainsAABB(const Frustum: TFrustum; const TestAABB: TAABB)
  : TSpaceContains;
{ : Clips a position to an AABB }
function ClipToAABB(const V: TAffineVector; const AABB: TAABB): TAffineVector;
{ : Determines if one BSphere intersects another BSphere }
function BSphereIntersectsBSphere(const MainBSphere,
  TestBSphere: TBSphere): Boolean;

{ : Extend the clip rect to include given coordinate. }
procedure IncludeInClipRect(var ClipRect: TClipRect; X, Y: Single);
{ : Projects an AABB and determines the extent of its projection as a clip rect. }
function AABBToClipRect(const Aabb: TAABB; ModelViewProjection: TMatrix;
  ViewportSizeX, ViewportSizeY: Integer): TClipRect;

{ : Finds the intersection between a ray and an axis aligned bounding box. }
function RayCastAABBIntersect(const RayOrigin, RayDirection: TVector;
  const Aabb: TAABB; out TNear, TFar: Single): Boolean; overload;
function RayCastAABBIntersect(const RayOrigin, RayDirection: TVector;
  const Aabb: TAABB; IntersectPoint: PVector = nil): Boolean; overload;

type
  TPlanIndices = array [0 .. 3] of Integer;
  TPlanBB = array [0 .. 5] of TPlanIndices;
  TDirPlan = array [0 .. 5] of Integer;

const
  CBBFront: TPlanIndices = (0, 1, 2, 3);
  CBBBack: TPlanIndices = (4, 5, 6, 7);
  CBBLeft: TPlanIndices = (0, 4, 7, 3);
  CBBRight: TPlanIndices = (1, 5, 6, 2);
  CBBTop: TPlanIndices = (0, 1, 5, 4);
  CBBBottom: TPlanIndices = (2, 3, 7, 6);
  CBBPlans: TPlanBB = ((0, 1, 2, 3), (4, 5, 6, 7), (0, 4, 7, 3), (1, 5, 6, 2),
    (0, 1, 5, 4), (2, 3, 7, 6));
  CDirPlan: TDirPlan = (0, 0, 1, 1, 2, 2);

  // --------------------------------------------------------------
  // --------------------------------------------------------------
  // --------------------------------------------------------------
implementation

// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
uses SysUtils, GLVectorTypes;
// ------------------------------------------------------------------------------
// ----------------- BB functions -------------------------------------------
// ------------------------------------------------------------------------------

// SetPlanBB
//
procedure SetPlanBB(var BB: THmgBoundingBox; const NumPlan: Integer;
  const Valeur: Double);
var
  I: Integer;
begin
  for I := 0 to 3 do
  begin
    BB.BBox[CBBPlans[NumPlan][I]].V[CDirPlan[NumPlan]] := Valeur;
    BB.BBox[CBBPlans[NumPlan][I]].V[3] := 1;
  end;
end;

// BoundingBoxesAreEqual (copy)
//
function BoundingBoxesAreEqual(const ABoundingBox1, ABoundingBox2
  : THmgBoundingBox): Boolean;
begin
  Result := CompareMem(@ABoundingBox1, @ABoundingBox2, SizeOf(THmgBoundingBox));
end;

// BoundingBoxesAreEqual (direct)
//
function BoundingBoxesAreEqual(const ABoundingBox1, ABoundingBox2
  : PHmgBoundingBox): Boolean;
begin
  Result := CompareMem(ABoundingBox1, ABoundingBox2, SizeOf(THmgBoundingBox));
end;

// AddBB
//
function AddBB(var C1: THmgBoundingBox; const C2: THmgBoundingBox)
  : THmgBoundingBox;

var
  I, J: Integer;
begin
  for I := 0 to 7 do
  begin
    for J := 0 to 3 do
      if C1.BBox[CBBFront[J]].V[0] < C2.BBox[I].V[0] then
        SetPlanBB(C1, 0, C2.BBox[I].V[0]);
    for J := 0 to 3 do
      if C1.BBox[CBBBack[J]].V[0] > C2.BBox[I].V[0] then
        SetPlanBB(C1, 1, C2.BBox[I].V[0]);
    for J := 0 to 3 do
      if C1.BBox[CBBLeft[J]].V[1] < C2.BBox[I].V[1] then
        SetPlanBB(C1, 2, C2.BBox[I].V[1]);
    for J := 0 to 3 do
      if C1.BBox[CBBRight[J]].V[1] > C2.BBox[I].V[1] then
        SetPlanBB(C1, 3, C2.BBox[I].V[1]);
    for J := 0 to 3 do
      if C1.BBox[CBBTop[J]].V[2] < C2.BBox[I].V[2] then
        SetPlanBB(C1, 4, C2.BBox[I].V[2]);
    for J := 0 to 3 do
      if C1.BBox[CBBBottom[J]].V[2] > C2.BBox[I].V[2] then
        SetPlanBB(C1, 5, C2.BBox[I].V[2]);
  end;
  Result := C1;
end;

// AddAABB
//
procedure AddAABB(var Aabb: TAABB; const Aabb1: TAABB);
begin
  if Aabb1.Min.V[0] < Aabb.Min.V[0] then
    Aabb.Min.V[0] := Aabb1.Min.V[0];
  if Aabb1.Min.V[1] < Aabb.Min.V[1] then
    Aabb.Min.V[1] := Aabb1.Min.V[1];
  if Aabb1.Min.V[2] < Aabb.Min.V[2] then
    Aabb.Min.V[2] := Aabb1.Min.V[2];
  if Aabb1.Max.V[0] > Aabb.Max.V[0] then
    Aabb.Max.V[0] := Aabb1.Max.V[0];
  if Aabb1.Max.V[1] > Aabb.Max.V[1] then
    Aabb.Max.V[1] := Aabb1.Max.V[1];
  if Aabb1.Max.V[2] > Aabb.Max.V[2] then
    Aabb.Max.V[2] := Aabb1.Max.V[2];
end;

// SetBB
//
procedure SetBB(var C: THmgBoundingBox; const V: TVector);
begin
  SetPlanBB(C, 0, V.V[0]);
  SetPlanBB(C, 1, -V.V[0]);
  SetPlanBB(C, 2, V.V[1]);
  SetPlanBB(C, 3, -V.V[1]);
  SetPlanBB(C, 4, V.V[2]);
  SetPlanBB(C, 5, -V.V[2]);
end;

// SetAABB
//
procedure SetAABB(var Bb: TAABB; const V: TVector);
begin
  Bb.Max.V[0] := Abs(V.V[0]);
  Bb.Max.V[1] := Abs(V.V[1]);
  Bb.Max.V[2] := Abs(V.V[2]);
  Bb.Min.V[0] := -Bb.Max.V[0];
  Bb.Min.V[1] := -Bb.Max.V[1];
  Bb.Min.V[2] := -Bb.Max.V[2];
end;

// BBTransform
//
procedure BBTransform(var C: THmgBoundingBox; const M: TMatrix);
var
  I: Integer;
begin
  for I := 0 to 7 do
    C.BBox[I] := VectorTransform(C.BBox[I], M);
end;

// AABBTransform
//
procedure AABBTransform(var Bb: TAABB; const M: TMatrix);
var
  OldMin, OldMax: TAffineVector;
begin
  OldMin := Bb.Min;
  OldMax := Bb.Max;
  Bb.Min := VectorTransform(OldMin, M);
  Bb.Max := Bb.Min;
  AABBInclude(Bb, VectorTransform(AffineVectorMake(OldMin.V[0],
    OldMin.V[1], OldMax.V[2]), M));
  AABBInclude(Bb, VectorTransform(AffineVectorMake(OldMin.V[0],
    OldMax.V[1], OldMin.V[2]), M));
  AABBInclude(Bb, VectorTransform(AffineVectorMake(OldMin.V[0],
    OldMax.V[1], OldMax.V[2]), M));
  AABBInclude(Bb, VectorTransform(AffineVectorMake(OldMax.V[0],
    OldMin.V[1], OldMin.V[2]), M));
  AABBInclude(Bb, VectorTransform(AffineVectorMake(OldMax.V[0],
    OldMin.V[1], OldMax.V[2]), M));
  AABBInclude(Bb, VectorTransform(AffineVectorMake(OldMax.V[0],
    OldMax.V[1], OldMin.V[2]), M));
  AABBInclude(Bb, VectorTransform(OldMax, M));
end;

// AABBScale
//
procedure AABBScale(var Bb: TAABB; const V: TAffineVector);
begin
  ScaleVector(Bb.Min, V);
  ScaleVector(Bb.Max, V);
end;

// BBMinX
//
function BBMinX(const C: THmgBoundingBox): Single;
var
  I: Integer;
begin
  Result := C.BBox[0].V[0];
  for I := 1 to 7 do
    Result := MinFloat(Result, C.BBox[I].V[0]);
end;

// BBMaxX
//
function BBMaxX(const C: THmgBoundingBox): Single;
var
  I: Integer;
begin
  Result := C.BBox[0].V[0];
  for I := 1 to 7 do
    Result := MaxFloat(Result, C.BBox[I].V[0]);
end;

// BBMinY
//
function BBMinY(const C: THmgBoundingBox): Single;
var
  I: Integer;
begin
  Result := C.BBox[0].V[1];
  for I := 1 to 7 do
    Result := MinFloat(Result, C.BBox[I].V[1]);
end;

// BBMaxY
//
function BBMaxY(const C: THmgBoundingBox): Single;
var
  I: Integer;
begin
  Result := C.BBox[0].V[1];
  for I := 1 to 7 do
    Result := MaxFloat(Result, C.BBox[I].V[1]);
end;

// BBMinZ
//
function BBMinZ(const C: THmgBoundingBox): Single;
var
  I: Integer;
begin
  Result := C.BBox[0].V[2];
  for I := 1 to 7 do
    Result := MinFloat(Result, C.BBox[I].V[2]);
end;

// BBMaxZ
//
function BBMaxZ(const C: THmgBoundingBox): Single;
var
  I: Integer;
begin
  Result := C.BBox[0].V[2];
  for I := 1 to 7 do
    Result := MaxFloat(Result, C.BBox[I].V[2]);
end;

// AABBInclude
//
procedure AABBInclude(var Bb: TAABB; const P: TAffineVector);
begin
  if P.V[0] < Bb.Min.V[0] then
    Bb.Min.V[0] := P.V[0];
  if P.V[0] > Bb.Max.V[0] then
    Bb.Max.V[0] := P.V[0];
  if P.V[1] < Bb.Min.V[1] then
    Bb.Min.V[1] := P.V[1];
  if P.V[1] > Bb.Max.V[1] then
    Bb.Max.V[1] := P.V[1];
  if P.V[2] < Bb.Min.V[2] then
    Bb.Min.V[2] := P.V[2];
  if P.V[2] > Bb.Max.V[2] then
    Bb.Max.V[2] := P.V[2];
end;

// AABBFromSweep
//
procedure AABBFromSweep(var SweepAABB: TAABB; const Start, Dest: TVector;
  const Radius: Single);
begin
  if Start.V[0] < Dest.V[0] then
  begin
    SweepAABB.Min.V[0] := Start.V[0] - Radius;
    SweepAABB.Max.V[0] := Dest.V[0] + Radius;
  end
  else
  begin
    SweepAABB.Min.V[0] := Dest.V[0] - Radius;
    SweepAABB.Max.V[0] := Start.V[0] + Radius;
  end;

  if Start.V[1] < Dest.V[1] then
  begin
    SweepAABB.Min.V[1] := Start.V[1] - Radius;
    SweepAABB.Max.V[1] := Dest.V[1] + Radius;
  end
  else
  begin
    SweepAABB.Min.V[1] := Dest.V[1] - Radius;
    SweepAABB.Max.V[1] := Start.V[1] + Radius;
  end;

  if Start.V[2] < Dest.V[2] then
  begin
    SweepAABB.Min.V[2] := Start.V[2] - Radius;
    SweepAABB.Max.V[2] := Dest.V[2] + Radius;
  end
  else
  begin
    SweepAABB.Min.V[2] := Dest.V[2] - Radius;
    SweepAABB.Max.V[2] := Start.V[2] + Radius;
  end;
end;

// AABBIntersection
//
function AABBIntersection(const Aabb1, Aabb2: TAABB): TAABB;
var
  I: Integer;
begin
  for I := 0 to 2 do
  begin
    Result.Min.V[I] := MaxFloat(Aabb1.Min.V[I], Aabb2.Min.V[I]);
    Result.Max.V[I] := MinFloat(Aabb1.Max.V[I], Aabb2.Max.V[I]);
  end;
end;

// BBToAABB
//
function BBToAABB(const ABB: THmgBoundingBox): TAABB;
var
  I: Integer;
begin
  SetVector(Result.Min, ABB.BBox[0]);
  SetVector(Result.Max, ABB.BBox[0]);
  for I := 1 to 7 do
  begin
    if ABB.BBox[I].V[0] < Result.Min.V[0] then
      Result.Min.V[0] := ABB.BBox[I].V[0];
    if ABB.BBox[I].V[0] > Result.Max.V[0] then
      Result.Max.V[0] := ABB.BBox[I].V[0];
    if ABB.BBox[I].V[1] < Result.Min.V[1] then
      Result.Min.V[1] := ABB.BBox[I].V[1];
    if ABB.BBox[I].V[1] > Result.Max.V[1] then
      Result.Max.V[1] := ABB.BBox[I].V[1];
    if ABB.BBox[I].V[2] < Result.Min.V[2] then
      Result.Min.V[2] := ABB.BBox[I].V[2];
    if ABB.BBox[I].V[2] > Result.Max.V[2] then
      Result.Max.V[2] := ABB.BBox[I].V[2];
  end;
end;

// AABBToBB
//
function AABBToBB(const AnAABB: TAABB): THmgBoundingBox;
begin
  with AnAABB do
  begin
    SetPlanBB(Result, 0, Max.V[0]);
    SetPlanBB(Result, 1, Min.V[0]);
    SetPlanBB(Result, 2, Max.V[1]);
    SetPlanBB(Result, 3, Min.V[1]);
    SetPlanBB(Result, 4, Max.V[2]);
    SetPlanBB(Result, 5, Min.V[2]);
  end;
end;

// AABBToBB
//
function AABBToBB(const AnAABB: TAABB; const M: TMatrix): THmgBoundingBox;
begin
  Result := AABBToBB(AnAABB);
  BBTransform(Result, M);
end;

// OffsetAABB
//
procedure OffsetAABB(var Aabb: TAABB; const Delta: TAffineVector);
begin
  AddVector(Aabb.Min, Delta);
  AddVector(Aabb.Max, Delta);
end;

// OffsetAABB
//
procedure OffsetAABB(var Aabb: TAABB; const Delta: TVector);
begin
  AddVector(Aabb.Min, Delta);
  AddVector(Aabb.Max, Delta);
end;

// OffsetBB
//
procedure OffsetBB(var Bb: THmgBoundingBox; const Delta: TAffineVector);
var
  I: Integer;
  TempVector: TVector;
begin
  TempVector := VectorMake(Delta, 0);
  for I := 0 to 7 do
    AddVector(Bb.BBox[I], TempVector);
end;

// OffsetAABB
//
procedure OffsetBB(var Bb: THmgBoundingBox; const Delta: TVector);
var
  I: Integer;
begin
  for I := 0 to 7 do
    AddVector(Bb.BBox[I], Delta);
end;

// OffsetBBPoint
//
procedure OffsetBBPoint(var Bb: THmgBoundingBox; const Delta: TVector);
var
  I: Integer;
begin
  for I := 0 to 7 do
    AddPoint(Bb.BBox[I], Delta);
end;

// IntersectCubes (AABBs)
//
function IntersectAABBs(const Aabb1, Aabb2: TAABB;
  const M1To2, M2To1: TMatrix): Boolean;
const
  CWires: array [0 .. 11, 0 .. 1] of Integer // Points of the wire
    = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4),
    (1, 5), (2, 6), (3, 7));
  CPlanes: array [0 .. 5, 0 .. 3] of Integer // points of the planes
    = ((1, 2, 6, 5), (2, 3, 7, 6), (0, 1, 2, 3), (0, 3, 7, 4), (0, 1, 5, 4),
    (5, 6, 7, 4));

  procedure MakeAABBPoints(const AABB: TAABB; var Pt: array of TVertex);
  begin
    with AABB do
    begin
      SetVector(Pt[0], Min.V[0], Min.V[1], Min.V[2]);
      SetVector(Pt[1], Max.V[0], Min.V[1], Min.V[2]);
      SetVector(Pt[2], Max.V[0], Max.V[1], Min.V[2]);
      SetVector(Pt[3], Min.V[0], Max.V[1], Min.V[2]);
      SetVector(Pt[4], Min.V[0], Min.V[1], Max.V[2]);
      SetVector(Pt[5], Max.V[0], Min.V[1], Max.V[2]);
      SetVector(Pt[6], Max.V[0], Max.V[1], Max.V[2]);
      SetVector(Pt[7], Min.V[0], Max.V[1], Max.V[2]);
    end;
  end;

  procedure MakePlanes(const Pt: array of TVertex;
    var Planes: array of THmgPlane);
  var
    I: Integer;
  begin
    for I := 0 to 5 do
      Planes[I] := PlaneMake(Pt[CPlanes[I, 0]], Pt[CPlanes[I, 1]],
        Pt[CPlanes[I, 2]]);
  end;

var
  Pt1, Pt2: array [0 .. 7] of TVertex;
  Pt: TVertex;
  Planes2: array [0 .. 5] of THmgPlane;
  I, T: Integer;
  V: TVertex;
  P: TVector;
begin
  Result := False;

  // Build Points
  MakeAABBPoints(AABB1, Pt1);
  MakeAABBPoints(AABB2, Pt2);
  for I := 0 to 7 do
  begin
    Pt := VectorTransform(Pt2[I], M2To1);
    // check for inclusion (points of Obj2 in Obj1)
    if IsInRange(Pt.V[0], AABB1.Min.V[0], AABB1.Max.V[0]) and
      IsInRange(Pt.V[1], AABB1.Min.V[1], AABB1.Max.V[1]) and
      IsInRange(Pt.V[2], AABB1.Min.V[2], AABB1.Max.V[2]) then
    begin
      Result := True;
      Exit;
    end;
  end;

  for I := 0 to 7 do
  begin
    Pt1[I] := VectorTransform(Pt1[I], M1To2);
    // check for inclusion (points of Obj1 in Obj2)
    if IsInRange(Pt1[I].V[0], AABB2.Min.V[0], AABB2.Max.V[0]) and
      IsInRange(Pt1[I].V[1], AABB2.Min.V[1], AABB2.Max.V[1]) and
      IsInRange(Pt1[I].V[2], AABB2.Min.V[2], AABB2.Max.V[2]) then
    begin
      Result := True;
      Exit;
    end;
  end;

  // Build Planes
  MakePlanes(Pt2, Planes2);

  // Wire test
  for I := 0 to 11 do
  begin
    for T := 0 to 5 do
    begin
      // Build Vector of Ray
      V := VectorSubtract(Pt1[CWires[I, 0]], Pt1[CWires[I, 1]]);
      if IntersectLinePlane(VectorMake(Pt1[CWires[I, 0]]), VectorMake(V),
        Planes2[T], @P) = 1 then
      begin
        // check point in Wire
        if IsInRange(P.V[0], Pt1[CWires[I, 0]].V[0],
          Pt1[CWires[I, 1]].V[0]) and
          IsInRange(P.V[1], Pt1[CWires[I, 0]].V[1],
          Pt1[CWires[I, 1]].V[1]) and
          IsInRange(P.V[2], Pt1[CWires[I, 0]].V[2],
          Pt1[CWires[I, 1]].V[2]) then
        begin
          // check point in Plane
          if IsInRange(P.V[0], Pt2[CPlanes[T, 0]].V[0],
            Pt2[CPlanes[T, 2]].V[0]) and
            IsInRange(P.V[1], Pt2[CPlanes[T, 0]].V[1],
            Pt2[CPlanes[T, 2]].V[1]) and
            IsInRange(P.V[2], Pt2[CPlanes[T, 0]].V[2],
            Pt2[CPlanes[T, 2]].V[2]) then
          begin
            Result := True;
            Exit;
          end;
        end;
      end;
    end;
  end;
end;

// IntersectAABBsAbsoluteXY (AABBs)
//
function IntersectAABBsAbsoluteXY(const Aabb1, Aabb2: TAABB): Boolean;
begin
  Result := False;

  if (AABB2.Min.V[0] > AABB1.Max.V[0]) or
    (AABB2.Min.V[1] > AABB1.Max.V[1]) then
    Exit
  else if (AABB2.Max.V[0] < AABB1.Min.V[0]) or
    (AABB2.Max.V[1] < AABB1.Min.V[1]) then
    Exit
  else
    Result := True;

end;

function IntersectAABBsAbsoluteXZ(const Aabb1, Aabb2: TAABB): Boolean;
begin
  Result := ((AABB1.Min.V[0] < AABB2.Max.V[0]) and
    (AABB1.Min.V[2] < AABB2.Max.V[2]) and

    (AABB2.Min.V[0] < AABB1.Max.V[0]) and
    (AABB2.Min.V[2] < AABB1.Max.V[2]));
end;

// IntersectAABBsAbsolute
//
function IntersectAABBsAbsolute(const Aabb1, Aabb2: TAABB): Boolean;
begin
  Result := not((AABB1.Min.V[0] > AABB2.Max.V[0]) or
    (AABB1.Min.V[1] > AABB2.Max.V[1]) or
    (AABB1.Min.V[2] > AABB2.Max.V[2]) or

    (AABB2.Min.V[0] > AABB1.Max.V[0]) or
    (AABB2.Min.V[1] > AABB1.Max.V[1]) or
    (AABB2.Min.V[2] > AABB1.Max.V[2]));
end;

// IntersectAABBsAbsolute
//
function AABBFitsInAABBAbsolute(const Aabb1, Aabb2: TAABB): Boolean;
begin
  // AABB1 fits completely inside AABB2?
  // AABB1 min must be >= to AABB2 min
  // AABB1 max must be <= to AABB2 max

  Result := (AABB1.Min.V[0] >= AABB2.Min.V[0]) and
    (AABB1.Min.V[1] >= AABB2.Min.V[1]) and
    (AABB1.Min.V[2] >= AABB2.Min.V[2]) and

    (AABB1.Max.V[0] <= AABB2.Max.V[0]) and
    (AABB1.Max.V[1] <= AABB2.Max.V[1]) and
    (AABB1.Max.V[2] <= AABB2.Max.V[2]);
end;

// PointInAABB (affine)
//
function PointInAABB(const P: TAffineVector; const Aabb: TAABB): Boolean;
begin
  Result := (P.V[0] <= Aabb.Max.V[0]) and
    (P.V[0] >= Aabb.Min.V[0]) and (P.V[1] <= Aabb.Max.V[1]) and
    (P.V[1] >= Aabb.Min.V[1]) and (P.V[2] <= Aabb.Max.V[2]) and
    (P.V[2] >= Aabb.Min.V[2]);
end;

// PointInAABB (hmg)
//
function PointInAABB(const P: TVector; const Aabb: TAABB): Boolean;
begin
  Result := (P.V[0] <= Aabb.Max.V[0]) and
    (P.V[0] >= Aabb.Min.V[0]) and (P.V[1] <= Aabb.Max.V[1]) and
    (P.V[1] >= Aabb.Min.V[1]) and (P.V[2] <= Aabb.Max.V[2]) and
    (P.V[2] >= Aabb.Min.V[2]);
end;

// PlaneIntersectAABB
//
function PlaneIntersectAABB(Normal: TAffineVector; D: Single;
  Aabb: TAABB): Boolean;
var
  Vmax, Vmin: TAffineVector;
  I: Integer;
begin
  Result := False;
  for I := 0 to 2 do
    if Normal.V[I] > 0.0 then
    begin
      VMin.V[I] := Aabb.Min.V[I];
      VMax.V[I] := Aabb.Max.V[I];
    end
    else
    begin
      VMin.V[I] := Aabb.Max.V[I];
      VMax.V[I] := Aabb.Min.V[I];
    end;

  if VectorDotProduct(Normal, Vmin) + D > 0 then
    Exit;
  if VectorDotProduct(Normal, Vmax) + D >= 0 then
    Result := True;
end;

procedure FindMinMax(X0, X1, X2: Single; out Min, Max: Single);
begin
  Min := X0;
  Max := X0;
  if (X1 < Min) then
    Min := X1;
  if (X1 > Max) then
    Max := X1;
  if (X2 < Min) then
    Min := X2;
  if (X2 > Max) then
    Max := X2;
end;

// PlaneAABBIntersection
//
function PlaneAABBIntersection(const plane : THmgPlane;const AABB : TAABB) : TAffineVectorList;
var
  i, j, annexe : Integer;
  index : array[0..2] of Integer;
  vec, temp : TVector3f;
  box : array [0..1] of TVector3f;
  V: array [0..7] of TVector3f;
  function EdgesStripPlaneIntersection(const pt0, pt1, pt4, pt7: TVector3f;
    const plane : THmgPlane; var inter : TVector3f): Boolean;
  begin
    Result := True;
    if not SegmentPlaneIntersection(pt0, pt1, plane, inter) then
      if not SegmentPlaneIntersection(pt1, pt4, plane, inter) then
        if not SegmentPlaneIntersection(pt4, pt7, plane, inter) then
          Result := False;
  end;
begin
  box[0] := AABB.min;
  box[1] := AABB.max;

  Result := TAffineVectorList.Create;

  // loop on vertices
  for i := 0 to 7 do
  begin
    for j := 0 to 2 do
    begin
      index[j] := (i div (1 shl j)) mod 2;
      vec.V[j] := box[index[j]].V[j];
    end;

    // try to find the right orientation to proceed intersection
    if (i = 0) then
    begin
      // prepare V 0 -> 7 array
      V[0] := vec;
      for j := 0 to 5 do
      begin
        temp := vec;
        temp.V[j mod 3] := box[(index[j mod 3] + 1) mod 2].V[j mod 3];
        if (j div 3) > 0 then
        begin
          temp.V[(j+1) mod 3] := box[(index[(j+1) mod 3] + 1) mod 2].V[(j+1) mod 3];
          if (j div 3) > 1 then
          begin
            temp.V[(j+2) mod 3] := box[(index[(j+2) mod 3] + 1) mod 2].V[(j+2) mod 3];
          end;
        end;
        V[j+1] := temp;
      end;
      for j := 0 to 2 do
        vec.V[j] := box[(index[j]+1) mod 2].V[j];
      V[7] := vec;
    end;
  end;

  //compute edge plane intersections
  for j := 0 to 2 do
  begin
    if j = 0 then annexe := 6 else annexe := j+3;

    // computes intersection with annexe edge
    if SegmentPlaneIntersection(V[j+1], V[annexe], plane, temp) then
      Result.Add(temp);
    // computes intersection with edge strip from main vertex V0 to opposite vertex V7
    if EdgesStripPlaneIntersection(V[0], V[j+1], V[j+4], V[7], plane, temp) then
      Result.Add(temp);
  end;
end;



function PlaneBoxOverlap(const Normal: TAffineVector; D: Single;
  const Maxbox: TAffineVector): Boolean;
var
  Q: Integer;
  Vmin, Vmax: TAffineVector;
begin
  Result := False;

  for Q := 0 to 2 do
  begin
    if (Normal.V[Q] > 0.0) then
    begin
      Vmin.V[Q] := -Maxbox.V[Q];
      Vmax.V[Q] := Maxbox.V[Q];
    end
    else
    begin
      Vmin.V[Q] := Maxbox.V[Q];
      Vmax.V[Q] := -Maxbox.V[Q];
    end;
  end;

  if (VectorDotProduct(Normal, Vmin) + D) > 0 then
    Exit;

  if (VectorDotProduct(Normal, Vmax) + D) >= 0 then
    Result := True;
end;

// TriangleIntersectAABB
//
function TriangleIntersectAABB(const Aabb: TAABB;
  const V1, V2, V3: TAffineVector): Boolean;
// Original source code by Tomas Akenine-Möller
// Based on the paper "Fast 3D Triangle-Box Overlap Testing"
// http://www.cs.lth.se/home/Tomas_Akenine_Moller/pubs/tribox.pdf
// http://jgt.akpeters.com/papers/AkenineMoller01/ (code)

// use separating axis theorem to test overlap between triangle and box
// need to test for overlap in these directions:
// 1) the (x,y,z)-directions (actually, since we use the AABB of the triangle
// we do not even need to test these)
// 2) normal of the triangle
// 3) crossproduct(edge from tri, {x,y,z}-directin)
// this gives 3x3=9 more tests
var
  Boxcenter, Boxhalfsize: TAffineVector;
  Tv0, Tv1, Tv2: TAffineVector;
  Min, Max, D, P0, P1, P2, Rad, Fex, Fey, Fez: Single;
  Normal, E0, E1, E2: TAffineVector;
begin
  Result := False;

  Boxhalfsize := VectorSubtract(VectorScale(Aabb.Max, 0.5),
    VectorScale(Aabb.Min, 0.5));
  Boxcenter := VectorAdd(VectorScale(Aabb.Max, 0.5),
    VectorScale(Aabb.Min, 0.5));
  // move everything so that the boxcenter is in (0,0,0)
  VectorSubtract(V1, Boxcenter, Tv0);
  VectorSubtract(V2, Boxcenter, Tv1);
  VectorSubtract(V3, Boxcenter, Tv2);

  // compute triangle edges
  VectorSubtract(Tv1, Tv0, E0);
  VectorSubtract(Tv2, Tv1, E1);
  VectorSubtract(Tv0, Tv2, E2);

  // Bullet 3:
  // test the 9 tests first (this was faster)
  Fex := Abs(E0.V[0]);
  Fey := Abs(E0.V[1]);
  Fez := Abs(E0.V[2]);

  // AXISTEST_X01(e0[Z], e0[Y], fez, fey);
  P0 := E0.V[2] * Tv0.V[1] - E0.V[1] * Tv0.V[2];
  P2 := E0.V[2] * Tv2.V[1] - E0.V[1] * Tv2.V[2];
  Min := MinFloat(P0, P2);
  Max := MaxFloat(P0, P2);
  Rad := Fez * Boxhalfsize.V[1] + Fey * Boxhalfsize.V[2];
  if (Min > Rad) or (Max < -Rad) then
    Exit;

  // AXISTEST_Y02(e0[Z], e0[X], fez, fex);
  P0 := -E0.V[2] * Tv0.V[0] + E0.V[0] * Tv0.V[2];
  P2 := -E0.V[2] * Tv2.V[0] + E0.V[0] * Tv2.V[2];
  Min := MinFloat(P0, P2);
  Max := MaxFloat(P0, P2);
  Rad := Fez * Boxhalfsize.V[0] + Fex * Boxhalfsize.V[2];
  if (Min > Rad) or (Max < -Rad) then
    Exit;

  // AXISTEST_Z12(e0[Y], e0[X], fey, fex);
  P1 := E0.V[1] * Tv1.V[0] - E0.V[0] * Tv1.V[1];
  P2 := E0.V[1] * Tv2.V[0] - E0.V[0] * Tv2.V[1];
  Min := MinFloat(P1, P2);
  Max := MaxFloat(P1, P2);
  Rad := Fey * Boxhalfsize.V[0] + Fex * Boxhalfsize.V[1];
  if (Min > Rad) or (Max < -Rad) then
    Exit;

  Fex := Abs(E1.V[0]);
  Fey := Abs(E1.V[1]);
  Fez := Abs(E1.V[2]);
  // AXISTEST_X01(e1[Z], e1[Y], fez, fey);
  P0 := E1.V[2] * Tv0.V[1] - E1.V[1] * Tv0.V[2];
  P2 := E1.V[2] * Tv2.V[1] - E1.V[1] * Tv2.V[2];
  Min := MinFloat(P0, P2);
  Max := MaxFloat(P0, P2);
  Rad := Fez * Boxhalfsize.V[1] + Fey * Boxhalfsize.V[2];
  if (Min > Rad) or (Max < -Rad) then
    Exit;

  // AXISTEST_Y02(e1[Z], e1[X], fez, fex);
  P0 := -E1.V[2] * Tv0.V[0] + E1.V[0] * Tv0.V[2];
  P2 := -E1.V[2] * Tv2.V[0] + E1.V[0] * Tv2.V[2];
  Min := MinFloat(P0, P2);
  Max := MaxFloat(P0, P2);
  Rad := Fez * Boxhalfsize.V[0] + Fex * Boxhalfsize.V[2];
  if (Min > Rad) or (Max < -Rad) then
    Exit;

  // AXISTEST_Z0(e1[Y], e1[X], fey, fex);
  P0 := E1.V[1] * Tv0.V[0] - E1.V[0] * Tv0.V[1];
  P1 := E1.V[1] * Tv1.V[0] - E1.V[0] * Tv1.V[1];
  Min := MinFloat(P0, P1);
  Max := MaxFloat(P0, P1);
  Rad := Fey * Boxhalfsize.V[0] + Fex * Boxhalfsize.V[1];
  if (Min > Rad) or (Max < -Rad) then
    Exit;

  Fex := Abs(E2.V[0]);
  Fey := Abs(E2.V[1]);
  Fez := Abs(E2.V[2]);
  // AXISTEST_X2(e2[Z], e2[Y], fez, fey);
  P0 := E2.V[2] * Tv0.V[1] - E2.V[1] * Tv0.V[2];
  P1 := E2.V[2] * Tv1.V[1] - E2.V[1] * Tv1.V[2];
  Min := MinFloat(P0, P1);
  Max := MaxFloat(P0, P1);
  Rad := Fez * Boxhalfsize.V[1] + Fey * Boxhalfsize.V[2];
  if (Min > Rad) or (Max < -Rad) then
    Exit;

  // AXISTEST_Y1(e2[Z], e2[X], fez, fex);
  P0 := -E2.V[2] * Tv0.V[0] + E2.V[0] * Tv0.V[2];
  P1 := -E2.V[2] * Tv1.V[0] + E2.V[0] * Tv1.V[2];
  Min := MinFloat(P0, P1);
  Max := MaxFloat(P0, P1);
  Rad := Fez * Boxhalfsize.V[0] + Fex * Boxhalfsize.V[2];
  if (Min > Rad) or (Max < -Rad) then
    Exit;

  // AXISTEST_Z12(e2[Y], e2[X], fey, fex);
  P1 := E2.V[1] * Tv1.V[0] - E2.V[0] * Tv1.V[1];
  P2 := E2.V[1] * Tv2.V[0] - E2.V[0] * Tv2.V[1];
  Min := MinFloat(P1, P2);
  Max := MaxFloat(P1, P2);
  Rad := Fey * Boxhalfsize.V[0] + Fex * Boxhalfsize.V[1];
  if (Min > Rad) or (Max < -Rad) then
    Exit;

  // Bullet 1:
  // first test overlap in the {x,y,z}-directions
  // find min, max of the triangle each direction, and test for overlap in
  // that direction -- this is equivalent to testing a minimal AABB around
  // the triangle against the AABB

  // test in X-direction
  FindMinMax(Tv0.V[0], Tv1.V[0], Tv2.V[0], Min, Max);
  if (Min > Boxhalfsize.V[0]) or (Max < -Boxhalfsize.V[0]) then
    Exit;

  // test in Y-direction
  FindMinMax(Tv0.V[1], Tv1.V[1], Tv2.V[1], Min, Max);
  if (Min > Boxhalfsize.V[1]) or (Max < -Boxhalfsize.V[1]) then
    Exit;

  // test in Z-direction
  FindMinMax(Tv0.V[2], Tv1.V[2], Tv2.V[2], Min, Max);
  if (Min > Boxhalfsize.V[2]) or (Max < -Boxhalfsize.V[2]) then
    Exit;

  // Bullet 2:
  // test if the box intersects the plane of the triangle
  // compute plane equation of triangle: normal * x + d = 0
  VectorCrossProduct(E0, E1, Normal);
  D := -VectorDotProduct(Normal, Tv0); // plane eq: normal.x + d = 0
  if not PlaneBoxOverlap(Normal, D, Boxhalfsize) then
    Exit;

  // box and triangle overlaps
  Result := True;
end;

// ExtractAABBCorners
//
procedure ExtractAABBCorners(const AABB: TAABB; var AABBCorners: TAABBCorners);
begin
  MakeVector(AABBCorners[0], AABB.Min.V[0], AABB.Min.V[1],
    AABB.Min.V[2]);
  MakeVector(AABBCorners[1], AABB.Min.V[0], AABB.Min.V[1],
    AABB.Max.V[2]);
  MakeVector(AABBCorners[2], AABB.Min.V[0], AABB.Max.V[1],
    AABB.Min.V[2]);
  MakeVector(AABBCorners[3], AABB.Min.V[0], AABB.Max.V[1],
    AABB.Max.V[2]);

  MakeVector(AABBCorners[4], AABB.Max.V[0], AABB.Min.V[1],
    AABB.Min.V[2]);
  MakeVector(AABBCorners[5], AABB.Max.V[0], AABB.Min.V[1],
    AABB.Max.V[2]);
  MakeVector(AABBCorners[6], AABB.Max.V[0], AABB.Max.V[1],
    AABB.Min.V[2]);
  MakeVector(AABBCorners[7], AABB.Max.V[0], AABB.Max.V[1],
    AABB.Max.V[2]);
end;

// AABBToBSphere
//
procedure AABBToBSphere(const AABB: TAABB; var BSphere: TBSphere);
begin
  BSphere.Center := VectorScale(VectorAdd(AABB.Min, AABB.Max), 0.5);
  BSphere.Radius := VectorDistance(AABB.Min, AABB.Max) * 0.5;
end;

// BSphereToAABB (bsphere)
//
procedure BSphereToAABB(const BSphere: TBSphere; var AABB: TAABB);
begin
  AABB.Min := VectorSubtract(BSphere.Center, BSphere.Radius);
  AABB.Max := VectorAdd(BSphere.Center, BSphere.Radius);
end;

// BSphereToAABB (affine center, radius)
//
function BSphereToAABB(const Center: TAffineVector; Radius: Single): TAABB;
begin
  Result.Min := VectorSubtract(Center, Radius);
  Result.Max := VectorAdd(Center, Radius);
end;

// BSphereToAABB (hmg center, radius)
//
function BSphereToAABB(const Center: TVector; Radius: Single): TAABB;
begin
  SetVector(Result.Min, VectorSubtract(Center, Radius));
  SetVector(Result.Max, VectorAdd(Center, Radius));
end;

// AABBContainsAABB
//
function AABBContainsAABB(const MainAABB, TestAABB: TAABB): TSpaceContains;
begin
  // AABB1 fits completely inside AABB2?
  // AABB1 min must be >= to AABB2 min
  // AABB1 max must be <= to AABB2 max

  if ((MainAABB.Min.V[0] < TestAABB.Max.V[0]) and
    (MainAABB.Min.V[1] < TestAABB.Max.V[1]) and
    (MainAABB.Min.V[2] < TestAABB.Max.V[2]) and

    (TestAABB.Min.V[0] < MainAABB.Max.V[0]) and
    (TestAABB.Min.V[1] < MainAABB.Max.V[1]) and
    (TestAABB.Min.V[2] < MainAABB.Max.V[2])) then
  begin
    if (TestAABB.Min.V[0] >= MainAABB.Min.V[0]) and
      (TestAABB.Min.V[1] >= MainAABB.Min.V[1]) and
      (TestAABB.Min.V[2] >= MainAABB.Min.V[2]) and

      (TestAABB.Max.V[0] <= MainAABB.Max.V[0]) and
      (TestAABB.Max.V[1] <= MainAABB.Max.V[1]) and
      (TestAABB.Max.V[2] <= MainAABB.Max.V[2]) then
      Result := ScContainsFully
    else
      Result := ScContainsPartially;
  end
  else
    Result := ScNoOverlap;
end;

// AABBContainsBSphere
//
function AABBContainsBSphere(const MainAABB: TAABB; const TestBSphere: TBSphere)
  : TSpaceContains;
var
  TestAABB: TAABB;
begin
  BSphereToAABB(TestBSphere, TestAABB);
  Result := AABBContainsAABB(MainAABB, TestAABB);
end;

function PlaneContainsBSphere(const Location, Normal: TAffineVector;
  const TestBSphere: TBSphere): TSpaceContains;
var
  Dist: Single;
begin
  Dist := PointPlaneDistance(TestBSphere.Center, Location, Normal);

  if Dist > TestBSphere.Radius then
    Result := ScNoOverlap
  else if Abs(Dist) <= TestBSphere.Radius then
    Result := ScContainsPartially
  else
    Result := ScContainsFully;
end;

// FrustumContainsBSphere
//
function FrustumContainsBSphere(const Frustum: TFrustum;
  const TestBSphere: TBSphere): TSpaceContains;
var
  NegRadius: Single;
  HitCount: Integer;
  Distance: Single;
  I: Integer;
type
  TPlaneArray = array [0 .. 5] of THmgPlane;
begin
  NegRadius := -TestBSphere.Radius;

  HitCount := 0;

  // This would be fractionally faster to unroll, but oh so ugly!?
  for I := 0 to 5 do
  begin
    Distance := PlaneEvaluatePoint(TPlaneArray(Frustum)[I], TestBSphere.Center);
    if Distance < NegRadius then
    begin
      Result := ScNoOverlap;
      Exit;
    end
    else if Distance >= TestBSphere.Radius then
      Inc(HitCount);
  end; // }

  if HitCount = 6 then
    Result := ScContainsFully
  else
    Result := ScContainsPartially;
end;

// FrustumContainsBSphere
// see http://www.flipcode.com/articles/article_frustumculling.shtml
function FrustumContainsAABB(const Frustum: TFrustum; const TestAABB: TAABB)
  : TSpaceContains;
type
  TPlaneArray = array [0 .. 5] of THmgPlane;
var
  IPlane, ICorner: Integer;
  PointIn: Boolean;
  AABBCorners: TAABBCorners;
  InCount: Integer;
  TotalIn: Integer;
begin
  ExtractAABBCorners(TestAABB, AABBCorners);

  TotalIn := 0;
  // test all 8 corners against the 6 sides
  // if all points are behind 1 specific plane, we are out
  // if we are in with all points, then we are fully in

  // For each plane
  for IPlane := Low(TPlaneArray) to High(TPlaneArray) do
  begin
    // We're about to test 8 corners
    InCount := 8;
    PointIn := True;

    // For each corner
    for ICorner := Low(AABBCorners) to High(AABBCorners) do
    begin
      if PlaneEvaluatePoint(TPlaneArray(Frustum)[IPlane], AABBCorners[ICorner]
        ) < 0 then
      begin
        PointIn := False;
        Dec(InCount);
      end;
    end;

    if InCount = 0 then
    begin
      Result := ScNoOverlap;
      Exit;
    end

    else if PointIn then
      Inc(TotalIn);
  end;

  if TotalIn = 6 then
    Result := ScContainsFully
  else
    Result := ScContainsPartially;
end;

// BSphereContainsAABB
//
function BSphereContainsAABB(const MainBSphere: TBSphere; const TestAABB: TAABB)
  : TSpaceContains;
var
  R2: Single;
  ClippedCenter: TAffineVector;

  AABBCorners: TAABBCorners;
  CornerHitCount: Integer;
begin
  R2 := Sqr(MainBSphere.Radius);

  ClippedCenter := ClipToAABB(MainBSphere.Center, TestAABB);

  if VectorDistance2(ClippedCenter, MainBSphere.Center) < R2 then
  begin
    ExtractAABBCorners(TestAABB, AABBCorners);

    CornerHitCount := 0;
    // BSphere fully contains aabb if all corners of aabb are within bsphere.
    if (VectorDistance2(MainBSphere.Center, AABBCorners[0]) < R2) then
      Inc(CornerHitCount);
    if (VectorDistance2(MainBSphere.Center, AABBCorners[1]) < R2) then
      Inc(CornerHitCount);
    if (VectorDistance2(MainBSphere.Center, AABBCorners[2]) < R2) then
      Inc(CornerHitCount);
    if (VectorDistance2(MainBSphere.Center, AABBCorners[3]) < R2) then
      Inc(CornerHitCount);
    if (VectorDistance2(MainBSphere.Center, AABBCorners[4]) < R2) then
      Inc(CornerHitCount);
    if (VectorDistance2(MainBSphere.Center, AABBCorners[5]) < R2) then
      Inc(CornerHitCount);
    if (VectorDistance2(MainBSphere.Center, AABBCorners[6]) < R2) then
      Inc(CornerHitCount);
    if (VectorDistance2(MainBSphere.Center, AABBCorners[7]) < R2) then
      Inc(CornerHitCount);

    if CornerHitCount = 7 then
      Result := ScContainsFully
    else
      Result := ScContainsPartially;
  end
  else
    Result := ScNoOverlap;
end;

// BSphereContainsBSphere
//
function BSphereContainsBSphere(const MainBSphere, TestBSphere: TBSphere)
  : TSpaceContains;
var
  D2: Single;
begin
  D2 := VectorDistance2(MainBSphere.Center, TestBSphere.Center);

  if D2 < Sqr(MainBSphere.Radius + TestBSphere.Radius) then
  begin
    if D2 < Sqr(MainBSphere.Radius - TestBSphere.Radius) then
      Result := ScContainsFully
    else
      Result := ScContainsPartially;
  end
  else
    Result := ScNoOverlap;
end;

// BSphereIntersectsBSphere
//
function BSphereIntersectsBSphere(const MainBSphere,
  TestBSphere: TBSphere): Boolean;
begin
  Result := VectorDistance2(MainBSphere.Center, TestBSphere.Center) <
    Sqr(MainBSphere.Radius + TestBSphere.Radius);
end;

// ClipToAABB
//
function ClipToAABB(const V: TAffineVector; const AABB: TAABB): TAffineVector;
begin
  Result := V;

  if Result.V[0] < AABB.Min.V[0] then
    Result.V[0] := AABB.Min.V[0];
  if Result.V[1] < AABB.Min.V[1] then
    Result.V[1] := AABB.Min.V[1];
  if Result.V[2] < AABB.Min.V[2] then
    Result.V[2] := AABB.Min.V[2];

  if Result.V[0] > AABB.Max.V[0] then
    Result.V[0] := AABB.Max.V[0];
  if Result.V[1] > AABB.Max.V[1] then
    Result.V[1] := AABB.Max.V[1];
  if Result.V[2] > AABB.Max.V[2] then
    Result.V[2] := AABB.Max.V[2];
end;

// IncludeInClipRect
//
procedure IncludeInClipRect(var ClipRect: TClipRect; X, Y: Single);
begin
  with ClipRect do
  begin
    if X < Left then
      Left := X;
    if X > Right then
      Right := X;
    if Y < Top then
      Top := Y;
    if Y > Bottom then
      Bottom := Y;
  end;
end;

// AABBToClipRect
//
function AABBToClipRect(const Aabb: TAABB; ModelViewProjection: TMatrix;
  ViewportSizeX, ViewportSizeY: Integer): TClipRect;
var
  I: Integer;
  V, Vt: TVector;
  Minmax: array [0 .. 1] of PAffineVector;
begin
  Minmax[0] := @Aabb.Min;
  Minmax[1] := @Aabb.Max;
  V.V[3] := 1;
  for I := 0 to 7 do
  begin
    V.V[0] := Minmax[I and 1].V[0];
    V.V[1] := Minmax[(I shr 1) and 1].V[1];
    V.V[2] := Minmax[(I shr 2) and 1].V[2];

    // Project
    Vt := VectorTransform(V, ModelViewProjection);
    ScaleVector(Vt, 1 / Vt.V[3]);

    // Convert to screen coordinates
    if I > 0 then
      IncludeInClipRect(Result, ViewportSizeX * (Vt.V[0] + 1) * 0.5,
        ViewportSizeY * (Vt.V[1] + 1) * 0.5)
    else
    begin
      Result.Left := ViewportSizeX * (Vt.V[0] + 1) * 0.5;
      Result.Top := ViewportSizeY * (Vt.V[1] + 1) * 0.5;
      Result.Right := Result.Left;
      Result.Bottom := Result.Top;
    end;
  end;
end;

function RayCastAABBIntersect(const RayOrigin, RayDirection: TVector;
  const Aabb: TAABB; out TNear, TFar: Single): Boolean; overload;
const
  Infinity = 1.0 / 0.0;
var
  P: Integer;
  InvDir: Double;
  T0, T1, Tmp: Single;
begin
  Result := False;

  TNear := -Infinity;
  TFar := Infinity;

  for P := 0 to 2 do
  begin
    if (RayDirection.V[P] = 0) then
    begin
      if ((RayOrigin.V[P] < Aabb.Min.V[P]) or
        (RayOrigin.V[P] > Aabb.Max.V[P])) then
        Exit;
    end
    else
    begin
      InvDir := 1 / RayDirection.V[P];
      T0 := (Aabb.Min.V[P] - RayOrigin.V[P]) * InvDir;
      T1 := (Aabb.Max.V[P] - RayOrigin.V[P]) * InvDir;

      if (T0 > T1) then
      begin
        Tmp := T0;
        T0 := T1;
        T1 := Tmp;
      end;

      if (T0 > TNear) then
        TNear := T0;
      if (T1 < TFar) then
        TFar := T1;

      if ((TNear > TFar) or (TFar < 0)) then
        Exit;
    end;
  end;

  Result := True;
end;

function RayCastAABBIntersect(const RayOrigin, RayDirection: TVector;
  const Aabb: TAABB; IntersectPoint: PVector = nil): Boolean; overload;
var
  TNear, TFar: Single;
begin
  Result := RayCastAABBIntersect(RayOrigin, RayDirection, Aabb, TNear, TFar);

  if Result and Assigned(IntersectPoint) then
  begin
    if TNear >= 0 then
      // origin outside the box
      IntersectPoint^ := VectorCombine(RayOrigin, RayDirection, 1, TNear)
    else
      // origin inside the box, near is "behind", use far
      IntersectPoint^ := VectorCombine(RayOrigin, RayDirection, 1, TFar);
  end;
end;

end.
