//
// This unit is part of the GLScene Project, http://glscene.org
//
{
  Geometric objects.

   History :  
   10/11/12 - PW - Added CPP compatibility: changed vector arrays to records
   13/05/11 - Vince - Add ArrowArc object
   13/05/11 - Vince - Add StartAngle ,StopAngle and Parts attributes
                 to display a slice of TGLTorus between start and stop angles
   24/03/11 - Yar - Replaced TGLTorus primitives to triangles, added tangent and binormal attributes
   23/08/10 - Yar - Added OpenGLTokens to uses, replaced OpenGL1x functions to OpenGLAdapter
   22/04/10 - Yar - Fixes after GLState revision
   15/03/08 - DaStr - Deleted TGLFrustrum.AxisAlignedBoundingBox(),
                 now this function references the inherited function
   20/01/08 - DaStr - Corrected object centering in TGLFrustrum.BuildList()
                 (thanks Sandor Domokos) (BugTrackerID = 1864314)
  Added a TGLCapsule object (thanks Dave Gravel)
   18/11/07 - DaStr - Got rid of compiler warning in TGLCone.RayCastIntersect
   07/05/07 - DanB - Added TGLCone.RayCastIntersect
  Improved TGLDisk.RayCastIntersect
   30/03/07 - DaStr - Added $I GLScene.inc
   25/09/04 - Eric Pascual - Added AxisAlignedBoundingBox,
                 AxisAlignedBoundingBoxUnscaled,
                 AxisAlignedDimensionsUnscaled
   02/08/04 - LR, YHC - BCB corrections: use record instead array
   29/11/03 - MF - Added shadow silhouette code for TGLCylinderBase et al.
  Added GetTopRadius to facilitate silhouette.
   24/10/03 - NelC - Fixed TGLTorus texture coord. bug
   21/07/03 - EG - Creation from GLObjects split
   
}
unit GLGeomObjects;

{$I GLScene.inc}

interface

uses
  Classes,
  GLScene,
  GLVectorGeometry,
  OpenGLTokens,
  OpenGLAdapter,
  GLContext,
  GLObjects,
  GLSilhouette,
  GLVectorTypes,
  GLGeometryBB,
  GLRenderContextInfo;

type

  // TGLDisk
  //
  { : A Disk object.
    The disk may not be complete, it can have a hole (controled by the
    InnerRadius property) and can only be a slice (controled by the StartAngle
    and SweepAngle properties). }
  TGLDisk = class(TGLQuadricObject)
  private
     
    FStartAngle, FSweepAngle, FOuterRadius, FInnerRadius: TGLFloat;
    FSlices, FLoops: TGLInt;
    procedure SetOuterRadius(const aValue: Single);
    procedure SetInnerRadius(const aValue: Single);
    procedure SetSlices(aValue: TGLInt);
    procedure SetLoops(aValue: TGLInt);
    procedure SetStartAngle(const aValue: Single);
    procedure SetSweepAngle(const aValue: Single);

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;

    procedure Assign(Source: TPersistent); override;
    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;

  published
     
    { : Allows defining a "hole" in the disk. }
    property InnerRadius: TGLFloat read FInnerRadius write SetInnerRadius;
    { : Number of radial mesh subdivisions. }
    property Loops: TGLInt read FLoops write SetLoops default 2;
    { : Outer radius for the disk.
      If you leave InnerRadius at 0, this is the disk radius. }
    property OuterRadius: TGLFloat read FOuterRadius write SetOuterRadius;
    { : Number of mesh slices.
      For instance, if Slices=6, your disk will look like an hexagon. }
    property Slices: TGLInt read FSlices write SetSlices default 16;
    property StartAngle: TGLFloat read FStartAngle write SetStartAngle;
    property SweepAngle: TGLFloat read FSweepAngle write SetSweepAngle;
  end;

  // TGLCylinderBase
  //
  { : Base class to cylinder-like objects.
    Introduces the basic cylinder description properties.
    Be aware teh default slices and stacks make up for a high-poly cylinder,
    unless you're after high-quality lighting it is recommended to reduce the
    Stacks property to 1. }
  TGLCylinderBase = class(TGLQuadricObject)
  private
     
    FBottomRadius: TGLFloat;
    FSlices, FStacks, FLoops: TGLInt;
    FHeight: TGLFloat;

  protected
     
    procedure SetBottomRadius(const aValue: Single);
    procedure SetHeight(const aValue: Single);
    procedure SetSlices(aValue: TGLInt);
    procedure SetStacks(aValue: TGLInt);
    procedure SetLoops(aValue: TGLInt);
    function GetTopRadius: Single; virtual;
  public
     
    constructor Create(AOwner: TComponent); override;

    procedure Assign(Source: TPersistent); override;

    function GenerateSilhouette(const silhouetteParameters
      : TGLSilhouetteParameters): TGLSilhouette; override;
  published
     
    property BottomRadius: TGLFloat read FBottomRadius write SetBottomRadius;
    property Height: TGLFloat read FHeight write SetHeight;
    property Slices: TGLInt read FSlices write SetSlices default 16;
    property Stacks: TGLInt read FStacks write SetStacks default 4;
    { : Number of concentric rings for top/bottom disk(s). }
    property Loops: TGLInt read FLoops write SetLoops default 1;
  end;

  // TConePart
  //
  TConePart = (coSides, coBottom);
  TConeParts = set of TConePart;

  // TGLCone
  //
  { : A cone object. }
  TGLCone = class(TGLCylinderBase)
  private
     
    FParts: TConeParts;

  protected
     
    procedure SetParts(aValue: TConeParts);
    function GetTopRadius: Single; override;

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure Assign(Source: TPersistent); override;

    procedure BuildList(var rci: TGLRenderContextInfo); override;
    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;

  published
     
    property Parts: TConeParts read FParts write SetParts
      default [coSides, coBottom];
  end;

  // TCylinderPart
  //
  TCylinderPart = (cySides, cyBottom, cyTop);
  TCylinderParts = set of TCylinderPart;

  // TCylinderAlignment
  //
  TCylinderAlignment = (caCenter, caTop, caBottom);

  // TGLCylinder
  //
  { : Cylinder object, can also be used to make truncated cones }
  TGLCylinder = class(TGLCylinderBase)
  private
     
    FParts: TCylinderParts;
    FTopRadius: TGLFloat;
    FAlignment: TCylinderAlignment;

  protected
     
    procedure SetTopRadius(const aValue: Single);
    procedure SetParts(aValue: TCylinderParts);
    procedure SetAlignment(val: TCylinderAlignment);
    function GetTopRadius: Single; override;

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure Assign(Source: TPersistent); override;

    procedure BuildList(var rci: TGLRenderContextInfo); override;
    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;

    procedure Align(const startPoint, endPoint: TVector); overload;
    procedure Align(const startObj, endObj: TGLBaseSceneObject); overload;
    procedure Align(const startPoint, endPoint: TAffineVector); overload;

  published
     
    property TopRadius: TGLFloat read FTopRadius write SetTopRadius;
    property Parts: TCylinderParts read FParts write SetParts
      default [cySides, cyBottom, cyTop];
    property Alignment: TCylinderAlignment read FAlignment write SetAlignment
      default caCenter;
  end;

  { : Capsule object, can also be used to make truncated cones }
  TGLCapsule = class(TGLSceneObject)
  private
     
    FParts: TCylinderParts;
    FRadius: TGLFloat;
    FSlices: TGLInt;
    FStacks: TGLInt;
    FHeight: TGLFloat;
    FAlignment: TCylinderAlignment;
  protected
     
    procedure SetHeight(const aValue: Single);
    procedure SetRadius(const aValue: Single);
    procedure SetSlices(const aValue: integer);
    procedure SetStacks(const aValue: integer);
    procedure SetParts(aValue: TCylinderParts);
    procedure SetAlignment(val: TCylinderAlignment);
  public
     
    constructor Create(AOwner: TComponent); override;
    procedure Assign(Source: TPersistent); override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;
    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;
    procedure Align(const startPoint, endPoint: TVector); overload;
    procedure Align(const startObj, endObj: TGLBaseSceneObject); overload;
    procedure Align(const startPoint, endPoint: TAffineVector); overload;
  published
     
    property Height: TGLFloat read FHeight write SetHeight;
    property Slices: TGLInt read FSlices write SetSlices;
    property Stacks: TGLInt read FStacks write SetStacks;
    property Radius: TGLFloat read FRadius write SetRadius;
    property Parts: TCylinderParts read FParts write SetParts
      default [cySides, cyBottom, cyTop];
    property Alignment: TCylinderAlignment read FAlignment write SetAlignment
      default caCenter;
  end;

  // TAnnulusPart
  //
  TAnnulusPart = (anInnerSides, anOuterSides, anBottom, anTop);
  TAnnulusParts = set of TAnnulusPart;

  // TGLAnnulus
  //
  { : An annulus is a cylinder that can be made hollow (pipe-like). }
  TGLAnnulus = class(TGLCylinderBase)
  private
     
    FParts: TAnnulusParts;
    FBottomInnerRadius: TGLFloat;
    FTopInnerRadius: TGLFloat;
    FTopRadius: TGLFloat;

  protected
     
    procedure SetTopRadius(const aValue: Single);
    procedure SetTopInnerRadius(const aValue: Single);
    procedure SetBottomInnerRadius(const aValue: Single);
    procedure SetParts(aValue: TAnnulusParts);

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure Assign(Source: TPersistent); override;

    procedure BuildList(var rci: TGLRenderContextInfo); override;
    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;

  published
     
    property BottomInnerRadius: TGLFloat read FBottomInnerRadius
      write SetBottomInnerRadius;
    property TopInnerRadius: TGLFloat read FTopInnerRadius
      write SetTopInnerRadius;
    property TopRadius: TGLFloat read FTopRadius write SetTopRadius;
    property Parts: TAnnulusParts read FParts write SetParts
      default [anInnerSides, anOuterSides, anBottom, anTop];
  end;

  // TTorusPart
  //
  TTorusPart = (toSides, toStartDisk, toStopDisk);
  TTorusParts = set of TTorusPart;

  // TGLTorus
  //
  { : A Torus object. }
  TGLTorus = class(TGLSceneObject)
  private
     
    FParts: TTorusParts;
    FRings, FSides: Cardinal;
    FStartAngle, FStopAngle: Single;
    FMinorRadius, FMajorRadius: Single;
    FMesh: array of array of TVertexRec;
  protected
     
    procedure SetMajorRadius(const aValue: Single);
    procedure SetMinorRadius(const aValue: Single);
    procedure SetRings(aValue: Cardinal);
    procedure SetSides(aValue: Cardinal);
    procedure SetStartAngle(const aValue: Single);
    procedure SetStopAngle(const aValue: Single);
    procedure SetParts(aValue: TTorusParts);

  public
     
    constructor Create(AOwner: TComponent); override;

    procedure BuildList(var rci: TGLRenderContextInfo); override;
    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;

  published
     
    property MajorRadius: Single read FMajorRadius write SetMajorRadius;
    property MinorRadius: Single read FMinorRadius write SetMinorRadius;
    property Rings: Cardinal read FRings write SetRings default 25;
    property Sides: Cardinal read FSides write SetSides default 15;
    property StartAngle: Single read FStartAngle write SetStartAngle;
    property StopAngle: Single read FStopAngle write SetStopAngle;
    property Parts: TTorusParts read FParts write SetParts default [toSides];
  end;

  // TArrowLinePart
  //
  TArrowLinePart = (alLine, alTopArrow, alBottomArrow);
  TArrowLineParts = set of TArrowLinePart;

  // TArrowHeadStackingStyle
  //
  TArrowHeadStackingStyle = (ahssStacked, ahssCentered, ahssIncluded);

  // TGLArrowLine
  //
  { : Draws an arrowhead (cylinder + cone).
    The arrow head is a cone that shares the attributes of the cylinder
    (ie stacks/slices, materials etc). Seems to work ok. 
    This is useful for displaying a vector based field (eg velocity) or
    other arrows that might be required. 
    By default the bottom arrow is off }
  TGLArrowLine = class(TGLCylinderBase)
  private
     
    FParts: TArrowLineParts;
    FTopRadius: Single;
    fTopArrowHeadHeight: Single;
    fTopArrowHeadRadius: Single;
    fBottomArrowHeadHeight: Single;
    fBottomArrowHeadRadius: Single;
    FHeadStackingStyle: TArrowHeadStackingStyle;

  protected
     
    procedure SetTopRadius(const aValue: Single);
    procedure SetTopArrowHeadHeight(const aValue: Single);
    procedure SetTopArrowHeadRadius(const aValue: Single);
    procedure SetBottomArrowHeadHeight(const aValue: Single);
    procedure SetBottomArrowHeadRadius(const aValue: Single);
    procedure SetParts(aValue: TArrowLineParts);
    procedure SetHeadStackingStyle(const val: TArrowHeadStackingStyle);

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;
    procedure Assign(Source: TPersistent); override;

  published
     
    property TopRadius: TGLFloat read FTopRadius write SetTopRadius;
    property HeadStackingStyle: TArrowHeadStackingStyle read FHeadStackingStyle
      write SetHeadStackingStyle default ahssStacked;
    property Parts: TArrowLineParts read FParts write SetParts
      default [alLine, alTopArrow];
    property TopArrowHeadHeight: TGLFloat read fTopArrowHeadHeight
      write SetTopArrowHeadHeight;
    property TopArrowHeadRadius: TGLFloat read fTopArrowHeadRadius
      write SetTopArrowHeadRadius;
    property BottomArrowHeadHeight: TGLFloat read fBottomArrowHeadHeight
      write SetBottomArrowHeadHeight;
    property BottomArrowHeadRadius: TGLFloat read fBottomArrowHeadRadius
      write SetBottomArrowHeadRadius;
  end;

  // TArrowArcPart
  //
  TArrowArcPart = (aaArc, aaTopArrow, aaBottomArrow);
  TArrowArcParts = set of TArrowArcPart;

  // TGLArrowArc
  //
  { : Draws an arrowhead (Sliced Torus + cone).
    The arrow head is a cone that shares the attributes of the Torus
    (ie stacks/slices, materials etc). 
    This is useful for displaying a movement (eg twist) or
    other arc arrows that might be required. 
    By default the bottom arrow is off }
  TGLArrowArc = class(TGLCylinderBase)
  private
     
    fArcRadius: Single;
    FStartAngle: Single;
    FStopAngle: Single;
    FParts: TArrowArcParts;
    FTopRadius: Single;
    fTopArrowHeadHeight: Single;
    fTopArrowHeadRadius: Single;
    fBottomArrowHeadHeight: Single;
    fBottomArrowHeadRadius: Single;
    FHeadStackingStyle: TArrowHeadStackingStyle;
    FMesh: array of array of TVertexRec;

  protected
     
    procedure SetArcRadius(const aValue: Single);
    procedure SetStartAngle(const aValue: Single);
    procedure SetStopAngle(const aValue: Single);
    procedure SetTopRadius(const aValue: Single);
    procedure SetTopArrowHeadHeight(const aValue: Single);
    procedure SetTopArrowHeadRadius(const aValue: Single);
    procedure SetBottomArrowHeadHeight(const aValue: Single);
    procedure SetBottomArrowHeadRadius(const aValue: Single);
    procedure SetParts(aValue: TArrowArcParts);
    procedure SetHeadStackingStyle(const val: TArrowHeadStackingStyle);

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;
    procedure Assign(Source: TPersistent); override;

  published
     
    property ArcRadius: TGLFloat read fArcRadius write SetArcRadius;
    property StartAngle: TGLFloat read FStartAngle write SetStartAngle;
    property StopAngle: TGLFloat read FStopAngle write SetStopAngle;
    property TopRadius: TGLFloat read FTopRadius write SetTopRadius;
    property HeadStackingStyle: TArrowHeadStackingStyle read FHeadStackingStyle
      write SetHeadStackingStyle default ahssStacked;
    property Parts: TArrowArcParts read FParts write SetParts
      default [aaArc, aaTopArrow];
    property TopArrowHeadHeight: TGLFloat read fTopArrowHeadHeight
      write SetTopArrowHeadHeight;
    property TopArrowHeadRadius: TGLFloat read fTopArrowHeadRadius
      write SetTopArrowHeadRadius;
    property BottomArrowHeadHeight: TGLFloat read fBottomArrowHeadHeight
      write SetBottomArrowHeadHeight;
    property BottomArrowHeadRadius: TGLFloat read fBottomArrowHeadRadius
      write SetBottomArrowHeadRadius;
  end;

  // TPolygonParts
  //
  TPolygonPart = (ppTop, ppBottom);
  TPolygonParts = set of TPolygonPart;

  // TGLPolygon
  //
  { : A basic polygon object.
    The curve is described by the Nodes and SplineMode properties, should be
    planar and is automatically tessellated.
    Texture coordinates are deduced from X and Y coordinates only.
    This object allows only for polygons described by a single curve, if you
    need "complex polygons" with holes, patches and cutouts, see GLMultiPolygon. }
  TGLPolygon = class(TGLPolygonBase)
  private
     
    FParts: TPolygonParts;

  protected
     
    procedure SetParts(const val: TPolygonParts);

  public
     
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure Assign(Source: TPersistent); override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;

  published
     
    { : Parts of polygon.
      The 'top' of the polygon is the position were the curve describing
      the polygon spin counter-clockwise (i.e. right handed convention). }
    property Parts: TPolygonParts read FParts write SetParts
      default [ppTop, ppBottom];
  end;

  // TFrustrumParts
  //
  TFrustrumPart = (fpTop, fpBottom, fpFront, fpBack, fpLeft, fpRight);
  TFrustrumParts = set of TFrustrumPart;

const
  cAllFrustrumParts = [fpTop, fpBottom, fpFront, fpBack, fpLeft, fpRight];

type
  // TGLFrustrum
  //
  { A frustrum is a pyramid with the top chopped off.
    The height of the imaginary pyramid is ApexHeight, the height of the
    frustrum is Height. If ApexHeight and Height are the same, the frustrum
    degenerates into a pyramid. 
    Height cannot be greater than ApexHeight. }
  TGLFrustrum = class(TGLSceneObject)
  private
     
    FApexHeight, FBaseDepth, FBaseWidth, FHeight: TGLFloat;
    FParts: TFrustrumParts;
    FNormalDirection: TNormalDirection;
    procedure SetApexHeight(const aValue: Single);
    procedure SetBaseDepth(const aValue: Single);
    procedure SetBaseWidth(const aValue: Single);
    procedure SetHeight(const aValue: Single);
    procedure SetParts(aValue: TFrustrumParts);
    procedure SetNormalDirection(aValue: TNormalDirection);

  protected
     
    procedure DefineProperties(Filer: TFiler); override;
    procedure ReadData(Stream: TStream);
    procedure WriteData(Stream: TStream);

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;
    procedure Assign(Source: TPersistent); override;
    function TopDepth: TGLFloat;
    function TopWidth: TGLFloat;
    function AxisAlignedBoundingBoxUnscaled: TAABB;
    function AxisAlignedDimensionsUnscaled: TVector; override;
  published
     
    property ApexHeight: TGLFloat read FApexHeight write SetApexHeight
      stored False;
    property BaseDepth: TGLFloat read FBaseDepth write SetBaseDepth
      stored False;
    property BaseWidth: TGLFloat read FBaseWidth write SetBaseWidth
      stored False;
    property Height: TGLFloat read FHeight write SetHeight stored False;
    property NormalDirection: TNormalDirection read FNormalDirection
      write SetNormalDirection default ndOutside;
    property Parts: TFrustrumParts read FParts write SetParts
      default cAllFrustrumParts;
  end;

  // -------------------------------------------------------------
  // -------------------------------------------------------------
  // -------------------------------------------------------------
implementation

// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------

uses
  GLPolynomials,
  XOpenGL;

// ------------------
// ------------------ TGLDisk ------------------
// ------------------

// Create
//

constructor TGLDisk.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FOuterRadius := 0.5;
  FInnerRadius := 0;
  FSlices := 16;
  FLoops := 2;
  FStartAngle := 0;
  FSweepAngle := 360;
end;

// BuildList
//

procedure TGLDisk.BuildList(var rci: TGLRenderContextInfo);
var
  quadric: PGLUquadricObj;
begin
  quadric := gluNewQuadric();
  SetupQuadricParams(quadric);
  gluPartialDisk(quadric, FInnerRadius, FOuterRadius, FSlices, FLoops,
    FStartAngle, FSweepAngle);
  gluDeleteQuadric(quadric);
end;

// SetOuterRadius
//

procedure TGLDisk.SetOuterRadius(const aValue: Single);
begin
  if aValue <> FOuterRadius then
  begin
    FOuterRadius := aValue;
    StructureChanged;
  end;
end;

// SetInnerRadius
//

procedure TGLDisk.SetInnerRadius(const aValue: Single);
begin
  if aValue <> FInnerRadius then
  begin
    FInnerRadius := aValue;
    StructureChanged;
  end;
end;

// SetSlices
//

procedure TGLDisk.SetSlices(aValue: integer);
begin
  if aValue <> FSlices then
  begin
    FSlices := aValue;
    StructureChanged;
  end;
end;

// SetLoops
//

procedure TGLDisk.SetLoops(aValue: integer);
begin
  if aValue <> FLoops then
  begin
    FLoops := aValue;
    StructureChanged;
  end;
end;

// SetStartAngle
//

procedure TGLDisk.SetStartAngle(const aValue: Single);
begin
  if aValue <> FStartAngle then
  begin
    FStartAngle := aValue;
    StructureChanged;
  end;
end;

// SetSweepAngle
//

procedure TGLDisk.SetSweepAngle(const aValue: Single);
begin
  if aValue <> FSweepAngle then
  begin
    FSweepAngle := aValue;
    StructureChanged;
  end;
end;

 
//

procedure TGLDisk.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLDisk) then
  begin
    FOuterRadius := TGLDisk(Source).FOuterRadius;
    FInnerRadius := TGLDisk(Source).FInnerRadius;
    FSlices := TGLDisk(Source).FSlices;
    FLoops := TGLDisk(Source).FLoops;
    FStartAngle := TGLDisk(Source).FStartAngle;
    FSweepAngle := TGLDisk(Source).FSweepAngle;
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensions
//

function TGLDisk.AxisAlignedDimensionsUnscaled: TVector;
var
  r: TGLFloat;
begin
  r := Abs(FOuterRadius);
  Result := VectorMake(r, r, 0);
end;

// RayCastIntersect
//

function TGLDisk.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
var
  ip: TVector;
  d: Single;
  angle, beginAngle, endAngle: Single;
  localIntPoint: TVector;
begin
  Result := False;
  if SweepAngle > 0 then
    if RayCastPlaneIntersect(rayStart, rayVector, AbsolutePosition,
      AbsoluteDirection, @ip) then
    begin
      if Assigned(intersectPoint) then
        SetVector(intersectPoint^, ip);
      localIntPoint := AbsoluteToLocal(ip);
      d := VectorNorm(localIntPoint);
      if (d >= Sqr(InnerRadius)) and (d <= Sqr(OuterRadius)) then
      begin
        if SweepAngle >= 360 then
          Result := true
        else
        begin
          // arctan2 returns results between -pi and +pi, we want between 0 and 360
          angle := 180 / pi * arctan2(localIntPoint.V[0], localIntPoint.V[1]);
          if angle < 0 then
            angle := angle + 360;
          // we also want StartAngle and StartAngle+SweepAngle to be in this range
          beginAngle := Trunc(StartAngle) mod 360;
          endAngle := Trunc(StartAngle + SweepAngle) mod 360;
          // If beginAngle>endAngle then area crosses the boundary from 360=>0 degrees
          // therefore have 2 valid regions  (beginAngle to 360) & (0 to endAngle)
          // otherwise just 1 valid region (beginAngle to endAngle)
          if beginAngle > endAngle then
          begin
            if (angle > beginAngle) or (angle < endAngle) then
              Result := true;
          end
          else if (angle > beginAngle) and (angle < endAngle) then
            Result := true;
        end;
      end;
    end;
  if Result = true then
    if Assigned(intersectNormal) then
      SetVector(intersectNormal^, AbsoluteUp);

end;

// ------------------
// ------------------ TGLCylinderBase ------------------
// ------------------

// Create
//

constructor TGLCylinderBase.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FBottomRadius := 0.5;
  FHeight := 1;
  FSlices := 16;
  FStacks := 4;
  FLoops := 1;
end;

// SetBottomRadius
//

procedure TGLCylinderBase.SetBottomRadius(const aValue: Single);
begin
  if aValue <> FBottomRadius then
  begin
    FBottomRadius := aValue;
    StructureChanged;
  end;
end;

// GetTopRadius
//

function TGLCylinderBase.GetTopRadius: Single;
begin
  Result := FBottomRadius;
end;

// SetHeight
//

procedure TGLCylinderBase.SetHeight(const aValue: Single);
begin
  if aValue <> FHeight then
  begin
    FHeight := aValue;
    StructureChanged;
  end;
end;

// SetSlices
//

procedure TGLCylinderBase.SetSlices(aValue: TGLInt);
begin
  if aValue <> FSlices then
  begin
    FSlices := aValue;
    StructureChanged;
  end;
end;

// SetStack
//

procedure TGLCylinderBase.SetStacks(aValue: TGLInt);
begin
  if aValue <> FStacks then
  begin
    FStacks := aValue;
    StructureChanged;
  end;
end;

// SetLoops
//

procedure TGLCylinderBase.SetLoops(aValue: TGLInt);
begin
  if (aValue >= 1) and (aValue <> FLoops) then
  begin
    FLoops := aValue;
    StructureChanged;
  end;
end;

 
//

procedure TGLCylinderBase.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLCylinderBase) then
  begin
    FBottomRadius := TGLCylinderBase(Source).FBottomRadius;
    FSlices := TGLCylinderBase(Source).FSlices;
    FStacks := TGLCylinderBase(Source).FStacks;
    FLoops := TGLCylinderBase(Source).FLoops;
    FHeight := TGLCylinderBase(Source).FHeight;
  end;
  inherited Assign(Source);
end;

// GenerateSilhouette
//

function TGLCylinderBase.GenerateSilhouette(const silhouetteParameters
  : TGLSilhouetteParameters): TGLSilhouette;
var
  connectivity: TConnectivity;
  sil: TGLSilhouette;
  ShadowSlices: integer;

  i: integer;
  p: array [0 .. 3] of TVector3f;
  PiDivSlices: Single;
  a1, a2: Single;
  c1, c2: TVector3f;
  cosa1, cosa2, sina1, sina2: Single;
  HalfHeight: Single;
  ShadowTopRadius: Single;
begin
  connectivity := TConnectivity.Create(true);

  ShadowSlices := FSlices div 1;

  if FSlices < 5 then
    FSlices := 5;

  PiDivSlices := 2 * pi / ShadowSlices;

  a1 := 0;

  // Is this a speed improvement or just a waste of code?
  HalfHeight := FHeight / 2;

  MakeVector(c1, 0, -HalfHeight, 0);
  MakeVector(c2, 0, HalfHeight, 0);

  ShadowTopRadius := GetTopRadius;

  for i := 0 to ShadowSlices - 1 do
  begin
    a2 := a1 + PiDivSlices;

    // Is this a speed improvement or just a waste of code?
    cosa1 := cos(a1);
    cosa2 := cos(a2);
    sina1 := sin(a1);
    sina2 := sin(a2);

    // Generate the four "corners";
    // Bottom corners
    MakeVector(p[0], FBottomRadius * sina2, -HalfHeight, FBottomRadius * cosa2);
    MakeVector(p[1], FBottomRadius * sina1, -HalfHeight, FBottomRadius * cosa1);

    // Top corners
    MakeVector(p[2], ShadowTopRadius * sina1, HalfHeight,
      ShadowTopRadius * cosa1);
    MakeVector(p[3], ShadowTopRadius * sina2, HalfHeight,
      ShadowTopRadius * cosa2); // }

    // This should be optimized to use AddIndexedFace, because this method
    // searches for each of the vertices and adds them or re-uses them.

    // Skin
    connectivity.AddFace(p[2], p[1], p[0]);
    connectivity.AddFace(p[3], p[2], p[0]);

    // Sides / caps
    connectivity.AddFace(c1, p[0], p[1]);
    connectivity.AddFace(p[2], p[3], c2);

    a1 := a1 + PiDivSlices;
  end;

  sil := nil;
  connectivity.CreateSilhouette(silhouetteParameters, sil, False);

  Result := sil;

  connectivity.Free;
end;

// ------------------
// ------------------ TGLCone ------------------
// ------------------

// Create
//

constructor TGLCone.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FParts := [coSides, coBottom];
end;

// BuildList
//

procedure TGLCone.BuildList(var rci: TGLRenderContextInfo);
var
  quadric: PGLUquadricObj;
begin
  GL.PushMatrix;
  quadric := gluNewQuadric();
  SetupQuadricParams(quadric);
  GL.Rotated(-90, 1, 0, 0);
  GL.Translatef(0, 0, -FHeight * 0.5);
  if coSides in FParts then
    gluCylinder(quadric, BottomRadius, 0, Height, Slices, Stacks);
  if coBottom in FParts then
  begin
    // top of a disk is defined as outside
    SetInvertedQuadricOrientation(quadric);
    gluDisk(quadric, 0, BottomRadius, Slices, FLoops);
  end;
  gluDeleteQuadric(quadric);
  GL.PopMatrix;
end;

// SetParts
//

procedure TGLCone.SetParts(aValue: TConeParts);
begin
  if aValue <> FParts then
  begin
    FParts := aValue;
    StructureChanged;
  end;
end;

 
//

procedure TGLCone.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLCone) then
  begin
    FParts := TGLCone(Source).FParts;
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensions
//

function TGLCone.AxisAlignedDimensionsUnscaled: TVector;
var
  r: TGLFloat;
begin
  r := Abs(FBottomRadius);
  Result := VectorMake(r { *Scale.DirectX } , 0.5 * FHeight { *Scale.DirectY } ,
    r { *Scale.DirectZ } );
end;

// GetTopRadius
//

function TGLCone.GetTopRadius: Single;
begin
  Result := 0;
end;

// RayCastIntersect
//

function TGLCone.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
var
  ip, localRayStart, localRayVector: TVector;
  poly: array [0 .. 2] of Double;
  roots: TDoubleArray;
  minRoot: Double;
  d, t, hconst: Single;
begin
  Result := False;
  localRayStart := AbsoluteToLocal(rayStart);
  localRayVector := VectorNormalize(AbsoluteToLocal(rayVector));

  if coBottom in Parts then
  begin
    // bottom can only be raycast from beneath
    if localRayStart.V[1] < -FHeight * 0.5 then
    begin
      if RayCastPlaneIntersect(localRayStart, localRayVector,
        PointMake(0, -FHeight * 0.5, 0), YHmgVector, @ip) then
      begin
        d := VectorNorm(ip.V[0], ip.V[2]);
        if (d <= Sqr(BottomRadius)) then
        begin
          Result := true;
          if Assigned(intersectPoint) then
            SetVector(intersectPoint^, LocalToAbsolute(ip));
          if Assigned(intersectNormal) then
            SetVector(intersectNormal^, VectorNegate(AbsoluteUp));
          Exit;
        end;
      end;
    end;
  end;
  if coSides in Parts then
  begin
    hconst := -Sqr(BottomRadius) / Sqr(Height);
    // intersect against infinite cones (in positive and negative direction)
    poly[0] := Sqr(localRayStart.V[0]) + hconst *
               Sqr(localRayStart.V[1] - 0.5 * FHeight) +
               Sqr(localRayStart.V[2]);
    poly[1] := 2 * (localRayStart.V[0] * localRayVector.V[0] + hconst *
                   (localRayStart.V[1] - 0.5 * FHeight) * localRayVector.V[1] +
                    localRayStart.V[2]* localRayVector.V[2]);
    poly[2] := Sqr(localRayVector.V[0]) + hconst * Sqr(localRayVector.V[1]) +
               Sqr(localRayVector.V[2]);
    SetLength(roots, 0);
    roots := SolveQuadric(@poly);
    if MinPositiveCoef(roots, minRoot) then
    begin
      t := minRoot;
      ip := VectorCombine(localRayStart, localRayVector, 1, t);
      // check that intersection with infinite cone is within the range we want
      if (ip.V[1] > -FHeight * 0.5) and (ip.V[1] < FHeight * 0.5) then
      begin
        Result := true;
        if Assigned(intersectPoint) then
          intersectPoint^ := LocalToAbsolute(ip);
        if Assigned(intersectNormal) then
        begin
          ip.V[1] := hconst * (ip.V[1] - 0.5 * Height);
          ip.V[3] := 0;
          NormalizeVector(ip);
          intersectNormal^ := LocalToAbsolute(ip);
        end;
      end;
    end;
  end;
end;

// ------------------
// ------------------ TGLCylinder ------------------
// ------------------

// Create
//

constructor TGLCylinder.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FTopRadius := 0.5;
  FParts := [cySides, cyBottom, cyTop];
  FAlignment := caCenter;
end;

// BuildList
//

procedure TGLCylinder.BuildList(var rci: TGLRenderContextInfo);
var
  quadric: PGLUquadricObj;
begin
  GL.PushMatrix;
  quadric := gluNewQuadric;
  SetupQuadricParams(quadric);
  GL.Rotatef(-90, 1, 0, 0);
  case Alignment of
    caTop:
      GL.Translatef(0, 0, -FHeight);
    caBottom:
      ;
  else // caCenter
    GL.Translatef(0, 0, -FHeight * 0.5);
  end;
  if cySides in FParts then
    gluCylinder(quadric, FBottomRadius, FTopRadius, FHeight, FSlices, FStacks);
  if cyTop in FParts then
  begin
    GL.PushMatrix;
    GL.Translatef(0, 0, FHeight);
    gluDisk(quadric, 0, FTopRadius, FSlices, FLoops);
    GL.PopMatrix;
  end;
  if cyBottom in FParts then
  begin
    // swap quadric orientation because top of a disk is defined as outside
    SetInvertedQuadricOrientation(quadric);
    gluDisk(quadric, 0, FBottomRadius, FSlices, FLoops);
  end;
  gluDeleteQuadric(quadric);
  GL.PopMatrix;
end;

// SetTopRadius
//

procedure TGLCylinder.SetTopRadius(const aValue: Single);
begin
  if aValue <> FTopRadius then
  begin
    FTopRadius := aValue;
    StructureChanged;
  end;
end;

// GetTopRadius
//

function TGLCylinder.GetTopRadius: Single;
begin
  Result := FTopRadius;
end;

// SetParts
//

procedure TGLCylinder.SetParts(aValue: TCylinderParts);
begin
  if aValue <> FParts then
  begin
    FParts := aValue;
    StructureChanged;
  end;
end;

// SetAlignment
//

procedure TGLCylinder.SetAlignment(val: TCylinderAlignment);
begin
  if val <> FAlignment then
  begin
    FAlignment := val;
    StructureChanged;
  end;
end;

 
//

procedure TGLCylinder.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLCylinder) then
  begin
    FParts := TGLCylinder(Source).FParts;
    FTopRadius := TGLCylinder(Source).FTopRadius;
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensions
//

function TGLCylinder.AxisAlignedDimensionsUnscaled: TVector;
var
  r, r1: TGLFloat;
begin
  r := Abs(FBottomRadius);
  r1 := Abs(FTopRadius);
  if r1 > r then
    r := r1;
  Result := VectorMake(r, 0.5 * FHeight, r);
  // ScaleVector(Result, Scale.AsVector);
end;

// RayCastIntersect
//

function TGLCylinder.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
const
  cOne: Single = 1;
var
  locRayStart, locRayVector, ip: TVector;
  poly: array [0 .. 2] of Double;
  roots: TDoubleArray;
  minRoot: Double;
  t, tr2, invRayVector1, hTop, hBottom: Single;
  tPlaneMin, tPlaneMax: Single;
begin
  Result := False;
  locRayStart := AbsoluteToLocal(rayStart);
  locRayVector := AbsoluteToLocal(rayVector);

  case Alignment of
    caTop:
      begin
        hTop := 0;
        hBottom := -Height;
      end;
    caBottom:
      begin
        hTop := Height;
        hBottom := 0;
      end;
  else
    // caCenter
    hTop := Height * 0.5;
    hBottom := -hTop;
  end;

  if locRayVector.V[1] = 0 then
  begin
    // intersect if ray shot through the top/bottom planes
    if (locRayStart.V[0] > hTop) or (locRayStart.V[0] < hBottom) then
      Exit;
    tPlaneMin := -1E99;
    tPlaneMax := 1E99;
  end
  else
  begin
    invRayVector1 := cOne / locRayVector.V[1];
    tr2 := Sqr(TopRadius);

    // compute intersection with topPlane
    t := (hTop - locRayStart.V[1]) * invRayVector1;
    if (t > 0) and (cyTop in Parts) then
    begin
      ip.V[0] := locRayStart.V[0] + t * locRayVector.V[0];
      ip.V[2] := locRayStart.V[2] + t * locRayVector.V[2];
      if Sqr(ip.V[0]) + Sqr(ip.V[2]) <= tr2 then
      begin
        // intersect with top plane
        if Assigned(intersectPoint) then
          intersectPoint^ := LocalToAbsolute(VectorMake(ip.V[0], hTop, ip.V[2], 1));
        if Assigned(intersectNormal) then
          intersectNormal^ := LocalToAbsolute(YHmgVector);
        Result := true;
      end;
    end;
    tPlaneMin := t;
    tPlaneMax := t;
    // compute intersection with bottomPlane
    t := (hBottom - locRayStart.V[1]) * invRayVector1;
    if (t > 0) and (cyBottom in Parts) then
    begin
      ip.V[0] := locRayStart.V[0] + t * locRayVector.V[0];
      ip.V[2] := locRayStart.V[2] + t * locRayVector.V[2];
      if (t < tPlaneMin) or (not(cyTop in Parts)) then
      begin
        if Sqr(ip.V[0]) + Sqr(ip.V[2]) <= tr2 then
        begin
          // intersect with top plane
          if Assigned(intersectPoint) then
            intersectPoint^ := LocalToAbsolute(VectorMake(ip.V[0], hBottom,
              ip.V[2], 1));
          if Assigned(intersectNormal) then
            intersectNormal^ := LocalToAbsolute(VectorNegate(YHmgVector));
          Result := true;
        end;
      end;
    end;
    if t < tPlaneMin then
      tPlaneMin := t;
    if t > tPlaneMax then
      tPlaneMax := t;
  end;
  if cySides in Parts then
  begin
    // intersect against cylinder infinite cylinder
    poly[0] := Sqr(locRayStart.V[0]) + Sqr(locRayStart.V[2]) - Sqr(TopRadius);
    poly[1] := 2 * (locRayStart.V[0] * locRayVector.V[0] + locRayStart.V[2] *
      locRayVector.V[2]);
    poly[2] := Sqr(locRayVector.V[0]) + Sqr(locRayVector.V[2]);
    roots := SolveQuadric(@poly);
    if MinPositiveCoef(roots, minRoot) then
    begin
      t := minRoot;
      if (t >= tPlaneMin) and (t < tPlaneMax) then
      begin
        if Assigned(intersectPoint) or Assigned(intersectNormal) then
        begin
          ip := VectorCombine(locRayStart, locRayVector, 1, t);
          if Assigned(intersectPoint) then
            intersectPoint^ := LocalToAbsolute(ip);
          if Assigned(intersectNormal) then
          begin
            ip.V[1] := 0;
            ip.V[3] := 0;
            intersectNormal^ := LocalToAbsolute(ip);
          end;
        end;
        Result := true;
      end;
    end;
  end
  else
    SetLength(roots, 0);
end;

// Align
//

procedure TGLCylinder.Align(const startPoint, endPoint: TVector);
var
  dir: TAffineVector;
begin
  AbsolutePosition := startPoint;
  VectorSubtract(endPoint, startPoint, dir);
  if Parent <> nil then
    dir := Parent.AbsoluteToLocal(dir);
  Up.AsAffineVector := dir;
  Height := VectorLength(dir);
  Lift(Height * 0.5);
  Alignment := caCenter;
end;

// Align
//

procedure TGLCylinder.Align(const startObj, endObj: TGLBaseSceneObject);
begin
  Align(startObj.AbsolutePosition, endObj.AbsolutePosition);
end;

// Align
//

procedure TGLCylinder.Align(const startPoint, endPoint: TAffineVector);
begin
  Align(PointMake(startPoint), PointMake(endPoint));
end;

// ------------------
// ------------------ TGLCapsule ------------------
// ------------------

// Create
//

constructor TGLCapsule.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FHeight := 1;
  FRadius := 0.5;
  FSlices := 4;
  FStacks := 4;
  FParts := [cySides, cyBottom, cyTop];
  FAlignment := caCenter;
end;

// BuildList
//

procedure TGLCapsule.BuildList(var rci: TGLRenderContextInfo);
var
  i, j, n: integer;
  start_nx2: Single;
  start_ny2: Single;
  tmp, nx, ny, nz, start_nx, start_ny, a, ca, sa, l: Single;
  nx2, ny2, nz2: Single;
begin
  GL.PushMatrix;
  GL.Rotatef(-90, 0, 0, 1);
  case Alignment of
    caTop:
      GL.Translatef(0, 0, FHeight + 1);
    caBottom:
      GL.Translatef(0, 0, -FHeight);
  else // caCenter
    GL.Translatef(0, 0, 0.5);
  end;
  n := FSlices * FStacks;
  l := FHeight;
  l := l * 0.5;
  a := (pi * 2.0) / n;
  sa := sin(a);
  ca := cos(a);
  ny := 0;
  nz := 1;
  if cySides in FParts then
  begin
    GL.Begin_(GL_TRIANGLE_STRIP);
    for i := 0 to n do
    begin
      GL.Normal3d(ny, nz, 0);
      GL.TexCoord2f(i / n, 1);
      GL.Vertex3d(ny * FRadius, nz * FRadius, l - 0.5);
      GL.Normal3d(ny, nz, 0);
      GL.TexCoord2f(i / n, 0);
      GL.Vertex3d(ny * FRadius, nz * FRadius, -l - 0.5);
      tmp := ca * ny - sa * nz;
      nz := sa * ny + ca * nz;
      ny := tmp;
    end;
    GL.End_();
  end;
  //
  if cyTop in FParts then
  begin
    start_nx := 0;
    start_ny := 1;
    for j := 0 to (n div FStacks) do
    begin
      start_nx2 := ca * start_nx + sa * start_ny;
      start_ny2 := -sa * start_nx + ca * start_ny;
      nx := start_nx;
      ny := start_ny;
      nz := 0;
      nx2 := start_nx2;
      ny2 := start_ny2;
      nz2 := 0;
      GL.PushMatrix;
      GL.Translatef(0, 0, -0.5);
      GL.Begin_(GL_TRIANGLE_STRIP);
      for i := 0 to n do
      begin
        GL.Normal3d(ny2, nz2, nx2);
        GL.TexCoord2f(i / n, j / n);
        GL.Vertex3d(ny2 * FRadius, nz2 * FRadius, l + nx2 * FRadius);
        GL.Normal3d(ny, nz, nx);
        GL.TexCoord2f(i / n, (j - 1) / n);
        GL.Vertex3d(ny * FRadius, nz * FRadius, l + nx * FRadius);
        tmp := ca * ny - sa * nz;
        nz := sa * ny + ca * nz;
        ny := tmp;
        tmp := ca * ny2 - sa * nz2;
        nz2 := sa * ny2 + ca * nz2;
        ny2 := tmp;
      end;
      GL.End_();
      GL.PopMatrix;
      start_nx := start_nx2;
      start_ny := start_ny2;
    end;
  end;
  //
  if cyBottom in FParts then
  begin
    start_nx := 0;
    start_ny := 1;
    for j := 0 to (n div FStacks) do
    begin
      start_nx2 := ca * start_nx - sa * start_ny;
      start_ny2 := sa * start_nx + ca * start_ny;
      nx := start_nx;
      ny := start_ny;
      nz := 0;
      nx2 := start_nx2;
      ny2 := start_ny2;
      nz2 := 0;
      GL.PushMatrix;
      GL.Translatef(0, 0, -0.5);
      GL.Begin_(GL_TRIANGLE_STRIP);
      for i := 0 to n do
      begin
        GL.Normal3d(ny, nz, nx);
        GL.TexCoord2f(i / n, (j - 1) / n);
        GL.Vertex3d(ny * FRadius, nz * FRadius, -l + nx * FRadius);
        GL.Normal3d(ny2, nz2, nx2);
        GL.TexCoord2f(i / n, j / n);
        GL.Vertex3d(ny2 * FRadius, nz2 * FRadius, -l + nx2 * FRadius);
        tmp := ca * ny - sa * nz;
        nz := sa * ny + ca * nz;
        ny := tmp;
        tmp := ca * ny2 - sa * nz2;
        nz2 := sa * ny2 + ca * nz2;
        ny2 := tmp;
      end;
      GL.End_();
      GL.PopMatrix;
      start_nx := start_nx2;
      start_ny := start_ny2;
    end;
  end;
  GL.PopMatrix;
end;

// SetLength
//

procedure TGLCapsule.SetHeight(const aValue: Single);
begin
  if aValue <> FHeight then
  begin
    FHeight := aValue;
    StructureChanged;
  end;
end;

// SetRadius
//

procedure TGLCapsule.SetRadius(const aValue: Single);
begin
  if aValue <> FRadius then
  begin
    FRadius := aValue;
    StructureChanged;
  end;
end;

// SetSlices
//

procedure TGLCapsule.SetSlices(const aValue: integer);
begin
  if aValue <> FSlices then
  begin
    FSlices := aValue;
    StructureChanged;
  end;
end;

// SetStacks
//

procedure TGLCapsule.SetStacks(const aValue: integer);
begin
  if aValue <> FStacks then
  begin
    FStacks := aValue;
    StructureChanged;
  end;
end;

// SetParts
//

procedure TGLCapsule.SetParts(aValue: TCylinderParts);
begin
  if aValue <> FParts then
  begin
    FParts := aValue;
    StructureChanged;
  end;
end;

// SetAlignment
//

procedure TGLCapsule.SetAlignment(val: TCylinderAlignment);
begin
  if val <> FAlignment then
  begin
    FAlignment := val;
    StructureChanged;
  end;
end;

 
//

procedure TGLCapsule.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLCapsule) then
  begin
    FParts := TGLCapsule(Source).FParts;
    FRadius := TGLCapsule(Source).FRadius;
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensions
//

function TGLCapsule.AxisAlignedDimensionsUnscaled: TVector;
var
  r, r1: TGLFloat;
begin
  r := Abs(FRadius);
  r1 := Abs(FRadius);
  if r1 > r then
    r := r1;
  Result := VectorMake(r, 0.5 * FHeight, r);
  // ScaleVector(Result, Scale.AsVector);
end;

// RayCastIntersect
//

function TGLCapsule.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
const
  cOne: Single = 1;
var
  locRayStart, locRayVector, ip: TVector;
  poly: array [0 .. 2] of Double;
  roots: TDoubleArray;
  minRoot: Double;
  t, tr2, invRayVector1, hTop, hBottom: Single;
  tPlaneMin, tPlaneMax: Single;
begin
  Result := False;
  locRayStart := AbsoluteToLocal(rayStart);
  locRayVector := AbsoluteToLocal(rayVector);

  case Alignment of
    caTop:
      begin
        hTop := 0;
        hBottom := -FHeight;
      end;
    caBottom:
      begin
        hTop := FHeight;
        hBottom := 0;
      end;
  else
    // caCenter
    hTop := FHeight * 0.5;
    hBottom := -hTop;
  end;

  if locRayVector.V[1] = 0 then
  begin
    // intersect if ray shot through the top/bottom planes
    if (locRayStart.V[0] > hTop) or (locRayStart.V[0] < hBottom) then
      Exit;
    tPlaneMin := -1E99;
    tPlaneMax := 1E99;
  end
  else
  begin
    invRayVector1 := cOne / locRayVector.V[1];
    tr2 := Sqr(Radius);

    // compute intersection with topPlane
    t := (hTop - locRayStart.V[1]) * invRayVector1;
    if (t > 0) and (cyTop in Parts) then
    begin
      ip.V[0] := locRayStart.V[0] + t * locRayVector.V[0];
      ip.V[2] := locRayStart.V[2] + t * locRayVector.V[2];
      if Sqr(ip.V[0]) + Sqr(ip.V[2]) <= tr2 then
      begin
        // intersect with top plane
        if Assigned(intersectPoint) then
          intersectPoint^ := LocalToAbsolute(VectorMake(ip.V[0], hTop, ip.V[2], 1));
        if Assigned(intersectNormal) then
          intersectNormal^ := LocalToAbsolute(YHmgVector);
        Result := true;
      end;
    end;
    tPlaneMin := t;
    tPlaneMax := t;
    // compute intersection with bottomPlane
    t := (hBottom - locRayStart.V[1]) * invRayVector1;
    if (t > 0) and (cyBottom in Parts) then
    begin
      ip.V[0] := locRayStart.V[0] + t * locRayVector.V[0];
      ip.V[2] := locRayStart.V[2] + t * locRayVector.V[2];
      if (t < tPlaneMin) or (not(cyTop in Parts)) then
      begin
        if Sqr(ip.V[0]) + Sqr(ip.V[2]) <= tr2 then
        begin
          // intersect with top plane
          if Assigned(intersectPoint) then
            intersectPoint^ := LocalToAbsolute(VectorMake(ip.V[0], hBottom,
              ip.V[2], 1));
          if Assigned(intersectNormal) then
            intersectNormal^ := LocalToAbsolute(VectorNegate(YHmgVector));
          Result := true;
        end;
      end;
    end;
    if t < tPlaneMin then
      tPlaneMin := t;
    if t > tPlaneMax then
      tPlaneMax := t;
  end;
  if cySides in Parts then
  begin
    // intersect against cylinder infinite cylinder
    poly[0] := Sqr(locRayStart.V[0]) + Sqr(locRayStart.V[2]) - Sqr(Radius);
    poly[1] := 2 * (locRayStart.V[0] * locRayVector.V[0] +
                    locRayStart.V[2] * locRayVector.V[2]);
    poly[2] := Sqr(locRayVector.V[0]) + Sqr(locRayVector.V[2]);
    roots := SolveQuadric(@poly);
    if MinPositiveCoef(roots, minRoot) then
    begin
      t := minRoot;
      if (t >= tPlaneMin) and (t < tPlaneMax) then
      begin
        if Assigned(intersectPoint) or Assigned(intersectNormal) then
        begin
          ip := VectorCombine(locRayStart, locRayVector, 1, t);
          if Assigned(intersectPoint) then
            intersectPoint^ := LocalToAbsolute(ip);
          if Assigned(intersectNormal) then
          begin
            ip.V[1] := 0;
            ip.V[3] := 0;
            intersectNormal^ := LocalToAbsolute(ip);
          end;
        end;
        Result := true;
      end;
    end;
  end
  else
    SetLength(roots, 0);
end;

// Align
//

procedure TGLCapsule.Align(const startPoint, endPoint: TVector);
var
  dir: TAffineVector;
begin
  AbsolutePosition := startPoint;
  VectorSubtract(endPoint, startPoint, dir);
  if Parent <> nil then
    dir := Parent.AbsoluteToLocal(dir);
  Up.AsAffineVector := dir;
  FHeight := VectorLength(dir);
  Lift(FHeight * 0.5);
  Alignment := caCenter;
end;

// Align
//

procedure TGLCapsule.Align(const startObj, endObj: TGLBaseSceneObject);
begin
  Align(startObj.AbsolutePosition, endObj.AbsolutePosition);
end;

// Align
//

procedure TGLCapsule.Align(const startPoint, endPoint: TAffineVector);
begin
  Align(PointMake(startPoint), PointMake(endPoint));
end;

// ------------------
// ------------------ TGLAnnulus ------------------
// ------------------

// Create
//

constructor TGLAnnulus.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FBottomInnerRadius := 0.3;
  FTopInnerRadius := 0.3;
  FTopRadius := 0.5;
  FParts := [anInnerSides, anOuterSides, anBottom, anTop];
end;

// SetBottomInnerRadius
//

procedure TGLAnnulus.SetBottomInnerRadius(const aValue: Single);
begin
  if aValue <> FBottomInnerRadius then
  begin
    FBottomInnerRadius := aValue;
    StructureChanged;
  end;
end;

// SetTopRadius
//

procedure TGLAnnulus.SetTopRadius(const aValue: Single);
begin
  if aValue <> FTopRadius then
  begin
    FTopRadius := aValue;
    StructureChanged;
  end;
end;

// SetTopInnerRadius
//

procedure TGLAnnulus.SetTopInnerRadius(const aValue: Single);
begin
  if aValue <> FTopInnerRadius then
  begin
    FTopInnerRadius := aValue;
    StructureChanged;
  end;
end;

// SetParts
//

procedure TGLAnnulus.SetParts(aValue: TAnnulusParts);
begin
  if aValue <> FParts then
  begin
    FParts := aValue;
    StructureChanged;
  end;
end;

// BuildList
//

procedure TGLAnnulus.BuildList(var rci: TGLRenderContextInfo);
var
  quadric: PGLUquadricObj;
begin
  GL.PushMatrix;
  quadric := gluNewQuadric;
  SetupQuadricParams(quadric);
  GL.Rotatef(-90, 1, 0, 0);
  GL.Translatef(0, 0, -FHeight * 0.5);
  if anOuterSides in FParts then
    gluCylinder(quadric, FBottomRadius, FTopRadius, FHeight, FSlices, FStacks);
  if anTop in FParts then
  begin
    GL.PushMatrix;
    GL.Translatef(0, 0, FHeight);
    gluDisk(quadric, FTopInnerRadius, FTopRadius, FSlices, FLoops);
    GL.PopMatrix;
  end;
  if [anBottom, anInnerSides] * FParts <> [] then
  begin
    // swap quadric orientation because top of a disk is defined as outside
    SetInvertedQuadricOrientation(quadric);
    if anBottom in FParts then
      gluDisk(quadric, FBottomInnerRadius, FBottomRadius, FSlices, FLoops);
    if anInnerSides in FParts then
      gluCylinder(quadric, FBottomInnerRadius, FTopInnerRadius, FHeight,
        FSlices, FStacks);
  end;
  gluDeleteQuadric(quadric);
  GL.PopMatrix;
end;

 
//

procedure TGLAnnulus.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLAnnulus) then
  begin
    FParts := TGLAnnulus(Source).FParts;
    FTopRadius := TGLAnnulus(Source).FTopRadius;
    FTopInnerRadius := TGLAnnulus(Source).FTopInnerRadius;
    FBottomRadius := TGLAnnulus(Source).FBottomRadius;
    FBottomInnerRadius := TGLAnnulus(Source).FBottomInnerRadius;
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensions
//

function TGLAnnulus.AxisAlignedDimensionsUnscaled: TVector;
var
  r, r1: TGLFloat;
begin
  r := Abs(FBottomRadius);
  r1 := Abs(FTopRadius);
  if r1 > r then
    r := r1;
  Result := VectorMake(r, 0.5 * FHeight, r);
end;

// RayCastIntersect
//

function TGLAnnulus.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint, intersectNormal: PVector): Boolean;
const
  cOne: Single = 1;
var
  locRayStart, locRayVector, ip: TVector;
  poly: array [0 .. 2] of Double;
  t, tr2, invRayVector1: Single;
  tPlaneMin, tPlaneMax: Single;
  tir2, d2: Single;
  Root: Double;
  roots, tmpRoots: TDoubleArray;
  FirstIntersected: Boolean;
  h1, h2, hTop, hBot: Single;
  Draw1, Draw2: Boolean;
begin
  Result := False;
  FirstIntersected := False;
  SetLength(tmpRoots, 0);
  locRayStart := AbsoluteToLocal(rayStart);
  locRayVector := AbsoluteToLocal(rayVector);

  hTop := Height * 0.5;
  hBot := -hTop;
  if locRayVector.V[1] < 0 then
  begin // Sort the planes according to the direction of view
    h1 := hTop; // Height of the 1st plane
    h2 := hBot; // Height of the 2nd plane
    Draw1 := (anTop in Parts); // 1st "cap" Must be drawn?
    Draw2 := (anBottom in Parts);
  end
  else
  begin
    h1 := hBot;
    h2 := hTop;
    Draw1 := (anBottom in Parts);
    Draw2 := (anTop in Parts);
  end; // if

  if locRayVector.V[1] = 0 then
  begin
    // intersect if ray shot through the top/bottom planes
    if (locRayStart.V[0] > hTop) or (locRayStart.V[0] < hBot) then
      Exit;
    tPlaneMin := -1E99;
    tPlaneMax := 1E99;
  end
  else
  begin
    invRayVector1 := cOne / locRayVector.V[1];
    tr2 := Sqr(TopRadius);
    tir2 := Sqr(TopInnerRadius);
    FirstIntersected := False;

    // compute intersection with first plane
    t := (h1 - locRayStart.V[1]) * invRayVector1;
    if (t > 0) and Draw1 then
    begin
      ip.V[0] := locRayStart.V[0] + t * locRayVector.V[0];
      ip.V[2] := locRayStart.V[2] + t * locRayVector.V[2];
      d2 := Sqr(ip.V[0]) + Sqr(ip.V[2]);
      if (d2 <= tr2) and (d2 >= tir2) then
      begin
        // intersect with top plane
        FirstIntersected := true;
        if Assigned(intersectPoint) then
          intersectPoint^ := LocalToAbsolute(VectorMake(ip.V[0], h1, ip.V[2], 1));
        if Assigned(intersectNormal) then
          intersectNormal^ := LocalToAbsolute(YHmgVector);
        Result := true;
      end;
    end;
    tPlaneMin := t;
    tPlaneMax := t;

    // compute intersection with second plane
    t := (h2 - locRayStart.V[1]) * invRayVector1;
    if (t > 0) and Draw2 then
    begin
      ip.V[0] := locRayStart.V[0] + t * locRayVector.V[0];
      ip.V[2] := locRayStart.V[2] + t * locRayVector.V[2];
      d2 := Sqr(ip.V[0]) + Sqr(ip.V[2]);
      if (t < tPlaneMin) or (not FirstIntersected) then
      begin
        if (d2 <= tr2) and (d2 >= tir2) then
        begin
          // intersect with top plane
          if Assigned(intersectPoint) then
            intersectPoint^ := LocalToAbsolute(VectorMake(ip.V[0], h2, ip.V[2], 1));
          if Assigned(intersectNormal) then
            intersectNormal^ := LocalToAbsolute(VectorNegate(YHmgVector));
          Result := true;
        end;
      end;
    end;
    if t < tPlaneMin then
    begin
      tPlaneMin := t;
    end; // if
    if t > tPlaneMax then
      tPlaneMax := t;
  end;

  try
    SetLength(roots, 4);
    roots[0] := -1;
    roots[1] := -1;
    roots[2] := -1;
    roots[3] := -1; // By default, side is behind rayStart

    { Compute roots for outer cylinder }
    if anOuterSides in Parts then
    begin
      // intersect against infinite cylinder, will be cut by tPlaneMine and tPlaneMax
      poly[0] := Sqr(locRayStart.V[0]) + Sqr(locRayStart.V[2]) - Sqr(TopRadius);
      poly[1] := 2 * (locRayStart.V[0] * locRayVector.V[0] + locRayStart.V[2] *
        locRayVector.V[2]);
      poly[2] := Sqr(locRayVector.V[0]) + Sqr(locRayVector.V[2]);
      tmpRoots := SolveQuadric(@poly);
      // Intersect coordinates on rayVector (rayStart=0)
      if ( High(tmpRoots) >= 0) and // Does root exist?
        ((tmpRoots[0] > tPlaneMin) and not FirstIntersected) and
      // In the annulus and not masked by first cap
        ((tmpRoots[0] < tPlaneMax)) { // In the annulus } then
        roots[0] := tmpRoots[0];
      if ( High(tmpRoots) >= 1) and
        ((tmpRoots[1] > tPlaneMin) and not FirstIntersected) and
        ((tmpRoots[1] < tPlaneMax)) then
        roots[1] := tmpRoots[1];
    end; // if

    { Compute roots for inner cylinder }
    if anInnerSides in Parts then
    begin
      // intersect against infinite cylinder
      poly[0] := Sqr(locRayStart.V[0]) +
                 Sqr(locRayStart.V[2]) - Sqr(TopInnerRadius);
      poly[1] := 2 * (locRayStart.V[0] * locRayVector.V[0] +
                 locRayStart.V[2] * locRayVector.V[2]);
      poly[2] := Sqr(locRayVector.V[0]) + Sqr(locRayVector.V[2]);
                 tmpRoots := SolveQuadric(@poly);
      if ( High(tmpRoots) >= 0) and
        ((tmpRoots[0] > tPlaneMin) and not FirstIntersected) and
        ((tmpRoots[0] < tPlaneMax)) then
        roots[2] := tmpRoots[0];
      if ( High(tmpRoots) >= 1) and
        ((tmpRoots[1] > tPlaneMin) and not FirstIntersected) and
        ((tmpRoots[1] < tPlaneMax)) then
        roots[3] := tmpRoots[1];
    end; // if

    { Find the first intersection point and compute its coordinates and normal }
    if MinPositiveCoef(roots, Root) then
    begin
      t := Root;
      if (t >= tPlaneMin) and (t < tPlaneMax) then
      begin
        if Assigned(intersectPoint) or Assigned(intersectNormal) then
        begin
          ip := VectorCombine(locRayStart, locRayVector, 1, t);
          if Assigned(intersectPoint) then
            intersectPoint^ := LocalToAbsolute(ip);
          if Assigned(intersectNormal) then
          begin
            ip.V[1] := 0;
            ip.V[3] := 0;
            intersectNormal^ := LocalToAbsolute(ip);
          end;
        end;
        Result := true;
      end;
    end;

  finally
    roots := nil;
    tmpRoots := nil;
  end; // finally
end;

// ------------------
// ------------------ TGLTorus ------------------
// ------------------

// Create
//

constructor TGLTorus.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FRings := 25;
  FSides := 15;
  FMinorRadius := 0.1;
  FMajorRadius := 0.4;
  FStartAngle := 0.0;
  FStopAngle := 360.0;
  FParts := [toSides, toStartDisk, toStopDisk];
end;

// BuildList
//

procedure TGLTorus.BuildList(var rci: TGLRenderContextInfo);

  procedure EmitVertex(ptr: PVertexRec; L1, L2: integer);
  begin
    XGL.TexCoord2fv(@ptr^.TexCoord);
    with GL do
    begin
      Normal3fv(@ptr^.Normal);
      if L1 > -1 then
        VertexAttrib3fv(L1, @ptr.Tangent);
      if L2 > -1 then
        VertexAttrib3fv(L2, @ptr.Binormal);
      Vertex3fv(@ptr^.Position);
    end;
  end;

var
  i, j: integer;
  Theta, Phi, Theta1, cosPhi, sinPhi, dist: TGLFloat;
  cosTheta1, sinTheta1: TGLFloat;
  ringDelta, sideDelta: TGLFloat;
  ringDir: TAffineVector;
  iFact, jFact: Single;
  pVertex: PVertexRec;
  TanLoc, BinLoc: TGLInt;
  MeshSize: integer;
  MeshIndex: integer;
  Vertex: TVertexRec;
begin
  if FMesh = nil then
  begin
    MeshSize := 0;
    MeshIndex := 0;
    if toStartDisk in FParts then
      MeshSize := MeshSize + 1;
    if toStopDisk in FParts then
      MeshSize := MeshSize + 1;
    if toSides in FParts then
      MeshSize := MeshSize + Integer(FRings) + 1;
    SetLength(FMesh, MeshSize);
    // handle texture generation
    ringDelta := ((FStopAngle - FStartAngle) / 360) * c2PI / FRings;
    sideDelta := c2PI / FSides;

    iFact := 1 / FRings;
    jFact := 1 / FSides;
    if toSides in FParts then
    begin
      Theta := DegToRad(FStartAngle) - ringDelta;
      for i := FRings downto 0 do
      begin
        SetLength(FMesh[i], FSides + 1);
        Theta1 := Theta + ringDelta;
        SinCos(Theta1, sinTheta1, cosTheta1);
        Phi := 0;
        for j := FSides downto 0 do
        begin
          Phi := Phi + sideDelta;
          SinCos(Phi, sinPhi, cosPhi);
          dist := FMajorRadius + FMinorRadius * cosPhi;

          FMesh[i][j].Position := Vector3fMake(cosTheta1 * dist,
            -sinTheta1 * dist, FMinorRadius * sinPhi);
          ringDir := FMesh[i][j].Position;
          ringDir.V[2] := 0.0;
          NormalizeVector(ringDir);
          FMesh[i][j].Normal := Vector3fMake(cosTheta1 * cosPhi,
            -sinTheta1 * cosPhi, sinPhi);
          FMesh[i][j].Tangent := VectorCrossProduct(ZVector, ringDir);
          FMesh[i][j].Binormal := VectorCrossProduct(FMesh[i][j].Normal,
            FMesh[i][j].Tangent);
          FMesh[i][j].TexCoord := Vector2fMake(i * iFact, j * jFact);
        end;
        Theta := Theta1;
      end;
      MeshIndex := FRings + 1;
    end;

    if toStartDisk in FParts then
    begin
      SetLength(FMesh[MeshIndex], FSides + 1);
      Theta1 := DegToRad(FStartAngle);
      SinCos(Theta1, sinTheta1, cosTheta1);

      if toSides in FParts then
      begin
        for j := FSides downto 0 do
        begin
          FMesh[MeshIndex][j].Position := FMesh[MeshIndex - 1][j].Position;
          FMesh[MeshIndex][j].Normal := FMesh[MeshIndex - 1][j].Tangent;
          FMesh[MeshIndex][j].Tangent := FMesh[MeshIndex - 1][j].Position;
          FMesh[MeshIndex][j].Tangent.V[2] := 0;
          FMesh[MeshIndex][j].Binormal := ZVector;
          FMesh[MeshIndex][j].TexCoord := FMesh[MeshIndex - 1][j].TexCoord;
          FMesh[MeshIndex][j].TexCoord.V[0] := 0;
        end;
      end
      else
      begin
        Phi := 0;
        for j := FSides downto 0 do
        begin
          Phi := Phi + sideDelta;
          SinCos(Phi, sinPhi, cosPhi);
          dist := FMajorRadius + FMinorRadius * cosPhi;
          FMesh[MeshIndex][j].Position := Vector3fMake(cosTheta1 * dist,
            -sinTheta1 * dist, FMinorRadius * sinPhi);
          ringDir := FMesh[MeshIndex][j].Position;
          ringDir.V[2] := 0.0;
          NormalizeVector(ringDir);
          FMesh[MeshIndex][j].Normal := VectorCrossProduct(ZVector, ringDir);
          FMesh[MeshIndex][j].Tangent := ringDir;
          FMesh[MeshIndex][j].Binormal := ZVector;
          FMesh[MeshIndex][j].TexCoord := Vector2fMake(0, j * jFact);
        end;
      end;
      Vertex.Position := Vector3fMake(cosTheta1 * FMajorRadius,
        -sinTheta1 * FMajorRadius, 0);
      Vertex.Normal := FMesh[MeshIndex][0].Normal;
      Vertex.Tangent := FMesh[MeshIndex][0].Tangent;
      Vertex.Binormal := FMesh[MeshIndex][0].Binormal;
      Vertex.TexCoord := Vector2fMake(1, 1);
      MeshIndex := MeshIndex + 1;
    end;

    if toStopDisk in FParts then
    begin
      SetLength(FMesh[MeshIndex], FSides + 1);
      Theta1 := DegToRad(FStopAngle);
      SinCos(Theta1, sinTheta1, cosTheta1);

      if toSides in FParts then
      begin
        for j := FSides downto 0 do
        begin
          FMesh[MeshIndex][j].Position := FMesh[0][j].Position;
          FMesh[MeshIndex][j].Normal := VectorNegate(FMesh[0][j].Tangent);
          FMesh[MeshIndex][j].Tangent := FMesh[0][j].Position;
          FMesh[MeshIndex][j].Tangent.V[2] := 0;
          FMesh[MeshIndex][j].Binormal := VectorNegate(ZVector);
          FMesh[MeshIndex][j].TexCoord := FMesh[0][j].TexCoord;
          FMesh[MeshIndex][j].TexCoord.V[0] := 1;
        end;
      end
      else
      begin
        Phi := 0;
        for j := FSides downto 0 do
        begin
          Phi := Phi + sideDelta;
          SinCos(Phi, sinPhi, cosPhi);
          dist := FMajorRadius + FMinorRadius * cosPhi;
          FMesh[MeshIndex][j].Position := Vector3fMake(cosTheta1 * dist,
            -sinTheta1 * dist, FMinorRadius * sinPhi);
          ringDir := FMesh[MeshIndex][j].Position;
          ringDir.V[2] := 0.0;
          NormalizeVector(ringDir);
          FMesh[MeshIndex][j].Normal := VectorCrossProduct(ringDir, ZVector);
          FMesh[MeshIndex][j].Tangent := ringDir;
          FMesh[MeshIndex][j].Binormal := VectorNegate(ZVector);
          FMesh[MeshIndex][j].TexCoord := Vector2fMake(1, j * jFact);
        end;
      end;
      Vertex.Position := Vector3fMake(cosTheta1 * FMajorRadius,
        -sinTheta1 * FMajorRadius, 0);
      Vertex.Normal := FMesh[MeshIndex][0].Normal;
      Vertex.Tangent := FMesh[MeshIndex][0].Tangent;
      Vertex.Binormal := FMesh[MeshIndex][0].Binormal;
      Vertex.TexCoord := Vector2fMake(0, 0);
    end;
  end;

  with GL do
  begin
    if ARB_shader_objects and (rci.GLStates.CurrentProgram > 0) then
    begin
      TanLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
        PGLChar(TangentAttributeName));
      BinLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
        PGLChar(BinormalAttributeName));
    end
    else
    begin
      TanLoc := -1;
      BinLoc := TanLoc;
    end;

    MeshIndex := 0;

    if toSides in FParts then
    begin
      Begin_(GL_TRIANGLES);
      for i := FRings - 1 downto 0 do
        for j := FSides - 1 downto 0 do
        begin
          pVertex := @FMesh[i][j];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[i][j + 1];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[i + 1][j];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[i + 1][j + 1];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[i + 1][j];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[i][j + 1];
          EmitVertex(pVertex, TanLoc, BinLoc);
        end;
      End_;
      MeshIndex := FRings + 1;
    end;

    if toStartDisk in FParts then
    begin
      Begin_(GL_TRIANGLE_FAN);
      pVertex := @Vertex;
      EmitVertex(pVertex, TanLoc, BinLoc);
      for j := 0 to FSides do
      begin
        pVertex := @FMesh[MeshIndex][j];
        EmitVertex(pVertex, TanLoc, BinLoc);
      end;
      End_;
      MeshIndex := MeshIndex + 1;
    end;

    if toStopDisk in FParts then
    begin
      Begin_(GL_TRIANGLE_FAN);
      pVertex := @Vertex;
      EmitVertex(pVertex, TanLoc, BinLoc);
      for j := FSides downto 0 do
      begin
        pVertex := @FMesh[MeshIndex][j];
        EmitVertex(pVertex, TanLoc, BinLoc);
      end;
      End_;
    end;

  end;
end;

// SetMajorRadius
//

procedure TGLTorus.SetMajorRadius(const aValue: Single);
begin
  if FMajorRadius <> aValue then
  begin
    FMajorRadius := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetMinorRadius
//

procedure TGLTorus.SetMinorRadius(const aValue: Single);
begin
  if FMinorRadius <> aValue then
  begin
    FMinorRadius := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetRings
//

procedure TGLTorus.SetRings(aValue: Cardinal);
begin
  if FRings <> aValue then
  begin
    FRings := aValue;
    if FRings < 2 then
      FRings := 2;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetSides
//

procedure TGLTorus.SetSides(aValue: Cardinal);
begin
  if FSides <> aValue then
  begin
    FSides := aValue;
    if FSides < 3 then
      FSides := 3;
    FMesh := nil;
    StructureChanged;
  end;
end;

procedure TGLTorus.SetStartAngle(const aValue: Single);
begin
  if FStartAngle <> aValue then
  begin
    FStartAngle := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

procedure TGLTorus.SetStopAngle(const aValue: Single);
begin
  if FStopAngle <> aValue then
  begin
    FStopAngle := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

procedure TGLTorus.SetParts(aValue: TTorusParts);
begin
  if aValue <> FParts then
  begin
    FParts := aValue;
    StructureChanged;
  end;
end;

// AxisAlignedDimensionsUnscaled
//

function TGLTorus.AxisAlignedDimensionsUnscaled: TVector;
var
  r, r1: TGLFloat;
begin
  r := Abs(FMajorRadius);
  r1 := Abs(FMinorRadius);
  Result := VectorMake(r + r1, r + r1, r1); // Danb
end;

// RayCastIntersect
//

function TGLTorus.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
var
  i: integer;
  fRo2, fRi2, fDE, fVal, r, nearest: Double;
  polynom: array [0 .. 4] of Double;
  polyRoots: TDoubleArray;
  localStart, localVector: TVector;
  vi, vc: TVector;
begin
  // compute coefficients of quartic polynomial
  fRo2 := Sqr(MajorRadius);
  fRi2 := Sqr(MinorRadius);
  localStart := AbsoluteToLocal(rayStart);
  localVector := AbsoluteToLocal(rayVector);
  NormalizeVector(localVector);
  fDE := VectorDotProduct(localStart, localVector);
  fVal := VectorNorm(localStart) - (fRo2 + fRi2);

  polynom[0] := Sqr(fVal) - 4.0 * fRo2 * (fRi2 - Sqr(localStart.V[2]));
  polynom[1] := 4.0 * fDE * fVal + 8.0 * fRo2 * localVector.V[2] * localStart.V[2];
  polynom[2] := 2.0 * fVal + 4.0 * Sqr(fDE) + 4.0 * fRo2 * Sqr(localVector.V[2]);
  polynom[3] := 4.0 * fDE;
  polynom[4] := 1;

  // solve the quartic
  polyRoots := SolveQuartic(@polynom[0]);

  // search for closest point
  Result := (Length(polyRoots) > 0);
  if Result then
  begin
    nearest := 1E20;
    for i := 0 to High(polyRoots) do
    begin
      r := polyRoots[i];
      if (r > 0) and (r < nearest) then
      begin
        nearest := r;
        Result := true;
      end;
    end;
    vi := VectorCombine(localStart, localVector, 1, nearest);
    if Assigned(intersectPoint) then
      SetVector(intersectPoint^, LocalToAbsolute(vi));
    if Assigned(intersectNormal) then
    begin
      // project vi on local torus plane
      vc.V[0] := vi.V[0];
      vc.V[1] := vi.V[1];
      vc.V[2] := 0;
      // project vc on MajorRadius circle
      ScaleVector(vc, MajorRadius / (VectorLength(vc) + 0.000001));
      // calculate circle to intersect vector (gives normal);
      SubtractVector(vi, vc);
      // return to absolute coordinates and normalize
      vi.V[3] := 0;
      SetVector(intersectNormal^, LocalToAbsolute(vi));
    end;
  end;
end;

// ------------------
// ------------------ TGLArrowLine ------------------
// ------------------

// Create
//

constructor TGLArrowLine.Create(AOwner: TComponent);
begin
  inherited;
  FTopRadius := 0.1;
  BottomRadius := 0.1;
  fTopArrowHeadRadius := 0.2;
  fTopArrowHeadHeight := 0.5;
  fBottomArrowHeadRadius := 0.2;
  fBottomArrowHeadHeight := 0.5;
  FHeadStackingStyle := ahssStacked;
  { by default there is not much point having the top of the line (cylinder)
    showing as it is coincidental with the Toparrowhead bottom.
    Note I've defaulted to "vector" type arrows (arrow head on top only }
  FParts := [alLine, alTopArrow];
end;

// SetTopRadius
//

procedure TGLArrowLine.SetTopRadius(const aValue: Single);
begin
  if aValue <> FTopRadius then
  begin
    FTopRadius := aValue;
    StructureChanged;
  end;
end;

// SetTopArrowHeadHeight
//

procedure TGLArrowLine.SetTopArrowHeadHeight(const aValue: Single);
begin
  if aValue <> fTopArrowHeadHeight then
  begin
    fTopArrowHeadHeight := aValue;
    StructureChanged;
  end;
end;

// SetTopArrowHeadRadius
//

procedure TGLArrowLine.SetTopArrowHeadRadius(const aValue: Single);
begin
  if aValue <> fTopArrowHeadRadius then
  begin
    fTopArrowHeadRadius := aValue;
    StructureChanged;
  end;
end;

// SetBottomArrowHeadHeight
//

procedure TGLArrowLine.SetBottomArrowHeadHeight(const aValue: Single);
begin
  if aValue <> fBottomArrowHeadHeight then
  begin
    fBottomArrowHeadHeight := aValue;
    StructureChanged;
  end;
end;

// SetBottomArrowHeadRadius
//

procedure TGLArrowLine.SetBottomArrowHeadRadius(const aValue: Single);
begin
  if aValue <> fBottomArrowHeadRadius then
  begin
    fBottomArrowHeadRadius := aValue;
    StructureChanged;
  end;
end;

// SetParts
//

procedure TGLArrowLine.SetParts(aValue: TArrowLineParts);
begin
  if aValue <> FParts then
  begin
    FParts := aValue;
    StructureChanged;
  end;
end;

// SetHeadStackingStyle
//

procedure TGLArrowLine.SetHeadStackingStyle(const val: TArrowHeadStackingStyle);
begin
  if val <> FHeadStackingStyle then
  begin
    FHeadStackingStyle := val;
    StructureChanged;
  end;
end;

// BuildList
//

procedure TGLArrowLine.BuildList(var rci: TGLRenderContextInfo);
var
  quadric: PGLUquadricObj;
  cylHeight, cylOffset, headInfluence: Single;
begin
  case HeadStackingStyle of
    ahssCentered:
      headInfluence := 0.5;
    ahssIncluded:
      headInfluence := 1;
  else // ahssStacked
    headInfluence := 0;
  end;
  cylHeight := Height;
  cylOffset := -FHeight * 0.5;
  // create a new quadric
  quadric := gluNewQuadric;
  SetupQuadricParams(quadric);
  // does the top arrow part - the cone
  if alTopArrow in Parts then
  begin
    cylHeight := cylHeight - TopArrowHeadHeight * headInfluence;
    GL.PushMatrix;
    GL.Translatef(0, 0, Height * 0.5 - TopArrowHeadHeight * headInfluence);
    gluCylinder(quadric, fTopArrowHeadRadius, 0, fTopArrowHeadHeight,
      Slices, Stacks);
    // top of a disk is defined as outside
    SetInvertedQuadricOrientation(quadric);
    if alLine in Parts then
      gluDisk(quadric, FTopRadius, fTopArrowHeadRadius, Slices, FLoops)
    else
      gluDisk(quadric, 0, fTopArrowHeadRadius, Slices, FLoops);
    GL.PopMatrix;
  end;
  // does the bottom arrow part - another cone
  if alBottomArrow in Parts then
  begin
    cylHeight := cylHeight - BottomArrowHeadHeight * headInfluence;
    cylOffset := cylOffset + BottomArrowHeadHeight * headInfluence;
    GL.PushMatrix;
    // make the bottom arrow point in the other direction
    GL.Rotatef(180, 1, 0, 0);
    GL.Translatef(0, 0, Height * 0.5 - BottomArrowHeadHeight * headInfluence);
    SetNormalQuadricOrientation(quadric);
    gluCylinder(quadric, fBottomArrowHeadRadius, 0, fBottomArrowHeadHeight,
      Slices, Stacks);
    // top of a disk is defined as outside
    SetInvertedQuadricOrientation(quadric);
    if alLine in Parts then
      gluDisk(quadric, FBottomRadius, fBottomArrowHeadRadius, Slices, FLoops)
    else
      gluDisk(quadric, 0, fBottomArrowHeadRadius, Slices, FLoops);
    GL.PopMatrix;
  end;
  // does the cylinder that makes the line
  if (cylHeight > 0) and (alLine in Parts) then
  begin
    GL.PushMatrix;
    GL.Translatef(0, 0, cylOffset);
    SetNormalQuadricOrientation(quadric);
    gluCylinder(quadric, FBottomRadius, FTopRadius, cylHeight, FSlices,
      FStacks);
    if not(alTopArrow in Parts) then
    begin
      GL.PushMatrix;
      GL.Translatef(0, 0, cylHeight);
      gluDisk(quadric, 0, FTopRadius, FSlices, FLoops);
      GL.PopMatrix;
    end;
    if not(alBottomArrow in Parts) then
    begin
      // swap quadric orientation because top of a disk is defined as outside
      SetInvertedQuadricOrientation(quadric);
      gluDisk(quadric, 0, FBottomRadius, FSlices, FLoops);
    end;
    GL.PopMatrix;
  end;
  gluDeleteQuadric(quadric);
end;

 
//

procedure TGLArrowLine.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLArrowLine) then
  begin
    FParts := TGLArrowLine(Source).FParts;
    FTopRadius := TGLArrowLine(Source).FTopRadius;
    fTopArrowHeadHeight := TGLArrowLine(Source).fTopArrowHeadHeight;
    fTopArrowHeadRadius := TGLArrowLine(Source).fTopArrowHeadRadius;
    fBottomArrowHeadHeight := TGLArrowLine(Source).fBottomArrowHeadHeight;
    fBottomArrowHeadRadius := TGLArrowLine(Source).fBottomArrowHeadRadius;
    FHeadStackingStyle := TGLArrowLine(Source).FHeadStackingStyle;
  end;
  inherited Assign(Source);
end;

// ------------------
// ------------------ TGLArrowArc ------------------
// ------------------

// Create
//

constructor TGLArrowArc.Create(AOwner: TComponent);
begin
  inherited;
  FStacks := 16;
  fArcRadius := 0.5;
  FStartAngle := 0;
  FStopAngle := 360;
  FTopRadius := 0.1;
  BottomRadius := 0.1;
  fTopArrowHeadRadius := 0.2;
  fTopArrowHeadHeight := 0.5;
  fBottomArrowHeadRadius := 0.2;
  fBottomArrowHeadHeight := 0.5;
  FHeadStackingStyle := ahssStacked;
  FParts := [aaArc, aaTopArrow];
end;

// SetArcRadius
//

procedure TGLArrowArc.SetArcRadius(const aValue: Single);
begin
  if fArcRadius <> aValue then
  begin
    fArcRadius := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetStartAngle
//

procedure TGLArrowArc.SetStartAngle(const aValue: Single);
begin
  if FStartAngle <> aValue then
  begin
    FStartAngle := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetStopAngle
//

procedure TGLArrowArc.SetStopAngle(const aValue: Single);
begin
  if FStopAngle <> aValue then
  begin
    FStopAngle := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetTopRadius
//

procedure TGLArrowArc.SetTopRadius(const aValue: Single);
begin
  if aValue <> FTopRadius then
  begin
    FTopRadius := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetTopArrowHeadHeight
//

procedure TGLArrowArc.SetTopArrowHeadHeight(const aValue: Single);
begin
  if aValue <> fTopArrowHeadHeight then
  begin
    fTopArrowHeadHeight := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetTopArrowHeadRadius
//

procedure TGLArrowArc.SetTopArrowHeadRadius(const aValue: Single);
begin
  if aValue <> fTopArrowHeadRadius then
  begin
    fTopArrowHeadRadius := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetBottomArrowHeadHeight
//

procedure TGLArrowArc.SetBottomArrowHeadHeight(const aValue: Single);
begin
  if aValue <> fBottomArrowHeadHeight then
  begin
    fBottomArrowHeadHeight := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetBottomArrowHeadRadius
//

procedure TGLArrowArc.SetBottomArrowHeadRadius(const aValue: Single);
begin
  if aValue <> fBottomArrowHeadRadius then
  begin
    fBottomArrowHeadRadius := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetParts
//

procedure TGLArrowArc.SetParts(aValue: TArrowArcParts);
begin
  if aValue <> FParts then
  begin
    FParts := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetHeadStackingStyle
//

procedure TGLArrowArc.SetHeadStackingStyle(const val: TArrowHeadStackingStyle);
begin
  if val <> FHeadStackingStyle then
  begin
    FHeadStackingStyle := val;
    FMesh := nil;
    StructureChanged;
  end;
end;

// BuildList
//

procedure TGLArrowArc.BuildList(var rci: TGLRenderContextInfo);
  procedure EmitVertex(ptr: PVertexRec; L1, L2: integer);
  // {$IFDEF GLS_INLINE}inline;{$ENDIF}
  begin
    XGL.TexCoord2fv(@ptr^.TexCoord);
    with GL do
    begin
      Normal3fv(@ptr^.Normal);
      if L1 > -1 then
        VertexAttrib3fv(L1, @ptr.Tangent);
      if L2 > -1 then
        VertexAttrib3fv(L2, @ptr.Binormal);
      Vertex3fv(@ptr^.Position);
    end;
  end;

var
  i, j: integer;
  Theta, Phi, Theta1, cosPhi, sinPhi, dist: TGLFloat;
  cosTheta1, sinTheta1: TGLFloat;
  ringDelta, sideDelta: TGLFloat;
  ringDir: TAffineVector;
  iFact, jFact: Single;
  pVertex: PVertexRec;
  TanLoc, BinLoc: TGLInt;
  MeshSize: integer;
  MeshIndex: integer;
  ConeCenter: TVertexRec;
  StartOffset, StopOffset: Single;
begin
  if FMesh = nil then
  begin
    MeshIndex := 0;
    MeshSize := 0;
    // Check Parts
    if aaArc in FParts then
      MeshSize := MeshSize + FStacks + 1;
    if aaTopArrow in FParts then
      MeshSize := MeshSize + 3
    else
      MeshSize := MeshSize + 1;
    if aaBottomArrow in FParts then
      MeshSize := MeshSize + 3
    else
      MeshSize := MeshSize + 1;
    // Allocate Mesh
    SetLength(FMesh, MeshSize);

    case FHeadStackingStyle of
      ahssStacked:
        begin
          StartOffset := 0;
          StopOffset := 0;
        end;
      ahssCentered:
        begin
          if aaBottomArrow in Parts then
            StartOffset :=
              RadToDeg(ArcTan(0.5 * fBottomArrowHeadHeight / fArcRadius))
          else
            StartOffset :=0;
          if aaTopArrow in Parts then
            StopOffset :=
              RadToDeg(ArcTan(0.5 * fTopArrowHeadHeight / fArcRadius))
          else
            StopOffset :=0;
        end ;
      ahssIncluded:
        begin
          if aaBottomArrow in Parts then
            StartOffset := RadToDeg(ArcTan(fBottomArrowHeadHeight / fArcRadius))
          else
            StartOffset :=0;
          if aaTopArrow in Parts then
            StopOffset := RadToDeg(ArcTan(fTopArrowHeadHeight / fArcRadius))
          else
            StopOffset :=0;
        end ;
    end;

    // handle texture generation
    ringDelta := (((FStopAngle - StopOffset) - (FStartAngle + StartOffset)) /
      360) * c2PI / FStacks;
    sideDelta := c2PI / FSlices;

    iFact := 1 / FStacks;
    jFact := 1 / FSlices;
    if aaArc in FParts then
    begin
      Theta := DegToRad(FStartAngle + StartOffset) - ringDelta;
      for i := FStacks downto 0 do
      begin
        SetLength(FMesh[i], FSlices + 1);
        Theta1 := Theta + ringDelta;
        SinCos(Theta1, sinTheta1, cosTheta1);
        Phi := 0;
        for j := FSlices downto 0 do
        begin
          Phi := Phi + sideDelta;
          SinCos(Phi, sinPhi, cosPhi);
          dist := fArcRadius + Lerp(FTopRadius, FBottomRadius, i * iFact) * cosPhi;

          FMesh[i][j].Position := Vector3fMake(cosTheta1 * dist,
            -sinTheta1 * dist, Lerp(FTopRadius, FBottomRadius, i * iFact) * sinPhi);
          ringDir := FMesh[i][j].Position;
          ringDir.V[2] := 0.0;
          NormalizeVector(ringDir);
          FMesh[i][j].Normal := Vector3fMake(cosTheta1 * cosPhi,
            -sinTheta1 * cosPhi, sinPhi);
          FMesh[i][j].Tangent := VectorCrossProduct(ZVector, ringDir);
          FMesh[i][j].Binormal := VectorCrossProduct(FMesh[i][j].Normal,
            FMesh[i][j].Tangent);
          FMesh[i][j].TexCoord := Vector2fMake(i * iFact, j * jFact);
        end;
        Theta := Theta1;
      end;
      MeshIndex := FStacks + 1;
      with GL do
      begin
        if ARB_shader_objects and (rci.GLStates.CurrentProgram > 0) then
        begin
          TanLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(TangentAttributeName));
          BinLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(BinormalAttributeName));
        end
        else
        begin
          TanLoc := -1;
          BinLoc := TanLoc;
        end;

        Begin_(GL_TRIANGLES);
        for i := FStacks - 1 downto 0 do
          for j := FSlices - 1 downto 0 do
          begin
            pVertex := @FMesh[i][j];
            EmitVertex(pVertex, TanLoc, BinLoc);

            pVertex := @FMesh[i][j + 1];
            EmitVertex(pVertex, TanLoc, BinLoc);

            pVertex := @FMesh[i + 1][j];
            EmitVertex(pVertex, TanLoc, BinLoc);

            pVertex := @FMesh[i + 1][j + 1];
            EmitVertex(pVertex, TanLoc, BinLoc);

            pVertex := @FMesh[i + 1][j];
            EmitVertex(pVertex, TanLoc, BinLoc);

            pVertex := @FMesh[i][j + 1];
            EmitVertex(pVertex, TanLoc, BinLoc);
          end;
        End_;
      end;
    end;

    // Build Arrow or start cap
    if aaBottomArrow in FParts then
    begin
      SetLength(FMesh[MeshIndex], FSlices + 1);
      SetLength(FMesh[MeshIndex + 1], FSlices + 1);
      SetLength(FMesh[MeshIndex + 2], FSlices + 1);
      Theta1 := DegToRad(FStartAngle + StartOffset);
      SinCos(Theta1, sinTheta1, cosTheta1);

      ConeCenter.Position := Vector3fMake(cosTheta1 * fArcRadius,
        -sinTheta1 * fArcRadius, 0);

      Phi := 0;
      for j := FSlices downto 0 do
      begin
        Phi := Phi + sideDelta;
        SinCos(Phi, sinPhi, cosPhi);
        dist := fArcRadius + fBottomArrowHeadRadius * cosPhi;

        // Cap
        FMesh[MeshIndex][J].Position := Vector3fMake(cosTheta1 * dist,
          -sinTheta1 * dist, fBottomArrowHeadRadius * sinPhi);
        ringDir := FMesh[MeshIndex][j].Position;
        ringDir.V[2] := 0.0;
        NormalizeVector(ringDir);
        FMesh[MeshIndex][j].Normal := VectorCrossProduct(ringDir, ZVector);
        FMesh[MeshIndex][j].Tangent := ringDir;
        FMesh[MeshIndex][j].Binormal := ZVector;
        FMesh[MeshIndex][j].TexCoord := Vector2fMake(1, j * jFact);

        // Cone
        FMesh[MeshIndex+1][j].Position := Vector3fMake(cosTheta1 * dist,
          -sinTheta1 * dist, fBottomArrowHeadRadius * sinPhi);
        FMesh[MeshIndex+2][j].Position := VectorAdd(ConeCenter.Position,
          Vector3fMake(sinTheta1 * fBottomArrowHeadHeight,
          cosTheta1 * fBottomArrowHeadHeight, 0));

        FMesh[MeshIndex + 1][j].Tangent :=
          VectorNormalize(VectorSubtract(FMesh[MeshIndex + 1][j].Position,
          FMesh[MeshIndex + 2][j].Position));
        FMesh[MeshIndex + 2][j].Tangent := FMesh[MeshIndex + 1][j].Tangent;

        FMesh[MeshIndex + 1][j].Binormal := Vector3fMake(cosTheta1 * -sinPhi,
          sinTheta1 * sinPhi, cosPhi);
        FMesh[MeshIndex + 2][j].Binormal := FMesh[MeshIndex + 1][j].Binormal;

        FMesh[MeshIndex + 1][j].Normal :=
          VectorCrossProduct(FMesh[MeshIndex + 1][j].Binormal,
          FMesh[MeshIndex + 1][j].Tangent);
        FMesh[MeshIndex + 2][j].Normal := FMesh[MeshIndex + 1][j].Normal;

        FMesh[MeshIndex + 1][j].TexCoord := Vector2fMake(0, j * jFact);
        FMesh[MeshIndex + 2][j].TexCoord := Vector2fMake(1, j * jFact);
      end;

      ConeCenter.Normal := FMesh[MeshIndex][0].Normal;
      ConeCenter.Tangent := FMesh[MeshIndex][0].Tangent;
      ConeCenter.Binormal := FMesh[MeshIndex][0].Binormal;
      ConeCenter.TexCoord := Vector2fMake(0, 0);

      with GL do
      begin
        if ARB_shader_objects and (rci.GLStates.CurrentProgram > 0) then
        begin
          TanLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(TangentAttributeName));
          BinLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(BinormalAttributeName));
        end
        else
        begin
          TanLoc := -1;
          BinLoc := TanLoc;
        end;

        Begin_(GL_TRIANGLE_FAN);
        pVertex := @ConeCenter;
        EmitVertex(pVertex, TanLoc, BinLoc);
        for j := FSlices downto 0 do
        begin
          pVertex := @FMesh[MeshIndex][j];
          EmitVertex(pVertex, TanLoc, BinLoc);
        end;
        End_;

        Begin_(GL_TRIANGLES);

        for j := FSlices - 1 downto 0 do
        begin
          pVertex := @FMesh[MeshIndex + 1][j];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 1][j + 1];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 2][j];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 2][j + 1];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 2][j];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 1][j + 1];
          EmitVertex(pVertex, TanLoc, BinLoc);
        end;
        End_;

      end;
      MeshIndex := MeshIndex + 3;
    end
    else
    begin
      SetLength(FMesh[MeshIndex], FSlices + 1);
      Theta1 := DegToRad(FStartAngle);
      SinCos(Theta1, sinTheta1, cosTheta1);

      Phi := 0;
      for j := FSlices downto 0 do
      begin
        Phi := Phi + sideDelta;
        SinCos(Phi, sinPhi, cosPhi);
        dist := fArcRadius + fBottomRadius * cosPhi;
        FMesh[MeshIndex][j].Position := Vector3fMake(cosTheta1 * dist,
          -sinTheta1 * dist, FBottomRadius * sinPhi);
        ringDir := FMesh[MeshIndex][j].Position;
        ringDir.V[2] := 0.0;
        NormalizeVector(ringDir);
        FMesh[MeshIndex][j].Normal := VectorCrossProduct(ZVector, ringDir);
        FMesh[MeshIndex][j].Tangent := ringDir;
        FMesh[MeshIndex][j].Binormal := ZVector;
        FMesh[MeshIndex][j].TexCoord := Vector2fMake(0, j * jFact);
      end;

      ConeCenter.Position := Vector3fMake(cosTheta1 * fArcRadius,
        -sinTheta1 * fArcRadius, 0);
      ConeCenter.Normal := FMesh[MeshIndex][0].Normal;
      ConeCenter.Tangent := FMesh[MeshIndex][0].Tangent;
      ConeCenter.Binormal := FMesh[MeshIndex][0].Binormal;
      ConeCenter.TexCoord := Vector2fMake(1, 1);
      with GL do
      begin
        if ARB_shader_objects and (rci.GLStates.CurrentProgram > 0) then
        begin
          TanLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(TangentAttributeName));
          BinLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(BinormalAttributeName));
        end
        else
        begin
          TanLoc := -1;
          BinLoc := TanLoc;
        end;
        Begin_(GL_TRIANGLE_FAN);
        pVertex := @ConeCenter;
        EmitVertex(pVertex, TanLoc, BinLoc);
        for j := 0 to FSlices do
        begin
          pVertex := @FMesh[MeshIndex][j];
          EmitVertex(pVertex, TanLoc, BinLoc);
        end;
        End_;
      end;
      MeshIndex := MeshIndex + 1;
    end;

    if aaTopArrow in FParts then
    begin
      SetLength(FMesh[MeshIndex], FSlices + 1);
      SetLength(FMesh[MeshIndex + 1], FSlices + 1);
      SetLength(FMesh[MeshIndex + 2], FSlices + 1);
      Theta1 := DegToRad(FStopAngle - StopOffset);
      SinCos(Theta1, sinTheta1, cosTheta1);

      ConeCenter.Position := Vector3fMake(cosTheta1 * fArcRadius,
        -sinTheta1 * fArcRadius, 0);

      Phi := 0;
      for j := FSlices downto 0 do
      begin
        Phi := Phi + sideDelta;
        SinCos(Phi, sinPhi, cosPhi);
        dist := fArcRadius + fTopArrowHeadRadius * cosPhi;

        // Cap
        FMesh[MeshIndex][j].Position := Vector3fMake(cosTheta1 * dist,
          -sinTheta1 * dist, fTopArrowHeadRadius * sinPhi);
        ringDir := FMesh[MeshIndex][j].Position;
        ringDir.V[2] := 0.0;
        NormalizeVector(ringDir);
        FMesh[MeshIndex][j].Normal := VectorCrossProduct(ZVector, ringDir);
        FMesh[MeshIndex][j].Tangent := ringDir;
        FMesh[MeshIndex][j].Binormal := ZVector;
        FMesh[MeshIndex][j].TexCoord := Vector2fMake(0, j * jFact);

        // Cone
        FMesh[MeshIndex + 1][j].Position := Vector3fMake(cosTheta1 * dist,
          -sinTheta1 * dist, fTopArrowHeadRadius * sinPhi);
        FMesh[MeshIndex + 2][j].Position := VectorSubtract(ConeCenter.Position,
          Vector3fMake(sinTheta1 * fTopArrowHeadHeight,
          cosTheta1 * fTopArrowHeadHeight, 0));

        FMesh[MeshIndex + 1][j].Tangent :=
          VectorNormalize(VectorSubtract(FMesh[MeshIndex + 2][j].Position,
          FMesh[MeshIndex + 1][j].Position));
        FMesh[MeshIndex + 2][j].Tangent := FMesh[MeshIndex + 1][j].Tangent;

        FMesh[MeshIndex + 1][j].Binormal := Vector3fMake(cosTheta1 * -sinPhi,
          sinTheta1 * sinPhi, cosPhi);
        FMesh[MeshIndex + 2][j].Binormal := FMesh[MeshIndex + 1][j].Binormal;

        FMesh[MeshIndex + 1][j].Normal :=
          VectorCrossProduct(FMesh[MeshIndex + 1][j].Binormal,
          FMesh[MeshIndex + 1][j].Tangent);
        FMesh[MeshIndex + 2][j].Normal := FMesh[MeshIndex + 1][j].Normal;

        FMesh[MeshIndex + 1][j].TexCoord := Vector2fMake(1, j * jFact);
        FMesh[MeshIndex + 2][j].TexCoord := Vector2fMake(0, j * jFact);
      end;

      ConeCenter.Normal := FMesh[MeshIndex][0].Normal;
      ConeCenter.Tangent := FMesh[MeshIndex][0].Tangent;
      ConeCenter.Binormal := FMesh[MeshIndex][0].Binormal;
      ConeCenter.TexCoord := Vector2fMake(1, 1);

      with GL do
      begin
        if ARB_shader_objects and (rci.GLStates.CurrentProgram > 0) then
        begin
          TanLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(TangentAttributeName));
          BinLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(BinormalAttributeName));
        end
        else
        begin
          TanLoc := -1;
          BinLoc := TanLoc;
        end;

        Begin_(GL_TRIANGLE_FAN);
        pVertex := @ConeCenter;
        EmitVertex(pVertex, TanLoc, BinLoc);
        for j := 0 to FSlices do
        begin
          pVertex := @FMesh[MeshIndex][j];
          EmitVertex(pVertex, TanLoc, BinLoc);
        end;
        End_;

        Begin_(GL_TRIANGLES);

        for j := FSlices - 1 downto 0 do
        begin
          pVertex := @FMesh[MeshIndex + 2][j];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 2][j + 1];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 1][j];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 1][j + 1];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 1][j];
          EmitVertex(pVertex, TanLoc, BinLoc);

          pVertex := @FMesh[MeshIndex + 2][j + 1];
          EmitVertex(pVertex, TanLoc, BinLoc);
        end;
        End_;

      end;
    end
    else
    begin
      SetLength(FMesh[MeshIndex], FSlices + 1);
      Theta1 := DegToRad(FStopAngle);
      SinCos(Theta1, sinTheta1, cosTheta1);

      Phi := 0;
      for j := FSlices downto 0 do
      begin
        Phi := Phi + sideDelta;
        SinCos(Phi, sinPhi, cosPhi);
        dist := fArcRadius + fTopRadius * cosPhi;
        FMesh[MeshIndex][j].Position := Vector3fMake(cosTheta1 * dist,
          -sinTheta1 * dist, fTopRadius * sinPhi);
        ringDir := FMesh[MeshIndex][j].Position;
        ringDir.V[2] := 0.0;
        NormalizeVector(ringDir);
        FMesh[MeshIndex][j].Normal := VectorCrossProduct(ringDir, ZVector);
        FMesh[MeshIndex][j].Tangent := ringDir;
        FMesh[MeshIndex][j].Binormal := VectorNegate(ZVector);
        FMesh[MeshIndex][j].TexCoord := Vector2fMake(1, j * jFact);
      end;
      ConeCenter.Position := Vector3fMake(cosTheta1 * fArcRadius,
        -sinTheta1 * fArcRadius, 0);
      ConeCenter.Normal := FMesh[MeshIndex][0].Normal;
      ConeCenter.Tangent := FMesh[MeshIndex][0].Tangent;
      ConeCenter.Binormal := FMesh[MeshIndex][0].Binormal;
      ConeCenter.TexCoord := Vector2fMake(0, 0);
      with GL do
      begin
        if ARB_shader_objects and (rci.GLStates.CurrentProgram > 0) then
        begin
          TanLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(TangentAttributeName));
          BinLoc := GetAttribLocation(rci.GLStates.CurrentProgram,
            PGLChar(BinormalAttributeName));
        end
        else
        begin
          TanLoc := -1;
          BinLoc := TanLoc;
        end;
        Begin_(GL_TRIANGLE_FAN);
        pVertex := @ConeCenter;
        EmitVertex(pVertex, TanLoc, BinLoc);
        for j := FSlices downto 0 do
        begin
          pVertex := @FMesh[MeshIndex][j];
          EmitVertex(pVertex, TanLoc, BinLoc);
        end;
        End_;
      end;
    end;
  end;
end;

 
//

procedure TGLArrowArc.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLArrowLine) then
  begin
    FStartAngle := TGLArrowArc(Source).FStartAngle;
    FStopAngle := TGLArrowArc(Source).FStopAngle;
    fArcRadius := TGLArrowArc(Source).fArcRadius;
    FParts := TGLArrowArc(Source).FParts;
    FTopRadius := TGLArrowArc(Source).FTopRadius;
    fTopArrowHeadHeight := TGLArrowArc(Source).fTopArrowHeadHeight;
    fTopArrowHeadRadius := TGLArrowArc(Source).fTopArrowHeadRadius;
    fBottomArrowHeadHeight := TGLArrowArc(Source).fBottomArrowHeadHeight;
    fBottomArrowHeadRadius := TGLArrowArc(Source).fBottomArrowHeadRadius;
    FHeadStackingStyle := TGLArrowArc(Source).FHeadStackingStyle;
  end;
  inherited Assign(Source);
end;

// ------------------
// ------------------ TGLFrustrum ------------------
// ------------------

constructor TGLFrustrum.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FApexHeight := 1;
  FBaseWidth := 1;
  FBaseDepth := 1;
  FHeight := 0.5;
  FParts := cAllFrustrumParts;
  FNormalDirection := ndOutside;
end;

procedure TGLFrustrum.BuildList(var rci: TGLRenderContextInfo);
var
  HBW, HBD: TGLFloat; // half of width, half of depth at base
  HTW, HTD: TGLFloat; // half of width, half of depth at top of frustrum
  HFH: TGLFloat; // half of height, for align to center
  Sign: TGLFloat; // +1 or -1
  angle: TGLFloat; // in radians
  ASin, ACos: TGLFloat;
begin
  if FNormalDirection = ndInside then
    Sign := -1
  else
    Sign := 1;
  HBW := FBaseWidth * 0.5;
  HBD := FBaseDepth * 0.5;
  HTW := HBW * (FApexHeight - FHeight) / FApexHeight;
  HTD := HBD * (FApexHeight - FHeight) / FApexHeight;
  HFH := FHeight * 0.5;

  GL.Begin_(GL_QUADS);

  if [fpFront, fpBack] * FParts <> [] then
  begin
    angle := ArcTan(FApexHeight / HBD);
    // angle of front plane with bottom plane
    SinCos(angle, ASin, ACos);
    if fpFront in FParts then
    begin
      GL.Normal3f(0, Sign * ACos, Sign * ASin);
      XGL.TexCoord2fv(@XYTexPoint);
      GL.Vertex3f(HTW, HFH, HTD);
      XGL.TexCoord2fv(@YTexPoint);
      GL.Vertex3f(-HTW, HFH, HTD);
      XGL.TexCoord2fv(@NullTexPoint);
      GL.Vertex3f(-HBW, -HFH, HBD);
      XGL.TexCoord2fv(@XTexPoint);
      GL.Vertex3f(HBW, -HFH, HBD);
    end;
    if fpBack in FParts then
    begin
      GL.Normal3f(0, Sign * ACos, -Sign * ASin);
      XGL.TexCoord2fv(@YTexPoint);
      GL.Vertex3f(HTW, HFH, -HTD);
      XGL.TexCoord2fv(@NullTexPoint);
      GL.Vertex3f(HBW, -HFH, -HBD);
      XGL.TexCoord2fv(@XTexPoint);
      GL.Vertex3f(-HBW, -HFH, -HBD);
      XGL.TexCoord2fv(@XYTexPoint);
      GL.Vertex3f(-HTW, HFH, -HTD);
    end;
  end;

  if [fpLeft, fpRight] * FParts <> [] then
  begin
    angle := ArcTan(FApexHeight / HBW); // angle of side plane with bottom plane
    SinCos(angle, ASin, ACos);
    if fpLeft in FParts then
    begin
      GL.Normal3f(-Sign * ASin, Sign * ACos, 0);
      XGL.TexCoord2fv(@XYTexPoint);
      GL.Vertex3f(-HTW, HFH, HTD);
      XGL.TexCoord2fv(@YTexPoint);
      GL.Vertex3f(-HTW, HFH, -HTD);
      XGL.TexCoord2fv(@NullTexPoint);
      GL.Vertex3f(-HBW, -HFH, -HBD);
      XGL.TexCoord2fv(@XTexPoint);
      GL.Vertex3f(-HBW, -HFH, HBD);
    end;
    if fpRight in FParts then
    begin
      GL.Normal3f(Sign * ASin, Sign * ACos, 0);
      XGL.TexCoord2fv(@YTexPoint);
      GL.Vertex3f(HTW, HFH, HTD);
      XGL.TexCoord2fv(@NullTexPoint);
      GL.Vertex3f(HBW, -HFH, HBD);
      XGL.TexCoord2fv(@XTexPoint);
      GL.Vertex3f(HBW, -HFH, -HBD);
      XGL.TexCoord2fv(@XYTexPoint);
      GL.Vertex3f(HTW, HFH, -HTD);
    end;
  end;

  if (fpTop in FParts) and (FHeight < FApexHeight) then
  begin
    GL.Normal3f(0, Sign, 0);
    XGL.TexCoord2fv(@YTexPoint);
    GL.Vertex3f(-HTW, HFH, -HTD);
    XGL.TexCoord2fv(@NullTexPoint);
    GL.Vertex3f(-HTW, HFH, HTD);
    XGL.TexCoord2fv(@XTexPoint);
    GL.Vertex3f(HTW, HFH, HTD);
    XGL.TexCoord2fv(@XYTexPoint);
    GL.Vertex3f(HTW, HFH, -HTD);
  end;
  if fpBottom in FParts then
  begin
    GL.Normal3f(0, -Sign, 0);
    XGL.TexCoord2fv(@NullTexPoint);
    GL.Vertex3f(-HBW, -HFH, -HBD);
    XGL.TexCoord2fv(@XTexPoint);
    GL.Vertex3f(HBW, -HFH, -HBD);
    XGL.TexCoord2fv(@XYTexPoint);
    GL.Vertex3f(HBW, -HFH, HBD);
    XGL.TexCoord2fv(@YTexPoint);
    GL.Vertex3f(-HBW, -HFH, HBD);
  end;

  GL.End_;
end;

procedure TGLFrustrum.SetApexHeight(const aValue: Single);
begin
  if (aValue <> FApexHeight) and (aValue >= 0) then
  begin
    FApexHeight := aValue;
    if FHeight > aValue then
      FHeight := aValue;
    StructureChanged;
  end;
end;

procedure TGLFrustrum.SetBaseDepth(const aValue: Single);
begin
  if (aValue <> FBaseDepth) and (aValue >= 0) then
  begin
    FBaseDepth := aValue;
    StructureChanged;
  end;
end;

procedure TGLFrustrum.SetBaseWidth(const aValue: Single);
begin
  if (aValue <> FBaseWidth) and (aValue >= 0) then
  begin
    FBaseWidth := aValue;
    StructureChanged;
  end;
end;

procedure TGLFrustrum.SetHeight(const aValue: Single);
begin
  if (aValue <> FHeight) and (aValue >= 0) then
  begin
    FHeight := aValue;
    if FApexHeight < aValue then
      FApexHeight := aValue;
    StructureChanged;
  end;
end;

procedure TGLFrustrum.SetParts(aValue: TFrustrumParts);
begin
  if aValue <> FParts then
  begin
    FParts := aValue;
    StructureChanged;
  end;
end;

procedure TGLFrustrum.SetNormalDirection(aValue: TNormalDirection);
begin
  if aValue <> FNormalDirection then
  begin
    FNormalDirection := aValue;
    StructureChanged;
  end;
end;

procedure TGLFrustrum.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLFrustrum) then
  begin
    FApexHeight := TGLFrustrum(Source).FApexHeight;
    FBaseDepth := TGLFrustrum(Source).FBaseDepth;
    FBaseWidth := TGLFrustrum(Source).FBaseWidth;
    FHeight := TGLFrustrum(Source).FHeight;
    FParts := TGLFrustrum(Source).FParts;
    FNormalDirection := TGLFrustrum(Source).FNormalDirection;
  end;
  inherited Assign(Source);
end;

function TGLFrustrum.TopDepth: TGLFloat;
begin
  Result := FBaseDepth * (FApexHeight - FHeight) / FApexHeight;
end;

function TGLFrustrum.TopWidth: TGLFloat;
begin
  Result := FBaseWidth * (FApexHeight - FHeight) / FApexHeight;
end;

procedure TGLFrustrum.DefineProperties(Filer: TFiler);
begin
  inherited;
  Filer.DefineBinaryProperty('FrustrumSize', ReadData, WriteData,
    (FApexHeight <> 1) or (FBaseDepth <> 1) or (FBaseWidth <> 1) or
    (FHeight <> 0.5));
end;

procedure TGLFrustrum.ReadData(Stream: TStream);
begin
  with Stream do
  begin
    Read(FApexHeight, SizeOf(FApexHeight));
    Read(FBaseDepth, SizeOf(FBaseDepth));
    Read(FBaseWidth, SizeOf(FBaseWidth));
    Read(FHeight, SizeOf(FHeight));
  end;
end;

procedure TGLFrustrum.WriteData(Stream: TStream);
begin
  with Stream do
  begin
    Write(FApexHeight, SizeOf(FApexHeight));
    Write(FBaseDepth, SizeOf(FBaseDepth));
    Write(FBaseWidth, SizeOf(FBaseWidth));
    Write(FHeight, SizeOf(FHeight));
  end;
end;

function TGLFrustrum.AxisAlignedBoundingBoxUnscaled: TAABB;
var
  aabb: TAABB;
  child: TGLBaseSceneObject;
  i: integer;
begin
  SetAABB(Result, AxisAlignedDimensionsUnscaled);
  OffsetAABB(Result, VectorMake(0, FHeight * 0.5, 0));

  // not tested for child objects
  for i := 0 to Count - 1 do
  begin
    child := TGLBaseSceneObject(Children[i]);
    aabb := child.AxisAlignedBoundingBoxUnscaled;
    AABBTransform(aabb, child.Matrix);
    AddAABB(Result, aabb);
  end;
end;

function TGLFrustrum.AxisAlignedDimensionsUnscaled: TVector;
begin
  Result.V[0] := FBaseWidth * 0.5;
  Result.V[1] := FHeight * 0.5;
  Result.V[2] := FBaseDepth * 0.5;
  Result.V[3] := 0;
end;

// ------------------
// ------------------ TGLPolygon ------------------
// ------------------

// Create
//

constructor TGLPolygon.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FParts := [ppTop, ppBottom];
end;

// Destroy
//

destructor TGLPolygon.Destroy;
begin
  inherited Destroy;
end;

// SetParts
//

procedure TGLPolygon.SetParts(const val: TPolygonParts);
begin
  if FParts <> val then
  begin
    FParts := val;
    StructureChanged;
  end;
end;

 
//

procedure TGLPolygon.Assign(Source: TPersistent);
begin
  if Source is TGLPolygon then
  begin
    FParts := TGLPolygon(Source).FParts;
  end;
  inherited Assign(Source);
end;

// BuildList
//

procedure TGLPolygon.BuildList(var rci: TGLRenderContextInfo);
var
  Normal: TAffineVector;
  pNorm: PAffineVector;
begin
  if (Nodes.Count > 1) then
  begin
    Normal := Nodes.Normal;
    if VectorIsNull(Normal) then
      pNorm := nil
    else
      pNorm := @Normal;
    if ppTop in FParts then
    begin
      if SplineMode = lsmLines then
        Nodes.RenderTesselatedPolygon(true, pNorm, 1)
      else
        Nodes.RenderTesselatedPolygon(true, pNorm, Division);
    end;
    // tessellate bottom polygon
    if ppBottom in FParts then
    begin
      if Assigned(pNorm) then
        NegateVector(Normal);
      if SplineMode = lsmLines then
        Nodes.RenderTesselatedPolygon(true, pNorm, 1, true)
      else
        Nodes.RenderTesselatedPolygon(true, pNorm, Division, true);
    end;
  end;
end;

// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------

initialization

// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------

RegisterClasses([TGLCylinder, TGLCone, TGLTorus, TGLDisk, TGLArrowLine,
  TGLAnnulus, TGLFrustrum, TGLPolygon, TGLCapsule, TGLArrowArc]);

end.
