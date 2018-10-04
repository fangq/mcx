//
// This unit is part of the GLScene Project, http://glscene.org
//
{
  Implementation of basic scene objects plus some management routines.

  All objects declared in this unit are part of the basic GLScene package,
  these are only simple objects and should be kept simple and lightweight. 

  More complex or more specialized versions should be placed in dedicated
  units where they can grow and prosper untammed. "Generic" geometrical
  objects can be found GLGeomObjects.

   History :  
   12/03/13 - Yar - Added TGLSuperellipsoid (contributed by Eric Hardinge)
   10/03/13 - PW - Added OctahedronBuildList and TetrahedronBuildList
   20/11/12 - PW - CPP compatibility: replaced direct access to some properties with
                 getter and a setter methods
   23/03/11 - Yar - Bugfixed TGLPlane.Assign (thanks ltyrosine)
                       Replaced plane primitives to triangles, added tangent and binormal attributes
   29/11/10 - Yar - Bugfixed client color array enabling in TGLPoints.BuildList when it not used (thanks rbenetis)
   23/08/10 - Yar - Added OpenGLTokens to uses, replaced OpenGL1x functions to OpenGLAdapter
   29/06/10 - Yar - Added loColorLogicXor to TGLLines.Options
   22/04/10 - Yar - Fixes after GLState revision
   11/04/10 - Yar - Replaced glNewList to GLState.NewList in TGLDummyCube.DoRender
   05/03/10 - DanB - More state added to TGLStateCache
   22/02/10 - Yar - Removed NoZWrite in TGLPlane, TGLSprite
                 Now use Material.DepthProperties
   28/12/09 - DanB - Modifying TGLLineBase.LineColor now calls StructureChanged
   13/03/09 - DanB - ScreenRect now accepts a buffer parameter, rather than using CurrentBuffer
   05/10/08 - DaStr - Added lsmLoop support to TGLLines
                (thanks Alejandro Leon Escalera) (BugtrackerID = 2084250)
   22/01/08 - DaStr - Fixed rendering of TGLPoints
                (thanks Kapitan) (BugtrackerID = 1876920)
   06/06/07 - DaStr - Added GLColor to uses (BugtrackerID = 1732211)
   14/03/07 - DaStr - Added explicit pointer dereferencing
                 (thanks Burkhard Carstens) (Bugtracker ID = 1678644)
   15/02/07 - DaStr - Global $R- removed, added default values to
                 TGLSprite.NoZWrite, MirrorU, MirrorV
   14/01/07 - DaStr - Fixed TGLCube.BuildList. Bugtracker ID=1623743 (Thanks Pete Jones)
   19/10/06 - LC - Fixed IcosahedronBuildList. Bugtracker ID=1490784 (thanks EPA_Couzijn)
   19/10/06 - LC - Fixed TGLLineBase.Assign problem. Bugtracker ID=1549354 (thanks Zapology)
   08/10/05 - Mathx - Fixed TGLLines.nodes.assign problem (thanks to  Yong Yoon Kit);
                 Also fixed a TGLLineBase.assign problem (object being assigned to
                 was refering the base lists, not copying them).
                 Bugtracker ID=830846
   17/01/05 - SG - Added color support for bezier style TGLLines
   03/12/04 - MF - Added TGLSprite.AxisAlignedDimensionsUnscaled override
   06/07/04 - SG - TGLCube.RayCastIntersect fix (Eric Pascual)
   20/01/04 - SG - Added IcosahedronBuildList
   30/11/03 - MF - Added TGLSphere.GenerateSilhouette - it now takes the
                      stacks/slices of the sphere into account
   10/09/03 - EG - Introduced TGLNodedLines
   18/08/03 - SG - Added MirrorU and MirrorV to TGLSprite for mirroring textures
   21/07/03 - EG - TGLTeapot moved to new GLTeapot unit,
                      TGLDodecahedron moved to new GLPolyhedron unit,
                      TGLCylinder, TGLCone, TGLTorus, TGLDisk, TGLArrowLine,
                      TGLAnnulus, TGLFrustrum and TGLPolygon moved to new
                      GLGeomObjects unit
   16/07/03 - EG - Style changes and cleanups
   19/06/03 - MF - Added GenerateSilhouette to TGLCube and TGLPlane.
   13/06/03 - EG - Fixed TGLAnnulus.RayCastIntersect (Alexandre Hirzel)
   03/06/03 - EG - Added TGLAnnulus.RayCastIntersect (Alexandre Hirzel)
   01/05/03 - SG - Added NURBS Curve to TGLLines (color not supported yet)
   14/04/03 - SG - Added a Simple Bezier Spline to TGLLines (color not supported yet)
   02/04/03 - EG - TGLPlane.RayCastIntersect fix (Erick Schuitema)
   13/02/03 - DanB - added AxisAlignedDimensionsUnscaled functions
   22/01/03 - EG - TGLCube.RayCastIntersect fixes (Dan Bartlett)
   10/01/03 - EG - TGLCube.RayCastIntersect (Stuart Gooding)
   08/01/03 - RC - Added TGLPlane.XScope and YScope, to use just a part of the texture
   27/09/02 - EG - Added TGLPointParameters
   24/07/02 - EG - Added TGLCylinder.Alignment
   23/07/02 - EG - Added TGLPoints (experimental)
   20/07/02 - EG - TGLCylinder.RayCastIntersect and TGLPlane.RayCastIntersect
   18/07/02 - EG - Added TGLCylinder.Align methods
   07/07/02 - EG - Added TGLPlane.Style
   03/07/02 - EG - TGLPolygon now properly setups normals (filippo)
   17/03/02 - EG - Support for transparent lines
   02/02/02 - EG - Fixed TGLSprite change notification
   26/01/02 - EG - TGLPlane & TGLCube now osDirectDraw
   20/01/02 - EG - TGLSpaceText moved to GLSpaceText
   22/08/01 - EG - TGLTorus.RayCastIntersect fixes
   30/07/01 - EG - Updated AxisAlignedDimensions implems
   16/03/01 - EG - TGLCylinderBase, changed default Stacks from 8 to 4
   27/02/01 - EG - Fix in TGLCube texcoords, added TGLFrustrum (thx Robin Gerrets)
   22/02/01 - EG - Added AxisAlignedDimensions overrides by Uwe Raabe
   05/02/01 - EG - Minor changes to TGLCube.BuildList
   21/01/01 - EG - BaseProjectionMatrix fix for TGLHUDSprite (picking issue),
  TGLHUDSprite moved to GLHUDObjects
   14/01/01 - EG - Fixed TGLSphere texture coordinates
   13/01/01 - EG - TGLSprite matrix compatibility update
   09/01/01 - EG - TGLSpaceText now handles its TFont.OnFontChange
   08/01/01 - EG - Added TGLLinesNode (color support) and Node size control
   22/12/00 - EG - Sprites are no longer texture enabled by default,
                      updated TGLSprite.BuildList to work with new matrices
   14/11/00 - EG - Added TGLDummyCube.Destroy (thx Airatz)
   08/10/00 - EG - Fixed call to wglUseFontOutlines
   06/08/00 - EG - TRotationSolid renamed to TGLRevolutionSolid & moved to GLExtrusion
   04/08/00 - EG - Fixed sphere main body texture coords + slight speedup
   02/08/00 - EG - Added TGLPolygonBase
   19/07/00 - EG - Added TGLHUDSprite
   18/07/00 - EG - Added TGLRevolutionSolid
   15/07/00 - EG - Code reduction and minor speedup for all quadric objects,
                      Added TGLLineBase (split of TGLLines),
                      TGLDummyCube now uses osDirectDraw instead of special behaviour
   13/07/00 - EG - Added TGLArrowLine (code by Aaron Hochwimmer)
   28/06/00 - EG - Support for "ObjectStyle"
   23/06/00 - EG - Reduced default Loop count for TGLDisk
   18/06/00 - EG - TGLMesh and accompanying stuff moved to GLMesh
   14/06/00 - EG - Added Capacity to TGLVertexList
   09/06/00 - EG - First row of Geometry-related upgrades
   08/06/00 - EG - Added ReleaseFontManager, fixed TGLSpaceText DestroyList,
   01/06/00 - EG - Added TGLAnnulus (code by Aaron Hochwimmer)
   29/05/00 - EG - TGLLines now uses TGLNode/TGLNodes
   28/05/00 - EG - Added persistence ability to TGLLines,
                      Added defaults for all TGLLines properties
   27/05/00 - EG - Moved in RogerCao's TGLLines object, added a TLineNode
                      class (currently private) and various enhancements + fixes,
                      DodecahedronBuildList now available as a procedure,
                      CubeWireframeBuildList now available as a procedure
   26/05/00 - RoC - Added division property to TGLLines, and Spline supported
   26/05/00 - EG - Moved vectorfile remnants to GLVectorFiles
   14/05/00 - EG - Removed Top/Bottom checks for TGLSphere,
  Added mmTriangleStrip support in CalcNormals
   08/05/00 - EG - Uncommented DisableAutoTexture in TGLSpaceText.BuildList
   07/05/00 - RoC - TGLLines added, to show a list of vertex
   26/04/00 - EG - Reactivated stuff in SetupQuadricParams (thanks Nelson Chu)
   18/04/00 - EG - Overriden TGLDummyCube.Render
   16/04/00 - EG - FontManager now published and auto-creating
   12/04/00 - EG - Added TGLCylinderBase.Loops (fixes a bug, thanks Uwe)
   24/03/00 - EG - Added Rotation to TGLSprite, fixed sprite size
   20/03/00 - EG - Enhanced FontManager
   17/03/00 - EG - Fixed SpaceText glBaseList bug,
  TGLSprite now uses a transposition of the globalmatrix
   16/03/00 - EG - Enhanced TFontManager to allow lower quality
   14/03/00 - EG - Added subobjects Barycenter support for TGLDummyCube
   09/02/00 - EG - ObjectManager stuff moved to GLSceneRegister,
  FreeForm and vector file stuff moved to new GLVectorFileObjects
   08/02/00 - EG - Added TGLDummyCube
   05/02/00 - EG - Javadocisation, fixes and enhancements :
                      TGLVertexList.AddVertex, "default"s to properties
   
}
unit GLObjects;

interface

{$I GLScene.inc}

uses
  Classes, SysUtils,
  GLVectorGeometry, GLVectorTypes, GLScene, OpenGLAdapter,
  OpenGLTokens, GLVectorLists, GLCrossPlatform, GLContext, GLSilhouette,
  GLColor, GLRenderContextInfo, GLBaseClasses, GLNodes, GLCoordinates;

type

  // TGLVisibilityDeterminationEvent
  //
  TGLVisibilityDeterminationEvent = function(Sender: TObject;
    var rci: TGLRenderContextInfo): Boolean of object;

  PVertexRec = ^TVertexRec;
  TVertexRec = record
    Position: TVector3f;
    Normal: TVector3f;
    Binormal: TVector3f;
    Tangent: TVector3f;
    TexCoord: TVector2f;
  end;

  // TGLDummyCube
  //
  { : A simple cube, invisible at run-time.
    This is a usually non-visible object -except at design-time- used for
    building hierarchies or groups, when some kind of joint or movement
    mechanism needs be described, you can use DummyCubes. 
    DummyCube's barycenter is its children's barycenter. 
    The DummyCube can optionnally amalgamate all its children into a single
    display list (see Amalgamate property). }
  TGLDummyCube = class(TGLCameraInvariantObject)
  private
     
    FCubeSize: TGLFloat;
    FEdgeColor: TGLColor;
    FVisibleAtRunTime, FAmalgamate: Boolean;
    FGroupList: TGLListHandle;
    FOnVisibilityDetermination: TGLVisibilityDeterminationEvent;

  protected
     
    procedure SetCubeSize(const val: TGLFloat);
    procedure SetEdgeColor(const val: TGLColor);
    procedure SetVisibleAtRunTime(const val: Boolean);
    procedure SetAmalgamate(const val: Boolean);

  public
     
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;

    procedure Assign(Source: TPersistent); override;

    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;
    procedure DoRender(var rci: TGLRenderContextInfo;
      renderSelf, renderChildren: Boolean); override;
    procedure StructureChanged; override;
    function BarycenterAbsolutePosition: TVector; override;

  published
     
    property CubeSize: TGLFloat read FCubeSize write SetCubeSize;
    property EdgeColor: TGLColor read FEdgeColor write SetEdgeColor;
    { : If true the dummycube's edges will be visible at runtime.
      The default behaviour of the dummycube is to be visible at design-time
      only, and invisible at runtime. }
    property VisibleAtRunTime: Boolean read FVisibleAtRunTime
      write SetVisibleAtRunTime default False;
    { : Amalgamate the dummy's children in a single OpenGL entity.
      This activates a special rendering mode, which will compile
      the rendering of all of the dummycube's children objects into a
      single display list. This may provide a significant speed up in some
      situations, however, this means that changes to the children will
      be ignored untill you call StructureChanged on the dummy cube. 
      Some objects, that have their own display list management, may not
      be compatible with this behaviour. This will also prevents sorting
      and culling to operate as usual.
      In short, this features is best used for static, non-transparent
      geometry, or when the point of view won't change over a large
      number of frames. }
    property Amalgamate: Boolean read FAmalgamate write SetAmalgamate
      default False;
    { : Camera Invariance Options.
      These options allow to "deactivate" sensitivity to camera, f.i. by
      centering the object on the camera or ignoring camera orientation. }
    property CamInvarianceMode default cimNone;
    { : Event for custom visibility determination.
      Event handler should return True if the dummycube and its children
      are to be considered visible for the current render. }
    property OnVisibilityDetermination: TGLVisibilityDeterminationEvent
      read FOnVisibilityDetermination write FOnVisibilityDetermination;
  end;

  // TPlaneStyle
  //
  TPlaneStyle = (psSingleQuad, psTileTexture);
  TPlaneStyles = set of TPlaneStyle;

  // Plane
  //
  { : A simple plane object.
    Note that a plane is always made of a single quad (two triangles) and the
    tiling is only applied to texture coordinates. }
  TGLPlane = class(TGLSceneObject)
  private
     
    FXOffset, FYOffset: TGLFloat;
    FXScope, FYScope: TGLFloat;
    FWidth, FHeight: TGLFloat;
    FXTiles, FYTiles: Cardinal;
    FStyle: TPlaneStyles;
    FMesh: array of array of TVertexRec;
  protected
     
    procedure SetHeight(const aValue: Single);
    procedure SetWidth(const aValue: Single);
    procedure SetXOffset(const Value: TGLFloat);
    procedure SetXScope(const Value: TGLFloat);
    function StoreXScope: Boolean;
    procedure SetXTiles(const Value: Cardinal);
    procedure SetYOffset(const Value: TGLFloat);
    procedure SetYScope(const Value: TGLFloat);
    function StoreYScope: Boolean;
    procedure SetYTiles(const Value: Cardinal);
    procedure SetStyle(const val: TPlaneStyles);

  public
     
    constructor Create(AOwner: TComponent); override;

    procedure Assign(Source: TPersistent); override;

    procedure BuildList(var rci: TGLRenderContextInfo); override;
    function GenerateSilhouette(const silhouetteParameters
      : TGLSilhouetteParameters): TGLSilhouette; override;

    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;
    { : Computes the screen coordinates of the smallest rectangle encompassing the plane.
      Returned extents are NOT limited to any physical screen extents. }
    function ScreenRect(aBuffer: TGLSceneBuffer): TGLRect;

    { : Computes the signed distance to the point.
      Point coordinates are expected in absolute coordinates. }
    function PointDistance(const aPoint: TVector): Single;

  published
     
    property Height: TGLFloat read FHeight write SetHeight;
    property Width: TGLFloat read FWidth write SetWidth;
    property XOffset: TGLFloat read FXOffset write SetXOffset;
    property XScope: TGLFloat read FXScope write SetXScope stored StoreXScope;
    property XTiles: Cardinal read FXTiles write SetXTiles default 1;
    property YOffset: TGLFloat read FYOffset write SetYOffset;
    property YScope: TGLFloat read FYScope write SetYScope stored StoreYScope;
    property YTiles: Cardinal read FYTiles write SetYTiles default 1;
    property Style: TPlaneStyles read FStyle write SetStyle
      default [psSingleQuad, psTileTexture];
  end;

  // TGLSprite
  //
  { : A rectangular area, perspective projected, but always facing the camera.
    A TGLSprite is perspective projected and as such is scaled with distance,
    if you want a 2D sprite that does not get scaled, see TGLHUDSprite. }
  TGLSprite = class(TGLSceneObject)
  private
     
    FWidth: TGLFloat;
    FHeight: TGLFloat;
    FRotation: TGLFloat;
    FAlphaChannel: Single;
    FMirrorU, FMirrorV: Boolean;

  protected
     
    procedure SetWidth(const val: TGLFloat);
    procedure SetHeight(const val: TGLFloat);
    procedure SetRotation(const val: TGLFloat);
    procedure SetAlphaChannel(const val: Single);
    function StoreAlphaChannel: Boolean;
    procedure SetMirrorU(const val: Boolean);
    procedure SetMirrorV(const val: Boolean);

  public
     
    constructor Create(AOwner: TComponent); override;

    procedure Assign(Source: TPersistent); override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;

    function AxisAlignedDimensionsUnscaled: TVector; override;

    procedure SetSize(const Width, Height: TGLFloat);
    // : Set width and height to "size"
    procedure SetSquareSize(const Size: TGLFloat);

  published
     
    { : Sprite Width in 3D world units. }
    property Width: TGLFloat read FWidth write SetWidth;
    { : Sprite Height in 3D world units. }
    property Height: TGLFloat read FHeight write SetHeight;
    { : This the ON-SCREEN rotation of the sprite.
      Rotatation=0 is handled faster. }
    property Rotation: TGLFloat read FRotation write SetRotation;
    { : If different from 1, this value will replace that of Diffuse.Alpha }
    property AlphaChannel: Single read FAlphaChannel write SetAlphaChannel
      stored StoreAlphaChannel;
    { : Reverses the texture coordinates in the U and V direction to mirror
      the texture. }
    property MirrorU: Boolean read FMirrorU write SetMirrorU default False;
    property MirrorV: Boolean read FMirrorV write SetMirrorV default False;
  end;

  // TGLPointStyle
  //
  TGLPointStyle = (psSquare, psRound, psSmooth, psSmoothAdditive,
    psSquareAdditive);

  // TGLPointParameters
  //
  { : Point parameters as in ARB_point_parameters.
    Make sure to read the ARB_point_parameters spec if you want to understand
    what each parameter does. }
  TGLPointParameters = class(TGLUpdateAbleObject)
  private
     
    FEnabled: Boolean;
    FMinSize, FMaxSize: Single;
    FFadeTresholdSize: Single;
    FDistanceAttenuation: TGLCoordinates;

  protected
     
    procedure SetEnabled(const val: Boolean);
    procedure SetMinSize(const val: Single);
    procedure SetMaxSize(const val: Single);
    procedure SetFadeTresholdSize(const val: Single);
    procedure SetDistanceAttenuation(const val: TGLCoordinates);

    procedure DefineProperties(Filer: TFiler); override;
    procedure ReadData(Stream: TStream);
    procedure WriteData(Stream: TStream);

  public
     
    constructor Create(AOwner: TPersistent); override;
    destructor Destroy; override;

    procedure Assign(Source: TPersistent); override;

    procedure Apply;
    procedure UnApply;

  published
     
    property Enabled: Boolean read FEnabled write SetEnabled default False;
    property MinSize: Single read FMinSize write SetMinSize stored False;
    property MaxSize: Single read FMaxSize write SetMaxSize stored False;
    property FadeTresholdSize: Single read FFadeTresholdSize
      write SetFadeTresholdSize stored False;
    { : Components XYZ are for constant, linear and quadratic attenuation. }
    property DistanceAttenuation: TGLCoordinates read FDistanceAttenuation
      write SetDistanceAttenuation;
  end;

  // TGLPoints
  //
  { : Renders a set of non-transparent colored points.
    The points positions and their color are defined through the Positions
    and Colors properties. }
  TGLPoints = class(TGLImmaterialSceneObject)
  private
     
    FPositions: TAffineVectorList;
    FColors: TVectorList;
    FSize: Single;
    FStyle: TGLPointStyle;
    FPointParameters: TGLPointParameters;
    FStatic, FNoZWrite: Boolean;

  protected
     
    function StoreSize: Boolean;
    procedure SetNoZWrite(const val: Boolean);
    procedure SetStatic(const val: Boolean);
    procedure SetSize(const val: Single);
    procedure SetPositions(const val: TAffineVectorList);
    procedure SetColors(const val: TVectorList);
    procedure SetStyle(const val: TGLPointStyle);
    procedure SetPointParameters(const val: TGLPointParameters);

  public
     
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;

    procedure Assign(Source: TPersistent); override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;

    { : Points positions.
      If empty, a single point is assumed at (0, 0, 0) }
    property Positions: TAffineVectorList read FPositions write SetPositions;
    { : Defines the points colors.
       
       if empty, point color will be opaque white
       if contains a single color, all points will use that color
       if contains N colors, the first N points (at max) will be rendered
      using the corresponding colors.
        }
    property Colors: TVectorList read FColors write SetColors;

  published
     
    { : If true points do not write their Z to the depth buffer. }
    property NoZWrite: Boolean read FNoZWrite write SetNoZWrite;
    { : Tells the component if point coordinates are static.
      If static, changes to the positions should be notified via an
      explicit StructureChanged call, or may not refresh. 
      Static sets of points may render faster than dynamic ones. }
    property Static: Boolean read FStatic write SetStatic;
    { : Point size, all points have a fixed size. }
    property Size: Single read FSize write SetSize stored StoreSize;
    { : Points style. }
    property Style: TGLPointStyle read FStyle write SetStyle default psSquare;
    { : Point parameters as of ARB_point_parameters.
      Allows to vary the size and transparency of points depending
      on their distance to the observer. }
    property PointParameters: TGLPointParameters read FPointParameters
      write SetPointParameters;

  end;

  // TLineNodesAspect
  //
  { : Possible aspects for the nodes of a TLine. }
  TLineNodesAspect = (lnaInvisible, lnaAxes, lnaCube, lnaDodecahedron);

  // TGLLineSplineMode
  //
  { : Available spline modes for a TLine. }
  TGLLineSplineMode = (lsmLines, lsmCubicSpline, lsmBezierSpline, lsmNURBSCurve,
    lsmSegments, lsmLoop);

  // TGLLinesNode
  //
  { : Specialized Node for use in a TGLLines objects.
    Adds a Color property (TGLColor). }
  TGLLinesNode = class(TGLNode)
  private
     
    FColor: TGLColor;

  protected
     
    procedure SetColor(const val: TGLColor);
    procedure OnColorChange(Sender: TObject);
    function StoreColor: Boolean;

  public
     
    constructor Create(Collection: TCollection); override;
    destructor Destroy; override;
    procedure Assign(Source: TPersistent); override;

  published
     

    { : The node color.
      Can also defined the line color (interpolated between nodes) if
      loUseNodeColorForLines is set (in TGLLines). }
    property Color: TGLColor read FColor write SetColor stored StoreColor;
  end;

  // TGLLinesNodes
  //
  { : Specialized collection for Nodes in a TGLLines objects.
    Stores TGLLinesNode items. }
  TGLLinesNodes = class(TGLNodes)
  public
     
    constructor Create(AOwner: TComponent); overload;

    procedure NotifyChange; override;
  end;

  // TGLLineBase
  //
  { : Base class for line objects.
    Introduces line style properties (width, color...). }
  TGLLineBase = class(TGLImmaterialSceneObject)
  private
     
    FLineColor: TGLColor;
    FLinePattern: TGLushort;
    FLineWidth: Single;
    FAntiAliased: Boolean;

  protected
     
    procedure SetLineColor(const Value: TGLColor);
    procedure SetLinePattern(const Value: TGLushort);
    procedure SetLineWidth(const val: Single);
    function StoreLineWidth: Boolean;
    procedure SetAntiAliased(const val: Boolean);

    { : Setup OpenGL states according to line style.
      You must call RestoreLineStyle after drawing your lines.
      You may use nested calls with SetupLineStyle/RestoreLineStyle. }
    procedure SetupLineStyle(var rci: TGLRenderContextInfo);

  public
     
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure Assign(Source: TPersistent); override;
    procedure NotifyChange(Sender: TObject); override;

  published
     
    { : Indicates if OpenGL should smooth line edges.
      Smoothed lines looks better but are poorly implemented in most OpenGL
      drivers and take *lots* of rendering time. }
    property AntiAliased: Boolean read FAntiAliased write SetAntiAliased
      default False;
    { : Default color of the lines. }
    property LineColor: TGLColor read FLineColor write SetLineColor;
    { : Bitwise line pattern.
      For instance $FFFF (65535) is a white line (stipple disabled), $0000
      is a black line, $CCCC is the stipple used in axes and dummycube, etc. }
    property LinePattern: TGLushort read FLinePattern write SetLinePattern
      default $FFFF;
    { : Default width of the lines. }
    property LineWidth: Single read FLineWidth write SetLineWidth
      stored StoreLineWidth;
    property Visible;
  end;

  // TGLNodedLines
  //
  { : Class that defines lines via a series of nodes.
    Base class, does not render anything. }
  TGLNodedLines = class(TGLLineBase)
  private
     
    FNodes: TGLLinesNodes;
    FNodesAspect: TLineNodesAspect;
    FNodeColor: TGLColor;
    FNodeSize: Single;
    FOldNodeColor: TColorVector;

  protected
     
    procedure SetNodesAspect(const Value: TLineNodesAspect);
    procedure SetNodeColor(const Value: TGLColor);
    procedure OnNodeColorChanged(Sender: TObject);
    procedure SetNodes(const aNodes: TGLLinesNodes);
    procedure SetNodeSize(const val: Single);
    function StoreNodeSize: Boolean;

    procedure DrawNode(var rci: TGLRenderContextInfo; X, Y, Z: Single;
      Color: TGLColor);

  public
     
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure Assign(Source: TPersistent); override;

    function AxisAlignedDimensionsUnscaled: TVector; override;

    procedure AddNode(const coords: TGLCoordinates); overload;
    procedure AddNode(const X, Y, Z: TGLFloat); overload;
    procedure AddNode(const Value: TVector); overload;
    procedure AddNode(const Value: TAffineVector); overload;

  published
     
    { : Default color for nodes.
      lnaInvisible and lnaAxes ignore this setting. }
    property NodeColor: TGLColor read FNodeColor write SetNodeColor;
    { : The nodes list. }
    property Nodes: TGLLinesNodes read FNodes write SetNodes;

    { : Default aspect of line nodes.
      May help you materialize nodes, segments and control points. }
    property NodesAspect: TLineNodesAspect read FNodesAspect
      write SetNodesAspect default lnaAxes;
    { : Size for the various node aspects. }
    property NodeSize: Single read FNodeSize write SetNodeSize
      stored StoreNodeSize;
  end;

  // TLinesOptions
  //
  TLinesOption = (loUseNodeColorForLines, loColorLogicXor);
  TLinesOptions = set of TLinesOption;

  // TGLLines
  //
  { : Set of 3D line segments.
    You define a 3D Line by adding its nodes in the "Nodes" property. The line
    may be rendered as a set of segment or as a curve (nodes then act as spline
    control points).
    Alternatively, you can also use it to render a set of spacial nodes (points
    in space), just make the lines transparent and the nodes visible by picking
    the node aspect that suits you. }
  TGLLines = class(TGLNodedLines)
  private
     
    FDivision: Integer;
    FSplineMode: TGLLineSplineMode;
    FOptions: TLinesOptions;
    FNURBSOrder: Integer;
    FNURBSTolerance: Single;
    FNURBSKnots: TSingleList;

  protected
     
    procedure SetSplineMode(const val: TGLLineSplineMode);
    procedure SetDivision(const Value: Integer);
    procedure SetOptions(const val: TLinesOptions);
    procedure SetNURBSOrder(const val: Integer);
    procedure SetNURBSTolerance(const val: Single);

  public
     
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure Assign(Source: TPersistent); override;

    procedure BuildList(var rci: TGLRenderContextInfo); override;

    property NURBSKnots: TSingleList read FNURBSKnots;
    property NURBSOrder: Integer read FNURBSOrder write SetNURBSOrder;
    property NURBSTolerance: Single read FNURBSTolerance
      write SetNURBSTolerance;

  published
     
    { : Number of divisions for each segment in spline modes.
      Minimum 1 (disabled), ignored in lsmLines mode. }
    property Division: Integer read FDivision write SetDivision default 10;
    { : Default spline drawing mode. }
    property SplineMode: TGLLineSplineMode read FSplineMode write SetSplineMode
      default lsmLines;

    { : Rendering options for the line.
       
       loUseNodeColorForLines: if set lines will be drawn using node
      colors (and color interpolation between nodes), if not, LineColor
      will be used (single color).
      loColorLogicXor: enable logic operation for color of XOR type.
        }
    property Options: TLinesOptions read FOptions write SetOptions;
  end;

  TCubePart = (cpTop, cpBottom, cpFront, cpBack, cpLeft, cpRight);
  TCubeParts = set of TCubePart;

  // TGLCube
  //
  { : A simple cube object.
    This cube use the same material for each of its faces, ie. all faces look
    the same. If you want a multi-material cube, use a mesh in conjunction
    with a TGLFreeForm and a material library. }
  TGLCube = class(TGLSceneObject)
  private
     
    FCubeSize: TAffineVector;
    FParts: TCubeParts;
    FNormalDirection: TNormalDirection;
    function GetCubeWHD(const Index: Integer): TGLFloat;
    procedure SetCubeWHD(Index: Integer; AValue: TGLFloat);
    procedure SetParts(aValue: TCubeParts);
    procedure SetNormalDirection(aValue: TNormalDirection);
  protected
     
    procedure DefineProperties(Filer: TFiler); override;
    procedure ReadData(Stream: TStream);
    procedure WriteData(Stream: TStream);

  public
     
    constructor Create(AOwner: TComponent); override;

    function GenerateSilhouette(const silhouetteParameters
      : TGLSilhouetteParameters): TGLSilhouette; override;
    procedure BuildList(var rci: TGLRenderContextInfo); override;

    procedure Assign(Source: TPersistent); override;
    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;

  published
     
    property CubeWidth: TGLFloat index 0 read GetCubeWHD write SetCubeWHD
      stored False;
    property CubeHeight: TGLFloat index 1 read GetCubeWHD write SetCubeWHD
      stored False;
    property CubeDepth: TGLFloat index 2 read GetCubeWHD write SetCubeWHD
      stored False;
    property NormalDirection: TNormalDirection read FNormalDirection
      write SetNormalDirection default ndOutside;
    property Parts: TCubeParts read FParts write SetParts
      default [cpTop, cpBottom, cpFront, cpBack, cpLeft, cpRight];
  end;

  // TNormalSmoothing
  //
  { : Determines how and if normals are smoothed.
    - nsFlat : facetted look 
    - nsSmooth : smooth look 
    - nsNone : unlighted rendering, usefull for decla texturing }
  TNormalSmoothing = (nsFlat, nsSmooth, nsNone);

  // TGLQuadricObject
  //
  { : Base class for quadric objects.
    Introduces some basic Quadric interaction functions (the actual quadric
    math is part of the GLU library). }
  TGLQuadricObject = class(TGLSceneObject)
  private
     
    FNormals: TNormalSmoothing;
    FNormalDirection: TNormalDirection;

  protected
     
    procedure SetNormals(aValue: TNormalSmoothing);
    procedure SetNormalDirection(aValue: TNormalDirection);
    procedure SetupQuadricParams(quadric: PGLUquadricObj);
    procedure SetNormalQuadricOrientation(quadric: PGLUquadricObj);
    procedure SetInvertedQuadricOrientation(quadric: PGLUquadricObj);

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure Assign(Source: TPersistent); override;

  published
     
    property Normals: TNormalSmoothing read FNormals write SetNormals
      default nsSmooth;
    property NormalDirection: TNormalDirection read FNormalDirection
      write SetNormalDirection default ndOutside;
  end;

  TAngleLimit1 = -90 .. 90;
  TAngleLimit2 = 0 .. 360;
  TCapType = (ctNone, ctCenter, ctFlat);

  // TGLSphere
  //
  { : A sphere object.
    The sphere can have to and bottom caps, as well as being just a slice
    of sphere. }
  TGLSphere = class(TGLQuadricObject)
  private
     
    FRadius: TGLFloat;
    FSlices, FStacks: TGLInt;
    FTop: TAngleLimit1;
    FBottom: TAngleLimit1;
    FStart: TAngleLimit2;
    FStop: TAngleLimit2;
    FTopCap, FBottomCap: TCapType;
    procedure SetBottom(aValue: TAngleLimit1);
    procedure SetBottomCap(aValue: TCapType);
    procedure SetRadius(const aValue: TGLFloat);
    procedure SetSlices(aValue: TGLInt);
    procedure SetStart(aValue: TAngleLimit2);
    procedure SetStop(aValue: TAngleLimit2);
    procedure SetStacks(aValue: TGLInt);
    procedure SetTop(aValue: TAngleLimit1);
    procedure SetTopCap(aValue: TCapType);

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure Assign(Source: TPersistent); override;

    procedure BuildList(var rci: TGLRenderContextInfo); override;
    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;

    function GenerateSilhouette(const silhouetteParameters
      : TGLSilhouetteParameters): TGLSilhouette; override;
  published
     
    property Bottom: TAngleLimit1 read FBottom write SetBottom default -90;
    property BottomCap: TCapType read FBottomCap write SetBottomCap
      default ctNone;
    property Radius: TGLFloat read FRadius write SetRadius;
    property Slices: TGLInt read FSlices write SetSlices default 16;
    property Stacks: TGLInt read FStacks write SetStacks default 16;
    property Start: TAngleLimit2 read FStart write SetStart default 0;
    property Stop: TAngleLimit2 read FStop write SetStop default 360;
    property Top: TAngleLimit1 read FTop write SetTop default 90;
    property TopCap: TCapType read FTopCap write SetTopCap default ctNone;
  end;

  // TGLPolygonBase
  //
  { : Base class for objects based on a polygon. }
  TGLPolygonBase = class(TGLSceneObject)
  private
     
    FDivision: Integer;
    FSplineMode: TGLLineSplineMode;

  protected
     
    FNodes: TGLNodes;
    procedure CreateNodes; dynamic;
    procedure SetSplineMode(const val: TGLLineSplineMode);
    procedure SetDivision(const Value: Integer);
    procedure SetNodes(const aNodes: TGLNodes);

  public
     
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;
    procedure Assign(Source: TPersistent); override;
    procedure NotifyChange(Sender: TObject); override;

    procedure AddNode(const coords: TGLCoordinates); overload;
    procedure AddNode(const X, Y, Z: TGLFloat); overload;
    procedure AddNode(const Value: TVector); overload;
    procedure AddNode(const Value: TAffineVector); overload;

  published
     
    { : The nodes list. }
    property Nodes: TGLNodes read FNodes write SetNodes;
    { : Number of divisions for each segment in spline modes.
      Minimum 1 (disabled), ignored in lsmLines mode. }
    property Division: Integer read FDivision write SetDivision default 10;
    { : Default spline drawing mode.
      This mode is used only for the curve, not for the rotation path. }
    property SplineMode: TGLLineSplineMode read FSplineMode write SetSplineMode
      default lsmLines;

  end;

  // TGLSuperellipsoid
  //
  { : A Superellipsoid object.
    The Superellipsoid can have top and bottom caps,
    as well as being just a slice of Superellipsoid. }
  TGLSuperellipsoid = class(TGLQuadricObject)
  private
     
    FRadius, FxyCurve, FzCurve: TGLFloat;
    FSlices, FStacks: TGLInt;
    FTop: TAngleLimit1;
    FBottom: TAngleLimit1;
    FStart: TAngleLimit2;
    FStop: TAngleLimit2;
    FTopCap, FBottomCap: TCapType;
    procedure SetBottom(aValue: TAngleLimit1);
    procedure SetBottomCap(aValue: TCapType);
    procedure SetRadius(const aValue: TGLFloat);
    procedure SetxyCurve(const aValue: TGLFloat);
    procedure SetzCurve(const aValue: TGLFloat);
    procedure SetSlices(aValue: TGLInt);
    procedure SetStart(aValue: TAngleLimit2);
    procedure SetStop(aValue: TAngleLimit2);
    procedure SetStacks(aValue: TGLInt);
    procedure SetTop(aValue: TAngleLimit1);
    procedure SetTopCap(aValue: TCapType);

  public
     
    constructor Create(AOwner: TComponent); override;
    procedure Assign(Source: TPersistent); override;

    procedure BuildList(var rci: TGLRenderContextInfo); override;
    function AxisAlignedDimensionsUnscaled: TVector; override;
    function RayCastIntersect(const rayStart, rayVector: TVector;
      intersectPoint: PVector = nil; intersectNormal: PVector = nil)
      : Boolean; override;

    function GenerateSilhouette(const silhouetteParameters
      : TGLSilhouetteParameters): TGLSilhouette; override;
  published
     
    property Bottom: TAngleLimit1 read FBottom write SetBottom default -90;
    property BottomCap: TCapType read FBottomCap write SetBottomCap
      default ctNone;
    property Radius: TGLFloat read FRadius write SetRadius;
    property xyCurve: TGLFloat read FxyCurve write SetxyCurve;
    property zCurve: TGLFloat read FzCurve write SetzCurve;
    property Slices: TGLInt read FSlices write SetSlices default 16;
    property Stacks: TGLInt read FStacks write SetStacks default 16;
    property Start: TAngleLimit2 read FStart write SetStart default 0;
    property Stop: TAngleLimit2 read FStop write SetStop default 360;
    property Top: TAngleLimit1 read FTop write SetTop default 90;
    property TopCap: TCapType read FTopCap write SetTopCap default ctNone;
  end;


{ : Issues OpenGL for a unit-size cube stippled wireframe. }
procedure CubeWireframeBuildList(var rci: TGLRenderContextInfo; Size: TGLFloat;
  Stipple: Boolean; const Color: TColorVector);
{ : Issues OpenGL for a unit-size dodecahedron. }
procedure DodecahedronBuildList;
{ : Issues OpenGL for a unit-size icosahedron. }
procedure IcosahedronBuildList;
{ : Issues OpenGL for a unit-size octahedron. }
procedure OctahedronBuildList;
{ : Issues OpenGL for a unit-size tetrahedron. }
procedure TetrahedronBuildList;



var
  TangentAttributeName: AnsiString = 'Tangent';
  BinormalAttributeName: AnsiString = 'Binormal';

// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------
implementation

// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------

uses
  GLSpline,
  XOpenGL,
  GLState;

const
  cDefaultPointSize: Single = 1.0;

  // CubeWireframeBuildList
  //

procedure CubeWireframeBuildList(var rci: TGLRenderContextInfo; Size: TGLFloat;
  Stipple: Boolean; const Color: TColorVector);
var
  mi, ma: Single;
begin
{$IFDEF GLS_OPENGL_DEBUG}
  if GL.GREMEDY_string_marker then
    GL.StringMarkerGREMEDY(22, 'CubeWireframeBuildList');
{$ENDIF}
  rci.GLStates.Disable(stLighting);
  rci.GLStates.Enable(stLineSmooth);
  if stipple then
  begin
    rci.GLStates.Enable(stLineStipple);
    rci.GLStates.Enable(stBlend);
    rci.GLStates.SetBlendFunc(bfSrcAlpha, bfOneMinusSrcAlpha);
    rci.GLStates.LineStippleFactor := 1;
    rci.GLStates.LineStipplePattern := $CCCC;
  end;
  rci.GLStates.LineWidth := 1;
  ma := 0.5 * Size;
  mi := -ma;

  GL.Color4fv(@Color);
  GL.Begin_(GL_LINE_STRIP);
  // front face
  GL.Vertex3f(ma, mi, mi);
  GL.Vertex3f(ma, ma, mi);
  GL.Vertex3f(ma, ma, ma);
  GL.Vertex3f(ma, mi, ma);
  GL.Vertex3f(ma, mi, mi);
  // partial up back face
  GL.Vertex3f(mi, mi, mi);
  GL.Vertex3f(mi, mi, ma);
  GL.Vertex3f(mi, ma, ma);
  GL.Vertex3f(mi, ma, mi);
  // right side low
  GL.Vertex3f(ma, ma, mi);
  GL.End_;
  GL.Begin_(GL_LINES);
  // right high
  GL.Vertex3f(ma, ma, ma);
  GL.Vertex3f(mi, ma, ma);
  // back low
  GL.Vertex3f(mi, mi, mi);
  GL.Vertex3f(mi, ma, mi);
  // left high
  GL.Vertex3f(ma, mi, ma);
  GL.Vertex3f(mi, mi, ma);
  GL.End_;
end;

// DodecahedronBuildList
//
procedure DodecahedronBuildList;
const
  A = 1.61803398875 * 0.3; // (Sqrt(5)+1)/2
  B = 0.61803398875 * 0.3; // (Sqrt(5)-1)/2
  C = 1 * 0.3;
const
  Vertices: packed array [0 .. 19] of TAffineVector = ((X: - A; Y: 0; Z: B),
    (X: - A; Y: 0; Z: - B), (X: A; Y: 0; Z: - B), (X: A; Y: 0; Z: B), (X: B;
    Y: - A; Z: 0), (X: - B; Y: - A; Z: 0), (X: - B; Y: A; Z: 0), (X: B; Y: A;
    Z: 0), (X: 0; Y: B; Z: - A), (X: 0; Y: - B; Z: - A), (X: 0; Y: - B; Z: A),
    (X: 0; Y: B; Z: A), (X: - C; Y: - C; Z: C), (X: - C; Y: - C; Z: - C), (X: C;
    Y: - C; Z: - C), (X: C; Y: - C; Z: C), (X: - C; Y: C; Z: C), (X: - C; Y: C;
    Z: - C), (X: C; Y: C; Z: - C), (X: C; Y: C; Z: C));

  Polygons: packed array [0 .. 11] of packed array [0 .. 4]
    of Byte = ((0, 12, 10, 11, 16), (1, 17, 8, 9, 13), (2, 14, 9, 8, 18),
    (3, 19, 11, 10, 15), (4, 14, 2, 3, 15), (5, 12, 0, 1, 13),
    (6, 17, 1, 0, 16), (7, 19, 3, 2, 18), (8, 17, 6, 7, 18), (9, 14, 4, 5, 13),
    (10, 12, 5, 4, 15), (11, 19, 7, 6, 16));
var
  i, j: Integer;
  n: TAffineVector;
  faceIndices: PByteArray;
begin
  for i := 0 to 11 do
  begin
    faceIndices := @polygons[i, 0];

    n := CalcPlaneNormal(vertices[faceIndices^[0]], vertices[faceIndices^[1]],
      vertices[faceIndices^[2]]);
    GL.Normal3fv(@n);

//    GL.Begin_(GL_TRIANGLE_FAN);
//    for j := 0 to 4 do
//      GL.Vertex3fv(@vertices[faceIndices^[j]]);
//    GL.End_;

    GL.Begin_(GL_TRIANGLES);

    for j := 1 to 3 do
    begin
      GL.Vertex3fv(@vertices[faceIndices^[0]]);
      GL.Vertex3fv(@vertices[faceIndices^[j]]);
      GL.Vertex3fv(@vertices[faceIndices^[j+1]]);
    end;
    GL.End_;
  end;
end;

// IcosahedronBuildList
//
procedure IcosahedronBuildList;
const
  A = 0.5;
  B = 0.30901699437; // 1/(1+Sqrt(5))
const
  Vertices: packed array [0 .. 11] of TAffineVector = ((X: 0; Y: - B; Z: - A),
    (X: 0; Y: - B; Z: A), (X: 0; Y: B; Z: - A), (X: 0; Y: B; Z: A), (X: - A;
    Y: 0; Z: - B), (X: - A; Y: 0; Z: B), (X: A; Y: 0; Z: - B), (X: A; Y: 0;
    Z: B), (X: - B; Y: - A; Z: 0), (X: - B; Y: A; Z: 0), (X: B; Y: - A; Z: 0),
    (X: B; Y: A; Z: 0));
  Triangles: packed array [0 .. 19] of packed array [0 .. 2]
    of Byte = ((2, 9, 11), (3, 11, 9), (3, 5, 1), (3, 1, 7), (2, 6, 0),
    (2, 0, 4), (1, 8, 10), (0, 10, 8), (9, 4, 5), (8, 5, 4), (11, 7, 6),
    (10, 6, 7), (3, 9, 5), (3, 7, 11), (2, 4, 9), (2, 11, 6), (0, 8, 4),
    (0, 6, 10), (1, 5, 8), (1, 10, 7));

var
  i, j: Integer;
  n: TAffineVector;
  faceIndices: PByteArray;
begin
  for i := 0 to 19 do
  begin
    faceIndices := @triangles[i, 0];

    n := CalcPlaneNormal(vertices[faceIndices^[0]], vertices[faceIndices^[1]],
      vertices[faceIndices^[2]]);
    GL.Normal3fv(@n);

    GL.Begin_(GL_TRIANGLES);
    for j := 0 to 2 do
      GL.Vertex3fv(@vertices[faceIndices^[j]]);
    GL.End_;
  end;
end;

// OctahedronBuildList
//
procedure OctahedronBuildList;
const
  Vertices: packed array [0 .. 5] of TAffineVector =
      ((X: 1.0; Y: 0.0; Z: 0.0),
       (X: -1.0; Y: 0.0; Z: 0.0),
       (X: 0.0; Y: 1.0; Z: 0.0),
       (X: 0.0; Y: -1.0; Z: 0.0),
       (X: 0.0; Y: 0.0; Z: 1.0),
       (X: 0.0; Y: 0.0; Z: -1.0));

  Triangles: packed array [0 .. 7] of packed array [0 .. 2]
    of Byte = ((0, 4, 2), (1, 2, 4), (0, 3, 4), (1, 4, 3),
               (0, 2, 5), (1, 5, 2), (0, 5, 3), (1, 3, 5));

var
  i, j: Integer;
  n: TAffineVector;
  faceIndices: PByteArray;
begin
  for i := 0 to 7 do
  begin
    faceIndices := @triangles[i, 0];

    n := CalcPlaneNormal(vertices[faceIndices^[0]], vertices[faceIndices^[1]],
      vertices[faceIndices^[2]]);
    GL.Normal3fv(@n);

    GL.Begin_(GL_TRIANGLES);
    for j := 0 to 2 do
      GL.Vertex3fv(@vertices[faceIndices^[j]]);
    GL.End_;
  end;
end;

// TetrahedronBuildList
//
procedure TetrahedronBuildList;
const
  TetT = 1.73205080756887729;
const
  Vertices: packed array [0 .. 3] of TAffineVector =
{
       ((X: TetT;  Y: TetT;  Z: TetT),
        (X: TetT;  Y: -TetT; Z: -TetT),
        (X: -TetT; Y: TetT;  Z: -TetT),
        (X: -TetT; Y: -TetT; Z: TetT));
}
       ((X: 1.0;  Y: 1.0;  Z: 1.0),
        (X: 1.0;  Y: -1.0; Z: -1.0),
        (X: -1.0; Y: 1.0;  Z: -1.0),
        (X: -1.0; Y: -1.0; Z: 1.0));

  Triangles: packed array [0 .. 3] of packed array [0 .. 2]
    of Byte = ((0, 1, 3), (2, 1, 0), (3, 2, 0), (1, 2, 3));

var
  i, j: Integer;
  n: TAffineVector;
  faceIndices: PByteArray;
begin
  for i := 0 to 3 do
  begin
    faceIndices := @triangles[i, 0];

    n := CalcPlaneNormal(vertices[faceIndices^[0]], vertices[faceIndices^[1]],
      vertices[faceIndices^[2]]);
    GL.Normal3fv(@n);

    GL.Begin_(GL_TRIANGLES);
    for j := 0 to 2 do
      GL.Vertex3fv(@vertices[faceIndices^[j]]);
    GL.End_;
  end;
end;

// ------------------
// ------------------ TGLDummyCube ------------------
// ------------------

// Create
//

constructor TGLDummyCube.Create(AOwner: TComponent);
begin
  inherited;
  ObjectStyle := ObjectStyle + [osDirectDraw];
  FCubeSize := 1;
  FEdgeColor := TGLColor.Create(Self);
  FEdgeColor.Initialize(clrWhite);
  FGroupList := TGLListHandle.Create;
  CamInvarianceMode := cimNone;
end;

// Destroy
//

destructor TGLDummyCube.Destroy;
begin
  FGroupList.Free;
  FEdgeColor.Free;
  inherited;
end;

 
//

procedure TGLDummyCube.Assign(Source: TPersistent);
begin
  if Source is TGLDummyCube then
  begin
    FCubeSize := TGLDummyCube(Source).FCubeSize;
    FEdgeColor.Color := TGLDummyCube(Source).FEdgeColor.Color;
    FVisibleAtRunTime := TGLDummyCube(Source).FVisibleAtRunTime;
    NotifyChange(Self);
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensionsUnscaled
//

function TGLDummyCube.AxisAlignedDimensionsUnscaled: TVector;
begin
  Result.X := 0.5 * Abs(FCubeSize);
  Result.Y := Result.X;
  Result.Z := Result.X;
  Result.W := 0;
end;

// RayCastIntersect
//

function TGLDummyCube.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
begin
  Result := False;
end;

// BuildList
//

procedure TGLDummyCube.BuildList(var rci: TGLRenderContextInfo);
begin
  if (csDesigning in ComponentState) or (FVisibleAtRunTime) then
    CubeWireframeBuildList(rci, FCubeSize, True, EdgeColor.Color);
end;

// DoRender
//

procedure TGLDummyCube.DoRender(var rci: TGLRenderContextInfo;
  renderSelf, renderChildren: Boolean);
begin
  if Assigned(FOnVisibilityDetermination) then
    if not FOnVisibilityDetermination(Self, rci) then
      Exit;
  if FAmalgamate and (not rci.amalgamating) then
  begin
    if FGroupList.Handle = 0 then
    begin
      FGroupList.AllocateHandle;
      Assert(FGroupList.Handle <> 0, 'Handle=0 for ' + ClassName);
      rci.GLStates.NewList(FGroupList.Handle, GL_COMPILE);
      rci.amalgamating := True;
      try
        inherited;
      finally
        rci.amalgamating := False;
        rci.GLStates.EndList;
      end;
    end;
    rci.GLStates.CallList(FGroupList.Handle);
  end
  else
  begin
    // proceed as usual
    inherited;
  end;
end;

// StructureChanged
//

procedure TGLDummyCube.StructureChanged;
begin
  if FAmalgamate then
    FGroupList.DestroyHandle;
  inherited;
end;

// BarycenterAbsolutePosition
//

function TGLDummyCube.BarycenterAbsolutePosition: TVector;
var
  i: Integer;
begin
  if Count > 0 then
  begin
    Result := Children[0].BarycenterAbsolutePosition;
    for i := 1 to Count - 1 do
      Result := VectorAdd(Result, Children[i].BarycenterAbsolutePosition);
    ScaleVector(Result, 1 / Count);
  end
  else
    Result := AbsolutePosition;
end;

// SetCubeSize
//

procedure TGLDummyCube.SetCubeSize(const val: TGLFloat);
begin
  if val <> FCubeSize then
  begin
    FCubeSize := val;
    StructureChanged;
  end;
end;

// SetEdgeColor
//

procedure TGLDummyCube.SetEdgeColor(const val: TGLColor);
begin
  if val <> FEdgeColor then
  begin
    FEdgeColor.Assign(val);
    StructureChanged;
  end;
end;

// SetVisibleAtRunTime
//

procedure TGLDummyCube.SetVisibleAtRunTime(const val: Boolean);
begin
  if val <> FVisibleAtRunTime then
  begin
    FVisibleAtRunTime := val;
    StructureChanged;
  end;
end;

// SetAmalgamate
//

procedure TGLDummyCube.SetAmalgamate(const val: Boolean);
begin
  if val <> FAmalgamate then
  begin
    FAmalgamate := val;
    if not val then
      FGroupList.DestroyHandle;
    inherited StructureChanged;
  end;
end;

// ------------------
// ------------------ TGLPlane ------------------
// ------------------

// Create
//

constructor TGLPlane.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FWidth := 1;
  FHeight := 1;
  FXTiles := 1;
  FYTiles := 1;
  FXScope := 1;
  FYScope := 1;
  ObjectStyle := ObjectStyle + [osDirectDraw];
  FStyle := [psSingleQuad, psTileTexture];
end;

 
//

procedure TGLPlane.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLPlane) then
  begin
    FWidth := TGLPlane(Source).FWidth;
    FHeight := TGLPlane(Source).FHeight;
    FXOffset := TGLPlane(Source).FXOffset;
    FXScope := TGLPlane(Source).FXScope;
    FXTiles := TGLPlane(Source).FXTiles;
    FYOffset := TGLPlane(Source).FYOffset;
    FYScope := TGLPlane(Source).FYScope;
    FYTiles := TGLPlane(Source).FYTiles;
    FStyle := TGLPlane(Source).FStyle;
    StructureChanged;
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensions
//

function TGLPlane.AxisAlignedDimensionsUnscaled: TVector;
begin
  Result.V[0] := 0.5 * Abs(FWidth);
  Result.V[1] := 0.5 * Abs(FHeight);
  Result.V[2] := 0;
end;

// RayCastIntersect
//

function TGLPlane.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
var
  locRayStart, locRayVector, ip: TVector;
  t: Single;
begin
  locRayStart := AbsoluteToLocal(rayStart);
  locRayVector := AbsoluteToLocal(rayVector);
  if locRayStart.V[2] >= 0 then
  begin
    // ray start over plane
    if locRayVector.V[2] < 0 then
    begin
      t := locRayStart.V[2] / locRayVector.V[2];
      ip.V[0] := locRayStart.V[0] - t * locRayVector.V[0];
      ip.V[1] := locRayStart.V[1] - t * locRayVector.V[1];
      if (Abs(ip.V[0]) <= 0.5 * Width) and (Abs(ip.V[1]) <= 0.5 * Height) then
      begin
        Result := True;
        if Assigned(intersectNormal) then
          intersectNormal^ := AbsoluteDirection;
      end
      else
        Result := False;
    end
    else
      Result := False;
  end
  else
  begin
    // ray start below plane
    if locRayVector.V[2] > 0 then
    begin
      t := locRayStart.V[2] / locRayVector.V[2];
      ip.V[0] := locRayStart.V[0] - t * locRayVector.V[0];
      ip.V[1] := locRayStart.V[1] - t * locRayVector.V[1];
      if (Abs(ip.V[0]) <= 0.5 * Width) and (Abs(ip.V[1]) <= 0.5 * Height) then
      begin
        Result := True;
        if Assigned(intersectNormal) then
          intersectNormal^ := VectorNegate(AbsoluteDirection);
      end
      else
        Result := False;
    end
    else
      Result := False;
  end;
  if Result and Assigned(intersectPoint) then
  begin
    ip.V[2] := 0;
    ip.V[3] := 1;
    intersectPoint^ := LocalToAbsolute(ip);
  end;
end;

// GenerateSilhouette
//

function TGLPlane.GenerateSilhouette(const silhouetteParameters
  : TGLSilhouetteParameters): TGLSilhouette;
var
  hw, hh: Single;
begin
  Result := TGLSilhouette.Create;

  hw := FWidth * 0.5;
  hh := FHeight * 0.5;

  with Result.vertices do
  begin
    AddPoint(hw, hh);
    AddPoint(hw, -hh);
    AddPoint(-hw, -hh);
    AddPoint(-hw, hh);
  end;

  with Result.Indices do
  begin
    Add(0, 1);
    Add(1, 2);
    Add(2, 3);
    Add(3, 0);
  end;

  if silhouetteParameters.CappingRequired then
    with Result.CapIndices do
    begin
      Add(0, 1, 2);
      Add(2, 3, 0);
    end;
end;

// BuildList
//

procedure TGLPlane.BuildList(var rci: TGLRenderContextInfo);

  procedure EmitVertex(ptr: PVertexRec); {$IFDEF GLS_INLINE}inline;{$ENDIF}
  begin
    XGL.TexCoord2fv(@ptr^.TexCoord);
    GL.Vertex3fv(@ptr^.Position);
  end;

var
  hw, hh, posXFact, posYFact, pX, pY1: TGLFloat;
  tx0, tx1, ty0, ty1, texSFact, texTFact: TGLFloat;
  texS, texT1: TGLFloat;
  X, Y: Integer;
  TanLoc, BinLoc: Integer;
  pVertex: PVertexRec;
begin
  hw := FWidth * 0.5;
  hh := FHeight * 0.5;

  with GL do
  begin
    Normal3fv(@ZVector);
    if ARB_shader_objects and (rci.GLStates.CurrentProgram > 0) then
    begin
      TanLoc := GetAttribLocation(rci.GLStates.CurrentProgram, PGLChar(TangentAttributeName));
      BinLoc := GetAttribLocation(rci.GLStates.CurrentProgram, PGLChar(BinormalAttributeName));
      if TanLoc > -1 then
        VertexAttrib3fv(TanLoc, @XVector);
      if BinLoc > -1 then
        VertexAttrib3fv(BinLoc, @YVector);
    end;
  end;
  // determine tex coords extents
  if psTileTexture in FStyle then
  begin
    tx0 := FXOffset;
    tx1 := FXTiles * FXScope + FXOffset;
    ty0 := FYOffset;
    ty1 := FYTiles * FYScope + FYOffset;
  end
  else
  begin
    tx0 := 0;
    ty0 := tx0;
    tx1 := FXScope;
    ty1 := FYScope;
  end;

  if psSingleQuad in FStyle then
  begin
    // single quad plane
    GL.Begin_(GL_TRIANGLES);
    xgl.TexCoord2f(tx1, ty1);
    GL.Vertex2f(hw, hh);
    xgl.TexCoord2f(tx0, ty1);
    GL.Vertex2f(-hw, hh);
    xgl.TexCoord2f(tx0, ty0);
    GL.Vertex2f(-hw, -hh);

    GL.Vertex2f(-hw, -hh);
    xgl.TexCoord2f(tx1, ty0);
    GL.Vertex2f(hw, -hh);
    xgl.TexCoord2f(tx1, ty1);
    GL.Vertex2f(hw, hh);
    GL.End_;
    exit;
  end
  else
  begin
    // multi-quad plane (actually built from tri-strips)
    texSFact := (tx1 - tx0) / FXTiles;
    texTFact := (ty1 - ty0) / FYTiles;
    posXFact := FWidth / FXTiles;
    posYFact := FHeight / FYTiles;
    if FMesh = nil then
    begin
      SetLength(FMesh, FYTiles+1, FXTiles+1);
      for Y := 0 to FYTiles do
      begin
        texT1 := Y * texTFact;
        pY1 := Y * posYFact - hh;
        for X := 0 to FXTiles do
        begin
          texS := X * texSFact;
          pX := X * posXFact - hw;
          FMesh[Y][X].Position := Vector3fMake(pX, pY1, 0.0);
          FMesh[Y][X].TexCoord := Vector2fMake(texS, texT1);
        end;
      end;
    end;
  end;

  with GL do
  begin
    Begin_(GL_TRIANGLES);
    for Y := 0 to FYTiles-1 do
    begin
      for X := 0 to FXTiles-1 do
      begin
        pVertex := @FMesh[Y][X];
        EmitVertex(pVertex);

        pVertex := @FMesh[Y][X+1];
        EmitVertex(pVertex);

        pVertex := @FMesh[Y+1][X];
        EmitVertex(pVertex);

        pVertex := @FMesh[Y+1][X+1];
        EmitVertex(pVertex);

        pVertex := @FMesh[Y+1][X];
        EmitVertex(pVertex);

        pVertex := @FMesh[Y][X+1];
        EmitVertex(pVertex);
      end;
    end;
    End_;
  end;
end;

// SetWidth
//

procedure TGLPlane.SetWidth(const aValue: Single);
begin
  if aValue <> FWidth then
  begin
    FWidth := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// ScreenRect
//

function TGLPlane.ScreenRect(aBuffer: TGLSceneBuffer): TGLRect;
var
  v: array [0 .. 3] of TVector;
  buf: TGLSceneBuffer;
  hw, hh: TGLFloat;
begin
  buf := aBuffer;
  if Assigned(buf) then
  begin
    hw := FWidth * 0.5;
    hh := FHeight * 0.5;
    v[0] := LocalToAbsolute(PointMake(-hw, -hh, 0));
    v[1] := LocalToAbsolute(PointMake(hw, -hh, 0));
    v[2] := LocalToAbsolute(PointMake(hw, hh, 0));
    v[3] := LocalToAbsolute(PointMake(-hw, hh, 0));
    buf.WorldToScreen(@v[0], 4);
    Result.Left := Round(MinFloat([v[0].V[0], v[1].V[0], v[2].V[0], v[3].V[0]]));
    Result.Right := Round(MaxFloat([v[0].V[0], v[1].V[0], v[2].V[0], v[3].V[0]]));
    Result.Top := Round(MinFloat([v[0].V[1], v[1].V[1], v[2].V[1], v[3].V[1]]));
    Result.Bottom := Round(MaxFloat([v[0].V[1], v[1].V[1], v[2].V[1], v[3].V[1]]));
  end
  else
    FillChar(Result, SizeOf(TGLRect), 0);
end;

// PointDistance
//

function TGLPlane.PointDistance(const aPoint: TVector): Single;
begin
  Result := VectorDotProduct(VectorSubtract(aPoint, AbsolutePosition),
    AbsoluteDirection);
end;

// SetHeight
//

procedure TGLPlane.SetHeight(const aValue: Single);
begin
  if aValue <> FHeight then
  begin
    FHeight := aValue;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetXOffset
//

procedure TGLPlane.SetXOffset(const Value: TGLFloat);
begin
  if Value <> FXOffset then
  begin
    FXOffset := Value;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetXScope
//

procedure TGLPlane.SetXScope(const Value: TGLFloat);
begin
  if Value <> FXScope then
  begin
    FXScope := Value;
    if FXScope > 1 then
      FXScope := 1;
    FMesh := nil;
    StructureChanged;
  end;
end;

// StoreXScope
//

function TGLPlane.StoreXScope: Boolean;
begin
  Result := (FXScope <> 1);
end;

// SetXTiles
//

procedure TGLPlane.SetXTiles(const Value: Cardinal);
begin
  if Value <> FXTiles then
  begin
    FXTiles := Value;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetYOffset
//

procedure TGLPlane.SetYOffset(const Value: TGLFloat);
begin
  if Value <> FYOffset then
  begin
    FYOffset := Value;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetYScope
//

procedure TGLPlane.SetYScope(const Value: TGLFloat);
begin
  if Value <> FYScope then
  begin
    FYScope := Value;
    if FYScope > 1 then
      FYScope := 1;
    FMesh := nil;
    StructureChanged;
  end;
end;

// StoreYScope
//

function TGLPlane.StoreYScope: Boolean;
begin
  Result := (FYScope <> 1);
end;

// SetYTiles
//

procedure TGLPlane.SetYTiles(const Value: Cardinal);
begin
  if Value <> FYTiles then
  begin
    FYTiles := Value;
    FMesh := nil;
    StructureChanged;
  end;
end;

// SetStyle
//

procedure TGLPlane.SetStyle(const val: TPlaneStyles);
begin
  if val <> FStyle then
  begin
    FStyle := val;
    StructureChanged;
  end;
end;

// ------------------
// ------------------ TGLSprite ------------------
// ------------------

// Create
//

constructor TGLSprite.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  ObjectStyle := ObjectStyle + [osDirectDraw, osNoVisibilityCulling];
  FAlphaChannel := 1;
  FWidth := 1;
  FHeight := 1;
end;

 
//

procedure TGLSprite.Assign(Source: TPersistent);
begin
  if Source is TGLSprite then
  begin
    FWidth := TGLSprite(Source).FWidth;
    FHeight := TGLSprite(Source).FHeight;
    FRotation := TGLSprite(Source).FRotation;
    FAlphaChannel := TGLSprite(Source).FAlphaChannel;
  end;
  inherited Assign(Source);
end;

function TGLSprite.AxisAlignedDimensionsUnscaled: TVector;
begin
  Result.V[0] := 0.5 * Abs(FWidth);
  Result.V[1] := 0.5 * Abs(FHeight);
  // Sprites turn with the camera and can be considered to have the same depth
  // as width
  Result.V[2] := 0.5 * Abs(FWidth);
end;

// BuildList
//

procedure TGLSprite.BuildList(var rci: TGLRenderContextInfo);
var
  vx, vy: TAffineVector;
  w, h: Single;
  mat: TMatrix;
  u0, v0, u1, v1: Integer;
begin
  if FAlphaChannel <> 1 then
    rci.GLStates.SetGLMaterialAlphaChannel(GL_FRONT, FAlphaChannel);

  mat := rci.PipelineTransformation.ModelViewMatrix;
  // extraction of the "vecteurs directeurs de la matrice"
  // (dunno how they are named in english)
  w := FWidth * 0.5;
  h := FHeight * 0.5;
  vx.V[0] := mat.V[0].V[0];
  vy.V[0] := mat.V[0].V[1];
  vx.V[1] := mat.V[1].V[0];
  vy.V[1] := mat.V[1].V[1];
  vx.V[2] := mat.V[2].V[0];
  vy.V[2] := mat.V[2].V[1];
  ScaleVector(vx, w / VectorLength(vx));
  ScaleVector(vy, h / VectorLength(vy));
  if FMirrorU then
  begin
    u0 := 1;
    u1 := 0;
  end
  else
  begin
    u0 := 0;
    u1 := 1;
  end;
  if FMirrorV then
  begin
    v0 := 1;
    v1 := 0;
  end
  else
  begin
    v0 := 0;
    v1 := 1;
  end;

  if FRotation <> 0 then
  begin
    GL.PushMatrix;
    GL.Rotatef(FRotation, mat.V[0].V[2], mat.V[1].V[2], mat.V[2].V[2]);
  end;
  GL.Begin_(GL_QUADS);
  xgl.TexCoord2f(u1, v1);
  GL.Vertex3f(vx.V[0] + vy.V[0], vx.V[1] + vy.V[1], vx.V[2] + vy.V[2]);
  xgl.TexCoord2f(u0, v1);
  GL.Vertex3f(-vx.V[0] + vy.V[0], -vx.V[1] + vy.V[1], -vx.V[2] + vy.V[2]);
  xgl.TexCoord2f(u0, v0);
  GL.Vertex3f(-vx.V[0] - vy.V[0], -vx.V[1] - vy.V[1], -vx.V[2] - vy.V[2]);
  xgl.TexCoord2f(u1, v0);
  GL.Vertex3f(vx.V[0] - vy.V[0], vx.V[1] - vy.V[1], vx.V[2] - vy.V[2]);
  GL.End_;
  if FRotation <> 0 then
    GL.PopMatrix;
end;

// SetWidth
//

procedure TGLSprite.SetWidth(const val: TGLFloat);
begin
  if FWidth <> val then
  begin
    FWidth := val;
    NotifyChange(Self);
  end;
end;

// SetHeight
//

procedure TGLSprite.SetHeight(const val: TGLFloat);
begin
  if FHeight <> val then
  begin
    FHeight := val;
    NotifyChange(Self);
  end;
end;

// SetRotation
//

procedure TGLSprite.SetRotation(const val: TGLFloat);
begin
  if FRotation <> val then
  begin
    FRotation := val;
    NotifyChange(Self);
  end;
end;

// SetAlphaChannel
//

procedure TGLSprite.SetAlphaChannel(const val: Single);
begin
  if val <> FAlphaChannel then
  begin
    if val < 0 then
      FAlphaChannel := 0
    else if val > 1 then
      FAlphaChannel := 1
    else
      FAlphaChannel := val;
    NotifyChange(Self);
  end;
end;

// StoreAlphaChannel
//

function TGLSprite.StoreAlphaChannel: Boolean;
begin
  Result := (FAlphaChannel <> 1);
end;

// SetMirrorU
//

procedure TGLSprite.SetMirrorU(const val: Boolean);
begin
  FMirrorU := val;
  NotifyChange(Self);
end;

// SetMirrorV
//

procedure TGLSprite.SetMirrorV(const val: Boolean);
begin
  FMirrorV := val;
  NotifyChange(Self);
end;

// SetSize
//

procedure TGLSprite.SetSize(const Width, Height: TGLFloat);
begin
  FWidth := Width;
  FHeight := Height;
  NotifyChange(Self);
end;

// SetSquareSize
//

procedure TGLSprite.SetSquareSize(const Size: TGLFloat);
begin
  FWidth := Size;
  FHeight := Size;
  NotifyChange(Self);
end;

// ------------------
// ------------------ TGLPointParameters ------------------
// ------------------

// Create
//

constructor TGLPointParameters.Create(AOwner: TPersistent);
begin
  inherited Create(AOwner);
  FMinSize := 0;
  FMaxSize := 128;
  FFadeTresholdSize := 1;
  FDistanceAttenuation := TGLCoordinates.CreateInitialized(Self, XHmgVector,
    csVector);
end;

// Destroy
//

destructor TGLPointParameters.Destroy;
begin
  FDistanceAttenuation.Free;
  inherited;
end;

 
//

procedure TGLPointParameters.Assign(Source: TPersistent);
begin
  if Source is TGLPointParameters then
  begin
    FMinSize := TGLPointParameters(Source).FMinSize;
    FMaxSize := TGLPointParameters(Source).FMaxSize;
    FFadeTresholdSize := TGLPointParameters(Source).FFadeTresholdSize;
    FDistanceAttenuation.Assign(TGLPointParameters(Source).DistanceAttenuation);
  end;
end;

// DefineProperties
//

procedure TGLPointParameters.DefineProperties(Filer: TFiler);
var
  defaultParams: Boolean;
begin
  inherited;
  defaultParams := (FMaxSize = 128) and (FMinSize = 0) and
    (FFadeTresholdSize = 1);
  Filer.DefineBinaryProperty('PointParams', ReadData, WriteData,
    not defaultParams);
end;

// ReadData
//

procedure TGLPointParameters.ReadData(Stream: TStream);
begin
  with Stream do
  begin
    Read(FMinSize, SizeOf(Single));
    Read(FMaxSize, SizeOf(Single));
    Read(FFadeTresholdSize, SizeOf(Single));
  end;
end;

// WriteData
//

procedure TGLPointParameters.WriteData(Stream: TStream);
begin
  with Stream do
  begin
    Write(FMinSize, SizeOf(Single));
    Write(FMaxSize, SizeOf(Single));
    Write(FFadeTresholdSize, SizeOf(Single));
  end;
end;

// Apply
//

procedure TGLPointParameters.Apply;
begin
  if Enabled and GL.ARB_point_parameters then
  begin
    GL.PointParameterf(GL_POINT_SIZE_MIN_ARB, FMinSize);
    GL.PointParameterf(GL_POINT_SIZE_MAX_ARB, FMaxSize);
    GL.PointParameterf(GL_POINT_FADE_THRESHOLD_SIZE_ARB, FFadeTresholdSize);
    GL.PointParameterfv(GL_DISTANCE_ATTENUATION_ARB,
      FDistanceAttenuation.AsAddress);
  end;
end;

// UnApply
//

procedure TGLPointParameters.UnApply;
begin
  if Enabled and GL.ARB_point_parameters then
  begin
    GL.PointParameterf(GL_POINT_SIZE_MIN_ARB, 0);
    GL.PointParameterf(GL_POINT_SIZE_MAX_ARB, 128);
    GL.PointParameterf(GL_POINT_FADE_THRESHOLD_SIZE_ARB, 1);
    GL.PointParameterfv(GL_DISTANCE_ATTENUATION_ARB, @XVector);
  end;
end;

// SetEnabled
//

procedure TGLPointParameters.SetEnabled(const val: Boolean);
begin
  if val <> FEnabled then
  begin
    FEnabled := val;
    NotifyChange(Self);
  end;
end;

// SetMinSize
//

procedure TGLPointParameters.SetMinSize(const val: Single);
begin
  if val <> FMinSize then
  begin
    if val < 0 then
      FMinSize := 0
    else
      FMinSize := val;
    NotifyChange(Self);
  end;
end;

// SetMaxSize
//

procedure TGLPointParameters.SetMaxSize(const val: Single);
begin
  if val <> FMaxSize then
  begin
    if val < 0 then
      FMaxSize := 0
    else
      FMaxSize := val;
    NotifyChange(Self);
  end;
end;

// SetFadeTresholdSize
//

procedure TGLPointParameters.SetFadeTresholdSize(const val: Single);
begin
  if val <> FFadeTresholdSize then
  begin
    if val < 0 then
      FFadeTresholdSize := 0
    else
      FFadeTresholdSize := val;
    NotifyChange(Self);
  end;
end;

// SetDistanceAttenuation
//

procedure TGLPointParameters.SetDistanceAttenuation(const val: TGLCoordinates);
begin
  FDistanceAttenuation.Assign(val);
end;

// ------------------
// ------------------ TGLPoints ------------------
// ------------------

// Create
//

constructor TGLPoints.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  ObjectStyle := ObjectStyle + [osDirectDraw, osNoVisibilityCulling];
  FStyle := psSquare;
  FSize := cDefaultPointSize;
  FPositions := TAffineVectorList.Create;
  FPositions.Add(NullVector);
  FColors := TVectorList.Create;
  FPointParameters := TGLPointParameters.Create(Self);
end;

// Destroy
//

destructor TGLPoints.Destroy;
begin
  FPointParameters.Free;
  FColors.Free;
  FPositions.Free;
  inherited;
end;

 
//

procedure TGLPoints.Assign(Source: TPersistent);
begin
  if Source is TGLPoints then
  begin
    FSize := TGLPoints(Source).FSize;
    FStyle := TGLPoints(Source).FStyle;
    FPositions.Assign(TGLPoints(Source).FPositions);
    FColors.Assign(TGLPoints(Source).FColors);
    StructureChanged
  end;
  inherited Assign(Source);
end;

// BuildList
//

procedure TGLPoints.BuildList(var rci: TGLRenderContextInfo);
var
  n: Integer;
  v: TVector;
begin
  n := FPositions.Count;
  if n = 0 then
    Exit;

  case FColors.Count of
    0:
      GL.Color4f(1, 1, 1, 1);
    1:
      GL.Color4fv(PGLFloat(FColors.List));
  else
    if FColors.Count < n then
      n := FColors.Count;
    GL.ColorPointer(4, GL_FLOAT, 0, FColors.List);
    GL.EnableClientState(GL_COLOR_ARRAY);
  end;
  if FColors.Count < 2 then
    GL.DisableClientState(GL_COLOR_ARRAY);

  rci.GLStates.Disable(stLighting);
  if n = 0 then
  begin
    v := NullHmgPoint;
    GL.VertexPointer(3, GL_FLOAT, 0, @v);
    n := 1;
  end
  else
    GL.VertexPointer(3, GL_FLOAT, 0, FPositions.List);
  GL.EnableClientState(GL_VERTEX_ARRAY);

  if NoZWrite then
    rci.GLStates.DepthWriteMask := False;
  rci.GLStates.PointSize := FSize;
  PointParameters.Apply;
  if GL.EXT_compiled_vertex_array and (n > 64) then
    GL.LockArrays(0, n);
  case FStyle of
    psSquare:
      begin
        // square point (simplest method, fastest)
        rci.GLStates.Disable(stBlend);
      end;
    psRound:
      begin
        rci.GLStates.Enable(stPointSmooth);
        rci.GLStates.Enable(stAlphaTest);
        rci.GLStates.SetGLAlphaFunction(cfGreater, 0.5);
        rci.GLStates.Disable(stBlend);
      end;
    psSmooth:
      begin
        rci.GLStates.Enable(stPointSmooth);
        rci.GLStates.Enable(stAlphaTest);
        rci.GLStates.SetGLAlphaFunction(cfNotEqual, 0.0);
        rci.GLStates.Enable(stBlend);
        rci.GLStates.SetBlendFunc(bfSrcAlpha, bfOneMinusSrcAlpha);
      end;
    psSmoothAdditive:
      begin
        rci.GLStates.Enable(stPointSmooth);
        rci.GLStates.Enable(stAlphaTest);
        rci.GLStates.SetGLAlphaFunction(cfNotEqual, 0.0);
        rci.GLStates.Enable(stBlend);
        rci.GLStates.SetBlendFunc(bfSrcAlpha, bfOne);
      end;
    psSquareAdditive:
      begin
        rci.GLStates.Enable(stBlend);
        rci.GLStates.SetBlendFunc(bfSrcAlpha, bfOne);
      end;
  else
    Assert(False);
  end;
  GL.DrawArrays(GL_POINTS, 0, n);
  if GL.EXT_compiled_vertex_array and (n > 64) then
    GL.UnlockArrays;
  PointParameters.UnApply;
  GL.DisableClientState(GL_VERTEX_ARRAY);
  if FColors.Count > 1 then
    GL.DisableClientState(GL_COLOR_ARRAY);
end;

// StoreSize
//

function TGLPoints.StoreSize: Boolean;
begin
  Result := (FSize <> cDefaultPointSize);
end;

// SetNoZWrite
//

procedure TGLPoints.SetNoZWrite(const val: Boolean);
begin
  if FNoZWrite <> val then
  begin
    FNoZWrite := val;
    StructureChanged;
  end;
end;

// SetStatic
//

procedure TGLPoints.SetStatic(const val: Boolean);
begin
  if FStatic <> val then
  begin
    FStatic := val;
    if val then
      ObjectStyle := ObjectStyle - [osDirectDraw]
    else
      ObjectStyle := ObjectStyle + [osDirectDraw];
    StructureChanged;
  end;
end;

// SetSize
//

procedure TGLPoints.SetSize(const val: Single);
begin
  if FSize <> val then
  begin
    FSize := val;
    StructureChanged;
  end;
end;

// SetPositions
//

procedure TGLPoints.SetPositions(const val: TAffineVectorList);
begin
  FPositions.Assign(val);
  StructureChanged;
end;

// SetColors
//

procedure TGLPoints.SetColors(const val: TVectorList);
begin
  FColors.Assign(val);
  StructureChanged;
end;

// SetStyle
//

procedure TGLPoints.SetStyle(const val: TGLPointStyle);
begin
  if FStyle <> val then
  begin
    FStyle := val;
    StructureChanged;
  end;
end;

// SetPointParameters
//

procedure TGLPoints.SetPointParameters(const val: TGLPointParameters);
begin
  FPointParameters.Assign(val);
end;

// ------------------
// ------------------ TGLLineBase ------------------
// ------------------

// Create
//

constructor TGLLineBase.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FLineColor := TGLColor.Create(Self);
  FLineColor.Initialize(clrWhite);
  FLinePattern := $FFFF;
  FAntiAliased := False;
  FLineWidth := 1.0;
end;

// Destroy
//

destructor TGLLineBase.Destroy;
begin
  FLineColor.Free;
  inherited Destroy;
end;

procedure TGLLineBase.NotifyChange(Sender: TObject);
begin
  if Sender = FLineColor then
    StructureChanged;
  inherited;
end;

// SetLineColor
//

procedure TGLLineBase.SetLineColor(const Value: TGLColor);
begin
  FLineColor.Color := Value.Color;
  StructureChanged;
end;

// SetLinePattern
//

procedure TGLLineBase.SetLinePattern(const Value: TGLushort);
begin
  if FLinePattern <> Value then
  begin
    FLinePattern := Value;
    StructureChanged;
  end;
end;

// SetLineWidth
//

procedure TGLLineBase.SetLineWidth(const val: Single);
begin
  if FLineWidth <> val then
  begin
    FLineWidth := val;
    StructureChanged;
  end;
end;

// StoreLineWidth
//

function TGLLineBase.StoreLineWidth: Boolean;
begin
  Result := (FLineWidth <> 1.0);
end;

// SetAntiAliased
//

procedure TGLLineBase.SetAntiAliased(const val: Boolean);
begin
  if FAntiAliased <> val then
  begin
    FAntiAliased := val;
    StructureChanged;
  end;
end;

 
//

procedure TGLLineBase.Assign(Source: TPersistent);
begin
  if Source is TGLLineBase then
  begin
    LineColor := TGLLineBase(Source).FLineColor;
    LinePattern := TGLLineBase(Source).FLinePattern;
    LineWidth := TGLLineBase(Source).FLineWidth;
    AntiAliased := TGLLineBase(Source).FAntiAliased;
  end;
  inherited Assign(Source);
end;

// SetupLineStyle
//

procedure TGLLineBase.SetupLineStyle(var rci: TGLRenderContextInfo);
begin
  with rci.GLStates do
  begin
    Disable(stLighting);
    if FLinePattern <> $FFFF then
    begin
      Enable(stLineStipple);
      Enable(stBlend);
      SetBlendFunc(bfSrcAlpha, bfOneMinusSrcAlpha);
      LineStippleFactor := 1;
      LineStipplePattern := FLinePattern;
    end
    else
      Disable(stLineStipple);
    if FAntiAliased then
    begin
      Enable(stLineSmooth);
      Enable(stBlend);
      SetBlendFunc(bfSrcAlpha, bfOneMinusSrcAlpha);
    end
    else
      Disable(stLineSmooth);
    LineWidth := FLineWidth;

    if FLineColor.Alpha <> 1 then
    begin
      if not FAntiAliased then
      begin
        Enable(stBlend);
        SetBlendFunc(bfSrcAlpha, bfOneMinusSrcAlpha);
      end;
      GL.Color4fv(FLineColor.AsAddress);
    end
    else
      GL.Color3fv(FLineColor.AsAddress);

  end;
end;

// ------------------
// ------------------ TGLLinesNode ------------------
// ------------------

// Create
//

constructor TGLLinesNode.Create(Collection: TCollection);
begin
  inherited Create(Collection);
  FColor := TGLColor.Create(Self);
  FColor.Initialize((TGLLinesNodes(Collection).GetOwner as TGLLines)
    .NodeColor.Color);
  FColor.OnNotifyChange := OnColorChange;
end;

// Destroy
//

destructor TGLLinesNode.Destroy;
begin
  FColor.Free;
  inherited Destroy;
end;

 
//

procedure TGLLinesNode.Assign(Source: TPersistent);
begin
  if Source is TGLLinesNode then
    FColor.Assign(TGLLinesNode(Source).FColor);
  inherited;
end;

// SetColor
//

procedure TGLLinesNode.SetColor(const val: TGLColor);
begin
  FColor.Assign(val);
end;

// OnColorChange
//

procedure TGLLinesNode.OnColorChange(Sender: TObject);
begin
  (Collection as TGLNodes).NotifyChange;
end;

// StoreColor
//

function TGLLinesNode.StoreColor: Boolean;
begin
  Result := not VectorEquals((TGLLinesNodes(Collection).GetOwner as TGLLines)
    .NodeColor.Color, FColor.Color);
end;

// ------------------
// ------------------ TGLLinesNodes ------------------
// ------------------

// Create
//

constructor TGLLinesNodes.Create(AOwner: TComponent);
begin
  inherited Create(AOwner, TGLLinesNode);
end;

// NotifyChange
//

procedure TGLLinesNodes.NotifyChange;
begin
  if (GetOwner <> nil) then
    (GetOwner as TGLBaseSceneObject).StructureChanged;
end;

// ------------------
// ------------------ TGLNodedLines ------------------
// ------------------

// Create
//

constructor TGLNodedLines.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FNodes := TGLLinesNodes.Create(Self);
  FNodeColor := TGLColor.Create(Self);
  FNodeColor.Initialize(clrBlue);
  FNodeColor.OnNotifyChange := OnNodeColorChanged;
  FOldNodeColor := clrBlue;
  FNodesAspect := lnaAxes;
  FNodeSize := 1;
end;

// Destroy
//

destructor TGLNodedLines.Destroy;
begin
  FNodes.Free;
  FNodeColor.Free;
  inherited Destroy;
end;

// SetNodesAspect
//

procedure TGLNodedLines.SetNodesAspect(const Value: TLineNodesAspect);
begin
  if Value <> FNodesAspect then
  begin
    FNodesAspect := Value;
    StructureChanged;
  end;
end;

// SetNodeColor
//

procedure TGLNodedLines.SetNodeColor(const Value: TGLColor);
begin
  FNodeColor.Color := Value.Color;
  StructureChanged;
end;

// OnNodeColorChanged
//

procedure TGLNodedLines.OnNodeColorChanged(Sender: TObject);
var
  i: Integer;
begin
  // update color for nodes...
  for i := 0 to Nodes.Count - 1 do
    if VectorEquals(TGLLinesNode(Nodes[i]).Color.Color, FOldNodeColor) then
      TGLLinesNode(Nodes[i]).Color.Assign(FNodeColor);
  SetVector(FOldNodeColor, FNodeColor.Color);
end;

// SetNodes
//

procedure TGLNodedLines.SetNodes(const aNodes: TGLLinesNodes);
begin
  FNodes.Assign(aNodes);
  StructureChanged;
end;

// SetNodeSize
//

procedure TGLNodedLines.SetNodeSize(const val: Single);
begin
  if val <= 0 then
    FNodeSize := 1
  else
    FNodeSize := val;
  StructureChanged;
end;

// StoreNodeSize
//

function TGLNodedLines.StoreNodeSize: Boolean;
begin
  Result := FNodeSize <> 1;
end;

 
//

procedure TGLNodedLines.Assign(Source: TPersistent);
begin
  if Source is TGLNodedLines then
  begin
    SetNodes(TGLNodedLines(Source).FNodes);
    FNodesAspect := TGLNodedLines(Source).FNodesAspect;
    FNodeColor.Color := TGLNodedLines(Source).FNodeColor.Color;
    FNodeSize := TGLNodedLines(Source).FNodeSize;
  end;
  inherited Assign(Source);
end;

// DrawNode
//

procedure TGLNodedLines.DrawNode(var rci: TGLRenderContextInfo; X, Y, Z: Single;
  Color: TGLColor);
begin
  GL.PushMatrix;
  GL.Translatef(X, Y, Z);
  case NodesAspect of
    lnaAxes:
      AxesBuildList(rci, $CCCC, FNodeSize * 0.5);
    lnaCube:
      CubeWireframeBuildList(rci, FNodeSize, False, Color.Color);
    lnaDodecahedron:
      begin
        if FNodeSize <> 1 then
        begin
          GL.PushMatrix;
          GL.Scalef(FNodeSize, FNodeSize, FNodeSize);
          rci.GLStates.SetGLMaterialColors(cmFront, clrBlack, clrGray20,
            Color.Color, clrBlack, 0);
          DodecahedronBuildList;
          GL.PopMatrix;
        end
        else
        begin
          rci.GLStates.SetGLMaterialColors(cmFront, clrBlack, clrGray20,
            Color.Color, clrBlack, 0);
          DodecahedronBuildList;
        end;
      end;
  else
    Assert(False)
  end;
  GL.PopMatrix;
end;

// AxisAlignedDimensionsUnscaled
//

function TGLNodedLines.AxisAlignedDimensionsUnscaled: TVector;
var
  i: Integer;
begin
  RstVector(Result);
  for i := 0 to Nodes.Count - 1 do
    MaxVector(Result, VectorAbs(Nodes[i].AsVector));
  // EG: commented out, line below looks suspicious, since scale isn't taken
  // into account in previous loop, must have been hiding another bug... somewhere...
  // DivideVector(Result, Scale.AsVector);     //DanB ?
end;

// AddNode (coords)
//

procedure TGLNodedLines.AddNode(const coords: TGLCoordinates);
var
  n: TGLNode;
begin
  n := Nodes.Add;
  if Assigned(coords) then
    n.AsVector := coords.AsVector;
  StructureChanged;
end;

// AddNode (xyz)
//

procedure TGLNodedLines.AddNode(const X, Y, Z: TGLFloat);
var
  n: TGLNode;
begin
  n := Nodes.Add;
  n.AsVector := VectorMake(X, Y, Z, 1);
  StructureChanged;
end;

// AddNode (vector)
//

procedure TGLNodedLines.AddNode(const Value: TVector);
var
  n: TGLNode;
begin
  n := Nodes.Add;
  n.AsVector := Value;
  StructureChanged;
end;

// AddNode (affine vector)
//

procedure TGLNodedLines.AddNode(const Value: TAffineVector);
var
  n: TGLNode;
begin
  n := Nodes.Add;
  n.AsVector := VectorMake(Value);
  StructureChanged;
end;

// ------------------
// ------------------ TGLLines ------------------
// ------------------

// Create
//

constructor TGLLines.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FDivision := 10;
  FSplineMode := lsmLines;
  FNURBSKnots := TSingleList.Create;
  FNURBSOrder := 0;
  FNURBSTolerance := 50;
end;

// Destroy
//

destructor TGLLines.Destroy;
begin
  FNURBSKnots.Free;
  inherited Destroy;
end;

// SetDivision
//

procedure TGLLines.SetDivision(const Value: Integer);
begin
  if Value <> FDivision then
  begin
    if Value < 1 then
      FDivision := 1
    else
      FDivision := Value;
    StructureChanged;
  end;
end;

// SetOptions
//

procedure TGLLines.SetOptions(const val: TLinesOptions);
begin
  FOptions := val;
  StructureChanged;
end;

// SetSplineMode
//

procedure TGLLines.SetSplineMode(const val: TGLLineSplineMode);
begin
  if FSplineMode <> val then
  begin
    FSplineMode := val;
    StructureChanged;
  end;
end;

// SetNURBSOrder
//

procedure TGLLines.SetNURBSOrder(const val: Integer);
begin
  if val <> FNURBSOrder then
  begin
    FNURBSOrder := val;
    StructureChanged;
  end;
end;

// SetNURBSTolerance
//

procedure TGLLines.SetNURBSTolerance(const val: Single);
begin
  if val <> FNURBSTolerance then
  begin
    FNURBSTolerance := val;
    StructureChanged;
  end;
end;

 
//

procedure TGLLines.Assign(Source: TPersistent);
begin
  if Source is TGLLines then
  begin
    FDivision := TGLLines(Source).FDivision;
    FSplineMode := TGLLines(Source).FSplineMode;
    FOptions := TGLLines(Source).FOptions;
  end;
  inherited Assign(Source);
end;

// BuildList
//

procedure TGLLines.BuildList(var rci: TGLRenderContextInfo);
var
  i, n: Integer;
  A, B, C: TGLFloat;
  f: Single;
  Spline: TCubicSpline;
  vertexColor: TVector;
  nodeBuffer: array of TAffineVector;
  colorBuffer: array of TVector;
  nurbsRenderer: PGLUNurbs;
begin
  if Nodes.Count > 1 then
  begin
    // first, we setup the line color & stippling styles
    SetupLineStyle(rci);
    if rci.bufferDepthTest then
      rci.GLStates.Enable(stDepthTest);
    if loColorLogicXor in Options then
    begin
      rci.GLStates.Enable(stColorLogicOp);
      rci.GLStates.LogicOpMode := loXOr;
    end;
    // Set up the control point buffer for Bezier splines and NURBS curves.
    // If required this could be optimized by storing a cached node buffer.
    if (FSplineMode = lsmBezierSpline) or (FSplineMode = lsmNURBSCurve) then
    begin
      SetLength(nodeBuffer, Nodes.Count);
      SetLength(colorBuffer, Nodes.Count);
      for i := 0 to Nodes.Count - 1 do
        with TGLLinesNode(Nodes[i]) do
        begin
          nodeBuffer[i] := AsAffineVector;
          colorBuffer[i] := Color.Color;
        end;
    end;

    if FSplineMode = lsmBezierSpline then
    begin
      // map evaluator
      rci.GLStates.PushAttrib([sttEval]);
      GL.Enable(GL_MAP1_VERTEX_3);
      GL.Enable(GL_MAP1_COLOR_4);

      GL.Map1f(GL_MAP1_VERTEX_3, 0, 1, 3, Nodes.Count, @nodeBuffer[0]);
      GL.Map1f(GL_MAP1_COLOR_4, 0, 1, 4, Nodes.Count, @colorBuffer[0]);
    end;

    // start drawing the line
    if (FSplineMode = lsmNURBSCurve) and (FDivision >= 2) then
    begin
      if (FNURBSOrder > 0) and (FNURBSKnots.Count > 0) then
      begin

        nurbsRenderer := gluNewNurbsRenderer;
        try
          gluNurbsProperty(nurbsRenderer, GLU_SAMPLING_TOLERANCE,
            FNURBSTolerance);
          gluNurbsProperty(nurbsRenderer, GLU_DISPLAY_MODE, GLU_FILL);
          gluBeginCurve(nurbsRenderer);
          gluNurbsCurve(nurbsRenderer, FNURBSKnots.Count, @FNURBSKnots.List[0],
            3, @nodeBuffer[0], FNURBSOrder, GL_MAP1_VERTEX_3);
          gluEndCurve(nurbsRenderer);
        finally
          gluDeleteNurbsRenderer(nurbsRenderer);
        end;
      end;
    end
    else
    begin
      // lines, cubic splines or bezier
      if FSplineMode = lsmSegments then
        GL.Begin_(GL_LINES)
      else if FSplineMode = lsmLoop then
        GL.Begin_(GL_LINE_LOOP)
      else
        GL.Begin_(GL_LINE_STRIP);
      if (FDivision < 2) or (FSplineMode in [lsmLines, lsmSegments,
        lsmLoop]) then
      begin
        // standard line(s), draw directly
        if loUseNodeColorForLines in Options then
        begin
          // node color interpolation
          for i := 0 to Nodes.Count - 1 do
            with TGLLinesNode(Nodes[i]) do
            begin
              GL.Color4fv(Color.AsAddress);
              GL.Vertex3f(X, Y, Z);
            end;
        end
        else
        begin
          // single color
          for i := 0 to Nodes.Count - 1 do
            with Nodes[i] do
              GL.Vertex3f(X, Y, Z);
        end;
      end
      else if FSplineMode = lsmCubicSpline then
      begin
        // cubic spline
        Spline := Nodes.CreateNewCubicSpline;
        try
          f := 1 / FDivision;
          for i := 0 to (Nodes.Count - 1) * FDivision do
          begin
            Spline.SplineXYZ(i * f, A, B, C);
            if loUseNodeColorForLines in Options then
            begin
              n := (i div FDivision);
              if n < Nodes.Count - 1 then
                VectorLerp(TGLLinesNode(Nodes[n]).Color.Color,
                  TGLLinesNode(Nodes[n + 1]).Color.Color, (i mod FDivision) * f,
                  vertexColor)
              else
                SetVector(vertexColor, TGLLinesNode(Nodes[Nodes.Count - 1])
                  .Color.Color);
              GL.Color4fv(@vertexColor);
            end;
            GL.Vertex3f(A, B, C);
          end;
        finally
          Spline.Free;
        end;
      end
      else if FSplineMode = lsmBezierSpline then
      begin
        f := 1 / FDivision;
        for i := 0 to FDivision do
          GL.EvalCoord1f(i * f);
      end;
      GL.End_;
    end;
    rci.GLStates.Disable(stColorLogicOp);

    if FSplineMode = lsmBezierSpline then
      rci.GLStates.PopAttrib;
    if Length(nodeBuffer) > 0 then
    begin
      SetLength(nodeBuffer, 0);
      SetLength(colorBuffer, 0);
    end;

    if FNodesAspect <> lnaInvisible then
    begin
      if not rci.ignoreBlendingRequests then
      begin
        rci.GLStates.Enable(stBlend);
        rci.GLStates.SetBlendFunc(bfSrcAlpha, bfOneMinusSrcAlpha);
      end;

      for i := 0 to Nodes.Count - 1 do
        with TGLLinesNode(Nodes[i]) do
          DrawNode(rci, X, Y, Z, Color);
    end;
  end;
end;

// ------------------
// ------------------ TGLCube ------------------
// ------------------

// Create
//

constructor TGLCube.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FCubeSize := XYZVector;
  FParts := [cpTop, cpBottom, cpFront, cpBack, cpLeft, cpRight];
  FNormalDirection := ndOutside;
  ObjectStyle := ObjectStyle + [osDirectDraw];
end;

// BuildList
//

procedure TGLCube.BuildList(var rci: TGLRenderContextInfo);
var
  hw, hh, hd, nd: TGLFloat;
  TanLoc, BinLoc: Integer;
begin
  if FNormalDirection = ndInside then
    nd := -1
  else
    nd := 1;
  hw := FCubeSize.X * 0.5;
  hh := FCubeSize.Y * 0.5;
  hd := FCubeSize.Z * 0.5;

  with GL do
  begin
    if ARB_shader_objects and (rci.GLStates.CurrentProgram > 0) then
    begin
      TanLoc := GetAttribLocation(rci.GLStates.CurrentProgram, PGLChar(TangentAttributeName));
      BinLoc := GetAttribLocation(rci.GLStates.CurrentProgram, PGLChar(BinormalAttributeName));
    end
    else
    begin
      TanLoc := -1;
      BinLoc := -1;
    end;

    Begin_(GL_TRIANGLES);
    if cpFront in FParts then
    begin
      Normal3f(0, 0, nd);
      if TanLoc > -1 then
        VertexAttrib3f(TanLoc, nd, 0, 0);
      if BinLoc > -1 then
        VertexAttrib3f(BinLoc, 0, nd, 0);
      xgl.TexCoord2fv(@XYTexPoint);
      Vertex3f(hw, hh, hd);
      xgl.TexCoord2fv(@YTexPoint);
      Vertex3f(-hw * nd, hh * nd, hd);
      xgl.TexCoord2fv(@NullTexPoint);
      Vertex3f(-hw, -hh, hd);
      Vertex3f(-hw, -hh, hd);
      xgl.TexCoord2fv(@XTexPoint);
      Vertex3f(hw * nd, -hh * nd, hd);
      xgl.TexCoord2fv(@XYTexPoint);
      Vertex3f(hw, hh, hd);
    end;
    if cpBack in FParts then
    begin
      Normal3f(0, 0, -nd);
      if TanLoc > -1 then
        VertexAttrib3f(TanLoc, -nd, 0, 0);
      if BinLoc > -1 then
        VertexAttrib3f(BinLoc, 0, nd, 0);
      xgl.TexCoord2fv(@YTexPoint);
      Vertex3f(hw, hh, -hd);
      xgl.TexCoord2fv(@NullTexPoint);
      Vertex3f(hw * nd, -hh * nd, -hd);
      xgl.TexCoord2fv(@XTexPoint);
      Vertex3f(-hw, -hh, -hd);
      Vertex3f(-hw, -hh, -hd);
      xgl.TexCoord2fv(@XYTexPoint);
      Vertex3f(-hw * nd, hh * nd, -hd);
      xgl.TexCoord2fv(@YTexPoint);
      Vertex3f(hw, hh, -hd);
    end;
    if cpLeft in FParts then
    begin
      Normal3f(-nd, 0, 0);
      if TanLoc > -1 then
        VertexAttrib3f(TanLoc, 0, 0, nd);
      if BinLoc > -1 then
        VertexAttrib3f(BinLoc, 0, nd, 0);
      xgl.TexCoord2fv(@XYTexPoint);
      Vertex3f(-hw, hh, hd);
      xgl.TexCoord2fv(@YTexPoint);
      Vertex3f(-hw, hh * nd, -hd * nd);
      xgl.TexCoord2fv(@NullTexPoint);
      Vertex3f(-hw, -hh, -hd);
      Vertex3f(-hw, -hh, -hd);
      xgl.TexCoord2fv(@XTexPoint);
      Vertex3f(-hw, -hh * nd, hd * nd);
      xgl.TexCoord2fv(@XYTexPoint);
      Vertex3f(-hw, hh, hd);
    end;
    if cpRight in FParts then
    begin
      Normal3f(nd, 0, 0);
      if TanLoc > -1 then
        VertexAttrib3f(TanLoc, 0, 0, -nd);
      if BinLoc > -1 then
        VertexAttrib3f(BinLoc, 0, nd, 0);
      xgl.TexCoord2fv(@YTexPoint);
      Vertex3f(hw, hh, hd);
      xgl.TexCoord2fv(@NullTexPoint);
      Vertex3f(hw, -hh * nd, hd * nd);
      xgl.TexCoord2fv(@XTexPoint);
      Vertex3f(hw, -hh, -hd);
      Vertex3f(hw, -hh, -hd);
      xgl.TexCoord2fv(@XYTexPoint);
      Vertex3f(hw, hh * nd, -hd * nd);
      xgl.TexCoord2fv(@YTexPoint);
      Vertex3f(hw, hh, hd);
    end;
    if cpTop in FParts then
    begin
      Normal3f(0, nd, 0);
      if TanLoc > -1 then
        VertexAttrib3f(TanLoc, nd, 0, 0);
      if BinLoc > -1 then
        VertexAttrib3f(BinLoc, 0, 0, -nd);
      xgl.TexCoord2fv(@YTexPoint);
      Vertex3f(-hw, hh, -hd);
      xgl.TexCoord2fv(@NullTexPoint);
      Vertex3f(-hw * nd, hh, hd * nd);
      xgl.TexCoord2fv(@XTexPoint);
      Vertex3f(hw, hh, hd);
      Vertex3f(hw, hh, hd);
      xgl.TexCoord2fv(@XYTexPoint);
      Vertex3f(hw * nd, hh, -hd * nd);
      xgl.TexCoord2fv(@YTexPoint);
      Vertex3f(-hw, hh, -hd);
    end;
    if cpBottom in FParts then
    begin
      Normal3f(0, -nd, 0);
      if TanLoc > -1 then
        VertexAttrib3f(TanLoc, -nd, 0, 0);
      if BinLoc > -1 then
        VertexAttrib3f(BinLoc, 0, 0, nd);
      xgl.TexCoord2fv(@NullTexPoint);
      Vertex3f(-hw, -hh, -hd);
      xgl.TexCoord2fv(@XTexPoint);
      Vertex3f(hw * nd, -hh, -hd * nd);
      xgl.TexCoord2fv(@XYTexPoint);
      Vertex3f(hw, -hh, hd);
      Vertex3f(hw, -hh, hd);
      xgl.TexCoord2fv(@YTexPoint);
      Vertex3f(-hw * nd, -hh, hd * nd);
      xgl.TexCoord2fv(@NullTexPoint);
      Vertex3f(-hw, -hh, -hd);
    end;
    End_;
  end;
end;

// GenerateSilhouette
//

function TGLCube.GenerateSilhouette(const silhouetteParameters
  : TGLSilhouetteParameters): TGLSilhouette;
var
  hw, hh, hd: TGLFloat;
  connectivity: TConnectivity;
  sil: TGLSilhouette;
begin
  connectivity := TConnectivity.Create(True);

  hw := FCubeSize.X * 0.5;
  hh := FCubeSize.Y * 0.5;
  hd := FCubeSize.Z * 0.5;

  if cpFront in FParts then
  begin
    connectivity.AddQuad(AffineVectorMake(hw, hh, hd),
      AffineVectorMake(-hw, hh, hd), AffineVectorMake(-hw, -hh, hd),
      AffineVectorMake(hw, -hh, hd));
  end;
  if cpBack in FParts then
  begin
    connectivity.AddQuad(AffineVectorMake(hw, hh, -hd),
      AffineVectorMake(hw, -hh, -hd), AffineVectorMake(-hw, -hh, -hd),
      AffineVectorMake(-hw, hh, -hd));
  end;
  if cpLeft in FParts then
  begin
    connectivity.AddQuad(AffineVectorMake(-hw, hh, hd),
      AffineVectorMake(-hw, hh, -hd), AffineVectorMake(-hw, -hh, -hd),
      AffineVectorMake(-hw, -hh, hd));
  end;
  if cpRight in FParts then
  begin
    connectivity.AddQuad(AffineVectorMake(hw, hh, hd),
      AffineVectorMake(hw, -hh, hd), AffineVectorMake(hw, -hh, -hd),
      AffineVectorMake(hw, hh, -hd));
  end;
  if cpTop in FParts then
  begin
    connectivity.AddQuad(AffineVectorMake(-hw, hh, -hd),
      AffineVectorMake(-hw, hh, hd), AffineVectorMake(hw, hh, hd),
      AffineVectorMake(hw, hh, -hd));
  end;
  if cpBottom in FParts then
  begin
    connectivity.AddQuad(AffineVectorMake(-hw, -hh, -hd),
      AffineVectorMake(hw, -hh, -hd), AffineVectorMake(hw, -hh, hd),
      AffineVectorMake(-hw, -hh, hd));
  end;

  sil := nil;
  connectivity.CreateSilhouette(silhouetteParameters, sil, False);

  Result := sil;

  connectivity.Free;
end;

// GetCubeWHD
//
function TGLCube.GetCubeWHD(const Index: Integer): TGLFloat;
begin
  Result := FCubeSize.V[index];
end;


// SetCubeWHD
//
procedure TGLCube.SetCubeWHD(Index: Integer; AValue: TGLFloat);
begin
  if AValue <> FCubeSize.V[index] then
  begin
    FCubeSize.V[index] := AValue;
    StructureChanged;
  end;
end;


// SetParts
//
procedure TGLCube.SetParts(aValue: TCubeParts);
begin
  if aValue <> FParts then
  begin
    FParts := aValue;
    StructureChanged;
  end;
end;

// SetNormalDirection
//

procedure TGLCube.SetNormalDirection(aValue: TNormalDirection);
begin
  if aValue <> FNormalDirection then
  begin
    FNormalDirection := aValue;
    StructureChanged;
  end;
end;

 
//

procedure TGLCube.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLCube) then
  begin
    FCubeSize := TGLCube(Source).FCubeSize;
    FParts := TGLCube(Source).FParts;
    FNormalDirection := TGLCube(Source).FNormalDirection;
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensions
//

function TGLCube.AxisAlignedDimensionsUnscaled: TVector;
begin
  Result.X := FCubeSize.X * 0.5;
  Result.Y := FCubeSize.Y * 0.5;
  Result.Z := FCubeSize.Z * 0.5;
  Result.W := 0;
end;

// RayCastIntersect
//

function TGLCube.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
var
  p: array [0 .. 5] of TVector;
  rv: TVector;
  rs, r: TVector;
  i: Integer;
  t, e: Single;
  eSize: TAffineVector;
begin
  rs := AbsoluteToLocal(rayStart);
  SetVector(rv, VectorNormalize(AbsoluteToLocal(rayVector)));
  e := 0.5 + 0.0001; // Small value for floating point imprecisions
  eSize.X := FCubeSize.X * e;
  eSize.Y := FCubeSize.Y * e;
  eSize.Z := FCubeSize.Z * e;
  p[0] := XHmgVector;
  p[1] := YHmgVector;
  p[2] := ZHmgVector;
  SetVector(p[3], -1, 0, 0);
  SetVector(p[4], 0, -1, 0);
  SetVector(p[5], 0, 0, -1);
  for i := 0 to 5 do
  begin
    if VectorDotProduct(p[i], rv) > 0 then
    begin
      t := -(p[i].X * rs.X + p[i].Y * rs.Y +
             p[i].Z * rs.Z + 0.5 *
        FCubeSize.V[i mod 3]) / (p[i].X * rv.X +
                                 p[i].Y * rv.Y +
                                 p[i].Z * rv.Z);
      MakePoint(r, rs.V[0] + t * rv.X, rs.Y +
                             t * rv.Y, rs.Z +
                             t * rv.Z);
      if (Abs(r.X) <= eSize.X) and
         (Abs(r.Y) <= eSize.Y) and
         (Abs(r.Z) <= eSize.Z) and
        (VectorDotProduct(VectorSubtract(r, rs), rv) > 0) then
      begin
        if Assigned(intersectPoint) then
          MakePoint(intersectPoint^, LocalToAbsolute(r));
        if Assigned(intersectNormal) then
          MakeVector(intersectNormal^, LocalToAbsolute(VectorNegate(p[i])));
        Result := True;
        Exit;
      end;
    end;
  end;
  Result := False;
end;

// DefineProperties
//

procedure TGLCube.DefineProperties(Filer: TFiler);
begin
  inherited;
  Filer.DefineBinaryProperty('CubeSize', ReadData, WriteData,
    (FCubeSize.V[0] <> 1) or (FCubeSize.V[1] <> 1) or (FCubeSize.V[2] <> 1));
end;

// ReadData
//

procedure TGLCube.ReadData(Stream: TStream);
begin
  with Stream do
  begin
    Read(FCubeSize, SizeOf(TAffineVector));
  end;
end;

// WriteData
//

procedure TGLCube.WriteData(Stream: TStream);
begin
  with Stream do
  begin
    Write(FCubeSize, SizeOf(TAffineVector));
  end;
end;

// ------------------
// ------------------ TGLQuadricObject ------------------
// ------------------

// Create
//

constructor TGLQuadricObject.Create(AOwner: TComponent);
begin
  inherited;
  FNormals := nsSmooth;
  FNormalDirection := ndOutside;
end;

// SetNormals
//

procedure TGLQuadricObject.SetNormals(aValue: TNormalSmoothing);
begin
  if aValue <> FNormals then
  begin
    FNormals := aValue;
    StructureChanged;
  end;
end;

// SetNormalDirection
//

procedure TGLQuadricObject.SetNormalDirection(aValue: TNormalDirection);
begin
  if aValue <> FNormalDirection then
  begin
    FNormalDirection := aValue;
    StructureChanged;
  end;
end;

// SetupQuadricParams
//

procedure TGLQuadricObject.SetupQuadricParams(quadric: PGLUquadricObj);
const
  cNormalSmoothinToEnum: array [nsFlat .. nsNone] of TGLEnum = (GLU_FLAT,
    GLU_SMOOTH, GLU_NONE);
begin
  gluQuadricDrawStyle(quadric, GLU_FILL);
  gluQuadricNormals(quadric, cNormalSmoothinToEnum[FNormals]);
  SetNormalQuadricOrientation(quadric);
  gluQuadricTexture(quadric, True);
end;

// SetNormalQuadricOrientation
//

procedure TGLQuadricObject.SetNormalQuadricOrientation(quadric: PGLUquadricObj);
const
  cNormalDirectionToEnum: array [ndInside .. ndOutside] of TGLEnum =
    (GLU_INSIDE, GLU_OUTSIDE);
begin
  gluQuadricOrientation(quadric, cNormalDirectionToEnum[FNormalDirection]);
end;

// SetInvertedQuadricOrientation
//

procedure TGLQuadricObject.SetInvertedQuadricOrientation
  (quadric: PGLUquadricObj);
const
  cNormalDirectionToEnum: array [ndInside .. ndOutside] of TGLEnum =
    (GLU_OUTSIDE, GLU_INSIDE);
begin
  gluQuadricOrientation(quadric, cNormalDirectionToEnum[FNormalDirection]);
end;

 
//

procedure TGLQuadricObject.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLQuadricObject) then
  begin
    FNormals := TGLQuadricObject(Source).FNormals;
    FNormalDirection := TGLQuadricObject(Source).FNormalDirection;
  end;
  inherited Assign(Source);
end;

// ------------------
// ------------------ TGLSphere ------------------
// ------------------

// Create
//

constructor TGLSphere.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FRadius := 0.5;
  FSlices := 16;
  FStacks := 16;
  FTop := 90;
  FBottom := -90;
  FStart := 0;
  FStop := 360;
end;

// BuildList
//

procedure TGLSphere.BuildList(var rci: TGLRenderContextInfo);
var
  v1, V2, N1: TAffineVector;
  AngTop, AngBottom, AngStart, AngStop, StepV, StepH: Double;
  SinP, CosP, SinP2, CosP2, SinT, CosT, Phi, Phi2, Theta: Double;
  uTexCoord, uTexFactor, vTexFactor, vTexCoord0, vTexCoord1: Single;
  i, j: Integer;
  DoReverse: Boolean;
begin
  DoReverse := (FNormalDirection = ndInside);
  rci.GLStates.PushAttrib([sttPolygon]);
  if DoReverse then
    rci.GLStates.InvertGLFrontFace;

  // common settings
  AngTop := DegToRad(1.0 * FTop);
  AngBottom := DegToRad(1.0 * FBottom);
  AngStart := DegToRad(1.0 * FStart);
  AngStop := DegToRad(1.0 * FStop);
  StepH := (AngStop - AngStart) / FSlices;
  StepV := (AngTop - AngBottom) / FStacks;
  GL.PushMatrix;
  GL.Scalef(Radius, Radius, Radius);

  // top cap
  if (FTop < 90) and (FTopCap in [ctCenter, ctFlat]) then
  begin
    GL.Begin_(GL_TRIANGLE_FAN);
    GLVectorGeometry.SinCos(AngTop, SinP, CosP);
    xgl.TexCoord2f(0.5, 0.5);
    if DoReverse then
      GL.Normal3f(0, -1, 0)
    else
      GL.Normal3f(0, 1, 0);
    if FTopCap = ctCenter then
      GL.Vertex3f(0, 0, 0)
    else
    begin
      GL.Vertex3f(0, SinP, 0);
      N1 := YVector;
      if DoReverse then
        N1.V[1] := -N1.V[1];
    end;
    v1.V[1] := SinP;
    Theta := AngStart;
    for i := 0 to FSlices do
    begin
      GLVectorGeometry.SinCos(Theta, SinT, CosT);
      v1.V[0] := CosP * SinT;
      v1.V[2] := CosP * CosT;
      if FTopCap = ctCenter then
      begin
        N1 := VectorPerpendicular(YVector, v1);
        if DoReverse then
          NegateVector(N1);
      end;
      xgl.TexCoord2f(SinT * 0.5 + 0.5, CosT * 0.5 + 0.5);
      GL.Normal3fv(@N1);
      GL.Vertex3fv(@v1);
      Theta := Theta + StepH;
    end;
    GL.End_;
  end;

  // main body
  Phi := AngTop;
  Phi2 := Phi - StepV;
  uTexFactor := 1 / FSlices;
  vTexFactor := 1 / FStacks;

  for j := 0 to FStacks - 1 do
  begin
    Theta := AngStart;
    GLVectorGeometry.SinCos(Phi, SinP, CosP);
    GLVectorGeometry.SinCos(Phi2, SinP2, CosP2);
    v1.V[1] := SinP;
    V2.V[1] := SinP2;
    vTexCoord0 := 1 - j * vTexFactor;
    vTexCoord1 := 1 - (j + 1) * vTexFactor;

    GL.Begin_(GL_TRIANGLE_STRIP);
    for i := 0 to FSlices do
    begin

      SinCos(Theta, SinT, CosT);
      v1.V[0] := CosP * SinT;
      V2.V[0] := CosP2 * SinT;
      v1.V[2] := CosP * CosT;
      V2.V[2] := CosP2 * CosT;

      uTexCoord := i * uTexFactor;
      xgl.TexCoord2f(uTexCoord, vTexCoord0);
      if DoReverse then
      begin
        N1 := VectorNegate(v1);
        GL.Normal3fv(@N1);
      end
      else
        GL.Normal3fv(@v1);
      GL.Vertex3fv(@v1);

      xgl.TexCoord2f(uTexCoord, vTexCoord1);
      if DoReverse then
      begin
        N1 := VectorNegate(V2);
        GL.Normal3fv(@N1);
      end
      else
        GL.Normal3fv(@V2);
      GL.Vertex3fv(@V2);

      Theta := Theta + StepH;
    end;
    GL.End_;
    Phi := Phi2;
    Phi2 := Phi2 - StepV;
  end;

  // bottom cap
  if (FBottom > -90) and (FBottomCap in [ctCenter, ctFlat]) then
  begin
    GL.Begin_(GL_TRIANGLE_FAN);
    SinCos(AngBottom, SinP, CosP);
    xgl.TexCoord2f(0.5, 0.5);
    if DoReverse then
      GL.Normal3f(0, 1, 0)
    else
      GL.Normal3f(0, -1, 0);
    if FBottomCap = ctCenter then
      GL.Vertex3f(0, 0, 0)
    else
    begin
      GL.Vertex3f(0, SinP, 0);
      if DoReverse then
        MakeVector(N1, 0, -1, 0)
      else
      begin
        N1 := YVector;
        NegateVector(N1); 
      end;
    end;
    v1.V[1] := SinP;
    Theta := AngStop;
    for i := 0 to FSlices do
    begin
      SinCos(Theta, SinT, CosT);
      v1.V[0] := CosP * SinT;
      v1.V[2] := CosP * CosT;
      if FBottomCap = ctCenter then
      begin
        N1 := VectorPerpendicular(AffineVectorMake(0, -1, 0), v1);
        if DoReverse then
          NegateVector(N1);
      end;
      xgl.TexCoord2f(SinT * 0.5 + 0.5, CosT * 0.5 + 0.5);
      GL.Normal3fv(@N1);
      GL.Vertex3fv(@v1);
      Theta := Theta - StepH;
    end;
    GL.End_;
  end;
  if DoReverse then
    rci.GLStates.InvertGLFrontFace;
  GL.PopMatrix;
  rci.GLStates.PopAttrib;
end;

// RayCastIntersect
//

function TGLSphere.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
var
  i1, i2: TVector;
  localStart, localVector: TVector;
begin
  // compute coefficients of quartic polynomial
  SetVector(localStart, AbsoluteToLocal(rayStart));
  SetVector(localVector, AbsoluteToLocal(rayVector));
  NormalizeVector(localVector);
  if RayCastSphereIntersect(localStart, localVector, NullHmgVector, Radius, i1,
    i2) > 0 then
  begin
    Result := True;
    if Assigned(intersectPoint) then
      SetVector(intersectPoint^, LocalToAbsolute(i1));
    if Assigned(intersectNormal) then
    begin
      i1.V[3] := 0; // vector transform
      SetVector(intersectNormal^, LocalToAbsolute(i1));
    end;
  end
  else
    Result := False;
end;

// GenerateSilhouette
//

function TGLSphere.GenerateSilhouette(const silhouetteParameters
  : TGLSilhouetteParameters): TGLSilhouette;
var
  i, j: Integer;
  s, C, angleFactor: Single;
  sVec, tVec: TAffineVector;
  Segments: Integer;
begin
  Segments := MaxInteger(FStacks, FSlices);

  // determine a local orthonormal matrix, viewer-oriented
  sVec := VectorCrossProduct(silhouetteParameters.SeenFrom, XVector);
  if VectorLength(sVec) < 1E-3 then
    sVec := VectorCrossProduct(silhouetteParameters.SeenFrom, YVector);
  tVec := VectorCrossProduct(silhouetteParameters.SeenFrom, sVec);
  NormalizeVector(sVec);
  NormalizeVector(tVec);
  // generate the silhouette (outline and capping)
  Result := TGLSilhouette.Create;
  angleFactor := (2 * PI) / Segments;
  for i := 0 to Segments - 1 do
  begin
    SinCos(i * angleFactor, FRadius, s, C);
    Result.vertices.AddPoint(VectorCombine(sVec, tVec, s, C));
    j := (i + 1) mod Segments;
    Result.Indices.Add(i, j);
    if silhouetteParameters.CappingRequired then
      Result.CapIndices.Add(Segments, i, j)
  end;
  if silhouetteParameters.CappingRequired then
    Result.vertices.Add(NullHmgPoint);
end;

// SetBottom
//

procedure TGLSphere.SetBottom(aValue: TAngleLimit1);
begin
  if FBottom <> aValue then
  begin
    FBottom := aValue;
    StructureChanged;
  end;
end;

// SetBottomCap
//

procedure TGLSphere.SetBottomCap(aValue: TCapType);
begin
  if FBottomCap <> aValue then
  begin
    FBottomCap := aValue;
    StructureChanged;
  end;
end;

// SetRadius
//

procedure TGLSphere.SetRadius(const aValue: TGLFloat);
begin
  if aValue <> FRadius then
  begin
    FRadius := aValue;
    StructureChanged;
  end;
end;

// SetSlices
//

procedure TGLSphere.SetSlices(aValue: Integer);
begin
  if aValue <> FSlices then
  begin
    if aValue <= 0 then
      FSlices := 1
    else
      FSlices := aValue;
    StructureChanged;
  end;
end;

// SetStacks
//

procedure TGLSphere.SetStacks(aValue: TGLInt);
begin
  if aValue <> FStacks then
  begin
    if aValue <= 0 then
      FStacks := 1
    else
      FStacks := aValue;
    StructureChanged;
  end;
end;

// SetStart
//

procedure TGLSphere.SetStart(aValue: TAngleLimit2);
begin
  if FStart <> aValue then
  begin
    Assert(aValue <= FStop);
    FStart := aValue;
    StructureChanged;
  end;
end;

// SetStop
//

procedure TGLSphere.SetStop(aValue: TAngleLimit2);
begin
  if FStop <> aValue then
  begin
    Assert(aValue >= FStart);
    FStop := aValue;
    StructureChanged;
  end;
end;

// SetTop
//

procedure TGLSphere.SetTop(aValue: TAngleLimit1);
begin
  if FTop <> aValue then
  begin
    FTop := aValue;
    StructureChanged;
  end;
end;

// SetTopCap
//

procedure TGLSphere.SetTopCap(aValue: TCapType);
begin
  if FTopCap <> aValue then
  begin
    FTopCap := aValue;
    StructureChanged;
  end;
end;

 
//

procedure TGLSphere.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLSphere) then
  begin
    FRadius := TGLSphere(Source).FRadius;
    FSlices := TGLSphere(Source).FSlices;
    FStacks := TGLSphere(Source).FStacks;
    FBottom := TGLSphere(Source).FBottom;
    FTop := TGLSphere(Source).FTop;
    FStart := TGLSphere(Source).FStart;
    FStop := TGLSphere(Source).FStop;
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensions
//

function TGLSphere.AxisAlignedDimensionsUnscaled: TVector;
begin
  Result.V[0] := Abs(FRadius);
  Result.V[1] := Result.V[0];
  Result.V[2] := Result.V[0];
  Result.V[3] := 0;
end;

// ------------------
// ------------------ TGLPolygonBase ------------------
// ------------------

// Create
//

constructor TGLPolygonBase.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  CreateNodes;
  FDivision := 10;
  FSplineMode := lsmLines;
end;

// CreateNodes
//

procedure TGLPolygonBase.CreateNodes;
begin
  FNodes := TGLNodes.Create(Self);
end;

// Destroy
//

destructor TGLPolygonBase.Destroy;
begin
  FNodes.Free;
  inherited Destroy;
end;

 
//

procedure TGLPolygonBase.Assign(Source: TPersistent);
begin
  if Source is TGLPolygonBase then
  begin
    SetNodes(TGLPolygonBase(Source).FNodes);
    FDivision := TGLPolygonBase(Source).FDivision;
    FSplineMode := TGLPolygonBase(Source).FSplineMode;
  end;
  inherited Assign(Source);
end;

// NotifyChange
//

procedure TGLPolygonBase.NotifyChange(Sender: TObject);
begin
  if Sender = Nodes then
    StructureChanged;
  inherited;
end;

// SetDivision
//

procedure TGLPolygonBase.SetDivision(const Value: Integer);
begin
  if Value <> FDivision then
  begin
    if Value < 1 then
      FDivision := 1
    else
      FDivision := Value;
    StructureChanged;
  end;
end;

// SetNodes
//

procedure TGLPolygonBase.SetNodes(const aNodes: TGLNodes);
begin
  FNodes.Assign(aNodes);
  StructureChanged;
end;

// SetSplineMode
//

procedure TGLPolygonBase.SetSplineMode(const val: TGLLineSplineMode);
begin
  if FSplineMode <> val then
  begin
    FSplineMode := val;
    StructureChanged;
  end;
end;

// AddNode (coords)
//

procedure TGLPolygonBase.AddNode(const coords: TGLCoordinates);
var
  n: TGLNode;
begin
  n := Nodes.Add;
  if Assigned(coords) then
    n.AsVector := coords.AsVector;
  StructureChanged;
end;

// AddNode (xyz)
//

procedure TGLPolygonBase.AddNode(const X, Y, Z: TGLFloat);
var
  n: TGLNode;
begin
  n := Nodes.Add;
  n.AsVector := VectorMake(X, Y, Z, 1);
  StructureChanged;
end;

// AddNode (vector)
//

procedure TGLPolygonBase.AddNode(const Value: TVector);
var
  n: TGLNode;
begin
  n := Nodes.Add;
  n.AsVector := Value;
  StructureChanged;
end;

// AddNode (affine vector)
//

procedure TGLPolygonBase.AddNode(const Value: TAffineVector);
var
  n: TGLNode;
begin
  n := Nodes.Add;
  n.AsVector := VectorMake(Value);
  StructureChanged;
end;

// ------------------
// ------------------ TGLSuperellipsoid ------------------
// ------------------

// Create
//

constructor TGLSuperellipsoid.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FRadius := 0.5;
  FxyCurve := 1.0;
  FzCurve := 1.0;
  FSlices := 16;
  FStacks := 16;
  FTop := 90;
  FBottom := -90;
  FStart := 0;
  FStop := 360;
end;

// BuildList
//

procedure TGLSuperellipsoid.BuildList(var rci: TGLRenderContextInfo);
var
  CosPc1, SinPc1, CosTc2, SinTc2: Double;

  tc1, tc2: integer;
  v1, v2, vs, N1: TAffineVector;
  AngTop, AngBottom, AngStart, AngStop, StepV, StepH: Double;
  SinP, CosP, SinP2, CosP2, SinT, CosT, Phi, Phi2, Theta: Double;
  uTexCoord, uTexFactor, vTexFactor, vTexCoord0, vTexCoord1: Double;
  i, j: Integer;
  DoReverse: Boolean;

begin
  DoReverse := (FNormalDirection = ndInside);
  if DoReverse then
    rci.GLStates.InvertGLFrontFace;

  // common settings
  AngTop := DegToRad(1.0 * FTop);
  AngBottom := DegToRad(1.0 * FBottom);
  AngStart := DegToRad(1.0 * FStart);
  AngStop := DegToRad(1.0 * FStop);
  StepH := (AngStop - AngStart) / FSlices;
  StepV := (AngTop - AngBottom) / FStacks;

  { Even integer used with the Power function, only produce positive points }
  tc1 := trunc(xyCurve);
  tc2 := trunc(zCurve);
  if tc1 mod 2 = 0 then
    xyCurve := xyCurve + 1e-6;
  if tc2 mod 2 = 0 then
    zCurve := zCurve - 1e-6;

  // top cap
  if (FTop < 90) and (FTopCap in [ctCenter, ctFlat]) then
  begin
    GL.Begin_(GL_TRIANGLE_FAN);
    SinCos(AngTop, SinP, CosP);
    xgl.TexCoord2f(0.5, 0.5);
    if DoReverse then
      GL.Normal3f(0, -1, 0)
    else
      GL.Normal3f(0, 1, 0);

    if FTopCap = ctCenter then
      GL.Vertex3f(0, 0, 0)
    else
    begin { FTopCap = ctFlat }
      if (Sign(SinP) = 1) or (tc1 = xyCurve) then
        SinPc1 := Power(SinP, xyCurve)
      else
        SinPc1 := -Power(-SinP, xyCurve);
      GL.Vertex3f(0, SinPc1*Radius, 0);

      N1 := YVector;
      if DoReverse then
        N1.Y := -N1.Y;
    end; { FTopCap = ctFlat }

    //  v1.Y := SinP;
    if (Sign(SinP) = 1) or (tc1 = xyCurve) then
      SinPc1 := Power(SinP, xyCurve)
    else
      SinPc1 := -Power(-SinP, xyCurve);
    v1.Y := SinPc1;

    Theta := AngStart;

    for i := 0 to FSlices do
    begin
      SinCos(Theta, SinT, CosT);
      //    v1.X := CosP * SinT;
      if (Sign(CosP) = 1) or (tc1 = xyCurve) then
        CosPc1 := Power(CosP, xyCurve)
      else
        CosPc1 := -Power(-CosP, xyCurve);

      if (Sign(SinT) = 1) or (tc2 = zCurve) then
        SinTc2 := Power(SinT, zCurve)
      else
        SinTc2 := -Power(-SinT, zCurve);
      v1.X := CosPc1 * SinTc2;

      //    v1.Z := CosP * CosT;
      if (Sign(CosT) = 1) or (tc2 = zCurve) then
        CosTc2 := Power(CosT, zCurve)
      else
        CosTc2 := -Power(-CosT, zCurve);
      v1.Z := CosPc1 * CosTc2;

      if FTopCap = ctCenter then
      begin
        N1 := VectorPerpendicular(YVector, v1);
        if DoReverse then
          NegateVector(N1);
      end;
      //    xgl.TexCoord2f(SinT * 0.5 + 0.5, CosT * 0.5 + 0.5);
      xgl.TexCoord2f(SinTc2 * 0.5 + 0.5, CosTc2 * 0.5 + 0.5);
      GL.Normal3fv(@N1);
      vs := v1;
      ScaleVector(vs, Radius);
      GL.Vertex3fv(@vs);
      Theta := Theta + StepH;
    end;
    GL.End_;
  end;

  // main body
  Phi := AngTop;
  Phi2 := Phi - StepV;
  uTexFactor := 1 / FSlices;
  vTexFactor := 1 / FStacks;

  for j := 0 to FStacks - 1 do
  begin
    Theta := AngStart;
    SinCos(Phi, SinP, CosP);
    SinCos(Phi2, SinP2, CosP2);

    if (Sign(SinP) = 1) or (tc1 = xyCurve) then
      SinPc1 := Power(SinP, xyCurve)
    else
      SinPc1 := -Power(-SinP, xyCurve);
    v1.Y := SinPc1;

    if (Sign(SinP2) = 1) or (tc1 = xyCurve) then
      SinPc1 := Power(SinP2, xyCurve)
    else
      SinPc1 := -Power(-SinP2, xyCurve);
    v2.Y := SinPc1;

    vTexCoord0 := 1 - j * vTexFactor;
    vTexCoord1 := 1 - (j + 1) * vTexFactor;

    GL.Begin_(GL_TRIANGLE_STRIP);
    for i := 0 to FSlices do
    begin
      SinCos(Theta, SinT, CosT);

      if (Sign(CosP) = 1) or (tc1 = xyCurve) then
        CosPc1 := Power(CosP, xyCurve)
      else
        CosPc1 := -Power(-CosP, xyCurve);

      if (Sign(SinT) = 1) or (tc2 = zCurve) then
        SinTc2 := Power(SinT, zCurve)
      else
        SinTc2 := -Power(-SinT, zCurve);
      v1.X := CosPc1 * SinTc2;

      if (Sign(CosP2) = 1) or (tc1 = xyCurve) then
        CosPc1 := Power(CosP2, xyCurve)
      else
        CosPc1 := -Power(-CosP2, xyCurve);
      V2.X := CosPc1 * SinTc2;

      if (Sign(CosP) = 1) or (tc1 = xyCurve) then
        CosPc1 := Power(CosP, xyCurve)
      else
        CosPc1 := -Power(-CosP, xyCurve);

      if (Sign(CosT) = 1) or (tc2 = zCurve) then
        CosTc2 := Power(CosT, zCurve)
      else
        CosTc2 := -Power(-CosT, zCurve);
      v1.Z := CosPc1 * CosTc2;

      if (Sign(CosP2) = 1) or (tc1 = xyCurve) then
        CosPc1 := Power(CosP2, xyCurve)
      else
        CosPc1 := -Power(-CosP2, xyCurve);
      V2.Z := CosPc1 * CosTc2;

      uTexCoord := i * uTexFactor;
      xgl.TexCoord2f(uTexCoord, vTexCoord0);
      if DoReverse then
      begin
        N1 := VectorNegate(v1);
        GL.Normal3fv(@N1);
      end
      else
        GL.Normal3fv(@v1);
      vs := v1;
      ScaleVector(vs, Radius);
      GL.Vertex3fv(@vs);

      xgl.TexCoord2f(uTexCoord, vTexCoord1);
      if DoReverse then
      begin
        N1 := VectorNegate(V2);
        GL.Normal3fv(@N1);
      end
      else
        GL.Normal3fv(@v2);
      vs := v2;
      ScaleVector(vs, Radius);
      GL.Vertex3fv(@vs);

      Theta := Theta + StepH;
    end;
    GL.End_;
    Phi := Phi2;
    Phi2 := Phi2 - StepV;
  end;

  // bottom cap
  if (FBottom > -90) and (FBottomCap in [ctCenter, ctFlat]) then
  begin
    GL.Begin_(GL_TRIANGLE_FAN);
    SinCos(AngBottom, SinP, CosP);
    xgl.TexCoord2f(0.5, 0.5);
    if DoReverse then
      GL.Normal3f(0, 1, 0)
    else
      GL.Normal3f(0, -1, 0);
    if FBottomCap = ctCenter then
      GL.Vertex3f(0, 0, 0)
    else
    begin { FTopCap = ctFlat }
      if (Sign(SinP) = 1) or (tc1 = xyCurve) then
        SinPc1 := Power(SinP, xyCurve)
      else
        SinPc1 := -Power(-SinP, xyCurve);
      GL.Vertex3f(0, SinPc1*Radius, 0);

      if DoReverse then
        MakeVector(N1, 0, -1, 0)
      else
        N1 := YVector;
    end;
    //  v1.Y := SinP;
    if (Sign(SinP) = 1) or (tc1 = xyCurve) then
      SinPc1 := Power(SinP, xyCurve)
    else
      SinPc1 := -Power(-SinP, xyCurve);
    v1.Y := SinPc1;

    Theta := AngStop;
    for i := 0 to FSlices do
    begin
      SinCos(Theta, SinT, CosT);
      //    v1.X := CosP * SinT;
      if (Sign(CosP) = 1) or (tc1 = xyCurve) then
        CosPc1 := Power(CosP, xyCurve)
      else
        CosPc1 := -Power(-CosP, xyCurve);

      if (Sign(SinT) = 1) or (tc2 = zCurve) then
        SinTc2 := Power(SinT, zCurve)
      else
        SinTc2 := -Power(-SinT, zCurve);
      v1.X := CosPc1 * SinTc2;

      //    v1.Z := CosP * CosT;
      if (Sign(CosT) = 1) or (tc2 = zCurve) then
        CosTc2 := Power(CosT, zCurve)
      else
        CosTc2 := -Power(-CosT, zCurve);
      v1.Z := CosPc1 * CosTc2;

      if FBottomCap = ctCenter then
      begin
        N1 := VectorPerpendicular(AffineVectorMake(0, -1, 0), v1);
        if DoReverse then
          NegateVector(N1);
        GL.Normal3fv(@N1);
      end;
      //    xgl.TexCoord2f(SinT * 0.5 + 0.5, CosT * 0.5 + 0.5);
      xgl.TexCoord2f(SinTc2 * 0.5 + 0.5, CosTc2 * 0.5 + 0.5);
      vs := v1;
      ScaleVector(vs, Radius);
      GL.Vertex3fv(@vs);
      Theta := Theta - StepH;
    end;
    GL.End_;
  end;
  if DoReverse then
    rci.GLStates.InvertGLFrontFace;
end;

// RayCastIntersect
// This will probably not work, karamba
// RayCastSphereIntersect -> RayCastSuperellipsoidIntersect ??????

function TGLSuperellipsoid.RayCastIntersect(const rayStart, rayVector: TVector;
  intersectPoint: PVector = nil; intersectNormal: PVector = nil): Boolean;
var
  i1, i2: TVector;
  localStart, localVector: TVector;
begin
  // compute coefficients of quartic polynomial
  SetVector(localStart, AbsoluteToLocal(rayStart));
  SetVector(localVector, AbsoluteToLocal(rayVector));
  NormalizeVector(localVector);
  if RayCastSphereIntersect(localStart, localVector, NullHmgVector, Radius, i1,
    i2) > 0 then
  begin
    Result := True;
    if Assigned(intersectPoint) then
      SetVector(intersectPoint^, LocalToAbsolute(i1));
    if Assigned(intersectNormal) then
    begin
      i1.W := 0; // vector transform
      SetVector(intersectNormal^, LocalToAbsolute(i1));
    end;
  end
  else
    Result := False;
end;

// GenerateSilhouette
// This will probably not work;

function TGLSuperellipsoid.GenerateSilhouette(const silhouetteParameters
  : TGLSilhouetteParameters): TGLSilhouette;
var
  i, j: Integer;
  s, C, angleFactor: Single;
  sVec, tVec: TAffineVector;
  Segments: Integer;
begin
  Segments := MaxInteger(FStacks, FSlices);

  // determine a local orthonormal matrix, viewer-oriented
  sVec := VectorCrossProduct(silhouetteParameters.SeenFrom, XVector);
  if VectorLength(sVec) < 1E-3 then
    sVec := VectorCrossProduct(silhouetteParameters.SeenFrom, YVector);
  tVec := VectorCrossProduct(silhouetteParameters.SeenFrom, sVec);
  NormalizeVector(sVec);
  NormalizeVector(tVec);
  // generate the silhouette (outline and capping)
  Result := TGLSilhouette.Create;
  angleFactor := (2 * PI) / Segments;
  for i := 0 to Segments - 1 do
  begin
    SinCos(i * angleFactor, FRadius, s, C);
    Result.vertices.AddPoint(VectorCombine(sVec, tVec, s, C));
    j := (i + 1) mod Segments;
    Result.Indices.Add(i, j);
    if silhouetteParameters.CappingRequired then
      Result.CapIndices.Add(Segments, i, j)
  end;
  if silhouetteParameters.CappingRequired then
    Result.vertices.Add(NullHmgPoint);
end;

// SetBottom
//

procedure TGLSuperellipsoid.SetBottom(aValue: TAngleLimit1);
begin
  if FBottom <> aValue then
  begin
    FBottom := aValue;
    StructureChanged;
  end;
end;

// SetBottomCap
//

procedure TGLSuperellipsoid.SetBottomCap(aValue: TCapType);
begin
  if FBottomCap <> aValue then
  begin
    FBottomCap := aValue;
    StructureChanged;
  end;
end;

// SetRadius
//

procedure TGLSuperellipsoid.SetRadius(const aValue: TGLFloat);
begin
  if aValue <> FRadius then
  begin
    FRadius := aValue;
    StructureChanged;
  end;
end;

// SetxyCurve
//

procedure TGLSuperellipsoid.SetxyCurve(const aValue: TGLFloat);
begin
  if aValue <> FxyCurve then
  begin
    FxyCurve := aValue;
    StructureChanged;
  end;
end;

// SetzCurve
//

procedure TGLSuperellipsoid.SetzCurve(const aValue: TGLFloat);
begin
  if aValue <> FzCurve then
  begin
    FzCurve := aValue;
    StructureChanged;
  end;
end;

// SetSlices
//

procedure TGLSuperellipsoid.SetSlices(aValue: Integer);
begin
  if aValue <> FSlices then
  begin
    if aValue <= 0 then
      FSlices := 1
    else
      FSlices := aValue;
    StructureChanged;
  end;
end;

// SetStacks
//

procedure TGLSuperellipsoid.SetStacks(aValue: TGLInt);
begin
  if aValue <> FStacks then
  begin
    if aValue <= 0 then
      FStacks := 1
    else
      FStacks := aValue;
    StructureChanged;
  end;
end;

// SetStart
//

procedure TGLSuperellipsoid.SetStart(aValue: TAngleLimit2);
begin
  if FStart <> aValue then
  begin
    Assert(aValue <= FStop);
    FStart := aValue;
    StructureChanged;
  end;
end;

// SetStop
//

procedure TGLSuperellipsoid.SetStop(aValue: TAngleLimit2);
begin
  if FStop <> aValue then
  begin
    Assert(aValue >= FStart);
    FStop := aValue;
    StructureChanged;
  end;
end;

// SetTop
//

procedure TGLSuperellipsoid.SetTop(aValue: TAngleLimit1);
begin
  if FTop <> aValue then
  begin
    FTop := aValue;
    StructureChanged;
  end;
end;

// SetTopCap
//

procedure TGLSuperellipsoid.SetTopCap(aValue: TCapType);
begin
  if FTopCap <> aValue then
  begin
    FTopCap := aValue;
    StructureChanged;
  end;
end;

 
//

procedure TGLSuperellipsoid.Assign(Source: TPersistent);
begin
  if Assigned(Source) and (Source is TGLSuperellipsoid) then
  begin
    FRadius := TGLSuperellipsoid(Source).FRadius;
    FSlices := TGLSuperellipsoid(Source).FSlices;
    FStacks := TGLSuperellipsoid(Source).FStacks;
    FBottom := TGLSuperellipsoid(Source).FBottom;
    FTop := TGLSuperellipsoid(Source).FTop;
    FStart := TGLSuperellipsoid(Source).FStart;
    FStop := TGLSuperellipsoid(Source).FStop;
  end;
  inherited Assign(Source);
end;

// AxisAlignedDimensions
//

function TGLSuperellipsoid.AxisAlignedDimensionsUnscaled: TVector;
begin
  Result.X := Abs(FRadius);
  Result.Y := Result.X;
  Result.Z := Result.X;
  Result.W := 0;
end;

// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------

initialization

// -------------------------------------------------------------
// -------------------------------------------------------------
// -------------------------------------------------------------

RegisterClasses([TGLSphere, TGLCube, TGLPlane, TGLSprite, TGLPoints,
  TGLDummyCube, TGLLines, TGLSuperellipsoid]);

end.
