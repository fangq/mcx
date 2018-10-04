//
// This unit is part of the GLScene Project, http://glscene.org
//
{
    Handles all the color and texture stuff.

  History :  
        04/01/13 - PW - Added ReleaseBitmap32 in TGLBlankImage destructor to remove
                           a memory leak (thanks to Lars Nebel)
        04/01/13 - PW - Moved cubic map texture consts CmtPX..CmtNZ from GLColor unit to here
        10/11/12 - PW - Added CPPB compatibility: used dummy instead abstract methods,
                           restored definition of TGLCubeMapTarget as integer type
        12/05/11 - Yar - Added KeepImageAfterTransfer for TGLTexture
        04/10/10 - Yar - Improved multycontext features for TGLTexture
        23/08/10 - Yar - Added OpenGLTokens to uses
        21/05/10 - Yar - Removed TGLFloatDataImage, replace OpenGL1x functions to OpenGLAdapter
        16/05/10 - Yar - Added protected method IsSelfLoading and LoadTexture to TGLTextureImage
        14/05/10 - Yar - Fixed UnpackAlignment in PrepareParams
        09/05/10 - Yar - Fixed texture compression (thanks Hacker)
        22/04/10 - Yar - Fixes after GLState revision
        05/03/10 - DanB - Removed disabling Texture Rect/CubeMap/3D, since disabling will
                             cause errors on hardware that doesn't support them
        05/03/10 - DanB - More state added to TGLStateCache
        23/01/10 - Yar  - Added TextureFormatEx to TGLTexture
                             and tfExtended to TGLTextureFormat (thanks mif for idea)
        22/01/10 - Yar  - Added GLTextureFormat to uses,
                             1D, 3D, array, cube map array target support,
                             texture error indication,
                             TGLTextureImage ResorceName property,
                             Depth property,
                             NativeTextureTarget becomes property
       07/01/10 - DaStr - Added tmAdd TextureMode and enhanced documentation
                             (thanks DungeonLords)
                             Removed IncludeTrailingBackslash function
       10/11/09 - DaStr - Added more texture formats (thanks YarUnderoaker)
       04/06/09 - DanB - Delphi 5 fix
       17/10/08 - DanB - changed some NotifyChange(Sender) calls to NotifyChange(Self)
       08/10/08 - DanB - split materials related stuff into GLMaterial.pas
       06/10/08 - DanB - added Assert check for trying to create texture images
       05/10/08 - DanB - separated texture image editor from texture unit
                            moved color related stuff to GLColor.pas
                            moved TGLRenderContextInfo into separate unit
       12/04/08 - DaStr - Bugfixed TGLTextureExItem.Create()
                              (thanks dAlex) (BugTracker ID = 1940451)
       10/04/08 - DaStr - Added a Delpi 5 interface bug work-around to
                              TGLMaterial (BugTracker ID = 1938988)
       08/02/08 - Mrqzzz - Added tiaBottomRightPointColorTransparent
       29/07/07 - LC - Modified how tmmEyeLinear is applied, see
                          Bugtracker ID = 1762966.
       06/06/07 - DaStr - Moved all color types, constants and functions
                              to GLColor.pas (Bugtracker ID = 1732211)
       31/03/07 - DaStr - Bugfixed TGLTexture.Assign (missed some properties)
                              (Bugtracker ID = 1692012) (thanks Zapology)
       28/03/07 - DaStr - Added explicit pointer dereferencing
                             (thanks Burkhard Carstens) (Bugtracker ID = 1678644)
       28/03/07 - DaStr - Renamed parameters in some methods
                             (thanks Burkhard Carstens) (Bugtracker ID = 1678658)
       23/03/07 - DaStr - Added missing parameters in procedure's implementation
                              (thanks Burkhard Carstens) (Bugtracker ID = 1681409)
       06/03/07 - DaStr - Removed obsolete FPC IFDEF's
                             (thanks Burkhard Carstens) (Bugtracker ID = 1678642)
       14/03/07 - DaStr - TGLPicFileImage now provides correct Width and Height
                                                        (BugtrackerID = 1680742)
       09/03/07 - DaStr - Added TGLMaterial.GetActualPrimaryMaterial, GetLibMaterial
                             Bugfixed TGLColor.Initialize and TGLColor.Destroy
                              (thanks Burkhard Carstens) (BugtrackerID = 1678650)
       04/03/07 - DaStr - Added TGLTextureTarget, [Encode/Decode]GLTextureTarget
       23/02/07 - DaStr - Added TGLShaderClass, TGLShaderFailedInitAction,
                              EGLShaderException
                             Added TGLShader.HandleFailedInitialization, ShaderSupported,
                              GetStardardNotSupportedMessage, FailedInitAction
                             Added default value for TGLShader.ShaderStyle
                             Fixed TGLShader.InitializeShader
                             Fixed TGLTextureExItem.Create (TGLCoordinatesStyle stuff)
       16/02/07 - DaStr - Global $Q- removed
                             Added TGLLibMaterials.GetTextureIndex, GetMaterialIndex,
                               GetNameOfTexture, GetNameOfLibMaterial
                             Added TGLMaterialLibrary.TextureByName,
                               GetNameOfTexture, GetNameOfLibMaterial
       01/02/07 - LIN - Added TGLLibMaterial.IsUsed : true if texture has registered users
       23/01/07 - LIN - Added TGLTextureImage.AssignToBitmap : Converts the TextureImage to a TBitmap
       23/01/07 - LIN - Added TGLTextureImage.AsBitmap : Returns the TextureImage as a TBitmap
       22/01/07 - DaStr - IGLMaterialLibrarySupported abstracted
                             TGLLibMaterial.TextureOffset/TextureScale.FStyle bugfxed (thanks Ian Mac)
       20/12/06 - DaStr - TGLColorManager.Enumcolors overloaded
                             TGLShader.Apply bugfixed, TGLShader.Assign added
       19/10/06 - LC - Fixed TGLLibMaterial.UnApply so it doesn't unapply a 2nd
                          texture that was never applied. Bugtracker ID=1234085
       19/10/06 - LC - Fixed TGLLibMaterial.Assign. Bugtracker ID=1549843 (thanks Zapology)
       15/09/06 - NC - TGLShader.handle as Integer -> Cardinal
       12/09/06 - NC - Added GetFloatTexImage and SetFloatTexImage
       06/03/05 - EG - FTextureEx now autocreated (like FTexture)
       30/11/04 - EG - No longer stores TextureEx if empty
       06/10/04 - NC - Corrected filtering param. setting for float texture,
                          Now keep using GL_TEXTURE_RECTANGLE_NV for TGLFloatDataImage
       05/10/04 - SG - Added Material.TextureEx (texture extension)
       04/10/04 - NC - Added TGLFloatDataImage
       02/08/04 - LR, YHC - BCB corrections: Added dummy method for the abstract problem
                          Changed type of TGLCubeMapTarget to integer
       03/07/04 - LR - Move InitWinColors to GLCrossPlatform
                          Replace TGraphics, TBitmap by TGLGraphics, TGLBitmap
       29/06/04 - SG - Added bmModulate blending mode
       08/04/04 - EG - Added AddMaterialsFromXxxx logic
       04/09/03 - EG - Added TGLShader.Enabled
       02/09/03 - EG - Added TGLColor.HSVA
       28/07/03 - aidave - Added TGLColor.RandomColor
       24/07/03 - EG - Introduced TGLTextureImageEditor mechanism
       04/07/03 - EG - Material.Texture now autocreating,
                          added per-texture brightness and gamma correction
       13/06/03 - EG - cubemap images can now be saved/restored as a whole
       05/06/03 - EG - Assign fixes (Andrzej Kaluza)
       23/05/03 - EG - More generic libmaterial registration
       08/12/02 - EG - Added tiaInverseLuminance
       13/11/02 - EG - Added tmmCubeMapLight0
       18/10/02 - EG - CubeMap texture matrix now setup for 2nd texture unit too
       24/07/02 - EG - Added TGLLibMaterials.DeleteUnusedMaterials
       13/07/02 - EG - Improved materials when lighting is off
       10/07/02 - EG - Added basic protection against cyclic material refs
       08/07/02 - EG - Multipass support
       18/06/02 - EG - Added TGLShader
       26/01/02 - EG - Makes use of new xglBegin/EndUpdate mechanism
       24/01/02 - EG - Added vUseDefaultSets mechanism,
                          TGLPictureImage no longer systematically creates a TPicture
       21/01/02 - EG - Fixed OnTextureNeeded calls (Leonel)
       20/01/02 - EG - Fixed texture memory use report error
       10/01/02 - EG - Added Material.FaceCulling, default texture filters
                          are now Linear/MipMap
       07/01/02 - EG - Added renderDPI to rci
       16/12/01 - EG - Added support for cube maps (texture and mappings)
       30/11/01 - EG - Texture-compression related errors now ignored (unsupported formats)
       14/09/01 - EG - Use of vFileStreamClass
       06/09/01 - EG - No longers depends on 'Windows'
       04/09/01 - EG - Texture binding cache
       31/08/01 - EG - tiaDefault wasn't honoured (Rene Lindsay)
       25/08/01 - EG - Added TGLBlankImage
       16/08/01 - EG - drawState now part of TGLRenderContextInfo
       15/08/01 - EG - TexGen support (object_linear, eye_linear and sphere_map)
       13/08/01 - EG - Fixed OnTextureNeeded handling (paths for mat lib)
       12/08/01 - EG - Completely rewritten handles management
       27/07/01 - EG - TGLLibMaterials now a TOwnedCollection
       19/07/01 - EG - Added "Enabled" to TGLTexture
       28/06/01 - EG - Added AddTextureMaterial TGraphic variant
       14/03/01 - EG - Streaming fixes by Uwe Raabe
       08/03/01 - EG - TGLPicFileImage.GetBitmap32 now resets filename if not found
       01/03/01 - EG - Fixed TGLMaterial.DestroyHandle,
                          Added Texture2 notifications and material cacheing
       26/02/01 - EG - Added support for GL_EXT_texture_filter_anisotropic
       23/02/01 - EG - Fixed texture matrix messup (second was using first)
       21/02/01 - EG - Minor fix for TextureImageRequiredMemory,
                          TexGen calls now based on XOpenGL
       14/02/01 - EG - Added support for texture format & texture compression
       31/01/01 - EG - Added Multitexture support
       28/01/01 - EG - Added MaterialOptions
       15/01/01 - EG - Enhanced TGLPicFileImage.LoadFromFile
       13/01/01 - EG - New helper functions for TGLMaterialLibrary
       08/01/01 - EG - Not-so-clean fix for TGLTexture.Destroy... better fix
                          will require awareness of rendering contexts...
       06/12/00 - EG - Added PrepareBuildList mechanism
       16/10/00 - EG - Fix in TGLPictureImage.Assign
       25/09/00 - EG - New texture management implemented
       13/08/00 - EG - Added AddTextureMaterial
       06/08/00 - EG - File not found error now happens only once per texture,
                          also added some more doc and texture transforms support
                          to TGLLibMaterial
       27/07/00 - EG - TGLPictureImage.Assign now accepts TGraphic & TPicture,
                          Added max texture size clamping
       15/07/00 - EG - Upgrade for new list/handle destruction scheme
       05/07/00 - EG - Added tiaTopLeftPointColorTransparent
       28/06/00 - EG - Added asserts for missing texture files
       01/06/00 - EG - Added ReloadTexture (support for texture library),
                          Fixed persistence of material names in a texture library
       28/05/00 - EG - TGLColor now has NotifyChange support for TGLBaseSceneObject
       23/04/00 - EG - Fixed bugs with TGLPicFileImage & TGLPersistentImage,
                          Added tiaOpaque
       17/04/00 - EG - Added Assign to DummyCube and Sprite
       16/04/00 - EG - Added TGLPicFileImage.Assign
       26/03/00 - EG - Finally fixed nasty bug in TGLMaterial.Free
       22/03/00 - EG - Added BeginUpdate/EndUpdate to TGLPictureImage,
          Made use of [Un]SetGLState in TGLMaterial
          (gain = 7-10% on T&L intensive rendering),
                          TGLTexBaseClass is no more (RIP)
       21/03/00 - EG - TGLMaterial props are now longer stored when it is
                          linked to a material library entry,
                          Added TGLPictureImage (split from TGLPersistentImage),
                          TGLPicFileImage has been updated and reactivated,
                          ColorManager is now autocreated and non longer force-linked.
       19/03/00 - EG - Added SaveToXxxx & LoadFromXxxx to TGLMaterialLibrary
       18/03/00 - EG - Added GetGLTextureImageClassesAsStrings,
                          Added FindGLTextureImageClassByFriendlyName,
                          FChanges states now ignored in TGLTexture.GetHandle,
                          Added SaveToFile/LoadFromFile to TextureImage
       17/03/00 - EG - Added tiaLuminance
       14/03/00 - EG - Added RegisterGLTextureImageClass stuff,
                          Added ImageAlpha
       13/03/00 - EG - Changed TGLTextureImage image persistence again,
                          Added "Edit" method for texture image classes,
                          TMagFilter/TMinFilter -> TGLMagFilter/TGLMinFilter
       03/03/00 - EG - Removed TImagePath,
                          Started major rework of the whole TGLTextureImage stuff,
                          Fixed and optimized TGLTexture.PrepareImage
       12/02/00 - EG - Added Material Library
       10/02/00 - EG - Fixed crash when texture is empty
       08/02/00 - EG - Added AsWinColor & DeclareCurrentAsDefault to TGLColor,
                          fixed notification on material property setXxx methods,
                          Objects now begin with 'TGL'
       07/02/00 - EG - "Update"s renamed to "NotifyChange"s
       06/02/00 - EG - RoundUpToPowerOf2, RoundDownToPowerOf2 and
                          IsPowerOf2 moved to GLMisc, added TGLPersistentImage.Assign,
                          fixed TGLMaterial.Assign,
                          disable inheritance stuff in TGLFaceProperties.Apply (needs fixing),
                          Diffuse & ambient color now default to openGL values
       05/02/00 - EG - Javadocisation, fixes and enhancements : 
                          TGLColor.Update, ConvertWinColor, TPicImage,
          TGLMaterial.Apply
    
}
unit GLTexture;

interface

{$I GLScene.inc}

uses
  Classes, SysUtils,

  GLStrings, GLCrossPlatform, GLBaseClasses, OpenGLTokens,
  GLVectorGeometry, GLGraphics, GLContext, GLState, GLColor, GLCoordinates,
  GLRenderContextInfo, GLTextureFormat, GLApplicationFileIO, GLUtils;

const
  cDefaultNormalMapScale = 0.125;

  CmtPX = 0;
  CmtNX = 1;
  CmtPY = 2;
  CmtNY = 3;
  CmtPZ = 4;
  CmtNZ = 5;

type
  TGLMinFilter =
  (
    miNearest,
    miLinear,
    miNearestMipmapNearest,
    miLinearMipmapNearest,
    miNearestMipmapLinear,
    miLinearMipmapLinear
  );

  TGLMagFilter = (maNearest, maLinear);

  TGLTextureMode = (tmDecal, tmModulate, tmBlend, tmReplace, tmAdd);
  TGLTextureWrap = (twBoth, twNone, twVertical, twHorizontal, twSeparate);

  // Specifies how depth values should be treated
  // during filtering and texture application
  TGLDepthTextureMode = (dtmLuminance, dtmIntensity, dtmAlpha);

  // Specifies the depth comparison function.
  TGLDepthCompareFunc = TDepthFunction;

  { Texture format for OpenGL (rendering) use.
  Internally, GLScene handles all "base" images as 32 Bits RGBA, but you can
  specify a generic format to reduce OpenGL texture memory use: }
  TGLTextureFormat = (
    tfDefault,
    tfRGB, // = tfRGB8
    tfRGBA, // = tfRGBA8
    tfRGB16, // = tfRGB5
    tfRGBA16, // = tfRGBA4
    tfAlpha, // = tfALPHA8
    tfLuminance, // = tfLUMINANCE8
    tfLuminanceAlpha, // = tfLUMINANCE8_ALPHA8
    tfIntensity, // = tfINTENSITY8
    tfNormalMap, // = tfRGB8
    tfRGBAFloat16, // = tfRGBA_FLOAT16_ATI
    tfRGBAFloat32, // = tfRGBA_FLOAT32_ATI
    tfExtended);

  // TGLTextureCompression
  //
  TGLTextureCompression = TGLInternalCompression;

  TGLTexture = class;

  IGLTextureNotifyAble = interface(IGLNotifyAble)
    ['{0D9DC0B0-ECE4-4513-A8A1-5AE7022C9426}']
    procedure NotifyTexMapChange(Sender: TObject);
  end;

  // TTextureNeededEvent
  //
  TTextureNeededEvent = procedure(Sender: TObject; var textureFileName: string)
    of object;

  TGLTextureChange = (tcImage, tcParams);
  TGLTextureChanges = set of TGLTextureChange;

  { Defines how and if Alpha channel is defined for a texture image. 
    tiaDefault : uses the alpha channel in the image if any
    tiaAlphaFromIntensity : the alpha channel value is deduced from other
    RGB components intensity (the brighter, the more opaque)
    tiaSuperBlackTransparent : pixels with a RGB color of (0, 0, 0) are
    completely transparent, others are completely opaque
    tiaLuminance : the luminance value is calculated for each pixel
    and used for RGB and Alpha values
    tiaLuminanceSqrt : same as tiaLuminance but with an Sqrt(Luminance)
        tiaOpaque : alpha channel is uniformously set to 1.0
        tiaTopLeftPointColorTransparent : points of the same color as the
          top left point of the bitmap are transparent, others are opaque.
        
    }
  TGLTextureImageAlpha =
  (
    tiaDefault,
    tiaAlphaFromIntensity,
    tiaSuperBlackTransparent,
    tiaLuminance,
    tiaLuminanceSqrt,
    tiaOpaque,
    tiaTopLeftPointColorTransparent,
    tiaInverseLuminance,
    tiaInverseLuminanceSqrt,
    tiaBottomRightPointColorTransparent
  );

  // TGLTextureImage
  //
  { Base class for texture image data.
   Basicly, subclasses are to be considered as different ways of getting
   a HBitmap (interfacing the actual source). 
   SubClasses should be registered using RegisterGLTextureImageClass to allow
   proper persistence and editability in the IDE experts. }
  TGLTextureImage = class(TGLUpdateAbleObject)
  private
    function GetResourceName: string;
  protected
    FOwnerTexture: TGLTexture;
    FOnTextureNeeded: TTextureNeededEvent;
    FResourceFile: string;
    class function IsSelfLoading: Boolean; virtual;
    procedure LoadTexture(AInternalFormat: TGLInternalFormat); virtual;
    function GetTextureTarget: TGLTextureTarget; virtual; abstract;
    function GetHeight: Integer; virtual; abstract;
    function GetWidth: Integer; virtual; abstract;
    function GetDepth: Integer; virtual; abstract;

    property OnTextureNeeded: TTextureNeededEvent read FOnTextureNeeded write
      FOnTextureNeeded;
  public
    { Public Properties }
    constructor Create(AOwner: TPersistent); override;
    destructor Destroy; override;

    property OwnerTexture: TGLTexture read FOwnerTexture write FOwnerTexture;
    procedure NotifyChange(Sender: TObject); override;

    { Save textureImage to file.
     This may not save a picture, but for instance, parameters, if the
     textureImage is a procedural texture. }
    procedure SaveToFile(const fileName: string); dynamic;abstract;
    { Load textureImage from a file.
     This may not load a picture, but for instance, parameters, if the
     textureImage is a procedural texture. 
             Subclasses should invoke inherited which will take care of the
             "OnTextureNeeded" stuff. }
    procedure LoadFromFile(const fileName: string); dynamic;
    { Returns a user-friendly denomination for the class.
     This denomination is used for picking a texture image class
     in the IDE expert. }
    class function FriendlyName: string; virtual;abstract;
    { Returns a user-friendly description for the class.
     This denomination is used for helping the user when picking a
     texture image class in the IDE expert. If it's not overriden,
     takes its value from FriendlyName. }
    class function FriendlyDescription: string; virtual;

    { Request reload/refresh of data upon next use. }
    procedure Invalidate; dynamic;

    { Returns image's bitmap handle.
     If the actual image is not a windows bitmap (BMP), descendants should
     take care of properly converting to bitmap. }
    function GetBitmap32: TGLImage; virtual; abstract;
    { Request for unloading bitmapData, to free some memory.
     This one is invoked when GLScene no longer needs the Bitmap data
     it got through a call to GetHBitmap. 
     Subclasses may ignore this call if the HBitmap was obtained at
     no particular memory cost. }
    procedure ReleaseBitmap32; virtual;
    //{ AsBitmap : Returns the TextureImage as a TBitmap }
    function AsBitmap: TGLBitmap;
    procedure AssignToBitmap(aBitmap: TGLBitmap);

    property Width: Integer read GetWidth;
    property Height: Integer read GetHeight;
    property Depth: Integer read GetDepth;
    { Native opengl texture target. }
    property NativeTextureTarget: TGLTextureTarget read GetTextureTarget;
    property ResourceName: string read GetResourceName;
  end;

  TGLTextureImageClass = class of TGLTextureImage;

  // TGLBlankImage
  //
  { A texture image with no specified content, only a size.
       This texture image type is of use if the context of your texture is
       calculated at run-time (with a TGLMemoryViewer for instance). }
  TGLBlankImage = class(TGLTextureImage)
  private
     
    procedure SetWidth(val: Integer);
    procedure SetHeight(val: Integer);
    procedure SetDepth(val: Integer);
    procedure SetCubeMap(const val: Boolean);
    procedure SetArray(const val: Boolean);
  protected
     
    fBitmap: TGLImage;

    fWidth, fHeight, fDepth: Integer;
    { Store a icolor format, because fBitmap is not always defined}
    fColorFormat: GLenum;
    { Blank Cube Map }
    fCubeMap: Boolean;
    { Flag to interparate depth as layer }
    fArray: Boolean;

    function GetWidth: Integer; override;
    function GetHeight: Integer; override;
    function GetDepth: Integer; override;
    function GetTextureTarget: TGLTextureTarget; override;
  public
     
    constructor Create(AOwner: TPersistent); override;
    destructor Destroy; override;

    procedure Assign(Source: TPersistent); override;

    function GetBitmap32: TGLImage; override;
    procedure ReleaseBitmap32; override;

    procedure SaveToFile(const fileName: string); override;
    procedure LoadFromFile(const fileName: string); override;
    class function FriendlyName: string; override;
    class function FriendlyDescription: string; override;

  published
     
    { Width, heigth and depth of the blank image (for memory allocation). }
    property Width: Integer read GetWidth write SetWidth default 256;
    property Height: Integer read GetHeight write SetHeight default 256;
    property Depth: Integer read GetDepth write SetDepth default 0;
    property CubeMap: Boolean read fCubeMap write SetCubeMap default false;
    property TextureArray: Boolean read fArray write SetArray default false;
    property ColorFormat: GLenum read fColorFormat write fColorFormat;
  end;

  // TGLPictureImage
  //
  { Base class for image data classes internally based on a TPicture. }
  TGLPictureImage = class(TGLTextureImage)
  private
     
    FBitmap: TGLImage;
    FGLPicture: TGLPicture;
    FUpdateCounter: Integer;

  protected
     
    function GetHeight: Integer; override;
    function GetWidth: Integer; override;
    function GetDepth: Integer; override;
    function GetTextureTarget: TGLTextureTarget; override;

    function GetPicture: TGLPicture;
    procedure SetPicture(const aPicture: TGLPicture);
    procedure PictureChanged(Sender: TObject);

  public
     
    constructor Create(AOwner: TPersistent); override;
    destructor Destroy; override;

    procedure Assign(Source: TPersistent); override;

    { Use this function if you are going to modify the Picture directly.
     Each invokation MUST be balanced by a call to EndUpdate. }
    procedure BeginUpdate;
    { Ends a direct picture modification session.
       Follows a BeginUpdate. }
    procedure EndUpdate;
    function GetBitmap32: TGLImage; override;
    procedure ReleaseBitmap32; override;

    { Holds the image content. }
    property Picture: TGLPicture read GetPicture write SetPicture;
  end;

  // TGLPersistentImage
  //
  { Stores any image compatible with Delphi's TPicture mechanism.
   The picture's data is actually stored into the DFM, the original
   picture name or path is not remembered. It is similar in behaviour
   to Delphi's TImage.
   Note that if original image is for instance JPEG format, only the JPEG
   data will be stored in the DFM (compact) }
  TGLPersistentImage = class(TGLPictureImage)
  private

  public
     
    constructor Create(AOwner: TPersistent); override;
    destructor Destroy; override;

    procedure SaveToFile(const fileName: string); override;
    procedure LoadFromFile(const fileName: string); override;
    class function FriendlyName: string; override;
    class function FriendlyDescription: string; override;
    property NativeTextureTarget;
  published
     
    property Picture;
  end;

  // TGLPicFileImage
  //
  { Uses a picture whose data is found in a file (only filename is stored).
       The image is unloaded after upload to OpenGL. }
  TGLPicFileImage = class(TGLPictureImage)
  private
    FPictureFileName: string;
    FAlreadyWarnedAboutMissingFile: Boolean;
    FWidth: Integer;
    FHeight: Integer;

  protected
    procedure SetPictureFileName(const val: string);
    function GetWidth: Integer; override;
    function GetHeight: Integer; override;
    function GetDepth: Integer; override;

  public
     
    constructor Create(AOwner: TPersistent); override;
    destructor Destroy; override;

    procedure Assign(Source: TPersistent); override;

    // Only picture file name is saved
    procedure SaveToFile(const fileName: string); override;
    { Load picture file name or use fileName as picture filename.
       The autodetection is based on the filelength and presence of zeros. }
    procedure LoadFromFile(const fileName: string); override;
    class function FriendlyName: string; override;
    class function FriendlyDescription: string; override;
    property NativeTextureTarget;

    function GetBitmap32: TGLImage; override;
    procedure Invalidate; override;

  published
    { Filename of the picture to use. }
    property PictureFileName: string read FPictureFileName write
      SetPictureFileName;
  end;


  // TGLCubeMapTarget
  //
 TGLCubeMapTarget = Integer;

  // TGLCubeMapImage
  //
  { A texture image used for specifying and stroing a cube map.
       Not unlike TGLPictureImage, but storing 6 of them instead of just one. 
       Saving & loading as a whole currently not supported. }
  TGLCubeMapImage = class(TGLTextureImage)
  private
     
    FImage: TGLImage;
    FUpdateCounter: Integer;
    FPicture: array[cmtPX..cmtNZ] of TGLPicture;
  protected
     
    function GetWidth: Integer; override;
    function GetHeight: Integer; override;
    function GetDepth: Integer; override;
    procedure SetPicture(index: TGLCubeMapTarget; const val: TGLPicture);
    function GetPicture(index: TGLCubeMapTarget): TGLPicture;
    function GetTextureTarget: TGLTextureTarget; override;

    procedure PictureChanged(Sender: TObject);

  public
     
    constructor Create(AOwner: TPersistent); override;
    destructor Destroy; override;

    procedure Assign(Source: TPersistent); override;

    function GetBitmap32: TGLImage; override;
    procedure ReleaseBitmap32; override;

    { Use this function if you are going to modify the Picture directly.
     Each invokation MUST be balanced by a call to EndUpdate. }
    procedure BeginUpdate;
    procedure EndUpdate;

    procedure SaveToFile(const fileName: string); override;
    procedure LoadFromFile(const fileName: string); override;
    class function FriendlyName: string; override;
    class function FriendlyDescription: string; override;
    property NativeTextureTarget;

    { Indexed access to the cube map's sub pictures. }
    property Picture[index: TGLCubeMapTarget]: TGLPicture read GetPicture write
    SetPicture;

  published
     
    property PicturePX: TGLPicture index cmtPX read GetPicture write SetPicture;
    property PictureNX: TGLPicture index cmtNX read GetPicture write SetPicture;
    property PicturePY: TGLPicture index cmtPY read GetPicture write SetPicture;
    property PictureNY: TGLPicture index cmtNY read GetPicture write SetPicture;
    property PicturePZ: TGLPicture index cmtPZ read GetPicture write SetPicture;
    property PictureNZ: TGLPicture index cmtNZ read GetPicture write SetPicture;
  end;

  // TGLTextureMappingMode
  //
  TGLTextureMappingMode = (tmmUser, tmmObjectLinear, tmmEyeLinear, tmmSphere,
    tmmCubeMapReflection, tmmCubeMapNormal,
    tmmCubeMapLight0, tmmCubeMapCamera);

  // TGLTexture
  //
    { Defines basic texturing properties.
       You can control texture wrapping, smoothing/filtering and of course define
       the texture map (note that texturing is disabled by default).
       A built-in mechanism (through ImageAlpha) allows auto-generation of an
       Alpha channel for all bitmaps (see TGLTextureImageAlpha). }
  TGLTexture = class(TGLUpdateAbleObject)
  private
     
    FTextureHandle: TGLTextureHandle;
    FSamplerHandle: TGLVirtualHandle;
    FTextureFormat: TGLInternalFormat;
    FTextureMode: TGLTextureMode;
    FTextureWrap: TGLTextureWrap;
    FMinFilter: TGLMinFilter;
    FMagFilter: TGLMagFilter;
    FDisabled: Boolean;
    FImage: TGLTextureImage;
    FImageAlpha: TGLTextureImageAlpha;
    FImageBrightness: Single;
    FImageGamma: Single;
    FMappingMode: TGLTextureMappingMode;
    FMapSCoordinates: TGLCoordinates4;
    FMapTCoordinates: TGLCoordinates4;
    FMapRCoordinates: TGLCoordinates4;
    FMapQCoordinates: TGLCoordinates4;
    FOnTextureNeeded: TTextureNeededEvent;
    FCompression: TGLTextureCompression;
    FRequiredMemorySize: Integer;
    FFilteringQuality: TGLTextureFilteringQuality;
    FTexWidth: Integer;
    FTexHeight: Integer;
    FTexDepth: Integer;
    FEnvColor: TGLColor;
    FBorderColor: TGLColor;
    FNormalMapScale: Single;
    FTextureWrapS: TGLSeparateTextureWrap;
    FTextureWrapT: TGLSeparateTextureWrap;
    FTextureWrapR: TGLSeparateTextureWrap;
    fTextureCompareMode: TGLTextureCompareMode;
    fTextureCompareFunc: TGLDepthCompareFunc;
    fDepthTextureMode: TGLDepthTextureMode;
    FKeepImageAfterTransfer: Boolean;
  protected
     
    procedure SetImage(AValue: TGLTextureImage);
    procedure SetImageAlpha(const val: TGLTextureImageAlpha);
    procedure SetImageBrightness(const val: Single);
    function StoreBrightness: Boolean;
    procedure SetImageGamma(const val: Single);
    function StoreGamma: Boolean;
    procedure SetMagFilter(AValue: TGLMagFilter);
    procedure SetMinFilter(AValue: TGLMinFilter);
    procedure SetTextureMode(AValue: TGLTextureMode);
    procedure SetTextureWrap(AValue: TGLTextureWrap);
    procedure SetTextureWrapS(AValue: TGLSeparateTextureWrap);
    procedure SetTextureWrapT(AValue: TGLSeparateTextureWrap);
    procedure SetTextureWrapR(AValue: TGLSeparateTextureWrap);
    function GetTextureFormat: TGLTextureFormat;
    procedure SetTextureFormat(const val: TGLTextureFormat);
    procedure SetTextureFormatEx(const val: TGLInternalFormat);
    function StoreTextureFormatEx: Boolean;
    procedure SetCompression(const val: TGLTextureCompression);
    procedure SetFilteringQuality(const val: TGLTextureFilteringQuality);
    procedure SetMappingMode(const val: TGLTextureMappingMode);
    function GetMappingSCoordinates: TGLCoordinates4;
    procedure SetMappingSCoordinates(const val: TGLCoordinates4);
    function StoreMappingSCoordinates: Boolean;
    function GetMappingTCoordinates: TGLCoordinates4;
    procedure SetMappingTCoordinates(const val: TGLCoordinates4);
    function StoreMappingTCoordinates: Boolean;
    function GetMappingRCoordinates: TGLCoordinates4;
    procedure SetMappingRCoordinates(const val: TGLCoordinates4);
    function StoreMappingRCoordinates: Boolean;
    function GetMappingQCoordinates: TGLCoordinates4;
    procedure SetMappingQCoordinates(const val: TGLCoordinates4);
    function StoreMappingQCoordinates: Boolean;
    procedure SetDisabled(AValue: Boolean);
    procedure SetEnabled(const val: Boolean);
    function GetEnabled: Boolean;
    procedure SetEnvColor(const val: TGLColor);
    procedure SetBorderColor(const val: TGLColor);
    procedure SetNormalMapScale(const val: Single);
    procedure SetTextureCompareMode(const val: TGLTextureCompareMode);
    procedure SetTextureCompareFunc(const val: TGLDepthCompareFunc);
    procedure SetDepthTextureMode(const val: TGLDepthTextureMode);
    function StoreNormalMapScale: Boolean;

    function StoreImageClassName: Boolean;

    function GetHandle: TGLuint; virtual;
    // Load texture to OpenGL subsystem
    procedure PrepareImage(target: TGLUInt); virtual;
    // Setup OpenGL texture parameters
    procedure PrepareParams(target: TGLUInt); virtual;

    procedure DoOnTextureNeeded(Sender: TObject; var textureFileName: string);
    procedure OnSamplerAllocate(Sender: TGLVirtualHandle; var Handle: Cardinal);
    procedure OnSamplerDestroy(Sender: TGLVirtualHandle; var Handle: Cardinal);
    // Shows a special image that indicates an error
    procedure SetTextureErrorImage;
  public
     
    constructor Create(AOwner: TPersistent); override;
    destructor Destroy; override;

    property OnTextureNeeded: TTextureNeededEvent read FOnTextureNeeded write
      FOnTextureNeeded;

    procedure PrepareBuildList;
    procedure ApplyMappingMode;
    procedure UnApplyMappingMode;
    procedure Apply(var rci: TGLRenderContextInfo);
    procedure UnApply(var rci: TGLRenderContextInfo);
    { Applies to TEXTURE1 }
    procedure ApplyAsTexture2(var rci: TGLRenderContextInfo; textureMatrix: PMatrix
      = nil);
    procedure UnApplyAsTexture2(var rci: TGLRenderContextInfo;
      reloadIdentityTextureMatrix: boolean);
    { N=1 for TEXTURE0, N=2 for TEXTURE1, etc. }
    procedure ApplyAsTextureN(n: Integer; var rci: TGLRenderContextInfo;
      textureMatrix: PMatrix = nil);
    procedure UnApplyAsTextureN(n: Integer; var rci: TGLRenderContextInfo;
      reloadIdentityTextureMatrix: boolean);

    procedure Assign(Source: TPersistent); override;
    procedure NotifyChange(Sender: TObject); override;
    procedure NotifyImageChange;
    procedure NotifyParamsChange;

    procedure DestroyHandles;

    procedure SetImageClassName(const val: string);
    function GetImageClassName: string;

    { Returns the OpenGL memory used by the texture.
      The compressed size is returned if, and only if texture compression
      if active and possible, and the texture has been allocated (Handle
      is defined), otherwise the estimated size (from TextureFormat
      specification) is returned. }
    function TextureImageRequiredMemory: Integer;
    { Allocates the texture handle if not already allocated.
      The texture is binded and parameters are setup, but no image data
      is initialized by this call - for expert use only. }
    function AllocateHandle: TGLuint;
    function IsHandleAllocated: Boolean;
    { Returns OpenGL texture format corresponding to current options. }
    function OpenGLTextureFormat: Integer;
    { Returns if of float data type}
    function IsFloatType: Boolean;
    { Is the texture enabled?.
      Always equals to 'not Disabled'. }
    property Enabled: Boolean read GetEnabled write SetEnabled;
    { Handle to the OpenGL texture object.
      If the handle hasn't already been allocated, it will be allocated
      by this call (ie. do not use if no OpenGL context is active!) }
    property Handle: TGLuint read GetHandle;
    property TextureHandle: TGLTextureHandle read FTextureHandle;

    { Actual width, height and depth used for last texture
      specification binding. }
    property TexWidth: Integer read FTexWidth;
    property TexHeight: Integer read FTexHeight;
    property TexDepth: Integer read FTexDepth;
    { Give texture rendering context }
  published
     

    { Image ClassName for enabling True polymorphism.
    This is ugly, but since the default streaming mechanism does a
    really bad job at storing	polymorphic owned-object properties,
    and neither TFiler nor TPicture allow proper use of the built-in
    streaming, that's the only way I found to allow a user-extensible
    mechanism. }
    property ImageClassName: string read GetImageClassName write
      SetImageClassName stored StoreImageClassName;
    { Image data for the texture. }
    property Image: TGLTextureImage read FImage write SetImage;

    { Automatic Image Alpha setting.
    Allows to control how and if the image's Alpha channel (transparency)
    is computed. }
    property ImageAlpha: TGLTextureImageAlpha read FImageAlpha write
      SetImageAlpha default tiaDefault;
    { Texture brightness correction.
    This correction is applied upon loading a TGLTextureImage, it's a
    simple saturating scaling applied to the RGB components of
    the 32 bits image, before it is passed to OpenGL, and before
    gamma correction (if any). }
    property ImageBrightness: Single read FImageBrightness write
      SetImageBrightness stored StoreBrightness;
    { Texture gamma correction.
    The gamma correction is applied upon loading a TGLTextureImage,
    applied to the RGB components of the 32 bits image, before it is
    passed to OpenGL, after brightness correction (if any). }
    property ImageGamma: Single read FImageGamma write SetImageGamma stored
      StoreGamma;

    { Texture magnification filter. }
    property MagFilter: TGLMagFilter read FMagFilter write SetMagFilter default
      maLinear;
    { Texture minification filter. }
    property MinFilter: TGLMinFilter read FMinFilter write SetMinFilter default
      miLinearMipMapLinear;
    { Texture application mode. }
    property TextureMode: TGLTextureMode read FTextureMode write SetTextureMode
      default tmDecal;
    { Wrapping mode for the texture. }
    property TextureWrap: TGLTextureWrap read FTextureWrap write SetTextureWrap
      default twBoth;
    { Wrapping mode for the texture when TextureWrap=twSeparate. }
    property TextureWrapS: TGLSeparateTextureWrap read FTextureWrapS write
      SetTextureWrapS default twRepeat;
    property TextureWrapT: TGLSeparateTextureWrap read FTextureWrapT write
      SetTextureWrapT default twRepeat;
    property TextureWrapR: TGLSeparateTextureWrap read FTextureWrapR write
      SetTextureWrapR default twRepeat;

    { Texture format for use by the renderer.
    See TGLTextureFormat for details. }
    property TextureFormat: TGLTextureFormat read GetTextureFormat write
      SetTextureFormat default tfDefault;
    property TextureFormatEx: TGLInternalFormat read FTextureFormat write
      SetTextureFormatEx stored StoreTextureFormatEx;

    { Texture compression control.
    If True the compressed TextureFormat variant (the OpenGL ICD must
    support GL_ARB_texture_compression, or this option is ignored). }
    property Compression: TGLTextureCompression read FCompression write
      SetCompression default tcDefault;
    { Specifies texture filtering quality.
    You can choose between bilinear and trilinear filetring (anisotropic).
    The OpenGL ICD must support GL_EXT_texture_filter_anisotropic or
    this property is ignored. }
    property FilteringQuality: TGLTextureFilteringQuality read FFilteringQuality
      write SetFilteringQuality default tfIsotropic;

    { Texture coordinates mapping mode.
    This property controls automatic texture coordinates generation. }
    property MappingMode: TGLTextureMappingMode read FMappingMode write
      SetMappingMode default tmmUser;
    { Texture mapping coordinates mode for S, T, R and Q axis.
    This property stores the coordinates for automatic texture
    coordinates generation. }
    property MappingSCoordinates: TGLCoordinates4 read GetMappingSCoordinates
      write SetMappingSCoordinates stored StoreMappingSCoordinates;
    property MappingTCoordinates: TGLCoordinates4 read GetMappingTCoordinates
      write SetMappingTCoordinates stored StoreMappingTCoordinates;
    property MappingRCoordinates: TGLCoordinates4 read GetMappingRCoordinates
      write SetMappingRCoordinates stored StoreMappingRCoordinates;
    property MappingQCoordinates: TGLCoordinates4 read GetMappingQCoordinates
      write SetMappingQCoordinates stored StoreMappingQCoordinates;

    { Texture Environment color. }
    property EnvColor: TGLColor read FEnvColor write SetEnvColor;
    { Texture Border color. }
    property BorderColor: TGLColor read FBorderColor write SetBorderColor;
    { If true, the texture is disabled (not used). }
    property Disabled: Boolean read FDisabled write SetDisabled default True;

    { Normal Map scaling.
    Only applies when TextureFormat is tfNormalMap, this property defines
    the scaling that is applied during normal map generation (ie. controls
    the intensity of the bumps). }
    property NormalMapScale: Single read FNormalMapScale write SetNormalMapScale
      stored StoreNormalMapScale;

    property TextureCompareMode: TGLTextureCompareMode read fTextureCompareMode
      write SetTextureCompareMode default tcmNone;
    property TextureCompareFunc: TGLDepthCompareFunc read fTextureCompareFunc
      write SetTextureCompareFunc default cfLequal;
    property DepthTextureMode: TGLDepthTextureMode read fDepthTextureMode write
      SetDepthTextureMode default dtmLuminance;

    { Disable image release after transfering it to VGA. }
    property KeepImageAfterTransfer: Boolean read FKeepImageAfterTransfer
      write FKeepImageAfterTransfer default False;
  end;

  // TGLTextureExItem
  //
  TGLTextureExItem = class(TCollectionItem, IGLTextureNotifyAble)
  private
    { Private Decalarations }
    FTexture: TGLTexture;
    FTextureIndex: Integer;
    FTextureOffset, FTextureScale: TGLCoordinates;
    FTextureMatrixIsIdentity: Boolean;
    FTextureMatrix: TMatrix;
    FApplied: Boolean;

    //implementing IInterface


      function QueryInterface(constref IID: TGUID; out Obj): HResult; {$IFNDEF WINDOWS}cdecl{$ELSE}stdcall{$ENDIF};
      function _AddRef: Integer; {$IFNDEF WINDOWS}cdecl{$ELSE}stdcall{$ENDIF};
      function _Release: Integer; {$IFNDEF WINDOWS}cdecl{$ELSE}stdcall{$ENDIF};

  protected
    { Protected Decalarations }
    function GetDisplayName: string; override;
    function GetOwner: TPersistent; override;
    procedure SetTexture(const Value: TGLTexture);
    procedure SetTextureIndex(const Value: Integer);
    procedure SetTextureOffset(const Value: TGLCoordinates);
    procedure SetTextureScale(const Value: TGLCoordinates);
    procedure NotifyTexMapChange(Sender: TObject);

    procedure CalculateTextureMatrix;

    procedure OnNotifyChange(Sender: TObject);

  public
    { Public Decalarations }
    constructor Create(ACollection: TCollection); override;
    destructor Destroy; override;

    procedure Assign(Source: TPersistent); override;
    procedure NotifyChange(Sender: TObject);

    procedure Apply(var rci: TGLRenderContextInfo);
    procedure UnApply(var rci: TGLRenderContextInfo);

  published
    { Published Decalarations }
    property Texture: TGLTexture read FTexture write SetTexture;
    property TextureIndex: Integer read FTextureIndex write SetTextureIndex;
    property TextureOffset: TGLCoordinates read FTextureOffset write
      SetTextureOffset;
    property TextureScale: TGLCoordinates read FTextureScale write
      SetTextureScale;

  end;

  // TGLTextureEx
  //
  TGLTextureEx = class(TCollection)
  private
    FOwner: TGLUpdateAbleObject;

  protected
    { Protected Decalarations }
    procedure SetItems(index: Integer; const Value: TGLTextureExItem);
    function GetItems(index: Integer): TGLTextureExItem;
    function GetOwner: TPersistent; override;
  public
    { Public Decalarations }
    constructor Create(AOwner: TGLUpdateAbleObject);

    procedure NotifyChange(Sender: TObject);
    procedure Apply(var rci: TGLRenderContextInfo);
    procedure UnApply(var rci: TGLRenderContextInfo);
    function IsTextureEnabled(Index: Integer): Boolean;

    function Add: TGLTextureExItem;

    property Items[index: Integer]: TGLTextureExItem read GetItems write
    SetItems; default;
    procedure Loaded;
  end;

  ETexture = class(Exception);
  EGLShaderException = class(Exception);

  // Register a TGLTextureImageClass (used for persistence and IDE purposes)
procedure RegisterGLTextureImageClass(textureImageClass: TGLTextureImageClass);
// Finds a registerer TGLTextureImageClass using its classname
function FindGLTextureImageClass(const className: string): TGLTextureImageClass;
// Finds a registerer TGLTextureImageClass using its FriendlyName
function FindGLTextureImageClassByFriendlyName(const friendlyName: string):
  TGLTextureImageClass;
// Defines a TStrings with the list of registered TGLTextureImageClass.
procedure SetGLTextureImageClassesToStrings(aStrings: TStrings);
{ Creates a TStrings with the list of registered TGLTextureImageClass.
 To be freed by caller. }
function GetGLTextureImageClassesAsStrings: TStrings;

procedure RegisterTGraphicClassFileExtension(const extension: string;
  const aClass: TGraphicClass);
function CreateGraphicFromFile(const fileName: string): TGLGraphic;

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
implementation
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

// TODO: remove dependancy on GLScene.pas unit (related to tmmCubeMapLight0)

uses
  GLScene,
  XOpenGL,
  PictureRegisteredFormats
  , GLVectorTypes;

const
  cTextureMode: array[tmDecal..tmAdd] of TGLEnum =
    (GL_DECAL, GL_MODULATE, GL_BLEND, GL_REPLACE, GL_ADD);

  cOldTextureFormatToInternalFormat: array[tfRGB..tfRGBAFloat32] of
    TGLInternalFormat = (
    tfRGB8,
    tfRGBA8,
    tfRGB5,
    tfRGBA4,
    tfALPHA8,
    tfLUMINANCE8,
    tfLUMINANCE8_ALPHA8,
    tfINTENSITY8,
    tfRGB8,
    tfRGBA_FLOAT16,
    tfRGBA_FLOAT32);

var
  vGLTextureImageClasses: TList;
  vTGraphicFileExtension: array of string;
  vTGraphicClass: array of TGraphicClass;

type
  TFriendlyImage = class(TGLBaseImage);



{%region%=====  'Helper functions ================================}

  // RegisterTGraphicClassFileExtension
  //

procedure RegisterTGraphicClassFileExtension(const extension: string;
  const aClass: TGraphicClass);
var
  n: Integer;
begin
  n := Length(vTGraphicFileExtension);
  SetLength(vTGraphicFileExtension, n + 1);
  SetLength(vTGraphicClass, n + 1);
  vTGraphicFileExtension[n] := LowerCase(extension);
  vTGraphicClass[n] := aClass;
end;

// CreateGraphicFromFile
//

function CreateGraphicFromFile(const fileName: string): TGLGraphic;
var
  i: Integer;
  ext: string;
  fs: TStream;
  graphicClass: TGraphicClass;
begin
  Result := nil;
  if FileStreamExists(fileName) then
  begin
    graphicClass := nil;
    ext := LowerCase(ExtractFileExt(fileName));
    for i := 0 to High(vTGraphicFileExtension) do
    begin
      if vTGraphicFileExtension[i] = ext then
      begin
        graphicClass := TGraphicClass(vTGraphicClass[i]);
        Break;
      end;
    end;
    if graphicClass = nil then
      graphicClass := GraphicClassForExtension(ext);
    if graphicClass <> nil then
    begin
      Result := graphicClass.Create;
      try
        fs := CreateFileStream(fileName, fmOpenRead);
        try
          Result.LoadFromStream(fs);
        finally
          fs.Free;
        end;
      except
        FreeAndNil(Result);
        raise;
      end;
    end;
  end;
end;

// RegisterGLTextureImageClass
//

procedure RegisterGLTextureImageClass(textureImageClass: TGLTextureImageClass);
begin
  if not Assigned(vGLTextureImageClasses) then
    vGLTextureImageClasses := TList.Create;
  vGLTextureImageClasses.Add(textureImageClass);
end;

// FindGLTextureImageClass
//

function FindGLTextureImageClass(const className: string): TGLTextureImageClass;
var
  i: Integer;
  tic: TGLTextureImageClass;
begin
  Result := nil;
  if Assigned(vGLTextureImageClasses) then
    for i := 0 to vGLTextureImageClasses.Count - 1 do
    begin
      tic := TGLTextureImageClass(vGLTextureImageClasses[i]);
      if tic.ClassName = className then
      begin
        Result := tic;
        Break;
      end;
    end;

end;

// FindGLTextureImageClassByFriendlyName
//

function FindGLTextureImageClassByFriendlyName(const friendlyName: string):
  TGLTextureImageClass;
var
  i: Integer;
  tic: TGLTextureImageClass;
begin
  Result := nil;
  if Assigned(vGLTextureImageClasses) then
    for i := 0 to vGLTextureImageClasses.Count - 1 do
    begin
      tic := TGLTextureImageClass(vGLTextureImageClasses[i]);
      if tic.FriendlyName = friendlyName then
      begin
        Result := tic;
        Break;
      end;
    end;
end;

// SetGLTextureImageClassesToStrings
//

procedure SetGLTextureImageClassesToStrings(aStrings: TStrings);
var
  i: Integer;
  tic: TGLTextureImageClass;
begin
  with aStrings do
  begin
    BeginUpdate;
    Clear;
    if Assigned(vGLTextureImageClasses) then
      for i := 0 to vGLTextureImageClasses.Count - 1 do
      begin
        tic := TGLTextureImageClass(vGLTextureImageClasses[i]);
        AddObject(tic.FriendlyName, TObject(Pointer(tic)));
      end;
    EndUpdate;
  end;
end;

// GetGLTextureImageClassesAsStrings
//

function GetGLTextureImageClassesAsStrings: TStrings;
begin
  Result := TStringList.Create;
  SetGLTextureImageClassesToStrings(Result);
end;

{%endregion%}

{%region%=====  'TGLTextureImage ================================}

// Create
//

constructor TGLTextureImage.Create(AOwner: TPersistent);
begin
  inherited;
  FOwnerTexture := (AOwner as TGLTexture);
end;

// Destroy
//

destructor TGLTextureImage.Destroy;
begin
  inherited Destroy;
end;

// FriendlyDescription
//

class function TGLTextureImage.FriendlyDescription: string;
begin
  Result := FriendlyName;
end;

procedure TGLTextureImage.Invalidate;
begin
  ReleaseBitmap32;
  NotifyChange(Self);
end;

// ReleaseBitmap32
//

procedure TGLTextureImage.ReleaseBitmap32;
begin
  // nothing here.
end;

// AsBitmap : Returns the TextureImage as a TBitmap
// WARNING: This Creates a new bitmap. Remember to free it, to prevent leaks.
// If possible, rather use AssignToBitmap.
//

function TGLTextureImage.AsBitmap: TGLBitmap;
begin
  result := self.GetBitmap32.Create32BitsBitmap;
end;

// AssignToBitmap
//

procedure TGLTextureImage.AssignToBitmap(aBitmap: TGLBitmap);
begin
  Self.GetBitmap32.AssignToBitmap(aBitmap);
end;

// NotifyChange
//

procedure TGLTextureImage.NotifyChange(Sender: TObject);
begin
  if Assigned(FOwnerTexture) then
  begin
    FOwnerTexture.FTextureHandle.NotifyChangesOfData;
    FOwnerTexture.FSamplerHandle.NotifyChangesOfData;
    // Check for texture target change
    GetTextureTarget;
    FOwnerTexture.NotifyChange(Self);
  end;
end;

 
//

procedure TGLTextureImage.LoadFromFile(const fileName: string);
var
  buf: string;
begin
  if Assigned(FOnTextureNeeded) then
  begin
    buf := fileName;
    FOnTextureNeeded(Self, buf);
  end;
end;

// GetResourceFile
//

function TGLTextureImage.GetResourceName: string;
begin
  Result := FResourceFile;
end;

class function TGLTextureImage.IsSelfLoading: Boolean;
begin
  Result := False;
end;

procedure TGLTextureImage.LoadTexture(AInternalFormat: TGLInternalFormat);
begin
end;

{%endregion%}

{%region%=====  'TGLBlankImage ================================}


constructor TGLBlankImage.Create(AOwner: TPersistent);
begin
  inherited;
  fWidth := 256;
  fHeight := 256;
  fDepth := 0;
  fColorFormat := GL_RGBA;
end;

destructor TGLBlankImage.Destroy;
begin
  ReleaseBitmap32;
  inherited Destroy;
end;

procedure TGLBlankImage.Assign(Source: TPersistent);
var
  img: TGLBlankImage;
begin
  if Assigned(Source) then
  begin
    if (Source is TGLBlankImage) then
    begin
      img := Source as TGLBlankImage;
      FWidth := img.Width;
      FHeight := img.Height;
      FDepth := img.Depth;
      FCubeMap := img.fCubeMap;
      FArray := img.fArray;
      fColorFormat := img.ColorFormat;
      FResourceFile := img.ResourceName;
      Invalidate;
    end
    else
      GetBitmap32.Assign(Source);
    NotifyChange(Self);
  end
  else
    inherited;
end;

procedure TGLBlankImage.SetWidth(val: Integer);
begin
  if val <> FWidth then
  begin
    FWidth := val;
    if FWidth < 1 then
      FWidth := 1;
    Invalidate;
  end;
end;

function TGLBlankImage.GetWidth: Integer;
begin
  Result := FWidth;
end;

procedure TGLBlankImage.SetHeight(val: Integer);
begin
  if val <> FHeight then
  begin
    FHeight := val;
    if FHeight < 1 then
      FHeight := 1;
    Invalidate;
  end;
end;

function TGLBlankImage.GetHeight: Integer;
begin
  Result := FHeight;
end;

procedure TGLBlankImage.SetDepth(val: Integer);
begin
  if val <> FDepth then
  begin
    FDepth := val;
    if FDepth < 0 then
      FDepth := 0;
    Invalidate;
  end;
end;

function TGLBlankImage.GetDepth: Integer;
begin
  Result := fDepth;
end;

procedure TGLBlankImage.SetCubeMap(const val: Boolean);
begin
  if val <> fCubeMap then
  begin
    fCubeMap := val;
    Invalidate;
  end;
end;

procedure TGLBlankImage.SetArray(const val: Boolean);
begin
  if val <> fArray then
  begin
    fArray := val;
    Invalidate;
  end;
end;

function TGLBlankImage.GetBitmap32: TGLImage;
begin
  if not Assigned(FBitmap) then
  begin
    fBitmap := TGLImage.Create;
    fBitmap.Width := FWidth;
    fBitmap.Height := FHeight;
    fBitmap.Depth := FDepth;
    fBitmap.CubeMap := FCubeMap;
    fBitmap.TextureArray := FArray;
    fBitmap.SetColorFormatDataType(FColorFormat, GL_UNSIGNED_BYTE);
  end;
  Result := FBitmap;
end;

procedure TGLBlankImage.ReleaseBitmap32;
begin
  if Assigned(FBitmap) then
  begin
    FBitmap.Free;
    FBitmap := nil;
  end;
end;

procedure TGLBlankImage.SaveToFile(const fileName: string);
begin
  SaveAnsiStringToFile(fileName, AnsiString(
    '[BlankImage]'#13#10'Width=' + IntToStr(Width) +
    #13#10'Height=' + IntToStr(Height) +
    #13#10'Depth=' + IntToStr(Depth)));
end;

procedure TGLBlankImage.LoadFromFile(const fileName: string);
var
  sl: TStringList;
  buf, temp: string;
begin
  buf := fileName;
  if Assigned(FOnTextureNeeded) then
    FOnTextureNeeded(Self, buf);
  if FileExists(buf) then
  begin
    sl := TStringList.Create;
    try
      sl.LoadFromFile(buf{$IFDEF GLS_DELPHI_2009_UP}, TEncoding.ASCII{$ENDIF});
      FWidth := StrToInt(sl.Values['Width']);
      FHeight := StrToInt(sl.Values['Height']);
      temp := sl.Values['Depth'];
      if Length(temp) > 0 then
        FDepth := StrToInt(temp)
      else
        FDepth := 1;
    finally
      sl.Free;
    end;
  end
  else
  begin
    Assert(False, Format(glsFailedOpenFile, [fileName]));
  end;
end;

class function TGLBlankImage.FriendlyName: string;
begin
  Result := 'Blank Image';
end;

class function TGLBlankImage.FriendlyDescription: string;
begin
  Result := 'Blank Image (Width x Height x Depth)';
end;

function TGLBlankImage.GetTextureTarget: TGLTextureTarget;
begin
  Result := ttTexture2D;
  // Choose a texture target
  if Assigned(fBitmap) then
  begin
    FWidth := fBitmap.Width;
    FHeight := fBitmap.Height;
    FDepth := fBitmap.Depth;
    FCubeMap := fBitmap.CubeMap;
    FArray := fBitmap.TextureArray;
  end;

  if FHeight = 1 then
    Result := ttTexture1D;
  if FCubeMap then
    Result := ttTextureCube;
  if FDepth > 0 then
    Result := ttTexture3D;
  if FArray then
  begin
    if FDepth < 2 then
      Result := ttTexture1DArray
    else
      Result := ttTexture2DArray;
    if FCubeMap then
      Result := ttTextureCubeArray;
  end;

  if Assigned(FOwnerTexture) then
  begin
    if ((FOwnerTexture.FTextureFormat >= tfFLOAT_R16)
      and (FOwnerTexture.FTextureFormat <= tfFLOAT_RGBA32)) then
      Result := ttTextureRect;
  end;
end;

{%endregion%}

{%region%=====  'TGLPictureImage ================================}

// Create
//

constructor TGLPictureImage.Create(AOwner: TPersistent);
begin
  inherited;
end;

// Destroy
//

destructor TGLPictureImage.Destroy;
begin
  ReleaseBitmap32;
  FGLPicture.Free;
  inherited Destroy;
end;

 
//

procedure TGLPictureImage.Assign(Source: TPersistent);
var
  bmp: TGLBitmap;
begin
  if Assigned(Source) then
  begin
    if (Source is TGLPersistentImage) then
      Picture.Assign(TGLPersistentImage(Source).Picture)
    else if (Source is TGLGraphic) then
      Picture.Assign(Source)
    else if (Source is TGLPicture) then
      Picture.Assign(Source)
    else if (Source is TGLImage) then
    begin
      bmp := TGLImage(Source).Create32BitsBitmap;
      Picture.Graphic := bmp;
      bmp.Free;
      FResourceFile := TGLImage(Source).ResourceName;
    end
    else
      inherited;
  end
  else
    inherited;
end;

// BeginUpdate
//

procedure TGLPictureImage.BeginUpdate;
begin
  Inc(FUpdateCounter);
  Picture.OnChange := nil;
end;

// EndUpdate
//

procedure TGLPictureImage.EndUpdate;
begin
  Assert(FUpdateCounter > 0, ClassName + ': Unbalanced Begin/EndUpdate');
  Dec(FUpdateCounter);
  Picture.OnChange := PictureChanged;
  if FUpdateCounter = 0 then
    PictureChanged(Picture);
end;

// GetHeight
//

function TGLPictureImage.GetHeight: Integer;
begin
  Result := Picture.Height;
end;

// GetWidth
//

function TGLPictureImage.GetWidth: Integer;
begin
  Result := Picture.Width;
end;

// GetDepth
//

function TGLPictureImage.GetDepth: Integer;
begin
  Result := 0;
end;

// GetBitmap32
//

function TGLPictureImage.GetBitmap32: TGLImage;
begin
  if not Assigned(FBitmap) then
  begin
    FBitmap := TGLImage.Create;
    // we need to deactivate OnChange, due to a "glitch" in some TGraphics,
    // for instance, TJPegImage triggers an OnChange when it is drawn...
    if Assigned(Picture.Graphic) then
    begin
      if Assigned(Picture.OnChange) then
      begin
        Picture.OnChange := nil;
        try
          FBitmap.Assign(Picture.Graphic);
        finally
          Picture.OnChange := PictureChanged;
        end;
      end
      else
        FBitmap.Assign(Picture.Graphic);
    end
    else
      FBitmap.SetErrorImage;
  end;
  Result := FBitmap;
end;

// ReleaseBitmap32
//

procedure TGLPictureImage.ReleaseBitmap32;
begin
  if Assigned(FBitmap) then
  begin
    FBitmap.Free;
    FBitmap := nil;
  end;
end;

// PictureChanged
//

procedure TGLPictureImage.PictureChanged(Sender: TObject);
begin
  Invalidate;
end;

// GetPicture
//

function TGLPictureImage.GetPicture: TGLPicture;
begin
  if not Assigned(FGLPicture) then
  begin
    FGLPicture := TGLPicture.Create;
    FGLPicture.OnChange := PictureChanged;
  end;
  Result := FGLPicture;
end;

// SetPicture
//

procedure TGLPictureImage.SetPicture(const aPicture: TGLPicture);
begin
  Picture.Assign(aPicture);
end;

// GetTextureTarget
//

function TGLPictureImage.GetTextureTarget: TGLTextureTarget;
begin
  Result := ttTexture2D;
end;

{%endregion%}

{%region%=====  'TGLPersistentImage ================================}

// Create
//

constructor TGLPersistentImage.Create(AOwner: TPersistent);
begin
  inherited;
end;

// Destroy
//

destructor TGLPersistentImage.Destroy;
begin
  inherited Destroy;
end;

// SaveToFile
//

procedure TGLPersistentImage.SaveToFile(const fileName: string);
begin
  Picture.SaveToFile(fileName);
  FResourceFile := fileName;
end;

 
//

procedure TGLPersistentImage.LoadFromFile(const fileName: string);
var
  buf: string;
  gr: TGLGraphic;
begin
  buf := fileName;
  FResourceFile := fileName;
  if Assigned(FOnTextureNeeded) then
    FOnTextureNeeded(Self, buf);
  if ApplicationFileIODefined then
  begin
    gr := CreateGraphicFromFile(buf);
    if Assigned(gr) then
    begin
      Picture.Graphic := gr;
      gr.Free;
      Exit;
    end;
  end
  else if FileExists(buf) then
  begin
    Picture.LoadFromFile(buf);
    Exit;
  end;
  Picture.Graphic := nil;
  raise ETexture.CreateFmt(glsFailedOpenFile, [fileName]);
end;

 
//

class function TGLPersistentImage.FriendlyName: string;
begin
  Result := 'Persistent Image';
end;

// FriendlyDescription
//

class function TGLPersistentImage.FriendlyDescription: string;
begin
  Result := 'Image data is stored in its original format with other form resources,'
    + 'ie. in the DFM at design-time, and embedded in the EXE at run-time.';
end;

{%endregion%}


{%region%=====  'TGLPicFileImage ================================}

// Create
//

constructor TGLPicFileImage.Create(AOwner: TPersistent);
begin
  inherited;
end;

// Destroy
//

destructor TGLPicFileImage.Destroy;
begin
  inherited;
end;

 
//

procedure TGLPicFileImage.Assign(Source: TPersistent);
begin
  if Source is TGLPicFileImage then
  begin
    FPictureFileName := TGLPicFileImage(Source).FPictureFileName;
    FResourceFile := TGLPicFileImage(Source).ResourceName;
  end
  else
    inherited;
end;

// SetPictureFileName
//

procedure TGLPicFileImage.SetPictureFileName(const val: string);
begin
  if val <> FPictureFileName then
  begin
    FPictureFileName := val;
    FResourceFile := val;
    FAlreadyWarnedAboutMissingFile := False;
    Invalidate;
  end;
end;

procedure TGLPicFileImage.Invalidate;
begin
  Picture.OnChange := nil;
  try
    Picture.Assign(nil);
    FBitmap := nil;
  finally
    Picture.OnChange := PictureChanged;
  end;
  inherited;
end;

function TGLPicFileImage.GetHeight: Integer;
begin
  Result := FHeight;
end;

// GetWidth
//

function TGLPicFileImage.GetWidth: Integer;
begin
  Result := FWidth;
end;

function TGLPicFileImage.GetDepth: Integer;
begin
  Result := 0;
end;

function TGLPicFileImage.GetBitmap32: TGLImage;
var
  buf: string;
  gr: TGLGraphic;
begin
  if (GetWidth <= 0) and (PictureFileName <> '') then
  begin
    Picture.OnChange := nil;
    try
      buf := PictureFileName;
      SetExeDirectory;
      if Assigned(FOnTextureNeeded) then
        FOnTextureNeeded(Self, buf);
      if FileStreamExists(buf) then
      begin
        gr := CreateGraphicFromFile(buf);
        Picture.Graphic := gr;
        gr.Free;
      end
      else
      begin
        Picture.Graphic := nil;
        if not FAlreadyWarnedAboutMissingFile then
        begin
          FAlreadyWarnedAboutMissingFile := True;
          GLOKMessageBox(Format(glsFailedOpenFileFromCurrentDir, [PictureFileName, GetCurrentDir]),glsError);
        end;
      end;
      Result := inherited GetBitmap32;
      FWidth := Result.Width;
      FHeight := Result.Height;
      Picture.Graphic := nil;
    finally
      Picture.OnChange := PictureChanged;
    end;
  end
  else
    Result := inherited GetBitmap32;
end;

// SaveToFile
//

procedure TGLPicFileImage.SaveToFile(const fileName: string);
begin
  FResourceFile := fileName;
  SaveAnsiStringToFile(fileName, AnsiString(PictureFileName));
end;

 
//

procedure TGLPicFileImage.LoadFromFile(const fileName: string);
var
  buf: string;
begin
  inherited;
  // attempt to autodetect if we are pointed to a file containing
  // a filename or directly to an image
  if SizeOfFile(fileName) < 512 then
  begin
    buf := string(LoadAnsiStringFromFile(fileName));
    if Pos(#0, buf) > 0 then
      PictureFileName := fileName
    else
      PictureFileName := buf;
  end
  else
    PictureFileName := fileName;
  FResourceFile := FPictureFileName;
end;

 
//

class function TGLPicFileImage.FriendlyName: string;
begin
  Result := 'PicFile Image';
end;

// FriendlyDescription
//

class function TGLPicFileImage.FriendlyDescription: string;
begin
  Result := 'Image data is retrieved from a file.';
end;

{%endregion%}

{%region%=====  'TGLCubeMapImage ================================}

// Create
//

constructor TGLCubeMapImage.Create(AOwner: TPersistent);
var
  i: TGLCubeMapTarget;
begin
  inherited;
  for i := Low(FPicture) to High(FPicture) do
  begin
    FPicture[i] := TGLPicture.Create;
    FPicture[i].OnChange := PictureChanged;
  end;
end;

// Destroy
//

destructor TGLCubeMapImage.Destroy;
var
  i: TGLCubeMapTarget;
begin
  ReleaseBitmap32;
  for i := Low(FPicture) to High(FPicture) do
    FPicture[i].Free;
  inherited Destroy;
end;

 
//

procedure TGLCubeMapImage.Assign(Source: TPersistent);
var
  i: TGLCubeMapTarget;
begin
  if Assigned(Source) then
  begin
    if (Source is TGLCubeMapImage) then
    begin
      for i := Low(FPicture) to High(FPicture) do
        FPicture[i].Assign(TGLCubeMapImage(Source).FPicture[i]);
      Invalidate;
    end
    else
      inherited;
  end
  else
    inherited;
end;

// GetWidth
//

function TGLCubeMapImage.GetWidth: Integer;
begin
  Result := FPicture[cmtPX].Width;
end;

// GetHeight
//

function TGLCubeMapImage.GetHeight: Integer;
begin
  Result := FPicture[cmtPX].Height;
end;

// GetDepth
//

function TGLCubeMapImage.GetDepth: Integer;
begin
  Result := 0;
end;

// GetBitmap32
//

function TGLCubeMapImage.GetBitmap32: TGLImage;
var
  I: Integer;
  LImage: TGLImage;
begin
  if Assigned(FImage) then
    FImage.Free;
  LImage := TGLImage.Create;
  LImage.VerticalReverseOnAssignFromBitmap := True;

  try
    for I := 0 to 5 do
    begin
      FPicture[TGLCubeMapTarget(I)].OnChange := nil;
      try
        LImage.Assign(FPicture[TGLCubeMapTarget(I)].Graphic);
        if not Assigned(FImage) then
        begin
          FImage := TGLImage.Create;
          FImage.Blank := True;
          FImage.Width := LImage.Width;
          FImage.Height := LImage.Height;
          FImage.SetColorFormatDataType(LImage.ColorFormat, LImage.DataType);
          FImage.CubeMap := True;
          FImage.Blank := False;
        end;
        Move(LImage.Data^, TFriendlyImage(FImage).GetLevelAddress(0, I)^, LImage.LevelSizeInByte[0]);
      finally
        FPicture[TGLCubeMapTarget(I)].OnChange := PictureChanged;
      end;
    end;
  finally
    LImage.Destroy;
  end;
  Result := FImage;
end;

// ReleaseBitmap32
//

procedure TGLCubeMapImage.ReleaseBitmap32;
begin
  if Assigned(FImage) then
  begin
    FImage.Free;
    FImage := nil;
  end;
end;

// BeginUpdate
//

procedure TGLCubeMapImage.BeginUpdate;
var
  i: TGLCubeMapTarget;
begin
  Inc(FUpdateCounter);
  for i := Low(FPicture) to High(FPicture) do
    FPicture[i].OnChange := nil;
end;

// EndUpdate
//

procedure TGLCubeMapImage.EndUpdate;
var
  i: TGLCubeMapTarget;
begin
  Assert(FUpdateCounter > 0, ClassName + ': Unbalanced Begin/EndUpdate');
  Dec(FUpdateCounter);
  for i := Low(FPicture) to High(FPicture) do
    FPicture[i].OnChange := PictureChanged;
  if FUpdateCounter = 0 then
    PictureChanged(FPicture[cmtPX]);
end;

// SaveToFile
//

procedure TGLCubeMapImage.SaveToFile(const fileName: string);
var
  fs: TFileStream;
  bmp: TGLBitmap;
  i: TGLCubeMapTarget;
  version: Word;
begin
  fs := TFileStream.Create(fileName, fmCreate);
  bmp := TGLBitmap.Create;
  try
    version := $0100;
    fs.Write(version, 2);
    for i := Low(FPicture) to High(FPicture) do
    begin
      bmp.Assign(FPicture[i].Graphic);
      bmp.SaveToStream(fs);
    end;
  finally
    bmp.Free;
    fs.Free;
  end;
end;

 
//

procedure TGLCubeMapImage.LoadFromFile(const fileName: string);
var
  fs: TFileStream;
  bmp: TGLBitmap;
  i: TGLCubeMapTarget;
  version: Word;
begin
  fs := TFileStream.Create(fileName, fmOpenRead + fmShareDenyWrite);
  bmp := TGLBitmap.Create;
  try
    fs.Read(version, 2);
    Assert(version = $0100);
    for i := Low(FPicture) to High(FPicture) do
    begin
      bmp.LoadFromStream(fs);
      FPicture[i].Graphic := bmp;
    end;
  finally
    bmp.Free;
    fs.Free;
  end;
end;

 
//

class function TGLCubeMapImage.FriendlyName: string;
begin
  Result := 'CubeMap Image';
end;

// FriendlyDescription
//

class function TGLCubeMapImage.FriendlyDescription: string;
begin
  Result := 'Image data is contain 6 pictures of cubemap faces.';
end;

// PictureChanged
//

procedure TGLCubeMapImage.PictureChanged(Sender: TObject);
begin
  Invalidate;
end;

// GetTextureTarget
//

function TGLCubeMapImage.GetTextureTarget: TGLTextureTarget;
begin
  Result := ttTextureCube;
end;

// SetPicture
//

procedure TGLCubeMapImage.SetPicture(index: TGLCubeMapTarget; const val:
  TGLPicture);
begin
  FPicture[index].Assign(val);
end;

function TGLCubeMapImage.GetPicture(index: TGLCubeMapTarget): TGLPicture;
begin
  Result := FPicture[index];
end;

{%endregion%}


{%region%=====  'TGLTexture ================================}


constructor TGLTexture.Create(AOwner: TPersistent);
begin
  inherited;
  FDisabled := True;
  FImage := TGLPersistentImage.Create(Self);
  FImage.OnTextureNeeded := DoOnTextureNeeded;
  FImageAlpha := tiaDefault;
  FImageBrightness := 1.0;
  FImageGamma := 1.0;
  FMagFilter := maLinear;
  FMinFilter := miLinearMipMapLinear;
  FFilteringQuality := tfIsotropic;
  FRequiredMemorySize := -1;
  FTextureHandle := TGLTextureHandle.Create;
  FSamplerHandle := TGLVirtualHandle.Create;
  FSamplerHandle.OnAllocate := OnSamplerAllocate;
  FSamplerHandle.OnDestroy := OnSamplerDestroy;
  FMappingMode := tmmUser;
  FEnvColor := TGLColor.CreateInitialized(Self, clrTransparent);
  FBorderColor := TGLColor.CreateInitialized(Self, clrTransparent);
  FNormalMapScale := cDefaultNormalMapScale;
  FTextureCompareMode := tcmNone;
  FTextureCompareFunc := cfLequal;
  FDepthTextureMode := dtmLuminance;
  TextureFormat := tfDefault;
  FCompression := tcDefault;
  FKeepImageAfterTransfer := False;
end;

destructor TGLTexture.Destroy;
begin
  FEnvColor.Free;
  FBorderColor.Free;
  FMapSCoordinates.Free;
  FMapTCoordinates.Free;
  FMapRCoordinates.Free;
  FMapQCoordinates.Free;
  DestroyHandles;
  FTextureHandle.Free;
  FSamplerHandle.Free;
  FImage.Free;
  inherited Destroy;
end;

procedure TGLTexture.Assign(Source: TPersistent);
begin
  if Assigned(Source) then
  begin
    if (Source is TGLTexture) then
    begin
      if Source <> Self then
      begin
        FImageAlpha := TGLTexture(Source).FImageAlpha;
        FTextureMode := TGLTexture(Source).FTextureMode;
        FTextureWrap := TGLTexture(Source).FTextureWrap;
        FTextureFormat := TGLTexture(Source).FTextureFormat;
        FCompression := TGLTexture(Source).FCompression;
        FMinFilter := TGLTexture(Source).FMinFilter;
        FMagFilter := TGLTexture(Source).FMagFilter;
        FMappingMode := TGLTexture(Source).FMappingMode;
        MappingSCoordinates.Assign(TGLTexture(Source).MappingSCoordinates);
        MappingTCoordinates.Assign(TGLTexture(Source).MappingTCoordinates);
        MappingRCoordinates.Assign(TGLTexture(Source).MappingRCoordinates);
        MappingQCoordinates.Assign(TGLTexture(Source).MappingQCoordinates);
        FDisabled := TGLTexture(Source).FDisabled;
        SetImage(TGLTexture(Source).FImage);
        FImageBrightness := TGLTexture(Source).FImageBrightness;
        FImageGamma := TGLTexture(Source).FImageGamma;
        FFilteringQuality := TGLTexture(Source).FFilteringQuality;
        FEnvColor.Assign(TGLTexture(Source).FEnvColor);
        FBorderColor.Assign(TGLTexture(Source).FBorderColor);
        FNormalMapScale := TGLTexture(Source).FNormalMapScale;
        // Probably don't need to assign these....
        // FOnTextureNeeded := TGLTexture(Source).FImageGamma;
        // FRequiredMemorySize  : Integer;
        // FTexWidth, FTexHeight : Integer;
        FTextureHandle.NotifyChangesOfData;
        FSamplerHandle.NotifyChangesOfData;
      end;
    end
    else if (Source is TGLGraphic) then
      Image.Assign(Source)
    else if (Source is TGLPicture) then
      Image.Assign(TGLPicture(Source).Graphic)
    else
      inherited Assign(Source);
  end
  else
  begin
    FDisabled := True;
    SetImage(nil);
    FTextureHandle.NotifyChangesOfData;
    FSamplerHandle.NotifyChangesOfData;
  end;
end;

procedure TGLTexture.NotifyChange(Sender: TObject);
begin
  if Assigned(Owner) then
  begin
    if Owner is TGLTextureExItem then
      TGLTextureExItem(Owner).NotifyChange(Self);
  end;
  if Sender is TGLTextureImage then
    FTextureHandle.NotifyChangesOfData;

  inherited;
end;

procedure TGLTexture.NotifyImageChange;
begin
  FTextureHandle.NotifyChangesOfData;
  NotifyChange(Self);
end;

procedure TGLTexture.NotifyParamsChange;
begin
  FSamplerHandle.NotifyChangesOfData;
  NotifyChange(Self);
end;

procedure TGLTexture.SetImage(AValue: TGLTextureImage);
begin
  if Assigned(aValue) then
  begin
    if FImage.ClassType <> AValue.ClassType then
    begin
      FImage.Free;
      FImage := TGLTextureImageClass(AValue.ClassType).Create(Self);
      FImage.OnTextureNeeded := DoOnTextureNeeded;
    end;
    FImage.Assign(AValue);
  end
  else
  begin
    FImage.Free;
    FImage := TGLPersistentImage.Create(Self);
    FImage.OnTextureNeeded := DoOnTextureNeeded;
  end;
end;

procedure TGLTexture.SetImageClassName(const val: string);
var
  newImage: TGLTextureImage;
  newImageClass: TGLTextureImageClass;
begin
  if val <> '' then
    if FImage.ClassName <> val then
    begin
      newImageClass := FindGLTextureImageClass(val);
      Assert(newImageClass <> nil, 'Make sure you include the unit for ' + val +
        ' in your uses clause');
      if newImageClass = nil then
        exit;
      newImage := newImageClass.Create(Self);
      newImage.OnTextureNeeded := DoOnTextureNeeded;
      FImage.Free;
      FImage := newImage;
    end;
end;

function TGLTexture.GetImageClassName: string;
begin
  Result := FImage.ClassName;
end;

function TGLTexture.TextureImageRequiredMemory: Integer;
var
  w, h, e, levelSize: Integer;
begin
  if FRequiredMemorySize < 0 then
  begin
    if IsCompressedFormat(fTextureFormat) then
    begin
      w := (Image.Width + 3) div 4;
      h := (Image.Height + 3) div 4;
    end
    else
    begin
      w := Image.Width;
      h := Image.Height;
    end;

    e := GetTextureElementSize(fTextureFormat);
    FRequiredMemorySize := w * h * e;
    if Image.Depth > 0 then
      FRequiredMemorySize := FRequiredMemorySize * Image.Depth;

    if not (MinFilter in [miNearest, miLinear]) then
    begin
      levelSize := FRequiredMemorySize;
      while e < levelSize do
      begin
        levelSize := levelSize div 4;
        FRequiredMemorySize := FRequiredMemorySize + levelSize;
      end;
    end;

    if Image.NativeTextureTarget = ttTextureCube then
      FRequiredMemorySize := FRequiredMemorySize * 6;
  end;
  Result := FRequiredMemorySize;
end;

procedure TGLTexture.SetImageAlpha(const val: TGLTextureImageAlpha);
begin
  if FImageAlpha <> val then
  begin
    FImageAlpha := val;
    NotifyImageChange;
  end;
end;

procedure TGLTexture.SetImageBrightness(const val: Single);
begin
  if FImageBrightness <> val then
  begin
    FImageBrightness := val;
    NotifyImageChange;
  end;
end;

function TGLTexture.StoreBrightness: Boolean;
begin
  Result := (FImageBrightness <> 1.0);
end;

procedure TGLTexture.SetImageGamma(const val: Single);
begin
  if FImageGamma <> val then
  begin
    FImageGamma := val;
    NotifyImageChange;
  end;
end;

function TGLTexture.StoreGamma: Boolean;
begin
  Result := (FImageGamma <> 1.0);
end;

procedure TGLTexture.SetMagFilter(AValue: TGLMagFilter);
begin
  if AValue <> FMagFilter then
  begin
    FMagFilter := AValue;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetMinFilter(AValue: TGLMinFilter);
begin
  if AValue <> FMinFilter then
  begin
    FMinFilter := AValue;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetTextureMode(AValue: TGLTextureMode);
begin
  if AValue <> FTextureMode then
  begin
    FTextureMode := AValue;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetDisabled(AValue: Boolean);
var
  intf: IGLTextureNotifyAble;
begin
  if AValue <> FDisabled then
  begin
    FDisabled := AValue;
    if Supports(Owner, IGLTextureNotifyAble, intf) then
      intf.NotifyTexMapChange(Self)
    else
      NotifyChange(Self);
  end;
end;

procedure TGLTexture.SetEnabled(const val: Boolean);
begin
  Disabled := not val;
end;

function TGLTexture.GetEnabled: Boolean;
begin
  Result := not Disabled;
end;

procedure TGLTexture.SetEnvColor(const val: TGLColor);
begin
  FEnvColor.Assign(val);
  NotifyParamsChange;
end;

procedure TGLTexture.SetBorderColor(const val: TGLColor);
begin
  FBorderColor.Assign(val);
  NotifyParamsChange;
end;

procedure TGLTexture.SetNormalMapScale(const val: Single);
begin
  if val <> FNormalMapScale then
  begin
    FNormalMapScale := val;
    if TextureFormat = tfNormalMap then
      NotifyImageChange;
  end;
end;

function TGLTexture.StoreNormalMapScale: Boolean;
begin
  Result := (FNormalMapScale <> cDefaultNormalMapScale);
end;

procedure TGLTexture.SetTextureWrap(AValue: TGLTextureWrap);
begin
  if AValue <> FTextureWrap then
  begin
    FTextureWrap := AValue;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetTextureWrapS(AValue: TGLSeparateTextureWrap);
begin
  if AValue <> FTextureWrapS then
  begin
    FTextureWrapS := AValue;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetTextureWrapT(AValue: TGLSeparateTextureWrap);
begin
  if AValue <> FTextureWrapT then
  begin
    FTextureWrapT := AValue;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetTextureWrapR(AValue: TGLSeparateTextureWrap);
begin
  if AValue <> FTextureWrapR then
  begin
    FTextureWrapR := AValue;
    NotifyParamsChange;
  end;
end;

function TGLTexture.GetTextureFormat: TGLTextureFormat;
var
  i: TGLTextureFormat;
begin
  if vDefaultTextureFormat = FTextureFormat then
  begin
    Result := tfDefault;
    Exit;
  end;
  for i := tfRGB to tfRGBAFloat32 do
  begin
    if cOldTextureFormatToInternalFormat[i] = FTextureFormat then
    begin
      Result := i;
      Exit;
    end;
  end;
  Result := tfExtended;
end;

procedure TGLTexture.SetTextureFormat(const val: TGLTextureFormat);
begin
  if val = tfDefault then
  begin
    FTextureFormat := vDefaultTextureFormat;
  end
  else if val < tfExtended then
  begin
    FTextureFormat := cOldTextureFormatToInternalFormat[val];
  end;
end;

procedure TGLTexture.SetTextureFormatEx(const val: TGLInternalFormat);
begin
  if val <> FTextureFormat then
  begin
    FTextureFormat := val;
    NotifyImageChange;
  end;
end;

function TGLTexture.StoreTextureFormatEx: Boolean;
begin
  Result := GetTextureFormat >= tfExtended;
end;

procedure TGLTexture.SetCompression(const val: TGLTextureCompression);
begin
  if val <> FCompression then
  begin
    FCompression := val;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetFilteringQuality(const val: TGLTextureFilteringQuality);
begin
  if val <> FFilteringQuality then
  begin
    FFilteringQuality := val;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetMappingMode(const val: TGLTextureMappingMode);
var
  texMapChange: Boolean;
  intf: IGLTextureNotifyAble;
begin
  if val <> FMappingMode then
  begin
    texMapChange := ((val = tmmUser) and (FMappingMode <> tmmUser))
      or ((val = tmmUser) and (FMappingMode <> tmmUser));
    FMappingMode := val;
    if texMapChange then
    begin
      // when switching between texGen modes and user mode, the geometry
      // must be rebuilt in whole (to specify/remove texCoord data!)
      if Supports(Owner, IGLTextureNotifyAble, intf) then
        intf.NotifyTexMapChange(Self);
    end
    else
      NotifyChange(Self);
  end;
end;

procedure TGLTexture.SetMappingSCoordinates(const val: TGLCoordinates4);
begin
  MappingSCoordinates.Assign(val);
end;

function TGLTexture.GetMappingSCoordinates: TGLCoordinates4;
begin
  if not Assigned(FMapSCoordinates) then
    FMapSCoordinates := TGLCoordinates4.CreateInitialized(Self, XHmgVector,
      csVector);
  Result := FMapSCoordinates;
end;


function TGLTexture.StoreMappingSCoordinates: Boolean;
begin
  if Assigned(FMapSCoordinates) then
    Result := not VectorEquals(FMapSCoordinates.AsVector, XHmgVector)
  else
    Result := false;
end;

procedure TGLTexture.SetMappingTCoordinates(const val: TGLCoordinates4);
begin
  MappingTCoordinates.Assign(val);
end;

function TGLTexture.GetMappingTCoordinates: TGLCoordinates4;
begin
  if not Assigned(FMapTCoordinates) then
    FMapTCoordinates := TGLCoordinates4.CreateInitialized(Self, YHmgVector,
      csVector);
  Result := FMapTCoordinates;
end;

function TGLTexture.StoreMappingTCoordinates: Boolean;
begin
  if Assigned(FMapTCoordinates) then
    Result := not VectorEquals(FMapTCoordinates.AsVector, YHmgVector)
  else
    Result := false;
end;


procedure TGLTexture.SetMappingRCoordinates(const val: TGLCoordinates4);
begin
  MappingRCoordinates.Assign(val);
end;

function TGLTexture.GetMappingRCoordinates: TGLCoordinates4;
begin
  if not Assigned(FMapRCoordinates) then
    FMapRCoordinates := TGLCoordinates4.CreateInitialized(Self, ZHmgVector,
      csVector);
  Result := FMapRCoordinates;
end;

function TGLTexture.StoreMappingRCoordinates: Boolean;
begin
  if Assigned(FMapRCoordinates) then
    Result := not VectorEquals(FMapRCoordinates.AsVector, ZHmgVector)
  else
    Result := false;
end;

procedure TGLTexture.SetMappingQCoordinates(const val: TGLCoordinates4);
begin
  MappingQCoordinates.Assign(val);
end;

function TGLTexture.GetMappingQCoordinates: TGLCoordinates4;
begin
  if not Assigned(FMapQCoordinates) then
    FMapQCoordinates := TGLCoordinates4.CreateInitialized(Self, WHmgVector,
      csVector);
  Result := FMapQCoordinates;
end;

function TGLTexture.StoreMappingQCoordinates: Boolean;
begin
  if Assigned(FMapQCoordinates) then
    Result := not VectorEquals(FMapQCoordinates.AsVector, WHmgVector)
  else
    Result := false;
end;

function TGLTexture.StoreImageClassName: Boolean;
begin
  Result := (FImage.ClassName <> TGLPersistentImage.ClassName);
end;

procedure TGLTexture.SetTextureCompareMode(const val: TGLTextureCompareMode);
begin
  if val <> fTextureCompareMode then
  begin
    fTextureCompareMode := val;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetTextureCompareFunc(const val: TGLDepthCompareFunc);
begin
  if val <> fTextureCompareFunc then
  begin
    fTextureCompareFunc := val;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.SetDepthTextureMode(const val: TGLDepthTextureMode);
begin
  if val <> fDepthTextureMode then
  begin
    fDepthTextureMode := val;
    NotifyParamsChange;
  end;
end;

procedure TGLTexture.PrepareBuildList;
begin
  GetHandle;
end;


procedure TGLTexture.ApplyMappingMode;
var
  R_Dim: Boolean;
begin
  R_Dim := GL.ARB_texture_cube_map or GL.EXT_texture3D;
  case MappingMode of
    tmmUser: ; // nothing to do, but checked first (common case)
    tmmObjectLinear:
      begin
        GL.TexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        GL.TexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
        GL.TexGenfv(GL_S, GL_OBJECT_PLANE, @MappingSCoordinates.DirectVector);
        GL.TexGenfv(GL_T, GL_OBJECT_PLANE, @MappingTCoordinates.DirectVector);
        GL.Enable(GL_TEXTURE_GEN_S);
        GL.Enable(GL_TEXTURE_GEN_T);

        if R_Dim then
        begin
          GL.TexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
          GL.TexGeni(GL_Q, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
          GL.TexGenfv(GL_R, GL_OBJECT_PLANE, @MappingRCoordinates.DirectVector);
          GL.TexGenfv(GL_Q, GL_OBJECT_PLANE, @MappingQCoordinates.DirectVector);
          GL.Enable(GL_TEXTURE_GEN_R);
          GL.Enable(GL_TEXTURE_GEN_Q);
        end;
      end;
    tmmEyeLinear:
      begin
        GL.TexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
        GL.TexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
        // specify planes in eye space, not world space
        GL.MatrixMode(GL_MODELVIEW);
        GL.PushMatrix;
        GL.LoadIdentity;
        GL.TexGenfv(GL_S, GL_EYE_PLANE, @MappingSCoordinates.DirectVector);
        GL.TexGenfv(GL_T, GL_EYE_PLANE, @MappingTCoordinates.DirectVector);
        GL.Enable(GL_TEXTURE_GEN_S);
        GL.Enable(GL_TEXTURE_GEN_T);
        if R_Dim then
        begin
          GL.TexGenfv(GL_R, GL_EYE_PLANE, @MappingRCoordinates.DirectVector);
          GL.TexGenfv(GL_Q, GL_EYE_PLANE, @MappingQCoordinates.DirectVector);
          GL.Enable(GL_TEXTURE_GEN_R);
          GL.Enable(GL_TEXTURE_GEN_Q);
        end;
        GL.PopMatrix;
      end;
    tmmSphere:
      begin
        GL.TexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
        GL.TexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_SPHERE_MAP);
        GL.Enable(GL_TEXTURE_GEN_S);
        GL.Enable(GL_TEXTURE_GEN_T);
      end;
    tmmCubeMapReflection, tmmCubeMapCamera: if GL.ARB_texture_cube_map then
      begin
        GL.TexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
        GL.TexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
        GL.TexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_REFLECTION_MAP);
        GL.Enable(GL_TEXTURE_GEN_S);
        GL.Enable(GL_TEXTURE_GEN_T);
        GL.Enable(GL_TEXTURE_GEN_R);
      end;
    tmmCubeMapNormal, tmmCubeMapLight0: if GL.ARB_texture_cube_map then
      begin
        GL.TexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP);
        GL.TexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP);
        GL.TexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP);
        GL.Enable(GL_TEXTURE_GEN_S);
        GL.Enable(GL_TEXTURE_GEN_T);
        GL.Enable(GL_TEXTURE_GEN_R);
      end;
  else
    Assert(False);
  end;
end;

procedure TGLTexture.UnApplyMappingMode;
begin
  if MappingMode <> tmmUser then
  begin
    GL.Disable(GL_TEXTURE_GEN_S);
    GL.Disable(GL_TEXTURE_GEN_T);
    if GL.EXT_texture3D or GL.ARB_texture_cube_map then
    begin
      GL.Disable(GL_TEXTURE_GEN_R);
      GL.Disable(GL_TEXTURE_GEN_Q);
    end;
  end;
end;

procedure TGLTexture.Apply(var rci: TGLRenderContextInfo);

  procedure SetCubeMapTextureMatrix;
  var
    m, mm: TMatrix;
  begin
    // compute model view matrix for proper viewing
    case MappingMode of
      tmmCubeMapReflection, tmmCubeMapNormal:
        begin
          m := rci.PipelineTransformation.ViewMatrix;
          NormalizeMatrix(m);
          TransposeMatrix(m);
          rci.GLStates.SetGLTextureMatrix(m);
        end;
      tmmCubeMapLight0:
        begin
          with TGLScene(rci.scene).Lights do
            if Count > 0 then
            begin
              m := TGLLightSource(Items[0]).AbsoluteMatrix;
              NormalizeMatrix(m);
              mm := rci.PipelineTransformation.ViewMatrix;
              NormalizeMatrix(mm);
              TransposeMatrix(mm);
              m := MatrixMultiply(m, mm);
              rci.GLStates.SetGLTextureMatrix(m);
            end;
        end;
      tmmCubeMapCamera:
        begin
          m.V[0] := VectorCrossProduct(rci.cameraUp, rci.cameraDirection);
          m.V[1] := VectorNegate(rci.cameraDirection);
          m.V[2] := rci.cameraUp;
          m.V[3] := WHmgPoint;
          mm := rci.PipelineTransformation.ViewMatrix;
          NormalizeMatrix(mm);
          TransposeMatrix(mm);
          m := MatrixMultiply(m, mm);
          rci.GLStates.SetGLTextureMatrix(m);
        end;
    end;
  end;
var
  H : TGLuint;
begin
  // Multisample image do not work with FFP
  if (FTextureHandle.Target = ttTexture2DMultisample) or
    (FTextureHandle.Target = ttTexture2DMultisampleArray) then
    exit;

  H := Handle;
  if not Disabled and (H > 0) then
  begin
    with rci.GLStates do
    begin
      ActiveTexture := 0;
      TextureBinding[0, FTextureHandle.Target] := H;
      ActiveTextureEnabled[FTextureHandle.Target] := True;
    end;

    if not rci.GLStates.ForwardContext then
    begin
      if FTextureHandle.Target = ttTextureCube then
        SetCubeMapTextureMatrix;
      GL.TexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE,
        cTextureMode[FTextureMode]);
      GL.TexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, FEnvColor.AsAddress);
      ApplyMappingMode;
      xgl.MapTexCoordToMain;
    end;
  end
  else if not rci.GLStates.ForwardContext then
  begin // default
    xgl.MapTexCoordToMain;
  end;
end;

procedure TGLTexture.UnApply(var rci: TGLRenderContextInfo);
begin
  if not Disabled
    and not rci.GLStates.ForwardContext then
  begin
    // Multisample image do not work with FFP
    if FTextureHandle.Target in [ttNoShape, ttTexture2DMultisample, ttTexture2DMultisampleArray] then
      exit;
    with rci.GLStates do
    begin
      ActiveTexture := 0;
      ActiveTextureEnabled[FTextureHandle.Target] := False;
      if FTextureHandle.Target = ttTextureCube then
        ResetGLTextureMatrix;
    end;
    UnApplyMappingMode;
  end;
end;

procedure TGLTexture.ApplyAsTexture2(var rci: TGLRenderContextInfo; textureMatrix:
  PMatrix = nil);
begin
  ApplyAsTextureN(2, rci, textureMatrix);
end;

procedure TGLTexture.UnApplyAsTexture2(var rci: TGLRenderContextInfo;
  reloadIdentityTextureMatrix: boolean);
begin
  UnApplyAsTextureN(2, rci, reloadIdentityTextureMatrix);
end;

procedure TGLTexture.ApplyAsTextureN(n: Integer; var rci: TGLRenderContextInfo;
  textureMatrix: PMatrix = nil);
var
  m: TMatrix;
begin
  if not Disabled then
  begin
    // Multisample image do not work with FFP
    if (FTextureHandle.Target = ttTexture2DMultisample) or
      (FTextureHandle.Target = ttTexture2DMultisampleArray) then
      exit;
    with rci.GLStates do
    begin
      ActiveTexture := n - 1;
      TextureBinding[n - 1, FTextureHandle.Target] := Handle;
      ActiveTextureEnabled[FTextureHandle.Target] := True;
      if Assigned(textureMatrix) then
        SetGLTextureMatrix(textureMatrix^)
      else if FTextureHandle.Target = ttTextureCube then
      begin
        m := rci.PipelineTransformation.ModelViewMatrix;
        NormalizeMatrix(m);
        TransposeMatrix(m);
        rci.GLStates.SetGLTextureMatrix(m);
      end;

      if not ForwardContext then
      begin
        GL.TexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, cTextureMode[FTextureMode]);
        GL.TexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, FEnvColor.AsAddress);
        ApplyMappingMode;
        ActiveTexture := 0;
      end;
    end;
  end;
end;

procedure TGLTexture.UnApplyAsTextureN(n: Integer; var rci: TGLRenderContextInfo;
  reloadIdentityTextureMatrix: boolean);
begin
  if not rci.GLStates.ForwardContext then
  begin
    // Multisample image do not work with FFP
    if (FTextureHandle.Target = ttTexture2DMultisample) or
      (FTextureHandle.Target = ttTexture2DMultisampleArray) then
      exit;
    with rci.GLStates do
    begin
      ActiveTexture := n - 1;
      ActiveTextureEnabled[FTextureHandle.Target] := False;
      UnApplyMappingMode;
      if (FTextureHandle.Target = ttTextureCube) or reloadIdentityTextureMatrix then
        ResetGLTextureMatrix;
      ActiveTexture := 0;
    end;
  end;
end;

function TGLTexture.AllocateHandle: TGLuint;
var
  vTarget: TGLTextureTarget;
begin
  vTarget := Image.NativeTextureTarget;
  if (vTarget <> ttNoShape) and (FTextureHandle.Target <> vTarget) then
    FTextureHandle.DestroyHandle;

  Result := FTextureHandle.Handle;
  if Result = 0 then
  begin
    FTextureHandle.AllocateHandle;
    Result := FTextureHandle.Handle;
  end;
  if FTextureHandle.IsDataNeedUpdate then
  begin
    FTextureHandle.Target := vTarget;
    FSamplerHandle.NotifyChangesOfData;
  end;
  if FSamplerHandle.Handle = 0 then
    FSamplerHandle.AllocateHandle;

  // bind texture
  if (FTextureHandle.Target <> ttNoShape) and
    IsTargetSupported(FTextureHandle.Target) then
  begin
    if FSamplerHandle.IsDataNeedUpdate then
    begin
      with CurrentGLContext.GLStates do
        TextureBinding[ActiveTexture, FTextureHandle.Target] := Result;
      PrepareParams(DecodeGLTextureTarget(FTextureHandle.Target));
      FSamplerHandle.NotifyDataUpdated;
    end;
  end
  else
    Result := 0;
end;

function TGLTexture.IsHandleAllocated: Boolean;
begin
  Result := (FTextureHandle.Handle <> 0);
end;


function TGLTexture.GetHandle: TGLuint;
var
  target: TGLUInt;
  LBinding: array[TGLTextureTarget] of TGLuint;

  procedure StoreBindings;
  var
    t: TGLTextureTarget;
  begin
    with CurrentGLContext.GLStates do
    begin
      if TextureBinding[ActiveTexture, FTextureHandle.Target] = FTextureHandle.Handle then
        TextureBinding[ActiveTexture, FTextureHandle.Target] := 0;
      for t := Low(TGLTextureTarget) to High(TGLTextureTarget) do
        LBinding[t] := TextureBinding[ActiveTexture, t];
    end;
  end;

  procedure RestoreBindings;
  var
    t: TGLTextureTarget;
  begin
    with CurrentGLContext.GLStates do
      for t := Low(TGLTextureTarget) to High(TGLTextureTarget) do
        TextureBinding[ActiveTexture, t] := LBinding[t];
  end;

begin
  with CurrentGLContext.GLStates do
  begin
    StoreBindings;
    try
      Result := AllocateHandle;
      if FTextureHandle.IsDataNeedUpdate then
      begin
        FTextureHandle.NotifyDataUpdated;
        // Check supporting
        target := DecodeGLTextureTarget(Image.NativeTextureTarget);
        if not IsTargetSupported(target) or not IsFormatSupported(TextureFormatEx) then
        begin
          SetTextureErrorImage;
          target := GL_TEXTURE_2D;
        end;
        // Load images
        if not GL.EXT_direct_state_access then
          TextureBinding[ActiveTexture, FTextureHandle.Target] := Result;
        PrepareImage(target);
      end;
    finally
      RestoreBindings;
    end;
  end;
end;

procedure TGLTexture.DestroyHandles;
begin
  FTextureHandle.DestroyHandle;
  FSamplerHandle.DestroyHandle;
  FRequiredMemorySize := -1;
end;

function TGLTexture.IsFloatType: Boolean;
begin
  Result := IsFloatFormat(TextureFormatEx);
end;

function TGLTexture.OpenGLTextureFormat: Integer;
var
  texComp: TGLTextureCompression;
begin
  if GL.ARB_texture_compression then
  begin
    if Compression = tcDefault then
      if vDefaultTextureCompression = tcDefault then
        texComp := tcNone
      else
        texComp := vDefaultTextureCompression
    else
      texComp := Compression;
  end
  else
    texComp := tcNone;

  if IsFloatType then
    texComp := tcNone; // no compression support for float_type

  if (texComp <> tcNone) and (TextureFormat <= tfNormalMap) then
    with CurrentGLContext.GLStates do
    begin
      case texComp of
        tcStandard: TextureCompressionHint := hintDontCare;
        tcHighQuality: TextureCompressionHint := hintNicest;
        tcHighSpeed: TextureCompressionHint := hintFastest;
      else
        Assert(False);
      end;
      Result := CompressedInternalFormatToOpenGL(TextureFormatEx);
    end
  else
    Result := InternalFormatToOpenGLFormat(TextureFormatEx);
end;

procedure TGLTexture.PrepareImage(target: TGLUInt);
var
  bitmap32: TGLImage;
  texComp: TGLTextureCompression;
  glFormat: TGLEnum;
begin
  if Image.IsSelfLoading then
  begin
    Image.LoadTexture(FTextureFormat);
  end
  else
  begin

    bitmap32 := Image.GetBitmap32;

    if (bitmap32 = nil) or bitmap32.IsEmpty then
      Exit;

    if TextureFormat = tfNormalMap then
      bitmap32.GrayScaleToNormalMap(NormalMapScale,
        TextureWrap in [twBoth, twHorizontal],
        TextureWrap in [twBoth, twVertical]);
    // prepare AlphaChannel
    case ImageAlpha of
      tiaDefault: ; // nothing to do
      tiaAlphaFromIntensity:
        bitmap32.SetAlphaFromIntensity;
      tiaSuperBlackTransparent:
        bitmap32.SetAlphaTransparentForColor($000000);
      tiaLuminance:
        bitmap32.SetAlphaFromIntensity;
      tiaLuminanceSqrt:
        begin
          bitmap32.SetAlphaFromIntensity;
          bitmap32.SqrtAlpha;
        end;
      tiaOpaque:
        bitmap32.SetAlphaToValue(255);
      tiaTopLeftPointColorTransparent:
        begin
          bitmap32.Narrow;
          bitmap32.SetAlphaTransparentForColor(bitmap32.Data^[0]);
        end;
      tiaInverseLuminance:
        begin
          bitmap32.SetAlphaFromIntensity;
          bitmap32.InvertAlpha;
        end;
      tiaInverseLuminanceSqrt:
        begin
          bitmap32.SetAlphaFromIntensity;
          bitmap32.SqrtAlpha;
          bitmap32.InvertAlpha;
        end;
      tiaBottomRightPointColorTransparent:
        begin
          bitmap32.Narrow;
          bitmap32.SetAlphaTransparentForColor(bitmap32.Data^[bitmap32.Width - 1]);
        end;
    else
      Assert(False);
    end;
    // apply brightness correction
    if FImageBrightness <> 1.0 then
      bitmap32.BrightnessCorrection(FImageBrightness);
    // apply gamma correction
    if FImageGamma <> 1.0 then
      bitmap32.GammaCorrection(FImageGamma);

    if GL.ARB_texture_compression
      and (TextureFormat <> tfExtended) then
    begin
      if Compression = tcDefault then
        if vDefaultTextureCompression = tcDefault then
          texComp := tcNone
        else
          texComp := vDefaultTextureCompression
      else
        texComp := Compression;
      if IsFloatType then
        texComp := tcNone;

    end
    else
      texComp := tcNone;

    if (texComp <> tcNone) and (TextureFormat <= tfNormalMap) then
      with CurrentGLContext.GLStates do
      begin
        case texComp of
          tcStandard: TextureCompressionHint := hintDontCare;
          tcHighQuality: TextureCompressionHint := hintNicest;
          tcHighSpeed: TextureCompressionHint := hintFastest;
        else
          Assert(False, glsErrorEx + glsUnknownType);
        end;
        glFormat := CompressedInternalFormatToOpenGL(FTextureFormat);
      end
    else
      glFormat := InternalFormatToOpenGLFormat(FTextureFormat);

    bitmap32.RegisterAsOpenGLTexture(
      FTextureHandle,
      not (FMinFilter in [miNearest, miLinear]),
      glFormat,
      FTexWidth,
      FTexHeight,
      FTexDepth);
  end;

  if GL.GetError <> GL_NO_ERROR then
  begin
    GL.ClearError;
    SetTextureErrorImage;
  end
  else
  begin
    FRequiredMemorySize := -1;
    TextureImageRequiredMemory;
    if not IsDesignTime and not FKeepImageAfterTransfer then
      Image.ReleaseBitmap32;
  end;
end;

procedure TGLTexture.PrepareParams(target: TGLUInt);
const
  cTextureSWrap: array[twBoth..twHorizontal] of TGLEnum =
    (GL_REPEAT, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_REPEAT);
  cTextureTWrap: array[twBoth..twHorizontal] of TGLEnum =
    (GL_REPEAT, GL_CLAMP_TO_EDGE, GL_REPEAT, GL_CLAMP_TO_EDGE);
  cTextureRWrap: array[twBoth..twHorizontal] of TGLEnum =
    (GL_REPEAT, GL_CLAMP_TO_EDGE, GL_REPEAT, GL_CLAMP_TO_EDGE);
  cTextureSWrapOld: array[twBoth..twHorizontal] of TGLEnum =
    (GL_REPEAT, GL_CLAMP, GL_CLAMP, GL_REPEAT);
  cTextureTWrapOld: array[twBoth..twHorizontal] of TGLEnum =
    (GL_REPEAT, GL_CLAMP, GL_REPEAT, GL_CLAMP);
  cTextureMagFilter: array[maNearest..maLinear] of TGLEnum =
    (GL_NEAREST, GL_LINEAR);
  cTextureMinFilter: array[miNearest..miLinearMipmapLinear] of TGLEnum =
    (GL_NEAREST, GL_LINEAR, GL_NEAREST_MIPMAP_NEAREST,
    GL_LINEAR_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR);
  cFilteringQuality: array[tfIsotropic..tfAnisotropic] of Integer = (1, 2);
  cSeparateTextureWrap: array[twRepeat..twMirrorClampToBorder] of TGLenum =
    (GL_REPEAT, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_BORDER,
    GL_MIRRORED_REPEAT, GL_MIRROR_CLAMP_TO_EDGE_ATI, GL_MIRROR_CLAMP_TO_BORDER_EXT);
  cTextureCompareMode: array[tcmNone..tcmCompareRtoTexture] of TGLenum =
    (GL_NONE, GL_COMPARE_R_TO_TEXTURE);
  cDepthTextureMode: array[dtmLuminance..dtmAlpha] of TGLenum =
    (GL_LUMINANCE, GL_INTENSITY, GL_ALPHA);

var
  R_Dim: Boolean;
  lMinFilter: TGLMinFilter;
begin
  if (target = GL_TEXTURE_2D_MULTISAMPLE)
    or (target = GL_TEXTURE_2D_MULTISAMPLE_ARRAY) then
    Exit;

  R_Dim := GL.ARB_texture_cube_map or GL.EXT_texture3D;

  with CurrentGLContext.GLStates do
  begin
    UnpackAlignment := 1;
    UnpackRowLength := 0;
    UnpackSkipRows := 0;
    UnpackSkipPixels := 0;
  end;

  GL.TexParameterfv(target, GL_TEXTURE_BORDER_COLOR, FBorderColor.AsAddress);

  if (GL.VERSION_1_2 or GL.EXT_texture_edge_clamp) then
  begin
    if FTextureWrap = twSeparate then
    begin
      GL.TexParameteri(target, GL_TEXTURE_WRAP_S,
        cSeparateTextureWrap[FTextureWrapS]);
      GL.TexParameteri(target, GL_TEXTURE_WRAP_T,
        cSeparateTextureWrap[FTextureWrapT]);
      if R_Dim then
        GL.TexParameteri(target, GL_TEXTURE_WRAP_R,
          cSeparateTextureWrap[FTextureWrapR]);
    end
    else
    begin
      GL.TexParameteri(target, GL_TEXTURE_WRAP_S, cTextureSWrap[FTextureWrap]);
      GL.TexParameteri(target, GL_TEXTURE_WRAP_T, cTextureTWrap[FTextureWrap]);
      if R_Dim then
        GL.TexParameteri(target, GL_TEXTURE_WRAP_R, cTextureRWrap[FTextureWrap]);
    end;
  end
  else
  begin
    GL.TexParameteri(target, GL_TEXTURE_WRAP_S, cTextureSWrapOld[FTextureWrap]);
    GL.TexParameteri(target, GL_TEXTURE_WRAP_T, cTextureTWrapOld[FTextureWrap]);
  end;

  lMinFilter := FMinFilter;
  // Down paramenter to rectangular texture supported
  if (target = GL_TEXTURE_RECTANGLE)
    or not (GL.EXT_texture_lod or GL.SGIS_texture_lod) then
  begin
    if lMinFilter in [miNearestMipmapNearest, miNearestMipmapLinear] then
      lMinFilter := miNearest;
    if FMinFilter in [miLinearMipmapNearest, miLinearMipmapLinear] then
      lMinFilter := miLinear;
  end;

  GL.TexParameteri(target, GL_TEXTURE_MIN_FILTER, cTextureMinFilter[lMinFilter]);
  GL.TexParameteri(target, GL_TEXTURE_MAG_FILTER, cTextureMagFilter[FMagFilter]);

  if GL.EXT_texture_filter_anisotropic then
    GL.TexParameteri(target, GL_TEXTURE_MAX_ANISOTROPY_EXT,
      cFilteringQuality[FFilteringQuality]);

  if IsDepthFormat(fTextureFormat) then
  begin
    GL.TexParameteri(target, GL_TEXTURE_COMPARE_MODE,
      cTextureCompareMode[fTextureCompareMode]);
    GL.TexParameteri(target, GL_TEXTURE_COMPARE_FUNC,
      cGLComparisonFunctionToGLEnum[fTextureCompareFunc]);
    if not FTextureHandle.RenderingContext.GLStates.ForwardContext then
      GL.TexParameteri(target, GL_DEPTH_TEXTURE_MODE,
        cDepthTextureMode[fDepthTextureMode]);
  end;
end;

procedure TGLTexture.DoOnTextureNeeded(Sender: TObject; var textureFileName:
  string);
begin
  if Assigned(FOnTextureNeeded) then
    FOnTextureNeeded(Sender, textureFileName);
end;

procedure TGLTexture.OnSamplerAllocate(Sender: TGLVirtualHandle; var Handle: Cardinal);
begin
  Handle := 1;
end;

procedure TGLTexture.OnSamplerDestroy(Sender: TGLVirtualHandle; var Handle: Cardinal);
begin
  Handle := 0;
end;

procedure TGLTexture.SetTextureErrorImage;
var
  img: TGLImage;
begin
  img := TGLImage.Create;
  img.SetErrorImage;

  ImageClassName := TGLBlankImage.className;
  TGLBlankImage(Image).Assign(img);
  img.Free;

  MagFilter := maNearest;
  MinFilter := miNearest;
  TextureWrap := twBoth;
  MappingMode := tmmUser;
  Compression := tcNone;
  AllocateHandle;
end;


{%endregion%}

{%region%=====  'TGLTextureExItem ================================}

constructor TGLTextureExItem.Create(ACollection: TCollection);
begin
  inherited;

  FTexture := TGLTexture.Create(Self);
  FTextureOffset := TGLCoordinates.CreateInitialized(Self, NullHMGVector,
    csPoint);
  FTextureOffset.OnNotifyChange := OnNotifyChange;
  FTextureScale := TGLCoordinates.CreateInitialized(Self, XYZHmgVector,
    csPoint);
  FTextureScale.OnNotifyChange := OnNotifyChange;

  FTextureIndex := ID;
  FTextureMatrix := IdentityHMGMatrix;

  //DanB - hmmm, not very flexible code, assumes it's owned by a material,
  // that has a Texture property, but may need to re-implement it somehow
{  if ACollection is TGLTextureEx then
    if TGLTextureEx(ACollection).FOwner <> nil then
      FTexture.OnTextureNeeded := TGLTextureEx(ACollection).FOwner.Texture.OnTextureNeeded;
      }
end;


destructor TGLTextureExItem.Destroy;
begin
  FTexture.Free;
  FTextureOffset.Free;
  FTextureScale.Free;

  inherited;
end;


  function TGLTextureExItem.QueryInterface(constref IID: TGUID; out Obj): HResult; {$IFNDEF WINDOWS}cdecl{$ELSE}stdcall{$ENDIF};


begin
  if GetInterface(IID, Obj) then
    Result := S_OK
  else
    Result := E_NOINTERFACE;
end;


  function TGLTextureExItem._AddRef: Integer; {$IFNDEF WINDOWS}cdecl{$ELSE}stdcall{$ENDIF};


begin
  Result := -1; //ignore
end;

  function TGLTextureExItem._Release: Integer; {$IFNDEF WINDOWS}cdecl{$ELSE}stdcall{$ENDIF};


begin
  Result := -1; //ignore
end;

procedure TGLTextureExItem.Assign(Source: TPersistent);
begin
  if Source is TGLTextureExItem then
  begin
    Texture := TGLTextureExItem(Source).Texture;
    TextureIndex := TGLTextureExItem(Source).TextureIndex;
    TextureOffset := TGLTextureExItem(Source).TextureOffset;
    TextureScale := TGLTextureExItem(Source).TextureScale;
    NotifyChange(Self);
  end
  else
    inherited;
end;

procedure TGLTextureExItem.NotifyChange(Sender: TObject);
begin
  if Assigned(Collection) then
    TGLTextureEx(Collection).NotifyChange(Self);
end;

procedure TGLTextureExItem.Apply(var rci: TGLRenderContextInfo);
begin
  FApplied := False;
  if FTexture.Enabled then
  begin
    rci.GLStates.ActiveTexture := FTextureIndex;
    GL.MatrixMode(GL_TEXTURE);
    GL.PushMatrix;
    if FTextureMatrixIsIdentity then
      GL.LoadIdentity
    else
      GL.LoadMatrixf(@FTextureMatrix.V[0].V[0]);
    GL.MatrixMode(GL_MODELVIEW);
    rci.GLStates.ActiveTexture := 0;
    if FTextureIndex = 0 then
      FTexture.Apply(rci)
    else if FTextureIndex = 1 then
      FTexture.ApplyAsTexture2(rci, nil)
    else if FTextureIndex >= 2 then
      FTexture.ApplyAsTextureN(FTextureIndex + 1, rci, nil);
    FApplied := True;
  end;
end;

procedure TGLTextureExItem.UnApply(var rci: TGLRenderContextInfo);
begin
  if FApplied then
  begin
    if FTextureIndex = 0 then
      FTexture.UnApply(rci)
    else if FTextureIndex = 1 then
      FTexture.UnApplyAsTexture2(rci, false)
    else if FTextureIndex >= 2 then
      FTexture.UnApplyAsTextureN(FTextureIndex + 1, rci, false);
    rci.GLStates.ActiveTexture := FTextureIndex;
    GL.MatrixMode(GL_TEXTURE);
    GL.PopMatrix;
    GL.MatrixMode(GL_MODELVIEW);
    rci.GLStates.ActiveTexture := 0;
    FApplied := False;
  end;
end;

function TGLTextureExItem.GetDisplayName: string;
begin
  Result := Format('Tex [%d]', [FTextureIndex]);
end;

function TGLTextureExItem.GetOwner: TPersistent;
begin
  Result := Collection;
end;

procedure TGLTextureExItem.NotifyTexMapChange(Sender: TObject);
var
  intf: IGLTextureNotifyAble;
begin
  if Supports(TObject(TGLTextureEx(Collection).FOwner), IGLTextureNotifyAble,
    intf) then
    intf.NotifyTexMapChange(Sender);
end;

procedure TGLTextureExItem.SetTexture(const Value: TGLTexture);
begin
  FTexture.Assign(Value);
  NotifyChange(Self);
end;

procedure TGLTextureExItem.SetTextureIndex(const Value: Integer);
var
  temp: Integer;
begin
  temp := Value;
  if temp < 0 then
    temp := 0;
  if temp <> FTextureIndex then
  begin
    FTextureIndex := temp;
    NotifyChange(Self);
  end;
end;

procedure TGLTextureExItem.SetTextureOffset(const Value: TGLCoordinates);
begin
  FTextureOffset.Assign(Value);
  NotifyChange(Self);
end;

procedure TGLTextureExItem.SetTextureScale(const Value: TGLCoordinates);
begin
  FTextureScale.Assign(Value);
  NotifyChange(Self);
end;

procedure TGLTextureExItem.CalculateTextureMatrix;
begin
  if TextureOffset.Equals(NullHmgVector) and TextureScale.Equals(XYZHmgVector) then
    FTextureMatrixIsIdentity := True
  else
  begin
    FTextureMatrixIsIdentity := False;
    FTextureMatrix := CreateScaleAndTranslationMatrix(TextureScale.AsVector,
      TextureOffset.AsVector);
  end;
  NotifyChange(Self);
end;

procedure TGLTextureExItem.OnNotifyChange(Sender: TObject);
begin
  CalculateTextureMatrix;
end;

{%endregion%}

{%region%===== TGLTextureEx ================================}


constructor TGLTextureEx.Create(AOwner: TGLUpdateAbleObject);
begin
  inherited Create(TGLTextureExItem);

  FOwner := AOwner;
end;


procedure TGLTextureEx.NotifyChange(Sender: TObject);
begin
  if Assigned(FOwner) then
    FOwner.NotifyChange(Self);
end;


procedure TGLTextureEx.Apply(var rci: TGLRenderContextInfo);
var
  i, texUnits: Integer;
  units: Cardinal;
begin
  if not GL.ARB_multitexture then
    exit;

  units := 0;
  GL.GetIntegerv(GL_MAX_TEXTURE_UNITS, @texUnits);
  for i := 0 to Count - 1 do
  begin
    if Items[i].TextureIndex < texUnits then
    begin
      Items[i].Apply(rci);
      if Items[i].FApplied then
        if (Items[i].TextureIndex > 0) and (Items[i].Texture.MappingMode =
          tmmUser) then
          units := units or (1 shl Items[i].TextureIndex);
    end;
  end;
  if units > 0 then
    xgl.MapTexCoordToArbitraryAdd(units);
end;


procedure TGLTextureEx.UnApply(var rci: TGLRenderContextInfo);
var
  i: Integer;
begin
  if not GL.ARB_multitexture then
    exit;
  for i := 0 to Count - 1 do
    Items[i].UnApply(rci);
end;

function TGLTextureEx.Add: TGLTextureExItem;
begin
  Result := TGLTextureExItem(inherited Add);
end;

procedure TGLTextureEx.Loaded;
var
  i: Integer;
begin
  for i := 0 to Count - 1 do
    Items[i].CalculateTextureMatrix;
end;

function TGLTextureEx.GetOwner: TPersistent;
begin
  Result := FOwner;
end;

procedure TGLTextureEx.SetItems(index: Integer; const Value: TGLTextureExItem);
begin
  inherited SetItem(index, Value);
end;

function TGLTextureEx.GetItems(index: Integer): TGLTextureExItem;
begin
  Result := TGLTextureExItem(inherited GetItem(index));
end;

function TGLTextureEx.IsTextureEnabled(Index: Integer): Boolean;
var
  i: Integer;
begin
  Result := False;
  if Self = nil then
    Exit;
  for i := 0 to Count - 1 do
    if Items[i].TextureIndex = Index then
      Result := Result or Items[i].Texture.Enabled;
end;

{%endregion%}

initialization

  RegisterGLTextureImageClass(TGLBlankImage);
  RegisterGLTextureImageClass(TGLPersistentImage);
  RegisterGLTextureImageClass(TGLPicFileImage);
  RegisterGLTextureImageClass(TGLCubeMapImage);

  RegisterTGraphicClassFileExtension('.bmp', TGLBitmap);

finalization

  vGLTextureImageClasses.Free;
  vGLTextureImageClasses := nil;

end.
