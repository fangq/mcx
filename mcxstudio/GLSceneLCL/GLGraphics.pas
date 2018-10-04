//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Utility class and functions to manipulate a bitmap in OpenGL's default
   byte order (GL_RGBA vs TBitmap's GL_BGRA)

   Note: TGLBitmap32 has support for Alex Denissov's Graphics32 library
   (http://www.g32.org), just make sure the GLS_Graphics32_SUPPORT conditionnal
   is active in GLScene.inc and recompile.

   Note: TGLBitmap32 has support for Alex Denissov's Graphics32 library
   (http://www.g32.org), just make sure the GLS_Graphics32_SUPPORT conditionnal
   is active in GLScene.inc and recompile.

 History :  
       25/11/11 - YP - Assertion removed from AssignFromBitmap32
       10/05/11 - Yar - Now VerticalReverseOnAssignFromBitmap works for AssignToBitmap
       04/11/10 - DaStr - Restored Delphi5 and Delphi6 compatibility
       18/07/10 - Yar - Raname TGLBitmap32 to TGLImage
       20/06/10 - Yar - Added in TRasterFileFormatsList.FindFromStream JPG singnature
       14/06/10 - YP  - PngImage support added (thanks to Sergio Alexandre Gianezini)
       20/05/10 - Yar - Fixes for Linux x64
                           Replace OpenGL1x functions to OpenGLAdapter
       22/04/10 - Yar - Fixes after GLState revision
       28/03/10 - Yar - Added glGenerateMipmap when used forward context
       08/03/10 - Yar - Added in TRasterFileFormatsList.FindFromStream PNG singnature
                           Added forward context cheking to circumvent deprecations
       01/03/10 - Yar - Bugfix when texture, which has lower mip-levels than the standard number is not rendered
                           (thanks to Controller)
       23/02/10 - Yar - Solved problem of TGLBitmap with width of which is not a multiple of four.
                           Added in AssignFrom24BitsBitmap, AssignFrom32BitsBitmap using extension GL_EXT_bgra
       22/02/10 - Yar - Added FindFromStream to TRasterFileFormatsList
                           (thanks to mif)
       10/02/10 - Yar   - Bugfix in RegisterAsOpenGLTexture with Cubemap mipmaping
       27/01/10 - Yar   - Bugfix in BlockOffset with negative result
                             Return to GL_SGIS_generate_mipmap
       23/01/10 - Yar   - Added to AssignFromTexture CurrentFormat parameter
       22/01/10 - Yar   - Added TRasterFileFormat, TGLBaseImage classes
                             TGLBitmap32 now derived from TGLBaseImage
                             and can contain all types of image
                             (with mipmap level, texture array, volume texture)
                             any kind of color and data formats of OpenGL
                             remake RegisterAsOpenGLTexture
       10/11/09 - DaStr - Updated TGLBitmap32.RegisterAsOpenGLTexture() - fixed
                               TextureFormat and ColorFormat (thanks YarUnderoaker)
                             Improved FPC compatibility
                              (BugtrackerID = 2893580) (thanks Predator)
       24/03/07 - DaStr - Moved TGLMinFilter and TGLMagFilter from GLUtils.pas
                              to GLGraphics.pas (BugTracker ID = 1923844)
       06/06/07 - DaStr - Added GLColor to uses (BugtrackerID = 1732211)
       14/03/07 - DaStr - Added explicit pointer dereferencing
                             (thanks Burkhard Carstens) (Bugtracker ID = 1678644)
       23/01/07 - LIN- Added TGLBitmap32.AssignToBitmap : Converts a TGLBitmap32 back into a TBitmap
       12/09/06 - NC - Added TGLBitmap32.Blank
       18/10/05 - NC - GL_ARB_texture_non_power_of_two, GL_TEXTURE_2D for float
                          texture
       06/10/04 - NC - Now uses GL_TEXTURE_RECTANGLE_NV for all float texture types
       04/10/04 - NC - Added support for float texture
       05/09/03 - EG - Added TGLBitmap32.DownSampleByFactor2
       04/07/03 - EG - Added RGBA brightness/gamma correction support
       13/05/03 - EG - Added GrayScaleToNormalMap
       26/08/02 - EG - Fixed loading of 1D horiz textures from 24 bits bitmaps
       16/06/02 - EG - AssignFrom32BitsBitmap fix for single-line bitmaps
       29/01/02 - EG - Fixed ScanLine Index bug with empty bitmaps
       20/01/02 - EG - Fixed BGR24/RGB24 last pixel transfer
       17/01/02 - EG - Faster assignments from bitmaps (approx. x2),
                          Added AssignFromBitmap24WithoutRGBSwap
       28/12/01 - EG - Graphics32 support added
       15/12/01 - EG - Texture target support
       14/09/01 - EG - Use of vFileStreamClass
       31/08/01 - EG - 24bits Bitmaps are now made opaque by default
       12/08/01 - EG - Now detects and uses GL_SGIS_generate_mipmap
       20/02/01 - EG - Fixed SetHeight & SetWidth (thx Nelson Chu)
       14/02/01 - EG - Simplified RegisterAsOpenGLTexture
       15/01/01 - EG - Fixed RegisterAsOpenGLTexture (clamping)
       14/01/01 - EG - Fixed isEmpty (was invalid for rectangles)
       08/10/00 - EG - Fixed RegisterAsOpenGLTexture and Assign(nil)
       25/09/00 - EG - First operational code
       19/08/00 - EG - Creation
  
}
unit GLGraphics;

interface

{$I GLScene.inc}

uses
{$IFDEF MSWINDOWS}
  Windows,
{$ENDIF}
  Classes, SysUtils, SyncObjs, Graphics, Math,
{$IFDEF GLS_Graphics32_SUPPORT}
  GR32,
{$ENDIF}
{$IFDEF GLS_PngImage_SUPPORT}
  Pngimage,
{$ENDIF}
  fpimage,
  Intfgraphics,
  GraphType,
  GLApplicationFileIO,
  GLPersistentClasses,
  OpenGLTokens,
  GLContext,
  GLImageUtils,
  GLUtils,
  GLCrossPlatform,
  GLColor,
  GLTextureFormat,
  GLVectorGeometry,
  GLStrings,
  GLSLog;


type

  // TGLPixel24
  //
  TGLPixel24 = packed record
    r, g, b: Byte;
  end;
  PGLPixel24 = ^TGLPixel24;

  // TGLPixel32
  //
  TGLPixel32 = packed record
    r, g, b, a: Byte;
  end;
  PGLPixel32 = ^TGLPixel32;

  TGLPixel32Array = array[0..MaxInt shr 3] of TGLPixel32;
  PGLPixel32Array = ^TGLPixel32Array;

  TGLLODStreamingState = (ssKeeping, ssLoading, ssLoaded, ssTransfered);

  // TGLImageLevelDesc
  //
  TGLImageLevelDesc = record
    Width: Integer;
    Height: Integer;
    Depth: Integer;
    PBO: TGLUnpackPBOHandle;
    MapAddress: Pointer;
    Offset: LongWord;
    StreamOffset: LongWord;
    Size: LongWord;
    State: TGLLODStreamingState;
  end;

  TGLImageLODRange = 0..15;

  TGLImagePiramid = array[TGLImageLODRange] of TGLImageLevelDesc;

  // TGLBaseImage
  //
  TGLBaseImage = class(TGLDataFile)
  private
    FSourceStream: TStream;
    FStreamLevel: TGLImageLODRange;
    FFinishEvent: TFinishTaskEvent;
{$IFDEF GLS_SERVICE_CONTEXT}
    procedure ImageStreamingTask; stdcall;
{$ENDIF}
  protected
    fData: PGLPixel32Array;
    FLOD: TGLImagePiramid;
    fLevelCount: TGLImageLODRange;
    fColorFormat: TGLEnum;
    fInternalFormat: TGLInternalFormat;
    fDataType: TGLEnum;
    fElementSize: Integer;
    fCubeMap: Boolean;
    fTextureArray: Boolean;

    function GetData: PGLPixel32Array; virtual;
    function GetWidth: Integer;
    function GetHeight: Integer;
    function GetDepth: Integer;
    function GetLevelAddress(ALevel: Byte): Pointer; overload;
    function GetLevelAddress(ALevel, AFace: Byte): Pointer; overload;

    function GetLevelWidth(ALOD: TGLImageLODRange): Integer;
    function GetLevelHeight(ALOD: TGLImageLODRange): Integer;
    function GetLevelDepth(ALOD: TGLImageLODRange): Integer;
    function GetLevelPBO(ALOD: TGLImageLODRange): TGLUnpackPBOHandle;
    function GetLevelOffset(ALOD: TGLImageLODRange): Integer;
    function GetLevelSizeInByte(ALOD: TGLImageLODRange): Integer;
    function GetLevelStreamingState(ALOD: TGLImageLODRange): TGLLODStreamingState;
    procedure SetLevelStreamingState(ALOD: TGLImageLODRange; AState: TGLLODStreamingState);

    procedure SaveHeader;
    procedure LoadHeader;
    procedure StartStreaming;
    procedure DoStreaming;
  public
    constructor Create; reintroduce; virtual;
    destructor Destroy; override;
    procedure Assign(Source: TPersistent); override;

    function GetTextureTarget: TGLTextureTarget;

    { Registers the bitmap's content as an OpenGL texture map. }
    procedure RegisterAsOpenGLTexture(
      AHandle: TGLTextureHandle;
      aMipmapGen: Boolean;
      aTexFormat: TGLEnum;
      out texWidth: integer;
      out texHeight: integer;
      out texDepth: integer); virtual;

    { Assigns from any Texture.}
    function AssignFromTexture(
      AHandle: TGLTextureHandle;
      const CastToFormat: Boolean;
      const intFormat: TGLInternalFormat = tfRGBA8;
      const colorFormat: TGLenum = 0;
      const dataType: TGLenum = 0): Boolean; virtual;

    { Convert vertical cross format of non compressed, non mipmaped image
       to six face of cube map }
    function ConvertCrossToCubeMap: Boolean;
    { Convert flat image to volume by dividing it into slice. }
    function ConvertToVolume(const col, row: Integer; const MakeArray: Boolean): Boolean;
    { Return size in byte of all image }
    function DataSize: PtrUint;
    { True if the bitmap is empty (ie. width or height is zero). }
    function IsEmpty: Boolean;
    function IsCompressed: Boolean;
    function IsVolume: Boolean;
    { Narrow image data to simple RGBA8 ubyte }
    procedure Narrow;
    { Generate LOD pyramid }
    procedure GenerateMipmap(AFilter: TImageFilterFunction); virtual;
    { Leave top level and remove other }
    procedure UnMipmap; virtual;
    { Direct Access to image data}
    property Data: PGLPixel32Array read GetData;
    { Set image of error. }
    procedure SetErrorImage;

    { Recalculate levels information based on first level. }
    procedure UpdateLevelsInfo;

    property LevelWidth[ALOD: TGLImageLODRange]: Integer
      read GetLevelWidth;
    property LevelHeight[ALOD: TGLImageLODRange]: Integer
      read GetLevelHeight;
    property LevelDepth[ALOD: TGLImageLODRange]: Integer
      read GetLevelDepth;
    property LevelPixelBuffer[ALOD: TGLImageLODRange]: TGLUnpackPBOHandle
      read GetLevelPBO;
    { LOD offset in byte }
    property LevelOffset[ALOD: TGLImageLODRange]: Integer
      read GetLevelOffset;
    { LOD size in byte }
    property LevelSizeInByte[ALOD: TGLImageLODRange]: Integer
      read GetLevelSizeInByte;
    property LevelStreamingState[ALOD: TGLImageLODRange]: TGLLODStreamingState
      read GetLevelStreamingState write SetLevelStreamingState;
    { Number of levels. }
    property LevelCount: TGLImageLODRange read fLevelCount;

    property InternalFormat: TGLInternalFormat read FInternalFormat;
    property ColorFormat: TGLenum read fColorFormat;
    property DataType: GLenum read fDataType;
    property ElementSize: Integer read fElementSize;
    property CubeMap: Boolean read fCubeMap;
    property TextureArray: Boolean read fTextureArray;
  end;

  TGLBaseImageClass = class of TGLBaseImage;

  // TGLImage
  //
    { Contains and manipulates a 32 bits (24+8) bitmap.
       This is the base class for preparing and manipulating textures in GLScene,
       this function does not rely on a windows handle and should be used for
       in-memory manipulations only. 
       16 bits textures are automatically converted to 24 bits and an opaque (255)
       alpha channel is assumed for all planes, the byte order is as specified
       in GL_RGBA. If 32 bits is used in this class, it can however output 16 bits texture
       data for use in OpenGL.
       The class has support for registering its content as a texture, as well
       as for directly drawing/reading from the current OpenGL buffer. }
  TGLImage = class(TGLBaseImage)
  private
     
    FVerticalReverseOnAssignFromBitmap: Boolean;
    FBlank: boolean;
    fOldColorFormat: GLenum;
    fOldDataType: GLenum;
    procedure DataConvertTask;
  protected
     
    procedure SetWidth(val: Integer);
    procedure SetHeight(const val: Integer);
    procedure SetDepth(const val: Integer);
    procedure SetBlank(const Value: boolean);
    procedure SetCubeMap(const val: Boolean);
    procedure SetArray(const val: Boolean);
    function GetScanLine(index: Integer): PGLPixel32Array;
    procedure AssignFrom24BitsBitmap(aBitmap: TGLBitmap);
    procedure AssignFrom32BitsBitmap(aBitmap: TGLBitmap);
{$IFDEF GLS_Graphics32_SUPPORT}
    procedure AssignFromBitmap32(aBitmap32: TBitmap32);
{$ENDIF}
{$IFDEF GLS_PngImage_SUPPORT}
    procedure AssignFromPngImage(aPngImage: TPngImage);
{$ENDIF}

  public
     
    constructor Create; override;
    destructor Destroy; override;
    { Accepts TGLImage and TGraphic subclasses. }
    procedure Assign(Source: TPersistent); override;
    { Assigns from a 24 bits bitmap without swapping RGB.
      This is faster than a regular assignment, but R and B channels
      will be reversed (from what you would view in a TImage). Suitable
      if you do your own drawing and reverse RGB on the drawing side. 
      If you're after speed, don't forget to set the bitmap's dimensions
      to a power of two! }
    procedure AssignFromBitmap24WithoutRGBSwap(aBitmap: TGLBitmap);
    { Assigns from a 2D Texture.
      The context which holds the texture must be active and the texture
      handle valid. }
    procedure AssignFromTexture2D(textureHandle: Cardinal); overload;
    { Assigns from a Texture handle.
      If the handle is invalid, the bitmap32 will be empty. }
    procedure AssignFromTexture2D(textureHandle: TGLTextureHandle); overload;

    { Create a 32 bits TBitmap from self content. }
    function Create32BitsBitmap: TGLBitmap;

    { Width of the bitmap. }
    property Width: Integer read GetWidth write SetWidth;
    { Height of the bitmap. }
    property Height: Integer read GetHeight write SetHeight;
    { Depth of the bitmap. }
    property Depth: Integer read GetDepth write SetDepth;
    { OpenGL color format }
    property ColorFormat: GLenum read fColorFormat;
    { Recommended texture internal format }
    property InternalFormat: TGLInternalFormat read FInternalFormat write
      FInternalFormat;
    { OpenGL data type }
    property DataType: GLenum read fDataType;
    { Size in bytes of pixel or block }
    property ElementSize: Integer read fElementSize;

    property CubeMap: Boolean read fCubeMap write SetCubeMap;

    property TextureArray: Boolean read fTextureArray write SetArray;
    { Access to a specific Bitmap ScanLine.
      index should be in the [0; Height[ range.
      Warning : this function is NOT protected against invalid indexes,
      and invoking it is invalid if the bitmap is Empty. }
    property ScanLine[index: Integer]: PGLPixel32Array read GetScanLine;

    property VerticalReverseOnAssignFromBitmap: Boolean read
      FVerticalReverseOnAssignFromBitmap write
      FVerticalReverseOnAssignFromBitmap;

    { Set Blank to true if you actually don't need to allocate data in main
      menory.
      Useful for textures that are generated by the GPU on the fly. }
    property Blank: boolean read FBlank write SetBlank;

    { Recast image OpenGL data type and color format. }
    procedure SetColorFormatDataType(const AColorFormat, ADataType: GLenum);
    { Set Alpha channel values to the pixel intensity.
      The intensity is calculated as the mean of RGB components. }
    procedure SetAlphaFromIntensity;
    { Set Alpha channel to 0 for pixels of given color, 255 for others).
      This makes pixels of given color totally transparent while the others
      are completely opaque. }
    procedure SetAlphaTransparentForColor(const aColor: TColor); overload;
    procedure SetAlphaTransparentForColor(const aColor: TGLPixel32); overload;
    procedure SetAlphaTransparentForColor(const aColor: TGLPixel24); overload;
    { Set Alpha channel values to given byte value. }
    procedure SetAlphaToValue(const aValue: Byte);
    { Set Alpha channel values to given float [0..1] value. }
    procedure SetAlphaToFloatValue(const aValue: Single);
    { Inverts the AlphaChannel component.
      What was transparent becomes opaque and vice-versa. }
    procedure InvertAlpha;
    { AlphaChannel components are replaced by their sqrt. }
    procedure SqrtAlpha;

    { Apply a brightness (scaled saturating) correction to the RGB components. }
    procedure BrightnessCorrection(const factor: Single);
    { Apply a gamma correction to the RGB components. }
    procedure GammaCorrection(const gamma: Single);

    { Downsample the bitmap by a factor of 2 in both dimensions.
      If one of the dimensions is 1 or less, does nothing. }
    procedure DownSampleByFactor2;

    { Reads the given area from the current active OpenGL rendering context.
      The best spot for reading pixels is within a SceneViewer's PostRender
      event : the scene has been fully rendered and the OpenGL context
      is still active. }
    procedure ReadPixels(const area: TGLRect);
    { Draws the whole bitmap at given position in the current OpenGL context.
      This function must be called with a rendering context active.
      Blending and Alpha channel functions are not altered by this function
      and must be adjusted separately. }
    procedure DrawPixels(const x, y: Single);

    { Converts a grayscale 'elevation' bitmap to normal map.
      Actually, only the Green component in the original bitmap is used. }
    procedure GrayScaleToNormalMap(const scale: Single;
      wrapX: Boolean = True; wrapY: Boolean = True);
    { Assumes the bitmap content is a normal map and normalizes all pixels. }
    procedure NormalizeNormalMap;
    procedure AssignToBitmap(aBitmap: TGLBitmap);
    { Generate level of detail. }
    procedure GenerateMipmap(AFilter: TImageFilterFunction); override;
    { Clear all levels except first. }
    procedure UnMipmap; override;
  end;

  TGLBitmap32 = TGLImage;

  // TRasterFileFormat
  //
  TRasterFileFormat = class
  public
    BaseImageClass: TGLBaseImageClass;
    Extension: string;
    Description: string;
    DescResID: Integer;
  end;

  // TRasterFileFormatsList
  //
  { Stores registered raster file formats. }
  TRasterFileFormatsList = class(TPersistentObjectList)
  public
     
    destructor Destroy; override;

    procedure Add(const Ext, Desc: string; DescID: Integer; AClass:
      TGLBaseImageClass);
    function FindExt(ext: string): TGLBaseImageClass;
    function FindFromFileName(const fileName: string): TGLBaseImageClass;
    function FindFromStream(const AStream: TStream): TGLBaseImageClass;
    procedure Remove(AClass: TGLBaseImageClass);
    procedure BuildFilterStrings(imageFileClass: TGLBaseImageClass;
      var descriptions, filters: string;
      formatsThatCanBeOpened: Boolean = True;
      formatsThatCanBeSaved: Boolean = False);
    function FindExtByIndex(index: Integer;
      formatsThatCanBeOpened: Boolean = True;
      formatsThatCanBeSaved: Boolean = False): string;
  end;

  EInvalidRasterFile = class(Exception);

procedure Div2(var Value: Integer);

procedure BGR24ToRGB24(src, dest: Pointer; pixelCount: Integer);
procedure BGR24ToRGBA32(src, dest: Pointer; pixelCount: Integer);
procedure RGB24ToRGBA32(src, dest: Pointer; pixelCount: Integer);
procedure BGRA32ToRGBA32(src, dest: Pointer; pixelCount: Integer);

procedure GammaCorrectRGBArray(base: Pointer; pixelCount: Integer;
  gamma: Single);
procedure BrightenRGBArray(base: Pointer; pixelCount: Integer;
  factor: Single);
// Read access to the list of registered vector file formats
function GetRasterFileFormats: TRasterFileFormatsList;
{ Returns an extension by its index
   in the internal image files dialogs filter.
   Use InternalImageFileFormatsFilter to obtain the filter. }
function RasterFileFormatExtensionByIndex(index: Integer): string;

procedure RegisterRasterFormat(const AExtension, ADescription: string;
  AClass: TGLBaseImageClass);
procedure UnregisterRasterFormat(AClass: TGLBaseImageClass);
// Return an optimal number of texture pyramid
function GetImageLodNumber(w, h, d: integer; IsVolume: Boolean): Integer;

var
  vVerticalFlipDDS: Boolean = true;
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

resourcestring
  glsCantConvertImg = '%s: can''t convert image to RGBA8 format';

var
  vRasterFileFormats: TRasterFileFormatsList;

{$IFDEF GLS_REGIONS}{%REGION  'Raster File Registries'}{$ENDIF}

  // GetRasterFileFormats
  //

function GetRasterFileFormats: TRasterFileFormatsList;
begin
  if not Assigned(vRasterFileFormats) then
    vRasterFileFormats := TRasterFileFormatsList.Create;
  Result := vRasterFileFormats;
end;

// RegisterRasterFormat
//

procedure RegisterRasterFormat(const AExtension, ADescription: string;
  AClass: TGLBaseImageClass);
begin
  RegisterClass(AClass);
  GetRasterFileFormats.Add(AExtension, ADescription, 0, AClass);
end;

// UnregisterRasterFormat
//

procedure UnregisterRasterFormat(AClass: TGLBaseImageClass);
begin
  if Assigned(vRasterFileFormats) then
    vRasterFileFormats.Remove(AClass);
end;

// RasterFileFormatExtensionByIndex
//

function RasterFileFormatExtensionByIndex(index: Integer): string;
begin
  Result := GetRasterFileFormats.FindExtByIndex(index);
end;

// TRasterFileFormatsList.Destroy
//

destructor TRasterFileFormatsList.Destroy;
begin
  Clean;
  inherited;
end;

// Add
//

procedure TRasterFileFormatsList.Add(const Ext, Desc: string; DescID: Integer;
  AClass: TGLBaseImageClass);
var
  newRec: TRasterFileFormat;
begin
  newRec := TRasterFileFormat.Create;
  with newRec do
  begin
    Extension := AnsiLowerCase(Ext);
    BaseImageClass := AClass;
    Description := Desc;
    DescResID := DescID;
  end;
  inherited Add(newRec);
end;

// FindExt
//

function TRasterFileFormatsList.FindExt(ext: string): TGLBaseImageClass;
var
  i: Integer;
begin
  ext := AnsiLowerCase(ext);
  for i := Count - 1 downto 0 do
    with TRasterFileFormat(Items[I]) do
    begin
      if Extension = ext then
      begin
        Result := BaseImageClass;
        Exit;
      end;
    end;
  Result := nil;
end;

// FindFromFileName
//

function TRasterFileFormatsList.FindFromFileName(const fileName: string):
  TGLBaseImageClass;
var
  ext: string;
begin
  ext := ExtractFileExt(Filename);
  System.Delete(ext, 1, 1);
  Result := FindExt(ext);
  if not Assigned(Result) then
    raise EInvalidRasterFile.CreateFmt(glsUnknownExtension,
      [ext, 'GLFile' + UpperCase(ext)]);
end;

// FindFromStream
//

function TRasterFileFormatsList.FindFromStream(const AStream: TStream): TGLBaseImageClass;
var
  ext: string;
  magic: array[0..1] of LongWord;
begin
  magic[0] := 0;
  magic[1] := 1;
  AStream.ReadBuffer(magic, 2 * SizeOf(Longword));
  AStream.Seek(-2 * SizeOf(Longword), 1);

  if magic[0] = $20534444 then
    ext := 'DDS'
  else if magic[1] = $4354334F then
    ext := 'O3TC'
  else if (magic[0] and $0000FFFF) = $00003F23 then
    ext := 'HDR'
  else if (magic[0] = $474E5089) and (magic[1] = $0A1A0A0D) then
    ext := 'PNG'
  else if (magic[0] = $E0FFD8FF) and (magic[1] = $464A1000) then
    ext := 'JPG';

  Result := FindExt(ext);
  if not Assigned(Result) then
    raise EInvalidRasterFile.CreateFmt(glsUnknownExtension,
      [ext, 'GLFile' + UpperCase(ext)]);
end;

// Remove
//

procedure TRasterFileFormatsList.Remove(AClass: TGLBaseImageClass);
var
  i: Integer;
begin
  for i := Count - 1 downto 0 do
  begin
    if TRasterFileFormat(Items[i]).BaseImageClass.InheritsFrom(AClass) then
      DeleteAndFree(i);
  end;
end;

// BuildFilterStrings
//

procedure TRasterFileFormatsList.BuildFilterStrings(
  imageFileClass: TGLBaseImageClass;
  var descriptions, filters: string;
  formatsThatCanBeOpened: Boolean = True;
  formatsThatCanBeSaved: Boolean = False);
var
  k, i: Integer;
  p: TRasterFileFormat;
begin
  descriptions := '';
  filters := '';
  k := 0;
  for i := 0 to Count - 1 do
  begin
    p := TRasterFileFormat(Items[i]);
    if p.BaseImageClass.InheritsFrom(imageFileClass) and (p.Extension <> '')
      and ((formatsThatCanBeOpened and (dfcRead in
      p.BaseImageClass.Capabilities))
      or (formatsThatCanBeSaved and (dfcWrite in p.BaseImageClass.Capabilities))) then
    begin
      with p do
      begin
        if k <> 0 then
        begin
          descriptions := descriptions + '|';
          filters := filters + ';';
        end;
        if (Description = '') and (DescResID <> 0) then
          Description := LoadStr(DescResID);
        FmtStr(descriptions, '%s%s (*.%s)|*.%2:s', [descriptions, Description,
          Extension]);
        filters := filters + '*.' + Extension;
        Inc(k);
      end;
    end;
  end;
  if (k > 1) and (not formatsThatCanBeSaved) then
    FmtStr(descriptions, '%s (%s)|%1:s|%s',
      [glsAllFilter, filters, descriptions]);
end;

// FindExtByIndex
//

function TRasterFileFormatsList.FindExtByIndex(index: Integer;
  formatsThatCanBeOpened: Boolean = True;
  formatsThatCanBeSaved: Boolean = False): string;
var
  i: Integer;
  p: TRasterFileFormat;
begin
  Result := '';
  if index > 0 then
  begin
    for i := 0 to Count - 1 do
    begin
      p := TRasterFileFormat(Items[i]);
      if (formatsThatCanBeOpened and (dfcRead in p.BaseImageClass.Capabilities))
        or (formatsThatCanBeSaved and (dfcWrite in
        p.BaseImageClass.Capabilities)) then
      begin
        if index = 1 then
        begin
          Result := p.Extension;
          Break;
        end
        else
          Dec(index);
      end;
    end;
  end;
end;

procedure Div2(var Value: Integer);
begin
  Value := Value div 2;
  if Value = 0 then
    Inc(Value);
end;

function GetImageLodNumber(w, h, d: integer; IsVolume: Boolean): Integer;
var
  L: Integer;
begin
  L := 1;
  d := MaxInteger(d, 1);
  while ((w > 1) or (h > 1) or (d > 1)) do
  begin
    Div2(w);
    Div2(h);
    if IsVolume then
      Div2(d);
    Inc(L);
  end;
  Result := L;
end;

procedure CalcImagePiramid(var APiramid: TGLImagePiramid);
begin

end;

{$IFDEF GLS_REGIONS}{$ENDREGION}{$ENDIF}

{$IFDEF GLS_REGIONS}{$REGION 'RGBA Utils'}{$ENDIF}
// GammaCorrectRGBArray
//

procedure GammaCorrectRGBArray(base: Pointer; pixelCount: Integer;
  gamma: Single);
var
  vGammaLUT: array[0..255] of Byte;
  invGamma: Single;
  i: Integer;
  ptr: PByte;
begin
  if pixelCount < 1 then
    Exit;
  Assert(gamma > 0);
  // build LUT
  if gamma < 0.1 then
    invGamma := 10
  else
    invGamma := 1 / gamma;
  for i := 0 to 255 do
    vGammaLUT[i] := Round(255 * Power(i * (1 / 255), InvGamma));
  // perform correction
  ptr := base;
  for i := 0 to pixelCount * 3 - 1 do
  begin
    ptr^ := vGammaLUT[ptr^];
    Inc(ptr);
  end;
end;

// GammaCorrectRGBAArray
//

procedure GammaCorrectRGBAArray(base: Pointer; pixelCount: Integer;
  gamma: Single);
var
  vGammaLUT: array[0..255] of Byte;
  pLUT: PByteArray;
  invGamma: Single;
  i: Integer;
  ptr: PByte;
begin
  if pixelCount < 1 then
    Exit;
  Assert(gamma > 0);
  // build LUT
  if gamma < 0.1 then
    invGamma := 10
  else
    invGamma := 1 / gamma;
  for i := 0 to 255 do
    vGammaLUT[i] := Round(255 * Power(i * (1 / 255), InvGamma));
  // perform correction
  ptr := base;
  pLUT := @vGammaLUT[0];
  for i := 0 to pixelCount - 1 do
  begin
    ptr^ := pLUT^[ptr^];
    Inc(ptr);
    ptr^ := pLUT^[ptr^];
    Inc(ptr);
    ptr^ := pLUT^[ptr^];
    Inc(ptr, 2);
  end;
end;

// BrightenRGBArray
//

procedure BrightenRGBArray(base: Pointer; pixelCount: Integer;
  factor: Single);
var
  vBrightnessLUT: array[0..255] of Byte;
  i, k: Integer;
  ptr: PByte;
begin
  if pixelCount < 1 then
    Exit;
  Assert(factor >= 0);
  // build LUT
  for i := 0 to 255 do
  begin
    k := Round(factor * i);
    if k > 255 then
      k := 255;
    vBrightnessLUT[i] := Byte(k);
  end;
  // perform correction
  ptr := base;
  for i := 0 to pixelCount * 3 - 1 do
  begin
    ptr^ := vBrightnessLUT[ptr^];
    Inc(ptr);
  end;
end;

// BrightenRGBAArray
//

procedure BrightenRGBAArray(base: Pointer; pixelCount: Integer;
  factor: Single);
var
  vBrightnessLUT: array[0..255] of Byte;
  pLUT: PByteArray;
  i: Integer;
  ptr: PByte;
  k: Integer;
begin
  if pixelCount < 1 then
    Exit;
  Assert(factor >= 0);
  // build LUT
  for i := 0 to 255 do
  begin
    k := Round(factor * i);
    if k > 255 then
      k := 255;
    vBrightnessLUT[i] := k;
  end;
  // perform correction
  ptr := base;
  pLUT := @vBrightnessLUT[0];
  for i := 0 to pixelCount - 1 do
  begin
    ptr^ := pLUT^[ptr^];
    Inc(ptr);
    ptr^ := pLUT^[ptr^];
    Inc(ptr);
    ptr^ := pLUT^[ptr^];
    Inc(ptr, 2);
  end;
end;

// BGR24ToRGB24
//

procedure BGR24ToRGB24(src, dest: Pointer; pixelCount: Integer); register;
begin
  while pixelCount > 0 do
  begin
    PAnsiChar(dest)[0] := PAnsiChar(src)[2];
    PAnsiChar(dest)[1] := PAnsiChar(src)[1];
    PAnsiChar(dest)[2] := PAnsiChar(src)[0];
    Inc(PAnsiChar(dest), 3);
    Inc(PAnsiChar(src), 3);
    Dec(pixelCount);
  end;
end;

// BGR24ToRGBA32
//
{$IFNDEF GEOMETRY_NO_ASM}

procedure BGR24ToRGBA32(src, dest: Pointer; pixelCount: Integer); register;
// EAX stores src
// EDX stores dest
// ECX stores pixelCount
asm
         push  edi
         cmp   ecx, 0
         jle   @@Done
         mov   edi, eax
         dec   ecx
         jz    @@Last
@@Loop:
         mov   eax, [edi]
         shl   eax, 8
         or    eax, $FF
         bswap eax
         mov   [edx], eax
         add   edi, 3
         add   edx, 4
         dec   ecx
         jnz   @@Loop
@@Last:
         mov   cx, [edi+1]
         shl   ecx, 16
         mov   ah, [edi]
         mov   al, $FF
         and   eax, $FFFF
         or    eax, ecx
         bswap eax
         mov   [edx], eax
@@Done:
         pop   edi
end;
{$ELSE}

procedure BGR24ToRGBA32(src, dest: Pointer; pixelCount: Integer);
begin
  while pixelCount > 0 do
  begin
    PAnsiChar(dest)[0] := PAnsiChar(src)[2];
    PAnsiChar(dest)[1] := PAnsiChar(src)[1];
    PAnsiChar(dest)[2] := PAnsiChar(src)[0];
    PAnsiChar(dest)[3] := #255;
    Inc(PAnsiChar(dest), 4);
    Inc(PAnsiChar(src), 3);
    Dec(pixelCount);
  end;
end;
{$ENDIF}

// RGB24ToRGBA32
//
{$IFNDEF GEOMETRY_NO_ASM}

procedure RGB24ToRGBA32(src, dest: Pointer; pixelCount: Integer); register;
// EAX stores src
// EDX stores dest
// ECX stores pixelCount
asm
         push  edi
         cmp   ecx, 0
         jle   @@Done
         mov   edi, eax
         dec   ecx
         jz    @@Last
@@Loop:
         mov   eax, [edi]
         or    eax, $FF000000
         mov   [edx], eax
         add   edi, 3
         add   edx, 4
         dec   ecx
         jnz   @@Loop
@@Last:
         mov   ax, [edi+1]
         shl   eax, 8
         mov   al, [edi];
         or    eax, $FF000000
         mov   [edx], eax
@@Done:
         pop   edi
end;
{$ELSE}

procedure RGB24ToRGBA32(src, dest: Pointer; pixelCount: Integer);
begin
  while pixelCount > 0 do
  begin
    PAnsiChar(dest)[0] := PAnsiChar(src)[0];
    PAnsiChar(dest)[1] := PAnsiChar(src)[1];
    PAnsiChar(dest)[2] := PAnsiChar(src)[2];
    PAnsiChar(dest)[3] := #255;
    Inc(PAnsiChar(dest), 4);
    Inc(PAnsiChar(src), 3);
    Dec(pixelCount);
  end;
end;
{$ENDIF}

// BGRA32ToRGBA32
//
{$IFNDEF GEOMETRY_NO_ASM}

procedure BGRA32ToRGBA32(src, dest: Pointer; pixelCount: Integer); register;
// EAX stores src
// EDX stores dest
// ECX stores pixelCount
asm
         push  edi
         cmp   ecx, 0
         jle   @@Done
         mov   edi, eax
@@Loop:
         mov   eax, [edi]
         shl   eax, 8
         mov   al, [edi+3]
         bswap eax
         mov   [edx], eax
         add   edi, 4
         add   edx, 4
         dec   ecx
         jnz   @@Loop
@@Done:
         pop   edi
end;
{$ELSE}

procedure BGRA32ToRGBA32(src, dest: Pointer; pixelCount: Integer);
begin
  while pixelCount > 0 do
  begin
    PAnsiChar(dest)[0] := PAnsiChar(src)[2];
    PAnsiChar(dest)[1] := PAnsiChar(src)[1];
    PAnsiChar(dest)[2] := PAnsiChar(src)[0];
    PAnsiChar(dest)[3] := PAnsiChar(src)[3];
    Inc(PAnsiChar(dest), 4);
    Inc(PAnsiChar(src), 4);
    Dec(pixelCount);
  end;
end;
{$ENDIF}

{$IFDEF GLS_REGIONS}{$ENDREGION}{$ENDIF}

// ------------------
// ------------------ TGLBaseImage ------------------
// ------------------

{$IFDEF GLS_REGIONS}{$REGION 'TGLBaseImage'}{$ENDIF}
// Create
//

constructor TGLBaseImage.Create;
begin
  inherited Create(Self);
  FillChar(FLOD, SizeOf(TGLImagePiramid), $00);
  fLevelCount := 1; // first level always is present
  fColorFormat := GL_RGBA;
  fInternalFormat := tfRGBA8;
  fDataType := GL_UNSIGNED_BYTE;
  fElementSize := 4;
  fCubeMap := false;
  fTextureArray := false;
end;

// Destroy
//

destructor TGLBaseImage.Destroy;
var
  level: TGLImageLODRange;
begin
  if Assigned(fData) then
  begin
    FreeMem(fData);
    fData := nil;
  end;
  FreeAndNil(FFinishEvent);
  for level := 0 to High(TGLImageLODRange) do
  begin
    FLOD[level].PBO.Free;
  end;
  FSourceStream.Free;
  inherited Destroy;
end;

 
//

procedure TGLBaseImage.Assign(Source: TPersistent);
var
  img: TGLBaseImage;
  size: integer;
begin
  if Source is TGLBaseImage then
  begin
    img := Source as TGLBaseImage;
    FLOD := img.FLOD;
    fLevelCount := img.fLevelCount;
    fColorFormat := img.fColorFormat;
    fInternalFormat := img.fInternalFormat;
    fDataType := img.fDataType;
    fElementSize := img.fElementSize;
    fCubeMap := img.fCubeMap;
    fTextureArray := img.fTextureArray;
    size := img.DataSize;
    ReallocMem(FData, size);
    Move(img.fData^, fData^, size);
  end
  else if Source <> nil then
    inherited; // raise AssingError
end;

function TGLBaseImage.GetTextureTarget: TGLTextureTarget;
begin
  Result := ttTexture2D;
  // Choose a texture target
  if GetHeight = 1 then
    Result := ttTexture1D;
  if FCubeMap then
    Result := ttTextureCube;
  if IsVolume then
    Result := ttTexture3D;
  if FTextureArray then
  begin
    if (GetDepth < 2) then
      Result := ttTexture1Darray
    else
      Result := ttTexture2DArray;
    if FCubeMap then
      Result := ttTextureCubeArray;
  end;
  if ((FInternalFormat >= tfFLOAT_R16)
    and (FInternalFormat <= tfFLOAT_RGBA32)) then
    Result := ttTextureRect;
end;

// DataSize
//

function TGLBaseImage.DataSize: PtrUint;
var
  l: TGLImageLODRange;
  s: PtrUint;
begin
  s := 0;
  if not IsEmpty then
  begin
    UpdateLevelsInfo;
    for l := 0 to FLevelCount - 1 do
      s := s + FLOD[l].Size;
  end;
  Result := s;
end;

function TGLBaseImage.GetWidth: Integer;
begin
  Result := FLOD[0].Width;
end;

function TGLBaseImage.GetDepth: Integer;
begin
  Result := FLOD[0].Depth;
end;

function TGLBaseImage.GetHeight: Integer;
begin
  Result := FLOD[0].Height;
end;

function TGLBaseImage.GetLevelAddress(ALevel: Byte): Pointer;
begin
  Result := FData;
  Inc(PByte(Result), FLOD[ALevel].Offset);
end;

function TGLBaseImage.GetLevelAddress(ALevel, AFace: Byte): Pointer;
begin
  Result := FData;
  Inc(PByte(Result), FLOD[ALevel].Offset);
  Inc(PByte(Result), AFace*(FLOD[ALevel].Size div 6));
end;

function TGLBaseImage.GetLevelDepth(ALOD: TGLImageLODRange): Integer;
begin
  Result := FLOD[ALOD].Depth;
end;

function TGLBaseImage.GetLevelHeight(ALOD: TGLImageLODRange): Integer;
begin
  Result := FLOD[ALOD].Height;
end;

function TGLBaseImage.GetLevelOffset(ALOD: TGLImageLODRange): Integer;
begin
  Result := FLOD[ALOD].Offset;
end;

function TGLBaseImage.GetLevelPBO(ALOD: TGLImageLODRange): TGLUnpackPBOHandle;
begin
  Result := FLOD[ALOD].PBO;
end;

function TGLBaseImage.GetLevelSizeInByte(ALOD: TGLImageLODRange): Integer;
begin
  Result := FLOD[ALOD].Size;
end;

function TGLBaseImage.GetLevelStreamingState(ALOD: TGLImageLODRange): TGLLODStreamingState;
begin
  Result := FLOD[ALOD].State;
end;

function TGLBaseImage.GetLevelWidth(ALOD: TGLImageLODRange): Integer;
begin
  Result := FLOD[ALOD].Width;
end;

// IsEmpty
//

function TGLBaseImage.IsEmpty: Boolean;
begin
  Result := (GetWidth = 0) or (GetHeight = 0);
end;

// IsCompressed
//

function TGLBaseImage.IsCompressed: Boolean;
begin
  Result := IsCompressedFormat(fInternalFormat);
end;

// IsVolume
//

function TGLBaseImage.IsVolume: boolean;
begin
  Result := (GetDepth > 0) and not fTextureArray and not fCubeMap;
end;


// ConvertCrossToCubemap
//

function TGLBaseImage.ConvertCrossToCubemap: Boolean;
var
  fW, fH, cubeSize, realCubeSize, e: integer;
  lData: PByteArray;
  ptr: PGLubyte;
  i, j: integer;
  bGenMipmap: Boolean;
begin
  Result := False;
  // Can't already be a cubemap
  if fCubeMap or fTextureArray then
    Exit;
  //this function only supports vertical cross format for now (3 wide by 4 high)
  if (GetWidth div 3 <> GetHeight div 4)
    or (GetWidth mod 3 <> 0)
    or (GetHeight mod 4 <> 0)
    or (GetDepth > 0) then
    Exit;

  bGenMipmap := FLevelCount > 1;
  UnMipmap;

  // Get the source data
  lData := PByteArray(fData);
  if IsCompressed then
  begin
    fW := (GetWidth + 3) div 4;
    fH := (GetHeight + 3) div 4;
    realCubeSize := (fH div 4) * 4;
  end
  else
  begin
    fW := GetWidth;
    fH := GetHeight;
    realCubeSize := fH div 4;
  end;
  cubeSize := fH;
  GetMem(fData, fW * fH * fElementSize);
  FLOD[0].Width := realCubeSize;
  FLOD[0].Height := realCubeSize;
  FLOD[0].Depth := 6;

  // Extract the faces
  ptr := PGLubyte(fData);
  // positive X
  for j := 0 to cubeSize - 1 do
  begin
    e := ((fH - (cubeSize + j + 1)) * fW + 2 * cubeSize) * fElementSize;
    Move(lData[E], ptr^, cubeSize * fElementSize);
    Inc(ptr, cubeSize * fElementSize);
  end;
  // negative X
  for j := 0 to cubeSize - 1 do
  begin
    Move(lData[(fH - (cubeSize + j + 1)) * fW * fElementSize],
      ptr^, cubeSize * fElementSize);
    Inc(ptr, cubeSize * fElementSize);
  end;
  // positive Y
  for j := 0 to cubeSize - 1 do
  begin
    e := ((4 * cubeSize - j - 1) * fW + cubeSize) * fElementSize;
    Move(lData[e], ptr^, cubeSize * fElementSize);
    Inc(ptr, cubeSize * fElementSize);
  end;
  // negative Y
  for j := 0 to cubeSize - 1 do
  begin
    e := ((2 * cubeSize - j - 1) * fW + cubeSize) * fElementSize;
    Move(lData[e], ptr^, cubeSize * fElementSize);
    Inc(ptr, cubeSize * fElementSize);
  end;
  // positive Z
  for j := 0 to cubeSize - 1 do
  begin
    e := ((fH - (cubeSize + j + 1)) * fW + cubeSize) * fElementSize;
    Move(lData[e], ptr^, cubeSize * fElementSize);
    Inc(ptr, cubeSize * fElementSize);
  end;
  // negative Z
  for j := 0 to cubeSize - 1 do
    for i := 0 to cubeSize - 1 do
    begin
      e := (j * fW + 2 * cubeSize - (i + 1)) * fElementSize;
      Move(lData[e], ptr^, fElementSize);
      Inc(ptr, fElementSize);
    end;
  // Set the new # of faces, width and height
  fCubeMap := true;
  FreeMem(lData);

  if bGenMipmap then
    GenerateMipmap(ImageTriangleFilter);

  Result := true;
end;

// ConvertToVolume
//

function TGLBaseImage.ConvertToVolume(const col, row: Integer; const MakeArray:
  Boolean): Boolean;
var
  fW, fH, sW, sH, sD: Integer;
  lData: PByteArray;
  ptr: PGLubyte;
  i, j, k: Integer;
begin
  Result := false;
  if fCubeMap then
    Exit;

  if (GetDepth > 0) and not fTextureArray and MakeArray then
  begin
    // Let volume be array
    fTextureArray := true;
    Result := true;
    exit;
  end;
  if fTextureArray and not MakeArray then
  begin
    // Let array be volume
    fTextureArray := false;
    Result := true;
    exit;
  end;

  Result := MakeArray;

  // Check sizes
  sD := col * row;
  if sD < 1 then
    Exit;
  if IsCompressed then
  begin
    fW := (GetWidth + 3) div 4;
    fH := (GetHeight + 3) div 4;
  end
  else
  begin
    fW := GetWidth;
    fH := GetHeight;
  end;
  sW := fW div col;
  sH := fH div row;
  if (sW = 0) or (sH = 0) then
  begin
    Result := False;
    Exit;
  end;

  // Mipmaps are not supported
  UnMipmap;
  // Get the source data
  lData := PByteArray(fData);
  GetMem(fData, sW * sH * sD * fElementSize);
  ptr := PGLubyte(fData);
  for i := 0 to row - 1 do
    for j := 0 to col - 1 do
      for k := 0 to sH - 1 do
      begin
        Move(lData[(i * fW * sH + j * sW + k * fW) * fElementSize],
          ptr^, sW * fElementSize);
        Inc(ptr, sW * fElementSize);
      end;

  if IsCompressed then
  begin
    FLOD[0].Width := sW * 4;
    FLOD[0].Height := sH * 4;
  end
  else
  begin
    FLOD[0].Width := sW;
    FLOD[0].Height := sH;
  end;
  FLOD[0].Depth := sD;
  fTextureArray := Result;
  FreeMem(lData);
  Result := True;
end;

procedure TGLBaseImage.SetErrorImage;
const
{$I TextureError.inc}
begin
  UnMipmap;
  FLOD[0].Width := 64;
  FLOD[0].Height := 64;
  FLOD[0].Depth := 0;
  fColorFormat := GL_RGBA;
  fInternalFormat := tfRGB8;
  fDataType := GL_UNSIGNED_BYTE;
  fElementSize := 4;
  fCubeMap := false;
  fTextureArray := false;
  FColorFormat := GL_RGB;
  ReallocMem(FData, DataSize);
  Move(cTextureError[0], FData[0], DataSize);
end;

procedure TGLBaseImage.SetLevelStreamingState(ALOD: TGLImageLODRange;
  AState: TGLLODStreamingState);
begin
  FLOD[ALOD].State := AState;
end;

// Narrow
//

procedure TGLBaseImage.Narrow;
var
  size: Integer;
  newData: Pointer;
begin
  // Check for already norrow
  if (fColorFormat = GL_RGBA)
    and (GetDepth = 0)
    and (fDataType = GL_UNSIGNED_BYTE)
    and (FLevelCount = 1)
    and not (fTextureArray or fCubeMap) then
    Exit;

  UnMipmap;
  // Use GLScene image utils
  size := GetWidth * GetHeight * 4;
  GetMem(newData, size);
  try
    ConvertImage(
      fData, newData,
      fColorFormat, GL_RGBA,
      fDataType, GL_UNSIGNED_BYTE,
      GetWidth, GetHeight);
  except
    GLSLogger.LogError(Format(glsCantConvertImg, [ClassName]));
    SetErrorImage;
    FreeMem(newData);
    exit;
  end;
  fInternalFormat := tfRGBA8;
  fColorFormat := GL_RGBA;
  fDataType := GL_UNSIGNED_BYTE;
  fElementSize := 4;
  fTextureArray := False;
  fCubeMap := False;
  FreeMem(fData);
  fData := newData;
end;

// GemerateMipmap
//

procedure TGLBaseImage.GenerateMipmap(AFilter: TImageFilterFunction);
var
  LAddresses: TPointerArray;
  level, slice, d: Integer;
begin
  UnMipmap;
  if IsVolume then
  begin
    fLevelCount := GetImageLodNumber(GetWidth, GetHeight, GetDepth, True);
    UpdateLevelsInfo;
    ReallocMem(FData, DataSize);
    {Message Hint 'TGLBaseImage.GenerateMipmap not yet implemented for volume images' }
  end
  else
  begin
    fLevelCount := GetImageLodNumber(GetWidth, GetHeight, GetDepth, False);
    ReallocMem(FData, DataSize);

    SetLength(LAddresses, fLevelCount-1);
    for level := 1 to fLevelCount-1 do
      LAddresses[level-1] := GetLevelAddress(level);
    d := MaxInteger(GetDepth, 1);
    for slice := 0 to d - 1 do
    begin
      Build2DMipmap(
        GetLevelAddress(0),
        LAddresses,
        fColorFormat,
        fDataType,
        AFilter,
        GetWidth,
        GetHeight);
      for level := 1 to fLevelCount-1 do
        Inc(PByte(LAddresses[level-1]), GetLevelSizeInByte(level) div d);
    end;
  end;
end;

// UnMipmap
//

procedure TGLBaseImage.UnMipmap;
var
  level: TGLImageLODRange;
begin
  for level := 1 to High(TGLImageLODRange) do
  begin
    FLOD[level].Width := 0;
    FLOD[level].Height := 0;
    FLOD[level].Depth := 0;
  end;
  FLevelCount := 1;
end;

procedure TGLBaseImage.UpdateLevelsInfo;
var
  level: TGLImageLODRange;
  w, h, d: Integer;

  function GetSize(const level: Integer): integer;
  var
    ld, bw, bh, lsize: integer;
  begin
    if fTextureArray then
      ld := FLOD[0].Depth
    else
      ld := d;
    if ld = 0 then
      ld := 1;

    if IsCompressed then
    begin
      bw := (w + 3) div 4;
      bh := (h + 3) div 4;
    end
    else
    begin
      bw := w;
      bh := h;
    end;
    if bh = 0 then
      bh := 1;

    lsize := bw * bh * ld * fElementSize;
    if fCubeMap and not fTextureArray then
      lsize := lsize * 6;
    // Align to Double Word
    if (lsize and 3) <> 0 then
      lsize := 4 * (1 + lsize div 4);
    Result := lsize;
  end;

begin
  w := FLOD[0].Width;
  h := FLOD[0].Height;
  d := FLOD[0].Depth;
  FLOD[0].Size := GetSize(0);
  FLOD[0].Offset := 0;

  for level := 1 to High(TGLImageLODRange) do
  begin
    Div2(w);
    Div2(h);
    if not fTextureArray then
      d := d div 2;
    FLOD[level].Width := w;
    FLOD[level].Height := h;
    FLOD[level].Depth := d;
    FLOD[level].Offset := FLOD[level - 1].Offset + FLOD[level - 1].Size;
    FLOD[level].Size := GetSize(level);
  end;
end;

function TGLBaseImage.GetData: PGLPixel32Array;
begin
  Result := fData;
end;

// RegisterAsOpenGLTexture
//

procedure TGLBaseImage.RegisterAsOpenGLTexture(
  AHandle: TGLTextureHandle;
  aMipmapGen: Boolean;
  aTexFormat: TGLEnum;
  out texWidth: integer;
  out texHeight: integer;
  out texDepth: integer);
var
  glTarget: TGLEnum;
  glHandle: TGLuint;
  Level: integer;
  LLevelCount, face: integer;
  bCompress, bBlank: boolean;
  w, h, d, cw, ch, maxSize: GLsizei;
  p, buffer: Pointer;
  vtcBuffer, top, bottom: PGLubyte;
  i, j, k: Integer;
  transferMethod: 0..3;

  function blockOffset(x, y, z: Integer): Integer;
  begin

    if z >= (d and -4) then
      Result := fElementSize * (cw * ch * (d and -4) + x +
        cw * (y + ch * (z - 4 * ch)))
    else
      Result := fElementSize * (4 * (x + cw * (y + ch * floor(z / 4))) + (z and
        3));
    if Result < 0 then
      Result := 0;
  end;

begin
  if AHandle.Target = ttNoShape then
    exit;

  with GL do
  begin
    UpdateLevelsInfo;

    if Self is TGLImage then
      bBlank := TGLImage(Self).Blank
    else
      bBlank := False;
    // Check for Non-power-of-two
    if not ARB_texture_non_power_of_two then
    begin
      w := RoundUpToPowerOf2(GetWidth);
      h := RoundUpToPowerOf2(GetHeight);
      d := RoundUpToPowerOf2(GetDepth);
      if GetDepth = 0 then
        d := 0;
    end
    else
    begin
      w := GetWidth;
      h := GetHeight;
      d := GetDepth;
    end;

    // Check maximum dimension
    maxSize := CurrentGLContext.GLStates.MaxTextureSize;
    if w > maxSize then
      w := maxSize;
    if h > maxSize then
      h := maxSize;
    texWidth := w;
    texHeight := h;
    texDepth := d;
    LLevelCount := FLevelCount;
    bCompress := IsCompressed;

    // Rescale if need and can
    buffer := nil;
    if (w <> GetWidth) or (h <> GetHeight) then
    begin
      if not ((d > 0) // not volume
        or bCompress // not compressed
        or bBlank) then // not blank
      begin
        GetMem(buffer, w * h * fElementSize);
        try
          RescaleImage(
            FData,
            buffer,
            FColorFormat,
            FDataType,
            ImageLanczos3Filter,
            GetWidth, GetHeight,
            w, h);
          LLevelCount := 1;
        except
          bBlank := true;
        end;
      end
      else
        bBlank := true;
    end;
    if Self is TGLImage then
      TGLImage(Self).FBlank := bBlank;

    glHandle := AHandle.Handle;
    glTarget := DecodeGLTextureTarget(AHandle.Target);

    // Hardware mipmap autogeneration
    aMipmapGen := aMipmapGen and IsTargetSupportMipmap(glTarget);
    aMipmapGen := aMipmapGen and (LLevelCount = 1);
    if aMipmapGen then
    begin
      if SGIS_generate_mipmap then
      begin
      {  if EXT_direct_state_access then
          TextureParameterf(
            glHandle,
            glTarget,
            GL_GENERATE_MIPMAP_SGIS,
            GL_TRUE)
        else }
          TexParameteri(glTarget, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
      end
      else
      begin
        // Software LODs generation
        Self.GenerateMipmap(ImageTriangleFilter);
        LLevelCount := LevelCount;
      end;
    end;

    // Setup top limitation of LODs
    if SGIS_texture_lod and (LLevelCount > 1) then
   {   if EXT_direct_state_access then
        TextureParameterf(
          glHandle,
          glTarget,
          GL_TEXTURE_MAX_LEVEL_SGIS,
          LLevelCount - 1)
      else }
        TexParameteri(glTarget, GL_TEXTURE_MAX_LEVEL_SGIS, LLevelCount - 1);

    // Select transfer method
    if bCompress then
      transferMethod := 1
    else
      transferMethod := 0;
    if EXT_direct_state_access then
      transferMethod := transferMethod + 2;

    // if image is blank then doing only allocatation texture in videomemory
    vtcBuffer := nil;
    case AHandle.Target of

      ttTexture1D:
        for Level := 0 to LLevelCount - 1 do
        begin
          if Assigned(buffer) then
            p := buffer
          else if not bBlank then
            p := GetLevelAddress(Level)
          else
            p := nil;

          case transferMethod of
            0: TexImage1D(glTarget, Level, aTexFormat, w, 0, FColorFormat, FDataType, p);
            1: CompressedTexImage1D(glTarget, Level, aTexFormat, w, 0, GetLevelSizeInByte(Level), p);
            2: TextureImage1D(glHandle, glTarget, Level, aTexFormat, w, 0, FColorFormat, FDataType, p);
            3: CompressedTextureImage1D(glHandle, glTarget, Level, aTexFormat, w, 0, GetLevelSizeInByte(Level), p)
          end;

          Div2(w);
        end;

      ttTexture2D:
        for Level := 0 to LLevelCount - 1 do
        begin
          if Assigned(buffer) then
            p := buffer
          else if not bBlank then
            p := GetLevelAddress(Level)
          else
            p := nil;

          case transferMethod of
            0: TexImage2D(glTarget, Level, aTexFormat, w, h, 0, FColorFormat, FDataType, p);
            1: CompressedTexImage2D(glTarget, Level, aTexFormat, w, h, 0, GetLevelSizeInByte(Level), p);
            2: TextureImage2D(glHandle, glTarget, Level, aTexFormat, w, h, 0, FColorFormat, FDataType, p);
            3: CompressedTextureImage2D(glHandle, glTarget, Level, aTexFormat, w, h, 0, GetLevelSizeInByte(Level), p);
          end;

          Div2(w);
          Div2(h);
        end;

      ttTextureRect:
        begin
          if Assigned(buffer) then
            p := buffer
          else if not bBlank then
            p := GetLevelAddress(0)
          else
            p := nil;

          case transferMethod of
            0: TexImage2D(glTarget, 0, aTexFormat, w, h, 0, FColorFormat, FDataType, p);
            1: CompressedTexImage2D(glTarget, 0, aTexFormat, w, h, 0, GetLevelSizeInByte(0), p);
            2: TextureImage2D(glHandle, glTarget, 0, aTexFormat, w, h, 0, FColorFormat, FDataType, p);
            3: CompressedTextureImage2D(glHandle, glTarget, 0, aTexFormat, w, h, 0, GetLevelSizeInByte(0), p);
          end;
        end;

      ttTexture3D:
        for Level := 0 to LLevelCount - 1 do
        begin
          if Assigned(buffer) then
            p := buffer
          else if not bBlank then
            p := GetLevelAddress(Level)
          else
            p := nil;

          if GL.NV_texture_compression_vtc and bCompress then
          begin
            // Shufle blocks for Volume Texture Compression
            if Assigned(p) then
            begin
              cw := (w + 3) div 4;
              ch := (h + 3) div 4;
              if Level = 0 then
                GetMem(vtcBuffer, GetLevelSizeInByte(0));
              top := p;
              for k := 0 to d - 1 do
                for i := 0 to ch - 1 do
                  for j := 0 to cw - 1 do
                  begin
                    bottom := vtcBuffer;
                    Inc(bottom, blockOffset(j, i, k));
                    Move(top^, bottom^, fElementSize);
                    Inc(top, fElementSize);
                  end;
            end;
            if EXT_direct_state_access then
              CompressedTextureImage3D(glHandle, glTarget, Level, aTexFormat, w, h, d, 0, GetLevelSizeInByte(Level), vtcBuffer)
            else
              CompressedTexImage3D(glTarget, Level, aTexFormat, w, h, d, 0, GetLevelSizeInByte(Level), vtcBuffer);
          end
          else
          begin
            // Normal compression
            case transferMethod of
              0: TexImage3D(glTarget, Level, aTexFormat, w, h, d, 0, FColorFormat, FDataType, p);
              1: CompressedTexImage3D(glTarget, Level, aTexFormat, w, h, d, 0, GetLevelSizeInByte(Level), p);
              2: TextureImage3D(glHandle, glTarget, Level, aTexFormat, w, h, d, 0, FColorFormat, FDataType, p);
              3: CompressedTextureImage3D(glHandle, glTarget, Level, aTexFormat, w, h, d, 0, GetLevelSizeInByte(Level), p);
            end;

          end;
          Div2(w);
          Div2(h);
          Div2(d);
        end;

      ttTextureCube:
        for Level := 0 to LLevelCount - 1 do
        begin
          for face := GL_TEXTURE_CUBE_MAP_POSITIVE_X to
            GL_TEXTURE_CUBE_MAP_NEGATIVE_Z do
          begin
            if Assigned(buffer) then
              p := buffer
            else if not bBlank then
              p := GetLevelAddress(Level, face - GL_TEXTURE_CUBE_MAP_POSITIVE_X)
            else
              p := nil;

            case transferMethod of
              0: TexImage2D(face, Level, aTexFormat, w, h, 0, FColorFormat, FDataType, p);
              1: CompressedTexImage2D(face, Level, aTexFormat, w, h, 0, GetLevelSizeInByte(Level) div 6, p);
              2: TextureImage2D(glHandle, face, Level, aTexFormat, w, h, 0, FColorFormat, FDataType, p);
              3: CompressedTextureImage2D(glHandle, face, Level, aTexFormat, w, h, 0, GetLevelSizeInByte(Level) div 6, p);
            end;

          end;
          Div2(w);
          Div2(h);
        end;

      ttTexture1DArray:
        for Level := 0 to LLevelCount - 1 do
        begin
          if Assigned(buffer) then
            p := buffer
          else if not bBlank then
            p := GetLevelAddress(Level)
          else
            p := nil;

          case transferMethod of
            0: TexImage2D(glTarget, Level, aTexFormat, w, h, 0, FColorFormat, FDataType, p);
            1: CompressedTexImage2D(glTarget, Level, aTexFormat, w, h, 0, GetLevelSizeInByte(Level), p);
            2: TextureImage2D(glHandle, glTarget, Level, aTexFormat, w, h, 0, FColorFormat, FDataType, p);
            3: CompressedTextureImage2D(glHandle, glTarget, Level, aTexFormat, w, h, 0, GetLevelSizeInByte(Level), p);
          end;

          Div2(w);
        end;

      ttTexture2DArray, ttTextureCubeArray:
        for Level := 0 to LLevelCount - 1 do
        begin
          if Assigned(buffer) then
            p := buffer
          else if not bBlank then
            p := GetLevelAddress(Level)
          else
            p := nil;

          case transferMethod of
            0: TexImage3D(glTarget, Level, aTexFormat, w, h, d, 0, FColorFormat, FDataType, p);
            1: CompressedTexImage3D(glTarget, Level, aTexFormat, w, h, d, 0, GetLevelSizeInByte(Level), p);
            2: TextureImage3D(glHandle, glTarget, Level, aTexFormat, w, h, d, 0, FColorFormat, FDataType, p);
            3: CompressedTextureImage3D(glHandle, glTarget, Level, aTexFormat, w, h, d, 0, GetLevelSizeInByte(Level), p);
          end;

          Div2(w);
          Div2(h);
        end;
    end; // of case

    if Assigned(buffer) then
      FreeMem(buffer);
    if Assigned(vtcBuffer) then
      FreeMem(vtcBuffer);

  end; // of with GL
end;

// AssignFromTexture
//

function TGLBaseImage.AssignFromTexture(
  AHandle: TGLTextureHandle;
  const CastToFormat: Boolean;
  const intFormat: TGLInternalFormat = tfRGBA8;
  const colorFormat: TGLenum = 0;
  const dataType: TGLenum = 0): Boolean;
var
  LContext: TGLContext;
  texFormat, texLod, optLod: Cardinal;
  glTarget: TGLEnum;
  level, maxFace, face: Integer;
  lData: PGLubyte;
  residentFormat: TGLInternalFormat;
  bCompressed: Boolean;
  vtcBuffer, top, bottom: PGLubyte;
  i, j, k: Integer;
  w, d, h, cw, ch: Integer;

  function blockOffset(x, y, z: Integer): Integer;
  begin

    if z >= (d and -4) then
      Result := fElementSize * (cw * ch * (d and -4) + x +
        cw * (y + ch * (z - 4 * ch)))
    else
      Result := fElementSize * (4 * (x + cw * (y + ch * floor(z / 4))) + (z and
        3));
  end;

begin
  Result := False;
  LContext := CurrentGLContext;
  if LContext = nil then
  begin
    LContext := AHandle.RenderingContext;
    if LContext = nil then
      exit;
  end;

  LContext.Activate;
  if AHandle.IsDataNeedUpdate then
  begin
    LContext.Deactivate;
    exit;
  end;

  glTarget := DecodeGLTextureTarget(AHandle.Target);

  try
    LContext.GLStates.TextureBinding[0, AHandle.Target] := AHandle.Handle;

    FLevelCount := 0;
    GL.GetTexParameteriv(glTarget, GL_TEXTURE_MAX_LEVEL, @texLod);
    if glTarget = GL_TEXTURE_CUBE_MAP then
    begin
      fCubeMap := true;
      maxFace := 5;
      glTarget := GL_TEXTURE_CUBE_MAP_POSITIVE_X;
    end
    else
    begin
      fCubeMap := false;
      maxFace := 0;
    end;
    fTextureArray := (glTarget = GL_TEXTURE_1D_ARRAY)
      or (glTarget = GL_TEXTURE_2D_ARRAY)
      or (glTarget = GL_TEXTURE_CUBE_MAP_ARRAY);

    with GL do
    repeat
      // Check level existence
      GL.GetTexLevelParameteriv(glTarget, FLevelCount,
        GL_TEXTURE_INTERNAL_FORMAT,
        @texFormat);
      if texFormat = 1 then
        Break;
      Inc(FLevelCount);
      if FLevelCount = 1 then
      begin
        GetTexLevelParameteriv(glTarget, 0, GL_TEXTURE_WIDTH, @FLOD[0].Width);
        GetTexLevelParameteriv(glTarget, 0, GL_TEXTURE_HEIGHT,@FLOD[0].Height);
        FLOD[0].Depth := 0;
        if (glTarget = GL_TEXTURE_3D)
          or (glTarget = GL_TEXTURE_2D_ARRAY)
          or (glTarget = GL_TEXTURE_CUBE_MAP_ARRAY) then
          GetTexLevelParameteriv(glTarget, 0, GL_TEXTURE_DEPTH, @FLOD[0].Depth);
        residentFormat := OpenGLFormatToInternalFormat(texFormat);
        if CastToFormat then
          fInternalFormat := residentFormat
        else
          fInternalFormat := intFormat;
        FindCompatibleDataFormat(fInternalFormat, fColorFormat, fDataType);
        // Substitute properties if need
        if colorFormat > 0 then
          fColorFormat := colorFormat;
        if dataType > 0 then
          fDataType := dataType;
        // Get optimal number or MipMap levels
        optLod := GetImageLodNumber(GetWidth, GetHeight, GetDepth, glTarget = GL_TEXTURE_3D);
        if texLod > optLod then
          texLod := optLod;
        // Check for MipMap posibility
        if ((fInternalFormat >= tfFLOAT_R16)
          and (fInternalFormat <= tfFLOAT_RGBA32)) then
          texLod := 1;
      end;
    until FLevelCount = Integer(texLod);

    if FLevelCount > 0 then
    with GL do
    begin
      fElementSize := GetTextureElementSize(fColorFormat, fDataType);
      UpdateLevelsInfo;

      ReallocMem(FData, DataSize);
      lData := PGLubyte(fData);
      bCompressed := IsCompressed;
      vtcBuffer := nil;
      w := GetWidth;
      h := GetHeight;
      d := GetDepth;

      for face := 0 to maxFace do
      begin
        if fCubeMap then
          glTarget := face + GL_TEXTURE_CUBE_MAP_POSITIVE_X;
        for level := 0 to FLevelCount - 1 do
        begin
          if bCompressed then
          begin

            if NV_texture_compression_vtc and (d > 1) and not fTextureArray then
            begin
              if level = 0 then
                GetMem(vtcBuffer, GetLevelSizeInByte(0));
              GetCompressedTexImage(glTarget, level, vtcBuffer);
              // Shufle blocks from VTC to S3TC
              cw := (w + 3) div 4;
              ch := (h + 3) div 4;
              top := lData;
              for k := 0 to d - 1 do
                for i := 0 to ch - 1 do
                  for j := 0 to cw - 1 do
                  begin
                    bottom := vtcBuffer;
                    Inc(bottom, blockOffset(j, i, k));
                    Move(bottom^, top^, fElementSize);
                    Inc(top, fElementSize);
                  end;
              Div2(w);
              Div2(h);
              Div2(d);
            end
            else
              GetCompressedTexImage(glTarget, level, lData);
          end
          else
            GetTexImage(glTarget, level, fColorFormat, fDataType, lData);

          Inc(lData, GetLevelSizeInByte(level));
        end; // for level
      end; // for face
      if Assigned(vtcBuffer) then
        FreeMem(vtcBuffer);
      // Check memory corruption
      ReallocMem(FData, DataSize);
    end;

    if Self is TGLImage then
    begin
      TGLImage(Self).FBlank := FLevelCount = 0;
      if FLevelCount = 0 then
      begin
        UnMipmap;
        FreeMem(fData);
        fData := nil;
      end;
    end;

    GL.CheckError;
    Result := True;

  finally
    LContext.Deactivate;
  end;
end;

procedure TGLBaseImage.SaveHeader;
var
  Temp: Integer;
  LStream: TStream;
begin
  Temp := 0;
  LStream := nil;
  try
    LStream := CreateFileStream(ResourceName, fmOpenWrite or fmCreate);
    with LStream do
    begin
      Write(Temp, SizeOf(Integer)); // Version
      Write(FLOD[0].Width, SizeOf(Integer));
      Write(FLOD[0].Height, SizeOf(Integer));
      Write(FLOD[0].Depth, SizeOf(Integer));
      Write(fColorFormat, SizeOf(GLenum));
      Temp := Integer(fInternalFormat);
      Write(Temp, SizeOf(Integer));
      Write(fDataType, SizeOf(GLenum));
      Write(fElementSize, SizeOf(Integer));
      Write(fLevelCount, SizeOf(TGLImageLODRange));
      Temp := Integer(fCubeMap);
      Write(Temp, SizeOf(Integer));
      Temp := Integer(fTextureArray);
      Write(Temp, SizeOf(Integer));
    end;
  finally
    LStream.Free;
  end;
end;

procedure TGLBaseImage.LoadHeader;
var
  Temp: Integer;
  LStream: TStream;
begin
  LStream := nil;
  try
    LStream := CreateFileStream(ResourceName, fmOpenRead);
    with LStream do
    begin
      Read(Temp, SizeOf(Integer)); // Version
      if Temp > 0 then
      begin
        GLSLogger.LogError(Format(glsUnknownArchive, [Self.ClassType, Temp]));
        Abort;
      end;
      Read(FLOD[0].Width, SizeOf(Integer));
      Read(FLOD[0].Height, SizeOf(Integer));
      Read(FLOD[0].Depth, SizeOf(Integer));
      Read(fColorFormat, SizeOf(GLenum));
      Read(Temp, SizeOf(Integer));
      fInternalFormat := TGLInternalFormat(Temp);
      Read(fDataType, SizeOf(GLenum));
      Read(fElementSize, SizeOf(Integer));
      Read(fLevelCount, SizeOf(TGLImageLODRange));
      Read(Temp, SizeOf(Integer));
      fCubeMap := Boolean(Temp);
      Read(Temp, SizeOf(Integer));
      fTextureArray := Boolean(Temp);
      UpdateLevelsInfo;
    end;
  finally
    LStream.Free;
  end;
end;

var
  vGlobalStreamingTaskCounter: Integer = 0;

procedure TGLBaseImage.StartStreaming;
var
  level: TGLImageLODRange;
begin
  FStreamLevel := fLevelCount - 1;
  for level := 0 to High(TGLImageLODRange) do
    FLOD[level].State := ssKeeping;
end;

procedure TGLBaseImage.DoStreaming;
begin
{$IFDEF GLS_SERVICE_CONTEXT}
  if Assigned(FFinishEvent) then
  begin
    if FFinishEvent.WaitFor(0) <> wrSignaled then
      exit;
  end
  else
    FFinishEvent := TFinishTaskEvent.Create;

  Inc(vGlobalStreamingTaskCounter);
  AddTaskForServiceContext(ImageStreamingTask, FFinishEvent);
{$ENDIF}
end;

{$IFDEF GLS_SERVICE_CONTEXT}
procedure TGLBaseImage.ImageStreamingTask;
var
  readSize: Integer;
  ptr: PByte;
begin
  with FLOD[FStreamLevel] do
  begin
    if PBO = nil then
      PBO := TGLUnpackPBOHandle.Create;

    PBO.AllocateHandle;
    if PBO.IsDataNeedUpdate then
    begin
      { This may work with multiple unshared context, but never tested
        because unlikely. }
      PBO.BindBufferData(nil, MaxInteger(Size, 1024), GL_STREAM_DRAW);
      if Assigned(MapAddress) then
        if not PBO.UnmapBuffer then
          exit;
      MapAddress := PBO.MapBuffer(GL_WRITE_ONLY);
      StreamOffset := 0;
      PBO.UnBind;
      PBO.NotifyDataUpdated;
    end;

    if FSourceStream = nil then
    begin
      FSourceStream := CreateFileStream(ResourceName + IntToHex(FStreamLevel, 2));
    end;

    // Move to position of next piece and read it
    readSize := MinInteger(Cardinal(8192 div vGlobalStreamingTaskCounter),
      Cardinal(Size - StreamOffset));
    if readSize > 0 then
    begin
      ptr := PByte(MapAddress);
      Inc(ptr, StreamOffset);
      FSourceStream.Read(ptr^, readSize);
      Inc(StreamOffset, readSize);
    end;

    Dec(vGlobalStreamingTaskCounter);

    if StreamOffset >= Size then
    begin
      PBO.Bind;
      if PBO.UnmapBuffer then
        State := ssLoaded;
      PBO.UnBind;
      if State <> ssLoaded then
        exit; // Can't unmap
      MapAddress := nil;
      StreamOffset := 0;
      if FStreamLevel > 0 then
        Dec(FStreamLevel);
      FSourceStream.Destroy;
      FSourceStream := nil;
    end;
  end;
end;
{$ENDIF}

{$IFDEF GLS_REGIONS}{$ENDREGION}{$ENDIF}

// ------------------
// ------------------ TGLImage ------------------
// ------------------

{$IFDEF GLS_REGIONS}{$REGION 'TGLImage'}{$ENDIF}
// Create
//

constructor TGLImage.Create;
begin
  inherited Create;
  SetBlank(false);
end;

// Destroy
//

destructor TGLImage.Destroy;
begin
  inherited Destroy;
end;

 
//

procedure TGLImage.Assign(Source: TPersistent);
var
  bmp: TGLBitmap;
  graphic: TGLGraphic;
begin
  if (Source is TGLImage) or (Source is TGLBaseImage) then
  begin
    if Source is TGLImage then
      FBlank := TGLImage(Source).fBlank
    else
      FBlank := false;

    if not FBlank then
      inherited
    else
    begin
      FLOD := TGLImage(Source).FLOD;
      FLevelCount := TGLImage(Source).FLevelCount;
      fCubeMap := TGLImage(Source).fCubeMap;
      fColorFormat := TGLImage(Source).fColorFormat;
      fInternalFormat := TGLImage(Source).fInternalFormat;
      fDataType := TGLImage(Source).fDataType;
      fElementSize := TGLImage(Source).fElementSize;
      fTextureArray := TGLImage(Source).fTextureArray;
    end;
  end
  else if Source is TGLGraphic then
  begin
    if (Source is TGLBitmap) and (((TGLBitmap(Source).Width and 3) = 0) or GL.EXT_bgra) then
    begin
      // in FPC this is pixel wize and reads all pixelFormats.
      AssignFrom24BitsBitmap(TGLBitmap(Source));
    end
    else
    begin
      graphic := TGLGraphic(Source);
      bmp := TGLBitmap.Create;
      try
        // crossbuilder: useless to set pixelformat before setting the size ?
        //               or maybe just useless at all on gtk .. as soon as
        //               bmp.canvas is touched, it's the pixelformat of the device
        //               no matter what was adjusted before ??
        // bmp.PixelFormat:=glpf24bit;
        // bmp.Height:=graphic.Height;
        // crossbuilder: using setsize because setting width or height while
        // the other one is zero results in not setting with/hight
        bmp.PixelFormat := glpf24bit;
        bmp.Height := graphic.Height;
        if (graphic.Width and 3) = 0 then
        begin
          bmp.Width := graphic.Width;
          bmp.Canvas.Draw(0, 0, graphic);
        end
        else
        begin
          bmp.Width := (graphic.Width and $FFFC) + 4;
          bmp.Canvas.StretchDraw(Rect(0, 0, bmp.Width, bmp.Height), graphic);
        end;
        AssignFrom24BitsBitmap(bmp);
      finally
        bmp.Free;
      end;
    end;
  end
  else
    inherited;
end;

// AssignFrom24BitsBitmap
//

procedure TGLImage.AssignFrom24BitsBitmap(aBitmap: TGLBitmap);
var
  y, x: Integer;
  pDest: PByte;
  pixel: TFPColor;
  IntfImg: TLazIntfImage;
begin
  UnMipmap;
  FLOD[0].Width := aBitmap.Width;
  FLOD[0].Height := aBitmap.Height;
  FLOD[0].Depth := 0;
  FLevelCount := 1;
  fColorFormat := GL_RGBA;
  fInternalFormat := tfRGBA8;
  fDataType := GL_UNSIGNED_BYTE;
  fElementSize := 4;
  fCubeMap := false;
  fTextureArray := false;
  ReallocMem(FData, DataSize);

  IntfImg := aBitmap.CreateIntfImage;
  try
    pDest := PByte(FData);
    if not (VerticalReverseOnAssignFromBitmap) then
    begin
      for y := GetHeight - 1 downto 0 do
        for x := 0 to GetWidth - 1 do
        begin
          pixel := IntfImg.Colors[x, y];
          pDest^ := pixel.red shr 8;
          inc(pDest);
          pDest^ := pixel.green shr 8;
          inc(pDest);
          pDest^ := pixel.blue shr 8;
          inc(pDest);
          pDest^ := pixel.alpha shr 8;
          inc(pDest);
        end;
    end
    else
    begin
      for y := 0 to GetHeight - 1 do
        for x := 0 to GetWidth - 1 do
        begin
          pixel := IntfImg.Colors[x, y];
          pDest^ := pixel.red shr 8;
          inc(pDest);
          pDest^ := pixel.green shr 8;
          inc(pDest);
          pDest^ := pixel.blue shr 8;
          inc(pDest);
          pDest^ := pixel.alpha shr 8;
          inc(pDest);
        end;
    end;
  finally
    IntfImg.Free;
  end;
end;

// AssignFromBitmap24WithoutRGBSwap
//

procedure TGLImage.AssignFromBitmap24WithoutRGBSwap(aBitmap: TGLBitmap);
var
  y: Integer;
  rowOffset: Int64;
  pSrc, pDest: PAnsiChar;
begin
  Assert(aBitmap.PixelFormat = glpf24bit);
  Assert((aBitmap.Width and 3) = 0);
  UnMipmap;
  FLOD[0].Width := aBitmap.Width;
  FLOD[0].Height := aBitmap.Height;
  FLOD[0].Depth := 0;
  fColorFormat := GL_RGBA;
  fInternalFormat := tfRGBA8;
  fDataType := GL_UNSIGNED_BYTE;
  fElementSize := 4;
  fCubeMap := false;
  fTextureArray := false;
  ReallocMem(FData, DataSize);
  FBlank := false;
  if Height > 0 then
  begin
    pDest := @PAnsiChar(FData)[Width * 4 * (Height - 1)];
    if Height = 1 then
    begin
      RGB24ToRGBA32(BitmapScanLine(aBitmap, 0), pDest, GetWidth);
    end
    else
    begin
      if VerticalReverseOnAssignFromBitmap then
      begin
        pSrc := BitmapScanLine(aBitmap, GetHeight - 1);
        rowOffset := PtrUInt(BitmapScanLine(aBitmap, GetHeight - 2));
        Dec(rowOffset, PtrUInt(pSrc));
      end
      else
      begin
        pSrc := BitmapScanLine(aBitmap, 0);
        rowOffset := PtrUInt(BitmapScanLine(aBitmap, 1));
        Dec(rowOffset, PtrUInt(pSrc));
      end;
      for y := 0 to Height - 1 do
      begin
        RGB24ToRGBA32(pSrc, pDest, GetWidth);
        Dec(pDest, GetWidth * 4);
        Inc(pSrc, rowOffset);
      end;
    end;
  end;
end;

// AssignFrom32BitsBitmap
//

procedure TGLImage.AssignFrom32BitsBitmap(aBitmap: TGLBitmap);
var
  y: Integer;
  rowOffset: Int64;
  pSrc, pDest: PAnsiChar;
begin
  Assert(aBitmap.PixelFormat = glpf32bit);
  UnMipmap;
  FLOD[0].Width := aBitmap.Width;
  FLOD[0].Height := aBitmap.Height;
  FLOD[0].Depth := 0;
  if GL.EXT_bgra then
    fColorFormat := GL_BGRA
  else
  begin
    Assert((aBitmap.Width and 3) = 0);
    fColorFormat := GL_RGBA;
  end;
  fInternalFormat := tfRGBA8;
  fDataType := GL_UNSIGNED_BYTE;
  fElementSize := 4;
  fCubeMap := false;
  fTextureArray := false;
  ReallocMem(FData, DataSize);
  FBlank := false;
  if Height > 0 then
  begin
    pDest := @PAnsiChar(FData)[Width * 4 * (Height - 1)];
    if VerticalReverseOnAssignFromBitmap then
    begin
      pSrc := BitmapScanLine(aBitmap, Height - 1);
      if Height > 1 then
      begin
        rowOffset := PtrUInt(BitmapScanLine(aBitmap, Height - 2));
        Dec(rowOffset, PtrUInt(pSrc));
      end
      else
        rowOffset := 0;
    end
    else
    begin
      pSrc := BitmapScanLine(aBitmap, 0);
      if Height > 1 then
      begin
        rowOffset := PtrUInt(BitmapScanLine(aBitmap, 1));
        Dec(rowOffset, PtrUInt(pSrc));
      end
      else
        rowOffset := 0;
    end;
    if GL.EXT_bgra then
    begin
      for y := 0 to Height - 1 do
      begin
        Move(pSrc^, pDest^, Width * 4);
        Dec(pDest, Width * 4);
        Inc(pSrc, rowOffset);
      end;
    end
    else
    begin
      for y := 0 to Height - 1 do
      begin
        BGRA32ToRGBA32(pSrc, pDest, Width);
        Dec(pDest, Width * 4);
        Inc(pSrc, rowOffset);
      end;
    end;
  end;
end;


{$IFDEF GLS_PngImage_SUPPORT}
// AlphaChannel Support

procedure TGLImage.AssignFromPngImage(aPngImage: TPngImage);
var
  i, j: Integer;
  SourceScan: PRGBLine;
  DestScan: PGLPixel32Array;
{$IFDEF GLS_DELPHI_XE2_UP}
  AlphaScan: VCL.Imaging.Pngimage.pByteArray;
{$ELSE}
  AlphaScan: Pngimage.pByteArray;
{$ENDIF}
  Pixel: Integer;
begin
{$IFDEF GLS_PngImage_RESIZENEAREST}
  if (aPngImage.Width and 3) > 0 then
    aPngImage.Resize((aPngImage.Width and $FFFC) + 4, aPngImage.Height);
{$ENDIF}
  UnMipmap;
  FLOD[0].Width := aPngImage.Width;
  FLOD[0].Height := aPngImage.Height;
  FLOD[0].Depth := 0;
  fColorFormat := GL_RGBA;
  fInternalFormat := tfRGBA8;
  fDataType := GL_UNSIGNED_BYTE;
  fElementSize := 4;
  fCubeMap := false;
  fTextureArray := false;
  ReallocMem(FData, DataSize);
  FBlank := False;
  case aPngImage.Header.ColorType of
    { Direct ScanLine (24 Bits) }
    COLOR_RGB, COLOR_RGBALPHA: for j := 1 to aPngImage.Height do
      begin
        SourceScan := aPngImage.Scanline[aPngImage.Height - j];
        AlphaScan := aPngImage.AlphaScanline[aPngImage.Height - j];
        DestScan := ScanLine[Pred(j)];
        for i := 0 to Pred(aPngImage.Width) do
        begin
          DestScan^[i].r := SourceScan^[i].rgbtRed;
          DestScan^[i].g := SourceScan^[i].rgbtGreen;
          DestScan^[i].b := SourceScan^[i].rgbtBlue;
          if Assigned(AlphaScan) then
            DestScan^[i].a := AlphaScan^[i]
          else
            DestScan^[i].a := $FF;
        end;
      end;
  else
    { Internal Decode TColor - Palette }
    for j := 1 to aPngImage.Height do
    begin
      AlphaScan := aPngImage.AlphaScanline[aPngImage.Height - j];
      DestScan := ScanLine[Pred(j)];
      for i := 0 to Pred(aPngImage.Width) do
      begin
        Pixel := aPngImage.Pixels[i, aPngImage.Height - j];
        DestScan^[i].r := Pixel and $FF;
        DestScan^[i].g := (Pixel shr 8) and $FF;
        DestScan^[i].b := (Pixel shr 16) and $FF;
        if Assigned(AlphaScan) then
          DestScan^[i].a := AlphaScan^[i]
        else
          DestScan^[i].a := $FF;
      end;
    end;
  end;
end;
{$ENDIF}

// AssignFromTexture2D
//

procedure TGLImage.AssignFromTexture2D(textureHandle: Cardinal);
var
  oldTex: Cardinal;
begin
  UnMipmap;

  with CurrentGLContext.GLStates do
  begin
    oldTex := TextureBinding[ActiveTexture, ttTexture2D];
    TextureBinding[ActiveTexture, ttTexture2D] := textureHandle;

    GL.GetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, @FLOD[0].Width);
    GL.GetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, @FLOD[0].Height);
    FLOD[0].Depth := 0;
    fColorFormat := GL_RGBA;
    fInternalFormat := tfRGBA8;
    fDataType := GL_UNSIGNED_BYTE;
    fElementSize := 4;
    fCubeMap := false;
    fTextureArray := false;
    ReallocMem(FData, DataSize);
    GL.GetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, FData);
    FBlank := false;

    TextureBinding[ActiveTexture, ttTexture2D] := oldTex;
  end;
end;

// AssignFromTexture2D
//

procedure TGLImage.AssignFromTexture2D(textureHandle: TGLTextureHandle);
var
  oldContext: TGLContext;
  contextActivate: Boolean;
begin
  if Assigned(textureHandle) and (textureHandle.Handle <> 0) then
  begin
    oldContext := CurrentGLContext;
    contextActivate := (oldContext <> textureHandle.RenderingContext);
    if contextActivate then
    begin
      if Assigned(oldContext) then
        oldContext.Deactivate;
      textureHandle.RenderingContext.Activate;
    end;

    try
      AssignFromTexture2D(textureHandle.Handle);
    finally
      if contextActivate then
      begin
        textureHandle.RenderingContext.Deactivate;
        if Assigned(oldContext) then
          oldContext.Activate;
      end;
    end;
  end
  else
  begin
    // Make image empty
    UnMipmap;
    FLOD[0].Width := 0;
    FLOD[0].Height := 0;
    FLOD[0].Depth := 0;
    fColorFormat := GL_RGBA;
    fInternalFormat := tfRGBA8;
    fDataType := GL_UNSIGNED_BYTE;
    fElementSize := 4;
    fCubeMap := false;
    fTextureArray := false;
    ReallocMem(FData, DataSize);
  end;
end;

// Create32BitsBitmap
//

function TGLImage.Create32BitsBitmap: TGLBitmap;
var
  RIMG: TRawImage;
begin
  if FBlank then
  begin
    Result := nil;
    exit;
  end;
  Narrow;

  Result := TGLBitmap.Create;
  Result.PixelFormat := glpf32bit;
  Result.Width := Width;
  Result.Height := Height;

  if Height > 0 then
  begin

    RIMG.Init;
    rimg.Description.Init_BPP32_B8G8R8A8_BIO_TTB(Width, Height);
    rimg.Description.RedShift := 0;
    rimg.Description.BlueShift := 16;
    rimg.Description.LineOrder := riloBottomToTop;
    RIMG.DataSize := DataSize;
    rimg.Data := PByte(FData);
    { READ THIS PLEASE: !!!!!!!!!!!!  }
    { If you get a compile time error on the following line, you could either
      A) update your lazarus to >= 0.9.27-r18329, or
      B) comment the next line and un-comment the "Workaround for older Lazarus "
         part after the next line }

    result.LoadFromRawImage(rimg, false);

    { -- "Workaround for older Lazarus "
        LIntfImg:=TLazIntfImage.Create(rimg,False);
        try
          result.LoadFromIntfImage(LIntfImg);
        finally
          FreeAndNil(LIntfImg);
        end;
      -- End of "Workaround for older Lazarus " }

  end;
end;

// SetWidth
//

procedure TGLImage.SetWidth(val: Integer);
begin
  if val <> FLOD[0].Width then
  begin
    Assert(val >= 0);
    FLOD[0].Width := val;
    FBlank := true;
  end;
end;

// SetHeight
//

procedure TGLImage.SetHeight(const val: Integer);
begin
  if val <> FLOD[0].Height then
  begin
    Assert(val >= 0);
    FLOD[0].Height := val;
    FBlank := true;
  end;
end;

// SetDepth
//

procedure TGLImage.SetDepth(const val: Integer);
begin
  if val <> FLOD[0].Depth then
  begin
    Assert(val >= 0);
    FLOD[0].Depth := val;
    FBlank := true;
  end;
end;

// SetCubeMap
//

procedure TGLImage.SetCubeMap(const val: Boolean);
begin
  if val <> fCubeMap then
  begin
    fCubeMap := val;
    FBlank := true;
  end;
end;

// SetArray
//

procedure TGLImage.SetArray(const val: Boolean);
begin
  if val <> fTextureArray then
  begin
    fTextureArray := val;
    FBlank := true;
  end;
end;

// SetColorFormatDataType
//

procedure TGLImage.SetColorFormatDataType(const AColorFormat, ADataType: GLenum);
begin
  if fBlank then
  begin
    fDataType := ADataType;
    fColorFormat := AColorFormat;
    exit;
  end;
  fOldDataType := fDataType;
  fOldColorFormat := fColorFormat;
  fDataType := ADataType;
  fColorFormat := AColorFormat;
  fElementSize := GetTextureElementSize(fColorFormat, fDataType);
  DataConvertTask;
end;

// GetScanLine
//

function TGLImage.GetScanLine(index: Integer): PGLPixel32Array;
begin
  Narrow;
  Result := PGLPixel32Array(@FData[index * GetWidth]);
end;

// SetAlphaFromIntensity
//

procedure TGLImage.SetAlphaFromIntensity;
var
  i: Integer;
begin
  Narrow;
  for i := 0 to (DataSize div 4) - 1 do
    with FData^[i] do
      a := (Integer(r) + Integer(g) + Integer(b)) div 3;
end;

// SetAlphaTransparentForColor
//

procedure TGLImage.SetAlphaTransparentForColor(const aColor: TColor);
var
  color: TGLPixel24;
begin
  color.r := GetRValue(aColor);
  color.g := GetGValue(aColor);
  color.b := GetBValue(aColor);
  SetAlphaTransparentForColor(color);
end;

// SetAlphaTransparentForColor
//

procedure TGLImage.SetAlphaTransparentForColor(const aColor: TGLPixel32);
var
  color: TGLPixel24;
begin
  color.r := aColor.r;
  color.g := aColor.g;
  color.b := aColor.b;
  SetAlphaTransparentForColor(color);
end;

// SetAlphaTransparentForColor
//

procedure TGLImage.SetAlphaTransparentForColor(const aColor: TGLPixel24);
var
  i: Integer;
  intCol: Integer;
begin
  Narrow;
  intCol := (PInteger(@aColor)^) and $FFFFFF;
  for i := 0 to (DataSize div 4) - 1 do
    if PInteger(@FData[i])^ and $FFFFFF = intCol then
      FData^[i].a := 0
    else
      FData^[i].a := 255;
end;

// SetAlphaToValue
//

procedure TGLImage.SetAlphaToValue(const aValue: Byte);
var
  i: Integer;
begin
  Narrow;
  for i := 0 to (DataSize div 4) - 1 do
    FData^[i].a := aValue
end;

// SetAlphaToFloatValue
//

procedure TGLImage.SetAlphaToFloatValue(const aValue: Single);
begin
  SetAlphaToValue(Byte(Trunc(aValue * 255) and 255));
end;

// InvertAlpha
//

procedure TGLImage.InvertAlpha;
var
  i: Integer;
begin
  Narrow;
  for i := (DataSize div 4) - 1 downto 0 do
    FData^[i].a := 255 - FData^[i].a;
end;

// SqrtAlpha
//

procedure TGLImage.SqrtAlpha;
var
  i: Integer;
  sqrt255Array: PSqrt255Array;
begin
  Narrow;
  sqrt255Array := GetSqrt255Array;
  for i := 0 to (DataSize div 4) - 1 do
    with FData^[i] do
      a := sqrt255Array^[(Integer(r) + Integer(g) + Integer(b)) div 3];
end;

// BrightnessCorrection
//

procedure TGLImage.BrightnessCorrection(const factor: Single);
begin
  if Assigned(FData) then
  begin
    Narrow;
    BrightenRGBAArray(Data, DataSize div 4, factor);
  end;
end;

// GammaCorrection
//

procedure TGLImage.GammaCorrection(const gamma: Single);
begin
  if Assigned(FData) then
  begin
    Narrow;
    GammaCorrectRGBAArray(Data, DataSize div 4, gamma);
  end;
end;

// DownSampleByFactor2
//

procedure TGLImage.DownSampleByFactor2;
type
  T2Pixel32 = packed array[0..1] of TGLPixel32;
  P2Pixel32 = ^T2Pixel32;

{$IFNDEF GEOMETRY_NO_ASM}
  procedure ProcessRow3DNow(pDest: PGLPixel32; pLineA, pLineB: P2Pixel32; n:
    Integer);
  asm     // 3DNow! version 30% faster
      db $0F,$EF,$C0           /// pxor        mm0, mm0          // set mm0 to [0, 0, 0, 0]

@@Loop:
      db $0F,$0D,$81,$00,$01,$00,$00/// prefetch    [ecx+256]

      db $0F,$6F,$0A           /// movq        mm1, [edx]
      db $0F,$6F,$11           /// movq        mm2, [ecx]

      db $0F,$6F,$D9           /// movq        mm3, mm1
      db $0F,$6F,$E2           /// movq        mm4, mm2

      db $0F,$60,$C8           /// punpcklbw   mm1, mm0          // promote to 16 bits and add LineA pixels
      db $0F,$68,$D8           /// punpckhbw   mm3, mm0
      db $0F,$FD,$CB           /// paddw       mm1, mm3

      db $0F,$60,$D0           /// punpcklbw   mm2, mm0          // promote to 16 bits and add LineB pixels
      db $0F,$68,$E0           /// punpckhbw   mm4, mm0
      db $0F,$FD,$D4           /// paddw       mm2, mm4

      db $0F,$FD,$CA           /// paddw       mm1, mm2          // add LineA and LineB pixels

      db $0F,$71,$D1,$02       /// psrlw       mm1, 2            // divide by 4
      db $0F,$67,$C9           /// packuswb    mm1, mm1          // reduce to 8 bits and store point
      db $0F,$7E,$08           /// movd        [eax], mm1

      add         edx, 8
      add         ecx, 8
      add         eax, 4

      dec         [n]
      jnz         @@Loop

      db $0F,$0E               /// femms
  end;
{$ENDIF}

  procedure ProcessRowPascal(pDest: PGLPixel32; pLineA, pLineB: P2Pixel32; n:
    Integer);
  var
    i: Integer;
  begin
    for i := 0 to n - 1 do
    begin
      pDest^.r := (pLineA^[0].r + pLineA^[1].r + pLineB^[0].r + pLineB^[1].r)
        shr
        2;
      pDest^.g := (pLineA^[0].g + pLineA^[1].g + pLineB^[0].g + pLineB^[1].g)
        shr
        2;
      pDest^.b := (pLineA^[0].b + pLineA^[1].b + pLineB^[0].b + pLineB^[1].b)
        shr
        2;
      pDest^.a := (pLineA^[0].a + pLineA^[1].a + pLineB^[0].a + pLineB^[1].a)
        shr
        2;
      Inc(pLineA);
      Inc(pLineB);
      Inc(pDest);
    end;
  end; // }

var
  y, w2, h2: Integer;
  pDest: PGLPixel32;
  pLineA, pLineB: P2Pixel32;
begin
  if (GetWidth <= 1) or (GetHeight <= 1) then
    Exit;
  Narrow;
  w2 := GetWidth shr 1;
  h2 := GetHeight shr 1;
  pDest := @FData[0];
  pLineA := @FData[0];
  pLineB := @FData[Width];
{$IFNDEF GEOMETRY_NO_ASM}
  if vSIMD = 1 then
  begin
    for y := 0 to h2 - 1 do
    begin
      ProcessRow3DNow(pDest, pLineA, pLineB, w2);
      Inc(pDest, w2);
      Inc(pLineA, Width);
      Inc(pLineB, Width);
    end;
  end
  else
{$ENDIF}
  begin
    for y := 0 to h2 - 1 do
    begin
      ProcessRowPascal(pDest, pLineA, pLineB, w2);
      Inc(pDest, w2);
      Inc(pLineA, Width);
      Inc(pLineB, Width);
    end;
  end;
  FLOD[0].Width := w2;
  FLOD[0].Height := h2;
  ReallocMem(FData, DataSize);
end;

// ReadPixels
//

procedure TGLImage.ReadPixels(const area: TGLRect);
begin
  UnMipmap;
  FLOD[0].Width := (area.Right - area.Left) and $FFFC;
  FLOD[0].Height := (area.Bottom - area.Top);
  FLOD[0].Depth := 0;
  fColorFormat := GL_RGBA;
  fInternalFormat := tfRGBA8;
  fDataType := GL_UNSIGNED_BYTE;
  fElementSize := 4;
  fCubeMap := false;
  fTextureArray := false;
  fBlank := false;
  ReallocMem(FData, DataSize);
  GL.ReadPixels(0, 0, GetWidth, GetHeight, GL_RGBA, GL_UNSIGNED_BYTE, FData);
end;

// DrawPixels
//

procedure TGLImage.DrawPixels(const x, y: Single);
begin
  if fBlank or IsEmpty then
    Exit;
  Assert(not CurrentGLContext.GLStates.ForwardContext);
  GL.RasterPos2f(x, y);
  GL.DrawPixels(Width, Height, fColorFormat, fDataType, FData);
end;

// TGLImage
//

procedure TGLImage.GrayScaleToNormalMap(const scale: Single;
  wrapX: Boolean = True; wrapY: Boolean = True);
var
  x, y: Integer;
  dcx, dcy: Single;
  invLen: Single;
  maskX, maskY: Integer;
  curRow, nextRow, prevRow: PGLPixel32Array;
  normalMapBuffer: PGLPixel32Array;
  p: PGLPixel32;
begin
  if Assigned(FData) then
  begin
    Narrow;
    GetMem(normalMapBuffer, DataSize);
    try
      maskX := Width - 1;
      maskY := Height - 1;
      p := @normalMapBuffer[0];
      for y := 0 to Height - 1 do
      begin
        curRow := GetScanLine(y);
        if wrapY then
        begin
          prevRow := GetScanLine((y - 1) and maskY);
          nextRow := GetScanLine((y + 1) and maskY);
        end
        else
        begin
          if y > 0 then
            prevRow := GetScanLine(y - 1)
          else
            prevRow := curRow;
          if y < Height - 1 then
            nextRow := GetScanLine(y + 1)
          else
            nextRow := curRow;
        end;
        for x := 0 to Width - 1 do
        begin
          if wrapX then
            dcx := scale * (curRow^[(x - 1) and maskX].g - curRow^[(x + 1) and
              maskX].g)
          else
          begin
            if x = 0 then
              dcx := scale * (curRow^[x].g - curRow^[x + 1].g)
            else if x < Width - 1 then
              dcx := scale * (curRow^[x - 1].g - curRow^[x].g)
            else
              dcx := scale * (curRow^[x - 1].g - curRow^[x + 1].g);
          end;
          dcy := scale * (prevRow^[x].g - nextRow^[x].g);
          invLen := 127 * RSqrt(dcx * dcx + dcy * dcy + 1);
          with p^ do
          begin
            r := Integer(Round(128 + ClampValue(dcx * invLen, -128, 127)));
            g := Integer(Round(128 + ClampValue(dcy * invLen, -128, 127)));
            b := Integer(Round(128 + ClampValue(invLen, -128, 127)));
            a := 255;
          end;
          Inc(p);
        end;
      end;
      Move(normalMapBuffer^, FData^, DataSize);
    finally
      FreeMem(normalMapBuffer);
    end;
  end;
end;

// NormalizeNormalMap
//

procedure TGLImage.NormalizeNormalMap;
var
  x, y: Integer;
  sr, sg, sb: Single;
  invLen: Single;
  curRow: PGLPixel32Array;
  p: PGLPixel32;
const
  cInv128: Single = 1 / 128;
begin
  if not IsEmpty and not Blank then
  begin
    Narrow;
    for y := 0 to Height - 1 do
    begin
      curRow := @FData[y * GetWidth];
      for x := 0 to GetWidth - 1 do
      begin
        p := @curRow[x];
        sr := (p^.r - 128) * cInv128;
        sg := (p^.g - 128) * cInv128;
        sb := (p^.b - 128) * cInv128;
        invLen := RSqrt(sr * sr + sg * sg + sb * sb);
        p^.r := Round(128 + 127 * ClampValue(sr * invLen, -1, 1));
        p^.g := Round(128 + 127 * ClampValue(sg * invLen, -1, 1));
        p^.b := Round(128 + 127 * ClampValue(sb * invLen, -1, 1));
      end;
    end;
  end;
end;

procedure TGLImage.SetBlank(const Value: boolean);
begin
  if not Value and not IsEmpty then
    ReallocMem(FData, DataSize);
  FBlank := Value;
end;

//Converts a TGLImage back into a TBitmap
//

procedure TGLImage.AssignToBitmap(aBitmap: TGLBitmap); //TGLBitmap = TBitmap
var
  y: integer;
  pSrc, pDest: PAnsiChar;
begin
  Narrow;
  aBitmap.Width := GetWidth;
  aBitmap.Height := GetHeight;
  aBitmap.PixelFormat := glpf32bit;
{$IFDEF FPC}
{$WARNING Crossbuilder: cvs version uses the above line instead of the following, but our TBitmap doesn't support Scanline}
{$NOTE BitmapScanline will generate an Assertion in FPC }
{$ENDIF}
  if FVerticalReverseOnAssignFromBitmap then
  begin
    for y := 0 to GetHeight - 1 do
    begin
      pSrc := @PAnsiChar(FData)[y * (GetWidth * 4)];
      pDest := BitmapScanLine(aBitmap, y);
      BGRA32ToRGBA32(pSrc, pDest, GetWidth);
    end;
  end
  else
  begin
    for y := 0 to GetHeight - 1 do
    begin
      pSrc := @PAnsiChar(FData)[y * (GetWidth * 4)];
      pDest := BitmapScanLine(aBitmap, GetHeight - 1 - y);
      BGRA32ToRGBA32(pSrc, pDest, GetWidth);
    end;
  end;
end;

// GenerateMipmap
//

procedure TGLImage.GenerateMipmap(AFilter: TImageFilterFunction);
begin
  if not FBlank then
    inherited GenerateMipmap(AFilter);
end;

// UnMipmap
//

procedure TGLImage.UnMipmap;
begin
  inherited UnMipmap;
  if not (fBlank or IsEmpty) then
    ReallocMem(FData, DataSize);
end;

procedure TGLImage.DataConvertTask;
var
  oldLOD: TGLImagePiramid;
  newData: Pointer;
  ptr: PByte;
  L: TGLImageLODRange;
  d: Integer;
begin
  oldLOD := FLOD;
  if IsVolume then
  begin
    {Message Hint 'TGLImage.DataConvertTask not yet implemented for volume images' }
  end
  else
  begin
    GetMem(newData, DataSize);
    d := MaxInteger(GetDepth, 1);

    try
      for L := FLevelCount - 1 downto 0 do
      begin
        ptr := newData;
        Inc(ptr, oldLOD[L].Offset);
        ConvertImage(
          GetLevelAddress(L), ptr,
          fOldColorFormat, fColorFormat,
          fOldDataType, fDataType,
          oldLOD[L].Width, oldLOD[L].Height * d);
      end;
      FreeMem(fData);
      fData := newData;
    except
      FreeMem(newData);
      GLSLogger.LogError(Format(glsCantConvertImg, [ClassName]));
      SetErrorImage;
    end;
  end;
end;

{$IFDEF GLS_REGIONS}{%ENDREGION}{$ENDIF}

initialization

finalization
  FreeAndNil(vRasterFileFormats);

end.

