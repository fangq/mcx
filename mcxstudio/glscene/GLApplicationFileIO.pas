//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Components and fonction that abstract file I/O access for an application. 
   Allows re-routing file reads to reads from a single archive file f.i.

  History :  
       10/11/12 - PW - Added CPPB compatibility: used TAFIOFileStreamEvent as procedure
                     instead of function for GLS_CPPB
       25/08/10 - DaStr - Fixed compiler warnings
       25/07/10 - Yar - Added TGLSResourceStream class and CreateResourceStream string
       23/01/10 - Yar - Change LoadFromStream to dynamic
       29/01/07 - DaStr - Moved registration to GLSceneRegister.pas
       02/08/04 - LR, YHC - BCB corrections: fixed BCB Compiler error "E2370 Simple type name expected"
       05/06/03 - EG - TGLDataFile moved in from GLMisc
       31/01/03 - EG - Added FileExists mechanism
       21/11/02 - EG - Creation
  
}
unit GLApplicationFileIO;

interface

{$I GLScene.inc}

uses
  Classes,
  SysUtils,
  GLBaseClasses,
  LResources,GLSLog;


const
  GLS_RC_DDS_Type =  'DDS';
  GLS_RC_JPG_Type =  'JPG';
  GLS_RC_XML_Type = 'XML';
  GLS_RC_String_Type = 'STR';

type

  TGLSApplicationResource = (
    aresNone,
    aresSplash,
    aresTexture,
    aresMaterial,
    aresSampler,
    aresFont,
    aresMesh);

  // TAFIOCreateFileStream
  //
  TAFIOCreateFileStream = function(const fileName: string; mode: Word): TStream;

  // TAFIOFileStreamExists
  //
  TAFIOFileStreamExists = function(const fileName: string): Boolean;

  // TAFIOFileStreamEvent
  //
   TAFIOFileStreamEvent = procedure (const fileName : String; mode : Word;var stream : TStream) of object;

  // TAFIOFileStreamExistsEvent
  //
  TAFIOFileStreamExistsEvent = function(const fileName: string): Boolean of object;

  // TGLApplicationFileIO
  //
    { Allows specifying a custom behaviour for GLApplicationFileIO's CreateFileStream.
       The component should be considered a helper only, you can directly specify
       a function via the vAFIOCreateFileStream variable. 
       If multiple TGLApplicationFileIO components exist in the application,
       the last one created will be the active one. }
  TGLApplicationFileIO = class(TComponent)
  private
     
    FOnFileStream: TAFIOFileStreamEvent;
    FOnFileStreamExists: TAFIOFileStreamExistsEvent;

  protected
     

  public
     
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;

  published
     
      { Event that allows you to specify a stream for the file.
         Destruction of the stream is at the discretion of the code that
         invoked CreateFileStream. Return nil to let the default mechanism
         take place (ie. attempt a regular file system access). }
    property OnFileStream: TAFIOFileStreamEvent read FOnFileStream write FOnFileStream;
    { Event that allows you to specify if a stream for the file exists. }
    property OnFileStreamExists: TAFIOFileStreamExistsEvent read FOnFileStreamExists write FOnFileStreamExists;
  end;

  // TGLDataFileCapabilities
  //
  TGLDataFileCapability = (dfcRead, dfcWrite);
  TGLDataFileCapabilities = set of TGLDataFileCapability;

  // TGLDataFile
  //
  { Abstract base class for data file formats interfaces.
     This class declares base file-related behaviours, ie. ability to load/save
     from a file or a stream.
     It is highly recommended to overload ONLY the stream based methods, as the
     file-based one just call these, and stream-based behaviours allow for more
     enhancement (such as other I/O abilities, compression, cacheing, etc.)
     to this class, without the need to rewrite subclasses. }
  TGLDataFile = class(TGLUpdateAbleObject)
  private
    FResourceName: string;
    procedure SetResourceName(const AName: string);
  public
    { Describes what the TGLDataFile is capable of.
       Default value is [dfcRead]. }
    class function Capabilities: TGLDataFileCapabilities; virtual;

    { Duplicates Self and returns a copy.
       Subclasses should override this method to duplicate their data. }
    function CreateCopy(AOwner: TPersistent): TGLDataFile; dynamic;

    procedure LoadFromFile(const fileName: string); dynamic;
    procedure SaveToFile(const fileName: string); dynamic;
    procedure LoadFromStream(stream: TStream); dynamic;
    procedure SaveToStream(stream: TStream); dynamic;
    procedure Initialize; dynamic;
    { Optionnal resource name.
       When using LoadFromFile/SaveToFile, the filename is placed in it,
       when using the Stream variants, the caller may place the resource
       name in it for parser use. }
    property ResourceName: string read FResourceName write SetResourceName;
  end;

  TGLDataFileClass = class of TGLDataFile;
  TGLSResourceStream = {$IFNDEF FPC}TResourceStream{$ELSE}TLazarusResourceStream{$ENDIF};

  // Returns true if an GLApplicationFileIO has been defined
function ApplicationFileIODefined: Boolean;

{ Creates a file stream corresponding to the fileName.
   If the file does not exists, an exception will be triggered. 
   Default mechanism creates a regular TFileStream, the 'mode' parameter
   is similar to the one for TFileStream. }
function CreateFileStream(const fileName: string;
  mode: Word = fmOpenRead + fmShareDenyNone): TStream;
{ Queries is a file stream corresponding to the fileName exists. }
function FileStreamExists(const fileName: string): Boolean;

{ Create a resource stream. }
function CreateResourceStream(const ResName: string; ResType: PChar): TGLSResourceStream;

function StrToGLSResType(const AStrRes: string): TGLSApplicationResource;

var
  vAFIOCreateFileStream: TAFIOCreateFileStream = nil;
  vAFIOFileStreamExists: TAFIOFileStreamExists = nil;

// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
implementation
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------

var
  vAFIO: TGLApplicationFileIO = nil;

function ApplicationFileIODefined: Boolean;
begin
  Result := (Assigned(vAFIOCreateFileStream) and Assigned(vAFIOFileStreamExists))
    or Assigned(vAFIO);
end;

function CreateFileStream(const fileName: string;
  mode: Word = fmOpenRead + fmShareDenyNone): TStream;
begin
  if Assigned(vAFIOCreateFileStream) then
    Result := vAFIOCreateFileStream(fileName, mode)
  else
  begin
      Result:=nil;
      if Assigned(vAFIO) and Assigned(vAFIO.FOnFileStream) then
         vAFIO.FOnFileStream(fileName, mode, Result);
      if not Assigned(Result) then begin
         if ((mode and fmCreate)=fmCreate) or FileExists(fileName) then
            Result:=TFileStream.Create(fileName, mode)
         else raise Exception.Create('File not found: "'+fileName+'"');
      end;
   end;
end;

function FileStreamExists(const fileName: string): Boolean;
begin
  if Assigned(vAFIOFileStreamExists) then
    Result := vAFIOFileStreamExists(fileName)
  else
  begin
    if Assigned(vAFIO) and Assigned(vAFIO.FOnFileStreamExists) then
      Result := vAFIO.FOnFileStreamExists(fileName)
    else
      Result := FileExists(fileName);
  end;
end;

function CreateResourceStream(const ResName: string; ResType: PChar): TGLSResourceStream;

var
  FPResource: TFPResourceHandle;
  function IsResourceExist: Boolean;
  begin
    FPResource := FindResource(HInstance, PChar(ResName), ResType);
    Result := FPResource <> 0;
  end;

begin
  Result := nil;

  if LazarusResources.Find(ResName, ResType) <> nil then
    Result := TLazarusResourceStream.Create(ResName, ResType)

  else if IsResourceExist then
    Result := TLazarusResourceStream.CreateFromHandle(HInstance, FPResource)

  else
    GLSLogger.LogError(Format('Can''t create stream of application resource "%s"', [ResName]));
end;

// ------------------
// ------------------ TGLApplicationFileIO ------------------
// ------------------

constructor TGLApplicationFileIO.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  vAFIO := Self;
end;

destructor TGLApplicationFileIO.Destroy;
begin
  vAFIO := nil;
  inherited Destroy;
end;


// ------------------
// ------------------ TGLDataFile ------------------
// ------------------

class function TGLDataFile.Capabilities: TGLDataFileCapabilities;
begin
  Result := [dfcRead];
end;


function TGLDataFile.CreateCopy(AOwner: TPersistent): TGLDataFile;
begin
  if Self <> nil then
    Result := TGLDataFileClass(Self.ClassType).Create(AOwner)
  else
    Result := nil;
end;

procedure TGLDataFile.LoadFromFile(const fileName: string);
var
  fs: TStream;
begin
  ResourceName := ExtractFileName(fileName);
  fs := CreateFileStream(fileName, fmOpenRead + fmShareDenyNone);
  try
    LoadFromStream(fs);
  finally
    fs.Free;
  end;
end;

procedure TGLDataFile.SaveToFile(const fileName: string);
var
  fs: TStream;
begin
  ResourceName := ExtractFileName(fileName);
  fs := CreateFileStream(fileName, fmCreate);
  try
    SaveToStream(fs);
  finally
    fs.Free;
  end;
end;

procedure TGLDataFile.LoadFromStream(stream: TStream);
begin
  Assert(False, 'Imaport for ' + ClassName + ' to ' + stream.ClassName + ' not available.');
end;

procedure TGLDataFile.SaveToStream(stream: TStream);
begin
  Assert(False, 'Export for ' + ClassName + ' to ' + stream.ClassName + ' not available.');
end;

procedure TGLDataFile.Initialize;
begin
end;

procedure TGLDataFile.SetResourceName(const AName: string);
begin
  FResourceName := AName;
end;

function StrToGLSResType(const AStrRes: string): TGLSApplicationResource;
begin
  if AStrRes = '[SAMPLERS]' then
  begin
    Result := aresSampler;
  end
  else if AStrRes = '[TEXTURES]' then
  begin
    Result := aresTexture;
  end
  else if AStrRes = '[MATERIALS]' then
  begin
    Result := aresMaterial;
  end
  else if AStrRes = '[STATIC MESHES]' then
  begin
    Result := aresMesh;
  end
  else if AStrRes = '[SPLASH]' then
  begin
    Result := aresSplash;
  end
  else if AStrRes = '[FONTS]' then
  begin
    Result := aresFont;
  end
  else
    Result := aresNone;
end;

end.

