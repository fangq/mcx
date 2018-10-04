//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Hacks into the VCL to access the list of TPicture registered TGraphic formats

    History :  
       19/06/11 - Yar - Improved for Lazarus (thanks to Johannes Pretorius, Bugtracker ID = 1586936)
       04/11/10 - DaStr - Restored Delphi5 and Delphi6 compatibility
       19/09/10 - YP - Range check auto disabled in HackTPictureRegisteredFormats
       31/05/10 - Yar - Fixes for Linux x64
       25/01/10 - DaStr - Updated warning about a possible crash while using the
                              'Use Debug DCUs' compiler option (BugTrackerID=1586936)
       10/11/09 - DaStr - Replaced all Delphi2005+ IFDEFs with a single one
       07/11/09 - DaStr - Improved FPC compatibility
                              (BugtrackerID = 2893580) (thanks Predator)
       16/10/08 - UweR - Added IFDEF for Delphi 2009
       06/04/08 - DanB - Change to HackTPictureRegisteredFormats due to Char changing
                            size in Delphi 2009
       06/04/08 - DaStr - Added IFDEFs for Delphi 5 compatibility
       20/12/06 - DaStr - Added a warning about optimization turned off
                             in HackTPictureRegisteredFormats (BugTrackerID=1586936)
       08/03/06 - ur - Added Delphi 2006 support
       28/02/05 - EG - Added BPL support
       24/02/05 - EG - Creation
    
}
unit PictureRegisteredFormats;

interface

{$I GLScene.inc}

uses
  Classes, Graphics,
  GLCrossPlatform;


 {$define PRF_HACK_PASSES} // FPC


{ Returns the TGraphicClass associated to the extension, if any.
   Accepts anExtension with or without the '.' }
function GraphicClassForExtension(const anExtension: string): TGraphicClass;

{ Adds to the passed TStrings the list of registered formats.
   Convention is "extension=description" for the string, the Objects hold
   the corresponding TGraphicClass (extensions do not include the '.'). }
procedure HackTPictureRegisteredFormats(destList: TStrings);

// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

type
  PInteger = ^integer;

// GraphicClassForExtension

function GraphicClassForExtension(const anExtension: string): TGraphicClass;
var

  buf: string;
begin
  Result := nil;
  if anExtension = '' then
    Exit;
  if anExtension[1] = '.' then
    buf := Copy(anExtension, 2, MaxInt)
  else
    buf := anExtension;
  Result := TPicture.Create.FindGraphicClassWithFileExt(buf, False);
end;

type
  PFileFormat = ^TFileFormat;

  TFileFormat = record
    GraphicClass: TGraphicClass;
    Extension: string;
    Description: string;
    DescResID: integer;
  end;

// HackTPictureRegisteredFormats
{$ifopt R+}
  {$define HackTPictureRegisteredFormats_Disable_RangeCheck}
  {$R-}
{$endif}
procedure HackTPictureRegisteredFormats(destList: TStrings);
{$IFNDEF FPC}
var
  pRegisterFileFormat, pCallGetFileFormat, pGetFileFormats, pFileFormats: PAnsiChar;
  iCall: cardinal;
  i: integer;
  list: TList;
  fileFormat: PFileFormat;
{$ENDIF}
begin
  {$IFDEF FPC}
  {$MESSAGE WARN 'HackTPictureRegisteredFormats not suppose to get here at all. GraphicClassForExtension must handle this for you.'}
  destList.Clear;
  {$ELSE}
  {$MESSAGE WARN 'HackTPictureRegisteredFormats will crash when Graphics.pas is compiled with the 'Use Debug DCUs' option'}

  pRegisterFileFormat := PAnsiChar(@TPicture.RegisterFileFormat);
  if pRegisterFileFormat[0] = #$FF then // in case of BPL redirector
    pRegisterFileFormat := PAnsiChar(PCardinal(PCardinal(@pRegisterFileFormat[2])^)^);
  pCallGetFileFormat := @pRegisterFileFormat[16];
  iCall := PCardinal(pCallGetFileFormat)^;
  pGetFileFormats := @pCallGetFileFormat[iCall + 4];
  pFileFormats := PAnsiChar(PCardinal(@pGetFileFormats[2])^);
  list := TList(PCardinal(pFileFormats)^);

  if list <> nil then
  begin
    for i := 0 to list.Count - 1 do
    begin
      fileFormat := PFileFormat(list[i]);
      destList.AddObject(fileFormat.Extension + '=' + fileFormat.Description,
        TObject(fileFormat.GraphicClass));
    end;
  end;
  {$ENDIF}
end;

{$ifdef HackTPictureRegisteredFormats_Disable_RangeCheck}
  {$R+}
  {$undef HackTPictureRegisteredFormats_Disable_RangeCheck}
{$endif}

end.

