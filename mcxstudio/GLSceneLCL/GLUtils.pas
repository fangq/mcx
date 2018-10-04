//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   Miscellaneous support utilities & classes.

  History :  
       02/01/13 - Yar - Added SetGLSceneMediaDir
       07/01/11 - Yar - Added SaveModelDialog, OpenModelDialog
       04/03/10 - DanB - Now uses CharInSet
       27/05/09 - DanB - re-added TryStrToFloat, since it ignores user's locale.
       24/03/09 - DanB - removed TryStrToFloat (exists in SysUtils or GLCrossPlatform already)
                            changed StrToFloatDef to accept only 1 param + now overloaded
       24/03/09 - DanB - Moved Dialog utilities here from GLCrossPlatform, because
                            they work on all platforms (with FPC)
       16/10/08 - UweR - corrected typo in TryStringToColorAdvanced parameter
       16/10/08 - DanB - renamed Save/LoadStringFromFile to Save/LoadAnsiStringFromFile
       24/03/08 - DaStr - Removed OpenGL1x dependancy
                             Moved TGLMinFilter and TGLMagFilter from GLUtils.pas
                              to GLGraphics.pas (BugTracker ID = 1923844)
       25/03/07 - DaStr - Replaced StrUtils with GLCrossPlatform
       23/03/07 - DaStr - Removed compiler warnings caused by
                               SaveComponentToFile and LoadComponentFromFile
       22/03/07 - DaStr - Added SaveComponentToFile, LoadComponentFromFile
       07/02/07 - DaStr - Added StringToColorAdvanced() functions
       05/09/03 - EG - Creation from GLMisc split
    
}
unit GLUtils;

interface

{$I GLScene.inc}

uses
  Classes, SysUtils, types,
  Graphics, Controls, FileUtil, LazUtf8, lazfileutils, Dialogs, ExtDlgs,
  // GLScene
  GLVectorGeometry, GLCrossPlatform;

type
  EGLUtilsException = class(Exception);

  TSqrt255Array = array[0..255] of Byte;
  PSqrt255Array = ^TSqrt255Array;

  // Copies the values of Source to Dest (converting word values to integer values)
procedure WordToIntegerArray(Source: PWordArray; Dest: PIntegerArray; Count: Cardinal);
// Round ups to the nearest power of two, value must be positive
function RoundUpToPowerOf2(value: Integer): Integer;
// Round down to the nearest power of two, value must be strictly positive
function RoundDownToPowerOf2(value: Integer): Integer;
// Returns True if value is a true power of two
function IsPowerOf2(value: Integer): Boolean;
{ Read a CRLF terminated string from a stream.
   The CRLF is NOT in the returned string. }
function ReadCRLFString(aStream: TStream): AnsiString;
// Write the string and a CRLF in the stream
procedure WriteCRLFString(aStream: TStream; const aString: AnsiString);
// Similar to SysUtils.TryStrToFloat, but ignores user's locale
function TryStrToFloat(const strValue: string; var val: Extended): Boolean;
// Similar to SysUtils.StrToFloatDef, but ignores user's locale
function StrToFloatDef(const strValue: string; defValue: Extended = 0): Extended;

// Converts a string into color
function StringToColorAdvancedSafe(const Str: string; const Default: TColor): TColor;
// Converts a string into color
function TryStringToColorAdvanced(const Str: string; var OutColor: TColor): Boolean;
// Converts a string into color
function StringToColorAdvanced(const Str: string): TColor;

{ Parses the next integer in the string.
   Initial non-numeric characters are skipper, p is altered, returns 0 if none
   found. '+' and '-' are acknowledged. }
function ParseInteger(var p: PChar): Integer;
{ Parses the next integer in the string.
   Initial non-numeric characters are skipper, p is altered, returns 0 if none
   found. Both '.' and ',' are accepted as decimal separators. }
function ParseFloat(var p: PChar): Extended;

{ Saves "data" to "filename". }
procedure SaveAnsiStringToFile(const fileName: string; const data: AnsiString);
{ Returns the content of "filename". }
function LoadAnsiStringFromFile(const fileName: string): AnsiString;

{ Saves component to a file. }
procedure SaveComponentToFile(const Component: TComponent; const FileName: string; const AsText: Boolean = True);
{ Loads component from a file. }
procedure LoadComponentFromFile(const Component: TComponent; const FileName: string; const AsText: Boolean = True);

{ Returns the size of "filename".
   Returns 0 (zero) is file does not exists. }
function SizeOfFile(const fileName: string): Int64;

{ Returns a pointer to an array containing the results of "255*sqrt(i/255)". }
function GetSqrt255Array: PSqrt255Array;

{ Pops up a simple dialog with msg and an Ok button. }
procedure InformationDlg(const msg: string);
{ Pops up a simple question dialog with msg and yes/no buttons.
   Returns True if answer was "yes". }
function QuestionDlg(const msg: string): Boolean;
{ Posp a simple dialog with a string input. }
function InputDlg(const aCaption, aPrompt, aDefault: string): string;

{ Pops up a simple save picture dialog. }
function SavePictureDialog(var aFileName: string; const aTitle: string = ''): Boolean;
{ Pops up a simple open picture dialog. }
function OpenPictureDialog(var aFileName: string; const aTitle: string = ''): Boolean;

//procedure SetGLSceneMediaDir();
Function SetGLSceneMediaDir:string;

var MediaPath:String;
//------------------------------------------------------
//------------------------------------------------------
//------------------------------------------------------
implementation
//------------------------------------------------------
//------------------------------------------------------
//------------------------------------------------------

uses
  GLApplicationFileIO;



var
  vSqrt255: TSqrt255Array;

resourcestring
  gluInvalidColor = '''%s'' is not a valid color format!';

  // WordToIntegerArray
  //
{$IFNDEF GEOMETRY_NO_ASM}

procedure WordToIntegerArray(Source: PWordArray; Dest: PIntegerArray; Count: Cardinal); assembler;
// EAX contains Source
// EDX contains Dest
// ECX contains Count
asm
              JECXZ @@Finish
              PUSH ESI
              PUSH EDI
              MOV ESI,EAX
              MOV EDI,EDX
              XOR EAX,EAX
@@1:          LODSW
              STOSD
              DEC ECX
              JNZ @@1
              POP EDI
              POP ESI
@@Finish:
end;
{$ELSE}

procedure WordToIntegerArray(Source: PWordArray; Dest: PIntegerArray; Count: Cardinal);
var
  i: integer;
begin
  for i := 0 to Count - 1 do
    Dest^[i] := Source^[i];
end;
{$ENDIF}

// RoundUpToPowerOf2
//

function RoundUpToPowerOf2(value: Integer): Integer;
begin
  Result := 1;
  while (Result < value) do
    Result := Result shl 1;
end;

// RoundDownToPowerOf2
//

function RoundDownToPowerOf2(value: Integer): Integer;
begin
  if value > 0 then
  begin
    Result := 1 shl 30;
    while Result > value do
      Result := Result shr 1;
  end
  else
    Result := 1;
end;

// IsPowerOf2
//

function IsPowerOf2(value: Integer): Boolean;
begin
  Result := (RoundUpToPowerOf2(value) = value);
end;

// ReadCRLFString
//

function ReadCRLFString(aStream: TStream): AnsiString;
var
  c: AnsiChar;
begin
  Result := '';
  while Copy(Result, Length(Result) - 1, 2) <> #13#10 do
  begin
    aStream.Read(c, 1);
    Result := Result + c;
  end;
  Result := Copy(Result, 1, Length(Result) - 2);
end;

// WriteCRLFString
//

procedure WriteCRLFString(aStream: TStream; const aString: AnsiString);
const
  cCRLF: Integer = $0A0D;
begin
  with aStream do
  begin
    Write(aString[1], Length(aString));
    Write(cCRLF, 2);
  end;
end;

// TryStrToFloat
//

function TryStrToFloat(const strValue: string; var val: Extended): Boolean;
var
  i, j, divider, lLen, exponent: Integer;
  c: Char;
  v: Extended;
begin
  if strValue = '' then
  begin
    Result := False;
    Exit;
  end
  else
    v := 0;
  lLen := Length(strValue);
  while (lLen > 0) and (strValue[lLen] = ' ') do
    Dec(lLen);
  divider := lLen + 1;
  exponent := 0;
  for i := 1 to lLen do
  begin
    c := strValue[i];
    case c of
      ' ': if v <> 0 then
        begin
          Result := False;
          Exit;
        end;
      '0'..'9': v := (v * 10) + Integer(c) - Integer('0');
      ',', '.':
        begin
          if (divider > lLen) then
            divider := i + 1
          else
          begin
            Result := False;
            Exit;
          end;
        end;
      '-', '+': if i > 1 then
        begin
          Result := False;
          Exit;
        end;
      'e', 'E':
        begin
          if i + 1 > lLen then
          begin
            Result := False;
            Exit;
          end;
          for j := i + 1 to lLen do
          begin
            c := strValue[j];
            case c of
              '-', '+': if j <> i + 1 then
                begin
                  Result := False;
                  Exit;
                end;
              '0'..'9': exponent := (exponent * 10) + Integer(c) - Integer('0');
            else
              Result := False;
              Exit;
            end;
          end;
          if strValue[i + 1] <> '-' then
            exponent := -exponent;
          exponent := exponent - 1;
          lLen := i;
          if divider > lLen then
            divider := lLen;
          Break;
        end;
    else
      Result := False;
      Exit;
    end;
  end;
  divider := lLen - divider + exponent + 1;
  if strValue[1] = '-' then
  begin
    v := -v;
  end;
  if divider <> 0 then
    v := v * Exp(-divider * Ln(10));
  val := v;
  Result := True;
end;

// StrToFloatDef
//

function StrToFloatDef(const strValue: string; defValue: Extended = 0): Extended;
begin
  if not TryStrToFloat(strValue, Result) then
    result := defValue;
end;

// StringToColorAdvancedSafe
//

function StringToColorAdvancedSafe(const Str: string; const Default: TColor): TColor;
begin
  if not TryStringToColorAdvanced(Str, Result) then
    Result := Default;
end;

// StringToColorAdvanced
//

function StringToColorAdvanced(const Str: string): TColor;
begin
  if not TryStringToColorAdvanced(Str, Result) then
    raise EGLUtilsException.CreateResFmt(@gluInvalidColor, [Str]);
end;

// TryStringToColorAdvanced
//

function TryStringToColorAdvanced(const Str: string; var OutColor: TColor): Boolean;
var
  Code, I: Integer;
  Temp: string;
begin
  Result := True;
  Temp := Str;

  Val(Temp, I, Code); //to see if it is a number
  if Code = 0 then
    OutColor := TColor(I) //Str = $0000FF
  else
  begin
    if not IdentToColor(Temp, Longint(OutColor)) then //Str = clRed
    begin
      if AnsiStartsText('clr', Temp) then //Str = clrRed
      begin
        Delete(Temp, 3, 1);
        if not IdentToColor(Temp, Longint(OutColor)) then
          Result := False;
      end
      else if not IdentToColor('cl' + Temp, Longint(OutColor)) then //Str = Red
        Result := False;
    end;
  end;
end;

// ParseInteger
//

function ParseInteger(var p: PChar): Integer;
var
  neg: Boolean;
  c: Char;
begin
  Result := 0;
  if p = nil then
    Exit;
  neg := False;
  // skip non-numerics
  while not CharInSet(p^, [#0, '0'..'9', '+', '-']) do
    Inc(p);
  c := p^;
  if c = '+' then
    Inc(p)
  else if c = '-' then
  begin
    neg := True;
    Inc(p);
  end;
  // Parse numerics
  while True do
  begin
    c := p^;
    if not CharInSet(c, ['0'..'9']) then
      Break;
    Result := Result * 10 + Integer(c) - Integer('0');
    Inc(p);
  end;
  if neg then
    Result := -Result;
end;

// ParseFloat
//

function ParseFloat(var p: PChar): Extended;
var
  decimals, expSign, exponent: Integer;
  c: Char;
  neg: Boolean;
begin
  Result := 0;
  if p = nil then
    Exit;
  // skip non-numerics
  while not CharInSet(p^, [#0, '0'..'9', '+', '-']) do
    Inc(p);
  c := p^;
  if c = '+' then
  begin
    neg := False;
    Inc(p);
  end
  else if c = '-' then
  begin
    neg := True;
    Inc(p);
  end
  else
    neg := False;
  // parse numbers
  while CharInSet(p^, ['0'..'9']) do
  begin
    Result := Result * 10 + (Integer(p^) - Integer('0'));
    Inc(p);
  end;
  // parse dot, then decimals, if any
  decimals := 0;
  if (p^ = '.') then
  begin
    Inc(p);
    while CharInSet(p^, ['0'..'9']) do
    begin
      Result := Result * 10 + (Integer(p^) - Integer('0'));
      Inc(p);
      Dec(decimals);
    end;
  end;
  // parse exponent, if any
  if CharInSet(p^, ['e', 'E']) then
  begin
    Inc(p);
    // parse exponent sign
    c := p^;
    if c = '-' then
    begin
      expSign := -1;
      Inc(p);
    end
    else if c = '+' then
    begin
      expSign := 1;
      Inc(p);
    end
    else
      expSign := 1;
    // parse exponent
    exponent := 0;
    while CharInSet(p^, ['0'..'9']) do
    begin
      exponent := exponent * 10 + (Integer(p^) - Integer('0'));
      Inc(p);
    end;
    decimals := decimals + expSign * exponent;
  end;
  if decimals <> 0 then
    Result := Result * Exp(decimals * Ln(10));
  if neg then
    Result := -Result;
end;

// SaveStringToFile
//

procedure SaveAnsiStringToFile(const fileName: string; const data: AnsiString);
var
  n: Cardinal;
  fs: TStream;
begin
  fs := CreateFileStream(fileName, fmCreate);
  try
    n := Length(data);
    if n > 0 then
      fs.Write(data[1], n);
  finally
    fs.Free;
  end;
end;

// LoadStringFromFile
//

function LoadAnsiStringFromFile(const fileName: string): AnsiString;
var
  n: Cardinal;
  fs: TStream;
begin
  if FileExists(fileName) then
  begin
    fs := CreateFileStream(fileName, fmOpenRead + fmShareDenyNone);
    try
      n := fs.Size;
      SetLength(Result, n);
      if n > 0 then
        fs.Read(Result[1], n);
    finally
      fs.Free;
    end;
  end
  else
    Result := '';
end;

// SaveComponentToFile
//

procedure SaveComponentToFile(const Component: TComponent; const FileName: string; const AsText: Boolean);
var
  Stream: TStream;
  MemStream: TMemoryStream;
begin
  Stream := CreateFileStream(FileName, fmCreate);
  try
    if AsText then
    begin
      MemStream := TMemoryStream.Create;
      try
        MemStream.WriteComponent(Component);
        MemStream.Position := 0;
        ObjectBinaryToText(MemStream, Stream);
      finally
        MemStream.Free;
      end;
    end
    else
      Stream.WriteComponent(Component);
  finally
    Stream.Free;
  end;
end;

// LoadComponentFromFile
//

procedure LoadComponentFromFile(const Component: TComponent; const FileName: string; const AsText: Boolean = True);
var
  Stream: TStream;
  MemStream: TMemoryStream;
begin
  Stream := CreateFileStream(FileName, fmOpenRead);
  try
    if AsText then
    begin
      MemStream := TMemoryStream.Create;
      try
        ObjectTextToBinary(Stream, MemStream);
        MemStream.Position := 0;
        MemStream.ReadComponent(Component);
      finally
        MemStream.Free;
      end;
    end
    else
      Stream.ReadComponent(Component);
  finally
    Stream.Free;
  end;
end;

// SizeOfFile
//

function SizeOfFile(const fileName: string): Int64;
var
  fs: TStream;
begin
  if FileExists(fileName) then
  begin
    fs := CreateFileStream(fileName, fmOpenRead + fmShareDenyNone);
    try
      Result := fs.Size;
    finally
      fs.Free;
    end;
  end
  else
    Result := 0;
end;

// GetSqrt255Array
//

function GetSqrt255Array: PSqrt255Array;
const
  cOneDiv255 = 1 / 255;
var
  i: Integer;
begin
  if vSqrt255[255] <> 255 then
  begin
    for i := 0 to 255 do
      vSqrt255[i] := Integer(Trunc(255 * Sqrt(i * cOneDiv255)));
  end;
  Result := @vSqrt255;
end;

// InformationDlg
//

procedure InformationDlg(const msg: string);
begin
  ShowMessage(msg);
end;

// QuestionDlg
//

function QuestionDlg(const msg: string): Boolean;
begin
  Result := (MessageDlg(msg, mtConfirmation, [mbYes, mbNo], 0) = mrYes);
end;

// InputDlg
//

function InputDlg(const aCaption, aPrompt, aDefault: string): string;
begin
  Result := InputBox(aCaption, aPrompt, aDefault);
end;

// SavePictureDialog
//

function SavePictureDialog(var aFileName: string; const aTitle: string = ''): Boolean;
var
  saveDialog: TSavePictureDialog;
begin
  saveDialog := TSavePictureDialog.Create(nil);
  try
    with saveDialog do
    begin
      Options := [ofHideReadOnly, ofNoReadOnlyReturn];
      if aTitle <> '' then
        Title := aTitle;
      FileName := aFileName;
      Result := Execute;
      if Result then
        aFileName := FileName;
    end;
  finally
    saveDialog.Free;
  end;
end;

// OpenPictureDialog
//

function OpenPictureDialog(var aFileName: string; const aTitle: string = ''): Boolean;
var
  openDialog: TOpenPictureDialog;
begin
  openDialog := TOpenPictureDialog.Create(nil);
  try
    with openDialog do
    begin
      Options := [ofHideReadOnly, ofNoReadOnlyReturn];
      if aTitle <> '' then
        Title := aTitle;
      FileName := aFileName;
      Result := Execute;
      if Result then
        aFileName := FileName;
    end;
  finally
    openDialog.Free;
  end;
end;

Function SetGLSceneMediaDir:string;
var
  path: UTF8String;
  p: integer;
begin
   result:='';

   // We need to lower case path because the functions are case sensible
//   path := lowercase(ExtractFilePath(ParamStrUTF8(0)));
   path := lowercase(ExtractFilePath(ParamStr(0)));
   p := Pos('samples', path);
   Delete(path, p + 7, Length(path));
   path := IncludeTrailingPathDelimiter(IncludeTrailingPathDelimiter(path) + 'media');
   SetCurrentDir(path);
   // SetCurrentDirUTF8(path) -->  NOT WORKING ON W10 64Bits !
     // We need to store the result in a global var "MediaPath"
     // The function SetCurrentDirUTF8 return TRUE but we are always in the application's folder
     // NB These functions provide from LazFileUtils unit and not from deprecated functions in FileUtils unit.

   MediaPath:=Path ;
   result:=path;

end;

end.

