//
// This unit is part of the GLScene Project, http://glscene.org
//
{
   TFont Import into a BitmapFont using variable width...

    History :  
       04/12/14 - PW - Corrected the usage of pixel formats for Lazarus (by Gabriel Corneanu)
        29/05/11 - Yar - Unicode support for Unix OSes (by Gabriel Corneanu)
       16/05/11 - Yar - Redesign to use multiple textures (by Gabriel Corneanu)
       13/05/11 - Yar - Adapted to unicode (by Gabriel Corneanu)
       23/08/10 - Yar - Added OpenGLTokens to uses, replaced OpenGL1x functions to OpenGLAdapter
       06/06/10 - Yar - Added "VectorTypes.pas" unit to uses
       25/01/10 - Yar - Bugfix in LoadWindowsFont with zero width of char
                          (thanks olkondr)
                          Replace Char to AnsiChar
       11/11/09 - DaStr - Added Delphi 2009 compatibility (thanks mal)
       17/03/07 - DaStr - Dropped Kylix support in favor of FPC (BugTracekrID=1681585)
       12/15/04 - Eugene Kryukov - Added TGLStoredBitmapFont
       03/07/04 - LR - Added ifdef for Graphics uses
       29/09/02 - EG - Fixed transparency, style fixes, prop defaults fixed,
                          dropped interface dependency, texture size auto computed,
                          fixed italics spacing, uses LUM+ALPHA texture
       06/09/02 - JAJ - Fixed alot of bugs... Expecially designtime updating bugs..
       12/08/02 - JAJ - Made into a standalone unit...
  
}
unit GLWindowsFont;

{$mode delphi}

interface

{$INCLUDE GLScene.inc}

uses
{$IFDEF MSWINDOWS}
  Windows,
{$ENDIF}

  LCLIntf, LCLType, LCLProc, LazUTF8,

  GLBitmapFont,
  GLRenderContextInfo,
  Classes,
  GLScene,
  GLTexture,

  Graphics, Types,
  GLVectorLists,
  GLCrossPlatform;

type

  // TGLWindowsBitmapFont
  //
  { A bitmap font automatically built from a TFont.
     It works like a TGLBitmapfont, you set ranges and which chars are assigned
     to which indexes, however here you also set the Font property to any TFont
     available to the system and it renders in GLScene as close to that font
     as posible, on some font types this is 100% on some a slight difference
     in spacing can occur at most 1 pixel per char on some char combinations.
     Ranges must be sorted in ascending ASCII order and should not overlap.
     As the font texture is automatically layed out, the Ranges StartGlyphIdx
     property is ignored and replaced appropriately. }
  TGLWindowsBitmapFont = class(TGLCustomBitmapFont)
  private
     
    FFont: TFont;
    procedure SetList(const AList : TIntegerList);
  protected
     
    procedure SetFont(value: TFont);
    procedure LoadWindowsFont; virtual;
    function  StoreRanges: Boolean;

    procedure PrepareImage(var ARci: TGLRenderContextInfo); override;
    function  TextureFormat: Integer; override;
    procedure StreamlineRanges;
  public
     
    constructor Create(AOwner: TComponent); override;
    destructor Destroy; override;

    procedure NotifyChange(Sender: TObject); override;

    function FontTextureWidth: Integer;
    function FontTextureHeight: Integer;

    procedure EnsureString(const s : UnicodeString); overload;
    procedure EnsureChars(const AStart, AEnd: widechar);

    property Glyphs;

  published
     
      { The font used to prepare the texture.
         Note: the font color is ignored. }
    property Font: TFont read FFont write SetFont;

    property HSpace;
    property VSpace;
    property MagFilter;
    property MinFilter;
    property Ranges stored StoreRanges;
  end;

  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
implementation
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// ------------------------------------------------------------------

uses
  GLUtils,
  Math,
  SysUtils,
  GLVectorGeometry,
  OpenGLTokens,
  GLApplicationFileIO,
  GLVectorTypes;

const
  cDefaultLast = '}';

{$IFDEF MSWINDOWS}
Var
  Win32PlatformIsUnicode : Boolean;
{$ENDIF}

// ------------------
// ------------------ TGLWindowsBitmapFont ------------------
// ------------------

// Create
//

constructor TGLWindowsBitmapFont.Create(AOwner: TComponent);
begin
  inherited;

  FFont := TFont.Create;
  FFont.Color := clWhite;
  FFont.OnChange := NotifyChange;
  GlyphsAlpha := tiaAlphaFromIntensity;
  EnsureChars(' ', cDefaultLast);
end;

// Destroy
//

destructor TGLWindowsBitmapFont.Destroy;
begin
  FFont.Free;
  Ranges.Clear; 
  inherited;
end;

// FontTextureWidth
//

function TGLWindowsBitmapFont.FontTextureWidth: Integer;
begin
  Result := Glyphs.Width;
end;

// FontTextureHeight
//

function TGLWindowsBitmapFont.FontTextureHeight: Integer;
begin
  Result := Glyphs.Height;
end;

// SetFont
//

procedure TGLWindowsBitmapFont.SetFont(value: TFont);
begin
  FFont.Assign(value);
end;

// NotifyChange
//

procedure TGLWindowsBitmapFont.NotifyChange(Sender: TObject);
begin
  StreamlineRanges;
  FreeTextureHandle;
  InvalidateUsers;
  inherited;
end;

// LoadWindowsFont
//

procedure TGLWindowsBitmapFont.LoadWindowsFont;

  procedure ComputeCharRects(bitmap: TGLBitmap);
  var
    px, py, cw, n, x, y: Integer;
    PaddedHeight : integer;
    buffer : array[0..2] of WideChar;
    p : PCharInfo;
    r : TGLRect;
{$IFNDEF MSWINDOWS}
    utfbuffer: array[0..5] of Char;
    i: SizeUInt;
{$ENDIF}
  begin
    buffer[1] := WideChar(#32);
    buffer[2] := WideChar(#0);
    PaddedHeight:= CharHeight + GlyphsIntervalY;
    x := bitmap.Width; y := bitmap.Height;
    px := 0;
    py := 0;
    if y < CharHeight then px := x;
    p  := @FChars[0];
    for n := 0 to CharacterCount - 1 do
    begin
      cw := p.w;
      if cw > 0 then
      begin
        Inc(cw, GlyphsIntervalX);

        if px + cw > x then
        begin
          px := 0;
          Inc(py, PaddedHeight);
          if py + PaddedHeight > y then
          begin
            py := bitmap.Height;
            y  := py + TextureHeight;
            bitmap.Height := y;
            with bitmap.Canvas do
            begin
              Brush.Style := bsSolid;
              Brush.Color := clBlack;
              FillRect(Rect(0, py, x, y));
            end;
          end;
        end;

        if Assigned(bitmap) then
        begin
          //+1 makes right align (padding left);
          // I prefer padding right for compatibility with bitmap font...
          p.l := px;
          //should make it consistent, same as above
          p.t := py;

          r.Left := px;
          r.Top  := py;
          r.Right  := px + cw;
          r.Bottom := py + PaddedHeight;
          buffer[0] := TileIndexToChar(n);
          // Draw the Char, the trailing space is to properly handle the italics.
{$IFDEF MSWINDOWS}
          // credits to the Unicode version of SynEdit for this function call. GPL/MPL as GLScene
          Windows.ExtTextOutW(bitmap.Canvas.Handle, p.l, p.t, ETO_CLIPPED, @r, buffer, 1, nil);
{$ELSE}
          ConvertUTF16ToUTF8(utfbuffer, 5, buffer, 1,  [toInvalidCharToSymbol], i);
          LCLIntf.ExtTextOut(bitmap.Canvas.Handle, p.l, p.t, ETO_CLIPPED, @r, utfbuffer, i-1, nil);
{$ENDIF}
        end;
        Inc(px, cw);
      end
      else
      begin
        p.l := 0;
        p.t := 0;
      end;
      inc(p);
    end;
  end;

  // credits to the Unicode version of SynEdit for this function. GPL/MPL as GLScene
  function GetTextSize(DC: HDC; Str: PWideChar; Count: Integer): TSize;

    {$IFDEF MSWINDOWS}

     var tm: LPTEXTMETRICW;

    {$ELSE}
    var LString: array[0..5] of char; //here we always have 1 char, so it's safe
      i : SizeUInt;
    {$ENDIF}

  begin
    Result.cx := 0;
    Result.cy := 0;
{$IFDEF MSWINDOWS}

    GetTextExtentPoint32W(DC, Str, Count, Result);
    if not Win32PlatformIsUnicode then
    begin
      GetTextMetricsW(DC, tm);
      if tm.tmPitchAndFamily and TMPF_TRUETYPE <> 0 then
        Result.cx := Result.cx - tm.tmOverhang
      else
        Result.cx := tm.tmAveCharWidth * Count;
    end;
{$ELSE}
    if ConvertUTF16ToUTF8(LString, 5, Str, Count,  [toInvalidCharToSymbol], i) = trNoError then
      GetTextExtentPoint32(DC, LString, i-1, Result);
{$ENDIF}
  end;

var
  bitmap: TGLBitmap;
  ch: widechar;
  i, cw, nbChars, n: Integer;
begin
  InvalidateUsers;
  Glyphs.OnChange := nil;
  //accessing Bitmap might trigger onchange
  bitmap := Glyphs.Bitmap;

  bitmap.Height      := 0;
  {$IFDEF MSWINDOWS}
   //due to lazarus doesn't properly support pixel formats
     bitmap.PixelFormat := glpf32bit;
  {$ENDIF}
  with bitmap.Canvas do
  begin
    Font := Self.Font;
    Font.Color := clWhite;
    // get characters dimensions for the font
    // character size without padding; paddings are used from GlyphsInterval
    CharWidth  := Round(MaxInteger(TextWidth('M'), TextWidth('W'), TextWidth('_')));
    CharHeight := TextHeight('"_pI|,');
    // used for padding
    GlyphsIntervalX := 1;
    GlyphsIntervalY := 1;
    if fsItalic in Font.Style then
    begin
      // italics aren't properly acknowledged in font width
      HSpaceFix := -(CharWidth div 3);
      CharWidth := CharWidth - HSpaceFix;
    end
    else
      HSpaceFix := 0;
  end;

  nbChars := CharacterCount;

  // Retrieve width of all characters (texture width)
  ResetCharWidths(0);
  n := 0;
  for i := 0 to nbChars - 1 do
  begin
    ch := TileIndexToChar(i);
    cw := GetTextSize(bitmap.canvas.Handle, @ch, 1).cx-HSpaceFix;
    n  := n + cw + GlyphsIntervalX;
    SetCharWidths(i, cw);
  end;
  //try to make best guess...
  //~total pixels, including some waste (10%)
  n := n * (CharHeight + GlyphsIntervalY) * 11 div 10;
  TextureWidth := min(512, RoundUpToPowerOf2( round(sqrt(n)) ));
  TextureHeight := min(512, RoundUpToPowerOf2( n div TextureWidth));

  bitmap.Width := TextureWidth;

  ComputeCharRects(bitmap);
  FCharsLoaded := true;
  Glyphs.OnChange := OnGlyphsChanged;
end;

// StoreRanges
//

function TGLWindowsBitmapFont.StoreRanges: Boolean;
begin
  Result := (Ranges.Count <> 1) or (Ranges[0].StartASCII[1] <> ' ') or (Ranges[0].StopASCII[1] <> cDefaultLast);
end;

type
  TFriendlyRange = class(TGLBitmapFontRange);

procedure TGLWindowsBitmapFont.StreamlineRanges;
var
  I, C: Integer;
begin
  C := 0;
  for I := 0 to Ranges.Count - 1 do
  begin
    TFriendlyRange(Ranges[I]).FStartGlyphIdx := C;
    Inc(C, Ranges[I].CharCount);
    TFriendlyRange(Ranges[I]).FStopGlyphIdx := MaxInteger(C - 1, 0);
  end;
end;

procedure TGLWindowsBitmapFont.SetList(const AList: TIntegerList);
var
  i : integer;
  f, n, s : integer;
begin
  //add existing ranges
  for I := 0 to Ranges.Count - 1 do
    with Ranges.Items[I] do
      AList.AddSerie(integer(StartASCII[1]), 1, CharCount);

  AList.SortAndRemoveDuplicates;

  Ranges.Clear;
  Ranges.BeginUpdate;
  if AList.Count > 0 then
  begin
    i := 0;
    while (i < AList.Count) and (AList[i] < 32) do inc(i);
    while i < AList.Count do
    begin
      f := AList[i]; n := f; s := Ranges.CharacterCount;
      while (i < AList.Count) and (n = AList[i]) do
      begin
        inc(i);
        inc(n);
      end;
      Ranges.Add(widechar(f), widechar(pred(n))).StartGlyphIdx := s;
    end;
  end;

  Ranges.EndUpdate;
  TextureChanged;
  InvalidateUsers;
end;

//add characters to internal list
procedure TGLWindowsBitmapFont.EnsureChars(const AStart, AEnd: widechar);
var
  c : widechar;
  ACharList : TIntegerList;
begin
  ACharList := TIntegerList.Create;
  for c := AStart to AEnd do
      ACharList.Add(integer(c));
  SetList(ACharList);
  ACharList.Free;
end;

//add characters to internal list
procedure TGLWindowsBitmapFont.EnsureString(const s: UnicodeString);
var
  i : integer;
  ACharList : TIntegerList;
begin
  ACharList := TIntegerList.Create;
  for i := 1 to length(s) do
      ACharList.Add(integer(s[i]));
  SetList(ACharList);
  ACharList.Free;
end;

// PrepareImage
//

procedure TGLWindowsBitmapFont.PrepareImage(var ARci: TGLRenderContextInfo);
begin
  LoadWindowsFont;
  inherited PrepareImage(ARci);
end;

// TextureFormat
//

function TGLWindowsBitmapFont.TextureFormat: Integer;
begin
  Result := GL_ALPHA;
end;

initialization
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
{$IFDEF MSWINDOWS}
  Win32PlatformIsUnicode := (Win32Platform = VER_PLATFORM_WIN32_NT);
{$ENDIF}

   // class registrations
  RegisterClasses([TGLWindowsBitmapFont]);

end.

