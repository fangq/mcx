{ mcxstudio2 - Toolbar, menu and shape-tree icons.

  The artwork is MCX Studio's own icon set, icons/svg in this repository --
  the flat coloured badges the Lazarus GUI has used since 2017.  "make icons"
  rasterises each one to a 96-pixel PNG and embeds it in mcxstudio2.res; both
  the PNGs and the .res are committed, so an ordinary build needs neither
  inkscape nor fpcres.

  The masters are 96 pixels because the display scale is not known until the
  program runs: this unit area-averages each badge down to whatever size the
  image list asks for, which on a 130-dpi screen is 22 and on a 192-dpi one
  is 32.  Resampling thirty icons at startup costs a few milliseconds.

  Seven glyphs the old set never had -- new, wizard, bench, fit, source,
  detector, media -- were drawn for mcxstudio2 in the same style, reusing the
  set's own badge outline, and live alongside the originals in icons/svg.

  Two icons are still drawn rather than rasterised: the navigator's collapsed
  and expanded chevrons.  A coloured badge is the wrong thing beside a
  heading, and a drawn chevron takes the theme's text colour, so it stays
  legible when the rest of the window turns dark. }
unit mcxicons;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Graphics, Controls, ImgList,
  FPImage, FPReadPNG, FPWritePNG;

const
  McxWindowIconRes = 'MCXICONPNG';   { see packaging/windows/mcxstudio2.rc }

  { Every badge is embedded under this prefix plus its upper-cased name, so
    the resource script is generated from the same packaging/icons.map the
    rasteriser reads and nothing has to be listed twice. }
  McxIconResPrefix = 'MCXICON_';

type
  { The names are the action ids they belong to, lower-cased, so a caller can
    ask for an icon by action name and get nil-safe behaviour when there is
    none. }
  TMcxIconName = string;

{ Fills AImages with one bitmap per name in ANames, in that order, so the
  index of a name is its index in the list.  Returns the list for chaining. }
function McxBuildIconList(AImages: TImageList; const ANames: array of string;
  AColour: TColor): TImageList;

{ Index of ANAme in the list built by McxBuildIconList, or -1. }
function McxIconIndex(const AName: string): Integer;

{ Every icon this unit can draw, in the canonical order used by the image
  list the application builds at startup. }
function McxIconNames: TStringArray;

{ Puts the application's own logo on the window and the task bar entry.

  It reads the PNG copy of the artwork that packaging/windows/led.rc embeds,
  rather than the MAINICON in the same binary.  MAINICON is the same picture
  and the LCL reads it back correctly in process -- all seven sizes -- but
  what reaches the window manager from it has its colour channels striped
  and its alpha forced opaque.  A PNG assigned to Application.Icon arrives
  intact.  Does nothing if the resource is missing, because a build without
  it should start with no icon rather than not start. }
procedure McxApplyWindowIcon;

{ The application's own logo, at whatever size the resource holds, or nil when
  this build has no artwork.  The caller owns the result. }
function McxAppLogo: TPortableNetworkGraphic;

{ One icon as a PNG at ASize pixels, area-averaged from the embedded 96-pixel
  master, or nil when this build has no artwork under that name -- which is
  the normal answer for the two drawn chevrons.  The caller owns the result. }
function McxIconPng(const AName: string; ASize: Integer): TPortableNetworkGraphic;

{ Draws one icon into ABitmap, which must already be sized.  Only the drawn
  glyphs -- the chevrons -- have anything to draw; everything else is
  artwork and comes back from McxIconPng. }
procedure McxDrawIcon(ABitmap: TBitmap; const AName: string; AColour: TColor);

{ One icon as a 16x16 bitmap with a transparent background, for the controls
  that take a Glyph rather than an image list index.  The caller owns the
  result only through the control it assigns it to -- TSpeedButton.Glyph
  copies, so the bitmap is freed here. }
function McxIconBitmap(const AName: string; AColour: TColor;
  ASize: Integer = 16): TBitmap;

{ Renders every icon into one PNG contact sheet, ASize pixels per cell.  Used
  by --dump-icons to review the artwork, and by the self-test to prove the
  glyph table and the name table stay in step. }
function McxSaveIconSheet(const AFileName: string; ASize: Integer = 32;
  AColour: TColor = clBlack): Boolean;

implementation

uses
  Forms, LCLType;

const
  { The background the icons are drawn on and then masked out.  Magenta
    because nothing in an icon is ever legitimately this colour. }
  MaskColour = TColor($00FF00FF);

  { Kept in one place so the toolbar, the menus and the shape tree all agree on
    what index means what.  Appending is safe; inserting is not, because an
    ImageIndex in a form file is an absolute position. }
  IconNames: array[0..33] of string = (
    { Session and files. }
    'new', 'open', 'save', 'saveas',
    { Running. }
    'run', 'stop', 'gpu', 'bench',
    { Modes and view. }
    'wizard', 'expert', 'preview', 'reset', 'fit',
    { Shape primitives -- these double as the shape-tree node glyphs. }
    'grid', 'box', 'subgrid', 'sphere', 'cylinder',
    'xlayers', 'ylayers', 'zlayers', 'xslabs', 'yslabs', 'zslabs',
    { Optodes and media. }
    'source', 'detector', 'media',
    { Editing. }
    'add', 'delete',
    { Help and appearance. }
    'about', 'help', 'theme',
    { Accordion headers.  A glyph rather than a caption prefix, because
      TSpeedButton centres its caption and has no Alignment -- but it does
      place a glyph at Margin, and the caption then follows it, which is what
      left-aligns the header text. }
    'collapsed', 'expanded'
  );

procedure McxApplyWindowIcon;
var
  Stream: TResourceStream;
  Png: TPortableNetworkGraphic;
begin
  Stream := nil;
  Png := nil;
  try
    try
      Stream := TResourceStream.Create(HInstance, McxWindowIconRes, RT_RCDATA);
      Png := TPortableNetworkGraphic.Create;
      Png.LoadFromStream(Stream);
      Application.Icon.Assign(Png);
    except
      { A missing or unreadable resource is not worth failing startup over. }
    end;
  finally
    Png.Free;
    Stream.Free;
  end;
end;

function McxAppLogo: TPortableNetworkGraphic;
var
  Stream: TResourceStream;
begin
  Result := nil;
  Stream := nil;
  try
    try
      Stream := TResourceStream.Create(HInstance, McxWindowIconRes, RT_RCDATA);
      Result := TPortableNetworkGraphic.Create;
      Result.LoadFromStream(Stream);
    except
      FreeAndNil(Result);
    end;
  finally
    Stream.Free;
  end;
end;

function McxIconNames: TStringArray;
var
  i: Integer;
begin
  Result := nil;
  SetLength(Result, Length(IconNames));
  for i := 0 to High(IconNames) do Result[i] := IconNames[i];
end;

function McxIconIndex(const AName: string): Integer;
var
  i: Integer;
begin
  for i := 0 to High(IconNames) do
    if SameText(IconNames[i], AName) then Exit(i);
  Result := -1;
end;

type
  { A tiny drawing context that takes coordinates on the 16x16 design grid and
    puts them where they belong at the real size, so a drawn glyph does not
    have to know how big it is being drawn.

    Line, Box and Ellipse went with the rest of the drawn set when the SVG
    artwork took over; the chevrons need a filled polygon and nothing else.
    Put them back from git if another glyph ever has to be drawn. }
  TPen16 = object
    C: TCanvas;
    S: Double;          // pixels per design unit
    procedure Init(ACanvas: TCanvas; ASize: Integer; AColour: TColor);
    function X(V: Double): Integer;
    procedure Poly(const APts: array of Double; AFill: Boolean = False);
  end;

procedure TPen16.Init(ACanvas: TCanvas; ASize: Integer; AColour: TColor);
begin
  C := ACanvas;
  S := ASize / 16;
  C.Pen.Color := AColour;
  C.Pen.Width := Round(S * 1.2);
  if C.Pen.Width < 1 then C.Pen.Width := 1;
  C.Pen.EndCap := pecSquare;
  C.Brush.Color := AColour;
  C.Brush.Style := bsClear;
end;

function TPen16.X(V: Double): Integer;
begin
  Result := Round(V * S);
end;

procedure TPen16.Poly(const APts: array of Double; AFill: Boolean);
var
  P: array of TPoint;
  i: Integer;
begin
  SetLength(P, Length(APts) div 2);
  for i := 0 to High(P) do
    P[i] := Point(X(APts[i * 2]), X(APts[i * 2 + 1]));
  if AFill then
  begin
    C.Brush.Style := bsSolid;
    C.Polygon(P);
    C.Brush.Style := bsClear;
  end
  else
    C.Polyline(P);
end;

{ Only the navigator's two chevrons are drawn.  Everything else is the SVG
  artwork, resampled by McxIconPng; a name that reaches here and matches
  nothing leaves the bitmap as the caller prepared it, which is a blank slot
  in the image list and a blank cell on the contact sheet. }
procedure McxDrawIcon(ABitmap: TBitmap; const AName: string; AColour: TColor);
var
  P: TPen16;
begin
  P.Init(ABitmap.Canvas, ABitmap.Width, AColour);
  case LowerCase(AName) of
    'collapsed': P.Poly([6, 3.5, 11, 8, 6, 12.5], True);
    'expanded':  P.Poly([3.5, 6, 12.5, 6, 8, 11], True);
  end;
end;

{ ---------------------------------------------------------- artwork ------- }

{ Area-averages ASrc down to ASize x ASize.

  The averaging is alpha-weighted.  A PNG writer leaves the colour of a fully
  transparent pixel undefined -- usually black -- so averaging colour and
  alpha independently drags that black into every partly covered pixel and
  rings each badge with a dark halo.  Weighting colour by alpha and dividing
  by the alpha that was actually there is what removes it. }
function ResampleImage(ASrc: TFPCustomImage; ASize: Integer): TFPMemoryImage;
var
  dx, dy, sx, sy, x0, x1, y0, y1, n: Integer;
  sr, sg, sb, sa: Int64;
  C: TFPColor;
begin
  Result := TFPMemoryImage.Create(ASize, ASize);
  Result.UsePalette := False;
  for dy := 0 to ASize - 1 do
  begin
    y0 := dy * ASrc.Height div ASize;
    y1 := (dy + 1) * ASrc.Height div ASize;
    if y1 <= y0 then y1 := y0 + 1;
    for dx := 0 to ASize - 1 do
    begin
      x0 := dx * ASrc.Width div ASize;
      x1 := (dx + 1) * ASrc.Width div ASize;
      if x1 <= x0 then x1 := x0 + 1;

      sr := 0; sg := 0; sb := 0; sa := 0; n := 0;
      for sy := y0 to y1 - 1 do
        for sx := x0 to x1 - 1 do
        begin
          C := ASrc.Colors[sx, sy];
          Inc(sr, Int64(C.Red) * C.Alpha div alphaOpaque);
          Inc(sg, Int64(C.Green) * C.Alpha div alphaOpaque);
          Inc(sb, Int64(C.Blue) * C.Alpha div alphaOpaque);
          Inc(sa, C.Alpha);
          Inc(n);
        end;

      if sa = 0 then
        C := colTransparent
      else
      begin
        C.Red := sr * alphaOpaque div sa;
        C.Green := sg * alphaOpaque div sa;
        C.Blue := sb * alphaOpaque div sa;
        C.Alpha := sa div n;
      end;
      Result.Colors[dx, dy] := C;
    end;
  end;
end;

function McxIconPng(const AName: string; ASize: Integer): TPortableNetworkGraphic;
var
  Res: TResourceStream;
  Raw, Small: TFPMemoryImage;
  Reader: TFPReaderPNG;
  Writer: TFPWriterPNG;
  Mem: TMemoryStream;
begin
  Result := nil;
  if ASize < 1 then ASize := 16;
  Res := nil;
  Raw := nil;
  Small := nil;
  Reader := nil;
  Writer := nil;
  Mem := nil;
  try
    try
      { A name with no artwork is an ordinary answer, not an error: the
        chevrons are drawn, and a half-built resource should degrade to a
        blank slot rather than stop the program. }
      Res := TResourceStream.Create(HInstance,
        McxIconResPrefix + UpperCase(AName), RT_RCDATA);
    except
      Exit;
    end;

    Raw := TFPMemoryImage.Create(0, 0);
    Reader := TFPReaderPNG.Create;
    Raw.LoadFromStream(Res, Reader);

    Small := ResampleImage(Raw, ASize);

    { Back out through a PNG rather than poking at a TBitmap: the LCL reads a
      PNG'"'"'s alpha correctly on every widget set, whereas assembling a 32-bit
      TBitmap by hand means caring which byte order the current one uses. }
    Mem := TMemoryStream.Create;
    Writer := TFPWriterPNG.Create;
    Writer.UseAlpha := True;
    Writer.WordSized := False;
    Small.SaveToStream(Mem, Writer);
    Mem.Position := 0;

    Result := TPortableNetworkGraphic.Create;
    try
      Result.LoadFromStream(Mem);
    except
      FreeAndNil(Result);
    end;
  finally
    Mem.Free;
    Writer.Free;
    Small.Free;
    Reader.Free;
    Raw.Free;
    Res.Free;
  end;
end;

var
  FGlyph: TBitmap = nil;

{ Copies artwork into the shared glyph bitmap.  A TPortableNetworkGraphic is
  a TCustomBitmap but not a TBitmap, and TSpeedButton.Glyph wants the
  latter. }
function GlyphFromArt(AArt: TPortableNetworkGraphic; ASize: Integer): TBitmap;
begin
  if FGlyph = nil then FGlyph := TBitmap.Create;
  FGlyph.SetSize(ASize, ASize);
  FGlyph.Assign(AArt);
  Result := FGlyph;
end;

function McxIconBitmap(const AName: string; AColour: TColor;
  ASize: Integer): TBitmap;
var
  Art: TPortableNetworkGraphic;
begin
  if ASize < 1 then ASize := 16;

  { Artwork first, for the same reason the image list takes it first: since
    the SVG set arrived McxDrawIcon knows only the two chevrons, so a caller
    asking here for "add" got a blank bitmap and a button with nothing on
    it. }
  Art := McxIconPng(AName, ASize);
  if Art <> nil then
    try
      Result := GlyphFromArt(Art, ASize);
      Exit;
    finally
      Art.Free;
    end;

  { One bitmap reused for every call: Glyph.Assign copies, so nothing outside
    keeps a reference, and this avoids leaking one per button.  Resized when
    the caller asks for a different size, which on a scaled display it does.
    This was fixed at sixteen, so the file browser's navigation buttons grew
    with the rest of the pane and kept a sixteen-pixel glyph rattling about
    inside them. }
  if FGlyph = nil then
  begin
    FGlyph := TBitmap.Create;
    FGlyph.PixelFormat := pf24bit;
  end;
  if (FGlyph.Width <> ASize) or (FGlyph.Height <> ASize) then
    FGlyph.SetSize(ASize, ASize);
  FGlyph.Canvas.Brush.Color := MaskColour;
  FGlyph.Canvas.Brush.Style := bsSolid;
  FGlyph.Canvas.FillRect(0, 0, ASize, ASize);
  FGlyph.Canvas.AntialiasingMode := amOff;
  McxDrawIcon(FGlyph, AName, AColour);
  FGlyph.TransparentColor := MaskColour;
  FGlyph.Transparent := True;
  Result := FGlyph;
end;

function McxBuildIconList(AImages: TImageList; const ANames: array of string;
  AColour: TColor): TImageList;
var
  Png: TPortableNetworkGraphic;
  Bmp: TBitmap;
  i: Integer;
begin
  Result := AImages;
  AImages.Clear;
  for i := 0 to High(ANames) do
  begin
    Png := McxIconPng(ANames[i], AImages.Width);
    if Png <> nil then
      try
        AImages.Add(Png, nil);
        Continue;
      finally
        Png.Free;
      end;

    { No artwork under that name, so it is one of the drawn glyphs. }
    Bmp := TBitmap.Create;
    try
      { 24-bit, not 32.  AddMasked compares whole pixels, and a 32-bit
        bitmap carries an alpha byte that the canvas leaves at zero while
        the mask colour is spelled with alpha 255, so nothing ever matches
        and every icon keeps a solid magenta square behind it. }
      Bmp.PixelFormat := pf24bit;
      Bmp.SetSize(AImages.Width, AImages.Height);
      Bmp.Canvas.Brush.Color := MaskColour;
      Bmp.Canvas.Brush.Style := bsSolid;
      Bmp.Canvas.FillRect(0, 0, Bmp.Width, Bmp.Height);

      { Antialiasing has to stay off for the same reason.  A masked bitmap
        is transparent only where the pixel matches exactly, so smoothed
        edges would blend the glyph into the mask colour and leave a magenta
        fringe around it. }
      Bmp.Canvas.AntialiasingMode := amOff;

      McxDrawIcon(Bmp, ANames[i], AColour);
      AImages.AddMasked(Bmp, MaskColour);
    finally
      Bmp.Free;
    end;
  end;
end;


function McxSaveIconSheet(const AFileName: string; ASize: Integer;
  AColour: TColor): Boolean;
const
  Cols = 8;
  Pad  = 6;
var
  Names: TStringArray;
  Sheet, Glyph: TBitmap;
  Png, Art: TPortableNetworkGraphic;
  i, Rows, Cell, X, Y: Integer;
begin
  Result := False;
  if ASize < 1 then ASize := 32;
  Names := McxIconNames;
  if Length(Names) = 0 then Exit;
  Cell := ASize + Pad * 2;
  Rows := (Length(Names) + Cols - 1) div Cols;

  Sheet := TBitmap.Create;
  Png := TPortableNetworkGraphic.Create;
  try
    Sheet.PixelFormat := pf24bit;
    Sheet.SetSize(Cols * Cell, Rows * Cell);
    Sheet.Canvas.Brush.Color := clWhite;
    Sheet.Canvas.Brush.Style := bsSolid;
    Sheet.Canvas.FillRect(0, 0, Sheet.Width, Sheet.Height);

    for i := 0 to High(Names) do
    begin
      { Artwork where there is artwork, the drawn glyph otherwise, so the
        sheet shows exactly what the toolbar will show -- including a blank
        cell if a name has neither, which is the point of looking at it.

        A bitmap per cell, not McxIconBitmap's shared one: that caches a
        transparency mask on first use, so reusing it drew glyph zero in every
        cell.  Painted on white rather than the mask colour, because a contact
        sheet is looked at rather than masked. }
      X := (i mod Cols) * Cell + Pad;
      Y := (i div Cols) * Cell + Pad;
      Art := McxIconPng(Names[i], ASize);
      if Art <> nil then
        try
          Sheet.Canvas.Draw(X, Y, Art);
        finally
          Art.Free;
        end
      else
      begin
        Glyph := TBitmap.Create;
        try
          Glyph.PixelFormat := pf24bit;
          Glyph.SetSize(ASize, ASize);
          Glyph.Canvas.Brush.Color := clWhite;
          Glyph.Canvas.Brush.Style := bsSolid;
          Glyph.Canvas.FillRect(0, 0, ASize, ASize);
          Glyph.Canvas.AntialiasingMode := amOff;
          McxDrawIcon(Glyph, Names[i], AColour);
          Sheet.Canvas.Draw(X, Y, Glyph);
        finally
          Glyph.Free;
        end;
      end;
    end;

    Png.Assign(Sheet);
    Png.SaveToFile(AFileName);
    Result := True;
  finally
    Png.Free;
    Sheet.Free;
  end;
end;

finalization
  FGlyph.Free;



end.
