{ mcxstudio2 - Toolbar, menu and shape-tree icons, drawn rather than shipped.

  Ported from the sibling Lazarus project led, whose header explains the
  reasoning: bundling a PNG set means artwork to license, to scale for HiDPI
  and to keep in step with the actions.  Drawing the icons from a handful of
  primitives instead means nothing to install, they follow the requested size
  exactly, and adding one is a short case branch rather than a trip to an
  image editor.

  Every icon is designed on a nominal 16x16 grid and scaled to the size the
  image list asks for, so the same code serves 16, 24 and 32 pixel toolbars.
  Call McxBuildIconList once the display scale is known -- see McxScale96 in
  mcxdpi -- and again if it changes. }
unit mcxicons;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Graphics, Controls, ImgList;

const
  McxWindowIconRes = 'MCXICONPNG';   { see packaging/windows/led.rc }

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

{ Draws one icon into ABitmap, which must already be sized. }
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
  IconNames: array[0..29] of string = (
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
    'add', 'delete', 'about'
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
    puts them where they belong at the real size.  Everything below is written
    against this, so no icon has to know how big it is being drawn. }
  TPen16 = object
    C: TCanvas;
    S: Double;          // pixels per design unit
    procedure Init(ACanvas: TCanvas; ASize: Integer; AColour: TColor);
    function X(V: Double): Integer;
    procedure Line(X1, Y1, X2, Y2: Double);
    procedure Box(X1, Y1, X2, Y2: Double; AFill: Boolean = False);
    procedure Ellipse(X1, Y1, X2, Y2: Double; AFill: Boolean = False);
    procedure Poly(const APts: array of Double; AFill: Boolean = False);
    procedure Colour(AColour: TColor);
    procedure Width(AUnits: Double);
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

procedure TPen16.Colour(AColour: TColor);
begin
  C.Pen.Color := AColour;
  C.Brush.Color := AColour;
end;

procedure TPen16.Width(AUnits: Double);
begin
  C.Pen.Width := Round(AUnits * S);
  if C.Pen.Width < 1 then C.Pen.Width := 1;
end;

procedure TPen16.Line(X1, Y1, X2, Y2: Double);
begin
  C.Line(X(X1), X(Y1), X(X2), X(Y2));
end;

procedure TPen16.Box(X1, Y1, X2, Y2: Double; AFill: Boolean);
begin
  if AFill then C.Brush.Style := bsSolid else C.Brush.Style := bsClear;
  C.Rectangle(X(X1), X(Y1), X(X2), X(Y2));
  C.Brush.Style := bsClear;
end;

procedure TPen16.Ellipse(X1, Y1, X2, Y2: Double; AFill: Boolean);
begin
  if AFill then C.Brush.Style := bsSolid else C.Brush.Style := bsClear;
  C.Ellipse(X(X1), X(Y1), X(X2), X(Y2));
  C.Brush.Style := bsClear;
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

{ A cube drawn in oblique projection: the shared base of every domain glyph.
  AFront fills the near face so that a solid shape reads differently from a
  region marker at 16 pixels, where shading is the only cue that survives. }
procedure DrawCube(var P: TPen16; AFill: Boolean = False);
begin
  P.Poly([2, 6, 2, 13.5, 9.5, 13.5, 9.5, 6, 2, 6], AFill);
  P.Poly([2, 6, 6.5, 2.5, 14, 2.5, 9.5, 6]);
  P.Line(14, 2.5, 14, 10);
  P.Line(9.5, 13.5, 14, 10);
end;

{ A stack of three slabs, rotated by the caller to mean X, Y or Z. }
procedure DrawStack(var P: TPen16; AVertical: Boolean);
var
  i: Integer;
begin
  for i := 0 to 2 do
    if AVertical then
      P.Box(2.5, 2.5 + i * 4, 13.5, 5.5 + i * 4)
    else
      P.Box(2.5 + i * 4, 2.5, 5.5 + i * 4, 13.5);
end;

procedure McxDrawIcon(ABitmap: TBitmap; const AName: string; AColour: TColor);
var
  P: TPen16;
  N: string;
begin
  P.Init(ABitmap.Canvas, ABitmap.Width, AColour);
  N := LowerCase(AName);

  case N of
    { ---- session and files ---------------------------------------------- }
    'new':
      begin
        P.Poly([3.5, 1.5, 3.5, 14.5, 12.5, 14.5, 12.5, 5, 9, 1.5, 3.5, 1.5]);
        P.Poly([9, 1.5, 9, 5, 12.5, 5]);
      end;
    'open':
      begin
        P.Poly([1.5, 13.5, 1.5, 3.5, 6, 3.5, 7.5, 5.5, 12.5, 5.5, 12.5, 7.5]);
        P.Poly([1.5, 13.5, 4.5, 7.5, 15, 7.5, 12, 13.5, 1.5, 13.5]);
      end;
    'save':
      begin
        { A floppy disk: still the only universally read save glyph. }
        P.Box(2, 2, 14, 14);
        P.Box(5, 2, 11, 6, True);
        P.Box(4, 9, 12, 14);
      end;
    'saveas':
      begin
        P.Box(2, 2, 11, 11);
        P.Box(4.5, 2, 8.5, 5, True);
        P.Line(11, 14, 14.5, 10.5);
        P.Line(14.5, 10.5, 12.5, 14.5);
      end;

    { ---- running --------------------------------------------------------- }
    'run':
      P.Poly([4, 2.5, 13.5, 8, 4, 13.5, 4, 2.5], True);
    'stop':
      P.Box(3.5, 3.5, 12.5, 12.5, True);
    'gpu':
      begin
        { A die with pins: the board, not a monitor, so it is not confused
          with the preview glyph. }
        P.Box(3.5, 3.5, 12.5, 12.5);
        P.Box(6, 6, 10, 10, True);
        P.Line(6, 1.5, 6, 3.5);
        P.Line(10, 1.5, 10, 3.5);
        P.Line(6, 12.5, 6, 14.5);
        P.Line(10, 12.5, 10, 14.5);
        P.Line(1.5, 6, 3.5, 6);
        P.Line(1.5, 10, 3.5, 10);
        P.Line(12.5, 6, 14.5, 6);
        P.Line(12.5, 10, 14.5, 10);
      end;
    'bench':
      begin
        { A bar chart: the built-in benchmark list. }
        P.Line(2, 14, 14, 14);
        P.Box(3, 9, 5.5, 14, True);
        P.Box(6.75, 5, 9.25, 14, True);
        P.Box(10.5, 2.5, 13, 14, True);
      end;

    { ---- modes and view -------------------------------------------------- }
    'wizard':
      begin
        { A wand with a spark: the guided path. }
        P.Line(2.5, 13.5, 10.5, 5.5);
        P.Line(12.5, 1.5, 12.5, 5);
        P.Line(10.75, 3.25, 14.25, 3.25);
        P.Line(11.3, 2.05, 13.7, 4.45);
        P.Line(13.7, 2.05, 11.3, 4.45);
      end;
    'expert':
      begin
        { Sliders: every setting at once. }
        P.Line(2, 4, 14, 4);
        P.Line(2, 8, 14, 8);
        P.Line(2, 12, 14, 12);
        P.Ellipse(4, 2.5, 6.5, 5.5, True);
        P.Ellipse(9, 6.5, 11.5, 9.5, True);
        P.Ellipse(5.5, 10.5, 8, 13.5, True);
      end;
    'preview':
      begin
        DrawCube(P);
        P.Ellipse(6.5, 7, 8.5, 9, True);
      end;
    'reset':
      begin
        { An open circular arrow. }
        P.Poly([12.5, 5, 12.5, 8, 10, 10.5, 6, 10.5, 3.5, 8, 3.5, 5, 6, 2.5, 10, 2.5]);
        P.Poly([10, 0.5, 12.5, 2.5, 10, 4.5]);
      end;
    'fit':
      begin
        { Corner brackets: zoom to extents. }
        P.Poly([2, 5.5, 2, 2, 5.5, 2]);
        P.Poly([10.5, 2, 14, 2, 14, 5.5]);
        P.Poly([14, 10.5, 14, 14, 10.5, 14]);
        P.Poly([5.5, 14, 2, 14, 2, 10.5]);
        P.Box(5.5, 5.5, 10.5, 10.5);
      end;

    { ---- shape primitives ------------------------------------------------ }
    'grid':
      begin
        P.Box(2, 2, 14, 14);
        P.Line(6, 2, 6, 14);
        P.Line(10, 2, 10, 14);
        P.Line(2, 6, 14, 6);
        P.Line(2, 10, 14, 10);
      end;
    'box':
      DrawCube(P);
    'subgrid':
      begin
        DrawCube(P);
        P.Box(4, 8, 7.5, 11.5, True);
      end;
    'sphere':
      begin
        P.Ellipse(2, 2, 14, 14);
        { Two ellipses turn a flat disc into a sphere. }
        P.Ellipse(2, 6, 14, 10);
        P.Line(8, 2, 8, 14);
      end;
    'cylinder':
      begin
        P.Ellipse(3.5, 1.5, 12.5, 5);
        P.Line(3.5, 3.25, 3.5, 12.75);
        P.Line(12.5, 3.25, 12.5, 12.75);
        P.Ellipse(3.5, 11, 12.5, 14.5);
      end;
    'xlayers':
      DrawStack(P, False);
    'ylayers':
      DrawStack(P, True);
    'zlayers':
      begin
        { Depth reads as an oblique stack rather than a flat one. }
        P.Poly([2, 9, 6, 5.5, 14, 5.5, 10, 9, 2, 9]);
        P.Poly([2, 12.5, 6, 9, 14, 9, 10, 12.5, 2, 12.5]);
      end;
    'xslabs':
      begin
        P.Box(2.5, 2.5, 5.5, 13.5, True);
        P.Box(9, 2.5, 12, 13.5);
      end;
    'yslabs':
      begin
        P.Box(2.5, 2.5, 13.5, 5.5, True);
        P.Box(2.5, 9, 13.5, 12);
      end;
    'zslabs':
      begin
        P.Poly([2, 8, 6, 4.5, 14, 4.5, 10, 8, 2, 8], True);
        P.Poly([2, 13, 6, 9.5, 14, 9.5, 10, 13, 2, 13]);
      end;

    { ---- optodes and media ----------------------------------------------- }
    'source':
      begin
        { A point with rays leaving it, pointing the way the launch does. }
        P.Ellipse(6, 1.5, 10, 5.5, True);
        P.Line(8, 5.5, 8, 14);
        P.Line(8, 14, 5, 10.5);
        P.Line(8, 14, 11, 10.5);
      end;
    'detector':
      begin
        { An open half-shell facing the surface. }
        P.Poly([2.5, 5, 2.5, 8, 8, 13.5, 13.5, 8, 13.5, 5]);
        P.Line(2.5, 5, 13.5, 5);
        P.Ellipse(6.5, 6, 9.5, 9);
      end;
    'media':
      begin
        { Stacked property rows: the media table. }
        P.Box(2, 3, 14, 13);
        P.Line(2, 6.5, 14, 6.5);
        P.Line(2, 9.75, 14, 9.75);
        P.Line(6, 3, 6, 13);
      end;

    { ---- editing --------------------------------------------------------- }
    'add':
      begin
        P.Line(8, 3, 8, 13);
        P.Line(3, 8, 13, 8);
      end;
    'delete':
      begin
        P.Line(4, 4, 12, 12);
        P.Line(12, 4, 4, 12);
      end;
    'about':
      begin
        P.Ellipse(2, 2, 14, 14);
        P.Line(8, 6.5, 8, 11.5);
        P.Line(8, 4.25, 8, 4.75);
      end;
  end;
end;

var
  FGlyph: TBitmap = nil;

function McxIconBitmap(const AName: string; AColour: TColor;
  ASize: Integer): TBitmap;
begin
  if ASize < 1 then ASize := 16;
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
  Bmp: TBitmap;
  i: Integer;
begin
  Result := AImages;
  AImages.Clear;
  for i := 0 to High(ANames) do
  begin
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
        edges would blend the icon into the mask colour and leave a magenta
        fringe around every glyph.  At 16 pixels crisp is the better
        trade anyway. }
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
  Png: TPortableNetworkGraphic;
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
      { A bitmap per cell, not McxIconBitmap's shared one: that caches a
        transparency mask on first use, so reusing it drew glyph zero in every
        cell.  Painted on white rather than the mask colour, because a contact
        sheet is looked at rather than masked. }
      Glyph := TBitmap.Create;
      try
        Glyph.PixelFormat := pf24bit;
        Glyph.SetSize(ASize, ASize);
        Glyph.Canvas.Brush.Color := clWhite;
        Glyph.Canvas.Brush.Style := bsSolid;
        Glyph.Canvas.FillRect(0, 0, ASize, ASize);
        Glyph.Canvas.AntialiasingMode := amOff;
        McxDrawIcon(Glyph, Names[i], AColour);
        X := (i mod Cols) * Cell + Pad;
        Y := (i div Cols) * Cell + Pad;
        Sheet.Canvas.Draw(X, Y, Glyph);
      finally
        Glyph.Free;
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
