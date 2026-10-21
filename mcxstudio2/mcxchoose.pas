unit mcxchoose;

{ mcxstudio2 - a row of picture tiles to choose one of a few things from.

  A radio group is the right control for a list of words.  It is the wrong
  one for a list of kinds of thing: "mcx / mcxcl / mmc / mcx-hip" asks the
  reader to already know what those are, and no amount of relabelling fixes
  that, because the difference between them is a difference between pictures
  -- a grid, a tetrahedron, a curved surface -- and not between names.

  So: one tile per choice, each an icon over a caption, laid out left to
  right and wrapping.  Painted rather than assembled out of widgets, because
  a tile is a picture with a label under it and hit-testing a rectangle is
  less code than keeping twelve controls in step. }

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, Controls, StdCtrls, ExtCtrls, Graphics, Types,
  FPImage, IntfGraphics,
  mcxdpi, mcxicons, mcxtheme;

type
  TMcxTile = record
    Value: string;
    Caption: string;
    Icon: string;
    Hint: string;
    Enabled: Boolean;
    Why: string;      { why not, when it is not }
    Art: TPortableNetworkGraphic;
    { The same picture at a third of its opacity, for when the tile cannot be
      chosen.  Built once beside the full-strength one. }
    Dim: TPortableNetworkGraphic;
    Rect: TRect;
  end;

  TMcxChooser = class
  private
    FPanel: TPanel;
    FTitle: TLabel;
    FTiles: array of TMcxTile;
    FIndex: Integer;
    FHot: Integer;
    FIconSize: Integer;
    FOnChange: TNotifyEvent;
    procedure Paint(Sender: TObject);
    procedure MouseDown(Sender: TObject; Button: TMouseButton;
      Shift: TShiftState; X, Y: Integer);
    procedure MouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
    procedure MouseLeave(Sender: TObject);
    function  HitTest(X, Y: Integer): Integer;
    procedure Layout(C: TCanvas);
    function  GetValue: string;
    procedure SetValue(const AValue: string);
  public
    constructor Create(AHost: TWinControl; const ATitle: string);
    destructor Destroy; override;
    procedure Add(const AValue, ACaption, AIcon, AHint: string);
    { Greys a tile out and says why.  Used for the combinations that have no
      solver -- a surface mesh without an NVIDIA card -- which is better told
      by a tile you can see and cannot press than by one that is not there. }
    procedure Allow(const AValue: string; AEnabled: Boolean;
      const AWhy: string = '');
    function  IsAllowed(const AValue: string): Boolean;
    { The first enabled value at or after the current one, so a choice that
      has just become impossible does not stay selected. }
    function  FirstAllowed: string;
    procedure Repaint;
    property Value: string read GetValue write SetValue;
    property Panel: TPanel read FPanel;
    property OnChange: TNotifyEvent read FOnChange write FOnChange;
  end;

implementation

const
  IconAt96 = 44;     { the icon, at 96 dpi }
  PadAt96  = 8;
  TextAt96 = 36;     { room for two lines of caption }

{ A copy at a third of the original's opacity.  Alpha rather than a blend
  towards the background, so it stays right when the theme changes. }
function Faded(ASource: TPortableNetworkGraphic): TPortableNetworkGraphic;
var
  x, y: Integer;
  C: TFPColor;
  Img: TLazIntfImage;
begin
  Result := nil;
  if ASource = nil then Exit;
  Result := TPortableNetworkGraphic.Create;
  Result.Assign(ASource);
  Img := Result.CreateIntfImage;
  try
    for y := 0 to Img.Height - 1 do
      for x := 0 to Img.Width - 1 do
      begin
        C := Img.Colors[x, y];
        C.Alpha := C.Alpha div 3;
        Img.Colors[x, y] := C;
      end;
    Result.LoadFromIntfImage(Img);
  finally
    Img.Free;
  end;
end;

constructor TMcxChooser.Create(AHost: TWinControl; const ATitle: string);
begin
  FIndex := -1;
  FHot := -1;
  FIconSize := McxScale96(IconAt96);

  FPanel := TPanel.Create(AHost);
  FPanel.Parent := AHost;
  FPanel.Align := alTop;
  FPanel.BevelOuter := bvNone;
  FPanel.Caption := '';
  FPanel.Height := McxScale96(IconAt96 + TextAt96 + PadAt96 * 3 + 16);
  FPanel.OnPaint := @Paint;
  FPanel.OnMouseDown := @MouseDown;
  FPanel.OnMouseMove := @MouseMove;
  FPanel.OnMouseLeave := @MouseLeave;
  FPanel.ShowHint := True;

  FTitle := TLabel.Create(FPanel);
  FTitle.Parent := FPanel;
  FTitle.Align := alTop;
  FTitle.Caption := ATitle;
  FTitle.BorderSpacing.Left := McxScale96(4);
  FTitle.BorderSpacing.Bottom := McxScale96(2);
end;

destructor TMcxChooser.Destroy;
var
  i: Integer;
begin
  for i := 0 to High(FTiles) do
  begin
    FTiles[i].Art.Free;
    FTiles[i].Dim.Free;
  end;
  inherited Destroy;
end;

procedure TMcxChooser.Add(const AValue, ACaption, AIcon, AHint: string);
var
  n: Integer;
begin
  n := Length(FTiles);
  SetLength(FTiles, n + 1);
  FTiles[n].Value := AValue;
  FTiles[n].Caption := ACaption;
  FTiles[n].Icon := AIcon;
  FTiles[n].Hint := AHint;
  FTiles[n].Enabled := True;
  FTiles[n].Why := '';
  { Rasterised once, at the size the tiles will draw it. }
  FTiles[n].Art := McxIconPng(AIcon, FIconSize);
  FTiles[n].Dim := Faded(FTiles[n].Art);
  if FIndex < 0 then FIndex := n;
end;

procedure TMcxChooser.Allow(const AValue: string; AEnabled: Boolean;
  const AWhy: string);
var
  i: Integer;
begin
  for i := 0 to High(FTiles) do
    if FTiles[i].Value = AValue then
    begin
      FTiles[i].Enabled := AEnabled;
      FTiles[i].Why := AWhy;
    end;
  FPanel.Invalidate;
end;

function TMcxChooser.IsAllowed(const AValue: string): Boolean;
var
  i: Integer;
begin
  for i := 0 to High(FTiles) do
    if FTiles[i].Value = AValue then Exit(FTiles[i].Enabled);
  Result := False;
end;

function TMcxChooser.FirstAllowed: string;
var
  i: Integer;
begin
  if (FIndex >= 0) and (FIndex <= High(FTiles)) and FTiles[FIndex].Enabled then
    Exit(FTiles[FIndex].Value);
  for i := 0 to High(FTiles) do
    if FTiles[i].Enabled then Exit(FTiles[i].Value);
  Result := '';
end;

function TMcxChooser.GetValue: string;
begin
  if (FIndex >= 0) and (FIndex <= High(FTiles)) then Result := FTiles[FIndex].Value
  else Result := '';
end;

procedure TMcxChooser.SetValue(const AValue: string);
var
  i: Integer;
begin
  for i := 0 to High(FTiles) do
    if FTiles[i].Value = AValue then
    begin
      FIndex := i;
      FPanel.Invalidate;
      Exit;
    end;
end;

procedure TMcxChooser.Repaint;
begin
  FPanel.Invalidate;
end;

{ Tiles across the panel, wrapping when they run out of room.

  Wide enough for the longest caption and no wider, so the tile that is
  chosen is a badge around its own picture rather than a slab of colour a
  quarter of the card across.  Recomputed on every paint: it is a dozen
  rectangles, and the panel is resized by the docking and by the DPI sweep
  without telling anyone. }
procedure TMcxChooser.Layout(C: TCanvas);
var
  i, w, h, x, y, PerRow, Rows, Wanted: Integer;
begin
  if Length(FTiles) = 0 then Exit;
  h := FIconSize + McxScale96(TextAt96 + PadAt96);

  { Wide enough for the longest caption on one line, unless that would take
    more than its share of the row -- then it wraps to two, which is what the
    caption band is two lines tall for. }
  Wanted := FIconSize + McxScale96(PadAt96 * 2);
  for i := 0 to High(FTiles) do
  begin
    w := C.TextWidth(FTiles[i].Caption) + McxScale96(PadAt96 * 2);
    if w > Wanted then Wanted := w;
  end;
  w := Min(Wanted, Max(FIconSize + McxScale96(PadAt96 * 2),
                       FPanel.ClientWidth div Length(FTiles)));

  PerRow := Max(1, FPanel.ClientWidth div w);
  if PerRow > Length(FTiles) then PerRow := Length(FTiles);
  Rows := (Length(FTiles) + PerRow - 1) div PerRow;

  FPanel.Height := FTitle.Height + McxScale96(PadAt96) + Rows * h;
  for i := 0 to High(FTiles) do
  begin
    x := (i mod PerRow) * w;
    y := FTitle.Height + McxScale96(PadAt96) + (i div PerRow) * h;
    FTiles[i].Rect := Rect(x, y, x + w, y + h);
  end;
end;

procedure TMcxChooser.Paint(Sender: TObject);
var
  i, cx: Integer;
  C: TCanvas;
  R: TRect;
  Face, Word_: TColor;
  Art: TPortableNetworkGraphic;
  Style: TTextStyle;
begin
  FillChar(Style, SizeOf(Style), 0);
  Style.Alignment := taCenter;
  Style.Layout := tlTop;
  Style.Wordbreak := True;
  Style.SingleLine := False;
  Style.Clipping := True;
  C := FPanel.Canvas;
  C.Font := FPanel.Font;
  Layout(C);
  C.Brush.Style := bsSolid;
  C.Brush.Color := FPanel.Color;
  C.FillRect(FPanel.ClientRect);
  C.Font := FPanel.Font;

  for i := 0 to High(FTiles) do
  begin
    R := FTiles[i].Rect;
    InflateRect(R, -McxScale96(3), -McxScale96(2));

    { Three states and no more: chosen, under the pointer, neither.  The
      chosen one is filled with the accent, which is the only thing in this
      window that ever means "this one". }
    if i = FIndex then Face := McxBlend(McxBase, McxAccent, 34)
    else if (i = FHot) and FTiles[i].Enabled then
      Face := McxBlend(McxBase, McxText, 12)
    else Face := clNone;

    if Face <> clNone then
    begin
      C.Brush.Color := Face;
      C.Pen.Color := Face;
      C.AntialiasingMode := amOn;
      C.RoundRect(R.Left, R.Top, R.Right, R.Bottom,
        McxScale96(10), McxScale96(10));
    end;

    cx := (R.Left + R.Right - FIconSize) div 2;
    { A tile that cannot be chosen is drawn faint rather than left out: the
      choice still exists, it is this combination that does not, and a gap
      where a tile used to be does not say that. }
    if FTiles[i].Enabled then Art := FTiles[i].Art else Art := FTiles[i].Dim;
    if Art <> nil then C.Draw(cx, R.Top + McxScale96(PadAt96), Art);

    if not FTiles[i].Enabled then Word_ := McxBlend(McxBase, McxText, 42)
    else if i = FIndex then Word_ := McxReadable(Face)
    else Word_ := McxText;
    C.Font.Color := Word_;
    C.Brush.Style := bsClear;
    R.Top := R.Top + McxScale96(PadAt96) + FIconSize + McxScale96(2);
    C.TextRect(R, R.Left, R.Top, FTiles[i].Caption, Style);
  end;
end;

function TMcxChooser.HitTest(X, Y: Integer): Integer;
var
  i: Integer;
begin
  for i := 0 to High(FTiles) do
    if PtInRect(FTiles[i].Rect, Point(X, Y)) then Exit(i);
  Result := -1;
end;

procedure TMcxChooser.MouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
var
  i: Integer;
begin
  if Button <> mbLeft then Exit;
  i := HitTest(X, Y);
  if (i < 0) or (i = FIndex) or (not FTiles[i].Enabled) then Exit;
  FIndex := i;
  FPanel.Invalidate;
  if Assigned(FOnChange) then FOnChange(Self);
end;

procedure TMcxChooser.MouseMove(Sender: TObject; Shift: TShiftState;
  X, Y: Integer);
var
  i: Integer;
begin
  i := HitTest(X, Y);
  if i <> FHot then
  begin
    FHot := i;
    FPanel.Invalidate;
  end;
  { The hint follows the pointer from tile to tile, which one hint on the
    panel cannot do -- so it is replaced as the pointer moves. }
  if i >= 0 then
  begin
    if FTiles[i].Enabled then FPanel.Hint := FTiles[i].Hint
    else FPanel.Hint := FTiles[i].Why;
  end
  else FPanel.Hint := '';
end;

procedure TMcxChooser.MouseLeave(Sender: TObject);
begin
  if FHot >= 0 then
  begin
    FHot := -1;
    FPanel.Invalidate;
  end;
end;

end.
