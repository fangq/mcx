{ mcxstudio2 - the controls over how a result is displayed.

  These are not settings.  Nothing here is written to the .json, nothing here
  changes what a simulation computes; every one of them is a uniform in the
  volume shader, so moving one is a repaint and nothing more.  That is why
  they live beside the picture rather than in the settings pane, and why they
  are sliders when the settings deliberately are not: a threshold is found by
  dragging until the picture reads, not by knowing the number.

  The set is the one mcxcloud offers -- a colour map, a transparency
  threshold, and an x/y/z slab -- because someone arriving from the web
  viewer should not have to look for them. }
unit mcxdisp;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Controls, StdCtrls, ExtCtrls, ComCtrls, Graphics, Spin,
  mcxgl, mcxview, mcxdpi;

type
  { The strip under the 3-D view.  Owns its controls, drives one TMcxView. }
  TMcxDisplayBar = class
  private
    FView: TMcxView;
    FRoot: TPanel;
    FHead: TPanel;
    FTitle: TLabel;
    FBody: TPanel;
    FMap: TComboBox;
    FStyle: TComboBox;
    FLog: TCheckBox;
    FOpacity: TTrackBar;
    FFloor: TTrackBar;
    { Two per axis, low then high. }
    FLo: array[0..2] of TTrackBar;
    FHi: array[0..2] of TTrackBar;
    { Which photons of a trajectory to draw.  A bar each for sweeping and a
      number each for naming one exactly -- a bar over ten thousand
      identifiers cannot land on a chosen one, and "show me ray 4237" is the
      question this row exists to answer. }
    FIdRow, FIdBars: TPanel;
    FIdSpan: TCheckBox;
    FIdLoBar, FIdHiBar: TTrackBar;
    FIdLoNum, FIdHiNum: TSpinEdit;
    FIdFirst, FIdLast: Integer;
    FIdSyncing: Integer;
    FOpen: Boolean;
    FRows: Integer;
    function  AddRow(const ACaption: string): TPanel;
    function  AddSlider(AParent: TWinControl; APosition: Integer): TTrackBar;
    function  AddNumber(AParent: TWinControl; AValue: Integer): TSpinEdit;
    procedure HeadClick(Sender: TObject);
    procedure IdChanged(Sender: TObject);
    procedure PushIds(ALo, AHi: Integer);
    procedure Changed(Sender: TObject);
    procedure SlabChanged(Sender: TObject);
    procedure Layout;
  public
    constructor Create(AHost: TWinControl; AView: TMcxView);
    { Shown only when there is a volume: with no data loaded every control
      here does nothing, and a row of dead sliders reads as a broken window. }
    procedure SetHasVolume(AValue: Boolean);
    { Puts the slab back to the whole volume. }
    procedure ResetSlab;
    { Shows the photon range row and scales it to the file just loaded.
      ALast is capped at MaxPhotonId, so a run with a hundred thousand
      photons in it does not give a bar with a hundred thousand steps that
      moves a thousand at a time. }
    procedure SetTrajectory(AFirst, ALast: Integer);
    property Panel: TPanel read FRoot;
  end;

implementation

const
  { What the shader's ramp() knows, in its order. }
  MapNames = 'Jet (blue to red)'#10'Hot (black to white)'#10'Viridis'#10 +
    'Cool (cyan to magenta)'#10'Greyscale';
  StyleNames = 'Maximum intensity'#10'Accumulated (translucent)';
  AxisNames: array[0..2] of string = ('X range', 'Y range', 'Z range');
  RowHeight = 26;
  RowGap = 2;
  { How many paths may be drawn at once.  The point of the row is to pull one
    path, or a handful, out of a cloud of half a million segments that has no
    direction in it; past about this many there is nothing to see that the
    unfiltered picture did not already show. }
  MaxPhotonId = 10000;

  { mcx numbers its photons from zero; the row shows them from one, because a
    person counting paths starts at the first one. }
  IdOffset = 1;

constructor TMcxDisplayBar.Create(AHost: TWinControl; AView: TMcxView);
var
  Row: TPanel;
  i: Integer;
begin
  FView := AView;
  FOpen := True;

  FRoot := TPanel.Create(AHost);
  FRoot.Parent := AHost;
  FRoot.Align := alBottom;
  FRoot.BevelOuter := bvNone;
  FRoot.Caption := '';
  FRoot.Visible := False;

  { The header is the whole of the collapse control: a title that is also a
    button, because a separate button next to a title is two things saying
    one thing. }
  FHead := TPanel.Create(FRoot);
  FHead.Parent := FRoot;
  FHead.Top := 0;
  FHead.Align := alTop;
  FHead.Height := McxScale96(22);
  FHead.BevelOuter := bvNone;
  FHead.Caption := '';
  FHead.Cursor := crHandPoint;
  FHead.OnClick := @HeadClick;

  FTitle := TLabel.Create(FHead);
  FTitle.Parent := FHead;
  FTitle.Align := alClient;
  FTitle.Layout := tlCenter;
  FTitle.BorderSpacing.Left := McxScale96(6);
  FTitle.Font.Style := [fsBold];
  FTitle.Cursor := crHandPoint;
  FTitle.OnClick := @HeadClick;

  FBody := TPanel.Create(FRoot);
  FBody.Parent := FRoot;
  FBody.Top := 100;
  FBody.Align := alTop;
  FBody.BevelOuter := bvNone;
  FBody.Caption := '';

  Row := AddRow('Colour map');
  FMap := TComboBox.Create(Row);
  FMap.Parent := Row;
  FMap.Align := alClient;
  FMap.Style := csDropDownList;
  FMap.Text := '';
  FMap.Items.Text := MapNames;
  FMap.ItemIndex := 0;
  FMap.OnChange := @Changed;
  FMap.Hint := 'How values are coloured.  Viridis keeps its lightness even, ' +
    'so it does not invent an edge where the data has none.';
  FMap.ShowHint := True;

  Row := AddRow('Rendering');
  { Log first on the right, so the combo gets what is left. }
  FLog := TCheckBox.Create(Row);
  FLog.Parent := Row;
  FLog.Align := alRight;
  FLog.AutoSize := False;
  FLog.Width := McxScale96(86);
  FLog.Caption := 'Log scale';
  FLog.Checked := True;
  FLog.OnChange := @Changed;
  FLog.Hint := 'Fluence spans several decades; on a linear scale everything ' +
    'but the source is black.';
  FLog.ShowHint := True;

  FStyle := TComboBox.Create(Row);
  FStyle.Parent := Row;
  FStyle.Align := alClient;
  FStyle.Style := csDropDownList;
  FStyle.Items.Text := StyleNames;
  FStyle.ItemIndex := 0;
  FStyle.OnChange := @Changed;
  FStyle.Hint := 'Maximum intensity shows the brightest voxel along each ray; ' +
    'accumulated builds the ray up and needs the opacity below.';
  FStyle.ShowHint := True;

  Row := AddRow('Threshold');
  FFloor := AddSlider(Row, 0);
  FFloor.Align := alClient;
  FFloor.OnChange := @Changed;
  FFloor.Hint := 'Voxels below this fraction of the displayed range are not ' +
    'drawn at all.  A fluence map is mostly its own faint tail, and the tail ' +
    'hides what is behind it.';

  Row := AddRow('Opacity');
  FOpacity := AddSlider(Row, 25);
  FOpacity.Align := alClient;
  FOpacity.OnChange := @Changed;
  FOpacity.Hint := 'How much each sample along a ray contributes.  Only used ' +
    'by the accumulated rendering.';

  { The photon range, above the slab rows: it belongs with the paths rather
    than with the volume, and it is the one row that is not always there. }
  { Laid out by hand rather than by ChildSizing: the two numbers and the
    check box want a fixed width and only the two bars should take up the
    slack, which is one rule ChildSizing's equal shares cannot express. }
  FIdRow := AddRow('Photon IDs');

  FIdLoNum := AddNumber(FIdRow, 1);
  FIdLoNum.Align := alLeft;
  FIdLoNum.Width := McxScale96(82);
  FIdLoNum.BorderSpacing.Right := McxScale96(6);
  FIdLoNum.OnChange := @IdChanged;
  FIdLoNum.Hint := 'The photon to draw.  With "to" ticked this is the first ' +
    'of a range; without it, the only one.';

  FIdHiNum := AddNumber(FIdRow, 1);
  FIdHiNum.Left := 10000;
  FIdHiNum.Align := alRight;
  FIdHiNum.Width := McxScale96(82);
  FIdHiNum.BorderSpacing.Left := McxScale96(6);
  FIdHiNum.OnChange := @IdChanged;
  FIdHiNum.Hint := 'The last photon to draw.';

  FIdSpan := TCheckBox.Create(FIdRow);
  FIdSpan.Parent := FIdRow;
  FIdSpan.Left := 9000;
  FIdSpan.Align := alRight;
  FIdSpan.AutoSize := False;
  FIdSpan.Width := McxScale96(42);
  FIdSpan.BorderSpacing.Left := McxScale96(6);
  FIdSpan.Caption := 'to';
  FIdSpan.Checked := True;
  FIdSpan.ShowHint := True;
  FIdSpan.Hint := 'Off draws the one photon named on the left, which is the ' +
    'only way to see where a single path went.';
  FIdSpan.OnChange := @IdChanged;

  FIdBars := TPanel.Create(FIdRow);
  FIdBars.Parent := FIdRow;
  FIdBars.Align := alClient;
  FIdBars.BevelOuter := bvNone;
  FIdBars.Caption := '';
  FIdBars.ChildSizing.Layout := cclLeftToRightThenTopToBottom;
  FIdBars.ChildSizing.ControlsPerLine := 2;
  FIdBars.ChildSizing.EnlargeHorizontal := crsScaleChilds;
  FIdBars.ChildSizing.HorizontalSpacing := McxScale96(6);

  FIdLoBar := AddSlider(FIdBars, 0);
  FIdLoBar.OnChange := @IdChanged;
  FIdLoBar.Hint := FIdLoNum.Hint;
  FIdHiBar := AddSlider(FIdBars, 100);
  FIdHiBar.OnChange := @IdChanged;
  FIdHiBar.Hint := FIdHiNum.Hint;

  { Hidden until there is a trajectory; the row above it is the volume's. }
  FIdRow.Parent.Visible := False;

  for i := 0 to 2 do
  begin
    Row := AddRow(AxisNames[i]);
    { Two sliders sharing the row, left half and right half, without either
      of them having to know how wide the row is. }
    Row.ChildSizing.Layout := cclLeftToRightThenTopToBottom;
    Row.ChildSizing.ControlsPerLine := 2;
    Row.ChildSizing.EnlargeHorizontal := crsScaleChilds;
    Row.ChildSizing.HorizontalSpacing := McxScale96(6);
    FLo[i] := AddSlider(Row, 0);
    FLo[i].Tag := i;
    FLo[i].OnChange := @SlabChanged;
    FLo[i].Hint := 'The near face of the visible slab along ' +
      Copy(AxisNames[i], 1, 1) + '.';
    FHi[i] := AddSlider(Row, 100);
    FHi[i].Tag := i;
    FHi[i].OnChange := @SlabChanged;
    FHi[i].Hint := 'The far face of the visible slab along ' +
      Copy(AxisNames[i], 1, 1) + '.';
  end;

  Layout;
end;

{ A row of the body: a caption down the left at a fixed width, and whatever
  the caller adds filling the rest.  The same shape as a settings card's row,
  so the two read as one window. }
function TMcxDisplayBar.AddRow(const ACaption: string): TPanel;
var
  L: TLabel;
  Host: TPanel;
begin
  Result := TPanel.Create(FBody);
  Result.Parent := FBody;
  { Rising Top before Align: alTop siblings are ordered by the Top they have
    when they are aligned, not by the order they were created in. }
  Inc(FRows);
  Result.Top := FRows * 100;
  Result.Align := alTop;
  Result.Height := McxScale96(RowHeight);
  Result.BevelOuter := bvNone;
  Result.Caption := '';
  Result.BorderSpacing.Around := McxScale96(RowGap);

  L := TLabel.Create(Result);
  L.Parent := Result;
  L.Align := alLeft;
  L.Layout := tlCenter;
  L.AutoSize := False;
  L.Width := McxScale96(86);
  L.BorderSpacing.Left := McxScale96(6);
  L.Caption := ACaption;

  { The label is a sibling of whatever comes next, and ChildSizing on the row
    would lay it out too; so rows that use ChildSizing get a second panel to
    put their controls in.  Cheaper to always have one than to special-case. }
  Host := TPanel.Create(Result);
  Host.Parent := Result;
  Host.Align := alClient;
  Host.BevelOuter := bvNone;
  Host.Caption := '';
  Result := Host;
end;

{ A number box, for the identifier a bar cannot land on. }
function TMcxDisplayBar.AddNumber(AParent: TWinControl;
  AValue: Integer): TSpinEdit;
begin
  Result := TSpinEdit.Create(AParent);
  Result.Parent := AParent;
  Result.MinValue := 1;
  Result.MaxValue := MaxPhotonId;
  Result.Value := AValue;
  Result.Height := McxScale96(22);
  Result.ShowHint := True;
end;

function TMcxDisplayBar.AddSlider(AParent: TWinControl;
  APosition: Integer): TTrackBar;
begin
  Result := TTrackBar.Create(AParent);
  Result.Parent := AParent;
  Result.Min := 0;
  Result.Max := 100;
  Result.Position := APosition;
  Result.TickStyle := tsNone;
  Result.ShowSelRange := False;
  Result.Height := McxScale96(22);
  Result.ShowHint := True;
end;

{ The body's height is counted rather than taken from AutoSize: the rows are
  added before anything has been laid out, so at this point AutoSize has
  nothing to measure. }
procedure TMcxDisplayBar.Layout;
begin
  if FOpen then FTitle.Caption := '- Display'
  else FTitle.Caption := '+ Display';
  FBody.Height := FRows * McxScale96(RowHeight + 2 * RowGap);
  FBody.Visible := FOpen;
  if FOpen then FRoot.Height := FHead.Height + FBody.Height
  else FRoot.Height := FHead.Height;
end;

procedure TMcxDisplayBar.HeadClick(Sender: TObject);
begin
  FOpen := not FOpen;
  Layout;
end;

procedure TMcxDisplayBar.Changed(Sender: TObject);
begin
  if FView = nil then Exit;
  FView.Colormap := FMap.ItemIndex;
  FView.Style := FStyle.ItemIndex;
  FView.LogScale := FLog.Checked;
  FView.Threshold := FFloor.Position / 100;
  FView.Opacity := FOpacity.Position / 100;
  FView.Redraw;
end;

procedure TMcxDisplayBar.SlabChanged(Sender: TObject);
var
  i: Integer;
  Lo, Hi: TMcxVec3;
begin
  if FView = nil then Exit;
  { A slab with no thickness shows nothing, so the two sliders push each
    other rather than crossing. }
  i := TTrackBar(Sender).Tag;
  if FLo[i].Position > FHi[i].Position then
  begin
    if Sender = FLo[i] then FHi[i].Position := FLo[i].Position
    else FLo[i].Position := FHi[i].Position;
  end;
  Lo := McxVec3(FLo[0].Position / 100, FLo[1].Position / 100,
                FLo[2].Position / 100);
  Hi := McxVec3(FHi[0].Position / 100, FHi[1].Position / 100,
                FHi[2].Position / 100);
  FView.ClipLo := Lo;
  FView.ClipHi := Hi;
  FView.Redraw;
end;

{ One of the four moved; the other three follow it.

  Bar and number are the same value at different resolutions -- the bar to
  sweep with, the number to name one with -- so whichever was touched is the
  truth and the rest are written from it. }
procedure TMcxDisplayBar.IdChanged(Sender: TObject);
var
  Lo, Hi: Integer;
begin
  if FIdSyncing > 0 then Exit;
  Lo := FIdLoBar.Position;
  Hi := FIdHiBar.Position;
  if Sender = FIdLoNum then Lo := FIdLoNum.Value;
  if Sender = FIdHiNum then Hi := FIdHiNum.Value;

  { A range that has crossed itself shows nothing, so the bounds push each
    other rather than passing. }
  if Lo > Hi then
    if (Sender = FIdLoBar) or (Sender = FIdLoNum) then Hi := Lo else Lo := Hi;

  PushIds(Lo, Hi);
end;

procedure TMcxDisplayBar.PushIds(ALo, AHi: Integer);
begin
  if ALo < FIdFirst then ALo := FIdFirst;
  if ALo > FIdLast then ALo := FIdLast;

  { Without "to" there is no upper bound: the row names one photon, which is
    the only way to see where a single path actually went. }
  if not FIdSpan.Checked then
    AHi := ALo
  else
  begin
    if AHi > FIdLast then AHi := FIdLast;
    if AHi < ALo then AHi := ALo;
    { At most ten thousand paths at once.  The cap is on the width of the
      window rather than on where it sits, so every photon in the file can
      still be reached by moving the lower bound -- pinning the cap to the
      start of the file would have made everything past the ten-thousandth
      unselectable. }
    if AHi > ALo + MaxPhotonId - 1 then AHi := ALo + MaxPhotonId - 1;
  end;

  Inc(FIdSyncing);
  try
    FIdLoBar.Position := ALo;
    FIdHiBar.Position := AHi;
    FIdLoNum.Value := ALo;
    FIdHiNum.Value := AHi;
    { Hidden rather than disabled: a disabled spin edit on gtk2 draws as a
      filled block with its number gone, which reads as damage rather than as
      "not in use".  Going away also gives the remaining bar the full width. }
    FIdHiBar.Visible := FIdSpan.Checked;
    FIdHiNum.Visible := FIdSpan.Checked;
  finally
    Dec(FIdSyncing);
  end;
  { The controls count from one and mcx counts from zero. }
  if FView <> nil then
    FView.SetPhotonRange(ALo - IdOffset, AHi - IdOffset);
end;

procedure TMcxDisplayBar.SetTrajectory(AFirst, ALast: Integer);
var
  Bar: TTrackBar;
  Num: TSpinEdit;
  i: Integer;
begin
  FIdRow.Parent.Visible := AFirst <= ALast;
  if AFirst > ALast then Exit;
  FIdFirst := AFirst + IdOffset;
  FIdLast := ALast + IdOffset;

  Inc(FIdSyncing);
  try
    for i := 0 to 1 do
    begin
      if i = 0 then begin Bar := FIdLoBar; Num := FIdLoNum; end
      else begin Bar := FIdHiBar; Num := FIdHiNum; end;
      Bar.Min := FIdFirst;
      Bar.Max := FIdLast;
      Num.MinValue := FIdFirst;
      Num.MaxValue := FIdLast;
    end;
  finally
    Dec(FIdSyncing);
  end;
  PushIds(FIdFirst, FIdLast);
end;

procedure TMcxDisplayBar.ResetSlab;
var
  i: Integer;
begin
  for i := 0 to 2 do
  begin
    FLo[i].Position := 0;
    FHi[i].Position := 100;
  end;
end;

procedure TMcxDisplayBar.SetHasVolume(AValue: Boolean);
begin
  FRoot.Visible := AValue;
end;

end.
