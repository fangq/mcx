{ mcxstudio2 - the editable tables: optical properties, detectors, sources.

  Everything else on the form is one control per setting, placed in the
  designer and joined to a path by mcxdoc's table.  These two cannot be: how
  many media a simulation has, and how many detectors, is not known until a
  file is opened.  So they are built here -- which is the whole of what the
  plan meant by "built in code: only what is genuinely variable-length".

  One class serves media and detectors, because the shape is the same: a JSON
  array of objects, one row each, and a fixed set of fields across.  Sources
  are shaped the other way round -- four parallel arrays, Pos, Dir, Param1
  and Param2, with a row per source in each -- and get a class of their own. }
unit mcxtable;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Controls, StdCtrls, ExtCtrls, Buttons, Grids, Graphics, fpjson,
  LCLIntf, LCLType, mcxdoc, mcxicons;

type
  { A grid over an array of objects in the document.

    Columns are given as a spec string: 'mua,mus,g,n' names four scalar
    fields, and 'Pos[0],Pos[1],Pos[2],R' mixes elements of an array-valued
    field with a scalar one.  Writing the column list as text keeps a table
    one line to declare, the same way a binding is.

    A key ending in '?' is optional: a new row copies it only when the row
    above has it.  A detector's Dir is one -- mcx and mmc read it when it is
    there, and a Dir of zeros written into every new detector is not the same
    thing as no Dir at all. }
  TMcxTable = class
  private
    FGrid: TStringGrid;
    FBar: TPanel;
    FAdd: TSpeedButton;
    FDel: TSpeedButton;
    FDoc: TMcxDoc;
    FPath: string;
    FKeys: TStringList;
    FOptional: array of Boolean;
    FShown: Integer;
    FLoading: Integer;
    FOnChange: TNotifyEvent;
    function  Rows: TJSONArray;
    function  CellPath(ARow, ACol: Integer): string;
    procedure GridEdited(Sender: TObject; ACol, ARow: Integer;
      const AValue: string);
    procedure AddClick(Sender: TObject);
    procedure DelClick(Sender: TObject);
    procedure Changed;
  public
    { AHost is the card the table sits on; ATitles and ASpec are matching
      comma lists, the first what the user reads and the second where each
      column comes from. }
    constructor Create(AHost: TWinControl; const ATitles, ASpec: string;
      ARows: Integer);
    destructor Destroy; override;
    procedure Attach(ADoc: TMcxDoc; const APath: string);
    procedure Reload;
    { What the buttons and a typed cell do, callable by the self-test.
      Rows and columns count from nought, past the fixed ones. }
    procedure AddRow;
    procedure SetCell(ARow, ACol: Integer; const AText: string);
    property OnChange: TNotifyEvent read FOnChange write FOnChange;
  end;

  { A grid over parallel vectors: Optode.Source's Pos, Dir, Param1 and
    Param2, a row per source and four columns per vector.

    mcx reads each of them as either one vector -- a single source -- or a
    list of vectors, one per source, and the lists have to be the same
    length (mcx_utils.c: "Length of sub-elements of Pos/Dir/Param1/Param2
    must match").  Four separate editors could not keep that true; one grid
    whose add and remove act on every vector at once cannot break it.

    The file keeps its own shape: a single source stays a flat [x,y,z], it
    becomes a list of rows only when a second source is added, and goes back
    to flat when it is down to one again.  A vector the file does not have
    stays absent until a cell of it is typed into. }
  TMcxVecTable = class
  private
    FGrid: TStringGrid;
    FBar: TPanel;
    FAdd: TSpeedButton;
    FDel: TSpeedButton;
    FDoc: TMcxDoc;
    FPath: string;
    FVecs: TStringList;
    FWidth: Integer;
    FLoading: Integer;
    FOnChange: TNotifyEvent;
    FOnSelect: TNotifyEvent;
    function  VecPath(AVec: Integer): string;
    function  Vec(AVec: Integer): TJSONArray;
    function  IsRows(A: TJSONArray): Boolean;
    function  RowOf(AVec, ARow: Integer): TJSONArray;
    procedure ToRows(AVec, ACount: Integer);
    procedure GridEdited(Sender: TObject; ACol, ARow: Integer;
      const AValue: string);
    procedure GridSelected(Sender: TObject; ACol, ARow: Integer);
    procedure AddClick(Sender: TObject);
    procedure DelClick(Sender: TObject);
    procedure Changed;
    function  GetSelected: Integer;
    procedure SetSelected(AValue: Integer);
  public
    { AVecs names the vectors under the path, the first deciding how many
      rows there are; ATitles is one comma list across all their columns. }
    constructor Create(AHost: TWinControl; const AVecs, ATitles: string;
      AWidth, ARows: Integer);
    destructor Destroy; override;
    procedure Attach(ADoc: TMcxDoc; const APath: string);
    procedure Reload;
    { How many sources: 0 without a Pos, 1 for a flat one. }
    function  Count: Integer;
    procedure AddRow;
    procedure DeleteRow(ARow: Integer);
    procedure SetCell(ARow, ACol: Integer; const AText: string);
    property Grid: TStringGrid read FGrid;
    property Bar: TPanel read FBar;
    property Selected: Integer read GetSelected write SetSelected;
    property OnChange: TNotifyEvent read FOnChange write FOnChange;
    { The picked row changed, so whatever shows one source can follow it. }
    property OnSelect: TNotifyEvent read FOnSelect write FOnSelect;
  end;

{ Sizes a grid to show every row, at least AShown, so it never scrolls
  vertically and never has to decide it should.

  A grid left on ssAuto* finds out it needs a scroll bar while it is
  painting (TCustomGrid.Paint -> ResetSizes), and the bar arriving shrinks
  its client area, whose resize handler invalidates the grid -- inside the
  paint, which gtk2 refuses with "InvalidateRect refused invalidating during
  paint message: TStringGrid".  Adding a sixth medium was enough.  Turning
  ScrollBars to ssVertical once the rows outgrow a cap only moves the same
  thing to that moment, because the bar is still shown at the next paint.
  The settings page scrolls already, so a table as tall as its rows needs no
  bar of its own; AHorz keeps a horizontal one always present, which never
  changes either.  Called whenever the rows change, never during a paint. }
procedure McxFitGrid(AGrid: TStringGrid; AShown: Integer; AHorz: Boolean);

implementation

procedure McxFitGrid(AGrid: TStringGrid; AShown: Integer; AHorz: Boolean);
var
  Vis, H: Integer;
  Want: TScrollStyle;
begin
  Vis := AGrid.RowCount - AGrid.FixedRows;
  if Vis < AShown then Vis := AShown;
  H := AGrid.DefaultRowHeight * (Vis + AGrid.FixedRows) + 4;
  if AHorz then H := H + GetSystemMetrics(SM_CYHSCROLL);
  if AHorz then Want := ssHorizontal else Want := ssNone;
  if AGrid.ScrollBars <> Want then AGrid.ScrollBars := Want;
  if AGrid.Height <> H then AGrid.Height := H;
end;

{ Zeros up to, but not including, the element a path ending in [k] names.

  The document pads a gap in an array with nulls, which is right for a path
  it knows nothing about and wrong for a vector: typing the third element of
  a Dir that is not there yet would write [null, null, 1], and mcx reads a
  null as whatever valuedouble happens to hold.  Zeros are what an absent
  component means everywhere these tables are used. }
procedure PadVector(ADoc: TMcxDoc; const APath: string);
var
  b, k: Integer;
  A: TJSONData;
begin
  if (APath = '') or (APath[Length(APath)] <> ']') then Exit;
  b := LastDelimiter('[', APath);
  if b < 2 then Exit;
  k := StrToIntDef(Copy(APath, b + 1, Length(APath) - b - 1), -1);
  if k < 1 then Exit;
  A := ADoc.Find(Copy(APath, 1, b - 1));
  if A = nil then A := ADoc.Ensure(Copy(APath, 1, b - 1), jtArray);
  if (A = nil) or (A.JSONType <> jtArray) then Exit;
  while TJSONArray(A).Count < k do TJSONArray(A).Add(0);
end;

constructor TMcxTable.Create(AHost: TWinControl; const ATitles, ASpec: string;
  ARows: Integer);
var
  Titles: TStringList;
  i: Integer;
begin
  FKeys := TStringList.Create;
  FKeys.Delimiter := ',';
  FKeys.StrictDelimiter := True;
  FKeys.DelimitedText := ASpec;
  SetLength(FOptional, FKeys.Count);
  for i := 0 to FKeys.Count - 1 do
  begin
    FOptional[i] := (FKeys[i] <> '') and (FKeys[i][Length(FKeys[i])] = '?');
    if FOptional[i] then FKeys[i] := Copy(FKeys[i], 1, Length(FKeys[i]) - 1);
  end;

  Titles := TStringList.Create;
  try
    Titles.Delimiter := ',';
    Titles.StrictDelimiter := True;
    Titles.DelimitedText := ATitles;

    FGrid := TStringGrid.Create(AHost);
    FGrid.Parent := AHost;
    { Top before Align, and the bar below it: alTop siblings are ordered by
      the Top they have when they are aligned, so leaving both at zero puts
      the buttons wherever the control list happens to say. }
    FGrid.Top := 100;
    FGrid.Align := alTop;
    { Raw 96-dpi: built during FormCreate, so the startup sweep scales it. }
    FGrid.Height := 22 * (ARows + 1) + 4;
    FShown := ARows;
    FGrid.ColCount := FKeys.Count + 1;
    FGrid.RowCount := 1;
    FGrid.FixedCols := 1;
    FGrid.FixedRows := 1;
    { A number down the left because the row number is the medium's tag and
      the detector's index -- the thing the rest of the file refers to. }
    FGrid.ColWidths[0] := 34;
    FGrid.Options := FGrid.Options + [goEditing, goColSizing, goThumbTracking]
      - [goRangeSelect];
    FGrid.AutoFillColumns := True;
    FGrid.OnEditingDone := nil;
    FGrid.OnSetEditText := @GridEdited;
    for i := 0 to FKeys.Count - 1 do
      if i < Titles.Count then FGrid.Cells[i + 1, 0] := Titles[i]
      else FGrid.Cells[i + 1, 0] := FKeys[i];
  finally
    Titles.Free;
  end;

  { Add and remove, under the grid.  Two buttons rather than a context menu:
    a table nobody can see how to add a row to is a table nobody adds a row
    to. }
  FBar := TPanel.Create(AHost);
  FBar.Parent := AHost;
  FBar.Top := 200;
  FBar.Align := alTop;
  FBar.Height := 26;
  FBar.BevelOuter := bvNone;
  FBar.Caption := '';

  FAdd := TSpeedButton.Create(FBar);
  FAdd.Parent := FBar;
  FAdd.SetBounds(0, 2, 22, 22);
  FAdd.Flat := True;
  FAdd.Hint := 'Add a row';
  FAdd.ShowHint := True;
  FAdd.Glyph.Assign(McxIconBitmap('add', clBtnText, 16));
  FAdd.OnClick := @AddClick;

  FDel := TSpeedButton.Create(FBar);
  FDel.Parent := FBar;
  FDel.SetBounds(26, 2, 22, 22);
  FDel.Flat := True;
  FDel.Hint := 'Remove the selected row';
  FDel.ShowHint := True;
  FDel.Glyph.Assign(McxIconBitmap('delete', clBtnText, 16));
  FDel.OnClick := @DelClick;
end;

destructor TMcxTable.Destroy;
begin
  FKeys.Free;
  inherited Destroy;
end;

procedure TMcxTable.Attach(ADoc: TMcxDoc; const APath: string);
begin
  FDoc := ADoc;
  FPath := APath;
  Reload;
end;

function TMcxTable.Rows: TJSONArray;
var
  D: TJSONData;
begin
  Result := nil;
  if FDoc = nil then Exit;
  D := FDoc.Find(FPath);
  if (D <> nil) and (D.JSONType = jtArray) then Result := TJSONArray(D);
end;

{ The path of one cell: 'Domain.Media[2].mua', or with an array-valued
  field, 'Optode.Detector[0].Pos[1]'.

  Going through a path rather than reaching into the row object is what lets
  the document keep its own rules -- an integer stays an integer, a flag
  written as true stays true, a value that has not changed is not rewritten,
  and a float prints as short as it can.  mcxdoc resolves and creates both
  index forms, which was checked before this relied on it. }
function TMcxTable.CellPath(ARow, ACol: Integer): string;
begin
  Result := Format('%s[%d].%s', [FPath, ARow, FKeys[ACol]]);
end;

procedure TMcxTable.Reload;
var
  A: TJSONArray;
  r, c: Integer;
begin
  Inc(FLoading);
  try
    A := Rows;
    if A = nil then
    begin
      FGrid.RowCount := 1;
      McxFitGrid(FGrid, FShown, False);
      Exit;
    end;
    FGrid.RowCount := A.Count + 1;
    McxFitGrid(FGrid, FShown, False);
    for r := 0 to A.Count - 1 do
    begin
      FGrid.Cells[0, r + 1] := IntToStr(r);
      for c := 0 to FKeys.Count - 1 do
        FGrid.Cells[c + 1, r + 1] := FDoc.AsStr(CellPath(r, c), '');
    end;
  finally
    Dec(FLoading);
  end;
end;

procedure TMcxTable.Changed;
begin
  if (FLoading = 0) and Assigned(FOnChange) then FOnChange(Self);
end;

procedure TMcxTable.GridEdited(Sender: TObject; ACol, ARow: Integer;
  const AValue: string);
var
  A: TJSONArray;
  V: Double;
  Fs: TFormatSettings;
begin
  if FLoading > 0 then Exit;
  if (ACol < 1) or (ARow < 1) or (FDoc = nil) then Exit;
  A := Rows;
  if (A = nil) or (ARow - 1 >= A.Count) then Exit;

  Fs := DefaultFormatSettings;
  Fs.DecimalSeparator := '.';
  { Anything that is not a number is left alone rather than written as zero:
    a half-typed '0.' is not an instruction to clear the value. }
  if not TryStrToFloat(Trim(AValue), V, Fs) then Exit;

  PadVector(FDoc, CellPath(ARow - 1, ACol - 1));
  FDoc.SetNum(CellPath(ARow - 1, ACol - 1), V);
  Changed;
end;

procedure TMcxTable.AddClick(Sender: TObject);
var
  A: TJSONArray;
  n, c: Integer;
begin
  if FDoc = nil then Exit;
  A := Rows;
  if A = nil then
  begin
    FDoc.Ensure(FPath, jtArray);
    A := Rows;
    if A = nil then Exit;
  end;
  n := A.Count;

  { Writing into each of the new row's cells is what creates the row: the
    document builds whatever a path needs on the way to its leaf, arrays
    included.  A row carrying every column is editable straight away rather
    than only once something has been typed into it.

    What is written is the row above, not zeros.  A new medium is nearly
    always a variation on the last one -- the same n, a scattering
    coefficient in the same decade -- and a row of zeros is not a medium at
    all: mua and mus of nought with a refractive index of nought is a
    material light does not propagate in.  The first Studio did the same and
    said so in its log (mcxgui.pas:2469).  The first row of all has nothing
    to copy and still gets zeros. }
  for c := 0 to FKeys.Count - 1 do
    if FOptional[c] and
       ((n = 0) or not FDoc.Exists(Format('%s[%d].%s', [FPath, n - 1, FKeys[c]]))) then
      Continue
    else if n > 0 then
      FDoc.SetNum(Format('%s[%d].%s', [FPath, n, FKeys[c]]),
                  FDoc.AsNum(Format('%s[%d].%s', [FPath, n - 1, FKeys[c]]), 0))
    else
      FDoc.SetNum(Format('%s[%d].%s', [FPath, n, FKeys[c]]), 0);

  Reload;
  FGrid.Row := FGrid.RowCount - 1;
  Changed;
end;

procedure TMcxTable.AddRow;
begin
  AddClick(nil);
end;

procedure TMcxTable.SetCell(ARow, ACol: Integer; const AText: string);
begin
  GridEdited(nil, ACol + 1, ARow + 1, AText);
end;

procedure TMcxTable.DelClick(Sender: TObject);
var
  A: TJSONArray;
  r: Integer;
begin
  A := Rows;
  if A = nil then Exit;
  r := FGrid.Row - 1;
  if (r < 0) or (r >= A.Count) then Exit;
  A.Delete(r);
  Reload;
  Changed;
end;

{ ------------------------------------------------------------ TMcxVecTable }

constructor TMcxVecTable.Create(AHost: TWinControl; const AVecs,
  ATitles: string; AWidth, ARows: Integer);
var
  Titles: TStringList;
  i: Integer;
begin
  FWidth := AWidth;
  FVecs := TStringList.Create;
  FVecs.Delimiter := ',';
  FVecs.StrictDelimiter := True;
  FVecs.DelimitedText := AVecs;

  Titles := TStringList.Create;
  try
    Titles.Delimiter := ',';
    Titles.StrictDelimiter := True;
    Titles.DelimitedText := ATitles;

    FGrid := TStringGrid.Create(AHost);
    FGrid.Parent := AHost;
    FGrid.Top := 100;
    FGrid.Align := alTop;
    FGrid.Height := 22 * (ARows + 1) + 4 + 16;
    FGrid.ColCount := FVecs.Count * FWidth + 1;
    FGrid.RowCount := 1;
    FGrid.FixedCols := 1;
    FGrid.FixedRows := 1;
    FGrid.ColWidths[0] := 34;
    { Sixteen columns do not fit a settings card, so they scroll sideways
      rather than being squeezed until no number in them can be read. }
    FGrid.AutoFillColumns := False;
    FGrid.DefaultColWidth := 72;
    FGrid.ScrollBars := ssHorizontal;
    FGrid.Options := FGrid.Options + [goEditing, goColSizing, goThumbTracking]
      - [goRangeSelect];
    FGrid.OnSetEditText := @GridEdited;
    FGrid.OnSelection := @GridSelected;
    for i := 1 to FGrid.ColCount - 1 do
      if i - 1 < Titles.Count then FGrid.Cells[i, 0] := Titles[i - 1];
  finally
    Titles.Free;
  end;

  FBar := TPanel.Create(AHost);
  FBar.Parent := AHost;
  FBar.Top := 200;
  FBar.Align := alTop;
  FBar.Height := 26;
  FBar.BevelOuter := bvNone;
  FBar.Caption := '';

  FAdd := TSpeedButton.Create(FBar);
  FAdd.Parent := FBar;
  FAdd.SetBounds(0, 2, 22, 22);
  FAdd.Flat := True;
  FAdd.Hint := 'Add a source, a copy of the last one';
  FAdd.ShowHint := True;
  FAdd.Glyph.Assign(McxIconBitmap('add', clBtnText, 16));
  FAdd.OnClick := @AddClick;

  FDel := TSpeedButton.Create(FBar);
  FDel.Parent := FBar;
  FDel.SetBounds(26, 2, 22, 22);
  FDel.Flat := True;
  FDel.Hint := 'Remove the selected source';
  FDel.ShowHint := True;
  FDel.Glyph.Assign(McxIconBitmap('delete', clBtnText, 16));
  FDel.OnClick := @DelClick;
end;

destructor TMcxVecTable.Destroy;
begin
  FVecs.Free;
  inherited Destroy;
end;

procedure TMcxVecTable.Attach(ADoc: TMcxDoc; const APath: string);
begin
  FDoc := ADoc;
  FPath := APath;
  Reload;
end;

function TMcxVecTable.VecPath(AVec: Integer): string;
begin
  Result := FPath + '.' + FVecs[AVec];
end;

function TMcxVecTable.Vec(AVec: Integer): TJSONArray;
var
  D: TJSONData;
begin
  Result := nil;
  if FDoc = nil then Exit;
  D := FDoc.Find(VecPath(AVec));
  if (D <> nil) and (D.JSONType = jtArray) then Result := TJSONArray(D);
end;

{ A list of rows rather than one vector: what the first element is says. }
function TMcxVecTable.IsRows(A: TJSONArray): Boolean;
begin
  Result := (A <> nil) and (A.Count > 0) and (A.Items[0].JSONType = jtArray);
end;

function TMcxVecTable.Count: Integer;
var
  A: TJSONArray;
begin
  A := Vec(0);
  if A = nil then Result := 0
  else if IsRows(A) then Result := A.Count
  else Result := 1;
end;

{ One source's vector, whichever shape the file holds it in; nil if absent. }
function TMcxVecTable.RowOf(AVec, ARow: Integer): TJSONArray;
var
  A: TJSONArray;
begin
  Result := nil;
  A := Vec(AVec);
  if A = nil then Exit;
  if IsRows(A) then
  begin
    if (ARow < A.Count) and (A.Items[ARow].JSONType = jtArray) then
      Result := TJSONArray(A.Items[ARow]);
  end
  else if ARow = 0 then
    Result := A;
end;

{ Makes a vector a list of ACount rows, so that it can be written a row at a
  time.  A flat vector becomes the first row and is copied into the rest -- a
  Dir given once for several sources means the same Dir for each of them --
  and an absent one becomes rows of zeros. }
procedure TMcxVecTable.ToRows(AVec, ACount: Integer);
var
  A, Row: TJSONArray;
  i: Integer;
begin
  A := Vec(AVec);
  if IsRows(A) then
  begin
    Row := TJSONArray(A.Items[A.Count - 1]);
    while A.Count < ACount do A.Add(Row.Clone);
    Exit;
  end;
  if A = nil then
  begin
    Row := TJSONArray.Create;
    for i := 1 to FWidth do Row.Add(0);
  end
  else
    Row := TJSONArray(A.Clone);
  A := TJSONArray(FDoc.Ensure(VecPath(AVec), jtArray));
  A.Clear;
  A.Add(Row);
  for i := 2 to ACount do A.Add(Row.Clone);
end;

procedure TMcxVecTable.Reload;
var
  n, r, v, k: Integer;
  Row: TJSONArray;
  E: TJSONData;
begin
  Inc(FLoading);
  try
    n := Count;
    FGrid.RowCount := n + 1;
    { A row for every source.  Sixteen columns are wider than a card nearly
      always, so the sideways bar is always there. }
    McxFitGrid(FGrid, 3, True);
    for r := 0 to n - 1 do
    begin
      { From one, as the source bar used to count: the zero mcx counts from
        is its own business. }
      FGrid.Cells[0, r + 1] := IntToStr(r + 1);
      for v := 0 to FVecs.Count - 1 do
      begin
        Row := RowOf(v, r);
        for k := 0 to FWidth - 1 do
        begin
          E := nil;
          if (Row <> nil) and (k < Row.Count) then E := Row.Items[k];
          if (E <> nil) and (E.JSONType in [jtNumber, jtString, jtBoolean]) then
            FGrid.Cells[v * FWidth + k + 1, r + 1] := E.AsString
          else
            FGrid.Cells[v * FWidth + k + 1, r + 1] := '';
        end;
      end;
    end;
  finally
    Dec(FLoading);
  end;
end;

procedure TMcxVecTable.Changed;
begin
  if (FLoading = 0) and Assigned(FOnChange) then FOnChange(Self);
end;

function TMcxVecTable.GetSelected: Integer;
begin
  Result := FGrid.Row - 1;
  if Result < 0 then Result := 0;
end;

procedure TMcxVecTable.SetSelected(AValue: Integer);
begin
  if (AValue + 1 < FGrid.RowCount) and (AValue >= 0) and
     (FGrid.Row <> AValue + 1) then
  begin
    Inc(FLoading);
    try
      FGrid.Row := AValue + 1;
    finally
      Dec(FLoading);
    end;
  end;
end;

procedure TMcxVecTable.GridSelected(Sender: TObject; ACol, ARow: Integer);
begin
  if (FLoading = 0) and Assigned(FOnSelect) then FOnSelect(Self);
end;

procedure TMcxVecTable.GridEdited(Sender: TObject; ACol, ARow: Integer;
  const AValue: string);
var
  V: Double;
  Fs: TFormatSettings;
  n, vi, k: Integer;
  Path: string;
begin
  if FLoading > 0 then Exit;
  if (ACol < 1) or (ARow < 1) or (FDoc = nil) then Exit;
  n := Count;
  if ARow - 1 >= n then Exit;
  vi := (ACol - 1) div FWidth;
  k := (ACol - 1) mod FWidth;

  Fs := DefaultFormatSettings;
  Fs.DecimalSeparator := '.';
  if not TryStrToFloat(Trim(AValue), V, Fs) then Exit;

  if n > 1 then
  begin
    { Every vector of a multi-source input is a list as long as Pos; one
      typed into for the first time joins it at that length. }
    if not IsRows(Vec(vi)) or (Vec(vi).Count < n) then ToRows(vi, n);
    Path := Format('%s[%d][%d]', [VecPath(vi), ARow - 1, k]);
  end
  else
  begin
    if Vec(vi) = nil then ToRows(vi, 1);
    if IsRows(Vec(vi)) then
      Path := Format('%s[0][%d]', [VecPath(vi), k])
    else
      Path := Format('%s[%d]', [VecPath(vi), k]);
  end;
  { Missing elements before this one are zeros, not the nulls the document
    would pad with. }
  if k > 0 then PadVector(FDoc, Path);
  FDoc.SetNum(Path, V);
  Changed;
end;

procedure TMcxVecTable.AddClick(Sender: TObject);
var
  n, v: Integer;
  A: TJSONArray;
begin
  if FDoc = nil then Exit;
  n := Count;
  if n = 0 then
  begin
    { Nothing to copy: the source every new file starts with. }
    FDoc.Ensure(VecPath(0), jtArray);
    for v := 0 to 2 do FDoc.SetNum(Format('%s[%d]', [VecPath(0), v]), 0);
  end
  else
    { Each vector the file has, the same row longer; the new source is a
      copy of the last, which is nearly always where the next one starts
      from.  Absent vectors stay absent. }
    for v := 0 to FVecs.Count - 1 do
    begin
      A := Vec(v);
      if A = nil then Continue;
      ToRows(v, n + 1);
    end;
  FDoc.Modified := True;
  Reload;
  FGrid.Row := FGrid.RowCount - 1;
  Changed;
  if Assigned(FOnSelect) then FOnSelect(Self);
end;

procedure TMcxVecTable.AddRow;
begin
  AddClick(nil);
end;

procedure TMcxVecTable.DeleteRow(ARow: Integer);
begin
  if ARow + 1 < FGrid.RowCount then FGrid.Row := ARow + 1;
  DelClick(nil);
end;

procedure TMcxVecTable.SetCell(ARow, ACol: Integer; const AText: string);
begin
  GridEdited(nil, ACol + 1, ARow + 1, AText);
end;

procedure TMcxVecTable.DelClick(Sender: TObject);
var
  n, r, v: Integer;
  A, Row: TJSONArray;
begin
  n := Count;
  r := FGrid.Row - 1;
  { A simulation needs a source, so the last one stays. }
  if (n < 2) or (r < 0) or (r >= n) then Exit;
  for v := 0 to FVecs.Count - 1 do
  begin
    A := Vec(v);
    if not IsRows(A) then Continue;
    if r < A.Count then A.Delete(r);
    { Down to one: back to the flat vector a single source is written as. }
    if (n = 2) and (A.Count = 1) and (A.Items[0].JSONType = jtArray) then
    begin
      Row := TJSONArray(A.Items[0].Clone);
      A.Clear;
      while Row.Count > 0 do A.Add(Row.Extract(0));
      Row.Free;
    end;
  end;
  FDoc.Modified := True;
  Reload;
  Changed;
  if Assigned(FOnSelect) then FOnSelect(Self);
end;

end.
