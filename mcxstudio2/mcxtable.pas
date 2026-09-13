{ mcxstudio2 - the two editable tables: optical properties, and detectors.

  Everything else on the form is one control per setting, placed in the
  designer and joined to a path by mcxdoc's table.  These two cannot be: how
  many media a simulation has, and how many detectors, is not known until a
  file is opened.  So they are built here -- which is the whole of what the
  plan meant by "built in code: only what is genuinely variable-length".

  One class serves both, because the shape is the same: a JSON array of
  objects, one row each, and a fixed set of fields across. }
unit mcxtable;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Controls, ExtCtrls, Buttons, Grids, Graphics, fpjson,
  mcxdoc, mcxicons;

type
  { A grid over an array of objects in the document.

    Columns are given as a spec string: 'mua,mus,g,n' names four scalar
    fields, and 'Pos[0],Pos[1],Pos[2],R' mixes elements of an array-valued
    field with a scalar one.  Writing the column list as text keeps a table
    one line to declare, the same way a binding is. }
  TMcxTable = class
  private
    FGrid: TStringGrid;
    FBar: TPanel;
    FAdd: TSpeedButton;
    FDel: TSpeedButton;
    FDoc: TMcxDoc;
    FPath: string;
    FKeys: TStringList;
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
    property OnChange: TNotifyEvent read FOnChange write FOnChange;
  end;

implementation

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
      Exit;
    end;
    FGrid.RowCount := A.Count + 1;
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

  { Writing a zero into each of the new row's cells is what creates the row:
    the document builds whatever a path needs on the way to its leaf, arrays
    included.  A row carrying every column is editable straight away rather
    than only once something has been typed into it. }
  for c := 0 to FKeys.Count - 1 do
    FDoc.SetNum(Format('%s[%d].%s', [FPath, n, FKeys[c]]), 0);

  Reload;
  FGrid.Row := FGrid.RowCount - 1;
  Changed;
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

end.
