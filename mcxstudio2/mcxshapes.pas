{ mcxstudio2 - the Shapes editor: a list of commands, and the one selected.

  Shapes is not a set of objects, it is a sequence of commands rasterised in
  order, each painting into the grid over whatever came before -- so two boxes
  in the same place are the second one, and an Origin moves the frame for
  everything after it.  That is why this is an ordered list with Up and Down
  rather than a tree or a set: the order is part of the meaning.

  The types come from mcx's own ShapeTags[] (mcx_shapes.c:47) rather than from
  the first Studio's menu, which is the same rule the binding table follows --
  mcx grows shapes faster than a GUI grows dialogs, and the list of what it
  reads should come from the thing that reads it.  Fourteen of them, declared
  one line each below; adding one is a line, not a form.

  The first Studio put each shape in a modal TValueListEditor fed a
  "name=value|name=value" string, which knew nothing about the type it was
  editing: nothing said a Sphere needs O and R, so it could not default them,
  name them or check them.  Here the fields are declared, so the editor for a
  Cylinder is two end points and a radius with those words next to them. }
unit mcxshapes;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Types, Controls, Forms, StdCtrls, ExtCtrls, Buttons,
  Menus, Grids, Graphics, StrUtils, fpjson, mcxdoc, mcxicons, mcxdpi;

type
  { How a command carries its parameters.

    Most are an object of named fields -- {"Sphere":{"O":[..],"R":10}} -- but
    three shapes are not, and mcx reads them as they are: Name is a bare
    string, Origin is a bare three-vector, and the Layers commands are a bare
    list of [from, to, tag] rows with no object and no separate Tag. }
  TMcxShapeKind = (skObject, skText, skVector, skRows);

  TMcxShapeDef = record
    Name: string;
    Kind: TMcxShapeKind;
    { For skObject, one entry per field: Key:Count:Title:Default, where Count
      is how many numbers the field holds and Default is that many of them.
      A Count of nought is a field that is a list of rows rather than a fixed
      set of boxes -- a Slab bound is one pair or several (mcx_shapes.c:553)
      -- and then Title is the column headings.
      For skVector and skRows, one entry describing the bare value. }
    Fields: string;
    { What the command does, for the hint on its menu entry. }
    Says: string;
  end;

  TMcxShapeSelect = procedure(Sender: TObject; AIndex: Integer) of object;

  { The editor.  Builds itself into a panel the designer placed, the way the
    media and detector tables do. }
  TMcxShapes = class
  private
    FHost: TWinControl;
    FDoc: TMcxDoc;
    FList: TListBox;
    FBar: TPanel;
    FAdd, FDel, FUp, FDown: TSpeedButton;
    FMenu: TPopupMenu;
    FProps: TScrollBox;
    FTitle: TLabel;
    FNote: TLabel;
    FGrid: TStringGrid;
    FGridAdd, FGridDel: TSpeedButton;
    FGridPath: string;
    FGridCols: Integer;
    FEdits: array of record
      Ctl: TWinControl;
      Path: string;
    end;
    FLoading: Integer;
    FOnChange: TNotifyEvent;
    FOnSelect: TMcxShapeSelect;
    function  Arr: TJSONArray;
    function  Count: Integer;
    function  NameOf(AIndex: Integer): string;
    function  Base(AIndex: Integer): string;
    function  Summary(AIndex: Integer): string;
    procedure ClearProps;
    procedure BuildProps;
    procedure BuildGrid(const APath: string; ACols: Integer;
      const ATitles: string; ATop: Integer);
    procedure ReloadGrid;
    procedure EditChanged(Sender: TObject);
    procedure ListClick(Sender: TObject);
    procedure AddClick(Sender: TObject);
    procedure AddType(Sender: TObject);
    procedure DelClick(Sender: TObject);
    procedure MoveClick(Sender: TObject);
    procedure GridEdited(Sender: TObject; ACol, ARow: Integer;
      const AValue: string);
    procedure GridAddClick(Sender: TObject);
    procedure GridDelClick(Sender: TObject);
    procedure Changed;
  public
    constructor Create(AHost: TWinControl);
    procedure Attach(ADoc: TMcxDoc);
    { The document changed underneath -- reread the list and the editor. }
    procedure Reload;
    { Selects a command by its place in the array, for the 3-D view to call
      when one of its solids is clicked. }
    procedure SelectIndex(AIndex: Integer);
    function  SelectedIndex: Integer;
    property OnChange: TNotifyEvent read FOnChange write FOnChange;
    property OnSelect: TMcxShapeSelect read FOnSelect write FOnSelect;
  end;

{ Every command mcx knows, in the order its own table lists them. }
function McxShapeCount: Integer;
function McxShapeDef(AIndex: Integer): TMcxShapeDef;
function McxShapeFind(const AName: string; out ADef: TMcxShapeDef): Boolean;

implementation

const
  { mcx_shapes.c:47.  Defaults are the first Studio's, which are the ones a
    person has seen before, adjusted only where it had none. }
  Defs: array[0..13] of TMcxShapeDef = (
    (Name: 'Name';   Kind: skText;   Fields: '';
     Says: 'A label for the domain.  Rasterises nothing.'),
    (Name: 'Origin'; Kind: skVector; Fields: 'Origin:3:Origin (x, y, z):0 0 0';
     Says: 'Moves the frame every command after it is measured in.'),
    (Name: 'Grid';   Kind: skObject;
     Fields: 'Size:3:Size (x, y, z):60 60 60|Tag:1:Medium label:1';
     Says: 'The whole domain, filled with one medium.  Usually first.'),
    (Name: 'Subgrid'; Kind: skObject;
     Fields: 'O:3:Corner (x, y, z):30 30 30|Size:3:Size (x, y, z):10 10 10|' +
             'Tag:1:Medium label:1';
     Says: 'A rectangular block of the grid, by corner and size.'),
    (Name: 'Sphere'; Kind: skObject;
     Fields: 'O:3:Centre (x, y, z):30 30 30|R:1:Radius:10|Tag:1:Medium label:1';
     Says: 'A sphere.'),
    (Name: 'Box';    Kind: skObject;
     Fields: 'O:3:Corner (x, y, z):30 30 30|Size:3:Size (x, y, z):10 10 10|' +
             'Tag:1:Medium label:1';
     Says: 'A box, by its lowest corner and its size.'),
    (Name: 'XSlabs'; Kind: skObject;
     Fields: 'Bound:0:From,To:1 10|Tag:1:Medium label:1';
     Says: 'Slabs across x, one row per pair of planes.'),
    (Name: 'YSlabs'; Kind: skObject;
     Fields: 'Bound:0:From,To:1 10|Tag:1:Medium label:1';
     Says: 'Slabs across y, one row per pair of planes.'),
    (Name: 'ZSlabs'; Kind: skObject;
     Fields: 'Bound:0:From,To:1 10|Tag:1:Medium label:1';
     Says: 'Slabs across z, one row per pair of planes.'),
    (Name: 'XLayers'; Kind: skRows; Fields: 'From,To,Medium label:1 10 1';
     Says: 'Several slabs across x at once: one row each.'),
    (Name: 'YLayers'; Kind: skRows; Fields: 'From,To,Medium label:1 10 1';
     Says: 'Several slabs across y at once: one row each.'),
    (Name: 'ZLayers'; Kind: skRows; Fields: 'From,To,Medium label:1 10 1';
     Says: 'Several slabs across z at once: one row each.'),
    (Name: 'Cylinder'; Kind: skObject;
     Fields: 'C0:3:End 1 (x, y, z):30 30 0|C1:3:End 2 (x, y, z):30 30 60|' +
             'R:1:Radius:5|Tag:1:Medium label:1';
     Says: 'A cylinder, by the centres of its two ends and a radius.'),
    (Name: 'UpperSpace'; Kind: skObject;
     Fields: 'Coef:4:A, B, C, D:1 -1 0 0|Tag:1:Medium label:1';
     Says: 'Everything on one side of the plane Ax + By + Cz > D.')
  );

  RowH    = 30;
  LabelW  = 150;
  BarH    = 34;

function McxShapeCount: Integer;
begin
  Result := Length(Defs);
end;

function McxShapeDef(AIndex: Integer): TMcxShapeDef;
begin
  Result := Defs[AIndex];
end;

function McxShapeFind(const AName: string; out ADef: TMcxShapeDef): Boolean;
var
  i: Integer;
begin
  for i := 0 to High(Defs) do
    if Defs[i].Name = AName then
    begin
      ADef := Defs[i];
      Exit(True);
    end;
  Result := False;
end;

{ One field of a command, split out of its spec entry. }
procedure SplitField(const AEntry: string; out AKey, ATitle, ADefault: string;
  out ACount: Integer);
var
  P: TStringList;
begin
  AKey := '';
  ATitle := '';
  ADefault := '';
  ACount := 1;
  P := TStringList.Create;
  try
    P.StrictDelimiter := True;
    P.Delimiter := ':';
    P.DelimitedText := AEntry;
    if P.Count > 0 then AKey := P[0];
    if P.Count > 1 then ACount := StrToIntDef(P[1], 1);
    if P.Count > 2 then ATitle := P[2];
    if P.Count > 3 then ADefault := P[3];
  finally
    P.Free;
  end;
end;

{ The entries of a spec, as a list the caller frees. }
function SplitSpec(const ASpec: string): TStringList;
begin
  Result := TStringList.Create;
  Result.StrictDelimiter := True;
  Result.Delimiter := '|';
  Result.DelimitedText := ASpec;
end;

{ ------------------------------------------------------------------------- }

constructor TMcxShapes.Create(AHost: TWinControl);
var
  Left: TPanel;
begin
  FHost := AHost;
  FLoading := 0;

  { Two columns: the sequence on the left, the one selected on the right.
    A splitter would be a third thing to remember the position of; the list
    needs a fixed share and the editor the rest. }
  Left := TPanel.Create(AHost);
  Left.Parent := AHost;
  Left.Align := alLeft;
  Left.Width := McxScale96(260);
  Left.BevelOuter := bvNone;
  Left.ParentColor := True;

  { Top before Align: alTop siblings are ordered by the Top they have when
    they are aligned, and two left at zero come out in whatever order the
    control list happens to hold them. }
  FBar := TPanel.Create(Left);
  FBar.Parent := Left;
  FBar.Top := 10000;
  FBar.Align := alBottom;
  FBar.Height := McxScale96(BarH);
  FBar.BevelOuter := bvNone;
  FBar.ParentColor := True;

  FList := TListBox.Create(Left);
  FList.Parent := Left;
  FList.Top := 0;
  FList.Align := alClient;
  FList.OnClick := @ListClick;

  FAdd := TSpeedButton.Create(FBar);
  FAdd.Parent := FBar;
  FAdd.Left := 0;
  FAdd.Align := alLeft;
  FAdd.Width := McxScale96(BarH);
  FAdd.Flat := True;
  FAdd.Glyph := McxIconBitmap('add', clBtnText, McxScale96(20));
  FAdd.Hint := 'Add a shape command';
  FAdd.ShowHint := True;
  FAdd.OnClick := @AddClick;

  FDel := TSpeedButton.Create(FBar);
  FDel.Parent := FBar;
  FDel.Left := 100;
  FDel.Align := alLeft;
  FDel.Width := McxScale96(BarH);
  FDel.Flat := True;
  FDel.Glyph := McxIconBitmap('delete', clBtnText, McxScale96(20));
  FDel.Hint := 'Remove the selected command';
  FDel.ShowHint := True;
  FDel.OnClick := @DelClick;

  { Up and Down are not decoration.  A command paints over whatever came
    before it, so moving one is the only way to say which of two overlapping
    shapes wins. }
  FUp := TSpeedButton.Create(FBar);
  FUp.Parent := FBar;
  FUp.Left := 200;
  FUp.Align := alLeft;
  FUp.Width := McxScale96(BarH);
  FUp.Flat := True;
  FUp.Caption := #$E2#$96#$B2;
  FUp.Hint := 'Move earlier: a later command paints over this one';
  FUp.ShowHint := True;
  FUp.Tag := -1;
  FUp.OnClick := @MoveClick;

  FDown := TSpeedButton.Create(FBar);
  FDown.Parent := FBar;
  FDown.Left := 300;
  FDown.Align := alLeft;
  FDown.Width := McxScale96(BarH);
  FDown.Flat := True;
  FDown.Caption := #$E2#$96#$BC;
  FDown.Hint := 'Move later: this command paints over the one before it';
  FDown.ShowHint := True;
  FDown.Tag := 1;
  FDown.OnClick := @MoveClick;

  FMenu := TPopupMenu.Create(AHost);

  FProps := TScrollBox.Create(AHost);
  FProps.Parent := AHost;
  FProps.Align := alClient;
  FProps.BorderStyle := bsNone;
  FProps.ParentColor := True;
  { A row is as wide as the pane by construction, so there is never anything
    to scroll sideways to. }
  FProps.HorzScrollBar.Visible := False;

  FTitle := TLabel.Create(FProps);
  FTitle.Parent := FProps;
  FTitle.Top := 0;
  FTitle.Align := alTop;
  FTitle.BorderSpacing.Around := McxScale96(8);
  FTitle.Font.Style := [fsBold];
  FTitle.Caption := '';

  FNote := TLabel.Create(FProps);
  FNote.Parent := FProps;
  FNote.Top := 1;
  FNote.Align := alTop;
  FNote.BorderSpacing.Left := McxScale96(8);
  FNote.BorderSpacing.Bottom := McxScale96(6);
  FNote.WordWrap := True;
  FNote.Caption := '';
end;

procedure TMcxShapes.Attach(ADoc: TMcxDoc);
var
  i: Integer;
  Item: TMenuItem;
begin
  FDoc := ADoc;

  FMenu.Items.Clear;
  for i := 0 to High(Defs) do
  begin
    Item := TMenuItem.Create(FMenu);
    Item.Caption := Defs[i].Name;
    Item.Hint := Defs[i].Says;
    Item.Tag := i;
    Item.OnClick := @AddType;
    FMenu.Items.Add(Item);
  end;

  Reload;
end;

function TMcxShapes.Arr: TJSONArray;
var
  D: TJSONData;
begin
  Result := nil;
  if FDoc = nil then Exit;
  D := FDoc.Find('Shapes');
  if (D <> nil) and (D.JSONType = jtArray) then Result := TJSONArray(D);
end;

function TMcxShapes.Count: Integer;
var
  A: TJSONArray;
begin
  A := Arr;
  if A = nil then Result := 0 else Result := A.Count;
end;

{ The one key a command object carries is the command's name. }
function TMcxShapes.NameOf(AIndex: Integer): string;
var
  A: TJSONArray;
  D: TJSONData;
begin
  Result := '';
  A := Arr;
  if (A = nil) or (AIndex < 0) or (AIndex >= A.Count) then Exit;
  D := A.Items[AIndex];
  if (D.JSONType <> jtObject) or (TJSONObject(D).Count < 1) then Exit;
  Result := TJSONObject(D).Names[0];
end;

function TMcxShapes.Base(AIndex: Integer): string;
begin
  Result := Format('Shapes[%d].%s', [AIndex, NameOf(AIndex)]);
end;

{ The line in the list: what the command is, and enough of what it says to
  tell two of the same kind apart without selecting each. }
function TMcxShapes.Summary(AIndex: Integer): string;
var
  Def: TMcxShapeDef;
  Spec: TStringList;
  Key, Title, Deft, S, B: string;
  n, i, k: Integer;
begin
  S := NameOf(AIndex);
  Result := Format('%d.  %s', [AIndex + 1, S]);
  if not McxShapeFind(S, Def) then Exit;
  B := Base(AIndex);

  case Def.Kind of
    skText:
      Result := Result + '   ' + FDoc.AsStr(B, '');
    skVector:
      begin
        SplitField(Def.Fields, Key, Title, Deft, n);
        S := '';
        for k := 0 to n - 1 do
          S := S + Format('%g ', [FDoc.AsNum(Format('%s[%d]', [B, k]), 0)]);
        Result := Result + '   ' + Trim(S);
      end;
    skRows:
      begin
        k := 0;
        if FDoc.Find(B) is TJSONArray then k := TJSONArray(FDoc.Find(B)).Count;
        if (k > 0) and (TJSONArray(FDoc.Find(B)).Items[0].JSONType <> jtArray) then
          k := 1;
        Result := Result + Format('   %d row(s)', [k]);
      end;
  else
    begin
      Spec := SplitSpec(Def.Fields);
      try
        S := '';
        if FDoc.Exists(B + '.Tag') then
          S := Format('tag %d  ', [FDoc.AsInt(B + '.Tag', 0)]);
        for i := 0 to Spec.Count - 1 do
        begin
          SplitField(Spec[i], Key, Title, Deft, n);
          if (n = 1) and (Key <> 'Tag') then
            S := S + Format('%s %g  ', [LowerCase(Key), FDoc.AsNum(B + '.' + Key, 0)]);
        end;
        Result := Result + '   ' + Trim(S);
      finally
        Spec.Free;
      end;
    end;
  end;
end;

procedure TMcxShapes.Reload;
var
  i, Was: Integer;
begin
  if FDoc = nil then Exit;
  Was := FList.ItemIndex;
  Inc(FLoading);
  try
    FList.Items.BeginUpdate;
    FList.Items.Clear;
    for i := 0 to Count - 1 do FList.Items.Add(Summary(i));
    FList.Items.EndUpdate;
    if (Was >= 0) and (Was < FList.Items.Count) then FList.ItemIndex := Was
    else if FList.Items.Count > 0 then FList.ItemIndex := 0;
  finally
    Dec(FLoading);
  end;
  BuildProps;
end;

function TMcxShapes.SelectedIndex: Integer;
begin
  Result := FList.ItemIndex;
end;

procedure TMcxShapes.SelectIndex(AIndex: Integer);
begin
  if (AIndex < 0) or (AIndex >= FList.Items.Count) then Exit;
  if FList.ItemIndex = AIndex then Exit;
  FList.ItemIndex := AIndex;
  BuildProps;
end;

procedure TMcxShapes.ClearProps;
var
  i: Integer;
begin
  for i := FProps.ControlCount - 1 downto 0 do
    if (FProps.Controls[i] <> FTitle) and (FProps.Controls[i] <> FNote) then
      FProps.Controls[i].Free;
  SetLength(FEdits, 0);
  FGrid := nil;
  FGridAdd := nil;
  FGridDel := nil;
  FGridPath := '';
end;

{ One row of the editor: a caption and ACount boxes side by side, each one
  bound to a path in the document exactly as a setting on the form is. }
procedure TMcxShapes.BuildProps;
var
  Def: TMcxShapeDef;
  Spec: TStringList;
  Key, Title, Deft, B, P: string;
  i, k, n, Top: Integer;
  Row: TPanel;
  Cap: TLabel;
  Ed: TEdit;
  Box: TPanel;
begin
  ClearProps;
  if FDoc = nil then Exit;

  i := FList.ItemIndex;
  if (i < 0) or (i >= Count) then
  begin
    FTitle.Caption := '';
    FNote.Caption := 'No shape selected.';
    Exit;
  end;

  B := Base(i);
  if not McxShapeFind(NameOf(i), Def) then
  begin
    FTitle.Caption := NameOf(i);
    FNote.Caption := 'mcx does not list this command, so there is nothing '
      + 'here to edit it with.  It is left exactly as the file had it.';
    Exit;
  end;

  FTitle.Caption := Def.Name;
  FNote.Caption := Def.Says;

  { Stacking.  alTop siblings are ordered by the Top they hold when the
    aligner runs, and the aligner overwrites the Top of everything it packs
    -- so by the time a second row is created the first is no longer at the
    small number it was given but at its real position further down.  A row
    built with Top 2 after that sorts above a row already sitting at 40, and
    the editor comes out upside down with its heading underneath it.

    So all three ranks are numbered from well past anything the pane can be,
    and only their order matters.  The heading needs it as much as the rows
    do: it carries a BorderSpacing, so it packs to 8 rather than to 0, and a
    note re-asserted at 1 sorted above it.  Fourth time this rule has
    bitten; it is the one thing about Align worth knowing. }
  FTitle.Top := 1000;
  FNote.Top := 2000;
  Top := 10000;

  if Def.Kind = skText then
  begin
    Row := TPanel.Create(FProps);
    Row.Parent := FProps;
    Row.Top := Top;
    Row.Align := alTop;
    Row.Height := McxScale96(RowH);
    Row.BevelOuter := bvNone;
    Row.ParentColor := True;
    Row.BorderSpacing.Around := McxScale96(4);

    Cap := TLabel.Create(Row);
    Cap.Parent := Row;
    Cap.Align := alLeft;
    Cap.Layout := tlCenter;
    Cap.Caption := 'Name';
    Cap.Constraints.MinWidth := McxScale96(LabelW);

    Ed := TEdit.Create(Row);
    Ed.Parent := Row;
    Ed.Align := alClient;
    Ed.Text := FDoc.AsStr(B, '');
    Ed.OnChange := @EditChanged;
    SetLength(FEdits, 1);
    FEdits[0].Ctl := Ed;
    FEdits[0].Path := B;
    Exit;
  end;

  if Def.Kind = skRows then
  begin
    BuildGrid(B, 3, Def.Fields, Top);
    Exit;
  end;

  Spec := SplitSpec(Def.Fields);
  try
    for i := 0 to Spec.Count - 1 do
    begin
      SplitField(Spec[i], Key, Title, Deft, n);

      { A field that is a list of rows rather than a fixed set of boxes. }
      if n = 0 then
      begin
        BuildGrid(B + '.' + Key, WordCount(Title, [',']),
          Title + ':' + Deft, Top + i * 10);
        Continue;
      end;

      Row := TPanel.Create(FProps);
      Row.Parent := FProps;
      Row.Top := Top + i * 10;
      Row.Align := alTop;
      Row.Height := McxScale96(RowH);
      Row.BevelOuter := bvNone;
      Row.ParentColor := True;
      Row.BorderSpacing.Around := McxScale96(4);

      Cap := TLabel.Create(Row);
      Cap.Parent := Row;
      Cap.Align := alLeft;
      Cap.Layout := tlCenter;
      Cap.Caption := Title;
      Cap.Constraints.MinWidth := McxScale96(LabelW);

      Box := TPanel.Create(Row);
      Box.Parent := Row;
      Box.Align := alClient;
      Box.BevelOuter := bvNone;
      Box.ParentColor := True;
      Box.ChildSizing.ControlsPerLine := n;
      Box.ChildSizing.Layout := cclLeftToRightThenTopToBottom;
      Box.ChildSizing.HorizontalSpacing := McxScale96(4);
      Box.ChildSizing.EnlargeHorizontal := crsHomogenousChildResize;

      for k := 0 to n - 1 do
      begin
        if Def.Kind = skVector then P := Format('%s[%d]', [B, k])
        else if n = 1 then P := B + '.' + Key
        else P := Format('%s.%s[%d]', [B, Key, k]);

        Ed := TEdit.Create(Box);
        Ed.Parent := Box;
        Ed.Text := FDoc.AsStr(P, '');
        Ed.OnChange := @EditChanged;
        SetLength(FEdits, Length(FEdits) + 1);
        FEdits[High(FEdits)].Ctl := Ed;
        FEdits[High(FEdits)].Path := P;
      end;

      { skVector is the whole value, so there is only ever one row of it. }
      if Def.Kind = skVector then Break;
    end;
  finally
    Spec.Free;
  end;
end;

{ The Layers commands are a list of rows rather than a set of fields, so they
  get a grid.  Same reason the media table is one: how many there are is not
  known until the file is open. }
procedure TMcxShapes.BuildGrid(const APath: string; ACols: Integer;
  const ATitles: string; ATop: Integer);
var
  Titles: TStringList;
  Bar: TPanel;
  i: Integer;
  Head: string;
begin
  FGridPath := APath;
  FGridCols := ACols;

  { The grid before its button bar, not after.  A control created second is
    ranked against the position the first has already been packed to, so
    building the bar first put it above the thing it acts on. }
  FGrid := TStringGrid.Create(FProps);
  FGrid.Parent := FProps;
  FGrid.Top := ATop;
  FGrid.Align := alTop;
  FGrid.Height := McxScale96(150);
  FGrid.BorderSpacing.Around := McxScale96(4);
  FGrid.FixedRows := 1;
  FGrid.FixedCols := 0;
  FGrid.ColCount := ACols;
  FGrid.RowCount := 1;
  { The columns share the width rather than leaving a field of grey beside
    two narrow ones, which also takes the horizontal scroll bar away. }
  FGrid.AutoFillColumns := True;
  FGrid.ScrollBars := ssAutoVertical;
  FGrid.Options := FGrid.Options + [goEditing, goRowSelect];
  FGrid.OnSetEditText := @GridEdited;

  Bar := TPanel.Create(FProps);
  Bar.Parent := FProps;
  Bar.Top := ATop + 5;
  Bar.Align := alTop;
  Bar.Height := McxScale96(BarH);
  Bar.BevelOuter := bvNone;
  Bar.ParentColor := True;

  Head := ATitles;
  i := Pos(':', Head);
  if i > 0 then Head := Copy(Head, 1, i - 1);
  Titles := TStringList.Create;
  try
    Titles.StrictDelimiter := True;
    Titles.Delimiter := ',';
    Titles.DelimitedText := Head;
    for i := 0 to ACols - 1 do
      if i < Titles.Count then FGrid.Cells[i, 0] := Titles[i];
  finally
    Titles.Free;
  end;

  FGridAdd := TSpeedButton.Create(Bar);
  FGridAdd.Parent := Bar;
  FGridAdd.Left := 0;
  FGridAdd.Align := alLeft;
  FGridAdd.Width := McxScale96(BarH);
  FGridAdd.Flat := True;
  FGridAdd.Glyph := McxIconBitmap('add', clBtnText, McxScale96(20));
  FGridAdd.Hint := 'Add a layer';
  FGridAdd.ShowHint := True;
  FGridAdd.OnClick := @GridAddClick;

  FGridDel := TSpeedButton.Create(Bar);
  FGridDel.Parent := Bar;
  FGridDel.Left := 100;
  FGridDel.Align := alLeft;
  FGridDel.Width := McxScale96(BarH);
  FGridDel.Flat := True;
  FGridDel.Glyph := McxIconBitmap('delete', clBtnText, McxScale96(20));
  FGridDel.Hint := 'Remove the selected layer';
  FGridDel.ShowHint := True;
  FGridDel.OnClick := @GridDelClick;

  ReloadGrid;
  { Known defect: a dead horizontal scroll bar is drawn across the foot of
    the grid whatever this says.  It survives ssNone, it survives
    AutoFillColumns being off, and it survives the scroll box's own
    horizontal bar being hidden, so it is not what it looks like and it is
    not worth more of a hunt than that until it is the worst thing here. }
  FGrid.ScrollBars := ssAutoVertical;
end;

{ mcx takes a Layers value as either one row or a list of them
  (mcx_shapes.c:663), and a file that used the short form is not rewritten
  into the long one just because it was looked at. }
procedure TMcxShapes.ReloadGrid;
var
  A: TJSONData;
  Rows, r, c: Integer;
  Flat: Boolean;
begin
  if FGrid = nil then Exit;
  A := FDoc.Find(FGridPath);
  Rows := 0;
  Flat := False;
  if A is TJSONArray then
  begin
    Rows := TJSONArray(A).Count;
    Flat := (Rows > 0) and (TJSONArray(A).Items[0].JSONType <> jtArray);
    if Flat then Rows := 1;
  end;

  Inc(FLoading);
  try
    FGrid.RowCount := Rows + 1;
    for r := 0 to Rows - 1 do
      for c := 0 to FGridCols - 1 do
        if Flat then
          FGrid.Cells[c, r + 1] := FDoc.AsStr(Format('%s[%d]', [FGridPath, c]), '')
        else
          FGrid.Cells[c, r + 1] :=
            FDoc.AsStr(Format('%s[%d][%d]', [FGridPath, r, c]), '');
  finally
    Dec(FLoading);
  end;
end;

procedure TMcxShapes.GridEdited(Sender: TObject; ACol, ARow: Integer;
  const AValue: string);
var
  A: TJSONData;
  Flat: Boolean;
  P: string;
  V: Double;
begin
  if (FLoading > 0) or (FDoc = nil) or (ARow < 1) then Exit;
  A := FDoc.Find(FGridPath);
  Flat := (A is TJSONArray) and (TJSONArray(A).Count > 0) and
          (TJSONArray(A).Items[0].JSONType <> jtArray);
  if Flat then P := Format('%s[%d]', [FGridPath, ACol])
  else P := Format('%s[%d][%d]', [FGridPath, ARow - 1, ACol]);
  if TryStrToFloat(Trim(AValue), V) then
  begin
    FDoc.SetNum(P, V);
    Changed;
  end;
end;

procedure TMcxShapes.GridAddClick(Sender: TObject);
var
  A: TJSONData;
  r, c: Integer;
  Flat: Boolean;
begin
  if FDoc = nil then Exit;
  A := FDoc.Find(FGridPath);
  Flat := (A is TJSONArray) and (TJSONArray(A).Count > 0) and
          (TJSONArray(A).Items[0].JSONType <> jtArray);
  if Flat then
  begin
    { A second row cannot be written in the short form, so the short one is
      promoted to a list of one first. }
    FDoc.SetNum(Format('%s[0][0]', [FGridPath]), FDoc.AsNum(FGridPath + '[0]', 0));
    FDoc.SetNum(Format('%s[0][1]', [FGridPath]), FDoc.AsNum(FGridPath + '[1]', 0));
    FDoc.SetNum(Format('%s[0][2]', [FGridPath]), FDoc.AsNum(FGridPath + '[2]', 0));
    r := 1;
  end
  else if A is TJSONArray then r := TJSONArray(A).Count
  else r := 0;

  for c := 0 to FGridCols - 1 do
    FDoc.SetNum(Format('%s[%d][%d]', [FGridPath, r, c]), 0);
  ReloadGrid;
  Changed;
end;

procedure TMcxShapes.GridDelClick(Sender: TObject);
var
  r: Integer;
begin
  if (FDoc = nil) or (FGrid = nil) then Exit;
  r := FGrid.Row - 1;
  if r < 0 then Exit;
  FDoc.Delete(Format('%s[%d]', [FGridPath, r]));
  ReloadGrid;
  Changed;
end;

procedure TMcxShapes.EditChanged(Sender: TObject);
var
  i: Integer;
  S: string;
  V: Double;
begin
  if (FLoading > 0) or (FDoc = nil) then Exit;
  for i := 0 to High(FEdits) do
    if FEdits[i].Ctl = Sender then
    begin
      S := Trim(TEdit(Sender).Text);
      { A string field takes the text; everything else is a number, and a box
        being retyped passes through states that are not one -- "-", "1e" --
        which are left alone rather than written as zero. }
      if FEdits[i].Path = Base(FList.ItemIndex) then FDoc.SetStr(FEdits[i].Path, S)
      else if TryStrToFloat(S, V) then FDoc.SetNum(FEdits[i].Path, V)
      else Exit;
      Changed;
      Exit;
    end;
end;

{ The list line is rebuilt in place rather than the whole list, so that
  typing in a box does not take the selection away from under the cursor. }
procedure TMcxShapes.Changed;
var
  i: Integer;
begin
  i := FList.ItemIndex;
  if (i >= 0) and (i < FList.Items.Count) and (i < Count) then
  begin
    Inc(FLoading);
    try
      FList.Items[i] := Summary(i);
      FList.ItemIndex := i;
    finally
      Dec(FLoading);
    end;
  end;
  if Assigned(FOnChange) then FOnChange(Self);
end;

procedure TMcxShapes.ListClick(Sender: TObject);
begin
  if FLoading > 0 then Exit;
  BuildProps;
  if Assigned(FOnSelect) then FOnSelect(Self, FList.ItemIndex);
end;

procedure TMcxShapes.AddClick(Sender: TObject);
var
  P: TPoint;
begin
  P := FAdd.ClientToScreen(Point(0, FAdd.Height));
  FMenu.PopUp(P.x, P.y);
end;

{ A new command is written through the document's own paths, so it is built
  the way Ensure builds anything: writing to one past the end of the array
  creates the element, and the keys arrive in the order they are written. }
procedure TMcxShapes.AddType(Sender: TObject);
var
  Def: TMcxShapeDef;
  Spec: TStringList;
  Key, Title, Deft, B: string;
  Nums: TStringList;
  i, k, n, At: Integer;
begin
  if FDoc = nil then Exit;
  Def := Defs[TMenuItem(Sender).Tag];
  At := Count;
  B := Format('Shapes[%d].%s', [At, Def.Name]);

  case Def.Kind of
    skText: FDoc.SetStr(B, 'mcxdomain');
    skVector:
      begin
        SplitField(Def.Fields, Key, Title, Deft, n);
        Nums := TStringList.Create;
        try
          Nums.Delimiter := ' ';
          Nums.DelimitedText := Deft;
          for k := 0 to n - 1 do
            FDoc.SetNum(Format('%s[%d]', [B, k]),
              StrToFloatDef(Nums[k], 0));
        finally
          Nums.Free;
        end;
      end;
    skRows:
      begin
        Deft := Def.Fields;
        i := Pos(':', Deft);
        Deft := Copy(Deft, i + 1, Length(Deft));
        Nums := TStringList.Create;
        try
          Nums.Delimiter := ' ';
          Nums.DelimitedText := Deft;
          for k := 0 to Nums.Count - 1 do
            FDoc.SetNum(Format('%s[0][%d]', [B, k]),
              StrToFloatDef(Nums[k], 0));
        finally
          Nums.Free;
        end;
      end;
  else
    begin
      Spec := SplitSpec(Def.Fields);
      try
        for i := 0 to Spec.Count - 1 do
        begin
          SplitField(Spec[i], Key, Title, Deft, n);
          Nums := TStringList.Create;
          try
            Nums.Delimiter := ' ';
            Nums.DelimitedText := Deft;
            if n = 1 then
              FDoc.SetNum(B + '.' + Key, StrToFloatDef(Nums[0], 0))
            else
              for k := 0 to n - 1 do
                FDoc.SetNum(Format('%s.%s[%d]', [B, Key, k]),
                  StrToFloatDef(Nums[k], 0));
          finally
            Nums.Free;
          end;
        end;
      finally
        Spec.Free;
      end;
    end;
  end;

  Reload;
  SelectIndex(At);
  FList.ItemIndex := At;
  BuildProps;
  if Assigned(FOnChange) then FOnChange(Self);
  if Assigned(FOnSelect) then FOnSelect(Self, At);
end;

procedure TMcxShapes.DelClick(Sender: TObject);
var
  i: Integer;
begin
  if FDoc = nil then Exit;
  i := FList.ItemIndex;
  if (i < 0) or (i >= Count) then Exit;
  FDoc.Delete(Format('Shapes[%d]', [i]));
  Reload;
  if i >= FList.Items.Count then i := FList.Items.Count - 1;
  FList.ItemIndex := i;
  BuildProps;
  if Assigned(FOnChange) then FOnChange(Self);
  if Assigned(FOnSelect) then FOnSelect(Self, i);
end;

{ Moving a command is the only way to say which of two overlapping shapes
  wins, since the later one paints over the earlier.  fpjson will not
  reorder an array, so the two elements are extracted and put back the other
  way round -- Extract hands over ownership, which is what keeps this from
  freeing a node that is about to be reinserted. }
procedure TMcxShapes.MoveClick(Sender: TObject);
var
  A: TJSONArray;
  i, j: Integer;
  Node: TJSONData;
begin
  if FDoc = nil then Exit;
  A := Arr;
  if A = nil then Exit;
  i := FList.ItemIndex;
  j := i + TSpeedButton(Sender).Tag;
  if (i < 0) or (i >= A.Count) or (j < 0) or (j >= A.Count) then Exit;

  Node := A.Extract(i);
  A.Insert(j, Node);
  FDoc.Modified := True;

  Reload;
  FList.ItemIndex := j;
  BuildProps;
  if Assigned(FOnChange) then FOnChange(Self);
  if Assigned(FOnSelect) then FOnSelect(Self, j);
end;

end.
