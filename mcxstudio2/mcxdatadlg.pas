{ mcxstudio2 - the Edit Volume / Edit Mesh dialog.

  A volume or a mesh is shown as the JSON it is stored as, in an editor that
  can be copied from and pasted into: that is how one moves between inputs,
  and how an array printed by another tool comes in.  Load from file puts a
  file's contents in the editor the same way, so what is about to be stored
  can be looked at first.

  Nothing is stored until OK, and OK is refused -- with the reason, and the
  text left as it was -- while what is in the editor would not make a
  volume or a mesh. }
unit mcxdatadlg;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, StdCtrls, ExtCtrls, Dialogs, Clipbrd,
  SynEdit, SynEditTypes, SynGutterLineNumber, mcxjsonsyn, mcxdpi, mcxtheme;

type
  { Whether the editor's text makes the thing being edited; AError says why
    not.  Called on OK, before the dialog may close. }
  TMcxDataCheck = function(const AText: string; out AError: string): Boolean of object;
  { A file's contents as the text the editor shows. }
  TMcxDataLoad = function(const AFileName: string; out AText, AError: string): Boolean of object;

{ Shows the dialog.  True when OK closed it, with AText what was accepted. }
function McxEditData(AOwner: TComponent; const ATitle, AHelp, AFilter: string;
  var AText: string; ACheck: TMcxDataCheck; ALoad: TMcxDataLoad): Boolean;

implementation

type
  TMcxDataForm = class(TForm)
  private
    FEdit: TSynEdit;
    FCheck: TMcxDataCheck;
    FLoad: TMcxDataLoad;
    FFilter: string;
    procedure LoadClick(Sender: TObject);
    procedure CopyClick(Sender: TObject);
    procedure PasteClick(Sender: TObject);
    procedure FormCloseQuery(Sender: TObject; var CanClose: Boolean);
  end;

procedure TMcxDataForm.LoadClick(Sender: TObject);
var
  D: TOpenDialog;
  T, E: string;
begin
  D := TOpenDialog.Create(Self);
  try
    D.Filter := FFilter;
    D.Options := D.Options + [ofFileMustExist];
    if not D.Execute then Exit;
    Screen.Cursor := crHourGlass;
    try
      if not FLoad(D.FileName, T, E) then
      begin
        Screen.Cursor := crDefault;
        MessageDlg('MCX Studio', E, mtError, [mbOK], 0);
        Exit;
      end;
    finally
      Screen.Cursor := crDefault;
    end;
    FEdit.Text := T;
  finally
    D.Free;
  end;
end;

procedure TMcxDataForm.CopyClick(Sender: TObject);
begin
  Clipboard.AsText := FEdit.Text;
end;

procedure TMcxDataForm.PasteClick(Sender: TObject);
begin
  { The whole of it: what is pasted here is a value, not a fragment. }
  FEdit.Text := Clipboard.AsText;
end;

procedure TMcxDataForm.FormCloseQuery(Sender: TObject; var CanClose: Boolean);
var
  E: string;
begin
  if ModalResult <> mrOK then Exit;
  CanClose := FCheck(FEdit.Text, E);
  if not CanClose then
    MessageDlg('MCX Studio', E, mtError, [mbOK], 0);
end;

function McxEditData(AOwner: TComponent; const ATitle, AHelp, AFilter: string;
  var AText: string; ACheck: TMcxDataCheck; ALoad: TMcxDataLoad): Boolean;
var
  F: TMcxDataForm;
  Bar: TPanel;
  Help: TLabel;
  Syn: TMcxJsonSyn;
  i: Integer;

  function AddButton(const ACaption: string; AAlign: TAlign;
    AClick: TNotifyEvent; AResult: TModalResult): TButton;
  begin
    Result := TButton.Create(Bar);
    Result.Parent := Bar;
    Result.Align := AAlign;
    Result.BorderSpacing.Around := 6;
    Result.AutoSize := True;
    Result.Caption := ACaption;
    Result.OnClick := AClick;
    Result.ModalResult := AResult;
  end;

begin
  F := TMcxDataForm.CreateNew(AOwner);
  try
    F.Caption := ATitle;
    F.Position := poOwnerFormCenter;
    F.Width := 820;
    F.Height := 560;
    F.FCheck := ACheck;
    F.FLoad := ALoad;
    F.FFilter := AFilter;
    F.OnCloseQuery := @F.FormCloseQuery;

    Help := TLabel.Create(F);
    Help.Parent := F;
    Help.Align := alTop;
    Help.WordWrap := True;
    Help.BorderSpacing.Around := 8;
    Help.Caption := AHelp;

    Bar := TPanel.Create(F);
    Bar.Parent := F;
    Bar.Align := alBottom;
    Bar.Height := 44;
    Bar.BevelOuter := bvNone;
    Bar.Caption := '';
    { Right-aligned ones are placed right to left. }
    AddButton('Cancel', alRight, nil, mrCancel).Cancel := True;
    AddButton('OK', alRight, nil, mrOK).Default := True;
    AddButton('Load from file...', alLeft, @F.LoadClick, mrNone);
    AddButton('Copy all', alLeft, @F.CopyClick, mrNone);
    AddButton('Paste', alLeft, @F.PasteClick, mrNone);

    Syn := TMcxJsonSyn.Create(F);
    F.FEdit := TSynEdit.Create(F);
    F.FEdit.Parent := F;
    F.FEdit.Align := alClient;
    F.FEdit.BorderSpacing.Left := 8;
    F.FEdit.BorderSpacing.Right := 8;
    F.FEdit.Highlighter := Syn;
    F.FEdit.Font.Name := McxDefaultFontName;
    F.FEdit.Font.Size := McxScalePointSize(McxDefaultFontSize);
    F.FEdit.RightEdge := 0;
    F.FEdit.Options := F.FEdit.Options - [eoScrollPastEol];
    { The same paper, ink and colouring as the JSON pane (mcxmain
      ApplyTheme): a highlighter left on its defaults is dark blue on the
      dark theme's dark field. }
    F.FEdit.Color := McxBlend(McxBase, McxText, 4);
    F.FEdit.Font.Color := McxText;
    F.FEdit.SelectedColor.Background := McxAccent;
    F.FEdit.SelectedColor.Foreground := McxReadable(McxAccent);
    F.FEdit.Gutter.Color := McxBase;
    Syn.Recolour(F.FEdit.Color, McxText, McxAccent);
    F.FEdit.Gutter.Visible := True;
    for i := 0 to F.FEdit.Gutter.Parts.Count - 1 do
    begin
      F.FEdit.Gutter.Parts[i].Visible :=
        F.FEdit.Gutter.Parts[i] is TSynGutterLineNumber;
      if F.FEdit.Gutter.Parts[i] is TSynGutterLineNumber then
        with TSynGutterLineNumber(F.FEdit.Gutter.Parts[i]) do
        begin
          MarkupInfo.Background := McxBase;
          MarkupInfo.Foreground := McxBlend(McxBase, McxText, 45);
        end;
    end;
    F.FEdit.Text := AText;

    Result := F.ShowModal = mrOK;
    if Result then AText := F.FEdit.Text;
  finally
    F.Free;
  end;
end;

end.
