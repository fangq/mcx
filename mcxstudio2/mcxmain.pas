{ mcxstudio2 - the main window.

  The form is a design-time .lfm and is meant to stay that way.  Controls are
  placed and tuned in the Lazarus designer; mcxdoc's binding table connects
  each one to a path in the document by component Name, so adding a setting is
  one control plus one table row rather than an edit to four serialisers.

  Only things whose count is not known until runtime are built here in code:
  the media and detector tables, the shape list, the device list, and the
  source parameter slots, whose labels change with the source type.

  Sizes written in FormCreate are plain 96-dpi numbers, because the startup
  sweep in mcxdpi still has AutoAdjustLayout to run over the finished form.
  Anything created after that sweep scales itself with McxScale96. }
unit mcxmain;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, ComCtrls, ExtCtrls,
  StdCtrls, Buttons, ActnList, Menus, ImgList, ClipBrd,
  mcxdpi, mcxicons, mcxdoc;

type

  { TfmMain }

  TfmMain = class(TForm)
    acNew: TAction;
    acOpen: TAction;
    acSave: TAction;
    acSaveAs: TAction;
    acRun: TAction;
    acStop: TAction;
    acDevices: TAction;
    acToggleMode: TAction;
    acQuit: TAction;
    acAbout: TAction;
    alMain: TActionList;
    dlgOpen: TOpenDialog;
    dlgSave: TSaveDialog;
    ilIcons: TImageList;
    mmCommand: TMemo;
    mmJSON: TMemo;
    mmLog: TMemo;
    pcView: TPageControl;
    pnMain: TPanel;
    pnPreview: TPanel;
    sbMain: TStatusBar;
    sbSections: TScrollBox;
    spMain: TSplitter;
    tbMain: TToolBar;
    tbNew: TToolButton;
    tbOpen: TToolButton;
    tbSave: TToolButton;
    tbSep1: TToolButton;
    tbRun: TToolButton;
    tbStop: TToolButton;
    tbDevices: TToolButton;
    tbSep2: TToolButton;
    tbMode: TToolButton;
    tmRefresh: TTimer;
    tsCommand: TTabSheet;
    tsJSON: TTabSheet;
    tsLog: TTabSheet;
    tsPreview: TTabSheet;
    { The eight accordion sections.  Each is a plain TPanel holding a header
      button and a body panel, so the whole thing stays editable in the
      designer -- Lazarus 2.2 has no TCategoryPanel to place instead. }
    pnTypes: TPanel;     hdTypes: TSpeedButton;     bdTypes: TPanel;
    pnForward: TPanel;   hdForward: TSpeedButton;   bdForward: TPanel;
    pnSession: TPanel;   hdSession: TSpeedButton;   bdSession: TPanel;
    pnMedia: TPanel;     hdMedia: TSpeedButton;     bdMedia: TPanel;
    pnShapes: TPanel;    hdShapes: TSpeedButton;    bdShapes: TPanel;
    pnOptode: TPanel;    hdOptode: TSpeedButton;    bdOptode: TPanel;
    pnCompute: TPanel;   hdCompute: TSpeedButton;   bdCompute: TPanel;
    pnAdvanced: TPanel;  hdAdvanced: TSpeedButton;  bdAdvanced: TPanel;

    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormCloseQuery(Sender: TObject; var CanClose: Boolean);
    procedure acNewExecute(Sender: TObject);
    procedure acOpenExecute(Sender: TObject);
    procedure acSaveExecute(Sender: TObject);
    procedure acSaveAsExecute(Sender: TObject);
    procedure acToggleModeExecute(Sender: TObject);
    procedure acAboutExecute(Sender: TObject);
    procedure acQuitExecute(Sender: TObject);
    procedure HeaderClick(Sender: TObject);
    procedure tmRefreshTimer(Sender: TObject);
  private
    FDoc: TMcxDoc;
    FRun: TMcxDoc;
    FPanes: array of TPanel;
    FHeads: array of TSpeedButton;
    FBodies: array of TPanel;
    FExpanded: Integer;
    FWizard: Boolean;
    procedure BuildIcons;
    procedure CollectPanes;
    procedure SetExpanded(AIndex: Integer);
    procedure UpdateHeaderCaptions;
    procedure NewDocument;
    function  SaveAs: Boolean;
    function  ConfirmDiscard: Boolean;
    procedure UpdateTitle;
    procedure UpdateStatus;
  public
    { Queues one preview refresh.  Every edit calls this rather than
      re-rendering on the spot, so holding a key down does not rebuild the
      JSON on every character. }
    procedure SchedulePreview;
    procedure RefreshPreview;
    property Doc: TMcxDoc read FDoc;
  end;

var
  fmMain: TfmMain;

implementation

{$R *.lfm}

const
  SectionCaptions: array[0..7] of string = (
    'Types', 'Forward', 'Session', 'Properties',
    'Shapes', 'Optode', 'Compute', 'Advanced');

{ A minimal document: enough for mcx to run, and the same 60-cube every
  tutorial starts from. }
procedure SeedDocument(D: TMcxDoc);
begin
  D.LoadFromString(
    '{'#10 +
    '  "Session": { "ID": "mcx", "Photons": 1000000, "RNGSeed": 1648335518,' +
    ' "DoMismatch": 1, "DoAutoThread": 1 },'#10 +
    '  "Forward": { "T0": 0, "T1": 5e-09, "Dt": 5e-09 },'#10 +
    '  "Domain": { "OriginType": 1, "LengthUnit": 1, "Dim": [60, 60, 60],' +
    ' "Media": [ { "mua": 0, "mus": 0, "g": 1, "n": 1 },' +
    ' { "mua": 0.005, "mus": 1, "g": 0.01, "n": 1.37 } ] },'#10 +
    '  "Shapes": [ { "Grid": { "Tag": 1, "Size": [60, 60, 60] } } ],'#10 +
    '  "Optode": { "Source": { "Type": "pencil", "Pos": [30, 30, 0],' +
    ' "Dir": [0, 0, 1] }, "Detector": [] }'#10 +
    '}');
  D.FileName := '';
  D.Modified := False;
end;

{ TfmMain }

procedure TfmMain.FormCreate(Sender: TObject);
begin
  FDoc := TMcxDoc.Create;
  FRun := TMcxDoc.Create;
  FExpanded := -1;
  FWizard := True;

  BuildIcons;
  CollectPanes;

  mmJSON.Font.Name := McxDefaultFontName;
  mmCommand.Font.Assign(mmJSON.Font);
  mmLog.Font.Assign(mmJSON.Font);

  NewDocument;
  SetExpanded(0);
  UpdateHeaderCaptions;
end;

procedure TfmMain.FormDestroy(Sender: TObject);
begin
  FreeAndNil(FRun);
  FreeAndNil(FDoc);
end;

procedure TfmMain.FormCloseQuery(Sender: TObject; var CanClose: Boolean);
begin
  CanClose := ConfirmDiscard;
end;

{ The image list ships empty and is drawn here, once, at the display's scale
  and in the current theme's button-text colour, so the icons stay legible
  under a dark theme instead of being black on black. }
procedure TfmMain.BuildIcons;
var
  Size: Integer;
begin
  Size := McxScale96(16);
  ilIcons.Width := Size;
  ilIcons.Height := Size;
  McxBuildIconList(ilIcons, McxIconNames, clBtnText);

  acNew.ImageIndex := McxIconIndex('new');
  acOpen.ImageIndex := McxIconIndex('open');
  acSave.ImageIndex := McxIconIndex('save');
  acSaveAs.ImageIndex := McxIconIndex('saveas');
  acRun.ImageIndex := McxIconIndex('run');
  acStop.ImageIndex := McxIconIndex('stop');
  acDevices.ImageIndex := McxIconIndex('gpu');
  acToggleMode.ImageIndex := McxIconIndex('wizard');
  acAbout.ImageIndex := McxIconIndex('about');
end;

{ ------------------------------------------------------------ accordion --- }

procedure TfmMain.CollectPanes;
begin
  FPanes := [pnTypes, pnForward, pnSession, pnMedia,
             pnShapes, pnOptode, pnCompute, pnAdvanced];
  FHeads := [hdTypes, hdForward, hdSession, hdMedia,
             hdShapes, hdOptode, hdCompute, hdAdvanced];
  FBodies := [bdTypes, bdForward, bdSession, bdMedia,
              bdShapes, bdOptode, bdCompute, bdAdvanced];
end;

{ One section open at a time, which is what the expert form is meant to feel
  like.  Clicking the open one closes it, so everything can be collapsed. }
procedure TfmMain.SetExpanded(AIndex: Integer);
var
  i: Integer;
begin
  if AIndex >= Length(FPanes) then AIndex := -1;
  sbSections.DisableAlign;
  try
    for i := 0 to High(FPanes) do
      FBodies[i].Visible := (i = AIndex);
  finally
    sbSections.EnableAlign;
  end;
  FExpanded := AIndex;
  UpdateHeaderCaptions;
  if AIndex >= 0 then sbSections.ScrollInView(FPanes[AIndex]);
end;

{ The chevron is a glyph rather than a caption prefix because TSpeedButton
  centres its caption and has no Alignment -- but it does place a glyph at
  Margin, and the caption follows the glyph, which left-aligns the header. }
procedure TfmMain.UpdateHeaderCaptions;
var
  i, Closed, Open: Integer;
begin
  Closed := McxIconIndex('collapsed');
  Open := McxIconIndex('expanded');
  for i := 0 to High(FHeads) do
  begin
    FHeads[i].Caption := SectionCaptions[i];
    if i = FExpanded then
      FHeads[i].ImageIndex := Open
    else
      FHeads[i].ImageIndex := Closed;
  end;
end;

procedure TfmMain.HeaderClick(Sender: TObject);
var
  i: Integer;
begin
  for i := 0 to High(FHeads) do
    if FHeads[i] = Sender then
    begin
      if FExpanded = i then SetExpanded(-1) else SetExpanded(i);
      Exit;
    end;
end;

{ --------------------------------------------------------------- modes ---- }

procedure TfmMain.acToggleModeExecute(Sender: TObject);
begin
  FWizard := not FWizard;
  { Wizard mode is the same panels with the expert-only controls hidden, so
    there is one layout and no second form to keep in step.  The filtering
    itself arrives with the binding table. }
  if FWizard then
  begin
    acToggleMode.Caption := 'Wizard';
    acToggleMode.ImageIndex := McxIconIndex('wizard');
    acToggleMode.Hint := 'Showing key settings only. Click for every setting.';
  end
  else
  begin
    acToggleMode.Caption := 'Expert';
    acToggleMode.ImageIndex := McxIconIndex('expert');
    acToggleMode.Hint := 'Showing every setting. Click for key settings only.';
  end;
  UpdateStatus;
end;

{ ------------------------------------------------------------ documents --- }

procedure TfmMain.NewDocument;
begin
  SeedDocument(FDoc);
  FRun.Clear;
  UpdateTitle;
  RefreshPreview;
end;

function TfmMain.ConfirmDiscard: Boolean;
begin
  Result := True;
  if not FDoc.Modified then Exit;
  case MessageDlg('MCX Studio',
         'This simulation has unsaved changes.'#10#10 +
         'Save before continuing?',
         mtConfirmation, [mbYes, mbNo, mbCancel], 0) of
    mrYes:    Result := (FDoc.FileName <> '') and FDoc.SaveToFile(FDoc.FileName)
                        or SaveAs;
    mrNo:     Result := True;
  else
    Result := False;
  end;
end;

procedure TfmMain.acNewExecute(Sender: TObject);
begin
  if not ConfirmDiscard then Exit;
  NewDocument;
end;

procedure TfmMain.acOpenExecute(Sender: TObject);
begin
  if not ConfirmDiscard then Exit;
  if not dlgOpen.Execute then Exit;
  if not FDoc.LoadFromFile(dlgOpen.FileName) then
  begin
    MessageDlg('MCX Studio',
      'Could not read ' + ExtractFileName(dlgOpen.FileName) + ':'#10#10 +
      FDoc.LastError, mtError, [mbOK], 0);
    Exit;
  end;
  UpdateTitle;
  RefreshPreview;
end;

function TfmMain.SaveAs: Boolean;
begin
  Result := False;
  if FDoc.FileName <> '' then dlgSave.FileName := FDoc.FileName
  else dlgSave.FileName := FDoc.AsStr('Session.ID', 'mcx') + '.json';
  if not dlgSave.Execute then Exit;
  Result := FDoc.SaveToFile(dlgSave.FileName);
  if not Result then
    MessageDlg('MCX Studio', 'Could not write the file:'#10#10 + FDoc.LastError,
      mtError, [mbOK], 0);
  UpdateTitle;
end;

procedure TfmMain.acSaveExecute(Sender: TObject);
begin
  if FDoc.FileName = '' then
    SaveAs
  else if not FDoc.SaveToFile(FDoc.FileName) then
    MessageDlg('MCX Studio', 'Could not write the file:'#10#10 + FDoc.LastError,
      mtError, [mbOK], 0);
  UpdateTitle;
end;

procedure TfmMain.acSaveAsExecute(Sender: TObject);
begin
  SaveAs;
end;

procedure TfmMain.acQuitExecute(Sender: TObject);
begin
  Close;
end;

procedure TfmMain.acAboutExecute(Sender: TObject);
begin
  MessageDlg('MCX Studio',
    'MCX Studio' + #10#10 +
    'A graphical front end for MCX, MMC and MCX-CL.'#10 +
    'https://mcx.space'#10#10 +
    Format('Display scale: %d dpi', [McxDesiredPPI]),
    mtInformation, [mbOK], 0);
end;

{ --------------------------------------------------------------- preview -- }

procedure TfmMain.SchedulePreview;
begin
  { Restarting the timer is the debounce: the refresh happens once the edits
    stop, not once per keystroke. }
  tmRefresh.Enabled := False;
  tmRefresh.Enabled := True;
end;

procedure TfmMain.tmRefreshTimer(Sender: TObject);
begin
  tmRefresh.Enabled := False;
  RefreshPreview;
end;

procedure TfmMain.RefreshPreview;
var
  Input: string;
begin
  mmJSON.Text := FDoc.ToJSON(True);

  { The command the Run action will issue, shown so it can be copied, re-run
    and pasted into a bug report.  Almost everything lives in the JSON, so
    this stays short -- which is the whole reason the old CreateCmd's 230
    lines of flag assembly are not being ported. }
  if FDoc.FileName <> '' then
    Input := ExtractFileName(FDoc.FileName)
  else
    Input := FDoc.AsStr('Session.ID', 'mcx') + '.json';
  mmCommand.Text := Format('mcx -f %s -s %s',
    [Input, FDoc.AsStr('Session.ID', 'mcx')]);

  UpdateStatus;
end;

procedure TfmMain.UpdateTitle;
var
  Shown: string;
begin
  if FDoc.FileName <> '' then Shown := ExtractFileName(FDoc.FileName)
  else Shown := 'untitled';
  Caption := Shown + ' - MCX Studio';
  UpdateStatus;
end;

procedure TfmMain.UpdateStatus;
begin
  if FDoc.Modified then
    sbMain.Panels[0].Text := 'Modified'
  else
    sbMain.Panels[0].Text := 'Ready';
  if FWizard then
    sbMain.Panels[1].Text := 'Wizard'
  else
    sbMain.Panels[1].Text := 'Expert';
  sbMain.Panels[2].Text := Format('%d x %d x %d',
    [FDoc.AsInt('Domain.Dim[0]'), FDoc.AsInt('Domain.Dim[1]'),
     FDoc.AsInt('Domain.Dim[2]')]);
end;

end.
