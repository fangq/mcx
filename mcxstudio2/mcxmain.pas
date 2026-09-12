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
  StdCtrls, Buttons, ActnList, Menus, ImgList, ClipBrd, Spin, fpjson,
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
    { Every setting control the designer placed, paired with the caption that
      belongs to it: edPhotons with lbPhotons, ckMismatch with lbMismatch.  The
      binder finds each one by name through mcxdoc's table, so nothing here is
      referenced by the code directly -- they are declared because a form field
      is how the designer round-trips a control, and because a class only the
      .lfm mentions is otherwise smart-linked away. }
    gbSimulator: TGroupBox;
    rgBackend: TRadioGroup;
    rgDomainKind: TRadioGroup;
    lbMediaFormat: TLabel;
    cbMediaFormat: TComboBox;
    lbT0: TLabel;
    edT0: TEdit;
    lbT1: TLabel;
    edT1: TEdit;
    lbDt: TLabel;
    edDt: TEdit;
    lbSessionID: TLabel;
    edSessionID: TEdit;
    lbPhotons: TLabel;
    edPhotons: TEdit;
    lbOutFormat: TLabel;
    rgOutFormat: TRadioGroup;
    lbOutType: TLabel;
    cbOutType: TComboBox;
    lbSeed: TLabel;
    edSeed: TEdit;
    lbMismatch: TLabel;
    ckMismatch: TCheckBox;
    lbNormalize: TLabel;
    ckNormalize: TCheckBox;
    lbSaveVolume: TLabel;
    ckSaveVolume: TCheckBox;
    lbSaveDetp: TLabel;
    ckSaveDetp: TCheckBox;
    lbSaveExit: TLabel;
    ckSaveExit: TCheckBox;
    lbSaveSeed: TLabel;
    ckSaveSeed: TCheckBox;
    lbSaveRef: TLabel;
    ckSaveRef: TCheckBox;
    lbSpecular: TLabel;
    ckSpecular: TCheckBox;
    lbDCS: TLabel;
    ckDCS: TCheckBox;
    lbDim: TLabel;
    edDim: TEdit;
    lbUnit: TLabel;
    edUnit: TEdit;
    lbVolumeFile: TLabel;
    edVolumeFile: TEdit;
    lbOriginType: TLabel;
    ckOriginType: TCheckBox;
    lbSrcType: TLabel;
    cbSrcType: TComboBox;
    lbSrcPos: TLabel;
    edSrcPos: TEdit;
    lbSrcDir: TLabel;
    edSrcDir: TEdit;
    lbSrcParam1: TLabel;
    edSrcParam1: TEdit;
    lbSrcParam2: TLabel;
    edSrcParam2: TEdit;
    lbSrcFreq: TLabel;
    edSrcFreq: TEdit;
    lbSrcNum: TLabel;
    edSrcNum: TEdit;
    lbSrcWavelen: TLabel;
    edSrcWavelen: TEdit;
    lbAutoThread: TLabel;
    ckAutoThread: TCheckBox;
    lbThread: TLabel;
    edThread: TEdit;
    lbBlock: TLabel;
    edBlock: TEdit;
    lbWorkload: TLabel;
    edWorkload: TEdit;
    lbBC: TLabel;
    edBC: TEdit;
    lbDebug: TLabel;
    cgDebug: TCheckGroup;
    lbSaveMask: TLabel;
    cgSaveMask: TCheckGroup;
    lbMaxDetp: TLabel;
    edMaxDetp: TEdit;
    lbMinEnergy: TLabel;
    edMinEnergy: TEdit;
    lbRootPath: TLabel;
    edRootPath: TEdit;

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
    { One resolved control per row of mcxdoc's Binds table, in the same order.
      Nil where the designer has not grown that control yet, which is not an
      error: the table is allowed to run ahead of the form. }
    FBound: array of TControl;
    { The caption that belongs to each control, so it hides with it.  Found by
      convention: edPhotons pairs with lbPhotons, ckMismatch with lbMismatch. }
    FLabels: array of TControl;
    { What each control held immediately after it was loaded.  A binding whose
      key is absent from the file and whose control nobody has touched must not
      write anything, or merely opening a document would add every key the form
      knows about -- turning a three-key Session into a thirteen-key one. }
    FLoaded: array of string;
    FLoading: Integer;
    FMissing: TStringList;
    FPanes: array of TPanel;
    FHeads: array of TSpeedButton;
    FBodies: array of TPanel;
    FExpanded: Integer;
    FWizard: Boolean;
    procedure BuildIcons;
    procedure CollectPanes;
    procedure BindControls;
    procedure BindChanged(Sender: TObject);
    procedure CheckGroupClick(Sender: TObject; Index: Integer);
    function  DocFor(const APath: string): TMcxDoc;
    procedure LoadBinding(AIndex: Integer);
    procedure SaveBinding(AIndex: Integer);
    procedure LoadAllBindings;
    procedure ApplyBindingStates;
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
    { Drives every binding over a corpus of real input files and checks that
      loading a document into the form and writing it straight back changes
      nothing mcx would notice.  Returns the number of failures.

      This is the test the old architecture could not pass: it rebuilt the file
      from 46 stringified columns, so every key without a column -- Help, the
      JData volumes, anything a newer mcx had grown -- was dropped on save. }
    function  RunSelfTest(const ADir: string): Integer;
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

  FMissing := TStringList.Create;

  BuildIcons;
  CollectPanes;
  BindControls;

  mmJSON.Font.Name := McxDefaultFontName;
  mmCommand.Font.Assign(mmJSON.Font);
  mmLog.Font.Assign(mmJSON.Font);

  NewDocument;
  SetExpanded(0);
  UpdateHeaderCaptions;

  if FMissing.Count > 0 then
  begin
    { Not fatal: the table is allowed to run ahead of the form while sections
      are still being built out.  Reported so it cannot go unnoticed. }
    mmLog.Lines.Add(Format('%d binding(s) have no control on the form yet:',
      [FMissing.Count]));
    mmLog.Lines.AddStrings(FMissing);
  end;
end;

procedure TfmMain.FormDestroy(Sender: TObject);
begin
  FreeAndNil(FMissing);
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
  ApplyBindingStates;
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
  FRun.SetStr('@run.backend', 'mcx');
  FRun.SetStr('@run.domainkind', 'shapes');
  FRun.Modified := False;
  LoadAllBindings;
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
  LoadAllBindings;
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



function TfmMain.RunSelfTest(const ADir: string): Integer;
var
  Files: TStringList;
  Before: TMcxDoc;
  i, j, Checked: Integer;
  Diff: string;

  procedure Collect(const ADirName: string);
  var
    R: TSearchRec;
  begin
    if FindFirst(IncludeTrailingPathDelimiter(ADirName) + '*', faAnyFile, R) <> 0 then Exit;
    try
      repeat
        if (R.Name = '.') or (R.Name = '..') then Continue;
        if (R.Attr and faDirectory) <> 0 then
          Collect(IncludeTrailingPathDelimiter(ADirName) + R.Name)
        else if LowerCase(ExtractFileExt(R.Name)) = '.json' then
          Files.Add(IncludeTrailingPathDelimiter(ADirName) + R.Name);
      until FindNext(R) <> 0;
    finally
      FindClose(R);
    end;
  end;

begin
  Result := 0;
  Checked := 0;

  { Every row must name a control that exists, or the setting is unreachable
    however correct its path is. }
  for i := 0 to High(Binds) do
    if FBound[i] = nil then
    begin
      WriteLn('  FAIL  no control named ', Binds[i].Ctl,
              ' for ', Binds[i].Path);
      Inc(Result);
    end;

  { Two rows may not share a control or a path: either way one of them would
    silently never be written. }
  for i := 0 to High(Binds) do
    for j := i + 1 to High(Binds) do
    begin
      if SameText(Binds[i].Ctl, Binds[j].Ctl) then
      begin
        WriteLn('  FAIL  ', Binds[i].Ctl, ' is bound twice');
        Inc(Result);
      end;
      if SameText(Binds[i].Path, Binds[j].Path) then
      begin
        WriteLn('  FAIL  ', Binds[i].Path, ' is bound twice');
        Inc(Result);
      end;
    end;

  Files := TStringList.Create;
  Before := TMcxDoc.Create;
  try
    Collect(ADir);
    Files.Sort;
    for i := 0 to Files.Count - 1 do
    begin
      if not FDoc.LoadFromFile(Files[i]) then Continue;
      if not Before.LoadFromFile(Files[i]) then Continue;
      Inc(Checked);

      { Caught here rather than left to the LCL, which would put up a modal
        dialog and wait for a click -- turning a headless test run into a
        hang. }
      try
        LoadAllBindings;
        { Writing every binding back must be a no-op for a file nobody edited. }
        Inc(FLoading);
        try
          for j := 0 to High(Binds) do SaveBinding(j);
        finally
          Dec(FLoading);
        end;

        if not McxJSONSame(Before.Root, FDoc.Root, Diff) then
        begin
          WriteLn('  FAIL  ', ExtractFileName(Files[i]), ' -- ', Diff);
          Inc(Result);
        end;
      except
        on E: Exception do
        begin
          WriteLn('  FAIL  ', ExtractFileName(Files[i]), ' -- ',
                  E.ClassName, ': ', E.Message);
          Inc(Result);
        end;
      end;
      Flush(Output);
    end;
    WriteLn(Format('  %d files driven through every binding', [Checked]));
  finally
    Before.Free;
    Files.Free;
  end;
end;

{ ------------------------------------------------------------- binding ---- }

{ A path starting @run is a runtime choice -- which device, how many threads --
  and belongs to the preferences rather than to the simulation, so it is routed
  to a second document.  Everything else is the file itself. }
function TfmMain.DocFor(const APath: string): TMcxDoc;
begin
  if (APath <> '') and (APath[1] = '@') then Result := FRun else Result := FDoc;
end;

{ '60, 60, 60' <-> [60,60,60].  A vector is one edit rather than three, which
  keeps the form readable and the table short. }
function VecToText(A: TJSONData): string;
var
  i: Integer;
  E: TJSONData;
begin
  Result := '';
  if (A = nil) or (A.JSONType <> jtArray) then Exit;
  for i := 0 to TJSONArray(A).Count - 1 do
  begin
    E := TJSONArray(A).Items[i];
    { A multi-source input gives Pos and Dir a row per source, so the elements
      are arrays rather than numbers.  One edit box cannot show that, and
      flattening it would destroy the file -- so show nothing, and because the
      box then matches what it was loaded with, the save path leaves the value
      alone. }
    if not (E.JSONType in [jtNumber, jtString, jtBoolean]) then Exit('');
    if i > 0 then Result := Result + ', ';
    Result := Result + E.AsString;
  end;
end;

procedure TextToVec(D: TMcxDoc; const APath, AText: string);
var
  Parts: TStringList;
  i: Integer;
  V: Double;
  Arr: TJSONArray;
  Fs: TFormatSettings;
begin
  Fs := DefaultFormatSettings;
  Fs.DecimalSeparator := '.';
  Parts := TStringList.Create;
  try
    Parts.Delimiter := ',';
    Parts.StrictDelimiter := True;
    Parts.DelimitedText := AText;

    Arr := TJSONArray(D.Ensure(APath, jtArray));
    if Arr = nil then Exit;
    Arr.Clear;
    for i := 0 to Parts.Count - 1 do
    begin
      if Trim(Parts[i]) = '' then Continue;
      if not TryStrToFloat(Trim(Parts[i]), V, Fs) then Continue;
      if (Frac(V) = 0) and (Abs(V) < 1e15) then
        Arr.Add(Round(V))
      else
        Arr.Add(V);
    end;
  finally
    Parts.Free;
  end;
end;

procedure SetChoices(AItems: TStrings; const AList: string);
begin
  AItems.Delimiter := ',';
  AItems.StrictDelimiter := True;
  AItems.DelimitedText := AList;
end;

{ 'D:Detector ID,S:Scattering counts' -> the letters DS and the captions.  One
  table entry rather than two, so a flag cannot acquire a caption that belongs
  to a different letter. }
procedure SplitFlags(const AList: string; out ALetters: string;
  ACaptions: TStrings);
var
  Parts: TStringList;
  i, C: Integer;
begin
  ALetters := '';
  ACaptions.Clear;
  Parts := TStringList.Create;
  try
    SetChoices(Parts, AList);
    for i := 0 to Parts.Count - 1 do
    begin
      C := Pos(':', Parts[i]);
      if C < 2 then Continue;
      ALetters := ALetters + Parts[i][1];
      ACaptions.Add(Copy(Parts[i], C + 1, MaxInt));
    end;
  finally
    Parts.Free;
  end;
end;

{ The letters a flag set currently holds.  mcx writes these either as a string
  of letters or as the equivalent bitmask, and both turn up in the examples. }
function FlagsOf(D: TMcxDoc; const APath, ALetters: string): string;
var
  E: TJSONData;
  Mask: Int64;
  i: Integer;
begin
  Result := '';
  E := D.Find(APath);
  if E = nil then Exit;
  if E.JSONType = jtString then Exit(UpperCase(E.AsString));
  if E.JSONType <> jtNumber then Exit;
  Mask := E.AsInt64;
  for i := 1 to Length(ALetters) do
    if (Mask and (Int64(1) shl (i - 1))) <> 0 then Result := Result + ALetters[i];
end;

{ Resolves every row of the table to the control the designer named, and wires
  one shared handler to it.  The key is the component Name, which the designer
  already keeps unique -- unlike the old GUI, which hid the key in the first
  word of Hint and so could not have a hint edited or translated without
  breaking the file format. }
procedure TfmMain.BindControls;
var
  i: Integer;
  C: TComponent;
  Letters: string;
begin
  SetLength(FBound, Length(Binds));
  SetLength(FLabels, Length(Binds));
  SetLength(FLoaded, Length(Binds));
  for i := 0 to High(Binds) do
  begin
    C := FindComponent('lb' + Copy(Binds[i].Ctl, 3, MaxInt));
    if C is TControl then FLabels[i] := TControl(C) else FLabels[i] := nil;

    C := FindComponent(Binds[i].Ctl);
    if not (C is TControl) then
    begin
      FBound[i] := nil;
      FMissing.Add(Binds[i].Ctl + '  ->  ' + Binds[i].Path);
      Continue;
    end;
    FBound[i] := TControl(C);
    { Tag carries the row index, so the shared handler resolves its binding in
      one indirection.  Safe because the binder owns every control it wires. }
    FBound[i].Tag := i;

    if C is TCheckBox then
      TCheckBox(C).OnChange := @BindChanged
    else if C is TRadioGroup then
    begin
      if Binds[i].Choices <> '' then
        SetChoices(TRadioGroup(C).Items, Binds[i].Choices);
      TRadioGroup(C).OnClick := @BindChanged;
    end
    else if C is TCheckGroup then
    begin
      if Binds[i].Choices <> '' then
      begin
        SplitFlags(Binds[i].Choices, Letters, TCheckGroup(C).Items);
        { The letters ride on the group's Hint, so the load and save paths do
          not each have to re-split the table entry. }
        TCheckGroup(C).Hint := Letters;
      end;
      TCheckGroup(C).OnItemClick := @CheckGroupClick;
    end
    else if C is TComboBox then
    begin
      if Binds[i].Choices <> '' then
        SetChoices(TComboBox(C).Items, Binds[i].Choices);
      TComboBox(C).Style := csDropDownList;
      TComboBox(C).OnChange := @BindChanged;
    end
    else if C is TSpinEdit then
      TSpinEdit(C).OnChange := @BindChanged
    else if C is TEdit then
      { On editing done rather than on change: reformatting a number while it
        is still half-typed fights the person typing it. }
      TEdit(C).OnEditingDone := @BindChanged;
  end;
end;


{ The control's value as text, used both to detect "nobody touched this" and to
  keep the comparison in one place. }
function ControlText(C: TControl): string;
var
  i: Integer;
begin
  Result := '';
  if C = nil then Exit;
  if C is TCheckBox then
    Result := BoolToStr(TCheckBox(C).Checked, True)
  else if C is TRadioGroup then
    Result := IntToStr(TRadioGroup(C).ItemIndex)
  else if C is TCheckGroup then
  begin
    for i := 0 to TCheckGroup(C).Items.Count - 1 do
      if TCheckGroup(C).Checked[i] then Result := Result + '1' else Result := Result + '0';
  end
  else if C is TComboBox then
    Result := IntToStr(TComboBox(C).ItemIndex)
  else if C is TSpinEdit then
    Result := IntToStr(TSpinEdit(C).Value)
  else if C is TEdit then
    Result := TEdit(C).Text;
end;

procedure TfmMain.LoadBinding(AIndex: Integer);
var
  B: TMcxBind;
  C: TControl;
  D: TMcxDoc;
  E: TJSONData;
  S: string;
  N: Integer;
begin
  C := FBound[AIndex];
  if C = nil then Exit;
  B := Binds[AIndex];
  D := DocFor(B.Path);

  case B.Kind of
    mkBool:
      if C is TCheckBox then TCheckBox(C).Checked := D.AsBool(B.Path);
    mkChoice:
      begin
        S := D.AsStr(B.Path);
        if C is TRadioGroup then
        begin
          { An unknown value is added rather than silently dropped, so opening
            a file written by a newer mcx does not quietly rewrite it. }
          N := TRadioGroup(C).Items.IndexOf(S);
          if (N < 0) and (S <> '') then N := TRadioGroup(C).Items.Add(S);
          TRadioGroup(C).ItemIndex := N;
        end
        else if C is TComboBox then
        begin
          N := TComboBox(C).Items.IndexOf(S);
          if (N < 0) and (S <> '') then N := TComboBox(C).Items.Add(S);
          TComboBox(C).ItemIndex := N;
        end;
      end;
    mkFlags:
      if C is TCheckGroup then
      begin
        S := FlagsOf(D, B.Path, TCheckGroup(C).Hint);
        for N := 0 to TCheckGroup(C).Items.Count - 1 do
          TCheckGroup(C).Checked[N] :=
            (N < Length(TCheckGroup(C).Hint)) and
            (Pos(TCheckGroup(C).Hint[N + 1], S) > 0);
      end;
    mkInt:
      if C is TSpinEdit then TSpinEdit(C).Value := D.AsInt(B.Path)
      else if C is TEdit then TEdit(C).Text := IntToStr(D.AsInt(B.Path));
    mkFloat:
      if C is TEdit then
      begin
        { A scalar field may meet an array -- some files give DebugFlag or a
          length as a list -- and AsString raises on one.  Show it empty rather
          than taking the window down, and leave the value alone on save. }
        E := D.Find(B.Path);
        if (E <> nil) and (E.JSONType in [jtNumber, jtString, jtBoolean]) then
          TEdit(C).Text := E.AsString
        else
          TEdit(C).Text := '';
      end;
    mkVec:
      if C is TEdit then TEdit(C).Text := VecToText(D.Find(B.Path));
    mkText, mkFile:
      if C is TEdit then TEdit(C).Text := D.AsStr(B.Path);
  end;

  FLoaded[AIndex] := ControlText(C);
end;

procedure TfmMain.SaveBinding(AIndex: Integer);
var
  B: TMcxBind;
  C: TControl;
  D: TMcxDoc;
  V: Double;
  S: string;
  N: Integer;
  Fs: TFormatSettings;
begin
  C := FBound[AIndex];
  if C = nil then Exit;
  B := Binds[AIndex];
  D := DocFor(B.Path);

  { Untouched since it was loaded means there is nothing to save.  This is what
    makes opening a file and saving it a genuine no-op: absent keys stay absent
    rather than the form filling in everything it knows, and a value the form
    cannot represent -- a multi-source Pos, a DebugFlag given as a list -- is
    left exactly as it was found instead of being flattened. }
  if ControlText(C) = FLoaded[AIndex] then Exit;

  Fs := DefaultFormatSettings;
  Fs.DecimalSeparator := '.';

  case B.Kind of
    mkBool:
      if C is TCheckBox then D.SetBool(B.Path, TCheckBox(C).Checked);
    mkChoice:
      begin
        if (C is TRadioGroup) and (TRadioGroup(C).ItemIndex >= 0) then
          D.SetStr(B.Path, TRadioGroup(C).Items[TRadioGroup(C).ItemIndex])
        else if (C is TComboBox) and (TComboBox(C).ItemIndex >= 0) then
          D.SetStr(B.Path, TComboBox(C).Items[TComboBox(C).ItemIndex]);
      end;
    mkFlags:
      if C is TCheckGroup then
      begin
        S := '';
        for N := 0 to TCheckGroup(C).Items.Count - 1 do
          if TCheckGroup(C).Checked[N] and (N < Length(TCheckGroup(C).Hint)) then
            S := S + TCheckGroup(C).Hint[N + 1];
        { Written as letters, which is what mcx's own help and most of the
          examples use; a file that spelled it as a bitmask keeps that shape
          only until the flags are actually changed. }
        if S = '' then D.Delete(B.Path) else D.SetStr(B.Path, S);
      end;
    mkInt:
      if C is TSpinEdit then D.SetInt(B.Path, TSpinEdit(C).Value)
      else if C is TEdit then
      begin
        if Trim(TEdit(C).Text) = '' then D.Delete(B.Path)
        else if TryStrToFloat(Trim(TEdit(C).Text), V, Fs) then
          D.SetInt(B.Path, Round(V));
      end;
    mkFloat:
      if C is TEdit then
      begin
        { An emptied field removes the key rather than writing a zero, because
          absent and zero mean different things to mcx. }
        if Trim(TEdit(C).Text) = '' then D.Delete(B.Path)
        else if TryStrToFloat(Trim(TEdit(C).Text), V, Fs) then
          D.SetNum(B.Path, V);
      end;
    mkVec:
      if C is TEdit then TextToVec(D, B.Path, TEdit(C).Text);
    mkText, mkFile:
      if C is TEdit then
      begin
        if Trim(TEdit(C).Text) = '' then D.Delete(B.Path)
        else D.SetStr(B.Path, TEdit(C).Text);
      end;
  end;
end;

procedure TfmMain.LoadAllBindings;
var
  i: Integer;
begin
  Inc(FLoading);
  try
    for i := 0 to High(Binds) do LoadBinding(i);
  finally
    Dec(FLoading);
  end;
  ApplyBindingStates;
end;

{ EnableIf names a path and the value it has to hold, 'Session.DoAutoThread=0'.
  Comparing as text through the document keeps 0, false and "0" equivalent, so
  a file that spells a flag either way behaves the same.

  This is what the old GUI did with a chain of hand-written if SKey(..)= tests
  scattered through a 120-line OnChange handler. }
function ConditionHolds(D: TMcxDoc; const ACond: string): Boolean;
var
  E: Integer;
  Path, Want: string;
begin
  Result := True;
  if ACond = '' then Exit;
  E := Pos('=', ACond);
  if E < 1 then Exit;
  Path := Copy(ACond, 1, E - 1);
  Want := Copy(ACond, E + 1, MaxInt);
  if (Want = '0') or (Want = '1') then
    Result := D.AsBool(Path) = (Want = '1')
  else
    Result := SameText(D.AsStr(Path), Want);
end;

procedure TfmMain.ApplyBindingStates;
var
  i: Integer;
  B: TMcxBind;
  C: TControl;
  InMode: Boolean;
begin
  for i := 0 to High(Binds) do
  begin
    C := FBound[i];
    if C = nil then Continue;
    B := Binds[i];

    { The wizard is a strict subset of the expert form, so one filter serves
      both modes and there is no second layout to keep in step. }
    InMode := (not FWizard) or (B.Level = mlWizard);
    C.Visible := InMode;
    if InMode then
      C.Enabled := ConditionHolds(DocFor(B.EnableIf), B.EnableIf);
    if FLabels[i] <> nil then
    begin
      FLabels[i].Visible := InMode;
      FLabels[i].Enabled := C.Enabled;
    end;
  end;
end;

{ TCheckGroup reports which box moved; the binding does not care, so this just
  forwards to the shared handler. }
procedure TfmMain.CheckGroupClick(Sender: TObject; Index: Integer);
begin
  BindChanged(Sender);
end;

procedure TfmMain.BindChanged(Sender: TObject);
var
  i: Integer;
begin
  if FLoading > 0 then Exit;
  if not (Sender is TControl) then Exit;
  i := TControl(Sender).Tag;
  if (i < 0) or (i > High(Binds)) then Exit;
  if FBound[i] <> Sender then Exit;

  SaveBinding(i);
  { One sweep handles every dependent: cheap at forty rows, and it means a
    dependency is a table entry rather than another branch in a handler. }
  ApplyBindingStates;
  UpdateTitle;
  SchedulePreview;
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
