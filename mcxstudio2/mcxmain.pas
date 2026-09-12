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
  { One navigator entry below a section header: the group box on the detail
    page it points at, and the button that points there.  Built from the
    Sections table rather than placed, so the two cannot drift apart. }
  TMcxNavItem = record
    Section: Integer;
    Box: TPanel;
    Btn: TSpeedButton;
    { The card's own caption.  A card is a panel, so the heading is a label
      the navigator puts there rather than a frame property -- which is also
      why colouring it no longer touches anything else on the card. }
    Cap: TLabel;
  end;

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
    lbTodoGL: TLabel;
    mmCommand: TMemo;
    mmJSON: TMemo;
    mmLog: TMemo;
    pcView: TPageControl;
    pnGL: TPanel;
    pnMain: TPanel;
    pnPreview: TPanel;
    sbDetail: TScrollBox;
    sbMain: TStatusBar;
    sbNav: TScrollBox;
    spNav: TSplitter;
    spPreview: TSplitter;
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

    { The navigator, on the left: one header button and one body panel per
      section.  The bodies are empty in the designer on purpose -- their
      contents are one button per group box, built from the Sections table
      below, so a group box cannot be added to a page and forgotten here. }
    pnSimulator: TPanel;  hdSimulator: TSpeedButton;  bdSimulator: TPanel;
    pnDomain: TPanel;     hdDomain: TSpeedButton;     bdDomain: TPanel;
    pnShapes: TPanel;     hdShapes: TSpeedButton;     bdShapes: TPanel;
    pnOptode: TPanel;     hdOptode: TSpeedButton;     bdOptode: TPanel;
    pnSession: TPanel;    hdSession: TSpeedButton;    bdSession: TPanel;
    pnCompute: TPanel;    hdCompute: TSpeedButton;    bdCompute: TPanel;
    pnAdvanced: TPanel;   hdAdvanced: TSpeedButton;   bdAdvanced: TPanel;

    { The detail pane, in the middle: one page per section, and on each page
      the group boxes the navigator lists.  Selecting a section shows its
      page; selecting a subsection scrolls its group box to the top. }
    pgSimulator: TPanel;
    pgDomain: TPanel;
    pgShapes: TPanel;
    pgOptode: TPanel;
    pgSession: TPanel;
    pgCompute: TPanel;
    pgAdvanced: TPanel;

    gbEngine: TPanel;
    gbGrid: TPanel;
    gbVolume: TPanel;
    gbMedia: TPanel;
    gbShapeList: TPanel;
    gbSource: TPanel;
    gbSrcAdv: TPanel;
    gbDetector: TPanel;
    gbBasic: TPanel;
    gbTime: TPanel;
    gbOutput: TPanel;
    gbSwitches: TPanel;
    gbDetPhoton: TPanel;
    gbGPU: TPanel;
    gbDevices: TPanel;
    gbBoundary: TPanel;
    gbFlags: TPanel;

    { The editors that are not written yet say so, rather than leaving a
      group box that looks broken. }
    lbTodoMedia: TLabel;
    lbTodoShapes: TLabel;
    lbTodoDet: TLabel;
    lbTodoDevices: TLabel;

    { One panel per label-and-control row.  A row is a panel rather than two
      controls at hand-picked coordinates so that every width comes from
      Align: the label is alLeft at a fixed width, the control alClient.  It
      also gives the wizard something to hide -- hiding an edit on its own
      would leave its caption behind and a 27-pixel hole where the row was. }
    rwDim: TPanel;
    rwUnit: TPanel;
    rwMediaFormat: TPanel;
    rwVolumeFile: TPanel;
    rwSrcType: TPanel;
    rwSrcPos: TPanel;
    rwSrcDir: TPanel;
    rwSrcParam1: TPanel;
    rwSrcParam2: TPanel;
    rwSrcFreq: TPanel;
    rwSrcNum: TPanel;
    rwSrcWavelen: TPanel;
    rwSessionID: TPanel;
    rwPhotons: TPanel;
    rwSeed: TPanel;
    rwT0: TPanel;
    rwT1: TPanel;
    rwDt: TPanel;
    rwOutType: TPanel;
    rwRootPath: TPanel;
    rwMismatch: TPanel;
    rwSaveVolume: TPanel;
    rwSpecular: TPanel;
    rwSaveDetp: TPanel;
    rwMaxDetp: TPanel;
    rwThread: TPanel;
    rwBlock: TPanel;
    rwWorkload: TPanel;
    rwBC: TPanel;
    rwMinEnergy: TPanel;

    { Every setting control the designer placed, paired with the caption that
      belongs to it: edPhotons with lbPhotons, cbSrcType with lbSrcType.  A
      control that carries its own caption -- a check box, a radio group --
      has no label, and the binder is happy to find none.

      Nothing here is referenced by the code directly; they are declared
      because a form field is how the designer round-trips a control, and
      because a class only the .lfm mentions is otherwise smart-linked away. }
    rgBackend: TRadioGroup;
    rgDomainKind: TRadioGroup;
    lbDim: TLabel;
    edDim: TEdit;
    lbUnit: TLabel;
    edUnit: TEdit;
    ckOriginType: TCheckBox;
    lbMediaFormat: TLabel;
    cbMediaFormat: TComboBox;
    lbVolumeFile: TLabel;
    edVolumeFile: TEdit;
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
    edSrcNum: TSpinEdit;
    lbSrcWavelen: TLabel;
    edSrcWavelen: TEdit;
    lbSessionID: TLabel;
    edSessionID: TEdit;
    lbPhotons: TLabel;
    edPhotons: TEdit;
    lbSeed: TLabel;
    edSeed: TEdit;
    lbT0: TLabel;
    edT0: TEdit;
    lbT1: TLabel;
    edT1: TEdit;
    lbDt: TLabel;
    edDt: TEdit;
    rgOutFormat: TRadioGroup;
    lbOutType: TLabel;
    cbOutType: TComboBox;
    lbRootPath: TLabel;
    edRootPath: TEdit;
    ckMismatch: TCheckBox;
    ckNormalize: TCheckBox;
    ckSaveVolume: TCheckBox;
    ckSaveRef: TCheckBox;
    ckSpecular: TCheckBox;
    ckDCS: TCheckBox;
    ckSaveDetp: TCheckBox;
    ckSaveExit: TCheckBox;
    ckSaveSeed: TCheckBox;
    cgSaveMask: TCheckGroup;
    lbMaxDetp: TLabel;
    edMaxDetp: TEdit;
    ckAutoThread: TCheckBox;
    lbThread: TLabel;
    edThread: TSpinEdit;
    lbBlock: TLabel;
    edBlock: TSpinEdit;
    lbWorkload: TLabel;
    edWorkload: TEdit;
    lbBC: TLabel;
    edBC: TEdit;
    cgDebug: TCheckGroup;
    lbMinEnergy: TLabel;
    edMinEnergy: TEdit;

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
    { The row panel and the group box each control sits in.  Hiding a control
      is not enough on its own: the row it shares with its caption has to go
      too, or the wizard leaves a caption with nothing beside it, and a group
      box the wizard has emptied has to take its heading and its navigator
      entry with it rather than stand there as an empty frame. }
    FRows: array of TPanel;
    FGroups: array of TPanel;
    FRowList: array of TPanel;
    FLoading: Integer;
    FMissing: TStringList;
    { The navigator: one pane, header and body per section, plus one entry
      per subsection.  FSection is the section whose page the detail pane is
      showing -- always a real one, because an empty detail pane is not a
      state worth having. }
    FPanes: array of TPanel;
    FHeads: array of TSpeedButton;
    FBodies: array of TPanel;
    FPages: array of TPanel;
    FSubs: array of TMcxNavItem;
    { The section whose page is showing, and the subsection whose card is
      highlighted on it.  There is always one of each: an empty detail pane
      is not a state worth having, and neither is a page where nothing says
      where you are. }
    FSection: Integer;
    FSub: Integer;
    FWizard: Boolean;
    procedure BuildIcons;
    procedure CollectSections;
    procedure BuildNav;
    procedure BindControls;
    procedure BindChanged(Sender: TObject);
    procedure CheckGroupClick(Sender: TObject; Index: Integer);
    function  DocFor(const APath: string): TMcxDoc;
    procedure LoadBinding(AIndex: Integer);
    procedure SaveBinding(AIndex: Integer);
    procedure LoadAllBindings;
    procedure ApplyBindingStates;
    procedure SelectSection(AIndex: Integer);
    procedure SubClick(Sender: TObject);
    procedure CardPaint(Sender: TObject);
    procedure SetCard(ACard: TPanel; AColor: TColor);
    procedure ScrollToGroup(ABox: TPanel);
    function  FirstSubOf(ASection: Integer): Integer;
    procedure SelectSub(AIndex: Integer);
    procedure UpdateNavState;
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
  { Which group boxes belong to which section, in the order they appear on the
    page.  This is the whole of the navigator's content: the section captions
    are on the header buttons in the .lfm and the subsection captions are on
    the group boxes, so every string a reader sees is written once, where the
    designer shows it, and the existing .po extraction keeps working.

    The order here matches CollectSections. }
  { Three colours do the whole window.

    Two of them are the same neutral at two strengths and say what level a
    surface is: the band behind a section heading, the fill of a card.  The
    third is the theme's own selection colour, and it is the only thing that
    ever says "this one".

    Seven per-section accents were prettier and worse.  A colour that means
    identity cannot also mean selection, so nothing on a page could be
    highlighted without arguing with the section it sat on. }
  { Corner radius on the design grid, scaled with everything else. }
  CardRadius = 8;

  BandLevel  = 14;   { a closed section heading }
  CardLevel  =  7;   { a card nobody has picked }
  BandActive = 58;   { the open section }
  CardActive = 22;   { the card of the selected subsection }

  SectionGroups: array[0..6] of string = (
    { Simulator } 'gbEngine',
    { Domain    } 'gbGrid,gbVolume,gbMedia',
    { Shapes    } 'gbShapeList',
    { Optode    } 'gbSource,gbSrcAdv,gbDetector',
    { Session   } 'gbBasic,gbTime,gbOutput,gbSwitches,gbDetPhoton',
    { Compute   } 'gbGPU,gbDevices',
    { Advanced  } 'gbBoundary,gbFlags');

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
  FSection := -1;
  FSub := -1;
  FWizard := True;

  FMissing := TStringList.Create;

  BuildIcons;
  CollectSections;
  BuildNav;
  BindControls;

  mmJSON.Font.Name := McxDefaultFontName;
  mmCommand.Font.Assign(mmJSON.Font);
  mmLog.Font.Assign(mmJSON.Font);

  NewDocument;
  SelectSection(0);

  if FMissing.Count > 0 then
  begin
    { Not fatal: the table is allowed to run ahead of the form while sections
      are still being built out.  Reported so it cannot go unnoticed. }
    mmLog.Lines.Add(Format('%d table entr(ies) have no control on the form yet:',
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
  { 24 rather than 16 on the design grid: the artwork is a coloured badge
    rather than a line glyph, and a badge needs the pixels to read as one. }
  Size := McxScale96(24);
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

{ ------------------------------------------------------------ navigator --- }

{ A colour APercent of the way from A to B.  Every shade the navigator uses is
  mixed from the theme's own clBtnFace / clWindowText / clHighlight rather than
  written down as a grey, so the bands stay legible under a dark theme -- which
  is where the old GUI's hard-coded colours failed. }
function McxBlend(A, B: TColor; APercent: Integer): TColor;
var
  ra, ga, ba, rb, gb, bb: Byte;
begin
  RedGreenBlue(ColorToRGB(A), ra, ga, ba);
  RedGreenBlue(ColorToRGB(B), rb, gb, bb);
  Result := RGBToColor(
    ra + (Integer(rb) - ra) * APercent div 100,
    ga + (Integer(gb) - ga) * APercent div 100,
    ba + (Integer(bb) - ba) * APercent div 100);
end;

{ Selecting a section shows its page and lists its subsections underneath the
  header; selecting a subsection scrolls that group box to the top of the
  page.  Nothing is edited in the navigator itself -- it exists to keep forty
  settings from arriving as one long column. }

type
  { Color and ParentColor are protected in TControl and only published by the
    concrete classes; a descendant declared in this unit reaches them without
    caring which class it was handed. }
  TColourAccess = class(TWinControl);

{ Paints a card and everything nested in it.

  Recursive, and it sets the colour on each control rather than relying on
  ParentColor: a TPanel honours ParentColor, but a radio or check group asks
  the widget set for its own background whatever ParentColor says, and left
  alone the two of them sat on a tinted card as pale rectangles. }
procedure PaintCard(C: TWinControl; AColor: TColor);
var
  i: Integer;
begin
  if C = nil then Exit;
  TColourAccess(C).ParentColor := False;
  TColourAccess(C).Color := AColor;
  for i := 0 to C.ControlCount - 1 do
    if (C.Controls[i] is TPanel) or (C.Controls[i] is TCustomGroupBox) then
      PaintCard(TWinControl(C.Controls[i]), AColor);
end;

{ Draws a card as a rounded rectangle.

  This is the other half of why a card had to stop being a TGroupBox: a
  TPanel is a TCustomControl, so it has a Paint of its own to hook, and a
  group box -- being a TWinControl wrapping a native widget -- has none.

  The panel itself is painted in the page's colour so the four corners show
  the background through, and the card is the rounded shape drawn on top.
  Nothing inside reaches the corners: ChildSizing insets every row by ten
  pixels, which is wider than the radius. }
{ Gives a card its colour.  The card itself is painted in the page's own
  colour so the corners stay transparent to it, the fill rides in Tag for
  CardPaint to pick up, and everything inside is filled solid -- nothing in
  there comes near a corner. }
procedure TfmMain.SetCard(ACard: TPanel; AColor: TColor);
var
  i: Integer;
  W: TWinControl;
begin
  if ACard = nil then Exit;
  ACard.Tag := AColor;
  W := ACard;
  TColourAccess(W).ParentColor := False;
  TColourAccess(W).Color := clBtnFace;
  for i := 0 to ACard.ControlCount - 1 do
    if (ACard.Controls[i] is TPanel) or (ACard.Controls[i] is TCustomGroupBox) then
      PaintCard(TWinControl(ACard.Controls[i]), AColor);
  ACard.Invalidate;
end;

procedure TfmMain.CardPaint(Sender: TObject);
var
  P: TPanel;
  R: Integer;
begin
  if not (Sender is TPanel) then Exit;
  P := TPanel(Sender);
  R := McxScale96(CardRadius);
  P.Canvas.AntialiasingMode := amOn;
  P.Canvas.Brush.Style := bsSolid;
  P.Canvas.Brush.Color := TColor(P.Tag);
  P.Canvas.Pen.Style := psSolid;
  P.Canvas.Pen.Color := TColor(P.Tag);
  P.Canvas.RoundRect(0, 0, P.Width, P.Height, R, R);
end;

procedure TfmMain.CollectSections;
var
  i: Integer;
begin
  FPanes := [pnSimulator, pnDomain, pnShapes, pnOptode,
             pnSession, pnCompute, pnAdvanced];
  FHeads := [hdSimulator, hdDomain, hdShapes, hdOptode,
             hdSession, hdCompute, hdAdvanced];
  FBodies := [bdSimulator, bdDomain, bdShapes, bdOptode,
              bdSession, bdCompute, bdAdvanced];
  FPages := [pgSimulator, pgDomain, pgShapes, pgOptode,
             pgSession, pgCompute, pgAdvanced];

  { A flat TSpeedButton paints nothing of its own until it is hovered, so the
    band behind a section heading is the section panel's colour showing
    through.  The body panel underneath is painted back to the navigator's
    own background, which is what separates a heading from the subsections
    listed below it. }
  for i := 0 to High(FPanes) do
  begin
    FPanes[i].ParentColor := False;
    FBodies[i].ParentColor := False;
    FBodies[i].Color := clBtnFace;
  end;
  sbNav.ParentColor := False;
  sbNav.Color := clBtnFace;
end;

{ One button per group box, captioned from the group box itself so a title is
  written -- and translated -- once, in the .lfm.

  Built in FormCreate, which is before mcxdpi's startup sweep, so the sizes
  here are plain 96-dpi numbers and the sweep scales them with the rest of the
  form.  Anything built after the sweep would have to call McxScale96 itself. }
procedure TfmMain.BuildNav;
var
  s, g: Integer;
  Names: TStringList;
  C: TComponent;
  Card: TPanel;
  Cap: TLabel;
  B: TSpeedButton;
begin
  SetLength(FSubs, 0);
  Names := TStringList.Create;
  try
    Names.Delimiter := ',';
    Names.StrictDelimiter := True;
    for s := 0 to High(SectionGroups) do
    begin
      Names.DelimitedText := SectionGroups[s];
      for g := 0 to Names.Count - 1 do
      begin
        C := FindComponent(Names[g]);
        if not (C is TPanel) then
        begin
          { The table naming a card the form does not have is the same kind
            of drift the binding table guards against, so it is reported the
            same way rather than silently skipped. }
          FMissing.Add(Names[g] + '  ->  no card on ' + FPages[s].Name);
          Continue;
        end;
        Card := TPanel(C);

        B := TSpeedButton.Create(Self);
        B.Parent := FBodies[s];
        B.Height := 44;
        { Top before Align: alTop children are ordered by the Top they have
          when they are aligned, so leaving them all at zero lists the
          subsections in reverse. }
        B.Top := g * 44;
        B.Align := alTop;
        { Left-justified rather than centred: a column of centred titles under
          a centred heading reads as a poster, not as a list to scan.  A
          TSpeedButton honours Alignment only when Margin and Spacing are both
          set, and Margin is also what indents a subsection under its section. }
        B.Alignment := taLeftJustify;
        B.AllowAllUp := True;
        B.Caption := Card.Caption;
        B.Flat := True;
        B.GroupIndex := 2;
        B.Layout := blGlyphLeft;
        { Indented past where a section heading's own text starts, so the
          hierarchy is visible without a second glyph column. }
        B.Margin := 40;
        B.Spacing := 6;
        B.ParentFont := False;
        B.Font.Height := -28;
        B.Tag := Length(FSubs);
        B.OnClick := @SubClick;

        SetLength(FSubs, Length(FSubs) + 1);
        FSubs[High(FSubs)].Section := s;
        FSubs[High(FSubs)].Box := Card;
        FSubs[High(FSubs)].Btn := B;

        { The heading moves off the panel and onto a label of its own.

          A card is a flat filled surface, and TGroupBox cannot be one: on
          gtk2 it is a GtkFrame, so the etched line around it comes from the
          widget set rather than from any property.  A TPanel with BevelOuter
          off has no line -- but it centres its Caption, and a caption that
          belongs to the panel is also inherited by every control on it, so
          colouring it would colour the whole card's text.  A label answers
          both: it sits at the top left and it is the only thing that
          changes when the card is selected.

          Top before Align, for the same reason the navigator's buttons need
          it: alTop children are ordered by the Top they have when they are
          aligned, and every row on the card already starts at zero. }
        Cap := TLabel.Create(Self);
        Cap.Parent := Card;
        Cap.Top := -1;
        Cap.Align := alTop;
        Cap.BorderSpacing.Bottom := 4;
        Cap.Caption := Card.Caption;
        Cap.ParentFont := False;
        Cap.Font.Style := [fsBold];
        Card.Caption := '';
        Card.OnPaint := @CardPaint;
        FSubs[High(FSubs)].Cap := Cap;
      end;
    end;
  finally
    Names.Free;
  end;
end;

procedure TfmMain.SelectSection(AIndex: Integer);
var
  i: Integer;
begin
  if (AIndex < 0) or (AIndex > High(FPanes)) then Exit;

  { Both panes move together: the navigator opens one section's subsection
    list, the detail pane shows that section's page.  Alignment is frozen
    while eight panels change visibility, or the scroll box relayouts once
    per panel. }
  sbNav.DisableAlign;
  sbDetail.DisableAlign;
  try
    for i := 0 to High(FPanes) do
    begin
      FBodies[i].Visible := (i = AIndex);
      FPages[i].Visible := (i = AIndex);
    end;
  finally
    sbDetail.EnableAlign;
    sbNav.EnableAlign;
  end;

  FSection := AIndex;
  sbDetail.VertScrollBar.Position := 0;
  SelectSub(FirstSubOf(AIndex));
end;

{ Scrolls ABox to the top of the detail pane.  The offset is summed up the
  parent chain rather than read from ABox.Top, because a group box's Top is
  relative to its page, not to the scroll box. }
procedure TfmMain.ScrollToGroup(ABox: TPanel);
var
  C: TControl;
  Offset: Integer;
begin
  if ABox = nil then Exit;
  Offset := 0;
  C := ABox;
  while (C <> nil) and (C <> sbDetail) do
  begin
    Inc(Offset, C.Top);
    C := C.Parent;
  end;
  Dec(Offset, 6);
  if Offset < 0 then Offset := 0;
  sbDetail.VertScrollBar.Position := Offset;
end;

{ The first subsection of a section that the current mode still shows, so
  opening a section always lands somewhere real. }
function TfmMain.FirstSubOf(ASection: Integer): Integer;
var
  i: Integer;
begin
  Result := -1;
  for i := 0 to High(FSubs) do
    if (FSubs[i].Section = ASection) and FSubs[i].Btn.Visible then
      Exit(i);
end;

{ Highlights one card and scrolls it to the top.

  This used to be a flash that faded out over a third of a second, which
  answered "did my click do anything" but not "where am I now" -- and on a
  section whose groups all fit, the scroll said nothing either.  Holding the
  highlight answers both, and costs one repaint instead of four. }
procedure TfmMain.SelectSub(AIndex: Integer);
begin
  FSub := AIndex;
  UpdateNavState;
  if (AIndex >= 0) and (AIndex <= High(FSubs)) then
    ScrollToGroup(FSubs[AIndex].Box);
end;

procedure TfmMain.SubClick(Sender: TObject);
var
  i: Integer;
begin
  if not (Sender is TSpeedButton) then Exit;
  i := TSpeedButton(Sender).Tag;
  if (i < 0) or (i > High(FSubs)) then Exit;

  if FSubs[i].Section <> FSection then SelectSection(FSubs[i].Section);
  SelectSub(i);
end;

{ The chevron on a header says whether that section's subsections are showing.
  It is a glyph rather than a caption prefix because TSpeedButton centres its
  caption and has no Alignment -- but it does place a glyph at Margin, and the
  caption follows the glyph, which left-aligns the header. }
procedure TfmMain.UpdateNavState;
var
  i: Integer;
  Closed, Open: Integer;
begin
  Closed := McxIconIndex('collapsed');
  Open := McxIconIndex('expanded');
  for i := 0 to High(FHeads) do
  begin
    if i = FSection then FHeads[i].ImageIndex := Open
    else FHeads[i].ImageIndex := Closed;
    FHeads[i].Down := (i = FSection);
    { The open section is filled with the selection colour, the closed ones
      sit on a neutral band: level, then state, and nothing else. }
    if i = FSection then
      FPanes[i].Color := McxBlend(clBtnFace, clHighlight, BandActive)
    else
      FPanes[i].Color := McxBlend(clBtnFace, clWindowText, BandLevel);
  end;

  for i := 0 to High(FSubs) do
  begin
    FSubs[i].Btn.Down := (i = FSub);
    { The title takes the selection colour rather than a background, because
      a filled row under a filled heading reads as a second heading. }
    if i = FSub then
      FSubs[i].Btn.Font.Color := McxBlend(clWindowText, clHighlight, 80)
    else
      FSubs[i].Btn.Font.Color := clDefault;

    { The card the title points at is held highlighted for as long as it is
      the selected one, and its own heading goes with it. }
    if i = FSub then
    begin
      SetCard(FSubs[i].Box, McxBlend(clBtnFace, clHighlight, CardActive));
      FSubs[i].Cap.Font.Color := McxBlend(clWindowText, clHighlight, 80);
    end
    else
    begin
      SetCard(FSubs[i].Box, McxBlend(clBtnFace, clWindowText, CardLevel));
      FSubs[i].Cap.Font.Color := clDefault;
    end;
  end;
end;

procedure TfmMain.HeaderClick(Sender: TObject);
var
  i: Integer;
begin
  for i := 0 to High(FHeads) do
    if FHeads[i] = Sender then
    begin
      SelectSection(i);
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
  i, N: Integer;
  C: TComponent;
  P: TWinControl;
  Letters: string;

  function IndexOfRow(APanel: TPanel): Integer;
  begin
    for Result := 0 to High(FRowList) do
      if FRowList[Result] = APanel then Exit;
    Result := -1;
  end;

begin
  SetLength(FBound, Length(Binds));
  SetLength(FLabels, Length(Binds));
  SetLength(FLoaded, Length(Binds));
  SetLength(FRows, Length(Binds));
  SetLength(FGroups, Length(Binds));
  SetLength(FRowList, 0);
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
    FRows[i] := nil;
    FGroups[i] := nil;
    P := FBound[i].Parent;
    while (P <> nil) and (FGroups[i] = nil) do
    begin
      { A row panel and the card around it are both plain panels, so each is
        recognised by its name prefix rather than by its class -- which keeps
        the designer the only place either one is declared. }
      if (FRows[i] = nil) and (P is TPanel) and
         (Copy(P.Name, 1, 2) = 'rw') then
      begin
        FRows[i] := TPanel(P);
        N := IndexOfRow(FRows[i]);
        if N < 0 then
        begin
          SetLength(FRowList, Length(FRowList) + 1);
          FRowList[High(FRowList)] := FRows[i];
        end;
      end;
      if (P is TPanel) and (Copy(P.Name, 1, 2) = 'gb') then
        FGroups[i] := TPanel(P);
      P := P.Parent;
    end;
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
    begin
      { LCL clamps only when MaxValue is greater than MinValue
        (spinedit.inc:228), so a floor cannot be set without a ceiling.  A row
        that gives a floor and no ceiling gets the largest one there is --
        which enforces the floor and refuses nothing a file could hold. }
      TSpinEdit(C).MinValue := Round(Binds[i].Min);
      if Binds[i].Max > Binds[i].Min then
        TSpinEdit(C).MaxValue := Round(Binds[i].Max)
      else if Binds[i].Min > 0 then
        TSpinEdit(C).MaxValue := High(LongInt)
      else
        TSpinEdit(C).MaxValue := 0;
      TSpinEdit(C).OnChange := @BindChanged;
    end
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
  i, j, s, First, NBound, NShown: Integer;
  B: TMcxBind;
  C: TControl;
  InMode, Shown: Boolean;
  Live: array of Boolean;
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

  { A row whose every control the wizard hid goes with them, or its caption
    is left standing beside nothing.  A row shared by two check boxes stays
    as long as one of them is showing. }
  for j := 0 to High(FRowList) do
  begin
    NBound := 0;
    NShown := 0;
    for i := 0 to High(Binds) do
      if (FBound[i] <> nil) and (FRows[i] = FRowList[j]) then
      begin
        Inc(NBound);
        if FBound[i].Visible then Inc(NShown);
      end;
    if NBound > 0 then FRowList[j].Visible := NShown > 0;
  end;

  { A group box the wizard has emptied hides, and so does its entry in the
    navigator: an empty frame under a heading reads as something broken.

    A group holding no bound control at all -- the media table, the shape
    list, the device list -- always stays, because what it will hold is a key
    setting in either mode. }
  SetLength(Live, Length(FSubs));
  for j := 0 to High(FSubs) do
  begin
    NBound := 0;
    NShown := 0;
    for i := 0 to High(Binds) do
      if (FBound[i] <> nil) and (FGroups[i] = FSubs[j].Box) then
      begin
        Inc(NBound);
        if FBound[i].Visible then Inc(NShown);
      end;
    Live[j] := (NBound = 0) or (NShown > 0);
  end;

  for j := 0 to High(FSubs) do
  begin
    FSubs[j].Box.Visible := Live[j];
    FSubs[j].Btn.Visible := Live[j];
  end;

  { A section whose every group went away would leave a header that opens on
    nothing, so it goes too -- and if it was the one being shown, the first
    surviving section takes over. }
  First := -1;
  for s := 0 to High(FPanes) do
  begin
    Shown := False;
    for j := 0 to High(FSubs) do
      if (FSubs[j].Section = s) and Live[j] then Shown := True;
    FPanes[s].Visible := Shown;
    if Shown and (First < 0) then First := s;
  end;
  if (FSection >= 0) and (not FPanes[FSection].Visible) then
    SelectSection(First)
  else if (FSub < 0) or (FSub > High(FSubs)) or (not FSubs[FSub].Btn.Visible) then
    { The wizard took away the card that was highlighted, so the highlight
      moves to the first one the section still has rather than pointing at
      something nobody can see. }
    SelectSub(FirstSubOf(FSection));
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
