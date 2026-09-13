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
  StdCtrls, Buttons, ActnList, Menus, ImgList, ClipBrd, Spin, Grids, LCLType,
  fpjson,
  AnchorDocking, AnchorDockPanel, AnchorDockStorage, XMLPropStorage,
  mcxdpi, mcxicons, mcxtheme, mcxdoc, mcxhelp, mcxabout, mcxrun, mcxgl, mcxview,
  mcxdisp, mcxtable, mcxjd;

type
  { One navigator entry below a section header: the group box on the detail
    page it points at, and the button that points there.  Built from the
    Sections table rather than placed, so the two cannot drift apart. }
  { One set of alTop siblings and the order they belong in. }
  TMcxStack = record
    Parent: TWinControl;
    Items: array of TControl;
  end;

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
    acSaveImage: TAction;
    acBenchmark: TAction;
    acLoadResult: TAction;
    acResetLayout: TAction;
    acAbout: TAction;
    acHelp: TAction;
    acTheme: TAction;
    pmTheme: TPopupMenu;
    alMain: TActionList;
    dlgOpen: TOpenDialog;
    dlgResult: TOpenDialog;
    dlgImage: TSaveDialog;
    dlgSave: TSaveDialog;
    ilIcons: TImageList;
    lbMaxJump: TLabel;
    edMaxJump: TSpinEdit;
    rwMaxJump: TPanel;
    lbTodoGL: TLabel;
    mmCommand: TMemo;
    mmJSON: TMemo;
    mmLog: TMemo;
    pcView: TPageControl;
    pmBench: TPopupMenu;
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
    tbSep3: TToolButton;
    tbImage: TToolButton;
    tbBench: TToolButton;
    tbResult: TToolButton;
    tbDock: TToolButton;
    tbMode: TToolButton;
    tmRefresh: TTimer;
    tsSettings: TTabSheet;
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
    procedure FormShow(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormCloseQuery(Sender: TObject; var CanClose: Boolean);
    procedure acNewExecute(Sender: TObject);
    procedure acOpenExecute(Sender: TObject);
    procedure acSaveExecute(Sender: TObject);
    procedure acSaveAsExecute(Sender: TObject);
    procedure acToggleModeExecute(Sender: TObject);
    procedure acSaveImageExecute(Sender: TObject);
    procedure acBenchmarkExecute(Sender: TObject);
    procedure acLoadResultExecute(Sender: TObject);
    procedure acResetLayoutExecute(Sender: TObject);
    procedure acRunExecute(Sender: TObject);
    procedure acStopExecute(Sender: TObject);
    procedure acDevicesExecute(Sender: TObject);
    procedure acAboutExecute(Sender: TObject);
    procedure acHelpExecute(Sender: TObject);
    procedure acThemeExecute(Sender: TObject);
    procedure FormKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
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
    { What each item of a choice control actually writes.  Parallel to the
      control's own Items, which hold what the person reads. }
    FValues: array of TStringList;
    { The flag letters of a check group, one string per binding.  They used to
      ride on the control's Hint; a hint is a thing the person reads. }
    FFlags: array of string;
    { Which debug-flag list the check group is currently showing. }
    FDebugList: string;
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
    { The wizard's step bar, built rather than placed: it belongs to a mode
      rather than to the form, and in expert mode it is not there at all. }
    FStepBar: TPanel;
    FStepBack: TButton;
    FStepNext: TButton;
    FStepText: TLabel;
    { Every set of alTop siblings the wizard filter can hide something from,
      each in the order it is meant to appear.  See Restack. }
    FStacks: array of TMcxStack;
    { The dock site and the three panes in it.  See BuildDock. }
    FDockSite: TAnchorDockPanel;
    FPaneNav: TForm;
    FPaneView: TForm;
    { The simulation in flight, or nil.  One at a time: a second run would
      write over the first one's output files. }
    FRunner: TMcxRunner;
    FDevices: TMcxDevices;
    { The 3-D preview, which owns its own GL control, camera and geometry. }
    FView: TMcxView;
    { The two settings that are lists rather than values.  How many media a
      simulation has, and how many detectors, is not known until a file is
      opened, so these are the only editors built in code. }
    FMedia: TMcxTable;
    FDetectors: TMcxTable;
    FDisplay: TMcxDisplayBar;
    { The last focused control the help table knows about.  Kept because the
      Help button is on the toolbar: pressing it takes the focus off the
      setting whose help was wanted. }
    FLastHelp: TControl;
    FPendingResult: string;
    FDockRestored: Boolean;
    FDockSized: Boolean;
    FWizard: Boolean;
    procedure BuildIcons;
    procedure CollectSections;
    procedure BuildNav;
    function  MakePane(const AName, ACaption: string; AControl: TControl;
      AWidth, AHeight: Integer): TForm;
    procedure BuildDock;
    procedure SizePane(APane: TForm; AAlign: TAlign; AWanted: Integer);
    procedure ApplyPaneSizes(Data: PtrInt);
    procedure GuardCentreHeader;
    procedure DockCreateControl(Sender: TObject; aName: string;
      var AControl: TControl; DoDisableAutoSizing: boolean);
    procedure ViewLog(Sender: TObject; const AText: string);
    procedure ShowResult(const AFileName: string);
    procedure LogDetected(const AFileName: string);
    procedure BenchmarkClick(Sender: TObject);
    procedure ViewPick(Sender: TObject; const AInfo: TMcxPickInfo);
    procedure TableChanged(Sender: TObject);
    function  Environment: string;
    procedure ShowHelpForFocus;
    procedure ApplyTheme;
    procedure BuildThemeMenu;
    procedure ThemeClick(Sender: TObject);
    procedure FocusChanged(Sender: TObject; LastControl: TControl);
    procedure DropResult;
    procedure GuessRunSettings;
    function  CurrentBackend: TMcxBackend;
    function  CurrentExe: string;
    procedure UpdateRunActions;
    procedure RunLine(Sender: TObject; const ALine: string);
    procedure RunProgress(Sender: TObject; APercent: Integer);
    procedure RunDone(Sender: TObject; AExitCode: Integer);
    procedure Log(const AText: string);
    procedure SaveDockLayout;
    function  LoadDockLayout: Boolean;
    procedure CaptureStacks;
    procedure Restack;
    procedure BindControls;
    procedure BindChanged(Sender: TObject);
    procedure CheckGroupClick(Sender: TObject; Index: Integer);
    function  DocFor(const APath: string): TMcxDoc;
    procedure LoadBinding(AIndex: Integer);
    procedure SaveBinding(AIndex: Integer);
    procedure LoadAllBindings;
    procedure ApplyBindingStates;
    procedure BuildStepBar;
    procedure UpdateStepBar;
    procedure StepClick(Sender: TObject);
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
    { Loads a simulation, reporting any trouble the way the Open action does.
      Public because the program opens a file named on the command line. }
    procedure OpenDocument(const AFileName: string);
    { Shows a result once the window is up.  Not at once: the GL context does
      not exist until the control has been realised, and a volume cannot be
      uploaded before there is somewhere to upload it to. }
    procedure ShowResultLater(const AFileName: string);
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
  { How many photon positions to keep when trajectories are recorded.

    The number that matters is photons x path length: a photon in tissue
    scatters a few tens of times before it leaves, so a buffer smaller than
    that does not record fewer paths -- it records every path and cuts all of
    them off in mid-flight, which looks like photons dying in open tissue.
    Two million covers a few tens of thousands of photons, which is what you
    would launch if you wanted to look at the paths at all. }
  DefaultMaxJump = 2000000;

  { Corner radius on the design grid, scaled with everything else. }
  CardRadius = 14;

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
  CaptureStacks;

  mmJSON.Font.Name := McxDefaultFontName;
  mmCommand.Font.Assign(mmJSON.Font);
  mmLog.Font.Assign(mmJSON.Font);

  BuildDock;
  { The placeholders these replace go with them. }
  lbTodoMedia.Visible := False;
  lbTodoDet.Visible := False;
  BuildStepBar;
  FMedia := TMcxTable.Create(gbMedia, 'mua (1/mm),mus (1/mm),g,n',
    'mua,mus,g,n', 6);
  FMedia.OnChange := @TableChanged;
  FDetectors := TMcxTable.Create(gbDetector, 'x,y,z,radius',
    'Pos[0],Pos[1],Pos[2],R', 4);
  FDetectors.OnChange := @TableChanged;

  FView := TMcxView.Create(pnGL);
  FView.OnLog := @ViewLog;
  FView.OnPick := @ViewPick;
  FView.Document := FDoc;
  lbTodoGL.Visible := False;
  { On the page, not on pnGL: pnGL is the GL control's host and the control
    fills it. }
  FDisplay := TMcxDisplayBar.Create(tsPreview, FView);

  { F1 is answered by the form, not by each control: the keys reach here
    first, and the control that has focus is the question. }
  KeyPreview := True;
  OnKeyDown := @FormKeyDown;
  Screen.AddHandlerActiveControlChanged(@FocusChanged);

  LoadDockLayout;

  { After the layout, not before: restoring it hands the dock manager its
    saved settings, and the header style is one of them -- so a theme applied
    first had its header drawer quietly replaced by the one in the file. }
  McxLoadTheme;
  BuildThemeMenu;
  ApplyTheme;

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
  { Screen outlives the form, so the handler has to come off it. }
  Screen.RemoveHandlerActiveControlChanged(@FocusChanged);
  FreeAndNil(FView);
  FreeAndNil(FMedia);
  FreeAndNil(FDetectors);
  FreeAndNil(FMissing);
  FreeAndNil(FRun);
  FreeAndNil(FDoc);
end;

procedure TfmMain.FormCloseQuery(Sender: TObject; var CanClose: Boolean);
begin
  CanClose := ConfirmDiscard;
  if CanClose then SaveDockLayout;
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
  acResetLayout.ImageIndex := McxIconIndex('reset');
  acLoadResult.ImageIndex := McxIconIndex('preview');
  acBenchmark.ImageIndex := McxIconIndex('bench');
  acSaveImage.ImageIndex := McxIconIndex('saveas');
  acAbout.ImageIndex := McxIconIndex('about');
  acHelp.ImageIndex := McxIconIndex('help');
  acTheme.ImageIndex := McxIconIndex('theme');
end;

{ ------------------------------------------------------------ navigator --- }

{ Selecting a section shows its page and lists its subsections underneath the
  header; selecting a subsection scrolls that group box to the top of the
  page.  Nothing is edited in the navigator itself -- it exists to keep forty
  settings from arriving as one long column. }

type
  { Color and ParentColor are protected in TControl and only published by the
    concrete classes; a descendant declared in this unit reaches them without
    caring which class it was handed. }
  TColourAccess = class(TWinControl);
  { The same trick for Font and ParentFont, which TControl also keeps
    protected. }
  TFontAccess = class(TControl);

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
  TColourAccess(W).Color := McxBase;
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
    FBodies[i].Color := McxBase;
  end;
  sbNav.ParentColor := False;
  sbNav.Color := McxBase;
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
        Cap.Font.Height := -18;
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
  UpdateStepBar;
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
{ The header of a docked pane, drawn in the theme's colours.

  AnchorDocking's own styles fill the header with clForm and outline the grip
  with DrawEdge -- both system colours, so under a light theme on a dark
  desktop the one black bar left in the window was the pane header.  A style
  registered here paints the whole thing instead: registered with
  NeedDrawHeaderAfterText false, which is what stops the dock manager filling
  it with clForm first.

  The grip is worth keeping.  It is the only thing on a pane that says it can
  be dragged somewhere else. }
procedure DrawMcxDockHeader(Canvas: TCanvas; Style: TADHeaderStyleDesc;
  r: TRect; Horizontal: Boolean; Focused: Boolean);
var
  Face, Ink: TColor;
  i, x, y: Integer;
begin
  if Focused then Face := McxBlend(McxBase, McxAccent, 26)
  else Face := McxBlend(McxBase, McxText, 9);
  Ink := McxBlend(Face, McxText, 38);

  Canvas.Brush.Style := bsSolid;
  Canvas.Brush.Color := Face;
  Canvas.FillRect(r);

  { A hairline along the edge the pane's contents start at, so the header
    reads as a band rather than as a gap. }
  Canvas.Pen.Color := McxBlend(Face, McxText, 18);
  if Horizontal then
  begin
    Canvas.MoveTo(r.Left, r.Bottom - 1);
    Canvas.LineTo(r.Right, r.Bottom - 1);
  end
  else
  begin
    Canvas.MoveTo(r.Right - 1, r.Top);
    Canvas.LineTo(r.Right - 1, r.Bottom);
  end;

  Canvas.Pen.Color := Ink;
  for i := 0 to 2 do
    if Horizontal then
    begin
      y := (r.Top + r.Bottom) div 2 - 3 + i * 3;
      Canvas.MoveTo(r.Left + 6, y);
      Canvas.LineTo(r.Left + 22, y);
    end
    else
    begin
      x := (r.Left + r.Right) div 2 - 3 + i * 3;
      Canvas.MoveTo(x, r.Top + 6);
      Canvas.LineTo(x, r.Top + 22);
    end;
end;

{ Repaints the window in the current theme.

  Only the surfaces the program paints itself are touched.  The widgets --
  edit boxes, combo boxes, the scroll bars -- are the widget set's and are
  left alone deliberately: on gtk2 half of them ignore Color anyway, and a
  window with hand-painted panels and native controls looks better than one
  where every second control has missed the change. }
procedure TfmMain.ApplyTheme;

  { The text colour, on everything that draws its caption straight onto the
    surface behind it.

    Not on the native widgets -- an edit box, a combo box, a grid keep their
    own background from the widget set whatever we ask, and giving one of
    those the theme's ink is how you get dark text on a dark field.  The two
    memos are handled below, where their background is set as well. }
  procedure SetInk(AControl: TControl; AColor: TColor);
  begin
    TFontAccess(AControl).ParentFont := False;
    TFontAccess(AControl).Font.Color := AColor;
  end;

  procedure Ink(AControl: TControl);
  var
    k: Integer;
  begin
    if AControl = nil then Exit;
    if (AControl is TLabel) or (AControl is TSpeedButton) or
       (AControl is TCheckBox) or (AControl is TRadioButton) or
       (AControl is TPanel) or (AControl is TCustomGroupBox) or
       (AControl is TTabSheet) or (AControl is TPageControl) or
       (AControl is TToolBar) or (AControl is TStatusBar) or
       (AControl is TScrollBox) or (AControl is TAnchorDockHeader) then
      SetInk(AControl, McxText)
    else if (AControl is TCustomEdit) or (AControl is TCustomListBox) or
            (AControl is TStringGrid) then
    begin
      { A field carries its own background, so it gets both halves: the paper
        colour and the ink to go on it.  Giving it only the ink -- which is
        what inheriting the form's font does -- is how a window ends up with
        the theme's dark text on the desktop's dark field. }
      SetInk(AControl, McxText);
      TColourAccess(AControl).ParentColor := False;
      TColourAccess(AControl).Color := McxBlend(McxBase, McxText, 4);
      { A grid's fixed cells have a colour of their own, and left behind they
        are a dark header over a light table. }
      if AControl is TStringGrid then
        TStringGrid(AControl).FixedColor := McxBlend(McxBase, McxText, 14);
    end
    else if (AControl is TCustomButton) or (AControl is TCustomComboBox) then
      { Drawn by the widget set, frame and face together -- a push button
        entirely, and a drop-down list's closed face -- so there is no way to
        give either half a palette.  clDefault hands the pairing back to the
        only thing that knows both halves of it. }
      SetInk(AControl, clDefault);
    if AControl is TWinControl then
      for k := 0 to TWinControl(AControl).ControlCount - 1 do
        Ink(TWinControl(AControl).Controls[k]);
  end;

  { Everything that inherits its colours does so from the form; the ones that
    were given their own in the designer are named below. }
  procedure Paper(AControl: TWinControl);
  begin
    if AControl = nil then Exit;
    TColourAccess(AControl).ParentColor := False;
    TColourAccess(AControl).Color := McxBlend(McxBase, McxText, 4);
    TColourAccess(AControl).Font.Color := McxText;
  end;

var
  i: Integer;
  r, g, b: Byte;
begin
  { The dock manager's headers are painted by a procedure of ours rather than
    by one of its own; registering is idempotent, and the drawer reads the
    theme when it paints, so this only has to be asked for once. }
  DockMaster.RegisterHeaderStyle('MCX', @DrawMcxDockHeader, False, False);
  DockMaster.HeaderStyle := 'MCX';

  { First: the widget set's own resource style.  There is a layer of the
    window the LCL cannot colour -- a notebook's tab strip, the status bar's
    face, a docked pane's header, the field behind selected text -- and gtk2
    draws all of it from its own style whatever we set on the control in
    front.  Handing it the same three roles is the only way those follow. }
  if McxCurrentTheme = mtSystem then
    McxApplyChromeColours(clNone, clNone, clNone, clNone, clNone)
  else
    McxApplyChromeColours(McxBase, McxText, McxAccent,
      McxBlend(McxBase, McxText, 4), McxReadable(McxAccent));

  Color := McxBase;
  Font.Color := McxText;

  sbNav.Color := McxBase;
  for i := 0 to High(FBodies) do FBodies[i].Color := McxBase;

  { The bars and the notebook too, where the widget set lets them be told.
    Where it does not -- gtk2 draws a tab strip itself -- the request is
    harmless and the strip stays the desktop's. }
  tbMain.ParentColor := False;
  tbMain.Color := McxBlend(McxBase, McxText, 6);
  sbMain.ParentColor := False;
  sbMain.Color := McxBlend(McxBase, McxText, 6);
  TColourAccess(pcView).ParentColor := False;
  TColourAccess(pcView).Color := McxBase;
  for i := 0 to pcView.PageCount - 1 do
  begin
    TColourAccess(pcView.Pages[i]).ParentColor := False;
    TColourAccess(pcView.Pages[i]).Color := McxBase;
  end;
  if sbDetail <> nil then
  begin
    sbDetail.ParentColor := False;
    sbDetail.Color := McxBase;
  end;

  { Every form, not just this one.  AnchorDocking puts each pane in a form of
    its own and re-parents our panels into it, so half the window -- the pane
    headers, the notebook, the splitters -- is not in this form's control
    tree at all, and a walk from Self stops at the dock site. }
  for i := 0 to Screen.FormCount - 1 do
  begin
    Ink(Screen.Forms[i]);
    TColourAccess(Screen.Forms[i]).Color := McxBase;
    Screen.Forms[i].Font.Color := McxText;
  end;

  { The text panes read as paper: a shade off the surface, so a block of JSON
    is a thing on the window rather than the window itself. }
  Paper(mmJSON);
  Paper(mmCommand);
  Paper(mmLog);

  if FView <> nil then
  begin
    { Dark in every theme, only tinted by the surface.  The wireframe, the
      grid and the axis labels are drawn light, and a light background would
      lose all three -- and a 3-D view is read against its own contents, not
      against the window it sits in. }
    RedGreenBlue(ColorToRGB(McxBlend(clBlack, McxBase, 22)), r, g, b);
    FView.Background := McxVec3(r / 255, g / 255, b / 255);
    FView.Redraw;
  end;

  UpdateNavState;
  Invalidate;
end;

procedure TfmMain.ThemeClick(Sender: TObject);
begin
  McxSetTheme(TMcxTheme(TMenuItem(Sender).Tag));
  McxSaveTheme;
  BuildThemeMenu;
  ApplyTheme;
end;

{ Rebuilt rather than ticked in place, because it is short and because the
  check mark is the only state it has. }
procedure TfmMain.BuildThemeMenu;
var
  T: TMcxTheme;
  Item: TMenuItem;
begin
  pmTheme.Items.Clear;
  for T := Low(TMcxTheme) to High(TMcxTheme) do
  begin
    Item := TMenuItem.Create(pmTheme);
    Item.Caption := McxThemeNames[T];
    Item.Tag := Ord(T);
    Item.RadioItem := True;
    Item.Checked := (T = McxCurrentTheme);
    Item.OnClick := @ThemeClick;
    pmTheme.Items.Add(Item);
  end;
end;

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
      FPanes[i].Color := McxBlend(McxBase, McxAccent, BandActive)
    else
      FPanes[i].Color := McxBlend(McxBase, McxText, BandLevel);
    { The heading sits on the band, and the open band is the accent -- a
      fixed colour, where the surface is not.  So what can be read on it is
      decided from the band and not from the theme's text colour. }
    FHeads[i].Font.Color := McxReadable(FPanes[i].Color);
  end;

  for i := 0 to High(FSubs) do
  begin
    FSubs[i].Btn.Down := (i = FSub);
    { The title takes the selection colour rather than a background, because
      a filled row under a filled heading reads as a second heading. }
    if i = FSub then
      FSubs[i].Btn.Font.Color := McxBlend(McxText, McxAccent, 80)
    else
      FSubs[i].Btn.Font.Color := McxText;

    { The card the title points at is held highlighted for as long as it is
      the selected one, and its own heading goes with it. }
    if i = FSub then
    begin
      SetCard(FSubs[i].Box, McxBlend(McxBase, McxAccent, CardActive));
      FSubs[i].Cap.Font.Color := McxBlend(McxText, McxAccent, 80);
    end
    else
    begin
      SetCard(FSubs[i].Box, McxBlend(McxBase, McxText, CardLevel));
      FSubs[i].Cap.Font.Color := McxText;
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

{ --------------------------------------------------------------- wizard --- }

{ Back and Next across the sections, for the mode that is meant to be walked
  rather than browsed.

  The plan called for this over the same panels rather than a second layout,
  and that is what it is: the buttons move the same selection the navigator
  moves, and hiding the bar is the whole of turning the wizard off.  There is
  no wizard state to keep -- where you are is which section is open. }
procedure TfmMain.BuildStepBar;
begin
  FStepBar := TPanel.Create(Self);
  FStepBar.Parent := tsSettings;
  FStepBar.Align := alBottom;
  { Raw 96-dpi numbers: built in FormCreate, so the startup sweep scales it. }
  FStepBar.Height := 40;
  FStepBar.BevelOuter := bvNone;
  FStepBar.Caption := '';

  FStepText := TLabel.Create(FStepBar);
  FStepText.Parent := FStepBar;
  FStepText.Align := alLeft;
  FStepText.Layout := tlCenter;
  FStepText.BorderSpacing.Left := 10;

  FStepNext := TButton.Create(FStepBar);
  FStepNext.Parent := FStepBar;
  FStepNext.Align := alRight;
  FStepNext.Width := 90;
  FStepNext.BorderSpacing.Around := 6;
  FStepNext.Caption := 'Next >';
  FStepNext.OnClick := @StepClick;

  FStepBack := TButton.Create(FStepBar);
  FStepBack.Parent := FStepBar;
  FStepBack.Align := alRight;
  FStepBack.Width := 90;
  FStepBack.BorderSpacing.Around := 6;
  FStepBack.Caption := '< Back';
  FStepBack.OnClick := @StepClick;
end;

procedure TfmMain.UpdateStepBar;
var
  i, Shown, At: Integer;
begin
  if FStepBar = nil then Exit;
  FStepBar.Visible := FWizard;
  { The binding sweep runs once before any section has been chosen -- from
    NewDocument, which FormCreate calls before SelectSection -- so there is a
    moment when there is no step to name. }
  if (not FWizard) or (FSection < 0) or (FSection > High(FHeads)) then Exit;

  { Counted rather than remembered, because the wizard filter can take a
    whole section away and the step numbers have to follow. }
  Shown := 0;
  At := 0;
  for i := 0 to High(FPanes) do
    if FPanes[i].Visible then
    begin
      Inc(Shown);
      if i = FSection then At := Shown;
    end;

  if Shown = 0 then FStepText.Caption := ''
  else FStepText.Caption := Format('Step %d of %d:  %s',
    [At, Shown, FHeads[FSection].Caption]);

  FStepBack.Enabled := At > 1;
  FStepNext.Enabled := (At > 0) and (At < Shown);
end;

procedure TfmMain.StepClick(Sender: TObject);
var
  Dir, i: Integer;
begin
  if Sender = FStepBack then Dir := -1 else Dir := 1;
  i := FSection + Dir;
  { Straight past anything the filter has hidden, so Next never lands on a
    section with nothing in it. }
  while (i >= 0) and (i <= High(FPanes)) and (not FPanes[i].Visible) do
    Inc(i, Dir);
  if (i >= 0) and (i <= High(FPanes)) then SelectSection(i);
end;

{ --------------------------------------------------------------- modes ---- }

procedure TfmMain.acToggleModeExecute(Sender: TObject);
begin
  FWizard := not FWizard;
  ApplyBindingStates;
  UpdateStepBar;
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
  FRun.SetNum('@run.maxjumpdebug', DefaultMaxJump);
  FRun.Modified := False;
  DropResult;
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

{ Which simulator and which kind of domain a file is for.

  Neither is written in the file -- they are how it is run, not what it says
  -- so they are read off its shape.  A file with a Mesh block is an MMC
  input and nothing else: no other backend has one, and opening it as a
  voxel simulation shows an empty 0 x 0 x 0 domain and a Run button that
  would fail. }
{ Forgets whatever result is on the card. }
procedure TfmMain.DropResult;
begin
  if FView = nil then Exit;
  FView.ClearVolume;
  FView.ClearTrajectory;
  if FDisplay <> nil then
  begin
    FDisplay.SetHasVolume(False);
    { An empty range hides the photon row. }
    FDisplay.SetTrajectory(1, 0);
  end;
end;

procedure TfmMain.GuessRunSettings;
begin
  { Seeded on every open, because it is a run setting and a document does not
    carry one; without it the first run after opening a file would take mcx's
    ten-million default. }
  if FRun.AsInt('@run.maxjumpdebug') <= 0 then
    FRun.SetNum('@run.maxjumpdebug', DefaultMaxJump);
  if (FDoc.Find('Mesh') <> nil) or (FDoc.Find('Shapes.MeshNode') <> nil) then
  begin
    FRun.SetStr('@run.backend', 'mmc');
    FRun.SetStr('@run.domainkind', 'mesh');
  end
  else if FDoc.AsStr('Domain.VolumeFile', '') <> '' then
    FRun.SetStr('@run.domainkind', 'voxel')
  else
    FRun.SetStr('@run.domainkind', 'shapes');
  FRun.Modified := False;
end;

procedure TfmMain.OpenDocument(const AFileName: string);
begin
  if not FDoc.LoadFromFile(AFileName) then
  begin
    MessageDlg('MCX Studio',
      'Could not read ' + ExtractFileName(AFileName) + ':'#10#10 +
      FDoc.LastError, mtError, [mbOK], 0);
    Exit;
  end;
  GuessRunSettings;
  { The old file's result is not this file's.  Nothing cleared it before, so
    opening a second simulation left the first one's fluence hanging in the
    view with the new domain drawn around it. }
  DropResult;
  LoadAllBindings;
  UpdateTitle;
  RefreshPreview;
end;

procedure TfmMain.acOpenExecute(Sender: TObject);
begin
  if not ConfirmDiscard then Exit;
  if not dlgOpen.Execute then Exit;
  OpenDocument(dlgOpen.FileName);
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

{ The machine-specific half of the About box: which binaries this copy found
  and what the driver calls itself.  Collected here because only the form
  knows where the view is. }
function TfmMain.Environment: string;
const
  BackendNames: array[TMcxBackend] of string =
    ('mcx', 'mcxcl', 'mmc', 'mcx-hip');
var
  B: TMcxBackend;
  Exe: string;
begin
  Result := '';
  for B := Low(TMcxBackend) to High(TMcxBackend) do
  begin
    Exe := McxFindExe(B);
    if Exe = '' then Exe := '(not found)';
    { The backend's label, not its file name: the CUDA and the ROCm build are
      both called mcx, and two lines reading "mcx:" say nothing. }
    Result := Result + Format('%-9s %s'#10, [BackendNames[B] + ':', Exe]);
  end;
  if FView <> nil then Result := Result + 'OpenGL:  ' + FView.Description + #10;
end;

procedure TfmMain.acAboutExecute(Sender: TObject);
begin
  McxShowAbout(Self, Environment);
end;

{ F1 answers for whatever has the keyboard, which is the setting the person is
  looking at.  A control the help table does not know is walked up from --
  a spin edit's inner edit, a radio group's button -- and only when nothing on
  the way up is known does the overview come back. }
procedure TfmMain.FormKeyDown(Sender: TObject; var Key: Word;
  Shift: TShiftState);
begin
  if Key <> VK_F1 then Exit;
  Key := 0;
  ShowHelpForFocus;
end;

procedure TfmMain.acHelpExecute(Sender: TObject);
begin
  ShowHelpForFocus;
end;

{ The button itself drops the menu: a theme is a list to pick from, and there
  is nothing sensible for a plain click to do. }
procedure TfmMain.acThemeExecute(Sender: TObject);
begin
  pmTheme.PopUp;
end;

{ Remembered as the focus moves rather than read when help is asked for: by
  then the Help button has the focus and the setting has lost it. }
procedure TfmMain.FocusChanged(Sender: TObject; LastControl: TControl);
var
  C: TControl;
begin
  C := Screen.ActiveControl;
  while C <> nil do
  begin
    if McxHelpFor(C.Name) <> '' then
    begin
      FLastHelp := C;
      Exit;
    end;
    C := C.Parent;
  end;
end;

procedure TfmMain.ShowHelpForFocus;
var
  C: TControl;
  Body, Title: string;
begin
  Body := '';
  Title := '';
  C := ActiveControl;
  if (C = nil) or (McxHelpFor(C.Name) = '') then C := FLastHelp;
  while (C <> nil) and (Body = '') do
  begin
    Body := McxHelpFor(C.Name);
    if Body <> '' then
    begin
      { What the setting is called on screen, never the control's name: a
        person who needed to be told it was rgBackend would not have pressed
        F1 over it. }
      Title := '';
      if (C.Tag >= 0) and (C.Tag <= High(Binds)) and (FBound[C.Tag] = C) and
         (FLabels[C.Tag] <> nil) then
        Title := FLabels[C.Tag].Caption;
      if (Title = '') and ((C is TRadioGroup) or (C is TCheckGroup)) then
        Title := TCustomGroupBox(C).Caption;
      if Title = '' then Title := 'This setting';
    end;
    C := C.Parent;
  end;
  if Body = '' then
  begin
    McxShowHelp(Self, 'MCX Studio help', McxHelpOverview);
    Exit;
  end;
  McxShowHelp(Self, Title, Title + LineEnding + LineEnding + Body);
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

{ ------------------------------------------------------------- running ---- }

function TfmMain.CurrentBackend: TMcxBackend;
var
  S: string;
begin
  S := LowerCase(FRun.AsStr('@run.backend', 'mcx'));
  if S = 'mcxcl' then Result := mbMCXCL
  else if S = 'mmc' then Result := mbMMC
  else if S = 'mcx-hip' then Result := mbHIP
  else Result := mbMCX;
end;

function TfmMain.CurrentExe: string;
begin
  Result := McxFindExe(CurrentBackend);
end;

{ Run and Devices depend on a backend being installed, and which backend is
  asked for is itself a setting -- so this is re-checked whenever a setting
  changes rather than once at startup. }
procedure TfmMain.UpdateRunActions;
begin
  acRun.Enabled := (FRunner = nil) and (CurrentExe <> '');
  acStop.Enabled := FRunner <> nil;
  acDevices.Enabled := (CurrentExe <> '') and (CurrentBackend <> mbMMC);
end;

{ The view has no log of its own; this is where its few lines go. }
procedure TfmMain.ViewLog(Sender: TObject; const AText: string);
begin
  Log(AText);
end;

{ A table writes into the document itself, so there is nothing to save here --
  only the same tail every other edit runs: mark it modified, redraw the
  preview, and let the sections that depend on it catch up. }
procedure TfmMain.TableChanged(Sender: TObject);
begin
  UpdateTitle;
  SchedulePreview;
end;

procedure TfmMain.Log(const AText: string);
begin
  mmLog.Lines.Add(AText);
  { Follow the tail, which is where a running simulation writes. }
  mmLog.SelStart := Length(mmLog.Text);
end;

procedure TfmMain.RunLine(Sender: TObject; const ALine: string);
begin
  Log(ALine);
end;

procedure TfmMain.RunProgress(Sender: TObject; APercent: Integer);
begin
  sbMain.Panels[0].Text := Format('Running  %d%%', [APercent]);
end;

{ A line in the log saying how many photons each detector caught.

  The detected-photon file is the point of half the runs there are, and
  until now the only sign it had been written was the file appearing.  The
  counts are also the quickest way to see that a detector is in the wrong
  place: a zero next to three hundreds is a typo in a coordinate, and that
  is worth finding before the fitting rather than after. }
procedure TfmMain.LogDetected(const AFileName: string);
var
  Ids: TMcxArray;
  Counts: array of Int64;
  i, Det: Integer;
  n: Int64;
  Line: string;
begin
  if not McxLoadArray(AFileName, 'MCXData.PhotonData.detid', Ids) then Exit;
  n := McxArrayCount(Ids);
  if n <= 0 then Exit;
  SetLength(Counts, 1);
  for i := 0 to n - 1 do
  begin
    Det := Round(McxArrayValue(Ids, i));
    if Det < 0 then Continue;
    if Det > High(Counts) then SetLength(Counts, Det + 1);
    Inc(Counts[Det]);
  end;
  Line := '';
  { Detector numbers start at one in this file; a zero would mean a photon
    that reached no detector, which should not be in it at all. }
  for Det := 1 to High(Counts) do
  begin
    if Line <> '' then Line := Line + ', ';
    Line := Line + Format('%d: %d', [Det, Counts[Det]]);
  end;
  Log(Format('%s: %d detected photons (%s)',
    [ExtractFileName(AFileName), n, Line]));
end;

procedure TfmMain.RunDone(Sender: TObject; AExitCode: Integer);
const
  TrajExts: array[0..2] of string = ('.jdt', '.jdb', '.jdat');
var
  Guess: string;
  i: Integer;
begin
  if AExitCode = 0 then
  begin
    Log('-- finished');
    { A run that worked has written its result beside the input, named after
      the session.  Showing it is the whole point of having run it. }
    Guess := ExtractFilePath(FDoc.FileName) + FDoc.AsStr('Session.ID', 'mcx');
    if FileExists(Guess + '.jnii') then ShowResult(Guess + '.jnii')
    else if FileExists(Guess + '.bnii') then ShowResult(Guess + '.bnii');
    { What each detector caught, when the run was asked for that. }
    for i := 0 to High(TrajExts) do
      if FileExists(Guess + '_detp' + TrajExts[i]) then
      begin
        LogDetected(Guess + '_detp' + TrajExts[i]);
        Break;
      end;
    { And the paths, when the run was asked for them. }
    { Three spellings, all current: mcx writes .jdt for text JData and .jdb
      for binary, and mcxcl still writes the older .jdat. }
    for i := 0 to High(TrajExts) do
      if FileExists(Guess + '_traj' + TrajExts[i]) then
      begin
        ShowResult(Guess + '_traj' + TrajExts[i]);
        Break;
      end;
  end
  else
    Log(Format('-- stopped, exit code %d', [AExitCode]));
  { The thread is not FreeOnTerminate, so that this can read ExitStatus off
    it; it is released here instead. }
  FreeAndNil(FRunner);
  UpdateRunActions;
  UpdateStatus;
end;

procedure TfmMain.acRunExecute(Sender: TObject);
var
  Exe: string;
  Args: TStringList;
begin
  if FRunner <> nil then Exit;

  Exe := CurrentExe;
  if Exe = '' then
  begin
    MessageDlg('MCX Studio',
      'Could not find ' + McxExeName(CurrentBackend) + '.'#10#10 +
      'It is looked for beside this program, in an MCXStudio folder, and on '
      + 'PATH.', mtError, [mbOK], 0);
    Exit;
  end;

  { A run needs a file on disk: the simulation is the JSON, and paths inside
    it -- a volume file, an mmc mesh -- resolve against where it sits.

    Written only if it has been edited.  Saving unconditionally meant that
    opening someone's example and pressing Run rewrote their file: same
    simulation, but reindented, arrays exploded a number to a line, 5.0e-09
    become 5E-9.  Nobody asked for that, and in a checkout it shows up as a
    modified file. }
  if FDoc.FileName = '' then
    if not SaveAs then Exit;
  if FDoc.Modified then
  begin
    if not FDoc.SaveToFile(FDoc.FileName) then
    begin
      MessageDlg('MCX Studio', 'Could not write the file:'#10#10 +
        FDoc.LastError, mtError, [mbOK], 0);
      Exit;
    end;
    UpdateTitle;
  end;

  pcView.ActivePage := tsLog;
  mmLog.Clear;
  Log(McxCommandLine(Exe, FDoc.FileName, FDoc, FRun));
  Log('');

  Args := McxBuildArgs(FDoc.FileName, FDoc, FRun);
  try
    FRunner := TMcxRunner.Create(Exe, Args, ExtractFilePath(FDoc.FileName));
  finally
    Args.Free;
  end;
  FRunner.OnLine := @RunLine;
  FRunner.OnProgress := @RunProgress;
  FRunner.OnDone := @RunDone;

  UpdateRunActions;
  sbMain.Panels[0].Text := 'Running';
  FRunner.Start;
end;

procedure TfmMain.acStopExecute(Sender: TObject);
begin
  if FRunner = nil then Exit;
  Log('-- stopping');
  FRunner.Stop;
end;

procedure TfmMain.acDevicesExecute(Sender: TObject);
var
  Exe, Listing: string;
  i: Integer;
begin
  Exe := CurrentExe;
  pcView.ActivePage := tsLog;
  if Exe = '' then
  begin
    Log('No ' + McxExeName(CurrentBackend) + ' found.');
    Exit;
  end;

  if not McxQueryDevices(Exe, Listing) then
  begin
    Log(Exe + ' -L: could not be run.');
    Exit;
  end;

  FDevices := McxParseDevices(Listing);
  Log(Exe + ' -L');
  if Length(FDevices) = 0 then
    Log('  no devices reported')
  else
    for i := 0 to High(FDevices) do
      Log(Format('  #%d  %s  (auto-thread %d, auto-block %d)',
        [FDevices[i].Id, FDevices[i].Name,
         FDevices[i].AutoThread, FDevices[i].AutoBlock]));
end;

{ ------------------------------------------------------------- docking ---- }

{ Docking is AnchorDocking's, the package the Lazarus IDE docks itself with
  and the one led uses.  Drag a pane by its header onto any edge of any other
  pane, drop it on a tab strip to join that group, drag it to open screen to
  float it.  None of that is written here; what is written here is where the
  three panes start and how they come back.

  The site is a TAnchorDockPanel filling pnMain rather than the form itself.
  MakeDockSite(Form) is the other way round and is the wrong one here: it
  docks panes around whatever the form already contains, so the settings
  would not be a pane and could never be moved.  With a panel every pane is
  equal, the middle one included.

  AnchorDocking docks forms, so each of the three controls is re-parented
  into a plain host form.  Everything that refers to sbNav, sbDetail or
  pnPreview elsewhere keeps working: re-parenting changes which control owns
  the pixels, not which component owns the reference. }

function TfmMain.MakePane(const AName, ACaption: string; AControl: TControl;
  AWidth, AHeight: Integer): TForm;
begin
  Result := TForm.CreateNew(Self);
  { The Name is how a saved layout finds its pane again, so it has to stay
    stable across versions -- see DockCreateControl. }
  Result.Name := AName;
  Result.Caption := ACaption;
  { Raw 96-dpi sizes: mcxdpi's form scaler runs over every form as it becomes
    visible, so scaling here would scale it twice. }
  Result.Width := AWidth;
  Result.Height := AHeight;
  AControl.Parent := Result;
  AControl.Align := alClient;
end;

procedure TfmMain.BuildDock;
begin
  { AnchorDocking brings its own splitters, so the designer's two go. }
  FreeAndNil(spNav);
  FreeAndNil(spPreview);

  FDockSite := TAnchorDockPanel.Create(Self);
  FDockSite.Name := 'McxDockSite';
  FDockSite.Parent := pnMain;
  FDockSite.Align := alClient;

  { The settings are a page of the notebook rather than a pane of their own.
    Two columns instead of three: the navigator says what you are editing and
    everything else -- the form, the 3-D view, the JSON, the log -- shares one
    large area, which is the only way the renderer gets room to be worth
    looking at. }
  sbDetail.Parent := tsSettings;
  sbDetail.Align := alClient;
  pcView.ActivePage := tsSettings;

  FPaneNav := MakePane('PaneSections', 'Sections', sbNav, 340, 640);
  FPaneView := MakePane('PaneMain', 'Views', pnPreview, 900, 640);

  DockMaster.OnCreateControl := @DockCreateControl;
  DockMaster.MakeDockPanel(FDockSite, admrpChild);

  { The middle goes in first, because the navigator docks against it. }
  DockMaster.MakeDockable(FPaneView, True, True, False);
  DockMaster.ManualDock(DockMaster.GetAnchorSite(FPaneView), FDockSite,
    alClient);

  DockMaster.MakeDockable(FPaneNav, False, True, True);
  DockMaster.ManualDock(DockMaster.GetAnchorSite(FPaneNav), FDockSite, alLeft);
  DockMaster.ShowControl(FPaneNav.Name, False);

  GuardCentreHeader;
end;

{ The default sizes cannot be applied in FormCreate.

  Nothing is laid out yet at that point -- the dock site measures 170 by 50 --
  so a splitter has nothing to move and every pane comes out whatever
  DockAnotherControl guessed.  This runs once the window has been shown and
  the real geometry exists, and only when no saved layout was restored, since
  a restored one already carries the sizes the user chose. }
procedure TfmMain.ApplyPaneSizes(Data: PtrInt);
begin
  if FPendingResult <> '' then
  begin
    ShowResult(FPendingResult);
    FPendingResult := '';
  end;
  if FDockRestored then Exit;
  SizePane(FPaneNav, alLeft, McxScale96(340));
  GuardCentreHeader;
end;

procedure TfmMain.ShowResultLater(const AFileName: string);
begin
  FPendingResult := AFileName;
end;

procedure TfmMain.FormShow(Sender: TObject);
begin
  if FDockSized then Exit;
  FDockSized := True;
  { After this message, not during it: the form is shown but its children are
    still settling while OnShow runs. }
  Application.QueueAsyncCall(@ApplyPaneSizes, 0);
end;

{ Gives a docked pane the size it should have.

  AnchorDocking sizes a newly docked pane as Min(its own width, half its
  neighbour's) -- DockAnotherControl -- which on this window means half of
  everything, so Sections opened 900 pixels wide.  The size is not a property
  it exposes: a docked site is anchored to a splitter and recomputes its own
  bounds, so the way to resize it is to move that splitter.

  led hit the same thing and needed 140 lines, because it has several panes on
  an edge and they have to be fixed outside-in or each takes the space
  straight back off the last.  With one pane an edge, this is the same idea
  without the ordering. }
procedure TfmMain.SizePane(APane: TForm; AAlign: TAlign; AWanted: Integer);
var
  Site: TAnchorDockHostSite;
  Split: TAnchorDockSplitter;
  Side: TAnchorKind;
  Have, Delta, Pass: Integer;
begin
  if APane = nil then Exit;
  Site := DockMaster.GetAnchorSite(APane);
  if (Site = nil) or (Site.Parent = nil) then Exit;

  { Which side the splitter is on: a left pane is anchored to the splitter on
    its right, a right pane to the one on its left. }
  case AAlign of
    alLeft:  Side := akRight;
    alRight: Side := akLeft;
    alTop:   Side := akBottom;
  else
    Side := akTop;
  end;
  if not (Site.AnchorSide[Side].Control is TAnchorDockSplitter) then Exit;
  Split := TAnchorDockSplitter(Site.AnchorSide[Side].Control);

  { Twice: a splitter only moves as far as the control on its far side will
    give up at that moment, so the first pass is often short. }
  for Pass := 1 to 2 do
  begin
    if AAlign in [alLeft, alRight] then Have := Site.Width else Have := Site.Height;
    Delta := AWanted - Have;
    if Delta = 0 then Break;
    { A left or top pane grows when its splitter moves away from the edge; a
      right or bottom pane grows when it moves towards it. }
    if AAlign in [alLeft, alTop] then Split.MoveSplitter(Delta)
    else Split.MoveSplitter(-Delta);
  end;
end;

{ The settings pane keeps no header.  MakeDockable is asked for none, but
  AnchorDocking puts one back every time it rebuilds the site, so this is
  re-asserted rather than done once.

  A header there would be a drag handle for the one pane with nowhere to go,
  and a close button for the one pane that must not close. }
procedure TfmMain.GuardCentreHeader;
var
  Site: TAnchorDockHostSite;
begin
  if FPaneView = nil then Exit;
  Site := DockMaster.GetAnchorSite(FPaneView);
  if (Site = nil) or (Site.Header = nil) then Exit;
  Site.Header.Visible := False;
  if Site.Header.CloseButton <> nil then
    Site.Header.CloseButton.Visible := False;
end;

{ Restoring a layout asks for each pane by the Name it was saved under.  All
  three already exist, so this is a lookup rather than a factory -- and a name
  it does not recognise has to come back nil, or AnchorDocking keeps a hole in
  the layout that it cannot fill. }
procedure TfmMain.DockCreateControl(Sender: TObject; aName: string;
  var AControl: TControl; DoDisableAutoSizing: boolean);
begin
  AControl := nil;
  if SameText(aName, 'PaneSections') then AControl := FPaneNav
  else if SameText(aName, 'PaneMain') then AControl := FPaneView;
  if (AControl <> nil) and DoDisableAutoSizing then
    TWinControl(AControl).DisableAutoSizing;
end;

{ Where the layout is kept.  Beside the preferences rather than beside the
  simulation: which pane sits where is a property of this installation, not
  of the file being edited. }
function McxLayoutFile: string;
begin
  Result := IncludeTrailingPathDelimiter(GetAppConfigDir(False)) + 'layout.xml';
end;

procedure TfmMain.SaveDockLayout;
var
  Cfg: TXMLConfigStorage;
begin
  if FDockSite = nil then Exit;
  try
    ForceDirectories(ExtractFilePath(McxLayoutFile));
    Cfg := TXMLConfigStorage.Create(McxLayoutFile, False);
    try
      DockMaster.SaveLayoutToConfig(Cfg);
      DockMaster.SaveSettingsToConfig(Cfg);
      Cfg.WriteToDisk;
    finally
      Cfg.Free;
    end;
  except
    { Where the panes sit is not worth refusing to close over. }
  end;
end;

function TfmMain.LoadDockLayout: Boolean;
var
  Cfg: TXMLConfigStorage;
begin
  Result := False;
  if not FileExists(McxLayoutFile) then Exit;
  try
    Cfg := TXMLConfigStorage.Create(McxLayoutFile, True);
    try
      DockMaster.LoadSettingsFromConfig(Cfg);
      Result := DockMaster.LoadLayoutFromConfig(Cfg, True);
    finally
      Cfg.Free;
    end;
  except
    { A layout written by an older build is not worth refusing to start
      over; the panes simply stay where BuildDock put them. }
    Result := False;
  end;
  FDockRestored := Result;
  GuardCentreHeader;
end;

{ AnchorDocking will happily leave a pane somewhere with no route back -- off
  the edge of the screen, or closed -- so there has to be a way home. }
{ Puts a result into the 3-D view.  Reported rather than silent when it will
  not read: a file that turns out not to hold an array is the usual way this
  goes wrong, and an unchanged picture says nothing about why. }
procedure TfmMain.ShowResult(const AFileName: string);
var
  A: TMcxArray;
  i: Integer;
  Shape: string;
begin
  { A trajectory file is a different thing from a fluence map -- paths rather
    than a volume -- and it is told apart by its name, which is how mcx
    names it. }
  if (Pos('_traj.', LowerCase(ExtractFileName(AFileName))) > 0) then
  begin
    if FView.ShowTrajectory(AFileName) then
    begin
      { The range row is scaled to the file that was just read, so the bar
        covers the photons that are actually in it. }
      FDisplay.SetTrajectory(FView.PhotonFirst, FView.PhotonLast);
      FDisplay.SetHasVolume(True);
      pcView.ActivePage := tsPreview;
    end
    else
      Log('no photon paths in ' + ExtractFileName(AFileName));
    Exit;
  end;

  if not McxLoadArray(AFileName, '', A) then
  begin
    Log('could not read an array out of ' + ExtractFileName(AFileName));
    Exit;
  end;

  Shape := '';
  for i := 0 to High(A.Dims) do
  begin
    if i > 0 then Shape := Shape + ' x ';
    Shape := Shape + IntToStr(A.Dims[i]);
  end;
  Log(Format('%s: %s %s, %d values', [ExtractFileName(AFileName),
    Shape, McxArrayKindName(A.Kind), McxArrayCount(A)]));

  if FView.ShowVolume(A) then
  begin
    FDisplay.ResetSlab;
    FDisplay.SetHasVolume(True);
    pcView.ActivePage := tsPreview;
  end
  else
    Log('  the 3-D view would not take it');
end;

{ mcx carries a set of standard simulations and will print any of them as
  JSON, so a new user has somewhere to start that is not a blank cube.

  Three steps and no parsing of our own: ask for the list, ask for the one
  picked, load it as a document.  The benchmark is printed by --dumpjson 2,
  which stops before any GPU work, so this costs nothing and needs no card. }
{ Twice the size of the pane, because a picture that goes into a paper or a
  bug report wants more pixels than the window happened to have. }
procedure TfmMain.acSaveImageExecute(Sender: TObject);
begin
  if FDoc.FileName <> '' then
    dlgImage.FileName := ChangeFileExt(FDoc.FileName, '.png')
  else
    dlgImage.FileName := FDoc.AsStr('Session.ID', 'mcx') + '.png';
  if not dlgImage.Execute then Exit;
  if FView.SaveImage(dlgImage.FileName, pnGL.Width * 2, pnGL.Height * 2) then
    Log('wrote ' + dlgImage.FileName)
  else
    Log('could not write ' + dlgImage.FileName);
end;

{ What a click in the 3-D view landed on.  The status bar rather than a
  dialog: it answers a question nobody asked out loud. }
procedure TfmMain.ViewPick(Sender: TObject; const AInfo: TMcxPickInfo);
begin
  sbMain.Panels[0].Text := Format('Shapes[%d] %s, tag %d',
    [AInfo.Index, AInfo.Verb, AInfo.Tag]);
end;

procedure TfmMain.acBenchmarkExecute(Sender: TObject);
var
  Names: TStringList;
  Item: TMenuItem;
  P: TPoint;
  i: Integer;
begin
  if CurrentExe = '' then
  begin
    Log('No ' + McxExeName(CurrentBackend) + ' found, so no benchmarks.');
    pcView.ActivePage := tsLog;
    Exit;
  end;

  Names := TStringList.Create;
  try
    if not McxBenchmarks(CurrentExe, Names) then
    begin
      Log(CurrentExe + ' -Q: no benchmarks reported.');
      pcView.ActivePage := tsLog;
      Exit;
    end;
    pmBench.Items.Clear;
    for i := 0 to Names.Count - 1 do
    begin
      Item := TMenuItem.Create(pmBench);
      Item.Caption := Names[i];
      Item.OnClick := @BenchmarkClick;
      pmBench.Items.Add(Item);
    end;
  finally
    Names.Free;
  end;

  P := tbMain.ClientToScreen(Point(tbBench.Left,
    tbBench.Top + tbBench.Height));
  pmBench.PopUp(P.X, P.Y);
end;

procedure TfmMain.BenchmarkClick(Sender: TObject);
var
  JSON, Bench: string;
begin
  if not (Sender is TMenuItem) then Exit;
  if not ConfirmDiscard then Exit;
  Bench := TMenuItem(Sender).Caption;

  if not McxBenchmarkJSON(CurrentExe, Bench, JSON) then
  begin
    Log('could not read the benchmark ' + Bench);
    pcView.ActivePage := tsLog;
    Exit;
  end;
  if not FDoc.LoadFromString(JSON) then
  begin
    Log(Bench + ': ' + FDoc.LastError);
    pcView.ActivePage := tsLog;
    Exit;
  end;

  { A benchmark arrives with no file of its own, so it behaves like a new
    document: Save asks where to put it. }
  FDoc.FileName := '';
  FDoc.Modified := False;
  LoadAllBindings;
  UpdateTitle;
  RefreshPreview;
  pcView.ActivePage := tsSettings;
  Log('loaded the ' + Bench + ' benchmark');
end;

procedure TfmMain.acLoadResultExecute(Sender: TObject);
begin
  if FDoc.FileName <> '' then
    dlgResult.InitialDir := ExtractFilePath(FDoc.FileName);
  if not dlgResult.Execute then Exit;
  ShowResult(dlgResult.FileName);
end;

procedure TfmMain.acResetLayoutExecute(Sender: TObject);
begin
  if MessageDlg('MCX Studio',
       'Put the panes back where they started?',
       mtConfirmation, [mbYes, mbNo], 0) <> mrYes then Exit;

  DeleteFile(McxLayoutFile);
  FDockRestored := False;
  DockMaster.ManualDock(DockMaster.GetAnchorSite(FPaneView), FDockSite,
    alClient);
  DockMaster.ManualDock(DockMaster.GetAnchorSite(FPaneNav), FDockSite, alLeft);
  DockMaster.ShowControl(FPaneNav.Name, False);
  ApplyPaneSizes(0);
end;

{ ------------------------------------------------------------- stacking --- }

{ Records the order a set of alTop siblings is meant to appear in, while every
  one of them is still showing.

  A hidden control keeps whatever Top it had when it went away, and the
  aligner orders alTop siblings by Top -- so when the wizard filter puts one
  back, its stale Top decides where it lands.  That is how Advanced arrived
  third in the navigator: it starts hidden, so it still carried the 168 the
  designer gave it, while the sections around it had been re-laid out at a
  display scale that put Domain at 131 and Shapes at 202.

  Capturing has to happen while nothing is hidden yet, which is why this runs
  from FormCreate before the first ApplyBindingStates. }
procedure TfmMain.CaptureStacks;

  procedure Capture(P: TWinControl);
  var
    i, j, n: Integer;
    C: TControl;
    S: TMcxStack;
  begin
    if P = nil then Exit;
    S.Parent := P;
    SetLength(S.Items, 0);
    for i := 0 to P.ControlCount - 1 do
      if P.Controls[i].Align = alTop then
      begin
        SetLength(S.Items, Length(S.Items) + 1);
        S.Items[High(S.Items)] := P.Controls[i];
      end;
    if Length(S.Items) < 2 then Exit;

    { Insertion sort on Top: the list is short, and the order the children
      happen to be in is the designer's, not the layout's. }
    for i := 1 to High(S.Items) do
    begin
      C := S.Items[i];
      j := i - 1;
      while (j >= 0) and (S.Items[j].Top > C.Top) do
      begin
        S.Items[j + 1] := S.Items[j];
        Dec(j);
      end;
      S.Items[j + 1] := C;
    end;

    n := Length(FStacks);
    SetLength(FStacks, n + 1);
    FStacks[n] := S;
  end;

var
  i: Integer;
begin
  SetLength(FStacks, 0);
  Capture(sbNav);
  for i := 0 to High(FPanes) do Capture(FBodies[i]);
  for i := 0 to High(FPages) do Capture(FPages[i]);
  for i := 0 to High(FSubs) do Capture(FSubs[i].Box);
end;

{ Writes the captured order back as Tops, so the next align packs them in it
  whatever was hidden in between.  The values only have to be increasing --
  the aligner overwrites every visible one with its packed position. }
procedure TfmMain.Restack;
var
  i, j: Integer;
begin
  for i := 0 to High(FStacks) do
  begin
    FStacks[i].Parent.DisableAlign;
    try
      for j := 0 to High(FStacks[i].Items) do
        FStacks[i].Items[j].Top := j * 1000;
    finally
      FStacks[i].Parent.EnableAlign;
    end;
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

{ 'x=Fluence rate,f=Fluence' -> the values in AValues and what the person
  reads in AItems.  An entry with no equals sign is its own label, so a plain
  list still works.

  Splitting here rather than storing two tables means a value and its label
  cannot drift apart: they are one string in one place. }
procedure SetChoices(AItems, AValues: TStrings; const AList: string);
var
  Parts: TStringList;
  i, e: Integer;
begin
  AItems.Clear;
  AValues.Clear;
  Parts := TStringList.Create;
  try
    Parts.Delimiter := ',';
    Parts.StrictDelimiter := True;
    Parts.DelimitedText := AList;
    for i := 0 to Parts.Count - 1 do
    begin
      e := Pos('=', Parts[i]);
      if e > 1 then
      begin
        AValues.Add(Copy(Parts[i], 1, e - 1));
        AItems.Add(Copy(Parts[i], e + 1, MaxInt));
      end
      else
      begin
        AValues.Add(Parts[i]);
        AItems.Add(Parts[i]);
      end;
    end;
  finally
    Parts.Free;
  end;
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
    { A plain comma split: a flag entry is letter:caption, which SetChoices'
      value=label form does not apply to. }
    Parts.Delimiter := ',';
    Parts.StrictDelimiter := True;
    Parts.DelimitedText := AList;
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
  Letters, Help: string;

  function IndexOfRow(APanel: TPanel): Integer;
  begin
    for Result := 0 to High(FRowList) do
      if FRowList[Result] = APanel then Exit;
    Result := -1;
  end;

  { The control and whatever it is built out of.  A TRadioGroup's radio
    buttons are real windows in front of it, so the hint has to be on them
    too, and a child that already has one of its own keeps it. }
  procedure SetHint(AControl: TControl; const AText: string);
  var
    k: Integer;
  begin
    if AControl = nil then Exit;
    AControl.Hint := AText;
    AControl.ShowHint := True;
    if AControl is TWinControl then
      for k := 0 to TWinControl(AControl).ControlCount - 1 do
        if TWinControl(AControl).Controls[k].Hint = '' then
          SetHint(TWinControl(AControl).Controls[k], AText);
  end;

begin
  SetLength(FBound, Length(Binds));
  SetLength(FLabels, Length(Binds));
  SetLength(FLoaded, Length(Binds));
  SetLength(FRows, Length(Binds));
  SetLength(FGroups, Length(Binds));
  SetLength(FValues, Length(Binds));
  SetLength(FFlags, Length(Binds));
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
      begin
        FValues[i] := TStringList.Create;
        SetChoices(TRadioGroup(C).Items, FValues[i], Binds[i].Choices);
      end;
      TRadioGroup(C).OnClick := @BindChanged;
    end
    else if C is TCheckGroup then
    begin
      if Binds[i].Choices <> '' then
      begin
        SplitFlags(Binds[i].Choices, Letters, TCheckGroup(C).Items);
        FFlags[i] := Letters;
      end;
      TCheckGroup(C).OnItemClick := @CheckGroupClick;
    end
    else if C is TComboBox then
    begin
      if Binds[i].Choices <> '' then
      begin
        FValues[i] := TStringList.Create;
        SetChoices(TComboBox(C).Items, FValues[i], Binds[i].Choices);
      end;
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

    { The same sentence hovers and answers F1.  Last, because a radio group's
      buttons do not exist until its Items have been set, and they are what
      the pointer is actually over: a hint on the group alone shows only in
      the gaps between them.

      On the caption and the row as well, so that a setting whose control is
      a 16-pixel checkbox still has somewhere to hover. }
    Help := McxHelpFor(Binds[i].Ctl);
    if Help <> '' then
    begin
      Help := McxWrap(Help);
      SetHint(FBound[i], Help);
      SetHint(FLabels[i], Help);
      SetHint(FRows[i], Help);
    end;

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
        { Matched on the value in the file, not on what is displayed: the
          file says "f" and the form says "Fluence". }
        N := -1;
        if FValues[AIndex] <> nil then N := FValues[AIndex].IndexOf(S);
        if C is TRadioGroup then
        begin
          { A value this build has never heard of is added rather than
            silently dropped, so opening a file from a newer mcx does not
            quietly rewrite it. }
          if (N < 0) and (S <> '') then
          begin
            N := TRadioGroup(C).Items.Add(S);
            if FValues[AIndex] <> nil then FValues[AIndex].Add(S);
          end;
          TRadioGroup(C).ItemIndex := N;
        end
        else if C is TComboBox then
        begin
          if (N < 0) and (S <> '') then
          begin
            N := TComboBox(C).Items.Add(S);
            if FValues[AIndex] <> nil then FValues[AIndex].Add(S);
          end;
          TComboBox(C).ItemIndex := N;
        end;
      end;
    mkFlags:
      if C is TCheckGroup then
      begin
        S := FlagsOf(D, B.Path, FFlags[AIndex]);
        for N := 0 to TCheckGroup(C).Items.Count - 1 do
          TCheckGroup(C).Checked[N] :=
            (N < Length(FFlags[AIndex])) and
            (Pos(FFlags[AIndex][N + 1], S) > 0);
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
        N := -1;
        if C is TRadioGroup then N := TRadioGroup(C).ItemIndex
        else if C is TComboBox then N := TComboBox(C).ItemIndex;
        { The value, never the label.  mcx reads "f"; "Fluence" would be a
          setting it has never heard of. }
        if (N >= 0) and (FValues[AIndex] <> nil) and
           (N < FValues[AIndex].Count) then
          D.SetStr(B.Path, FValues[AIndex][N]);
      end;
    mkFlags:
      if C is TCheckGroup then
      begin
        S := '';
        for N := 0 to TCheckGroup(C).Items.Count - 1 do
          if TCheckGroup(C).Checked[N] and (N < Length(FFlags[AIndex])) then
            S := S + FFlags[AIndex][N + 1];
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
  { The tables read the document directly, so they are refreshed here rather
    than bound: there is no control-to-path row that could describe them. }
  if FMedia <> nil then FMedia.Attach(FDoc, 'Domain.Media');
  if FDetectors <> nil then FDetectors.Attach(FDoc, 'Optode.Detector');
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

  { '~' asks whether the value contains the text rather than equals it.  The
    flag settings are sets written as letters -- DebugFlag is 'MP' -- so "is
    M on" is not a comparison. }
  E := Pos('~', ACond);
  if E > 1 then
  begin
    Path := Copy(ACond, 1, E - 1);
    Want := Copy(ACond, E + 1, MaxInt);
    Exit(Pos(UpperCase(Want), UpperCase(D.AsStr(Path))) > 0);
  end;

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
  InMode, Shown, Moved: Boolean;
  Cap1, Cap2, Flags, Letters: string;
  Live: array of Boolean;

  { Restacking is only needed when something appeared or disappeared, and
    this runs on every keystroke that commits, so it is worth knowing. }
  procedure Show(AControl: TControl; AVisible: Boolean);
  begin
    if AControl = nil then Exit;
    if AControl.Visible <> AVisible then Moved := True;
    AControl.Visible := AVisible;
  end;

begin
  Moved := False;
  for i := 0 to High(Binds) do
  begin
    C := FBound[i];
    if C = nil then Continue;
    B := Binds[i];

    { The wizard is a strict subset of the expert form, so one filter serves
      both modes and there is no second layout to keep in step. }
    InMode := (not FWizard) or (B.Level = mlWizard);
    Show(C, InMode);
    if InMode then
      C.Enabled := ConditionHolds(DocFor(B.EnableIf), B.EnableIf);
    if FLabels[i] <> nil then
    begin
      Show(FLabels[i], InMode);
      FLabels[i].Enabled := C.Enabled;
    end;
  end;

  { The debug flags are the one setting whose meaning changes with the
    simulator: mmc shares only M and P with mcx.  Re-split rather than given
    a control of its own, so the letters that are ticked survive the switch
    -- they are stored as letters, and the two that matter mean the same in
    both. }
  Flags := FlagsDebug;
  if CurrentBackend = mbMMC then Flags := FlagsDebugMMC;
  if Flags <> FDebugList then
  begin
    FDebugList := Flags;
    for i := 0 to High(Binds) do
      if (FBound[i] is TCheckGroup) and (Binds[i].Path = 'Session.DebugFlag') then
      begin
        SplitFlags(Flags, Letters, TCheckGroup(FBound[i]).Items);
        FFlags[i] := Letters;
        LoadBinding(i);
      end;
  end;

  { Param1 and Param2 hold up to four numbers each, and which numbers they
    are is decided entirely by the source type -- so the caption is too, and
    a type that takes none loses the row rather than offering an empty box
    nobody can fill.  This is the one place the form is not a fixed picture
    of the binding table, and it is worth it: "Parameter 1" is a box you
    need the manual open to use. }
  McxSrcParams(FDoc.AsStr('Optode.Source.Type', 'pencil'), Cap1, Cap2);
  if Cap1 = '' then
  begin
    Show(edSrcParam1, False);
    Show(lbSrcParam1, False);
  end
  else
    lbSrcParam1.Caption := Cap1;
  if Cap2 = '' then
  begin
    Show(edSrcParam2, False);
    Show(lbSrcParam2, False);
  end
  else
    lbSrcParam2.Caption := Cap2;

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
    if NBound > 0 then Show(FRowList[j], NShown > 0);
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
    Show(FSubs[j].Box, Live[j]);
    Show(FSubs[j].Btn, Live[j]);
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
    Show(FPanes[s], Shown);
    if Shown and (First < 0) then First := s;
  end;
  if Moved then Restack;
  UpdateRunActions;
  UpdateStepBar;

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
    Input := FDoc.FileName
  else
    Input := FDoc.AsStr('Session.ID', 'mcx') + '.json';
  mmCommand.Text := McxCommandLine(CurrentExe, Input, FDoc, FRun);
  if FView <> nil then FView.Rebuild;

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
  { A mesh has no Dim, and "0 x 0 x 0" reads as a broken file rather than as
    a different kind of domain. }
  if FRun.AsStr('@run.domainkind', 'shapes') = 'mesh' then
    sbMain.Panels[2].Text := 'mesh ' + FDoc.AsStr('Mesh.MeshID', '(embedded)')
  else
    sbMain.Panels[2].Text := Format('%d x %d x %d',
      [FDoc.AsInt('Domain.Dim[0]'), FDoc.AsInt('Domain.Dim[1]'),
       FDoc.AsInt('Domain.Dim[2]')]);
end;

end.
