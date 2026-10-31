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
  SynEdit, SynEditTypes, SynGutterLineNumber, mcxjsonsyn,
  mcxdpi, mcxicons, mcxtheme, mcxdoc, mcxhelp, mcxabout, mcxrun, mcxgl, mcxview, mcxshapes,
  mcxdisp, mcxchoose, mcxtable, mcxjd, mcxmcxp, mcxdomain, mcxdatadlg, CheckLst,
  FileUtil, LazFileUtils, ToolWin;

type
  { The dock's splitters in the theme's colours.  The stock one paints a
    rebar band, which the visual styles on Windows draw light grey whatever
    the window around it is, so a dark theme had a pale bar between every
    pair of panes. }
  TMcxDockSplitter = class(TAnchorDockSplitter)
  protected
    procedure Paint; override;
  end;

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
    gbDomainData: TPanel;
    gbMesh: TPanel;
    rwMeshID: TPanel;
    lbMeshID: TLabel;
    edMeshID: TEdit;
    rwInitElem: TPanel;
    lbInitElem: TLabel;
    edInitElem: TSpinEdit;
    rwRayTracer: TPanel;
    lbRayTracer: TLabel;
    cbRayTracer: TComboBox;
    rwBasisOrder: TPanel;
    lbBasisOrder: TLabel;
    cbBasisOrder: TComboBox;
    lbMeshInfo: TLabel;
    gbMedia: TPanel;
    gbShapeList: TPanel;
    pnShapeEdit: TPanel;
    gbSource: TPanel;
    gbSrcAdv: TPanel;
    gbDetector: TPanel;
    gbBasic: TPanel;
    gbTime: TPanel;
    gbOutput: TPanel;
    gbSwitches: TPanel;
    gbDetPhoton: TPanel;
    gbSolver: TPanel;
    rwSimulator: TPanel;
    lbSimulator: TLabel;
    cbSimulator: TComboBox;
    rwExtraArgs: TPanel;
    lbExtraArgs: TLabel;
    edExtraArgs: TEdit;
    gbGPU: TPanel;
    gbDevices: TPanel;
    gbBoundary: TPanel;
    gbFlags: TPanel;

    { The editors that are not written yet say so, rather than leaving a
      group box that looks broken. }
    lbTodoMedia: TLabel;
    lbTodoDet: TLabel;
    rwDetParam1: TPanel;
    lbDetParam1: TLabel;
    edDetParam1: TEdit;
    rwDetParam2: TPanel;
    lbDetParam2: TLabel;
    edDetParam2: TEdit;
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
    procedure BuildDeviceCard;
    procedure BuildDomainCard;
    procedure UpdateDomainCard;
    procedure DomEditClick(Sender: TObject);
    procedure DomLoadClick(Sender: TObject);
    function  DomainIsMesh: Boolean;
    function  VolumeText: string;
    function  MeshText: string;
    procedure StoreVolume(AObj: TJSONObject);
    procedure StoreMesh(ANode, AElem, AMedia: TJSONData; AReplaceMedia: Boolean);
    procedure DomainChanged;
    function  RawWidth: Integer;
    function  CheckVolumeText(const AText: string; out AError: string): Boolean;
    function  CheckMeshText(const AText: string; out AError: string): Boolean;
    function  LoadVolumeText(const AFileName: string; out AText, AError: string): Boolean;
    function  LoadMeshText(const AFileName: string; out AText, AError: string): Boolean;
    procedure RefreshDevices;
    procedure DeviceTicked(Sender: TObject);
    procedure DeviceRefreshClick(Sender: TObject);
    procedure acAboutExecute(Sender: TObject);
    procedure acHelpExecute(Sender: TObject);
    procedure acThemeExecute(Sender: TObject);
    procedure FormKeyDown(Sender: TObject; var Key: Word; Shift: TShiftState);
    procedure acQuitExecute(Sender: TObject);
    procedure HeaderClick(Sender: TObject);
    procedure ShowSettingsPage;
    procedure BuildSourceBar;
    function  SourceRows: Integer;
    function  VecPath(const APath: string): string;
    function  MeshPath(const APath: string): string;
    procedure SourceChanged(Sender: TObject);
    procedure SourceTableChanged(Sender: TObject);
    procedure UpdateSourceBar;
    procedure UpdateMeshInfo;
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
    { Whether the selected section's subsection list is showing. }
    FOpen: Boolean;
    FTabBar: TPanel;
    FTabs: array of TSpeedButton;
    { The section heading the pointer is over, or -1. }
    FHeadHot: Integer;
    { The subsection the pointer is over, or -1.  The band behind it is the
      hover, since the button no longer paints one. }
    FSubHot: Integer;
    { Which source the four source vectors are showing, when the file has
      more than one.  See SourceRows. }
    FSource: Integer;
    { Every source at once, a row each; picking a row is what sets FSource. }
    FSources: TMcxVecTable;
    FSrcHead: TLabel;
    { Set while the table's own edit is being passed on, so the reload that
      follows leaves the grid alone: it is called on every keystroke, and
      rewriting the cell under the editor turns a half-typed 1.0 into 1. }
    FSrcEditing: Boolean;
    FPanes: array of TPanel;
    { One per section, between the section panel and its heading: the band
      behind the heading is painted on this.  See CollectSections. }
    FBands: array of TPanel;
    FBandCut: array of TPaintBox;
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
    FStepBack: TSpeedButton;
    FStepMode: TSpeedButton;
    FStepReset: TSpeedButton;
    FStepNext: TSpeedButton;
    FStepText: TLabel;
    { Every set of alTop siblings the wizard filter can hide something from,
      each in the order it is meant to appear.  See Restack. }
    FStacks: array of TMcxStack;
    { The dock site and the three panes in it.  See BuildDock. }
    FDockSite: TAnchorDockPanel;
    FPaneNav: TForm;
    FPaneView: TForm;
    FPaneLog: TForm;
    { Docked before there was a window to size it in; see ApplyPaneSizes. }
    FLogUnsized: Boolean;
    { The simulation in flight, or nil.  One at a time: a second run would
      write over the first one's output files. }
    FRunner: TMcxRunner;
    FDevices: TMcxDevices;
    { The Devices card: what -L reported, ticked for the ones a run uses.
      FDevFor names the simulator the list came from, so a list is asked for
      again when a different one is chosen rather than shown for the wrong
      program. }
    { The Domain volume and mesh card: what the domain is made of, and the
      one button that edits it -- a volume or a mesh in the data editor, or
      shapes in the Shapes editor. }
    FDomInfo: TLabel;
    FDomEdit: TButton;
    FDomLoad: TButton;
    FDevList: TCheckListBox;
    FDevNote: TLabel;
    FDevFor: string;
    { The 3-D preview, which owns its own GL control, camera and geometry. }
    FView: TMcxView;
    { The two settings that are lists rather than values.  How many media a
      simulation has, and how many detectors, is not known until a file is
      opened, so these are the only editors built in code. }
    FMedia: TMcxTable;
    FDetectors: TMcxTable;
    FShapes: TMcxShapes;
    { The JSON pane: a code editor rather than a memo, so the file can be
      read as a structure instead of as a wall of quotation marks. }
    FJson: TSynEdit;
    FJsonSyn: TMcxJsonSyn;
    FJsonBar: TPanel;
    FJsonApply, FJsonRevert: TSpeedButton;
    FJsonNote: TLabel;
    { Raised while the pane is being written from the document, so that doing
      so does not look like the person typing in it. }
    FJsonLoading: Integer;
    { Set when it does.  While it is set the document stops writing to the
      pane: the text on screen is the person's, not ours. }
    FJsonEdited: Boolean;
    { Set while the note is reporting a parse failure, so that the ordinary
      messages do not write over it. }
    FJsonFault: Boolean;
    FDisplay: TMcxDisplayBar;
    FStatus: TPaintBox;
    { The three picture choosers on the first card.  Which simulator runs is
      worked out from them rather than asked as a fourth question. }
    FProcessor, FDomainKind, FMedium: TMcxChooser;
    FPlanText: TLabel;
    { The last focused control the help table knows about.  Kept because the
      Help button is on the toolbar: pressing it takes the focus off the
      setting whose help was wanted. }
    FLastHelp: TControl;
    FPendingResult: string;
    { --run: start the simulation once the window is up; --quit: and exit when
      it ends, with its exit code, for a script that has nobody to watch it. }
    FRunPending: Boolean;
    FQuitAfterRun: Boolean;
    { --snapshot: the 3-D view written to this PNG once the window is up, and
      the program closed -- a thumbnail of a simulation, or a check that the
      view draws, with nobody at the screen. }
    FSnapshot: string;
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
    procedure StartRequestedRun;
    procedure QuitAfterRun(AExitCode: Integer; const AWhy: string);
    procedure CentreOnLaunchScreen;
    procedure GuardCentreHeader;
    procedure DockLog;
    function  LogShown: Boolean;
    procedure ShowLog;
    procedure DockCreateControl(Sender: TObject; aName: string;
      var AControl: TControl; DoDisableAutoSizing: boolean);
    procedure ViewLog(Sender: TObject; const AText: string);
    procedure ShowResult(const AFileName: string);
    procedure LogDetected(const AFileName: string);
    procedure BenchmarkClick(Sender: TObject);
    procedure ViewPick(Sender: TObject; const AInfo: TMcxPickInfo);
    procedure ShapeSelected(Sender: TObject; AIndex: Integer);
    procedure ShapeLabelAdded(Sender: TObject; ATag: Integer);
    procedure TableChanged(Sender: TObject);
    function  Environment: string;
    procedure ShowHelpForFocus;
    procedure ResetSettings(Sender: TObject);
    procedure ApplyTheme;
    procedure ApplyNativeChrome;
    procedure StatusPaint(Sender: TObject);
    { A status panel's text, in whichever bar is showing. }
    procedure SetStatus(AIndex: Integer; const AText: string);
    procedure BuildThemeMenu;
    procedure ThemeClick(Sender: TObject);
    procedure FocusChanged(Sender: TObject; LastControl: TControl);
    procedure BuildJsonPane;
    procedure JsonChanged(Sender: TObject);
    procedure JsonApply(Sender: TObject);
    procedure JsonRevert(Sender: TObject);
    procedure ShowJson;
    procedure UpdateJsonBar;
    procedure BuildChoosers;
    procedure ChoiceChanged(Sender: TObject);
    procedure ApplyPlan;
    procedure LoadChoosers;
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
    procedure BuildTabBar;
    procedure TabClick(Sender: TObject);
    procedure TabChanged(Sender: TObject);
    procedure UpdateTabBar;
    procedure UpdateHeadBands;
    procedure HeadEnter(Sender: TObject);
    procedure HeadLeave(Sender: TObject);
    procedure SubEnter(Sender: TObject);
    procedure SubLeave(Sender: TObject);
    procedure CardPaint(Sender: TObject);
    procedure SetCard(ACard: TPanel; AColor: TColor);
    procedure ScrollToGroup(ABox: TPanel);
    function  FirstSubOf(ASection: Integer): Integer;
    procedure SelectSub(AIndex: Integer);
    procedure UpdateNavState;

    procedure NavSubPaint(Sender: TObject);
    function  SwapHeading(AOld: TSpeedButton; AParent: TWinControl): TSpeedButton;
    procedure NavBandCut(Sender: TObject);
    procedure NewDocument;
    function  SaveAs: Boolean;
    function  OpenProject(const AFileName: string;
      out ACancelled: Boolean): Boolean;
    function  NeedsWorkFolder: Boolean;
    procedure RebaseFiles(const AOld, ANew: string);
    function  SaveDocTo(const AFileName: string): Boolean;
    function  ConfirmDiscard: Boolean;
    procedure UpdateTitle;
    procedure UpdateStatus;
  public
    { The domain card's actions, public for the self-test. }
    function  ApplyVolumeText(const AText: string; out AError: string): Boolean;
    function  ApplyMeshText(const AText: string; out AError: string): Boolean;
    function  LoadVolumeFile(const AFileName: string; out AError: string): Boolean;
    function  LoadMeshFile(const AFileName: string; out AError: string): Boolean;
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
    { --run: runs the open simulation as soon as the window has been shown,
      as if Run had been pressed.  AQuit (--quit) exits when it ends, with the
      simulator's exit code, and sends what it prints to stdout; a problem
      that would have been a dialog becomes a line on stderr instead.  Called
      before OpenDocument, so that a file that does not load is reported that
      way too. }
    procedure RunLater(AQuit: Boolean);
    { --processor: picks the processor tile (nvidia, amd, intel or cpu), which
      decides with the domain which solver runs -- mcx, mcxcl, mcx-hip, or mmc
      and its -c.  After OpenDocument, which resets the tiles from the file. }
    function  SetProcessor(const AValue: string): Boolean;
    procedure SnapshotLater(const AFileName: string);
    { --run --quit or --snapshot: nobody is at the screen to answer a dialog,
      so what would have asked goes to stderr instead. }
    function Unattended: Boolean;
    function  RunSelfTest(const ADir: string): Integer;
    function  SelfTestEditors: Integer;
    property Doc: TMcxDoc read FDoc;
  end;

var
  fmMain: TfmMain;

implementation

{$R *.lfm}

type
  { A flat TSpeedButton that does not paint its own background.

    A flat one still asks ThemeServices for the toolbar-button "hot" element
    when the pointer is over it (speedbutton.inc:798), and on gtk2 that is a
    pale fill from the desktop's theme -- white on the ones here.  It landed
    on top of the band painted behind the title and lost it, which is the
    opposite of a highlight.

    Nothing replaces it here because the band underneath is where the hover
    belongs: it is the shape the selection uses, so the two agree.  See
    NavSubPaint. }
  { A push button drawn as a filled rounded shape rather than as a bevel.

    The rest of the window is flat surfaces -- a card is a fill and a
    hairline, a navigator band is a fill -- and a widget-set button in the
    middle of that is the one thing with a raised edge on it.  Flat here
    means the same thing it means for a card: a fill that says where the
    thing is, a stronger fill when the pointer is on it, and the accent when
    it is pressed.

    TSpeedButton rather than TButton because only a TSpeedButton lets the
    background be painted: PaintBackground is virtual on it and there is no
    equivalent on a native button, which the widget set draws whole. }
  TMcxFlatButton = class(TSpeedButton)
  protected
    procedure PaintBackground(var PaintRect: TRect); override;
  end;

  TMcxNavButton = class(TSpeedButton)
  protected
    procedure PaintBackground(var PaintRect: TRect); override;
    procedure Paint; override;
  end;

procedure TMcxNavButton.PaintBackground(var PaintRect: TRect);
begin
  { Deliberately nothing: the caption and the glyph still draw. }
end;

{ Draws the caption itself, hard against the margin.

  A TSpeedButton cannot be told to left-justify one.  Margin does set where
  the text starts -- but for blGlyphLeft with a margin and a spacing the text
  rectangle is then stretched to the full width of the button
  (speedbutton.inc:803) and the text drawn into it with DT_CENTER, taken from
  Alignment, which defaults to taCenter and is not published before Lazarus
  2.3.  So the indent is real and then centred away, which is why a margin of
  40 still produced a column of centred titles.

  The stock painter still draws the glyph; the caption is switched off at the
  button and put back here. }
procedure TMcxNavButton.Paint;
var
  R: TRect;
  x: Integer;
  Style: TTextStyle;
begin
  inherited Paint;
  if Caption = '' then Exit;

  x := Margin;
  if x < 0 then x := 0;
  { Past the glyph, when there is one.  The subsections carry none, so their
    whole indent is the margin. }
  if (Images <> nil) and (ImageIndex >= 0) then
  begin
    Inc(x, Images.Width);
    if Spacing > 0 then Inc(x, Spacing);
  end;

  R := Rect(x, 0, Width, Height);
  FillChar(Style, SizeOf(Style), 0);
  Style.Alignment := taLeftJustify;
  Style.Layout := tlCenter;
  Style.SingleLine := True;
  Style.Clipping := True;

  Canvas.Font := Font;
  Canvas.Brush.Style := bsClear;
  Canvas.TextRect(R, R.Left, R.Top, Caption, Style);
end;

procedure TMcxFlatButton.PaintBackground(var PaintRect: TRect);
var
  Face: TColor;
  R: Integer;
begin
  if not Enabled then Face := McxBlend(McxBase, McxText, 6)
  else if FState in [bsDown, bsExclusive] then
    Face := McxBlend(McxBase, McxAccent, 62)
  else if MouseInClient then Face := McxBlend(McxBase, McxAccent, 26)
  else Face := McxBlend(McxBase, McxText, 12);

  R := McxScale96(8);
  Canvas.AntialiasingMode := amOn;
  Canvas.Brush.Style := bsSolid;
  Canvas.Brush.Color := Face;
  Canvas.Pen.Style := psSolid;
  Canvas.Pen.Color := Face;
  Canvas.RoundRect(0, 0, Width, Height, R, R);
  { The caption is drawn after this, onto this canvas, so what can be read on
    the face is decided here where the face is known. }
  if Enabled then Canvas.Font.Color := McxReadable(Face)
  else Canvas.Font.Color := McxBlend(Face, McxText, 40);
end;

{ Forward, because FormCreate has to register it before the saved layout is
  read and it is defined with the rest of the painting further down. }
procedure DrawMcxDockHeader(Canvas: TCanvas; Style: TADHeaderStyleDesc;
  r: TRect; Horizontal: Boolean; Focused: Boolean); forward;

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
  SubActive  = 32;   { the selected subsection's own band }
  SubHover   = 14;   { and the one the pointer is over }
  HeadHover  = 12;   { how far a hovered section band lifts towards the ink }
  { Neither navigator band has a radius of its own: both are pills, rounded
    by half their own height, which is the one radius that does not have to
    be chosen again when the heading font or the display scale moves -- and
    which makes the two look equally round without being the same number. }
  { A gap between one section and the next, so the bands read as a list of
    separate things rather than as a column divided into stripes. }
  BandGap    = 5;
  { The same between one subsection band and the next.  Smaller, because they
    are a list inside a section rather than the sections themselves. }
  SubGap     = 4;
  { Air between one card and the next on the settings page.  A card is told
    from the page by a shade, which is a quiet difference; the gap is what
    makes a group of settings look like a group at a glance. }
  CardGap    = 18;

  SectionGroups: array[0..6] of string = (
    { Simulator } 'gbEngine,gbDomainData',
    { Domain    } 'gbGrid,gbVolume,gbMesh,gbMedia',
    { Shapes    } 'gbShapeList',
    { Optode    } 'gbSource,gbSrcAdv,gbDetector',
    { Session   } 'gbBasic,gbTime,gbOutput,gbSwitches,gbDetPhoton',
    { Compute   } 'gbSolver,gbGPU,gbDevices',
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

  mmCommand.Font.Name := McxDefaultFontName;
  mmLog.Font.Name := McxDefaultFontName;
  BuildJsonPane;

  BuildDock;
  { The placeholders these replace go with them. }
  lbTodoMedia.Visible := False;
  lbTodoDet.Visible := False;
  BuildChoosers;
  BuildStepBar;
  FMedia := TMcxTable.Create(gbMedia, 'mua (1/mm),mus (1/mm),g,n',
    'mua,mus,g,n', 6);
  FMedia.OnChange := @TableChanged;
  { Dir is optional in mcx and mmc alike -- an acceptance direction where it
    is given -- so a new detector copies it only from a row that has one. }
  FDetectors := TMcxTable.Create(gbDetector,
    'x,y,z,radius,dir x,dir y,dir z,dir w',
    'Pos[0],Pos[1],Pos[2],R,Dir[0]?,Dir[1]?,Dir[2]?,Dir[3]?', 4);
  FDetectors.OnChange := @TableChanged;
  { The wide-field rows after the table rather than above it: the designer
    placed them at the top of a card the table did not exist on yet. }
  rwDetParam1.Top := 10000;
  rwDetParam2.Top := 10001;

  { The Shapes editor, like the two tables, reads the document directly:
    how many commands a domain is built from is not known until a file is
    open, so there is no control-to-path row that could describe it. }
  BuildTabBar;
  BuildSourceBar;
  BuildDeviceCard;
  BuildDomainCard;
  FShapes := TMcxShapes.Create(pnShapeEdit);
  FShapes.OnChange := @TableChanged;
  FShapes.OnSelect := @ShapeSelected;
  FShapes.OnNewLabel := @ShapeLabelAdded;

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

  { Registered before the layout is read, not with the rest of the theme.

    A saved layout carries the header style's name in it -- HeaderStyle="MCX"
    -- and restoring it hands that name back to the dock manager.  If nothing
    has registered a drawer under it by then, the manager is left holding a
    style with no procedure behind it, and the first header to paint calls
    through a nil pointer: an access violation at startup, on the second run
    and every run after, with both the window and the dialog reporting it
    coming up blank.  A clean profile started fine, which is what made it
    look like anything but this.

    Registering is idempotent and the drawer reads the theme when it paints,
    so doing it here costs nothing and ApplyTheme still asks for it again. }
  DockMaster.RegisterHeaderStyle('MCX', @DrawMcxDockHeader, False, False);

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
  { The choosers own a pair of bitmaps per tile -- the artwork and a faded
    copy of it -- and ten tiles across the three of them is twenty, which is
    most of the thirty gdiBitmaps the widget set reported unreleased on the
    way out.  They were built and never freed. }
  FreeAndNil(FProcessor);
  FreeAndNil(FDomainKind);
  FreeAndNil(FMedium);
  FreeAndNil(FMedia);
  FreeAndNil(FDetectors);
  FreeAndNil(FSources);
  FreeAndNil(FShapes);
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
  { 36 rather than 16 on the design grid: the artwork is a coloured badge
    rather than a line glyph, and a badge needs the pixels to read as one. }
  Size := McxScale96(36);
  ilIcons.Width := Size;
  ilIcons.Height := Size;
  { The theme's ink, not the desktop's button colour.  Only the two chevrons
    are drawn rather than rasterised, so they are the only ones this reaches
    -- and they are the ones that need it: a chevron in the desktop's colour
    on a light theme over a dark desktop is pale grey on pale grey.  Called
    again from ApplyTheme, which is why it reads the theme rather than being
    told a colour. }
  McxBuildIconList(ilIcons, McxIconNames, McxText);

  { The buttons are sized from the glyph rather than from the designer.

    ButtonWidth and ButtonHeight are not among the properties the DPI sweep
    touches -- nothing in the LCL claims them -- so the 32 the .lfm carried
    stayed 32 whatever the display did, while the icons above grew with it.
    On this 130-dpi screen that was a 32-pixel badge in a 32-pixel button,
    with no room left for the frame around it.  Tying the two together is
    also the whole of making the bar bigger: one number, and the padding
    keeps its proportion. }
  tbMain.ButtonWidth := Size + McxScale96(12);
  tbMain.ButtonHeight := Size + McxScale96(12);

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

{ The same again for the subsection that is selected, on the body panel its
  buttons sit on.

  Which button it is comes from the button's own parent rather than from
  working out which section this body belongs to: there is one selection in
  the whole navigator, so either it is on this panel or there is nothing to
  draw. }
{ Rounds the section band by taking its corners away again.

  The band itself is the band panel's fill, because a fill is the one thing
  that always reaches the screen: everything drawn on that panel is erased by
  it on the section that was just opened, which is the section whose band
  matters.  What survives the fill is whatever is drawn after it, and the
  heading's caption always did -- so this is a second child, created after
  the heading, and it is the last thing to touch the band.

  It cannot simply draw the rounded shape, because it would draw over the
  caption.  So it paints only what is outside that shape, in the navigator's
  own colour: a margin down each side, and a quarter disc at each corner,
  worked out a row at a time.  Nothing it touches is anywhere a caption
  reaches. }
procedure TfmMain.NavBandCut(Sender: TObject);
var
  B: TPaintBox;
  R, Pad, y, dx: Integer;
  C: TCanvas;

  { The rows of one corner, as the horizontal distance from the band's edge
    to the arc.  ATop and ALeft say which corner. }
  procedure Corner(ATop, ALeft: Boolean);
  var
    i, dy, x0: Integer;
  begin
    for i := 0 to R - 1 do
    begin
      { Distance of this row's centre from the centre of the arc. }
      dy := R - i;
      if R * R - dy * dy > 0 then
        dx := R - Round(Sqrt(R * R - dy * dy))
      else
        dx := R;
      if dx <= 0 then Continue;
      if ATop then y := i else y := B.Height - 1 - i;
      if ALeft then x0 := Pad else x0 := B.Width - Pad - dx;
      C.FillRect(x0, y, x0 + dx, y + 1);
    end;
  end;

begin
  if not (Sender is TPaintBox) then Exit;
  B := TPaintBox(Sender);
  C := B.Canvas;
  { Half the band's height, which makes each end a semicircle: a pill.  Taken
    from the band rather than written down, so it stays a pill when the
    heading font or the display scale changes the height. }
  R := B.Height div 2;
  Pad := McxScale96(4);

  C.Brush.Style := bsSolid;
  C.Brush.Color := McxBase;
  { The margin either side, so the band is a band and not the whole column. }
  C.FillRect(0, 0, Pad, B.Height);
  C.FillRect(B.Width - Pad, 0, B.Width, B.Height);

  Corner(True, True);
  Corner(True, False);
  Corner(False, True);
  Corner(False, False);
end;

procedure TfmMain.NavSubPaint(Sender: TObject);
var
  P: TPanel;
  R, Pad: Integer;

  { One band, in ATint per cent of the accent. }
  procedure Band(AIndex, ATint: Integer);
  var
    B: TSpeedButton;
  begin
    if (AIndex < 0) or (AIndex > High(FSubs)) then Exit;
    B := FSubs[AIndex].Btn;
    if (B = nil) or (B.Parent <> Sender) then Exit;
    P.Canvas.Brush.Style := bsSolid;
    P.Canvas.Brush.Color := McxBlend(McxBase, McxAccent, ATint);
    P.Canvas.Pen.Style := psSolid;
    P.Canvas.Pen.Color := P.Canvas.Brush.Color;
    { A pill.  RoundRect's last two arguments are the width and height of the
      ellipse the corners are cut from, not a radius -- so a pill wants the
      band's whole height, and half of it, which is what a radius would be,
      gives corners half as round as they should be.

      Indented past the band above, so the shape says what the indent of the
      title already says: this one is under that one.  A band starting where
      its parent's starts reads as a second section. }
    R := B.Height;
    P.Canvas.RoundRect(Pad + McxScale96(18), B.Top, P.Width - Pad,
      B.Top + B.Height, R, R);
  end;

begin
  if not (Sender is TPanel) then Exit;
  P := TPanel(Sender);
  Pad := McxScale96(4);
  P.Canvas.AntialiasingMode := amOn;

  { The pointer first, so that the selected one wins where they are the same
    title: a selection is a stronger statement than a pointer resting. }
  if FSubHot <> FSub then Band(FSubHot, SubHover);
  Band(FSub, SubActive);
end;

{ Replaces a designer-placed heading with one of ours on AParent, carrying
  over everything the .lfm set on it, and frees the original. }
function TfmMain.SwapHeading(AOld: TSpeedButton; AParent: TWinControl): TSpeedButton;
var
  B: TMcxNavButton;
begin
  B := TMcxNavButton.Create(Self);
  B.Parent := AParent;
  B.Align := AOld.Align;
  B.Height := AOld.Height;
  B.BorderSpacing.Bottom := AOld.BorderSpacing.Bottom;
  B.Caption := AOld.Caption;
  { The stock painter centres a caption whatever the margin says, so ours
    draws it; this stops it being drawn twice.  See TMcxNavButton.Paint. }
  B.ShowCaption := False;
  B.Flat := AOld.Flat;
  B.Images := AOld.Images;
  B.ImageIndex := AOld.ImageIndex;
  B.Layout := AOld.Layout;
  B.Margin := AOld.Margin;
  B.Spacing := AOld.Spacing;
  B.ParentFont := False;
  B.Font.Assign(AOld.Font);
  B.Hint := AOld.Hint;
  B.ShowHint := AOld.ShowHint;
  B.OnClick := AOld.OnClick;
  AOld.Free;
  Result := B;
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
  FOpen := True;
  FSubHot := -1;
  FHeadHot := -1;
  SetLength(FBands, Length(FPanes));
  SetLength(FBandCut, Length(FPanes));
  for i := 0 to High(FPanes) do
  begin
    FPanes[i].ParentColor := False;
    FPanes[i].BorderSpacing.Bottom := BandGap;

    { The heading moves onto a panel of its own, above the body, and the band
      is painted on that.  AutoSize, so the strip is whatever the heading
      turns out to be at this display's scale. }
    FBands[i] := TPanel.Create(FPanes[i]);
    FBands[i].Parent := FPanes[i];
    FBands[i].Top := 0;
    FBands[i].Align := alTop;
    { Not AutoSize: the band is measured from its heading once, here, and
      the DPI sweep scales it with everything else afterwards.  With AutoSize
      on, setting the paint box below to the band's size fed straight back
      into the band's size, and the LCL caught it as a ChangeBounds loop. }
    FBands[i].AutoSize := False;
    FBands[i].Height := FHeads[i].Height + FHeads[i].BorderSpacing.Bottom;
    FBands[i].BevelOuter := bvNone;
    FBands[i].ParentColor := False;
    { The band is the panel's own fill.

      It was meant to be a rounded shape painted on top of it, and that is
      still what the closed sections would get -- but not the open one, and
      the open one is the whole point.  The face of the band panel is filled
      by the widget set after the LCL has drawn on it and before the heading
      is drawn over it, on the one section whose height just changed: which
      is why a caption survived there and a band never did.  Painting it on
      the section panel, on a panel of its own, and on a paint box inside
      that all lost to the same fill, as did invalidating from
      UpdateNavState, from the message queue, from the resize itself, and
      with FullRepaint off.  A fill is the one thing that always arrives, so
      the band is a fill, and square. }

    FBandCut[i] := TPaintBox.Create(FBands[i]);
    FBandCut[i].Parent := FBands[i];
    FBandCut[i].SetBounds(0, 0, FBands[i].Width, FBands[i].Height);
    FBandCut[i].Anchors := [akLeft, akTop, akRight, akBottom];
    { It lies over the heading, so it must not take the heading's clicks.
      A disabled graphic control is passed over by the hit test and still
      paints whatever its OnPaint draws, which is all this one is for. }
    FBandCut[i].Enabled := False;

    { The heading is swapped for one that does not paint its own background,
      for the reason TMcxNavButton gives: a flat TSpeedButton asks the widget
      set for a "hot" face when the pointer is over it, and that face lands on
      top of the band.  The designer cannot place a class of ours -- it would
      have to be registered in a package, which is the trap the whole project
      avoids -- so the one it placed is copied and dropped.  Everything after
      this point knows it only through FHeads. }
    FHeads[i] := SwapHeading(FHeads[i], FBands[i]);
    FHeads[i].OnMouseEnter := @HeadEnter;
    FHeads[i].OnMouseLeave := @HeadLeave;
    FBandCut[i].OnPaint := @NavBandCut;

    FBodies[i].ParentColor := False;
    FBodies[i].Color := McxBase;
    FBodies[i].OnPaint := @NavSubPaint;
    { A heading is a thing you click, and nothing else about it says so:
      it has no frame, no arrow but the chevron, and it does not depress. }
    FHeads[i].Cursor := crHandPoint;
    { And it must not depress, which is the whole reason the open section
      was a square.  A grouped TSpeedButton latches Down when it is clicked
      and the widget set then fills its whole rectangle with the pressed
      face -- square, edge to edge, over the top of the band painted behind
      it.  With no group there is no latch, nothing reads Down, and the band
      is the only thing saying which section is open. }
    FHeads[i].GroupIndex := 0;
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

        { Air under each card.  The designer gives every one the same margin
          all round, which leaves adjacent cards nearly touching; a group of
          settings that has to be told from the next one by a shade alone is
          a group you have to look for. }
        Card.BorderSpacing.Bottom := CardGap;

        B := TMcxNavButton.Create(Self);
        B.Parent := FBodies[s];
        B.Height := 44;
        { The gap the band is drawn with room for.  Real spacing rather than
          an inset on the paint, so the pill is the button's own rectangle
          and the two cannot drift apart. }
        B.BorderSpacing.Bottom := SubGap;
        { Top before Align: alTop children are ordered by the Top they have
          when they are aligned, so leaving them all at zero lists the
          subsections in reverse. }
        B.Top := g * (44 + SubGap);
        B.Align := alTop;
        B.AllowAllUp := True;
        B.Caption := Card.Caption;
        { Drawn by TMcxNavButton.Paint, left-aligned, rather than by the
          stock painter, which centres it. }
        B.ShowCaption := False;
        B.Flat := True;
        { No group, so it cannot latch: see the headings above -- a pressed
          TSpeedButton paints a square face over the band behind it. }
        B.GroupIndex := 0;
        B.Layout := blGlyphLeft;
        { Indented past where a section heading's own text starts, so the
          hierarchy is visible without a second glyph column.  The button
          carries no glyph, so this margin is the whole of the indent. }
        B.Margin := 40;
        B.Spacing := 6;
        B.ParentFont := False;
        B.Font.Height := -28;
        B.Tag := Length(FSubs);
        B.OnClick := @SubClick;
        B.OnMouseEnter := @SubEnter;
        B.OnMouseLeave := @SubLeave;

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
      FBodies[i].Visible := (i = AIndex) and FOpen;
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
  { The device list is asked for the first time its page is opened, not at
    startup: -L initialises the driver, which takes a second or two that
    nobody opening a file should have to wait through. }
  if (FDevList <> nil) and (gbDevices.Parent = FPages[AIndex]) and
     (FDevFor <> McxExeName(CurrentBackend)) then
    RefreshDevices;
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

{ The pointer arriving over a subsection title, and leaving it.  Only the
  body panel it sits on has to be told: the band is painted there. }
procedure TfmMain.SubEnter(Sender: TObject);
begin
  if not (Sender is TSpeedButton) then Exit;
  FSubHot := TSpeedButton(Sender).Tag;
  if (FSubHot >= 0) and (FSubHot <= High(FSubs)) then
    FSubs[FSubHot].Btn.Parent.Invalidate;
end;

procedure TfmMain.SubLeave(Sender: TObject);
var
  Was: Integer;
begin
  Was := FSubHot;
  FSubHot := -1;
  if (Was >= 0) and (Was <= High(FSubs)) then
    FSubs[Was].Btn.Parent.Invalidate;
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
  ShowSettingsPage;
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
      { And no sunken frame around it.  The field is told apart from the card
        it sits on by being a shade off it, which is the same way a card is
        told from the page; a bevel as well is the widget set's idea of a
        field rather than this window's.  BorderStyle is published on every
        one of these, and bsNone is the only thing it has to say. }
      if AControl is TCustomEdit then
        TCustomEdit(AControl).BorderStyle := bsNone
      else if AControl is TCustomListBox then
        TCustomListBox(AControl).BorderStyle := bsNone
      else if AControl is TCustomDrawGrid then
        TCustomDrawGrid(AControl).BorderStyle := bsNone;
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
      SetInk(AControl, clDefault)
    else if AControl is TAnchorDockSplitter then
      AControl.Invalidate;
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
    McxApplyChromeColours(clNone, clNone, clNone, clNone, clNone, clNone)
  else
    McxApplyChromeColours(McxBase, McxText, McxAccent,
      McxBlend(McxBase, McxText, 4), McxReadable(McxAccent),
      { A lift towards the accent: enough to say the pointer is here, not so
        much that it competes with what is actually selected. }
      McxBlend(McxBase, McxAccent, 22));

  Color := McxBase;
  Font.Color := McxText;

  sbNav.Color := McxBase;
  for i := 0 to High(FBodies) do FBodies[i].Color := McxBase;

  { The bars and the notebook too, where the widget set lets them be told.
    Where it does not -- gtk2 draws a tab strip itself -- the request is
    harmless and the strip stays the desktop's. }
  tbMain.ParentColor := False;
  tbMain.Color := McxBlend(McxBase, McxText, 6);
  { The etched line under the bar is the desktop's two greys, one of them
    white: under a theme of our own the change of shade is the edge. }
  if McxCurrentTheme = mtSystem then tbMain.EdgeBorders := [ebBottom]
  else tbMain.EdgeBorders := [];
  sbMain.ParentColor := False;
  sbMain.Color := McxBlend(McxBase, McxText, 6);
  TColourAccess(pcView).ParentColor := False;
  TColourAccess(pcView).Color := McxBase;
  if FTabBar <> nil then FTabBar.Color := McxBase;
  for i := 0 to pcView.PageCount - 1 do
  begin
    TColourAccess(pcView.Pages[i]).ParentColor := False;
    TColourAccess(pcView.Pages[i]).Color := McxBase;
  end;
  if sbDetail <> nil then
  begin
    sbDetail.ParentColor := False;
    sbDetail.Color := McxBase;
    { Opaque under a theme of our own.  On Windows a control on a notebook
      page is drawn with the page's themed body behind it -- white -- unless
      it says it paints all of itself, and a scroll box never does: the cards
      covered that, and the page showed through wherever they ended. }
    if McxCurrentTheme = mtSystem then
      sbDetail.ControlStyle := sbDetail.ControlStyle - [csOpaque]
    else
      sbDetail.ControlStyle := sbDetail.ControlStyle + [csOpaque];
    sbDetail.Invalidate;
  end;

  { Under a theme of our own the status bar is a strip we paint, with the
    native one hidden and kept only for its panels' texts and widths.  On
    Windows the native bar's face, the frame round each panel and the gaps
    between them are the desktop's greys and white, whatever its Color says
    -- custom-drawn panels only reach the inside of each frame. }
  if FStatus = nil then
  begin
    FStatus := TPaintBox.Create(Self);
    FStatus.Parent := Self;
    { Below everything else along the bottom: alBottom siblings are stacked
      by the Top they have when they are aligned. }
    FStatus.Top := ClientHeight + 1000;
    FStatus.Align := alBottom;
    FStatus.Height := sbMain.Height;
    FStatus.OnPaint := @StatusPaint;
  end;
  FStatus.Visible := McxCurrentTheme <> mtSystem;
  sbMain.Visible := McxCurrentTheme = mtSystem;
  FStatus.Invalidate;

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
  { The pane and the colours the highlighter paints on it, together: the
    tokens are mixed from the same three roles as everything else, so the
    JSON belongs to whichever theme is on rather than to a palette of its
    own. }
  if FJson <> nil then
  begin
    FJson.Color := McxBlend(McxBase, McxText, 4);
    FJson.Font.Color := McxText;
    FJson.SelectedColor.Background := McxAccent;
    FJson.SelectedColor.Foreground := McxReadable(McxAccent);
    { The gutter is a margin rather than a second surface: the page's own
      colour, with the numbers faded enough to be countable and not read. }
    FJson.Gutter.Color := McxBase;
    for i := 0 to FJson.Gutter.Parts.Count - 1 do
      if FJson.Gutter.Parts[i] is TSynGutterLineNumber then
        with TSynGutterLineNumber(FJson.Gutter.Parts[i]) do
        begin
          MarkupInfo.Background := McxBase;
          MarkupInfo.Foreground := McxBlend(McxBase, McxText, 45);
        end;
    FJsonSyn.Recolour(FJson.Color, McxText, McxAccent);
    FJson.Invalidate;
    if FJsonNote <> nil then FJsonNote.Font.Color := McxText;
  end;
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

  { The drawn glyphs carry the theme's ink, so they are rebuilt with it. }
  BuildIcons;
  ApplyNativeChrome;

  UpdateNavState;
  Invalidate;
end;

{ What the widget set draws itself on Windows -- scroll bars, the status
  bar's face, the title bar -- told about the theme.  Separate from
  ApplyTheme because it needs the windows to exist, and at the first
  ApplyTheme, in FormCreate, they do not: it runs again once the form is
  shown. }
procedure TfmMain.ApplyNativeChrome;
var
  Custom, Dark: Boolean;
  r, g, b: Byte;
  i: Integer;
begin
  Custom := McxCurrentTheme <> mtSystem;
  RedGreenBlue(ColorToRGB(McxBase), r, g, b);
  Dark := Custom and (r * 299 + g * 587 + b * 114 < 128000);
  McxNativeScrollBars(sbDetail, Dark);
  McxNativeScrollBars(sbNav, Dark);
  McxNativeScrollBars(mmLog, Dark);
  McxNativeScrollBars(mmCommand, Dark);
  McxNativeScrollBars(FJson, Dark);
  for i := 0 to Screen.FormCount - 1 do
    McxNativeTitleBar(Screen.Forms[i], Dark);
end;

{ The painted status bar: sbMain's panels at sbMain's widths, the last one
  taking the rest, with a hairline between them rather than a frame round
  each. }
procedure TfmMain.StatusPaint(Sender: TObject);
var
  C: TCanvas;
  i, x, w, y: Integer;
begin
  C := FStatus.Canvas;
  C.Brush.Color := McxBlend(McxBase, McxText, 6);
  C.FillRect(0, 0, FStatus.Width, FStatus.Height);
  C.Font.Assign(sbMain.Font);
  C.Font.Color := McxText;
  y := (FStatus.Height - C.TextHeight('Ay')) div 2;
  x := 0;
  for i := 0 to sbMain.Panels.Count - 1 do
  begin
    w := sbMain.Panels[i].Width;
    if i = sbMain.Panels.Count - 1 then w := FStatus.Width - x;
    C.Brush.Style := bsClear;
    C.ClipRect := Rect(x, 0, x + w - McxScale96(4), FStatus.Height);
    C.Clipping := True;
    C.TextOut(x + McxScale96(6), y, sbMain.Panels[i].Text);
    C.Clipping := False;
    Inc(x, w);
    if i < sbMain.Panels.Count - 1 then
    begin
      C.Pen.Color := McxBlend(McxBase, McxText, 22);
      C.Line(x - 1, McxScale96(4), x - 1, FStatus.Height - McxScale96(4));
    end;
  end;
end;

procedure TfmMain.SetStatus(AIndex: Integer; const AText: string);
begin
  sbMain.Panels[AIndex].Text := AText;
  if (FStatus <> nil) and FStatus.Visible then FStatus.Invalidate;
end;

{ Every setting back to what a blank simulation opens with.

  Asked first, because it cannot be undone and the button is next to two that
  can.  The file name is kept: this is the same document with its contents
  reset, so Save still knows where it goes, and nothing on disk changes until
  it is saved. }
procedure TfmMain.ResetSettings(Sender: TObject);
var
  Keep: string;
begin
  if MessageDlg('MCX Studio',
       'Put every setting back to what a new simulation starts with?' +
       LineEnding + LineEnding +
       'Nothing on disk changes until this is saved.',
       mtConfirmation, [mbYes, mbNo], 0) <> mrYes then Exit;

  Keep := FDoc.FileName;
  SeedDocument(FDoc);
  FDoc.FileName := Keep;
  FDoc.Modified := True;

  FJsonEdited := False;
  FJsonFault := False;
  DropResult;
  GuessRunSettings;
  LoadAllBindings;
  UpdateTitle;
  RefreshPreview;
  Log('settings reset');
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

{ The colour of every section band, which is level, then state, and nothing
  else -- plus a lift for the one the pointer is over.

  Its own procedure because hovering has to redo it and nothing else, and
  because the band is the only hover a heading has: the button under the
  pointer no longer paints one.  See TMcxNavButton. }
{ The notebook's own tab strip, replaced by a row of flat buttons.

  A TPageControl's tabs are the widget set's, drawn with its notches and its
  edges, and there is no way to ask for anything else: the LCL offers no
  owner drawing for them.  What it does offer is ShowTabs, and with the tabs
  off the notebook is a stack of pages and the strip can be anything.  Here
  it is the same flat button the wizard bar uses, which is the point -- one
  idea of what a button looks like, in both places.

  Grouped, so that exactly one is down and the down one is drawn in the
  accent by TMcxFlatButton, which is the whole of showing which page is
  showing. }
procedure TfmMain.BuildTabBar;
var
  i: Integer;
  B: TMcxFlatButton;
begin
  pcView.ShowTabs := False;
  {$IFDEF WINDOWS}
  { With the tabs off the Windows notebook still draws its body: a light
    frame and a white margin round the page.  The button style has no body,
    and with no tabs it has no buttons either, so the page is all there is. }
  pcView.Style := tsFlatButtons;
  {$ENDIF}

  FTabBar := TPanel.Create(pnPreview);
  FTabBar.Parent := pnPreview;
  { Above the notebook, which is alClient and takes what is left. }
  FTabBar.Top := -1;
  FTabBar.Align := alTop;
  FTabBar.AutoSize := True;
  FTabBar.Canvas.Font.Assign(Font);
  FTabBar.BevelOuter := bvNone;
  FTabBar.ParentColor := False;

  SetLength(FTabs, pcView.PageCount);
  for i := 0 to pcView.PageCount - 1 do
  begin
    B := TMcxFlatButton.Create(FTabBar);
    B.Parent := FTabBar;
    { Numbered, for the reason every other row of aligned siblings here is:
      left at zero they come out in whatever order the control list holds. }
    B.Left := i * 1000;
    B.Align := alLeft;
    B.BorderSpacing.Around := 3;
    B.Caption := pcView.Pages[i].Caption;
    { Measured rather than AutoSized: a TSpeedButton sized to its caption is
      sized to exactly its caption, and a word touching both ends of the
      shape it sits in does not read as a button.  Raw 96-dpi here, because
      the startup sweep scales it with everything else. }
    B.Height := 26;
    B.Width := FTabBar.Canvas.TextWidth(B.Caption) + 24;
    B.GroupIndex := 3;
    B.AllowAllUp := False;
    B.Tag := i;
    B.Cursor := crHandPoint;
    B.OnClick := @TabClick;
    FTabs[i] := B;
  end;

  pcView.OnChange := @TabChanged;
  UpdateTabBar;
end;

procedure TfmMain.TabClick(Sender: TObject);
begin
  if not (Sender is TSpeedButton) then Exit;
  pcView.ActivePageIndex := TSpeedButton(Sender).Tag;
  UpdateTabBar;
end;

procedure TfmMain.TabChanged(Sender: TObject);
begin
  UpdateTabBar;
end;

{ The page can be changed from anywhere -- a run switches to the log, a
  result to the preview -- so which button is down follows the notebook
  rather than the other way about. }
procedure TfmMain.UpdateTabBar;
var
  i: Integer;
begin
  for i := 0 to High(FTabs) do
    FTabs[i].Down := (i = pcView.ActivePageIndex);
end;

procedure TfmMain.UpdateHeadBands;
var
  i: Integer;
begin
  for i := 0 to High(FBands) do
  begin
    if i = FSection then
      FBands[i].Color := McxBlend(McxBase, McxAccent, BandActive)
    else
      FBands[i].Color := McxBlend(McxBase, McxText, BandLevel);
    { A lift towards the ink, so it reads the same way on the accent band and
      on a neutral one: brighter than it was, whatever it was. }
    if i = FHeadHot then
      FBands[i].Color := McxBlend(FBands[i].Color, McxText, HeadHover);
    { Re-asserted here rather than once at startup: CollectSections runs
      before McxLoadTheme, so a colour read there is the desktop's rather
      than the theme's -- which left a pale gap between the bands. }
    FPanes[i].Color := McxBase;
    { The heading sits on the band, and the open band is the accent -- a
      fixed colour, where the surface is not.  So what can be read on it is
      decided from the band and not from the theme's text colour. }
    FHeads[i].Font.Color := McxReadable(FBands[i].Color);
    FBands[i].Invalidate;
    FBandCut[i].Invalidate;
  end;
end;

procedure TfmMain.HeadEnter(Sender: TObject);
var
  i: Integer;
begin
  for i := 0 to High(FHeads) do
    if FHeads[i] = Sender then
    begin
      FHeadHot := i;
      UpdateHeadBands;
      Exit;
    end;
end;

procedure TfmMain.HeadLeave(Sender: TObject);
begin
  if FHeadHot < 0 then Exit;
  FHeadHot := -1;
  UpdateHeadBands;
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
    if (i = FSection) and FOpen then FHeads[i].ImageIndex := Open
    else FHeads[i].ImageIndex := Closed;
  end;
  UpdateHeadBands;

  for i := 0 to High(FSubs) do
  begin
    FSubs[i].Btn.Cursor := crHandPoint;
    { The selected title sits on a band of its own -- weaker than the
      section's above it, so the two read as two levels rather than as two
      headings -- and takes whatever ink can be read on that. }
    if i = FSub then
      FSubs[i].Btn.Font.Color :=
        McxReadable(McxBlend(McxBase, McxAccent, SubActive))
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

  { The band under a title is painted by the body panel, so the panel is what
    has to be told the selection moved. }
  for i := 0 to High(FBodies) do FBodies[i].Invalidate;



  { And the repaint is asked for again once the layout has settled.

    Opening a section shows its body, which changes the section panel's
    AutoSize height -- and a resize has the widget set fill the new extent
    with the panel's own background, after the band has been drawn on it and
    without an LCL Paint to draw it again.  Invalidating here is too early:
    the resize has not happened yet.  So the band of the one section that
    just changed size came out as the panel's plain colour, which is what an
    open section looked like whatever colour it was given. }

end;

{ Picking a section in the navigator brings the settings forward with it.

  The navigator names what you are editing and the notebook holds what you
  edit it in, so a click on one that leaves the other showing the log or the
  3-D view answers a question nobody asked.  It is most of the way to a dead
  control: after a run the notebook sits on the log, and from there every
  section in the list looked like it did nothing. }
{ How many sources the file describes, or nought when it describes one.

  mcx takes Pos, Dir, Param1 and Param2 either as one vector or as a list of
  equal-length vectors, a row per source (mcx_utils.c reads both, and
  example/multisrc/multisrc.json is the second).  Pos decides, because a file
  with several sources has to give each one a position; the others follow it
  or are absent. }
function TfmMain.SourceRows: Integer;
var
  A: TJSONData;
begin
  Result := 0;
  if FDoc = nil then Exit;
  A := FDoc.Find('Optode.Source.Pos');
  if not (A is TJSONArray) then Exit;
  if TJSONArray(A).Count = 0 then Exit;
  if TJSONArray(A).Items[0].JSONType <> jtArray then Exit;
  Result := TJSONArray(A).Count;
end;

{ Where a Mesh binding reads and writes.  mmc takes its mesh block from
  "Mesh", or from "Shapes" when there is no Mesh -- which is how mmc's own
  --dumpjson writes it, and so how every exported test input carries it.
  Writing Mesh.InitElem into such a file would create a Mesh holding nothing
  but InitElem, and mmc would read that and lose the mesh. }
{ Where a vector binding actually reads and writes.

  One rule for every mkVec rather than four special cases: a vector that
  turns out to be a table of vectors is edited a row at a time, and which
  row is the source bar's business.  A vector that is a vector is untouched,
  which is every other one on the form. }
function TfmMain.MeshPath(const APath: string): string;
begin
  Result := APath;
  if (FDoc = nil) or (Pos('Mesh.', APath) <> 1) then Exit;
  if FDoc.Find('Mesh') <> nil then Exit;
  if FDoc.Find('Shapes') is TJSONObject then
    Result := 'Shapes.' + Copy(APath, 6, MaxInt);
end;

function TfmMain.VecPath(const APath: string): string;
var
  A: TJSONData;
begin
  Result := APath;
  if FDoc = nil then Exit;
  A := FDoc.Find(APath);
  if not (A is TJSONArray) then Exit;
  if TJSONArray(A).Count = 0 then Exit;
  if TJSONArray(A).Items[0].JSONType <> jtArray then Exit;
  if FSource >= TJSONArray(A).Count then Exit;
  Result := Format('%s[%d]', [APath, FSource]);
end;

{ The source table, under the four vectors on the source card.  Built here
  rather than in the designer because how many sources there are is not known
  until a file is open -- the same reason the media and detector tables are.

  The vectors above it show the row picked here, so the table is both the
  overview and the way to say which source the rest of the card is about. }
procedure TfmMain.BuildSourceBar;
begin
  FSrcHead := TLabel.Create(gbSource);
  FSrcHead.Parent := gbSource;
  FSrcHead.Top := 10000;
  FSrcHead.Align := alTop;
  FSrcHead.BorderSpacing.Top := 6;
  FSrcHead.BorderSpacing.Bottom := 3;
  FSrcHead.WordWrap := True;
  FSrcHead.Caption := '';

  FSources := TMcxVecTable.Create(gbSource, 'Pos,Dir,Param1,Param2',
    'x,y,z,w,dir x,dir y,dir z,focus,p1 x,p1 y,p1 z,p1 w,' +
    'p2 x,p2 y,p2 z,p2 w', 4, 3);
  FSources.OnChange := @SourceTableChanged;
  FSources.OnSelect := @SourceChanged;
  { After the heading, which is after the designer's rows, and the buttons
    after the grid. }
  FSources.Grid.Top := 10001;
  FSources.Bar.Top := 10002;
end;

{ Counts from one, because mcx counting its sources from zero is mcx's
  business and not the reader's. }
procedure TfmMain.UpdateSourceBar;
var
  n: Integer;
begin
  if FSources = nil then Exit;
  if not FSrcEditing then FSources.Attach(FDoc, 'Optode.Source');
  n := FSources.Count;
  if FSource >= n then FSource := n - 1;
  if FSource < 0 then FSource := 0;
  FSources.Selected := FSource;
  if n > 1 then
    FSrcHead.Caption := Format('%d sources.  The fields above show source %d; ' +
      'pick a row to edit another.', [n, FSource + 1])
  else
    FSrcHead.Caption := 'Sources.  Add a row for a second one: Pos, Dir, ' +
      'Param1 and Param2 become one row per source, all the same length.';
end;

procedure TfmMain.SourceChanged(Sender: TObject);
begin
  if FLoading > 0 then Exit;
  FSource := FSources.Selected;
  { Only the four vectors move, but reloading the lot is one call and costs
    nothing anybody can see. }
  LoadAllBindings;
end;

{ An edit in the table changes what the vectors above show as well as the
  file, so they are reloaded with it. }
procedure TfmMain.SourceTableChanged(Sender: TObject);
begin
  TableChanged(Sender);
  FSrcEditing := True;
  try
    LoadAllBindings;
  finally
    FSrcEditing := False;
  end;
end;

procedure TfmMain.ShowSettingsPage;
begin
  if pcView.ActivePage <> tsSettings then pcView.ActivePage := tsSettings;
end;

{ Clicking the open section closes it again.

  The grouped TSpeedButton used to do this on its own -- AllowAllUp let the
  latched one come back up -- and it went when the group did, taking the
  chevron with it: a section opened once stayed open and kept pointing down
  however often it was clicked.  Two lines of state instead, which is also
  the only reader of the chevron's direction.

  Only the subsection list closes.  The page stays where it is, because a
  detail pane showing nothing is not a state worth having, and because the
  section is still the one being edited whether or not its subsections are
  listed. }
procedure TfmMain.HeaderClick(Sender: TObject);
var
  i: Integer;
begin
  for i := 0 to High(FHeads) do
    if FHeads[i] = Sender then
    begin
      if i = FSection then FOpen := not FOpen else FOpen := True;
      SelectSection(i);
      ShowSettingsPage;
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

{ The first card: three rows of picture tiles.

  Built in code rather than in the designer because a tile is a picture and a
  caption, not a widget, and because the three rows are the same thing three
  times -- one declaration each is shorter than three panels of controls and
  cannot drift out of step with the rule that reads them. }

{ The JSON pane.

  Built in code for the reason the 3-D view is: the designer cannot place a
  class that is not registered in a package it knows, and registering one is
  the trap this project avoids.  The .lfm stays a form anybody can open.

  Read-only, because this pane is the document rendered rather than a second
  place to edit it -- everything here is written by a control on the settings
  page, and two ways to change one value is one way too many.  Selecting and
  copying still work, which is what it is actually used for. }
procedure TfmMain.BuildJsonPane;
var
  i: Integer;
begin
  FJsonSyn := TMcxJsonSyn.Create(Self);

  { The bar first, so that alBottom takes its space before the editor claims
    what is left. }
  FJsonBar := TPanel.Create(Self);
  FJsonBar.Parent := tsJSON;
  FJsonBar.Align := alBottom;
  FJsonBar.Height := 40;
  FJsonBar.BevelOuter := bvNone;
  FJsonBar.Caption := '';

  FJsonApply := TMcxFlatButton.Create(FJsonBar);
  FJsonApply.Parent := FJsonBar;
  { alRight siblings are ordered by the Left they hold when they are aligned,
    so a larger one is further right. }
  FJsonApply.Left := 2000;
  FJsonApply.Align := alRight;
  FJsonApply.Width := 130;
  FJsonApply.BorderSpacing.Around := 6;
  FJsonApply.Caption := 'Use this JSON';
  FJsonApply.Cursor := crHandPoint;
  FJsonApply.ShowHint := True;
  FJsonApply.Hint := 'Read what is in this pane and rebuild the whole ' +
    'simulation from it, replacing every setting on the Settings page.';
  FJsonApply.OnClick := @JsonApply;

  FJsonRevert := TMcxFlatButton.Create(FJsonBar);
  FJsonRevert.Parent := FJsonBar;
  FJsonRevert.Left := 1000;
  FJsonRevert.Align := alRight;
  FJsonRevert.Width := 90;
  FJsonRevert.BorderSpacing.Around := 6;
  FJsonRevert.Caption := 'Revert';
  FJsonRevert.Cursor := crHandPoint;
  FJsonRevert.ShowHint := True;
  FJsonRevert.Hint := 'Throw away the edits in this pane and show the ' +
    'simulation as it stands.';
  FJsonRevert.OnClick := @JsonRevert;

  FJsonNote := TLabel.Create(FJsonBar);
  FJsonNote.Parent := FJsonBar;
  FJsonNote.Align := alClient;
  FJsonNote.Layout := tlCenter;
  FJsonNote.BorderSpacing.Left := 10;
  FJsonNote.WordWrap := True;

  FJson := TSynEdit.Create(Self);
  FJson.Parent := tsJSON;
  FJson.Align := alClient;
  FJson.Highlighter := FJsonSyn;
  FJson.Font.Name := McxDefaultFontName;
  FJson.Font.Size := McxScalePointSize(McxDefaultFontSize);
  FJson.RightEdge := 0;
  FJson.BorderStyle := bsNone;
  FJson.Options := FJson.Options - [eoScrollPastEol];
  FJson.OnChange := @JsonChanged;

  { Line numbers and nothing else.  A gutter that also carries bookmarks,
    a fold tree and a change bar is four columns of furniture for a pane
    where only one of them means anything. }
  FJson.Gutter.Visible := True;
  for i := 0 to FJson.Gutter.Parts.Count - 1 do
    FJson.Gutter.Parts[i].Visible :=
      FJson.Gutter.Parts[i] is TSynGutterLineNumber;

  UpdateJsonBar;
end;

{ Writes the document into the pane.

  Not while it is being edited: the text on screen is then the person's
  rather than ours, and replacing it as they type is the one thing that
  makes an editable pane useless.  Where it is scrolled to survives the
  rewrite, or reading the file while changing a number on the settings page
  is impossible. }
procedure TfmMain.ShowJson;
var
  JsonTop: Integer;
begin
  if (FJson = nil) or FJsonEdited then Exit;
  JsonTop := FJson.TopLine;
  Inc(FJsonLoading);
  try
    FJson.Text := FDoc.ToJSON(True);
    FJson.Modified := False;
  finally
    Dec(FJsonLoading);
  end;
  FJson.TopLine := JsonTop;
  UpdateJsonBar;
end;

procedure TfmMain.JsonChanged(Sender: TObject);
begin
  if FJsonLoading > 0 then Exit;
  { The fault goes as soon as the text moves: it was about the old text. }
  FJsonFault := False;
  if FJsonEdited then
  begin
    UpdateJsonBar;
    Exit;
  end;
  FJsonEdited := True;
  UpdateJsonBar;
end;

procedure TfmMain.UpdateJsonBar;
begin
  if FJsonApply = nil then Exit;
  FJsonApply.Enabled := FJsonEdited;
  FJsonRevert.Enabled := FJsonEdited;
  { A parse failure has the floor until the text changes again. }
  if FJsonFault then Exit;
  FJsonNote.Font.Color := McxText;
  if FJsonEdited then
    FJsonNote.Caption := 'Edited here.  "Use this JSON" rebuilds the ' +
      'simulation from it.'
  else
    FJsonNote.Caption := 'The simulation as it stands.  Paste a file over it ' +
      'to start from that instead.';
end;

{ Rebuilds the whole simulation from what is in the pane.

  This is the point of the pane being editable: a simulation arrives as a
  file far more often than it is built a control at a time, and pasting one
  in should be the same as opening it.  So the document is replaced outright
  -- every setting on the settings page is reset from this text, and any that
  the text does not mention goes back to nothing. }
procedure TfmMain.JsonApply(Sender: TObject);
var
  Keep: string;
begin
  { The file name survives: this is new contents for the same document, in
    the same folder, which is what relative paths inside it resolve against. }
  Keep := FDoc.FileName;
  if not FDoc.LoadFromString(FJson.Text) then
  begin
    FJsonFault := True;
    FJsonNote.Caption := 'Not read: ' + FDoc.LastError;
    FJsonNote.Font.Color := clRed;
    Exit;
  end;
  FDoc.FileName := Keep;
  FDoc.Modified := True;

  FJsonEdited := False;
  FJsonFault := False;

  { The same sequence opening a file runs, and for the same reason: which
    simulator and which kind of domain are read off the new contents, every
    control is loaded from it, and the preview is redrawn. }
  DropResult;
  GuessRunSettings;
  LoadAllBindings;
  UpdateTitle;
  RefreshPreview;
  Log('rebuilt the simulation from the JSON pane');
end;

procedure TfmMain.JsonRevert(Sender: TObject);
begin
  FJsonEdited := False;
  FJsonFault := False;
  ShowJson;
end;

procedure TfmMain.BuildChoosers;
begin
  FMedium := TMcxChooser.Create(gbEngine, 'What kind of medium');
  FMedium.Add('diffusive', 'Diffusive media', 'diffusive',
    'Ordinary scattering tissue.  Any solver can do this.');
  FMedium.Add('lowscatter', 'Low-scattering, curved boundaries', 'lowscatter',
    'Light travels a long way between scattering events, so where the ' +
    'boundaries are matters more than what is between them.  On a voxel ' +
    'grid this needs split-voxel mode; a mesh describes the boundaries ' +
    'exactly and needs nothing extra.');
  FMedium.OnChange := @ChoiceChanged;

  FDomainKind := TMcxChooser.Create(gbEngine, 'How the domain is described');
  FDomainKind.Add('voxel', 'Voxel grid', 'grid',
    'A volume of labelled voxels, read from a file.');
  FDomainKind.Add('shapes', 'Primitive shapes', 'sphere',
    'Boxes, spheres, cylinders and layers, listed in this file and ' +
    'rasterised onto a grid when the run starts.');
  FDomainKind.Add('mesh', 'Tetrahedral mesh', 'tetmesh',
    'A volume filled with tetrahedra.  Boundaries are exact rather than ' +
    'stepped, which is what MMC is for.');
  FDomainKind.Add('surface', 'Surface mesh', 'surfmesh',
    'Closed triangulated surfaces with the medium between them, traced ' +
    'with OptiX instead of walked tetrahedron by tetrahedron.');
  FDomainKind.OnChange := @ChoiceChanged;

  FProcessor := TMcxChooser.Create(gbEngine, 'What to run it on');
  FProcessor.Add('nvidia', 'NVIDIA GPU', 'gpunvidia',
    'CUDA, and the only thing that can trace a surface mesh.');
  FProcessor.Add('amd', 'AMD GPU', 'gpuamd', 'ROCm, through the HIP build.');
  FProcessor.Add('intel', 'Intel GPU', 'gpuintel', 'OpenCL.');
  FProcessor.Add('cpu', 'CPU', 'cpu',
    'No graphics card needed.  Slower by a hundredfold or so, and always ' +
    'available.');
  FProcessor.OnChange := @ChoiceChanged;

  { Rising Top before Align: alTop siblings are ordered by the Top they have
    when they are aligned, not by the order they were created in. }
  FProcessor.Panel.Top := 100;
  FDomainKind.Panel.Top := 200;
  FMedium.Panel.Top := 300;

  { What the three of them add up to, said in words underneath.  The point of
    deriving the solver is that nobody has to choose it; the point of showing
    it is that nobody has to wonder what got chosen. }
  FPlanText := TLabel.Create(gbEngine);
  FPlanText.Parent := gbEngine;
  FPlanText.Top := 400;
  FPlanText.Align := alTop;
  FPlanText.BorderSpacing.Around := McxScale96(4);
  FPlanText.BorderSpacing.Left := McxScale96(6);
  FPlanText.WordWrap := True;
end;

{ A tile was pressed.  Everything downstream of the three choices is redone
  from them: which combinations are still possible, which solver runs, and
  what the rest of the form should be showing. }
procedure TfmMain.ChoiceChanged(Sender: TObject);
var
  Was: TMcxBackend;
begin
  Was := CurrentBackend;
  FRun.SetStr('@run.processor', FProcessor.Value);
  FRun.SetStr('@run.domainkind', FDomainKind.Value);
  FRun.SetStr('@run.media', FMedium.Value);
  ApplyPlan;
  UpdateDomainCard;
  { Another simulator has other defaults, and the fields the file leaves out
    are showing the old one's.  Everything typed is already in the
    document, so reloading loses nothing. }
  if CurrentBackend <> Was then
  begin
    LoadAllBindings;
    { Another program has other devices.  Asked again now if the list is
      on screen, otherwise the next time it is. }
    FDevFor := '';
    if (FDevList <> nil) and (FSection >= 0) and
       (gbDevices.Parent = FPages[FSection]) then
      RefreshDevices;
  end;
  ApplyBindingStates;
  UpdateTitle;
  UpdateStatus;
  SchedulePreview;
end;

{ Works the solver out from the three choices, greys out the tiles that would
  not lead to one, and writes down what the run will actually be. }
procedure TfmMain.ApplyPlan;
var
  Plan: TMcxPlan;
  Exe, Forced: string;
begin
  if FProcessor = nil then Exit;

  { A surface mesh is OptiX's and OptiX is NVIDIA's, so the tile says so
    rather than failing at the Run button. }
  FDomainKind.Allow('surface', FProcessor.Value = 'nvidia',
    'A surface mesh is traced with OptiX, which needs an NVIDIA card.');
  if not FDomainKind.IsAllowed(FDomainKind.Value) then
  begin
    FDomainKind.Value := FDomainKind.FirstAllowed;
    FRun.SetStr('@run.domainkind', FDomainKind.Value);
  end;

  { A simulator picked in the Compute section runs whatever the tiles say;
    the tiles still describe the domain it runs. }
  Forced := FRun.AsStr('@run.simulator', 'auto');
  if (Forced = '') or (Forced = 'auto') then
    Plan := McxPlanFor(FProcessor.Value, FDomainKind.Value, FMedium.Value)
  else
    Plan := McxForcedPlan(Forced, FDomainKind.Value, FMedium.Value);
  FRun.SetStr('@run.backend', McxBackendName(Plan.Backend));
  FRun.SetStr('@run.compute', Plan.Compute);

  { The one place a choice reaches into the document: a low-scattering
    medium on a grid is a split-voxel run, and that is a field mcx reads. }
  if Plan.Svmc then FDoc.SetStr('Domain.MediaFormat', 'svmc')
  else if FDoc.AsStr('Domain.MediaFormat', '') = 'svmc' then
    FDoc.Delete('Domain.MediaFormat');

  Exe := McxFindExe(Plan.Backend);
  if not Plan.Ok then
    FPlanText.Caption := 'No solver for this combination.  ' + Plan.Why
  else if Exe = '' then
    FPlanText.Caption := Plan.Name + ' -- but ' +
      McxExeName(Plan.Backend) + ' was not found beside this program or on PATH.'
  else
    FPlanText.Caption := 'Runs as: ' + Plan.Name;
  if (Forced <> '') and (Forced <> 'auto') then
    FPlanText.Caption := FPlanText.Caption +
      '  (chosen under Compute > Simulator)';
  FPlanText.Font.Color := McxText;

  UpdateRunActions;
end;

{ Puts the tiles where the run settings say, after a file has been opened. }
procedure TfmMain.LoadChoosers;
begin
  if FProcessor = nil then Exit;
  FProcessor.Value := FRun.AsStr('@run.processor', 'nvidia');
  FDomainKind.Value := FRun.AsStr('@run.domainkind', 'shapes');
  FMedium.Value := FRun.AsStr('@run.media', 'diffusive');
  ApplyPlan;
end;

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
  FStepText.Left := 1000;
  FStepText.Align := alLeft;
  FStepText.Layout := tlCenter;
  FStepText.BorderSpacing.Left := 14;

  { alRight siblings are laid out in order of the Left they have when they
    are aligned, so these are numbered to fix the order: the mode switch on
    the outside, then Back, then Next, which is the one the eye should land
    on last. }
  FStepNext := TMcxFlatButton.Create(FStepBar);
  FStepNext.Parent := FStepBar;
  FStepNext.Left := 3000;
  FStepNext.Align := alRight;
  FStepNext.Width := 90;
  FStepNext.BorderSpacing.Around := 6;
  FStepNext.Caption := 'Next >';
  FStepNext.OnClick := @StepClick;

  FStepBack := TMcxFlatButton.Create(FStepBar);
  FStepBack.Parent := FStepBar;
  FStepBack.Left := 2000;
  FStepBack.Align := alRight;
  FStepBack.Width := 90;
  FStepBack.BorderSpacing.Around := 6;
  FStepBack.Caption := '< Back';
  FStepBack.OnClick := @StepClick;

  { The mode switch, on the settings page itself rather than only as an icon
    on the toolbar.  Whichever section is showing, the question "is there
    more than this?" is asked here, and the answer should be next to the
    settings rather than up among the file buttons.

    On the left, away from Back and Next: it is not a step in the wizard, and
    those two come and go with the mode -- when they are hidden and shown
    again the aligner re-sorts that side of the bar, and a button that moves
    when you are not looking is a button you have to find twice.

    Captioned with the mode it goes to, because a button says what pressing
    it does. }
  FStepMode := TMcxFlatButton.Create(FStepBar);
  FStepMode.Parent := FStepBar;
  FStepMode.Left := 0;
  FStepMode.Align := alLeft;
  FStepMode.Width := 90;
  FStepMode.BorderSpacing.Around := 6;
  FStepMode.OnClick := @acToggleModeExecute;
  { The window opens in the simple mode, so the way out of it is what the
    button offers.  Kept in step with the toolbar action from here on. }
  FStepMode.Caption := 'Expert';
  FStepMode.Hint := 'Show every setting, not only the key ones';
  FStepMode.ShowHint := True;

  { Back to the settings a blank simulation opens with.  Beside the mode
    switch because it is the other thing that acts on the whole page rather
    than on one value, and away from Back and Next, which it is not a step
    in. }
  FStepReset := TMcxFlatButton.Create(FStepBar);
  FStepReset.Parent := FStepBar;
  FStepReset.Left := 1;
  FStepReset.Align := alLeft;
  FStepReset.Width := 90;
  FStepReset.BorderSpacing.Around := 6;
  FStepReset.Caption := 'Reset';
  FStepReset.Cursor := crHandPoint;
  FStepReset.ShowHint := True;
  FStepReset.Hint := 'Put every setting back to what a new simulation ' +
    'starts with.  The file this was opened from is not touched until it is ' +
    'saved.';
  FStepReset.OnClick := @ResetSettings;
end;

procedure TfmMain.UpdateStepBar;
var
  i, Shown, At: Integer;
begin
  if FStepBar = nil then Exit;
  { The bar stays whatever the mode is, because the mode switch lives on it:
    hiding the bar in expert mode took the only way back with it.  What goes
    is the wizard's own part of it -- the step count and Back and Next, which
    mean nothing when every section is showing at once. }
  FStepText.Visible := FWizard;
  FStepBack.Visible := FWizard;
  FStepNext.Visible := FWizard;
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
  if (i >= 0) and (i <= High(FPanes)) then
  begin
    SelectSection(i);
    { Back and Next are the navigator by another route, so they bring the
      settings forward the same way a click on a section does. }
    ShowSettingsPage;
  end;
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
    if FStepMode <> nil then
    begin
      FStepMode.Caption := 'Expert';
      FStepMode.Hint := 'Show every setting, not only the key ones';
      FStepMode.ShowHint := True;
    end;
  end
  else
  begin
    acToggleMode.Caption := 'Expert';
    acToggleMode.ImageIndex := McxIconIndex('expert');
    acToggleMode.Hint := 'Showing every setting. Click for key settings only.';
    if FStepMode <> nil then
    begin
      FStepMode.Caption := 'Simple';
      FStepMode.Hint := 'Show only the settings a simulation usually needs';
      FStepMode.ShowHint := True;
    end;
  end;
  UpdateStatus;
end;

{ ------------------------------------------------------------ documents --- }

procedure TfmMain.NewDocument;
begin
  SeedDocument(FDoc);
  FRun.Clear;
  FRun.SetStr('@run.processor', 'nvidia');
  FRun.SetStr('@run.domainkind', 'shapes');
  FRun.SetStr('@run.media', 'diffusive');
  FRun.SetNum('@run.maxjumpdebug', DefaultMaxJump);
  FRun.Modified := False;
  LoadChoosers;
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
  FView.ForgetOutput;
  if FDisplay <> nil then
    { An empty range hides the photon row, and the rest follows the view. }
    FDisplay.SetTrajectory(1, 0);
end;

procedure TfmMain.GuessRunSettings;
begin
  { Seeded on every open, because it is a run setting and a document does not
    carry one; without it the first run after opening a file would take mcx's
    ten-million default. }
  if FRun.AsInt('@run.maxjumpdebug') <= 0 then
    FRun.SetNum('@run.maxjumpdebug', DefaultMaxJump);

  { How the domain is described is written all over the file, so it is read
    off rather than asked.  A Mesh block is MMC's and nothing else's; a
    VolumeFile is a grid of voxels; anything else is shapes to rasterise.

    Tetrahedral rather than surface, always: both are stored as nodes and
    elements, and the file does not say which kind of tracing was meant.  It
    is one tile away if the answer is the other one.

    What to run it on is not read off, because it is not in the file at all
    -- it is a property of this machine, and keeping the last answer is
    better than guessing a new one. }
  if (FDoc.Find('Mesh') <> nil) or (FDoc.Find('Shapes.MeshNode') <> nil) then
    FRun.SetStr('@run.domainkind', 'mesh')
  else if (FDoc.AsStr('Domain.VolumeFile', '') <> '') or
          ((FDoc.Find('Shapes') is TJSONObject) and
           (TJSONObject(FDoc.Find('Shapes')).Find('_ArrayType_') <> nil)) then
    { A volume written into Shapes as a JData array is a voxel domain as
      much as one in a file -- every input mcxlab exports is one. }
    FRun.SetStr('@run.domainkind', 'voxel')
  else
    FRun.SetStr('@run.domainkind', 'shapes');

  { Split-voxel is how a low-scattering medium is run on a grid, so a file
    that asks for it is telling us which kind of medium it holds. }
  if FDoc.AsStr('Domain.MediaFormat', '') = 'svmc' then
    FRun.SetStr('@run.media', 'lowscatter')
  else
    FRun.SetStr('@run.media', 'diffusive');

  FRun.Modified := False;
  LoadChoosers;
end;

{ A session of a first-Studio project, converted to the mcx input it stood
  for (mcxmcxp).  With several sessions the person picks one; a headless run
  takes the first.  The document is given the name <session>.json beside
  the project, unsaved, so a run writes it there -- where its relative paths
  resolve -- unless that would overwrite a file, when Save As asks. }
function TfmMain.OpenProject(const AFileName: string;
  out ACancelled: Boolean): Boolean;
var
  Names, Warn: TStringList;
  k: Integer;
  Json, Target: string;
begin
  Result := False;
  ACancelled := False;
  Names := TStringList.Create;
  Warn := TStringList.Create;
  try
    if not McxProjectSessions(AFileName, Names) then Exit;
    k := 0;
    if (Names.Count > 1) and not Unattended then
    begin
      k := InputCombo('MCX Studio', ExtractFileName(AFileName) +
        ' holds several sessions. Open which?', Names);
      if k < 0 then
      begin
        ACancelled := True;
        Exit;
      end;
    end;
    Json := McxProjectSession(AFileName, Names[k], Warn);
    if (Json = '') or not FDoc.LoadFromString(Json) then Exit;
    Target := ExtractFilePath(ExpandFileName(AFileName)) + Names[k] + '.json';
    if FileExists(Target) then FDoc.FileName := '' else FDoc.FileName := Target;
    FDoc.Modified := True;
    Log(Format('%s: session %s of %d, converted from the first Studio''s project',
      [ExtractFileName(AFileName), Names[k], Names.Count]));
    for k := 0 to Warn.Count - 1 do Log('  ' + Warn[k]);
    Result := True;
  finally
    Warn.Free;
    Names.Free;
  end;
end;

procedure TfmMain.OpenDocument(const AFileName: string);
var
  Ok, Cancelled: Boolean;
begin
  if LowerCase(ExtractFileExt(AFileName)) = '.mcxp' then
  begin
    Ok := OpenProject(AFileName, Cancelled);
    if Cancelled then Exit;
  end
  else
    Ok := FDoc.LoadFromFile(AFileName);
  if not Ok then
  begin
    if Unattended then
    begin
      WriteLn(StdErr, 'Could not read ', AFileName, ': ', FDoc.LastError);
      Halt(2);
    end;
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

{ True when the document cannot be run where it is: never saved, or saved
  in a folder this user cannot write -- an example under the install folder
  is the usual one, and mcx writes its output beside its input. }
function TfmMain.NeedsWorkFolder: Boolean;
begin
  Result := (FDoc.FileName = '') or
    not DirectoryIsWritable(ExtractFilePath(ExpandFileName(FDoc.FileName)));
end;

{ After the document has moved from AOld to ANew, the files it names by a
  relative path are still in AOld.  A VolumeFile is pointed at where it is;
  mmc's mesh files -- node_<id>.dat and the rest, found beside the input --
  are copied, because MeshID is a tag with a prefix put in front of it and
  cannot carry a folder. }
procedure TfmMain.RebaseFiles(const AOld, ANew: string);
const
  MeshParts: array[0..6] of string =
    ('node_', 'elem_', 'face_', 'facenb_', 'velem_', 'evol_', 'prop_');
var
  Vf, Id: string;
  i: Integer;
begin
  if (AOld = '') or SameFileName(IncludeTrailingPathDelimiter(AOld),
     IncludeTrailingPathDelimiter(ANew)) then Exit;
  Vf := FDoc.AsStr('Domain.VolumeFile', '');
  if (Vf <> '') and not FilenameIsAbsolute(Vf) and
     FileExists(IncludeTrailingPathDelimiter(AOld) + Vf) then
    FDoc.SetStr('Domain.VolumeFile',
      ExpandFileName(IncludeTrailingPathDelimiter(AOld) + Vf));
  Id := FDoc.AsStr(MeshPath('Mesh.MeshID'), '');
  if Id <> '' then
    for i := 0 to High(MeshParts) do
      if FileExists(IncludeTrailingPathDelimiter(AOld) + MeshParts[i] + Id + '.dat') then
        CopyFile(IncludeTrailingPathDelimiter(AOld) + MeshParts[i] + Id + '.dat',
          IncludeTrailingPathDelimiter(ANew) + MeshParts[i] + Id + '.dat');
end;

{ Saves the document to AFileName, carrying along what it refers to when
  that is a different folder from the one it came from. }
function TfmMain.SaveDocTo(const AFileName: string): Boolean;
var
  Old: string;
begin
  Old := '';
  if FDoc.FileName <> '' then Old := ExtractFilePath(ExpandFileName(FDoc.FileName));
  RebaseFiles(Old, ExtractFilePath(ExpandFileName(AFileName)));
  Result := FDoc.SaveToFile(AFileName);
  if not Result then
    MessageDlg('MCX Studio', 'Could not write the file:'#10#10 + FDoc.LastError,
      mtError, [mbOK], 0);
  UpdateTitle;
end;

function TfmMain.SaveAs: Boolean;
var
  Session: string;
begin
  Result := False;
  Session := FDoc.AsStr('Session.ID', 'mcx');
  { Offered the session folder when there is nowhere better: a new document,
    or one in a folder that cannot be written. }
  if NeedsWorkFolder then
  begin
    dlgSave.InitialDir := McxWorkFolder(CurrentBackend, Session);
    dlgSave.FileName := IncludeTrailingPathDelimiter(dlgSave.InitialDir) +
      Session + '.json';
  end
  else
    dlgSave.FileName := FDoc.FileName;
  if not dlgSave.Execute then Exit;
  Result := SaveDocTo(dlgSave.FileName);
end;

procedure TfmMain.acSaveExecute(Sender: TObject);
begin
  { A file that cannot be written where it is gets Save As, offered the
    session folder, instead of an error. }
  if NeedsWorkFolder then
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
    ('mcx', 'mcxcl', 'mmc', 'mcx-hip', 'vkmmc', 'umcx');
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
  if FView <> nil then Result := Result + '3-D view:  ' + FView.Description + #10;
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
  Body, Title, HelpCtl: string;
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
      HelpCtl := C.Name;
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
    McxShowHelp(Self, 'MCX Studio help', McxHelpOverview, McxManualURL(''));
    Exit;
  end;
  McxShowHelp(Self, Title, Title + LineEnding + LineEnding + Body,
    McxManualURL(HelpCtl));
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
          { where it was raised: with the debug build's line info this names
            the binding or parser that took the bad value }
          DumpExceptionBackTrace(Output);
          Inc(Result);
        end;
      end;
      Flush(Output);
    end;
    WriteLn(Format('  %d files driven through every binding', [Checked]));
    Inc(Result, SelfTestEditors);
  finally
    Before.Free;
    Files.Free;
  end;
end;

{ The code-built editors, driven the way a click would drive them, with the
  document checked after each step.  The corpus pass above proves they leave
  a file alone; this proves they change it the way mcx and mmc expect. }
function TfmMain.SelfTestEditors: Integer;
var
  Fails: Integer;
  Url: string;
  Vol: TMcxArray;
  Vi: Integer;

  procedure Expect(const AWhat, AGot, AWant: string);
  begin
    if AGot = AWant then Exit;
    WriteLn('  FAIL  ', AWhat, ': got ', AGot, ', want ', AWant);
    Inc(Fails);
  end;

  function Js(const APath: string): string;
  var
    D: TJSONData;
  begin
    D := FDoc.Find(APath);
    if D = nil then Result := '(absent)' else Result := D.AsJSON;
  end;

  function BindIndex(const ACtl: string): Integer;
  begin
    for Result := 0 to High(Binds) do
      if Binds[Result].Ctl = ACtl then Exit;
    Result := -1;
  end;

  { The value at row-major (x, y, z) of the volume in Shapes. }
  function VoxelAt(x, y, z: Integer): string;
  var
    A: TMcxArray;
  begin
    Result := '?';
    if not (FDoc.Find('Shapes') is TJSONObject) then Exit;
    if not McxDecodeJData(TJSONObject(FDoc.Find('Shapes')), A) then Exit;
    Result := FloatToStr(McxArrayValue(A, (Int64(x) * A.Dims[1] + y) * A.Dims[2] + z));
  end;

  procedure DomainTests;
  var
    E, Tmp, Txt, Mm: string;
    F: TFileStream;
    B: array[0..23] of Byte;
    x, y, z: Integer;
    Ok: Boolean;
  begin
    FDoc.LoadFromString('{"Session":{"ID":"vol"},"Forward":{"T0":0,"T1":5e-9,"Dt":5e-9},' +
      '"Domain":{"Dim":[60,60,60],"MediaFormat":"short","VolumeFile":"x.bin",' +
      '"Media":[{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.005,"mus":1,"g":0.01,"n":1.37}]},' +
      '"Optode":{"Source":{"Pos":[1,1,0],"Dir":[0,0,1]}}}');
    FRun.SetStr('@run.domainkind', 'voxel');
    LoadAllBindings;
    Ok := ApplyVolumeText('[[[1,2],[3,4]],[[5,6],[7,8]]]', E);
    Expect('nested volume pasted', BoolToStr(Ok, True) + ' ' + E, 'True ');
    Expect('nested volume type', Js('Shapes._ArrayType_'), '"uint8"');
    Expect('nested volume size', Js('Shapes._ArraySize_'), '[2, 2, 2]');
    Expect('Dim follows the volume', Js('Domain.Dim'), '[2, 2, 2]');
    Expect('MediaFormat follows the type', Js('Domain.MediaFormat'), '"byte"');
    Expect('VolumeFile dropped', Js('Domain.VolumeFile'), '(absent)');
    Expect('row-major order kept', VoxelAt(1, 0, 1), '6');
    Expect('card says embedded', BoolToStr(Pos('embedded', FDomInfo.Caption) > 0, True), 'True');

    { mcx's raw volume is x fastest; value x + 2y + 6z at each voxel. }
    for z := 0 to 3 do
      for y := 0 to 2 do
        for x := 0 to 1 do
          B[x + 2 * y + 6 * z] := x + 2 * y + 6 * z;
    Tmp := IncludeTrailingPathDelimiter(GetTempDir) + 'mcxstudio_selftest.bin';
    F := TFileStream.Create(Tmp, fmCreate);
    try
      F.WriteBuffer(B, SizeOf(B));
    finally
      F.Free;
    end;
    FDoc.SetInt('Domain.Dim[0]', 2);
    FDoc.SetInt('Domain.Dim[1]', 3);
    FDoc.SetInt('Domain.Dim[2]', 4);
    Ok := LoadVolumeFile(Tmp, E);
    DeleteFile(Tmp);
    Expect('raw volume loaded', BoolToStr(Ok, True) + ' ' + E, 'True ');
    Expect('raw volume size', Js('Shapes._ArraySize_'), '[2, 3, 4]');
    Expect('raw volume voxel (1,2,3)', VoxelAt(1, 2, 3), IntToStr(1 + 4 + 18));
    Expect('raw volume voxel (0,1,2)', VoxelAt(0, 1, 2), IntToStr(2 + 12));

    { The same values pasted as a column-major JData array. }
    Txt := '{"_ArrayType_":"uint8","_ArraySize_":[2,3,4],"_ArrayOrder_":"c",' +
      '"_ArrayData_":[';
    for x := 0 to 23 do
    begin
      if x > 0 then Txt := Txt + ',';
      Txt := Txt + IntToStr(x);
    end;
    Txt := Txt + ']}';
    Ok := ApplyVolumeText(Txt, E);
    Expect('column-major JData pasted', BoolToStr(Ok, True) + ' ' + E, 'True ');
    Expect('column-major turned row-major', VoxelAt(1, 2, 3), IntToStr(1 + 4 + 18));
    Ok := ApplyVolumeText('{"not":"a volume"}', E);
    Expect('a non-volume is refused', BoolToStr(Ok, True), 'False');
    { A real one: 20 cubed, every voxel label 1. }
    Txt := '[';
    for x := 0 to 19 do
    begin
      if x > 0 then Txt := Txt + ',';
      Txt := Txt + '[';
      for y := 0 to 19 do
      begin
        if y > 0 then Txt := Txt + ',';
        Txt := Txt + '[';
        for z := 0 to 19 do
        begin
          if z > 0 then Txt := Txt + ',';
          Txt := Txt + '1';
        end;
        Txt := Txt + ']';
      end;
      Txt := Txt + ']';
    end;
    Txt := Txt + ']';
    FDoc.SetInt('Optode.Source.Pos[0]', 10);
    FDoc.SetInt('Optode.Source.Pos[1]', 10);
    Ok := ApplyVolumeText(Txt, E);
    Expect('20-cube volume pasted', Js('Domain.Dim') + ' ' + VoxelAt(19, 19, 19),
      '[20, 20, 20] 1');

    { A mesh, pasted as plain rows. }
    FDoc.LoadFromString('{"Session":{"ID":"mesh"},"Mesh":{"MeshID":"cube"},' +
      '"Optode":{"Source":{"Pos":[0.1,0.1,0.1],"Dir":[0,0,1]}}}');
    FRun.SetStr('@run.domainkind', 'mesh');
    LoadAllBindings;
    Ok := ApplyMeshText('{"MeshNode":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],' +
      '"MeshElem":[[1,2,3,4]]}', E);
    Expect('mesh pasted', BoolToStr(Ok, True) + ' ' + E, 'True ');
    Expect('MeshNode as single', Js('Mesh.MeshNode._ArrayType_') + ' ' +
      Js('Mesh.MeshNode._ArraySize_'), '"single" [4, 3]');
    Expect('MeshElem as int32 with a label', Js('Mesh.MeshElem._ArrayType_') + ' ' +
      Js('Mesh.MeshElem._ArraySize_'), '"int32" [1, 5]');
    Expect('MeshID dropped', Js('Mesh.MeshID'), '(absent)');
    Expect('InitElem given', Js('Mesh.InitElem'), '-1');
    Expect('card says embedded mesh', BoolToStr(Pos('4 nodes, 1 elements',
      FDomInfo.Caption) > 0, True), 'True');

    { mmc's own .dat pair, where the source tree has one. }
    Mm := ExpandFileName(McxProgramDir +
      '../../../mmc/examples/meshtest/node_mesh1.dat');
    if FileExists(Mm) then
    begin
      Ok := LoadMeshFile(Mm, E);
      Expect('mmc .dat mesh loaded', BoolToStr(Ok, True) + ' ' + E, 'True ');
      Expect('prop_<id>.dat becomes Domain.Media',
        BoolToStr(FDoc.Find('Domain.Media') is TJSONArray, True), 'True');
      with TStringList.Create do
      try
        LoadFromFile(Mm);
        { a header line, then a node a line }
        Expect('mmc .dat node count',
          IntToStr(McxMeshRows(FDoc.Find('Mesh.MeshNode'))), IntToStr(Count - 1));
      finally
        Free;
      end;
    end;
  end;

var
  i: Integer;
  WasWizard: Boolean;
  WasDomain, WasBackend: string;
begin
  Fails := 0;
  { Expert, because only a field that is showing has its enabled state kept. }
  WasWizard := FWizard;
  FWizard := False;
  WasDomain := FRun.AsStr('@run.domainkind', 'shapes');
  FRun.SetStr('@run.domainkind', 'mesh');
  FDoc.LoadFromString('{"Session":{"ID":"t"},"Optode":{"Source":{"Type":' +
    '"pencil","Pos":[1,2,3],"Dir":[0,0,1]},"Detector":[{"Pos":[5,5,0],"R":1}]},' +
    '"Mesh":{"MeshID":"cube","InitElem":1}}');
  LoadAllBindings;

  { Sources: one, then two -- every vector the file has becomes rows of the
    same length, and one it does not have stays absent. }
  Expect('sources at start', IntToStr(FSources.Count), '1');
  FSources.AddRow;
  Expect('Pos after add', Js('Optode.Source.Pos'), '[[1, 2, 3], [1, 2, 3]]');
  Expect('Dir after add', Js('Optode.Source.Dir'), '[[0, 0, 1], [0, 0, 1]]');
  Expect('Param1 after add', Js('Optode.Source.Param1'), '(absent)');
  { A cell of the second source: the vector fields follow the pick. }
  FSources.SetCell(1, 0, '7');
  Expect('Pos after edit', Js('Optode.Source.Pos'), '[[1, 2, 3], [7, 2, 3]]');
  FSources.SetCell(1, 11, '0.5');
  Expect('Param1 typed into', Js('Optode.Source.Param1'),
    '[[0, 0, 0, 0], [0, 0, 0, 0.5]]');
  FSources.SetCell(0, 3, '2');
  Expect('Pos w padded', Js('Optode.Source.Pos[0]'), '[1, 2, 3, 2]');
  FSources.Selected := 1;
  FSource := 1;
  LoadAllBindings;
  Expect('edSrcPos shows source 2', edSrcPos.Text, '7, 2, 3');
  FSources.DeleteRow(0);
  Expect('Pos back to flat', Js('Optode.Source.Pos'), '[7, 2, 3]');
  Expect('Param1 back to flat', Js('Optode.Source.Param1'), '[0, 0, 0, 0.5]');
  FSources.DeleteRow(0);
  Expect('last source kept', IntToStr(FSources.Count), '1');

  { Detectors: Dir is optional, and a typed component pads with zeros. }
  FDetectors.SetCell(0, 6, '1');
  Expect('detector Dir padded', Js('Optode.Detector[0].Dir'), '[0, 0, 1]');
  FDetectors.AddRow;
  Expect('new detector copies Dir', Js('Optode.Detector[1].Dir'), '[0, 0, 1]');
  FDoc.Delete('Optode.Detector[0].Dir');
  FDoc.Delete('Optode.Detector[1]');
  FDetectors.Reload;
  FDetectors.AddRow;
  Expect('no Dir copied from a row without one',
    Js('Optode.Detector[1]'), '{ "Pos" : [5, 5, 0], "R" : 1 }');

  { BasisOrder is a number in the file, RayTracer a letter. }
  i := BindIndex('cbBasisOrder');
  cbBasisOrder.ItemIndex := FValues[i].IndexOf('0');
  SaveBinding(i);
  Expect('BasisOrder', Js('Session.BasisOrder'), '0');
  i := BindIndex('cbRayTracer');
  cbRayTracer.ItemIndex := FValues[i].IndexOf('g');
  SaveBinding(i);
  Expect('RayTracer', Js('Session.RayTracer'), '"g"');
  Expect('MeshID shown', edMeshID.Text, 'cube');

  { The wide-field detector fields wait for a detector. }
  ApplyBindingStates;
  Expect('DetParam1 enabled with a detector',
    BoolToStr(edDetParam1.Enabled, True), 'True');
  FDoc.LoadFromString('{"Optode":{"Source":{"Pos":[0,0,0],"Dir":[0,0,1]}}}');
  LoadAllBindings;
  Expect('DetParam1 disabled without one',
    BoolToStr(edDetParam1.Enabled, True), 'False');

  { A new shape takes the label after the highest in use, and the media
    table grows a copy of its last row for every label it lacks. }
  FDoc.LoadFromString('{"Domain":{"Dim":[60,60,60],"Media":[' +
    '{"mua":0,"mus":0,"g":1,"n":1},{"mua":0.005,"mus":1,"g":0.01,"n":1.37}]},' +
    '"Shapes":[{"Grid":{"Size":[60,60,60],"Tag":1}}],' +
    '"Optode":{"Source":{"Pos":[30,30,0],"Dir":[0,0,1]}}}');
  LoadAllBindings;
  FMedia.Reload;
  FShapes.Reload;
  FShapes.AddShape('Sphere');
  Expect('new sphere takes the next label', Js('Shapes[1].Sphere.Tag'), '2');
  Expect('media grows to the new label', IntToStr(FDoc.Find('Domain.Media').Count), '3');
  Expect('new medium copies the last', Js('Domain.Media[2]'), Js('Domain.Media[1]'));
  FShapes.AddShape('ZLayers');
  Expect('new layer row takes the next label', Js('Shapes[2].ZLayers[0][2]'), '3');
  Expect('media grows for a layer', IntToStr(FDoc.Find('Domain.Media').Count), '4');
  FShapes.AddShape('Origin');
  Expect('Origin adds no medium', IntToStr(FDoc.Find('Domain.Media').Count), '4');
  FDoc.SetNum('Domain.Media[6].mua', 0.01);
  FShapes.AddShape('Box');
  Expect('box after the layers', Js('Shapes[4].Box.Tag'), '4');
  Expect('enough media already: none added', IntToStr(FDoc.Find('Domain.Media').Count), '7');

  { The 3-D view switches between a run's output and the input domain, and
    back, without the output being read again. }
  Vol.Kind := akSingle;
  SetLength(Vol.Dims, 3);
  Vol.Dims[0] := 2; Vol.Dims[1] := 2; Vol.Dims[2] := 2;
  SetLength(Vol.Data, 8 * SizeOf(Single));
  for Vi := 0 to 7 do PSingle(@Vol.Data[Vi * SizeOf(Single)])^ := Vi + 1;
  if FView.ShowVolume(Vol) then
  begin
    FDisplay.Sync;
    Expect('a result is output', BoolToStr(FView.HasOutput, True), 'True');
    FView.ShowInput := True;
    Expect('input view hides the output', BoolToStr(FView.HasVolume, True), 'False');
    FView.ShowInput := False;
    Expect('output view puts it back', BoolToStr(FView.HasVolume, True), 'True');
    DropResult;
    Expect('a new document has no output', BoolToStr(FView.HasOutput, True), 'False');
  end
  else
    WriteLn('  skip  output/input switch: the view took no volume');

  { A mesh block named Shapes, as mmc --dumpjson writes it, is edited where
    it is rather than shadowed by a new Mesh. }
  FDoc.LoadFromString('{"Optode":{"Source":{"Pos":[0,0,0],"Dir":[0,0,1]}},' +
    '"Shapes":{"MeshNode":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],' +
    '"MeshElem":[[1,2,3,4,1]],"InitElem":1}}');
  LoadAllBindings;
  Expect('InitElem read from Shapes', IntToStr(edInitElem.Value), '1');
  edInitElem.Value := 3;
  SaveBinding(BindIndex('edInitElem'));
  Expect('InitElem written to Shapes', Js('Shapes.InitElem'), '3');
  Expect('no Mesh block made', Js('Mesh'), '(absent)');
  Expect('embedded mesh summary', lbMeshInfo.Caption,
    'The mesh is embedded in this file: nodes 4, elements 1.');

  { Absent keys show the chosen simulator's own defaults, and saving an
    untouched form writes none of them. }
  WasBackend := FRun.AsStr('@run.backend', 'mcx');
  FDoc.LoadFromString('{"Optode":{"Source":{"Pos":[0,0,0],"Dir":[0,0,1]}}}');
  FRun.SetStr('@run.backend', 'mcx');
  LoadAllBindings;
  Expect('mcx DoPartialPath default', BoolToStr(ckSaveDetp.Checked, True), 'True');
  Expect('mcx OutputFormat default',
    FValues[BindIndex('rgOutFormat')][rgOutFormat.ItemIndex], 'jnii');
  Expect('mcx MinEnergy default', edMinEnergy.Text, '0');
  Expect('mcx thread count follows DoAutoThread',
    BoolToStr(edThread.Enabled, True), 'False');
  Expect('mcx DoSaveExit live with DoPartialPath on',
    BoolToStr(ckSaveExit.Enabled, True), 'True');
  FRun.SetStr('@run.backend', 'mcxcl');
  LoadAllBindings;
  Expect('mcxcl T1 default', edT1.Text, '5e-09');
  FRun.SetStr('@run.backend', 'mmc');
  LoadAllBindings;
  Expect('mmc DoPartialPath default', BoolToStr(ckSaveDetp.Checked, True), 'False');
  Expect('mmc OutputFormat default',
    FValues[BindIndex('rgOutFormat')][rgOutFormat.ItemIndex], 'ascii');
  Expect('mmc MinEnergy default', edMinEnergy.Text, '1e-06');
  Expect('mmc RayTracer default',
    FValues[BindIndex('cbRayTracer')][cbRayTracer.ItemIndex], 'g');
  Inc(FLoading);
  try
    for i := 0 to High(Binds) do SaveBinding(i);
  finally
    Dec(FLoading);
  end;
  Expect('defaults not written', FDoc.ToJSON(False),
    '{ "Optode" : { "Source" : { "Pos" : [0, 0, 0], "Dir" : [0, 0, 1] } } }');
  FRun.SetStr('@run.backend', WasBackend);

  { The help window's manual button: an option goes to its own entry of
    mcx_help.html, a setting with no option to the JSON input section, and
    mmc's own options, which that page does not describe, get no button. }
  Url := McxManualURL('edPhotons');
  Expect('manual link of an option', Copy(Url, Length(Url) - 6, 7), '#photon');
  Url := McxManualURL('edT0');
  Expect('manual link of an input-file setting',
    Copy(Url, Pos('#', Url), MaxInt), '#JSON_input_file_format');
  Expect('no manual link for an mmc option', McxManualURL('cbRayTracer'), '');

  { A domain written as a JData array goes on the 3-D card as a volume, and
    comes off again with the document that carried it.  Under a virtual X
    server the view may not have painted yet, and has nothing to draw into. }
  if (FView <> nil) and FView.Ready then
  begin
    FDoc.LoadFromString('{"Domain":{"Dim":[2,2,2]},"Shapes":{' +
      '"_ArrayType_":"uint8","_ArraySize_":[2,2,2],' +
      '"_ArrayData_":[1,1,1,1,1,1,1,2]}}');
    FView.Rebuild;
    Expect('JData domain previewed', BoolToStr(FView.HasVolume, True), 'True');
    FDoc.LoadFromString('{"Domain":{"Dim":[2,2,2]},"Shapes":[]}');
    FView.Rebuild;
    Expect('JData domain cleared', BoolToStr(FView.HasVolume, True), 'False');
  end;

  { The domain card's data: a volume pasted as a nested list goes into
    Shapes as row-major JData with Dim and MediaFormat to match; a raw .bin
    and a column-major JData array come out in the same row-major order; a
    pasted mesh becomes mmc's single/int32 annotated arrays. }
  DomainTests;

  { And none of it shows for a grid, whatever the file holds. }
  FRun.SetStr('@run.domainkind', 'voxel');
  ApplyBindingStates;
  Expect('mesh card hidden for a voxel grid',
    BoolToStr(gbMesh.Visible, True), 'False');
  Expect('detector card kept for a voxel grid',
    BoolToStr(gbDetector.Visible, True), 'True');

  FRun.SetStr('@run.domainkind', WasDomain);
  FWizard := WasWizard;
  ApplyBindingStates;
  WriteLn(Format('  code-built editors: %d failure(s)', [Fails]));
  Result := Fails;
end;

{ ------------------------------------------------------ domain data ---- }

{ The card under Engine and domain.  Built here, not in the designer: what it
  says and what its buttons do follow the domain tile picked above it. }
procedure TfmMain.BuildDomainCard;
var
  Bar: TPanel;
begin
  FDomInfo := TLabel.Create(gbDomainData);
  FDomInfo.Parent := gbDomainData;
  FDomInfo.Top := 1000;
  FDomInfo.Align := alTop;
  FDomInfo.WordWrap := True;
  FDomInfo.BorderSpacing.Bottom := 6;
  FDomInfo.Caption := '';

  Bar := TPanel.Create(gbDomainData);
  Bar.Parent := gbDomainData;
  Bar.Top := 1001;
  Bar.Align := alTop;
  Bar.Height := 34;
  Bar.BevelOuter := bvNone;
  Bar.Caption := '';

  FDomEdit := TButton.Create(Bar);
  FDomEdit.Parent := Bar;
  FDomEdit.Align := alLeft;
  FDomEdit.BorderSpacing.Right := 8;
  FDomEdit.Width := 150;
  FDomEdit.OnClick := @DomEditClick;

  FDomLoad := TButton.Create(Bar);
  FDomLoad.Parent := Bar;
  FDomLoad.Left := 200;
  FDomLoad.Align := alLeft;
  FDomLoad.Width := 150;
  FDomLoad.Caption := 'Load from file...';
  FDomLoad.OnClick := @DomLoadClick;
end;

function TfmMain.DomainIsMesh: Boolean;
var
  K: string;
begin
  K := FRun.AsStr('@run.domainkind', 'shapes');
  Result := (K = 'mesh') or (K = 'surface');
end;

{ The embedded volume, if Shapes is one. }
function EmbeddedVolume(ADoc: TMcxDoc): TJSONObject;
var
  D: TJSONData;
begin
  Result := nil;
  D := ADoc.Find('Shapes');
  if (D is TJSONObject) and (TJSONObject(D).Find('_ArrayType_') <> nil) then
    Result := TJSONObject(D);
end;

{ A path the document names, as the run will find it: relative ones beside
  the document, where mcx is started. }
function DocRelative(ADoc: TMcxDoc; const APath: string): string;
begin
  Result := APath;
  if (APath <> '') and not FilenameIsAbsolute(APath) and (ADoc.FileName <> '') then
    Result := ExtractFilePath(ExpandFileName(ADoc.FileName)) + APath;
end;

procedure TfmMain.UpdateDomainCard;
var
  K, Vf, Id, S: string;
  V: TJSONObject;
  nn, ne: Integer;
  Sh: TJSONData;
begin
  if FDomInfo = nil then Exit;
  K := FRun.AsStr('@run.domainkind', 'shapes');
  FDomLoad.Visible := K <> 'shapes';
  if K = 'shapes' then
  begin
    FDomEdit.Caption := 'Edit Shapes';
    Sh := FDoc.Find('Shapes');
    if Sh is TJSONArray then
      FDomInfo.Caption := Format('The domain is built from %d shape ' +
        'command(s) in this file.', [Sh.Count])
    else
      FDomInfo.Caption := 'The domain is built from shape commands; this ' +
        'file has none yet.';
  end
  else if DomainIsMesh then
  begin
    FDomEdit.Caption := 'Edit Mesh...';
    nn := McxMeshRows(FDoc.Find(MeshPath('Mesh.MeshNode')));
    ne := McxMeshRows(FDoc.Find(MeshPath('Mesh.MeshElem')));
    Id := FDoc.AsStr(MeshPath('Mesh.MeshID'), '');
    if (nn >= 0) and (ne >= 0) then
      FDomInfo.Caption := Format('The mesh is embedded in this file: %d ' +
        'nodes, %d elements.', [nn, ne])
    else if Id <> '' then
    begin
      S := DocRelative(FDoc, 'node_' + Id + '.dat');
      if FileExists(S) then
        FDomInfo.Caption := Format('The mesh is read from node_%s.dat and ' +
          'elem_%s.dat beside this file.', [Id, Id])
      else
        FDomInfo.Caption := Format('MeshID is %s, but node_%s.dat is not ' +
          'beside this file.', [Id, Id]);
    end
    else
      FDomInfo.Caption := 'No mesh yet. Load one from a file -- mmc''s ' +
        'node_/elem_ .dat files or a .jmsh -- or paste one with Edit Mesh.';
  end
  else
  begin
    FDomEdit.Caption := 'Edit Volume...';
    V := EmbeddedVolume(FDoc);
    Vf := FDoc.AsStr('Domain.VolumeFile', '');
    if V <> nil then
      FDomInfo.Caption := 'The volume is embedded in this file: ' +
        McxVolumeSummary(V) + '.'
    else if Vf <> '' then
    begin
      if FileExists(DocRelative(FDoc, Vf)) then
        FDomInfo.Caption := Format('The volume is read from %s (%d x %d x %d, %s).',
          [Vf, FDoc.AsInt('Domain.Dim[0]', 0), FDoc.AsInt('Domain.Dim[1]', 0),
           FDoc.AsInt('Domain.Dim[2]', 0), FDoc.AsStr('Domain.MediaFormat', 'byte')])
      else
        FDomInfo.Caption := Format('The volume file %s is not there.', [Vf]);
    end
    else
      FDomInfo.Caption := 'No volume yet. Load one from a file -- .bin, ' +
        '.nii, .jnii, .bnii or a JData .json -- or paste one with Edit Volume.';
  end;
end;

{ The byte width of a raw volume's labels, from MediaFormat. }
function TfmMain.RawWidth: Integer;
var
  F: string;
begin
  F := LowerCase(FDoc.AsStr('Domain.MediaFormat', 'byte'));
  if F = 'short' then Result := 2
  else if F = 'integer' then Result := 4
  else if F = 'byte' then Result := 1
  else Result := 0;
end;

{ What Edit Volume opens with: the embedded array, or the volume file's
  contents as the array they would become, or nothing. }
function TfmMain.VolumeText: string;
var
  V, O: TJSONObject;
  E, Vf: string;
begin
  Result := '';
  V := EmbeddedVolume(FDoc);
  if V <> nil then Exit(V.FormatJSON);
  Vf := DocRelative(FDoc, FDoc.AsStr('Domain.VolumeFile', ''));
  if (Vf <> '') and FileExists(Vf) and
     McxVolumeFromFile(Vf, RawWidth, FDoc.AsInt('Domain.Dim[0]', 0),
       FDoc.AsInt('Domain.Dim[1]', 0), FDoc.AsInt('Domain.Dim[2]', 0), O, E) then
  try
    Result := O.FormatJSON;
  finally
    O.Free;
  end;
end;

function TfmMain.MeshText: string;
var
  N, E, M: TJSONData;
  O: TJSONObject;
  Err, Id, F: string;
begin
  Result := '';
  N := FDoc.Find(MeshPath('Mesh.MeshNode'));
  E := FDoc.Find(MeshPath('Mesh.MeshElem'));
  if (N <> nil) and (E <> nil) then
  begin
    O := TJSONObject.Create;
    try
      O.Add('MeshNode', N.Clone);
      O.Add('MeshElem', E.Clone);
      Result := O.FormatJSON;
    finally
      O.Free;
    end;
    Exit;
  end;
  Id := FDoc.AsStr(MeshPath('Mesh.MeshID'), '');
  F := DocRelative(FDoc, 'node_' + Id + '.dat');
  if (Id <> '') and FileExists(F) and McxMeshFromFile(F, N, E, M, Err) then
  begin
    O := TJSONObject.Create;
    try
      O.Add('MeshNode', N);
      O.Add('MeshElem', E);
      { Only when this file has no media of its own, as mmc would read it. }
      if (M <> nil) and not ((FDoc.Find('Domain.Media') is TJSONArray) and
         (FDoc.Find('Domain.Media').Count > 0)) then
        O.Add('Media', M)
      else
        M.Free;
      Result := O.FormatJSON;
    finally
      O.Free;
    end;
  end;
end;

{ After the domain data changed: everything that reads it catches up. }
procedure TfmMain.DomainChanged;
begin
  FDoc.Modified := True;
  LoadAllBindings;
  UpdateTitle;
  SchedulePreview;
  RefreshPreview;
end;

{ A volume goes into Shapes, row-major JData, and is the only volume the
  file has: a VolumeFile left beside it would be read first and the array
  ignored.  Dim and MediaFormat are made to agree with it. }
procedure TfmMain.StoreVolume(AObj: TJSONObject);
var
  nx, ny, nz: Integer;
  Fmt: string;
begin
  McxVolumeShape(AObj, nx, ny, nz, Fmt);
  FDoc.Root.Elements['Shapes'] := AObj;
  FDoc.Delete('Domain.VolumeFile');
  if nx > 0 then
  begin
    FDoc.SetInt('Domain.Dim[0]', nx);
    FDoc.SetInt('Domain.Dim[1]', ny);
    FDoc.SetInt('Domain.Dim[2]', nz);
  end;
  if Fmt <> '' then FDoc.SetStr('Domain.MediaFormat', Fmt);
  DomainChanged;
end;

{ A mesh goes in as MeshNode and MeshElem, where mmc reads its mesh -- Mesh,
  or Shapes when that is where this file keeps it -- and MeshID goes, so
  there is one mesh and not a choice between two. }
procedure TfmMain.StoreMesh(ANode, AElem, AMedia: TJSONData;
  AReplaceMedia: Boolean);
var
  Base: string;
  B: TJSONObject;
  M: TJSONData;
begin
  { Media that came with the mesh: always when pasted with it, otherwise
    only into a file that has none of its own -- mmc prefers the file's. }
  if AMedia <> nil then
  begin
    M := FDoc.Find('Domain.Media');
    if AReplaceMedia or not (M is TJSONArray) or (M.Count = 0) then
      TJSONObject(FDoc.Ensure('Domain', jtObject)).Elements['Media'] := AMedia
    else
      AMedia.Free;
  end;
  Base := Copy(MeshPath('Mesh.MeshNode'), 1, Pos('.', MeshPath('Mesh.MeshNode')) - 1);
  B := TJSONObject(FDoc.Ensure(Base, jtObject));
  B.Elements['MeshNode'] := ANode;
  B.Elements['MeshElem'] := AElem;
  if B.Find('MeshID') <> nil then B.Delete('MeshID');
  { mmc refuses a mesh with no starting element; -1 has it search for the
    one the source is in. }
  if B.Find('InitElem') = nil then B.Add('InitElem', -1);
  DomainChanged;
end;

{ The dialog's OK check: would this text make a volume, or a mesh?  Only
  checked here; it is stored once the dialog has closed, and only if it was
  changed, so OK on what was there leaves the file as it was. }
function TfmMain.CheckVolumeText(const AText: string; out AError: string): Boolean;
var
  D: TJSONData;
  O: TJSONObject;
begin
  Result := False;
  AError := 'The editor is empty: paste a volume, or load one from a file.';
  if Trim(AText) = '' then Exit;
  try
    D := GetJSON(AText);
  except
    on E: Exception do
    begin
      AError := 'This is not valid JSON: ' + E.Message;
      Exit;
    end;
  end;
  try
    Result := McxVolumeFromJSON(D, O, AError);
    O.Free;
  finally
    D.Free;
  end;
end;

function TfmMain.CheckMeshText(const AText: string; out AError: string): Boolean;
var
  D, N, E: TJSONData;
begin
  Result := False;
  AError := 'The editor is empty: paste a mesh, or load one from a file.';
  if Trim(AText) = '' then Exit;
  try
    D := GetJSON(AText);
  except
    on Ex: Exception do
    begin
      AError := 'This is not valid JSON: ' + Ex.Message;
      Exit;
    end;
  end;
  try
    Result := McxMeshFromJSON(D, N, E, AError);
    N.Free;
    E.Free;
  finally
    D.Free;
  end;
end;

function TfmMain.ApplyVolumeText(const AText: string; out AError: string): Boolean;
var
  D: TJSONData;
  O: TJSONObject;
begin
  Result := False;
  AError := '';
  if Trim(AText) = '' then
  begin
    AError := 'The editor is empty: paste a volume, or load one from a file.';
    Exit;
  end;
  try
    D := GetJSON(AText);
  except
    on E: Exception do
    begin
      AError := 'This is not valid JSON: ' + E.Message;
      Exit;
    end;
  end;
  try
    if not McxVolumeFromJSON(D, O, AError) then Exit;
  finally
    D.Free;
  end;
  StoreVolume(O);
  Result := True;
end;

function TfmMain.ApplyMeshText(const AText: string; out AError: string): Boolean;
var
  D, N, E, M: TJSONData;
begin
  Result := False;
  AError := '';
  if Trim(AText) = '' then
  begin
    AError := 'The editor is empty: paste a mesh, or load one from a file.';
    Exit;
  end;
  try
    D := GetJSON(AText);
  except
    on Ex: Exception do
    begin
      AError := 'This is not valid JSON: ' + Ex.Message;
      Exit;
    end;
  end;
  M := nil;
  try
    if not McxMeshFromJSON(D, N, E, AError) then Exit;
    if (D is TJSONObject) and (TJSONObject(D).Find('Media') is TJSONArray) then
      M := TJSONObject(D).Find('Media').Clone;
  finally
    D.Free;
  end;
  StoreMesh(N, E, M, True);
  Result := True;
end;

function TfmMain.LoadVolumeText(const AFileName: string; out AText, AError: string): Boolean;
var
  O: TJSONObject;
begin
  AText := '';
  Result := McxVolumeFromFile(AFileName, RawWidth, FDoc.AsInt('Domain.Dim[0]', 0),
    FDoc.AsInt('Domain.Dim[1]', 0), FDoc.AsInt('Domain.Dim[2]', 0), O, AError);
  if Result then
  try
    AText := O.FormatJSON;
  finally
    O.Free;
  end;
end;

function TfmMain.LoadMeshText(const AFileName: string; out AText, AError: string): Boolean;
var
  N, E, M: TJSONData;
  O: TJSONObject;
begin
  AText := '';
  Result := McxMeshFromFile(AFileName, N, E, M, AError);
  if not Result then Exit;
  O := TJSONObject.Create;
  try
    O.Add('MeshNode', N);
    O.Add('MeshElem', E);
    if M <> nil then O.Add('Media', M);
    AText := O.FormatJSON;
  finally
    O.Free;
  end;
end;

function TfmMain.LoadVolumeFile(const AFileName: string; out AError: string): Boolean;
var
  O: TJSONObject;
begin
  Result := McxVolumeFromFile(AFileName, RawWidth, FDoc.AsInt('Domain.Dim[0]', 0),
    FDoc.AsInt('Domain.Dim[1]', 0), FDoc.AsInt('Domain.Dim[2]', 0), O, AError);
  if Result then StoreVolume(O);
end;

function TfmMain.LoadMeshFile(const AFileName: string; out AError: string): Boolean;
var
  N, E, M: TJSONData;
begin
  Result := McxMeshFromFile(AFileName, N, E, M, AError);
  if Result then StoreMesh(N, E, M, False);
end;

const
  VolumeFilter = 'Volumes (*.bin;*.raw;*.vol;*.nii;*.jnii;*.bnii;*.json)|' +
    '*.bin;*.raw;*.vol;*.nii;*.jnii;*.bnii;*.json|All files|*';
  MeshFilter = 'Meshes (node_*.dat;elem_*.dat;*.jmsh;*.json)|' +
    'node_*.dat;elem_*.dat;*.jmsh;*.json|All files|*';

procedure TfmMain.DomEditClick(Sender: TObject);
var
  T, T0, Err: string;
begin
  if FRun.AsStr('@run.domainkind', 'shapes') = 'shapes' then
  begin
    { Shapes have an editor of their own, command by command. }
    SelectSection(2);
    ShowSettingsPage;
    Exit;
  end;
  if DomainIsMesh then
  begin
    T := MeshText;
    T0 := T;
    if McxEditData(Self, 'Edit Mesh',
      'The mesh as mmc reads it from an input: MeshNode, a row of x y z per ' +
      'node, and MeshElem, four node numbers (from 1) and a region label ' +
      'per tetrahedron. Copy it to use elsewhere, or paste a mesh -- ' +
      'JData arrays or plain lists of rows, and the JMesh names ' +
      'MeshVertex3/MeshTet4 too. A Media list beside them, as Load from ' +
      'file adds from prop_<id>.dat, becomes the optical properties.  ' +
      'The whole of it may also sit under "Mesh", as in an mmc input:',
      '{"Mesh": {' + LineEnding +
      '   "MeshNode": [[x1, y1, z1], [x2, y2, z2], ...],' + LineEnding +
      '   "MeshElem": [[n1, n2, n3, n4, label], ...]},' + LineEnding +
      ' "Media": [{"mua": 0, "mus": 0, "g": 1, "n": 1},' + LineEnding +
      '           {"mua": 0.01, "mus": 10, "g": 0.9, "n": 1.37}, ...]}' + LineEnding +
      'MeshNode and MeshElem may each be a JData array instead, ' +
      '{"_ArrayType_": "double", "_ArraySize_": [N, 3], "_ArrayData_": [...]}; ' +
      'Media is optional.',
      MeshFilter, T, @CheckMeshText, @LoadMeshText) and (T <> T0) then
      ApplyMeshText(T, Err);
  end
  else
  begin
    T := VolumeText;
    T0 := T;
    if McxEditData(Self, 'Edit Volume',
      'The volume as mcx reads it from an input: a JData array (row-major, ' +
      'compressed), kept in Shapes. Copy it to use elsewhere, or paste a ' +
      'volume -- a JData array, or a nested list of numbers [[[...]]] in ' +
      'x, y, z order. OK makes it the domain, and sets Dim and the volume ' +
      'format to match.  Either of these, on its own (not inside Shapes):',
      '{"_ArrayType_": "uint8", "_ArraySize_": [Nx, Ny, Nz],' + LineEnding +
      ' "_ArrayData_": [v1, v2, ...]}       (or _ArrayZipType_/_ArrayZipData_)' +
      LineEnding + LineEnding +
      '[[[v111, v112, ...], [v121, ...], ...], [[v211, ...], ...], ...]' +
      LineEnding + 'Each value is a medium label: 0 is outside the domain.',
      VolumeFilter, T, @CheckVolumeText, @LoadVolumeText) and (T <> T0) then
      ApplyVolumeText(T, Err);
  end;
end;

procedure TfmMain.DomLoadClick(Sender: TObject);
var
  D: TOpenDialog;
  Ok: Boolean;
  E: string;
begin
  D := TOpenDialog.Create(Self);
  try
    if DomainIsMesh then D.Filter := MeshFilter else D.Filter := VolumeFilter;
    D.Options := D.Options + [ofFileMustExist];
    if FDoc.FileName <> '' then D.InitialDir := ExtractFilePath(FDoc.FileName);
    if not D.Execute then Exit;
    Screen.Cursor := crHourGlass;
    try
      if DomainIsMesh then Ok := LoadMeshFile(D.FileName, E)
      else Ok := LoadVolumeFile(D.FileName, E);
    finally
      Screen.Cursor := crDefault;
    end;
    if Ok then
      Log('Domain loaded from ' + D.FileName)
    else
      MessageDlg('MCX Studio', E, mtError, [mbOK], 0);
  finally
    D.Free;
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
  else if S = 'vkmmc' then Result := mbVKMMC
  else if S = 'umcx' then Result := mbUMCX
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
  acDevices.Enabled := CurrentExe <> '';
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
  { The wide-field detector fields wait for a first detector to exist. }
  ApplyBindingStates;
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
  if FQuitAfterRun then WriteLn(ALine);
end;

procedure TfmMain.RunProgress(Sender: TObject; APercent: Integer);
begin
  SetStatus(0, Format('Running  %d%%', [APercent]));
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
    { vkmmc writes its volume as <session>.jdat, the name mcxcl gives a
      trajectory only with _traj in front of it. }
    if FileExists(Guess + '.jnii') then ShowResult(Guess + '.jnii')
    else if FileExists(Guess + '.bnii') then ShowResult(Guess + '.bnii')
    else if (CurrentBackend = mbVKMMC) and FileExists(Guess + '.jdat') then
      ShowResult(Guess + '.jdat');
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
  if FQuitAfterRun then QuitAfterRun(AExitCode, '');
end;

procedure TfmMain.SnapshotLater(const AFileName: string);
begin
  FSnapshot := AFileName;
end;

function TfmMain.Unattended: Boolean;
begin
  Result := FQuitAfterRun or (FSnapshot <> '');
end;

procedure TfmMain.RunLater(AQuit: Boolean);
begin
  FRunPending := True;
  FQuitAfterRun := AQuit;
end;

function TfmMain.SetProcessor(const AValue: string): Boolean;
begin
  Result := (FProcessor <> nil) and
    ((AValue = 'nvidia') or (AValue = 'amd') or (AValue = 'intel') or (AValue = 'cpu'));
  if not Result then Exit;
  FProcessor.Value := AValue;
  ChoiceChanged(nil);
end;

{ The run --run asked for, now that the window exists.  The checks Run would
  make with a dialog are made here first when nobody is there to answer one. }
procedure TfmMain.StartRequestedRun;
begin
  if FQuitAfterRun then
  begin
    if FDoc.FileName = '' then
    begin
      QuitAfterRun(2, 'no input file to run');
      Exit;
    end;
    if CurrentExe = '' then
    begin
      QuitAfterRun(3, 'could not find ' + McxExeName(CurrentBackend) +
        ' beside this program, in an MCXStudio folder, or on PATH');
      Exit;
    end;
    WriteLn(McxCommandLine(CurrentExe, FDoc.FileName, FDoc, FRun));
  end;
  acRunExecute(nil);
  { Run did not start: it has already said why, in a dialog or not at all. }
  if FQuitAfterRun and (FRunner = nil) then
    QuitAfterRun(3, 'the simulation could not be started');
end;

procedure TfmMain.QuitAfterRun(AExitCode: Integer; const AWhy: string);
begin
  if AWhy <> '' then WriteLn(StdErr, 'mcxstudio: ', AWhy);
  Flush(Output);
  ExitCode := AExitCode;
  Application.Terminate;
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
  begin
    if not SaveAs then Exit;
  end
  else if NeedsWorkFolder then
  begin
    { Opened from somewhere read-only -- the examples the installer put
      under the program folder.  The run goes into the session folder, a
      copy of the input with it, rather than failing to write its output. }
    Log(Format('%s cannot be written here; running a copy in %s',
      [ExtractFileName(FDoc.FileName),
       McxWorkFolder(CurrentBackend, FDoc.AsStr('Session.ID', 'mcx'))]));
    if not SaveDocTo(IncludeTrailingPathDelimiter(
      McxWorkFolder(CurrentBackend, FDoc.AsStr('Session.ID', 'mcx'))) +
      ExtractFileName(FDoc.FileName)) then Exit;
  end;
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

  ShowLog;
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
  SetStatus(0, 'Running');
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
  i: Integer;
begin
  RefreshDevices;
  ShowLog;
  Log(CurrentExe + ' -L');
  if Length(FDevices) = 0 then
    Log('  no devices reported')
  else
    for i := 0 to High(FDevices) do
      Log(Format('  #%d  %s  (auto-thread %d, auto-block %d)',
        [FDevices[i].Id, FDevices[i].Name,
         FDevices[i].AutoThread, FDevices[i].AutoBlock]));
end;

{ The Devices card.  Built here because what it lists is whatever this
  machine has, which no designer session could know. }
procedure TfmMain.BuildDeviceCard;
var
  Bar: TPanel;
  Btn: TButton;
begin
  lbTodoDevices.Visible := False;

  FDevNote := TLabel.Create(gbDevices);
  FDevNote.Parent := gbDevices;
  FDevNote.Top := 1000;
  FDevNote.Align := alTop;
  FDevNote.WordWrap := True;
  FDevNote.BorderSpacing.Bottom := 4;
  FDevNote.Caption := 'Open this page, or press Find devices, to list them.';

  FDevList := TCheckListBox.Create(gbDevices);
  FDevList.Parent := gbDevices;
  FDevList.Top := 1001;
  FDevList.Align := alTop;
  FDevList.Height := 22 * 4 + 4;
  FDevList.OnClickCheck := @DeviceTicked;

  Bar := TPanel.Create(gbDevices);
  Bar.Parent := gbDevices;
  Bar.Top := 1002;
  Bar.Align := alTop;
  Bar.Height := 32;
  Bar.BevelOuter := bvNone;
  Bar.Caption := '';

  Btn := TButton.Create(Bar);
  Btn.Parent := Bar;
  Btn.Align := alLeft;
  Btn.BorderSpacing.Top := 3;
  Btn.Width := 130;
  Btn.Caption := 'Find devices';
  Btn.OnClick := @DeviceRefreshClick;
end;

procedure TfmMain.DeviceRefreshClick(Sender: TObject);
begin
  RefreshDevices;
end;

{ One line for a device: its number as -G counts it, its name, and the two
  facts that tell devices of one kind apart -- how much memory, how many
  compute units -- picked out of whatever else -L said. }
function DeviceLine(const ADev: TMcxDevice; AIndex: Integer): string;
var
  L: TStringList;
  i, c: Integer;
  K, V, Mem, Units: string;
  B: Double;
begin
  Mem := '';
  Units := '';
  L := TStringList.Create;
  try
    L.Text := ADev.Detail;
    for i := 0 to L.Count - 1 do
    begin
      c := Pos(':', L[i]);
      if c < 2 then Continue;
      K := LowerCase(Trim(Copy(L[i], 1, c - 1)));
      V := Trim(Copy(L[i], c + 1, MaxInt));
      if K = 'global memory' then
      begin
        B := StrToFloatDef(Trim(StringReplace(V, 'B', '', [])), 0);
        if B > 0 then Mem := Format('%.1f GB', [B / 1e9]);
      end
      else if (K = 'number of sms') or (K = 'compute units') then
        Units := StringReplace(V, ' core(s)', '', []) + ' compute units';
    end;
  finally
    L.Free;
  end;
  Result := Format('%d   %s', [AIndex + 1, ADev.Name]);
  if (Mem <> '') or (Units <> '') then
  begin
    Result := Result + '   (';
    if Units <> '' then Result := Result + Units;
    if (Units <> '') and (Mem <> '') then Result := Result + ', ';
    if Mem <> '' then Result := Result + Mem;
    Result := Result + ')';
  end;
end;

{ Asks the chosen simulator what it can run on, and ticks what the run is
  set to use.  -G takes either one device number or a string of 0s and 1s,
  one per device in this order (mcx -h), so the ticks are that string. }
procedure TfmMain.RefreshDevices;
var
  Exe, Listing, Mask: string;
  i: Integer;
begin
  if FDevList = nil then Exit;
  FDevFor := McxExeName(CurrentBackend);
  FDevList.Items.Clear;
  SetLength(FDevices, 0);
  Exe := CurrentExe;
  if Exe = '' then
  begin
    FDevNote.Caption := 'No ' + FDevFor + ' was found, so there is nothing ' +
      'to ask. Install it, or put it on the PATH.';
    Exit;
  end;
  if CurrentBackend = mbUMCX then
  begin
    FDevNote.Caption := 'umcx has no device listing; it runs on the device ' +
      'it was built for (the processor, unless built with GPU offload).';
    Exit;
  end;
  if (CurrentBackend = mbMMC) and (FRun.AsStr('@run.processor', '') = 'cpu') then
  begin
    FDevNote.Caption := 'mmc runs on the processor here (-c sse), which ' +
      'uses every core and needs no device.';
    Exit;
  end;

  Screen.Cursor := crHourGlass;
  try
    if not McxQueryDevices(Exe, Listing) then
    begin
      FDevNote.Caption := Exe + ' -L could not be run.';
      Exit;
    end;
  finally
    Screen.Cursor := crDefault;
  end;

  FDevices := McxParseDevices(Listing);
  if Length(FDevices) = 0 then
  begin
    FDevNote.Caption := FDevFor + ' -L reported no device it can use.';
    Exit;
  end;

  Mask := FRun.AsStr('@run.device', '');
  { mcx reads one character as a device number and anything longer as the
    0/1 string; nothing at all is the first device. }
  if (Length(Mask) = 1) and (StrToIntDef(Mask, 0) > 0) then
    Mask := StringOfChar('0', StrToInt(Mask) - 1) + '1';
  if Mask = '' then Mask := '1';
  for i := 0 to High(FDevices) do
  begin
    FDevList.Items.Add(DeviceLine(FDevices[i], i));
    FDevList.Checked[i] := (i < Length(Mask)) and (Mask[i + 1] = '1');
  end;
  FDevNote.Caption := Format('%s found %d device(s). Tick the ones a run ' +
    'uses; several share the photons between them.',
    [FDevFor, Length(FDevices)]);
end;

procedure TfmMain.DeviceTicked(Sender: TObject);
var
  i: Integer;
  Mask: string;
begin
  Mask := '';
  for i := 0 to FDevList.Items.Count - 1 do
    if FDevList.Checked[i] then Mask := Mask + '1' else Mask := Mask + '0';
  while (Mask <> '') and (Mask[Length(Mask)] = '0') do
    Delete(Mask, Length(Mask), 1);
  { The first device alone is what -G means when it is not given, so it is
    not given: the command stays as short as the person left it. }
  if (Mask = '') or (Mask = '1') then FRun.Delete('@run.device')
  else FRun.SetStr('@run.device', Mask);
  SchedulePreview;
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

procedure TMcxDockSplitter.Paint;
begin
  if (McxCurrentTheme = mtSystem) or not Enabled then
    inherited Paint
  else
  begin
    Canvas.Brush.Color := McxBlend(McxBase, McxText, 6);
    Canvas.FillRect(ClientRect);
  end;
end;

procedure TfmMain.BuildDock;
begin
  { AnchorDocking brings its own splitters, so the designer's two go. }
  FreeAndNil(spNav);
  FreeAndNil(spPreview);
  DockMaster.SplitterClass := TMcxDockSplitter;

  FDockSite := TAnchorDockPanel.Create(Self);
  FDockSite.Name := 'McxDockSite';
  FDockSite.Parent := pnMain;
  FDockSite.Align := alClient;
  { A TPanel, raised by default: a light edge drawn round all the panes. }
  FDockSite.BevelOuter := bvNone;

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
  { The log is a pane of its own rather than a page of the notebook, so the
    output of a run can be read beside the 3-D view of it.  Under the views
    to start with; its header drags it anywhere else, or onto the views to
    be a page of them again. }
  FPaneLog := MakePane('PaneLog', 'Log', mmLog, 900, 180);
  FreeAndNil(tsLog);

  DockMaster.OnCreateControl := @DockCreateControl;
  DockMaster.MakeDockPanel(FDockSite, admrpChild);

  { The middle goes in first, because the navigator docks against it.

    Both panes are made dockable hidden and shown once docked.  Shown here,
    the pane's dock site is mapped while it is still a floating window of
    its own -- a small empty one, mid-FormCreate -- and asking for it to be
    brought to front focuses it, which has GTK map this form too before it
    has a size or a position.  Either way something flashed at the corner
    of the screen before the window appeared in the middle. }
  DockMaster.MakeDockable(FPaneView, False, False, False);
  DockMaster.ManualDock(DockMaster.GetAnchorSite(FPaneView), FDockSite,
    alClient);
  DockMaster.ShowControl(FPaneView.Name, False);

  DockMaster.MakeDockable(FPaneNav, False, False, True);
  DockMaster.ManualDock(DockMaster.GetAnchorSite(FPaneNav), FDockSite, alLeft);
  DockMaster.ShowControl(FPaneNav.Name, False);

  DockMaster.MakeDockable(FPaneLog, False, False, True);
  DockLog;

  GuardCentreHeader;
end;

{ Docks the log under the views, at its starting height once there is a
  window to measure. }
procedure TfmMain.DockLog;
begin
  DockMaster.ManualDock(DockMaster.GetAnchorSite(FPaneLog), FDockSite,
    alBottom, DockMaster.GetAnchorSite(FPaneView));
  DockMaster.ShowControl(FPaneLog.Name, False);
  if FDockSized then SizePane(FPaneLog, alBottom, McxScale96(180))
  else FLogUnsized := True;
end;

{ Whether the log is anywhere to be seen: closed, or left out of a layout,
  it is in a dock site of its own that is hidden. }
function TfmMain.LogShown: Boolean;
var
  Site: TAnchorDockHostSite;
begin
  Site := DockMaster.GetAnchorSite(FPaneLog);
  Result := (Site <> nil) and Site.Visible and FPaneLog.Visible;
end;

{ Brings the log to the front, wherever it has been put -- floating, or a
  page beside the views -- and back under the views if it was closed. }
procedure TfmMain.ShowLog;
begin
  if FPaneLog = nil then Exit;
  if not LogShown then
  begin
    DockMaster.MakeDockable(FPaneLog, False, False, True);
    DockLog;
  end
  else
    DockMaster.ShowControl(FPaneLog.Name, True);
end;

{ The default sizes cannot be applied in FormCreate.

  Nothing is laid out yet at that point -- the dock site measures 170 by 50 --
  so a splitter has nothing to move and every pane comes out whatever
  DockAnotherControl guessed.  This runs once the window has been shown and
  the real geometry exists, and only when no saved layout was restored, since
  a restored one already carries the sizes the user chose. }
{ Opens on the screen the program was launched from.

  poScreenCenter is the centre of the primary monitor, which on a two-screen
  desk is the wrong screen whenever the work is on the other one -- the
  window arrives where nobody is looking.  The pointer is where the person
  just was, since something had to be clicked or typed to start this, so the
  monitor under it is the one to open on.

  Its work area rather than its bounds, so a panel or a dock does not get
  half the title bar.  And clamped, because a window can be taller than the
  space it is being centred in, and half of a negative overhang is a title
  bar off the top of the screen. }
procedure TfmMain.CentreOnLaunchScreen;
var
  M: TMonitor;
  R: TRect;
  x, y: Integer;
begin
  M := Screen.MonitorFromPoint(Mouse.CursorPos);
  if M = nil then M := Screen.PrimaryMonitor;
  if M = nil then Exit;
  R := M.WorkareaRect;
  x := R.Left + (R.Right - R.Left - Width) div 2;
  y := R.Top + (R.Bottom - R.Top - Height) div 2;
  if x < R.Left then x := R.Left;
  if y < R.Top then y := R.Top;
  { Off the designed position first, or the LCL puts the window back on the
    primary monitor the next time it thinks about it. }
  Position := poDesigned;
  SetBounds(x, y, Width, Height);
end;

procedure TfmMain.ApplyPaneSizes(Data: PtrInt);
begin
  { After the window has been shown and scaled, not before: the DPI sweep
    resizes it on the way up, and a centre worked out from the designed size
    centres the wrong rectangle. }
  CentreOnLaunchScreen;
  { The panes first: a result or a snapshot wants the 3-D view at the size it
    will have, not at the sliver the log leaves it before it is sized. }
  if not FDockRestored then SizePane(FPaneNav, alLeft, McxScale96(340));
  if FLogUnsized then SizePane(FPaneLog, alBottom, McxScale96(180));
  FLogUnsized := False;
  GuardCentreHeader;

  if FPendingResult <> '' then
  begin
    ShowResult(FPendingResult);
    FPendingResult := '';
  end;
  if FRunPending then
  begin
    FRunPending := False;
    StartRequestedRun;
  end;
  if FSnapshot <> '' then
  begin
    { The size the Save image action uses, twice the pane, so the picture
      is the one a person would have saved. }
    if FView.SaveImage(FSnapshot, pnGL.Width * 2, pnGL.Height * 2) then
    begin
      WriteLn('wrote ', FSnapshot, ' (', FView.Description, ')');
      ExitCode := 0;
    end
    else
    begin
      WriteLn(StdErr, 'mcxstudio: could not write ', FSnapshot);
      ExitCode := 1;
    end;
    FSnapshot := '';
    Application.Terminate;
  end;
end;

procedure TfmMain.ShowResultLater(const AFileName: string);
begin
  FPendingResult := AFileName;
end;

procedure TfmMain.FormShow(Sender: TObject);
begin
  if FDockSized then Exit;
  FDockSized := True;
  ApplyNativeChrome;
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
  else if SameText(aName, 'PaneMain') then AControl := FPaneView
  else if SameText(aName, 'PaneLog') then AControl := FPaneLog;
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
  { A layout saved before the log was a pane of its own leaves it out, and
    puts it away hidden. }
  if Result and not LogShown then
  begin
    DockMaster.MakeDockable(FPaneLog, False, False, True);
    DockLog;
  end;
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
  Prev: TTabSheet;
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

  { The 3-D page is brought up before the volume is handed to it, not after.
    On Windows a control on a page that has never been shown has no window
    yet, so no OpenGL context either, and the view turns the volume away: a
    run shows the log page while it runs, and its result was refused unless
    the 3-D page had been opened earlier in the session. }
  Prev := pcView.ActivePage;
  pcView.ActivePage := tsPreview;
  if FView.ShowVolume(A) then
  begin
    FDisplay.ResetSlab;
    FDisplay.Sync;
  end
  else
  begin
    pcView.ActivePage := Prev;
    Log('  the 3-D view would not take it');
  end;
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
{ Clicking a solid in the 3-D view selects its command in the editor, which
  is the other half of clicking a row and having it light up: the picture and
  the list are two views of one sequence, and picking in either should say
  the same thing. }
procedure TfmMain.ViewPick(Sender: TObject; const AInfo: TMcxPickInfo);
begin
  SetStatus(0, Format('Shapes[%d] %s, tag %d',
    [AInfo.Index, AInfo.Verb, AInfo.Tag]));
  if FShapes <> nil then FShapes.SelectIndex(AInfo.Index);
end;

{ And the other way: a row selected in the list is drawn nearly solid. }
procedure TfmMain.ShapeSelected(Sender: TObject; AIndex: Integer);
begin
  if FView <> nil then FView.SelectShape(AIndex);
end;

{ A shape with a label the media table has no row for is a domain mcx will
  not run, so the table grows to cover it: each new row a copy of the last,
  as the table's own Add button makes them, for the label to be edited from
  rather than typed in whole. }
procedure TfmMain.ShapeLabelAdded(Sender: TObject; ATag: Integer);
var
  M: TJSONData;
  i: Integer;
begin
  if FMedia = nil then Exit;
  { Bounded, not until-done: a Media that is not an array takes no rows. }
  for i := 0 to ATag do
  begin
    M := FDoc.Find('Domain.Media');
    if (M is TJSONArray) and (M.Count > ATag) then Break;
    FMedia.AddRow;
  end;
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
    ShowLog;
    Exit;
  end;

  Names := TStringList.Create;
  try
    if not McxBenchmarks(CurrentExe, Names) then
    begin
      Log(CurrentExe + ' -Q: no benchmarks reported.');
      ShowLog;
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
    ShowLog;
    Exit;
  end;
  if not FDoc.LoadFromString(JSON) then
  begin
    Log(Bench + ': ' + FDoc.LastError);
    ShowLog;
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
  DockMaster.MakeDockable(FPaneLog, False, False, True);
  DockLog;
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
    { A multi-source input gives a row per source, and not only to Pos and
      Dir: Param1 and Param2 take the same shape, one equal-length row each
      (example/multisrc/multisrc.json has all four).  All four are mkVec, so
      all four come through here.  One edit box cannot show a table, and
      flattening it would destroy the file -- so show nothing, and because
      the box then matches what it was loaded with, the save path leaves the
      value alone. }
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
  S, Def: string;
  N: Integer;
  V64: Int64;
  Absent: Boolean;
begin
  C := FBound[AIndex];
  if C = nil then Exit;
  B := Binds[AIndex];
  B.Path := MeshPath(B.Path);
  D := DocFor(B.Path);
  { A key the file leaves out shows what the chosen simulator will use for
    it (mcx_initcfg), not a blank or a zero that means nothing.  Shown only:
    FLoaded below records it, so an untouched default is never written. }
  Absent := D.Find(B.Path) = nil;
  Def := '';
  if Absent then Def := McxDefault(Binds[AIndex].Path, CurrentBackend);

  case B.Kind of
    mkBool:
      if C is TCheckBox then
      begin
        if Absent and (Def <> '') then TCheckBox(C).Checked := Def = '1'
        else TCheckBox(C).Checked := D.AsBool(B.Path);
      end;
    mkChoice:
      begin
        S := D.AsStr(B.Path);
        if Absent then S := Def;
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
        if Absent then S := Def
        else S := FlagsOf(D, B.Path, FFlags[AIndex]);
        for N := 0 to TCheckGroup(C).Items.Count - 1 do
          TCheckGroup(C).Checked[N] :=
            (N < Length(FFlags[AIndex])) and
            (Pos(FFlags[AIndex][N + 1], S) > 0);
      end;
    mkInt:
      { Value is an Integer and the file can hold any number: clamped, or a
        64-bit value is a range error here (and silently wrapped without range
        checks); the control then applies its own limits }
      if C is TSpinEdit then
      begin
        V64 := D.AsInt(B.Path, StrToIntDef(Def, 0));
        if V64 > High(Integer) then V64 := High(Integer)
        else if V64 < Low(Integer) then V64 := Low(Integer);
        TSpinEdit(C).Value := V64;
      end
      else if C is TEdit then
      begin
        if Absent then TEdit(C).Text := Def
        else TEdit(C).Text := IntToStr(D.AsInt(B.Path));
      end;
    mkFloat:
      if C is TEdit then
      begin
        { A scalar field may meet an array -- some files give DebugFlag or a
          length as a list -- and AsString raises on one.  Show it empty rather
          than taking the window down, and leave the value alone on save. }
        E := D.Find(B.Path);
        if (E <> nil) and (E.JSONType in [jtNumber, jtString, jtBoolean]) then
          TEdit(C).Text := E.AsString
        else if Absent then
          TEdit(C).Text := Def
        else
          TEdit(C).Text := '';
      end;
    mkVec:
      if C is TEdit then TEdit(C).Text := VecToText(D.Find(VecPath(B.Path)));
    mkText, mkFile:
      if C is TEdit then TEdit(C).Text := D.AsStr(B.Path, Def);
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
  N, Iv: Integer;
  Fs: TFormatSettings;
begin
  C := FBound[AIndex];
  if C = nil then Exit;
  B := Binds[AIndex];
  B.Path := MeshPath(B.Path);
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
        { A choice whose values are numbers -- BasisOrder is 0 or 1 -- is a
          number in the file; written as text, mmc's valueint reads it as
          nought whatever it said. }
        if (N >= 0) and (FValues[AIndex] <> nil) and
           (N < FValues[AIndex].Count) then
        begin
          if TryStrToInt(FValues[AIndex][N], Iv) and
             ((D.Find(B.Path) = nil) or (D.Find(B.Path).JSONType = jtNumber)) then
            D.SetInt(B.Path, Iv)
          else
            D.SetStr(B.Path, FValues[AIndex][N]);
        end;
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
      if C is TEdit then TextToVec(D, VecPath(B.Path), TEdit(C).Text);
    mkText, mkFile:
      if C is TEdit then
      begin
        if Trim(TEdit(C).Text) = '' then D.Delete(B.Path)
        else D.SetStr(B.Path, TEdit(C).Text);
      end;
  end;
end;

{ How many rows an embedded mesh array holds: a plain list of rows, or a
  JData-annotated one whose _ArraySize_ says it.  -1 when absent. }
function MeshRows(D: TJSONData): Integer;
var
  Sz: TJSONData;
begin
  Result := -1;
  if D = nil then Exit;
  if D.JSONType = jtArray then Exit(TJSONArray(D).Count);
  if D.JSONType <> jtObject then Exit;
  Sz := TJSONObject(D).Find('_ArraySize_');
  { a size that is not a number in range shows as no count rather than raising }
  if (Sz is TJSONArray) and (TJSONArray(Sz).Count > 0) and
     (TJSONArray(Sz).Items[0].JSONType = jtNumber) and
     (TJSONArray(Sz).Items[0].AsFloat >= 0) and (TJSONArray(Sz).Items[0].AsFloat <= MaxInt) then
    Result := Trunc(TJSONArray(Sz).Items[0].AsFloat);
end;

{ A mesh carried in the file is shown, not edited: thousands of nodes are
  not something to type into a form. }
procedure TfmMain.UpdateMeshInfo;
var
  nn, ne: Integer;
begin
  if lbMeshInfo = nil then Exit;
  nn := MeshRows(FDoc.Find(MeshPath('Mesh.MeshNode')));
  ne := MeshRows(FDoc.Find(MeshPath('Mesh.MeshElem')));
  if (nn >= 0) or (ne >= 0) then
  begin
    if nn < 0 then nn := 0;
    if ne < 0 then ne := 0;
    lbMeshInfo.Caption := Format('The mesh is embedded in this file: ' +
      'nodes %d, elements %d.', [nn, ne]);
  end
  else
    lbMeshInfo.Caption := 'MeshID names node_<id>.dat and elem_<id>.dat ' +
      'beside the input, or in the root path.';
end;

procedure TfmMain.LoadAllBindings;
var
  i: Integer;
begin
  { The tables read the document directly, so they are refreshed here rather
    than bound: there is no control-to-path row that could describe them. }
  if FMedia <> nil then FMedia.Attach(FDoc, 'Domain.Media');
  if FDetectors <> nil then FDetectors.Attach(FDoc, 'Optode.Detector');
  if FShapes <> nil then FShapes.Attach(FDoc);
  UpdateSourceBar;
  UpdateMeshInfo;
  UpdateDomainCard;
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
function ConditionHolds(D: TMcxDoc; const ACond: string;
  ABackend: TMcxBackend): Boolean;
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
    Exit(Pos(UpperCase(Want),
      UpperCase(D.AsStr(Path, McxDefault(Path, ABackend)))) > 0);
  end;

  { A trailing '?' asks only whether the path is there at all. }
  if ACond[Length(ACond)] = '?' then
    Exit(D.Exists(Copy(ACond, 1, Length(ACond) - 1)));

  E := Pos('=', ACond);
  if E < 1 then Exit;
  Path := Copy(ACond, 1, E - 1);
  Want := Copy(ACond, E + 1, MaxInt);
  { An absent key holds the simulator's default, the same value its field
    shows: DoPartialPath left out of an mcx input is on, and the settings
    that depend on it are live. }
  if (Want = '0') or (Want = '1') then
    Result := D.AsBool(Path, McxDefault(Path, ABackend) = '1') = (Want = '1')
  else
    Result := SameText(D.AsStr(Path, McxDefault(Path, ABackend)), Want);
end;


procedure TfmMain.ApplyBindingStates;
var
  i, j, s, First, NBound, NShown: Integer;
  B: TMcxBind;
  C: TControl;
  InMode, Shown, Moved: Boolean;
  Cap1, Cap2, Flags, Letters, DomName: string;
  Live: array of Boolean;
  Dom: TMcxDomain;
  Be: TMcxBackend;

  { Restacking is only needed when something appeared or disappeared, and
    this runs on every keystroke that commits, so it is worth knowing. }
  procedure Show(AControl: TControl; AVisible: Boolean);
  begin
    if AControl = nil then Exit;
    if AControl.Visible <> AVisible then Moved := True;
    AControl.Visible := AVisible;
  end;

begin
  DomName := FRun.AsStr('@run.domainkind', 'shapes');
  if DomName = 'voxel' then Dom := mdVoxel
  else if (DomName = 'mesh') or (DomName = 'surface') then Dom := mdMesh
  else Dom := mdShapes;
  Be := CurrentBackend;
  Moved := False;
  for i := 0 to High(Binds) do
  begin
    C := FBound[i];
    if C = nil then Continue;
    B := Binds[i];

    { The wizard is a strict subset of the expert form, so one filter serves
      both modes and there is no second layout to keep in step.  And a
      setting the chosen domain or simulator never reads is not offered:
      a mesh has no Dim and no volume file, a voxel grid no MeshID. }
    InMode := ((not FWizard) or (B.Level = mlWizard)) and
      ((B.Domains = []) or (Dom in B.Domains)) and
      ((B.Backends = []) or (Be in B.Backends));
    Show(C, InMode);
    if InMode then
      C.Enabled := ConditionHolds(DocFor(B.EnableIf), B.EnableIf,
        CurrentBackend);
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
    { The cards with a table on them stay whatever their few bound rows
      do: the detector card's wide-field rows are mmc's, and hiding them
      must not take the detector table with them. }
    Live[j] := (NBound = 0) or (NShown > 0) or
      (FSubs[j].Box = gbDetector) or (FSubs[j].Box = gbMedia);
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
  { Picking a simulator is what a tile press is: another program, with
    other defaults and other devices. }
  if Binds[i].Path = '@run.simulator' then
  begin
    ChoiceChanged(Sender);
    Exit;
  end;
  { A source vector typed above the source table is a cell of it too. }
  if (FSources <> nil) and (Pos('Optode.Source.', Binds[i].Path) = 1) then
    FSources.Reload;
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
  { The pane is rewritten on every edit that commits, so where it was
    scrolled to has to survive that -- otherwise reading the file while
    changing a number is impossible. }
  ShowJson;

  { The command the Run action will issue, shown so it can be copied, re-run
    and pasted into a bug report.  Almost everything lives in the JSON, so
    this stays short -- which is the whole reason the old CreateCmd's 230
    lines of flag assembly are not being ported. }
  if FDoc.FileName <> '' then
    Input := FDoc.FileName
  else
    Input := FDoc.AsStr('Session.ID', 'mcx') + '.json';
  mmCommand.Text := McxCommandLine(CurrentExe, Input, FDoc, FRun);
  if FView <> nil then
  begin
    FView.Rebuild;
    { A domain carried as data is drawn as a volume, and the display bar
      that styles a volume comes and goes with it. }
    if FView.VolumeChanged then
    begin
      FView.VolumeChanged := False;
      FDisplay.ResetSlab;
      FDisplay.Sync;
    end;
  end;

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
    SetStatus(0, 'Modified')
  else
    SetStatus(0, 'Ready');
  if FWizard then
    SetStatus(1, 'Wizard')
  else
    SetStatus(1, 'Expert');
  { A mesh has no Dim, and "0 x 0 x 0" reads as a broken file rather than as
    a different kind of domain. }
  if FRun.AsStr('@run.domainkind', 'shapes') = 'mesh' then
    SetStatus(2, 'mesh ' + FDoc.AsStr('Mesh.MeshID', '(embedded)'))
  else
    SetStatus(2, Format('%d x %d x %d',
      [FDoc.AsInt('Domain.Dim[0]'), FDoc.AsInt('Domain.Dim[1]'),
       FDoc.AsInt('Domain.Dim[2]')]));
end;

end.
