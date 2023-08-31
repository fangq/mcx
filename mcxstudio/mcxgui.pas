unit mcxgui;
{==============================================================================
    Monte Carlo eXtreme (MCX) Studio
-------------------------------------------------------------------------------
    Author: Qianqian Fang
    Email : q.fang at neu.edu
    Web   : http://mcx.space
    License: GNU General Public License version 3 (GPLv3)
===============================================================================}
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, process, FileUtil, SynEdit, math, ClipBrd, AnchorDocking,
  SynHighlighterAny, SynHighlighterPerl, synhighlighterunixshellscript, LclIntf,
  LResources, Forms, Controls, Graphics, Dialogs, StdCtrls, Menus, ComCtrls,
  ExtCtrls, Spin, EditBtn, Buttons, ActnList, lcltype, AsyncProcess, Grids,
  CheckLst, LazHelpHTML, ValEdit, inifiles, fpjson, jsonparser,jsonscanner {$IFDEF USE_SYNAPSE}, runssh{$ENDIF},
  strutils, RegExpr, OpenGLTokens, mcxabout, mcxshape, mcxnewsession, mcxsource,
  mcxrender, mcxview, mcxconfig, mcxstoprun, Types {$IFDEF WINDOWS}, registry, ShlObj{$ENDIF};

type

  { TfmMCX }

  TfmMCX = class(TForm)
    acEditShape: TActionList;
    mcxdoPlotJNIFTI: TAction;
    btSendCmd: TButton;
    btExpandOutput: TButton;
    Button2: TButton;
    ckDoRemote: TCheckBox;
    ckSharedFS: TCheckBox;
    ckShowProgress: TCheckBox;
    edCmdInput: TEdit;
    edBenchmark: TComboBox;
    edRemote: TComboBox;
    Image1: TImage;
    Label19: TLabel;
    Label5: TLabel;
    mcxdoConfig: TAction;
    ckbDet: TCheckListBox;
    edOutputType: TComboBox;
    grBC: TGroupBox;
    grDet: TGroupBox;
    MenuItem64: TMenuItem;
    MenuItem65: TMenuItem;
    Label14: TLabel;
    mcxdoWebURL: TAction;
    MenuItem33: TMenuItem;
    MenuItem34: TMenuItem;
    MenuItem35: TMenuItem;
    MenuItem36: TMenuItem;
    MenuItem37: TMenuItem;
    MenuItem38: TMenuItem;
    MenuItem39: TMenuItem;
    MenuItem40: TMenuItem;
    MenuItem41: TMenuItem;
    MenuItem42: TMenuItem;
    MenuItem43: TMenuItem;
    MenuItem44: TMenuItem;
    MenuItem45: TMenuItem;
    MenuItem46: TMenuItem;
    MenuItem47: TMenuItem;
    MenuItem48: TMenuItem;
    MenuItem49: TMenuItem;
    MenuItem50: TMenuItem;
    MenuItem51: TMenuItem;
    MenuItem52: TMenuItem;
    MenuItem53: TMenuItem;
    MenuItem54: TMenuItem;
    MenuItem55: TMenuItem;
    MenuItem56: TMenuItem;
    MenuItem57: TMenuItem;
    MenuItem58: TMenuItem;
    MenuItem59: TMenuItem;
    MenuItem60: TMenuItem;
    MenuItem61: TMenuItem;
    MenuItem62: TMenuItem;
    MenuItem63: TMenuItem;
    MenuItem66: TMenuItem;
    MenuItem67: TMenuItem;
    MenuItem68: TMenuItem;
    MenuItem69: TMenuItem;
    MenuItem70: TMenuItem;
    MenuItem71: TMenuItem;
    MenuItem72: TMenuItem;
    MenuItem73: TMenuItem;
    MenuItem74: TMenuItem;
    MenuItem75: TMenuItem;
    MenuItem76: TMenuItem;
    miExportJSON: TMenuItem;
    miClearLog: TMenuItem;
    miCopy: TMenuItem;
    mmOutput: TSynEdit;
    plConsole: TPanel;
    Panel2: TPanel;
    plOutputDock: TPanel;
    PopupMenu3: TPopupMenu;
    PopupMenu4: TPopupMenu;
    plSetting: TScrollBox;
    rbUseBench: TRadioButton;
    shapePreview: TAction;
    edOutputFormat: TComboBox;
    Label11: TLabel;
    mcxdoDownloadMask: TAction;
    mcxdoDownloadMCH: TAction;
    mcxdoDownloadMC2: TAction;
    htmlHelpDatabase: THTMLHelpDatabase;
    MenuItem28: TMenuItem;
    MenuItem30: TMenuItem;
    MenuItem31: TMenuItem;
    MenuItem32: TMenuItem;
    miUseMatlab: TMenuItem;
    MenuItem29: TMenuItem;
    Splitter5: TSplitter;
    Splitter6: TSplitter;
    SynUNIXShellScriptSyn1: TSynUNIXShellScriptSyn;
    ToolBar3: TToolBar;
    ToolButton34: TToolButton;
    ToolButton35: TToolButton;
    ToolButton38: TToolButton;
    ToolButton39: TToolButton;
    ToolButton40: TToolButton;
    ToolButton41: TToolButton;
    ToolButton42: TToolButton;
    ToolButton43: TToolButton;
    ToolButton44: TToolButton;
    ToolButton45: TToolButton;
    ToolButton46: TToolButton;
    ToolButton47: TToolButton;
    ToolButton48: TToolButton;
    ToolButton50: TToolButton;
    ToolButton51: TToolButton;
    ToolButton52: TToolButton;
    vlBC: TValueListEditor;
    webBrowser: THTMLBrowserHelpViewer;
    HTMLHelpDatabase1: THTMLHelpDatabase;
    mcxdoPlotMC2: TAction;
    mcxdoPlotMesh: TAction;
    mcxdoPlotNifty: TAction;
    mcxdoPlotVol: TAction;
    btGBExpand: TButton;
    ckLockGPU: TCheckBox;
    ckReflect: TCheckBox;
    ckSpecular: TCheckBox;
    edMoreParam: TEdit;
    grAtomic: TRadioGroup;
    MenuItem24: TMenuItem;
    MenuItem25: TMenuItem;
    MenuItem26: TMenuItem;
    MenuItem27: TMenuItem;
    PopupMenu1: TPopupMenu;
    ProgramIcon: TImageList;
    Label18: TLabel;
    mcxdoToggleView: TAction;
    mcxdoPaste: TAction;
    mcxdoCopy: TAction;
    ckDoReplay: TCheckBox;
    ckSaveMask: TCheckBox;
    ckbDebug: TCheckListBox;
    edReplayDet: TSpinEdit;
    JSONIcons: TImageList;
    ImageList3: TImageList;
    Label16: TLabel;
    Label17: TLabel;
    MenuItem19: TMenuItem;
    MenuItem20: TMenuItem;
    MenuItem21: TMenuItem;
    MenuItem22: TMenuItem;
    MenuItem23: TMenuItem;
    miClearLog1: TMenuItem;
    OpenHistoryFile: TOpenDialog;
    OpenVolume: TOpenDialog;
    PopupMenu2: TPopupMenu;
    grProgram: TRadioGroup;
    OpenDir: TSelectDirectoryDialog;
    shapePrint: TAction;
    shapeEdit: TAction;
    shapeAddCylinder: TAction;
    shapeAddZSlabs: TAction;
    shapeAddYSlabs: TAction;
    shapeAddXSlabs: TAction;
    shapeAddZLayers: TAction;
    shapeAddYLayers: TAction;
    shapeAddXLayers: TAction;
    shapeAddUpperSpace: TAction;
    shapeAddSubgrid: TAction;
    shapeAddBox: TAction;
    shapeAddSphere: TAction;
    shapeAddGrid: TAction;
    shapeAddOrigin: TAction;
    shapeAddName: TAction;
    shapeDelete: TAction;
    shapeReset: TAction;
    ckSaveSeed: TCheckBox;
    edGPUID: TCheckListBox;
    ckAutopilot: TCheckBox;
    ckNormalize: TCheckBox;
    ckSaveData: TCheckBox;
    ckSaveDetector: TCheckBox;
    ckSaveRef: TCheckBox;
    ckSkipVoid: TCheckBox;
    ckSrcFrom0: TCheckBox;
    edBlockSize: TComboBox;
    edBubble: TEdit;
    edConfigFile: TFileNameEdit;
    edDetectedNum: TEdit;
    edGate: TSpinEdit;
    edPhoton: TEdit;
    edRespin: TSpinEdit;
    edSeed: TEdit;
    edSession: TEdit;
    edThread: TComboBox;
    edUnitInMM: TEdit;
    edWorkload: TEdit;
    grAdvSettings: TGroupBox;
    grArray: TRadioGroup;
    grBasic: TGroupBox;
    grGPU: TGroupBox;
    grSwitches: TGroupBox;
    ImageList2: TImageList;
    Label1: TLabel;
    Label10: TLabel;
    lbBubble: TLabel;
    Label12: TLabel;
    Label13: TLabel;
    lbAtomic: TLabel;
    Label15: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    lbRespin: TLabel;
    Label6: TLabel;
    Label7: TLabel;
    Label8: TLabel;
    Label9: TLabel;
    lvJobs: TListView;
    mcxdoHelpOptions: TAction;
    OpenProject: TOpenDialog;
    pcSimuEditor: TPageControl;
    plOutput: TPanel;
    pMCX: TAsyncProcess;
    mcxSetCurrent: TAction;
    acInteract: TActionList;
    mcxdoDefault: TAction;
    mcxdoClearLog: TAction;
    mcxdoSave: TAction;
    mcxdoExit: TAction;
    mcxdoListGPU: TAction;
    mcxdoQuery: TAction;
    mcxdoWeb: TAction;
    mcxdoAbout: TAction;
    mcxdoHelp: TAction;
    mcxdoRunAll: TAction;
    mcxdoStop: TAction;
    mcxdoRun: TAction;
    mcxdoVerify: TAction;
    mcxdoDeleteItem: TAction;
    mcxdoAddItem: TAction;
    mcxdoOpen: TAction;
    mcxdoInitEnv: TAction;
    acMCX: TActionList;
    ImageList1: TImageList;
    MainMenu1: TMainMenu;
    MenuItem17: TMenuItem;
    MenuItem18: TMenuItem;
    MenuItem1: TMenuItem;
    MenuItem10: TMenuItem;
    MenuItem11: TMenuItem;
    MenuItem12: TMenuItem;
    MenuItem13: TMenuItem;
    MenuItem14: TMenuItem;
    MenuItem15: TMenuItem;
    MenuItem16: TMenuItem;
    MenuItem2: TMenuItem;
    MenuItem3: TMenuItem;
    MenuItem4: TMenuItem;
    MenuItem5: TMenuItem;
    MenuItem6: TMenuItem;
    MenuItem7: TMenuItem;
    MenuItem8: TMenuItem;
    MenuItem9: TMenuItem;
    rbUseFile: TRadioButton;
    rbUseDesigner: TRadioButton;
    SaveProject: TSaveDialog;
    sbInfo: TStatusBar;
    btLoadSeed: TSpeedButton;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    Splitter3: TSplitter;
    Splitter4: TSplitter;
    StaticText1: TStaticText;
    StaticText2: TStaticText;
    StaticText3: TStaticText;
    sgMedia: TStringGrid;
    sgDet: TStringGrid;
    sgConfig: TStringGrid;
    tabInputData: TTabSheet;
    tabVolumeDesigner: TTabSheet;
    tbtRun: TToolButton;
    tbtStop: TToolButton;
    tbtVerify: TToolButton;
    Timer1: TTimer;
    tmAnimation: TTimer;
    ToolBar1: TToolBar;
    ToolBar2: TToolBar;
    ToolButton1: TToolButton;
    ToolButton10: TToolButton;
    ToolButton11: TToolButton;
    ToolButton12: TToolButton;
    ToolButton13: TToolButton;
    ToolButton14: TToolButton;
    ToolButton15: TToolButton;
    ToolButton16: TToolButton;
    ToolButton17: TToolButton;
    ToolButton18: TToolButton;
    ToolButton2: TToolButton;
    ToolButton3: TToolButton;
    ToolButton31: TToolButton;
    ToolButton4: TToolButton;
    ToolButton5: TToolButton;
    ToolButton6: TToolButton;
    ToolButton7: TToolButton;
    ToolButton8: TToolButton;
    ToolButton9: TToolButton;
    tvShapes: TTreeView;
    procedure btExpandOutputClick(Sender: TObject);
    procedure btLoadSeedClick(Sender: TObject);
    procedure btGBExpandClick(Sender: TObject);
    procedure btSendCmdClick(Sender: TObject);
    procedure Button1Click(Sender: TObject);
    procedure ckLockGPUChange(Sender: TObject);
    procedure edCmdInputKeyPress(Sender: TObject; var Key: char);
    procedure edSessionEditingDone(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormShow(Sender: TObject);
    procedure grAdvSettingsClick(Sender: TObject);
    procedure grAdvSettingsDblClick(Sender: TObject);
    procedure grProgramSelectionChanged(Sender: TObject);
    procedure mcxdoAboutExecute(Sender: TObject);
    procedure mcxdoAddItemExecute(Sender: TObject);
    procedure mcxdoConfigExecute(Sender: TObject);
    procedure mcxdoCopyExecute(Sender: TObject);
    procedure mcxdoDefaultExecute(Sender: TObject);
    procedure mcxdoDeleteItemExecute(Sender: TObject);
    procedure mcxdoDownloadMaskExecute(Sender: TObject);
    procedure mcxdoDownloadMC2Execute(Sender: TObject);
    procedure mcxdoDownloadMCHExecute(Sender: TObject);
    procedure mcxdoExitExecute(Sender: TObject);
    procedure mcxdoHelpExecute(Sender: TObject);
    procedure mcxdoHelpOptionsExecute(Sender: TObject);
    procedure mcxdoOpenExecute(Sender: TObject);
    procedure mcxdoPasteExecute(Sender: TObject);
    procedure mcxdoPlotJNIFTIExecute(Sender: TObject);
    procedure mcxdoPlotMC2Execute(Sender: TObject);
    procedure mcxdoPlotNiftyExecute(Sender: TObject);
    procedure mcxdoPlotVolExecute(Sender: TObject);
    procedure mcxdoQueryExecute(Sender: TObject);
    procedure mcxdoRunExecute(Sender: TObject);
    procedure mcxdoSaveExecute(Sender: TObject);
    procedure mcxdoStopExecute(Sender: TObject);
    procedure mcxdoToggleViewExecute(Sender: TObject);
    procedure mcxdoVerifyExecute(Sender: TObject);
    procedure edConfigFileChange(Sender: TObject);
    procedure edRespinChange(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure lvJobsSelectItem(Sender: TObject; Item: TListItem;
      Selected: Boolean);
    procedure mcxdoWebExecute(Sender: TObject);
    procedure mcxdoWebURLExecute(Sender: TObject);
    procedure mcxSetCurrentExecute(Sender: TObject);
    procedure MenuItem22Click(Sender: TObject);
    procedure MenuItem76Click(Sender: TObject);
    procedure miExportJSONClick(Sender: TObject);
    procedure miClearLogClick(Sender: TObject);
    procedure miCopyClick(Sender: TObject);
    procedure miUseMatlabClick(Sender: TObject);
    procedure MenuItem9Click(Sender: TObject);
    procedure pMCXReadData(Sender: TObject);
    procedure pMCXTerminate(Sender: TObject);
    procedure rbUseFileChange(Sender: TObject);
    procedure sbInfoDrawPanel(StatusBar: TStatusBar; Panel: TStatusPanel;
      const Rect: TRect);
    procedure sgConfigDblClick(Sender: TObject);
    procedure sgConfigEditButtonClick(Sender: TObject);
    procedure sgConfigResize(Sender: TObject);
    procedure sgConfigSelectEditor(Sender: TObject; aCol, aRow: Integer;
      var Editor: TWinControl);
    procedure sgMediaDrawCell(Sender: TObject; aCol, aRow: Integer;
      aRect: TRect; aState: TGridDrawState);
    procedure sgMediaEditingDone(Sender: TObject);
    procedure shapeAddBoxExecute(Sender: TObject);
    procedure shapeAddCylinderExecute(Sender: TObject);
    procedure shapeAddGridExecute(Sender: TObject);
    procedure shapeAddNameExecute(Sender: TObject);
    procedure shapeAddOriginExecute(Sender: TObject);
    procedure shapeAddSphereExecute(Sender: TObject);
    procedure shapeAddSubgridExecute(Sender: TObject);
    procedure shapeAddUpperSpaceExecute(Sender: TObject);
    procedure shapeAddXLayersExecute(Sender: TObject);
    procedure shapeAddXSlabsExecute(Sender: TObject);
    procedure shapeAddYLayersExecute(Sender: TObject);
    procedure shapeAddYSlabsExecute(Sender: TObject);
    procedure shapeAddZLayersExecute(Sender: TObject);
    procedure shapeAddZSlabsExecute(Sender: TObject);
    procedure shapePrintExecute(Sender: TObject);
    procedure shapePreviewExecute(Sender: TObject);
    procedure shapeResetExecute(Sender: TObject);
    procedure shapeDeleteExecute(Sender: TObject);
    procedure Splitter6CanOffset(Sender: TObject; var NewOffset: Integer;
      var Accept: Boolean);
    procedure Splitter6CanResize(Sender: TObject; var NewSize: Integer;
      var Accept: Boolean);
    procedure StaticText2DblClick(Sender: TObject);
    procedure tbtRunClick(Sender: TObject);
    procedure Timer1Timer(Sender: TObject);
    procedure tmAnimationTimer(Sender: TObject);
    procedure tvShapesEdited(Sender: TObject; Node: TTreeNode; var S: string);
    procedure tvShapesSelectionChanged(Sender: TObject);
    procedure vlBCGetPickList(Sender: TObject; const KeyName: string;
      Values: TStrings);
    procedure LoadSessionFromJSON(jfile: string);
    procedure NewSessionFromJSON(jsonstr, folder: string);
  private
    { private declarations }
  public
    { public declarations }
    MapList, ConfigData, JSONstr, PassList : TStringList;
    JSONdata : TJSONData;
    RegEngine:TRegExpr;
    {$IFDEF USE_SYNAPSE}
    sshrun: TSSHThread;
    {$ENDIF}
    function CreateCmd(proc: TProcess=nil):AnsiString;
    function CreateCmdOnly:AnsiString;
    function SKey(str: AnsiString):AnsiString;
    procedure VerifyInput;
    procedure AddLog(str:AnsiString);
    procedure AddMultiLineLog(str:AnsiString; Sender: TObject);
    procedure ListToPanel2(node:TListItem);
    procedure PanelToList2(node:TListItem);
    procedure UpdateMCXActions(actlst: TActionList; ontag,offtag: string);
    procedure UpdateGPUList(Buffer:string);
    function  GetMCXOutput(Sender: TObject): string;
    procedure SaveTasksToIni(fname: string);
    procedure LoadTasksFromIni(fname: string);
    procedure RunExternalCmd(cmd: AnsiString);
    function  GetBrowserPath : AnsiString;
    function  GetFileBrowserPath : string;
    function  SearchForExe(fname : string; isremote: boolean = false) : string;
    function CreateWorkFolder(session: string; iscreate: boolean=true):string;
    function SaveJSONConfig(filename: string): AnsiString;
    function CheckListToStr(list: TCheckListBox) : string;
    function GridToStr(grid:TStringGrid) :AnsiString;
    procedure StrToGrid(str: string; grid:TStringGrid);
    procedure ShowJSONData(AParent : TTreeNode; Data : TJSONData; toplevel: boolean=false);
    procedure AddShapesWindow(shapeid: string; defaultval: TStringList; node: TTreeNode);
    procedure AddShapes(shapeid: string; defaultval: string);
    function UpdateShapeTag(root: TJSONData; doexport: boolean = false): integer;
    function RebuildLayeredObj(rootdata: TJSONData; out maxtag: integer): AnsiString;
    procedure SetModified;
    procedure LoadJSONShapeTree(shapejson: string);
    procedure GotoColRow(grid: TStringGrid; Col, Row: Integer);
    procedure SetSessionType(sessiontype: integer);
    function ResetMCX(exitcode: LongInt) : boolean;
    function ExpandPathMacro(path, app: string): string;
    function ExpandPassword(url: AnsiString): AnsiString;
    function GetAppRoot: string;
    procedure SwapState(old, new: TListItem);
    procedure RunSSHCmd(Sender: TObject; cmd: string; updategpu:boolean=false; doprogress: boolean=false);
    function CreateSSHDownloadCmd(suffix: string='.nii'): AnsiString;
  end;

var
  fmMCX: TfmMCX;
  fmDomain: TfmDomain;
  fmConfig: TfmConfig;
  fmStop: TfmStop;
  ProfileChanged: Boolean;
  MaxWait: integer;
  GotoCol, GotoRow: Integer;
  GotoGrid: TStringGrid;
  GotoGBox: TGroupBox;
  CurrentSession: TListItem;
  BCItemProp: TItemProp;
  UseUserFolder: Boolean;

implementation

Const
  ImageTypeMap : Array[TJSONtype] of Integer =
//      jtUnknown, jtNumber, jtString, jtBoolean, jtNull, jtArray, jtObject
     (-1,8,9,7,6,5,4);
  JSONTypeNames : Array[TJSONtype] of string =
     ('Unknown','Number','String','Boolean','Null','Array','Object');
  MCProgram : Array[0..2] of string =
     ('mcx','mmc','mcxcl');
  DebugFlags: string ='RMP';
  SaveDetFlags: string ='DSPMXVW';
  BCFlags: string = 'ARMC';
  OutputTypeFlags: string = 'XFEJPM';

{ TfmMCX }
procedure TfmMCX.AddLog(str:AnsiString);
begin
    mmOutput.Lines.Add(str);
    mmOutput.SelStart := length(mmOutput.Text);
    mmOutput.LeftChar:=0;
end;

procedure TfmMCX.AddMultiLineLog(str:AnsiString; Sender: TObject);
var
   sl: TStringList;
begin
    sl:=TStringList.Create;
    sl.StrictDelimiter:=true;
    sl.Delimiter:=#10;
    sl.DelimitedText:=str;
    if(Sender is TAsyncProcess) then begin
      if((Sender as TAsyncProcess)=pMCX) then begin
        mmOutput.Lines.AddStrings(sl);
        mmOutput.SelStart := length(mmOutput.Text);
        mmOutput.LeftChar:=0;
      end;
    end;
    sl.Free;
end;

procedure TfmMCX.edConfigFileChange(Sender: TObject);
begin
  if(Length(edSession.Text)=0) then
       edSession.Text:=ExtractFileName(edConfigFile.FileName);
end;
procedure TfmMCX.SetModified;
begin
    //if not (mcxdoRun.Enabled) then exit;
    UpdateMCXActions(acMCX,'','Work');
    UpdateMCXActions(acMCX,'','Run');
    mcxdoSave.Enabled:=true;
end;

procedure TfmMCX.edRespinChange(Sender: TObject);
var
    ed: TEdit;
    cb: TComboBox;
    ck: TCheckBox;
    se: TSpinEdit;
    gr: TRadioGroup;
    ckb: TCheckListBox;
    fed:TFileNameEdit;
    sg: TStringGrid;
    tv: TTreeView;
    idx: integer;
    node: TListItem;
    ss: string;
begin
    if(Sender=nil) or (CurrentSession=nil) then exit;
    if(lvJobs.Tag=1) then exit;
    try
    node:=CurrentSession;
    if(Sender is TSpinEdit) then begin
       se:=Sender as TSpinEdit;
       idx:=MapList.IndexOf(SKey(se.Hint));
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=IntToStr(se.Value);
    end else if(Sender is TEdit) then begin
       ed:=Sender as TEdit;
       idx:=MapList.IndexOf(SKey(ed.Hint));
       if(SKey(ed.Hint) = 'Session') then  begin
         node.Caption:=ed.Text;
         Caption:='MCX Studio - ['+node.Caption+']';
       end;
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=ed.Text;
    end else if(Sender is TRadioGroup) then begin
       gr:=Sender as TRadioGroup;
       idx:=MapList.IndexOf(SKey(gr.Hint));
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=IntToStr(gr.ItemIndex);
    end else if(Sender is TComboBox) then begin
       cb:=Sender as TComboBox;
       idx:=MapList.IndexOf(SKey(cb.Hint));
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=cb.Text;
    end else if(Sender is TCheckBox) then begin
       ck:=Sender as TCheckBox;
       idx:=MapList.IndexOf(SKey(ck.Hint));
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=IntToStr(Integer(ck.Checked));
       if(SKey(ck.Hint)='Autopilot') then begin
           edThread.Enabled:=not ck.Checked;
           edBlockSize.Enabled:=not ck.Checked;
       end;
       if(SKey(ck.Hint)='SaveSeed') or (SKey(ck.Hint)='SaveExit') then begin
           ckSaveDetector.Checked:=true;
       end;
       if(SKey(ck.Hint)='ShowProgress') then begin
           ckbDebug.Checked[2]:=ck.Checked;
       end;
       if(SKey(ck.Hint)='DoReflect') then begin
           grBC.Enabled:=not ck.Checked;
       end;
       if(SKey(ck.Hint)='SaveDetector') then begin
           edDetectedNum.Enabled:=ck.Checked;
           mcxdoDownloadMCH.Enabled:=ck.Checked;
           ckbDet.Enabled:=ck.Checked;
       end;
       if(SKey(ck.Hint)='DoSaveMask') then begin
           mcxdoDownloadMask.Enabled:=ck.Checked;
       end;
       if(SKey(ck.Hint)='DoReplay') then begin
           edReplayDet.Enabled:=ck.Checked;
       end;
       if(SKey(ck.Hint)='DoRemote') then begin
           edRemote.Enabled:=ck.Checked;
           ckSharedFS.Enabled:=ck.Checked;
           if(ck.Checked) then
              mcxdoQuery.Enabled:=true
           else
              mcxdoQuery.Enabled:=(SearchForExe(CreateCmdOnly) <> '');
           mcxdoDownloadMC2.Enabled:=ck.Checked and (not ckSharedFS.Checked);
       end;
       if(SKey(ck.Hint)='DoSharedFS') then begin
           mcxdoDownloadMC2.Enabled:=(not ck.Checked) and (ckDoRemote.Checked);
       end;
    end else if(Sender is TCheckListBox) then begin
       ckb:=Sender as TCheckListBox;
       idx:=MapList.IndexOf(SKey(ckb.Hint));
       if(idx>=0) then begin
           node.SubItems.Strings[idx]:=CheckListToStr(ckb);
       end;
       if(SKey(ckb.Hint)='DebugFlags') then
           ckShowProgress.Checked:=ckb.Checked[2];
    end else if(Sender is TFileNameEdit) then begin
       fed:=Sender as TFileNameEdit;
       idx:=MapList.IndexOf(SKey(fed.Hint));
       if(idx>=0) then
                  node.SubItems.Strings[idx]:=fed.Text;
    end else if(Sender is TStringGrid) then begin
       sg:=Sender as TStringGrid;
       idx:=MapList.IndexOf(SKey(sg.Hint));
       if(idx>=0) then
             node.SubItems.Strings[idx]:=GridToStr(sg);
    end else if(Sender is TTreeView) then begin
       tv:=Sender as TTreeView;
       idx:=MapList.IndexOf(SKey(tv.Hint));
       if(idx>=0) and (tv.Name='tvShapes') then  begin
           ss:= JSONData.AsJSON;
           if(JSONData.FindPath('Shapes') <> nil) then begin
               tv.Tag:=UpdateShapeTag(JSONData.FindPath('Shapes'));
           end else begin
               tv.Tag:=UpdateShapeTag(JSONData);
           end;
           node.SubItems.Strings[idx]:=TJSONData(JSONData).FormatJSON(AsJSONFormat);
       end;
    end;
    SetModified;
    except
    end;
end;

procedure TfmMCX.mcxdoExitExecute(Sender: TObject);
var
   ret:TModalResult;
begin
    if(mcxdoSave.Enabled) then begin
       ret:=MessageDlg('Confirmation', 'The current session has not been saved, do you want to save before exit?', mtConfirmation, [mbYes, mbNo, mbCancel],0);
       if (ret=mrYes) then
            mcxdoSaveExecute(Sender);
       if (ret=mrCancel) then
            exit;
    end;
    Close;
end;

procedure TfmMCX.mcxdoHelpExecute(Sender: TObject);
begin

end;

function TfmMCX.ExpandPassword(url: AnsiString): AnsiString;
var
    pass: string;
begin
    Result:=url;
    if(Pos('%PASSWORD%',url)>0) then begin
        pass:=PassList.Values[url];
        if(Length(pass)=0) then begin
            pass:=PasswordBox('Password', 'Please type in your password for the command:');
            PassList.Values[url]:=pass;
        end;
        Result:=StringReplace(url,'%PASSWORD%', pass,[rfReplaceAll]);
    end;
end;

procedure TfmMCX.mcxdoHelpOptionsExecute(Sender: TObject);
begin
    if(not pMCX.Running) then begin
          AddLog('"-- Run Command --"');
          if(ckDoRemote.Checked) then begin
              pMCX.CommandLine:=ExpandPassword(edRemote.Text)+' '+ CreateCmdOnly+' --help';
              AddLog(edRemote.Text+' '+ CreateCmdOnly+' --help');
          end else begin
              pMCX.CommandLine:=CreateCmdOnly+' --help';
              AddLog(pMCX.CommandLine);
          end;
          sbInfo.Panels[0].Text := 'Status: querying command line options';
          pMCX.Tag:=-2;
          AddLog('"-- Print MCX Command Line Options --"');
          pMCX.Execute;
    end;
end;

procedure TfmMCX.SetSessionType(sessiontype: integer);
begin
    grProgram.ItemIndex:=sessiontype;
end;

procedure TfmMCX.mcxdoAddItemExecute(Sender: TObject);
var
   node: TListItem;
   i:integer;
   sessionid: string;
   sessiontype: integer;
   fmNewSession: TfmNewSession;
begin
   if(not (Sender is TJSONObject)) then begin
     fmNewSession:=TfmNewSession.Create(self);
     fmnewSession.grProgram.Columns:=1;
     if (Sender is TEdit) then begin
         fmnewSession.grProgram.ItemIndex:=grProgram.ItemIndex;
         fmNewSession.edSession.Text:=edSession.Text;
         fmNewSession.Tag:=1;
     end;
     if(fmNewSession.ShowModal= mrOK) then begin
            sessionid:=Trim(fmNewSession.edSession.Text);
            sessiontype:=fmNewSession.grProgram.ItemIndex;
     end else begin
            fmNewSession.Free;
            exit;
     end;
     fmNewSession.Free;
   end else begin
     sessionid:=TJSONObject(Sender).Strings['ID'];
     sessiontype:=0;
     // append number if same name exist
     if(lvJobs.FindCaption(0,sessionid,true,true,true) <> nil) then begin
       for i:=1 to 100000 do begin
         if(lvJobs.FindCaption(0,sessionid+IntToStr(i),true,true,true) = nil) then begin
           sessionid:=sessionid+IntToStr(i);
           break;
         end;
       end;
     end;
   end;

   if (not (Sender is TEdit)) or (Sender is TJSONObject) then
       node:=lvJobs.Items.Add
   else
       node:=CurrentSession;
   for i:=1 to lvJobs.Columns.Count-1 do node.SubItems.Add('');
   node.Caption:=sessionid;
   node.ImageIndex:=sessiontype;
   plSetting.Enabled:=true;
   shapePreview.Enabled:=true;
   shapePreview.Enabled:=true;
   pcSimuEditor.Enabled:=true;
   lvJobs.Selected:=node;
   SwapState(CurrentSession,lvJobs.Selected);
   CurrentSession:=lvJobs.Selected;
   if(Sender is TJSONObject) then mcxdoSave.Enabled:=false;
   mcxdoDefaultExecute(nil);
   edSession.Text:=sessionid;
   SetSessionType(sessiontype);
   UpdateMCXActions(acMCX,'','Work');
   UpdateMCXActions(acMCX,'','Run');
   UpdateMCXActions(acMCX,'Preproc','');
   UpdateMCXActions(acMCX,'SelectedJob','');
end;

procedure TfmMCX.mcxdoConfigExecute(Sender: TObject);
begin
    fmConfig.ShowModal;
end;

procedure TfmMCX.mcxdoCopyExecute(Sender: TObject);
var
   setting: TStringList;
   j: integer;
begin
   if(lvJobs.Selected = nil) then exit;
   setting:=TStringList.Create;
   setting.Values['Session']:=lvJobs.Selected.Caption;
   for j:=1 to lvJobs.Columns.Count-1 do begin
           setting.Add(lvJobs.Columns.Items[j].Caption+'='+lvJobs.Selected.SubItems.Strings[j-1]);
   end;
   Clipboard.Open;
   try
      Clipboard.Clear;
      Clipboard.AsText := setting.Text;
   finally
      Clipboard.Close;
      setting.Free;
   end;
end;

procedure TfmMCX.mcxdoDefaultExecute(Sender: TObject);
begin

      if(mcxdoSave.Enabled) then begin
            if not (MessageDlg('Confirmation', 'Are you sure you want to discard the current setting?', mtConfirmation, [mbYes, mbNo, mbCancel],0)=mrYes) then
                exit;
      end;

      //edSession.Text:='';
      edConfigFile.FileName:='';
      edThread.Text:='16384';
      edPhoton.Text:='1e6';
      edBlockSize.Text:='64';
      edBubble.Text:='-2';
      edGate.Value:=100;
      edRespin.Value:=1;
      grArray.ItemIndex:=0;
      ckReflect.Checked:=true;   //-b
      ckSaveData.Checked:=true;   //-S
      ckNormalize.Checked:=true;   //-U
      ckSaveDetector.Checked:=true;   //-d
      ckSaveRef.Checked:=false;  //-X
      ckSrcFrom0.Checked:=true;  //-z
      ckSkipVoid.Checked:=true;  //-k
      ckAutopilot.Checked:=true;
      ckSaveSeed.Checked:=false;
      ckSaveMask.Checked:=false;
      edThread.Enabled:=false;
      edBlockSize.Enabled:=false;
      ckSpecular.Checked:=false;
      edWorkLoad.Text:='100';
      edMoreParam.Text:='';
      edUnitInMM.Text:='1';
      ckSharedFS.Checked:=false;
      if not (ckLockGPU.Checked) then begin
          edGPUID.CheckAll(cbUnchecked);
          if(edGPUID.Items.Count>0) then begin
              edGPUID.Checked[0]:=true;
          end;
      end;
      pcSimuEditor.ActivePage:=tabInputData;
      edDetectedNum.Text:='10000000';
      edSeed.Text:='1648335518';
      ckDoReplay.Checked:=false;
      ckbDebug.CheckAll(cbUnchecked);
      grAtomic.ItemIndex:=0;
      edReplayDet.Value:=0;
      rbUseDesigner.Checked:=true;
      sgMedia.RowCount:=3;
      sgMedia.Rows[1].CommaText:=',0,0,1,1';
      sgMedia.Rows[2].CommaText:=',0.005,1,0.01,1.37';
      sgMedia.RowCount:=20;
      sgMedia.FixedCols:=1;
      sgMedia.FixedRows:=1;

      sgDet.RowCount:=2;
      sgDet.Rows[1].CommaText:=',24,29,0,1';
      sgDet.RowCount:=20;
      sgDet.FixedCols:=1;
      sgDet.FixedRows:=1;

      sgConfig.Cols[2].CommaText:=ConfigData.CommaText;
      sgConfig.FixedCols:=2;
      sgConfig.FixedRows:=1;
{$IFDEF USE_SYNAPSE}
      edRemote.ItemIndex:=0;
{$ELSE}
      edRemote.ItemIndex:=1;
{$ENDIF}
      ckDoRemote.Checked:=false;
      ckSharedFS.Checked:=false;
      grBC.Enabled:=false;
      ckbDet.CheckAll(cbUnchecked);
      ckbDet.Checked[0]:=true;
      ckbDet.Checked[2]:=true;
      edOutputType.ItemIndex:=0;
      edOutputFormat.ItemIndex:=0;
      edBenchmark.ItemIndex:=0;
      vlBC.Values['x-'] := 'absorb';
      vlBC.Values['x+'] := 'absorb';
      vlBC.Values['y-'] := 'absorb';
      vlBC.Values['y+'] := 'absorb';
      vlBC.Values['z-'] := 'absorb';
      vlBC.Values['z+'] := 'absorb';
      vlBC.FixedCols:=1;

      if(grProgram.ItemIndex=1) then begin
          sgConfig.Rows[1].CommaText:='Domain,MeshID,';
          sgConfig.Rows[2].CommaText:='Domain,InitElem,';
          sgConfig.Rows[3].CommaText:='Session,RayTracer,g - Dual-grid MMC';
          rbUseBench.Enabled:=false;
          edBenchMark.Enabled:=false;
      end else begin
          sgConfig.Rows[1].CommaText:='Domain,VolumeFile,"See Volume Designer..."';
          sgConfig.Rows[2].CommaText:='Domain,Dim,"[60,60,60]"';
          sgConfig.Rows[3].CommaText:='Domain,MediaFormat,byte - 1 byte integer';
          rbUseBench.Enabled:=true;
          edBenchMark.Enabled:=true;
      end;
      LoadJSONShapeTree('[{"Grid":{"Tag":1,"Size":[60,60,60]}}]');
      if not (CurrentSession = nil) then
         PanelToList2(CurrentSession);
end;

procedure TfmMCX.UpdateMCXActions(actlst: TActionList; ontag,offtag: string);
var
   i: integer;
begin
   for i:=0 to actlst.ActionCount-1 do begin
       if (Length(ontag)>0) and (actlst.Actions[i].Category=ontag) then begin
             (actlst.Actions[i] as TAction).Enabled:=true;
       end else if (Length(offtag)>0) and (actlst.Actions[i].Category=offtag) then
             (actlst.Actions[i] as TAction).Enabled:=false;
   end;
end;


procedure TfmMCX.mcxdoAboutExecute(Sender: TObject);
var
    fmAbout:TfmAbout;
begin
     fmAbout:=TfmAbout.Create(Application);
     fmAbout.ShowModal;
     fmAbout.Free;
end;

procedure TfmMCX.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
      tvShapes.Enabled:=false;
end;

procedure TfmMCX.FormShow(Sender: TObject);
begin
    grGPU.Top:=grProgram.Height+grBasic.Height;
    grAdvSettings.Height:=self.Canvas.TextHeight('Ag')+btGBExpand.Height+2;
    sgMedia.FixedCols:=1;
    sgMedia.FixedRows:=1;
    sgDet.FixedCols:=1;
    sgDet.FixedRows:=1;
    vlBC.FixedCols:=1;
end;

procedure TfmMCX.grAdvSettingsClick(Sender: TObject);
begin

end;

procedure TfmMCX.btLoadSeedClick(Sender: TObject);
begin
      if(OpenHistoryFile.Execute) then begin
          edSeed.Text:=OpenHistoryFile.FileName;
          ckDoReplay.Checked:=true;
      end;
end;

procedure TfmMCX.btExpandOutputClick(Sender: TObject);
begin
      if(btExpandOutput.Tag=0) then begin // expand
          btExpandOutput.Tag:=plConsole.Height;
          btExpandOutput.Caption:=#9662;
          plConsole.Height:=lvJobs.Height;
          sbInfo.Top:=plConsole.Top+plConsole.Height;
      end else begin
          btExpandOutput.Caption:=#9653;
          plConsole.Height:=btExpandOutput.Tag;
          btExpandOutput.Tag:=0;
          sbInfo.Top:=plConsole.Top+plConsole.Height;
      end;
end;

procedure TfmMCX.btGBExpandClick(Sender: TObject);
var
     gr: TGroupBox;
begin
     if not (Sender is TButton) then exit;
     if not ((Sender as TButton).Parent is TGroupBox) then exit;
     gr:=((Sender as TButton).Parent as TGroupBox);
     GotoGBox:=gr;
     if(tmAnimation.Tag=1) then begin // collapse
         //gr.Align:=alTop;
         (Sender as TButton).Caption:=#9662;
         GotoGBox.Height:=self.Canvas.TextHeight('Ag')+btGBExpand.Height+2;
         //tmAnimation.Enabled:=true;
         tmAnimation.Tag:=0;
     end else begin
         (Sender as TButton).Caption:=#9653;
         GotoGBox.Height:=edMoreParam.Top+edMoreParam.Height+self.Canvas.TextHeight('Ag')+5;
         tmAnimation.Tag:=1;
         //tmAnimation.Enabled:=true;
     end;
end;

procedure TfmMCX.btSendCmdClick(Sender: TObject);
var
    cmd: string;
begin
    cmd:=edCmdInput.Text+#10;
    if(Length(cmd)=0) or (pMCX=nil) or (not pMCX.Running) then exit;
    pMCX.Input.Write(cmd[1], Length(cmd));
    mmOutput.Lines.Add('"User input:" '+cmd);
    edCmdInput.Text:='';
end;

procedure TfmMCX.Button1Click(Sender: TObject);
begin

end;

procedure TfmMCX.ckLockGPUChange(Sender: TObject);
begin
     edGPUID.Enabled:=not ckLockGPU.Checked;
     edRemote.Enabled:=not ckLockGPU.Checked;
     ckDoRemote.Enabled:=not ckLockGPU.Checked;
     ckSharedFS.Enabled:=not ckLockGPU.Checked;
end;

procedure TfmMCX.edCmdInputKeyPress(Sender: TObject; var Key: char);
begin
     if (Key = #13) or (Key = #10) then begin
       btSendCmdClick(Sender);
     end;
end;

procedure TfmMCX.edSessionEditingDone(Sender: TObject);
begin
   //if()
   edRespinChange(Sender);
end;

procedure TfmMCX.grAdvSettingsDblClick(Sender: TObject);
var
    i: integer;
begin
     if(grAdvSettings.Tag<0) then begin
       for i:=20 to plSetting.ClientHeight do begin
            grAdvSettings.Height:=i;
            Sleep(100);
       end;
       grAdvSettings.Tag:=0;
     end else begin
       for i:=plSetting.ClientHeight downto 20 do begin
              grAdvSettings.Height:=i;
              Sleep(100);
       end;
       grAdvSettings.Tag:=-1;
     end;
end;

procedure TfmMCX.grProgramSelectionChanged(Sender: TObject);
begin
  case grProgram.ItemIndex of
    0, 2: begin
        //grGPU.Top:=grProgram.Height+grBasic.Height;
        //grGPU.Visible:=true;
        tabVolumeDesigner.Enabled:=true;
        ckSpecular.Visible:=true;
        //ckSaveRef.Visible:=true;
        edRespin.Hint:='RespinNum';
        lbRespin.Caption:='Split into runs (-r)';
        edBubble.Hint:='BubbleSize';
        lbBubble.Caption:='Cache radius from src (-R)';
        //mcxdoQuery.Enabled:=true;
        ckSrcFrom0.Visible:=true;
        ckSaveMask.Visible:=true;
        edOutputFormat.ItemIndex:=0;
        grArray.Enabled:=true;
        edGate.Enabled:=true;
        edDetectedNum.Enabled:=true;
    end;
    1: begin
        //grGPU.Visible:=false;
        tabVolumeDesigner.Enabled:=false;
        ckSpecular.Visible:=true;
        //ckSaveRef.Visible:=false;
        edOutputFormat.ItemIndex:=0;
        edRespin.Hint:='BasicOrder';
        lbRespin.Caption:='Element order (-C)';
        edBubble.Hint:='DebugPhoton';
        lbBubble.Caption:='Debug photon index';
        //mcxdoQuery.Enabled:=false;
        ckSrcFrom0.Visible:=false;
        ckSaveMask.Visible:=false;
        grArray.Enabled:=false;
        edGate.Enabled:=false;
        edDetectedNum.Enabled:=false;
    end;
  end;
  if(grProgram.ItemIndex=1) then begin
        sgConfig.Rows[1].CommaText:='Domain,MeshID,';
        sgConfig.Rows[2].CommaText:='Domain,InitElem,';
        sgConfig.Rows[3].CommaText:='Session,RayTracer,g - Dual-grid MMC';
  end else begin
        sgConfig.Rows[1].CommaText:='Domain,VolumeFile,"See Volume Designer..."';
        sgConfig.Rows[2].CommaText:='Domain,Dim,"[60,60,60]"';
        sgConfig.Rows[3].CommaText:='Domain,MediaFormat,byte - 1 byte integer';
  end;
  if(CurrentSession <> nil) then
      CurrentSession.ImageIndex:=grProgram.ItemIndex+3;
  if(lvJobs.Tag=0) then
      edRespinChange(Sender);
end;
procedure TfmMCX.SwapState(old, new: TListItem);
begin
   if(old <> nil) then
      if(old.ImageIndex>=3) then old.ImageIndex:=old.ImageIndex-3;
   if(new <> nil) then
      if(new.ImageIndex<3)  then new.ImageIndex:=new.ImageIndex+3;
end;

procedure TfmMCX.mcxdoDeleteItemExecute(Sender: TObject);
begin
  if not (lvJobs.Selected = nil) then
  begin
        if not (MessageDlg('Confirmation', 'The selected configuration will be deleted, are you sure?', mtConfirmation, [mbYes, mbNo, mbCancel],0)=mrYes) then
            exit;
        if(CurrentSession=lvJobs.Selected) then
           CurrentSession:=nil;
        lvJobs.Items.Delete(lvJobs.Selected.Index);
        if(lvJobs.Items.Count>0) then begin
            lvJobs.Selected:=lvJobs.Items[0];
            SwapState(CurrentSession,lvJobs.Selected);
            CurrentSession:=lvJobs.Selected;
        end;
        if not (CurrentSession = nil) then
            ListToPanel2(CurrentSession)
        else
            mcxdoDeleteItem.Enabled:=false;
  end;
end;

procedure TfmMCX.mcxdoDownloadMaskExecute(Sender: TObject);
begin
  mcxdoDownloadMC2Execute(Sender);
end;

function TfmMCX.CreateSSHDownloadCmd(suffix: string='.nii'): string;
var
   rootpath, localfile, remotefile, url, cmd, scpcmd: string;
begin
   rootpath:='MCXOutput'+'/'+CreateCmdOnly+'sessions'+'/'+Trim(edSession.Text);
   localfile:=CreateWorkFolder(edSession.Text, true)+DirectorySeparator+edSession.Text+suffix;
   remotefile:=rootpath+'/'+edSession.Text+suffix;
   scpcmd:=edRemote.Text;
   scpcmd:=StringReplace(scpcmd,'plink', 'pscp',[rfReplaceAll]);
   scpcmd:=StringReplace(scpcmd,'ssh ', 'scp ',[rfReplaceAll]);
   url:=ExpandPassword(scpcmd);
   if(sscanf(url,'%s',[@cmd])=1) then begin
       Result:='"'+SearchForExe(cmd)+'"'+
          Copy(url,Length(cmd)+1,Length(url)-
          Length(cmd))+':"'+remotefile+'" "'+localfile+'"';
       AddLog('"'+SearchForExe(cmd)+'"'+
          Copy(scpcmd,Length(cmd)+1,Length(scpcmd)-
          Length(cmd))+':"'+remotefile+'" "'+localfile+'"');
       exit;
   end;
   Result:='';
end;

procedure TfmMCX.mcxdoDownloadMC2Execute(Sender: TObject);
var
    suffix: string;
begin
   if not (Sender is TAction) then exit;
   if(ResetMCX(0)) then begin
        suffix:=(Sender as TAction).Hint;
        if(Length(suffix)=0) then
           suffix:='.'+edOutputFormat.Text;
        pMCX.CommandLine:=CreateSSHDownloadCmd(suffix);
        if(Length(pMCX.CommandLine)=0) then exit;
        AddLog('"-- Downloading remote file --"');
        pMCX.Execute;
        mcxdoStop.Enabled:=true;
        mcxdoRun.Enabled:=false;
        sbInfo.Panels[0].Text := 'Status: downloading file';
        sbInfo.Panels[2].Text := '';
        pMCX.Tag:=-20;
        Application.ProcessMessages;
    end;
end;

procedure TfmMCX.mcxdoDownloadMCHExecute(Sender: TObject);
begin
  mcxdoDownloadMC2Execute(Sender);
end;

procedure TfmMCX.mcxdoOpenExecute(Sender: TObject);
var
    ret:TModalResult;
    TaskFile: string;
    fext: string;
    fmViewer: TfmViewer;
begin
  ret:=mrNo;
  if(OpenProject.Execute) then begin
    TaskFile:=OpenProject.FileName;
    fext:=ExtractFileExt(TaskFile);
    if(fext = '.json') then begin // adding new session
         LoadSessionFromJSON(TaskFile);
    end else if (AnsiIndexStr(fext, ['.tx3','.nii','.jnii']) >= 0) then begin
      try
            fmViewer:=TfmViewer.Create(self);
            Case AnsiIndexStr(fext, ['.tx3','.nii','.jnii']) of
                 0, 2:  fmViewer.LoadTexture(TaskFile);
                 1:  if(Pos(TaskFile, '_vol.nii') > 0) then
                         fmViewer.LoadTexture(TaskFile,0,0,0,2,352,GL_RGBA16I)
                     else
                         fmViewer.LoadTexture(TaskFile,0,0,0,0,352,GL_RGBA32F);
            else
            end;
            fmViewer.BringToFront;
            fmViewer.Show;
      except
          on E: Exception do
             ShowMessage('OpenGL Error: '+E.ClassName+#13#10 + E.Message);
      end;
    end else begin
      if(mcxdoSave.Enabled) then begin
         ret:=MessageDlg('Confirmation', 'The current session has not been saved, do you want to save it?',
             mtConfirmation, [mbYes, mbNo, mbCancel],0);
         if(ret=mrYes) then begin
               mcxdoSaveExecute(Sender);
               if(mcxdoSave.Enabled=true) then exit;
         end;
      end;
      if not (ret=mrCancel) then begin
        lvJobs.Items.Clear;
        CurrentSession:=nil;
        LoadTasksFromIni(TaskFile);
      end;
    end
  end
end;

procedure TfmMCX.mcxdoPasteExecute(Sender: TObject);
var
   setting: TStringList;
   j: integer;
   node: TListItem;
   fmNewSession: TfmNewSession;
begin
   setting:=TStringList.Create;
   setting.Text:=Clipboard.AsText;

   for j:=1 to 100000 do begin
       if(lvJobs.FindCaption(0,setting.Values['Session']+IntToStr(j),true,true,true) = nil) then
          break;
   end;

   fmNewSession:=TfmNewSession.Create(Application);
   fmnewSession.grProgram.ChildSizing.ControlsPerLine:=1;
   fmnewSession.edSession.Text:=setting.Values['Session'];
   if(Length(setting.Values['MCProgram'])>0) then
       fmnewSession.grProgram.ItemIndex:=StrToInt(setting.Values['MCProgram']);
   if(fmNewSession.ShowModal= mrOK) then begin
          setting.Values['Session']:=Trim(fmNewSession.edSession.Text);
          setting.Values['MCProgram']:=IntToStr(fmNewSession.grProgram.ItemIndex);
   end else begin
          fmNewSession.Free;
          setting.Free;
          exit;
   end;
   fmNewSession.Free;

   grProgram.ItemIndex:=StrToInt(setting.Values['MCProgram']);
   node:=lvJobs.Items.Add;
   node.Caption:=setting.Values['Session'];
   node.ImageIndex:=grProgram.ItemIndex;
   //node.ImageIndex:=-1;
   for j:=1 to lvJobs.Columns.Count-1 do
       node.SubItems.Add('');
   for j:=1 to lvJobs.Columns.Count-1 do begin
       node.SubItems.Strings[j-1]:=setting.Values[lvJobs.Columns.Items[j].Caption];
   end;
   setting.Free;
   lvJobs.Selected:=node;
   mcxSetCurrentExecute(Sender);
end;

procedure TfmMCX.mcxdoPlotJNIFTIExecute(Sender: TObject);
begin
  mcxdoPlotVolExecute(Sender);
end;

procedure TfmMCX.mcxdoPlotMC2Execute(Sender: TObject);
begin
  mcxdoPlotVolExecute(Sender);
end;

procedure TfmMCX.mcxdoPlotNiftyExecute(Sender: TObject);
begin
   mcxdoPlotVolExecute(Sender);
end;

procedure TfmMCX.mcxdoPlotVolExecute(Sender: TObject);
var
    outputfile: string;
    ftype: TAction;
    nx : integer = 0;
    ny,nz,nt: integer;
    fmViewer: TfmViewer;
    cmd: TStringList;
    singletype: LongWord;
    dref: string;
begin
    if(CurrentSession=nil) then exit;
    if (grProgram.ItemIndex=1) and (sgConfig.Cells[2,3] <> 'g') then begin
        MessageDlg('Warning', 'You must set Session::RayTracer to "g" for MMC to use this feature', mtError, [mbOK],0);
        exit;
    end;
    if not (Sender is TAction) then exit;
    ftype:=Sender as TAction;

    if (grProgram.ItemIndex <> 1) then begin
        outputfile:=CreateWorkFolder(edSession.Text, false)+DirectorySeparator+edSession.Text+ftype.Hint;
        singletype:=GL_RGBA32F;
    end else begin
        outputfile:=sgConfig.Cells[2,14]+DirectorySeparator+edSession.Text+ftype.Hint;
        singletype:=GL_DOUBLE_EXT;
    end;

    if(not FileExists(outputfile)) then begin
        MessageDlg('Warning', 'Specified file does not exists', mtError, [mbOK],0);
        exit;
    end;
    nt:=Round((StrToFloat(sgConfig.Cells[2,5])-StrToFloat(sgConfig.Cells[2,4]))/StrToFloat(sgConfig.Cells[2,6]));
    if (grProgram.ItemIndex <> 1) and (sscanf(sgConfig.Cells[2,2] ,'[%d,%d,%d]',[@nx,@ny,@nz])<>3) then begin
      MessageDlg('Warning', 'Domain size specifier contains incorrect format', mtError, [mbOK],0);
      exit;
    end;

    dref:='';
    if(ckSaveRef.Checked) then dref:=',dref';

    cmd:=TStringList.Create;
    cmd.Add(Format('%d %d %d %d',[nx,ny,nz,nt]));
    cmd.Add('%%%%%%%%% MATLAB/OCTAVE PLOTTING SCRIPT %%%%%%%%%');
    cmd.Add(Format('addpath(''%s'');',[GetAppRoot+
        'MCXSuite'+DirectorySeparator+'mcx'+DirectorySeparator+'utils']));
    Case AnsiIndexStr(ftype.Hint, ['.tx3','.mc2','.img','.nii','_vol.nii','.jnii']) of
          0:    cmd.Add(Format('[data%s]=loadmc2(''%s'',[%d,%d,%d,%d],''float'',16);', [dref,outputfile,nx,ny,nz,nt]));
          1..2: cmd.Add(Format('[data%s]=loadmc2(''%s'',[%d,%d,%d,%d],''float'');', [dref,outputfile,nx,ny,nz,nt]));
          3..4: cmd.Add(Format('img=mcxloadnii(''%s'');data=img.img;', [outputfile]));
          5: cmd.Add(Format('img=loadjson(''%s'');data=img.NIFTIData;', [outputfile]));
    else
    end;
    if(ckSaveDetector.Checked) then begin
        cmd.Add(Format('detps=loadmch(''%s'');', [ChangeFileExt(outputfile,'.mch')]));
        cmd.Add('%% call mcxdetphoton to parse it into subfields');
    end;
    if(ckbDebug.Checked[1]) then begin
        cmd.Add(Format('traj=loadmch(''%s'');', [ChangeFileExt(outputfile,'.mct')]));
        cmd.Add('% mcxplotphotons(traj); %% plot the trajectories');
    end;

    if not (ftype.Hint='_vol.nii') then
        cmd.Add('mcxplotvol(log10(data));')
    else
        cmd.Add('mcxplotvol(data);');

    cmd.Add('%%%%%%%%% END PLOTTING SCRIPT %%%%%%%%%');
    cmd.Delimiter:=#10;
    AddMultiLineLog(cmd.DelimitedText,pMCX);
    cmd.Free;

    if(miUseMatlab.Checked) then exit;
    try
          fmViewer:=TfmViewer.Create(self);
          Case AnsiIndexStr(ftype.Hint, ['.tx3','.mc2','.img','.nii','_vol.nii','.jnii']) of
               0:  fmViewer.LoadTexture(outputfile);
               1..2:  fmViewer.LoadTexture(outputfile,nx,ny,nz,nt,0,GL_RGBA32F);
               3:  fmViewer.LoadTexture(outputfile,nx,ny,nz,nt,352,singletype);
               4:  fmViewer.LoadTexture(outputfile,nx,ny,nz,2,352,GL_RGBA16I);
               5:  fmViewer.LoadTexture(outputfile);
          else
          end;
          fmViewer.BringToFront;
          fmViewer.Show;
    except
        on E: Exception do
           ShowMessage('OpenGL Error: '+E.ClassName+#13#10 + E.Message);
    end;
end;

procedure TfmMCX.RunSSHCmd(Sender: TObject; cmd: string; updategpu:boolean=false; doprogress: boolean=false);
var
    pass, host, username, url: string;
begin
    host:=fmConfig.cbHost.Text;
    username:=fmConfig.edUserName.Text;
    url:=username+'@'+host+':'+fmConfig.edPort.Text;
    pass:=PassList.Values[url];
    if(Length(pass)=0) then begin
            pass:=PasswordBox('SSH','Plese type your SSH password');
            PassList.Values[url]:=pass;
    end;
{$IFDEF USE_SYNAPSE}
    sshrun := TSSHThread.Create(host,fmConfig.edPort.Text,username,pass,cmd,doprogress,@pMCXTerminate,true);
    sshrun.isupdategpu:=updategpu;
    sshrun.OutputMemo:=mmOutput;
    sshrun.sbInfo:=sbInfo;
    sshrun.ProgressBar:=fmStop.pbProgress;
    sshrun.Resume;
{$ENDIF}
end;

procedure TfmMCX.mcxdoQueryExecute(Sender: TObject);
var
    cmd, url: string;
    AProcess : TProcess;
    Buffer   : string;
    BufStr   : string;
begin
    if(ResetMCX(0)) then begin
          AddLog('"-- Run Command --"');
          if(ckDoRemote.Checked) then begin
{$IFDEF USE_SYNAPSE}
              if(edRemote.ItemIndex=0) then
              begin
                url:=fmConfig.cbHost.Text;
                cmd:=fmConfig.edUserName.Text;
                if(url.IsEmpty) or (cmd.IsEmpty) then begin
                   if(MessageDlg('Question', 'You have not set up remote server information. Do you want to set now?', mtWarning,
                       [mbYes, mbNo, mbCancel],0) <> mrYes) then exit;
                   mcxdoConfigExecute(Sender);
                end;
                cmd:=CreateCmdOnly+' -L';
                RunSSHCmd(Sender, cmd, true, false);
                exit;
              end;
{$ENDIF}
              url:=ExpandPassword(edRemote.Text);
              if(sscanf(url,'%s',[@cmd])=1) then begin
                  pMCX.CommandLine:='"'+SearchForExe(cmd)+'"'+
                     Copy(url,Length(cmd)+1,Length(url)-
                     Length(cmd))+' '+ CreateCmdOnly+' -L';
                  AddLog('"'+SearchForExe(cmd)+'"'+
                     Copy(edRemote.Text,Length(cmd)+1,Length(edRemote.Text)-
                     Length(cmd))+' '+ CreateCmdOnly+' -L');
              end else
                  exit;
          end else begin
              cmd:=SearchForExe(CreateCmdOnly);
              if(Length(cmd)=0) then begin
                  MessageDlg('Warning', 'Program is not found', mtError, [mbOK],0);
                  exit;
              end;
              pMCX.CommandLine:='"'+cmd+'" -L';
              AddLog(pMCX.CommandLine);
          end;
          {$IFDEF DARWIN}
          AProcess := TProcess.Create(nil);
          try
            AProcess.CommandLine:=  pMCX.CommandLine;
            AProcess.Options := [poUsePipes,poStderrToOutput];
            AProcess.Execute;
            Buffer := '';
            repeat
              if AProcess.Output.NumBytesAvailable > 0 then
              begin
                SetLength(BufStr, AProcess.Output.NumBytesAvailable);
                AProcess.Output.Read(BufStr[1], Length(BufStr));

                BufStr:=StringReplace(BufStr,#8, '',[rfReplaceAll]);
                BufStr:=ReplaceRegExpr(#27'\[(\d+;)*\d+m',BufStr,'',false);
                Buffer := Buffer + BufStr;
              end;
            until not AProcess.Running;
          if AProcess.Output.NumBytesAvailable > 0 then
          begin
            SetLength(BufStr, AProcess.Output.NumBytesAvailable);
            AProcess.Output.Read(BufStr[1], Length(BufStr));
            BufStr:=StringReplace(BufStr,#8, '',[rfReplaceAll]);
            BufStr:=ReplaceRegExpr(#27'\[(\d+;)*\d+m',BufStr,'',false);
            Buffer := Buffer + BufStr;
            Application.ProcessMessages;
          end;
          finally
            AProcess.Free;
          end;
          AddLog('"-- Printing GPU Information --"');
          AddMultiLineLog(Buffer,pMCX);
          UpdateGPUList(Buffer);
          exit;
          {$ENDIF}
          sbInfo.Panels[0].Text := 'Status: querying GPU';
          pMCX.Tag:=-1;
          AddLog('"-- Printing GPU Information --"');
          pMCX.Execute;

          UpdateMCXActions(acMCX,'Run','');
    end;
end;
function TfmMCX.ResetMCX(exitcode: LongInt) : boolean;
begin
    Result:=false;
    if(pMCX.Running) then begin
        pMCX.Terminate(exitcode);
        AddLog('Program is still running, trying to terminate ...');
        Sleep(1000);
    end;
    if(pMCX.Running) then begin
        MessageDlg('Warning', 'Program is still running. Please wait or kill the program manually from your Task Manager', mtError, [mbOK],0);
        exit;
    end;
    Result:=true;
end;

procedure TfmMCX.mcxdoRunExecute(Sender: TObject);
var
    AProcess : TProcess;
    Buffer   : string;
    BufStr, url,cmd, fullcmd  : string;
    total: integer;
begin
    if(GetTickCount64-mcxdoRun.Tag<100) then
       exit;
    if(ResetMCX(0)) then begin
        fullcmd:=CreateCmd(pMCX);
        if(Sender=miExportJSON) then begin
            pMCX.Parameters.Add('--dumpjson');
            pMCX.Parameters.Add(edSession.Text+'_input.json');
            AddLog(pMCX.Executable+' '+pMCX.Parameters.CommaText);
        end;
        pMCX.CurrentDirectory:=ExtractFilePath(SearchForExe(CreateCmdOnly));
        AddLog('"-- Executing Simulation --"');
        if(ckbDebug.Checked[2]) then begin
            sbInfo.Panels[1].Text:='0%';
            sbInfo.Invalidate;
        end;

        fmStop.Show;

        mcxdoStop.Enabled:=true;
        mcxdoRun.Enabled:=false;
        sbInfo.Panels[0].Text := 'Status: running simulation';
        sbInfo.Panels[2].Text := '';
        pMCX.Tag:=-10;
        sbInfo.Color := clRed;

        UpdateMCXActions(acMCX,'Run','');
        mcxdoRun.Tag:=ptrint(GetTickCount64);
{$IFDEF USE_SYNAPSE}
        if(edRemote.ItemIndex=0) then
        begin
          url:=fmConfig.cbHost.Text;
          cmd:=fmConfig.edUserName.Text;
          if ckDoRemote.Checked and ( (url.IsEmpty) or (cmd.IsEmpty) ) then begin
             if(MessageDlg('Question', 'You have not set up remote server information. Do you want to set now?', mtWarning,
                 [mbYes, mbNo, mbCancel],0) <> mrYes) then exit;
             mcxdoConfigExecute(Sender);
          end;
          RunSSHCmd(Sender, fullcmd, false, ckbDebug.Checked[2]);
          exit;
        end;
{$ENDIF}

        {$IFDEF DARWIN}
        AProcess := TProcess.Create(nil);
        try
          AProcess.Executable:=  pMCX.Executable;
          AProcess.Parameters:=  pMCX.Parameters;
          AProcess.Options := [poUsePipes,poStderrToOutput];
          AProcess.Execute;
          Buffer := '';
          repeat
            if AProcess.Output.NumBytesAvailable > 0 then
            begin
              SetLength(BufStr, AProcess.Output.NumBytesAvailable);
              AProcess.Output.Read(BufStr[1], Length(BufStr));
              BufStr:=StringReplace(BufStr,#8, '',[rfReplaceAll]);
              BufStr:=ReplaceRegExpr(#27'\[(\d+;)*\d+m',BufStr,'',false);
              if (ckbDebug.Checked[2]) then begin
                       if RegEngine.Exec(ReverseString(BufStr)) then begin
                             if(sscanf(ReverseString(RegEngine.Match[0]),']%d\%', [@total])=1) then begin
                                sbInfo.Panels[1].Text:=Format('%d%%',[total]);
                                sbInfo.Tag:=total;
                                fmStop.pbProgress.Position:=total;
                                sbInfo.Repaint;
                                Application.ProcessMessages;
                             end;
                       end;
              end;
              Buffer := Buffer + BufStr;
              AddMultiLineLog(BufStr,pMCX);
              Application.ProcessMessages;
            end;
          until not AProcess.Running;
        if AProcess.Output.NumBytesAvailable > 0 then
        begin
          SetLength(BufStr, AProcess.Output.NumBytesAvailable);
          AProcess.Output.Read(BufStr[1], Length(BufStr));
          BufStr:=StringReplace(BufStr,#8, '',[rfReplaceAll]);
          BufStr:=ReplaceRegExpr(#27'\[(\d+;)*\d+m',BufStr,'',false);
          Buffer := Buffer + BufStr;
          AddMultiLineLog(BufStr,pMCX);
          Application.ProcessMessages;
        end;
        finally
          AProcess.Free;
        end;
        AddLog('"-- Command Completed --"');
        pMCXTerminate(nil);
        exit;
        {$ELSE}
        pMCX.Execute;
        Sleep(50);
        if(not pMCX.Running) then
           pMCXTerminate(pMCX);
        {$ENDIF}
        Application.ProcessMessages;
    end;
end;

procedure TfmMCX.mcxdoSaveExecute(Sender: TObject);
var
    TaskFile: string;
begin
  if(SaveProject.Execute) then begin
    TaskFile:=SaveProject.FileName;
    if(length(TaskFile) >0) then begin
        SaveTasksToIni(TaskFile);
        mcxdoSave.Enabled:=false;
    end;
  end;
end;

procedure TfmMCX.mcxdoStopExecute(Sender: TObject);
begin
{$IFDEF USE_SYNAPSE}
     if(ckDoRemote.Checked) and (sshrun<>nil) then
     begin
          sshrun.Terminate;
     end else if(pMCX.Running) then
{$ENDIF}
     begin
       pMCX.Terminate(0);
     end;
     Sleep(1000);
     if(not pMCX.Running) then begin
          mcxdoStop.Enabled:=false;
          if(mcxdoVerify.Enabled) then
             mcxdoRun.Enabled:=true;
          AddLog('"-- Terminated Simulation --"');
          if(ckbDebug.Checked[2]) then begin
              sbInfo.Panels[1].Text:='0%';
              sbInfo.Tag:=0;
              fmStop.pbProgress.Position:=0;
              sbInfo.Repaint;
          end;
          fmStop.Hide;
     end
end;

procedure TfmMCX.mcxdoToggleViewExecute(Sender: TObject);
begin
     if(lvJobs.ViewStyle=vsIcon) then
        lvJobs.ViewStyle:=vsSmallIcon
     else if (lvJobs.ViewStyle=vsSmallIcon) then
         lvJobs.ViewStyle:=vsReport
     else
         lvJobs.ViewStyle:=vsIcon;
end;

procedure TfmMCX.mcxdoVerifyExecute(Sender: TObject);
begin
    VerifyInput;
end;

procedure TfmMCX.LoadJSONShapeTree(shapejson: string);
var
    shaperoot: TTreeNode;
begin
    tvShapes.Enabled:=false;
    tvShapes.Items.BeginUpdate;
    tvShapes.Items.Clear;
    tvShapes.Items.EndUpdate;
    shaperoot:=tvShapes.Items.Add(nil,'Shapes');
    FreeAndNil(JSONData);
    JSONData:=GetJSON(shapejson);
    if(JSONData.FindPath('Shapes') <> nil) then
       ShowJSONData(shaperoot,JSONData.FindPath('Shapes'))
    else
       ShowJSONData(shaperoot,JSONData);
    tvShapes.Enabled:=true;
    tvShapes.FullExpand;
end;

procedure TfmMCX.FormCreate(Sender: TObject);
var
    i: integer;
    BrowserPath,BrowserParams, workdir: string;
begin
  {$IFDEF WINDOWS}
  with TRegistry.Create do
    try
      RootKey := HKEY_CURRENT_USER;
      if OpenKeyReadOnly('\Software\Classes\.mcxp')=false or Application.HasOption('r','registry') then begin
        if OpenKey('\Software\Classes\.mcxp', true) then
          WriteString('', 'MCXProject');
        if OpenKey('\Software\Classes\MCXProject', true) then
          WriteString('', 'MCX Project File');
        if OpenKey('\Software\Classes\MCXProject\DefaultIcon', true) then
          WriteString('', Application.ExeName);
        if OpenKey('\Software\Classes\MCXProject\shell\open\command', true) then
          WriteString('', Application.ExeName+' -p "%1"');
        SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nil, nil);
      end;
{        RootKey := HKEY_LOCAL_MACHINE;
        if OpenKeyReadOnly('\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe') then
           webBrowser.BrowserPath:=ReadString('');
}
    finally
      Free;
    end;
  {$ENDIF}

  {$IFDEF DARWIN}
    lvJobs.ViewStyle:=vsReport;
  {$ENDIF}
    DockMaster.MakeDockSite(Self,[akBottom,akLeft,akRight],admrpChild);

    fmDomain:=TfmDomain.Create(Self);
    fmConfig:=TfmConfig.Create(Self);
    fmStop:=TfmStop.Create(Self);
    fmDomain.FormStyle:=fsStayOnTop;
    fmStop.FormStyle:=fsStayOnTop;

    CurrentSession:=nil;
{$IFDEF USE_SYNAPSE}
    sshrun:=nil;
{$ENDIF}
    PassList:=TStringList.Create();
    MapList:=TStringList.Create();
    MapList.Clear;
    for i:=1 to lvJobs.Columns.Count-1 do begin
        MapList.Add(lvJobs.Columns.Items[i].Caption);
    end;

    JSONdata:=TJSONObject.Create;

    ConfigData:=TStringList.Create();
    ConfigData.Clear;
    ConfigData.CommaText:=sgConfig.Cols[2].CommaText;

    RegEngine:=TRegExpr.Create('%[0-9 ]{4}\]');

    btLoadSeed.Glyph.Assign(nil);
    JSONIcons.GetBitmap(2, btLoadSeed.Glyph);

    BCItemProp := TItemProp.Create(vlBC);
    BCItemProp.EditStyle := esPickList;
    BCItemProp.ReadOnly := True;
    BCItemProp.PickList.CommaText:='absorb,reflect,mirror,cyclic';

    vlBC.ItemProps['x-'] := BCItemProp;
    vlBC.ItemProps['x+'] := BCItemProp;
    vlBC.ItemProps['y-'] := BCItemProp;
    vlBC.ItemProps['y+'] := BCItemProp;
    vlBC.ItemProps['z-'] := BCItemProp;
    vlBC.ItemProps['z+'] := BCItemProp;

    ProfileChanged:=false;
    if not (SearchForExe(CreateCmdOnly) = '') then begin
        mcxdoQuery.Enabled:=true;
        mcxdoHelpOptions.Enabled:=true;
    end;
    LoadJSONShapeTree('[{"Grid":{"Tag":1,"Size":[60,60,60]}}]');
    if(Application.HasOption('p','project')) then
        LoadTasksFromIni(Application.GetOptionValue('p', 'project'));
    UseUserFolder:=false;
    if(Application.HasOption('u','user')) then
        UseUserFolder:=true;
    workdir:=GetAppRoot + 'MCXOutput';
    if(not DirectoryExists(workdir)) then
        ForceDirectories(workdir);
    if(not DirectoryExists(workdir)) then
        UseUserFolder:=true;
end;

procedure TfmMCX.FormDestroy(Sender: TObject);
begin
    MapList.Free;
    ConfigData.Free;
    RegEngine.Free;
    PassList.Free;
    BCItemProp.Free;

    fmDomain.Free;
    fmConfig.Free;
    fmStop.Free;

    FreeAndNil(JSONData);
end;

procedure TfmMCX.lvJobsSelectItem(Sender: TObject; Item: TListItem;
  Selected: Boolean);
begin
     if(Selected) then begin
          if not (lvJobs.Selected=nil) then begin
              mcxdoDeleteItem.Enabled:=true;
              mcxdoCopy.Enabled:=Selected;
              mcxdoPlotVol.Enabled:=Selected;
              mcxdoPlotMC2.Enabled:=Selected;
              mcxdoPlotNifty.Enabled:=Selected;
              mcxdoPlotMesh.Enabled:=Selected;
          end;
     end;
end;

procedure TfmMCX.RunExternalCmd(cmd: AnsiString);
var
  Proc : TProcess;
begin
  Proc := TProcess.Create(nil);
  try
    Proc.CommandLine := cmd;
    //PRoc.Options := Proc.Options + [poWaitOnExit];
    PRoc.Execute;
  finally
    Proc.free;
  end;
end;

function TfmMCX.ExpandPathMacro(path, app: string): string;
begin
  Result:=path;
  Result:=StringReplace(Result,'%MCXSTUDIO%', GetAppRoot,[rfReplaceAll]);
  Result:=StringReplace(Result,'%APP%', app, [rfReplaceAll]);
  Result:=StringReplace(Result,'$HOME', GetUserDir, [rfReplaceAll]);
  Result:=StringReplace(Result,'$PATH', GetEnvironmentVariable('PATH'), [rfReplaceAll]);
  {$IFDEF WINDOWS}
  Result:=StringReplace(Result,'/', DirectorySeparator, [rfReplaceAll]);
  {$ENDIF}
end;

function TfmMCX.GetAppRoot: string;
begin
      Result:=ExtractFilePath(Application.ExeName);
      {$IFDEF DARWIN}
      Result:=ReplaceRegExpr('/mcxstudio.app/Contents/MacOS/$',Result, '/', false);
      {$ENDIF}
end;

function TfmMCX.SearchForExe(fname : string; isremote: boolean = false) : string;
var
   path: string;
begin
   {$IFDEF WINDOWS}
   if(Pos('.cl',Trim(LowerCase(fname)))<=0) then
       if (Pos('.exe',Trim(LowerCase(fname)))<=0) or (Pos('.exe',Trim(LowerCase(fname))) <> Length(Trim(fname))-3) then
           fname:=fname+'.exe';
   {$ENDIF}

   if(fmConfig.ckUseManualPath.Checked) then begin
        if(fname='mcx') or (fname='mmc') or (fname='mcxcl') then begin
          if not (isremote) then begin
              path:=fmConfig.edLocalPath.Cols[0].CommaText;
              if(not path.IsEmpty) then begin
                  path:=ExpandPathMacro(path,fname);
                  Result := SearchFileInPath(fname, '', path,PathSeparator, [sffDontSearchInBasePath]);
                  if (not Result.IsEmpty) then exit;
              end;
          end else begin
              path:=fmConfig.edRemotePath.Text;
              if(not path.IsEmpty) and FileExists(path+DirectorySeparator+fname) then
                  Result:=path+DirectorySeparator+fname
              else
                  Result:=fname;
              exit;
          end;
        end else begin
              if(fname='ssh') or (fname='plink') then begin
                   path:=fmConfig.edSSHPath.Text;
                   if (not path.IsEmpty) and FileExists(path) then begin
                       Result:=path;
                       exit;
                   end;
              end else if (fname='scp') then begin
                    path:=fmConfig.edSCPPath.Text;
                    if (not path.IsEmpty) and FileExists(path) then begin
                        Result:=path;
                        exit;
                    end;
              end;

        end;
   end;

   Result :=
        SearchFileInPath(fname, '',
            GetAppRoot+'MCXSuite'+
            DirectorySeparator+MCProgram[grProgram.ItemIndex]+DirectorySeparator+
            'bin'+PathSeparator+GetAppRoot+PathSeparator+
            GetUserDir+DirectorySeparator+'MCXStudio'+PathSeparator+
            GetAppRoot+MCProgram[grProgram.ItemIndex]+
            DirectorySeparator+'bin'+PathSeparator+GetEnvironmentVariable('PATH'),
                         PathSeparator, [sffDontSearchInBasePath]);
   AddLog('EXEPATH='+Result);
end;

function TfmMCX.GetFileBrowserPath : string;
  {Return path to first browser found.}
begin
   Result := SearchForExe('xdg-open'); // linux
   if Result = '' then
     Result := SearchForExe('open'); // mac os
   if Result = '' then
     Result :=SearchForExe('explorer.exe');; // windows
end;

function TfmMCX.GetBrowserPath : AnsiString;
  {Return path to first browser found.}
var
   RootKey: string;
begin
   Result := SearchForExe('firefox');
   if Result = '' then begin
     {$IFDEF WINDOWS}
      with TRegistry.Create do
       try
         RootKey := HKEY_LOCAL_MACHINE;
         if OpenKeyReadOnly('\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe') then
           Result:=ReadString('');
       finally
         Free;
       end;
     {$ELSE}
       Result := SearchForExe('google-chrome');
     {$ENDIF}
   end;
   if Result = '' then
     Result := SearchForExe('konqueror');  {KDE browser}
   if Result = '' then
     Result := SearchForExe('epiphany');  {GNOME browser}
   if Result = '' then
     Result := SearchForExe('opera');
   if Result = '' then
     Result := SearchForExe('open'); // mac os
   if Result = '' then
     Result :='cmd /c start'; // windows
end;

procedure TfmMCX.mcxdoWebExecute(Sender: TObject);
begin
  OpenURL('http://mcx.space');
end;

procedure TfmMCX.mcxdoWebURLExecute(Sender: TObject);
begin
  if(Sender is TMenuItem) then begin
        OpenURL((Sender as TMenuItem).Hint);
  end;
end;

procedure TfmMCX.mcxSetCurrentExecute(Sender: TObject);
begin
     if not (lvJobs.Selected = nil) then begin
         lvJobs.Tag:=0; // loading
         SwapState(CurrentSession,lvJobs.Selected);
         CurrentSession:=lvJobs.Selected;
         ListToPanel2(CurrentSession);
         plSetting.Enabled:=true;
         pcSimuEditor.Enabled:=true;
         mcxdoVerify.Enabled:=true;
         mcxdoDefault.Enabled:=true;
         Caption:='MCX Studio - ['+CurrentSession.Caption+']';
         lvJobs.Tag:=0;
         //lvJobs.Selected.ImageIndex:=7;
     end;
end;

procedure TfmMCX.MenuItem22Click(Sender: TObject);
begin
  if(lvJobs.Selected <> nil) then
      RunExternalCmd('"'+GetFileBrowserPath + '" "'+CreateWorkFolder(lvJobs.Selected.Caption, true)+'"');
end;

procedure TfmMCX.MenuItem76Click(Sender: TObject);
var
   fmViewer: TfmViewer;
begin
  fmViewer:=TfmViewer.Create(self);
  fmViewer.BringToFront;
  fmViewer.Show;
end;

procedure TfmMCX.miExportJSONClick(Sender: TObject);
begin
  if(lvJobs.Selected <> nil) then
      mcxdoRunExecute(miExportJSON);
end;

procedure TfmMCX.miClearLogClick(Sender: TObject);
begin
  mmOutput.Lines.Clear;
end;

procedure TfmMCX.miCopyClick(Sender: TObject);
begin
   Clipboard.AsText:=mmOutput.SelText;
end;

procedure TfmMCX.miUseMatlabClick(Sender: TObject);
begin
     miUseMatlab.Checked:=not miUseMatlab.Checked;
end;

procedure TfmMCX.MenuItem9Click(Sender: TObject);
begin
    pcSimuEditor.ActivePage:=tabInputData;
end;

procedure TfmMCX.pMCXReadData(Sender: TObject);
begin
     AddMultiLineLog(GetMCXOutput(Sender), Sender);
     if ((Sender as TAsyncProcess)=pMCX)  and (not (pMCX.Running)) then
         pMCXTerminate(Sender);
end;

procedure TfmMCX.pMCXTerminate(Sender: TObject);
begin
  {$IFDEF USE_SYNAPSE}
     if (ckDoRemote.Checked) and (sshrun<>nil) and (sshrun.isupdategpu) then
         UpdateGPUList(sshrun.FullLog);
  {$ENDIF}
     if(not mcxdoStop.Enabled) then exit;
     if(Sender <> nil) and (Sender is TAsyncProcess) then
         AddMultiLineLog(GetMCXOutput(Sender), Sender);
     if(Sender <> nil) and (pMCX.Tag=-10) then begin
         sbInfo.Panels[2].Text:=Format('Last simulation used %.3f seconds', [(GetTickCount64-mcxdoRun.Tag)/1000.]);
         //if ckDoRemote.Checked and (not ckSharedFS.Checked) then
         //    mcxdoDownloadMC2Execute(Sender);
     end;

     fmStop.Hide;

     mcxdoRun.Tag:=0;
     mcxdoStop.Enabled:=false;
     if(mcxdoVerify.Enabled) then
         mcxdoRun.Enabled:=true;
     sbInfo.Panels[0].Text := 'Status: idle';
     pMCX.Tag:=0;
     sbInfo.Color := clBtnFace;
     AddLog('"-- Task completed --"');
     sbInfo.Panels[1].Text:='';
     sbInfo.Tag:=0;
     fmStop.pbProgress.Position:=0;
     sbInfo.Repaint;
     UpdateMCXActions(acMCX,'','Run');
end;

procedure TfmMCX.rbUseFileChange(Sender: TObject);
var
   btn: TRadioButton;
begin
   if(Sender is TRadioButton) then
   begin
        btn:= Sender as TRadioButton;
        if(btn.Checked) then
        begin
            edConfigFile.Enabled:=true;
            tabInputData.Enabled:=false;
        end else begin
            edConfigFile.Enabled:=false;
            tabInputData.Enabled:=true;
        end;
   end;
end;

procedure TfmMCX.sbInfoDrawPanel(StatusBar: TStatusBar; Panel: TStatusPanel;
  const Rect: TRect);
var
   perc: integer;
   newrect: TRect;
begin
   case Panel.Index of
      1: begin
          perc:=sbInfo.Tag;
          sbInfo.Canvas.Brush.style:= bsSolid;
          sbInfo.Canvas.Brush.color:= RGBToColor(230, 184, 156);
          newrect:=Rect;
{$IFDEF WINDOWS}
          newrect.Left:=sbInfo.Panels[0].Width;
          newrect.Right:=sbInfo.Panels[0].Width+sbInfo.Panels[1].Width;
{$ENDIF}
          newrect.Right:=Round(real((Rect.Right-Rect.Left)*perc)/100.0)+Rect.Left;
          sbInfo.Canvas.FillRect(newrect);
          sbInfo.Canvas.Brush.style:= bsClear;
          sbInfo.Canvas.TextOut(Rect.Left+Round(Real(Rect.Right-Rect.Left)*0.5)-Rect.Bottom,Rect.Top, sbInfo.Panels[1].Text);
      end;
   end;
end;

procedure TfmMCX.sgConfigDblClick(Sender: TObject);
var
   sg: TStringGrid;
begin

end;

procedure TfmMCX.sgConfigEditButtonClick(Sender: TObject);
var
   grid: TStringGrid;
   fmSrc: TfmSource;
   meshid: string;
begin
   if not(Sender is TStringGrid) then exit;
   grid:= Sender as TStringGrid;
   if(grid.Col=2) and (grid.Row=14) then begin
      if(OpenDir.Execute) then
         grid.Cells[grid.Col,grid.Row]:=OpenDir.FileName;
   end else if(grid.Col=2) and (grid.Row=1) then begin
      //if(grid.Cells[grid.Col,grid.Row]='See Volume Designer...') then
      //   pcSimuEditor.ActivePage:=tabVolumeDesigner
      //else
          if(OpenVolume.Execute) then begin
             if(grProgram.ItemIndex<>1) then begin
                 grid.Cells[grid.Col,grid.Row]:=OpenVolume.FileName
             end else begin
                 grid.Cells[2,14]:=ExtractFilePath(OpenVolume.FileName);
                 meshid:=ExtractFileName(OpenVolume.FileName);
                 meshid:=ReplaceRegExpr('^[a-z]{4}_',meshid,'',false);
                 grid.Cells[grid.Col,grid.Row]:=ReplaceRegExpr('.dat$',meshid,'',false);
             end;
          end;
   end else if(grid.Col=2) and (grid.Row>=10) and (grid.Row<=12) then begin
      fmSrc:=TfmSource.Create(Application);
      if(Length(grid.Cells[2,10])>0) then begin
             fmSrc.edSource.Text:=grid.Cells[2,10];
             fmSrc.edSourceEditingDone(fmSrc.edSource);
      end;
      if(fmSrc.ShowModal= mrOK) then begin
             grid.Cells[2,10]:=Trim(fmSrc.edSource.Text);
             grid.Cells[2,11]:='['+fmSrc.SrcParam1.CommaText+']';
             grid.Cells[2,12]:='['+fmSrc.SrcParam2.CommaText+']';
      end;
      fmSrc.Free;
   end;
   edRespinChange(Sender);
end;

procedure TfmMCX.sgConfigResize(Sender: TObject);
begin

end;

procedure TfmMCX.sgConfigSelectEditor(Sender: TObject; aCol, aRow: Integer;
  var Editor: TWinControl);
begin
   if (aCol=2) and (aRow=3) then begin
     if(grProgram.ItemIndex=1) then begin
       Editor := sgConfig.EditorByStyle(cbsPickList);
       TCustomComboBox(Editor).Items.Clear;
       TCustomComboBox(Editor).Items.Add('p - Plucker ray-tracer');
       TCustomComboBox(Editor).Items.Add('h - Havel ray-tracer');
       TCustomComboBox(Editor).Items.Add('b - Badouel ray-tracer');
       TCustomComboBox(Editor).Items.Add('s - Branchless Badouel ray-tracer');
       TCustomComboBox(Editor).Items.Add('g - Dual-grid MMC');
     end else begin
         Editor := sgConfig.EditorByStyle(cbsPickList);
         TCustomComboBox(Editor).Items.Clear;
         TCustomComboBox(Editor).Items.Add('byte - 1-byte integer');
         TCustomComboBox(Editor).Items.Add('short - 2-byte integer');
         TCustomComboBox(Editor).Items.Add('integer - 4-byte integer');
         TCustomComboBox(Editor).Items.Add('mixlabel - mix ratio of two labels at boundary');
         TCustomComboBox(Editor).Items.Add('labelplus - labels plus one continuous component');
         TCustomComboBox(Editor).Items.Add('muamus_float - per voxel mua/mus in float');
         TCustomComboBox(Editor).Items.Add('mua_float - per voxel mua in float');
         TCustomComboBox(Editor).Items.Add('muamus_half - per voxel mua/mus in half-float');
         TCustomComboBox(Editor).Items.Add('asgn_byte - per voxel mua/mus/g/n grayscale in byte');
         TCustomComboBox(Editor).Items.Add('muamus_short - per voxel mua/mus grayscale in short');
     end;
   end;
end;

procedure TfmMCX.sgMediaDrawCell(Sender: TObject; aCol, aRow: Integer;
  aRect: TRect; aState: TGridDrawState);
var
  LStrCell: string;
  LRect: TRect;
begin
  if(ACol>0) or (ARow=0) then exit;
  LStrCell := IntToStr(ARow-1);
  sgMedia.Canvas.FillRect(aRect); // clear the cell
  LRect := aRect;
  LRect.Top := LRect.Top + 3; // adjust top to center vertical
  // draw text
  DrawText(sgMedia.Canvas.Handle, PChar(LStrCell), Length(LStrCell), LRect, DT_CENTER);
end;

procedure TfmMCX.GotoColRow(grid: TStringGrid; Col, Row: Integer);
begin
  GotoCol := Col;
  GotoRow := Row;
  GotoGrid:= grid;
  Timer1.Enabled := True;
end;

procedure TfmMCX.sgMediaEditingDone(Sender: TObject);
var
     grid: TStringGrid;
     val: Extended;
begin
     if not(Sender is TStringGrid) then exit;
     grid:= Sender as TStringGrid;
     if(Length(grid.Cells[grid.Col,grid.Row])=0) then exit;
     if(not TryStrToFloat(grid.Cells[grid.Col,grid.Row], val)) then
     begin
        MessageDlg('Input Error', 'Input is not a number!', mtError, [mbOK],0);
        GotoColRow(grid, grid.Col,grid.Row);
        exit;
     end;
     edRespinChange(Sender);
end;

procedure TfmMCX.shapeAddBoxExecute(Sender: TObject);
begin
  AddShapes('Box',Format('Tag=%d|O=[30,30,30]|Size=[10,10,10]',[tvShapes.Tag+1]));
end;

procedure TfmMCX.shapeAddCylinderExecute(Sender: TObject);
begin
  AddShapes('Cylinder',Format('Tag=%d|C0=[30,30,0]|C1=[30,30,60]|R=5',[tvShapes.Tag+1]));
end;

procedure TfmMCX.AddShapesWindow(shapeid: string; defaultval: TStringList; node: TTreeNode);
var
   fmshape:TfmShapeEditor;
   ss: string;
   jdata: TJSONData;
begin
   if(node = nil) then exit;
   if(defaultval.Count=1) then begin
        ss:=InputBox('Edit Shape',shapeid, defaultval.Strings[0]);
        ss:=Trim(ss);
        if(Pos(',',ss)=0) then
            ss:='{"'+shapeid+'":"'+ss+'"}'
        else
            ss:='{"'+shapeid+'":'+ss+'}';
        jdata:= GetJSON(ss);
        TJSONArray(node.TreeView.Items[0].Data).Add(jdata);
        ShowJSONData(node,jdata,true);
        exit;
   end;

   fmshape:=TfmShapeEditor.Create(Application, defaultval);
   fmshape.Caption:='Add Shape: '+shapeid;
   if(Pos('Layer',shapeid)=2) or (Pos('Slab',shapeid)=2) then
        fmshape.plEditor.Options:=fmshape.plEditor.Options+[goAutoAddRows];
   if(fmshape.ShowModal= mrOK) then begin
        ss:=Trim(fmshape.JSON.DelimitedText);
        if(Pos('[',ss)=1) then
             ss:='{"'+shapeid+'":['+ss+']}'
        else
             ss:='{"'+shapeid+'":{'+ss+'}}';
        jdata:= GetJSON(ss);
        TJSONArray(node.TreeView.Items[0].Data).Add(jdata);
        ShowJSONData(node,jdata,true);
   end;
   fmshape.Free;
end;

procedure TfmMCX.AddShapes(shapeid: string; defaultval: string);
var
    fs: TStringList;
    node: TTreeNode;
begin
  try
    fs:=TStringList.Create;
    fs.StrictDelimiter := true;
    fs.Delimiter:='|';
    fs.DelimitedText:=defaultval;
    //node:=tvShapes.Items.AddChild(tvShapes.Items[0],shapeid);
    AddShapesWindow(shapeid, fs, tvShapes.Items[0]);
    edRespinChange(tvShapes);
  finally
    fs.Free;
  end;
end;

procedure TfmMCX.shapeAddGridExecute(Sender: TObject);
begin
     AddShapes('Grid',Format('Tag=%d|Size=[60,60,60]',[tvShapes.Tag+1]));
end;

procedure TfmMCX.shapeAddNameExecute(Sender: TObject);
begin
   AddShapes('Name','mcxdomain');
end;

procedure TfmMCX.shapeAddOriginExecute(Sender: TObject);
begin
   AddShapes('Origin','[0,0,0]');
end;

procedure TfmMCX.shapeAddSphereExecute(Sender: TObject);
begin
  AddShapes('Sphere',Format('Tag=%d|O=[30,30,30]|R=10',[tvShapes.Tag+1]));
end;

procedure TfmMCX.shapeAddSubgridExecute(Sender: TObject);
begin
  AddShapes('Subgrid',Format('Tag=%d|O=[30,30,30]|Size=[10,10,10]',[tvShapes.Tag+1]));
end;

procedure TfmMCX.shapeAddUpperSpaceExecute(Sender: TObject);
begin
  AddShapes('UpperSpace',Format('Tag=%d|Coef=[1,-1,0,0]|Equ="Ax+By+Cz>D"',[tvShapes.Tag+1]));
end;

procedure TfmMCX.shapeAddXLayersExecute(Sender: TObject);
begin
   AddShapes('XLayers',Format('Layer 1=[1,10,%d]|Layer 2=[11,30,%d]|Layer 3=[31,50,%d]',[tvShapes.Tag+1,tvShapes.Tag+2,tvShapes.Tag+3]));
end;

procedure TfmMCX.shapeAddXSlabsExecute(Sender: TObject);
begin
  AddShapes('XSlabs',Format('Tag=%d|Bound=[1,10]',[tvShapes.Tag+1]));
end;

procedure TfmMCX.shapeAddYLayersExecute(Sender: TObject);
begin
   AddShapes('YLayers',Format('Layer 1=[1,10,%d]|Layer 2=[11,30,%d]|Layer 3=[31,50,%d]',[tvShapes.Tag+1,tvShapes.Tag+2,tvShapes.Tag+3]));
end;

procedure TfmMCX.shapeAddYSlabsExecute(Sender: TObject);
begin
   AddShapes('YSlabs',Format('Tag=%d|Bound=[1,10]',[tvShapes.Tag+1]));
end;

procedure TfmMCX.shapeAddZLayersExecute(Sender: TObject);
begin
  AddShapes('ZLayers',Format('Layer 1=[1,10,%d]|Layer 2=[11,30,%d]|Layer 3=[31,50,%d]',[tvShapes.Tag+1,tvShapes.Tag+2,tvShapes.Tag+3]));
end;

procedure TfmMCX.shapeAddZSlabsExecute(Sender: TObject);
begin
  AddShapes('ZSlabs',Format('Tag=%d|Bound=[1,10]',[tvShapes.Tag+1]));
end;

function TfmMCX.RebuildLayeredObj(rootdata: TJSONData; out maxtag: integer): AnsiString;
var
    i,j: integer;
    val: extended;
    row, jobj: TJSONArray;
    root: TJSONObject;
begin
    if(rootdata = nil) then exit;
    root:=TJSONObject(TJSONObject(rootdata).Items[0]);
    jobj:= TJSONArray.Create;
    maxtag:=0;
    try
      for i:=0 to root.Count-1 do begin
        if(Pos('Layer ',root.Names[i])=1) then begin
           row:= TJSONArray.Create;
           for j:=0 to root.Items[i].Count-1 do begin
               val:=TJSONArray(root.Items[i]).Floats[j];
               if j=root.Items[i].Count-1 then
                   maxtag:=max(Round(val),maxtag);
               if(Frac(val)=0) then
                   row.Add(Round(val))
               else
                   row.Add(val);
           end;
           jobj.Add(row);
        end;
      end;
      Result:=jobj.AsJSON;
    finally
      FreeAndNil(jobj);
    end;
end;
// handle shape data without "Shapes" root node;
// root should always be an array
function TfmMCX.UpdateShapeTag(root: TJSONData; doexport: boolean = false): integer;
var
     i, j, maxlayertag, maxtag, lastgood: integer;
     ss: string;
     jobj: TJSONObject;
begin
     maxtag:=0;
     ss:= root.AsJSON;
     if(root.JSONType <> jtArray) then begin
        MessageDlg('JSON Error','Shape data root node should always be an array', mtError, [mbOK],0);
        exit;
     end;
     for i:=0 to root.Count-1 do begin
       jobj:=TJSONObject(root.Items[i]);
       ss:=jobj.AsJSON;
       if (jobj.Count=1) and (Pos('Layers',jobj.Names[0])=2) then begin
            if(doexport) then begin
                jobj.Objects[jobj.Names[0]]:= TJSONObject(GetJSON(RebuildLayeredObj(jobj,maxlayertag)));
            end else begin
                RebuildLayeredObj(jobj,maxlayertag);
            end;
            maxtag:=Max(maxtag,maxlayertag);
       end;
       if(jobj.Count=1) and (jobj.Items[0].Count>0) then
           jobj:=TJSONObject(jobj.Items[0]);
       ss:=jobj.AsJSON;

       if(jobj.FindPath('Tag') <> nil) then
          maxtag:=Max(maxtag,jobj.Integers['Tag']);
       if (root.JSONType=jtObject) then begin
          if (TJSONObject(root).Names[i]='Grid') and (jobj.FindPath('Size') <> nil) then
              sgConfig.Cells[2,2]:=jobj.Arrays['Size'].AsJSON;
       end else if (root.JSONType=jtArray) then begin
          if (jobj.Count>0) and (TJSONObject(root.Items[i]).Names[0]='Grid') and (jobj.FindPath('Size') <> nil) then
              sgConfig.Cells[2,2]:=jobj.FindPath('Size').AsJSON;
       end;
     end;

     lastgood:=0;
     for i:=sgMedia.RowCount-1 downto 1 do begin
         if(Length(sgMedia.Cells[1,i])>0) then begin
             lastgood:=i;
             break;
         end;
     end;

     if(maxtag > lastgood-2) then begin
          sgMedia.RowCount:=Max(maxtag+3,sgMedia.RowCount);
          for i:=lastgood+1 to maxtag+1 do begin
              AddLog('"WARNING:" copying media type #'+IntToStr(lastgood-1)+' to new media type #'+IntToStr(i-1));
              AddLog('Please edit the media setting to customize.');
              for j:=1 to sgMedia.ColCount-1 do begin
                  sgMedia.Cells[j,i]:=sgMedia.Cells[j,lastgood];
              end;
          end;
     end;

     edRespinChange(sgConfig);
     Result:=maxtag;
end;

procedure TfmMCX.shapePrintExecute(Sender: TObject);
begin
    if(tvShapes.Selected <> nil) then
       if(tvShapes.Selected=tvShapes.Items[0]) then begin
           AddMultiLineLog(JSONData.FormatJSON, pMCX);
       end else begin
           if(tvShapes.Selected.Data <> nil) then
               AddMultiLineLog(TJSONData(tvShapes.Selected.Data).FormatJSON, pMCX);
       end;
end;

procedure TfmMCX.shapePreviewExecute(Sender: TObject);
var
    shapejson: TJSONData;
    cmd: TStringList;
begin
    shapejson:=GetJSON(SaveJSONConfig(''));

    cmd:=TStringList.Create;
    cmd.Add('%%%%%%%%% MATLAB/OCTAVE PLOTTING SCRIPT %%%%%%%%%');
    cmd.Add(Format('addpath(''%s'');',[GetAppRoot+
        'MATLAB'+DirectorySeparator+'mcxlab']));
    cmd.Add(Format('cfg=json2mcx(''%s'')',[CreateWorkFolder(edSession.Text)+DirectorySeparator+Trim(edSession.Text)+'.json']));
    cmd.Add('mcxpreview(cfg)');
    cmd.Add('%%%%%%%%% END PLOTTING SCRIPT %%%%%%%%%');
    cmd.Delimiter:=#10;
    AddMultiLineLog(cmd.DelimitedText,pMCX);
    cmd.Free;

    if(miUseMatlab.Checked) then exit;

    try
        fmDomain.mmShapeJSON.Lines.Text:=shapejson.FormatJSON;
        freeandnil(shapejson);
        fmDomain.Show;
    except
        on E: Exception do
           ShowMessage('OpenGL Error: '+E.ClassName+#13#10 + E.Message);
    end;
end;


procedure TfmMCX.shapeResetExecute(Sender: TObject);
var
  ret:TModalResult;
begin
  ret:=MessageDlg('Confirmation', 'The current shape has not been saved, are you sure you want to clear?',
        mtConfirmation, [mbYes, mbNo, mbCancel],0);
  if (ret=IDYES) then begin
    LoadJSONShapeTree('[{"Grid":{"Tag":1,"Size":[60,60,60]}}]');
    edRespinChange(tvShapes);
  end;
  if (ret=mrCancel) then
       exit;
end;

procedure TfmMCX.shapeDeleteExecute(Sender: TObject);
var
     P: TJSONData;
     Node: TTreeNode;
begin
  Node:= tvShapes.Selected;
  if(Node <> nil) then begin
        if(Node.Parent = nil) or (Node.Parent.Data=nil) or (Node.Data=nil) then exit;
        P:=TJSONData(Node.Parent.Data);
        //ss:=P.AsJSON;
        If P.JSONType=jtArray then
          TJSONArray(P).Remove(P.Items[Node.Index])
        else If P.JSONType=jtObject then
          TJSONObject(P).Remove(P.Items[Node.Index]);
        //ss:=P.AsJSON;
        Node.Delete;
        edRespinChange(tvShapes);
  end;
end;

procedure TfmMCX.Splitter6CanOffset(Sender: TObject; var NewOffset: Integer;
  var Accept: Boolean);
begin
  grAdvSettings.Height:=grAdvSettings.Height+NewOffset;
  Accept:=true;
end;

procedure TfmMCX.Splitter6CanResize(Sender: TObject; var NewSize: Integer;
  var Accept: Boolean);
begin
    grAdvSettings.Height:=NewSize;
    Accept:=true;
end;

procedure TfmMCX.StaticText2DblClick(Sender: TObject);
begin
    GridToStr(sgMedia);
end;

procedure TfmMCX.tbtRunClick(Sender: TObject);
begin
  {$IFNDEF DARWIN}
  mcxdoRunExecute(Sender);
  {$ENDIF}
end;

procedure TfmMCX.Timer1Timer(Sender: TObject);
begin
  Timer1.Enabled := False;
  if(Assigned(GotoGrid)) then begin
      GotoGrid.Col := GotoCol;
      GotoGrid.Row := GotoRow;
      GotoGrid.EditorMode := True;
  end;
end;

procedure TfmMCX.tmAnimationTimer(Sender: TObject);
begin
    if not (Assigned(GotoGBox)) then exit;
    if(tmAnimation.Tag>0) then begin // collapse
         GotoGBox.Height:=GotoGBox.Height-10;
         if(GotoGBox.Height<=self.Canvas.TextHeight('Ag')+btGBExpand.Height+2) then begin
              //GotoGBox.Align:=alTop;
              GotoGBox.Height:=self.Canvas.TextHeight('Ag')+btGBExpand.Height+2;
              tmAnimation.Tag:=0;
              tmAnimation.Enabled:=false;
         end;
    end else begin
        GotoGBox.Height:=GotoGBox.Height+10;
        if(GotoGBox.Height>=edMoreParam.Top+edMoreParam.Height+self.Canvas.TextHeight('Ag')+5) then begin
             //GotoGBox.Align:=alClient;
             GotoGBox.Height:=edMoreParam.Top+edMoreParam.Height+self.Canvas.TextHeight('Ag')+5;
             tmAnimation.Tag:=1;
             tmAnimation.Enabled:=false;
        end;
    end;
end;

procedure TfmMCX.tvShapesEdited(Sender: TObject; Node: TTreeNode; var S: string
  );
var
     val: extended;
begin
     if(Node.Parent= nil) then exit;

     if(Node.ImageIndex=ImageTypeMap[jtNumber]) then begin
         if(not TryStrToFloat(S, val)) then begin
             MessageDlg('Input Error', 'The field must be a number', mtError, [mbOK],0);
             S:= Node.Text;
             exit;
         end;
         if (Pos('Layer ',Node.Parent.Text)=0) then begin
              if(TJSONData(Node.Parent.Data).JSONType=jtArray) then begin
                  if(Frac(val)=0) then
                      TJSONArray(Node.Parent.Data).Integers[Node.Index]:=Round(val)
                  else
                      TJSONArray(Node.Parent.Data).Floats[Node.Index]:=val;
              end else if TJSONData(Node.Parent.Data).JSONType=jtNumber then begin
                  if(Frac(val)=0) then
                      TJSONData(Node.Parent.Data).Value:=Round(val)
                  else
                      TJSONData(Node.Parent.Data).Value:=val;
              end;
         end else begin
              if(Frac(val)=0) then
                 TJSONArray(TJSONArray(Node.Parent.Data).Items[Node.Parent.Index]).Integers[Node.Index]:=Round(val)
              else
                 TJSONArray(TJSONArray(Node.Parent.Data).Items[Node.Parent.Index]).Floats[Node.Index]:=val;
         end;
     end else if(Node.ImageIndex=ImageTypeMap[jtString]) then begin
          if(Length(S)=0) then begin
               MessageDlg('Input Error', 'Input string can not be empty', mtError, [mbOK],0);
               S:=Node.Text;
               exit;
          end;
          if(Node.Parent <> nil) then begin
              if(TJSONData(Node.Parent.Data).JSONType=jtArray) then begin
                  TJSONArray(Node.Parent.Data).Strings[Node.Index]:=S;
              end else if TJSONData(Node.Parent.Data).JSONType=jtString then begin
                  //ss:= TJSONData(Node.Parent.Data).AsJSON;
                  TJSONData(Node.Parent.Data).Value:=S;
              end;
          end;
     end;
     edRespinChange(Sender);
end;

procedure TfmMCX.tvShapesSelectionChanged(Sender: TObject);
begin
    if(tvShapes.Selected <> nil) then  begin
       shapeDelete.Enabled:=tvShapes.Selected.Level=1;
       shapeEdit.Enabled:=tvShapes.Selected.Level=1;
       if(tvShapes.Selected.Count=0) then begin
           tvShapes.Options:=tvShapes.Options-[tvoReadOnly];
       end else begin
           tvShapes.Options:=tvShapes.Options+[tvoReadOnly];
       end;
    end;
end;

procedure TfmMCX.vlBCGetPickList(Sender: TObject; const KeyName: string;
  Values: TStrings);
begin
end;

procedure TfmMCX.UpdateGPUList(Buffer:string);
var
    list: TStringList;
    i, idx, total, namepos,gpucount: integer;
    gpuname, ss: string;

    {$IFDEF WINDOWS}
    Reg: TRegistry;
    RegKey: DWORD;
    Key: string;
    needfix: boolean;
    {$ENDIF}
begin
  list:=TStringList.Create;
  list.StrictDelimiter := true;
  list.Delimiter:=AnsiChar(#10);
  list.DelimitedText:=Buffer;
  gpucount:=0;
  for i:=0 to list.Count-1 do begin
    ss:= list.Strings[i];
    if(sscanf(ss+' ','Device %d of %d:%s', [@idx, @total, @gpuname])=3) then
    begin
           if(idx=1) then
               edGPUID.Items.Clear;
           namepos := Pos(gpuname, ss);
           edGPUID.Items.Add('#'+IntToStr(idx)+':'+Trim(copy(ss, namepos, Length(ss)-namepos+1)));
           gpucount:=gpucount+1;
    end;
  end;
  if(edGPUID.Items.Count>0) then
      edGPUID.Checked[0]:=true;
  {$IFDEF WINDOWS}
  if (not (ckDoRemote.Checked)) and (gpucount>=1) then begin
      Reg := TRegistry.Create;
      needfix:=true;
      try
        Reg.RootKey := HKEY_LOCAL_MACHINE;
        Key := '\SYSTEM\CurrentControlSet\Control\GraphicsDrivers';
        if Reg.OpenKeyReadOnly(Key) then
        begin
          if Reg.ValueExists('TdrDelay') then
          begin
            RegKey := Reg.ReadInteger('TdrDelay');
            needfix:=false;
          end;
        end;
        Reg.CloseKey;
        if(needfix) then begin
          if(MessageDlg('Question', 'If you run MCX on the GPU that is connected to your monitor, you may encouter an "Unspecified launch failure " error. Do you want to modify the "TdrDelay" registry key to allow MCX to run for more than 5 seconds?', mtWarning,
                  [mbYes, mbNo, mbCancel],0) = mrYes) then begin
                if Reg.OpenKey(Key, true) then  begin
                    Reg.WriteInteger('TdrDelay', 999999);
                    if(MessageDlg('Confirmation', 'Registry modification was successfully applied. You MUST reboot the computer to activate the settings, select Yes to reboot (strongly recommended), No to reboot manually.',
                         mtInformation, [mbYes, mbNo],0) = mrYes) then
                        RunExternalCmd('shutdown /r');
                end else
                    MessageDlg('Permission Error', 'You don''t have permission to modify registry. Please restart the program by right-clicking mcxstudio.exe and select "Run as Administrator"', mtError, [mbOK],0);
            end;
        end;
      finally
        Reg.Free
      end;
  end;
  {$ENDIF}
  list.Free;
end;

function TfmMCX.GetMCXOutput(Sender: TObject): string;
var
    Buffer, revbuf, percent: string;
    BytesAvailable: DWord;
    BytesRead:LongInt;

    total: integer;
    proc: TAsyncProcess;
begin
   if (Sender is TAsyncProcess) then
    proc:= Sender as TAsyncProcess;
    begin
      BytesAvailable := proc.Output.NumBytesAvailable;
      BytesRead := 0;
      while BytesAvailable>0 do
      begin
        SetLength(Buffer, BytesAvailable);
        BytesRead := proc.OutPut.Read(Buffer[1], BytesAvailable);
        Buffer:=StringReplace(Buffer,#8, '',[rfReplaceAll]);
        Buffer:=ReplaceRegExpr(#27'\[(\d+;)*\d+m',Buffer,'',false);
        //Buffer:=ReplaceRegExpr('\%Progress:',Buffer,'\%'+#13+'Progress',false);
        if(proc=pMCX) and (ckbDebug.Checked[2]) then begin
               revbuf:=ReverseString(Buffer);
               if RegEngine.Exec(revbuf) then begin
                     percent:=ReverseString(RegEngine.Match[0]);
                     if(sscanf(percent,']%d\%', [@total])=1) then begin
                        sbInfo.Panels[1].Text:=Format('%d%%',[total]);
                        sbInfo.Tag:=total;
                        fmStop.pbProgress.Position:=total;
                        sbInfo.Repaint;
                        Application.ProcessMessages;
                     end;
               end;
        end;
        Result := Result + copy(Buffer,1, BytesRead);
        BytesAvailable := proc.Output.NumBytesAvailable;
        Application.ProcessMessages;
      end;
    end;
    if(proc=pMCX) and (pMCX.Tag=-1) then begin
        UpdateGPUList(Buffer);
    end;
    Sleep(500);
end;

procedure TfmMCX.SaveTasksToIni(fname: string);
var
   inifile: TIniFile;
   i,j: integer;
begin
     DeleteFile(fname);
     inifile:=TIniFile.Create(fname);
     for i:=0 to lvJobs.Items.Count-1 do begin
          for j:=1 to lvJobs.Columns.Count-1 do begin
              inifile.WriteString(lvJobs.Items.Item[i].Caption,
                                  lvJobs.Columns.Items[j].Caption,
                                  lvJobs.Items.Item[i].SubItems.Strings[j-1]);
          end;
     end;
     inifile.UpdateFile;
     inifile.Free;
end;

procedure TfmMCX.LoadTasksFromIni(fname: string);
var
   sessions,vals:TStringList;
   inifile: TIniFile;
   i,j: integer;
   node: TListItem;
begin
     sessions:=TStringList.Create;
     vals:=TStringList.Create;
     inifile:=TIniFile.Create(fname);
     inifile.ReadSections(sessions);
     for i:=0 to sessions.Count-1 do begin
         node:=lvJobs.Items.Add;
         node.Caption:=sessions.Strings[i];
         node.ImageIndex:=0;
         inifile.ReadSectionValues(node.Caption,vals);
         for j:=1 to lvJobs.Columns.Count-1 do
             node.SubItems.Add('');
         for j:=1 to lvJobs.Columns.Count-1 do begin
             node.SubItems.Strings[j-1]:=vals.Values[lvJobs.Columns.Items[j].Caption];
         end;
         if(Length(vals.Values['MCProgram'])>0) then
             node.ImageIndex:=StrToInt(vals.Values['MCProgram']);
     end;
     inifile.Free;
     vals.Free;
     sessions.Free;
     AddLog(Format('Successfully loaded project %s. Please double click on session list to edit.',[fname]));
end;


procedure TfmMCX.VerifyInput;
var
    nthread, nblock: integer;
    nphoton, radius: extended;
    exepath: string;
begin
  try
    if rbUseFile.Checked and (Length(edConfigFile.FileName)=0) then
        raise Exception.Create('Config file must be specified');
    if rbUseFile.Checked and (not FileExists(edConfigFile.FileName)) then
        raise Exception.Create('Config file does not exist, please check the path');
    if ckDoReplay.Checked and (not FileExists(edSeed.Text)) then
        raise Exception.Create('An existing MCH file must be set as the seed when replay is desired');

    try
        nthread:=StrToInt(edThread.Text);
        nphoton:=StrToFloat(edPhoton.Text);
        radius:=StrToFloat(edBubble.Text);
        nblock:=StrToInt(edBlockSize.Text);
    except
        raise Exception.Create('Invalid numbers: check the values for thread (-t), block (-T), photon (-n) and cache radius (-R)');
    end;
    if(nthread<512) then
       AddLog('Warning: using over 20000 threads (-t) can usually give you the best speed');
    if(nphoton>1e9) then
       AddLog('Warning: you can increase respin number (-r) to get more photons');
    if(nblock<0) then
       raise Exception.Create('Thread block number (-T) can not be negative');

    exepath:=SearchForExe(CreateCmdOnly);
    if(exepath='') and (not ckDoRemote.Checked) then
       raise Exception.Create(Format('Can not find %s executable in the search path',[CreateCmdOnly]));

    if not (SaveJSONConfig('')='') then  begin
       UpdateMCXActions(acMCX,'Work','');
       AddLog('"-- Input is valid, please click [Run] to execute --"');

       if (MessageDlg('Confirmation', 'Input is valid. Do you want to execute the simulation?', mtConfirmation, [mbYes, mbCancel],0)=mrYes) then
           mcxdoRunExecute(nil);
    end;
  except
    On E : Exception do
      MessageDlg('Input Error', E.Message, mtError, [mbOK],0);
  end;
end;

function TfmMCX.CreateCmdOnly:string;
var
    cmd: string;
begin
    cmd:=MCProgram[grProgram.ItemIndex];
    Result:=cmd;
end;
function TfmMCX.SaveJSONConfig(filename: string): AnsiString;
var
    nthread, nblock,hitmax,seed, i: integer;
    bubbleradius,unitinmm,nphoton: extended;
    gpuid, section, key, val, ss, formatid: string;
    json, jobj, jmedium, jdet, joptode : TJSONObject;
    jdets, jmedia, jshape: TJSONArray;
    jsonlist: TStringList;
begin
  Result:='';
  try
      nthread:=StrToInt(edThread.Text);
      nphoton:=StrToFloat(edPhoton.Text);
      nblock:=StrToInt(edBlockSize.Text);
      bubbleradius:=StrToFloat(edBubble.Text);
      gpuid:=CheckListToStr(edGPUID);
      unitinmm:=StrToFloat(edUnitInMM.Text);
      hitmax:=StrToInt(edDetectedNum.Text);
      if not (ckDoReplay.Checked) then
          seed:=StrToInt(edSeed.Text);
  except
      MessageDlg('Input Error','Invalid numbers: check the values for thread, block, photon and time gate settings', mtError, [mbOK],0);
      exit;
  end;

  try
    try
      json:=TJSONObject.Create;

      if(json.Find('Session') = nil) then
          json.Objects['Session']:=TJSONObject.Create;
      jobj:= json.Objects['Session'];
      jobj.Floats['Photons']:=nphoton;
      if not (ckDoReplay.Checked) then
          jobj.Integers['RNGSeed']:=seed
      else
          jobj.Strings['RNGSeed']:=edSeed.Text;
      jobj.Strings['ID']:=edSession.Text;
      jobj.Integers['DoMismatch']:=Integer(ckReflect.Checked);
      jobj.Integers['DoNormalize']:=Integer(ckNormalize.Checked);
      jobj.Integers['DoPartialPath']:=Integer(ckSaveDetector.Checked);
      jobj.Integers['DoSaveSeed']:=Integer(ckSaveSeed.Checked);
      jobj.Integers['DoSaveRef']:=Integer(ckSaveRef.Checked);
      jobj.Strings['OutputType']:=OutputTypeFlags[edOutputType.ItemIndex+1];

      if(json.Find('Domain') = nil) then
          json.Objects['Domain']:=TJSONObject.Create;
      jobj:= json.Objects['Domain'];
      jobj.Integers['OriginType']:=Integer(ckSrcFrom0.Checked);
      jobj.Floats['LengthUnit']:=unitinmm;

      jmedia:=TJSONArray.Create;
      for i := sgMedia.FixedRows to sgMedia.RowCount - 1 do
      begin
              if (Length(sgMedia.Cells[1,i])=0) then break;
              jmedium:=TJSONObject.Create;
              jmedium.Add('mua',StrToFloat(sgMedia.Cells[1,i]));
              jmedium.Add('mus',StrToFloat(sgMedia.Cells[2,i]));
              jmedium.Add('g',StrToFloat(sgMedia.Cells[3,i]));
              jmedium.Add('n',StrToFloat(sgMedia.Cells[4,i]));
              jmedia.Add(jmedium);
      end;
      if(jmedia.Count>0) then
         jobj.Arrays['Media']:=jmedia
      else
         jmedia.Free;

      jdets:=TJSONArray.Create;
      for i := sgDet.FixedRows to sgDet.RowCount - 1 do
      begin
              if (Length(sgDet.Cells[1,i])=0) then break;
              jdet:=TJSONObject.Create;
              jdet.Arrays['Pos']:=TJSONArray.Create;
              jdet.Arrays['Pos'].Add(StrToFloat(sgDet.Cells[1,i]));
              jdet.Arrays['Pos'].Add(StrToFloat(sgDet.Cells[2,i]));
              jdet.Arrays['Pos'].Add(StrToFloat(sgDet.Cells[3,i]));
              jdet.Add('R',StrToFloat(sgDet.Cells[4,i]));
              jdets.Add(jdet);
      end;
      if(json.Find('Optode') = nil) then
         json.Objects['Optode']:=TJSONObject.Create
      else
         jdets.Free;

      joptode:=json.Objects['Optode'];

      if(ckSaveDetector.Checked) and (jdets.Count=0) then begin
          raise Exception.Create('You ask for saving detected photon data, but no detector is defined');
      end;
      if(jdets.Count>0) then
          joptode.Arrays['Detector']:=jdets;
      joptode.Objects['Source']:=TJSONObject.Create;

      json.Objects['Forward']:=TJSONObject.Create;
      for i := sgConfig.FixedRows to sgConfig.RowCount - 1 do
      begin
              if(Length(sgConfig.Cells[0,i])=0) and (i>sgConfig.FixedRows) then break;
              val:=sgConfig.Cells[2,i];
              if(Length(val)=0) and (i>sgConfig.FixedRows) then continue;
              section:= sgConfig.Cells[0,i];
              key:=sgConfig.Cells[1,i];
              if(section = 'Forward') then begin
                  json.Objects['Forward'].Floats[key]:=StrToFloat(val);
              end else if(section = 'Session') then begin
                  json.Objects['Session'].Strings[key]:=val;
              end else if(section = 'Domain') then begin
                  if (key='MeshID') then begin
                     json.Objects[section].Strings[key]:=val;
                     continue;
                  end else if(key='InitElem') then begin
                     json.Objects[section].Integers[key]:=StrToInt(val);
                     continue;
                  end else if(key='MediaFormat') or (key='RayTracer') then begin
                     if(sscanf(val,'%s',[@formatid])=1) then
                        json.Objects[section].Strings[key]:=formatid;
                     continue;
                  end;
                  if (key = 'VolumeFile') and ((val='See Volume Designer...') or (Length(val)=0)) then begin
                      if(JSONData.FindPath('Shapes') = nil) then
                          jshape:=TJSONArray(GetJSON(JSONData.AsJSON))
                      else
                          jshape:=TJSONArray(GetJSON(JSONData.FindPath('Shapes').AsJSON));
                      ss:=jshape.AsJSON;
                      tvShapes.Tag:=UpdateShapeTag(jshape,true);
                      json.Arrays['Shapes']:=jshape;
                      if(jmedia.Count<=tvShapes.Tag) then begin
                        raise Exception.Create('Insufficent media are defined');
                      end;
                  end else begin
                      if(key = 'VolumeFile') then begin
                          json.Objects['Domain'].Strings[key]:=ExtractRelativepath(ExtractFilePath(filename),ExtractFilePath(val))+DirectorySeparator+ExtractFilename(val)
                      end else
                          json.Objects['Domain'].Objects[key]:=TJSONObject(GetJSON(val));
                  end;
              end else if(section = 'Optode.Source') then begin
                  if (key = 'Type') then begin
                      joptode.Objects['Source'].Strings[key]:=val;
                  end else begin
                      joptode.Objects['Source'].Objects[key]:=TJSONObject(GetJSON(val));
                  end;
              end;
      end;

      AddLog('"-- JSON Input: --"');
      AddMultiLineLog(json.FormatJSON, pMCX);

      if(Length(filename)>0) then begin
          jsonlist:=TStringList.Create;
          jsonlist.Text:=json.FormatJSON;
          try
              jsonlist.SaveToFile(filename);
          finally
              jsonlist.Free;
          end;
      end;
      Result:=json.AsJSON;
    except
      on E: Exception do
          MessageDlg('Configuration Error', E.Message, mtError, [mbOK],0);
    end;
  finally
    FreeAndNil(json);
  end;
end;

function TfmMCX.CreateWorkFolder(session: string; iscreate: boolean=true) : string;
var
    path: string;
begin
    if(UseUserFolder) then
        path:=GetUserDir
           +'MCXOutput'+DirectorySeparator+CreateCmdOnly+'sessions'+DirectorySeparator+session
    else
        path:=GetAppRoot
            +'MCXOutput'+DirectorySeparator+CreateCmdOnly+'sessions'+DirectorySeparator+session;

    if fmConfig.ckUseManualPath.Checked then begin
         path:=fmConfig.edWorkPath.Text;
         path:=ExpandPathMacro(path,session);
         path:=path+DirectorySeparator+CreateCmdOnly+'sessions'+DirectorySeparator+session;
    end;
    path:=ExpandFileName(path);
    Result:=path;
    AddLog(Result);
    if(iscreate) then begin
        try
          if(not DirectoryExists(path)) then
              ForceDirectories(path);
        except
              On E : Exception do
                  MessageDlg('Input Error', E.Message, mtError, [mbOK],0);
        end;
    end;
end;

function TfmMCX.CheckListToStr(list: TCheckListBox) : string;
var
    i: integer;
begin
    Result:='';
     for i:=0 to list.Items.Count-1 do begin
          Result:=Result + IntToStr(Integer(list.Checked[i]));
     end;
end;

function TfmMCX.CreateCmd(proc: TProcess=nil):AnsiString;
var
    nthread, nblock,hitmax,seed, i: integer;
    bubbleradius,unitinmm,nphoton: extended;
    cmd, jsonfile, gpuid, debugflag, rootpath, inputjson, fname, savedetflag: string;
    shellscript, param: TStringList;
begin
    rootpath:='';
    cmd:=CreateCmdOnly;
    param:=TStringList.Create;
    param.StrictDelimiter:=true;
    param.Delimiter:=' ';
    if(proc<> nil) then begin
        proc.CommandLine:='';
        proc.Executable:=cmd;
        proc.Parameters.Clear;
    end;

    if(rbUseBench.Checked) then begin
        param.Add('--bench');
        param.Add(edBenchmark.Text);
        rootpath:=CreateWorkFolder(edSession.Text);
    end;

    if(Length(edSession.Text)>0) then begin
        param.Add('--session');
        param.Add(Trim(edSession.Text));
    end;
    if rbUseFile.Checked and (Length(edConfigFile.FileName)>0) then
    begin
        param.Add('--input');
        param.Add(Trim(edConfigFile.FileName));
        rootpath:=ExcludeTrailingPathDelimiter(ExtractFilePath(edConfigFile.FileName));
    end else if rbUseDesigner.Checked then begin
        jsonfile:=CreateWorkFolder(edSession.Text)+DirectorySeparator+Trim(edSession.Text)+'.json';
        inputjson:=StringReplace(SaveJSONConfig(jsonfile),'"','"',[rfReplaceAll]);
        {$IFDEF WINDOWS}
        inputjson:=StringReplace(inputjson,'"', '\"',[rfReplaceAll]);
        {$ENDIF}
        if(inputjson='') then
            exit;
        if(ckDoRemote.Checked) and (not (ckSharedFS.Checked)) then begin
            param.Add('--input');
            param.Add(''''+Trim(inputjson)+'''');
        end else begin
            param.Add('--input');
            param.Add(Trim(jsonfile));
            rootpath:=ExcludeTrailingPathDelimiter(ExtractFilePath(jsonfile));
        end;
    end;
    if(Length(sgConfig.Cells[2,14])>0) then
        rootpath:=sgConfig.Cells[2,14];
    if(ckDoRemote.Checked) then begin
        if(rootpath='') then
            rootpath:='MCXOutput'+'/'+CreateCmdOnly+'sessions'+'/'+Trim(edSession.Text);
    end;
    param.Add('--root');
    param.Add(rootpath);
    param.Add('--outputformat');
    param.Add(edOutputFormat.Text);

    try
        nthread:=StrToInt(edThread.Text);
        nphoton:=StrToFloat(edPhoton.Text);
        nblock:=StrToInt(edBlockSize.Text);
        bubbleradius:=StrToFloat(edBubble.Text);
        gpuid:=CheckListToStr(edGPUID);
        unitinmm:=StrToFloat(edUnitInMM.Text);
        hitmax:=StrToInt(edDetectedNum.Text);
        if not (ckDoReplay.Checked) then
            seed:=StrToInt(edSeed.Text);
    except
        MessageDlg('Input Error','Invalid numbers: check the values for thread, block, photon and time gate settings', mtError, [mbOK],0);
        exit;
    end;

    if(true) then begin
        param.Add('--gpu');
        param.Add(gpuid);
        if(ckAutopilot.Checked) then begin
          param.Add('--autopilot');
          param.Add('1');
        end else begin
          param.Add('--thread');
          param.Add(IntToStr(nthread));
          param.Add('--blocksize');
          param.Add(IntToStr(nblock));
        end;
    end;
    param.Add('--photon');
    param.Add(Format('%.0f',[nphoton]));
    param.Add('--normalize');
    param.Add(Format('%d',[Integer(ckNormalize.Checked)]));
    param.Add('--save2pt');
    param.Add(Format('%d',[Integer(ckSaveData.Checked)]));
    param.Add('--reflect');
    param.Add(Format('%d',[Integer(ckReflect.Checked)]));
    param.Add('--savedet');
    param.Add(Format('%d',[Integer(ckSaveDetector.Checked)]));
    param.Add('--unitinmm');
    param.Add(Format('%f',[unitinmm]));
    if(Length(edSeed.Text)>0) then begin
      param.Add('--seed');
      param.Add(Format('%s',[edSeed.Text]));
    end;
    if(edReplayDet.Enabled) then begin
      param.Add('--replaydet');
      param.Add(Format('%d',[edReplayDet.Value]));
    end;
    if(ckSaveDetector.Checked) then begin
      param.Add('--saveseed');
      param.Add(Format('%d',[Integer(ckSaveSeed.Checked)]));
    end;

    if(grProgram.ItemIndex>=1) then begin
      param.Add('--atomic');
      param.Add(Format('%d',[1 - grAtomic.ItemIndex]));
    end;
    param.Add('--specular');
    param.Add(Format('%d',[Integer(ckSpecular.Checked)]));
    if (grProgram.ItemIndex=1) then begin
         param.Add('--basisorder');
         param.Add(Format('%d',[edRespin.Value]));
    end;

    if(grAtomic.ItemIndex=0) then begin
           if(grProgram.ItemIndex=0) then begin
               param.Add('--skipradius');
               param.Add('-2');
           end;
    end;

    if(grProgram.ItemIndex<>1) then begin
        param.Add('--array');
        param.Add(Format('%d',[grArray.ItemIndex]));
        param.Add('--dumpmask');
        param.Add(Format('%d',[Integer(ckSaveMask.Checked)]));
        param.Add('--repeat');
        param.Add(Format('%d',[edRespin.Value]));
    end;

    if(grProgram.ItemIndex<>1) then begin
        if(ckSaveDetector.Checked) then begin
            savedetflag:='';
            for i:=0 to ckbDet.Items.Count-1 do begin
                 if(ckbDet.Checked[i]) then
                     savedetflag:=savedetflag+SaveDetFlags[i+1];
            end;
            if(Length(savedetflag)>0) then begin
                param.Add('--savedetflag');
                param.Add(savedetflag);
            end;
            param.Add('--maxdetphoton');
            param.Add(Format('%d',[hitmax]));
        end;

        if(not ckReflect.Checked) then begin
            savedetflag:='';
            for i:=0 to vlBC.Strings.Count-1 do begin
                savedetflag:=savedetflag+vlBC.Values[vlBC.Cells[0,i]][1];
            end;
            if(Length(savedetflag)>0) then begin
                param.Add('--bc');
                param.Add(savedetflag);
            end;
        end;
    end;
    if(not ckSkipVoid.Checked) then begin
        param.Add('--voidtime');
        param.Add('0');
    end;
    debugflag:='';
    for i:=0 to ckbDebug.Items.Count-1 do begin
         if(ckbDebug.Checked[i]) then
             debugflag:=debugflag+DebugFlags[i+1];
    end;
    if(Length(debugflag)>0) then begin
        param.Add('--debug');
        param.Add(debugflag);
    end;

    if(Length(edMoreParam.Text)>0) then begin
        shellscript:=TStringList.Create;
        shellscript.StrictDelimiter:=true;
        shellscript.Delimiter:=' ';
        shellscript.DelimitedText:=edMoreParam.Text;
        param.AddStrings(shellscript);
        shellscript.Free;
    end;

    AddLog('"-- Command: --"');
    AddLog(cmd+' '+param.DelimitedText);

    if(ckDoRemote.Checked) and (edRemote.ItemIndex<>0) then begin
        shellscript:=TStringList.Create;
        shellscript.StrictDelimiter:=true;
        shellscript.Delimiter:=' ';
        shellscript.DelimitedText:=ExpandPassword(Trim(edRemote.Text));
        shellscript.Add(cmd);
        if(shellscript.Count>1) then begin
          for i:=1 to shellscript.Count-1 do begin
            param.Insert(i-1,shellscript.Strings[i]);
          end;
          cmd:=shellscript.Strings[0];
        end;
        shellscript.Free;
        AddLog('Remote Command: '+edRemote.Text);
    end;

    if(Length(jsonfile)>0) then begin
         shellscript:=TStringList.Create;
         shellscript.Add('#!/bin/sh');
         shellscript.Add(cmd+' '+param.DelimitedText);
         shellscript.SaveToFile(ChangeFileExt(jsonfile,'.sh'));
         shellscript.Clear;
         shellscript.Add('@echo off');
         shellscript.Add(cmd+' '+param.DelimitedText);
         shellscript.SaveToFile(ChangeFileExt(jsonfile,'.bat'));
         shellscript.Free;
    end;
    if(proc<> nil) then begin
        proc.Executable:=SearchForExe(cmd);
        proc.Parameters.CommaText:=param.CommaText;
    end;
    param.QuoteChar:='''';
    Result:=cmd+' '+param.DelimitedText;
    param.Free;
end;

function TfmMCX.GridToStr(grid:TStringGrid):AnsiString;
var
    i: integer;
    json: TStrings;
begin
  json := TStringList.Create;
  json.StrictDelimiter:=true;
  Result:='';
  try
      try
          for i := grid.FixedRows to grid.RowCount - 1 do
          begin
              if (Length(grid.Cells[0,i])=0) and (Length(grid.Cells[2,i])=0) then break;
              json.Add(grid.Rows[i].CommaText);
          end;
      except
          On E : Exception do
              MessageDlg('Input Error', E.Message, mtError, [mbOK],0);
      end;
      json.Delimiter:='|';
      //json.QuoteChar:=' ';
      Result:=json.DelimitedText;
    finally
        json.Free;
    end;
end;


procedure TfmMCX.StrToGrid(str: string; grid:TStringGrid);
var
    i: integer;
    json,rowtext: TStrings;
begin
  json := TStringList.Create;
  json.StrictDelimiter:=true;
  json.Delimiter:='|';
  json.DelimitedText:=str;

  rowtext := TStringList.Create;

  try
      try
          if(grid.RowCount < json.Count+1) then
              grid.RowCount:= json.Count+1;
          for i := 0 to json.Count-1 do begin
              rowtext.CommaText:=json.Strings[i];
              if((grid.Name='sgMedia') or (grid.Name='sgDet') )and (rowtext.Count=4) then
                  json.Strings[i]:=','+json.Strings[i];
              grid.Rows[i+grid.FixedRows].CommaText:=json.Strings[i];
          end;
      except
          On E : Exception do
              MessageDlg('Input Error', E.Message, mtError, [mbOK],0);
      end;
    finally
        json.Free;
        rowtext.Free;
    end;
end;
procedure TfmMCX.ShowJSONData(AParent : TTreeNode; Data : TJSONData; toplevel: boolean=false);

Var
  N,N2 : TTreeNode;
  I : Integer;
  D : TJSONData;
  C : String;
  S : TStringList;
begin
  N:=nil;
  if Assigned(Data) then begin
    case Data.JSONType of
      jtArray,
      jtObject:
      begin
        if(N =nil) then N:=AParent;
        S:=TstringList.Create;
        try
          for I:=0 to Data.Count-1 do
            if Data.JSONtype=jtArray then
              S.AddObject('_ARRAY_',Data.items[i])
            else
              S.AddObject(TJSONObject(Data).Names[i],Data.items[i]);
          for I:=0 to S.Count-1 do begin
            if(not (S[i] = '_ARRAY_')) then begin
               N2:=AParent.TreeView.Items.AddChild(N,S[i]);
            end else begin
               N2:=N;
            end;
            D:=TJSONData(S.Objects[i]);
            N2.ImageIndex:=ImageTypeMap[D.JSONType];
            N2.SelectedIndex:=ImageTypeMap[D.JSONType];
            N2.Data:=D;
            ShowJSONData(N2,D);
          end
        finally
          S.Free;
        end;
      end;
      jtNull:
        N:=AParent.TreeView.Items.AddChild(AParent,'null');
    else
      N:=AParent.TreeView.Items.AddChild(AParent,Data.AsString);
    end;
    If toplevel=false and Assigned(N) then begin
      N.ImageIndex:=ImageTypeMap[Data.JSONType];
      N.SelectedIndex:=ImageTypeMap[Data.JSONType];
      N.Data:=Data;
    end;
  end;
end;

procedure TfmMCX.PanelToList2(node:TListItem);
var
    ed: TEdit;
    cb: TComboBox;
    ck: TCheckBox;
    cg: TCheckBox;
    se: TSpinEdit;
    gr: TRadioGroup;
    ckb: TCheckListBox;
    gb: TGroupBox;
    sg: TStringGrid;
    vl: TValueListEditor;
    i,id,idx: integer;
begin
    if(node=nil) then exit;
    for i:=0 to plSetting.ControlCount-1 do
    begin
      if(plSetting.Controls[i] is TGroupBox) then begin
       gb:= plSetting.Controls[i] as TGroupBox;
       for id:=0 to gb.ControlCount-1 do
        try
        if(gb.Controls[id] is TSpinEdit) then begin
           se:=gb.Controls[id] as TSpinEdit;
           idx:=MapList.IndexOf(SKey(se.Hint));
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(se.Value);
           continue;
        end;
        if(gb.Controls[id] is TEdit) then begin
           ed:=gb.Controls[id] as TEdit;
           idx:=MapList.IndexOf(SKey(ed.Hint));
           if(idx>=0) then node.SubItems.Strings[idx]:=ed.Text;
           continue;
        end;
        if(gb.Controls[id] is TRadioGroup) then begin
           gr:=gb.Controls[id] as TRadioGroup;
           idx:=MapList.IndexOf(SKey(gr.Hint));
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(gr.ItemIndex);
           continue;
        end;
        if(gb.Controls[id] is TComboBox) then begin
           cb:=gb.Controls[id] as TComboBox;
           idx:=MapList.IndexOf(SKey(cb.Hint));
           if(idx>=0) then node.SubItems.Strings[idx]:=cb.Text;
           continue;
        end;
        if(gb.Controls[id] is TCheckBox) then begin
           ck:=gb.Controls[id] as TCheckBox;
           idx:=MapList.IndexOf(SKey(ck.Hint));
           if(idx>=0) then node.SubItems.Strings[idx]:=IntToStr(Integer(ck.Checked));
           continue;
        end;
        if(gb.Controls[id] is TCheckListBox) then begin
           ckb:=gb.Controls[id] as TCheckListBox;
           idx:=MapList.IndexOf(SKey(ckb.Hint));
           if(idx>=0) then node.SubItems.Strings[idx]:=CheckListToStr(ckb);
           continue;
        end;
        if(gb.Controls[id] is TValueListEditor) then begin
           vl:=gb.Controls[id] as TValueListEditor;
           idx:=MapList.IndexOf(SKey(vl.Hint));
           if(idx>=0) then node.SubItems.Strings[idx]:=vl.Strings.CommaText;
           continue;
        end;
        except
        end;
      end;
    end;
    for i:=0 to tabInputData.ControlCount-1 do
    begin
        try
          if(tabInputData.Controls[i] is TStringGrid) then begin
             sg:=tabInputData.Controls[i] as TStringGrid;
             idx:=MapList.IndexOf(SKey(sg.Hint));
             if(idx>=0) then node.SubItems.Strings[idx]:=GridToStr(sg);
             continue;
          end;
        finally
        end;
    end;
    idx:=MapList.IndexOf(SKey(tvShapes.Hint));
    if(idx>=0) then
      if(JSONData.FindPath('Shapes') <> nil) then begin
        tvShapes.Tag:=UpdateShapeTag(JSONData.FindPath('Shapes'));
        node.SubItems.Strings[idx]:=JSONData.FormatJSON(AsJSONFormat);
      end else begin
        tvShapes.Tag:=UpdateShapeTag(JSONData);
        node.SubItems.Strings[idx]:='{"Shapes":'+JSONData.FormatJSON(AsJSONFormat)+'}';
      end;
end;

procedure TfmMCX.ListToPanel2(node:TListItem);
var
    ed: TEdit;
    cb: TComboBox;
    ck: TCheckBox;
    se: TSpinEdit;
    gr: TRadioGroup;
    ckb: TCheckListBox;
    sg: TStringGrid;
    gb: TGroupBox;
    fed:TFileNameEdit;
    vl: TValueListEditor;
    i,id,j,idx: integer;
    ss: string;
    slist: TStringList;
begin
    if(node=nil) then exit;
    edSession.Text:=node.Caption;

    for i:=0 to plSetting.ControlCount-1 do
    begin
      if(plSetting.Controls[i] is TRadioGroup) then begin
           gr:=plSetting.Controls[i] as TRadioGroup;
           idx:=MapList.IndexOf(SKey(gr.Hint));
           if(idx>=0) and (Length(node.SubItems.Strings[idx])>0) then begin
                try
                      gr.ItemIndex:=StrToInt(node.SubItems.Strings[idx]);
                except
                end;
           end;
           edRespinChange(ck);
           continue;
      end else if(plSetting.Controls[i] is TGroupBox) then begin
       gb:= plSetting.Controls[i] as TGroupBox;
       for id:=0 to gb.ControlCount-1 do begin
        if(gb.Controls[id] is TSpinEdit) then begin
           se:=gb.Controls[id] as TSpinEdit;
           idx:=MapList.IndexOf(SKey(se.Hint));
           if(idx>=0) then begin
             if(Length(node.SubItems.Strings[idx])>0) then begin
                try
                      se.Value:=StrToInt(node.SubItems.Strings[idx]);
                except
                end;
             end;
           end;
           continue;
        end;
        if(gb.Controls[id] is TEdit) then begin
           ed:=gb.Controls[id] as TEdit;
           idx:=MapList.IndexOf(SKey(ed.Hint));
           if(idx>=0) then ed.Text:=node.SubItems.Strings[idx];
           continue;
        end;
        if(gb.Controls[id] is TFileNameEdit) then begin
           fed:=gb.Controls[id] as TFileNameEdit;
           idx:=MapList.IndexOf(SKey(fed.Hint));
           if(idx>=0) then fed.Text:=node.SubItems.Strings[idx];
           continue;
        end;
        if(gb.Controls[id] is TValueListEditor) then begin
           vl:=gb.Controls[id] as TValueListEditor;
           idx:=MapList.IndexOf(SKey(vl.Hint));
           if(SKey(vl.Hint)='Boundary') then
           begin
               slist:=TStringList.Create;
               slist.CommaText:=node.SubItems.Strings[idx];
               vl.Values['x-']:=slist.Values['x-'];
               vl.Values['x+']:=slist.Values['x+'];
               vl.Values['y-']:=slist.Values['y-'];
               vl.Values['y+']:=slist.Values['y+'];
               vl.Values['z-']:=slist.Values['z-'];
               vl.Values['z+']:=slist.Values['z+'];
               slist.Free;
           end else begin
               if(idx>=0) then vl.Strings.CommaText:=node.SubItems.Strings[idx];
           end;
           continue;
        end;
        if(gb.Controls[id] is TRadioGroup) then begin
           gr:=gb.Controls[id] as TRadioGroup;
           idx:=MapList.IndexOf(SKey(gr.Hint));
           if(idx>=0) and (Length(node.SubItems.Strings[idx])>0) then begin
                try
                      gr.ItemIndex:=StrToInt(node.SubItems.Strings[idx]);
                except
                end;
           end;
           continue;
        end;
        if(gb.Controls[id] is TComboBox) then begin
           cb:=gb.Controls[id] as TComboBox;
           if(Length(SKey(cb.Hint))=0) then continue;
           idx:=MapList.IndexOf(SKey(cb.Hint));
           if(idx>=0) and (SKey(cb.Hint)='RemoteCmd') then begin
               if not (ckLockGPU.Checked) then begin
                   cb.Text:=node.SubItems.Strings[idx];
               end;
               continue;
           end;
           if(idx>=0) then begin
             if(node.SubItems.Strings[idx].IsEmpty) then
                 cb.ItemIndex:=0
             else
                 cb.Text:=node.SubItems.Strings[idx];
           end;
           continue;
        end;
        if(gb.Controls[id] is TCheckBox) then begin
           ck:=gb.Controls[id] as TCheckBox;
           if(Length(SKey(ck.Hint))=0) then continue;
           idx:=MapList.IndexOf(SKey(ck.Hint));
           if(idx>=0) and ((SKey(ck.Hint)='DoRemote') or (SKey(ck.Hint)='DoSharedFS')) then begin
               if not (ckLockGPU.Checked) then begin
                   ck.Checked:=(node.SubItems.Strings[idx]='1');
               end;
               continue;
           end;
           if(idx>=0) then ck.Checked:=(node.SubItems.Strings[idx]='1');
           edRespinChange(ck);
           continue;
        end;
        if(gb.Controls[id] is TCheckListBox) then begin
           ckb:=gb.Controls[id] as TCheckListBox;
           idx:=MapList.IndexOf(SKey(ckb.Hint));
           if(idx>=0) then begin
             ss:= node.SubItems.Strings[idx];
             if(SKey(ckb.Hint)='GPUID') then begin
               if(not ckLockGPU.Checked) then begin
                   ckb.Items.Clear;
                   for j:=0 to Length(node.SubItems.Strings[idx])-1 do begin
                       ckb.Items.Add('GPU#'+IntToStr(j+1));
                       if(ss[j+1]='1') then
                           ckb.Checked[j]:=true;
                   end;
               end;
             end else if(SKey(ckb.Hint)='DebugFlags') or (SKey(ckb.Hint)='SaveDetFlag') then begin
               ckb.CheckAll(cbUnchecked);
               for j:=0 to Min(ckb.Items.Count, Length(node.SubItems.Strings[idx]))-1 do begin
                   if(ss[j+1]='1') then
                       ckb.Checked[j]:=true;
               end;
             end;
           end;
           continue;
        end;
       end;
      end;
    end;
    for i:=0 to tabInputData.ControlCount-1 do
    begin
        try
          if(tabInputData.Controls[i] is TStringGrid) then begin
             sg:=tabInputData.Controls[i] as TStringGrid;
             idx:=MapList.IndexOf(SKey(sg.Hint));
             if(idx>=0) then begin
                 if((sg.Name='sgMedia') or (sg.Name='sgDet') ) then begin
                     sg.RowCount:=1;
                     sg.RowCount:=20;
                 end;
                 StrToGrid(node.SubItems.Strings[idx],sg);
             end;
             continue;
          end;
        finally
        end;
    end;
    idx:=MapList.IndexOf(SKey(tvShapes.Hint));
    if(idx>=0) then
        LoadJSONShapeTree(node.SubItems.Strings[idx]);
end;

function TfmMCX.SKey(str: AnsiString):AnsiString;
begin
  if(sscanf(str,'%s',[@SKey])<>1) then
      SKey:='';
end;

procedure TfmMCX.LoadSessionFromJSON(jfile: string);
var
    sl: TStringList;
begin
    sl:=TStringList.Create;
    try
      sl.LoadFromFile(jfile);
      NewSessionFromJSON(StringReplace(sl.Text, #10, '',[rfReplaceAll]), ExtractFilePath(jfile));
    finally
      sl.Free;
    end;
end;

procedure TfmMCX.NewSessionFromJSON(jsonstr, folder: string);
var
    js: TJSONData;
    root, jobj, jsrc : TJSONObject;
    media, jarr: TJSONArray;
    idx, i: integer;
    key, jfile: string;
    sl: TStringList;
begin
    try
      js:=GetJSON(jsonstr);
      if(js.JSONType <> jtObject) then
         raise Exception.Create('Invalid JSON input string');
      root:=TJSONObject(js);

      // Session Section
      jobj:=root.Objects['Session'];
      if(jobj = nil) then
         raise Exception.Create('Root-level object Session is required');

      mcxdoAddItemExecute(jobj);

      if(jobj.FindPath('Photons') <> nil) then
         edPhoton.Text:=jobj.FindPath('Photons').AsString;
      if(jobj.FindPath('RNGSeed') <> nil) then
         edSeed.Text:=jobj.FindPath('RNGSeed').AsString;
      if(jobj.FindPath('DoMismatch') <> nil) then
         ckReflect.Checked:=(jobj.FindPath('DoMismatch').AsInteger=1);
      if(jobj.FindPath('DoSaveVolume') <> nil) then
         ckSaveData.Checked:=(jobj.FindPath('DoSaveVolume').AsInteger=1);
      if(jobj.FindPath('DoNormalize') <> nil) then
         ckNormalize.Checked:=(jobj.FindPath('DoNormalize').AsInteger=1);
      if(jobj.FindPath('DoPartialPath') <> nil) then
         ckSaveDetector.Checked:=(jobj.FindPath('DoPartialPath').AsInteger=1);
      if(jobj.FindPath('DoSaveRef') <> nil) then
         ckSaveRef.Checked:=(jobj.FindPath('DoSaveRef').AsInteger=1);
      if(jobj.FindPath('DoSaveExit') <> nil) then begin
         ckbDet.Checked[4]:=(jobj.FindPath('DoSaveExit').AsInteger=1);
         ckbDet.Checked[5]:=ckbDet.Checked[4];
      end;
      if(jobj.FindPath('DoSaveSeed') <> nil) then
         ckSaveSeed.Checked:=(jobj.FindPath('DoSaveSeed').AsInteger=1);
      if(jobj.FindPath('DoAutoThread') <> nil) then
         ckAutopilot.Checked:=(jobj.FindPath('DoAutoThread').AsInteger=1);
      if(jobj.FindPath('DoDCS') <> nil) then
         ckbDet.Checked[3]:=(jobj.FindPath('DoDCS').AsInteger=1);
      if(jobj.FindPath('DoSpecular') <> nil) then
         ckSpecular.Checked:=(jobj.FindPath('DoSpecular').AsInteger=1);
      if(jobj.FindPath('OutputFormat') <> nil) then
         edOutputFormat.Text:=jobj.FindPath('OutputFormat').AsString;
      if(jobj.FindPath('OutputType') <> nil) then begin
         idx:=Pos(LowerCase(jobj.FindPath('OutputType').AsString),LowerCase(OutputTypeFlags));
         if(idx>0) then edOutputType.ItemIndex:=idx-1;
      end;
      if(jobj.FindPath('RootPath') <> nil) then
         sgConfig.Cells[2,14]:=jobj.FindPath('RootPath').AsString;
      if(jobj.FindPath('SaveDataMask') <> nil) then begin
        key:=jobj.FindPath('SaveDataMask').AsString;
        if(Length(key)>0) then begin
          for i:=0 to ckbDet.Items.Count-1 do begin
             ckbDet.Checked[i]:=(Pos(SaveDetFlags[i+1],UpperCase(key))>0);
          end;
        end;
      end;

      // Forward Section
      jobj:=root.Objects['Forward'];
      if(jobj <> nil) then begin
         if(jobj.FindPath('T0') <> nil) then
            sgConfig.Cells[2,4]:=Format('%g',[jobj.FindPath('T0').AsFloat]);
         if(jobj.FindPath('T1') <> nil) then
            sgConfig.Cells[2,5]:=Format('%g',[jobj.FindPath('T1').AsFloat]);
         if(jobj.FindPath('Dt') <> nil) then
            sgConfig.Cells[2,6]:=Format('%g',[jobj.FindPath('Dt').AsFloat]);
      end;

      // Domain Section
      jobj:=root.Objects['Domain'];
      if(jobj <> nil) then begin
         if(jobj.FindPath('VolumeFile') <> nil) then begin
            jfile:=jobj.FindPath('VolumeFile').AsString;
            if(Pos('.json', jfile) >0) then begin
              if(Pos('.'+PathDelim, jfile) >0) or (Length(ExtractFileName(jfile))=Length(jfile)) then
                  jfile:=folder+jfile;
              if(FileExists(jfile)) then begin
                try
                  sl:=TStringList.Create();
                  sl.LoadFromFile(jfile);
                  LoadJSONShapeTree(StringReplace(sl.Text, #10, '',[rfReplaceAll]));
                finally
                  sl.Free;
                end;
              end else begin
                sgConfig.Cells[2,1]:=jobj.FindPath('VolumeFile').AsString;
              end
            end else begin
              sgConfig.Cells[2,1]:=jobj.FindPath('VolumeFile').AsString;
            end;
         end;
         if(jobj.FindPath('MediaFormat') <> nil) then
            sgConfig.Cells[2,2]:=jobj.FindPath('MediaFormat').AsString;
         if(jobj.FindPath('LengthUnit') <> nil) then
            edUnitInMM.Text:=jobj.FindPath('LengthUnit').AsString;
         if(jobj.FindPath('OriginType') <> nil) then
            ckSrcFrom0.Checked:=(jobj.FindPath('OriginType').AsInteger=1);
         if(jobj.FindPath('Media') <> nil) then begin
            media:=TJSONArray(jobj.FindPath('Media'));
            for i:=0 to media.Count-1 do begin
              sgMedia.Cells[1,i+1]:=Format('%g',[media.Items[i].FindPath('mua').AsFloat]);
              sgMedia.Cells[2,i+1]:=Format('%g',[media.Items[i].FindPath('mus').AsFloat]);
              sgMedia.Cells[3,i+1]:=Format('%g',[media.Items[i].FindPath('g').AsFloat]);
              sgMedia.Cells[4,i+1]:=Format('%g',[media.Items[i].FindPath('n').AsFloat]);
            end;
         end;
         if(jobj.FindPath('Dim') <> nil) then
            sgConfig.Cells[2,2]:=jobj.Arrays['Dim'].AsJSON;
      end;

      // Optode Section
      jobj:=root.Objects['Optode'];
      if(jobj <> nil) then begin
         jsrc:=TJSONObject(jobj.FindPath('Source'));
         if(jsrc <> nil) then begin
             if(jsrc.FindPath('Type') <> nil) then
                 sgConfig.Cells[2,10]:= jsrc.Strings['Type'];
             if(jsrc.FindPath('Pos') <> nil) then
                 sgConfig.Cells[2,8]:= jsrc.Arrays['Pos'].AsJSON;
             if(jsrc.FindPath('Dir') <> nil) then
                 sgConfig.Cells[2,9]:= jsrc.Arrays['Dir'].AsJSON;
             if(jsrc.FindPath('Param1') <> nil) then
                 sgConfig.Cells[2,11]:= jsrc.Arrays['Param1'].AsJSON;
             if(jsrc.FindPath('Param2') <> nil) then
                 sgConfig.Cells[2,12]:= jsrc.Arrays['Param2'].AsJSON;
         end;

         if(jobj.Find('Detector') <> nil) then begin
           if(jobj.Find('Detector').JSONType = jtArray) then begin
             jarr:=TJSONArray(jobj.Find('Detector'));
             for i:=0 to jarr.Count-1 do begin
               sgDet.Cells[1,i+1]:=Format('%g',[TJSONArray(jarr.Items[i].FindPath('Pos')).Items[0].AsFloat]);
               sgDet.Cells[2,i+1]:=Format('%g',[TJSONArray(jarr.Items[i].FindPath('Pos')).Items[1].AsFloat]);
               sgDet.Cells[3,i+1]:=Format('%g',[TJSONArray(jarr.Items[i].FindPath('Pos')).Items[2].AsFloat]);
               sgDet.Cells[4,i+1]:=Format('%g',[TJSONArray(jarr.Items[i].FindPath('R')).AsFloat]);
             end;
           end else if(jobj.Find('Detector').JSONType = jtObject) then begin
             sgDet.Cells[1,i+1]:=Format('%g',[TJSONArray(jobj.FindPath('Detector.Pos')).Items[0].AsFloat]);
             sgDet.Cells[2,i+1]:=Format('%g',[TJSONArray(jobj.FindPath('Detector.Pos')).Items[1].AsFloat]);
             sgDet.Cells[3,i+1]:=Format('%g',[TJSONArray(jobj.FindPath('Detector.Pos')).Items[2].AsFloat]);
             sgDet.Cells[4,i+1]:=Format('%g',[jobj.FindPath('Detector.R').AsFloat]);
           end;
         end;

         if(root.FindPath('Shapes') <> nil) then begin
              LoadJSONShapeTree(root.Arrays['Shapes'].AsJSON);
         end;
      end;
    finally
      if not (CurrentSession = nil) then
         PanelToList2(CurrentSession);
    end;
end;
initialization
  {$I mcxgui.lrs}
end.

